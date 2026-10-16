#!/usr/bin/env bash
# Create an Always Free ARM VM (VM.Standard.A1.Flex, 4 OCPU / 24 GB) with a public IP and ports 22/80/443.
#   OCI_PROFILE=itantra scripts/oracle_create_vm.sh
# Needs: `oci session authenticate --profile-name itantra --region <home region>` done first.
set -euo pipefail
OCI="${OCI:-$HOME/.oci-cli-venv/bin/oci} --auth security_token --profile ${OCI_PROFILE:-itantra}"
NAME=itantra-stt
SSH_PUB="$HOME/.ssh/itantra_oci.pub"
q() { $OCI "$@"; }

TENANCY=$(grep -A10 "^\[${OCI_PROFILE:-itantra}\]" ~/.oci/config | awk -F= '/^tenancy/{print $2; exit}' | tr -d ' ')
COMP=${COMPARTMENT:-$TENANCY}
echo "compartment: $COMP"

VCN=$(q network vcn list -c "$COMP" --display-name "$NAME-vcn" --query 'data[0].id' --raw-output 2>/dev/null || true)
if [ -z "$VCN" ] || [ "$VCN" = "null" ]; then
  VCN=$(q network vcn create -c "$COMP" --display-name "$NAME-vcn" --cidr-blocks '["10.0.0.0/16"]' \
        --wait-for-state AVAILABLE --query 'data.id' --raw-output)
fi
IGW=$(q network internet-gateway list -c "$COMP" --vcn-id "$VCN" --query 'data[0].id' --raw-output 2>/dev/null || true)
if [ -z "$IGW" ] || [ "$IGW" = "null" ]; then
  IGW=$(q network internet-gateway create -c "$COMP" --vcn-id "$VCN" --is-enabled true --display-name "$NAME-igw" \
        --wait-for-state AVAILABLE --query 'data.id' --raw-output)
fi
RT=$(q network vcn get --vcn-id "$VCN" --query 'data."default-route-table-id"' --raw-output)
q network route-table update --rt-id "$RT" --force \
  --route-rules "[{\"destination\":\"0.0.0.0/0\",\"destinationType\":\"CIDR_BLOCK\",\"networkEntityId\":\"$IGW\"}]" >/dev/null
SL=$(q network vcn get --vcn-id "$VCN" --query 'data."default-security-list-id"' --raw-output)
rule() { echo "{\"source\":\"0.0.0.0/0\",\"protocol\":\"6\",\"tcpOptions\":{\"destinationPortRange\":{\"min\":$1,\"max\":$1}}}"; }
q network security-list update --security-list-id "$SL" --force \
  --ingress-security-rules "[$(rule 22),$(rule 80),$(rule 443)]" >/dev/null
SUBNET=$(q network subnet list -c "$COMP" --vcn-id "$VCN" --query 'data[0].id' --raw-output 2>/dev/null || true)
if [ -z "$SUBNET" ] || [ "$SUBNET" = "null" ]; then
  SUBNET=$(q network subnet create -c "$COMP" --vcn-id "$VCN" --cidr-block 10.0.0.0/24 --display-name "$NAME-subnet" \
           --wait-for-state AVAILABLE --query 'data.id' --raw-output)
fi

IMAGE=$(q compute image list -c "$COMP" --operating-system "Canonical Ubuntu" --operating-system-version "24.04" \
        --shape VM.Standard.A1.Flex --sort-by TIMECREATED --sort-order DESC --query 'data[0].id' --raw-output)
echo "image: $IMAGE"

for AD in $(q iam availability-domain list -c "$COMP" --query 'data[].name' --raw-output | tr -d '[]",'); do
  echo "trying $AD …"
  if ID=$(q compute instance launch -c "$COMP" --availability-domain "$AD" --display-name "$NAME" \
        --shape VM.Standard.A1.Flex --shape-config '{"ocpus":4,"memoryInGBs":24}' \
        --image-id "$IMAGE" --subnet-id "$SUBNET" --assign-public-ip true --boot-volume-size-in-gbs 60 \
        --ssh-authorized-keys-file "$SSH_PUB" --wait-for-state RUNNING --query 'data.id' --raw-output 2>/tmp/oci-launch.err); then
    IP=$(q compute instance list-vnics --instance-id "$ID" --query 'data[0]."public-ip"' --raw-output)
    echo "VM running: $IP"
    exit 0
  fi
  grep -o 'Out of host capacity[^"]*' /tmp/oci-launch.err || tail -3 /tmp/oci-launch.err
done
echo "No capacity in any availability domain right now; retry later (common for free ARM VMs)."
exit 1
