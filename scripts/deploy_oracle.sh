#!/usr/bin/env bash
# Copy the app + models to the Oracle VM and (re)start it behind HTTPS.
#   IP=<public ip> scripts/deploy_oracle.sh
set -euo pipefail
IP=${IP:?set IP=<vm public ip>}
USER_=${VM_USER:-ubuntu}
KEY=${KEY:-$HOME/.ssh/itantra_oci}
SITE="${IP//./-}.sslip.io"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new $USER_@$IP"

[ -f build/cloudrun/Dockerfile ] || .venv/bin/python scripts/publish.py cloudrun
cp deploy/oracle/setup_server.sh build/cloudrun/setup_server.sh
$SSH "sudo mkdir -p /opt/itantra && sudo chown $USER_ /opt/itantra"
rsync -az --info=progress2 -e "ssh -i $KEY" build/cloudrun/ "$USER_@$IP:/opt/itantra/"
$SSH "cd /opt/itantra && SITE=$SITE bash setup_server.sh"
echo "open https://$SITE (first load after a restart: ~20 s while models load)"
