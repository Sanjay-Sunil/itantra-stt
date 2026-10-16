#!/usr/bin/env bash
# Runs ON the VM (Ubuntu 24.04 ARM) as the default user, from /opt/itantra. Idempotent.
#   SITE=<ip-with-dashes>.sslip.io bash deploy/oracle/setup_server.sh
set -euo pipefail
SITE=${SITE:?set SITE, e.g. 1-2-3-4.sslip.io}
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-dev build-essential libsndfile1 caddy

# Ubuntu images on OCI ship iptables rules that only allow SSH: open 80/443
for p in 80 443; do
  sudo iptables -C INPUT -p tcp --dport $p -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 5 -p tcp --dport $p -j ACCEPT
done
sudo sh -c 'command -v netfilter-persistent >/dev/null && netfilter-persistent save' || true

cd /opt/itantra
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/pip install -q --no-deps --force-reinstall .

sudo tee /etc/systemd/system/itantra.service >/dev/null <<UNIT
[Unit]
Description=iTantra STT dashboard
After=network-online.target

[Service]
User=$USER
WorkingDirectory=/opt/itantra
Environment=ITANTRA_CLOUD=1 INDICSTT_MODELS_DIR=/opt/itantra/models PYTHONUNBUFFERED=1
ExecStart=/opt/itantra/.venv/bin/streamlit run app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

# HTTPS via Let's Encrypt on a free sslip.io name (browsers only allow microphone recording over HTTPS)
sudo tee /etc/caddy/Caddyfile >/dev/null <<CADDY
$SITE {
    encode gzip
    reverse_proxy 127.0.0.1:8501
}
CADDY

sudo systemctl daemon-reload
sudo systemctl enable --now itantra
sudo systemctl restart itantra caddy
echo "deployed: https://$SITE"
