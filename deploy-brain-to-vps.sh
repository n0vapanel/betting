#!/bin/bash
# deploy-brain-to-vps.sh — deploy betting brain to VPS 45.86.62.152

set -e

VPS_IP="45.86.62.152"
VPS_USER="root"
VPS_KEY="${SSH_KEY_PATH:-.ssh/id_ed25519_basicrmm}"  # fallback key if env not set

echo "=== Deploying brain to VPS ${VPS_IP} ==="
echo "SSH key: ~/${VPS_KEY}"

# 1. Test SSH connectivity
echo "[1/5] Testing SSH connectivity..."
if ! ssh -i ~/"${VPS_KEY}" -o ConnectTimeout=5 "${VPS_USER}"@"${VPS_IP}" "echo OK" 2>/dev/null; then
    echo "❌ SSH to ${VPS_IP} failed. Check IP, key, and firewall."
    exit 1
fi
echo "✓ SSH connected"

# 2. Create brain directories on VPS
echo "[2/5] Creating brain directory structure..."
ssh -i ~/"${VPS_KEY}" "${VPS_USER}"@"${VPS_IP}" << 'EOF'
    mkdir -p /root/betting-brain/{models,logs,data,ledger,state}
    mkdir -p /root/betting-brain/backtest/{multisport/{mma,soccer,nba,nfl,nhl,mlb,tennis}}
    chmod 755 /root/betting-brain
    echo "Brain dirs ready"
EOF

# 3. Deploy repo files (brain + models)
echo "[3/5] Uploading brain code and models..."
scp -i ~/"${VPS_KEY}" -r brain/ "${VPS_USER}"@"${VPS_IP}":/root/betting-brain/
scp -i ~/"${VPS_KEY}" -r backtest/multisport/mma/models/ "${VPS_USER}"@"${VPS_IP}":/root/betting-brain/models/mma_v2/
scp -i ~/"${VPS_KEY}" .env "${VPS_USER}"@"${VPS_IP}":/root/betting-brain/.env.vps
echo "✓ Files uploaded"

# 4. Set up Python venv + dependencies on VPS
echo "[4/5] Setting up Python environment on VPS..."
ssh -i ~/"${VPS_KEY}" "${VPS_USER}"@"${VPS_IP}" << 'EOF'
    cd /root/betting-brain
    python3.11 -m venv venv 2>/dev/null || python3 -m venv venv
    source venv/bin/activate
    pip install -q --upgrade pip
    pip install -q telegram kaggle pandas numpy scikit-learn xgboost lightgbm 2>&1 | tail -1
    python -c "import telegram, pandas, sklearn; print('✓ Core libs OK')"
EOF

# 5. Deploy Telegram bot + brain service
echo "[5/5] Installing Telegram bot service..."
ssh -i ~/"${VPS_KEY}" "${VPS_USER}"@"${VPS_IP}" << 'EOFSERVICE'
    cat > /etc/systemd/system/betting-brain-bot.service << 'EOFUNIT'
[Unit]
Description=Betting Brain Telegram Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/betting-brain
EnvironmentFile=/root/betting-brain/.env.vps
ExecStart=/root/betting-brain/venv/bin/python -u brain/bot_telegram.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOFUNIT

    systemctl daemon-reload
    systemctl enable betting-brain-bot.service
    echo "✓ Service installed"
EOFSERVICE

echo ""
echo "=== ✅ Deployment complete ==="
echo ""
echo "Next steps:"
echo "  1. SSH to VPS: ssh root@${VPS_IP}"
echo "  2. Check venv: source /root/betting-brain/venv/bin/activate"
echo "  3. Start bot: systemctl start betting-brain-bot"
echo "  4. Check logs: journalctl -u betting-brain-bot -f"
echo "  5. Test Telegram: /start in your chat"
echo ""
