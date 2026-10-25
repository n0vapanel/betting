#!/usr/bin/env bash
# deploy/setup_vps.sh — install or update the Telegram betting bot on a Linux VPS.
#
# The bot gets its own small, sandboxed slice of the server so it can't disturb
# anything else running there:
#   - its own system user (bettingbot) with no login shell
#   - its own folder (/opt/bettingbot) and Python virtualenv
#   - a systemd service capped at 512 MB RAM and half a CPU core, with a
#     read-only view of the rest of the system
# It only makes outbound connections (Telegram long polling), so no ports are opened.
#
# Usage, as root on the VPS:
#   curl -fsSL https://raw.githubusercontent.com/n0vapanel/betting/main/deploy/setup_vps.sh -o setup_vps.sh
#   bash setup_vps.sh
#
# The first run asks for your bot token and chat ID and stores them in
# /opt/bettingbot/app/.env (readable only by root and the bot). Run it again any
# time to pull the latest code and restart. Overrides: BRANCH=..., REPO_URL=...
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/n0vapanel/betting.git}"
BRANCH="${BRANCH:-main}"
BOT_USER="bettingbot"
BASE="/opt/bettingbot"
APP="$BASE/app"
VENV="$BASE/venv"
ENV_FILE="$APP/.env"
SERVICE="bettingbot"

# ask VAR PROMPT PATTERN [secret]: read VAR from the terminal until it matches PATTERN.
# A value already set in the environment is used instead of asking.
ask() {
  local var="$1" prompt="$2" pattern="$3" secret="${4:-}"
  local value="${!var:-}"
  if [ -n "$value" ]; then
    [[ "$value" =~ $pattern ]] || { echo "  $var looks wrong."; exit 1; }
  else
    while :; do
      if [ -n "$secret" ]; then
        read -r -s -p "  $prompt: " value < /dev/tty; echo
      else
        read -r -p "  $prompt: " value < /dev/tty
      fi
      [[ "$value" =~ $pattern ]] && break
      echo "  That doesn't look right, try again."
    done
  fi
  printf -v "$var" '%s' "$value"
}

main() {
  [ "$(id -u)" -eq 0 ] || { echo "Run this as root (or with sudo)."; exit 1; }
  command -v systemctl >/dev/null || { echo "This needs a Linux server that uses systemd (Ubuntu, Debian, Fedora...)."; exit 1; }

  echo "[1/6] Installing system packages (git, Python)..."
  if command -v apt-get >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    # --no-upgrade: leave already-installed packages alone, other services may rely on them
    apt-get install -y -qq --no-upgrade git python3 python3-venv >/dev/null
  elif command -v dnf >/dev/null; then
    dnf install -y -q git python3
  elif command -v yum >/dev/null; then
    yum install -y -q git python3
  else
    echo "Couldn't find apt, dnf or yum. Install git and Python 3.10+, then run this again."; exit 1
  fi
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || { echo "The bot needs Python 3.10 or newer; this server has $(python3 --version 2>&1)."; exit 1; }

  echo "[2/6] Creating the $BOT_USER user..."
  id "$BOT_USER" >/dev/null 2>&1 \
    || useradd --system --home-dir "$BASE" --no-create-home --shell /usr/sbin/nologin "$BOT_USER"
  mkdir -p "$BASE"

  echo "[3/6] Getting the bot code ($BRANCH)..."
  if [ -d "$APP/.git" ]; then
    git -C "$APP" fetch --quiet origin "$BRANCH"
    git -C "$APP" checkout --quiet --force -B "$BRANCH" "origin/$BRANCH"
  else
    git clone --quiet --branch "$BRANCH" "$REPO_URL" "$APP"
  fi

  echo "[4/6] Installing Python packages (can take a minute)..."
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --disable-pip-version-check -r "$APP/requirements.txt"

  echo "[5/6] Bot settings..."
  if [ -f "$ENV_FILE" ]; then
    echo "  Keeping your existing settings in $ENV_FILE"
  else
    ask TELEGRAM_BOT_TOKEN "Telegram bot token (hidden while you paste)" '^[0-9]{6,12}:[A-Za-z0-9_-]{30,}$' secret
    ask TELEGRAM_CHANNEL_ID "Chat ID for bot notifications" '^-?[0-9]{5,20}$'
    echo "  API keys for real predictions are optional: press Enter to skip, add them to $ENV_FILE later."
    ask API_SPORTS_KEY "API-Sports key" '^[A-Za-z0-9_-]*$'
    ask THE_ODDS_API_KEY "The Odds API key" '^[A-Za-z0-9_-]*$'
    ask NEWS_API_KEY "NewsAPI key" '^[A-Za-z0-9_-]*$'
    ( umask 077
      cat > "$ENV_FILE" <<EOF
TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN
TELEGRAM_CHANNEL_ID=$TELEGRAM_CHANNEL_ID
API_SPORTS_KEY=$API_SPORTS_KEY
THE_ODDS_API_KEY=$THE_ODDS_API_KEY
NEWS_API_KEY=$NEWS_API_KEY
EOF
    )
    chown "root:$BOT_USER" "$ENV_FILE"
    chmod 640 "$ENV_FILE"
  fi

  echo "[6/6] Starting the bot as a background service..."
  cat > "/etc/systemd/system/$SERVICE.service" <<EOF
[Unit]
Description=Telegram betting bot
After=network-online.target
Wants=network-online.target
# Stop retrying after 5 crashes in 5 minutes (e.g. a wrong token)
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
User=$BOT_USER
Group=$BOT_USER
WorkingDirectory=$APP
EnvironmentFile=$ENV_FILE
ExecStart=$VENV/bin/python -m telegram_betting_bot_simple.main
Restart=on-failure
RestartSec=10
# The bot's slice of the server: capped resources, read-only view of everything else
MemoryMax=512M
CPUQuota=50%
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable --quiet "$SERVICE"
  systemctl reset-failed "$SERVICE" 2>/dev/null || true
  systemctl restart "$SERVICE"

  sleep 8
  if ! systemctl is-active --quiet "$SERVICE"; then
    echo "The bot didn't start. Last log lines:"
    journalctl -u "$SERVICE" -n 30 --no-pager
    exit 1
  fi
  echo "The bot is running."
  # Say hello through the notify bridge so you know messages reach your chat
  # shellcheck source=/dev/null
  if ( set -a; . "$ENV_FILE"; set +a
       "$VENV/bin/python" "$APP/brain/notify.py" --emoji "✅" --sport "Betting bot" \
         --msg "is live on your VPS. Send /start to try it." ) >/dev/null; then
    echo "  Sent a test message to your Telegram chat."
  else
    echo "  Couldn't message your chat: open your bot in Telegram, press Start, then run this again."
  fi

  cat <<EOF

Done. Open your bot in Telegram and send /start.

Useful commands:
  journalctl -u $SERVICE -f     live logs (Ctrl+C stops watching, the bot keeps running)
  systemctl restart $SERVICE    restart after editing settings
  nano $ENV_FILE   change the token or add API keys
  bash setup_vps.sh               update to the latest code from GitHub
EOF
}

main "$@"
