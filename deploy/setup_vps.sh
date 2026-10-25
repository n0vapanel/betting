#!/usr/bin/env bash
# deploy/setup_vps.sh — install or update the Telegram betting bot on a Linux VPS.
#
# The bot gets its own small, sandboxed slice of the server so it can't disturb
# anything else running there:
#   - its own system user (bettingbot) with no login shell
#   - its own folder (/opt/bettingbot) and Python virtualenv
#   - systemd services (the bot and the web dashboard) each capped at 512 MB RAM and
#     half a CPU core, with a read-only view of the rest of the system
# The bot only makes outbound connections; the dashboard listens on one port
# (DASHBOARD_PORT, default 8787) behind a password.
#
# Usage, as root on the VPS:
#   curl -fsSL https://raw.githubusercontent.com/n0vapanel/betting/main/deploy/setup_vps.sh -o setup_vps.sh
#   bash setup_vps.sh
#
# The first run asks for your bot token, chat ID, Odds API key and a dashboard
# password and stores them in /opt/bettingbot/app/.env (readable only by root
# and the bot). Run it again any
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
DASHBOARD="bettingdash"
# Secrets may be pre-set in the environment: keep them away from apt, git and pip.
export -n TELEGRAM_BOT_TOKEN TELEGRAM_CHANNEL_ID API_SPORTS_KEY THE_ODDS_API_KEY NEWS_API_KEY DASHBOARD_PASSWORD

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
  # The bot user must be able to read what root creates here, even if root's umask is stricter
  umask 022

  echo "[1/7] Installing system packages (git, Python)..."
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

  echo "[2/7] Creating the $BOT_USER user..."
  id "$BOT_USER" >/dev/null 2>&1 \
    || useradd --system --home-dir "$BASE" --no-create-home --shell /usr/sbin/nologin "$BOT_USER"
  mkdir -p "$BASE"
  chmod 755 "$BASE"

  echo "[3/7] Getting the bot code ($BRANCH)..."
  if [ -d "$APP/.git" ]; then
    git -C "$APP" fetch --quiet origin "$BRANCH"
    git -C "$APP" checkout --quiet --force -B "$BRANCH" "origin/$BRANCH"
  else
    git clone --quiet --branch "$BRANCH" "$REPO_URL" "$APP"
  fi

  echo "[4/7] Installing Python packages (can take a minute)..."
  # Install as the bot user so package install scripts never run as root. The service
  # still can't modify its own packages: ProtectSystem=strict makes them read-only.
  install -d -o "$BOT_USER" -g "$BOT_USER" "$VENV"
  [ -x "$VENV/bin/python" ] || runuser -u "$BOT_USER" -- python3 -m venv "$VENV"
  runuser -u "$BOT_USER" -- env HOME="$BASE" "$VENV/bin/pip" install --quiet --no-cache-dir \
    --disable-pip-version-check -r "$APP/requirements.txt"

  echo "[5/7] Settings..."
  if [ -f "$ENV_FILE" ]; then
    echo "  Keeping your existing settings in $ENV_FILE"
  else
    ask TELEGRAM_BOT_TOKEN "Telegram bot token (hidden while you paste)" '^[0-9]{6,12}:[A-Za-z0-9_-]{30,}$' secret
    ask TELEGRAM_CHANNEL_ID "Chat ID for bot notifications" '^-?[0-9]{5,20}$'
    echo "  Games and odds come from The Odds API (free key at the-odds-api.com)."
    echo "  Press Enter to skip for now and add it to $ENV_FILE later."
    ask THE_ODDS_API_KEY "The Odds API key" '^[A-Za-z0-9_-]*$'
    ( umask 077
      cat > "$ENV_FILE" <<EOF
TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN
TELEGRAM_CHANNEL_ID=$TELEGRAM_CHANNEL_ID
THE_ODDS_API_KEY=$THE_ODDS_API_KEY
# Timezone for game start times (IANA name, e.g. America/Chicago or Europe/London)
TIMEZONE=America/New_York
EOF
    )
  fi
  # The dashboard's settings (added to older installs too)
  local generated=""
  if ! grep -q '^DASHBOARD_PASSWORD=' "$ENV_FILE"; then
    ask DASHBOARD_PASSWORD "Dashboard password (8+ letters/digits, Enter to generate one)" '^([A-Za-z0-9_.@%+=:-]{8,64})?$' secret
    if [ -z "$DASHBOARD_PASSWORD" ]; then
      DASHBOARD_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')"
      generated="$DASHBOARD_PASSWORD"
    fi
    printf 'DASHBOARD_PASSWORD=%s\nDASHBOARD_PORT=%s\n' "$DASHBOARD_PASSWORD" "${DASHBOARD_PORT:-8787}" >> "$ENV_FILE"
  fi
  chown "root:$BOT_USER" "$ENV_FILE"
  chmod 640 "$ENV_FILE"
  local port
  port="$(grep -E '^DASHBOARD_PORT=' "$ENV_FILE" | tail -1 | cut -d= -f2)"
  port="${port:-8787}"

  echo "[6/7] Starting the bot as a background service..."
  write_unit "$SERVICE" "Telegram betting bot" "telegram_betting_bot_simple.main"
  start_unit "$SERVICE" || exit 1
  echo "  The bot is running."
  # Say hello through the notify bridge so you know messages reach your chat
  # shellcheck source=/dev/null
  if ( set -a; . "$ENV_FILE"; set +a
       "$VENV/bin/python" "$APP/brain/notify.py" --emoji "✅" --sport "Betting bot" \
         --msg "is live on your VPS. Send /start to try it." ) >/dev/null; then
    echo "  Sent a test message to your Telegram chat."
  else
    echo "  Couldn't message your chat: open your bot in Telegram, press Start, then run this again."
  fi

  echo "[7/7] Starting the dashboard on port $port..."
  write_unit "$DASHBOARD" "Betting dashboard (web)" "telegram_betting_bot_simple.dashboard"
  start_unit "$DASHBOARD" || exit 1
  if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "^Status: active"; then
    ufw allow "$port/tcp" comment "betting dashboard" >/dev/null && echo "  Opened port $port in the ufw firewall."
  elif command -v firewall-cmd >/dev/null && firewall-cmd --state >/dev/null 2>&1; then
    firewall-cmd --quiet --permanent --add-port="$port/tcp" && firewall-cmd --quiet --reload \
      && echo "  Opened port $port in firewalld."
  fi
  echo "  The dashboard is running."

  local address
  address="$(hostname -I 2>/dev/null | awk '{print $1}')"
  cat <<EOF

Done.
  Telegram: open your bot and send /start.
  Dashboard: http://${address:-YOUR-SERVER-IP}:$port  (any username; the dashboard password)
EOF
  if [ -n "$generated" ]; then
    echo "  Your dashboard password is: $generated  (it's also in $ENV_FILE)"
  fi
  cat <<EOF
  If the page doesn't load, allow port $port in your hosting provider's firewall.

Useful commands:
  journalctl -u $SERVICE -f        bot logs (Ctrl+C stops watching; the bot keeps running)
  journalctl -u $DASHBOARD -f      dashboard logs
  systemctl restart $SERVICE $DASHBOARD    restart after editing settings
  nano $ENV_FILE      change keys and passwords
  bash setup_vps.sh                  update to the latest code from GitHub
EOF
}

# write_unit NAME DESCRIPTION MODULE: a sandboxed systemd service running "python -m MODULE"
write_unit() {
  cat > "/etc/systemd/system/$1.service" <<EOF
[Unit]
Description=$2
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
ExecStart=$VENV/bin/python -m $3
Restart=on-failure
RestartSec=10
# Its slice of the server: capped resources, read-only view of everything else
MemoryMax=512M
CPUQuota=50%
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
# The one writable folder (/var/lib/$BOT_USER), shared by the bot and the dashboard: saved odds,
# quota log and bot state. systemd passes its path in STATE_DIRECTORY
StateDirectory=$BOT_USER

[Install]
WantedBy=multi-user.target
EOF
}

# start_unit NAME: (re)start a service and show its logs if it doesn't stay up
start_unit() {
  systemctl daemon-reload
  systemctl enable --quiet "$1"
  systemctl reset-failed "$1" 2>/dev/null || true
  systemctl restart "$1"
  sleep 8
  if ! systemctl is-active --quiet "$1"; then
    echo "  $1 didn't start. Last log lines:"
    journalctl -u "$1" -n 30 --no-pager
    return 1
  fi
}

main "$@"
