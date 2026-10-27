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
# With a domain name (DOMAIN=bets.example.com, see below) the dashboard is served
# over HTTPS by Caddy, which gets and renews the certificate by itself. Telegram
# only opens HTTPS pages as Mini Apps, so this is what puts an "Open app" button
# in the bot. The dashboard then listens on 127.0.0.1 only, behind Caddy.
#
# Usage, as root on the VPS:
#   curl -fsSL https://raw.githubusercontent.com/n0vapanel/betting/main/deploy/setup_vps.sh -o setup_vps.sh
#   bash setup_vps.sh
#
# The first run asks for your bot token, chat ID, Odds API key and a dashboard
# password and stores them in /opt/bettingbot/app/.env (readable only by root
# and the bot). Run it again any time to pull the latest code and restart; it
# keeps your settings and only asks for ones that are missing.
#
# Keys put in front of the command are saved without asking, and replace saved
# ones (a new server, or switching to a paid API key). Start the line with a
# space so the keys stay out of the shell history:
#    TELEGRAM_BOT_TOKEN=... TELEGRAM_CHANNEL_ID=... THE_ODDS_API_KEY=... bash setup_vps.sh
# Also: API_SPORTS_KEY, DASHBOARD_PASSWORD, BRANCH (default main), REPO_URL.
#
# HTTPS and the Telegram Mini App (optional): point a domain's DNS A record at this
# server, then run
#    DOMAIN=bets.example.com bash setup_vps.sh
# The domain is saved, so later runs keep serving it. DOMAIN=off goes back to the
# plain http://<server>:8787 address.
#
# The app is private: anyone on Telegram can find the bot, so only the Telegram
# users listed in TELEGRAM_ALLOWED_IDS can open it there (the dashboard password
# still works in a browser). The first run lists you, when bot notifications go to
# your private chat. Everyone else is shown their Telegram id; to let friends in,
# list everyone (this replaces the saved list):
#    TELEGRAM_ALLOWED_IDS=11111111,22222222 bash setup_vps.sh
#
# AI analysts (optional; each call costs money on that API account):
#   ANTHROPIC_API_KEY=sk-ant-...  Claude (console.anthropic.com)
#   DEEPSEEK_API_KEY=sk-...       DeepSeek (platform.deepseek.com)
#   AI_WEB_SEARCH=1               Claude checks the latest news before each forecast (about $0.01 a search)
#   AI_DAILY_LIMIT=25             calls per analyst per 24 hours
#   AI_EFFORT=medium              how hard Claude thinks: low, medium, high, xhigh or max
#   ANTHROPIC_MODEL, DEEPSEEK_MODEL to pick other models
# The AI chat (uses ANTHROPIC_API_KEY), in US cents of Claude a day:
#   AI_CHAT_DAILY_CENTS=100       each person's allowance
#   AI_CHAT_WEB_DAILY_CENTS=300   everyone signed in with the dashboard password, together (default: 3 people's)
#   AI_CHAT_DAILY_TOTAL_CENTS=1000  the whole app (default: 10 people's)
# The markets brain ("Markets say"; optional, the defaults are fine):
#   ODDS_BOOKMAKERS=pinnacle,betfair_ex_eu,...  the odds board's bookmakers, at most 10 (10 cost one region:
#                                 3 credits a board); "none" asks for ODDS_REGIONS instead
#   SHARP_BOOKS=pinnacle,betfair_ex_eu,matchbook  books that only anchor the consensus (never a price offered)
#   ODDS_PRICE_BOOKS=draftkings,fanduel  the books "best price" and every ticket use (default: the rest)
#   MARKETS_ENABLED=0             turn off Kalshi, Polymarket, ESPN's predictor and the ratings (free; on by default)
#
# The dashboard password should be 12 or more characters: a shorter one still works, but the setup
# warns about it.
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
CADDYFILE="/etc/caddy/Caddyfile"
# The bot's own Caddy site, imported from the main Caddyfile so other sites served by Caddy stay as they are
CADDY_SITE="/etc/caddy/bettingbot.caddy"
# Secrets may be pre-set in the environment: keep them away from apt, git and pip.
export -n TELEGRAM_BOT_TOKEN TELEGRAM_CHANNEL_ID API_SPORTS_KEY THE_ODDS_API_KEY NEWS_API_KEY DASHBOARD_PASSWORD \
  ANTHROPIC_API_KEY DEEPSEEK_API_KEY

# Settings that can be given on the command line, and what a valid value looks like
declare -A PATTERNS=(
  [TELEGRAM_BOT_TOKEN]='^[0-9]{6,12}:[A-Za-z0-9_-]{30,}$'
  [TELEGRAM_CHANNEL_ID]='^-?[0-9]{5,20}$'
  [THE_ODDS_API_KEY]='^[A-Za-z0-9_-]{16,64}$'
  [API_SPORTS_KEY]='^[A-Za-z0-9_-]{16,64}$'
  [DASHBOARD_PASSWORD]='^[A-Za-z0-9_.@%+=:-]{8,64}$'
  [ANTHROPIC_API_KEY]='^sk-ant-[A-Za-z0-9_-]{20,300}$'
  [DEEPSEEK_API_KEY]='^sk-[A-Za-z0-9]{16,100}$'
  [AI_WEB_SEARCH]='^(0|1|true|false)$'
  [AI_DAILY_LIMIT]='^[0-9]{1,4}$'
  [AI_EFFORT]='^(low|medium|high|xhigh|max)$'
  [AI_CHAT_DAILY_CENTS]='^[0-9]{1,6}$'
  [AI_CHAT_WEB_DAILY_CENTS]='^[0-9]{1,6}$'
  [AI_CHAT_DAILY_TOTAL_CENTS]='^[0-9]{1,7}$'
  [ANTHROPIC_MODEL]='^claude-[a-z0-9.-]{3,60}$'
  [DEEPSEEK_MODEL]='^deepseek-[a-z0-9.-]{2,60}$'
  [TELEGRAM_ALLOWED_IDS]='^[0-9]{1,20}(,[0-9]{1,20})*$'
  [ODDS_BOOKMAKERS]='^([a-z0-9][a-z0-9_]{1,39}(,[a-z0-9][a-z0-9_]{1,39}){0,9}|none)$'
  [SHARP_BOOKS]='^[a-z0-9][a-z0-9_]{1,39}(,[a-z0-9][a-z0-9_]{1,39}){0,9}$'
  [ODDS_PRICE_BOOKS]='^[a-z0-9][a-z0-9_]{1,39}(,[a-z0-9][a-z0-9_]{1,39}){0,9}$'
  [MARKETS_ENABLED]='^(0|1|true|false)$'
)
# DOMAIN is checked and saved on its own, since it can also be "off"
DOMAIN_PATTERN='^([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{0,61}[a-z0-9]$'
PRESET=()
for var in "${!PATTERNS[@]}"; do
  if [ -n "${!var:-}" ]; then PRESET+=("$var"); fi
done

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

# warn_short_password VALUE: passwords of 8-11 characters keep working, but say plainly that 12+ is safer.
# Only the length is printed, never the password
MIN_SAFE_PASSWORD=12
warn_short_password() {
  if [ "${#1}" -lt "$MIN_SAFE_PASSWORD" ]; then
    echo "  WARNING: the dashboard password is only ${#1} characters long. It works, but anyone who finds the"
    echo "  dashboard's address can try to guess it: use $MIN_SAFE_PASSWORD or more characters. To change it, run"
    echo "  this script again with DASHBOARD_PASSWORD=<new password> in front (start the line with a space)."
  fi
}

# set_env VAR VALUE: save VAR=VALUE in the .env file, replacing an earlier value
set_env() {
  sed -i "/^$1=/d" "$ENV_FILE"
  echo "$1=$2" >> "$ENV_FILE"
}

main() {
  [ "$(id -u)" -eq 0 ] || { echo "Run this as root (or with sudo)."; exit 1; }
  command -v systemctl >/dev/null || { echo "This needs a Linux server that uses systemd (Ubuntu, Debian, Fedora...)."; exit 1; }
  # The bot user must be able to read what root creates here, even if root's umask is stricter
  umask 022
  # A domain given on an earlier run is saved in .env and keeps being served; DOMAIN=off stops that.
  # Defaulted first: under "set -u" an unset DOMAIN (a first install) would stop the script
  DOMAIN="${DOMAIN:-}"
  if [ -z "$DOMAIN" ] && [ -f "$ENV_FILE" ]; then
    DOMAIN="$(grep -E '^DOMAIN=' "$ENV_FILE" | tail -1 | cut -d= -f2 || true)"
  fi
  DOMAIN="${DOMAIN,,}"
  local domain_off=""
  if [ "$DOMAIN" = "off" ]; then
    domain_off=1
    DOMAIN=""
  fi
  if [ -n "$DOMAIN" ] && ! [[ "$DOMAIN" =~ $DOMAIN_PATTERN ]]; then
    echo "DOMAIN looks wrong: give just the name, e.g. DOMAIN=bets.example.com (or DOMAIN=off)"; exit 1
  fi

  echo "[1/8] Installing system packages (git, Python${DOMAIN:+, Caddy})..."
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
  # Before any settings change, so a server where Caddy can't be installed or can't have ports 80
  # and 443 is left as it was
  if [ -n "$DOMAIN" ]; then
    check_web_ports
    install_caddy
  fi

  echo "[2/8] Creating the $BOT_USER user..."
  id "$BOT_USER" >/dev/null 2>&1 \
    || useradd --system --home-dir "$BASE" --no-create-home --shell /usr/sbin/nologin "$BOT_USER"
  mkdir -p "$BASE"
  chmod 755 "$BASE"

  echo "[3/8] Getting the bot code ($BRANCH)..."
  if [ -d "$APP/.git" ]; then
    git -C "$APP" fetch --quiet origin "$BRANCH"
    git -C "$APP" checkout --quiet --force -B "$BRANCH" "origin/$BRANCH"
  else
    git clone --quiet --branch "$BRANCH" "$REPO_URL" "$APP"
  fi

  echo "[4/8] Installing Python packages (can take a minute)..."
  # Install as the bot user so package install scripts never run as root. The service
  # still can't modify its own packages: ProtectSystem=strict makes them read-only.
  install -d -o "$BOT_USER" -g "$BOT_USER" "$VENV"
  [ -x "$VENV/bin/python" ] || runuser -u "$BOT_USER" -- python3 -m venv "$VENV"
  runuser -u "$BOT_USER" -- env HOME="$BASE" "$VENV/bin/pip" install --quiet --no-cache-dir \
    --disable-pip-version-check -r "$APP/requirements.txt"

  echo "[5/8] Settings..."
  if [ -f "$ENV_FILE" ]; then
    echo "  Keeping your existing settings in $ENV_FILE"
  else
    ask TELEGRAM_BOT_TOKEN "Telegram bot token (hidden while you paste)" '^[0-9]{6,12}:[A-Za-z0-9_-]{30,}$' secret
    ask TELEGRAM_CHANNEL_ID "Chat ID for bot notifications" '^-?[0-9]{5,20}$'
    ( umask 077
      cat > "$ENV_FILE" <<EOF
TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN
TELEGRAM_CHANNEL_ID=$TELEGRAM_CHANNEL_ID
# Timezone for game start times (IANA name, e.g. America/Chicago or Europe/London)
TIMEZONE=America/New_York
EOF
    )
  fi
  # The bot and dashboard can't show games without it, so it's asked for again if an earlier run left it out
  if ! grep -qE '^THE_ODDS_API_KEY=[A-Za-z0-9_-]+' "$ENV_FILE"; then
    echo "  Games and odds come from The Odds API (free key at the-odds-api.com)."
    ask THE_ODDS_API_KEY "The Odds API key" '^[A-Za-z0-9_-]{16,64}$'
    sed -i '/^THE_ODDS_API_KEY=/d' "$ENV_FILE"
    echo "THE_ODDS_API_KEY=$THE_ODDS_API_KEY" >> "$ENV_FILE"
  fi
  # The dashboard's settings (added to older installs too)
  local generated=""
  if ! grep -q '^DASHBOARD_PASSWORD=' "$ENV_FILE"; then
    ask DASHBOARD_PASSWORD "Dashboard password (12+ letters/digits, Enter to generate one)" '^([A-Za-z0-9_.@%+=:-]{8,64})?$' secret
    if [ -z "$DASHBOARD_PASSWORD" ]; then
      DASHBOARD_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')"
      generated="$DASHBOARD_PASSWORD"
    else
      warn_short_password "$DASHBOARD_PASSWORD"
    fi
    printf 'DASHBOARD_PASSWORD=%s\nDASHBOARD_PORT=%s\n' "$DASHBOARD_PASSWORD" "${DASHBOARD_PORT:-8787}" >> "$ENV_FILE"
  fi
  # Keys given on the command line replace the saved ones
  local var saved_password
  saved_password="$(grep -E '^DASHBOARD_PASSWORD=' "$ENV_FILE" | tail -1 | cut -d= -f2- || true)"
  for var in ${PRESET[@]+"${PRESET[@]}"}; do
    [[ "${!var}" =~ ${PATTERNS[$var]} ]] || { echo "  $var looks wrong."; exit 1; }
    set_env "$var" "${!var}"
  done
  # A password given on the command line replaces the saved one: warn when the new one is short
  if [ -n "${DASHBOARD_PASSWORD:-}" ] && [ "$DASHBOARD_PASSWORD" != "$saved_password" ]; then
    warn_short_password "$DASHBOARD_PASSWORD"
  fi
  if [ ${#PRESET[@]} -gt 0 ]; then echo "  Saved from the command line: ${PRESET[*]}"; fi
  # Behind Caddy: the bot links to the HTTPS address, and the dashboard only listens to Caddy
  if [ -n "$DOMAIN" ]; then
    set_env DOMAIN "$DOMAIN"
    set_env PUBLIC_URL "https://$DOMAIN"
    set_env DASHBOARD_HOST "127.0.0.1"
  elif [ -n "$domain_off" ]; then
    sed -i -E '/^(DOMAIN|PUBLIC_URL|DASHBOARD_HOST)=/d' "$ENV_FILE"
    echo "  HTTPS turned off: the dashboard goes back to its own port."
  fi
  # Who may open the app from Telegram. A private chat's id is that person's Telegram id, so when
  # notifications go to the owner's own chat, the owner is on the list from the start
  if ! grep -q '^TELEGRAM_ALLOWED_IDS=' "$ENV_FILE"; then
    local chat
    chat="$(grep -E '^TELEGRAM_CHANNEL_ID=' "$ENV_FILE" | tail -1 | cut -d= -f2 || true)"
    if [[ "$chat" =~ ^[0-9]{5,20}$ ]]; then
      set_env TELEGRAM_ALLOWED_IDS "$chat"
      echo "  Only you (Telegram id $chat) can open the app from Telegram for now; see TELEGRAM_ALLOWED_IDS."
    fi
  fi
  chown "root:$BOT_USER" "$ENV_FILE"
  chmod 640 "$ENV_FILE"
  local port
  port="$(grep -E '^DASHBOARD_PORT=' "$ENV_FILE" | tail -1 | cut -d= -f2)"
  port="${port:-8787}"

  echo "[6/8] Starting the bot as a background service..."
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

  echo "[7/8] Starting the dashboard on port $port..."
  write_unit "$DASHBOARD" "Betting dashboard (web)" "telegram_betting_bot_simple.dashboard"
  start_unit "$DASHBOARD" || exit 1
  # Behind Caddy the port stays closed: only Caddy (on this server) talks to it
  if [ -n "$DOMAIN" ]; then
    :
  elif ufw_active; then
    ufw allow "$port/tcp" comment "betting dashboard" >/dev/null && echo "  Opened port $port in the ufw firewall."
  elif firewalld_active; then
    firewall-cmd --quiet --permanent --add-port="$port/tcp" && firewall-cmd --quiet --reload \
      && echo "  Opened port $port in firewalld."
  fi
  echo "  The dashboard is running."

  if [ -n "$DOMAIN" ]; then
    echo "[8/8] Serving the dashboard over HTTPS at $DOMAIN (Caddy)..."
    setup_caddy "$port"
  elif [ -n "$domain_off" ]; then
    echo "[8/8] HTTPS: turned off (DOMAIN=off)."
    remove_caddy_site
  else
    echo "[8/8] HTTPS: skipped (no DOMAIN given)."
  fi

  local address
  address="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [ -n "$DOMAIN" ]; then
    local allowed
    allowed="$(grep -E '^TELEGRAM_ALLOWED_IDS=' "$ENV_FILE" | tail -1 | cut -d= -f2 || true)"
    cat <<EOF

Done.
  Telegram: open your bot, send /start and tap "📱 Open app" (or the menu button).
  Dashboard: https://$DOMAIN  (sign in with the dashboard password)
  Telegram users who can open the app: ${allowed:-nobody yet}. Anyone else is shown their
  Telegram id; to let them in, list everyone: TELEGRAM_ALLOWED_IDS=id1,id2 bash setup_vps.sh
EOF
  else
    cat <<EOF

Done.
  Telegram: open your bot and send /start.
  Dashboard: http://${address:-YOUR-SERVER-IP}:$port  (sign in with the dashboard password)
EOF
  fi
  if [ -n "$generated" ]; then
    echo "  Your dashboard password is: $generated  (it's also in $ENV_FILE)"
  fi
  if [ -n "$DOMAIN" ]; then
    cat <<EOF
  If the page doesn't load: check that $DOMAIN's DNS A record points at ${address:-this server} and
  allow ports 80 and 443 in your hosting provider's firewall. Caddy keeps retrying the certificate.
EOF
  else
    cat <<EOF
  If the page doesn't load, allow port $port in your hosting provider's firewall.
  For HTTPS and the Telegram app button, point a domain at this server and run:
    DOMAIN=bets.example.com bash setup_vps.sh
EOF
  fi
  cat <<EOF

Useful commands:
  journalctl -u $SERVICE -f        bot logs (Ctrl+C stops watching; the bot keeps running)
  journalctl -u $DASHBOARD -f      dashboard logs
  systemctl restart $SERVICE $DASHBOARD    restart after editing settings
  nano $ENV_FILE      change keys and passwords
  bash setup_vps.sh                  update to the latest code from GitHub
EOF
  if [ -n "$DOMAIN" ]; then echo "  journalctl -u caddy -f           HTTPS (Caddy) logs"; fi
}

ufw_active() {
  command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "^Status: active"
}

firewalld_active() {
  command -v firewall-cmd >/dev/null && firewall-cmd --state >/dev/null 2>&1
}

# check_web_ports: stop, before anything is changed, if a program other than Caddy listens on port
# 80 or 443 (e.g. nginx or Apache on a shared server): Caddy couldn't start or get a certificate
check_web_ports() {
  command -v ss >/dev/null || return 0
  local others names
  others="$(ss -Hltnp '( sport = :80 or sport = :443 )' 2>/dev/null | grep -v '"caddy"' || true)"
  [ -z "$others" ] && return 0
  names="$(printf '%s\n' "$others" | grep -o '(("[^"]*"' | cut -d'"' -f2 | sort -u | paste -sd, - | sed 's/,/, /g' || true)"
  echo "  HTTPS needs ports 80 and 443 for Caddy, but they are in use by ${names:-another program}."
  echo "  Nothing was changed. Free those ports (or run this without DOMAIN), then run this again."
  exit 1
}

# install_caddy: Caddy from its official packages (https://caddyserver.com/docs/install)
install_caddy() {
  if command -v caddy >/dev/null; then
    echo "  Caddy is already installed."
    return
  fi
  if ! install_caddy_package; then
    echo "  Couldn't install Caddy (for HTTPS); nothing else was changed."
    echo "  Install it by hand (https://caddyserver.com/docs/install), then run this again."
    exit 1
  fi
  echo "  Installed Caddy."
}

# install_caddy_package: the apt repository on Ubuntu/Debian, COPR on Fedora/RHEL. Each step is
# chained with && because "set -e" is off inside a function called as a condition
install_caddy_package() {
  if command -v apt-get >/dev/null; then
    apt-get install -y -qq --no-upgrade debian-keyring debian-archive-keyring apt-transport-https curl gnupg >/dev/null \
      && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg \
      && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        > /etc/apt/sources.list.d/caddy-stable.list \
      && chmod o+r /usr/share/keyrings/caddy-stable-archive-keyring.gpg /etc/apt/sources.list.d/caddy-stable.list \
      && apt-get update -qq \
      && apt-get install -y -qq caddy >/dev/null
  elif command -v dnf >/dev/null; then
    dnf install -y -q 'dnf-command(copr)' && dnf copr enable -y -q @caddy/caddy && dnf install -y -q caddy
  elif command -v yum >/dev/null; then
    yum install -y -q yum-plugin-copr && yum copr enable -y -q @caddy/caddy && yum install -y -q caddy
  else
    return 1
  fi
}

# stock_caddyfile FILE: true if FILE is the example page Caddy's package ships (":80", static files)
stock_caddyfile() {
  [ "$(grep -vE '^[[:space:]]*(#|$)' "$1" | tr -d '[:space:]')" = ':80{root*/usr/share/caddyfile_server}' ]
}

# setup_caddy PORT: serve DOMAIN over HTTPS, passing requests to the dashboard on 127.0.0.1:PORT
setup_caddy() {
  local port="$1"
  # Caddy's packages make this folder; a Caddy installed by hand may not have it
  mkdir -p "$(dirname "$CADDY_SITE")"
  cat > "$CADDY_SITE" <<CADDY
# Written by setup_vps.sh: the betting dashboard over HTTPS. Caddy gets and renews the certificate.
$DOMAIN {
	# Compress responses here, off the app's event loop (the app compresses only requests that reach it
	# directly; the slate's ready-made gzip passes through as it is)
	encode zstd gzip
	# A pasted bet slip's screenshot (POST /api/chat/slip) may be up to 2.5 MB; every other request is a few
	# kilobytes. The app refuses bigger ones too, path by path.
	@slip path /api/chat/slip
	handle @slip {
		request_body {
			max_size 2500KB
		}
		reverse_proxy 127.0.0.1:$port
	}
	handle {
		request_body {
			max_size 64KB
		}
		reverse_proxy 127.0.0.1:$port
	}
}
CADDY
  if [ ! -f "$CADDYFILE" ] || stock_caddyfile "$CADDYFILE"; then
    if [ -f "$CADDYFILE" ]; then cp "$CADDYFILE" "$CADDYFILE.orig"; fi
    echo "import $CADDY_SITE" > "$CADDYFILE"
  elif ! grep -qxF "import $CADDY_SITE" "$CADDYFILE"; then
    # Caddy already serves other sites from this server: add ours next to them
    printf '\nimport %s\n' "$CADDY_SITE" >> "$CADDYFILE"
    echo "  Added the dashboard to your existing $CADDYFILE."
  fi
  if ! caddy validate --config "$CADDYFILE" --adapter caddyfile >/dev/null 2>&1; then
    echo "  Caddy doesn't accept $CADDYFILE:"
    # "|| true": with pipefail, caddy's own failure would stop the script before the clean-up below
    caddy validate --config "$CADDYFILE" --adapter caddyfile 2>&1 | tail -5 || true
    # Take our site back out, so the next Caddy restart doesn't fail on it
    remove_caddy_site >/dev/null
    echo "  Fix that and run this again, or go back to the plain address: DOMAIN=off bash setup_vps.sh"
    exit 1
  fi
  getent hosts "$DOMAIN" >/dev/null \
    || echo "  $DOMAIN doesn't resolve yet: add a DNS A record pointing at this server."

  # Caddy needs 80 (certificate checks, redirect to HTTPS) and 443; the dashboard's own port closes
  if ufw_active; then
    ufw allow 80/tcp comment "caddy http" >/dev/null
    ufw allow 443/tcp comment "caddy https" >/dev/null
    ufw delete allow "$port/tcp" >/dev/null 2>&1 || true
    echo "  Opened ports 80 and 443 and closed port $port in the ufw firewall."
  elif firewalld_active; then
    firewall-cmd --quiet --permanent --add-service=http --add-service=https
    firewall-cmd --quiet --permanent --remove-port="$port/tcp" >/dev/null 2>&1 || true
    firewall-cmd --quiet --reload
    echo "  Opened ports 80 and 443 and closed port $port in firewalld."
  fi
  # SELinux (Fedora/RHEL) lets web servers such as Caddy connect to web ports only, not the dashboard's
  if command -v getenforce >/dev/null && [ "$(getenforce 2>/dev/null)" = "Enforcing" ]; then
    setsebool -P httpd_can_network_connect 1 || echo "  Couldn't let Caddy through SELinux (setsebool failed)."
  fi
  systemctl enable --quiet caddy
  if ! systemctl reload-or-restart caddy; then
    echo "  Caddy didn't start (is another web server using ports 80 or 443?). Last log lines:"
    journalctl -u caddy -n 20 --no-pager
    echo "  To go back to the plain address: DOMAIN=off bash setup_vps.sh"
    exit 1
  fi
  echo "  Caddy serves https://$DOMAIN"
}

# remove_caddy_site: stop serving the dashboard through Caddy. Caddy and any other sites it serves stay
remove_caddy_site() {
  [ -f "$CADDY_SITE" ] || return 0
  if [ -f "$CADDYFILE" ]; then
    sed -i "\|^import $CADDY_SITE\$|d" "$CADDYFILE"
    # Only our site was there: put back the example page Caddy came with
    if [ -f "$CADDYFILE.orig" ] && ! grep -qvE '^[[:space:]]*$' "$CADDYFILE"; then
      mv "$CADDYFILE.orig" "$CADDYFILE"
    fi
  fi
  rm -f "$CADDY_SITE"
  if systemctl is-active --quiet caddy; then
    systemctl reload caddy || echo "  Caddy didn't reload; it keeps serving the dashboard until it restarts."
  fi
  echo "  Caddy no longer serves the dashboard."
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
