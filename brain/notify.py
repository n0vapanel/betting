#!/usr/bin/env python3
"""Telegram notification bridge for the betting brain.

Usage:
    notify.py --sport MMA --msg "Updated: 80.2% acc, +40.3% ROI"
    notify.py --ledger "MMA/paper: 10 open, 50 settled, +5.2% ROI"
"""
import os, sys, argparse, json
from urllib.request import Request, urlopen
from urllib.error import URLError

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHANNEL_ID = os.environ.get("TELEGRAM_CHANNEL_ID", "").strip()


def send_message(text, disable_preview=True):
    """Send a message to the Telegram channel."""
    if not BOT_TOKEN or not CHANNEL_ID:
        print(f"[notify] TELEGRAM_BOT_TOKEN or CHANNEL_ID not set; skipping send")
        print(f"  >> {text[:100]}")
        return False
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHANNEL_ID,
        "text": text,
        "disable_web_page_preview": disable_preview,
        "parse_mode": "Markdown"
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                return True
            print(f"[notify] Telegram error: {result.get('description')}", file=sys.stderr)
            return False
    except URLError as e:
        print(f"[notify] Network error: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="")
    ap.add_argument("--msg", default="")
    ap.add_argument("--ledger", default="")
    ap.add_argument("--emoji", default="")
    a = ap.parse_args()
    
    if a.msg:
        text = f"{a.emoji or ''} {a.sport}: {a.msg}".strip()
    elif a.ledger:
        text = f"📊 {a.ledger}"
    else:
        text = "Test notification"
    
    success = send_message(text)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
