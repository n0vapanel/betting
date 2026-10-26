"""Run the web dashboard: python -m telegram_betting_bot_simple.dashboard"""

from __future__ import annotations

import logging
import sys
import threading
import time

import uvicorn

from ..config import load_config
from ..services import build_service
from .app import create_app
from .service import DashboardService

GRADE_EVERY_SECONDS = 600
ANALYZE_EVERY_SECONDS = 600


def grade_forever(service: DashboardService) -> None:
    """Score people's calls and the AI's predictions as games finish (ESPN only, no odds quota)."""
    while True:
        try:
            service.grade_due()
        except Exception:
            logging.exception("Grading finished games failed")
        time.sleep(GRADE_EVERY_SECONDS)


def analyze_forever(service: DashboardService) -> None:
    """Ask the AI analysts about NBA games starting in the next few hours (within their daily limits)."""
    while True:
        try:
            done = service.analysts.run_due(service)
            if done:
                logging.info("AI analysts made %d new forecasts", done)
        except Exception:
            logging.exception("The AI analysts' run failed")
        time.sleep(ANALYZE_EVERY_SECONDS)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    if not config.dashboard_password:
        sys.exit("Set DASHBOARD_PASSWORD in the .env file to start the dashboard.")
    service = build_service(config)
    if service is not None:
        threading.Thread(target=grade_forever, args=(service,), name="grader", daemon=True).start()
        if service.analysts is not None:
            threading.Thread(target=analyze_forever, args=(service,), name="analysts", daemon=True).start()
    # The bot token checks Telegram Mini App sign-ins, and only the listed Telegram users get in.
    # Behind Caddy, DASHBOARD_HOST=127.0.0.1 keeps the plain-HTTP port off the internet; uvicorn
    # trusts Caddy's X-Forwarded-Proto from 127.0.0.1, so cookies are still marked Secure
    if config.mini_app_url and not config.telegram_user_ids:
        logging.warning("TELEGRAM_ALLOWED_IDS is empty, so nobody can open the app from Telegram yet")
    app = create_app(service, config.dashboard_password, bot_token=config.telegram_token,
                     telegram_ids=config.telegram_user_ids)
    uvicorn.run(app, host=config.dashboard_host, port=config.dashboard_port, access_log=False)


if __name__ == "__main__":
    main()
