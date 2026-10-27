"""Run the web dashboard: python -m telegram_betting_bot_simple.dashboard"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Callable, Optional

import uvicorn

from ..config import load_config
from ..data.cache import stats_line
from ..services import build_service, build_slate
from .app import create_app
from .service import DashboardService

GRADE_EVERY_SECONDS = 600
ANALYZE_EVERY_SECONDS = 600
CACHE_LOG_EVERY_SECONDS = 3600


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


def log_cache_forever(every: float = CACHE_LOG_EVERY_SECONDS, rounds: Optional[int] = None,
                      sleep: Callable[[float], None] = time.sleep) -> None:
    """Log the shared cache's one-line summary (data/cache.py's stats_line: reads, hit shares, loads, failures,
    memory size) every hour; ``rounds`` stops it early (tests)."""
    done = 0
    while rounds is None or done < rounds:
        sleep(every)
        try:
            logging.info(stats_line())
        except Exception:
            logging.exception("Summarising the cache failed")
        done += 1


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    if not config.dashboard_password:
        sys.exit("Set DASHBOARD_PASSWORD in the .env file to start the dashboard.")
    service = build_service(config)
    slate = None
    if service is not None:
        threading.Thread(target=grade_forever, args=(service,), name="grader", daemon=True).start()
        threading.Thread(target=log_cache_forever, name="cache-summary", daemon=True).start()
        if service.analysts is not None:
            threading.Thread(target=analyze_forever, args=(service,), name="analysts", daemon=True).start()
        # Today's all-sports slate: the last one saved is served at once, then ESPN's scoreboards keep it
        # current (it never spends Odds API credits; see data/slate.py)
        slate = build_slate(config, service)
        if slate.load():
            logging.info("Serving the saved slate until the first refresh finishes")
        # The markets brain: the slate's own background worker refreshes Kalshi, Polymarket, ESPN's predictor and
        # the ratings (free; MARKETS_ENABLED=0 turns them off), and every game's consensus is logged before its
        # start and graded with the rest (grade_forever)
        logging.info("Markets brain: %s", "books, prediction markets and ratings" if slate.kalshi or slate.polymarket
                     or slate.ratings else "books and the model only (MARKETS_ENABLED is off)")
        slate.start()
    # The bot token checks Telegram Mini App sign-ins, and only the listed Telegram users get in.
    # Behind Caddy, DASHBOARD_HOST=127.0.0.1 keeps the plain-HTTP port off the internet; uvicorn
    # trusts Caddy's X-Forwarded-Proto from 127.0.0.1, so cookies are still marked Secure
    if config.mini_app_url and not config.telegram_user_ids:
        logging.warning("TELEGRAM_ALLOWED_IDS is empty, so nobody can open the app from Telegram yet")
    app = create_app(service, config.dashboard_password, bot_token=config.telegram_token,
                     telegram_ids=config.telegram_user_ids, slate=slate)
    uvicorn.run(app, host=config.dashboard_host, port=config.dashboard_port, access_log=False)


if __name__ == "__main__":
    main()
