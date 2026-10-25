"""Run the web dashboard: python -m telegram_betting_bot_simple.dashboard"""

from __future__ import annotations

import logging
import sys
from zoneinfo import ZoneInfo

import uvicorn

from ..config import load_config
from ..data.cache import TTLCache
from ..data.espn_client import EspnClient
from ..data.odds_service import OddsService
from .app import create_app
from .service import DashboardService


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    if not config.dashboard_password:
        sys.exit("Set DASHBOARD_PASSWORD in the .env file to start the dashboard.")
    service = None
    if config.odds_api_key:
        cache = TTLCache()
        odds = OddsService(config.odds_api_key, cache, config.odds_cache_minutes, config.odds_regions)
        service = DashboardService(odds, EspnClient(cache), ZoneInfo(config.timezone))
    app = create_app(service, config.dashboard_password)
    uvicorn.run(app, host="0.0.0.0", port=config.dashboard_port, access_log=False)


if __name__ == "__main__":
    main()
