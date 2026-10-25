"""Builds the odds, ESPN and prediction services the same way for the bot and the dashboard."""

from __future__ import annotations

import os
import tempfile
from typing import Optional
from zoneinfo import ZoneInfo

from .config import Config
from .dashboard.service import DashboardService
from .data.cache import TTLCache
from .data.espn_client import EspnClient
from .data.odds_service import OddsService
from .data.store import Store


def build_service(config: Config) -> Optional[DashboardService]:
    """The shared games/predictions/slip service, or None without an Odds API key."""
    if not config.odds_api_key:
        return None
    cache = TTLCache()
    # On the server systemd provides the state folder; elsewhere (a laptop) a temp folder will do
    state_dir = config.state_dir or os.path.join(tempfile.gettempdir(), "sportsbet-ai")
    os.makedirs(state_dir, exist_ok=True)
    store = Store(os.path.join(state_dir, "betting.db"))
    odds = OddsService(
        config.odds_api_key, cache, config.odds_cache_minutes, config.odds_regions,
        store=store, reserve=config.odds_reserve,
    )
    return DashboardService(odds, EspnClient(cache), ZoneInfo(config.timezone), store=store)
