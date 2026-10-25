"""Builds the odds, ESPN and prediction services the same way for the bot and the dashboard."""

from __future__ import annotations

import os
import tempfile
from typing import Optional
from zoneinfo import ZoneInfo

from .ai.desk import AnalystDesk
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
    return DashboardService(odds, EspnClient(cache), ZoneInfo(config.timezone), store=store,
                            analysts=build_analysts(config, store))


def build_analysts(config: Config, store: Store) -> Optional[AnalystDesk]:
    """The AI analysts whose API keys are set (None when there are none)."""
    analysts = []
    if config.anthropic_api_key:
        from .ai.claude import ClaudeAnalyst  # the SDK is only needed with a key
        analysts.append(ClaudeAnalyst(config.anthropic_api_key, config.anthropic_model, config.ai_effort,
                                      config.ai_web_search))
    if config.deepseek_api_key:
        from .ai.deepseek import DeepSeekAnalyst
        analysts.append(DeepSeekAnalyst(config.deepseek_api_key, config.deepseek_model))
    if not analysts:
        return None
    return AnalystDesk(store, analysts, config.ai_daily_limit, config.ai_lookahead_hours)
