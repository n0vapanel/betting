"""Builds the odds, ESPN and prediction services the same way for the bot and the dashboard.

The per-person features (dashboard/lab.py: tickets, saved bets, plans, players and the AI chat) hang off
the games service as ``service.lab``, so the bot and the dashboard reach them the same way.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Optional
from zoneinfo import ZoneInfo

from .ai.desk import AnalystDesk
from .config import Config
from .dashboard.lab import DAILY_AI_CENTS, Lab
from .dashboard.service import AI_LEAGUES, DashboardService
from .data.cache import TTLCache
from .data.espn_client import EspnClient
from .data.espn_players import PlayerData
from .data.odds_service import OddsService, configure_books
from .data.slate import SlateBuilder
from .data.store import Store


def build_service(config: Config) -> Optional[DashboardService]:
    """The shared games/predictions/slip service (with its Lab as ``service.lab``), or None without an
    Odds API key."""
    if not config.odds_api_key:
        return None
    cache = TTLCache()
    # On the server systemd provides the state folder; elsewhere (a laptop) a temp folder will do
    state_dir = config.state_dir or os.path.join(tempfile.gettempdir(), "sportsbet-ai")
    os.makedirs(state_dir, exist_ok=True)
    store = Store(os.path.join(state_dir, "betting.db"))
    bookmakers = tuple(getattr(config, "bookmakers", ()) or ())
    # Which books anchor the consensus and which a person can be offered (best price, slips, tickets): the whole
    # process follows these (data/odds_service.py's BookRoles)
    configure_books(bookmakers or None, getattr(config, "sharp_book_keys", None) or None,
                    getattr(config, "price_book_keys", None) or None)
    odds = OddsService(
        config.odds_api_key, cache, config.odds_cache_minutes, config.odds_regions,
        store=store, reserve=config.odds_reserve, bookmakers=bookmakers,
    )
    espn = EspnClient(cache)
    service = DashboardService(odds, espn, ZoneInfo(config.timezone), store=store,
                               analysts=build_analysts(config, store))
    lab = Lab(service, store, PlayerData(espn), public_url=config.public_url,
              daily_cents=getattr(config, "ai_chat_daily_cents", DAILY_AI_CENTS))
    lab.assistant = build_chat_assistant(config, lab, store)
    return service


def build_slate(config: Config, service: DashboardService) -> SlateBuilder:
    """Today's all-sports slate over the service's odds, ESPN and store (the dashboard starts its thread), with
    the markets brain's free outside sources unless MARKETS_ENABLED is off. The service's game route reads the
    slate's "Markets say" (``service.slate``)."""
    polymarket = kalshi = ratings = None
    if getattr(config, "markets_enabled", True):
        try:
            from .data.markets import Kalshi, Polymarket, Ratings
            polymarket = Polymarket(service.cache, service.store)
            kalshi = Kalshi(service.cache, service.store)
            ratings = Ratings(service.espn, service.cache, service.store)
        except Exception:  # the app works without them: books and the model still make a blend
            logging.exception("The prediction markets and ratings couldn't start; the markets brain uses the books only")
            polymarket = kalshi = ratings = None
    slate = SlateBuilder(
        service.odds, service.espn, service.store, service=service, tz=service.tz,
        live_seconds=config.slate_live_seconds, today_seconds=config.slate_today_seconds,
        idle_seconds=config.slate_idle_seconds, future_seconds=config.slate_future_seconds,
        odds_per_day=config.slate_odds_per_day, ai_leagues=AI_LEAGUES,
        polymarket=polymarket, kalshi=kalshi, ratings=ratings,
    )
    service.slate = slate
    return slate


def build_chat_assistant(config: Config, lab: Lab, store: Store) -> Optional[Any]:
    """The AI chat assistant, or None without ANTHROPIC_API_KEY (or if it can't start: the app works without it)."""
    if not config.anthropic_api_key:
        return None
    try:
        from .ai.assistant import build_assistant  # the SDK is only needed with a key
    except ImportError as exc:
        logging.warning("The AI chat assistant isn't available (%s); the app runs without it", exc)
        return None
    try:
        return build_assistant(config, lab, store)
    except Exception:
        logging.exception("The AI chat assistant couldn't start; the app runs without it")
        return None


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
