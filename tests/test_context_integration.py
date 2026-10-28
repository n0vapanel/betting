"""Stage 4 integration: the parts that only meet when everything runs together, over the real modules (the venue
table, the Open-Meteo and MLB StatsAPI clients with only their HTTP faked, data/context.py's ContextService, the real
SlateBuilder, TTLCache and Store, DashboardService, Lab and the Builder):

- a forecast that changes after the cards were built reaches every place the brain's chance of the totals is used
  at the next context round (the game page's card, the league's cards, odds_get_prices, the slip and the Builder),
  not a cache lifetime later, so the Builder and the slip never disagree about the same leg;
- the bot's process (another TTLCache on the same store, no worker of its own) follows the blocks the dashboard's
  worker writes, rather than the first copy it read;
- an unknown venue is placed at its city in the country ESPN names (not the first place of that name anywhere);
- services.build_context hands DashboardService a ContextService that tells it of every changed adjustment.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from telegram_betting_bot_simple.ai.tools import Toolbox
from telegram_betting_bot_simple.dashboard.service import DashboardService
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.context import READ_AGAIN, ContextService, total_adjustment
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.data.weather import WeatherClient
from telegram_betting_bot_simple.engine import context as engine
from tests.test_context_wiring import (AFTER, BEFORE, NFL, NOW, NY, UTC, VENUES, Response, World, default_roles,
                                       no_network)

AUTOUSE = (default_roles, no_network)  # the wiring tests' autouse fixtures (no network, the default book roles): here too
OVER = f"{NFL}:nfl1:totals:over:41.5"


def calm_world(tmp_path) -> World:
    """The wiring tests' world with the odds fetched between two calm forecasts: no adjustment yet, every card and
    the Builder's legs built and held."""
    w = World(tmp_path)
    w.round()  # T0: the forecast before the odds were fetched (ODDS_AT)
    w.clock["t"] = NOW
    w.cache.invalidate_prefix(("weather",))  # the hour is up: the next round reads the forecast again
    w.round()  # NOW: the same forecast after the odds: nothing moves
    assert w.context.for_game(NFL, "nfl1")["adjustments"] == []
    return w


def change_forecast(w: World, seconds: float = 5.0) -> None:
    """Seconds later the forecast for Soldier Field picks up (its copy refreshed): the next round adjusts."""
    w.clock["t"] += seconds
    w.meteo.conditions = dict(AFTER)
    w.cache.invalidate_prefix(("weather",))
    again(w)


def again(w: World) -> None:
    """A context round now (the worker's next one, without waiting out its minute)."""
    w.slate._context_next = 0.0
    w.round()


def over_card(w: World) -> Dict[str, Any]:
    card = w.service.game(NFL, "nfl1")
    return card["markets"]["totals"]["over"]


def test_a_forecast_change_reaches_every_chance_at_the_next_round(tmp_path) -> None:
    w = calm_world(tmp_path)
    uid = "tg:5150"
    w.store.get_or_create_user(uid, now=NOW)
    w.store.confirm_age(uid, now=NOW)
    slip = {"type": "single", "stake": "10", "legs": [{"league": NFL, "id": "nfl1", "market": "totals",
                                                        "side": "over"}]}
    toolbox = Toolbox(w.lab, uid, NY, now=lambda: datetime.datetime.fromtimestamp(w.clock["t"], UTC))

    # Built and held while nothing moves: the books' chance everywhere
    assert over_card(w)["ai"] == 0.5 and "adjusted" not in over_card(w)
    league_card = next(c for c in w.service.games(NFL)["games"] if c["id"] == "nfl1")
    assert league_card["markets"]["totals"]["over"]["ai"] == 0.5
    assert w.service.builder_candidates(now=w.clock["t"])["by_key"][OVER].chance == 0.5
    held = w.lab.price(uid, slip)["legs"][0]
    assert held["ai_prob"] is None and held["market_prob"] == 0.5 and "adjusted" not in held  # the market's chance

    change_forecast(w)  # 5 seconds on: every one of those entries is still within its time to live
    moved = total_adjustment(w.context.for_game(NFL, "nfl1"))
    assert moved is not None and moved["applied"] == -2.0
    after = engine.adjusted_over("nfl", 41.5, 0.5, moved["applied"])

    over = over_card(w)
    assert over["ai"] == pytest.approx(after, abs=1e-12) and over["adjusted"]["applied"] == -2.0
    league_card = next(c for c in w.service.games(NFL)["games"] if c["id"] == "nfl1")
    assert league_card["markets"]["totals"]["over"]["ai"] == pytest.approx(after, abs=1e-12)
    leg = w.service.builder_candidates(now=w.clock["t"])["by_key"][OVER]
    assert leg.chance == pytest.approx(after, abs=1e-12) and leg.adjusted["applied"] == -2.0
    priced = w.lab.price(uid, slip)["legs"][0]
    assert priced["ai_prob"] == pytest.approx(after, abs=1e-4) and priced["adjusted"]["applied"] == -2.0
    run = toolbox.run("odds_get_prices", {"league": NFL, "event_id": "nfl1"})
    assert run.ok, run.content
    offered = next(o for o in run.content["outcomes"] if o["ref"]["market"] == "totals" and o["ref"]["side"] == "over")
    assert offered["ai_chance"] == round(after, 4)
    # The Builder's shared results follow too: a run now prices the over with the adjusted chance
    client = w.client()
    result = client.post("/api/builder/run", json={"controls": {"preset": "risky", "leagues": [NFL],
                                                                "markets": ["totals"]}}).json()
    legs = [leg for ticket in result.get("tickets") or [] for leg in ticket["legs"] if leg["ref"]["id"] == "nfl1"]
    assert all(leg["adjusted"]["applied"] == -2.0 for leg in legs)

    # The forecast settles back: the adjustment goes, and the chances return to the books' at once
    w.clock["t"] += 5
    w.meteo.conditions = dict(BEFORE)
    w.cache.invalidate_prefix(("weather",))
    again(w)
    assert w.context.for_game(NFL, "nfl1")["adjustments"] == []
    assert over_card(w)["ai"] == 0.5 and "adjusted" not in over_card(w)
    assert w.service.builder_candidates(now=w.clock["t"])["by_key"][OVER].chance == 0.5
    back = w.lab.price(uid, slip)["legs"][0]
    assert back["ai_prob"] is None and "adjusted" not in back


def test_the_service_hears_of_each_changed_adjustment_once(tmp_path) -> None:
    w = calm_world(tmp_path)
    heard: List[Any] = []
    real = w.service.adjustment_changed
    w.context.on_adjust = lambda league, game_id: heard.append((league, game_id)) or real(league, game_id)
    mark = w.service.context_mark()
    change_forecast(w)
    # Soldier Field's wind and the Emirates' warmth changed: each game is passed on once
    assert sorted(heard) == [("americanfootball_nfl", "nfl1"), ("soccer_epl", "epl1")]
    assert w.service.context_mark() != mark
    w.clock["t"] += 5
    again(w)  # the same adjustments again: nothing to pass on
    assert len(heard) == 2


def test_the_bot_process_follows_the_dashboards_blocks(tmp_path) -> None:
    w = calm_world(tmp_path)
    # The bot's process: its own cache on the same store, a ContextService without a worker or a weather client
    bot_cache = TTLCache(store=w.store, clock=lambda: w.clock["t"], name="bot")
    bot = ContextService(VENUES, None, None, w.store, bot_cache, now=lambda: w.clock["t"])
    first = bot.for_game(NFL, "nfl1")
    assert first["status"] == "ready" and first["adjustments"] == []

    change_forecast(w)  # the dashboard's worker writes the adjusted block
    # Within READ_AGAIN the bot keeps its copy (one table read a minute at most); past it, it reads the new one
    assert bot.for_game(NFL, "nfl1")["adjustments"] == []
    w.clock["t"] += READ_AGAIN + 1
    seen = bot.for_game(NFL, "nfl1")
    assert seen["adjustments"] and seen["adjustments"][0]["applied"] == -2.0
    assert total_adjustment(bot.held(NFL, "nfl1"))["applied"] == -2.0

    # The bot's game card prices the totals with it too (its DashboardService over the bot's cache)
    bot_service = DashboardService(w.odds, w.service.espn, NY, store=w.store, cache=bot_cache)
    bot_service.context = bot
    over = bot_service.game(NFL, "nfl1")["markets"]["totals"]["over"]
    assert over["adjusted"]["applied"] == -2.0
    assert over["ai"] == pytest.approx(engine.adjusted_over("nfl", 41.5, 0.5, -2.0), abs=1e-12)

    # The dashboard's own reads never go back to the table: its last block is the newest
    reads: List[Any] = []
    real_peek = w.cache.peek
    w.cache.peek = lambda *a, **k: reads.append(a) or real_peek(*a, **k)  # type: ignore[method-assign]
    assert w.context.for_game(NFL, "nfl1")["adjustments"][0]["applied"] == -2.0
    assert reads == []


def test_peek_rechecks_the_shared_table_only_past_its_age(tmp_path) -> None:
    store = Store(str(tmp_path / "s.db"))
    clock = {"t": 1_000_000.0}
    writer = TTLCache(store=store, clock=lambda: clock["t"], name="writer")
    reader = TTLCache(store=store, clock=lambda: clock["t"], name="reader")
    key = ("game_context", NFL, "x1")
    writer.put(key, {"v": 1}, ttl=900, stale=900, kind="game_context")
    assert reader.peek(key)[1] == {"v": 1}
    clock["t"] += 10
    writer.put(key, {"v": 2}, ttl=900, stale=900, kind="game_context")
    assert reader.peek(key)[1] == {"v": 1}  # no recheck asked: the memory copy
    assert reader.peek(key, recheck=60)[1] == {"v": 1}  # its copy is 10 s old
    clock["t"] += 61
    assert reader.peek(key, recheck=60)[1] == {"v": 2}  # past 60 s: the table's newer copy
    assert reader.peek(key, shared=False, recheck=60)[1] == {"v": 2}
    writer.invalidate(key)  # dropped there (a status change): the reader keeps what it has
    clock["t"] += 61
    assert reader.peek(key, recheck=60)[1] == {"v": 2}


class Geocoder:
    """Open-Meteo's geocoder answering "Reading" with the Pennsylvania town first and the English one second."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append({"url": url, **dict(params or {})})
        if "geocoding" not in url:
            raise AssertionError("only the geocoder is expected here")
        return Response({"results": [
            {"name": "Reading", "latitude": 40.3357, "longitude": -75.9269, "elevation": 85.0, "country_code": "US",
             "admin1": "Pennsylvania", "timezone": "America/New_York", "population": 95112},
            {"name": "Reading", "latitude": 51.4562, "longitude": -0.9711, "elevation": 61.0, "country_code": "GB",
             "admin1": "England", "timezone": "Europe/London", "population": 244070}]})


def test_an_unknown_venue_is_placed_in_the_country_espn_names(tmp_path) -> None:
    store = Store(str(tmp_path / "g.db"))
    cache = TTLCache(store=store, clock=lambda: NOW, name="geo")
    geocoder = Geocoder()
    ctx = ContextService(VENUES, WeatherClient(cache, store, session=geocoder), None, store, cache,
                         now=lambda: NOW)
    game = {"league": "soccer_efl_champ", "odds_id": "ch9", "venue": {"name": "Select Car Leasing Stadium",
                                                                       "city": "Reading", "indoor": False},
            "_place": {"id": "99999", "country": "England"}, "_ids": ("99", "98")}
    row = ctx._venue(game, rich=True)
    assert row is not None and row["approx"] is True and row["country"] == "GB"
    assert row["lat"] == pytest.approx(51.4562) and row["tz"] == "Europe/London" and row["roof"] == "open"
    assert len(geocoder.calls) == 1 and geocoder.calls[0]["name"] == "Reading"


def test_build_context_tells_the_service_of_changed_adjustments(tmp_path, monkeypatch) -> None:
    from telegram_betting_bot_simple import services
    store = Store(str(tmp_path / "b.db"))
    cache = TTLCache(store=store, name="svc")
    config = SimpleNamespace(weather_enabled=True, weather_daily_cap=250)
    context = services.build_context(config, store, cache, espn=SimpleNamespace())
    assert isinstance(context, ContextService) and context.weather.daily_cap == 250
    service = DashboardService(SimpleNamespace(), SimpleNamespace(), NY, store=store, cache=cache)
    assert service.context is None and service.context_mark() == ""
    service.context = context
    assert context.on_adjust == service.adjustment_changed
    # A service handed one already wired elsewhere keeps that wiring
    other = DashboardService(SimpleNamespace(), SimpleNamespace(), NY, store=store, cache=cache)
    other.context = context
    assert context.on_adjust == service.adjustment_changed
    # Off: no forecasts, the venue, rest and travel stay
    off = services.build_context(SimpleNamespace(weather_enabled=False, weather_daily_cap=1000), store, cache,
                                 espn=SimpleNamespace())
    assert off.weather is None


def test_a_game_asked_for_brings_the_round_forward(tmp_path) -> None:
    """The game page re-checks a "loading" block once, 20 seconds on: the worker's next round must not be the
    regular one a minute (or, on an idle slate, ten minutes) away."""
    w = World(tmp_path)
    w.context.window = 3600  # only games within the hour are warmed by themselves
    w.round()
    w.slate._wake.clear()
    w.slate._context_next = w.clock["t"] + 3600  # the regular round is far off
    assert w.context.for_game(NFL, "nfl1") == {"status": "loading"}
    assert w.slate._context_next <= w.clock["t"] and w.slate._wake.is_set()  # a slate round now, with a context round
    w.slate._wake.clear()
    # Another game asked for within CONTEXT_SOON: queued, no second wake-up (one round takes both)
    assert w.context.for_game(NFL, "nfl2") == {"status": "loading"}
    assert not w.slate._wake.is_set() and w.context.pending()
    w.clock["t"] += 2  # the same moment, give or take: the clock hasn't reached the next regular round
    w.slate.refresh()
    w.slate.wait_context(20)
    assert w.context.for_game(NFL, "nfl1")["status"] == "ready" and w.context.for_game(NFL, "nfl2")["status"] == "ready"
    assert not w.context.pending()
    # A game queued by a status change (invalidate) waits for the regular round: no wake-up, nothing pending
    w.slate._context_next = w.clock["t"] + 3600
    w.slate._wake.clear()
    w.context.invalidate(NFL, "nfl1")
    assert not w.slate._wake.is_set() and not w.context.pending()
    # The bot's process (no slate): asking queues it and nothing else
    bot = ContextService(VENUES, None, None, w.store, TTLCache(), now=lambda: w.clock["t"])
    assert bot.for_game(NFL, "nfl9") == {"status": "loading"} and bot.pending()


def test_a_game_asked_for_during_a_round_gets_the_next_one_at_once(tmp_path) -> None:
    w = World(tmp_path)
    w.round()
    real = w.context.warm

    def warm(games):
        out = real(games)
        w.context.queue(NFL, "late", soon=True)  # asked while the round ran
        return out
    w.context.warm = warm  # type: ignore[method-assign]
    w.slate._wake.clear()
    w.slate._context_next = 0.0
    w.slate.refresh()
    w.slate.wait_context(20)
    assert w.slate._context_next <= w.clock["t"] and w.slate._wake.is_set()
