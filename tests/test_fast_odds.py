"""Fast odds: the prices are there before anyone asks (scratchpad/build/SPEC_FAST_ODDS.md).

- data/slate.py preloads the odds boards of leagues with a game soon, through request_odds' one loader, within a daily
  budget: the month's credits above the reserve spread over the days left (preloads may use 70% of a day's boards,
  requests all of them), counted per UTC day in the shared store.
- dashboard/lab.py's _ensure_prices: the Builder, Ask AI's ticket build and the top picks wait a few seconds for the
  boards they need that are on their way, instead of saying "ask again".
- A league still without prices is named honestly: only the games of the ask's own window ("tonight's 7 games"), one
  of four reasons, never "ask again in a few minutes".

Fakes and real Stores in temporary folders: no network (requests' Session.request fails loudly). The Odds API,
Anthropic and DeepSeek are never called: a board "loads" by saving a snapshot (tests/test_slate.py's FakeOdds) and
Claude is a scripted fake.
"""

from __future__ import annotations

import contextlib
import datetime
import html
import json
import logging
import sqlite3
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import requests

from telegram_betting_bot_simple.ai.assistant import Assistant
from telegram_betting_bot_simple.ai.explain import ASK_AGAIN_PROBLEM, asks_again, builder_template, picks_template
from telegram_betting_bot_simple.ai.tools import Toolbox
from telegram_betting_bot_simple.dashboard import lab as lab_module
from telegram_betting_bot_simple.dashboard.lab import PRICE_WAIT, Lab, no_prices_line, prices_note
from telegram_betting_bot_simple.dashboard.service import DashboardService
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data import odds_service
from telegram_betting_bot_simple.data.espn_client import ESPN_DAY_TZ
from telegram_betting_bot_simple.data.odds_service import (DIRECT, OddsApiError, OddsService, boards_per_day, budget_day,
                                                           configure_books, day_boards, days_left)
from telegram_betting_bot_simple.data.slate import ODDS_PER_DAY, PRELOAD_HOURS, PRELOAD_SHARE, SlateBuilder
from telegram_betting_bot_simple.data.store import CLAIM_SECONDS, Store
from tests.test_best_effort_wiring import bot_turn, reply, say
from tests.test_builder_wiring import (EPL, NBA, NHL, NOW, NY, PLAYER_A, TODAY, NoEspn, QuietEspn, StoreOdds, World,
                                       epl_board, espn_boards, in_2030, make_world, nba_board, run)
from tests.test_slate import FakeEspn, FakeOdds, PayingOdds, at, espn_game, hold_the_odds_pool, odds_event, wait_for

LIGA, SERIE, BUNDES, ODI = "soccer_spain_la_liga", "soccer_italy_serie_a", "soccer_germany_bundesliga", "cricket_odi"
LISTING = [{"key": key, "group": group, "title": title, "active": True, "has_outrights": False}
           for key, group, title in ((NBA, "Basketball", "NBA"), (NHL, "Ice Hockey", "NHL"), (EPL, "Soccer", "EPL"),
                                     (LIGA, "Soccer", "La Liga"), (SERIE, "Soccer", "Serie A"),
                                     (BUNDES, "Soccer", "Bundesliga"), (ODI, "Cricket", "One Day Internationals"))]
DAY = "2026-10-26"  # NOW's UTC day (6 pm in New York is 10 pm UTC)
UID = "web:" + PLAYER_A["X-Player"]
SLATE_LOG = "telegram_betting_bot_simple.data.slate"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(requests.Session, "request", refuse)


@pytest.fixture(autouse=True)
def default_roles():
    configure_books()
    yield
    configure_books()


@pytest.fixture()
def store(tmp_path):
    return Store(str(tmp_path / "betting.db"))


def utc(*parts: int) -> float:
    return datetime.datetime(*parts, tzinfo=datetime.timezone.utc).timestamp()


def preloader(store, boards, odds=None, now=lambda: NOW) -> "tuple[SlateBuilder, FakeOdds]":
    """A slate that preloads, over fake ESPN boards and a fake odds service whose boards "load" by being saved."""
    odds = odds if odds is not None else FakeOdds(store, listing=LISTING, now=now)
    return SlateBuilder(odds, FakeEspn(boards), store, now=now, workers=2, preload=True), odds


def preload_lines(caplog) -> List[str]:
    return [record.getMessage() for record in caplog.records if "preloading" in record.getMessage()]


# --- the preloader: which leagues, in which order -----------------------------------------------------------------

def test_the_preloader_loads_leagues_with_a_game_soon_the_soonest_first(store, caplog) -> None:
    boards = {
        (NBA, TODAY): [espn_game("n1", "Boston Celtics", "New York Knicks", at(3)),
                       espn_game("n2", "Miami Heat", "Orlando Magic", at(5), ids=("3", "4"))],
        (NHL, TODAY): [espn_game("h1", "Boston Bruins", "Toronto Maple Leafs", at(3), sport="nhl")],
        (EPL, TODAY): [espn_game("s1", "Arsenal", "Everton", at(1), sport="soccer")],
        (LIGA, TODAY): [espn_game("l1", "Real Madrid", "Getafe", at(17.5), sport="soccer")],
        (SERIE, TODAY): [espn_game("i1", "Inter", "Roma", at(PRELOAD_HOURS + 0.5), sport="soccer")],  # too far off
        # ESPN shows a WNBA game, but The Odds API has no WNBA market now: out of season
        ("basketball_wnba", TODAY): [espn_game("w1", "Las Vegas Aces", "Seattle Storm", at(2), sport="wnba")],
    }  # Bundesliga: in season, and ESPN shows no game
    # The ODIs are odds-only (ESPN doesn't cover them): an old board's game is no game ESPN shows
    store.save_odds_snapshot(ODI, [odds_event(ODI, "c1", "India", "Australia", at(2))], NOW - 3 * 3600)
    builder, odds = preloader(store, boards)
    odds.api_key = "odds-key-never-logged"
    with caplog.at_level(logging.INFO, logger=SLATE_LOG):
        builder.refresh()  # a round queues at most PRELOAD_BACKLOG (2) preloads; the next tops it up
        wait_for(builder)
        assert odds.fetches == [EPL, NBA]
        builder.refresh()
        wait_for(builder)
    # The soonest first game first (EPL at +1h); NBA and the NHL both start at +3h, and the NBA has two games then
    assert odds.fetches == [EPL, NBA, NHL, LIGA]
    lines = preload_lines(caplog)
    assert len(lines) == 4 and all("(3 credits; 300 left this month" in line for line in lines)
    assert [line.split()[2] for line in lines] == [EPL, NBA, NHL, LIGA]
    assert "odds-key-never-logged" not in caplog.text
    # The next round: every board is inside the odds' 2 hours, so nothing loads again
    builder.refresh()
    wait_for(builder)
    assert odds.fetches == [EPL, NBA, NHL, LIGA]


def test_a_preload_needs_stale_saved_odds_and_a_stale_board_is_loaded_again(store) -> None:
    clock = {"now": NOW}
    boards = {(NBA, TODAY): [espn_game("n1", "Boston Celtics", "New York Knicks", at(3))],
              (NHL, TODAY): [espn_game("h1", "Boston Bruins", "Toronto Maple Leafs", at(4), sport="nhl")]}
    store.save_odds_snapshot(NBA, [], NOW - 3600)  # an hour old: fresh, so only the NHL loads
    builder, odds = preloader(store, boards, now=lambda: clock["now"])
    builder.refresh()
    wait_for(builder)
    assert odds.fetches == [NHL]
    clock["now"] += 3600 + 60  # the NBA's board is now over 2 hours old, the NHL's an hour
    builder.refresh()
    wait_for(builder)
    assert odds.fetches == [NHL, NBA]


def test_no_preload_without_the_season_listing_or_with_preload_off(store) -> None:
    boards = {(NBA, TODAY): [espn_game("n1", "Boston Celtics", "New York Knicks", at(3))]}
    off = SlateBuilder(FakeOdds(store, listing=LISTING), FakeEspn(boards), store, now=lambda: NOW, workers=2)
    off.refresh()  # SlateBuilder's default: no preloads (services.build_slate turns them on: SLATE_PRELOAD)
    wait_for(off)
    assert off.odds.fetches == []

    class NoListing(FakeOdds):
        def sports(self):
            raise RuntimeError("the listing is down")

    builder, odds = preloader(store, boards, odds=NoListing(store))
    builder.refresh()
    wait_for(builder)
    assert odds.fetches == []  # nothing is known to be in season: nothing is preloaded


def test_services_turn_the_preloader_on_unless_slate_preload_is_off(tmp_path) -> None:
    from telegram_betting_bot_simple import services
    store = Store(str(tmp_path / "betting.db"))
    service = DashboardService(StoreOdds(store), NoEspn(), NY, store=store, cache=TTLCache())
    config = SimpleNamespace(markets_enabled=False, slate_live_seconds=20, slate_today_seconds=60,
                             slate_idle_seconds=600, slate_future_seconds=3600, slate_odds_per_day=12)
    assert services.build_slate(config, service).preload is True
    assert services.build_slate(SimpleNamespace(**vars(config), slate_preload=False), service).preload is False
    from telegram_betting_bot_simple.config import Config
    assert Config(TELEGRAM_BOT_TOKEN="123456:test").slate_preload is True
    assert Config(TELEGRAM_BOT_TOKEN="123456:test", SLATE_PRELOAD="0").slate_preload is False


def test_a_board_is_never_fetched_twice_across_processes_or_inside_the_two_hours(tmp_path) -> None:
    path = str(tmp_path / "betting.db")
    clock = {"now": NOW}
    boards = {(NBA, TODAY): [espn_game("n1", "Boston Celtics", "New York Knicks", at(3))]}
    dashboard, other = Store(path), Store(path)
    other.holder = "another-process"  # the second process on the same file (a deploy's overlap, say)
    first, odds = preloader(dashboard, boards, now=lambda: clock["now"])
    second, other_odds = preloader(other, boards, now=lambda: clock["now"])
    gate = hold_the_odds_pool(first)  # the first process's board is on its way ...
    first.refresh()
    second.refresh()  # ... when the second's round looks: the lease is taken
    assert first.request_odds(NBA)["code"] == "loading"  # a request rides on the board on its way
    assert second.request_odds(NBA)["code"] == "throttled"
    gate.set()
    wait_for(first)
    wait_for(second)
    assert odds.fetches == [NBA] and other_odds.fetches == []
    assert dashboard.odds_boards(DAY) == {"preload": 1}  # the second's refused board was given back
    for builder in (first, second):
        builder.refresh()
        wait_for(builder)
    assert odds.fetches == [NBA] and other_odds.fetches == []  # inside the odds' 2 hours
    clock["now"] += 2 * 3600 + 60
    for builder in (first, second):
        builder.refresh()
        wait_for(builder)
    assert odds.fetches == [NBA, NBA] and other_odds.fetches == []  # stale: loaded again, once


# --- the daily budget -----------------------------------------------------------------------------------------------

def test_the_free_plan_and_the_30_dollar_plan_on_the_first_of_the_month() -> None:
    first = utc(2026, 10, 1, 12)
    assert days_left(first) == 31
    assert boards_per_day(500, 50, 3, 31) == 4  # the free plan: (500 - 50) / 31 / 3
    assert boards_per_day(20_000, 50, 3, 31) == 214  # the $30 plan: about 210
    for remaining, boards, preloads in ((500, 4, 2), (20_000, 214, 149)):
        builder = SlateBuilder(FakeOdds(None, remaining=remaining), FakeEspn(), None, now=lambda: first)
        budget = builder._budget(first)
        assert (budget["boards"], budget["preloads"], budget["spent"]) == (boards, preloads, 0)
        assert preloads == int(boards * PRELOAD_SHARE)
    # Regions cost more per board: two regions (6 credits) halve the boards
    builder = SlateBuilder(PayingOdds(None, remaining=20_000), FakeEspn(), None, now=lambda: first)
    builder.odds.regions = "us,uk"
    assert builder._budget(first)["boards"] == 107


def test_days_left_counts_utc_days_to_the_first_of_next_month() -> None:
    assert days_left(utc(2026, 10, 31, 0, 0)) == days_left(utc(2026, 10, 31, 23, 59)) == 1
    assert days_left(utc(2026, 11, 1, 0, 0)) == 30 and days_left(utc(2026, 12, 31, 12)) == 1
    assert days_left(utc(2027, 2, 1)) == 28 and days_left(utc(2028, 2, 1)) == 29
    # 11:30 pm in New York on 31 October is 1 November in UTC: a new month's 30 days
    assert days_left(datetime.datetime(2026, 10, 31, 23, 30, tzinfo=NY).timestamp()) == 30
    # On the month's last day the day may use every credit above the reserve, and never one below it
    assert boards_per_day(62, 50, 3, 1) == 4 and boards_per_day(50, 50, 3, 1) == 0 and boards_per_day(40, 50, 3, 1) == 0
    last = SlateBuilder(FakeOdds(None, remaining=62), FakeEspn(), None, now=lambda: utc(2026, 10, 31, 20))
    assert last._budget(utc(2026, 10, 31, 20))["boards"] == 4


def test_the_day_keeps_its_allowance_as_it_is_spent(store) -> None:
    # 20,000 credits on the 1st: 214 boards; after 100 are loaded (300 credits paid) the day still has 214, not 211
    first = utc(2026, 10, 1, 12)
    odds = FakeOdds(store, remaining=20_000 - 300)
    builder = SlateBuilder(odds, FakeEspn(), store, now=lambda: first)
    for _ in range(100):
        assert store.take_odds_board("2026-10-01", "preload", 999)
    assert builder._budget(first)["boards"] == 214


def test_preloads_use_70_percent_requests_the_rest_and_the_reserve_is_never_crossed(store, monkeypatch) -> None:
    monkeypatch.setattr(odds_service, "days_left", lambda now: 1)  # the month's last day: 12 credits over the reserve
    leagues = [EPL, LIGA, SERIE, BUNDES, NBA, NHL]
    sports = {NBA: "nba", NHL: "nhl"}
    boards = {(league, TODAY): [espn_game(f"g{n}", "Home Side", "Away Side", at(2 + n), ids=(f"{n}1", f"{n}2"),
                                          sport=sports.get(league, "soccer"))]
              for n, league in enumerate(leagues)}
    odds = PayingOdds(store, listing=LISTING, remaining=62)  # each board pays 3 credits off the reported count
    builder, _ = preloader(store, boards, odds=odds)
    builder.refresh()
    wait_for(builder)
    assert builder._budget(NOW)["boards"] == 4
    assert odds.fetches == [EPL, LIGA] and odds.quota["remaining"] == 56  # preloads: 70% of today's 4 boards
    builder.refresh()
    wait_for(builder)
    assert odds.fetches == [EPL, LIGA]  # the preloads' share is used up, though four leagues are still due
    gate = hold_the_odds_pool(builder)
    answers = [builder.request_odds(league)["code"] for league in (SERIE, BUNDES, NBA)]
    gate.set()
    wait_for(builder)
    # Requests may use the rest of the day; the third would leave less than the reserve (boards on their way count)
    assert answers == ["queued", "queued", "reserve"]
    assert odds.fetches == [EPL, LIGA, SERIE, BUNDES] and odds.quota["remaining"] == 50
    assert builder.request_odds(NHL)["code"] == "reserve"
    assert store.odds_boards(DAY) == {"preload": 2, "request": 2}


def test_with_the_count_unknown_a_day_has_the_old_12_boards(store) -> None:
    boards = {(NBA, TODAY): [espn_game("n1", "Boston Celtics", "New York Knicks", at(3))]}
    odds = FakeOdds(store, listing=LISTING, remaining=None)  # a new install: no request has reported a count yet
    builder = SlateBuilder(odds, FakeEspn(boards), store, now=lambda: NOW, workers=2)
    assert ODDS_PER_DAY == 12
    assert builder._budget(NOW)["boards"] == 12 and builder._budget(NOW)["preloads"] == 8
    for _ in range(12):
        assert store.take_odds_board(DAY, "request", 99)
    builder.refresh()
    assert builder.request_odds(NBA)["code"] == "daily_cap"


def test_the_day_s_count_is_shared_by_two_store_handles_and_a_restart(tmp_path) -> None:
    path = str(tmp_path / "betting.db")
    dashboard, bot = Store(path), Store(path)
    bot.holder = "bot"
    assert dashboard.take_odds_board(DAY, "preload", 3) and dashboard.take_odds_board(DAY, "request", 3)
    assert bot.odds_boards(DAY) == {"preload": 1, "request": 1}
    assert bot.take_odds_board(DAY, "request", 3)
    assert not dashboard.take_odds_board(DAY, "preload", 3)  # three counted, both kinds together: the limit
    bot.give_back_odds_board(DAY, "request")
    assert dashboard.odds_boards(DAY) == {"preload": 1, "request": 1} and dashboard.odds_boards("2026-10-27") == {}
    # Two handles taking at once never pass the limit together
    results: List[bool] = []
    handles = [Store(path) for _ in range(4)]

    def take(handle: Store) -> None:
        for _ in range(5):
            results.append(handle.take_odds_board("2026-10-27", "request", 7))
    threads = [threading.Thread(target=take, args=(handle,)) for handle in handles]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert results.count(True) == 7 and dashboard.odds_boards("2026-10-27") == {"request": 7}
    # A dashboard and the one that replaces it after a restart agree on what today spent: 1 board a day here
    boards = {(NBA, TODAY): [espn_game("n1", "Boston Celtics", "New York Knicks", at(3))],
              (NHL, TODAY): [espn_game("h1", "Boston Bruins", "Toronto Maple Leafs", at(4), sport="nhl")]}
    fresh = str(tmp_path / "fresh.db")
    before = SlateBuilder(FakeOdds(Store(fresh), listing=LISTING, remaining=70), FakeEspn(boards), Store(fresh),
                          now=lambda: NOW, workers=2)
    before.refresh()
    assert before.request_odds(NBA)["code"] == "queued"
    wait_for(before)
    restarted_store = Store(fresh)
    restarted_store.holder = "restarted"
    after = SlateBuilder(FakeOdds(restarted_store, listing=LISTING, remaining=70), FakeEspn(boards), restarted_store,
                         now=lambda: NOW, workers=2)
    after.refresh()
    assert after.request_odds(NHL)["code"] == "daily_cap"


# --- waiting for prices: the Builder, Ask AI's ticket build and the top picks ----------------------------------------

class SlowOdds(FakeOdds):
    """The loader's odds service: league_games "fetches" the league's board (tests/test_builder_wiring's NBA or EPL
    games) by saving it after ``delay`` seconds; with ``finishes`` False it never comes (held until the test ends)."""

    def __init__(self, store: Store, delay: float = 1.0, finishes: bool = True) -> None:
        super().__init__(store)
        self.delay, self.finishes = delay, finishes
        self.release = threading.Event()

    def league_games(self, league, refresh=False):
        self.fetches.append(league)
        if not self.finishes:
            self.release.wait(60)
            return []
        time.sleep(self.delay)
        self.store.save_odds_snapshot(league, {NBA: nba_board(), EPL: epl_board()}.get(league, []), NOW)
        return []


@pytest.fixture()
def slow(tmp_path):
    """make(delay, finishes): the Builder's world (tests/test_builder_wiring.py's games) with EPL priced, the NBA on
    ESPN only, and the slate's real loader and thread over SlowOdds: a board asked for arrives after ``delay`` seconds,
    or never."""
    worlds: List[World] = []

    def make(delay: float = 1.0, finishes: bool = True) -> World:
        store = Store(str(tmp_path / f"betting-{len(worlds)}.db"))
        store.save_odds_snapshot(EPL, epl_board(), NOW - 600)
        odds = SlowOdds(store, delay, finishes)
        slate = SlateBuilder(odds, FakeEspn(espn_boards()), store, now=lambda: NOW, workers=2)
        slate.refresh()
        service = DashboardService(StoreOdds(store), NoEspn(), NY, store=store, cache=TTLCache())
        service.slate = slate
        lab = Lab(service, store, now=lambda: NOW)
        slate.start()  # the loader wakes it when a board arrives: the slate's games then carry the prices
        world = World(store, slate, service, lab, odds)
        worlds.append(world)
        return world

    yield make
    for world in worlds:
        world.odds.release.set()
        world.slate.stop()


def legs_leagues(data: Dict[str, Any]) -> set:
    return {leg["ref"]["league"] for ticket in data["tickets"] for leg in ticket["legs"]}


def test_a_board_that_arrives_in_a_second_is_waited_for_and_the_build_has_the_nba(slow) -> None:
    world = slow(delay=1.0)
    client = world.client()
    started = time.monotonic()
    data = run(client, {"preset": "risky", "leagues": [NBA], "legs": 2})
    took = time.monotonic() - started
    assert world.odds.fetches == [NBA]
    assert data["tickets"] and legs_leagues(data) == {NBA}
    assert data["unpriced"] == [] and data["blocked"] is None and "prices_note" not in data
    assert 1.0 <= took < 1.0 + 2.0  # the board's second, never PRICE_WAIT


def test_a_board_that_never_comes_lets_the_build_go_ahead_without_it_after_price_wait(slow) -> None:
    world = slow(finishes=False)
    client = world.client()
    started = time.monotonic()
    data = run(client, {"preset": "risky", "leagues": [NBA, EPL]})
    took = time.monotonic() - started
    assert PRICE_WAIT == 8.0 and PRICE_WAIT <= took < PRICE_WAIT + 1.0
    assert world.odds.fetches == [NBA]
    assert data["tickets"] and legs_leagues(data) == {EPL}  # the ask allows it: the league that has prices
    [entry] = data["unpriced"]
    assert entry["league"] == NBA and entry["games"] == 4
    assert entry["reason"] == "NBA: no prices loaded (tonight's 4 games); they're still on their way"
    assert data["prices_note"] == "NBA prices are still on their way; here's the best I can do with EPL games now."
    assert not asks_again(json.dumps(data))
    # Asked again a moment later (the board still on its way): the same honest words, no second board
    world.lab.price_wait = 0.3
    again = run(client, {"preset": "risky", "leagues": [NBA, EPL]})
    assert again["prices_note"] == data["prices_note"] and world.odds.fetches == [NBA]


def test_a_stale_board_on_its_way_is_waited_for_too(slow) -> None:
    world = slow(delay=1.0)
    # The NBA's saved board is 3.5 hours old: past the Builder's 3-hour price limit, so on its own nothing could use it
    world.store.save_odds_snapshot(NBA, nba_board(), NOW - 3.5 * 3600)
    world.slate.refresh()
    assert slate_entry(world, NBA)["odds"] == "stale"
    data = run(world.client(), {"preset": "risky", "leagues": [NBA], "legs": 2})
    assert world.odds.fetches == [NBA]  # asked for, and waited for
    assert data["tickets"] and legs_leagues(data) == {NBA} and "prices_note" not in data
    assert all(leg["price_age_s"] == 0 and leg["age"] == "under 1 min"  # the new board's prices, not the old one's
               for ticket in data["tickets"] for leg in ticket["legs"])


def slate_entry(world: World, league: str) -> Dict[str, Any]:
    return next(entry for entry in world.slate.current().data["leagues"] if entry["key"] == league)


def test_with_nothing_to_wait_for_a_run_costs_no_time(slow) -> None:
    world = slow(delay=1.0)
    lab = world.lab
    assembled = world.service.builder_candidates(now=NOW)
    started = time.monotonic()
    assert lab._ensure_prices(assembled, [EPL]) is assembled  # priced, and fresh
    assert lab._ensure_prices(assembled, None) is assembled  # every league: some have prices, so the run goes ahead
    world.slate._requested[NHL] = NOW  # asked for lately and not on its way: the loader answers "throttled"
    assert lab._ensure_prices(assembled, [NHL]) is assembled
    assert time.monotonic() - started < 0.2 and world.odds.fetches == []
    # The bot's process has no slate to wait on: the saved boards are all it reads
    bot_service = DashboardService(StoreOdds(world.store), NoEspn(), NY, store=world.store, cache=TTLCache())
    assert Lab(bot_service, world.store, now=lambda: NOW)._ensure_prices(assembled, [NBA]) is assembled
    started = time.monotonic()
    data = run(world.client(), {"preset": "risky", "leagues": [EPL]})
    assert data["tickets"] and time.monotonic() - started < 2.0 and world.odds.fetches == []


def test_the_top_picks_wait_for_a_board_on_its_way_too(slow) -> None:
    world = slow(delay=1.0)
    world.client()  # PLAYER_A confirms their age
    started = time.monotonic()
    done = Toolbox(world.lab, UID, NY, leagues={NBA}).run("brain_top_picks", {"leagues": [NBA], "n": 3})
    assert done.ok and done.content["picks"] and all(p["ref"]["league"] == NBA for p in done.content["picks"])
    assert "prices" not in done.content and 1.0 <= time.monotonic() - started < PRICE_WAIT
    assert world.odds.fetches == [NBA]


def test_the_top_picks_say_in_one_line_that_a_league_s_prices_are_still_coming(slow) -> None:
    world = slow(finishes=False)
    world.lab.price_wait = 0.5
    world.client()
    done = Toolbox(world.lab, UID, NY, leagues={NBA, EPL}).run("brain_top_picks", {"leagues": [NBA, EPL], "n": 3})
    assert done.ok and done.content["picks"] and all(p["ref"]["league"] == EPL for p in done.content["picks"])
    assert done.content["prices"] == "NBA prices are still on their way; here's the best I can do with EPL games now."
    assert done.content["hint"].startswith("Open with prices in one line")
    assert done.cards[0]["prices_note"] == done.content["prices"]
    nothing = Toolbox(world.lab, UID, NY, leagues={NBA}).run("brain_top_picks", {"leagues": [NBA]})
    assert nothing.content["picks"] == [] and nothing.cards == []
    assert nothing.content["prices"] == "No prices are loaded for tonight's 4 NBA games: they haven't arrived yet."


# --- the words: the ask's own games, one of four reasons, never "ask again" -------------------------------------------

def unpriced_of(lab: Lab, games: List[str], controls: Any = None) -> Dict[str, Any]:
    leagues = {NHL: {"title": "NHL", "games": games, "priced": 0, "odds": "none"}}
    engine = lab_module._engine_builder()
    [entry] = lab._unpriced({"leagues": leagues}, controls or engine.Controls(), None)
    return entry


def test_tonight_s_count_is_tonight_s_games_never_every_game_to_come(tmp_path) -> None:
    world = make_world(tmp_path)
    world.slate.request_odds = lambda league: {"queued": False, "code": "throttled", "reason": "Asked lately."}
    tonight = [at(1), at(2), at(3)]  # 7, 8 and 9 pm in New York
    later = [at(25), at(49), at(72), at(96), at(120)]
    entry = unpriced_of(world.lab, tonight + later)
    assert (entry["games"], entry["when"]) == (3, "tonight's")  # it said 8: every game this week
    assert entry["reason"] == "NHL: no prices loaded (tonight's 3 games); odds were asked for less than 2 hours ago"
    assert unpriced_of(world.lab, later)["reason"].startswith("NHL: no prices loaded (tomorrow's 1 game)")
    assert unpriced_of(world.lab, [at(72), at(73)])["when"] == "Thursday's"
    morning = Lab(world.service, world.store, now=lambda: NOW - 8 * 3600)  # 10 am: a game at noon is today's
    assert unpriced_of(morning, [at(-6), at(1)])["reason"].startswith("NHL: no prices loaded (today's 2 games)")
    # A window over several days counts its games; one within a day names it
    engine = lab_module._engine_builder()
    week = engine.Controls(window={"from": at(0), "to": at(100)})
    assert unpriced_of(world.lab, tonight + later, week)["reason"].startswith("NHL: no prices loaded (7 games in "
                                                                               "your window)")
    evening = engine.Controls(window={"from": at(0), "to": at(5)})
    assert unpriced_of(world.lab, tonight + later, evening)["games"] == 3
    # Read in the person's time zone: 9 pm in Los Angeles is midnight in New York, tomorrow there
    west = engine.Controls(tz="America/Los_Angeles")
    assert unpriced_of(world.lab, [at(6)], west)["when"] == "tonight's"
    assert unpriced_of(world.lab, [at(6)])["when"] == "tomorrow's"


def test_the_one_line_reasons_and_the_built_from_others_line() -> None:
    entry = {"league": NBA, "title": "NBA", "games": 7, "when": "tonight's", "why": "on_way"}
    lines = {
        "on_way": "No prices are loaded for tonight's 7 NBA games: they haven't arrived yet.",
        "arriving": "No prices are loaded for tonight's 7 NBA games: they haven't arrived yet.",
        "reserve": "No prices are loaded for tonight's 7 NBA games: the month's odds allowance is at its reserve.",
        "daily_cap": "No prices are loaded for tonight's 7 NBA games: today's odds allowance is used up.",
    }
    for why, line in lines.items():
        assert no_prices_line([{**entry, "why": why}]) == line
    assert no_prices_line([], ["NBA"]) == "There are no NBA games coming up this week."
    assert no_prices_line([], ["NBA", "NHL"], windowed=True) == "There are no NBA and NHL games in your window."
    two = [entry, {"league": NHL, "title": "NHL", "games": 1, "when": "tonight's", "why": "on_way"}]
    assert no_prices_line(two) == "No prices are loaded for tonight's 7 NBA games and tonight's 1 NHL game: they " \
                                  "haven't arrived yet."
    assert prices_note([entry], ["NHL", "NFL"]) == ("NBA prices are still on their way; here's the best I can do "
                                                    "with NHL and NFL games now.")
    assert prices_note([{**entry, "why": "arriving"}], ["NHL"]) == ("NBA prices haven't arrived yet; here's the best "
                                                                    "I can do with NHL games now.")
    assert prices_note([{**entry, "why": "daily_cap"}], ["NHL"]) == ("No NBA prices (today's odds allowance is used "
                                                                     "up); here's the best I can do with NHL games now.")
    assert prices_note([entry]) == "NBA prices are still on their way."
    for line in list(lines.values()) + [prices_note([entry], ["NHL"])]:
        assert not asks_again(line)


def test_ask_again_in_a_few_minutes_is_caught_and_honest_lines_pass() -> None:
    for text in ("Ask again in a few minutes and I'll build the two most likely picks.", "Try again in a minute.",
                 "Check back shortly for the NBA.", "Give it a few minutes.", "ask me again in 5 minutes",
                 "Come back in a couple of minutes"):
        assert asks_again(text), text
    for text in ("Today's odds allowance is used up: ask again tomorrow.", "Tap Build again once they're in.",
                 "NBA prices are still on their way; here's the best I can do with NHL games now.", "Try again later."):
        assert not asks_again(text), text


def test_the_templates_lead_with_the_prices_line() -> None:
    note = "NBA prices are still on their way; here's the best I can do with NHL games now."
    ticket = {"legs": [{"label": "Boston Bruins to win"}], "odds": {"decimal": 1.8}, "joint": {"chance": 0.55}}
    words = builder_template([{"title": "Risky ticket #1", "ticket": ticket, "prices_note": note}])
    assert words.split("\n")[0] == note and words.split("\n")[1].startswith("Risky ticket #1: Boston Bruins to win")
    picks = {"picks": [{"rank": 1, "label": "Boston Bruins to win", "chance": 0.6, "price": "1.65",
                        "book": "DraftKings", "why": ["Home ice"]}], "prices_note": note}
    assert picks_template(picks).split("\n")[:2] == [note, "The most likely winners, strongest first:"]
    assert picks_template({"picks": [], "prices_note": "No prices are loaded for tonight's 4 NBA games: they haven't "
                                                        "arrived yet."}).startswith("No prices are loaded for tonight's")
    assert picks_template({"picks": []}) == "No game in scope has prices loaded right now: try another sport."


# --- the owner's case, end to end ----------------------------------------------------------------------------------

OWNER = "build me a 2 leg NBA ticket"
OWNER_ROUTE = {"intent": "build", "leagues": [NBA], "stake": None, "target": None, "max_legs": None, "tier": "none",
               "target_odds": None, "target_chance": None, "legs": 2, "top_picks": False, "days": None,
               "min_odds": None, "legs_per_day": None, "players": [], "risk_flags": []}


class OwnerClaude:
    """The owner's ask as Claude takes it: the router reads a 2-leg NBA build, the chat model calls ticket_build with
    it and answers from the tool's facts: the ticket's picks, or the tool's line on why there is none (``stubborn``:
    always "ask again in a few minutes" instead, as the owner was told)."""

    def __init__(self, stubborn: bool = False) -> None:
        self.stubborn = stubborn
        self.requests: List[Dict[str, Any]] = []
        self.facts: List[Dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(json.loads(json.dumps(request, default=lambda o: getattr(o, "__dict__", str(o)))))
        if request["model"] == "claude-haiku-5-5":
            return reply([say(json.dumps(OWNER_ROUTE))], model="claude-haiku-5-5")
        results = [block for message in request["messages"] if isinstance(message["content"], list)
                   for block in message["content"] if isinstance(block, dict) and block.get("type") == "tool_result"]
        if not results:
            return reply([SimpleNamespace(type="tool_use", id="toolu_1", name="ticket_build",
                                          input={"leagues": [NBA], "legs": 2})], stop="tool_use")
        facts = json.loads(results[-1]["content"])
        self.facts.append(facts)
        if self.stubborn:
            return reply([say("Tonight's NBA prices haven't loaded yet. Ask again in a few minutes.")])
        if facts.get("tickets"):
            legs = facts["tickets"][0]["legs"]
            return reply([say(f"Here's your 2-leg NBA ticket: {legs[0]['label']} and {legs[1]['label']}. The card "
                              "below has the details.")])
        return reply([say(facts.get("prices") or facts.get("blocked") or "Nothing could be built.")])


def facts_text(facts: Dict[str, Any]) -> str:
    """What the tool told Claude, without its hint (the hint itself says never to ask again)."""
    return json.dumps({key: value for key, value in facts.items() if key != "hint"})


def test_the_owner_s_two_leg_nba_ticket_waits_for_the_board_and_ends_with_a_ticket(slow) -> None:
    world = slow(delay=1.0)
    world.client()  # PLAYER_A confirms their age
    claude = OwnerClaude()
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    started = time.monotonic()
    answer = world.lab.chat(UID, {"message": OWNER})["reply"]
    took = time.monotonic() - started
    assert world.odds.fetches == [NBA] and 1.0 <= took < PRICE_WAIT
    assert answer["kind"] == "answer", answer
    assert answer["text"].startswith("Here's your 2-leg NBA ticket: ")
    tickets = [card for card in answer["cards"] if card["type"] == "ticket"]
    assert tickets and all(len(card["refs"]) == 2 and {ref["league"] for ref in card["refs"]} == {NBA}
                           for card in tickets)
    [facts] = claude.facts
    assert facts["tickets"] and "unpriced" not in facts and "blocked" not in facts and "prices" not in facts
    for text in (answer["text"], facts_text(facts)):
        assert not asks_again(text) and "few minutes" not in text.lower() and "in the background" not in text


def test_the_owner_s_ticket_in_the_telegram_chat(slow) -> None:
    world = slow(delay=1.0)
    sent, _ = bot_turn(world, OwnerClaude(), OWNER)
    text = html.unescape(sent.text)  # bot/chat.py's Telegram HTML
    assert "Here's your 2-leg NBA ticket: " in text and text.count("🧾") >= 1 and world.odds.fetches == [NBA]
    assert not asks_again(text) and "few minutes" not in text and "no prices" not in text.lower()


def test_after_a_wait_a_reply_that_says_ask_again_is_never_shown(slow) -> None:
    world = slow(finishes=False)
    world.lab.price_wait = 0.5
    world.client()
    claude = OwnerClaude(stubborn=True)
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    answer = world.lab.chat(UID, {"message": OWNER})["reply"]
    line = "No prices are loaded for tonight's 4 NBA games: they haven't arrived yet."
    assert claude.facts[0]["prices"] == claude.facts[0]["blocked"] == line
    assert "never tell them to ask again in a few minutes" in claude.facts[0]["hint"]
    # The reply asked again: it was sent back once, said it again, and code's own line answers instead
    corrections = [message["content"] for request in claude.requests for message in request["messages"]
                   if message["role"] == "user" and isinstance(message["content"], str)
                   and ASK_AGAIN_PROBLEM in message["content"]]
    assert corrections
    assert answer["kind"] == "template" and answer["text"] == line
    assert not asks_again(answer["text"])


# --- every board counted: the odds service's own loads (a game page, a league's games, the bot) ----------------------
#
# The real OddsService over a real Store, its _get answering as The Odds API would and counting every call (no network,
# no key sent anywhere): the day's budget holds at the one place that fetches, whichever page or process loads.

LISTING_RAW = [{"key": key, "group": "x", "title": key, "active": True, "has_outrights": False} for key in (NBA, NHL)]


class CountingApi:
    """OddsService._get's stand-in: a board request answers the league's events as raw Odds API events, /sports the
    listing; every call is recorded (``boards``: the board requests)."""

    def __init__(self, events: List[Dict[str, Any]]) -> None:
        self.events = events
        self.calls: List[str] = []
        self.lock = threading.Lock()

    @property
    def boards(self) -> List[str]:
        return [path for path in self.calls if path.endswith("/odds")]

    def get(self, path: str, params: Dict[str, Any]) -> Any:
        with self.lock:
            self.calls.append(path)
        if not path.endswith("/odds"):
            return json.loads(json.dumps(LISTING_RAW))
        league = path.split("/")[2]
        return [{"id": event["id"], "sport_key": league, "commence_time": event["commence_time"],
                 "home_team": event["home"], "away_team": event["away"], "bookmakers": []} for event in self.events]


def odds_world(tmp_path, monkeypatch, remaining: Any = 400, name: str = "betting.db") -> SimpleNamespace:
    """The NBA's board saved 3 hours ago (stale) in a real Store, the month's count ``remaining`` (None: unknown), the
    real OddsService and games service over it, and the counting API."""
    store = Store(str(tmp_path / name))
    board = nba_board(in_2030)
    store.save_odds_snapshot(NBA, board, time.time() - 3 * 3600)
    if remaining is not None:
        store.record_usage("/sports", 0, remaining, 500 - remaining)
    api = CountingApi(board)
    monkeypatch.setattr(OddsService, "_get", lambda self, path, params: api.get(path, params))
    odds = OddsService("key", cache=TTLCache(name=f"odds-{name}"), store=store)
    service = DashboardService(odds, QuietEspn(), NY, store=store, cache=TTLCache(name=f"games-{name}"))
    return SimpleNamespace(store=store, odds=odds, service=service, api=api, board=board, game=board[0]["id"])


def use_up_today(store: Store, remaining: Any) -> str:
    """Count boards against today until the day's budget (day_boards, the rule every loader uses) has none left."""
    day = budget_day(time.time())
    while sum(store.odds_boards(day).values()) < day_boards(remaining, 50, 3, sum(store.odds_boards(day).values()),
                                                            time.time()):
        assert store.take_odds_board(day, "request", 10_000)
    return day


def reload(w: SimpleNamespace) -> None:
    """Forget the in-memory copies (the odds service's minute, the games service's league list): the next read loads."""
    w.odds.cache.invalidate(("odds", NBA))
    w.service.cache.invalidate(("games", NBA))


def test_past_the_day_s_budget_no_page_or_list_fetches_a_stale_board(tmp_path, monkeypatch) -> None:
    w = odds_world(tmp_path, monkeypatch)
    day = use_up_today(w.store, 400)
    counted = w.store.odds_boards(day)
    # The game page (/api/game runs DashboardService.game before the page asks /api/game/picks), a league's games
    # (Today's league list, the bot's list and card), a read that asks for a refresh, and a league never loaded at all
    assert w.service.game(NBA, w.game)["home"]  # served from the saved board
    assert {card["id"] for card in w.service.games(NBA)["games"]} == {event["id"] for event in w.board}
    assert [g["id"] for g in w.odds.league_games(NBA, refresh=True)] == [event["id"] for event in w.board]
    assert w.odds.league_games(NHL) == []  # nothing saved and no board left today: no games, and no request
    # Not one board request (the league list's title reads the free /sports listing, which costs no credits), and
    # nothing more counted
    assert w.api.boards == [] and set(w.api.calls) <= {"/sports"} and w.store.odds_boards(day) == counted
    # The reviewer's repro (scratchpad/review-gp/checks/q_gameroute.py): one /api/game open asks nothing past the day
    reload(w)
    w.service.game(NBA, w.game)
    assert w.api.boards == []


def test_with_a_board_left_a_page_s_load_asks_once_and_is_counted(tmp_path, monkeypatch) -> None:
    w = odds_world(tmp_path, monkeypatch)
    day = budget_day(time.time())
    w.service.game(NBA, w.game)  # the review's repro with room in the day: one request, the count +1
    assert w.api.boards == [f"/sports/{NBA}/odds"] and w.store.odds_boards(day) == {DIRECT: 1}
    reload(w)
    w.service.games(NBA)  # the new board is fresh: read, not fetched, and nothing counted
    assert w.api.boards == [f"/sports/{NBA}/odds"] and w.store.odds_boards(day) == {DIRECT: 1}
    # A request The Odds API refuses spends no credits: its board is given back
    w.store.save_odds_snapshot(NBA, w.board, time.time() - 3 * 3600)

    def refused(self, path, params):
        w.api.calls.append(path)
        raise OddsApiError(f"The Odds API returned HTTP 500 for {path}", 500)
    monkeypatch.setattr(OddsService, "_get", refused)
    reload(w)
    assert w.odds.league_games(NBA)  # the saved board
    assert w.api.boards == [f"/sports/{NBA}/odds"] * 2 and w.store.odds_boards(day) == {DIRECT: 1}


def test_with_the_count_unknown_the_odds_service_s_day_has_12_boards(tmp_path, monkeypatch) -> None:
    w = odds_world(tmp_path, monkeypatch, remaining=None)
    assert w.odds.quota["remaining"] is None and w.odds.odds_per_day == ODDS_PER_DAY == 12
    day = budget_day(time.time())
    for _ in range(11):
        assert w.store.take_odds_board(day, "request", 99)
    w.service.game(NBA, w.game)  # the 12th board
    assert w.api.boards == [f"/sports/{NBA}/odds"] and w.store.odds_boards(day) == {"request": 11, DIRECT: 1}
    w.store.save_odds_snapshot(NBA, w.board, time.time() - 3 * 3600)
    reload(w)
    w.service.game(NBA, w.game)  # a 13th: refused, the saved board served
    assert w.api.boards == [f"/sports/{NBA}/odds"] and sum(w.store.odds_boards(day).values()) == 12
    # SLATE_ODDS_PER_DAY reaches the odds service of both processes (services.build_service)
    from telegram_betting_bot_simple import services
    from telegram_betting_bot_simple.config import Config
    config = Config(TELEGRAM_BOT_TOKEN="123456:test", THE_ODDS_API_KEY="test-only", STATE_DIR=str(tmp_path / "state"),
                    SLATE_ODDS_PER_DAY="5")
    assert services.build_service(config).odds.odds_per_day == 5


def slate_over(w: SimpleNamespace) -> SlateBuilder:
    """The real slate over the world's real odds and games services (the NBA's load goes through the games service, as
    the AI league's does), with tonight's NBA game on ESPN, on the real clock."""
    now = time.time()
    tonight = datetime.datetime.fromtimestamp(now + 3 * 3600, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    day = datetime.datetime.fromtimestamp(now, ESPN_DAY_TZ).date()
    espn = FakeEspn({(NBA, day): [espn_game("e1", "Boston Celtics", "New York Knicks", tonight)]})
    slate = SlateBuilder(w.odds, espn, w.store, service=w.service, workers=2)
    slate.refresh()
    return slate


def test_a_board_the_slate_queues_is_counted_once(tmp_path, monkeypatch) -> None:
    w = odds_world(tmp_path, monkeypatch)
    slate = slate_over(w)
    day = budget_day(time.time())
    assert slate.request_odds(NBA)["code"] == "queued"
    wait_for(slate)
    # The slate counted the board when it queued it; the odds service's fetch takes no second one
    assert w.api.boards == [f"/sports/{NBA}/odds"] and w.store.odds_boards(day) == {"request": 1}
    assert slate.request_odds(NBA)["code"] == "fresh"


def test_a_queued_board_that_loads_nothing_is_given_back(tmp_path, monkeypatch) -> None:
    w = odds_world(tmp_path, monkeypatch)
    slate = slate_over(w)
    day = budget_day(time.time())
    gate = hold_the_odds_pool(slate)
    assert slate.request_odds(NBA)["code"] == "queued"
    assert w.store.odds_boards(day) == {"request": 1}
    w.store.save_odds_snapshot(NBA, w.board, time.time())  # meanwhile the other process loaded it
    gate.set()
    wait_for(slate)
    assert w.api.boards == [] and w.store.odds_boards(day) == {}  # nothing fetched, so nothing spent


def test_a_queued_board_that_did_load_keeps_its_count_whoever_loaded_it(tmp_path, monkeypatch) -> None:
    # An odds service that saves the board some other way (the kit's fakes do) never marks the slate's board as used:
    # the board still came, so it stays counted (only a load that brought nothing gives its board back)
    w = odds_world(tmp_path, monkeypatch)
    slate = slate_over(w)
    day = budget_day(time.time())

    def saved_elsewhere(league, refresh=False):
        w.store.save_odds_snapshot(league, w.board, time.time())
        return w.board
    w.odds.league_games = saved_elsewhere
    assert slate.request_odds(NBA)["code"] == "queued"
    wait_for(slate)
    assert w.api.boards == [] and w.store.odds_boards(day) == {"request": 1}

def test_two_processes_can_t_both_take_the_last_board(tmp_path, monkeypatch) -> None:
    for attempt in range(3):
        path = str(tmp_path / f"shared-{attempt}.db")
        dashboard, bot = Store(path), Store(path)
        bot.holder = "bot"
        board = nba_board(in_2030)
        for league in (NBA, NHL):
            dashboard.save_odds_snapshot(league, board, time.time() - 3 * 3600)  # both stale
        api = CountingApi(board)
        monkeypatch.setattr(OddsService, "_get", lambda self, p, q: api.get(p, q))
        # The count unknown and SLATE_ODDS_PER_DAY at 1: one board today, for both processes together
        services_ = [OddsService("key", cache=TTLCache(name=f"{who}-{attempt}"), store=store, odds_per_day=1)
                     for who, store in (("dashboard", dashboard), ("bot", bot))]
        start = threading.Barrier(2)

        def load(service: OddsService, league: str) -> None:
            start.wait(5)
            service.league_games(league)
        threads = [threading.Thread(target=load, args=(services_[0], NBA)),
                   threading.Thread(target=load, args=(services_[1], NHL))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert len(api.boards) == 1, (attempt, api.boards)
        assert dashboard.odds_boards(budget_day(time.time())) == {DIRECT: 1}


# --- the final check's four fixes ----------------------------------------------------------------------------------
#
# A fake The Odds API at requests.get, so the real OddsService._get runs and records the usage headers as in production:
# a board answers after ``delay`` seconds with the count down by its cost. No network.

NFL = "americanfootball_nfl"


class Answer:
    def __init__(self, headers: Dict[str, str], body: Any) -> None:
        self.status_code, self.headers, self._body = 200, headers, body
        self.ok = True

    def json(self) -> Any:
        return self._body


class FakeOddsApi:
    def __init__(self, remaining: int, delay: float = 0.0, headers: bool = True) -> None:
        self.remaining, self.used, self.delay, self.headers = remaining, 500 - remaining, delay, headers
        self.calls: List[str] = []
        self.boards: List[str] = []
        self.lock = threading.Lock()
        self.events = nba_board(in_2030)

    def get(self, url: str, params: Any = None, timeout: Any = None) -> Answer:
        path = url[len(odds_service.BASE_URL):]
        with self.lock:
            self.calls.append(path)
        if path.endswith("/odds"):
            time.sleep(self.delay)  # a board takes a moment to arrive
            league = path.split("/")[2]
            with self.lock:
                self.boards.append(league)
                self.remaining, self.used = self.remaining - 3, self.used + 3
                headers = self._headers(3)
            return Answer(headers, [{"id": e["id"], "sport_key": league, "commence_time": e["commence_time"],
                                     "home_team": e["home"], "away_team": e["away"], "bookmakers": []}
                                    for e in self.events])
        with self.lock:
            headers = self._headers(0)
        return Answer(headers, [{"key": key, "group": "x", "title": key, "active": True, "has_outrights": False}
                                for key in (NBA, NHL, NFL)])

    def _headers(self, cost: int) -> Dict[str, str]:
        if not self.headers:
            return {}
        return {"x-requests-remaining": str(self.remaining), "x-requests-used": str(self.used),
                "x-requests-last": str(cost)}


def shared_odds(tmp_path, monkeypatch, remaining: int, delay: float = 0.6, counted: int = 0, days: int = 30):
    """Two processes' stores on one file (the dashboard's and the bot's), every league's board saved 3 hours ago (stale),
    ``counted`` boards already counted today, the month's count ``remaining`` and ``days`` days left in it."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = str(tmp_path / "shared.db")
    stores = [Store(path), Store(path)]
    stores[1].holder = "bot"
    for league in (NBA, NHL, NFL):
        stores[0].save_odds_snapshot(league, nba_board(in_2030), time.time() - 3 * 3600)
    day = budget_day(time.time())
    for _ in range(counted):
        assert stores[0].take_odds_board(day, "request", 10_000)
    stores[0].record_usage("/sports", 0, remaining, 500 - remaining)
    api = FakeOddsApi(remaining, delay)
    monkeypatch.setattr(odds_service.requests, "get", api.get)
    monkeypatch.setattr(odds_service, "days_left", lambda now: days)
    return stores, api, day


def overlapping(loads: List[Any], stagger: float = 0.1) -> None:
    """Start each load ``stagger`` seconds after the one before, while the earlier boards are still on their way."""
    start = threading.Barrier(len(loads))

    def run(i: int, load: Any) -> None:
        start.wait(5)
        time.sleep(i * stagger)
        load()
    threads = [threading.Thread(target=run, args=(i, load)) for i, load in enumerate(loads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)


def test_fix1_overlapping_loads_never_pass_the_day_or_the_reserve(tmp_path, monkeypatch) -> None:
    # The day's last board, two processes and two threads: 488 left and 3 boards counted (497 as the day began), 30
    # days left: 4 boards today. The second load starts while the first board is on its way (the count hasn't dropped)
    for processes in (2, 1):
        stores, api, day = shared_odds(tmp_path / f"day-{processes}", monkeypatch, 488, counted=3)
        services_ = [OddsService("key", cache=TTLCache(name=f"day-{processes}-{i}"), store=stores[i % processes])
                     for i in range(2)]
        overlapping([lambda: services_[0].league_games(NBA), lambda: services_[1].league_games(NHL)])
        assert api.boards == [NBA], (processes, api.boards)  # it read 2 boards before the fix
        assert sum(stores[0].odds_boards(day).values()) == 4 and stores[0].odds_day(day)["limit"] == 4
    # The month's last day, 1 board above the reserve: three overlapping loads in two processes load one board, and the
    # count stops at the reserve (three boards took it to 44 before the fix)
    stores, api, day = shared_odds(tmp_path / "last", monkeypatch, 53, days=1)
    services_ = [OddsService("key", cache=TTLCache(name=f"last-{i}"), store=stores[i % 2]) for i in range(3)]
    overlapping([lambda s=s, league=league: s.league_games(league) for s, league in zip(services_, (NBA, NHL, NFL))])
    assert api.boards == [NBA] and api.remaining == 50 and stores[0].latest_quota()["remaining"] == 50
    assert stores[0].odds_boards(day) == {DIRECT: 1}


def test_fix1_the_slate_s_board_and_a_page_s_overlap_within_the_day(tmp_path, monkeypatch) -> None:
    stores, api, day = shared_odds(tmp_path, monkeypatch, 488, counted=3)  # one board left today
    odds = OddsService("key", cache=TTLCache(name="slate-and-page"), store=stores[0])
    now = time.time()
    tonight = datetime.datetime.fromtimestamp(now + 3 * 3600, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    espn_day = datetime.datetime.fromtimestamp(now, ESPN_DAY_TZ).date()
    slate = SlateBuilder(odds, FakeEspn({(NHL, espn_day): [espn_game("h1", "Boston Bruins", "New York Rangers", tonight,
                                                                     sport="nhl")]}), stores[0], workers=2)
    slate.refresh()
    assert slate.request_odds(NHL)["code"] == "queued"  # the day's last board, on its way
    time.sleep(0.1)
    odds.league_games(NBA)  # a page opens a stale NBA board meanwhile: no board left today
    wait_for(slate)
    assert api.boards == [NHL] and stores[0].odds_boards(day) == {"request": 4}  # two boards before the fix


def test_fix1_claims_fix_the_day_and_hold_the_reserve_with_boards_on_their_way(tmp_path) -> None:
    path = str(tmp_path / "claims.db")
    a, b = Store(path), Store(path)
    b.holder = "bot"
    day = "2026-10-31"
    first, why = a.claim_odds_board(day, DIRECT, remaining=497, limit=4, reserve=50, cost=3)
    assert first and why == "ok" and a.odds_day(day) == {"start_remaining": 497, "limit": 4}
    # Later claims can't make the day bigger, whatever their own sums say (a board on its way still shows in the count)
    taken = [b.claim_odds_board(day, "request", remaining=497, limit=40, reserve=50, cost=3) for _ in range(4)]
    assert [why for _, why in taken] == ["ok", "ok", "ok", "daily_cap"] and a.odds_day(day)["limit"] == 4
    # A preload's part of the day: 70% of a 10-board day is 7
    other = "2026-10-30"
    parts = [a.claim_odds_board(other, "preload", remaining=1000, limit=10, share=PRELOAD_SHARE, reserve=50, cost=3)
             for _ in range(8)]
    assert [why for _, why in parts].count("ok") == 7 and parts[-1][1] == "daily_cap"
    for claim, _ in [(first, "ok")] + taken + parts:
        a.finish_odds_board(claim)  # those have all loaded
    # The reserve with the boards on their way: 56 left and 2 boards on their way, so a third would leave 47
    tight = "2026-10-29"
    on_way = [a.claim_odds_board(tight, DIRECT, remaining=56, limit=99, reserve=50, cost=3) for _ in range(3)]
    assert [why for _, why in on_way] == ["ok", "ok", "reserve"]
    b.finish_odds_board(on_way[0][0])  # the first has loaded (its cost is in the count the API reports now)
    assert b.claim_odds_board(tight, DIRECT, remaining=53, limit=99, reserve=50, cost=3)[1] == "reserve"
    a.give_back_odds_board(tight, DIRECT, on_way[1][0])  # the second's request failed: nothing was spent
    late, why = b.claim_odds_board(tight, DIRECT, remaining=53, limit=99, reserve=50, cost=3)
    assert late and why == "ok"
    # A higher count (the month's reset, a new key) fixes the day again
    reset, why = a.claim_odds_board(day, DIRECT, remaining=20_000, limit=214, reserve=50, cost=3)
    assert reset and why == "ok" and a.odds_day(day) == {"start_remaining": 20_000, "limit": 214}
    for claim in (late, reset):
        a.finish_odds_board(claim)
    # A claim never ended (its process died) stops counting as on its way after CLAIM_SECONDS
    stale = "2026-10-28"
    dead, why = a.claim_odds_board(stale, DIRECT, remaining=53, limit=9, reserve=50, cost=3,
                                   now=time.time() - CLAIM_SECONDS - 1)
    assert dead and why == "ok"
    assert a.claim_odds_board(stale, DIRECT, remaining=53, limit=9, reserve=50, cost=3)[1] == "ok"  # 47 if it counted


def last_month_count(path: str, remaining: int) -> None:
    """The store's latest count, reported 30 minutes before this UTC month began."""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    first = datetime.datetime(today.year, today.month, 1, tzinfo=datetime.timezone.utc).timestamp()
    with contextlib.closing(sqlite3.connect(path)) as db:
        db.execute("INSERT INTO api_usage (ts, endpoint, cost, remaining, used) VALUES (?, ?, ?, ?, ?)",
                   (first - 1800, "/sports/basketball_nba/odds", 3, remaining, 500 - remaining))
        db.commit()


def test_fix2_on_the_1st_last_month_s_count_is_read_again_and_a_page_loads(tmp_path, monkeypatch) -> None:
    # Last month's budget was spent: its last count, 52, is just above the reserve (no board today by that count).
    # The Odds API's real count has reset to 500
    path = str(tmp_path / "month.db")
    store = Store(path)
    store.save_odds_snapshot(NBA, nba_board(in_2030), time.time() - 3 * 3600)
    last_month_count(path, 52)
    api = FakeOddsApi(500)
    monkeypatch.setattr(odds_service.requests, "get", api.get)
    odds = OddsService("key", cache=TTLCache(name="month"), store=store)
    assert day_boards(52, 50, 3, 0, time.time()) == 0
    odds.league_games(NBA)  # a page opens a stale board: the count is read again for free, then the board loads
    assert api.calls == ["/sports", f"/sports/{NBA}/odds"] and odds.quota["remaining"] == 497  # [] and 52 before
    # The check is free and rare: once read, this month's count stands
    assert odds.recheck_quota() is False


def test_fix2_a_count_above_the_reserve_that_gives_no_board_is_read_again(tmp_path, monkeypatch) -> None:
    # The review's month_start case: 52 reported (this month), so day_boards gives 0 although the count is above the
    # reserve: read again (free, at most hourly), and the board loads
    path = str(tmp_path / "zero.db")
    store = Store(path)
    store.save_odds_snapshot(NBA, nba_board(in_2030), time.time() - 3 * 3600)
    store.record_usage("/sports/x/odds", 3, 52, 448)
    api = FakeOddsApi(500)
    monkeypatch.setattr(odds_service.requests, "get", api.get)
    odds = OddsService("key", cache=TTLCache(name="zero"), store=store)
    odds.league_games(NBA)
    assert api.boards == [NBA] and odds.quota["remaining"] == 497


def test_fix2_the_slate_s_first_round_after_midnight_loads_again(tmp_path, monkeypatch) -> None:
    path = str(tmp_path / "round.db")
    store = Store(path)
    store.save_odds_snapshot(NHL, nba_board(in_2030), time.time() - 3 * 3600)
    api = FakeOddsApi(500, headers=False)
    monkeypatch.setattr(odds_service.requests, "get", api.get)
    odds = OddsService("key", cache=TTLCache(name="round"), store=store)
    odds.sports()  # the in-season listing, cached for hours (and here reporting no count)
    last_month_count(path, 52)
    api.headers = True
    now = time.time()
    tonight = datetime.datetime.fromtimestamp(now + 3 * 3600, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    espn_day = datetime.datetime.fromtimestamp(now, ESPN_DAY_TZ).date()
    slate = SlateBuilder(odds, FakeEspn({(NHL, espn_day): [espn_game("h1", "Boston Bruins", "New York Rangers", tonight,
                                                                     sport="nhl")]}), store, workers=2, preload=True)
    slate.refresh()  # the first round: the count is read again (the free listing), and the NHL's board preloads
    wait_for(slate)
    assert api.calls == ["/sports", "/sports", f"/sports/{NHL}/odds"] and odds.quota["remaining"] == 497


class PlanClaude:
    """The router reads the owner's 2-leg NBA build; the chat model calls each tool of ``plan`` in turn, then answers."""

    def __init__(self, plan: List[Any]) -> None:
        self.plan = plan
        self.results = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        if request["model"] == "claude-haiku-5-5":
            return reply([say(json.dumps(OWNER_ROUTE))], model="claude-haiku-5-5")
        done = sum(1 for message in request["messages"] if isinstance(message["content"], list)
                   for block in message["content"] if isinstance(block, dict) and block.get("type") == "tool_result")
        self.results = done
        if done < len(self.plan):
            name, args = self.plan[done]
            return reply([SimpleNamespace(type="tool_use", id=f"toolu_{done}", name=name, input=args)],
                         stop="tool_use")
        return reply([say("Here is what I found.")])


def test_fix3_a_chat_turn_s_tools_share_one_price_wait(slow) -> None:
    world = slow(finishes=False)  # the NBA's board never comes: each call would wait PRICE_WAIT on its own
    world.client()
    claude = PlanClaude([("ticket_build", {"leagues": [NBA], "legs": 2}), ("ticket_build", {"leagues": [NBA], "legs": 3}),
                         ("brain_top_picks", {"leagues": [NBA]})])
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    started = time.monotonic()
    out = world.lab.chat(UID, {"message": OWNER})
    took = time.monotonic() - started
    assert claude.results == 3 and out["reply"]["kind"] == "answer"
    assert PRICE_WAIT <= took < PRICE_WAIT + 1.5, took  # 24 s before the fix
    assert world.odds.fetches == [NBA] and lab_module.PRICE_BUDGET.get() is None
    # Outside a chat turn the Builder and the top picks still wait on their own, each its full wait
    world.lab.price_wait = 0.4
    for call in (lambda: world.lab.builder_run(UID, {"controls": {"preset": "risky", "leagues": [NBA]}}),
                 lambda: world.lab.builder_top_picks(UID, [NBA])):
        started = time.monotonic()
        call()
        assert time.monotonic() - started >= 0.4


def test_fix4_a_person_s_request_goes_first_and_preloads_stay_two_deep(store) -> None:
    order = [EPL, LIGA, SERIE, BUNDES, NHL]
    boards = {(league, TODAY): [espn_game(f"g{n}", "Home Side", "Away Side", at(1 + n), ids=(f"{n}1", f"{n}2"),
                                          sport="nhl" if league == NHL else "soccer")] for n, league in enumerate(order)}
    boards[(NBA, TODAY)] = [espn_game("n1", "Boston Celtics", "New York Knicks", at(8))]
    builder, odds = preloader(store, boards)
    gate = hold_the_odds_pool(builder)  # a board is loading
    builder.refresh()
    assert sorted(builder._loading) == sorted([EPL, LIGA])  # at most 2 preloads queued or loading (5 due: 12 before)
    builder.refresh()
    assert sorted(builder._loading) == sorted([EPL, LIGA])  # full: the next round adds none
    assert builder.request_odds(NBA)["code"] == "queued"
    gate.set()
    wait_for(builder)
    assert odds.fetches == [NBA, EPL, LIGA]  # the person's request starts next, before the queued preloads
    builder.refresh()  # the next round tops the preloads up
    wait_for(builder)
    assert odds.fetches == [NBA, EPL, LIGA, SERIE, BUNDES]
