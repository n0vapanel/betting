"""The odds allowance has no daily limit (the owner's call, 8 Nov 2026: "there should not be limits ... if api credits
run out i will pay more").

- ODDS_UNLIMITED (on by default) lifts the day's board ration: however many boards the day has already loaded, a
  league still gets a fresh board when its prices are needed, so the Builder prices a ticket at any hour instead of
  answering daily_cap at 21:00 (4 boards a day were spent by 03:09 UTC).
- The month's count is still read and every board is still counted (the quota line and the burn rate stay honest), and
  nothing is fetched at zero credits left: the API would refuse it anyway.
- The other brakes stay: a league's board is reused for its odds TTL, and the loader takes one board at a time (a
  person's request before a preload).
"""

from __future__ import annotations

import time
from typing import Optional

import pytest
import requests

from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.odds_service import UNLIMITED_BOARDS, OddsService, budget_day
from telegram_betting_bot_simple.data.slate import SlateBuilder
from telegram_betting_bot_simple.data.store import Store
from tests.test_fast_odds import DAY, NBA, NHL, preloader, wait_for
from tests.test_slate import TODAY, at, espn_game


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(requests.Session, "request", refuse)


@pytest.fixture()
def store(tmp_path) -> Store:
    return Store(str(tmp_path / "betting.db"))


def spent(store: Store, boards: int, day: str) -> str:
    for _ in range(boards):
        assert store.take_odds_board(day, "request", boards)
    return day


def service_with(store: Store, *, unlimited: bool, remaining: int = 369) -> OddsService:
    odds = OddsService("key", cache=TTLCache(name=f"unlimited-{unlimited}"), store=store, reserve=50,
                       unlimited=unlimited)
    odds._quota.update({"remaining": remaining, "used": 500 - remaining})
    return odds


def slate_with_preloader(store: Store, boards):
    builder, odds = preloader(store, boards)
    builder.preload = False  # these tests drive the request path themselves
    builder.refresh()
    wait_for(builder)
    return builder, odds


def spend_the_day(store: Store, builder: SlateBuilder) -> int:
    """Spend the whole of the slate's day, as 7 Nov did by 03:09 UTC. The day's boards are worked out again with
    what has been spent (so the allowance doesn't shrink as it is used): spend until the count passes it."""
    for _ in range(200):
        budget = builder._budget(builder._now())
        if budget["spent"] >= budget["boards"]:
            return budget["spent"]
        assert store.take_odds_board(DAY, "request", 10_000)
    raise AssertionError("the day never filled up")


# --- the odds service's own loading -----------------------------------------------------------------------------

def test_a_board_still_loads_once_the_day_s_ration_is_spent_when_unlimited(store) -> None:
    spent(store, 4, budget_day(time.time()))
    odds = service_with(store, unlimited=True)
    allowed, board = odds._take_board(NBA)
    assert allowed is True
    board.done()


def test_the_daily_ration_still_stops_a_board_when_the_limit_is_on(store) -> None:
    """The old rule, kept as an opt-out (ODDS_UNLIMITED=0)."""
    spent(store, 4, budget_day(time.time()))
    odds = service_with(store, unlimited=False)
    allowed, _ = odds._take_board(NBA)
    assert allowed is False


def test_nothing_is_fetched_with_no_credits_left(store) -> None:
    """Paying more does not mean asking an API for what it will refuse: at zero credits the saved board stands."""
    spent(store, 4, budget_day(time.time()))
    odds = service_with(store, unlimited=True, remaining=0)
    allowed, _ = odds._take_board(NBA)
    assert allowed is False


def test_the_month_still_counts_what_unlimited_loading_spends(store) -> None:
    day = spent(store, 4, budget_day(time.time()))
    odds = service_with(store, unlimited=True)
    for _ in range(5):
        allowed, board = odds._take_board(NBA)
        assert allowed and board
        board.done()
    assert sum(store.odds_boards(day).values()) == 9  # 4 spent before, 5 now: the burn rate stays visible
    assert UNLIMITED_BOARDS > 1000


# --- the slate's loader (what the Builder's ask goes through) -----------------------------------------------------

def test_the_slate_s_loader_loads_past_the_day_s_ration_when_unlimited(store) -> None:
    boards = {(NBA, TODAY): [espn_game("n1", "Boston Celtics", "New York Knicks", at(3))]}
    builder, odds = slate_with_preloader(store, boards)
    spend_the_day(store, builder)  # the ration, worked out with the limit still on
    odds.unlimited = True
    assert builder.request_odds(NBA)["code"] == "queued"
    wait_for(builder)
    assert odds.fetches == [NBA]


def test_the_slate_still_refuses_past_the_ration_without_the_flag(store) -> None:
    boards = {(NBA, TODAY): [espn_game("n1", "Boston Celtics", "New York Knicks", at(3))]}
    builder, odds = slate_with_preloader(store, boards)
    spend_the_day(store, builder)
    assert builder.request_odds(NBA)["code"] == "daily_cap"
    wait_for(builder)
    assert odds.fetches == []


def test_a_day_frozen_at_the_old_ration_does_not_stop_unlimited_loading(store) -> None:
    """The bug after the first deploy: the day was already frozen at 4 boards (odds_days, 8 Nov), and the store's
    fixed limit still refused every preload (2 of 2 preload boards). Unlimited must ignore the frozen cap."""
    day = budget_day(time.time())
    first, why = store.claim_odds_board(day, "preload", remaining=369, limit=4, share=0.7, reserve=50, cost=3)
    assert first and why == "ok" and store.odds_day(day)["limit"] == 4
    store.finish_odds_board(first)
    # The frozen day gives 2 preload boards: the second is the last, the third is refused
    second, why = store.claim_odds_board(day, "preload", remaining=369, limit=4, share=0.7, reserve=50, cost=3)
    assert second and why == "ok"
    store.finish_odds_board(second)
    assert store.claim_odds_board(day, "preload", remaining=369, limit=4, share=0.7, reserve=50, cost=3)[1] == "daily_cap"
    # With no limit wanted, the frozen cap is not consulted (only the month's count is)
    allowed, why = store.claim_odds_board(day, "preload", remaining=369, limit=UNLIMITED_BOARDS, share=1.0,
                                          reserve=0, cost=3, unlimited=True)
    assert allowed and why == "ok"
    store.finish_odds_board(allowed)

# --- the refresh window (ODDS_REFRESH_WINDOW_HOURS) ---------------------------------------------------------------
#
# Unlimited boards burn ~1,150 credits a day if every league with a game in the next 18 hours is refreshed every two
# hours. The window keeps the burn to the games a person can bet: a league is refreshed only while it has a game
# starting within ODDS_REFRESH_WINDOW_HOURS (6 by default); a league whose next game is days out keeps its saved board.

def test_a_league_is_only_preloaded_when_its_game_is_inside_the_refresh_window(store) -> None:
    boards = {(NBA, TODAY): [espn_game("n1", "Boston Celtics", "New York Knicks", at(3))],
              (NHL, TODAY): [espn_game("h1", "Boston Bruins", "Toronto Maple Leafs", at(10), sport="nhl")]}
    builder, odds = preloader(store, boards)
    builder.preload = True
    builder.refresh_window_hours = 6
    builder.refresh()
    wait_for(builder)
    assert odds.fetches == [NBA]  # the 3-hour game is inside the window, the 10-hour one is not


def test_a_person_s_ask_is_never_held_back_by_the_refresh_window(store) -> None:
    """The window rations the automatic refresh only: an explicit ask still loads, and still jumps the queue."""
    boards = {(NHL, TODAY): [espn_game("h1", "Boston Bruins", "Toronto Maple Leafs", at(10), sport="nhl")]}
    builder, odds = preloader(store, boards)
    builder.refresh_window_hours = 6
    builder.refresh()
    wait_for(builder)
    assert odds.fetches == []  # nothing preloaded: its game is 10 hours out
    assert builder.request_odds(NHL)["code"] == "queued"
    wait_for(builder)
    assert odds.fetches == [NHL]


def test_without_a_window_the_old_18_hour_preload_still_applies(store) -> None:
    """ODDS_REFRESH_WINDOW_HOURS=0 keeps the old rule (PRELOAD_HOURS), for a bigger plan."""
    boards = {(NBA, TODAY): [espn_game("n1", "Boston Celtics", "New York Knicks", at(3))],
              (NHL, TODAY): [espn_game("h1", "Boston Bruins", "Toronto Maple Leafs", at(10), sport="nhl")]}
    builder, odds = preloader(store, boards)
    builder.refresh_window_hours = 0
    builder.refresh()
    wait_for(builder)
    builder.refresh()
    wait_for(builder)
    assert sorted(odds.fetches) == sorted([NBA, NHL])
