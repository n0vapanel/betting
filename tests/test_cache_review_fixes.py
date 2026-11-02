"""Regressions for the v3 Stage 2b review of the shared short-term memory: a game summary is never kept as a final one
while its copy says the game is on (and an old copy never moves a game's state back), a process holding a copy waits
out a dead holder's lease once, calls close at tip-off on a cached card, failures are remembered for error_ttl, a
copy keeps the TTL it was fetched with (a game-day game log after the game), memory is pruned and summaries are
lean, a computed card lives no longer than its inputs, an own copy in the table is not decompressed, and a request's
scoreboard is served at once past its TTL. Fake loaders, fake clocks and a real Store: nothing touches the network."""

from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path

import pytest

from telegram_betting_bot_simple.dashboard import service as service_module
from telegram_betting_bot_simple.dashboard.service import CallRejected
from telegram_betting_bot_simple.data import cache as cache_module
from telegram_betting_bot_simple.data.cache import PARTIAL_KEEP, Policy, TTLCache, key_text, ttl_for
from telegram_betting_bot_simple.data.espn_client import SUMMARY_KEEP
from telegram_betting_bot_simple.data.espn_players import PlayerData
from telegram_betting_bot_simple.data.store import Store
from tests.test_cache import Clock as SleepClock, Loader, counts
from tests.test_cache_wiring import (NBA, NOW, TODAY, Clock, FakeOdds, at, board, counted_service, counting_espn,
                                     event, summary)

FIXTURES = Path(__file__).parent / "fixtures" / "espn"
GAME = ("espn", "summary", NBA, "401810001")


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "betting.db")


@pytest.fixture(autouse=True)
def no_secrets(monkeypatch):
    monkeypatch.setattr(cache_module, "_SECRETS", frozenset())


# --- a game summary and the game's state ---------------------------------------------------------------------

def test_a_pre_game_summary_is_reloaded_once_the_board_says_final_and_the_grader_sees_the_result() -> None:
    clock = Clock()
    answers = {"/summary": summary("pre", start=at(1)), "/scoreboard": {"events": board("pre", start=at(1))}}
    espn = counting_espn(TTLCache(clock=clock), answers, clock=clock)
    espn.fetch_scoreboard(NBA, TODAY)
    espn.summary(NBA, "e1")  # the slate's predictor worker, an hour before tip-off
    # The game is played; the slate's board says final half an hour after the whistle
    clock.now += 4 * 3600
    answers["/scoreboard"] = {"events": board("post", 101, 99, start=at(1))}
    answers["/summary"] = summary("post", start=at(1), home=101, away=99)
    espn.fetch_scoreboard(NBA, TODAY)
    assert espn.event_state(NBA, "e1")["state"] == "final"
    calls = len(espn.http_calls)
    assert espn.game_result(NBA, "e1") == {"state": "final", "home": 101, "away": 99}
    assert len(espn.http_calls) == calls + 1  # the pre-game copy wasn't kept as a "final" one for a day
    assert espn.event_state(NBA, "e1") == {"state": "final", "home": 101, "away": 99}
    espn.game_result(NBA, "e1")
    assert len(espn.http_calls) == calls + 1  # the final copy itself keeps


def test_a_live_copy_is_reloaded_after_the_final_whistle_in_both_processes(db) -> None:
    clock = Clock()
    answers = {"/summary": summary("in", start=at(-2), home=90, away=88),
               "/scoreboard": {"events": board("in", 90, 88, start=at(-2))}}
    dashboard = counting_espn(TTLCache(Store(db), name="dashboard", clock=clock), answers, clock=clock)
    bot = counting_espn(TTLCache(Store(db), name="bot", clock=clock), answers, clock=clock)
    assert dashboard.game_result(NBA, "e1")["state"] == "pending"  # someone opened the game while it was on
    clock.now += 20 * 60
    answers["/scoreboard"] = {"events": board("post", 101, 99, start=at(-2))}
    answers["/summary"] = summary("post", start=at(-2), home=101, away=99)
    for espn in (dashboard, bot):
        espn.fetch_scoreboard(NBA, TODAY)
    # The live copy (in the dashboard's memory, and in the table for the bot) is no "final" copy
    assert dashboard.game_result(NBA, "e1") == {"state": "final", "home": 101, "away": 99}
    assert bot.game_result(NBA, "e1") == {"state": "final", "home": 101, "away": 99}
    assert bot.http_calls == [] and dashboard.http_calls[-1].endswith("/summary")  # the board and the final
    # summary came from the dashboard's rows


def test_an_old_copy_never_moves_a_game_back() -> None:
    clock = Clock()
    answers = {"/summary": summary("in", start=at(-2), home=99, away=99),
               "/scoreboard": {"events": board("in", 99, 99, start=at(-2))}}
    espn = counting_espn(TTLCache(clock=clock), answers, clock=clock)
    espn.summary(NBA, "e1")
    clock.now += 5
    answers["/scoreboard"] = {"events": board("post", 101, 99, start=at(-2))}
    espn.fetch_scoreboard(NBA, TODAY)  # the board, fetched after the summary, has the final
    clock.now += 5
    espn.summary(NBA, "e1")  # the summary copy (fetched before the board) is still served ...
    assert len(espn.http_calls) == 2
    assert espn.event_state(NBA, "e1") == {"state": "final", "home": 101, "away": 99}  # ... but can't undo it


def test_summaries_are_cached_lean() -> None:
    full = {**summary("post", start=at(-3), home=101, away=99), "predictor": {"homeTeam": {"id": "1"}},
            "boxscore": {"players": []}, "pickcenter": [{"spread": -3.5}], "plays": [{"text": "x"}] * 50,
            "winprobability": [{"homeWinPercentage": 0.5}] * 50, "news": {"articles": []}, "leaders": []}
    espn = counting_espn(TTLCache(), {"/summary": full})
    data = espn.summary(NBA, "401")
    assert set(data) == {"header", "predictor", "injuries", "boxscore", "pickcenter"} == set(SUMMARY_KEEP)
    assert espn.closing_line(NBA, "401")["home_spread"] == -3.5 and espn.box_score(NBA, "401") == {}


# --- the lease of a holder that died ---------------------------------------------------------------------------

def test_a_process_with_a_copy_waits_out_another_holders_lease_once(db) -> None:
    clock = SleepClock()
    store = Store(db)
    bot = TTLCache(Store(db), name="bot", clock=clock, sleep=clock.sleep, lease_wait=10)
    load = Loader({"v": 1}, {"v": 2})
    bot.get(GAME, 60, load)
    clock.now += 61
    start = clock.now
    assert store.cache_lease(key_text(GAME), "dashboard:dead", 30, start)  # it died mid-load: the lease stays
    assert bot.get(GAME, 60, load) == {"v": 1} and clock.now - start == pytest.approx(10, abs=0.3)  # one wait
    waited = clock.now
    for _ in range(3):  # the readers after it get the last good copy at once, not 10 s each in turn
        assert bot.get(GAME, 60, load) == {"v": 1}
    assert clock.now == waited and load.calls == 1 and counts(bot)["lease_waits"] == 1
    clock.now = start + 31  # the lease lapses: the next reader takes it and loads
    assert bot.get(GAME, 60, load) == {"v": 2} and load.calls == 2


# --- calls close at tip-off on a cached card ---------------------------------------------------------------------

def test_calls_close_at_tip_off_even_on_a_card_built_before_it(tmp_path, monkeypatch) -> None:
    store = Store(str(tmp_path / "betting.db"))
    start = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0) + datetime.timedelta(minutes=5)
    games = [event("g1", "Boston Celtics", "New York Knicks", start.strftime("%Y-%m-%dT%H:%M:%SZ"))]
    service = counted_service(FakeOdds(games), TTLCache(store=store), store)
    card, info, _ = service.detail(NBA, "g1")  # someone opens the game before tip-off
    assert card["live"] is False and info["prediction"] is not None
    monkeypatch.setattr(service_module, "_now_utc", lambda: start + datetime.timedelta(seconds=1))
    with pytest.raises(CallRejected, match="calls close at tip-off"):
        service.save_call("web:abc12345", "Sam", NBA, "g1", {"winner": "home", "confidence": 0.6})
    assert service.game(NBA, "g1")["live"] is True
    assert service.builds == ["g1"]  # the same cached card: only whether the game has started is worked out anew


# --- a copy keeps the TTL it was fetched with --------------------------------------------------------------------

def test_a_copy_fetched_under_a_short_ttl_is_not_fresh_under_a_longer_one(db) -> None:
    clock = SleepClock()
    memory = TTLCache(clock=clock)
    load = Loader({"v": 1}, {"v": 2})
    memory.get(GAME, 3600, load, stale=3600)  # fetched on game day: an hour
    clock.now += 3 * 3600
    assert memory.get(GAME, 6 * 3600, load, stale=3600) == {"v": 2}  # the day's over (6 h now): still reloaded
    # Through the table: the other process judges the copy by the life it was saved with
    dashboard, bot = TTLCache(Store(db), clock=clock), TTLCache(Store(db), clock=clock)
    shared = Loader({"v": 1}, {"v": 2})
    dashboard.get(GAME, 3600, shared, stale=3600)
    clock.now += 3 * 3600
    assert bot.get(GAME, 6 * 3600, shared, stale=3600) == {"v": 2} and shared.calls == 2
    cache = dashboard
    # A slate's 5-second reuse with a longer life: a request keeps the copy for the life
    reuse = Loader({"board": 1}, {"board": 2})
    cache.get(("b",), 5, reuse, life=Policy(60, 60))
    clock.now += 30
    assert cache.get(("b",), 60, reuse) == {"board": 1} and reuse.calls == 1
    assert cache.get(("b",), 5, reuse, life=lambda value: Policy(60, 60)) == {"board": 2}


def test_a_game_day_game_log_is_fetched_again_after_the_game() -> None:
    et = datetime.timezone(datetime.timedelta(hours=-4))
    clock = Clock(datetime.datetime(2026, 10, 27, 17, 0, tzinfo=et).timestamp())  # 5 pm on his game day
    tip, after = "2026-10-27T23:00Z", "2026-10-29T23:30Z"
    status = {"game": "pre"}

    def overview():
        data = json.loads((FIXTURES / "nba_overview_1966.json").read_text())
        data["nextGame"]["league"]["events"] = [{
            "id": "777", "date": tip, "shortName": "BOS @ PHI", "status": status["game"],
            "competitors": [
                {"id": "20", "homeAway": "home", "abbreviation": "PHI", "displayName": "Philadelphia 76ers"},
                {"id": "2", "homeAway": "away", "abbreviation": "BOS", "displayName": "Boston Celtics"}]}]
        return data

    def game(event_id, date, state):
        return {"id": event_id, "date": date, "shortName": "BOS @ PHI", "competitions": [{
            "status": {"type": {"state": state}}, "competitors": [
                {"homeAway": "home", "team": {"id": "20", "abbreviation": "PHI", "displayName": "Philadelphia 76ers"}},
                {"homeAway": "away", "team": {"id": "2", "abbreviation": "BOS", "displayName": "Boston Celtics"}}]}]}

    espn = counting_espn(TTLCache(clock=clock), {}, clock=clock)
    fixtures = {"/athletes/1966": "nba_athlete_1966", "/gamelog": "nba_gamelog_1966", "/news/players": "news_fba_1966"}

    def get(url, params=None):
        espn.http_calls.append(url)
        if url.endswith("/overview"):
            return overview()
        if url.endswith("/schedule"):
            return {"events": [game("777", tip, status["game"]), game("778", after, "pre")]}
        for ending, name in fixtures.items():
            if url.endswith(ending):
                return json.loads((FIXTURES / f"{name}.json").read_text())
        raise RuntimeError(f"ESPN returned HTTP 404 for {url}")

    espn._get = get
    players = PlayerData(espn, now=lambda: datetime.datetime.fromtimestamp(clock(), datetime.timezone.utc))
    players.card(NBA, "1966")
    logs = lambda: sum(url.endswith("/gamelog") for url in espn.http_calls)  # noqa: E731
    before = logs()
    assert before >= 1
    status["game"] = "post"  # 7 pm tip-off, final at 9:30
    clock.now += 5 * 3600  # 10 pm: someone opens his card
    card = players.card(NBA, "1966")
    assert card["next_game"]["date"].startswith("2026-10-29")  # not a game day any more (6 h policy) ...
    assert logs() == before + 1  # ... but the log fetched before tip-off (an hour's) is fetched again
    espn.cache.drain(5)


# --- memory ------------------------------------------------------------------------------------------------------

def test_the_prune_drops_memory_copies_a_day_past_their_stale_window() -> None:
    clock = SleepClock()
    cache = TTLCache(clock=clock)
    cache.get(("short",), 60, lambda: 1, stale=60)
    cache.get(("long",), 7 * 86400, lambda: 2)
    cache.seed(("seeded",), 3, clock.now)
    clock.now += 120 + 86400 + 1
    cache.prune()
    assert cache.loaded_at(("short",)) is None and ("short",) not in cache._key_locks
    assert cache.loaded_at(("long",)) is not None and cache.loaded_at(("seeded",)) is not None
    assert cache.stats()["kinds"]["short"]["expired"] == 1
    # The prune runs by itself every PRUNE_WRITES copies, with or without a store
    cache.PRUNE_WRITES, cache.PRUNE_EVERY = 2, 30 * 86400
    cache.get(("a",), 60, lambda: 4)
    clock.now += 2 * 86400
    cache.get(("b",), 60, lambda: 5)
    cache.get(("c",), 60, lambda: 6)
    assert cache.loaded_at(("a",)) is None and cache.loaded_at(("c",)) is not None


def test_a_copy_no_newer_than_this_processes_is_not_decompressed(db, monkeypatch) -> None:
    clock = SleepClock()
    store = Store(db)
    cache = TTLCache(store, clock=clock)
    cache.get(GAME, 20, lambda: {"v": 1})
    unpacked = []
    real = cache_module._unpack
    monkeypatch.setattr(cache_module, "_unpack", lambda blob: unpacked.append(1) or real(blob))
    for value in range(2, 6):  # past its TTL: the table holds only this process's own copy, then each reload's
        clock.now += 21
        assert cache.get(GAME, 20, lambda value=value: {"v": value}) == {"v": value}
    assert unpacked == []
    text = key_text(GAME)
    assert store.cache_get(text, clock.now) is None and store.cache_get(text, clock.now - 1)[0] == clock.now
    other = TTLCache(Store(db), clock=clock)
    assert other.get(GAME, 20, lambda: pytest.fail("shared")) == {"v": 5} and len(unpacked) == 1


# --- a computed card lives no longer than its inputs -------------------------------------------------------------

def test_a_computed_card_lives_no_longer_than_the_inputs_it_was_built_from() -> None:
    clock = SleepClock()
    cache = TTLCache(clock=clock)
    bio = Loader({"status": "Active"}, {"status": "Out"})

    def build():
        return {"injury": cache.get(("bio",), 600, bio, stale=600)["status"]}

    def card():
        return cache.get_computed(("card",), 600, build, lambda value: False, stale=600)["injury"]

    assert card() == "Active"
    clock.now += 601  # both past their TTL; ESPN now lists him out
    assert card() == "Active"  # at once, while the card is rebuilt in the background ...
    assert cache.drain(5) and bio.calls == 2  # ... from the bio served stale (its own refresh ran too)
    clock.now += PARTIAL_KEEP + 1  # so that card is fresh only briefly: the next reader's rebuild has him out
    card()
    assert cache.drain(5) and card() == "Out"
    # An input fresh for 100 s more caps a card's own 600 s
    cache.get(("odds",), 100, lambda: 1)
    clock.now += 50
    built = Loader({"n": 1}, {"n": 2})
    cache.get_computed(("card2",), 600, lambda: {**built(), "odds": cache.get(("odds",), 100, lambda: 2)},
                       lambda value: False)
    clock.now += 51
    assert cache.get_computed(("card2",), 600, built, lambda value: False)["n"] == 2


def test_a_players_injury_keeps_for_the_injuries_policy() -> None:
    assert ttl_for("bio") == ttl_for("injuries") == Policy(600, 600)


# --- a request's scoreboard ---------------------------------------------------------------------------------------

def test_a_requests_scoreboard_is_served_at_once_past_its_ttl_and_keeps_the_slates_pace() -> None:
    clock = Clock()
    cache = TTLCache(clock=clock)
    answers = {"/scoreboard": {"events": board("pre", start=at(1))}}  # games to come today: a minute
    espn = counting_espn(cache, answers, clock=clock)
    espn.fetch_scoreboard(NBA, TODAY)  # the slate's fetch
    commence = at(1).replace("Z", ":00Z")
    clock.now += 30  # past SLATE_REUSE, inside the board's minute: the slate's copy
    assert espn.find_game(NBA, "Boston Celtics", "New York Knicks", commence)["event_id"] == "e1"
    assert len(espn.http_calls) == 1
    clock.now += 40  # past the minute: the page doesn't wait for ESPN
    misses = counts(cache, "scoreboard")["misses"]
    assert espn.find_game(NBA, "Boston Celtics", "New York Knicks", commence)["event_id"] == "e1"
    row = counts(cache, "scoreboard")
    assert row["misses"] == misses and row["stale_hits"] == 1
    assert cache.drain(5) and len(espn.http_calls) == 2  # one background refresh
    assert copy.deepcopy(espn.scoreboard(NBA, TODAY))[0]["id"] == "e1" and NOW < clock.now
