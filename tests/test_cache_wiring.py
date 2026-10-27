"""The shared short-term memory wired through the app (v3 Stage 2b): one TTLCache per process reaches every client
and service, ESPN's game summaries keep for their game's state, the slate's scoreboards are shared, computed results
(a game's card, "Markets say", the team context, a player's card) are built once for everyone and dropped when the
slate sees a game's status or score change, GET /api/cache/stats and the hourly summary line. Fake loaders, fake
clocks and a real Store in a temporary folder: nothing touches the network."""

from __future__ import annotations

import copy
import datetime
import json
import logging
import threading
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from telegram_betting_bot_simple import services
from telegram_betting_bot_simple.config import Config
from telegram_betting_bot_simple.dashboard import __main__ as dashboard_main
from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.dashboard.service import DashboardService
from telegram_betting_bot_simple.data import cache as cache_module
from telegram_betting_bot_simple.data import utils
from telegram_betting_bot_simple.data.apis import ApiSportsClient
from telegram_betting_bot_simple.data.cache import SecretInKey, TTLCache, ttl_for
from telegram_betting_bot_simple.data.espn_client import ESPN_DAY_TZ, SLATE_REUSE, EspnClient, lean_events
from telegram_betting_bot_simple.data.espn_players import PlayerData
from telegram_betting_bot_simple.data.odds_service import OddsService, normalize_event
from telegram_betting_bot_simple.data.slate import SlateBuilder
from telegram_betting_bot_simple.data.store import Store

NBA, NHL = "basketball_nba", "icehockey_nhl"
# 6 pm in New York on Monday 26 October 2026
NOW = datetime.datetime(2026, 10, 26, 18, 0, tzinfo=ESPN_DAY_TZ).timestamp()
TODAY = datetime.date(2026, 10, 26)


def at(hours: float) -> str:
    moment = datetime.datetime.fromtimestamp(NOW + hours * 3600, datetime.timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%MZ")


class Clock:
    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def clean_globals(monkeypatch):
    """Each test's secrets and fetch_json cache are put back afterwards."""
    monkeypatch.setattr(cache_module, "_SECRETS", cache_module._SECRETS)
    monkeypatch.setattr(utils, "_cache", None)


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "betting.db"))


def summary(state: str = "pre", start: str = None, home: int = 0, away: int = 0) -> Dict[str, Any]:
    """An ESPN game summary's header in a state ("pre", "in" or "post")."""
    return {"header": {"competitions": [{
        "date": start or at(2),
        "status": {"type": {"name": "STATUS", "state": state, "completed": state == "post"}},
        "competitors": [{"homeAway": "home", "score": str(home)}, {"homeAway": "away", "score": str(away)}],
    }]}, "injuries": []}


def counting_espn(cache: TTLCache, answers: Dict[str, Any], clock: Optional[Clock] = None,
                  delay: float = 0.0) -> EspnClient:
    """A real EspnClient whose one HTTP helper answers from ``answers`` (by URL ending) and counts its calls."""
    espn = EspnClient(cache, clock=clock)
    espn.http_calls = []

    def get(url, params=None):
        espn.http_calls.append(url)
        if delay:
            time.sleep(delay)
        for ending, answer in answers.items():
            if url.endswith(ending):
                if isinstance(answer, Exception):
                    raise answer
                return copy.deepcopy(answer)
        raise RuntimeError(f"ESPN returned HTTP 404 for {url}")

    espn._get = get
    return espn


# --- one cache per process -------------------------------------------------------------------------------

def test_one_cache_per_process_reaches_every_client_and_service(tmp_path, monkeypatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "MARKETS_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    config = Config(TELEGRAM_BOT_TOKEN="123456:test-token", THE_ODDS_API_KEY="odds-secret-0123456789",
                    STATE_DIRECTORY=str(tmp_path), _env_file=None)
    service = services.build_service(config)
    cache = service.cache
    assert isinstance(cache, TTLCache) and cache.store is service.store and cache.name == "bot"
    assert service.odds.cache is cache and service.espn.cache is cache and service.lab.players.cache is cache
    assert utils.shared_cache() is cache  # fetch_json and clients built without a cache
    # The settings' secrets can't reach a cache key
    with pytest.raises(SecretInKey):
        cache.get(("odds", "odds-secret-0123456789"), 60, lambda: 1)
    # The slate drops a game's computed entries through the service
    slate = services.build_slate(config, service)
    assert slate.on_change == service.invalidate_game


# --- ESPN: one load per key, state-aware summaries, the SQLite tier ------------------------------------------------

def test_two_people_and_twenty_threads_asking_for_one_game_cause_one_load() -> None:
    espn = counting_espn(TTLCache(), {"/summary": summary()}, delay=0.05)
    first, second = espn.summary(NBA, "401"), espn.summary(NBA, "401")
    assert first == second and len(espn.http_calls) == 1
    other = counting_espn(TTLCache(), {"/summary": summary()}, delay=0.05)
    answers: List[Any] = []
    threads = [threading.Thread(target=lambda: answers.append(other.summary(NBA, "402"))) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(answers) == 20 and len(other.http_calls) == 1


def test_a_summary_keeps_for_its_games_state() -> None:
    clock = Clock()
    cache = TTLCache(clock=clock)
    answers = {"/summary": summary("pre", start=at(2))}
    espn = counting_espn(cache, answers, clock=clock)
    espn.summary(NBA, "401")
    assert espn.event_state(NBA, "401") == {"state": "pre", "home": 0, "away": 0}
    pre = ttl_for("summary", "pre")
    clock.now += pre.ttl - 1
    espn.summary(NBA, "401")
    assert len(espn.http_calls) == 1  # a pre-game summary keeps for minutes
    # Live: seconds
    answers["/summary"] = summary("in", start=at(-1), home=10, away=8)
    clock.now += 2
    espn.summary(NBA, "401")
    cache.drain(5)  # inside the stale window: served at once, refreshed in the background
    assert len(espn.http_calls) == 2 and espn.game_state(NBA, "401") == "live"
    live = ttl_for("summary", "live")
    clock.now += live.ttl + live.stale + 1
    assert espn.summary(NBA, "401")["header"]["competitions"][0]["competitors"][0]["score"] == "10"
    assert len(espn.http_calls) == 3
    # Final: a day
    answers["/summary"] = summary("post", start=at(-3), home=101, away=99)
    clock.now += live.ttl + live.stale + 1
    espn.summary(NBA, "401")
    assert espn.event_state(NBA, "401") == {"state": "final", "home": 101, "away": 99}
    clock.now += 6 * 3600
    espn.summary(NBA, "401")
    assert len(espn.http_calls) == 4


def test_a_pre_game_summary_whose_start_has_passed_keeps_only_for_seconds() -> None:
    clock = Clock()
    espn = counting_espn(TTLCache(clock=clock), {"/summary": summary("pre", start=at(0.05))}, clock=clock)
    espn.summary(NBA, "401")
    assert espn.game_state(NBA, "401") == "pre"
    clock.now += 0.05 * 3600 + 1  # tip-off passed; ESPN's copy still says "pre"
    assert espn.game_state(NBA, "401") == "live"


def test_the_bot_and_the_dashboard_share_one_load_through_the_sqlite_file(store) -> None:
    dashboard = counting_espn(TTLCache(store=store, name="dashboard"), {"/summary": summary()})
    bot = counting_espn(TTLCache(store=store, name="bot"), {"/summary": summary()})
    assert dashboard.summary(NBA, "401") == bot.summary(NBA, "401")
    assert len(dashboard.http_calls) == 1 and bot.http_calls == []
    assert bot.cache.stats()["kinds"]["summary"]["shared_hits"] == 1


# --- scoreboards: the slate's fetch is shared ------------------------------------------------------------------

def board(state: str = "pre", home: int = 0, away: int = 0, start: str = None) -> List[Dict[str, Any]]:
    return [{"id": "e1", "date": start or at(1), "links": [{"href": "x" * 50}], "competitions": [{
        "id": "e1", "date": start or at(1), "odds": [{"details": "BOS -3"}], "headlines": [{"video": [1, 2, 3]}],
        "status": {"type": {"name": "STATUS", "state": state, "completed": state == "post"}},
        "competitors": [
            {"homeAway": "home", "score": str(home), "leaders": [{"x": 1}], "statistics": [{"name": "pts"}],
             "team": {"id": "1", "displayName": "Boston Celtics", "links": [{"href": "y"}]}},
            {"homeAway": "away", "score": str(away),
             "team": {"id": "2", "displayName": "New York Knicks", "links": [{"href": "z"}]}}],
    }]}]


def test_the_slates_scoreboard_fetch_is_shared_lean_and_still_reports_failures() -> None:
    clock = Clock()
    cache = TTLCache(clock=clock)
    answers: Dict[str, Any] = {"/scoreboard": {"events": board()}}
    espn = counting_espn(cache, answers, clock=clock)
    events = espn.fetch_scoreboard(NBA, TODAY)
    assert "links" not in events[0] and "odds" not in events[0]["competitions"][0]
    assert "leaders" not in events[0]["competitions"][0]["competitors"][0]
    assert events[0]["competitions"][0]["competitors"][0]["team"] == {"id": "1", "displayName": "Boston Celtics"}
    assert events == lean_events(board())
    # A request finding the game reads the slate's copy
    assert espn.find_game(NBA, "Boston Celtics", "New York Knicks", at(1).replace("Z", ":00Z"))["event_id"] == "e1"
    assert len(espn.http_calls) == 1
    # The slate's next round reuses a copy younger than SLATE_REUSE, else loads
    clock.now += SLATE_REUSE - 1
    espn.fetch_scoreboard(NBA, TODAY)
    assert len(espn.http_calls) == 1
    answers["/scoreboard"] = {"events": board("in", 12, 9)}
    clock.now += 2
    espn.fetch_scoreboard(NBA, TODAY)
    assert len(espn.http_calls) == 2 and espn.event_state(NBA, "e1") == {"state": "live", "home": 12, "away": 9}
    # ESPN fails: the slate hears about it (its own back-off and fallback), though the cache holds a good copy
    answers["/scoreboard"] = RuntimeError("ESPN returned HTTP 503")
    clock.now += 30
    with pytest.raises(RuntimeError):
        espn.fetch_scoreboard(NBA, TODAY)
    assert espn.scoreboard(NBA, TODAY)  # a request still gets the last good board


def test_a_requests_scoreboard_keeps_as_long_as_the_slate_paces_that_board() -> None:
    espn = EspnClient(TTLCache(), clock=Clock())
    assert espn.board_state(TODAY, lean_events(board("in"))) == "live"
    assert espn.board_state(TODAY, lean_events(board("pre", start=at(0.1)))) == "live"  # about to start
    assert espn.board_state(TODAY, lean_events(board("pre", start=at(3)))) == "today"
    assert espn.board_state(TODAY, lean_events(board("post"))) == "idle"
    assert espn.board_state(TODAY + datetime.timedelta(days=2)) == "future"
    assert espn.board_state(TODAY - datetime.timedelta(days=3)) == "past"


# --- computed results: a game's card --------------------------------------------------------------------

def event(event_id: str, home: str, away: str, start: str, home_ml: float = 1.8) -> Dict[str, Any]:
    markets = [{"key": "h2h", "outcomes": [{"name": home, "price": home_ml}, {"name": away, "price": 2.1}]},
               {"key": "totals", "outcomes": [{"name": "Over", "price": 1.91, "point": 220.5},
                                              {"name": "Under", "price": 1.91, "point": 220.5}]}]
    return normalize_event({"id": event_id, "sport_key": NBA, "sport_title": "NBA", "commence_time": start,
                            "home_team": home, "away_team": away,
                            "bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": markets}]})


class FakeOdds:
    quota = {"remaining": 400, "used": 100}

    def __init__(self, games: List[Dict[str, Any]], loaded_at: float = NOW - 600) -> None:
        self.games, self.loaded_at = games, loaded_at

    def sports(self):
        return [{"key": NBA, "group": "Basketball", "title": "NBA", "active": True}]

    def league_games(self, league, refresh=False):
        return copy.deepcopy(self.games)

    def odds_loaded_at(self, league):
        return self.loaded_at

    def saved_games(self, league):
        return self.loaded_at, copy.deepcopy(self.games)


class QuietEspn:
    def find_game(self, *args):
        return None

    def standings(self, league, season=None):
        return {}


def counted_service(odds: FakeOdds, cache: TTLCache, store: Optional[Store] = None) -> DashboardService:
    service = DashboardService(odds, QuietEspn(), ZoneInfo("America/New_York"), store=store, cache=cache)
    service.builds = []
    real = service._card

    def card(league, game, context):
        service.builds.append(game["id"])
        return real(league, game, context)

    service._card = card
    return service


def test_a_games_card_is_built_once_for_everyone_and_shared_with_the_other_process(store) -> None:
    games = [event("g1", "Boston Celtics", "New York Knicks", "2030-01-01T00:00:00Z")]
    dashboard = counted_service(FakeOdds(games), TTLCache(store=store, name="dashboard"), store)
    first = dashboard.game(NBA, "g1")
    first["markets"]["h2h"]["home"]["price"] = 99.0  # a caller's change doesn't reach the cached card
    second = dashboard.game(NBA, "g1")
    card, info, players = dashboard.detail(NBA, "g1")  # the call judge, save_call and the projections share
    assert dashboard.builds == ["g1"] and second["markets"]["h2h"]["home"]["price"] == 1.8
    assert card["prediction"] == second["prediction"] and info["prediction"] is not None
    # The bot's process: its own cache on the same file gets the dashboard's card without building it
    bot = counted_service(FakeOdds(games), TTLCache(store=store, name="bot"), store)
    assert bot.game(NBA, "g1") == {**second, "consensus": bot.game(NBA, "g1")["consensus"]}
    assert bot.builds == [] and bot.cache.stats()["kinds"]["game_card"]["shared_hits"] >= 1
    # New prices are new inputs: built again, once
    dashboard.odds.games = [event("g1", "Boston Celtics", "New York Knicks", "2030-01-01T00:00:00Z", 1.7)]
    assert dashboard.game(NBA, "g1")["markets"]["h2h"]["home"]["price"] == 1.7
    dashboard.game(NBA, "g1")
    assert dashboard.builds == ["g1", "g1"]


def test_invalidate_game_drops_its_entries_so_the_next_view_rebuilds_once(store) -> None:
    games = [event("g1", "Boston Celtics", "New York Knicks", "2030-01-01T00:00:00Z"),
             event("g2", "Miami Heat", "Utah Jazz", "2030-01-01T01:00:00Z")]
    service = counted_service(FakeOdds(games), TTLCache(store=store), store)
    service.game(NBA, "g1")
    service.game(NBA, "g2")
    service.games(NBA)
    builds = len(service.builds)
    assert service.invalidate_game(NBA, odds_id="g1", espn_id="401", board=False) >= 2  # card and parts
    service.game(NBA, "g1")
    service.game(NBA, "g1")
    service.game(NBA, "g2")  # another game's card stays
    service.games(NBA)  # the board stays without board=True
    assert len(service.builds) == builds + 1
    service.invalidate_game(NBA, odds_id="g2", board=True)  # a status change: the league's cards go too
    service.games(NBA)
    assert len(service.builds) == builds + 1 + len(games)
    assert service.invalidate_game(NBA, odds_id=None, espn_id="402") == 0  # no odds id: nothing of ours


def test_a_card_built_while_espn_failed_is_kept_only_seconds_then_rebuilt() -> None:
    class FailingEspn(QuietEspn):
        summaries = 0

        def find_game(self, *args):
            return {"event_id": "401", "home_id": "1", "away_id": "2", "day": TODAY}

        def summary(self, league, event_id):
            FailingEspn.summaries += 1
            raise RuntimeError("ESPN returned HTTP 503")

        def played_on(self, league, day):
            return set()

    games = [event("g1", "Boston Celtics", "New York Knicks", "2030-01-01T00:00:00Z")]
    clock = Clock()
    service = counted_service(FakeOdds(games), TTLCache(clock=clock))
    service.espn = FailingEspn()
    for _ in range(5):  # five people meanwhile: one build, shared
        service.game(NBA, "g1")
    assert service.builds == ["g1"]
    clock.now += cache_module.PARTIAL_KEEP  # then the next view rebuilds it (whole again once ESPN is back)
    service.game(NBA, "g1")
    assert service.builds == ["g1", "g1"] and FailingEspn.summaries == 2


# --- the slate tells the service when a game's status or score changes -----------------------------------------

class BoardsEspn:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def fetch_scoreboard(self, league, day):
        return copy.deepcopy(self.events) if league == NBA and day == TODAY else []


class SlateOdds:
    odds_ttl, reserve, quota = 7200, 50, {"remaining": 300, "used": 200}

    def __init__(self, store: Store) -> None:
        self.store = store

    def sports(self):
        return [{"key": NBA, "group": "Basketball", "title": "NBA", "active": True, "has_outrights": False}]

    def league_games(self, league, refresh=False):
        return []


def espn_event(state: str, home: int = 0, away: int = 0) -> Dict[str, Any]:
    def side(where, name, score, team_id):
        words = name.split()
        return {"homeAway": where, "score": str(score), "team": {
            "id": team_id, "displayName": name, "shortDisplayName": words[-1], "name": words[-1],
            "location": " ".join(words[:-1]), "abbreviation": words[-1][:3].upper()}}
    return {"id": "e1", "date": at(1), "competitions": [{
        "id": "e1", "date": at(1),
        "status": {"type": {"name": "STATUS", "state": state, "completed": state == "post"}},
        "competitors": [side("home", "Boston Celtics", home, "1"), side("away", "New York Knicks", away, "2")]}]}


def test_the_slate_drops_a_games_computed_entries_when_its_status_or_score_changes(store) -> None:
    store.save_odds_snapshot(NBA, [event("o1", "Boston Celtics", "New York Knicks", at(1).replace("Z", ":00Z"))],
                             NOW - 60)
    espn = BoardsEspn()
    changes: List[Any] = []
    slate = SlateBuilder(SlateOdds(store), espn, store, now=lambda: NOW, workers=1, log_consensus=False,
                         on_change=lambda league, **ids: changes.append((league, ids)))

    def round_with(*events):
        espn.events = list(events)
        slate._boards.clear()  # every board due again
        slate.refresh()

    round_with(espn_event("pre"))
    round_with(espn_event("pre"))
    assert changes == []  # the first round only notes each game; nothing moved in the second
    round_with(espn_event("in", 2, 0))
    assert changes == [(NBA, {"odds_id": "o1", "espn_id": "e1", "board": True})]
    round_with(espn_event("in", 4, 0))
    assert changes[-1] == (NBA, {"odds_id": "o1", "espn_id": "e1", "board": False})  # a score: the board stays
    round_with(espn_event("in", 4, 0))
    assert len(changes) == 2
    # A failing hook costs only itself
    slate.on_change = lambda league, **ids: 1 / 0
    round_with(espn_event("post", 6, 0))
    assert slate.current() is not None


# --- computed results: "Markets say", the team context, a player's card ------------------------------------------

def test_markets_say_without_a_slate_is_one_computed_entry_per_game(store) -> None:
    games = [event("g1", "Boston Celtics", "New York Knicks", "2030-01-01T00:00:00Z")]
    service = DashboardService(FakeOdds(games), QuietEspn(), ZoneInfo("America/New_York"), store=store,
                               cache=TTLCache())
    worked: List[str] = []
    real = service._markets_detail
    service._markets_detail = lambda *args: worked.append(args[1]) or real(*args)
    first = service.markets_detail(NBA, "g1")
    assert first["sources"] and service.markets_detail(NBA, "g1") == first and worked == ["g1"]
    assert service.markets_detail(NBA, "g1", now=NOW) is not None and worked == ["g1", "g1"]  # a test clock: afresh

    class Slate:
        def markets_for(self, league, odds_id=None, espn_id=None):
            return {"blend": None, "sources": [], "links": [], "from": "slate"}

    service.slate = Slate()
    assert service.markets_detail(NBA, "g1")["from"] == "slate" and worked == ["g1", "g1"]


def test_the_team_context_is_one_entry_per_league_and_board_shared_by_both_processes(store) -> None:
    class Standings(QuietEspn):
        calls = 0

        def standings(self, league, season=None):
            Standings.calls += 1
            return {"1": {"name": "Boston Celtics", "games": 4, "wins": 3, "losses": 1, "points_for": 115.0,
                          "points_against": 108.0}}

    odds = FakeOdds([event("g1", "Boston Celtics", "New York Knicks", "2030-01-01T00:00:00Z")])
    dashboard = DashboardService(odds, Standings(), ZoneInfo("America/New_York"), cache=TTLCache(store=store))
    bot = DashboardService(odds, Standings(), ZoneInfo("America/New_York"), cache=TTLCache(store=store))
    context = dashboard._league_context(NBA)
    assert bot._league_context(NBA) == context and dashboard._league_context(NBA) == context
    assert Standings.calls == 2  # this season and last, once
    odds.loaded_at += 3600  # a new board: its totals set the scoring level again
    dashboard._league_context(NBA)
    assert Standings.calls == 4


LEBRON_BIO = {"athlete": {"id": "1966", "displayName": "LeBron James", "team": {"id": "13", "displayName": "Lakers"}}}


def test_a_players_card_is_built_once_and_props_are_shared_as_json_with_exact_prices(store) -> None:
    from telegram_betting_bot_simple.data import espn_players

    class Fake:
        def __init__(self, cache):
            self.cache, self.calls = cache, []

        def _get(self, url, params=None):
            self.calls.append(url)
            if url.endswith("/propBets"):
                return {"pageCount": 1, "items": [{
                    "athlete": {"$ref": "http://x/athletes/1966"}, "type": {"name": "Total Points"},
                    "current": {"target": {"value": 25.5, "displayValue": "25.5"}},
                    "odds": {"american": {"value": "-110"}}, "side": "Over"}]}
            raise RuntimeError(f"ESPN returned HTTP 404 for {url}")

    dashboard = Fake(TTLCache(store=store, name="dashboard"))
    data = PlayerData(dashboard, now=lambda: datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc))
    first = data.card(NBA, "1966")
    data.card(NBA, "1966")
    assert first["reason"] and len(dashboard.calls) == 4  # bio, overview, game log, news: once (404s: nothing)
    props = data.props(NBA, "401")
    bot = Fake(TTLCache(store=store, name="bot"))
    shared = PlayerData(bot).props(NBA, "401")
    assert bot.calls == [] and shared["players"] == props["players"]
    prices = [prop["over_price"] for prop in shared["players"].get("1966", [])]
    assert prices and all(isinstance(price, Decimal) for price in prices) and prices[0] == Decimal("1.9091")
    # A card built while ESPN failed (not merely had nothing) is kept only PARTIAL_KEEP seconds, then rebuilt
    clock = Clock()
    failing = Fake(TTLCache(clock=clock))
    failing._get = lambda url, params=None: failing.calls.append(url) or (_ for _ in ()).throw(
        RuntimeError("ESPN timed out"))
    people = PlayerData(failing)
    built: List[str] = []
    real_card = people._card
    people._card = lambda league, athlete_id: built.append(athlete_id) or real_card(league, athlete_id)
    for _ in range(5):
        assert "couldn't be reached" in people.card(NBA, "1966")["reason"]
    assert built == ["1966"] and len(failing.calls) == 4  # five people, one build, each part asked once
    clock.now += cache_module.PARTIAL_KEEP
    people.card(NBA, "1966")
    assert built == ["1966", "1966"] and len(failing.calls) == 4  # rebuilt; the parts are still backing off
    assert espn_players.KINDS["props"] == "props"


# --- the other clients ---------------------------------------------------------------------------------------

def test_fetch_json_and_api_sports_go_through_the_cache_without_their_keys(monkeypatch) -> None:
    import requests

    calls: List[Any] = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": [{"id": 1}]}

    monkeypatch.setattr(utils.requests, "get", lambda url, **kwargs: calls.append((url, kwargs)) or Response())
    cache = TTLCache()
    first = utils.fetch_json("https://api.example.test/v1/x?season=2026", params={"apiKey": "SECRET-12345"},
                             cache=cache)
    assert utils.fetch_json("https://api.example.test/v1/x?season=2026", params={"apiKey": "OTHER-12345"},
                            cache=cache) == first
    assert len(calls) == 1
    assert utils.cache_key("https://u:p@api.example.test/v1/x?apiKey=S&b=2", {"token": "T", "a": "1"}) == (
        "http", "https://api.example.test/v1/x", (("a", "1"), ("b", "2")))
    assert "SECRET-12345" not in json.dumps([list(map(str, key)) for key in cache._entries])

    sessions: List[Any] = []
    monkeypatch.setattr(requests.Session, "get", lambda self, url, **kwargs: sessions.append(url) or Response())
    client = ApiSportsClient("api-sports-secret", cache=cache)
    assert client.get_injuries("soccer", league_id="39") == client.get_injuries("soccer", league_id="39")
    assert len(sessions) == 1 and "api-sports-secret" not in str(list(cache._entries))
    assert cache.stats()["kinds"]["api_sports"]["hits"] == 1


def test_the_odds_listing_is_cached_and_the_board_stays_out_of_the_shared_table(store, monkeypatch) -> None:
    from telegram_betting_bot_simple.data import odds_service

    calls: List[str] = []

    class Response:
        ok, status_code, headers = True, 200, {}

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    def get(url, params=None, timeout=None):
        calls.append(url)
        return Response([{"key": NBA, "title": "NBA"}] if url.endswith("/sports") else [])

    monkeypatch.setattr(odds_service.requests, "get", get)
    cache = TTLCache(store=store)
    odds = OddsService("odds-secret-abcdef", cache)
    assert odds.sports() == odds.sports() and calls == ["https://api.the-odds-api.com/v4/sports"]
    odds.league_games(NBA)
    with store._connect() as db:
        kinds = {row[0] for row in db.execute("SELECT kind FROM shared_cache")}
    assert kinds == {"odds_sports"}  # the board is the odds snapshot's business


# --- visibility: GET /api/cache/stats and the hourly line -----------------------------------------------------

def test_cache_stats_are_for_signed_in_people_only(store) -> None:
    games = [event("g1", "Boston Celtics", "New York Knicks", "2030-01-01T00:00:00Z")]
    service = DashboardService(FakeOdds(games), QuietEspn(), ZoneInfo("America/New_York"), store=store,
                               cache=TTLCache(store=store, name="dashboard"))
    service.game(NBA, "g1")
    service.game(NBA, "g1")
    client = TestClient(create_app(service, "s3cret", login_delay=0))
    assert client.get("/api/cache/stats").status_code == 401
    assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
    body = client.get("/api/cache/stats").json()
    assert body["cache"]["name"] == "dashboard" and body["cache"]["kinds"]["game_card"]["hits"] >= 1
    assert body["cache"]["table"]["rows"] >= 1 and body["line"].startswith("cache: ")
    assert any(cache["name"] == "dashboard" for cache in body["process"]["caches"])
    assert "s3cret" not in json.dumps(body)


def test_the_dashboard_logs_a_one_line_summary_every_hour(caplog) -> None:
    waits: List[float] = []
    with caplog.at_level(logging.INFO):
        dashboard_main.log_cache_forever(rounds=2, sleep=waits.append)
    assert waits == [3600, 3600]
    lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith("cache: ")]
    assert len(lines) == 2 and "reads" in lines[0] and "\n" not in lines[0]
