"""The shared short-term memory end to end (v3 Stage 2b integration): the app's real stack in two "processes" (two
Stores, two TTLCaches, two of every client and service on one SQLite file, as the bot and the dashboard run), with
only the HTTP layer faked. requests' Session.request answers ESPN from canned JSON and counts every call; anything
else (The Odds API included) is refused, so these tests also prove no credits and no AI are spent.

- a game page and a player card fetched by the dashboard are served to the bot from the shared table: no fetch,
  no rebuild;
- the real SlateBuilder sees a score change and, through DashboardService.invalidate_game, drops the game's shared
  card; the next view rebuilds it once for both processes;
- with ESPN down and a copy past its TTL, everyone gets the last good copy at once and ESPN is asked once per
  back-off, not once per person;
- the process cache stays under its cap through the real ESPN client.
"""

from __future__ import annotations

import datetime
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import pytest
import requests

from telegram_betting_bot_simple.dashboard.service import DashboardService
from telegram_betting_bot_simple.data.cache import TTLCache, key_text
from telegram_betting_bot_simple.data.espn_client import ESPN_DAY_TZ, EspnClient
from telegram_betting_bot_simple.data.espn_players import PlayerData
from telegram_betting_bot_simple.data.odds_service import OddsService, normalize_event
from telegram_betting_bot_simple.data.slate import SlateBuilder
from telegram_betting_bot_simple.data.store import Store

NBA = "basketball_nba"
TZ = ZoneInfo("America/New_York")
FIXTURES = Path(__file__).parent / "fixtures" / "espn"
HOME, AWAY = ("2", "Boston Celtics", "BOS"), ("18", "New York Knicks", "NY")
LABELS = ["GP", "GS", "MIN", "FG", "FG%", "3PT", "3P%", "FT", "FT%", "OR", "DR", "REB", "AST", "BLK", "STL", "PF",
          "TO", "PTS"]


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _start_clock() -> Clock:
    """Real time (the service reads the real clock in places), moved two hours back late in ESPN's day so tonight's
    game, two minutes away, falls on the same ESPN day."""
    now = datetime.datetime.now(datetime.timezone.utc).timestamp()
    if datetime.datetime.fromtimestamp(now, ESPN_DAY_TZ).hour >= 23:
        now -= 2 * 3600
    return Clock(float(int(now)))


def _team(team: Tuple[str, str, str]) -> Dict[str, Any]:
    team_id, name, abbr = team
    words = name.split()
    return {"id": team_id, "displayName": name, "shortDisplayName": words[-1], "name": words[-1],
            "location": " ".join(words[:-1]), "abbreviation": abbr}


class Espn:
    """ESPN's answers by URL (a Celtics-Knicks night and LeBron's fixtures), counting every request. ``state`` and
    ``score`` move the game on; ``down`` makes every ESPN request a 503."""

    def __init__(self, start: float) -> None:
        self.start_iso = datetime.datetime.fromtimestamp(start, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        self.day = datetime.datetime.fromtimestamp(start, ESPN_DAY_TZ).strftime("%Y%m%d")
        self.state, self.score = "pre", (0, 0)
        self.down = False
        self.calls: List[str] = []
        self.refused: List[str] = []
        self.lock = threading.Lock()

    def competition(self, event_id: str = "401") -> Dict[str, Any]:
        return {"id": event_id, "date": self.start_iso,
                "status": {"type": {"name": "STATUS", "state": self.state, "completed": self.state == "post"}},
                "competitors": [{"homeAway": "home", "score": str(self.score[0]), "team": _team(HOME)},
                                {"homeAway": "away", "score": str(self.score[1]), "team": _team(AWAY)}]}

    def request(self, session, method, url, params=None, **kwargs) -> requests.Response:
        url = str(url)
        response = requests.Response()
        response.url = url
        if "espn.com" not in url:
            self.refused.append(url.split("?")[0])
            raise AssertionError(f"no request may leave for {url.split('/')[2]}")
        with self.lock:
            self.calls.append(url + "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items())))
        response.status_code = 503 if self.down else 200
        body = {} if self.down else self.body(url, params or {})
        response._content = json.dumps(body).encode()
        return response

    def body(self, url: str, params: Dict[str, Any]) -> Any:
        if url.endswith("/scoreboard"):
            if str(params.get("dates")) != self.day:
                return {"events": []}
            return {"events": [{"id": "401", "date": self.start_iso, "competitions": [self.competition()]}]}
        if url.endswith("/summary"):
            return {"header": {"id": params.get("event"), "competitions": [self.competition(str(params.get("event")))]},
                    "predictor": {"homeTeam": {"id": HOME[0], "gameProjection": "61.5"}}, "injuries": []}
        if url.endswith("/standings"):
            return {"children": [{"standings": {"entries": [
                {"team": {"id": team_id, "displayName": name}, "stats": [
                    {"name": "wins", "value": 3}, {"name": "losses", "value": 2},
                    {"name": "avgPointsFor", "value": 114.5}, {"name": "avgPointsAgainst", "value": 111.0}]}
                for team_id, name, _ in (HOME, AWAY)]}}]}
        if url.endswith("/roster"):
            team_id = url.split("/teams/")[1].split("/")[0]
            return {"athletes": [{"id": f"{team_id}0{i}", "displayName": f"Player {i}",
                                  "position": {"abbreviation": "G"}} for i in range(3)]}
        if url.endswith("/stats"):
            stats = ["60", "60", "30.0", "7-15", "47", "2-5", "36", "3-4", "79", "1", "4", "5", "4", "0.5", "1", "2",
                     "2", "18.0"]
            return {"categories": [{"name": "averages", "labels": LABELS, "statistics": [
                {"season": {"displayName": season}, "stats": stats} for season in ("2025-26", "2026-27")]}]}
        for ending, name in (("/overview", "nba_overview_1966"), ("/gamelog", "nba_gamelog_1966"),
                             ("/splits", "nba_splits_1966"), ("/news/players", "news_fba_1966"),
                             ("/schedule", "nba_schedule_20"), ("/athletes/1966", "nba_athlete_1966")):
            if url.endswith(ending):
                return json.loads((FIXTURES / f"{name}.json").read_text())
        return {}


class Process:
    """One process's stack, built as services.build_service builds it: its own Store object on the shared file, its
    own TTLCache over it, the real OddsService (over the saved snapshot), EspnClient, PlayerData and service."""

    def __init__(self, name: str, path: str, clock: Clock) -> None:
        self.store = Store(path)
        self.cache = TTLCache(store=self.store, name=name, clock=clock)
        self.odds = OddsService("odds-key-not-real-1234", self.cache, store=self.store)
        self.espn = EspnClient(self.cache, clock=clock)
        self.service = DashboardService(self.odds, self.espn, TZ, store=self.store, cache=self.cache)
        self.players = PlayerData(self.espn, now=lambda: datetime.datetime.fromtimestamp(clock(), datetime.timezone.utc))
        self.builds: List[str] = []
        payload = self.service._game_payload
        self.service._game_payload = lambda *args: self.builds.append(args[1]) or payload(*args)

    def counts(self, field: str, kind: str) -> int:
        return (self.cache.stats()["kinds"].get(kind) or {}).get(field, 0)


@pytest.fixture
def world(tmp_path, monkeypatch):
    clock = _start_clock()
    start = clock.now + 120  # tip-off in two minutes
    espn = Espn(start)
    monkeypatch.setattr(requests.sessions.Session, "request",
                        lambda session, method, url, **kwargs: espn.request(session, method, url, **kwargs))
    path = str(tmp_path / "betting.db")
    when = datetime.datetime.fromtimestamp(start, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    markets = [{"key": "h2h", "outcomes": [{"name": HOME[1], "price": 1.65}, {"name": AWAY[1], "price": 2.3}]},
               {"key": "totals", "outcomes": [{"name": "Over", "price": 1.91, "point": 224.5},
                                              {"name": "Under", "price": 1.91, "point": 224.5}]}]
    game = normalize_event({"id": "g1", "sport_key": NBA, "sport_title": "NBA", "commence_time": when,
                            "home_team": HOME[1], "away_team": AWAY[1],
                            "bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": markets}]})
    Store(path).save_odds_snapshot(NBA, [game], clock.now - 300)  # a fresh board: nobody asks The Odds API
    dashboard, bot = Process("dashboard", path, clock), Process("bot", path, clock)
    yield clock, espn, dashboard, bot
    assert espn.refused == []  # no Odds API, no AI, nothing but the canned ESPN


def test_a_game_page_and_a_player_card_are_fetched_once_for_the_bot_and_the_dashboard(world) -> None:
    clock, espn, dashboard, bot = world
    page = dashboard.service.game(NBA, "g1")
    card = dashboard.players.card(NBA, "1966")
    assert page["espn_home_win"] == 0.615 and page["prediction"] and card["name"] == "LeBron James"
    fetched = len(espn.calls)
    assert fetched >= 8 and len(set(espn.calls)) == fetched  # scoreboard, summary, standings, rosters, careers, ...
    # Another person, in the bot's process: everything from the shared table, nothing fetched, nothing rebuilt
    other = bot.service.game(NBA, "g1")
    assert {k: v for k, v in other.items() if k != "consensus"} == {k: v for k, v in page.items() if k != "consensus"}
    assert bot.players.card(NBA, "1966") == card
    assert len(espn.calls) == fetched and bot.builds == [] and dashboard.builds == ["g1"]
    assert bot.counts("shared_hits", "game_card") == 1 and bot.counts("shared_hits", "player_card") == 1
    # ... and a third person in each process is answered from memory
    dashboard.service.game(NBA, "g1")
    bot.players.card(NBA, "1966")
    assert len(espn.calls) == fetched and dashboard.counts("hits", "game_card") >= 1


def test_the_slate_drops_a_games_card_for_everyone_when_the_score_changes(world) -> None:
    clock, espn, dashboard, bot = world
    dashboard.cache.put(("odds", "sports"), [{"key": NBA, "group": "Basketball", "title": "NBA", "active": True,
                                              "has_outrights": False}], kind="odds_sports")  # the free listing
    slate = SlateBuilder(dashboard.odds, dashboard.espn, dashboard.store, service=dashboard.service, tz=TZ,
                         now=clock, workers=2, log_consensus=False, on_change=dashboard.service.invalidate_game)
    dashboard.service.slate = slate  # as services.build_slate
    slate.refresh()  # the first round: the boards, shared through the table
    first = dashboard.service.game(NBA, "g1")
    assert bot.service.game(NBA, "g1")["prediction"] == first["prediction"] and bot.builds == []
    old_rows = [key for key in _game_rows(dashboard.store)]
    assert old_rows and dashboard.builds == ["g1"]
    # Tip-off and a basket: the slate's next round sees the change and drops the game's shared card
    espn.state, espn.score = "in", (2, 0)
    clock.now += 21
    slate.refresh()
    assert dashboard.espn.event_state(NBA, "401") == {"state": "live", "home": 2, "away": 0}
    assert not any(row in _game_rows(dashboard.store) for row in old_rows)
    live = dashboard.service.game(NBA, "g1")
    assert dashboard.builds == ["g1", "g1"] and live["prediction"] is not None
    # It was built from the pre-game summary, served past its time to live (the game is on) while one background
    # refresh renews it: that card is fresh only PARTIAL_KEEP seconds (a card lives no longer than its inputs)
    assert dashboard.cache.drain(5)
    # The bot sees the live game once its own scoreboard copy (20 s while a game is on) is renewed, from the
    # slate's copy in the table, and is served the dashboard's card at once
    clock.now += 19
    slate.refresh()  # no change: nothing dropped
    calls = len(espn.calls)
    other = bot.service.game(NBA, "g1")
    assert bot.espn.event_state(NBA, "401")["state"] == "live"
    assert {k: v for k, v in other.items() if k != "consensus"} == {k: v for k, v in live.items() if k != "consensus"}
    # ... and, that card being past its short life, one background rebuild from the renewed summary in the table
    assert bot.cache.drain(5) and bot.builds == ["g1"]
    assert len(espn.calls) == calls  # the scoreboard, the card and the summary came from the table


def _game_rows(store: Store) -> List[str]:
    with store._connect() as db:
        return [row[0] for row in db.execute("SELECT key FROM shared_cache WHERE kind = 'game_card'")]


def test_with_espn_down_everyone_gets_the_last_good_copy_and_espn_is_asked_once_per_back_off(world) -> None:
    clock, espn, dashboard, bot = world
    good = dashboard.espn.summary(NBA, "401")
    card = dashboard.players.card(NBA, "1966")
    clock.now += 25 * 60  # past the pre-game summary's 5 minutes and its 10-minute stale window
    espn.down = True
    before = len(espn.calls)
    answers = []
    people = [threading.Thread(target=lambda: answers.append(dashboard.espn.summary(NBA, "401"))) for _ in range(20)]
    for person in people:
        person.start()
    for person in people:
        person.join()
    assert answers == [good] * 20
    assert len(espn.calls) == before + 1  # one try, not one per person
    assert bot.espn.summary(NBA, "401") == good  # the bot: the table's copy is old too; one try of its own
    assert len(espn.calls) == before + 2
    clock.now += 10
    for _ in range(5):
        assert dashboard.espn.summary(NBA, "401") == good and bot.espn.summary(NBA, "401") == good
    assert len(espn.calls) == before + 2  # still inside the 30 s back-off in both processes
    # The player card's parts keep their last good copies too (the card is never built from failures)
    clock.now += 7 * 3600
    assert dashboard.players.card(NBA, "1966")["name"] == card["name"]
    # ESPN is back: once the back-off is over the next reader gets a fresh copy
    espn.down = False
    clock.now += 61
    assert dashboard.espn.summary(NBA, "401")["header"]["id"] == "401"
    assert dashboard.cache.failing(("espn", "summary", NBA, "401")) is None


def test_the_process_cache_stays_under_its_cap_through_the_real_client(world) -> None:
    clock, espn, dashboard, bot = world
    small = TTLCache(store=dashboard.store, max_entries=30, name="small", clock=clock)
    client = EspnClient(small, clock=clock)
    for event_id in range(1000, 1120):
        client.summary(NBA, str(event_id))
    stats = small.stats()
    assert stats["entries"] == 30 and len(small._key_locks) <= 30 and stats["total"]["evictions"] == 90
    # An evicted summary comes back from the shared table, not from ESPN
    calls = len(espn.calls)
    assert client.summary(NBA, "1000")["header"]["id"] == "1000" and len(espn.calls) == calls
    assert small.stats()["kinds"]["summary"]["shared_hits"] == 1
    assert dashboard.store.cache_get(key_text(("espn", "summary", NBA, "1119")))
