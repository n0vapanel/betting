"""The capture worker (v4 P1, plan section 3.3): capture/worker.py Capturer and capture/plan.py, on a fake clock against a
fake world (ESPN's site, core and fightcenter APIs, MLB StatsAPI, GitHub, football-data and Kalshi answering capture's
URLs through PoliteHttp's one send). The real fetchers (capture/sources), the real EspnClient (summary_raw and
scoreboard_raw seeding its lean cache) and the real RawStore run; nothing touches the network, and The Odds API is never
asked."""

from __future__ import annotations

import collections
import datetime
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.parse
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

import pytest
import requests

from telegram_betting_bot_simple.capture import http as capture_http
from telegram_betting_bot_simple.capture import plan, raw_store
from telegram_betting_bot_simple.capture import worker as worker_module
from telegram_betting_bot_simple.capture.http import HttpResponse, PoliteHttp
from telegram_betting_bot_simple.capture.raw_store import Job, RawStore
from telegram_betting_bot_simple.capture.sources import scoreboard_key
from telegram_betting_bot_simple.capture.worker import Capturer, summary_state
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.espn_client import EspnClient, lean_summary
from telegram_betting_bot_simple.data.mlb_stats import MlbStats

ET = ZoneInfo("America/New_York")
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "capture"
DAY = datetime.date(2026, 10, 30)  # a Friday (US Eastern, ESPN's day)
NBA, NFL, NHL, MLB, EPL, MLS, NCAAB = ("basketball_nba", "americanfootball_nfl", "icehockey_nhl", "baseball_mlb",
                                       "soccer_epl", "soccer_usa_mls", "basketball_ncaab")
UFC, ATP = "mma_mixed_martial_arts", "tennis_atp"
Usage = namedtuple("Usage", "total used free")


@pytest.fixture(autouse=True)
def roomy_disk(monkeypatch):
    """This machine's own disk may be under the capture's floor: here every test has room unless it says not."""
    monkeypatch.setattr(raw_store.shutil, "disk_usage", lambda folder: Usage(100e9, 10e9, 90e9))


@pytest.fixture(autouse=True)
def no_shared_door():
    """No test leaves the process a door (capture.http.set_shared): a later test's HistHttp would wait on it."""
    capture_http.set_shared(None)
    yield
    capture_http.set_shared(None)


def at(day: datetime.date, hour: int, minute: int = 0) -> float:
    return datetime.datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET).timestamp()


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


class Clock:
    def __init__(self, now: float) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


STATUS = {"pre": {"name": "STATUS_SCHEDULED", "state": "pre", "completed": False},
          "live": {"name": "STATUS_IN_PROGRESS", "state": "in", "completed": False},
          "final": {"name": "STATUS_FINAL", "state": "post", "completed": True},
          "postponed": {"name": "STATUS_POSTPONED", "state": "post", "completed": False}}
LEAGUE_OF = {path: league for league, path in plan.PATHS.items()}


@dataclass
class Game:
    league: str
    event_id: str
    start: float
    home: str = "Boston Celtics"
    away: str = "Chicago Bulls"
    state: str = "pre"
    version: int = 0
    fixture: Optional[str] = None  # a real summary (tests/fixtures/capture) served once the game is final
    runs: float = 2.5 * 3600  # how long it lasts (the world's clock moves it: see World.play)


@dataclass
class Bout:
    id: str
    competitors: Tuple[str, str]
    state: str = "pre"
    result: Optional[str] = "kotko"


@dataclass
class Card:
    event_id: str
    start: float
    bouts: List[Bout]
    runs: float = 6 * 3600


@dataclass
class Match:
    comp_id: str
    tour: str
    start: float
    state: str = "pre"
    doubles: bool = False


class World:
    """Answers capture's URLs (the fake PoliteHttp send) and the lean client's and MLB schedule's (their _get)."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.games: Dict[str, Game] = {}
        self.cards: Dict[str, Card] = {}
        self.matches: Dict[str, Match] = {}
        self.sent: List[Tuple[float, str]] = []  # through PoliteHttp
        self.lean: List[str] = []  # the lean client's own fetches (EspnClient._get) and MLB's schedule
        self.status_for: Dict[str, int] = {}  # URL part -> the status to answer
        self.unreachable: Set[str] = set()
        self.plays_pages = 5
        self.schedule: Dict[str, List[Dict[str, Any]]] = {}
        self.propbets_moves = True  # each snapshot's prices differ
        self.prop_calls = 0
        self.backfilling = False  # set while the older-games backfill's run_night runs (see tag_backfill)
        self.tagged: List[Tuple[float, str, str]] = []  # (time, "forward" | "backfill", url) of each request

    def add(self, *things: Any) -> None:
        for thing in things:
            if isinstance(thing, Game):
                self.games[thing.event_id] = thing
            elif isinstance(thing, Card):
                self.cards[thing.event_id] = thing
            elif isinstance(thing, Match):
                self.matches[thing.comp_id] = thing

    def play(self, now: Optional[float] = None) -> None:
        """Move every game to its state at ``now`` (pre, live, final; postponed games stay postponed)."""
        now = self.clock() if now is None else now
        for game in self.games.values():
            if game.state != "postponed":
                game.state = "pre" if now < game.start else ("live" if now < game.start + game.runs else "final")
        for card in self.cards.values():
            for n, bout in enumerate(card.bouts):
                start = card.start + n * card.runs / max(1, len(card.bouts))
                bout.state = "pre" if now < start else ("live" if now < card.start + card.runs else "final")
        for match in self.matches.values():
            match.state = "pre" if now < match.start else ("live" if now < match.start + 2 * 3600 else "final")

    def urls(self, part: str = "") -> List[str]:
        return [url for _t, url in self.sent if part in url]

    # -- the fake send --

    def send(self, request: Any) -> HttpResponse:
        url = request.full_url
        self.sent.append((self.clock(), url))
        self.tagged.append((self.clock(), "backfill" if self.backfilling else "forward", url))
        if any(part in url for part in self.unreachable):
            raise ConnectionResetError("reset by peer")
        for part, status in self.status_for.items():
            if part in url:
                return HttpResponse(status, json.dumps({"code": status}).encode(), {})
        status, body = self.route(url)
        return HttpResponse(status, body, {"content-type": "application/json"})

    def route(self, url: str) -> Tuple[int, bytes]:
        parts = urllib.parse.urlsplit(url)
        query = dict(urllib.parse.parse_qsl(parts.query))
        host, path = parts.hostname, parts.path
        if "the-odds-api" in str(host):
            raise AssertionError("capture asked The Odds API")
        if host == "site.api.espn.com":
            found = re.fullmatch(r"/apis/site/v2/sports/([^/]+)/([^/]+)/(summary|scoreboard)", path)
            league = LEAGUE_OF[(found[1], found[2])]
            if found[3] == "summary":
                return self.summary(query["event"])
            return self.board(league, query["dates"])
        if host == "sports.core.api.espn.com":
            found = re.fullmatch(r"/v2/sports/[^/]+/leagues/[^/]+/events/([^/]+)/competitions/([^/]+)/(.+)", path)
            rest = found[3]
            if rest == "plays":
                return self.plays(found[1], int(query["page"]))
            if rest == "status":
                return 200, json.dumps({"type": STATUS["final"], "result": {"name": "decision---unanimous"},
                                        "bout": found[2]}).encode()
            stats = re.fullmatch(r"competitors/([^/]+)/statistics", rest)
            if stats:
                return 200, json.dumps({"splits": {"categories": [{"stats": [{"name": "sigStrikesLanded",
                                                                                "value": 41}]}]},
                                        "athlete": stats[1]}).encode()
            if rest == "odds/100/propBets":
                return self.propbets(found[1], int(query["page"]))
        if host == "site.web.api.espn.com":
            found = re.fullmatch(r"/apis/common/v3/sports/mma/ufc/fightcenter/([^/]+)", path)
            if found and found[1] in self.cards:
                return 200, self.fightcenter(self.cards[found[1]])
        if host == "statsapi.mlb.com":
            found = re.fullmatch(r"/api/v1\.1/game/(\d+)/feed/live", path)
            if found:
                return 200, json.dumps({"gamePk": int(found[1]), "liveData": {"plays": {"allPlays": [
                    {"result": {"eventType": "strikeout"}}]}}}).encode()
        if host == "api.elections.kalshi.com":
            return 200, json.dumps({"events": [{"event_ticker": f"{query.get('series_ticker')}-26OCT30",
                                                "markets": [{"yes_bid": int(self.clock()) % 97}]}],
                                    "cursor": ""}).encode()
        if host == "github.com" and path.endswith(".csv.gz"):
            return 200, (FIXTURES / "syn_play_by_play_2024.csv.gz").read_bytes()
        if host in ("github.com", "raw.githubusercontent.com") and path.endswith(".csv"):
            return 200, f"file,{path.rsplit('/', 1)[1]}\n1,2\n".encode()
        if host == "football-data.co.uk":
            return 200, f"Div,HomeTeam,AwayTeam\n{path.rsplit('/', 1)[1][:-4]},Arsenal,Spurs\n".encode()
        return 404, b'{"code": 404, "message": "not found"}'

    # -- ESPN's answers --

    def _competitors(self, game: Game) -> List[Dict[str, Any]]:
        return [{"homeAway": "home", "team": {"id": "1", "displayName": game.home}, "score": "101"},
                {"homeAway": "away", "team": {"id": "2", "displayName": game.away}, "score": "99"}]

    def summary(self, event_id: str) -> Tuple[int, bytes]:
        game = self.games.get(event_id)
        if game is None:
            return 404, b'{"code": 404}'
        if game.fixture and game.state == "final":
            data = json.loads((FIXTURES / game.fixture).read_text())
            data["header"]["id"] = event_id
            if game.version:
                data["boxscore"]["corrected"] = game.version
            return 200, json.dumps(data).encode()
        return 200, json.dumps({
            "header": {"id": event_id, "competitions": [{"id": event_id, "date": iso(game.start),
                                                         "status": {"type": STATUS[game.state]},
                                                         "competitors": self._competitors(game)}]},
            "boxscore": {"teams": [{"statistics": [{"name": "wonCorners", "displayValue": "7"}]}],
                         "version": game.version},
            "plays": [{"id": "1", "type": {"id": "64", "text": "Traveling"}}] if game.state != "pre" else [],
            "winprobability": [{"playId": "1", "homeWinPercentage": 0.61}], "pickcenter": [],
            # before the start the inputs change: the injury report hour by hour, the confirmed line-ups 75 minutes out
            "injuries": [{"team": game.home, "asOf": int(self.clock() // 3600)}] if game.state == "pre" else [],
            "rosters": [{"team": game.home, "starters": 11}] if game.start - self.clock() <= 75 * 60 else [],
            "gameInfo": {"officials": [{"displayName": "Ref A"}]}}).encode()

    def board(self, league: str, dates: str) -> Tuple[int, bytes]:
        if not re.fullmatch(r"\d{8}", dates):
            return 400, b'{"code":400,"message":"Failed to get events endpoint."}'
        day = datetime.datetime.strptime(dates, "%Y%m%d").date()

        def on(ts: float) -> bool:
            return datetime.datetime.fromtimestamp(ts, ET).date() == day

        events: List[Dict[str, Any]] = []
        sport = plan.PATHS[league][0]
        if sport == "mma":
            for card in self.cards.values():
                if on(card.start):
                    events.append({"id": card.event_id, "date": iso(card.start), "competitions": [
                        {"id": bout.id, "date": iso(card.start), "status": {"type": STATUS[bout.state]},
                         "competitors": [{"id": who, "athlete": {"id": who}} for who in bout.competitors],
                         "details": [{"type": {"text": "Takedown Attempt"}}]} for bout in card.bouts]})
        elif sport == "tennis":
            matches = [m for m in self.matches.values() if m.tour == league and on(m.start)]
            if matches:
                events.append({"id": "13-2026", "name": "Rolex Paris Masters", "groupings": [
                    {"grouping": {"slug": slug}, "competitions": [
                        {"id": m.comp_id, "date": iso(m.start), "status": {"type": STATUS[m.state]},
                         "notes": [{"text": "won 6-4 6-4"}]} for m in matches if m.doubles == (slug != "mens-singles")]}
                    for slug in ("mens-singles", "mens-doubles")]})
        else:
            for game in self.games.values():
                if game.league == league and on(game.start):
                    events.append({"id": game.event_id, "date": iso(game.start), "competitions": [
                        {"id": game.event_id, "date": iso(game.start), "status": {"type": STATUS[game.state]},
                         "competitors": self._competitors(game), "leaders": [{"name": "points"}]}]})
        return 200, json.dumps({"leagues": [{"slug": league}], "events": events}).encode()

    def plays(self, event_id: str, page: int) -> Tuple[int, bytes]:
        data = json.loads((FIXTURES / "core_epl_plays.json").read_text())
        data.update({"pageIndex": page, "pageCount": self.plays_pages,
                     "items": [{**item, "id": f"{event_id}-{page}-{n}"} for n, item in enumerate(data["items"])]})
        if page > self.plays_pages:
            data["items"] = []
        return 200, json.dumps(data).encode()

    def propbets(self, event_id: str, page: int) -> Tuple[int, bytes]:
        self.prop_calls += 1
        price = -110 - (self.prop_calls if self.propbets_moves else 0)  # the line moves between snapshots
        return 200, json.dumps({"count": 2, "pageIndex": page, "pageSize": 500, "pageCount": 1, "items": [
            {"athlete": {"$ref": "athletes/1966"}, "type": {"id": "1", "name": "Total Points"},
             "current": {"target": {"value": 25.5}}, "odds": {"american": str(price)}},
            {"athlete": {"$ref": "athletes/3112335"}, "type": {"id": "2", "name": "Total Rebounds"},
             "current": {"target": {"value": 9.5}}, "odds": {"american": "-120"}}]}).encode()

    def fightcenter(self, card: Card) -> bytes:
        def bout(b: Bout) -> Dict[str, Any]:
            status: Dict[str, Any] = {"type": STATUS[b.state]}
            if b.state == "final" and b.result:
                status["result"] = {"name": b.result, "displayName": "KO/TKO"}
            return {"id": b.id, "status": status, "competitors": [{"id": who, "athlete": {"id": who}}
                                                                   for who in b.competitors]}
        return json.dumps({"cards": {"main": {"competitions": [bout(b) for b in card.bouts[:1]]},
                                     "prelims1": {"competitions": [bout(b) for b in card.bouts[1:]]}},
                           "bets": [{"name": "Total Rounds"}], "asOf": iso(self.clock())}).encode()

    # -- the lean client's and MLB schedule's own fetches (requests, monkeypatched) --

    def lean_get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        full = url + ("?" + urllib.parse.urlencode(params or {}) if params else "")
        self.lean.append(full)
        status, body = self.route(full)
        if status != 200:
            raise RuntimeError(f"ESPN returned HTTP {status} for {url}")
        return json.loads(body)

    def schedule_get(self, url: str, params: Dict[str, Any]) -> Any:
        self.lean.append(f"{url}?date={params['date']}")
        return {"dates": [{"games": self.schedule.get(params["date"], [])}]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    clock = Clock(at(DAY, 8))
    world = World(clock)
    monkeypatch.setattr(EspnClient, "_get", lambda self, url, params=None: world.lean_get(url, params))
    monkeypatch.setattr(MlbStats, "_get", lambda self, url, params: world.schedule_get(url, params))
    calls: List[str] = []

    def no_requests(session, method, url, *args, **kwargs):
        calls.append(str(url))
        raise AssertionError(f"capture made a requests call: {url}")

    monkeypatch.setattr(requests.sessions.Session, "request", no_requests)

    def make(name: str = "raw.db", *, per_day: int = 3000, options: Optional[plan.Options] = None,
             kalshi: Sequence[str] = (), backfill: Any = None, config: Any = None, lease: Any = None,
             cache: Optional[TTLCache] = None) -> Capturer:
        raw = RawStore(str(tmp_path / name), clock=clock)
        cache = cache or TTLCache(name="capture-test", clock=clock)
        http = PoliteHttp(clock=clock, sleep=clock.sleep)
        http._send = world.send  # type: ignore[method-assign]
        capturer = Capturer(None, raw, EspnClient(cache, clock=clock), http, clock,
                            config=config if config is not None else SimpleNamespace(capture_backfill=False),
                            options=options, per_day=per_day, kalshi_series=kalshi, backfill=backfill,
                            mlb=MlbStats(cache, clock=clock), lease=lease or (lambda: True))
        return capturer

    return SimpleNamespace(clock=clock, world=world, make=make, tmp=tmp_path, requests_calls=calls)


def row(thing: Any, status: Optional[str] = None, league: Optional[str] = None) -> Dict[str, Any]:
    """A slate row (data/slate.py's capture hand-off) for a game, a bout or a tennis match."""
    if isinstance(thing, Match):
        return {"id": f"m{thing.comp_id}", "league": league or f"{thing.tour}_paris_masters", "espn_league": thing.tour,
                "espn_id": thing.comp_id, "start": iso(thing.start), "_ts": thing.start,
                "status": status or thing.state, "_time_ok": True}
    return {"id": f"g{thing.event_id}", "league": thing.league, "espn_league": thing.league, "espn_id": thing.event_id,
            "start": iso(thing.start), "_ts": thing.start, "status": status or thing.state, "_time_ok": True}


def bout_rows(card: Card) -> List[Dict[str, Any]]:
    return [{"id": f"b{bout.id}", "league": UFC, "espn_league": UFC, "espn_id": bout.id, "start": iso(card.start),
             "_ts": card.start, "status": bout.state} for bout in card.bouts]


def kept(raw: RawStore, game_key: str, state: Optional[str] = None) -> Set[Tuple[str, str]]:
    """(source, endpoint) of a game's 2xx payloads (of one state)."""
    return {(p.source, p.endpoint) for p in raw.bundle(game_key) if state is None or p.state == state}


def rows_of(raw: RawStore, game_key: str) -> List[Tuple[str, str, str, Optional[str], str]]:
    with raw._connect() as db:
        return [(r[0], r[1], r[2], r[3], r[4]) for r in db.execute(
            "SELECT source, endpoint, state, horizon, params FROM raw_payloads WHERE game_key = ? ORDER BY id",
            (game_key,))]


# --- the plan ----------------------------------------------------------------------------------------------------------

def test_the_plan_has_the_58_paths_and_section_3_3s_kinds() -> None:
    assert len(plan.PATHS) == 58 and {ATP, "tennis_wta", UFC} <= set(plan.PATHS)
    assert plan.KINDS == {
        "basketball": {"final": ["summary"], "pre": ["summary", "propbets"]},
        "football": {"final": ["summary"], "pre": ["propbets"]},
        "hockey": {"final": ["summary"], "pre": ["propbets"]},
        "baseball": {"final": ["summary", "feed_live"], "pre": ["propbets"]},
        "soccer": {"final": ["summary", "core_plays"], "pre": ["summary"]},
        "mma": {"final": ["fightcenter", "core_status", "core_competitor_stats", "scoreboard"], "pre": ["fightcenter"]},
        "tennis": {"final": ["scoreboard"], "pre": []}}
    assert plan.CORE_PLAYS_LEAGUES == plan.WAVE1_SOCCER and len(plan.WAVE1_SOCCER) == 10
    assert plan.kinds_for(EPL, "final") == ["summary", "core_plays"] and plan.kinds_for(MLS, "final") == ["summary"]
    assert plan.pre_items(NBA) == [("summary", "t90"), ("propbets", "open"), ("propbets", "t24"), ("propbets", "t90"),
                                   ("propbets", "close")]
    assert plan.pre_items(EPL) == [("summary", "t24"), ("summary", "close")]
    assert plan.pre_items(NCAAB) == [] and plan.pre_items(MLS) == [] and plan.pre_items(ATP) == []
    assert plan.pre_items(UFC) == [("fightcenter", "t24")]
    assert [kind for kind, _h in plan.pre_items(NFL)] == ["propbets"] * 4
    assert plan.pre_items(NBA, plan.Options(propbets=False)) == [("summary", "t90")]
    assert ("kalshi", "close") in plan.pre_items(EPL, plan.Options(kalshi_leagues=(EPL,)))
    assert plan.required_final(UFC) == ["fightcenter", "scoreboard"]
    assert plan.FINAL_DELAYS == (1800.0, 7200.0, 43200.0, 129600.0) and plan.GIVE_UP == 4 * 86400.0
    assert plan.HORIZON_LEAD == {"t24": 86400.0, "t90": 5400.0, "close": 600.0}
    assert plan.espn_league("tennis_atp_paris_masters") == ATP and plan.espn_league("cricket_odi") is None
    from telegram_betting_bot_simple.capture.sources import espn_propbets, kalshi_raw
    assert plan.PROPBETS_LEAGUES == espn_propbets.LEAGUES
    assert set(plan.PRE_HORIZONS[("*", "propbets")]) == set(espn_propbets.HORIZONS)
    # The settings' defaults are the plan's
    from telegram_betting_bot_simple import config as config_module
    assert config_module.CORE_PLAYS_DEFAULT == ",".join(plan.WAVE1_SOCCER)
    assert tuple(config_module.KALSHI_SERIES_DEFAULT.split(",")) == kalshi_raw.DEFAULT_SERIES


def test_the_static_schedule() -> None:
    by_name = {entry.name: entry for entry in plan.STATIC}
    assert set(by_name) == {"greco1899", "mcp", "football_data", "nflverse"}
    now = at(datetime.date(2026, 11, 3), 11)  # a Tuesday
    assert plan.static_due(by_name["greco1899"], None, now)
    assert not plan.static_due(by_name["greco1899"], {"at": now - 6 * 86400}, now)
    assert plan.static_due(by_name["greco1899"], {"at": now - 7 * 86400}, now)
    assert not plan.static_due(by_name["mcp"], {"at": now - 29 * 86400}, now)
    assert plan.static_due(by_name["mcp"], {"at": now - 30 * 86400}, now)
    assert not plan.static_due(by_name["football_data"], {"at": now - 3 * 86400}, now)
    assert plan.static_due(by_name["football_data"], {"at": now - 3.5 * 86400}, now)  # twice a week
    tz = ET
    period = plan.static_period(by_name["nflverse"], now, tz)
    assert period == "2026-W45" and not plan.static_due(by_name["nflverse"], {"period": period}, now, tz)
    assert plan.static_period(by_name["nflverse"], at(datetime.date(2026, 11, 3), 9), tz) == "2026-W44"  # before 10:00
    assert plan.static_due(by_name["nflverse"], {"period": "2026-W44"}, at(datetime.date(2026, 11, 3), 10), tz)


# --- one recorded day with a final in every sport ---------------------------------------------------------------------

def recorded_day(world: World) -> Dict[str, Any]:
    nba = Game(NBA, "401909278", at(DAY, 19, 30), fixture="sum_nba.json", runs=2.3 * 3600)
    ncaab = Game(NCAAB, "401800001", at(DAY, 19), "Duke Blue Devils", "Kansas Jayhawks", runs=2 * 3600)
    nfl = Game(NFL, "401873017", at(DAY, 13), "Atlanta Falcons", "San Francisco 49ers", fixture="sum_nfl.json",
               runs=3.2 * 3600)
    nhl = Game(NHL, "401892612", at(DAY, 19), "Florida Panthers", "Winnipeg Jets", fixture="sum_nhl.json",
               runs=2.6 * 3600)
    mlb = Game(MLB, "401908046", at(DAY, 20, 8), "Tampa Bay Rays", "Milwaukee Brewers", fixture="sum_mlb.json",
               runs=3 * 3600)
    epl = Game(EPL, "401878765", at(DAY, 10), "Liverpool", "Brighton & Hove Albion", fixture="sum_epl.json",
               runs=1.9 * 3600)
    mls = Game(MLS, "401700001", at(DAY, 19, 30), "Inter Miami CF", "Orlando City SC", runs=1.9 * 3600)
    card = Card("600061267", at(DAY, 14), [Bout("401911632", ("3949584", "4024000")),
                                           Bout("401911640", ("5000001", "5000002"), result=None),
                                           Bout("401911645", ("5000003", "5000004"))])
    singles = Match("187334", ATP, at(DAY, 6))
    doubles = Match("187294", ATP, at(DAY, 7), doubles=True)
    world.add(nba, ncaab, nfl, nhl, mlb, epl, mls, card, singles, doubles)
    world.schedule[DAY.isoformat()] = [{"gamePk": 777079, "gameDate": iso(mlb.start + 300), "status": {},
                                        "teams": {"home": {"team": {"name": "Tampa Bay Rays", "id": 139}},
                                                  "away": {"team": {"name": "Milwaukee Brewers", "id": 158}}}},
                                       {"gamePk": 777080, "gameDate": iso(mlb.start + 300), "status": {},
                                        "teams": {"home": {"team": {"name": "New York Yankees", "id": 147}},
                                                  "away": {"team": {"name": "Boston Red Sox", "id": 111}}}}]
    return {"nba": nba, "ncaab": ncaab, "nfl": nfl, "nhl": nhl, "mlb": mlb, "epl": epl, "mls": mls, "card": card,
            "singles": singles, "doubles": doubles}


def slate_rows(day: Dict[str, Any]) -> List[Dict[str, Any]]:
    games = [day[k] for k in ("nba", "ncaab", "nfl", "nhl", "mlb", "epl", "mls")]
    return [row(g) for g in games] + bout_rows(day["card"]) + [row(day["singles"])]


def run_day(capturer: Capturer, world: World, day: Dict[str, Any], until: float, step: float = 600.0) -> None:
    """The dashboard's loops on the fake clock: each step the world moves on, the slate hands its rows over (as its
    hook does every round) and the capture thread ticks."""
    clock = world.clock
    while clock.now < until:
        world.play()
        capturer.schedule_from_slate(slate_rows(day))
        capturer.tick()
        clock.now = max(clock.now, (clock.now // step + 1) * step)


def test_one_recorded_day_stores_the_plans_kinds_for_every_sport(env) -> None:
    world = env.world
    day = recorded_day(world)
    capturer = env.make()
    run_day(capturer, world, day, at(DAY + datetime.timedelta(days=1), 7))
    raw = capturer.raw
    final = lambda key: kept(raw, key, "final")  # noqa: E731
    assert final("espn:401909278") == {("espn_site", "summary")}  # NBA
    assert final("espn:401800001") == {("espn_site", "summary")}  # college basketball
    assert final("espn:401873017") == {("espn_site", "summary")}  # NFL
    assert final("espn:401892612") == {("espn_site", "summary")}  # NHL
    assert final("espn:401908046") == {("espn_site", "summary"), ("mlb_statsapi", "feed_live")}  # MLB
    assert final("espn:401878765") == {("espn_site", "summary"), ("espn_core", "core_plays")}  # EPL (wave 1)
    assert final("espn:401700001") == {("espn_site", "summary")}  # MLS: no core plays
    assert final("espn:600061267") == {("espn_fightcenter", "fightcenter")}  # the UFC card
    assert final("espn-bout:401911640") == {("espn_core", "core_status"), ("espn_core", "core_competitor_stats")}
    assert raw.bundle("espn-bout:401911632").payloads == () and raw.bundle("espn-bout:401911645").payloads == ()
    assert len(raw.bundle("espn-bout:401911640").all("espn_core", "core_competitor_stats")) == 2  # both fighters
    # Untrimmed: the summaries are the whole answers (plays, win probability, keyEvents...), not the lean copies
    nba = raw.latest("espn:401909278", "espn_site", "summary").json()
    assert {"plays", "winprobability", "boxscore", "gameInfo"} <= set(nba)
    assert {"keyEvents", "commentary", "rosters"} <= set(raw.latest("espn:401878765", "espn_site", "summary").json())
    assert [p.params["page"] for p in raw.bundle("espn:401878765").pages("espn_core", "core_plays")] == [1, 2, 3, 4, 5]
    assert raw.latest("espn:401908046", "mlb_statsapi", "feed_live").params == {"game_pk": 777079}
    # The day's boards, raw (a UFC bout's details, tennis notes): the card's and the tours' record
    ufc_board = raw.latest(scoreboard_key(UFC, "20261030"), "espn_site", "scoreboard", {"dates": "20261030"})
    assert ufc_board.json()["events"][0]["competitions"][0]["details"]
    atp_board = raw.latest(scoreboard_key(ATP, "20261030"), "espn_site", "scoreboard")
    assert atp_board.state == "final" and "notes" in json.dumps(atp_board.json())
    # Every job is done; tennis matches (the slate's and the one nobody opened) from the board
    jobs = {job.game_key: job for job in raw.jobs()}
    assert {key: job.state for key, job in jobs.items() if job.state != "done"} == {}
    assert {"espn-match:187334", "espn:600061267"} <= set(jobs) and "espn-match:187294" not in jobs  # singles only
    assert jobs["espn:600061267"].kinds_done == ["final:core_competitor_stats", "final:core_status",
                                                 "final:fightcenter", "final:scoreboard", "pre:t24:fightcenter"]
    # The pre-start kinds: NBA's t90 summary and four propBets snapshots, EPL's t24 and close summaries
    assert ("espn_site", "summary", "pre", "t90", "{}") in rows_of(raw, "espn:401909278")
    props = [r for r in rows_of(raw, "espn:401909278") if r[0] == "espn_propbets"]
    assert [r[3] for r in props] == ["t24", "t90", "close"]  # first seen at 08:00: open and t24 in one snapshot
    epl = [r[3] for r in rows_of(raw, "espn:401878765") if r[2] == "pre"]
    assert epl == ["t24", "close"]
    assert not [r for r in rows_of(raw, "espn:401800001") if r[2] == "pre"]  # wave 3: no pre-start fetch
    # Never The Odds API, and nothing outside the polite door: the lean client fetched only the UFC board it reads
    # the cards from and MLB's schedule (both through the cache)
    assert not any("odds-api" in url for url in world.urls() + world.lean) and env.requests_calls == []
    assert all("scoreboard" in url and "/mma/" in url or "statsapi.mlb.com/api/v1/schedule" in url
               for url in world.lean)


def test_capture_is_idempotent_and_a_corrected_body_is_a_new_version(env) -> None:
    world, clock = env.world, env.clock
    day = recorded_day(world)
    capturer = env.make()
    run_day(capturer, world, day, at(DAY + datetime.timedelta(days=1), 7))
    raw = capturer.raw
    rows, asked = raw.payload_count(), len(world.sent)
    # Running on stores nothing and asks nothing more (every job done, the sweep marked)
    run_day(capturer, world, day, clock.now + 3 * 3600)
    assert raw.payload_count() == rows and len(world.sent) == asked
    # A restarted process over the same raw.db: the same
    again = env.make()
    again.schedule_from_slate(slate_rows(day))
    assert again.run_due(1000)["fetched"] == 0 and raw.payload_count() == rows
    # The same games fetched again (the jobs reopened): identical bodies add no row
    for key in ("espn:401892612", "espn:401878765"):
        raw.modify_job(key, lambda job: Job(job.game_key, job.league, job.espn_id, job.start, clock.now))
    report = again.run_due(1000)
    assert report["fetched"] == 2 and report["stored"] == 0 and raw.payload_count() == rows  # (plays pages in hand
    # aren't asked again)
    # A corrected summary (ESPN fixed a stat): a new row, the old one kept, latest() gives the new
    day["nba"].version = 2
    raw.modify_job("espn:401909278", lambda job: Job(job.game_key, job.league, job.espn_id, job.start, clock.now))
    assert again.run_due(1000)["stored"] == 1
    versions = [p for p in raw.bundle("espn:401909278", ok_only=True).all("espn_site", "summary")]
    assert raw.latest("espn:401909278", "espn_site", "summary").json()["boxscore"]["corrected"] == 2
    with raw._connect() as db:
        finals = db.execute("SELECT COUNT(*) FROM raw_payloads WHERE game_key = 'espn:401909278' AND source = "
                            "'espn_site' AND state = 'final'").fetchone()[0]
    assert finals == 2 and len(versions) == 1


# --- the grader and the lean cache ------------------------------------------------------------------------------------

def test_one_fetch_serves_both_stores_and_the_grader_sees_the_same_bytes(env) -> None:
    world, clock = env.world, env.clock
    game = Game(NBA, "401909278", at(DAY, 19, 30), fixture="sum_nba.json", state="final")
    world.add(game)
    clock.now = at(DAY, 23)
    cache = TTLCache(name="capture-test", clock=clock)
    capturer = env.make(cache=cache)
    capturer.schedule_from_slate([row(game, "final")])
    clock.now += 1800
    assert capturer.run_due(100)["stored"] == 1
    assert world.urls("/summary") and world.lean == []  # one fetch, through the polite door
    # The grader's next summary() is a hit: no request at all, the lean copy of the same answer
    seeded = capturer.espn
    lean = seeded.summary(NBA, "401909278")
    assert world.lean == [] and len(world.urls("/summary")) == 1
    body = (FIXTURES / "sum_nba.json").read_bytes()
    assert lean == lean_summary(json.loads(body))
    # ...byte-identical to what the grader's own fetch would have cached, and its answers the same
    fresh = EspnClient(TTLCache(name="fresh", clock=clock), clock=clock)
    assert json.dumps(fresh.summary(NBA, "401909278"), sort_keys=True) == json.dumps(lean, sort_keys=True)
    assert len(world.lean) == 1  # that one was the fresh client's own load
    for read in ("game_result", "box_score", "closing_line"):
        assert getattr(seeded, read)(NBA, "401909278") == getattr(fresh, read)(NBA, "401909278")
    key = ("espn", "summary", NBA, "401909278")
    assert json.dumps(cache.peek(key)[1], sort_keys=True) == json.dumps(fresh.cache.peek(key)[1], sort_keys=True)
    # raw.db has the whole answer
    assert capturer.raw.latest("espn:401909278", "espn_site", "summary").raw_bytes() == world.summary(
        "401909278")[1]


def test_the_sweep_seeds_the_lean_boards_too(env) -> None:
    world, clock = env.world, env.clock
    world.add(Game(NBA, "401909300", at(DAY, 19), state="final"))
    clock.now = at(DAY + datetime.timedelta(days=1), 6)
    capturer = env.make()
    capturer.sweep(DAY)
    events = capturer.espn.scoreboard(NBA, DAY)
    assert world.lean == [] and [e["id"] for e in events] == ["401909300"]
    assert "leaders" not in json.dumps(events)  # the lean copy (LEAN_DROP), as the cache always keeps it
    assert "leaders" in json.dumps(capturer.raw.latest(scoreboard_key(NBA, "20261030"), "espn_site",
                                                       "scoreboard").json())  # the raw one keeps everything


# --- The Odds API: never ---------------------------------------------------------------------------------------------

def test_capture_and_the_backfill_make_no_odds_api_request(env, tmp_path) -> None:
    """Counting test: a whole day of forward capture, the sweep, the static files and a night of the real backfill
    make zero Odds API requests, and the store's quota log stays empty."""
    from telegram_betting_bot_simple.data.store import Store
    world, clock = env.world, env.clock
    day = recorded_day(world)
    store = Store(str(tmp_path / "betting.db"))
    backfill_raw = RawStore(str(tmp_path / "raw.db"), clock=clock)
    capturer = env.make(config=SimpleNamespace(capture_backfill=True, capture_backfill_per_night=40,
                                               capture_backfill_seasons=1))
    capturer.store = store
    capturer.lease = lambda: bool(store.try_lease(worker_module.LEASE, worker_module.LEASE_SECONDS))
    assert capturer.raw.path == backfill_raw.path
    capturer.tz = ET
    capturer._backfill_tried = False
    capturer._backfill = None
    run_day(capturer, world, day, at(DAY + datetime.timedelta(days=1), 7), step=1800)
    hosts = {urllib.parse.urlsplit(url).hostname for url in world.urls() + world.lean}
    assert hosts and not any("odds-api" in host for host in hosts)
    assert env.requests_calls == []  # nothing through requests at all (the Odds API clients use it)
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM api_usage").fetchone()[0] == 0
    assert capturer.raw.spent("backfill", "2026-10-31") > 0  # the backfill did run that night


# --- pre-start horizons ---------------------------------------------------------------------------------------------------

def test_pre_horizons_open_t24_t90_and_close_then_nothing_once_it_starts(env) -> None:
    world, clock = env.world, env.clock
    start = at(DAY, 19, 30)
    game = Game(NBA, "401909400", start)
    world.add(game)
    clock.now = start - 30 * 3600
    capturer = env.make()
    capturer.schedule_from_slate([row(game)])
    for minutes in range(0, 31 * 60, 5):
        clock.now = start - 30 * 3600 + minutes * 60
        world.play()
        capturer.run_due(100)
    with capturer.raw._connect() as db:
        rows = db.execute("SELECT source, horizon, fetched_at, state FROM raw_payloads WHERE game_key = ? ORDER BY "
                          "fetched_at, id", ("espn:401909400",)).fetchall()
    pre = [(source, horizon, round((fetched - start) / 60)) for source, horizon, fetched, state in rows
           if state == "pre"]
    assert pre == [("espn_propbets", "open", -1800), ("espn_propbets", "t24", -1440), ("espn_site", "t90", -90),
                   ("espn_propbets", "t90", -90), ("espn_propbets", "close", -10)]
    assert all(fetched < start for _s, _h, fetched, state in rows if state == "pre")  # inputs, never outcomes
    assert [at_ for at_, url in world.sent if "propBets" in url and at_ >= start] == []  # nothing once it starts
    assert capturer.raw.job("espn:401909400").kinds_done == [
        "pre:close:propbets", "pre:open:propbets", "pre:t24:propbets", "pre:t90:propbets", "pre:t90:summary"]


def test_a_late_first_sight_takes_one_snapshot_for_the_horizons_already_due(env) -> None:
    world, clock = env.world, env.clock
    start = at(DAY, 19)
    late = Game(NHL, "401892700", start, "Florida Panthers", "Winnipeg Jets")
    gone = Game(NHL, "401892701", at(DAY, 7), "Boston Bruins", "Ottawa Senators", state="live")
    world.add(late, gone)
    clock.now = start - 5 * 3600  # 5 hours before: open and t24 are both due
    capturer = env.make()
    assert capturer.schedule_from_slate([row(late), row(gone)]) == 2
    capturer.run_due(100)
    assert [r[3] for r in rows_of(capturer.raw, "espn:401892700")] == ["t24"]
    assert capturer.raw.job("espn:401892700").kinds_done == ["pre:open:propbets", "pre:t24:propbets"]
    assert rows_of(capturer.raw, "espn:401892701") == []  # first seen after its start: no pre-start fetch
    assert capturer.raw.job("espn:401892700").due_at == start - 90 * 60  # next: t90


def test_soccer_t24_and_close_summaries_and_ufcs_t24_fightcenter(env) -> None:
    world, clock = env.world, env.clock
    epl = Game(EPL, "401878900", at(DAY, 15), "Arsenal", "Chelsea")
    card = Card("600061300", at(DAY, 18), [Bout("401911700", ("1", "2")), Bout("401911701", ("3", "4"))])
    world.add(epl, card)
    clock.now = at(DAY, 15) - 30 * 3600
    capturer = env.make()
    for minutes in range(0, 34 * 60, 10):
        clock.now = at(DAY, 15) - 30 * 3600 + minutes * 60
        world.play()
        capturer.schedule_from_slate([row(epl)] + bout_rows(card))
        capturer.run_due(100)
    assert [(r[2], r[3]) for r in rows_of(capturer.raw, "espn:401878900") if r[2] == "pre"] == [("pre", "t24"),
                                                                                              ("pre", "close")]
    fight = [r for r in rows_of(capturer.raw, "espn:600061300") if r[2] == "pre"]
    assert fight == [("espn_fightcenter", "fightcenter", "pre", "t24", "{}")]


def test_kalshi_series_are_snapshotted_at_their_leagues_horizons_once_per_ten_minutes(env) -> None:
    world, clock = env.world, env.clock
    one = Game(EPL, "401879001", at(DAY, 15), "Arsenal", "Chelsea")
    two = Game(EPL, "401879002", at(DAY, 15), "Everton", "Fulham")
    other = Game("soccer_spain_la_liga", "401879003", at(DAY, 15), "Real Madrid", "Barcelona")
    world.add(one, two, other)
    clock.now = at(DAY, 15) - 30 * 3600
    capturer = env.make(kalshi=("KXEPLCORNERS", "KXUCLSOT"),
                        options=plan.Options(kalshi_leagues=(EPL, "soccer_uefa_champs_league")))
    capturer.schedule_from_slate([row(one), row(two), row(other)])
    capturer.run_due(100)
    kalshi = world.urls("api.elections.kalshi.com")
    assert len(kalshi) == 1 and "series_ticker=KXEPLCORNERS" in kalshi[0]  # two games, one snapshot; no UCL game
    snapshot = capturer.raw.latest("static:kalshi:KXEPLCORNERS", "kalshi", "events")
    assert snapshot.state == "pre" and snapshot.league == EPL
    assert capturer.raw.spent("prices", raw_store.utc_day(clock.now)) == 1
    assert capturer.raw.spent("capture", raw_store.utc_day(clock.now)) >= 1  # inside CAPTURE_PER_DAY too


# --- the final: timing, retries, giving up ----------------------------------------------------------------------------------

def test_the_final_waits_for_the_whistle_and_retries_at_2h_12h_and_36h_then_gives_up(env) -> None:
    world, clock = env.world, env.clock
    start = at(DAY, 19)
    game = Game(NCAAB, "401800100", start, "Duke Blue Devils", "Kansas Jayhawks", state="live")
    world.add(game)
    clock.now = start + 600
    capturer = env.make()
    capturer.schedule_from_slate([row(game, "live")])
    job = capturer.raw.job("espn:401800100")
    end = plan.estimated_end(NCAAB, start)
    assert job.due_at == end + 1800 and job.state == "todo"  # the usual end + 30 min
    tries = []
    for _ in range(6):
        clock.now = capturer.raw.job("espn:401800100").due_at
        capturer.run_due(10)
        job = capturer.raw.job("espn:401800100")
        tries.append((round((clock.now - end) / 3600, 2), job.attempts, job.state))
        if job.state != "todo" and job.state != "partial":
            break
    assert tries == [(0.5, 1, "todo"), (2.0, 2, "todo"), (12.0, 3, "todo"), (36.0, 4, "gave_up")]
    assert "isn't over yet" in job.last_error
    assert len(world.urls("/summary")) == 4  # one summary per attempt; the live ones are kept as "live"
    assert {r[2] for r in rows_of(capturer.raw, "espn:401800100")} == {"live"}


def test_the_slate_moves_the_first_final_fetch(env) -> None:
    world, clock = env.world, env.clock
    start = at(DAY, 19)
    game = Game(NBA, "401909500", start)
    world.add(game)
    clock.now = start - 3600
    capturer = env.make(options=plan.Options(propbets=False))
    capturer.schedule_from_slate([row(game, "pre")])
    clock.now = start + 3 * 3600  # running long: the slate still says live
    capturer.schedule_from_slate([row(game, "live")])
    assert capturer.raw.job("espn:401909500").due_at == clock.now + worker_module.LIVE_HOLD
    clock.now += 600  # the final whistle, seen on the slate's next round
    capturer.schedule_from_slate([row(game, "final")])
    assert capturer.raw.job("espn:401909500").due_at == pytest.approx(clock.now + 1800, abs=1)
    clock.now += 900
    capturer.schedule_from_slate([row(game, "final")])  # later rounds don't push it back
    assert capturer.raw.job("espn:401909500").due_at == pytest.approx(clock.now + 900, abs=1)
    # an early whistle comes forward of the usual end
    early = Game(NBA, "401909501", start)
    world.add(early)
    clock.now = start + 3600
    capturer.schedule_from_slate([row(early, "final")])
    assert capturer.raw.job("espn:401909501").due_at == clock.now + 1800


def test_a_job_gives_up_four_days_after_its_start_whatever_happened(env) -> None:
    world, clock = env.world, env.clock
    game = Game(NHL, "401892800", at(DAY, 19), state="final")
    world.add(game)
    world.unreachable.add("site.api.espn.com")
    clock.now = at(DAY, 23)
    capturer = env.make(options=plan.Options(propbets=False))
    capturer.schedule_from_slate([row(game, "final")])
    clock.now = game.start + 4 * 86400 + 1
    capturer.run_due(10)
    assert capturer.raw.job("espn:401892800").state == "gave_up"


def test_a_postponed_game_is_void_and_comes_back_if_it_is_rescheduled(env) -> None:
    world, clock = env.world, env.clock
    game = Game(MLB, "401908100", at(DAY, 19), "New York Yankees", "Boston Red Sox", state="postponed")
    world.add(game)
    clock.now = at(DAY, 19, 30)
    capturer = env.make(options=plan.Options(propbets=False))
    capturer.schedule_from_slate([row(game, "postponed")])
    clock.now += 60
    capturer.run_due(10)
    job = capturer.raw.job("espn:401908100")
    assert job.state == "void" and job.kinds_done == ["final:summary"]
    assert not world.urls("statsapi.mlb.com")  # nothing more for a void game
    # Rescheduled two days later under the same id: captured again
    game.start, game.state = at(DAY + datetime.timedelta(days=2), 13), "pre"
    capturer.schedule_from_slate([row(game, "pre")])
    job = capturer.raw.job("espn:401908100")
    assert job.state == "todo" and job.kinds_done == [] and job.start == game.start


# --- restarts, budgets, counters --------------------------------------------------------------------------------------------

def test_a_restart_mid_run_resumes_where_it_stopped(env) -> None:
    world, clock = env.world, env.clock
    game = Game(EPL, "401879100", at(DAY, 10), "Arsenal", "Chelsea", state="final", fixture="sum_epl.json")
    world.add(game)
    clock.now = at(DAY, 13)
    first = env.make(options=plan.Options(propbets=False))
    first.schedule_from_slate([row(game, "final")])
    clock.now += 1800
    report = first.run_due(3)  # the summary and two pages of plays, then this call's budget is spent
    assert report == {"fetched": 3, "stored": 3, "failed": 0, "jobs_done": 0, "stopped": "budget"}
    job = first.raw.job("espn:401879100")
    assert job.state == "partial" and job.kinds_done == ["final:summary"] and job.attempts == 0
    asked = len(world.sent)
    # The process restarts: a new Capturer over the same raw.db carries on with plays page 3
    second = env.make(options=plan.Options(propbets=False))
    assert second.run_due(100)["jobs_done"] == 1
    later = [url for url in world.urls()[asked:]]
    assert [urllib.parse.parse_qs(urllib.parse.urlsplit(u).query)["page"][0] for u in later] == ["3", "4", "5"]
    assert not any("/summary" in u for u in later)
    assert [p.params["page"] for p in second.raw.bundle("espn:401879100").pages("espn_core", "core_plays")] == \
        [1, 2, 3, 4, 5]


def test_capture_per_day_caps_the_days_requests(env) -> None:
    world, clock = env.world, env.clock
    games = [Game(NHL, f"40189290{n}", at(DAY, 19), state="final") for n in range(8)]
    world.add(*games)
    clock.now = at(DAY, 23, 30)
    capturer = env.make(per_day=5, options=plan.Options(propbets=False))
    capturer.schedule_from_slate([row(g, "final") for g in games])
    clock.now += 1800
    report = capturer.run_due(100)
    day = raw_store.utc_day(clock.now)
    assert report["fetched"] == 5 and report["stopped"] == "day" and capturer.raw.spent("capture", day) == 5
    assert not capturer.has_runnable_due()  # due jobs, but no budget left today: the backfill may run
    assert capturer.run_due(100)["fetched"] == 0
    clock.now = at(DAY + datetime.timedelta(days=1), 21)  # a new UTC day (21:00 US Eastern is 01:00 UTC)
    assert raw_store.utc_day(clock.now) != day and capturer.run_due(100)["fetched"] == 3
    assert all(job.state == "done" for job in capturer.raw.jobs())


def test_every_request_is_counted_ok_or_error_and_prices_count_twice(env) -> None:
    world, clock = env.world, env.clock
    good = Game(NBA, "401909600", at(DAY, 19), state="pre")
    bad = Game(NHL, "401892999", at(DAY, 12), state="final")
    lost = Game(NFL, "401873999", at(DAY, 13), state="final")
    world.add(good, bad, lost)
    world.status_for["event=401892999"] = 503
    world.unreachable.add("event=401873999")
    clock.now = at(DAY, 18, 0)
    capturer = env.make()
    capturer.schedule_from_slate([row(good), row(bad, "final"), row(lost, "final")])
    capturer.run_due(100)
    day = raw_store.utc_day(clock.now)
    ok, error = capturer.raw.spent("http_ok", day), capturer.raw.spent("http_error", day)
    assert error == 2 and ok == capturer.raw.spent("capture", day) - 2
    assert capturer.raw.spent("prices", day) == len(world.urls("propBets"))
    errored = capturer.raw.latest("espn:401892999", "espn_site", "summary", ok_only=False)
    assert errored.http_status == 503 and errored.horizon is None  # the error answer is kept with its status
    assert "Unreachable" in capturer.raw.job("espn:401873999").last_error or "reset" in \
        capturer.raw.job("espn:401873999").last_error


def test_a_backing_off_host_holds_its_jobs_not_the_others(env) -> None:
    world, clock = env.world, env.clock
    soccer = Game(EPL, "401879200", at(DAY, 10), "Arsenal", "Chelsea", state="final", fixture="sum_epl.json")
    world.add(soccer)
    world.status_for["/plays"] = 503
    clock.now = at(DAY, 13)
    capturer = env.make(options=plan.Options(propbets=False))
    capturer.schedule_from_slate([row(soccer, "final")])
    clock.now += 1800
    capturer.run_due(100)
    job = capturer.raw.job("espn:401879200")
    assert job.kinds_done == ["final:summary"] and job.attempts == 1  # the plays failed: retried later
    assert capturer.http.backing_off("sports.core.api.espn.com")
    # while the core host backs off, a job only needing it doesn't count as runnable ...
    clock.now = job.due_at
    capturer.http._buckets["sports.core.api.espn.com"].closed_until = clock.now + 3600
    assert not capturer.has_runnable_due()
    # ... but a job on another host does
    other = Game(NHL, "401892801", at(DAY, 10), state="final")
    world.add(other)
    capturer.schedule_from_slate([row(other, "final")])
    clock.now += 1801
    assert capturer.has_runnable_due()


# --- the sweep ------------------------------------------------------------------------------------------------------------------

def test_the_sweep_reads_yesterdays_58_boards_and_adds_the_finals_nobody_opened(env) -> None:
    world, clock = env.world, env.clock
    seen = Game(NBA, "401909700", at(DAY, 19), state="final")
    missed = Game(NBA, "401909701", at(DAY, 22), state="final")
    unfinished = Game(MLB, "401908200", at(DAY, 23), state="live")
    college = Game("americanfootball_ncaaf", "401750001", at(DAY, 19), state="final")
    card = Card("600061400", at(DAY, 18), [Bout("401911800", ("1", "2"), state="final")])
    match = Match("187400", "tennis_wta", at(DAY, 9), state="final")
    world.add(seen, missed, unfinished, college, card, match)
    clock.now = at(DAY, 23)
    capturer = env.make(options=plan.Options(propbets=False))
    capturer.schedule_from_slate([row(seen, "final")])
    clock.now = at(DAY + datetime.timedelta(days=1), 4, 59)
    assert capturer.sweep_day() is None  # yesterday is swept from 05:00 US Eastern
    clock.now = at(DAY + datetime.timedelta(days=1), 5)
    assert capturer.sweep_day() == DAY
    made = capturer.sweep(DAY)
    boards = world.urls("/scoreboard")
    assert len(boards) == 58 and all("dates=20261030" in url for url in boards)
    assert made == 4  # the missed NBA game, the college game, the UFC card and the WTA match (made done)
    jobs = {job.game_key: job for job in capturer.raw.jobs()}
    assert set(jobs) == {"espn:401909700", "espn:401909701", "espn:401750001", "espn:600061400", "espn-match:187400"}
    assert jobs["espn-match:187400"].state == "done" and jobs["espn:600061400"].kinds_done == ["final:scoreboard"]
    assert jobs["espn:401909701"].due_at <= clock.now  # due at once: it finished hours ago
    for league in plan.PATHS:
        board = capturer.raw.latest(scoreboard_key(league, "20261030"), "espn_site", "scoreboard", {"dates": "20261030"})
        assert board is not None and board.state == "final" and board.league == league
    assert capturer.sweep_day() is None  # once a day
    capturer.run_due(100)
    assert {job.game_key: job.state for job in capturer.raw.jobs()} == {
        "espn:401909700": "done", "espn:401909701": "done", "espn:401750001": "done", "espn:600061400": "done",
        "espn-match:187400": "done"}


def test_a_sweep_stopped_by_the_budget_carries_on_without_reading_a_board_twice(env) -> None:
    world, clock = env.world, env.clock
    clock.now = at(DAY + datetime.timedelta(days=1), 6)
    capturer = env.make(per_day=20)
    capturer.sweep(DAY)
    assert len(world.urls("/scoreboard")) == 20 and capturer.sweep_day() == DAY  # not finished
    clock.now = at(DAY + datetime.timedelta(days=1), 21)  # a new UTC day: 20 more
    capturer.sweep(DAY)
    assert len(world.urls("/scoreboard")) == 40 and capturer.sweep_day() == DAY
    clock.now += 86400
    capturer.sweep(DAY)
    boards = world.urls("/scoreboard")
    assert len(boards) == 58 and len(set(boards)) == 58 and capturer.raw.get_state("sweep:last") == DAY.isoformat()


def test_failed_boards_are_tried_again_later_three_sweeps_at_most(env) -> None:
    world, clock = env.world, env.clock
    world.status_for["/nhl/scoreboard"] = 500
    clock.now = at(DAY + datetime.timedelta(days=1), 6)
    capturer = env.make()
    capturer.sweep(DAY)
    assert capturer.sweep_day() is None  # retried after SWEEP_RETRY
    clock.now += worker_module.SWEEP_RETRY
    assert capturer.sweep_day() == DAY
    capturer.sweep(DAY)
    assert len(world.urls("/nhl/scoreboard")) == 2 and len(world.urls("/nba/scoreboard")) == 1
    clock.now += worker_module.SWEEP_RETRY
    capturer.sweep(DAY)
    clock.now += worker_module.SWEEP_RETRY
    assert capturer.sweep_day() is None  # given up after three


# --- the slate's rows ---------------------------------------------------------------------------------------------------------

def test_slate_rows_become_jobs_under_the_right_keys(env) -> None:
    world, clock = env.world, env.clock
    clock.now = at(DAY, 8)
    nba = Game(NBA, "401909800", at(DAY, 19))
    card = Card("600061500", at(DAY, 18), [Bout("401911900", ("1", "2")), Bout("401911901", ("3", "4"))])
    match = Match("187500", ATP, at(DAY, 12))
    world.add(nba, card, match)
    capturer = env.make()
    rows = [row(nba), *bout_rows(card), row(match),
            {"league": "cricket_odi", "espn_id": "1", "_ts": at(DAY, 9), "status": "pre"},  # not on ESPN's paths
            {"league": NBA, "espn_id": "", "_ts": at(DAY, 9)}, {"league": NBA, "espn_id": "x y", "_ts": at(DAY, 9)},
            {"league": NBA, "espn_id": "401909801"}, "not a row"]
    assert capturer.schedule_from_slate(rows) == 3
    jobs = {job.game_key: job for job in capturer.raw.jobs()}
    assert set(jobs) == {"espn:401909800", "espn:600061500", "espn-match:187500"}
    assert jobs["espn-match:187500"].league == ATP and jobs["espn:600061500"].league == UFC
    assert jobs["espn:600061500"].espn_id == "600061500" and jobs["espn:600061500"].start == card.start
    assert jobs["espn-match:187500"].due_at > at(DAY + datetime.timedelta(days=1), 5)  # after its day's sweep
    assert capturer.schedule_from_slate(rows) == 0  # the same rows again: nothing new
    # a game moved an hour later: its horizons follow it
    before = capturer.raw.job("espn:401909800")
    nba.start += 3600
    capturer.schedule_from_slate([row(nba)])
    moved = capturer.raw.job("espn:401909800")
    assert moved.start == before.start + 3600 and moved.kinds_done == before.kinds_done


def test_a_card_not_on_the_cached_board_waits_for_the_sweep(env) -> None:
    clock = env.clock
    clock.now = at(DAY, 8)
    capturer = env.make()
    rows = [{"league": UFC, "espn_league": UFC, "espn_id": "401999999", "_ts": at(DAY, 18), "status": "pre"}]
    assert capturer.schedule_from_slate(rows) == 0 and capturer.raw.jobs() == []


# --- static history files ---------------------------------------------------------------------------------------------------------

def test_static_sources_run_on_their_schedules_under_their_keys(env) -> None:
    world, clock = env.world, env.clock
    clock.now = at(datetime.date(2026, 11, 3), 11)  # a Tuesday
    capturer = env.make()
    capturer.tz = ET
    report = capturer.run_static()
    assert set(report) == {"greco1899", "mcp", "football_data", "nflverse"} and all(r["ok"] for r in report.values())
    keys = set(capturer.raw.iter_games())
    assert {"static:greco1899:ufc_event_details.csv", "static:mcp:charting-m-matches.csv",
            "static:football_data:E0:2627", "static:football_data:USA", "static:nflverse:pbp:2026",
            "static:nflverse:officials"} <= keys
    pbp = capturer.raw.latest("static:nflverse:pbp:2026", "nflverse", "pbp_season")
    assert pbp.state == "static" and pbp.codec == "identity" and pbp.raw_bytes()[:2] == b"\x1f\x8b"
    assert len(world.urls("raw.githubusercontent.com/Greco1899")) == 6 and len(world.urls("football-data")) == 10
    asked = len(world.sent)
    clock.now += 3600
    assert capturer.run_static() == {} and len(world.sent) == asked  # nothing due an hour later
    clock.now = at(datetime.date(2026, 11, 7), 0)  # Saturday: football-data again (twice a week)
    assert set(capturer.run_static()) == {"football_data"}
    clock.now = at(datetime.date(2026, 11, 10), 12, 30)  # the next Tuesday: Greco1899 weekly, nflverse
    assert set(capturer.run_static()) == {"greco1899", "nflverse", "football_data"}
    clock.now = at(datetime.date(2026, 12, 3), 12)
    assert "mcp" in capturer.run_static()  # monthly


def test_a_failing_static_source_is_tried_again_after_an_hour(env) -> None:
    world, clock = env.world, env.clock
    world.status_for["JeffSackmann"] = 503
    clock.now = at(datetime.date(2026, 11, 3), 11)
    capturer = env.make()
    assert capturer.run_static(["mcp"])["mcp"]["ok"] is False
    clock.now += 600
    assert capturer.run_static(["mcp"]) == {}
    del world.status_for["JeffSackmann"]
    clock.now += 3000
    assert capturer.run_static(["mcp"])["mcp"]["ok"] is True


# --- the backfill: inside its window, forward first -------------------------------------------------------------------------------

class StubBackfill:
    """RawBackfill's calling shape: in_window(), run_night(budget) checking forward_due before each request."""

    def __init__(self, capturer: Capturer, world: World, window: Tuple[float, float]) -> None:
        self.capturer, self.world, self.window = capturer, world, window
        self.calls: List[Dict[str, Any]] = []
        self.requests = 0
        self.left = 120

    def in_window(self, ts: Optional[float] = None) -> bool:
        now = self.world.clock() if ts is None else ts
        return self.window[0] <= now < self.window[1]

    def run_night(self, budget: Optional[int] = None) -> Dict[str, Any]:
        made = 0
        stopped = "budget"
        while made < (budget or 0):
            if not self.in_window():
                stopped = "window"
                break
            if self.capturer.has_runnable_due():
                stopped = "forward"
                break
            if not self.left:
                stopped = "done"
                break
            self.world.sent.append((self.world.clock(), "https://site.api.espn.com/backfill"))
            self.world.clock.sleep(1.0)
            made += 1
            self.left -= 1
            self.requests += 1
            self.on_request(self.requests)
        self.calls.append({"budget": budget, "requests": made, "stopped": stopped})
        return {"ran": True, "requests": made, "stopped": stopped, "stored": made}

    def on_request(self, n: int) -> None:
        pass


def test_the_backfill_runs_in_its_window_and_a_due_job_always_goes_first(env) -> None:
    world, clock = env.world, env.clock
    night = at(DAY + datetime.timedelta(days=1), 1)
    clock.now = night
    capturer = env.make(options=plan.Options(propbets=False))
    stub = StubBackfill(capturer, world, (night, night + 5 * 3600))
    capturer._backfill, capturer.backfill_enabled = stub, True
    late = Game(NHL, "401893000", night - 3 * 3600, state="final")
    world.add(late)

    def mid_run(n: int) -> None:  # a game's final comes due in the middle of the backfill
        if n == 30:
            capturer.schedule_from_slate([row(late, "final")])
            capturer.raw.modify_job("espn:401893000", lambda job: Job(job.game_key, job.league, job.espn_id,
                                                                      job.start, clock.now))

    stub.on_request = mid_run  # type: ignore[method-assign]
    report = capturer.tick()
    assert report["backfill"]["requests"] == 120 and report["backfill"]["stopped"] == "done"
    order = [url for url in world.urls() if "/backfill" in url or "/summary" in url]  # (the static files ran first)
    summary_at = next(n for n, url in enumerate(order) if "/summary" in url)
    assert order[:30] == ["https://site.api.espn.com/backfill"] * 30 and summary_at == 30  # the due job jumped in
    assert capturer.raw.job("espn:401893000").state == "done"
    assert [(c["requests"], c["stopped"]) for c in stub.calls] == [(30, "forward"), (50, "budget"), (40, "done")]
    assert all(c["budget"] == worker_module.BACKFILL_CHUNK for c in stub.calls)
    # Outside the window: nothing
    clock.now = night + 6 * 3600
    assert capturer.run_backfill() is None


def test_a_spent_day_budget_never_starves_the_backfill(env) -> None:
    world, clock = env.world, env.clock
    night = at(DAY + datetime.timedelta(days=1), 2)
    clock.now = night
    capturer = env.make(per_day=0, options=plan.Options(propbets=False))
    stuck = Game(NHL, "401893100", night - 4 * 3600, state="final")
    world.add(stuck)
    capturer.schedule_from_slate([row(stuck, "final")])
    clock.now += 1800
    assert capturer.raw.next_due() <= clock.now and not capturer.has_runnable_due()
    stub = StubBackfill(capturer, world, (night, night + 4 * 3600))
    stub.left = 10
    capturer._backfill, capturer.backfill_enabled = stub, True
    assert capturer.run_backfill()["requests"] == 10  # the past-due job can't run today: the backfill goes on


def test_a_job_on_a_backing_off_host_lets_the_backfill_run_but_one_on_another_host_comes_first(env) -> None:
    world, clock = env.world, env.clock
    night = at(DAY + datetime.timedelta(days=1), 2)
    clock.now = night - 3 * 3600
    capturer = env.make(options=plan.Options(propbets=False))
    soccer = Game(EPL, "401879300", night - 6 * 3600, "Arsenal", "Chelsea", state="final", fixture="sum_epl.json")
    world.add(soccer)
    world.status_for["/plays"] = 503
    capturer.schedule_from_slate([row(soccer, "final")])
    clock.now = capturer.raw.job("espn:401879300").due_at
    capturer.run_due(100)  # the summary is kept; the plays answer 503: sports.core backs off
    clock.now = night
    capturer.raw.modify_job("espn:401879300", lambda job: Job(**{**job.__dict__, "due_at": clock.now}))
    capturer.http._buckets["sports.core.api.espn.com"].closed_until = clock.now + 3 * 3600
    assert capturer.raw.next_due() <= clock.now and not capturer.has_runnable_due()
    stub = StubBackfill(capturer, world, (night, night + 4 * 3600))
    stub.left = 20
    capturer._backfill, capturer.backfill_enabled = stub, True
    late = Game(NHL, "401893500", night - 4 * 3600, state="final")
    world.add(late)

    def due_mid_run(n: int) -> None:  # a game on another host comes due while the backfill runs
        if n == 5:
            capturer.schedule_from_slate([row(late, "final")])
            capturer.raw.modify_job("espn:401893500", lambda job: Job(**{**job.__dict__, "due_at": clock.now}))

    stub.on_request = due_mid_run  # type: ignore[method-assign]
    total = capturer.run_backfill()
    assert total["requests"] == 20  # the job held by its backing-off host never held the backfill back ...
    order = [url for url in world.urls() if "/backfill" in url or "event=401893500" in url]
    assert order.index(next(u for u in order if "event=401893500" in u)) == 5  # ... the other host's job came first
    assert capturer.raw.job("espn:401893500").state == "done"
    assert capturer.raw.job("espn:401879300").kinds_done == ["final:summary"]  # still waiting for its host


def test_the_real_backfill_stops_for_a_due_job_and_carries_on_after_it(env) -> None:
    world, clock = env.world, env.clock
    clock.now = at(DAY + datetime.timedelta(days=1), 2)
    capturer = env.make(config=SimpleNamespace(capture_backfill=True, capture_backfill_per_night=30,
                                               capture_backfill_seasons=1), options=plan.Options(propbets=False))
    capturer.tz = ET
    backfill = capturer.backfill
    assert backfill is not None and backfill.forward_due == capturer.has_runnable_due
    due = Game(NHL, "401893200", at(DAY, 19), state="final")
    world.add(due)
    capturer.schedule_from_slate([row(due, "final")])
    asked = len(world.sent)
    report = backfill.run_night(budget=50)
    assert report["stopped"] == "forward" and len(world.sent) == asked  # not one backfill request
    total = capturer.run_backfill()
    assert capturer.raw.job("espn:401893200").state == "done"
    assert total["requests"] == 30 and capturer.raw.spent("backfill", backfill.night()) == 30
    first_backfill = next(n for n, url in enumerate(world.urls()[asked:]) if "401893200" not in url)
    assert any("event=401893200" in url for url in world.urls()[asked:asked + first_backfill + 1])


# --- the disk floor ----------------------------------------------------------------------------------------------------------------

def test_below_the_disk_floor_nothing_is_fetched_and_it_resumes_with_room(env, monkeypatch) -> None:
    world, clock = env.world, env.clock
    night = at(DAY + datetime.timedelta(days=1), 2)
    clock.now = night
    capturer = env.make(options=plan.Options(propbets=False))
    stub = StubBackfill(capturer, world, (night, night + 4 * 3600))
    stub.left = 5
    capturer._backfill, capturer.backfill_enabled = stub, True
    game = Game(NHL, "401893300", night - 4 * 3600, state="final")
    world.add(game)
    capturer.schedule_from_slate([row(game, "final")])
    clock.now += 1800
    monkeypatch.setattr(raw_store.shutil, "disk_usage", lambda folder: Usage(100e9, 95e9, 5e9))  # 5% free
    assert capturer.run_due(100)["stopped"] == "disk"
    assert capturer.sweep(DAY) == 0 and capturer.run_static() == {"stopped": "disk"}
    assert capturer.run_backfill()["stopped"] == "disk"
    report = capturer.tick()
    assert report["due"]["stopped"] == "disk" and world.sent == []  # zero requests, forward and backfill
    monkeypatch.setattr(raw_store.shutil, "disk_usage", lambda folder: Usage(100e9, 50e9, 50e9))
    clock.now += 61  # room() reads the disk again after a minute
    assert capturer.run_due(100)["fetched"] == 1
    assert capturer.run_backfill()["requests"] == 5

    def unreadable(folder: str) -> Any:
        raise PermissionError("no")

    monkeypatch.setattr(raw_store.shutil, "disk_usage", unreadable)
    clock.now += 61
    assert capturer.room()  # can't tell: the capture runs


# --- the thread, the lease, the dashboard ---------------------------------------------------------------------------------------

def test_only_the_process_holding_the_capture_lease_captures(env, tmp_path) -> None:
    from telegram_betting_bot_simple.data.store import Store
    world, clock = env.world, env.clock
    path = str(tmp_path / "betting.db")
    mine, theirs = Store(path), Store(path)
    theirs.holder = "another-process"
    assert theirs.try_lease(worker_module.LEASE, worker_module.LEASE_SECONDS)
    capturer = env.make(lease=lambda: bool(mine.try_lease(worker_module.LEASE, worker_module.LEASE_SECONDS)))
    world.add(Game(NHL, "401893400", at(DAY, 1), state="final"))
    capturer.schedule_from_slate([row(world.games["401893400"], "final")])
    clock.now = at(DAY, 12)
    assert capturer.tick()["skipped"] == "another process holds the capture lease" and world.sent == []
    rounds: List[float] = []
    worker_module.capture_forever(capturer, every=5.0, rounds=3, sleep=rounds.append)
    assert rounds == [5.0, 5.0]


def test_the_dashboard_starts_the_capture_thread_and_hands_it_the_slate(tmp_path, monkeypatch) -> None:
    from telegram_betting_bot_simple.config import Config
    from telegram_betting_bot_simple.dashboard import __main__ as dashboard_main
    from telegram_betting_bot_simple.data.store import Store
    monkeypatch.delenv("RAW_DB_PATH", raising=False)
    started: List[Any] = []

    class FakeThread:
        def __init__(self, target: Any = None, args: Tuple[Any, ...] = (), name: str = "", daemon: bool = False):
            self.target, self.args, self.name = target, args, name

        def start(self) -> None:
            started.append(self)

    def boot(**settings: str) -> SimpleNamespace:
        started.clear()
        config = Config(TELEGRAM_BOT_TOKEN="123456:test", DASHBOARD_PASSWORD="s3cret", _env_file=None, **settings)
        service = SimpleNamespace(store=Store(str(tmp_path / "betting.db")), cache=TTLCache(), analysts=None,
                                  espn=EspnClient(TTLCache()))
        slate = SimpleNamespace(load=lambda: False, start=lambda: None, kalshi=None, polymarket=None, ratings=None,
                                capturer=None)
        monkeypatch.setattr(dashboard_main, "load_config", lambda: config)
        monkeypatch.setattr(dashboard_main, "build_service", lambda config: service)
        monkeypatch.setattr(dashboard_main, "build_slate", lambda config, service: slate)
        monkeypatch.setattr(dashboard_main, "build_learner", lambda config, service: None)
        monkeypatch.setattr(dashboard_main, "create_app", lambda *args, **kwargs: "app")
        monkeypatch.setattr(dashboard_main.uvicorn, "run", lambda app, **kwargs: None)
        monkeypatch.setattr(dashboard_main, "threading", SimpleNamespace(Thread=FakeThread))
        dashboard_main.main()
        return slate

    slate = boot()
    capture = [thread for thread in started if thread.name == "capture"]
    assert len(capture) == 1 and slate.capturer is capture[0].args[0]
    capturer = capture[0].args[0]
    assert isinstance(capturer, Capturer) and capturer.raw.path == str(tmp_path / "raw.db")
    assert capturer.per_day == 3000 and capturer.raw.min_free == 2 * 1024 ** 3
    assert capturer.kalshi_series == ("KXEPLCORNERS", "KXUCLSOT", "KXUCLSAVE")
    assert capturer.options.core_plays_leagues == plan.WAVE1_SOCCER and capturer.options.propbets
    assert capture[0].target is dashboard_main.capture_forever
    assert capturer.share_door  # the process's door while it holds the capture lease (FIX2 Major 1)
    # CAPTURE_ENABLED=0: no capturer, no thread, the slate as it was
    slate = boot(CAPTURE_ENABLED="0")
    assert slate.capturer is None and not [t for t in started if t.name == "capture"]
    # the settings reach it
    slate = boot(CAPTURE_PER_DAY="100", CAPTURE_PROPBETS="0", CAPTURE_KALSHI_SERIES="none",
                 CAPTURE_CORE_PLAYS_LEAGUES="soccer_epl", RAW_DB_PATH=str(tmp_path / "elsewhere" / "r.db"),
                 CAPTURE_MIN_FREE_GB="5")
    capturer = slate.capturer
    assert capturer.per_day == 100 and not capturer.options.propbets and capturer.kalshi_series == ()
    assert capturer.options.core_plays_leagues == (EPL,) and capturer.raw.path == str(tmp_path / "elsewhere" / "r.db")
    assert capturer.raw.min_free == 5 * 1024 ** 3


def test_the_capture_settings_and_their_defaults(monkeypatch) -> None:
    from telegram_betting_bot_simple.config import Config
    for name in ("CAPTURE_ENABLED", "RAW_DB_PATH", "CAPTURE_PER_DAY", "CAPTURE_CORE_PLAYS_LEAGUES", "CAPTURE_PROPBETS",
                 "CAPTURE_KALSHI_SERIES", "CAPTURE_BACKFILL", "CAPTURE_BACKFILL_PER_NIGHT", "CAPTURE_BACKFILL_SEASONS",
                 "CAPTURE_MIN_FREE_GB"):
        monkeypatch.delenv(name, raising=False)
    config = Config(TELEGRAM_BOT_TOKEN="123456:test", _env_file=None)
    assert (config.capture_enabled, config.raw_db_path, config.capture_per_day, config.capture_propbets,
            config.capture_backfill, config.capture_backfill_per_night, config.capture_backfill_seasons,
            config.capture_min_free_gb) == (True, None, 3000, True, True, 6000, 3, 2.0)
    assert config.capture_core_plays_leagues == ",".join(plan.WAVE1_SOCCER)
    assert config.capture_kalshi_series == "KXEPLCORNERS,KXUCLSOT,KXUCLSAVE"
    blank = Config(TELEGRAM_BOT_TOKEN="123456:test", _env_file=None, CAPTURE_ENABLED="", RAW_DB_PATH=" ",
                   CAPTURE_PER_DAY="", CAPTURE_CORE_PLAYS_LEAGUES="", CAPTURE_PROPBETS="", CAPTURE_KALSHI_SERIES="",
                   CAPTURE_BACKFILL="", CAPTURE_BACKFILL_PER_NIGHT="", CAPTURE_BACKFILL_SEASONS="",
                   CAPTURE_MIN_FREE_GB="")
    assert (blank.capture_enabled, blank.raw_db_path, blank.capture_per_day, blank.capture_core_plays_leagues,
            blank.capture_propbets, blank.capture_kalshi_series, blank.capture_backfill_seasons,
            blank.capture_min_free_gb) == (True, None, 3000, config.capture_core_plays_leagues, True,
                                           config.capture_kalshi_series, 3, 2.0)
    off = Config(TELEGRAM_BOT_TOKEN="123456:test", _env_file=None, CAPTURE_ENABLED="0",
                 CAPTURE_CORE_PLAYS_LEAGUES="none", CAPTURE_KALSHI_SERIES="none", CAPTURE_PROPBETS="false",
                 CAPTURE_BACKFILL="0")
    assert not off.capture_enabled and off.capture_core_plays_leagues == "none" and off.capture_kalshi_series == ""
    assert worker_module.options_from(off).core_plays_leagues == ()
    picked = Config(TELEGRAM_BOT_TOKEN="123456:test", _env_file=None,
                    CAPTURE_CORE_PLAYS_LEAGUES="Soccer_EPL, soccer_usa_mls,soccer_epl", CAPTURE_KALSHI_SERIES="kxeplcorners")
    assert picked.capture_core_plays_leagues == "soccer_epl,soccer_usa_mls"
    assert picked.capture_kalshi_series == "KXEPLCORNERS"
    assert Config(TELEGRAM_BOT_TOKEN="123456:test", _env_file=None, CAPTURE_ENABLED="Yes", CAPTURE_PROPBETS="OFF",
                  CAPTURE_MIN_FREE_GB="0.5", CAPTURE_PER_DAY=" 100 ").capture_min_free_gb == 0.5


# The reviewer's bad values (scratchpad/review-p1/checks/c05_bad_env.py): each crashed both services at startup
BAD_CAPTURE_VALUES = [
    ("CAPTURE_PER_DAY", "abc", "capture_per_day", 3000), ("CAPTURE_PER_DAY", "3,000", "capture_per_day", 3000),
    ("CAPTURE_PER_DAY", "-5", "capture_per_day", 3000), ("CAPTURE_ENABLED", "maybe", "capture_enabled", True),
    ("CAPTURE_ENABLED", "yes please", "capture_enabled", True), ("CAPTURE_PROPBETS", "on?", "capture_propbets", True),
    ("CAPTURE_BACKFILL_PER_NIGHT", "6k", "capture_backfill_per_night", 6000),
    ("CAPTURE_BACKFILL_SEASONS", "20", "capture_backfill_seasons", 3),
    ("CAPTURE_BACKFILL_SEASONS", "0", "capture_backfill_seasons", 3),
    ("CAPTURE_MIN_FREE_GB", "-1", "capture_min_free_gb", 2.0), ("CAPTURE_MIN_FREE_GB", "2GB", "capture_min_free_gb", 2.0),
    ("CAPTURE_CORE_PLAYS_LEAGUES", "soccer_epl,soccer_xyz", "capture_core_plays_leagues", "soccer_epl"),
    ("CAPTURE_KALSHI_SERIES", "EPLCORNERS", "capture_kalshi_series", "KXEPLCORNERS,KXUCLSOT,KXUCLSAVE"),
    ("RAW_DB_PATH", "/var/lib/bettingbot/betting.db", "raw_db_path", None),
    # and a few more of the same kinds
    ("CAPTURE_MIN_FREE_GB", "nan", "capture_min_free_gb", 2.0), ("CAPTURE_PER_DAY", "1e3", "capture_per_day", 3000),
    ("CAPTURE_BACKFILL", "2", "capture_backfill", True),
    ("CAPTURE_CORE_PLAYS_LEAGUES", "soccer_xyz", "capture_core_plays_leagues", ",".join(plan.WAVE1_SOCCER)),
    ("CAPTURE_KALSHI_SERIES", "KXEPLCORNERS,EPL CORNERS", "capture_kalshi_series", "KXEPLCORNERS"),
]


@pytest.mark.parametrize("name, value, field, default", BAD_CAPTURE_VALUES)
def test_a_bad_capture_setting_never_stops_the_bot_or_the_dashboard(tmp_path, monkeypatch, caplog, name, value,
                                                                     field, default) -> None:
    """M3: load_config (the bot's and the dashboard's) starts with any capture value; the bad one logs one warning
    naming the setting, never the value, and its default (or the rest of its list) is used."""
    from telegram_betting_bot_simple.config import load_config
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test")
    monkeypatch.delenv("STATE_DIRECTORY", raising=False)
    monkeypatch.setenv(name, value)
    caplog.set_level("WARNING", logger="telegram_betting_bot_simple.config")
    config = load_config(str(tmp_path / ".env"))
    assert getattr(config, field) == default
    warnings = [record.getMessage() for record in caplog.records if record.name == "telegram_betting_bot_simple.config"]
    assert len(warnings) == 1 and name in warnings[0], warnings
    if name not in ("CAPTURE_CORE_PLAYS_LEAGUES", "CAPTURE_KALSHI_SERIES", "RAW_DB_PATH") and any(
            ch.isalpha() for ch in value):
        assert value not in warnings[0]  # (a key pasted into the wrong line never reaches the logs)


def test_the_setup_script_documents_and_checks_every_capture_setting() -> None:
    text = (Path(__file__).resolve().parent.parent / "deploy" / "setup_vps.sh").read_text()
    for name, default in (("CAPTURE_ENABLED", "0"), ("RAW_DB_PATH", "/var/lib/bettingbot/raw.db"),
                          ("CAPTURE_PER_DAY", "3000"), ("CAPTURE_CORE_PLAYS_LEAGUES", "soccer_epl"),
                          ("CAPTURE_PROPBETS", "0"), ("CAPTURE_KALSHI_SERIES", "KXEPLCORNERS,KXUCLSOT,KXUCLSAVE"),
                          ("CAPTURE_BACKFILL", "0"), ("CAPTURE_BACKFILL_PER_NIGHT", "6000"),
                          ("CAPTURE_BACKFILL_SEASONS", "3"), ("CAPTURE_MIN_FREE_GB", "2")):
        assert re.search(rf"^#   {name}={re.escape(default)}", text, re.M), name  # documented with its default
        assert f"  [{name}]='^" in text, name  # can be given on the command line, checked
    import subprocess
    import shutil
    bash = shutil.which("bash")
    if bash:
        patterns = dict(re.findall(r"^  \[(CAPTURE_\w+|RAW_DB_PATH)\]='(.*)'$", text, re.M))
        for name, value, ok in (("RAW_DB_PATH", "/var/lib/bettingbot/raw.db", True),
                                ("RAW_DB_PATH", "/etc/raw.db", False), ("CAPTURE_MIN_FREE_GB", "2.5", True),
                                ("CAPTURE_KALSHI_SERIES", "none", True), ("CAPTURE_BACKFILL_SEASONS", "0", False),
                                ("CAPTURE_CORE_PLAYS_LEAGUES", "soccer_epl,soccer_spl", True)):
            run = subprocess.run([bash, "-c", '[[ "$1" =~ $2 ]]', "-", value, patterns[name]])
            assert (run.returncode == 0) is ok, (name, value)


def test_summary_state_reads_the_header() -> None:
    def summary(name: str, state: str, completed: bool = False) -> Dict[str, Any]:
        return {"header": {"competitions": [{"status": {"type": {"name": name, "state": state,
                                                                 "completed": completed}}}]}}
    assert summary_state(summary("STATUS_FINAL", "post", True)) == "final"
    assert summary_state(summary("STATUS_FULL_TIME", "post", True)) == "final"
    assert summary_state(summary("STATUS_POSTPONED", "post")) == "void"
    assert summary_state(summary("STATUS_CANCELED", "post")) == "void"
    assert summary_state(summary("STATUS_IN_PROGRESS", "in")) == "live"
    assert summary_state(summary("STATUS_SUSPENDED", "in")) == "live"  # resumes later: not void
    assert summary_state(summary("STATUS_SCHEDULED", "pre")) == "pre"
    assert summary_state({"header": {}}) is None and summary_state("nope") is None


def test_an_unchanged_slate_hand_off_takes_no_write_lock(env) -> None:
    world, clock = env.world, env.clock
    clock.now = at(DAY, 8)
    games = [Game(NHL, f"40189500{n}", at(DAY, 19), state="pre") for n in range(6)]
    world.add(*games)
    capturer = env.make()
    assert capturer.schedule_from_slate([row(g) for g in games]) == 6
    writes: List[str] = []
    real = capturer.raw.modify_job

    def counted(key: str, change: Any, **kwargs: Any) -> Any:
        writes.append(key)
        return real(key, change, **kwargs)

    capturer.raw.modify_job = counted  # type: ignore[method-assign]
    clock.now += 60
    assert capturer.schedule_from_slate([row(g) for g in games]) == 0 and writes == []  # one read, no write
    games[0].start += 3600  # one game moved: only its job is written
    assert capturer.schedule_from_slate([row(g) for g in games]) == 0 and writes == ["espn:401895000"]


def test_the_graders_output_is_byte_identical_and_it_asks_espn_nothing_after_a_capture(env, tmp_path) -> None:
    """The grader (DashboardService.grade_due) grades the same call to the same bytes whether the capture fetched the
    game's summary (seeding the lean cache) or the grader fetched it itself; after a capture it makes no request."""
    from telegram_betting_bot_simple.dashboard.service import DashboardService
    from telegram_betting_bot_simple.data.store import Store

    class NoOdds:
        quota = {"remaining": 480, "used": 20}

        def sports(self) -> List[Any]:
            return []

        def league_games(self, league: str, refresh: bool = False) -> List[Any]:
            return []

        def odds_loaded_at(self, league: str) -> None:
            return None

    world, clock = env.world, env.clock
    world.add(Game(NBA, "401909278", at(DAY, 19, 30), "Boston Celtics", "Chicago Bulls", state="final",
                   fixture="sum_nba.json"))
    clock.now = at(DAY, 23)

    def graded(captured: bool) -> Tuple[str, List[str]]:
        cache = TTLCache(name=f"grader-{captured}", clock=clock)
        espn = EspnClient(cache, clock=clock)
        if captured:
            capturer = env.make(f"raw-{captured}.db", cache=cache)
            capturer.espn = espn
            capturer.schedule_from_slate([row(world.games["401909278"], "final")])
            clock.now += 1800
            assert capturer.run_due(10)["stored"] == 1
        store = Store(str(tmp_path / f"betting-{captured}.db"))
        store.save_prediction({"player": "web:abc12345", "name": "Sam", "league": NBA, "event_id": "o1",
                               "espn_id": "401909278", "home": "Boston Celtics", "away": "Chicago Bulls",
                               "start": "2026-10-30T23:30:00Z",
                               "call": {"winner": "home", "confidence": 0.7, "home_score": 112, "away_score": 104},
                               "ai": {"home_win": 0.64, "margin": 4.5, "total": 221.5}})
        service = DashboardService(NoOdds(), espn, ET, store=store)
        lean_before = len(world.lean)
        assert service.grade_due() == 1
        with store._connect() as db:
            rows = db.execute("SELECT status, result, user_points, ai_points FROM predictions").fetchall()
        return json.dumps(rows, sort_keys=True), world.lean[lean_before:]

    plain, asked_plain = graded(False)
    captured, asked_captured = graded(True)
    assert captured == plain and json.loads(plain)[0][0] == "graded"
    assert len(asked_plain) == 1 and "/summary" in asked_plain[0]  # the grader's own fetch...
    assert asked_captured == []  # ...not needed after the capture: the seeded cache answered


def test_a_definitive_pre_answer_serves_its_horizon_and_a_server_error_is_retried(env) -> None:
    world, clock = env.world, env.clock
    start = at(DAY + datetime.timedelta(days=3), 19)
    early = Game(NBA, "401909900", start)  # three days out: no props posted yet
    busy = Game(NBA, "401909901", start)
    world.add(early, busy)
    world.status_for["events/401909900/competitions/401909900/odds"] = 404
    world.status_for["events/401909901/competitions/401909901/odds"] = 503
    clock.now = start - 3 * 86400
    capturer = env.make()
    capturer.schedule_from_slate([row(early), row(busy)])
    capturer.run_due(10)
    assert capturer.raw.job("espn:401909900").kinds_done == ["pre:open:propbets"]  # answered: not asked again
    retry = capturer.raw.job("espn:401909901")
    assert retry.kinds_done == [] and retry.due_at == pytest.approx(clock.now + worker_module.PRE_RETRY)
    asked = len(world.urls("propBets"))
    clock.now += worker_module.PRE_RETRY + 60
    capturer.run_due(10)
    assert len(world.urls("propBets")) == asked + 1  # only the 503 one again (after the host's back-off)
    assert worker_module.answered(404) and not worker_module.answered(429) and not worker_module.answered(503)


# --- the fix round: counting, a full disk, the learn backfill, the window's time zone ----------------------------------

def tag_backfill(capturer: Capturer, world: World) -> Any:
    """The world tags the requests the backfill's run_night makes (independently of how the worker counts them)."""
    backfill = capturer.backfill
    real = backfill.run_night

    def run_night(*args: Any, **kwargs: Any) -> Any:
        world.backfilling = True
        try:
            return real(*args, **kwargs)
        finally:
            world.backfilling = False

    backfill.run_night = run_night  # type: ignore[method-assign]
    return backfill


def test_a_forward_day_and_a_backfill_night_are_each_counted_once_under_their_own_kinds(env) -> None:
    """F1: http_ok / http_error count the forward requests only (the health block's alarm), backfill_http_ok /
    backfill_http_error the backfill's; every request once, by UTC day."""
    world = env.world
    day = recorded_day(world)
    world.status_for["event=401800001"] = 503  # some forward errors too
    capturer = env.make(config=SimpleNamespace(capture_backfill=True, capture_backfill_per_night=60,
                                               capture_backfill_seasons=1))
    capturer.tz = ET
    tag_backfill(capturer, world)
    run_day(capturer, world, day, at(DAY + datetime.timedelta(days=1), 7), step=1800)
    days = sorted({raw_store.utc_day(ts) for ts, _who, _url in world.tagged})
    totals = collections.Counter(who for _ts, who, _url in world.tagged)
    assert totals["forward"] > 0 and totals["backfill"] > 0
    raw = capturer.raw
    for utc in days:
        forward = sum(1 for ts, who, _u in world.tagged if who == "forward" and raw_store.utc_day(ts) == utc)
        backfill = sum(1 for ts, who, _u in world.tagged if who == "backfill" and raw_store.utc_day(ts) == utc)
        assert raw.spent("http_ok", utc) + raw.spent("http_error", utc) == forward, utc
        assert raw.spent("backfill_http_ok", utc) + raw.spent("backfill_http_error", utc) == backfill, utc
    assert sum(raw.spent("http_error", utc) for utc in days) >= 1  # the 503s are forward errors


def test_a_full_raw_db_wastes_at_most_one_request_a_tick(env, monkeypatch) -> None:
    """F8: SQLite's "database or disk is full" (free blocks still say there is room) turns room() off for 5 minutes:
    after the first answer raw.db can't store, nothing more is asked (the reviewer's c10: 63 requests in 5 ticks)."""
    world, clock = env.world, env.clock
    games = [Game(NBA, f"40190{n:04d}", at(DAY, 12), state="final", fixture="sum_nba.json") for n in range(12)]
    world.add(*games)
    clock.now = at(DAY, 18)
    capturer = env.make(options=plan.Options(propbets=False))
    for entry in plan.STATIC:  # the static files already ran today, and yesterday's boards were swept
        capturer.raw.set_state(f"static:{entry.name}", {"at": clock.now, "ok": True,
                                                        "period": plan.static_period(entry, clock.now)})
    capturer.raw.set_state("sweep:last", (DAY - datetime.timedelta(days=1)).isoformat())
    capturer.schedule_from_slate([row(g, "final") for g in games])
    clock.now = at(DAY, 23)  # every final due
    with capturer.raw._connect() as db:
        pages = db.execute("PRAGMA page_count").fetchone()[0]
    real = RawStore._connect

    def capped(self: RawStore) -> Any:  # no page more: a new summary can't be stored
        db = real(self)
        db.execute(f"PRAGMA max_page_count = {pages}")
        return db

    monkeypatch.setattr(RawStore, "_connect", capped)
    per_tick = []
    for _ in range(6):
        before = len(world.sent)
        report = capturer.tick()
        per_tick.append(len(world.sent) - before)
        clock.now += 61
    assert max(per_tick) <= 1 and sum(per_tick) <= 2, per_tick  # the first failure, then one try after 5 minutes
    assert report["due"]["stopped"] == "disk" or per_tick[-1] == 1
    assert all(job.state == "todo" and job.attempts == 0 for job in capturer.raw.jobs())  # no attempt spent
    monkeypatch.setattr(RawStore, "_connect", real)
    clock.now += raw_store.FULL_PAUSE
    assert capturer.run_due(100)["jobs_done"] == 12  # room again: everything is captured


def test_a_job_whose_own_row_raw_db_cant_write_stops_the_run_and_waits(env, monkeypatch) -> None:
    """F8: a job's answers were stored but its row can't be written (SQLite full): the run stops with "disk" (no other
    job is asked, nothing raised), and the job is captured once there is room."""
    world, clock = env.world, env.clock
    games = [Game(NBA, f"40192{n:04d}", at(DAY, 12), state="final", fixture="sum_nba.json") for n in range(3)]
    world.add(*games)
    clock.now = at(DAY, 18)
    capturer = env.make(options=plan.Options(propbets=False))
    capturer.schedule_from_slate([row(g, "final") for g in games])
    clock.now = at(DAY, 23)  # every final due
    raw, real_modify = capturer.raw, capturer.raw.modify_job

    def refused(*args: Any, **kwargs: Any) -> Any:  # as RawStore._writing does with SQLite's SQLITE_FULL
        raw._full()
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(raw, "modify_job", refused)
    before = len(world.sent)
    assert capturer.run_due(100)["stopped"] == "disk"
    asked = {g.event_id for g in games if any(g.event_id in url for url in world.urls()[before:])}
    assert len(asked) == 1  # the first job only
    monkeypatch.setattr(raw, "modify_job", real_modify)
    clock.now += raw_store.FULL_PAUSE + 1
    assert capturer.run_due(100)["jobs_done"] == 3


@pytest.mark.parametrize("case", ["body", "jobs", "state", "floor"])
def test_the_sweep_stops_at_the_first_board_raw_db_cant_store(env, monkeypatch, caplog, case) -> None:
    """F8: the sweep stops at the first board raw.db can't take, whichever write SQLite refuses (the board itself, the
    jobs of its finals, the sweep's progress) or when the disk crossed its floor after the first board: no other board
    is asked for, nothing is logged as a failure, no try is spent, and the next sweep with room makes every job."""
    world, clock = env.world, env.clock
    games = [Game(NBA, f"40191{n:04d}", at(DAY, 19), state="final") for n in range(150)]  # a big first board
    if case == "body":
        games[0].home = os.urandom(40_000).hex()  # nothing compresses it: the board needs pages raw.db can't add
    world.add(*games)
    clock.now = at(DAY + datetime.timedelta(days=1), 6)
    capturer = env.make()
    raw = capturer.raw
    with raw._connect() as db:
        pages = db.execute("PRAGMA page_count").fetchone()[0]
    real, real_set, full = RawStore._connect, raw.set_state, [True]

    def capped(self: RawStore) -> Any:  # no page more
        db = real(self)
        if full[0]:
            db.execute(f"PRAGMA max_page_count = {pages}")
        return db

    def refused(key: str, value: Any) -> None:  # the sweep's progress can't be written (as RawStore._writing does)
        if full[0] and key.startswith("sweep:"):
            raw._full()
            raise sqlite3.OperationalError("database or disk is full")
        real_set(key, value)

    if case in ("body", "jobs"):
        monkeypatch.setattr(RawStore, "_connect", capped)
    elif case == "state":
        monkeypatch.setattr(raw, "set_state", refused)
    else:  # the disk's floor: crossed once the first board was asked for
        monkeypatch.setattr(raw, "room", lambda: not full[0] or not world.urls("/scoreboard"))
    caplog.set_level(logging.WARNING)
    capturer.sweep(DAY)
    assert len(world.urls("/scoreboard")) == 1  # not the other 57
    assert capturer.last["sweep"]["stopped"] == "disk"
    board = raw.latest(scoreboard_key(NBA, DAY.strftime("%Y%m%d")), "espn_site", "scoreboard")
    assert (board is None) == (case == "body")  # each case's write is the one refused
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], caplog.text  # a pause, not a failure
    assert capturer.sweep_day() == DAY  # no try was spent: swept once there is room
    full[0] = False
    clock.now += raw_store.FULL_PAUSE + 1
    capturer.sweep(DAY)
    assert len(set(world.urls("/scoreboard"))) == 58 and raw.get_state("sweep:last") == DAY.isoformat()
    assert sorted(job.espn_id for job in raw.jobs() if job.league == NBA) == sorted(g.event_id for g in games)


def test_a_full_raw_db_stops_the_static_files_at_the_first_one_it_cant_store(env, monkeypatch) -> None:
    """F8: the static files stop at the first answer raw.db can't store (no other file or source is asked for that
    tick), and none is marked as run: they are fetched once there is room."""
    world, clock = env.world, env.clock
    capturer = env.make()
    route = world.route

    def incompressible(url: str) -> Tuple[int, bytes]:  # each file as big as its bytes: compression can't make it fit
        status, body = route(url)
        return status, os.urandom(60_000) if status == 200 else body

    monkeypatch.setattr(world, "route", incompressible)
    with capturer.raw._connect() as db:
        pages = db.execute("PRAGMA page_count").fetchone()[0]
    real = RawStore._connect

    def capped(self: RawStore) -> Any:  # no page more: no file fits
        db = real(self)
        db.execute(f"PRAGMA max_page_count = {pages}")
        return db

    monkeypatch.setattr(RawStore, "_connect", capped)
    report = capturer.run_static(force=True)
    assert report["stopped"] == "disk" and len(world.sent) == 1, report
    assert capturer.run_static(force=True) == {"stopped": "disk"} and len(world.sent) == 1  # room() is off
    assert all(capturer.raw.get_state(f"static:{entry.name}") is None for entry in plan.STATIC)
    monkeypatch.setattr(RawStore, "_connect", real)
    clock.now += raw_store.FULL_PAUSE + 1
    report = capturer.run_static()
    assert all(report[entry.name]["ok"] for entry in plan.STATIC), report


def test_while_the_learn_backfill_runs_the_capture_backfill_waits_for_the_next_tick(env) -> None:
    """M2 (the worker's half): RawBackfill stops with "learn" while learn.history's backfill runs; the worker stops its
    chunks for that tick (it doesn't spin) and tries again on the next one."""
    clock = env.clock
    night = at(DAY + datetime.timedelta(days=1), 2)
    clock.now = night
    capturer = env.make()
    calls: List[int] = []

    class LearnBusy:
        stopped, requests = "learn", 0

        def in_window(self, ts: Optional[float] = None) -> bool:
            return True

        def run_night(self, budget: Optional[int] = None) -> Dict[str, Any]:
            calls.append(budget or 0)
            assert len(calls) <= 20, "the worker spins: it asks the backfill again and again in one tick"
            return {"ran": self.stopped != "learn", "requests": self.requests, "stopped": self.stopped}

    stub = LearnBusy()
    capturer._backfill, capturer.backfill_enabled = stub, True
    report = capturer.run_backfill()
    assert report["stopped"] == "learn" and calls == [worker_module.BACKFILL_CHUNK]  # one call, then the tick moves on
    stub.stopped = "done"
    clock.now += 60
    assert capturer.run_backfill()["stopped"] == "done" and len(calls) == 2
    # The learn backfill started part-way through a chunk (a few requests made): the tick's chunks stop as well
    stub.stopped, stub.requests = "learn", 3
    assert capturer.run_backfill()["stopped"] == "learn" and len(calls) == 3
    # "backoff" (only retries left whose wait isn't over) stops the tick's chunks the same way
    stub.stopped, stub.requests = "backoff", 0
    assert capturer.run_backfill()["stopped"] == "backoff" and len(calls) == 4


def test_the_backfill_window_follows_the_owners_time_zone(env, caplog) -> None:
    """Decision 1: RawBackfill reads its 01:00-06:00 window in TIMEZONE (a UTC server's 01:00 is New York's evening)."""
    from zoneinfo import ZoneInfo
    capturer = env.make(config=SimpleNamespace(capture_backfill=True, timezone="America/Chicago"))
    assert capturer.tz is None and capturer.backfill.tz == ZoneInfo("America/Chicago")
    default = env.make("raw2.db", config=SimpleNamespace(capture_backfill=True))  # no TIMEZONE: New York
    assert default.backfill.tz == ZoneInfo("America/New_York")
    caplog.set_level("WARNING")
    bad = env.make("raw3.db", config=SimpleNamespace(capture_backfill=True, timezone="Not/AZone"))
    assert bad.backfill.tz is None  # server-local time, with one warning
    given = env.make("raw4.db", config=SimpleNamespace(capture_backfill=True, timezone="America/Chicago"))
    given.tz = datetime.timezone.utc  # an explicit zone (the tests' rigs) wins
    assert given.backfill.tz == datetime.timezone.utc


def test_an_unchanged_close_snapshot_is_found_at_close(env) -> None:
    """F3 end to end: propBets that didn't move between t90 and close are stored once, and latest(horizon="close")
    finds the snapshot seen at close."""
    world, clock = env.world, env.clock
    world.propbets_moves = False
    start = at(DAY, 19, 30)
    game = Game(NBA, "401909950", start)
    world.add(game)
    clock.now = start - 2 * 3600
    capturer = env.make()
    capturer.schedule_from_slate([row(game)])
    for minutes in range(0, 120, 5):
        clock.now = start - 2 * 3600 + minutes * 60
        capturer.run_due(100)
    close = capturer.raw.latest("espn:401909950", "espn_propbets", "propbets", horizon="close")
    t90 = capturer.raw.latest("espn:401909950", "espn_propbets", "propbets", horizon="t90")
    assert close is not None and t90 is not None and close.id == t90.id
    assert start - 10 * 60 <= close.seen_at < start and close.fetched_at < close.seen_at
    assert [r[3] for r in rows_of(capturer.raw, "espn:401909950") if r[0] == "espn_propbets"] == ["t24"]


# --- FIX2: one bucket per host for the process, forward first to the request, E1's summary half ---------------------------

def quiet_tick(capturer: Capturer, clock: Clock) -> None:
    """Nothing for a tick to do but the lease: the static files ran today and yesterday's boards were swept."""
    for entry in plan.STATIC:
        capturer.raw.set_state(f"static:{entry.name}", {"at": clock.now, "ok": True,
                                                        "period": plan.static_period(entry, clock.now)})
    capturer.raw.set_state("sweep:last", (worker_module.espn_day(clock.now) - datetime.timedelta(days=1)).isoformat())


def test_the_dashboards_capturer_lends_its_door_while_it_holds_the_capture_lease(env) -> None:
    """Major 1: build_capturer's capturer (share_door) makes its PoliteHttp the process's door at each tick it holds
    the capture lease, and withdraws it when another process holds it; a capturer that isn't the dashboard's lends
    nothing, and a door that isn't this capturer's is never withdrawn by it."""
    clock = env.clock
    held = [True]
    capturer = env.make(lease=lambda: held[0])
    quiet_tick(capturer, clock)
    assert not capturer.share_door
    capturer.tick()
    assert capture_http.shared() is None
    capturer.share_door = True
    capturer.tick()
    assert capture_http.shared() is capturer.http
    held[0] = False
    assert capturer.tick()["skipped"] and capture_http.shared() is None
    other = PoliteHttp()
    capture_http.set_shared(other)
    capturer.tick()
    assert capture_http.shared() is other


def test_a_final_that_falls_due_during_a_backfill_requests_token_wait_has_no_backfill_request_ahead(env) -> None:
    """Minor 2 at the worker (the final check's c16: 6 of 4,892 backfill requests): a final falls due while a backfill
    request waits for its token. That request is refused, and the final's request is the next one sent."""
    world, clock = env.world, env.clock
    clock.now = at(DAY + datetime.timedelta(days=1), 2)  # 02:00 New York: the backfill's window
    capturer = env.make(config=SimpleNamespace(capture_backfill=True, capture_backfill_per_night=400,
                                               capture_backfill_seasons=1))
    tag_backfill(capturer, world)
    game = Game(NBA, "401909990", at(DAY, 19), state="final", fixture="sum_nba.json")
    world.add(game)
    fell_due: List[float] = []
    real_sleep = capturer.http.sleep

    def sleep(seconds: float) -> None:  # the final falls due half-way through a backfill request's wait
        if world.backfilling and seconds > 0 and not fell_due:
            fell_due.append(clock.now + seconds / 2)
            capturer.raw.add_job(f"espn:{game.event_id}", NBA, game.event_id, game.start, fell_due[0], now=clock.now)
        real_sleep(seconds)

    capturer.http.sleep = sleep  # type: ignore[method-assign]
    report = capturer.run_backfill(max_seconds=300.0)
    assert report is not None and fell_due
    first = min(t for t, who, url in world.tagged if who == "forward" and f"event={game.event_id}" in url)
    ahead = [url for t, who, url in world.tagged if who == "backfill" and fell_due[0] <= t <= first]
    assert ahead == [], ahead
    assert sum(1 for _t, who, _url in world.tagged if who == "backfill") > 3  # the backfill ran before and after
    assert capturer.raw.job(f"espn:{game.event_id}").state == "done"


LEARN_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "learn"


def test_forward_capture_and_the_learn_backfill_on_two_threads_keep_to_one_bucket_per_host(env, tmp_path) -> None:
    """Major 1 (the final check's c13, as a test): finals due at the learner's 02:30 on a UTC server while the learn
    backfill (its raw-first hook on) runs, each on a real thread, every pace scaled x20. The capture's PoliteHttp is
    the process's door, so site.api.espn.com stays within its bucket (1 a second, a burst of 3) over the run and over
    every 60 s, both doors asking it all along (c13 without the door: 1.25/s, 76 requests in 60 s)."""
    from telegram_betting_bot_simple.data.store import Store
    from telegram_betting_bot_simple.learn import history
    from telegram_betting_bot_simple.learn.sources.http import HistHttp
    from telegram_betting_bot_simple.learn.sources.http import Refused as LearnRefused
    scale, finals, utc = 20.0, 60, datetime.timezone.utc
    t0 = datetime.datetime(2026, 11, 3, 2, 30, tzinfo=utc).timestamp()
    real0 = time.monotonic()

    def moving() -> float:  # the app's clock, moving with real time from 02:30 UTC
        return t0 + (time.monotonic() - real0)

    world = World(moving)  # type: ignore[arg-type]
    for n in range(finals):
        world.add(Game(NBA, f"40193{n:04d}", t0 - 4 * 3600, state="final", fixture="sum_nba.json"))
    raw = RawStore(str(tmp_path / "c13-raw.db"), clock=moving)
    for game in world.games.values():
        raw.add_job(f"espn:{game.event_id}", NBA, game.event_id, game.start, t0 - 60.0, now=t0 - 60.0)
    sent: List[Tuple[float, str, str]] = []
    lock = threading.Lock()

    def note(door: str, url: str) -> None:
        with lock:
            sent.append((time.monotonic(), door, urllib.parse.urlsplit(url).hostname or ""))

    http = PoliteHttp(rates={"site.api.espn.com": (scale, 3), "sports.core.api.espn.com": (scale, 3)})
    http._send = lambda request: (note("forward", request.full_url), world.send(request))[1]  # type: ignore
    capturer = Capturer(None, raw, EspnClient(TTLCache(name="c13", clock=moving), clock=moving), http, moving,
                        config=SimpleNamespace(capture_backfill=False), options=plan.Options(propbets=False),
                        per_day=10 ** 6, tz=utc, lease=lambda: True, share_door=True)
    capturer._lend_door(True)  # as its tick does while it holds the capture lease
    stop = threading.Event()

    def learn_transport(url: str, params: Any, timeout: float) -> Tuple[int, bytes]:
        note("learn", url)
        if url.endswith("/events"):  # 30 games a day: work for the whole run
            base = int(str(params.get("dates", "0")).split("-")[0]) * 100
            items = [{"$ref": f"http://sports.core.api.espn.pvt/v2/sports/basketball/leagues/nba/events/{base + n}"}
                     for n in range(30)]
            return 200, json.dumps({"count": 30, "pageIndex": 1, "pageCount": 1, "items": items}).encode()
        if url.endswith("/odds"):
            return 200, (LEARN_FIXTURES / "espn_nba_odds.json").read_bytes()
        if url.endswith("/summary"):
            return 200, (LEARN_FIXTURES / "espn_nba_summary.json").read_bytes()
        return 404, b"{}"

    store = Store(str(tmp_path / "c13-betting.db"))
    hist = HistHttp(pace=history.PACE / scale, transport=learn_transport)
    learn = history.Backfill(store, TTLCache(store, name="c13-learn", clock=moving), moving, http=hist, tz=utc,
                             per_night=100000, sources=["espn"], leagues=[NBA], raw=raw)
    assert learn.in_window(t0)
    own_allow = hist.allow

    def allow() -> None:  # the learner thread stops at its next request once the test is done
        if stop.is_set():
            raise LearnRefused("stopped")
        own_allow()

    hist.allow = allow

    def forward() -> None:
        while not stop.is_set() and raw.due_jobs(moving()):
            capturer.run_due(300)

    def learner() -> None:
        while not stop.is_set():
            learn.run_night(budget=100000)

    threads = [threading.Thread(target=forward, daemon=True), threading.Thread(target=learner, daemon=True)]
    start = time.monotonic()
    for thread in threads:
        thread.start()
    threads[0].join(timeout=60)
    end = time.monotonic()
    stop.set()
    threads[1].join(timeout=60)
    assert not any(thread.is_alive() for thread in threads)
    site = [(at_, door) for at_, door, host in sent if host == "site.api.espn.com" and start <= at_ <= end]
    doors = collections.Counter(door for _at, door in site)
    assert doors["forward"] == finals and doors["learn"] >= 3, doors  # both doors asked it at once
    times = sorted(at_ for at_, _door in site)
    span, window = end - start, 60.0 / scale  # 60 s at the real rate
    rate = (len(times) - 3) / span / scale
    assert len(times) <= 3 + scale * span + 2, f"{rate:.2f}/s over {span * scale:.0f} s"
    busiest = max(sum(1 for other in times if at_ <= other <= at_ + window) for at_ in times)
    assert busiest <= 3 + 60 + 2, busiest
    assert all(job.state == "done" for job in raw.jobs())


def test_a_2xx_summary_that_isnt_json_is_kept_raw_seeds_nothing_and_its_job_tries_again(env, monkeypatch) -> None:
    """E1's summary half: ESPN answers a final's summary with a 2xx that isn't JSON (a maintenance page). summary_raw
    gives it back as it came and seeds nothing (no lean copy, no state noted); the worker keeps it raw as an unreadable
    copy (state "live", no horizon) and the job tries again later, as a board's does, and is done once ESPN answers."""
    world, clock = env.world, env.clock
    game = Game(NBA, "401909980", at(DAY, 12), state="final", fixture="sum_nba.json")
    world.add(game)
    clock.now = at(DAY, 18)
    capturer = env.make(options=plan.Options(propbets=False))
    capturer.schedule_from_slate([row(game, "final")])
    page, broken = b"<html><body>ESPN is down for maintenance</body></html>", [True]
    real_summary = world.summary
    monkeypatch.setattr(world, "summary", lambda event_id: (200, page) if broken[0] and event_id == game.event_id
                        else real_summary(event_id))
    key, lean = f"espn:{game.event_id}", ("espn", "summary", NBA, game.event_id)
    assert capturer.espn.summary_raw(NBA, game.event_id, http=capturer.http) == page
    assert capturer.espn.cache.peek(lean) is None and (NBA, game.event_id) not in capturer.espn._states
    clock.now = at(DAY, 23)  # the final is due
    capturer.run_due(100)
    kept = capturer.raw.latest(key, "espn_site", "summary")
    assert kept is not None and kept.raw_bytes() == page and kept.http_status == 200
    assert kept.state == "live" and kept.horizon is None  # unreadable: not taken for the final record
    job = capturer.raw.job(key)
    assert job.state == "todo" and job.attempts == 1 and "unreadable" in (job.last_error or "")
    assert capturer.espn.cache.peek(lean) is None and (NBA, game.event_id) not in capturer.espn._states
    broken[0] = False
    clock.now = job.due_at + 1
    capturer.run_due(100)
    assert capturer.raw.job(key).state == "done"
    final = capturer.raw.latest(key, "espn_site", "summary")
    assert final is not None and final.state == "final" and final.horizon == "final"
    assert capturer.espn.cache.peek(lean) is not None  # the readable answer seeds the grader's copy
