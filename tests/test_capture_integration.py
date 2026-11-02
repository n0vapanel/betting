"""P1 integration: the raw capture and the older-games backfill working as one system, over the real modules.

Everything here runs together for real: RawStore on a temp dir, one PoliteHttp whose transport (``_send``) is the
fake ``World`` below (it serves tests/fixtures/capture/* by URL and records every request with its fake-clock time),
the Capturer (capture/worker.py), RawBackfill (capture/backfill.py, built by the worker as the dashboard builds it),
EspnClient.summary_raw / scoreboard_raw seeding the lean cache, the grader (DashboardService.grade_due), the learn
backfill (learn/history.Backfill) with its raw-first hook as learn/loop.py wires it, and the Brain progress page's
capture block (learn/progress.py) over the raw.db the run produced. The clock and every sleep are fake; the disk is
patched (this machine's own disk may be under the capture's floor). Nothing touches the network: requests, urllib's
opener and every door are refused or faked, and The Odds API, Anthropic and DeepSeek are counted at zero.
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import datetime
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

import pytest
import requests

from telegram_betting_bot_simple.capture import backfill as bf
from telegram_betting_bot_simple.capture import plan, raw_store
from telegram_betting_bot_simple.capture import worker as worker_module
from telegram_betting_bot_simple.capture.http import DEFAULT_RATE, GITHUB, RATES, HttpResponse, PoliteHttp, is_github
from telegram_betting_bot_simple.capture.raw_store import RawStore
from telegram_betting_bot_simple.capture.sources import scoreboard_key
from telegram_betting_bot_simple.capture.worker import Capturer
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.espn_client import EspnClient
from telegram_betting_bot_simple.data.mlb_stats import MlbStats
from telegram_betting_bot_simple.data.store import Store

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "capture"
LEARN_FIX = ROOT / "tests" / "fixtures" / "learn"
ESPN_FIX = ROOT / "tests" / "fixtures" / "espn"
MARKETS_FIX = ROOT / "tests" / "fixtures" / "markets"
ET = ZoneInfo("America/New_York")
UTC = datetime.timezone.utc
DAY = datetime.date(2026, 10, 30)  # the recorded day: a Friday on ESPN's (US Eastern) calendar
NBA, NFL, NHL, MLB, EPL, UCL = ("basketball_nba", "americanfootball_nfl", "icehockey_nhl", "baseball_mlb",
                                "soccer_epl", "soccer_uefa_champs_league")
UFC, ATP = "mma_mixed_martial_arts", "tennis_atp"
FORBIDDEN = ("the-odds-api.com", "anthropic.com", "deepseek.com")  # never asked, by anything here
ESPN_HOSTS = {"site.api.espn.com", "sports.core.api.espn.com", "site.web.api.espn.com"}
LEAGUE_OF = {path: league for league, path in plan.PATHS.items()}
Usage = collections.namedtuple("Usage", "total used free")
ROOMY = Usage(100e9, 10e9, 90e9)


@pytest.fixture(autouse=True)
def disk(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """raw.db's disk (shutil.disk_usage: capture/raw_store.disk_room and learn/progress.py read it): plenty of room
    unless a test sets ``disk.usage``."""
    state = SimpleNamespace(usage=ROOMY)
    monkeypatch.setattr(raw_store.shutil, "disk_usage", lambda folder: state.usage)
    return state


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Every real door to the network is shut: a call fails the test (and is listed)."""
    calls: List[str] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        calls.append(repr(args[:3])[:200])
        raise AssertionError(f"a real network call: {args[:3]!r}")

    monkeypatch.setattr(requests.sessions.Session, "request", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", refuse)
    return calls


def at(day: datetime.date, hour: int, minute: int = 0, tz: datetime.tzinfo = ET) -> float:
    return datetime.datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz).timestamp()


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%MZ")


class Clock:
    """Fake epoch seconds; PoliteHttp's waits for a token (and HistHttp's pace) move it on."""

    def __init__(self, ts: float) -> None:
        self.now = float(ts)

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


def load(name: str) -> Any:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


STATUS = {"pre": {"id": "1", "name": "STATUS_SCHEDULED", "state": "pre", "completed": False,
                  "description": "Scheduled"},
          "live": {"id": "2", "name": "STATUS_IN_PROGRESS", "state": "in", "completed": False,
                   "description": "In Progress"},
          "final": {"id": "3", "name": "STATUS_FINAL", "state": "post", "completed": True, "description": "Final"},
          "postponed": {"id": "6", "name": "STATUS_POSTPONED", "state": "post", "completed": False,
                        "description": "Postponed"}}
BOARD_OF = {"basketball": "sb_nba.json", "football": "sb_nfl.json", "hockey": "sb_nhl.json", "baseball": "sb_mlb.json",
            "soccer": "sb_epl.json", "mma": "sb_ufc.json", "tennis": "sb_atp.json"}
SUMMARY_OF = {"basketball": "sum_nba.json", "football": "sum_nfl.json", "hockey": "sum_nhl.json",
              "baseball": "sum_mlb.json", "soccer": "sum_epl.json"}
PLAY_PAGES = 2
GAME_PK = 777079  # syn_feed_live.json's gamePk; the MLB schedule maps the recorded day's game to it
# The backfill's fake seasons: a short window (25-31 October) of every year, a few games a week, for the wave 1-2
# leagues (wave 3's walk has its own tests: tests/test_capture_backfill.py)
WINDOW = ((10, 25), (10, 31))
CODE = {league: 10 + n for n, league in enumerate(bf.WAVES_1_2)}


def sport_of(league: str) -> str:
    return plan.PATHS[league][0]


@dataclasses.dataclass
class Game:
    """A forward game: its final summary is a real one (tests/fixtures/capture, or learn:<file>)."""

    league: str
    event_id: str
    start: float
    summary: str
    runs: float = 3 * 3600.0
    postponed: bool = False
    version: int = 0

    def state(self, now: float) -> str:
        if self.postponed:
            return "postponed"
        return "pre" if now < self.start else ("live" if now < self.start + self.runs else "final")


@dataclasses.dataclass
class Card:
    event_id: str
    start: float
    runs: float = 6 * 3600.0

    def state(self, now: float) -> str:
        return "pre" if now < self.start else ("live" if now < self.start + self.runs else "final")


def is_card_day(day: datetime.date) -> bool:
    return day.weekday() == 5  # the backfill seasons' cards: Saturdays


def backfill_games(league: str, day: datetime.date) -> List[str]:
    """The fake past seasons' games of a day (event ids: digits only), inside the short WINDOW only."""
    if league not in CODE or not WINDOW[0] <= (day.month, day.day) <= WINDOW[1] or day == DAY:
        return []
    weekday = day.weekday()
    count = 0
    if league == NBA:
        count = 1 if weekday in (1, 4, 6) else 0
    elif league == NFL:
        count = 1 if weekday == 6 else 0
    elif league == UCL:
        count = 1 if weekday == 1 else 0
    elif league.startswith("soccer_"):
        count = 1 if weekday == 5 else 0
    elif league == NHL:
        count = 1 if weekday in (2, 5) else 0
    elif league == MLB:
        count = 1 if weekday in (0, 3) else 0
    elif league == UFC:
        count = 1 if is_card_day(day) else 0
    return [f"9{CODE[league]}{day:%Y%m%d}{n}" for n in range(count)]


def backfill_id_day(event_id: str) -> Optional[Tuple[str, datetime.date]]:
    """(league, day) of a fake past-season event id, or None."""
    if len(event_id) != 12 or not event_id.startswith("9") or not event_id.isdigit():
        return None
    code, day = int(event_id[1:3]), datetime.datetime.strptime(event_id[3:11], "%Y%m%d").date()
    league = next((name for name, number in CODE.items() if number == code), None)
    return (league, day) if league else None


class World:
    """The internet as the capture sees it, served from the fixtures. Four doors land here, each request recorded as
    (fake-clock time, door, host, url): "polite" (PoliteHttp._send: capture and backfill), "lean" (EspnClient._get and
    MlbStats._get: the lean cache's own loads), and "learn" (the learn backfill's HistHttp transport).

    - Forward games (``games``, ``cards``, ``tennis_day``) on their days, real summaries once final;
    - three past seasons of every backfill league (``backfill_games``) when ``seasons`` is on;
    - ``status_for`` (URL part -> status), ``unreachable`` (URL parts that reset), ``garbage`` (URL parts answered
      200 with a body that isn't JSON)."""

    def __init__(self, clock: Clock, *, seasons: bool = False, latency: float = 0.0) -> None:
        self.clock = clock
        self.seasons = seasons
        self.latency = latency  # how long each polite answer takes (fake seconds)
        self.sent: List[Tuple[float, str, str, str]] = []
        self.tagger: Optional[Callable[[], str]] = None  # who sent a polite request ("forward", "backfill")
        self.tags: List[Tuple[float, str, str]] = []  # (time, tag, url) of each polite request
        self.games: Dict[str, Game] = {}
        self.cards: Dict[str, Card] = {}
        self.tennis_day: Optional[datetime.date] = None
        self.status_for: Dict[str, int] = {}
        self.unreachable: Set[str] = set()
        self.garbage: Set[str] = set()
        self.boards = {sport: load(name) for sport, name in BOARD_OF.items()}
        self.summaries = {sport: (FIX / name).read_bytes() for sport, name in SUMMARY_OF.items()}
        plays = load("core_epl_plays.json")
        self.pages = {n: json.dumps({**plays, "pageIndex": n, "pageCount": PLAY_PAGES}).encode()
                      for n in range(1, PLAY_PAGES + 1)}
        fight = load("syn_fightcenter.json")
        for part in fight["cards"].values():  # every bout of the card is over (the prelims without a result)
            for bout in part["competitions"]:
                bout["status"]["type"] = dict(STATUS["final"])
        self.fightcenter = json.dumps(fight).encode()
        self.prop_calls = 0

    # -- the doors --

    def send(self, request: Any) -> HttpResponse:
        """PoliteHttp._send."""
        url = request.full_url
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        self.sent.append((self.clock(), "polite", host, url))
        if self.tagger is not None:
            self.tags.append((self.clock(), self.tagger(), url))
        self.clock.now += self.latency
        if any(word in host for word in FORBIDDEN):
            raise AssertionError(f"asked a forbidden host: {url}")
        if any(part in url for part in self.unreachable):
            raise ConnectionResetError("connection reset by peer")
        if any(part in url for part in self.garbage):
            return HttpResponse(200, b"<html>maintenance</html>", {"content-type": "text/html"})
        for part, status in self.status_for.items():
            if part in url:
                return HttpResponse(status, json.dumps({"code": status}).encode(), {})
        status, body = self.route(url)
        return HttpResponse(status, body, {"content-type": "application/json"})

    def lean_get(self, url: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        """EspnClient._get (the lean client's own loads, through requests in the app)."""
        full = url + ("?" + urllib.parse.urlencode(dict(params or {})) if params else "")
        self.sent.append((self.clock(), "lean", urllib.parse.urlsplit(full).hostname or "", full))
        status, body = self.route(full)
        if status != 200:
            raise RuntimeError(f"ESPN returned HTTP {status} for {url}")
        return json.loads(body)

    def schedule_get(self, url: str, params: Mapping[str, Any]) -> Any:
        """MlbStats._get: the day's StatsAPI schedule (the worker maps an ESPN game to its gamePk with it)."""
        self.sent.append((self.clock(), "lean", "statsapi.mlb.com", f"{url}?date={params.get('date')}"))
        day = datetime.date.fromisoformat(str(params.get("date")))
        games = []
        for game in self.games.values():
            if game.league == MLB and datetime.datetime.fromtimestamp(game.start, ET).date() == day:
                games.append({"gamePk": GAME_PK, "gameDate": iso(game.start),
                              "status": {"abstractGameState": "Final"},
                              "teams": {"home": {"team": {"name": "Tampa Bay Rays", "id": 139}},
                                        "away": {"team": {"name": "Milwaukee Brewers", "id": 158}}}})
        return {"dates": [{"games": games}]}

    def learn_transport(self, url: str, params: Mapping[str, Any], timeout: float) -> Tuple[int, bytes]:
        """The learn backfill's HistHttp transport: its NBA week (tests/fixtures/learn)."""
        full = url + ("?" + urllib.parse.urlencode(dict(params or {})) if params else "")
        host = urllib.parse.urlsplit(url).hostname or ""
        self.sent.append((self.clock(), "learn", host, full))
        if any(word in host for word in FORBIDDEN):
            raise AssertionError(f"asked a forbidden host: {url}")
        if host == "sports.core.api.espn.com" and url.endswith("/events"):
            if "/nba/" in url:
                return 200, (LEARN_FIX / "espn_nba_events.json").read_bytes()
            return 200, json.dumps({"count": 0, "pageIndex": 1, "pageCount": 1, "items": []}).encode()
        if url.endswith("/odds"):
            return 200, (LEARN_FIX / "espn_nba_odds.json").read_bytes()
        if url.endswith("/summary"):
            return 200, (LEARN_FIX / "espn_nba_summary.json").read_bytes()
        return 404, b"{}"

    # -- what was asked --

    def asked(self, door: Optional[str] = None, part: str = "") -> List[Tuple[float, str, str, str]]:
        return [item for item in self.sent if (door is None or item[1] == door) and part in item[3]]

    # -- answers --

    def route(self, url: str) -> Tuple[int, bytes]:
        parts = urllib.parse.urlsplit(url)
        query = dict(urllib.parse.parse_qsl(parts.query))
        host, segs = parts.hostname or "", parts.path.split("/")
        if host == "site.api.espn.com" and len(segs) == 8 and segs[1:5] == ["apis", "site", "v2", "sports"]:
            league = LEAGUE_OF.get((segs[5], segs[6]))
            if league and segs[7] == "summary":
                return self.summary(league, query.get("event", ""))
            if league and segs[7] == "scoreboard" and query.get("dates"):
                return 200, self.board(league, query["dates"])
        if host == "sports.core.api.espn.com" and len(segs) >= 11 and segs[6] == "events":
            rest = "/".join(segs[10:])
            if rest == "plays":
                page = int(query.get("page", "1"))
                return (200, self.pages[page]) if page in self.pages else (404, b'{"code":404}')
            if rest == "status":
                return 200, (FIX / "core_ufc_status.json").read_bytes()
            if rest.startswith("competitors/") and rest.endswith("/statistics"):
                return 200, (FIX / "core_ufc_stats.json").read_bytes()
            if rest == "odds/100/propBets":
                self.prop_calls += 1
                data = json.loads((ESPN_FIX / "props_nba_401909866.json").read_text())
                data["items"][0]["snapshot"] = self.prop_calls  # the lines move between snapshots
                return 200, json.dumps(data).encode()
        if host == "site.web.api.espn.com" and "/fightcenter/" in parts.path:
            card = segs[-1]
            known = card in self.cards or (backfill_id_day(card) or ("",))[0] == UFC
            return (200, self.fightcenter) if known else (404, b'{"code":404}')
        if host == "statsapi.mlb.com" and parts.path == f"/api/v1.1/game/{GAME_PK}/feed/live":
            return 200, (FIX / "syn_feed_live.json").read_bytes()
        if host == "github.com" and parts.path.endswith(".csv.gz"):
            return 200, (FIX / "syn_play_by_play_2024.csv.gz").read_bytes()
        if host == "github.com" and parts.path.endswith("/officials.csv"):
            return 200, (FIX / "syn_officials.csv").read_bytes()
        if host == "raw.githubusercontent.com" and parts.path.endswith(".csv"):
            name = parts.path.rsplit("/", 1)[1]
            synthetic = FIX / f"syn_{name}"
            return 200, synthetic.read_bytes() if synthetic.exists() else f"file\n{name}\n".encode()
        if host == "football-data.co.uk":
            if parts.path == "/new/USA.csv":
                return 200, (LEARN_FIX / "fd_new_USA.csv").read_bytes()
            return 200, (FIX / "syn_E0_2425.csv").read_bytes()
        if host == "api.elections.kalshi.com" and parts.path == "/trade-api/v2/events":
            data = json.loads((MARKETS_FIX / "kalshi_epl_game.json").read_text())
            data["series_ticker"] = query.get("series_ticker")
            data["as_of"] = int(self.clock())  # the prices move
            return 200, json.dumps(data).encode()
        return 404, b'{"code":404,"message":"not found"}'

    def summary(self, league: str, event_id: str) -> Tuple[int, bytes]:
        game = self.games.get(event_id)
        if game is not None:
            raw = (LEARN_FIX / game.summary[6:]).read_bytes() if game.summary.startswith("learn:") else \
                (FIX / game.summary).read_bytes()
            state = game.state(self.clock())
            if state == "final" and not game.version:
                return 200, raw
            data = json.loads(raw)
            data["header"]["competitions"][0]["status"]["type"] = dict(STATUS[state])
            if state == "pre":
                data["asOf"] = int(self.clock() // 3600)  # the injury report and line-ups change before the start
            if game.version:
                data["corrected"] = game.version  # ESPN corrected the record: a new body
            return 200, json.dumps(data).encode()
        found = backfill_id_day(event_id) if self.seasons else None
        if found is not None and found[0] == league and sport_of(league) in self.summaries:
            return 200, self.summaries[sport_of(league)]
        return 404, b'{"code":404}'

    def board(self, league: str, dates: str) -> bytes:
        day = datetime.datetime.strptime(dates, "%Y%m%d").date()
        sport = sport_of(league)
        data = copy.deepcopy(self.boards[sport])
        proto = data["events"][0]
        now = self.clock()
        events: List[Dict[str, Any]] = []

        def team_event(event_id: str, start: float, state: str) -> Dict[str, Any]:
            event = copy.deepcopy(proto)
            event.update({"id": event_id, "uid": f"s:x~e:{event_id}", "date": iso(start)})
            competition = event["competitions"][0]
            competition.update({"id": event_id, "date": iso(start)})
            competition["status"]["type"] = dict(STATUS[state])
            return event

        def card_event(event_id: str, start: float, state: str) -> Dict[str, Any]:
            event = copy.deepcopy(proto)
            event.update({"id": event_id, "date": iso(start)})
            event["status"]["type"] = dict(STATUS[state])
            for competition in event["competitions"]:
                competition["date"] = iso(start)
                competition["status"]["type"] = dict(STATUS[state])
            return event

        if sport == "tennis":
            if day == self.tennis_day:
                event = proto
                for grouping in event["groupings"]:
                    for n, competition in enumerate(grouping["competitions"]):
                        competition["date"] = iso(at(day, 6 + n))
                data["events"] = [event]
            else:
                data["events"] = []
            return json.dumps(data).encode()
        if sport == "mma":
            for card in self.cards.values():
                if datetime.datetime.fromtimestamp(card.start, ET).date() == day:
                    events.append(card_event(card.event_id, card.start, card.state(now)))
        else:
            for game in self.games.values():
                if game.league == league and datetime.datetime.fromtimestamp(game.start, ET).date() == day:
                    events.append(team_event(game.event_id, game.start, game.state(now)))
        if self.seasons:
            for event_id in backfill_games(league, day):
                start = at(day, 19)
                events.append(card_event(event_id, start, "final") if sport == "mma" else
                              team_event(event_id, start, "final"))
        data["events"] = events
        season = data.get("leagues", [{}])[0].get("season")
        if isinstance(season, dict):
            season["year"] = day.year
        if sport == "mma":
            data["leagues"][0]["calendar"] = self.ufc_calendar(day.year)
        return json.dumps(data).encode()

    def ufc_calendar(self, year: int) -> List[Dict[str, Any]]:
        """A UFC season's cards as ESPN lists them (each with its event link)."""
        entries = []
        day = datetime.date(year, 1, 1)
        while day.year == year:
            ids = backfill_games(UFC, day) if self.seasons else []
            ids += [card.event_id for card in self.cards.values()
                    if datetime.datetime.fromtimestamp(card.start, ET).date() == day]
            for card in ids:
                entries.append({"label": f"UFC {day}", "startDate": f"{day + datetime.timedelta(days=1)}T00:00Z",
                                "event": {"$ref": f"http://sports.core.api.espn.pvt/v2/sports/mma/leagues/ufc/"
                                                  f"events/{card}?lang=en&region=us"}})
            day += datetime.timedelta(days=1)
        return entries


def recorded_day(world: World) -> Dict[str, str]:
    """One final in every sport on DAY (the real fixtures' games): {sport: game key}."""
    for league, event_id, (hour, minute), summary in (
            (NBA, "401909278", (19, 30), "sum_nba.json"), (NFL, "401873017", (13, 0), "sum_nfl.json"),
            (NHL, "401892612", (19, 0), "sum_nhl.json"), (MLB, "401908046", (16, 0), "sum_mlb.json"),
            (EPL, "401878765", (10, 0), "sum_epl.json")):
        world.games[event_id] = Game(league, event_id, at(DAY, hour, minute), summary)
    world.cards["600061267"] = Card("600061267", at(DAY, 14))
    world.tennis_day = DAY
    return {"basketball": "espn:401909278", "football": "espn:401873017", "hockey": "espn:401892612",
            "baseball": "espn:401908046", "soccer": "espn:401878765", "mma": "espn:600061267"}


TENNIS_SINGLES = ("187149", "187148", "187154")  # sb_atp.json's men's singles, all final
UFC_BOUTS_WITHOUT_RESULT = ("401911640", "401911645")  # syn_fightcenter.json's prelims (no method given)


class Rig:
    """The dashboard's capture as build_capturer makes it, on a fake clock: one raw.db, one shared cache (over a
    betting.db store, so its rows are in the shared table), one PoliteHttp (its _send is the world's), one EspnClient
    and the Capturer over them. ``config`` takes the CAPTURE_* settings (the backfill is built by the worker from
    them, with tz UTC as the server's)."""

    def __init__(self, tmp: Path, world: World, monkeypatch: pytest.MonkeyPatch, *, name: str = "raw",
                 per_day: int = 3000, options: Optional[plan.Options] = None, kalshi: Sequence[str] = (),
                 config: Any = None, lease: Optional[Callable[[], bool]] = None, store: Any = None,
                 cache: Optional[TTLCache] = None) -> None:
        self.world, self.clock = world, world.clock
        monkeypatch.setattr(EspnClient, "_get", lambda client, url, params=None: world.lean_get(url, params))
        monkeypatch.setattr(MlbStats, "_get", lambda client, url, params: world.schedule_get(url, params))
        self.store = store if store is not None else Store(str(tmp / f"{name}-betting.db"))
        self.cache = cache if cache is not None else TTLCache(self.store, name=f"{name}-cache", clock=self.clock)
        self.raw = RawStore(str(tmp / f"{name}.db"), clock=self.clock)
        self.http = PoliteHttp(clock=self.clock, sleep=self.clock.sleep)
        self.http._send = world.send  # type: ignore[method-assign]
        self.espn = EspnClient(self.cache, clock=self.clock)
        self.capturer = Capturer(self.store, self.raw, self.espn, self.http, self.clock,
                                 config=config if config is not None else SimpleNamespace(capture_backfill=False),
                                 options=options, per_day=per_day, kalshi_series=kalshi,
                                 mlb=MlbStats(self.cache, clock=self.clock), tz=UTC,
                                 lease=lease or (lambda: True))

    def drain(self, budget: int = 300, rounds: int = 40) -> Dict[str, int]:
        """run_due until nothing is left to fetch now (or ``rounds``), the clock moving on to each due time."""
        total = collections.Counter()
        for _ in range(rounds):
            report = self.capturer.run_due(budget)
            total.update({k: v for k, v in report.items() if isinstance(v, int)})
            due = self.raw.next_due()
            if due is None:
                break
            if due > self.clock():
                self.clock.now = due + 1.0
        return dict(total)


def payload_rows(raw: RawStore) -> List[Tuple[Any, ...]]:
    """Every raw row's identity and labels, in a stable order (not the ids or fetch times)."""
    db = sqlite3.connect(raw.path)
    try:
        return sorted(db.execute("SELECT game_key, league, source, endpoint, params, state, horizon, http_status, "
                                 "sha256, bytes_raw, codec FROM raw_payloads").fetchall())
    finally:
        db.close()


def kinds_of(raw: RawStore, game_key: str, state: Optional[str] = "final") -> Set[Tuple[str, str]]:
    return {(p.source, p.endpoint) for p in raw.bundle(game_key) if state is None or p.state == state}


# Section 3.3's politeness table (requests a second, burst), written out here so a changed RATES can't pass
PLAN_RATES = {"site.api.espn.com": (1.0, 3), "sports.core.api.espn.com": (1.0, 3), "site.web.api.espn.com": (0.5, 1),
              "statsapi.mlb.com": (0.5, 1), GITHUB: (0.2, 1)}
PLAN_DEFAULT_RATE = (0.5, 1)


def test_the_politeness_table_is_section_3_3s() -> None:
    assert RATES == PLAN_RATES and DEFAULT_RATE == PLAN_DEFAULT_RATE


def bucket_overruns(sent: Iterable[Tuple[float, str, str, str]]) -> List[str]:
    """Replays the requests PoliteHttp sent against section 3.3's token buckets (a host's rate and burst; the GitHub
    hosts share one bucket; any other host 0.5/s): every request that came before its token is listed."""
    tokens: Dict[str, Tuple[float, float]] = {}  # bucket -> (tokens, at)
    late: List[str] = []
    for ts, door, host, url in sorted(sent):
        if door != "polite":
            continue
        name = GITHUB if is_github(host) else host
        rate, burst = PLAN_RATES.get(name, PLAN_DEFAULT_RATE)
        held, then = tokens.get(name, (float(burst), ts))
        held = min(float(burst), held + max(0.0, ts - then) * rate) - 1.0
        if held < -1e-6:
            late.append(f"{name} at {ts:.3f}: {url}")
        tokens[name] = (held, ts)
    return late


def lean_copy(cache: TTLCache, store: Store, key: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """A lean cache entry as the grader reads it: (value as JSON text, kind, fresh for, usable for) in memory, and the
    shared table's (blob, fresh and usable offsets from its fetch time)."""
    from telegram_betting_bot_simple.data.cache import key_text
    entry = cache._entries[key]
    row = store.cache_get(key_text(key))
    assert row is not None, f"{key} is in the shared table too"
    return (json.dumps(entry[2], sort_keys=True), entry[3], entry[4], entry[5], bytes(row[3]), row[1] - row[0],
            row[2] - row[0])


def expected_final(league: str, options: Optional[plan.Options] = None) -> Set[Tuple[str, str]]:
    """(source, endpoint) of plan.KINDS' final kinds of a league."""
    return {(plan.SOURCE_OF[kind], plan.endpoint_of(kind)) for kind in plan.kinds_for(league, "final", options)}


# --- forward capture: one recorded day, every sport ------------------------------------------------------------------

def test_one_recorded_day_with_a_final_in_every_sport_stores_the_plans_kinds(tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock(at(DAY + datetime.timedelta(days=1), 5, 30))  # the next ESPN day, past the sweep's 05:00
    world = World(clock)
    keys = recorded_day(world)
    rig = Rig(tmp_path, world, monkeypatch)
    capturer, raw = rig.capturer, rig.raw
    assert capturer.sweep_day() == DAY
    made = capturer.sweep(DAY)
    boards = world.asked("polite", "/scoreboard?")
    assert len(boards) == len(plan.PATHS) == 58 and len({url for *_r, url in boards}) == 58
    assert all("dates=20261030" in url for *_r, url in boards)  # one-day boards: ESPN refuses ranges
    for league in plan.PATHS:  # every board kept raw, final once its day is over
        payload = raw.latest(scoreboard_key(league, "20261030"), "espn_site", "scoreboard", {"dates": "20261030"})
        assert payload is not None and (payload.state, payload.horizon) == ("final", "final"), league
    assert made == 5 + 1 + len(TENNIS_SINGLES)  # the five team games, the card, the tennis singles
    report = rig.drain()
    assert report["failed"] == 0 and report["fetched"] == 15  # 5 summaries, the feed, 2 plays pages, the card's 7
    jobs = raw.jobs()
    assert jobs and all(job.state == "done" for job in jobs), [(j.game_key, j.state, j.kinds_done) for j in jobs]

    # The plan's final kinds for every sport, each under its section 3.3 key
    for sport, league in (("basketball", NBA), ("football", NFL), ("hockey", NHL), ("baseball", MLB),
                          ("soccer", EPL)):
        assert kinds_of(raw, keys[sport]) == expected_final(league), sport
    feed = raw.latest(keys["baseball"], "mlb_statsapi", "feed_live")
    assert dict(feed.params) == {"game_pk": GAME_PK} and isinstance(feed.params["game_pk"], int)
    pages = raw.bundle(keys["soccer"]).pages("espn_core", "core_plays")
    assert [dict(p.params) for p in pages] == [{"limit": 300, "page": n} for n in range(1, PLAY_PAGES + 1)]
    # MMA: the card's fightcenter under espn:<cardEventId> (league mma), each bout with no result its core status
    # and both fighters' statistics under espn-bout:<competitionId>, the day's board raw (each bout's details kept)
    card = raw.bundle(keys["mma"])
    assert card.league == UFC and kinds_of(raw, keys["mma"]) == {("espn_fightcenter", "fightcenter")}
    stored = set(kinds_of(raw, keys["mma"]))
    for bout in UFC_BOUTS_WITHOUT_RESULT:
        bundle = raw.bundle(f"espn-bout:{bout}")
        assert {(p.source, p.endpoint) for p in bundle} == {("espn_core", "core_status"),
                                                           ("espn_core", "core_competitor_stats")}
        assert len(bundle.all("espn_core", "core_competitor_stats")) == 2
        stored |= {(p.source, p.endpoint) for p in bundle}
    ufc_board = raw.latest(scoreboard_key(UFC, "20261030"), "espn_site", "scoreboard")
    stored.add((ufc_board.source, ufc_board.endpoint))
    assert stored == expected_final(UFC)
    assert raw.job(keys["mma"]).kinds_done == ["final:core_competitor_stats", "final:core_status",
                                               "final:fightcenter", "final:scoreboard"]
    # Tennis: the day's ATP board raw is the matches' record; their jobs are done from it
    for match in TENNIS_SINGLES:
        assert raw.job(f"espn-match:{match}").kinds_done == ["final:scoreboard"]
    atp_board = raw.latest(scoreboard_key(ATP, "20261030"), "espn_site", "scoreboard")
    assert {(atp_board.source, atp_board.endpoint)} == expected_final(ATP)
    assert atp_board.json()["events"][0]["groupings"]
    # Raw means raw: what LEAN_DROP takes out of the cache's boards is in raw.db (MMA details, tennis notes)
    lean_board = rig.espn.scoreboard(UFC, DAY)
    assert "details" in json.dumps(ufc_board.json()) and "details" not in json.dumps(lean_board)
    # Every request went through the one polite door, to ESPN and StatsAPI only (the gamePk came from the cached
    # StatsAPI schedule, data/mlb_stats.py)
    assert {host for _t, door, host, _u in world.sent if door == "polite"} == ESPN_HOSTS | {"statsapi.mlb.com"}
    assert [url for _t, door, _h, url in world.sent if door == "lean"] == [
        "https://statsapi.mlb.com/api/v1/schedule?date=2026-10-30"]
    assert bucket_overruns(world.sent) == []


def test_the_slates_games_get_every_pre_start_kind_at_its_horizons(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """The slate's hook (data/slate.py, the real SlateBuilder) hands its games to schedule_from_slate; each game then
    gets section 3.3's pre-start kinds at open / t24 / t90 / close, and its finals after the whistle."""
    from telegram_betting_bot_simple.data.slate import SlateBuilder
    day = DAY + datetime.timedelta(days=5)  # Wednesday 4 November
    clock = Clock(at(day, 15) - 30 * 3600)  # first sight 30 hours before the earliest start
    world = World(clock)
    games = {NBA: Game(NBA, "401909278", at(day, 19, 30), "sum_nba.json"),
             NFL: Game(NFL, "401873017", at(day, 20, 15), "sum_nfl.json"),
             NHL: Game(NHL, "401892612", at(day, 19), "sum_nhl.json"),
             MLB: Game(MLB, "401908046", at(day, 19, 5), "sum_mlb.json"),
             EPL: Game(EPL, "401878765", at(day, 15), "sum_epl.json")}
    world.games.update({game.event_id: game for game in games.values()})
    world.cards["600061267"] = Card("600061267", at(day, 18))
    rig = Rig(tmp_path, world, monkeypatch, kalshi=("KXEPLCORNERS",))
    slate = SlateBuilder(None, None, None, now=clock, capturer=rig.capturer)
    ufc_bouts = [c["id"] for c in load("sb_ufc.json")["events"][0]["competitions"]]

    def slate_game(league: str, espn_id: str, start: float, status: str = "pre") -> Dict[str, Any]:
        return {"id": f"{league}:o{espn_id}", "league": league, "espn_id": espn_id, "odds_id": f"o{espn_id}",
                "start": iso(start), "status": status, "home": {"name": "Home"}, "away": {"name": "Away"},
                "venue": None, "_ts": start, "_time_ok": True, "_ids": ("1", "2"), "_neutral": False,
                "_books": [{"key": "pinnacle"}], "badges": None}

    rows = [slate_game(g.league, g.event_id, g.start) for g in games.values()]
    rows += [slate_game(UFC, bout, at(day, 18)) for bout in ufc_bouts]
    slate._schedule_capture(rows, clock())
    slate.wait_capture()
    assert {job.game_key for job in rig.raw.jobs()} == {f"espn:{g.event_id}" for g in games.values()} | \
        {"espn:600061267"}
    rig.drain(rounds=120)
    expect = {NBA: {"pre:t90:summary"} | {f"pre:{h}:propbets" for h in ("open", "t24", "t90", "close")},
              NFL: {f"pre:{h}:propbets" for h in ("open", "t24", "t90", "close")},
              EPL: {"pre:t24:summary", "pre:close:summary"} | {f"pre:{h}:kalshi" for h in ("open", "t24", "t90",
                                                                                             "close")}}
    expect[NHL] = expect[MLB] = expect[NFL]
    for league, game in games.items():
        job = rig.raw.job(f"espn:{game.event_id}")
        assert job.state == "done" and expect[league] <= set(job.kinds_done), (league, job.kinds_done)
        assert {name for name in job.kinds_done if name.startswith("pre:")} == expect[league]
    assert "pre:t24:fightcenter" in rig.raw.job("espn:600061267").kinds_done
    # Each snapshot is kept with its horizon, before the start (inputs, never outcomes)
    assert rig.raw.latest("espn:401909278", "espn_site", "summary").state == "final"
    with sqlite3.connect(rig.raw.path) as db:
        props = db.execute("SELECT horizon, state, fetched_at FROM raw_payloads WHERE game_key = ? AND source = "
                           "'espn_propbets' ORDER BY fetched_at", ("espn:401909278",)).fetchall()
        kalshi = db.execute("SELECT horizon, state FROM raw_payloads WHERE game_key = 'static:kalshi:KXEPLCORNERS' "
                            "ORDER BY fetched_at").fetchall()
        pre_summaries = db.execute("SELECT game_key, horizon, fetched_at FROM raw_payloads WHERE source = "
                                   "'espn_site' AND endpoint = 'summary' AND state = 'pre' ORDER BY fetched_at"
                                   ).fetchall()
    assert [h for h, _s, _t in props] == ["open", "t24", "t90", "close"] and all(s == "pre" for _h, s, _t in props)
    assert all(t < games[NBA].start for _h, _s, t in props)
    assert [h for h, _s in kalshi] == ["open", "t24", "t90", "close"]
    assert [(k, h) for k, h, _t in pre_summaries] == [("espn:401878765", "t24"), ("espn:401878765", "close"),
                                                       ("espn:401909278", "t90")]
    asked_prices = len(world.asked("polite", "propBets")) + len(world.asked("polite", "kalshi"))
    assert sum(row["used"] for row in rig.raw.counts(["prices"])) == asked_prices  # propBets and Kalshi
    assert bucket_overruns(world.sent) == []


def test_a_rerun_stores_nothing_new_and_a_corrected_body_is_a_new_version(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock(at(DAY + datetime.timedelta(days=1), 5, 30))
    world = World(clock)
    keys = recorded_day(world)
    rig = Rig(tmp_path, world, monkeypatch)
    rig.capturer.sweep(DAY)
    rig.drain()
    first = payload_rows(rig.raw)
    asked = len(world.sent)
    # The same work again from scratch (the jobs and the sweep's marks gone; raw.db's payloads kept): every
    # answer is the same, so nothing new is stored
    with sqlite3.connect(rig.raw.path) as db:
        db.execute("DELETE FROM capture_jobs")
        db.execute("DELETE FROM capture_state")
    again = Rig(tmp_path, world, monkeypatch, store=rig.store, cache=rig.cache)
    assert again.raw.path == rig.raw.path
    assert again.capturer.sweep(DAY) == 5 + 1 + len(TENNIS_SINGLES)
    report = again.drain()
    assert len(world.sent) > asked and report["fetched"] > 0
    assert report["stored"] == 0 and payload_rows(again.raw) == first
    # ESPN corrects one game's record: a new sha256, a new row; the old one is kept and latest() is the new one
    world.games["401909278"].version = 1
    job = again.raw.job(keys["basketball"])
    job.state, job.due_at, job.kinds_done, job.attempts = "todo", clock(), [], 0
    again.raw.save_job(job, now=clock())
    report = again.drain()
    rows = payload_rows(again.raw)
    assert report["stored"] == 1 and len(rows) == len(first) + 1
    summaries = [row for row in rows if row[0] == keys["basketball"] and row[3] == "summary"]
    assert len(summaries) == 2 and len({row[8] for row in summaries}) == 2
    assert again.raw.latest(keys["basketball"], "espn_site", "summary").json()["corrected"] == 1


def test_a_restart_mid_run_resumes_where_it_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stopped every few requests (a new process each time, on the same raw.db), the run ends with the same raw rows
    as one uninterrupted run, and no request is made twice."""
    def run(name: str, step: Optional[int]) -> Tuple[List[Tuple[Any, ...]], List[str]]:
        clock = Clock(at(DAY + datetime.timedelta(days=1), 5, 30))
        world = World(clock)
        recorded_day(world)
        made = Rig(tmp_path, world, monkeypatch, name=name)
        made.capturer.sweep(DAY)
        if step is None:
            made.drain()
        else:
            for _ in range(60):
                fresh = Rig(tmp_path, world, monkeypatch, name=name, store=made.store, cache=made.cache)
                report = fresh.capturer.run_due(step)  # a budget stop part-way through a job: then a "restart"
                if report["fetched"] == 0 and fresh.raw.next_due() is None:
                    break
                due = fresh.raw.next_due()
                if due is not None and due > clock():
                    clock.now = due + 1.0
        return payload_rows(Rig(tmp_path, world, monkeypatch, name=name, store=made.store, cache=made.cache).raw), \
            [url for _t, door, _h, url in world.sent if door == "polite"]

    whole, whole_urls = run("whole", None)
    for step in (1, 2, 3):
        pieces, urls = run(f"step{step}", step)
        assert pieces == whole and sorted(urls) == sorted(whole_urls), step
        assert len(urls) == len(set(urls)), "nothing fetched twice"


# --- the grader and the lean cache ------------------------------------------------------------------------------------

class NoOdds:
    """The grader's odds door: nothing (the capture never needs it, and nothing here may call The Odds API)."""

    quota = {"remaining": 480, "used": 20}

    def sports(self) -> List[Any]:
        return []

    def league_games(self, league: str, refresh: bool = False) -> List[Any]:
        return []

    def odds_loaded_at(self, league: str) -> None:
        return None


def teams_of(summary: str) -> Tuple[str, str]:
    competitors = load(summary)["header"]["competitions"][0]["competitors"]
    side = {c["homeAway"]: c["team"]["displayName"] for c in competitors}
    return side["home"], side["away"]


def test_the_graders_output_and_the_lean_cache_are_byte_identical_and_its_read_is_a_hit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The grader (DashboardService.grade_due) grades every sport's game to the same bytes whether the capture
    fetched the summaries first (summary_raw seeding the lean cache from the same answer) or the grader loaded them
    itself; the lean cache holds the same entries, in memory and in the shared table; after a capture the grader asks
    ESPN nothing."""
    from telegram_betting_bot_simple.dashboard.service import DashboardService
    games = [(NBA, "401909278", "sum_nba.json"), (NFL, "401873017", "sum_nfl.json"),
             (NHL, "401892612", "sum_nhl.json"), (MLB, "401908046", "sum_mlb.json"),
             (EPL, "401878765", "sum_epl.json")]

    def graded(captured: bool) -> Tuple[str, Dict[Any, Any], List[str], List[str]]:
        clock = Clock(at(DAY + datetime.timedelta(days=1), 5, 30))
        world = World(clock)
        recorded_day(world)
        rig = Rig(tmp_path, world, monkeypatch, name=f"grade-{captured}")
        if captured:
            rig.capturer.sweep(DAY)
            rig.drain()
        before = len(world.sent)
        for n, (league, event_id, summary) in enumerate(games):
            home, away = teams_of(summary)
            rig.store.save_prediction({"player": "web:abc12345", "name": "Sam", "league": league,
                                       "event_id": f"o{n}", "espn_id": event_id, "home": home, "away": away,
                                       "start": iso(world.games[event_id].start),
                                       "call": {"winner": "home", "confidence": 0.6, "home_score": 3,
                                                "away_score": 1},
                                       "ai": {"home_win": 0.55, "margin": 1.5, "total": 4.5}})
        service = DashboardService(NoOdds(), rig.espn, ET, store=rig.store)
        assert service.grade_due() == len(games)
        with rig.store._connect() as db:
            rows = db.execute("SELECT espn_id, status, result, user_points, ai_points FROM predictions "
                              "ORDER BY espn_id").fetchall()
        copies = {league: lean_copy(rig.cache, rig.store, ("espn", "summary", league, event_id))
                  for league, event_id, _s in games}
        asked = world.sent[before:]
        return (json.dumps(rows, sort_keys=True), copies, [u for _t, d, _h, u in asked if d == "lean"],
                [u for _t, d, _h, u in asked if d == "polite"])

    plain_rows, plain_cache, plain_lean, plain_polite = graded(False)
    rows, cache, lean, polite = graded(True)
    assert rows == plain_rows and all(row[1] == "graded" for row in json.loads(rows))
    assert cache == plain_cache  # byte for byte: value, kind, life, and the shared table's blob and life
    assert len([u for u in plain_lean if "/summary?" in u]) == len(games) and plain_polite == []
    assert lean == [] and polite == []  # after the capture the grader's reads were all hits


# --- the whole system over three nights -------------------------------------------------------------------------------

START = at(datetime.date(2026, 10, 31), 5, 30)  # Saturday 31 October, 05:30 US Eastern (09:30 UTC): the sweep is due
END = datetime.datetime(2026, 11, 3, 6, 30, tzinfo=UTC).timestamp()
LEARN_AT = datetime.datetime(2026, 11, 3, 3, 0, tzinfo=UTC).timestamp()  # the learn backfill's night slot
PER_NIGHT = 200
PRE_ID, PRE_START = "401909281", datetime.datetime(2026, 11, 1, 2, 32, tzinfo=UTC).timestamp()  # its t90 is 01:02 UTC
# More pre games whose t90 snapshots fall due all through the first night's backfill (some in the middle of a run of
# requests, where only the check before each request can stop it): {event id: start}
PRE_GAMES = {PRE_ID: PRE_START, **{
    f"4019093{n:02d}": datetime.datetime(2026, 11, 1, 1, 0, tzinfo=UTC).timestamp() + offset + 5400.0
    for n, offset in enumerate((40.5, 65.5, 91.5, 151.5, 172.5, 193.5))}}
LEARN_IDS = ("401810326", "401810327", "401810328", "401810329")  # tests/fixtures/learn/espn_nba_events.json


def unit_of(url: str, world: World) -> Optional[Tuple[str, int]]:
    """The backfill unit (league, season) a request belongs to (from its URL), or None."""
    parts = urllib.parse.urlsplit(url)
    query = dict(urllib.parse.parse_qsl(parts.query))
    segs = parts.path.split("/")
    if parts.hostname == "github.com" and "play_by_play_" in parts.path:
        return NFL, int(parts.path.rsplit("_", 1)[1].split(".")[0])
    if parts.hostname == "site.api.espn.com" and len(segs) == 8:
        league = LEAGUE_OF.get((segs[5], segs[6]))
        if segs[7] == "scoreboard" and league:
            return league, int(query["dates"][:4])
        event = query.get("event", "")
    elif parts.hostname == "sports.core.api.espn.com" and len(segs) > 7:
        event = segs[7]
    elif parts.hostname == "site.web.api.espn.com":
        event = segs[-1]
    else:
        return None
    found = backfill_id_day(event)
    if found:
        return found[0], found[1].year
    game = world.games.get(event) or world.cards.get(event)
    if game is None:
        return None
    league = getattr(game, "league", UFC)
    return league, datetime.datetime.fromtimestamp(game.start, ET).year


def learn_backfill(world: World, store: Store, cache: TTLCache, raw: Any) -> Tuple[Dict[str, Any], List[str]]:
    """The nightly learn backfill (learn/history.Backfill) over the learn NBA week, with ``raw`` as learn/loop.py
    hands it; returns its report and the requests it made."""
    from telegram_betting_bot_simple.learn.history import Backfill
    from telegram_betting_bot_simple.learn.sources.http import HistHttp
    before = len(world.sent)
    http = HistHttp(clock=world.clock, sleep=world.clock.sleep, transport=world.learn_transport)
    options = {} if raw is None else {"raw": raw}
    report = Backfill(store, cache, world.clock, http=http, tz=UTC, per_night=500, seasons=1, sources=["espn"],
                      leagues=[NBA], **options).run_night()
    return report, [url for _t, door, _h, url in world.sent[before:] if door == "learn"]


def hist_dump(store: Store) -> Dict[str, List[Tuple[Any, ...]]]:
    from telegram_betting_bot_simple.learn import history_store as hs
    db = store._connect()
    hs.init_history_schema(db)
    try:
        games = db.execute("SELECT league, game_key, espn_id, start, home, away, home_score, away_score, status, "
                           "season, source, home_key, away_key, info FROM hist_games").fetchall()  # not fetched_at
        return {"hist_games": sorted(games, key=repr),
                **{table: sorted(db.execute(f"SELECT * FROM {table}").fetchall(), key=repr)
                   for table in ("hist_lines", "hist_stats")}}
    finally:
        db.close()


@pytest.fixture(scope="module")
def night(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Three days of the dashboard's capture thread on a fake clock, from Saturday 31 October 09:30 UTC (the server's
    time zone is UTC): the slate hands over a pre game, a tick a minute (every half hour outside the nights) runs
    yesterday's sweep, the due jobs, the static files and, from 01:00 to 06:00, the backfill of 3 past seasons of
    every wave 1-2 league (short 25-31 October seasons) at PER_NIGHT requests a night. Each night starts in a new
    process (every object rebuilt over the same raw.db, betting.db and cache table). On the third night the learn
    backfill runs with the raw-first hook as learn/loop.py wires it, and once more without it on a fresh store."""
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("night")
    try:
        run = _three_nights(tmp, mp)
    finally:
        mp.undo()  # nothing the run patched outlives it: the tests only read what it left
    yield run


def _three_nights(tmp: Path, mp: pytest.MonkeyPatch) -> SimpleNamespace:
    from telegram_betting_bot_simple.capture.raw_store import resolve_raw_db_path
    refused: List[str] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        refused.append(repr(args[:3])[:200])
        raise AssertionError("a real network call")

    mp.setattr(raw_store.shutil, "disk_usage", lambda folder: ROOMY)
    mp.setattr(requests.sessions.Session, "request", refuse)
    mp.setattr(urllib.request, "urlopen", refuse)
    mp.setattr(urllib.request.OpenerDirector, "open", refuse)
    mp.setattr(bf, "_projected", True)
    mp.setattr(bf, "ORDER", bf.WAVES_1_2)  # the backfill of the wave 1-2 leagues, as the run's world has them
    for league in bf.ORDER:
        mp.setitem(bf.WINDOWS, league, WINDOW)
    mp.setenv("CAPTURE_BACKFILL_PER_NIGHT", str(PER_NIGHT))
    mp.delenv("RAW_DB_PATH", raising=False)
    mp.delenv("CAPTURE_MIN_FREE_GB", raising=False)
    clock = Clock(START)
    world = World(clock, seasons=True, latency=1.0)
    keys = recorded_day(world)
    world.games[LEARN_IDS[0]] = Game(NBA, LEARN_IDS[0], at(DAY, 22), "learn:espn_nba_summary.json")  # forward
    world.games[LEARN_IDS[1]] = Game(NBA, LEARN_IDS[1], at(DAY - datetime.timedelta(days=3), 19),
                                     "learn:espn_nba_summary.json")  # only the backfill sees it (27 October)
    for event_id, start in PRE_GAMES.items():
        world.games[event_id] = Game(NBA, event_id, start, "sum_nba3.json")
    config = SimpleNamespace(capture_enabled=True, raw_db_path=None, capture_backfill=True,
                             capture_backfill_seasons=3, capture_backfill_per_night=PER_NIGHT)
    store = Store(str(tmp / "betting.db"))
    cache = TTLCache(store, name="night", clock=clock)
    holder: Dict[str, Rig] = {}

    def restart() -> Rig:
        holder["rig"] = Rig(tmp, world, mp, name="raw", config=config, kalshi=("KXEPLCORNERS",), store=store,
                            cache=cache)
        return holder["rig"]

    def tag() -> str:
        rig = holder["rig"]
        backfill = rig.capturer._backfill
        return "backfill" if backfill is not None and rig.http._guard.get() == backfill._allow else "forward"

    world.tagger = tag
    rig = restart()
    assert rig.raw.path == resolve_raw_db_path(config, store)  # the file the health block and the learn hook read
    rig.capturer.schedule_from_slate([{"league": NBA, "espn_league": NBA, "espn_id": event_id, "start": iso(start),
                                       "_ts": start, "status": "pre", "_time_ok": True}
                                      for event_id, start in PRE_GAMES.items()])
    nights: Set[str] = set()
    ticks: List[Dict[str, Any]] = []
    learn: Dict[str, Any] = {}
    progress: Dict[str, List[Dict[str, Any]]] = {}  # night -> backfill_progress as the night began
    while clock() < END:
        backfill = rig.capturer.backfill
        if backfill.in_window() and backfill.night() not in nights:
            nights.add(backfill.night())
            progress[backfill.night()] = rig.raw.progress()
            rig = restart()  # a new night, a new process: the cursors and budgets are all in raw.db
            backfill = rig.capturer.backfill
        ticks.append(rig.capturer.tick())
        if not learn and clock() >= LEARN_AT:
            hooked_raw = RawStore(resolve_raw_db_path(config, store))  # as learn/loop.py opens it
            learn["hooked"] = learn_backfill(world, store, cache, hooked_raw)
            plain_store = Store(str(tmp / "plain.db"))
            learn["plain"] = learn_backfill(world, plain_store, TTLCache(plain_store, name="plain", clock=clock), None)
            learn["dumps"] = (hist_dump(store), hist_dump(plain_store))
        working = backfill.in_window() and rig.raw.spent("backfill", backfill.night()) < PER_NIGHT
        upcoming = clock() + (60.0 if working else 1800.0)
        opens = datetime.datetime.combine(datetime.datetime.fromtimestamp(clock(), UTC).date(),
                                          datetime.time(1, 0, 30), UTC).timestamp()
        opens += 86400.0 if opens <= clock() else 0.0
        due = rig.raw.next_due()
        for moment in (opens, due + 1.0 if due is not None else None):  # a tick a minute: none of these is missed
            if moment is not None and clock() < moment < upcoming:
                upcoming = moment
        clock.now = max(clock() + 1.0, upcoming)
    return SimpleNamespace(world=world, store=store, cache=cache, config=config, keys=keys, ticks=ticks,
                           nights=sorted(nights), learn=learn, rig=rig, raw=rig.raw, refused=refused, tmp=tmp,
                           now=clock(), progress=progress)


def test_the_night_backfill_walks_newest_season_first_in_wave_order(night: Any) -> None:
    expected = [(league, season) for season in (2026, 2025, 2024) for league in bf.WAVES_1_2]
    walked: Dict[str, List[Tuple[str, int]]] = collections.defaultdict(list)  # night -> units in request order
    for ts, who, url in night.world.tags:
        unit = unit_of(url, night.world) if who == "backfill" else None
        if unit is not None:
            walked[datetime.datetime.fromtimestamp(ts, UTC).date().isoformat()].append(unit)
    for day, units in walked.items():  # each night goes down the list: newest season first, leagues in wave order
        places = [expected.index(unit) for unit in units]
        assert places == sorted(places), day
    firsts = list(dict.fromkeys(walked["2026-11-01"]))
    assert firsts == [unit for unit in expected if unit in firsts] and firsts[0] == (NBA, 2026)
    everything = {unit for units in walked.values() for unit in units}
    assert len(everything) >= 40 and {league for league, _s in everything} == set(bf.WAVES_1_2)
    # within a season the newest day first: each unit's boards come newest first (the days a season still being
    # played settled later come after, as a new top)
    for unit in firsts:
        days = [urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["dates"][0]
                for _ts, who, url in night.world.tags
                if who == "backfill" and "/scoreboard?" in url and unit_of(url, night.world) == unit]
        first_night = [day for day in days if day <= "20261030"]
        assert first_night == sorted(first_night, reverse=True), unit
    # and every league-season is done
    progress = {(row["league"], row["season"]): row for row in night.raw.progress()}
    assert set(progress) == set(expected) and all(row["done"] for row in progress.values())


def test_the_night_backfill_keeps_to_its_budget_and_its_window(night: Any) -> None:
    by_night: Dict[str, int] = collections.Counter()
    for ts, who, _url in night.world.tags:
        if who != "backfill":
            continue
        moment = datetime.datetime.fromtimestamp(ts, UTC)
        assert 1 <= moment.hour < 6, f"a backfill request at {moment}"  # 01:00-06:00 server time (UTC here)
        by_night[moment.date().isoformat()] += 1
    assert night.nights == ["2026-11-01", "2026-11-02", "2026-11-03"] and sorted(by_night) == night.nights[:2]
    assert all(n <= PER_NIGHT for n in by_night.values()) and by_night["2026-11-01"] == PER_NIGHT
    for day, n in by_night.items():  # the night's counter is exactly the requests made
        assert night.raw.spent("backfill", day) == n
    assert all(report.get("backfill") is None or report["backfill"]["stopped"] in
               ("budget", "done", "forward", None) for report in night.ticks)


def test_the_night_backfill_resumes_mid_season_after_a_restart(night: Any) -> None:
    """Each night was a new process: the units cut off by a night's budget carry on the next night from their
    cursor, and no request of the backfill is ever made twice."""
    urls = [url for _ts, who, url in night.world.tags if who == "backfill"]
    assert len(urls) == len(set(urls))
    spans: Dict[Tuple[str, int], Set[str]] = collections.defaultdict(set)
    for ts, who, url in night.world.tags:
        unit = unit_of(url, night.world) if who == "backfill" else None
        if unit is not None:
            spans[unit].add(datetime.datetime.fromtimestamp(ts, UTC).date().isoformat())
    cut = [unit for unit, days in spans.items() if len(days) > 1]
    assert cut, "a unit was cut off by a night's budget and finished on a later night"
    # The new process found where each cut unit stood in its cursor (backfill_progress), saved after every step
    before_night2 = {(row["league"], row["season"]): row for row in night.progress["2026-11-02"]}
    for unit in cut:
        row = before_night2[unit]
        last_board = max((url for ts, who, url in night.world.tags if who == "backfill" and "/scoreboard?" in url
                          and unit_of(url, night.world) == unit
                          and datetime.datetime.fromtimestamp(ts, UTC).date().isoformat() == "2026-11-01"),
                         key=lambda url: [t for t, _w, u in night.world.tags if u == url][0], default=None)
        assert not row["done"] and isinstance(row["cursor"], dict) and row["games_seen"] >= 0
        if last_board is not None:
            day = urllib.parse.parse_qs(urllib.parse.urlsplit(last_board).query)["dates"][0]
            assert row["cursor"]["day"] == f"{day[:4]}-{day[4:6]}-{day[6:]}", unit  # the last board it read


def test_the_backfill_skips_what_forward_capture_stored(night: Any) -> None:
    """Every game's summary and plays page was fetched once in all (forward capture or the backfill, never both),
    and the backfill read the boards the sweep had kept from raw.db instead of asking again."""
    summaries = [url for _ts, _who, url in night.world.tags if ("/summary?" in url or "/plays?" in url)
                 and not any(event_id in url for event_id in PRE_GAMES)]  # (a t90 snapshot and the final)
    assert len(summaries) == len(set(summaries))
    forward = {url for _ts, who, url in night.world.tags if who == "forward"}
    backfill = {url for _ts, who, url in night.world.tags if who == "backfill"}
    assert not forward & backfill
    # the recorded day's and 31 October's boards were the sweep's: the backfill never asked for them
    assert not [url for url in backfill if "dates=20261030" in url or "dates=20261031" in url]
    assert any("dates=20261031" in url for url in forward)
    # 31 October's games (soccer, NHL, a UFC card) came by the sweep's jobs, and the backfill counted them stored
    october31 = [f"espn:{eid}" for league in bf.WAVES_1_2
                 for eid in backfill_games(league, datetime.date(2026, 10, 31))]
    assert october31 and all(night.raw.has_final(key) or night.raw.has_final(key, "espn_fightcenter", "fightcenter")
                             for key in october31)


def test_a_due_forward_job_is_never_delayed_by_the_backfill(night: Any) -> None:
    """Each pre game's t90 snapshot falls due while the first night's backfill runs: from that moment nothing of the
    backfill goes out before the game's own requests, which go within seconds."""
    tags = night.world.tags
    for event_id, start in PRE_GAMES.items():
        due = start - 90 * 60
        later = [(ts, who, url) for ts, who, url in tags if ts >= due]
        mine = next(i for i, (_ts, _who, url) in enumerate(later) if event_id in url)
        assert all(who == "forward" for _ts, who, _url in later[:mine]), (event_id, later[:mine + 1])
        assert later[mine][0] - due < 10.0
        assert any(who == "backfill" for ts, who, _u in tags if due - 5.0 < ts < due)  # the backfill was running
        assert any(who == "backfill" for _ts, who, _u in later)  # and carried on after it
        job = night.raw.job(f"espn:{event_id}")
        assert {"pre:t90:summary", "pre:t90:propbets", "pre:close:propbets"} <= set(job.kinds_done)


def test_the_learn_hook_asks_nothing_for_a_game_already_in_raw_db(night: Any) -> None:
    hooked_report, hooked = night.learn["hooked"]
    plain_report, plain = night.learn["plain"]
    asked = [urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("event", [None])[0] for url in hooked
             if "/summary?" in url]
    assert sorted(asked) == sorted(LEARN_IDS[2:])  # the forward-captured and the backfilled games: no request
    assert sorted(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["event"][0] for url in plain
                  if "/summary?" in url) == sorted(LEARN_IDS)
    assert [url for url in hooked if "/summary?" not in url] == [url for url in plain if "/summary?" not in url]
    assert hooked_report["stopped"] == plain_report["stopped"] == "done"
    assert hooked_report["requests"] == plain_report["requests"] - 2
    hooked_dump, plain_dump = night.learn["dumps"]
    assert hooked_dump == plain_dump  # the same history either way
    # the two it fetched are in raw.db now too (one fetch served both stores)
    for event_id in LEARN_IDS:
        payload = night.raw.latest(f"espn:{event_id}", "espn_site", "summary")
        assert payload is not None and payload.state == "final"


def test_the_whole_run_never_asked_the_odds_api_anthropic_or_deepseek(night: Any) -> None:
    hosts = collections.Counter(host for _ts, _door, host, _url in night.world.sent)
    assert not [host for host in hosts if any(word in host for word in FORBIDDEN)]
    assert night.refused == []  # nothing tried a real door (requests, urlopen, an opener)
    with night.store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM api_usage").fetchone()[0] == 0  # The Odds API's quota log
    allowed = ESPN_HOSTS | {"statsapi.mlb.com", "github.com", "raw.githubusercontent.com", "football-data.co.uk",
                            "api.elections.kalshi.com"}
    assert set(hosts) <= allowed and {"github.com", "raw.githubusercontent.com", "football-data.co.uk",
                                      "statsapi.mlb.com"} <= set(hosts)


def test_no_host_ever_went_above_its_bucket(night: Any) -> None:
    assert len(night.world.asked("polite")) > 600
    assert bucket_overruns(night.world.sent) == []


def test_every_capture_request_is_counted_once(night: Any) -> None:
    """http_ok + http_error (the health block's HTTP alarm) count each request the polite door sent exactly once,
    forward and backfill alike."""
    by_day: Dict[str, int] = collections.Counter(raw_store.utc_day(ts) for ts, door, _h, _u in night.world.sent
                                                 if door == "polite")
    for day, n in by_day.items():
        counted = sum(night.raw.spent(kind, day) for kind in ("http_ok", "http_error", "backfill_http_ok",
                                                              "backfill_http_error"))
        assert counted == n, day
        tagged = collections.Counter(who for ts, who, _u in night.world.tags if raw_store.utc_day(ts) == day)
        assert night.raw.spent("http_ok", day) + night.raw.spent("http_error", day) == tagged["forward"], day
        assert night.raw.spent("backfill_http_ok", day) + night.raw.spent("backfill_http_error", day) == \
            tagged["backfill"], day


# --- the Brain page's capture block, over the raw.db the run produced -------------------------------------------------

def test_the_health_block_reports_coverage_size_disk_and_the_backfill_of_the_run(
        night: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from telegram_betting_bot_simple.learn import progress as progress_module
    monkeypatch.setenv("CAPTURE_BACKFILL_PER_NIGHT", str(PER_NIGHT))  # the run's settings, as the server has them
    monkeypatch.delenv("RAW_DB_PATH", raising=False)
    for league in bf.ORDER:
        monkeypatch.setitem(bf.WINDOWS, league, WINDOW)
    page = progress_module.progress(night.store, None, now=night.now)  # GET /api/brain/progress
    block = page["capture"]
    assert block["enabled"] is True and block["stored"] is True and not block.get("unreadable")
    # Coverage: every final of the last 7 days stored within 36 h of its end, per sport
    by_sport = {row["sport"]: row for row in block["coverage"]["sports"]}
    finals = {"basketball": 2 + len(PRE_GAMES), "football": 1, "hockey": 2, "baseball": 1, "soccer": 10, "mma": 2,
              "tennis": 3}
    assert {sport: row["finals"] for sport, row in by_sport.items()} == finals
    assert all(row["covered"] == row["finals"] and row["share"] == 1.0 and not row["missed"] and not row["pending"]
               for row in by_sport.values())
    assert block["coverage"]["total"]["share"] == 1.0 and block["coverage"]["target"] == 0.98
    # Size and disk: raw.db with its WAL, the (patched) disk, no pause
    size = block["size"]
    assert os.path.getsize(night.raw.path) <= size["raw_bytes"] <= night.raw.size_bytes()  # (+ the reader's -shm)
    assert (size["disk_total"], size["disk_free"], size["disk_full"], size["paused"]) == (
        int(ROOMY.total), int(ROOMY.free), 0.1, False)
    assert size["floor_bytes"] == int(max(2 * 1024 ** 3, 0.10 * ROOMY.total))
    # Requests: today's (UTC) against the cap, the backfill's last night against its own
    today = raw_store.utc_day(night.now)
    sent_today = [url for ts, who, url in night.world.tags if raw_store.utc_day(ts) == today and who == "forward"]
    requests_block = block["requests"]
    assert requests_block["day"] == today and requests_block["capture"] == len(sent_today)
    assert requests_block["per_day"] == 3000 and requests_block["per_night"] == PER_NIGHT
    assert requests_block["backfill_night"] == "2026-11-02"
    assert requests_block["backfill"] == night.raw.spent("backfill", "2026-11-02") > 0
    # HTTP: every request of today and yesterday counted once, none failed
    days = {today, raw_store.utc_day(night.now - 86400)}
    forward = [ts for ts, who, _u in night.world.tags if who == "forward" and raw_store.utc_day(ts) in days]
    backfill = [ts for ts, who, _u in night.world.tags if who == "backfill" and raw_store.utc_day(ts) in days]
    assert block["http"]["counted"] == "requests" and block["http"]["requests"] == len(forward)
    assert block["http"]["errors"] == 0 and block["http"]["share"] == 0.0
    assert block["backfill_http"]["requests"] == len(backfill) and block["backfill_http"]["errors"] == 0
    # The backfill per league and season, in wave order, all done
    assert [row["league"] for row in block["backfill"]] == list(bf.WAVES_1_2)
    for row in block["backfill"]:
        assert row["begun"] == row["done"] == 3 and [s["season"] for s in row["seasons"]] == [2026, 2025, 2024]
        assert row["planned"] == 3  # the backfill's own scope: 3 completed seasons, none being played here
    stored_games = sum(row["games_stored"] for row in block["backfill"])
    assert stored_games >= len([url for _t, who, url in night.world.tags if who == "backfill" and "/summary?" in url])
    # Every shape as the plan's payloads have it, and so no alarm and no error
    assert block["shapes"] and all(shape["ok"] for shape in block["shapes"])
    assert block["errors"] == [] and block["alarms"] == []
    json.dumps(page)  # the route's JSON


def _copy_raw_db(source: str, target: Path) -> str:
    """A consistent copy of a raw.db (its WAL included), to change without touching the run's own."""
    src, dst = sqlite3.connect(source), sqlite3.connect(str(target))
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    return str(target)


def test_the_health_block_raises_its_alarms_and_survives_an_empty_raw_db(night: Any, tmp_path: Path,
                                                                         disk: SimpleNamespace) -> None:
    from telegram_betting_bot_simple.learn.progress import capture_health
    path = _copy_raw_db(night.raw.path, tmp_path / "raw.db")
    copy_store = RawStore(path, clock=lambda: night.now)
    now = night.now
    # A payload-shape change: ESPN's newest game summary lost its box score
    copy_store.put(game_key="espn:401999999", league=NBA, source="espn_site", endpoint="summary", params={},
                   body=json.dumps({"header": {"id": "401999999"}}).encode(), state="final", http_status=200,
                   horizon="final", fetched_at=now)
    # A failing host: 300 errors today among the counted requests
    copy_store.spend("http_error", raw_store.utc_day(now), 300)
    # And the disk: 95% full, under the 10% floor: capture and the backfill are paused
    disk.usage = Usage(100e9, 95e9, 5e9)
    block = capture_health(path, now, settings={"per_day": 3000, "per_night": PER_NIGHT, "seasons": 3})
    kinds = [alarm["kind"] for alarm in block["alarms"]]
    assert sorted(kinds) == ["disk", "http", "shape"], block["alarms"]
    texts = {alarm["kind"]: alarm["text"] for alarm in block["alarms"]}
    assert texts["disk"] == ("The disk is 95% full: 4.7 GB free. Capture and the backfill are paused until 9.3 GB "
                             "is free.")
    assert texts["shape"].startswith("ESPN game summaries changed shape: boxscore is missing from the newest answer")
    assert texts["http"].endswith(f"failed today and yesterday (UTC): 300 of {block['http']['requests']:,}.")
    assert block["size"]["paused"] is True
    # At 75% full with room above the floor: the 70% alarm, not paused
    disk.usage = Usage(100e9, 75e9, 25e9)
    block = capture_health(path, now, settings={"per_day": 3000})
    assert [a["text"] for a in block["alarms"] if a["kind"] == "disk"] == [
        f"The disk is 75% full: 23.3 GB free, raw.db is {progress_bytes(block['size']['raw_bytes'])}."]
    assert block["size"]["paused"] is False
    # No raw.db yet: nothing stored, nothing created, no alarm
    missing = tmp_path / "none" / "raw.db"
    empty = capture_health(str(missing), now)
    assert (empty["stored"], empty["alarms"], empty["size"]) == (False, [], None) and not missing.exists()
    # A raw.db just created by the worker (its tables, no rows): an honest empty block
    fresh = RawStore(str(tmp_path / "fresh.db"))
    disk.usage = ROOMY
    block = capture_health(fresh.path, now)
    assert block["stored"] is True and block["coverage"]["total"]["finals"] == 0 and block["alarms"] == []
    assert block["backfill"] == [] and block["errors"] == [] and block["http"]["requests"] == 0


def progress_bytes(n: int) -> str:
    from telegram_betting_bot_simple.learn.progress import _bytes_text
    return _bytes_text(n)


# --- capture off: nothing outside capture changes ----------------------------------------------------------------------

class FakeThread:
    started: List["FakeThread"] = []

    def __init__(self, target: Any = None, args: Tuple[Any, ...] = (), name: str = "", daemon: bool = False) -> None:
        self.target, self.args, self.name = target, args, name

    def start(self) -> None:
        FakeThread.started.append(self)


def test_capture_off_creates_no_raw_db_and_starts_no_capture_thread(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """The dashboard's main() with CAPTURE_ENABLED=0 (the real settings, the real build_slate): no capturer, no
    "capture" thread, the slate without a capturer, the same threads as before P1, and no raw.db anywhere."""
    from telegram_betting_bot_simple.config import Config
    from telegram_betting_bot_simple.dashboard import __main__ as dashboard_main
    from telegram_betting_bot_simple.dashboard.service import DashboardService
    monkeypatch.delenv("RAW_DB_PATH", raising=False)
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path))

    def boot(**settings: str) -> Tuple[List[str], Any]:
        FakeThread.started = []
        config = Config(TELEGRAM_BOT_TOKEN="123456:test", DASHBOARD_PASSWORD="s3cret", _env_file=None,
                        MARKETS_ENABLED="0", **settings)
        store = Store(str(tmp_path / "betting.db"))
        cache = TTLCache(store, name="off")
        service = DashboardService(NoOdds(), EspnClient(cache), ET, store=store, cache=cache)
        built: Dict[str, Any] = {}
        real_build_slate = dashboard_main.build_slate

        def build_slate(config: Any, service: Any) -> Any:
            slate = built["slate"] = real_build_slate(config, service)
            monkeypatch.setattr(slate, "start", lambda: None)  # its own threads aren't the point here
            return slate

        monkeypatch.setattr(dashboard_main, "load_config", lambda: config)
        monkeypatch.setattr(dashboard_main, "build_service", lambda config: service)
        monkeypatch.setattr(dashboard_main, "build_slate", build_slate)
        monkeypatch.setattr(dashboard_main, "build_learner", lambda config, service: None)
        monkeypatch.setattr(dashboard_main, "create_app", lambda *args, **kwargs: "app")
        monkeypatch.setattr(dashboard_main.uvicorn, "run", lambda app, **kwargs: None)
        monkeypatch.setattr(dashboard_main, "threading", SimpleNamespace(Thread=FakeThread))
        dashboard_main.main()
        return [thread.name for thread in FakeThread.started], built["slate"]

    names, slate = boot(CAPTURE_ENABLED="0")
    assert names == ["grader", "cache-summary"]  # the threads main() started before P1 (no learner, no analysts)
    assert slate.capturer is None and not list(tmp_path.rglob("raw.db*"))
    on_names, on_slate = boot()
    assert on_names == ["grader", "cache-summary", "capture"] and on_slate.capturer is not None
    assert (tmp_path / "raw.db").exists()  # beside betting.db, only when capture is on


def test_capture_off_leaves_the_slate_round_as_it_was(tmp_path: Path) -> None:
    """A slate round without a capturer (CAPTURE_ENABLED=0) serves the same bytes as the round before P1 (the hook's
    call removed), and starts no hand-off thread."""
    from telegram_betting_bot_simple.data.slate import SlateBuilder

    day = datetime.date(2026, 10, 26)
    now = at(day, 18)

    def competition(event_id: str, home: str, away: str, start: float) -> Dict[str, Any]:
        def side(where: str, label: str, team: str) -> Dict[str, Any]:
            return {"homeAway": where, "score": "0", "team": {"id": team, "displayName": label,
                                                             "shortDisplayName": label.split()[-1],
                                                             "abbreviation": label[:3].upper()}}
        return {"id": event_id, "date": iso(start), "competitions": [{
            "id": event_id, "date": iso(start), "venue": {"fullName": "Arena", "address": {"city": "Boston"}},
            "status": {"period": 0, "displayClock": "0.0", "type": {"name": "STATUS_SCHEDULED", "state": "pre",
                                                                    "completed": False, "shortDetail": "7 PM"}},
            "competitors": [side("home", home, "1"), side("away", away, "2")]}]}

    class Espn:
        def fetch_scoreboard(self, league: str, board_day: datetime.date) -> List[Dict[str, Any]]:
            if board_day != day:
                return []
            return {NBA: [competition("401", "Boston Celtics", "New York Knicks", now + 3600)],
                    NHL: [competition("402", "Boston Bruins", "Toronto Maple Leafs", now + 7200)]}.get(league, [])

    class Odds(NoOdds):
        odds_ttl, reserve = 7200, 50

        def sports(self) -> List[Dict[str, Any]]:
            return [{"key": NBA, "group": "Basketball", "title": "NBA", "active": True, "has_outrights": False},
                    {"key": NHL, "group": "Ice Hockey", "title": "NHL", "active": True, "has_outrights": False}]

    def round_bytes(name: str, hooked_call: bool) -> bytes:
        slate = SlateBuilder(Odds(), Espn(), Store(str(tmp_path / f"{name}.db")), now=lambda: now, workers=2)
        if not hooked_call:
            slate._schedule_capture = lambda games, now: None  # type: ignore[method-assign]  # c0c3dd3's round
        before = {t.name for t in threading.enumerate()}
        slate.refresh()
        assert slate.capturer is None and slate._capture_thread is None
        assert "slate-capture" not in {t.name for t in threading.enumerate()} - before
        return slate.current().raw

    assert round_bytes("off", True) == round_bytes("before", False)


def test_capture_off_leaves_the_learn_backfill_as_it_was(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """learn/loop.py's nightly backfill with CAPTURE_ENABLED=0 makes the same requests and writes the same history as
    the call before P1 (``Backfill(store, cache, clock, tz=..., per_night=...)``), and opens no raw.db; with capture
    on and raw.db empty, too (the hook only adds the raw copies)."""
    from telegram_betting_bot_simple.learn import analyst, history, loop
    from telegram_betting_bot_simple.learn.sources.http import HistHttp
    clock = Clock(datetime.datetime(2026, 10, 28, 2, 0, tzinfo=UTC).timestamp())
    world = World(clock)
    monkeypatch.delenv("RAW_DB_PATH", raising=False)

    class FakeClockHistHttp(HistHttp):  # the learn backfill's own door, on the fake clock and the world
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.update(clock=clock, sleep=clock.sleep, transport=world.learn_transport)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(history, "HistHttp", FakeClockHistHttp)

    def via_learner(name: str, config: Any) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
        clock.now = datetime.datetime(2026, 10, 28, 2, 0, tzinfo=UTC).timestamp()
        folder = tmp_path / name
        folder.mkdir()
        store = Store(str(folder / "betting.db"))
        learner = loop.Learner(store, config=config, clock=clock, tz=UTC, use_notify=False,
                               analyst=analyst.BrainAnalyst(store, mode="off", clock=clock, tz=UTC),
                               lease=lambda: True)
        learner.cache = TTLCache(store, name=name, clock=clock)
        before = len(world.sent)
        report = learner._night_backfill(clock(), "2026-10-28")
        return report, [url for _t, door, _h, url in world.sent[before:] if door == "learn"], hist_dump(store)

    def as_before(name: str) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
        clock.now = datetime.datetime(2026, 10, 28, 2, 0, tzinfo=UTC).timestamp()
        folder = tmp_path / name
        folder.mkdir()
        store = Store(str(folder / "betting.db"))
        before = len(world.sent)
        report = history.Backfill(store, TTLCache(store, name=name, clock=clock), clock, tz=UTC,
                                  per_night=history.per_night_budget()).run_night()
        report = {k: report.get(k) for k in ("ran", "stopped", "used", "requests", "games", "budget")}
        return report, [url for _t, door, _h, url in world.sent[before:] if door == "learn"], hist_dump(store)

    base_report, base_urls, base_history = as_before("base")
    assert base_report["ran"] and any("/summary?" in url for url in base_urls)
    off = via_learner("off", SimpleNamespace(capture_enabled=False, raw_db_path=None))
    assert off == (base_report, base_urls, base_history)
    assert not list((tmp_path / "off").glob("raw.db*"))
    on = via_learner("on", SimpleNamespace(capture_enabled=True, raw_db_path=None))
    assert on == (base_report, base_urls, base_history)  # raw.db empty: the same requests, the same history
    raw = RawStore(str(tmp_path / "on" / "raw.db"))
    summaries = [url for url in base_urls if "/summary?" in url]
    assert raw.payload_count() == len(set(summaries)) > 0  # ... and the raw bodies kept


# --- two processes, one capture ----------------------------------------------------------------------------------------

TWO_PROCESS_SCRIPT = textwrap.dedent('''
    """One dashboard process's capture thread (a few ticks), over a shared betting.db and raw.db, its network faked."""
    import collections, json, os, sys, time
    from types import SimpleNamespace
    sys.path.insert(0, sys.argv[1])
    from telegram_betting_bot_simple.capture import plan, raw_store
    from telegram_betting_bot_simple.capture.http import HttpResponse, PoliteHttp
    from telegram_betting_bot_simple.capture.raw_store import RawStore
    from telegram_betting_bot_simple.capture.worker import Capturer
    from telegram_betting_bot_simple.data.cache import TTLCache
    from telegram_betting_bot_simple.data.espn_client import EspnClient
    from telegram_betting_bot_simple.data.store import Store
    root, betting, raw_path, log_path, go, ticks = sys.argv[1:7]
    raw_store.shutil.disk_usage = lambda folder: collections.namedtuple("U", "total used free")(100e9, 10e9, 90e9)
    body = open(os.path.join(root, "tests", "fixtures", "capture", "sum_nhl.json"), "rb").read()
    sent = []
    def send(request):
        sent.append((time.time(), request.full_url))
        time.sleep(0.02)
        return HttpResponse(200, body, {})
    http = PoliteHttp(rates={"site.api.espn.com": (50.0, 5)})
    http._send = send
    store = Store(betting)
    capturer = Capturer(store, RawStore(raw_path), EspnClient(TTLCache()), http, time.time,
                        config=SimpleNamespace(capture_backfill=False), options=plan.Options(propbets=False))
    while not os.path.exists(go):
        time.sleep(0.01)
    with open(log_path, "a") as log:
        for _ in range(int(ticks)):
            mark = len(sent)
            start = time.time()
            report = capturer.tick()
            log.write(json.dumps({"pid": os.getpid(), "start": start, "end": time.time(),
                                  "sent": sent[mark:], "skipped": "skipped" in report}) + "\\n")
            log.flush()
            time.sleep(0.05)
''')


def test_two_dashboard_processes_never_capture_at_the_same_time(tmp_path: Path) -> None:
    """Two real processes on one betting.db and raw.db (two dashboards): the store lease "capture" lets one of them
    capture; every tick of the other is skipped, so their requests never overlap."""
    from telegram_betting_bot_simple.capture.plan import STATIC, static_period
    script = tmp_path / "capture_process.py"
    script.write_text(TWO_PROCESS_SCRIPT)
    betting, raw_path, log, go = (str(tmp_path / n) for n in ("betting.db", "raw.db", "ticks.log", "go"))
    Store(betting)
    raw = RawStore(raw_path)
    now = datetime.datetime.now(UTC).timestamp()
    # 30 finished NHL games due now; yesterday's sweep and the static files already done (no other work)
    yesterday = worker_module.espn_day(now) - datetime.timedelta(days=1)
    raw.set_state("sweep:last", yesterday.isoformat())
    for entry in STATIC:
        raw.set_state(f"static:{entry.name}", {"at": now, "ok": True, "period": static_period(entry, now)})
    for n in range(30):
        raw.add_job(f"espn:4018926{n:02d}", NHL, f"4018926{n:02d}", start=now - 6 * 3600, due_at=now - 60, now=now)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    procs = [subprocess.Popen([sys.executable, "-I", "-B", str(script), str(ROOT), betting, raw_path, log, go, "8"],
                              env=env) for _ in range(2)]
    try:
        Path(go).write_text("go")
        assert all(proc.wait(timeout=120) == 0 for proc in procs)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
    ticks = [json.loads(line) for line in Path(log).read_text().splitlines()]
    pids = {tick["pid"] for tick in ticks}
    assert len(pids) == 2 and len(ticks) == 16
    capturing = {tick["pid"] for tick in ticks if not tick["skipped"]}
    assert len(capturing) == 1, "exactly one process holds the capture lease"
    other = (pids - capturing).pop()
    assert all(tick["skipped"] and not tick["sent"] for tick in ticks if tick["pid"] == other)
    sent = [item for tick in ticks for item in tick["sent"]]
    assert len(sent) == 30 and len({url for _t, url in sent}) == 30  # every game once, by the one holder
    assert all(job.state == "done" for job in raw.jobs())


class FakeTime:
    """data/store.py's ``time`` on the fake clock (its leases' expiry); everything else real."""

    def __init__(self, clock: Clock) -> None:
        import time as real
        self._real, self._clock = real, clock

    def time(self) -> float:
        return self._clock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def test_two_lease_holders_never_capture_at_once_even_through_a_long_tick(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """Two capturers on one betting.db with different lease holders (two processes): while one holds the "capture"
    lease the other never fetches, even when one tick runs longer than the lease (slow answers); a holder that dies
    is replaced once its lease runs out."""
    from telegram_betting_bot_simple.data import store as store_module
    clock = Clock(at(DAY, 12))
    monkeypatch.setattr(store_module, "time", FakeTime(clock))
    world = World(clock, latency=4.0)  # ESPN answering slowly: a tick's 300 summaries take over 20 minutes
    for n in range(320):
        event_id = f"4018926{n:03d}"
        world.games[event_id] = Game(NHL, event_id, at(DAY, 1), "sum_nhl.json")
    path = str(tmp_path / "betting.db")
    mine, theirs = Store(path), Store(path)
    theirs.holder = "another-process"
    a = Rig(tmp_path, world, monkeypatch, name="raw", store=mine,
            lease=lambda: bool(mine.try_lease(worker_module.LEASE, worker_module.LEASE_SECONDS)))
    b = Rig(tmp_path, world, monkeypatch, name="raw", store=theirs, cache=a.cache,
            lease=lambda: bool(theirs.try_lease(worker_module.LEASE, worker_module.LEASE_SECONDS)))
    for rig in (a, b):
        rig.raw.set_state("sweep:last", (DAY - datetime.timedelta(days=1)).isoformat())
        for entry in plan.STATIC:
            rig.raw.set_state(f"static:{entry.name}", {"at": clock(), "ok": True,
                                                       "period": plan.static_period(entry, clock(), UTC)})
    a.capturer.schedule_from_slate([{"league": NHL, "espn_league": NHL, "espn_id": g.event_id, "start": iso(g.start),
                                     "_ts": g.start, "status": "final"} for g in world.games.values()])
    clock.now = at(DAY, 12, 40)
    who: List[Tuple[float, str]] = []
    a.http._send = lambda request: (who.append((clock(), "a")), world.send(request))[1]  # type: ignore
    b_ticks: List[Dict[str, Any]] = []

    def b_send(request: Any) -> HttpResponse:
        who.append((clock(), "b"))
        return world.send(request)

    b.http._send = b_send  # type: ignore[method-assign]
    next_b = [clock() + 60.0]

    def a_send(request: Any) -> HttpResponse:  # the other process ticks every minute while this one works
        who.append((clock(), "a"))
        while clock() >= next_b[0]:
            next_b[0] += 60.0
            b_ticks.append(b.capturer.tick())
        return world.send(request)

    a.http._send = a_send  # type: ignore[method-assign]
    started = clock()
    report = a.capturer.tick()
    assert clock() - started > worker_module.LEASE_SECONDS  # one tick longer than the lease
    assert report["due"]["fetched"] == worker_module.TICK_BUDGET
    assert b_ticks and all(tick.get("skipped") for tick in b_ticks), "the other process waited its turn"
    assert {name for _t, name in who} == {"a"}
    # The holder dies (no more ticks): the other takes over once the lease has run out, never before
    died = clock()
    with sqlite3.connect(path) as db:
        holder, expires = db.execute("SELECT holder, expires FROM leases WHERE key = 'capture'").fetchone()
    assert holder == mine.holder and expires > died + worker_module.LEASE_SECONDS - worker_module.LEASE_RENEW - 60
    clock.now = expires - 30
    assert b.capturer.tick().get("skipped")
    clock.now = expires + 1
    report = b.capturer.tick()
    assert not report.get("skipped") and report["due"]["fetched"] == 20  # the rest of the games
    assert [name for t, name in who if t > died] == ["b"] * 20


# --- a board that isn't JSON, and the disk floor ------------------------------------------------------------------------

def test_a_board_that_isnt_json_costs_the_sweep_only_itself(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ESPN answers one league's board with a 2xx that isn't JSON (a maintenance page): the sweep keeps it raw, reads
    the other 57 boards and their games, and tries that league again later a few times, not every minute."""
    clock = Clock(at(DAY + datetime.timedelta(days=1), 5, 30))
    world = World(clock)
    keys = recorded_day(world)
    world.garbage.add("/basketball/nba/scoreboard")
    rig = Rig(tmp_path, world, monkeypatch)
    for _ in range(4 * 60):  # four hours of ticks, a minute apart
        rig.capturer.tick()
        clock.now += 60.0
    nba = [url for url in (u for _t, d, _h, u in world.sent if d == "polite") if "/basketball/nba/scoreboard" in url]
    assert 1 <= len(nba) <= worker_module.SWEEP_TRIES
    boards = {url for _t, d, _h, url in world.sent if d == "polite" and "/scoreboard?" in url}
    assert len(boards) == 58
    # the other sports' games of the day were all captured from their boards
    for sport in ("football", "hockey", "baseball", "soccer", "mma"):
        assert rig.raw.job(keys[sport]).state == "done", sport
    assert rig.raw.job(keys["basketball"]) is None  # its board never read: the slate's jobs and tomorrow's cover it
    kept = rig.raw.latest(scoreboard_key(NBA, "20261030"), "espn_site", "scoreboard")
    assert kept is not None and kept.raw_bytes().startswith(b"<html>")  # kept as it came
    progress = rig.raw.get_state("sweep:2026-10-30")
    assert progress["failed"] == [NBA] and progress["tries"] == worker_module.SWEEP_TRIES
    assert rig.raw.get_state("sweep:last") == "2026-10-30"
    assert rig.espn.cache.peek(("espn", "scoreboard", NBA, "2026-10-30")) is None  # nothing seeded from it


def test_without_room_on_the_disk_nothing_is_fetched_and_the_learn_hook_writes_nothing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disk: SimpleNamespace) -> None:
    """Under the floor (CAPTURE_MIN_FREE_GB, 2 GiB, and 10% of the disk free): the sweep, the due jobs, the static
    files and the backfill fetch nothing, the learn hook still reads raw.db but keeps no new body, and the health
    block says capture and the backfill are paused. When room comes back (within a minute) everything resumes."""
    from telegram_betting_bot_simple.learn.progress import capture_health
    monkeypatch.setattr(bf, "_projected", True)
    for league in bf.ORDER:
        monkeypatch.setitem(bf.WINDOWS, league, WINDOW)
    clock = Clock(at(DAY + datetime.timedelta(days=1), 5, 30))
    world = World(clock, seasons=True)
    recorded_day(world)
    config = SimpleNamespace(capture_backfill=True, capture_backfill_seasons=1, capture_backfill_per_night=50)
    rig = Rig(tmp_path, world, monkeypatch, config=config)
    rig.raw.put(game_key=f"espn:{LEARN_IDS[0]}", league=NBA, source="espn_site", endpoint="summary", params={},
                body=(LEARN_FIX / "espn_nba_summary.json").read_bytes(), state="final", http_status=200,
                horizon="final", fetched_at=clock())
    rows = rig.raw.payload_count()
    disk.usage = Usage(100e9, 92e9, 8e9)  # 8% free: under the 10% floor
    report = rig.capturer.tick()
    assert world.asked("polite") == [] and rig.capturer.last["sweep"]["stopped"] == "disk"
    assert report["due"]["stopped"] == "disk" and report["static"] == {"stopped": "disk"}
    clock.now = datetime.datetime(2026, 11, 1, 2, 0, tzinfo=UTC).timestamp()  # inside the backfill's night
    report = rig.capturer.tick()
    assert report["backfill"]["stopped"] == "disk" and world.asked("polite") == []
    night = rig.capturer.backfill.run_night(budget=15)
    assert (night["ran"], night["stopped"], night["requests"]) == (False, "disk", 0)
    # The learn hook: the stored game read from raw.db, the others fetched the usual way, nothing written
    store = Store(str(tmp_path / "learn.db"))
    clock.now = datetime.datetime(2026, 11, 1, 3, 0, tzinfo=UTC).timestamp()
    learn_report, asked = learn_backfill(world, store, TTLCache(store, name="learn", clock=clock), rig.raw)
    assert learn_report["stopped"] == "done" and rig.raw.payload_count() == rows
    summaries = sorted(urllib.parse.parse_qs(urllib.parse.urlsplit(u).query)["event"][0] for u in asked
                       if "/summary?" in u)
    assert summaries == sorted(LEARN_IDS[1:])
    block = capture_health(rig.raw.path, clock(), settings={"per_day": 3000})
    assert [a["text"] for a in block["alarms"] if a["kind"] == "disk"] == [
        "The disk is 92% full: 7.5 GB free. Capture and the backfill are paused until 9.3 GB is free."]
    # Room again: within RawStore.room()'s minute everything carries on
    disk.usage = ROOMY
    clock.now += raw_store.ROOM_EVERY + 1
    report = rig.capturer.tick()
    assert world.asked("polite") and report["backfill"]["requests"] > 0
    assert rig.capturer.last["sweep"]["boards"] == 58


def test_an_unexpected_error_on_one_board_costs_the_sweep_only_that_board(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever goes wrong with one league's board (here an error nobody expected), the sweep reads the other 57 and
    tries that one again a few times later, instead of starting over every minute."""
    clock = Clock(at(DAY + datetime.timedelta(days=1), 5, 30))
    world = World(clock)
    keys = recorded_day(world)
    rig = Rig(tmp_path, world, monkeypatch)
    real = EspnClient.scoreboard_raw
    tried: List[str] = []

    def scoreboard_raw(client: Any, league: str, day: datetime.date, *, http: Any = None) -> bytes:
        if league == NHL:
            tried.append(league)
            raise KeyError("an unexpected bug")
        return real(client, league, day, http=http)

    monkeypatch.setattr(EspnClient, "scoreboard_raw", scoreboard_raw)
    for _ in range(4 * 60):
        rig.capturer.tick()
        clock.now += 60.0
    assert len(tried) == worker_module.SWEEP_TRIES
    assert len({url for _t, d, _h, url in world.sent if d == "polite" and "/scoreboard?" in url}) == 57
    assert rig.raw.get_state("sweep:2026-10-30")["failed"] == [NHL]
    assert all(rig.raw.job(keys[sport]).state == "done" for sport in ("basketball", "football", "soccer", "mma"))


def test_the_backfill_never_runs_past_its_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker starts the backfill just before 06:00 (server time) with plenty left to do: its last request goes
    before 06:00, even in the middle of a chunk, and the next night carries on."""
    monkeypatch.setattr(bf, "_projected", True)
    for league in bf.ORDER:
        monkeypatch.setitem(bf.WINDOWS, league, WINDOW)
    clock = Clock(datetime.datetime(2026, 11, 1, 5, 59, 30, tzinfo=UTC).timestamp())  # a chunk takes ~50 s
    world = World(clock, seasons=True, latency=1.0)
    config = SimpleNamespace(capture_backfill=True, capture_backfill_seasons=3, capture_backfill_per_night=6000)
    rig = Rig(tmp_path, world, monkeypatch, config=config)
    rig.raw.set_state("sweep:last", "2026-10-31")
    for entry in plan.STATIC:
        rig.raw.set_state(f"static:{entry.name}", {"at": clock(), "ok": True,
                                                   "period": plan.static_period(entry, clock(), UTC)})
    closes = datetime.datetime(2026, 11, 1, 6, 0, tzinfo=UTC).timestamp()
    while clock() < closes + 600:
        rig.capturer.tick()
        clock.now += 60.0
    times = [ts for ts, _door, _h, _u in world.sent]
    assert times and closes - 5 < max(times) < closes and len(times) > 20  # it ran up to the window's end
    assert rig.raw.spent("backfill", "2026-11-01") == len(times)
    assert not all(row["done"] for row in rig.raw.progress())  # plenty left for the next night


def test_a_summary_error_is_labelled_a_game_summarys(tmp_path: Path) -> None:
    from telegram_betting_bot_simple.learn.progress import capture_health
    raw = RawStore(str(tmp_path / "raw.db"))
    now = at(DAY, 12)
    raw.put(game_key="espn:401888888", league=NHL, source="espn_site", endpoint="summary", params={},
            body=b'{"code": 503}', state="final", http_status=503, fetched_at=now - 60)
    errors = capture_health(raw.path, now)["errors"]
    assert [(e["what"], e["error"]) for e in errors] == [("ESPN game summaries", "HTTP 503")]


def test_the_cards_planned_seasons_follow_the_backfills_own_scope() -> None:
    """The card's "done of planned" for a league is the backfill's own scope (capture/backfill.seasons_for): between
    seasons there is no season being played, so 3 of 3 is all done (not 3 of 4 forever)."""
    from telegram_betting_bot_simple.learn.progress import _planned_seasons
    in_season = datetime.datetime(2026, 11, 1, 18, 0, tzinfo=UTC).timestamp()
    summer = datetime.datetime(2026, 8, 15, 18, 0, tzinfo=UTC).timestamp()
    assert _planned_seasons(NBA, 3, in_season) == 4 and _planned_seasons(NBA, 3, summer) == 3
    assert _planned_seasons(MLB, 3, datetime.datetime(2026, 12, 20, tzinfo=UTC).timestamp()) == 3
    assert _planned_seasons(MLB, 3, summer) == 4 and _planned_seasons(UFC, 2, summer) == 3
    assert _planned_seasons("soccer_zzz_other", 3, summer) == 4 and _planned_seasons(NBA, None, summer) is None


def test_the_static_files_and_the_backfills_hosts_are_keyed_as_decided(night: Any) -> None:
    """The coordinator's keys for the static units (the worker's weekly and monthly files, the backfill's NFL
    play-by-play), and the backfill's hosts: ESPN and nflverse's GitHub release only (never StatsAPI, whose terms are
    non-bulk, nor Greco1899, the worker's weekly job)."""
    with sqlite3.connect(night.raw.path) as db:
        keys = {key for (key,) in db.execute("SELECT DISTINCT game_key FROM raw_payloads WHERE game_key LIKE 'static:%' "
                                             "AND game_key NOT LIKE 'static:espn_site:scoreboard:%'")}
    from telegram_betting_bot_simple.capture.sources import football_data_raw, greco, mcp
    assert keys == ({f"static:nflverse:pbp:{season}" for season in (2024, 2025, 2026)} | {"static:nflverse:officials"}
                    | {f"static:greco1899:{name}" for name in greco.FILES} | {f"static:mcp:{name}" for name in mcp.FILES}
                    | {f"static:football_data:{code}:2627" for code in football_data_raw.CODES}
                    | {"static:football_data:USA"})
    boards = [url for _ts, who, url in night.world.tags if "/scoreboard?" in url]
    assert boards and all(len(urllib.parse.parse_qs(urllib.parse.urlsplit(u).query)["dates"][0]) == 8 for u in boards)
    backfill_hosts = {urllib.parse.urlsplit(url).hostname for _ts, who, url in night.world.tags if who == "backfill"}
    assert backfill_hosts == ESPN_HOSTS | {"github.com"}
