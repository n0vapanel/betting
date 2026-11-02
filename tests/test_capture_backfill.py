"""The backfill of older games into raw.db (v4 P1, Contract D: capture/backfill.py) and the learn backfill's raw-first
hook (learn/history.py, wired in learn/loop.py). Nothing here touches the network: the backfill's requests go through
the real PoliteHttp (its per-host token buckets, on a fake clock) to a fake ESPN world (``World``: three past seasons
and the one being played, built from the trimmed audit payloads in tests/fixtures/capture/), and the learn backfill's
to a fake transport over tests/fixtures/learn/."""

from __future__ import annotations

import collections
import copy
import datetime
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from telegram_betting_bot_simple.capture import backfill as bf
from telegram_betting_bot_simple.capture import plan, raw_store
from telegram_betting_bot_simple.capture.backfill import RawBackfill
from telegram_betting_bot_simple.capture.http import HttpResponse, PoliteHttp
from telegram_betting_bot_simple.capture.raw_store import RawStore
from telegram_betting_bot_simple.capture.sources import nflverse, scoreboard_key
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.espn_client import LEAGUES, SCOREBOARD_ONLY
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.learn import history_store as hs
from telegram_betting_bot_simple.learn.history import LEARN_BACKFILL_RUNNING, Backfill, HistoryReader
from telegram_betting_bot_simple.learn.sources.http import HistHttp

FIX = Path(__file__).parent / "fixtures" / "capture"
LEARN_FIX = Path(__file__).parent / "fixtures" / "learn"
UTC = datetime.timezone.utc
NOW = datetime.datetime(2026, 11, 1, 2, 0, tzinfo=UTC).timestamp()  # a Sunday, 02:00 server time (UTC here)
NBA, NFL, EPL, UCL, NHL, UFC, MLB = ("basketball_nba", "americanfootball_nfl", "soccer_epl",
                                    "soccer_uefa_champs_league", "icehockey_nhl", "mma_mixed_martial_arts",
                                    "baseball_mlb")
ESPN_HOSTS = {"site.api.espn.com", "sports.core.api.espn.com", "site.web.api.espn.com"}


DiskUsage = collections.namedtuple("DiskUsage", "total used free")


@pytest.fixture(autouse=True)
def no_learn_backfill() -> Any:
    """The learn backfill's running flag (a process-wide Event) is clear before and after each test."""
    LEARN_BACKFILL_RUNNING.clear()
    yield
    LEARN_BACKFILL_RUNNING.clear()


@pytest.fixture(autouse=True)
def healthy_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """raw.db's disk has plenty of room, whatever the machine running the tests has (the disk-floor tests say
    otherwise)."""
    monkeypatch.setattr(raw_store.shutil, "disk_usage", lambda path: DiskUsage(100e9, 10e9, 90e9))


def at(*parts: int) -> float:
    return datetime.datetime(*parts, tzinfo=UTC).timestamp()


def load(name: str) -> Any:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


class Clock:
    """A fake clock (epoch seconds): PoliteHttp's waits for a token move it on."""

    def __init__(self, ts: float) -> None:
        self.ts = float(ts)

    def __call__(self) -> float:
        return self.ts

    def sleep(self, seconds: float) -> None:
        self.ts += max(0.0, float(seconds))


# --- the fake ESPN world -------------------------------------------------------------------------------------------

PATHS = {path: league for league, path in {**LEAGUES, **SCOREBOARD_ONLY}.items()}
CODE = {league: 10 + n for n, league in enumerate(bf.ORDER)}
FINAL = {"id": "3", "name": "STATUS_FINAL", "state": "post", "completed": True, "description": "Final"}
POSTPONED = {"id": "6", "name": "STATUS_POSTPONED", "state": "post", "completed": False, "description": "Postponed"}
BOARDS = {"basketball": "sb_nba.json", "football": "sb_nfl.json", "soccer": "sb_epl.json", "hockey": "sb_nhl.json",
          "baseball": "sb_mlb.json", "mma": "sb_ufc.json", "tennis": "sb_atp.json"}
SUMMARIES = {"basketball": "sum_nba.json", "football": "sum_nfl.json", "soccer": "sum_epl.json",
             "hockey": "sum_nhl.json", "baseball": "sum_mlb.json"}
PLAY_PAGES = 2


def sport(league: str) -> str:
    return {**LEAGUES, **SCOREBOARD_ONLY}[league][0]


def is_card_day(day: datetime.date) -> bool:
    return day.weekday() == 5 and (day.toordinal() // 7) % 2 == 0  # every other Saturday


def games_on(league: str, day: datetime.date) -> int:
    weekday = day.weekday()
    if league == NBA:
        return 2 if weekday in (1, 4, 6) else 0
    if league == NFL:
        return {6: 2, 0: 1}.get(weekday, 0)
    if league == UCL:
        return 1 if weekday == 1 else 0
    if league.startswith("soccer_"):
        return 2 if weekday == 5 else 0
    if league == NHL:
        return 1 if weekday in (2, 5) else 0
    if league == MLB:
        return 1
    if league == UFC:
        return 1 if is_card_day(day) else 0
    if league in plan.TENNIS:
        return 3 if weekday < 6 else 0  # singles matches every day but Sunday
    return 0


def event_ids(league: str, day: datetime.date, count: Optional[int] = None) -> List[str]:
    return [f"{CODE[league]}{day:%Y%m%d}{n}" for n in range(games_on(league, day) if count is None else count)]


def espn_time(day: datetime.date, hour: int = 7) -> str:
    """A day as ESPN's calendars write it: midnight Pacific in UTC (07:00Z in summer, 08:00Z in winter)."""
    return f"{day:%Y-%m-%d}T{hour:02d}:00Z"


def ufc_calendar(year: int) -> List[Dict[str, Any]]:
    """A season's cards as ESPN lists them: the start in UTC is 01:00 the next day (21:00 or 20:00 US Eastern)."""
    day, entries = datetime.date(year, 1, 1), []
    while day.year == year:
        if is_card_day(day):
            card = event_ids(UFC, day)[0]
            entries.append({"label": f"UFC Fight Night {day}",
                            "startDate": f"{day + datetime.timedelta(days=1)}T01:00Z",
                            "endDate": f"{day + datetime.timedelta(days=1)}T07:59Z",
                            "event": {"$ref": f"http://sports.core.api.espn.pvt/v2/sports/mma/leagues/ufc/events/"
                                              f"{card}?lang=en&region=us"}})
        day += datetime.timedelta(days=1)
    return entries


class World:
    """ESPN's site, core and fightcenter APIs and nflverse's GitHub releases, faked for PoliteHttp._send: boards for
    any day (games on fixed weekdays, every game of a past day final), each game's summary, two pages of core plays,
    a UFC card every other Saturday with its fightcenter and the season's calendar on every UFC board, and a
    play-by-play file per NFL season, a tennis day's singles matches in one tournament. ``calls`` records (kind,
    league, what, host, time); ``fail`` maps (kind, what) to answers given before the real one (a status, or "reset"
    for a dropped connection); ``postponed`` holds event ids whose games were postponed; ``schedule`` maps a league
    to its games on a day (instead of games_on) and ``calendars`` to its board's season and calendar on a day
    (merged into ``leagues[0]``; None: a board with neither)."""

    def __init__(self, clock: Clock, per_request: float = 0.0) -> None:
        self.clock = clock
        self.per_request = per_request
        self.calls: List[Tuple[str, str, str, str, float]] = []
        self.fail: Dict[Tuple[str, str], List[Any]] = {}
        self.fail_always: Dict[str, int] = {}
        self.outage: Dict[Tuple[str, str], float] = {}  # (kind, what) -> answers 503 until this time
        self.postponed: set = set()
        self.schedule: Dict[str, Callable[[datetime.date], int]] = {}
        self.calendars: Dict[str, Callable[[datetime.date], Optional[Dict[str, Any]]]] = {}
        self.boards = {name: load(file) for name, file in BOARDS.items()}
        self.summaries = {name: (FIX / file).read_bytes() for name, file in SUMMARIES.items()}
        plays = load("core_epl_plays.json")
        self.pages = {n: json.dumps({**plays, "pageIndex": n, "pageCount": PLAY_PAGES}).encode()
                      for n in range(1, PLAY_PAGES + 1)}
        self.fightcenter = (FIX / "syn_fightcenter.json").read_bytes()
        self.pbp = (FIX / "syn_play_by_play_2024.csv.gz").read_bytes()

    def send(self, request: Any) -> HttpResponse:
        parts = urllib.parse.urlsplit(request.full_url)
        query = dict(urllib.parse.parse_qsl(parts.query))
        host, seg = parts.hostname or "", parts.path.split("/")
        kind, league, what = "other", "", parts.path
        if host == "site.api.espn.com" and len(seg) == 8:
            league, kind = PATHS.get((seg[5], seg[6]), ""), seg[7]
            what = query.get("dates", "") if kind == "scoreboard" else query.get("event", "")
            kind = "board" if kind == "scoreboard" else kind
        elif host == "sports.core.api.espn.com" and len(seg) == 11 and seg[10] == "plays":
            league, kind, what = PATHS.get((seg[3], seg[5]), ""), "plays", f"{seg[7]}:{query.get('page')}"
        elif host == "site.web.api.espn.com" and "fightcenter" in seg:
            league, kind, what = UFC, "fightcenter", seg[-1]
        elif host == "github.com" and parts.path.endswith(".csv.gz"):
            league, kind, what = NFL, "pbp", parts.path.rsplit("_", 1)[-1].split(".")[0]
        self.calls.append((kind, league, what, host, self.clock()))
        self.clock.ts += self.per_request
        planned = self.fail.get((kind, what))
        if planned:
            answer = planned.pop(0)
            if answer == "reset":
                raise ConnectionResetError("connection reset by peer")
            return HttpResponse(int(answer), b'{"code":%d}' % int(answer), {})
        if kind in self.fail_always:
            return HttpResponse(self.fail_always[kind], b'{"code":404,"message":"Not Found"}', {})
        if self.clock() < self.outage.get((kind, what), -1.0):
            return HttpResponse(503, b'{"code":503,"message":"Service Unavailable"}', {})
        if kind == "board":
            return HttpResponse(200, self.board(league, what), {})
        if kind == "summary" and league != UFC:
            return HttpResponse(200, self.summaries[sport(league)], {})
        if kind == "plays":
            page = int(what.split(":")[1])
            return HttpResponse(200, self.pages[page], {}) if page in self.pages else HttpResponse(404, b"{}", {})
        if kind == "fightcenter":
            return HttpResponse(200, self.fightcenter, {})
        if kind == "pbp":
            return HttpResponse(200, self.pbp, {})
        return HttpResponse(404, b'{"code":404}', {})

    def board(self, league: str, dates: str) -> bytes:
        day = datetime.datetime.strptime(dates, "%Y%m%d").date()
        data = copy.deepcopy(self.boards[sport(league)])
        ids = event_ids(league, day, self.schedule[league](day) if league in self.schedule else None)
        if league in self.calendars:
            for key in ("calendar", "calendarType", "calendarIsWhitelist", "calendarStartDate", "calendarEndDate"):
                data["leagues"][0].pop(key, None)
            data["leagues"][0]["season"] = {"year": day.year}
            data["leagues"][0].update(self.calendars[league](day) or {})
        if sport(league) == "tennis":
            return self.tennis_board(data, day, ids)
        proto = data["events"][0]
        events = []
        for event_id in ids:
            event = copy.deepcopy(proto)
            status = POSTPONED if event_id in self.postponed else FINAL
            event.update({"id": event_id, "date": f"{day}T23:00Z"})
            competition = event["competitions"][0]
            competition.update({"id": event_id, "date": f"{day}T23:00Z"})
            competition["status"]["type"] = dict(status)
            if league == UFC:
                event["status"]["type"] = dict(status)
            events.append(event)
        data["events"] = events
        data["leagues"][0]["season"]["year"] = day.year
        if league == UFC:
            data["leagues"][0]["calendar"] = ufc_calendar(day.year)
        return json.dumps(data).encode()

    @staticmethod
    def tennis_board(data: Dict[str, Any], day: datetime.date, ids: List[str]) -> bytes:
        """A tennis day (sb_atp.json's shape): one tournament, its singles matches that day, every one final."""
        event = data["events"][0]
        singles = next(g for g in event["groupings"] if g["grouping"]["slug"] == "mens-singles")
        proto = singles["competitions"][0]
        singles["competitions"] = []
        for match_id in ids:
            match = copy.deepcopy(proto)
            match.update({"id": match_id, "date": f"{day}T15:00Z", "startDate": f"{day}T15:00Z"})
            match["status"]["type"] = dict(FINAL)
            singles["competitions"].append(match)
        event["groupings"] = [singles]
        data["events"] = [event] if ids else []
        return json.dumps(data).encode()

    # -- what was asked --

    def of(self, kind: str, league: Optional[str] = None) -> List[Tuple[str, str, str, str, float]]:
        return [call for call in self.calls if call[0] == kind and (league is None or call[1] == league)]


class Env:
    """One raw.db and one fake world; ``backfill()`` is a fresh RawBackfill over a fresh PoliteHttp each time (as
    after a restart)."""

    def __init__(self, tmp_path: Path, leagues: List[str], *, ts: float = NOW, per_request: float = 0.0,
                 name: str = "raw.db", **options: Any) -> None:
        self.clock = Clock(ts)
        self.world = World(self.clock, per_request)
        self.raw = RawStore(str(tmp_path / name), clock=self.clock)
        self.leagues = leagues
        self.options = options
        self.http_made: List[PoliteHttp] = []

    def http(self) -> PoliteHttp:
        http = PoliteHttp(clock=self.clock, sleep=self.clock.sleep)
        http._send = self.world.send  # type: ignore[method-assign]
        self.http_made.append(http)
        return http

    def backfill(self, **extra: Any) -> RawBackfill:
        options = {"leagues": self.leagues, "tz": UTC, **self.options, **extra}
        return RawBackfill(self.raw, self.http(), self.clock, **options)

    def rows(self) -> List[Tuple[Any, ...]]:
        db = sqlite3.connect(self.raw.path)
        try:
            return sorted(db.execute("SELECT game_key, league, source, endpoint, params, state, horizon, http_status, "
                                     "sha256 FROM raw_payloads").fetchall())
        finally:
            db.close()


@pytest.fixture
def short(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Shorter season windows for the leagues given (the real ones are tested on their own)."""

    def apply(window: Tuple[Tuple[int, int], Tuple[int, int]], *leagues: str) -> None:
        for league in leagues or bf.ORDER:
            monkeypatch.setitem(bf.WINDOWS, league, window)
    return apply


def days(first: datetime.date, last: datetime.date) -> List[str]:
    return [f"{first + datetime.timedelta(days=n):%Y%m%d}" for n in range((last - first).days + 1)]


# --- season windows and scope ------------------------------------------------------------------------------------

def test_the_season_windows_follow_the_real_calendars() -> None:
    assert bf.ORDER[:15] == bf.WAVES_1_2 == (
        NBA, NFL, "soccer_epl", "soccer_efl_champ", "soccer_spain_la_liga", "soccer_germany_bundesliga",
        "soccer_italy_serie_a", "soccer_france_ligue_one", "soccer_spl", "soccer_netherlands_eredivisie",
        "soccer_portugal_primeira_liga", UCL, NHL, UFC, MLB)
    assert set(bf.WINDOWS) == set(bf.ORDER)

    def inside(league: str, season: int, *moments: Tuple[int, int, int]) -> None:
        first, last = bf.season_dates(league, season)
        for moment in moments:
            assert first <= datetime.date(*moment) <= last, (league, season, moment)

    # The learn backfill's windows for the NBA, NHL and MLB (learn/sources/espn_hist.LEAGUES)
    assert bf.season_dates(NBA, 2025) == (datetime.date(2025, 10, 1), datetime.date(2026, 6, 30))
    assert bf.season_dates(MLB, 2025) == (datetime.date(2025, 3, 15), datetime.date(2025, 11, 15))
    inside(NBA, 2025, (2025, 10, 21), (2026, 6, 19))          # opening night; a Finals game in June
    inside(NHL, 2025, (2025, 10, 7), (2026, 6, 15))
    inside(MLB, 2025, (2025, 3, 18), (2025, 11, 1))           # the Tokyo Series; World Series game 7
    # NFL: Week 1 after Labor Day to the Super Bowl (LX: 8 February 2026)
    assert bf.season_dates(NFL, 2025) == (datetime.date(2025, 9, 1), datetime.date(2026, 2, 20))
    inside(NFL, 2025, (2025, 9, 4), (2026, 1, 4), (2026, 2, 8))
    inside(NFL, 2024, (2024, 9, 5), (2025, 2, 9))
    # European soccer August to May (the Scottish Premiership opens first), the Champions League September to June
    inside("soccer_epl", 2025, (2025, 8, 15), (2026, 5, 24))
    inside("soccer_spl", 2025, (2025, 8, 2), (2026, 5, 17))
    inside("soccer_efl_champ", 2025, (2025, 8, 8), (2026, 5, 23))  # the play-off final at Wembley
    inside(UCL, 2025, (2025, 9, 16), (2026, 5, 30))
    inside(UCL, 2023, (2023, 9, 19), (2024, 6, 1))
    # UFC: the calendar year
    assert bf.season_dates(UFC, 2025) == (datetime.date(2025, 1, 1), datetime.date(2025, 12, 31))


def test_the_scope_is_the_season_being_played_and_the_last_three_completed() -> None:
    settled = bf.settled_day(NOW)
    assert settled == datetime.date(2026, 10, 30)  # 36 hours before 22:00 US Eastern on 31 October
    nba = bf.seasons_for(NBA, settled, 3)
    assert nba == [(2026, datetime.date(2026, 10, 1), settled, True),
                   (2025, datetime.date(2025, 10, 1), datetime.date(2026, 6, 30), False),
                   (2024, datetime.date(2024, 10, 1), datetime.date(2025, 6, 30), False),
                   (2023, datetime.date(2023, 10, 1), datetime.date(2024, 6, 30), False)]
    assert [s[0] for s in bf.seasons_for(MLB, settled, 3)] == [2026, 2025, 2024, 2023]  # 2026 runs to 15 Nov
    assert bf.seasons_for(UFC, settled, 3)[0] == (2026, datetime.date(2026, 1, 1), settled, True)
    # Off-season: no season being played, the last three completed
    july = datetime.date(2026, 7, 15)
    assert [(s[0], s[3]) for s in bf.seasons_for(NBA, july, 3)] == [(2025, False), (2024, False), (2023, False)]
    assert [(s[0], s[3]) for s in bf.seasons_for(NFL, datetime.date(2027, 3, 1), 3)] == \
        [(2026, False), (2025, False), (2024, False)]
    # The Super Bowl's week: still the 2026 season being played
    assert bf.seasons_for(NFL, datetime.date(2027, 2, 8), 1)[0] == (2026, datetime.date(2026, 9, 1),
                                                                     datetime.date(2027, 2, 8), True)
    # Newest season first, leagues in wave order
    env_units = RawBackfill(SimpleNamespace(), PoliteHttp(), lambda: NOW).units()
    assert [(u.tier, u.rank) for u in env_units] == sorted((u.tier, u.rank) for u in env_units)
    assert [(u.league, u.season) for u in env_units[:3]] == [(NBA, 2026), (NFL, 2026), ("soccer_epl", 2026)]
    assert [(u.league, u.season) for u in env_units[58:60]] == [(NBA, 2025), (NFL, 2025)]  # after every newest one
    assert len([u for u in env_units if u.league in bf.WAVES_1_2]) == 15 * 4
    assert {sum(u.league == league for u in env_units) for league in bf.ORDER} == {3, 4}  # 3: none being played
    with pytest.raises(ValueError):  # a tennis tournament's own key is not one of the 58 paths (its tour is)
        RawBackfill(SimpleNamespace(), PoliteHttp(), lambda: NOW, leagues=["tennis_atp_paris_masters"])


def test_the_projected_total_is_logged_once(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
                                            tmp_path: Path) -> None:
    monkeypatch.setattr(bf, "_projected", False)
    raw = RawStore(str(tmp_path / "raw.db"))
    with caplog.at_level(logging.INFO, logger=bf.__name__):
        first = RawBackfill(raw, PoliteHttp(), lambda: NOW)
        RawBackfill(raw, PoliteHttp(), lambda: NOW)
    lines = [r.getMessage() for r in caplog.records if "projected" in r.getMessage()]
    assert len(lines) == 1 and "requests" in lines[0] and "GB" in lines[0]
    found = first.projection()
    # Waves 1-2: about 85,000 requests of summaries and core plays plus the one-day boards (~14,000), about 4 GB
    early = [found["waves"][1], found["waves"][2]]
    requests = sum(wave["requests"] for wave in early)
    assert 90_000 <= requests <= 110_000 and 3.5 <= sum(wave["gb"] for wave in early) <= 4.5, found
    assert set(found["leagues"]) == set(bf.ORDER)
    assert all(entry["seasons"] == 4 for league, entry in found["leagues"].items() if league in bf.WAVES_1_2)
    assert 15 <= sum(wave["nights"] for wave in early) <= 19  # at 6,000 a night
    soccer = sum(found["leagues"][league]["requests"] for league in bf.PLAYS_LEAGUES)
    assert soccer > requests / 2  # core plays are most of it
    assert str(f"{found['requests']:,}") in lines[0]


# --- wave 3: every other ESPN path (v4 P1b) -----------------------------------------------------------------------

MLS, ATP, WTA = "soccer_usa_mls", "tennis_atp", "tennis_wta"
EASTERN = bf.EASTERN


def test_the_order_lists_every_one_of_the_58_paths_once_wave_3_after_waves_1_and_2() -> None:
    assert bf.ORDER == bf.WAVES_1_2 + bf.WAVE3 and bf.WAVES_1_2 == plan.WAVE1 + plan.WAVE2
    assert bf.WAVE3 == (
        "basketball_ncaab", "americanfootball_ncaaf", "basketball_wnba", "soccer_uefa_europa_league", MLS,
        "soccer_mexico_ligamx", "soccer_brazil_campeonato", "soccer_argentina_primera_division",
        "soccer_turkey_super_league", "soccer_belgium_first_div", "soccer_uefa_europa_conference_league",
        "soccer_conmebol_copa_libertadores", "soccer_saudi_arabia_pro_league", "soccer_japan_j_league",
        "soccer_australia_aleague", "soccer_austria_bundesliga", "soccer_switzerland_superleague",
        "soccer_denmark_superliga", "soccer_greece_super_league", "soccer_norway_eliteserien",
        "soccer_sweden_allsvenskan", "soccer_russia_premier_league", "soccer_chile_campeonato",
        "soccer_spain_segunda_division", "soccer_italy_serie_b", "soccer_germany_bundesliga2",
        "soccer_france_ligue_two", "soccer_england_league1", "soccer_england_league2", "soccer_brazil_serie_b",
        "soccer_conmebol_copa_sudamericana", "soccer_england_efl_cup", "soccer_germany_dfb_pokal",
        "soccer_uefa_nations_league", "soccer_uefa_champs_league_women", "soccer_sweden_superettan",
        "soccer_finland_veikkausliiga", "soccer_league_of_ireland", "basketball_wncaab", "americanfootball_cfl",
        "basketball_nbl", ATP, WTA)
    assert len(bf.ORDER) == len(set(bf.ORDER)) == len(plan.PATHS) == 58 and sorted(bf.ORDER) == sorted(plan.PATHS)
    assert [bf.wave_of(league) for league in bf.ORDER] == [1] * 12 + [2] * 3 + [3] * 43
    assert set(bf.WINDOWS) == set(bf.GAMES) == set(bf.ORDER) and set(bf.PLAIN) == set(bf.CALENDAR_LEAGUES) == \
        set(bf.WAVE3)
    # The walk takes every league's newest season in that order, and the default scope is all 58
    backfill = RawBackfill(SimpleNamespace(), PoliteHttp(), lambda: NOW)
    assert backfill.leagues == bf.ORDER and [u.league for u in backfill.units() if u.tier == 0] == list(bf.ORDER)


def test_wave_3s_tight_windows_hold_its_real_seasons() -> None:
    def inside(league: str, season: int, *moments: Tuple[int, int, int]) -> None:
        first, last = bf.season_dates(league, season)
        for moment in moments:
            assert first <= datetime.date(*moment) <= last, (league, season, moment)

    inside("basketball_ncaab", 2025, (2025, 11, 3), (2026, 4, 6))   # opening night; the title game
    inside("basketball_wncaab", 2025, (2025, 11, 3), (2026, 4, 5))
    inside("americanfootball_ncaaf", 2025, (2025, 8, 23), (2026, 1, 19))  # Week 0; the CFP final
    inside("basketball_wnba", 2025, (2025, 5, 16), (2025, 10, 10))  # opening night; the Finals' last game
    inside("americanfootball_cfl", 2025, (2025, 6, 5), (2025, 11, 16))  # Week 1; the Grey Cup
    inside("basketball_nbl", 2025, (2025, 9, 18), (2026, 3, 15))
    inside(MLS, 2025, (2025, 2, 22), (2025, 12, 6))                  # opening day; MLS Cup
    inside("soccer_brazil_campeonato", 2026, (2026, 1, 28))          # 2026 opened in January
    inside("soccer_brazil_campeonato", 2025, (2025, 3, 29), (2025, 12, 7))
    inside("soccer_japan_j_league", 2025, (2025, 2, 14), (2025, 12, 6))
    inside("soccer_mexico_ligamx", 2023, (2023, 6, 30), (2024, 5, 26))  # an Apertura from 30 June
    inside("soccer_uefa_europa_conference_league", 2022, (2023, 6, 7))
    inside("soccer_uefa_nations_league", 2022, (2023, 6, 18))       # the 2023 final
    inside("soccer_spain_segunda_division", 2024, (2024, 8, 15), (2025, 6, 22))  # the play-off final
    inside("soccer_australia_aleague", 2022, (2022, 10, 7), (2023, 6, 3))  # the 2023 grand final
    inside("soccer_conmebol_copa_libertadores", 2025, (2025, 2, 5), (2025, 11, 29))
    inside(ATP, 2025, (2025, 1, 1), (2025, 11, 16))                  # the ATP Finals' final
    # Tight: never longer than the plain window of its kind (calendar year, August-May, ...) by more than the July
    # openers and June finals the plain August-May one would miss, and together much shorter
    tight = plain = 0
    for league in bf.WAVE3:
        first, last = bf.season_dates(league, 2025)
        p_first, p_last = bf._window_days(bf.PLAIN[league], 2025)
        tight, plain = tight + (last - first).days + 1, plain + (p_last - p_first).days + 1
        assert (last - first).days <= (p_last - p_first).days + 40, league
    assert tight < plain - 1000


def test_espns_calendar_on_the_audits_boards() -> None:
    day = datetime.date
    # Soccer: a whitelist of match days inside a June-June season (sb_epl.json: 2026-27 opens on 21 August)
    assert bf.calendar_bound(load("sb_epl.json")) == (day(2026, 6, 1), day(2027, 5, 31), day(2026, 8, 21),
                                                      day(2027, 5, 30))
    # Basketball: the days with games so far (the NBA's 2026-27 regular season: its play-offs aren't listed yet)
    assert bf.calendar_bound(load("sb_nba.json")) == (day(2026, 9, 30), day(2027, 6, 26), day(2026, 10, 3),
                                                      day(2027, 4, 11))
    # Football: a list of season parts with their dates, the off season left out
    nfl = bf.calendar_bound(load("sb_nfl.json"))
    assert nfl is not None and nfl[:3] == (day(2026, 8, 6), day(2027, 2, 16), day(2026, 8, 6))
    assert day(2027, 2, 7) <= nfl[3] <= day(2027, 2, 16)
    through_summer = load("sb_nfl.json")  # a season whose dates run on through its off season: still no game day
    through_summer["leagues"][0]["season"]["endDate"] = "2027-08-01T06:59Z"
    assert bf.calendar_bound(through_summer)[1:] == (day(2027, 8, 1), day(2026, 8, 6), nfl[3])
    # A calendar of days without games (MLB's is no whitelist): the season's own dates
    assert bf.calendar_bound(load("sb_mlb.json")) == (day(2026, 2, 19), day(2026, 11, 12), day(2026, 2, 19),
                                                      day(2026, 11, 12))
    assert bf.calendar_bound(load("sb_atp.json"))[2:] == (day(2026, 1, 1), day(2026, 11, 14))
    assert bf.calendar_bound({"leagues": [{"calendar": ["2026-08-21T07:00Z"]}]}) is None  # no season dates
    assert bf.calendar_bound({"events": []}) is None and bf.calendar_bound(None) is None


def test_the_calendar_trims_only_days_it_shows_have_no_game() -> None:
    d = datetime.date
    season = (d(2025, 1, 1), d(2025, 12, 31), d(2025, 2, 22), d(2025, 12, 6))
    assert bf.bounded_span(d(2025, 2, 15), d(2025, 12, 15), season) == (d(2025, 2, 22), d(2025, 12, 6))
    iso = [part.isoformat() for part in season]  # as a cursor keeps it
    assert bf.bounded_span(d(2025, 2, 15), d(2025, 12, 15), iso) == (d(2025, 2, 22), d(2025, 12, 6))
    # A season still being played: the tail is the newest settled day, whatever the calendar listed then
    assert bf.bounded_span(d(2025, 2, 15), d(2025, 12, 15), season, tail=False) == (d(2025, 2, 22), d(2025, 12, 15))
    # Days outside the calendar's season are another season's: kept
    autumn = (d(2025, 8, 1), d(2026, 1, 31), d(2025, 8, 9), d(2026, 1, 20))  # an Apertura-only season
    assert bf.bounded_span(d(2025, 6, 25), d(2026, 5, 31), autumn) == (d(2025, 6, 25), d(2026, 5, 31))
    assert bf.bounded_span(d(2025, 8, 1), d(2026, 5, 31), autumn) == (d(2025, 8, 9), d(2026, 5, 31))
    # A window wholly without games is empty; a bound that can't be read leaves the window
    first, last = bf.bounded_span(d(2025, 12, 10), d(2025, 12, 15), season)
    assert last < first
    assert bf.bounded_span(d(2025, 2, 15), d(2025, 12, 15), ["junk"]) == (d(2025, 2, 15), d(2025, 12, 15))


def utc_text(day: datetime.date, hour: int, minute: int = 0) -> str:
    moment = datetime.datetime.combine(day, datetime.time(hour, minute), EASTERN).astimezone(UTC)
    return moment.strftime("%Y-%m-%dT%H:%MZ")


def whitelist(first: datetime.date, last: datetime.date, game_days: List[datetime.date]) -> Dict[str, Any]:
    """A board's ``leagues[0]`` season (midnight to 23:59 US Eastern) and its days with games, as soccer's board
    carries them (sb_epl.json)."""
    return {"season": {"startDate": utc_text(first, 0), "endDate": utc_text(last, 23, 59)}, "calendarType": "day",
            "calendarIsWhitelist": True, "calendar": [espn_time(day) for day in sorted(game_days)]}


def mls_autumn(env: Env) -> Callable[[datetime.date], List[datetime.date]]:
    """MLS in the fake world: two games every Saturday from 10 October to 15 November, a calendar-year season whose
    calendar lists those Saturdays."""
    def saturdays(year: int) -> List[datetime.date]:
        day, out = datetime.date(year, 10, 10), []
        while day <= datetime.date(year, 11, 15):
            out += [day] if day.weekday() == 5 else []
            day += datetime.timedelta(days=1)
        return out

    env.world.schedule[MLS] = lambda day: 2 if day in saturdays(day.year) else 0
    env.world.calendars[MLS] = lambda day: whitelist(datetime.date(day.year, 1, 1), datetime.date(day.year, 12, 31),
                                                     saturdays(day.year))
    return lambda day: event_ids(MLS, day, env.world.schedule[MLS](day))


def test_a_wave_3_season_reads_only_the_days_its_espn_calendar_holds(tmp_path: Path, short: Any) -> None:
    short(((10, 1), (11, 30)), MLS)  # 2026: 1-30 October so far; 2025: 1 October - 30 November
    env = Env(tmp_path, [MLS], seasons=1)
    games = mls_autumn(env)
    report = env.backfill().run_night()
    assert report["stopped"] == "done", report
    boards = [what for kind, _league, what, _host, _ts in env.world.calls if kind == "board"]
    d = datetime.date
    # 2026, being played: the middle day's board first (its calendar), then 10-30 October newest first: the days
    # before the first Saturday with games are left out, the newest settled day is still the top
    first_season = ["20261015"] + [day for day in days(d(2026, 10, 10), d(2026, 10, 30))[::-1] if day != "20261015"]
    # 2025, completed: its middle day's board, then only its Saturdays' span, 11 October - 15 November
    second = ["20251031"] + [day for day in days(d(2025, 10, 11), d(2025, 11, 15))[::-1] if day != "20251031"]
    assert boards == first_season + second  # each board once: the probe's comes back from raw.db
    assert len(boards) == 21 + 36 < 30 + 61  # 34 empty days not paid for
    summaries = sorted(what for kind, _l, what, _h, _t in env.world.calls if kind == "summary")
    expected = [i for first, last in ((d(2026, 10, 1), d(2026, 10, 30)), (d(2025, 10, 1), d(2025, 11, 30)))
                for n in range((last - first).days + 1) for i in games(first + datetime.timedelta(days=n))]
    assert summaries == sorted(expected) and len(expected) == 6 + 12  # every game, none twice
    progress = {row["season"]: row for row in env.raw.progress()}
    assert progress[2025]["done"] is True and progress[2026]["done"] is False
    assert progress[2025]["games_stored"] == 12 and progress[2026]["games_stored"] == 6
    assert progress[2025]["cursor"]["bound"] == ["2025-01-01", "2025-12-31", "2025-10-11", "2025-11-15"]
    assert progress[2025]["cursor"]["bound_current"] is False and progress[2026]["cursor"]["bound_current"] is True
    # The next night: 31 October has settled (a Saturday with games): read, then nothing more
    env.clock.ts = at(2026, 11, 2, 2, 0)
    calls = len(env.world.calls)
    assert env.backfill().run_night()["stopped"] == "done"
    assert [call[2] for call in env.world.calls[calls:]] == ["20261031"] + games(d(2026, 10, 31))


def test_without_a_calendar_a_wave_3_season_keeps_its_tight_window(tmp_path: Path, short: Any) -> None:
    short(((10, 1), (10, 20)), MLS)  # 2026: 1-20 October, completed
    env = Env(tmp_path, [MLS], seasons=1)
    env.world.calendars[MLS] = lambda day: None  # a board with neither season dates nor a calendar
    assert env.backfill().run_night()["stopped"] == "done"
    boards = [what for kind, _l, what, _h, _t in env.world.calls if kind == "board"]
    every = days(datetime.date(2026, 10, 1), datetime.date(2026, 10, 20))
    assert boards[0] == "20261010" and sorted(boards) == every  # the probe, then the whole window, each day once
    assert env.raw.progress()[0]["cursor"]["bound"] == [] and env.raw.progress()[0]["done"] is True


def test_a_calendar_board_espn_refuses_leaves_the_window_and_an_outage_waits(tmp_path: Path, short: Any) -> None:
    short(((10, 1), (11, 30)), MLS)
    env = Env(tmp_path, [MLS], seasons=1, name="refused.db")
    mls_autumn(env)
    env.world.fail[("board", "20261015")] = [404] * bf.ITEM_TRIES  # the 2026 probe: ESPN refuses it three times
    env.world.fail[("board", "20251031")] = [503]                    # the 2025 probe: an outage, once
    first = env.backfill().run_night()
    assert first["stopped"] == "backoff" and first["deferred"] == 1, first
    boards = [what for kind, _l, what, _h, _t in env.world.calls if kind == "board"]
    assert boards[:3] == ["20261015"] * 3 and sorted(boards[3:-1]) == days(datetime.date(2026, 10, 1),
                                                                           datetime.date(2026, 10, 30))
    assert boards[-1] == "20251031"  # 2026 read its whole window; 2025 waits for its calendar
    progress = {row["season"]: row for row in env.raw.progress()}
    assert progress[2026]["cursor"]["bound"] == [] and "bound" not in progress[2025]["cursor"]
    assert progress[2025]["done"] is False and progress[2025]["cursor"]["bound_wait"][1] == 1
    env.clock.ts += bf.RETRY_WAIT + 60  # the wait is over: the calendar is read, then only its span
    calls = len(env.world.calls)
    assert env.backfill().run_night()["stopped"] == "done"
    again = [what for kind, _l, what, _h, _t in env.world.calls[calls:] if kind == "board"]
    assert again == ["20251031"] + [day for day in days(datetime.date(2025, 10, 11), datetime.date(2025, 11, 15))[::-1]
                                    if day != "20251031"]
    # An outage that outlasts PROBE_WAITS waits leaves the plain window too
    env2 = Env(tmp_path, [MLS], seasons=1, name="outage.db")
    mls_autumn(env2)
    env2.world.fail[("board", "20251031")] = [503] * bf.PROBE_WAITS
    for _ in range(bf.PROBE_WAITS):  # a night each: every wait (at most RETRY_WAIT_CAP) is over by the next
        env2.backfill().run_night()
        env2.clock.ts += 86400.0
    cursor = {row["season"]: row for row in env2.raw.progress()}[2025]["cursor"]
    assert cursor["bound"] == [] and "bound_wait" not in cursor


def test_tennis_reads_day_boards_only_and_counts_their_matches_from_them(tmp_path: Path, short: Any) -> None:
    short(((10, 20), (11, 5)), ATP, WTA)  # 2026: 20-30 October so far; 2025: 20 October - 5 November
    env = Env(tmp_path, [ATP, WTA], seasons=1)
    report = env.backfill().run_night()
    assert report["stopped"] == "done", report
    assert {call[0] for call in env.world.calls} == {"board"}  # no summary, no core page: no per-match fetch
    spans = ((datetime.date(2026, 10, 20), datetime.date(2026, 10, 30)),
             (datetime.date(2025, 10, 20), datetime.date(2025, 11, 5)))
    for league in (ATP, WTA):
        read = [what for kind, lg, what, _h, _t in env.world.calls if lg == league]
        assert sorted(read) == sorted(days(*spans[0]) + days(*spans[1])), league  # each day once
    assert {row[2:4] for row in env.rows()} == {("espn_site", "scoreboard")}
    assert {row[0] for row in env.rows()} == {scoreboard_key(league, what) for _k, league, what, _h, _t
                                              in env.world.calls}
    # The day's board is its matches' record: every finished singles match on it is seen and stored
    for row in env.raw.progress():
        first, last = spans[0] if row["season"] == 2026 else spans[1]
        matches = sum(games_on(row["league"], first + datetime.timedelta(days=n))
                      for n in range((last - first).days + 1))
        assert row["games_seen"] == row["games_stored"] == matches > 0, row
        assert row["done"] is (row["season"] == 2025)


def test_the_projection_counts_each_wave_and_logs_them_once(monkeypatch: pytest.MonkeyPatch,
                                                            caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    monkeypatch.setattr(bf, "_projected", False)
    raw = RawStore(str(tmp_path / "raw.db"))
    with caplog.at_level(logging.INFO, logger=bf.__name__):
        backfill = RawBackfill(raw, PoliteHttp(), lambda: NOW)
    [line] = [r.getMessage() for r in caplog.records if "projected" in r.getMessage()]
    found = backfill.projection()
    waves = found["waves"]
    assert [waves[wave]["leagues"] for wave in (1, 2, 3)] == [12, 3, 43]
    assert sum(waves[wave]["requests"] for wave in (1, 2, 3)) == found["requests"]
    assert abs(sum(waves[wave]["gb"] for wave in (1, 2, 3)) - found["gb"]) <= 0.02
    assert waves[1]["requests"] + waves[2]["requests"] == sum(n for n, _mb in PROJECTED_BEFORE_WAVE3.values())
    # Wave 3: about 115,000 requests (college basketball's ~5,800 games a season most of it), about 3 GB
    assert 100_000 <= waves[3]["requests"] <= 130_000 and 2.5 <= waves[3]["gb"] <= 3.5, waves
    assert waves[3]["nights"] == round(waves[3]["requests"] / 6000, 1) and found["nights"] == \
        round(found["requests"] / 6000, 1)
    for wave in (1, 2, 3):
        entry = waves[wave]
        assert f"wave {wave} ({entry['leagues']} leagues) {entry['requests']:,} requests {entry['gb']:.2f} GB, " \
               f"about {entry['nights']} nights" in line
    assert f"projected {found['requests']:,} requests" in line and f"save {found['saved']:,} requests" in line
    # Tennis: its day boards only; college basketball at 35 KB a game (plan section 9.4), the NBA at 60
    units = backfill.units()
    atp_days = sum((u.last - u.first).days + 1 for u in units if u.league == ATP)
    assert found["leagues"][ATP]["requests"] == atp_days
    assert abs(found["leagues"][ATP]["mb"] - atp_days * bf.BOARD_KB["tennis"] / 1024) <= 0.1
    ncaab = found["leagues"]["basketball_ncaab"]
    ncaab_days = sum((u.last - u.first).days + 1 for u in units if u.league == "basketball_ncaab")
    assert abs(ncaab["mb"] - (ncaab_days * 15 + 3 * 5800 * 35) / 1024) <= 0.1  # 3 whole seasons, none being played
    # Saved: the plain windows' days the tight windows leave out (no calendar read yet)
    settled = bf.settled_day(NOW)

    def span(first: datetime.date, last: datetime.date) -> set:
        return {first + datetime.timedelta(days=n) for n in range((last - first).days + 1)}

    saved = 0
    for u in units:
        if u.league in bf.WAVE3:
            plain = span(*bf._window_days(bf.PLAIN[u.league], u.season))
            saved += len({day for day in plain if not u.current or day <= settled} - span(u.first, u.last))
    assert found["saved"] == saved > 5000
    # Once a season's calendar is read, its bounded days count instead
    mls = next(u for u in units if u.league == MLS and not u.current)
    bound = [mls.first.isoformat(), mls.last.isoformat(), (mls.first + datetime.timedelta(days=10)).isoformat(),
             (mls.last - datetime.timedelta(days=5)).isoformat()]
    raw.set_cursor(MLS, mls.season, {"v": bf.CURSOR_VERSION, "bound": bound, "bound_current": False}, now=NOW)
    again = backfill.projection()
    assert again["leagues"][MLS]["requests"] == found["leagues"][MLS]["requests"] - 15
    assert again["saved"] == found["saved"] + 15 and again["requests"] == found["requests"] - 15


# --- the walk: windows, order, kinds ---------------------------------------------------------------------------------

def test_a_window_is_seven_one_day_boards_newest_first_then_its_games(tmp_path: Path, short: Any) -> None:
    short(((10, 10), (11, 20)), NBA)  # 2026: 10-30 Oct so far (three windows); 2025: 10 Oct-20 Nov (six)
    env = Env(tmp_path, [NBA], seasons=1)
    report = env.backfill().run_night()
    assert report["stopped"] == "done" and report["ran"] is True, report
    kinds = [(kind, what) for kind, _league, what, _host, _ts in env.world.calls]
    boards = [what for kind, what in kinds if kind == "board"]
    # Every board is one day (ESPN refuses a range), each read once, the newest season and week first
    assert all(len(what) == 8 and what.isdigit() for what in boards)
    assert len(boards) == len(set(boards)) == 21 + 42
    assert boards[:7] == days(datetime.date(2026, 10, 24), datetime.date(2026, 10, 30))[::-1]
    assert boards[7:14] == days(datetime.date(2026, 10, 17), datetime.date(2026, 10, 23))[::-1]
    assert boards[14:21] == days(datetime.date(2026, 10, 10), datetime.date(2026, 10, 16))[::-1]
    assert boards[21:28] == days(datetime.date(2025, 11, 14), datetime.date(2025, 11, 20))[::-1]
    assert boards[-1] == "20251010"
    # A window's games follow its seven boards, before the next window's boards
    first_window = set(boards[:7])
    between = kinds[kinds.index(("board", boards[6])) + 1:kinds.index(("board", boards[7]))]
    assert between and all(kind == "summary" and what[2:10] in first_window for kind, what in between)

    def games(first: datetime.date, last: datetime.date) -> List[str]:
        return [i for d in days(first, last) for i in event_ids(NBA, datetime.datetime.strptime(d, "%Y%m%d").date())]

    expected = games(datetime.date(2026, 10, 10), datetime.date(2026, 10, 30)) + \
        games(datetime.date(2025, 10, 10), datetime.date(2025, 11, 20))
    summaries = [what for kind, what in kinds if kind == "summary"]
    assert sorted(summaries) == sorted(expected) and len(summaries) == len(set(summaries))
    progress = {row["season"]: row for row in env.raw.progress()}
    assert progress[2025]["done"] is True and progress[2026]["done"] is False  # 2026 is still being played
    assert progress[2026]["games_seen"] == progress[2026]["games_stored"] == \
        len(games(datetime.date(2026, 10, 10), datetime.date(2026, 10, 30)))
    assert progress[2026]["cursor"]["days_read"] == 21 and progress[2026]["cursor"]["queue"] == []
    summary = {(row["league"], row["season"]): row for row in report["progress"]}  # the report's own summary
    assert summary[(NBA, 2026)]["games_stored"] == progress[2026]["games_stored"]
    assert summary[(NBA, 2026)]["day"] == "2026-10-10" and summary[(NBA, 2025)]["done"] is True
    # The next night, the day settled since is read first, then nothing more
    env.clock.ts = at(2026, 11, 2, 2, 0)
    calls = len(env.world.calls)
    assert env.backfill().run_night()["stopped"] == "done"
    assert [call[2] for call in env.world.calls[calls:] if call[0] == "board"] == ["20261031"]


@pytest.fixture(scope="module")
def every_path(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """One night's walk over all 58 paths, each with a short season (27 October - 2 November): 2026 being played
    (27-30 October) and 2025, 2024 and 2023 completed (the order and the waves 1-2 pin below read it)."""
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(raw_store.shutil, "disk_usage", lambda path: DiskUsage(100e9, 10e9, 90e9))
        for league in bf.ORDER:
            mp.setitem(bf.WINDOWS, league, ((10, 27), (11, 2)))
        env = Env(tmp_path_factory.mktemp("every_path"), list(bf.ORDER))
        report = env.backfill().run_night()
    finally:
        mp.undo()
    return SimpleNamespace(env=env, report=report)


def test_newest_season_first_in_wave_order_across_every_league(every_path: SimpleNamespace) -> None:
    env, report = every_path.env, every_path.report
    assert report["stopped"] == "done", report
    order = {league: rank for rank, league in enumerate(bf.ORDER)}

    def season_of(league: str, what: str) -> int:
        return int(what[:4])

    units = []  # (season tier, wave rank) of every board, in the order read
    for kind, league, what, _host, _ts in env.world.calls:
        if kind == "board" and league != UFC:
            units.append((2026 - season_of(league, what), order[league], what))
    assert [unit[:2] for unit in units] == sorted(unit[:2] for unit in units)
    for tier in range(4):
        for league in bf.ORDER:
            if league == UFC:
                continue
            seen = [what for t, rank, what in units if t == tier and rank == order[league]]
            if league in bf.CALENDAR_LEAGUES:  # wave 3 reads its calendar first, from its middle day's board
                probe, seen = seen[0], seen[1:]
                assert probe not in seen and seen[-1] < probe < seen[0], (league, tier)
            assert seen == sorted(seen, reverse=True) and seen, (league, tier)  # newest day first
    # Only ESPN and nflverse's GitHub: never The Odds API, never MLB StatsAPI
    hosts = [call[3] for call in env.world.calls]
    assert set(hosts) <= ESPN_HOSTS | {"github.com"} and "api.the-odds-api.com" not in hosts
    # Wave 1 (NBA, NFL, the ten soccer leagues), then wave 2 (NHL, UFC, MLB), then wave 3 within each season
    for year in ("2026", "2025", "2024", "2023"):
        at_ = [(bf.wave_of(call[1]), i) for i, call in enumerate(env.world.calls)
               if call[0] == "board" and call[2].startswith(year)]
        assert [wave for wave, _i in at_] == sorted(wave for wave, _i in at_), year
        assert {wave for wave, _i in at_} == {1, 2, 3}


def digest(found: Any) -> str:
    return hashlib.sha256(json.dumps(found).encode()).hexdigest()


# Waves 1-2 before wave 3 (head f02f12a): the every_path night with only their 15 leagues (its requests, in order,
# the rows they stored and their progress), their seasons and the projection's figures with the real windows
BEFORE_WAVE3 = {"calls": 598, "requests": "efb00eeac3cc6d2b5dce597acb3ff7a08d6275281c754f2ef218cab0470d7796",
                "rows": "e46220edc08fa8c0c3d7cde0c19506a460bb37aab28812c659f8dc0ab8357ea2",
                "progress": "b9e54b6d8fa0d0e13fc8ab017a7988d30ccc8deee8151f3b6784ab7e77dd9d8b",
                "units": "a5d6c93bdd8c0298316792c3a0cca64cf61c6b41d4b073f7cfd3bbbc8dfaa329"}
PROJECTED_BEFORE_WAVE3 = {
    "basketball_nba": (5173, 265.7), "americanfootball_nfl": (1540, 143.6), "soccer_epl": (8526, 278.4),
    "soccer_efl_champ": (12030, 406.7), "soccer_spain_la_liga": (8526, 278.4),
    "soccer_germany_bundesliga": (7101, 226.2), "soccer_italy_serie_a": (8526, 278.4),
    "soccer_france_ligue_one": (7101, 226.2), "soccer_spl": (5518, 168.2),
    "soccer_netherlands_eredivisie": (7062, 224.8), "soccer_portugal_primeira_liga": (7062, 224.8),
    "soccer_uefa_champs_league": (4563, 135.9), "icehockey_nhl": (5204, 246.3), UFC: (402, 9.4),
    "baseball_mlb": (11297, 926.8)}


def test_waves_1_and_2_ask_for_exactly_what_they_did_before_wave_3(every_path: SimpleNamespace) -> None:
    env = every_path.env
    calls = [[kind, league, what] for kind, league, what, _host, _ts in env.world.calls if league in bf.WAVES_1_2]
    rows = [list(map(str, row)) for row in env.rows() if row[1] in bf.WAVES_1_2]
    progress = sorted((row["league"], row["season"], row["done"], row["games_seen"], row["games_stored"])
                      for row in env.raw.progress() if row["league"] in bf.WAVES_1_2)
    assert len(calls) == BEFORE_WAVE3["calls"] and digest(calls) == BEFORE_WAVE3["requests"]
    assert digest(rows) == BEFORE_WAVE3["rows"] and digest(progress) == BEFORE_WAVE3["progress"]
    assert {call[1] for call in env.world.calls} == set(bf.ORDER)  # while wave 3 was walked in the same night
    backfill = RawBackfill(SimpleNamespace(), PoliteHttp(), lambda: NOW)
    units = [[u.league, u.season, u.first.isoformat(), u.last.isoformat(), u.current, u.tier, u.rank]
             for u in backfill.units() if u.league in bf.WAVES_1_2]
    assert len(units) == 60 and digest(units) == BEFORE_WAVE3["units"]
    found = backfill.projection()["leagues"]
    assert {league: (found[league]["requests"], found[league]["mb"]) for league in bf.WAVES_1_2} == \
        PROJECTED_BEFORE_WAVE3 and all(found[league]["seasons"] == 4 for league in bf.WAVES_1_2)


def test_every_kind_is_kept_raw_under_its_game_key(tmp_path: Path, short: Any) -> None:
    short(((10, 20), (11, 5)), NBA, EPL, NFL, MLB)  # 2026 being played, and 2025
    env = Env(tmp_path, [NBA, NFL, EPL, MLB], seasons=1)
    assert env.backfill().run_night()["stopped"] == "done"
    nba_game = event_ids(NBA, datetime.date(2026, 10, 27))[0]
    summary = env.raw.latest(f"espn:{nba_game}", "espn_site", "summary")
    assert summary is not None and (summary.state, summary.horizon, dict(summary.params)) == ("final", "final", {})
    assert summary.raw_bytes() == (FIX / "sum_nba.json").read_bytes()  # untrimmed: as ESPN sent it
    assert {"plays", "winprobability", "boxscore", "header"} <= set(summary.json())
    # The day's board, raw (not through LEAN_DROP), under the sweep's key
    board = env.raw.latest(scoreboard_key(NBA, "20261027"), "espn_site", "scoreboard")
    assert board is not None and dict(board.params) == {"dates": "20261027"} and board.state == "final"
    assert "leaders" in json.dumps(board.json())
    # Wave-1 soccer: every page of the core plays
    match = event_ids(EPL, datetime.date(2026, 10, 24))[0]
    pages = env.raw.bundle(f"espn:{match}").pages("espn_core", "core_plays")
    assert [dict(p.params) for p in pages] == [{"limit": 300, "page": 1}, {"limit": 300, "page": 2}]
    assert env.raw.has_final(f"espn:{match}")
    # nflverse: each season's play-by-play once, kept as the gzip file it is
    pbp = env.raw.latest(nflverse.pbp_key(2026), "nflverse", "pbp_season", {"season": 2026})
    assert pbp is not None and pbp.codec == "identity" and pbp.league == NFL and pbp.state == "static"
    assert [what for _k, _l, what, _h, _t in env.world.of("pbp")] == ["2026", "2025"]
    # MLB: ESPN only, never StatsAPI (its terms are non-bulk)
    assert env.world.of("summary", MLB) and all(call[3] in ESPN_HOSTS | {"github.com"} for call in env.world.calls)
    assert not [call for call in env.world.calls if "statsapi" in call[3]]


def test_ufc_reads_its_cards_from_the_calendar_not_every_day(tmp_path: Path) -> None:
    env = Env(tmp_path, [UFC], seasons=1)
    postponed_day = next(day for day in (datetime.date(2025, 6, 1) + datetime.timedelta(days=n) for n in range(28))
                         if is_card_day(day))
    env.world.postponed.add(event_ids(UFC, postponed_day)[0])
    report = env.backfill().run_night()
    assert report["stopped"] == "done", report
    boards = [what for _k, _l, what, _h, _t in env.world.of("board")]
    cards_2026 = [d for d in days(datetime.date(2026, 1, 1), datetime.date(2026, 10, 30))
                  if is_card_day(datetime.datetime.strptime(d, "%Y%m%d").date())]
    cards_2025 = [d for d in days(datetime.date(2025, 1, 1), datetime.date(2025, 12, 31))
                  if is_card_day(datetime.datetime.strptime(d, "%Y%m%d").date())]
    # One board for the calendar (the newest day; 30 December for a finished season), then the cards' days only
    assert boards[0] == "20261030" and boards[1:1 + len(cards_2026)] == cards_2026[::-1]
    assert boards[1 + len(cards_2026)] == "20251230"
    assert boards[2 + len(cards_2026):] == cards_2025[::-1]
    assert len(boards) == len(cards_2026) + len(cards_2025) + 2  # not 668 days of boards
    fights = [what for _k, _l, what, _h, _t in env.world.of("fightcenter")]
    assert len(fights) == len(cards_2026) + len(cards_2025) - 1  # the postponed card has none
    assert event_ids(UFC, postponed_day)[0] not in fights
    assert not env.world.of("summary")  # ESPN has no MMA summary (404): fightcenter is the card's record
    card = env.raw.latest(f"espn:{fights[0]}", "espn_fightcenter", "fightcenter")
    assert card is not None and card.league == UFC and card.state == "final"
    assert card.raw_bytes() == (FIX / "syn_fightcenter.json").read_bytes()
    # The calendar's start (01:00 UTC the next day) is the card's US Eastern day
    assert fights[0] == event_ids(UFC, datetime.datetime.strptime(cards_2026[-1], "%Y%m%d").date())[0]


def test_calendar_cards_reads_the_real_ufc_board() -> None:
    board = load("sb_ufc.json")
    cards = bf.calendar_cards(board, 2026)
    assert cards is not None and len(cards) == len(board["leagues"][0]["calendar"])
    assert cards == sorted(cards, reverse=True)
    by_id = {event_id: day for day, event_id in cards}
    assert by_id["600061267"] == "2026-10-24"  # UFC 333 (17:00 UTC)
    assert by_id["600057024"] == "2026-01-24"  # UFC 324: 01:30 UTC on the 25th is the 24th in New York
    assert bf.calendar_cards(board, 2025) is None and bf.calendar_cards({"events": []}, 2026) is None
    # The real boards' events read as finished games
    for name, card in (("sb_nba.json", False), ("sb_epl.json", False), ("sb_ufc.json", True)):
        events = bf.board_events(load(name))
        assert events and all(bf.completed(bf.event_status(e, card)) for e in events), name
    assert bf.summary_state((FIX / "sum_nba.json").read_bytes()) == ("final", "final")
    assert bf.summary_state(b"not json") == ("pre", None)
    assert not bf.completed(POSTPONED) and bf.completed(FINAL)


# --- budget and window -----------------------------------------------------------------------------------------------

def test_the_nights_budget_holds_across_calls_and_restarts(tmp_path: Path, short: Any) -> None:
    short(((10, 1), (11, 20)), NBA)
    env = Env(tmp_path, [NBA], per_night=30)
    report = env.backfill().run_night()
    assert (report["ran"], report["requests"], report["used"], report["stopped"]) == (True, 30, 30, "budget")
    assert env.raw.spent("backfill", "2026-11-01") == 30 and len(env.world.calls) == 30
    again = env.backfill().run_night()  # a restart the same night spends nothing more
    assert (again["requests"], again["stopped"], len(env.world.calls)) == (0, "budget", 30)
    clock = env.clock
    clock.ts = at(2026, 11, 2, 1, 0)  # the next night has its own budget, in slices
    slice_ = env.backfill().run_night(budget=12)
    assert (slice_["requests"], slice_["stopped"], slice_["used"]) == (12, "budget", 12)
    rest = env.backfill().run_night()
    assert rest["requests"] == 18 and rest["used"] == 30 and env.raw.spent("backfill", "2026-11-02") == 30
    assert env.raw.spent("backfill", "2026-11-01") == 30
    # The default night budget is CAPTURE_BACKFILL_PER_NIGHT's 6,000
    assert RawBackfill(env.raw, env.http(), clock).per_night == 6000 == bf.PER_NIGHT_DEFAULT


@pytest.mark.parametrize("hour,minute", [(0, 59), (6, 0), (12, 0), (23, 30)])
def test_nothing_is_fetched_outside_01_to_06(tmp_path: Path, hour: int, minute: int) -> None:
    env = Env(tmp_path, [NBA], ts=at(2026, 11, 1, hour, minute))
    report = env.backfill().run_night()
    assert report == {**report, "ran": False, "requests": 0, "stopped": "window"} and env.world.calls == []


def test_a_run_stops_when_the_window_closes_and_the_server_zone_decides(tmp_path: Path, short: Any) -> None:
    short(((10, 1), (11, 20)), NBA)
    env = Env(tmp_path, [NBA], ts=at(2026, 11, 1, 5, 58), per_request=20.0)
    report = env.backfill().run_night()
    assert report["stopped"] == "window" and 1 <= report["requests"] <= 7
    assert all(datetime.datetime.fromtimestamp(ts, UTC).hour < 6 for *_rest, ts in env.world.calls)
    # 02:00 UTC is 22:00 the day before at UTC-4: outside that server's window
    env2 = Env(tmp_path, [NBA], name="raw2.db")
    eastern = env2.backfill(tz=datetime.timezone(datetime.timedelta(hours=-4)))
    assert eastern.in_window() is False and eastern.run_night()["stopped"] == "window"
    assert env2.world.calls == []
    assert env2.backfill(window=(21, 23), tz=datetime.timezone(datetime.timedelta(hours=-4))).in_window()


# --- resume, idempotence ---------------------------------------------------------------------------------------------

def test_a_restart_mid_season_carries_on_where_it_stopped(tmp_path: Path, short: Any) -> None:
    short(((10, 10), (10, 30)), NBA, EPL, UFC)
    whole = Env(tmp_path, [NBA, EPL, UFC], seasons=1, name="whole.db")
    assert whole.backfill().run_night()["stopped"] == "done"
    pieces = Env(tmp_path, [NBA, EPL, UFC], seasons=1, name="pieces.db")
    pieces.world.fail[("summary", event_ids(NBA, datetime.date(2026, 10, 27))[0])] = ["reset"]
    runs = 0
    while True:
        runs += 1
        report = pieces.backfill().run_night(budget=7)  # a restart after every 7 requests
        if report["stopped"] == "done":
            break
        if report["stopped"] == "backoff":  # the reset made site.api.espn.com back off: a later call goes on
            pieces.clock.ts += 60
            continue
        assert report["stopped"] == "budget" and report["requests"] == 7
        assert runs < 200
    asked = sorted(call[:3] for call in pieces.world.calls)
    once = sorted(call[:3] for call in whole.world.calls)
    # The same work: one more request for the one that failed, nothing asked twice otherwise (a restart mid-window,
    # mid-queue and mid-pager included)
    failed = ("summary", NBA, event_ids(NBA, datetime.date(2026, 10, 27))[0])
    assert asked == sorted(once + [failed])
    assert pieces.rows() == whole.rows()
    strip = lambda rows: [{k: row[k] for k in ("league", "season", "done", "games_seen", "games_stored")}  # noqa: E731
                          for row in rows]
    assert strip(pieces.raw.progress()) == strip(whole.raw.progress())


def test_a_restart_between_the_store_and_the_cursor_fetches_nothing_twice(tmp_path: Path, short: Any,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    short(((10, 20), (10, 30)), EPL)
    env = Env(tmp_path, [EPL], seasons=1)
    real_insert, real_save = env.raw.insert, env.raw.set_cursor
    stored: Dict[str, Any] = {"summary": None, "crashed": False}

    def insert(**kwargs: Any) -> Tuple[int, bool]:
        if kwargs["endpoint"] == "summary":
            stored["summary"] = kwargs["game_key"]
        return real_insert(**kwargs)

    def save(*args: Any, **kwargs: Any) -> None:
        if stored["summary"] and not stored["crashed"]:  # the process dies after a summary was stored, before
            stored["crashed"] = True                    # its cursor was saved
            raise RuntimeError("killed")
        real_save(*args, **kwargs)

    monkeypatch.setattr(env.raw, "insert", insert)
    monkeypatch.setattr(env.raw, "set_cursor", save)
    with pytest.raises(RuntimeError):
        env.backfill().run_night()
    assert stored["crashed"] and env.raw.has_final(stored["summary"])
    assert env.backfill().run_night()["stopped"] == "done"
    asked = [call[:3] for call in env.world.calls]
    assert len(asked) == len(set(asked))  # nothing fetched twice
    assert env.raw.progress()[0]["games_stored"] == env.raw.progress()[0]["games_seen"] > 0


def test_games_already_stored_are_skipped_and_a_repeat_run_costs_nothing(tmp_path: Path, short: Any) -> None:
    short(((10, 20), (10, 30)), NBA, EPL)
    env = Env(tmp_path, [NBA, EPL], seasons=1)
    # The worker already captured some finals (a summary, a match's summary and both plays pages)
    nba_game = event_ids(NBA, datetime.date(2026, 10, 27))[1]
    match = event_ids(EPL, datetime.date(2026, 10, 24))[0]
    for game, league, body in ((nba_game, NBA, b'{"header":{"competitions":[{"status":{"type":{"completed":true}}}]}}'),
                               (match, EPL, (FIX / "sum_epl.json").read_bytes())):
        env.raw.put(game_key=f"espn:{game}", league=league, source="espn_site", endpoint="summary", params={},
                    body=body, state="final", http_status=200, horizon="final", fetched_at=NOW - 3600)
    for page, body in env.world.pages.items():
        env.raw.put(game_key=f"espn:{match}", league=EPL, source="espn_core", endpoint="core_plays",
                    params={"page": page, "limit": 300}, body=body, state="final", http_status=200, horizon="final")
    report = env.backfill().run_night()
    assert report["stopped"] == "done" and report["skipped"] >= 2
    asked = {what for kind, _l, what, _h, _t in env.world.calls if kind in ("summary", "plays")}
    assert nba_game not in asked and match not in asked and not any(w.startswith(match) for w in asked)
    first_rows = env.rows()
    progress = {(row["league"], row["season"]): row for row in env.raw.progress()}
    assert progress[(NBA, 2026)]["games_stored"] == progress[(NBA, 2026)]["games_seen"]
    # A repeat run, every cursor gone: boards are read back from raw.db, nothing is fetched, nothing new stored
    db = sqlite3.connect(env.raw.path)
    db.execute("DELETE FROM backfill_progress")
    db.commit()
    db.close()
    calls = len(env.world.calls)
    repeat = env.backfill().run_night()
    assert repeat["stopped"] == "done" and repeat["requests"] == 0 and len(env.world.calls) == calls
    assert env.rows() == first_rows and repeat["stored"] == 0


# --- forward capture first, disk, politeness ------------------------------------------------------------------------

def test_a_due_forward_job_is_never_delayed(tmp_path: Path, short: Any) -> None:
    short(((10, 1), (11, 20)), NBA)  # 2026 so far and 2025: a few hundred requests
    env = Env(tmp_path, [NBA], seasons=1, per_request=1.0)
    due_at = NOW + 40.0
    report = env.backfill(forward_due=lambda: env.clock() >= due_at).run_night()
    assert report["stopped"] == "forward" and report["requests"] > 10
    assert all(ts < due_at for *_rest, ts in env.world.calls)  # nothing once the job was due
    # With the worker's jobs in raw.db (the default check): a job due now stops the backfill before any request
    env.raw.add_job("espn:999", NBA, "999", start=env.clock() - 7200, due_at=env.clock() - 1, now=env.clock())
    calls = len(env.world.calls)
    blocked = env.backfill().run_night()
    assert (blocked["ran"], blocked["requests"], blocked["stopped"]) == (False, 0, "forward")
    assert len(env.world.calls) == calls
    # A job due a little later stops it at that moment
    job = env.raw.job("espn:999")
    job.state, job.due_at = "todo", env.clock() + 30.0
    env.raw.save_job(job, now=env.clock())
    later = env.backfill().run_night()
    assert later["stopped"] == "forward" and all(ts < job.due_at for *_r, ts in env.world.calls[calls:])
    # Once the worker has done it, the backfill carries on where it stopped, nothing fetched twice
    job = env.raw.job("espn:999")
    job.state = "done"
    env.raw.save_job(job, now=env.clock())
    assert env.backfill().run_night()["stopped"] == "done"
    asked = [call[:3] for call in env.world.calls]
    assert len(asked) == len(set(asked))


def test_the_disk_floor_stops_the_backfill(tmp_path: Path, short: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    short(((10, 1), (10, 30)), NBA)
    env = Env(tmp_path, [NBA], seasons=1)
    monkeypatch.setattr(env.raw, "room", lambda: False, raising=False)
    report = env.backfill().run_night()
    assert (report["ran"], report["requests"], report["stopped"]) == (False, 0, "disk") and env.world.calls == []
    # Room running out during a run stops it before the next request (the cursor kept)
    left = {"n": 9}

    def room() -> bool:
        left["n"] -= 1
        return left["n"] >= 0

    monkeypatch.setattr(env.raw, "room", room, raising=False)
    report = env.backfill().run_night()
    assert report["stopped"] == "disk" and 1 <= report["requests"] <= 8
    monkeypatch.setattr(env.raw, "room", lambda: True, raising=False)
    assert env.backfill().run_night()["stopped"] == "done"
    asked = [call[:3] for call in env.world.calls]
    assert len(asked) == len(set(asked))


@pytest.mark.parametrize("free,stopped", [(9e9, "disk"), (1.5 * 2 ** 30, "disk"), (11e9, "done")])
def test_the_disk_floor_from_the_disks_free_space(tmp_path: Path, short: Any, monkeypatch: pytest.MonkeyPatch,
                                                  free: float, stopped: str) -> None:
    short(((10, 25), (10, 30)), NBA)
    monkeypatch.setattr(raw_store.shutil, "disk_usage", lambda path: DiskUsage(100e9, 100e9 - free, free))
    env = Env(tmp_path, [NBA], seasons=0)  # 9 GB of 100 free is under the 10% floor; 1.5 GiB under the 2 GiB one
    report = env.backfill().run_night()
    assert report["stopped"] == stopped
    if stopped == "disk":
        assert report["requests"] == 0 and env.world.calls == [] and env.raw.payload_count() == 0


def test_requests_keep_to_each_hosts_token_bucket(tmp_path: Path, short: Any) -> None:
    short(((10, 20), (10, 30)), NBA, EPL, NFL)  # UFC keeps its calendar year: a card every other week
    env = Env(tmp_path, [NBA, NFL, EPL, UFC], seasons=1)
    assert env.backfill().run_night()["stopped"] == "done"
    rates = {"site.api.espn.com": (1.0, 3), "sports.core.api.espn.com": (1.0, 3), "site.web.api.espn.com": (0.5, 1),
             "github.com": (0.2, 1)}
    hosts = {call[3] for call in env.world.calls}
    assert hosts <= set(rates) and {"site.api.espn.com", "sports.core.api.espn.com", "site.web.api.espn.com"} <= hosts
    for host, (rate, burst) in rates.items():
        times = [ts for _k, _l, _w, h, ts in env.world.calls if h == host]
        for i in range(len(times)):
            for j in range(i, min(len(times), i + 40)):
                assert j - i + 1 <= burst + rate * (times[j] - times[i]) + 1e-6, (host, i, j)


def test_a_definitive_failure_is_given_up_after_three_tries_and_a_transient_one_never(tmp_path: Path,
                                                                                   short: Any) -> None:
    short(((10, 20), (10, 30)), NBA)
    env = Env(tmp_path, [NBA], seasons=1)
    flaky, broken = event_ids(NBA, datetime.date(2026, 10, 27))
    throttled = event_ids(NBA, datetime.date(2026, 10, 25))[0]
    env.world.fail[("summary", flaky)] = [503]
    env.world.fail[("summary", throttled)] = [429, 429, 429, 429]  # a 429 is never ESPN's definitive answer
    env.world.fail[("summary", broken)] = [404, 404, 404]
    backfill = env.backfill()
    report = backfill.run_night()
    for _ in range(40):  # the worker's next calls, a minute or more apart: the retries wait RETRY_WAIT, doubling
        if report["stopped"] == "done":
            break
        assert report["stopped"] == "backoff", report  # a host backing off, or retries still waiting
        env.clock.ts += 300
        report = backfill.run_night()
    assert report["stopped"] == "done", report
    asked = lambda game: len([c for c in env.world.of("summary") if c[2] == game])  # noqa: E731
    assert asked(flaky) == 2 and env.raw.has_final(f"espn:{flaky}")
    assert asked(throttled) == 5 and env.raw.has_final(f"espn:{throttled}")
    assert asked(broken) == 3 and not env.raw.has_final(f"espn:{broken}")
    cursor = env.raw.get_cursor(NBA, 2026)
    assert cursor["given_up"] == 1 and cursor["skipped"] == [[f"summary {broken}", "HTTP 404"]]
    assert cursor["retry"] == [] and env.raw.progress(NBA)[0]["done"] is True
    # The failed answers are kept for the health block's latest errors
    assert any(error.get("status") == 404 for error in env.raw.recent_errors())


def test_a_kind_failing_ten_times_in_a_row_stops_for_the_night(tmp_path: Path, short: Any) -> None:
    short(((10, 1), (10, 30)), NBA)
    env = Env(tmp_path, [NBA], seasons=1)
    env.world.fail_always["summary"] = 404  # ESPN changed its summary endpoint
    backfill = env.backfill()  # the worker keeps one for the process
    report = backfill.run_night()
    assert report["stopped"] == "errors" and report["failed"] == bf.STOP_AFTER_ERRORS
    assert report["given_up"] < bf.STOP_AFTER_ERRORS  # the rest wait for a later night instead of being given up
    calls = len(env.world.of("summary"))
    again = backfill.run_night()  # the worker's next call the same night: no more summary requests
    assert again["stopped"] == "errors" and len(env.world.of("summary")) == calls
    env.world.fail_always.clear()
    env.clock.ts = at(2026, 11, 2, 2, 0)  # the next night
    assert backfill.run_night()["stopped"] == "done"
    cursor = env.raw.get_cursor(NBA, 2026)
    assert cursor["queue"] == [] and cursor["given_up"] == report["given_up"]


def test_a_host_backing_off_holds_up_only_its_own_work(tmp_path: Path, short: Any) -> None:
    short(((10, 20), (10, 30)), NBA)
    env = Env(tmp_path, [UFC, NBA], seasons=1)
    http = env.http()
    http._settle("site.web.api.espn.com", True, retry_after=600)  # fightcenter's host backs off for 10 minutes
    backfill = RawBackfill(env.raw, http, env.clock, leagues=[UFC, NBA], tz=UTC, seasons=1)
    start = env.clock()
    report = backfill.run_night()
    assert report["stopped"] == "backoff"
    assert not env.world.of("fightcenter") and env.world.of("summary", NBA)  # the NBA went on
    assert env.clock() - start < 600  # never waited the back-off out
    env.clock.ts += 700
    assert backfill.run_night()["stopped"] == "done" and env.world.of("fightcenter")


def test_no_odds_api_and_no_statsapi_request(tmp_path: Path, short: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    import urllib.request
    outside: List[str] = []
    monkeypatch.setattr(requests, "get", lambda *a, **k: outside.append(str(a[:1])) or pytest.fail("requests.get"))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: outside.append(str(a[:1])) or pytest.fail("urlopen"))
    short(((10, 20), (11, 5)), NBA, NFL, EPL, MLB)  # with UFC's year: every host the backfill uses
    env = Env(tmp_path, [NBA, NFL, EPL, UFC, MLB], seasons=1)
    assert env.backfill().run_night()["stopped"] == "done"
    hosts = [call[3] for call in env.world.calls]
    assert set(hosts) == ESPN_HOSTS | {"github.com"}
    assert hosts and set(hosts) <= ESPN_HOSTS | {"github.com"}
    assert sum(host == "api.the-odds-api.com" for host in hosts) == 0 and outside == []
    assert sum("statsapi.mlb.com" in host for host in hosts) == 0
    sent = [made.requests for made in env.http_made]
    assert all(set(counts) <= ESPN_HOSTS | {"github.com"} for counts in sent)


# --- the learn backfill's raw-first hook (learn/history.py, learn/loop.py) -----------------------------------------

EMPTY_LIST = json.dumps({"count": 0, "pageIndex": 1, "pageCount": 1, "items": []}).encode()
LEARN_AT = at(2026, 10, 28, 2)


class LearnTransport:
    """The learn backfill's ESPN NBA week (tests/fixtures/learn/): the event list, each game's summary and odds."""

    def __init__(self, clock: Clock, summary: Optional[bytes] = None) -> None:
        self.clock = clock
        self.summary = summary if summary is not None else (LEARN_FIX / "espn_nba_summary.json").read_bytes()
        self.calls: List[Tuple[str, Dict[str, Any], float]] = []

    def __call__(self, url: str, params: Dict[str, Any], timeout: float) -> Tuple[int, bytes]:
        self.calls.append((url, dict(params), self.clock()))
        if "sports.core.api.espn.com" in url and url.endswith("/events"):
            return 200, (LEARN_FIX / "espn_nba_events.json").read_bytes() if "/nba/" in url else EMPTY_LIST
        if url.endswith("/odds"):
            return 200, (LEARN_FIX / "espn_nba_odds.json").read_bytes()
        if url.endswith("/summary"):
            return 200, self.summary
        return 404, b"{}"

    def summaries(self) -> List[str]:
        return [params.get("event") for url, params, _ts in self.calls if url.endswith("/summary")]


def learn_run(tmp_path: Path, name: str, raw: Any = None, cache: Optional[TTLCache] = None,
              summary: Optional[bytes] = None) -> Tuple[Store, LearnTransport, Dict[str, Any], TTLCache]:
    clock = Clock(LEARN_AT)
    transport = LearnTransport(clock, summary)
    store = Store(str(tmp_path / f"{name}.db"))
    cache = cache if cache is not None else TTLCache()
    http = HistHttp(clock=clock, sleep=clock.sleep, transport=transport)
    options = {} if raw is None else {"raw": raw}
    report = Backfill(store, cache, clock, http=http, tz=UTC, per_night=500, seasons=1, sources=["espn"],
                      leagues=["basketball_nba"], **options).run_night()
    return store, transport, report, cache


def dump(store: Store) -> Dict[str, List[Tuple[Any, ...]]]:
    db = store._connect()
    hs.init_history_schema(db)
    try:
        return {table: sorted(db.execute(f"SELECT * FROM {table}").fetchall(), key=repr)
                for table in ("hist_games", "hist_lines", "hist_stats", "hist_weather", "hist_progress",
                              "hist_budget")}
    finally:
        db.close()


def test_the_learn_hook_with_raw_db_empty_is_byte_identical(tmp_path: Path) -> None:
    before, plain, plain_report, _ = learn_run(tmp_path, "plain")
    raw = RawStore(str(tmp_path / "raw.db"))
    after, hooked, hooked_report, cache = learn_run(tmp_path, "hooked", raw)
    assert plain_report["stopped"] == "done" and hooked_report == plain_report
    assert hooked.calls == plain.calls  # the same requests, params, order and pace: one budget
    assert dump(after) == dump(before)
    assert len(plain.summaries()) == 4
    # ... and the raw bodies are in raw.db too: one fetch served both stores
    for event_id in plain.summaries():
        payload = raw.latest(f"espn:{event_id}", "espn_site", "summary")
        assert payload is not None and payload.state == "final" and dict(payload.params) == {}
        assert payload.raw_bytes() == (LEARN_FIX / "espn_nba_summary.json").read_bytes()
        # seed=False: an old game's lean copy never fills the grader's cache (in memory or in the shared table)
        assert cache.peek(("espn", "summary", "basketball_nba", event_id)) is None


def test_the_learn_hook_reads_a_stored_summary_with_no_request(tmp_path: Path) -> None:
    plain_store, plain, _report, _ = learn_run(tmp_path, "plain")
    raw = RawStore(str(tmp_path / "raw.db"))
    for event_id in plain.summaries():  # the raw backfill (or the worker) stored them
        raw.put(game_key=f"espn:{event_id}", league="basketball_nba", source="espn_site", endpoint="summary",
                params={}, body=(LEARN_FIX / "espn_nba_summary.json").read_bytes(), state="final", http_status=200,
                horizon="final")
    rows = raw.payload_count()
    store, hooked, report, _ = learn_run(tmp_path, "hooked", raw)
    assert report["stopped"] == "done" and hooked.summaries() == []  # zero extra requests: none at all
    assert report["requests"] == len(plain.calls) - len(plain.summaries())
    assert raw.payload_count() == rows
    # The same history (only the rows' write times differ: four fewer requests paced 2 seconds apart)
    records = list(HistoryReader(store).games("basketball_nba"))
    assert records == list(HistoryReader(plain_store).games("basketball_nba"))
    assert len(records) == 4 and records[0]["result"] == {"home": 123, "away": 132}
    assert dump(store)["hist_lines"] == dump(plain_store)["hist_lines"]
    assert dump(store)["hist_stats"] == dump(plain_store)["hist_stats"]


def test_the_learn_hook_writes_nothing_while_the_disk_is_short(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                                                ) -> None:
    before, plain, plain_report, _ = learn_run(tmp_path, "plain")
    raw = RawStore(str(tmp_path / "raw.db"))
    monkeypatch.setattr(raw, "room", lambda: False, raising=False)
    after, hooked, report, cache = learn_run(tmp_path, "hooked", raw)
    assert report == plain_report and hooked.calls == plain.calls and dump(after) == dump(before)
    assert raw.payload_count() == 0  # nothing written
    assert len(plain.summaries()) == 4  # summary_raw wasn't used: the lean cache wasn't seeded either
    assert all(cache.peek(("espn", "summary", "basketball_nba", event_id)) is None for event_id in plain.summaries())
    # A summary already there is still read (no request)
    raw.put(game_key=f"espn:{plain.summaries()[0]}", league="basketball_nba", source="espn_site",
            endpoint="summary", params={}, body=(LEARN_FIX / "espn_nba_summary.json").read_bytes(), state="final",
            http_status=200, horizon="final")
    _store, third, _report, _ = learn_run(tmp_path, "third", raw)
    assert third.summaries() == plain.summaries()[1:]


def test_the_learn_backfill_without_raw_never_opens_raw_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))  # beside the store's betting.db, and named by RAW_DB_PATH
    raw.put(game_key="espn:401909861", league="basketball_nba", source="espn_site", endpoint="summary", params={},
            body=(LEARN_FIX / "espn_nba_summary.json").read_bytes(), state="final", http_status=200, horizon="final")
    monkeypatch.setenv("RAW_DB_PATH", raw.path)
    _store, transport, report, _ = learn_run(tmp_path, "betting")
    assert report["stopped"] == "done" and len(transport.summaries()) == 4 and raw.payload_count() == 1


class FakeBackfill:
    made: List[Dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        FakeBackfill.made.append(kwargs)

    def run_night(self) -> Dict[str, Any]:
        return {"ran": True, "stopped": "done", "used": 0, "requests": 0, "games": 0, "budget": 1500}


@pytest.mark.parametrize("config,wired", [
    (SimpleNamespace(capture_enabled=True, raw_db_path=None), True),
    (SimpleNamespace(capture_enabled=True, raw_db_path="custom/raw.db"), True),
    (SimpleNamespace(capture_enabled=False, raw_db_path=None), False),
    (None, False),
])
def test_the_learner_hands_raw_db_to_the_backfill_only_while_capture_is_on(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: Any, wired: bool) -> None:
    from telegram_betting_bot_simple.learn import analyst, history, loop
    monkeypatch.delenv("RAW_DB_PATH", raising=False)
    monkeypatch.setattr(history, "Backfill", FakeBackfill)
    FakeBackfill.made = []
    if config is not None and config.raw_db_path:
        config.raw_db_path = str(tmp_path / config.raw_db_path)
    store = Store(str(tmp_path / "betting.db"))
    learner = loop.Learner(store, config=config, clock=lambda: LEARN_AT, tz=UTC, use_notify=False,
                           analyst=analyst.BrainAnalyst(store, mode="off", clock=lambda: LEARN_AT, tz=UTC),
                           lease=lambda: True)
    result = learner._night_backfill(LEARN_AT, "2026-10-28")
    assert result["stopped"] == "done" and len(FakeBackfill.made) == 1
    options = FakeBackfill.made[0]
    if not wired:
        assert "raw" not in options  # capture off: the backfill exactly as before
        return
    expected = config.raw_db_path or str(tmp_path / "raw.db")
    assert isinstance(options["raw"], RawStore) and options["raw"].path == expected


# --- the fix round (v4 P1 review): M1, M2, the window's time zone, F2, F4 ----------------------------------------

def test_the_learn_hook_leaves_the_graders_lean_cache_alone(tmp_path: Path) -> None:
    """M1 (the reviewer's checks/c04): 100 old games through the raw-first hook leave no lean summary in the cache,
    in memory or in betting.db's shared table; the learn history is the same as without raw.db."""
    games = 100
    summary = json.loads((FIX / "sum_nba.json").read_text(encoding="utf-8"))

    def transport(url: str, params: Dict[str, Any], timeout: float) -> Tuple[int, bytes]:
        if url.endswith("/events"):
            base = int(str(params.get("dates", "0")).split("-")[0]) * 1000
            items = [{"$ref": f"http://sports.core.api.espn.pvt/v2/sports/basketball/leagues/nba/events/{base + n}"}
                     for n in range(games)]
            return 200, json.dumps({"count": games, "pageIndex": 1, "pageCount": 1, "items": items}).encode()
        if url.endswith("/summary"):
            data = copy.deepcopy(summary)
            data["header"]["id"] = str(params.get("event"))
            return 200, json.dumps(data).encode()
        if url.endswith("/odds"):
            return 200, (LEARN_FIX / "espn_nba_odds.json").read_bytes()
        return 404, b"{}"

    def run(hook: bool) -> Tuple[Dict[str, Any], collections.Counter, Dict[str, int], Any, Optional[RawStore]]:
        store = Store(str(tmp_path / f"betting-{int(hook)}.db"))
        clock = Clock(at(2026, 11, 3, 2, 30))
        cache = TTLCache(store, name=f"m1-{int(hook)}", clock=clock)
        raw = RawStore(str(tmp_path / "raw.db")) if hook else None
        report = Backfill(store, cache, clock, http=HistHttp(pace=0, transport=transport), tz=UTC,
                          per_night=2 * games + 1, sources=["espn"], leagues=["basketball_nba"],
                          **({"raw": raw} if raw is not None else {})).run_night()
        kinds = collections.Counter(entry[3] for entry in cache._entries.values())
        db = sqlite3.connect(store.path)
        rows = dict(db.execute("SELECT kind, COUNT(*) FROM shared_cache GROUP BY kind").fetchall())
        db.close()
        return report, kinds, rows, dump(store), raw

    plain_report, plain_kinds, plain_rows, plain_dump, _ = run(False)
    report, kinds, rows, history, raw = run(True)
    assert report["games"] == games and report == plain_report
    assert kinds.get("summary", 0) == 0 and rows.get("summary", 0) == 0  # nothing for the grader's lean cache
    assert kinds == plain_kinds and rows == plain_rows and history == plain_dump
    assert raw is not None and raw.payload_count() == games  # the raw bodies are kept all the same


def test_the_capture_backfill_makes_no_request_while_the_learn_backfill_runs(tmp_path: Path, short: Any) -> None:
    """M2 (the reviewer's checks/c02): the two backfills never read ESPN at the same time."""
    short(((10, 20), (10, 30)), NBA)
    env = Env(tmp_path, [NBA], seasons=1)
    gate, inside = threading.Event(), threading.Event()

    def slow(url: str, params: Dict[str, Any], timeout: float) -> Tuple[int, bytes]:
        inside.set()
        gate.wait(10)
        return 200, EMPTY_LIST

    learn = Backfill(Store(str(tmp_path / "learn.db")), TTLCache(), env.clock, http=HistHttp(pace=0, transport=slow),
                     tz=UTC, per_night=3, seasons=1, sources=["espn"], leagues=["basketball_nba"])
    thread = threading.Thread(target=learn.run_night, daemon=True)  # the dashboard's "learner" thread
    thread.start()
    try:
        assert inside.wait(10) and LEARN_BACKFILL_RUNNING.is_set()
        backfill = env.backfill()
        report = backfill.run_night()
        assert (report["ran"], report["requests"], report["stopped"]) == (False, 0, "learn")
        assert env.world.calls == []
    finally:
        gate.set()
        thread.join(10)
    assert not thread.is_alive() and not LEARN_BACKFILL_RUNNING.is_set()
    assert backfill.run_night()["stopped"] == "done" and env.world.of("summary")  # then it goes on


def test_the_learn_backfill_starting_mid_run_stops_the_capture_backfill_at_its_next_request(tmp_path: Path,
                                                                                           short: Any) -> None:
    short(((10, 1), (11, 20)), NBA)
    env = Env(tmp_path, [NBA], seasons=1)
    send = env.world.send

    def learner_begins(request: Any) -> HttpResponse:
        answer = send(request)
        if len(env.world.calls) == 12:
            LEARN_BACKFILL_RUNNING.set()  # the learner's night run begins
        return answer

    env.world.send = learner_begins  # type: ignore[method-assign]
    report = env.backfill().run_night()
    assert report["stopped"] == "learn" and report["requests"] == 12 and len(env.world.calls) == 12
    LEARN_BACKFILL_RUNNING.clear()
    assert env.backfill().run_night()["stopped"] == "done"
    asked = [call[:3] for call in env.world.calls]
    assert len(asked) == len(set(asked))  # resumed from the cursor: nothing fetched twice


def test_the_learn_backfill_sets_its_flag_for_the_whole_run_and_always_clears_it(tmp_path: Path,
                                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    seen: List[bool] = []
    monkeypatch.setattr(Backfill, "_run_night", lambda self, budget: seen.append(LEARN_BACKFILL_RUNNING.is_set())
                        or {"stopped": "done"})
    learn = Backfill(Store(str(tmp_path / "learn.db")), TTLCache(), LEARN_AT, tz=UTC)
    assert learn.run_night() == {"stopped": "done"} and seen == [True] and not LEARN_BACKFILL_RUNNING.is_set()

    def boom(self: Any, budget: Any) -> Dict[str, Any]:
        raise RuntimeError("the learn backfill failed")

    monkeypatch.setattr(Backfill, "_run_night", boom)
    with pytest.raises(RuntimeError):
        learn.run_night()
    assert not LEARN_BACKFILL_RUNNING.is_set()


@pytest.fixture
def utc_server() -> Any:
    """A server whose own clock is UTC (TZ=UTC), as the VPS's usually is."""
    before = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    yield
    if before is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = before
    time.tzset()


def test_the_window_follows_the_owners_time_zone_on_a_utc_server(tmp_path: Path, utc_server: Any) -> None:
    from telegram_betting_bot_simple.config import Config
    tz = bf.backfill_tz(Config(TELEGRAM_BOT_TOKEN="123456:test", _env_file=None))  # TIMEZONE's default
    assert str(tz) == "America/New_York" == str(bf.backfill_tz(None))
    assert str(bf.backfill_tz(Config(TELEGRAM_BOT_TOKEN="123456:test", TIMEZONE="Europe/London",
                                     _env_file=None))) == "Europe/London"
    raw = RawStore(str(tmp_path / "raw.db"))
    clock = Clock(at(2026, 7, 15, 2, 30))  # 02:30 on the (UTC) server is 22:30 the evening before in New York
    owner, server = RawBackfill(raw, PoliteHttp(), clock, tz=tz), RawBackfill(raw, PoliteHttp(), clock)
    assert server.in_window() and not owner.in_window()
    assert owner.run_night()["stopped"] == "window"  # the US evening slate: no backfill
    clock.ts = at(2026, 7, 15, 6, 30)  # 02:30 in New York
    assert owner.in_window() and not server.in_window() and owner.night() == "2026-07-15"
    clock.ts = at(2026, 7, 15, 10, 0)  # 06:00 in New York: closed
    assert not owner.in_window()


@pytest.mark.parametrize("utc, inside, night", [
    # spring forward, Sunday 8 March 2026: 02:00 EST becomes 03:00 EDT at 07:00 UTC (a 4-hour window)
    ((2026, 3, 8, 5, 59), False, None), ((2026, 3, 8, 6, 0), True, "2026-03-08"),
    ((2026, 3, 8, 6, 59), True, "2026-03-08"), ((2026, 3, 8, 7, 0), True, "2026-03-08"),
    ((2026, 3, 8, 9, 59), True, "2026-03-08"), ((2026, 3, 8, 10, 0), False, None),
    # fall back, Sunday 1 November 2026: 02:00 EDT becomes 01:00 EST at 06:00 UTC (a 6-hour window)
    ((2026, 11, 1, 4, 59), False, None), ((2026, 11, 1, 5, 0), True, "2026-11-01"),
    ((2026, 11, 1, 5, 59), True, "2026-11-01"), ((2026, 11, 1, 6, 0), True, "2026-11-01"),
    ((2026, 11, 1, 10, 59), True, "2026-11-01"), ((2026, 11, 1, 11, 0), False, None)])
def test_the_window_keeps_new_york_time_across_both_dst_changes(tmp_path: Path, utc_server: Any,
                                                                utc: Tuple[int, ...], inside: bool,
                                                                night: Optional[str]) -> None:
    tz = bf.backfill_tz(SimpleNamespace(timezone="America/New_York"))
    backfill = RawBackfill(RawStore(str(tmp_path / "raw.db")), PoliteHttp(), lambda: at(*utc), tz=tz)
    assert backfill.in_window() is inside
    if inside:
        assert backfill.night() == night


def test_an_invalid_time_zone_falls_back_to_server_time_with_one_warning(tmp_path: Path, utc_server: Any,
                                                                         monkeypatch: pytest.MonkeyPatch,
                                                                         caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(bf, "_tz_warned", set())
    with caplog.at_level(logging.WARNING, logger=bf.__name__):
        for name in ("Mars/Olympus", "Mars/Olympus", "America", "../../etc/passwd"):
            assert bf.backfill_tz(SimpleNamespace(timezone=name)) is None
    warned = [record.getMessage() for record in caplog.records if "TIMEZONE" in record.getMessage()]
    assert len([text for text in warned if "Mars/Olympus" in text]) == 1 and len(warned) == 3
    backfill = RawBackfill(RawStore(str(tmp_path / "raw.db")), PoliteHttp(), lambda: at(2026, 7, 15, 2, 30),
                           tz=bf.backfill_tz(SimpleNamespace(timezone="Mars/Olympus")))
    assert backfill.tz is None and backfill.in_window()  # the server's own 02:30 (UTC here)


def test_an_outage_never_leaves_a_hole(tmp_path: Path, short: Any) -> None:
    """F2 (the reviewer's checks/c09): a board answering 503 all night is tried again later, never given up; the
    season isn't done without it, and the next night it is read with its games."""
    short(((10, 20), (10, 30)), NBA)
    env = Env(tmp_path, [NBA], seasons=1)
    day = datetime.date(2026, 10, 27)
    env.world.outage[("board", "20261027")] = at(2026, 11, 2, 0, 0)  # down the rest of this night
    backfill = env.backfill()
    report = backfill.run_night()
    while report["stopped"] != "window":  # the worker's ticks through the night
        assert report["stopped"] in ("backoff", "done"), report
        env.clock.ts += 300
        report = backfill.run_night()
    cursor = env.raw.get_cursor(NBA, 2026)
    assert cursor["given_up"] == 0 and cursor["skipped"] == []
    assert [entry[:2] for entry in cursor["retry"]] == [["board", "2026-10-27"]]
    assert env.raw.progress(NBA)[0]["done"] is False  # never done with a hole
    tries = len([call for call in env.world.of("board") if call[2] == "20261027"])
    assert 3 <= tries <= 6  # 10 minutes apart, then 20, 40, 80, 160: never hammered
    everything_else = [d for d in days(datetime.date(2026, 10, 20), datetime.date(2026, 10, 30)) if d != "20261027"]
    assert all(env.raw.latest(scoreboard_key(NBA, d), "espn_site", "scoreboard") for d in everything_else)
    # The next night ESPN answers: the board and its games come in, and the season is done
    env.clock.ts = at(2026, 11, 2, 1, 30)
    report = backfill.run_night()
    assert report["stopped"] == "done", report
    assert env.raw.latest(scoreboard_key(NBA, "20261027"), "espn_site", "scoreboard") is not None
    assert all(env.raw.has_final(f"espn:{game}") for game in event_ids(NBA, day)) and event_ids(NBA, day)
    cursor = env.raw.get_cursor(NBA, 2026)
    assert cursor["retry"] == [] and env.raw.progress(NBA)[0]["done"] is True


def test_raw_first_uses_a_stored_summary_only_once_it_has_settled(tmp_path: Path) -> None:
    """F4 (the reviewer's checks/c12): a final summary in raw.db stands for the learn backfill's own fetch only if
    it was last seen SETTLE_HOURS after the game's start; an earlier copy is fetched again, and ESPN's correction is
    written (and kept in raw.db as a new version)."""
    early = json.loads((LEARN_FIX / "espn_nba_summary.json").read_text(encoding="utf-8"))
    corrected = copy.deepcopy(early)
    for stat in corrected["boxscore"]["teams"][0]["statistics"]:
        if stat["name"] == "fieldGoalsMade-fieldGoalsAttempted":
            stat["displayValue"] = "41-82"  # ESPN's later correction of the away side's field goals
    start = bf.summary_start(early)
    assert start == at(2026, 10, 25, 21, 0)
    events = [entry.rsplit("/", 1)[-1].split("?")[0]
              for entry in (item["$ref"] for item in json.loads((LEARN_FIX / "espn_nba_events.json").read_text())
                            ["items"])]

    def fgm(store: Store) -> set:
        db = sqlite3.connect(store.path)
        try:
            return {row[0] for row in db.execute("SELECT value FROM hist_stats WHERE team_side = 'away' AND stat = "
                                                 "'fgm'")}
        finally:
            db.close()

    def raw_with(*seen: float) -> RawStore:
        raw = RawStore(str(tmp_path / f"raw-{'-'.join(str(int(s - start)) for s in seen)}.db"))
        for event_id in events:
            for when in seen:  # the same early body, seen at each of these times
                bf.keep_summary(raw, "basketball_nba", event_id, json.dumps(early).encode(), fetched_at=when)
        return raw

    # Forward capture's copy, 3 hours after the start: fetched again, the corrected record written
    raw = raw_with(start + 3 * 3600)
    store, transport, report, _ = learn_run(tmp_path, "early", raw, summary=json.dumps(corrected).encode())
    assert report["stopped"] == "done" and len(transport.summaries()) == 4 and fgm(store) == {41.0}
    newest = raw.latest(f"espn:{events[0]}", "espn_site", "summary")
    assert newest is not None and newest.json() == corrected and raw.payload_count() == 8  # a new version each
    # A copy seen 40 hours after the start has settled: no request, its record is written
    for seen in ((start + 40 * 3600,), (start + 3 * 3600, start + 40 * 3600)):  # (fetched then, or seen again then)
        store, transport, report, _ = learn_run(tmp_path, f"settled-{len(seen)}", raw_with(*seen),
                                                summary=json.dumps(corrected).encode())
        assert report["stopped"] == "done" and transport.summaries() == [] and fgm(store) == {40.0}


def test_sqlite_finding_raw_db_full_stops_the_backfill_with_disk(tmp_path: Path, short: Any,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """SQLite's "database or disk is full" on a store (as the reviewer's checks/c10 forces it: raw.db's pages capped)
    stops the run with "disk"; RawStore.room() is then off for FULL_PAUSE, and with room again the backfill carries
    on from its cursor, every board and game stored."""
    short(((10, 1), (10, 30)), NBA)
    env = Env(tmp_path, [NBA], seasons=1)
    assert env.backfill().run_night(budget=20)["stopped"] == "budget"
    db = sqlite3.connect(env.raw.path)
    pages = db.execute("PRAGMA page_count").fetchone()[0]
    db.close()
    real = env.raw._connect

    def capped() -> sqlite3.Connection:
        connection = real()
        connection.execute(f"PRAGMA max_page_count = {pages + 1}")
        return connection

    monkeypatch.setattr(env.raw, "_connect", capped)
    calls = len(env.world.calls)
    report = env.backfill().run_night()
    assert report["stopped"] == "disk" and len(env.world.calls) - calls <= 1  # at most the answer it couldn't keep
    assert env.raw.room() is False
    again = env.backfill().run_night()
    assert (again["ran"], again["requests"], again["stopped"]) == (False, 0, "disk")
    monkeypatch.setattr(env.raw, "_connect", real)  # room again, once the pause is over
    env.clock.ts += raw_store.FULL_PAUSE + 1
    assert env.backfill().run_night()["stopped"] == "done"
    for d in days(datetime.date(2026, 10, 1), datetime.date(2026, 10, 30)):
        assert env.raw.latest(scoreboard_key(NBA, d), "espn_site", "scoreboard") is not None, d
        for game in event_ids(NBA, datetime.datetime.strptime(d, "%Y%m%d").date()):
            assert env.raw.has_final(f"espn:{game}"), game

