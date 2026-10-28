"""History for the learning loop (v3, Contract 1): a gentle nightly backfill of old games, a point-in-time reader,
and the grader's live capture, all into the ``hist_*`` tables (learn/history_store.py).

``Backfill(store, cache, now).run_night(budget=None)``
    Spends at most ``budget`` requests a night (default LEARN_BACKFILL_PER_NIGHT, 1500), one every 2 seconds, only
    between 01:00 and 06:00 server time, newest seasons first:
    - football-data.co.uk: E0, E1, SP1, I1, D1, F1 from 2019/20 and MLS from the USA file (one request a file);
    - nflverse: games.csv (one request covers 2019 on);
    - ESPN: the NBA, NHL, college football and basketball and MLB, the last 3 seasons. Event ids by 7-day windows,
      then per game the summary (result, box stats) and the core odds (open and close). MLB uses one StatsAPI
      schedule request per season (results, innings, weather), ESPN's scoreboard per date for the ids, and ESPN odds
      only (no box scores: StatsAPI's terms are "non-bulk").
    Resumable: each unit's cursor is saved after every step (``hist_progress``) and the night's request count in
    ``hist_budget``, so a restart carries on where it stopped and never spends a night's budget twice. Every fetch
    goes through the shared TTLCache. Returns a report dict (see ``run_night``).

``HistoryReader(store).games(league, since=None, until=None)``
    Yields point-in-time records, oldest first: the game, its lines dated before the start only (``as_of < start``,
    never a post-start price), and, kept apart for grading, the result, box stats and weather.

``capture_final(store, league, row, final, summary=None, *, context=None, weather=None, now=None)``
    The grader's live capture: a finished game's result, box stats and the last pre-start lines (the consensus row's
    market sources, the context row's total line, the summary's pickcenter open/close) into the same tables, so the
    live record and the history share one shape.
"""

from __future__ import annotations

import datetime
import logging
import os
import sqlite3
import time
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from ..data.markets.aliases import team_key
from . import history_store as hs
from .sources import espn_hist, football_data, nflverse
from .sources.http import PACE, HistHttp, HttpError, Refused

log = logging.getLogger(__name__)

PER_NIGHT_DEFAULT = 1500
WINDOW_HOURS = (1, 6)        # 01:00 to 06:00 server time
SEASONS = 3                  # ESPN: the current season and the two before
CHUNK = 25                   # requests one unit spends before the next unit of the same season tier gets a turn
WINDOW_DAYS = 7              # ESPN event lists, a week at a time
SETTLE_HOURS = 36            # a day is listed once its games finished this long ago (US Eastern days)
REFRESH_SECONDS = 20 * 3600  # a season still being played is read again after this
ITEM_TRIES = 3               # a game (or a window) failing this often is skipped
STOP_AFTER_ERRORS = 10       # this many failed requests in a row stop the night (the source is down)
WRITE_BATCH = 200            # games of a file written per transaction
EASTERN = ZoneInfo("America/New_York")
SOURCES = ("nflverse", "football-data", "espn")
LIVE_SOURCE = "live"
LIVE_MARKET_KINDS = ("sharp", "us_books", "prediction_market")
# Soccer leagues played within one calendar year (their season is the year); the rest start in July/August
CALENDAR_SOCCER = ("soccer_usa_mls", "soccer_brazil", "soccer_argentina", "soccer_japan", "soccer_sweden",
                   "soccer_norway", "soccer_finland", "soccer_league_of_ireland", "soccer_chile", "soccer_conmebol")
# HistoryReader.sharp_line: the first book with a full set of prices wins (Pinnacle is never stored after its
# 2025-07-23 cut-off, so Betfair Exchange's close takes over there)
SHARP_ORDER = ("pinnacle", "betfair_ex", "median", "nflverse", "avg", "draftkings", "espn_bet", "bet365", "sharp",
               "books_fair", "us_books")


def per_night_budget(env: Optional[Mapping[str, str]] = None) -> int:
    """LEARN_BACKFILL_PER_NIGHT as a whole number of requests (default 1500; a bad value falls back to it)."""
    raw = (os.environ if env is None else env).get("LEARN_BACKFILL_PER_NIGHT", "")
    try:
        value = int(str(raw).strip())
    except ValueError:
        return PER_NIGHT_DEFAULT
    return value if value >= 0 else PER_NIGHT_DEFAULT


def _clock(now: Any) -> Callable[[], float]:
    if now is None:
        return time.time
    if callable(now):
        return lambda: float(hs.to_ts(now()) or time.time())
    fixed = hs.to_ts(now)
    if fixed is None:
        raise ValueError(f"unreadable now: {now!r}")
    return lambda: fixed


def _connect(store: Any) -> Tuple[sqlite3.Connection, bool]:
    """(an open connection to the store's file with the history tables, whether to close it after)."""
    if isinstance(store, sqlite3.Connection):
        if not store.in_transaction:
            hs.init_history_schema(store)
        return store, False
    if hasattr(store, "_connect"):
        db = store._connect()
    else:
        db = sqlite3.connect(str(getattr(store, "path", store)), timeout=10, isolation_level=None)
    hs.init_history_schema(db)
    return db, True


def season_for(league: str, start: Any) -> Optional[int]:
    """The season (its first calendar year) of a game starting at ``start``."""
    ts = hs.to_ts(start)
    if ts is None:
        return None
    day = datetime.datetime.fromtimestamp(ts, EASTERN).date()
    if league in espn_hist.LEAGUES:
        return espn_hist.current_season(league, day)
    if league == "americanfootball_nfl":
        return day.year if day.month >= 3 else day.year - 1
    if league.startswith("soccer_") and not league.startswith(CALENDAR_SOCCER):
        return day.year if day.month >= 7 else day.year - 1
    return day.year


def _keys(league: str, home: Any, away: Any) -> Tuple[Optional[str], Optional[str]]:
    return team_key(league, home), team_key(league, away)


# --- the backfill -------------------------------------------------------------------------------------------------

class _Unit:
    """One resumable piece of the backfill: a file, or one league-season of ESPN games."""

    source = ""
    cost = 1  # cheap units (one request) go first within a season tier

    def __init__(self, name: str, tier: int, current: bool) -> None:
        self.name = name
        self.tier = tier
        self.current = current

    def progress(self, db: sqlite3.Connection) -> Dict[str, Any]:
        return hs.get_progress(db, self.source, self.name) or {"status": "todo", "state": {}}

    def has_work(self, bf: "Backfill", db: sqlite3.Connection, now: float) -> bool:
        raise NotImplementedError

    def step(self, bf: "Backfill", db: sqlite3.Connection, now: float) -> int:
        """Do one piece of work (at most a few requests); returns the games written."""
        raise NotImplementedError


class _FileUnit(_Unit):
    """A whole file (a football-data league-season, the USA file, nflverse's games.csv): one request, then done
    (a finished season) or read again after REFRESH_SECONDS (a season still being played)."""

    def __init__(self, source: str, name: str, tier: int, current: bool,
                 load: Callable[[], List[Dict[str, Any]]], soccer: bool) -> None:
        super().__init__(name, tier, current)
        self.source = source
        self.load = load
        self.soccer = soccer

    def has_work(self, bf: "Backfill", db: sqlite3.Connection, now: float) -> bool:
        found = self.progress(db)
        if found["status"] == "done" or self.name in bf.failed_units:
            return False
        fetched = (found.get("state") or {}).get("fetched_at")
        return fetched is None or now - float(fetched) >= REFRESH_SECONDS

    def step(self, bf: "Backfill", db: sqlite3.Connection, now: float) -> int:
        before = bf.run_requests
        try:
            records = self.load()
        except Refused:
            raise
        except HttpError as exc:
            if exc.status != 404:
                return self._failed(bf, db, now, exc, before)
            # No file (yet): a season not published, or a league-season that doesn't exist
            hs.set_progress(db, self.source, self.name, status="current" if self.current else "done",
                            state={"fetched_at": now, "rows": 0}, requests=bf.run_requests - before,
                            note="no file published", now=now)
            bf.ok()
            return 0
        except Exception as exc:  # a timeout, a reset connection, an unreadable file: tried again next night
            return self._failed(bf, db, now, exc, before)
        bf.ok()
        written = 0
        for at in range(0, len(records), WRITE_BATCH):  # a few hundred games a transaction
            with hs.transaction(db):
                for item in records[at:at + WRITE_BATCH]:
                    written += bf.write(db, item, now, soccer=self.soccer)
        hs.set_progress(db, self.source, self.name, status="current" if self.current else "done",
                        state={"fetched_at": now, "rows": len(records)}, games=written,
                        requests=bf.run_requests - before, note="", now=now)
        return written

    def _failed(self, bf: "Backfill", db: sqlite3.Connection, now: float, exc: Exception, before: int) -> int:
        bf.failed_units.add(self.name)
        hs.set_progress(db, self.source, self.name, status="active" if self.progress(db)["status"] == "todo" else None,
                        requests=bf.run_requests - before, errors=1, note=f"failed: {type(exc).__name__}", now=now)
        bf.failed(exc)
        return 0


class _EspnUnit(_Unit):
    """One ESPN league-season: event ids a week at a time (newest days first, then back to the season's start), then
    per game its summary and odds. MLB lists one day at a time from ESPN's scoreboard instead, joins MLB's season
    schedule for the result and weather, and reads odds only."""

    source = "espn"
    cost = 2

    def __init__(self, league: str, season: int, tier: int, current: bool) -> None:
        super().__init__(f"{league}:{season}", tier, current)
        self.league = league
        self.season = season
        self.mlb = league == "baseball_mlb"

    def _span(self, now: float) -> Tuple[datetime.date, datetime.date]:
        first, last = espn_hist.season_dates(self.league, self.season)
        settled = (datetime.datetime.fromtimestamp(now, EASTERN) - datetime.timedelta(hours=SETTLE_HOURS)).date()
        return first, min(last, settled)

    def has_work(self, bf: "Backfill", db: sqlite3.Connection, now: float) -> bool:
        found = self.progress(db)
        if found["status"] == "done":
            return False
        state = found.get("state") or {}
        if state.get("queue"):
            return True
        if self._next_window(state, now) is not None:
            return True
        if not self.current and found["status"] == "current":  # a season that was being played has ended
            hs.set_progress(db, self.source, self.name, status="done", now=now)
        return False

    def _next_window(self, state: Mapping[str, Any], now: float) -> Optional[Tuple[datetime.date, datetime.date]]:
        """The next dates to list: the days after ``top`` up to the latest settled one (new games first), else the
        days before ``bottom`` back to the season's start."""
        first, latest = self._span(now)
        days = 1 if self.mlb else WINDOW_DAYS
        if latest < first:
            return None
        top = datetime.date.fromisoformat(state["top"]) if state.get("top") else None
        bottom = datetime.date.fromisoformat(state["bottom"]) if state.get("bottom") else None
        if top is None:
            return max(first, latest - datetime.timedelta(days=days - 1)), latest
        if top < latest:
            return top + datetime.timedelta(days=1), min(latest, top + datetime.timedelta(days=days))
        if bottom is not None and bottom > first:
            return max(first, bottom - datetime.timedelta(days=days)), bottom - datetime.timedelta(days=1)
        return None

    def step(self, bf: "Backfill", db: sqlite3.Connection, now: float) -> int:
        found = self.progress(db)
        state = dict(found.get("state") or {})
        queue: List[List[Any]] = [list(item) for item in state.get("queue") or []]
        before = bf.run_requests
        written = 0
        errors = 0
        note: Optional[str] = None
        if queue:
            event_id, stage, tries = queue[0][0], queue[0][1], int(queue[0][2] if len(queue[0]) > 2 else 0)
            try:
                written, next_stage = self._item(bf, db, now, str(event_id), stage)
                queue.pop(0)
                if next_stage:
                    queue.insert(0, [event_id, next_stage, 0])
                bf.ok()
            except Refused:
                raise
            except Exception as exc:  # a bad answer or a failed request: try the game again later, then skip it
                bf.failed(exc)
                errors = 1
                if tries + 1 >= ITEM_TRIES:
                    queue.pop(0)
                    note = f"skipped {event_id} ({stage}): {type(exc).__name__}"
                else:
                    queue[0] = [event_id, stage, tries + 1]
        else:
            window = self._next_window(state, now)
            if window is not None:
                a, b = window
                try:
                    if self.mlb:
                        listed, written = self._list_mlb(bf, db, now, a)
                    else:
                        listed = self._list(bf, db, now, a, b)
                    queue.extend([event_id, "odds" if self.mlb else "summary", 0] for event_id in listed)
                    state.pop("list_tries", None)
                    bf.ok()
                    moved = True
                except Refused:
                    raise
                except Exception as exc:
                    bf.failed(exc)
                    errors = 1
                    tries = int(state.get("list_tries") or 0) + 1
                    moved = tries >= ITEM_TRIES
                    state["list_tries"] = 0 if moved else tries
                    if moved:
                        note = f"skipped {a}..{b}: {type(exc).__name__}"
                if moved:
                    top = state.get("top")
                    if top is None or b.isoformat() > top:
                        state["top"] = b.isoformat()
                    if state.get("bottom") is None or a.isoformat() < state["bottom"]:
                        state["bottom"] = a.isoformat()
        state["queue"] = queue
        status = "active"
        if not queue and self._next_window(state, now) is None:
            status = "current" if self.current else "done"
        hs.set_progress(db, self.source, self.name, status=status, state=state, games=written,
                        requests=bf.run_requests - before, errors=errors, note=note, now=now)
        return written

    # -- listing --

    def _list(self, bf: "Backfill", db: sqlite3.Connection, now: float, a: datetime.date,
              b: datetime.date) -> List[str]:
        past = (datetime.datetime.fromtimestamp(now, EASTERN).date() - b).days >= 2
        ids = bf.espn.events(self.league, a, b, past=past)
        return [event_id for event_id in reversed(ids) if not bf.complete(db, self.league, f"espn:{event_id}")]

    def _list_mlb(self, bf: "Backfill", db: sqlite3.Connection, now: float,
                  day: datetime.date) -> Tuple[List[str], int]:
        past = (datetime.datetime.fromtimestamp(now, EASTERN).date() - day).days >= 2
        events = bf.espn.scoreboard(self.league, day, past=past)
        if not events:
            return [], 0
        season_games = bf.espn.mlb_season(self.season, current=self.current)
        by_teams: Dict[Tuple[Optional[str], Optional[str]], List[Dict[str, Any]]] = {}
        for game in season_games:
            if game.get("date") == day.isoformat():
                key = _keys(self.league, (game.get("home") or {}).get("name"), (game.get("away") or {}).get("name"))
                by_teams.setdefault(key, []).append(game)
        queued: List[str] = []
        written = 0
        for event in sorted(events, key=lambda e: (e.get("start") or "", e.get("id") or "")):
            if event.get("state") not in ("final", "void") or not event.get("home") or not event.get("away"):
                continue
            home, away = event["home"], event["away"]
            key = _keys(self.league, home.get("abbr") or home.get("name"), away.get("abbr") or away.get("name"))
            matches = by_teams.get(key) or []
            mlb = matches.pop(0) if matches else None
            game_key = f"espn:{event['id']}"
            record = _mlb_record(self.league, self.season, event, mlb, key)
            if record is None:
                continue
            written += bf.write(db, record, now)
            if event["state"] == "final" and not hs.has_close_lines(db, game_key):
                queued.append(str(event["id"]))
        return queued, written

    # -- one game --

    def _item(self, bf: "Backfill", db: sqlite3.Connection, now: float, event_id: str,
              stage: str) -> Tuple[int, Optional[str]]:
        key = f"espn:{event_id}"
        if stage == "summary":
            parsed = bf.espn.summary(self.league, event_id)
            if parsed.get("state") not in ("final", "void") or not parsed.get("home") or not parsed.get("away"):
                return 0, None  # not over (a suspended game): the live grader or a later night takes it
            record = _summary_record(self.league, self.season, event_id, parsed, espn_hist.SOURCE)
            if record is None:
                return 0, None
            written = bf.write(db, record, now)
            return written, ("odds" if parsed["state"] == "final" else None)
        if stage == "odds":
            game = hs.get_game(db, self.league, key)
            if game is None or game["status"] != "final":
                return 0, None
            start_ts = hs.to_ts(game["start"])
            odds = bf.espn.odds(self.league, event_id)
            lines = [{**line, "as_of": start_ts - float(line.get("lead") or espn_hist.CLOSE_LEAD)}
                     for line in odds.get("lines") or []]
            with hs.transaction(db):
                hs.add_lines(db, key, game["start"], lines, espn_hist.SOURCE)
            return 0, None
        return 0, None


def _summary_record(league: str, season: Optional[int], event_id: str, parsed: Mapping[str, Any],
                    source: str) -> Optional[Dict[str, Any]]:
    """A parsed ESPN summary as a history record."""
    home, away = parsed.get("home") or {}, parsed.get("away") or {}
    if hs.to_ts(parsed.get("start")) is None:
        return None
    state = parsed.get("state")
    hk, ak = _keys(league, home.get("abbr") or home.get("name"), away.get("abbr") or away.get("name"))
    info = {"status": parsed.get("status"), "venue": parsed.get("venue"),
            "neutral": True if parsed.get("neutral") else None, "after_extra_time": parsed.get("after_extra_time")}
    game = {"league": league, "game_key": f"espn:{event_id}", "espn_id": str(event_id), "start": parsed["start"],
            "home": home.get("name") or home.get("abbr") or "", "away": away.get("name") or away.get("abbr") or "",
            "home_score": home.get("score") if state == "final" else None,
            "away_score": away.get("score") if state == "final" else None,
            "status": "final" if state == "final" else "void", "season": season, "source": source,
            "home_key": hk or team_key(league, home.get("name")), "away_key": ak or team_key(league, away.get("name")),
            "info": {k: v for k, v in info.items() if v is not None}}
    stats = espn_hist.summary_stats(parsed) if state == "final" else []
    return {"game": game, "lines": [], "stats": stats, "weather": []}


def _mlb_record(league: str, season: int, event: Mapping[str, Any], mlb: Optional[Mapping[str, Any]],
                keys: Tuple[Optional[str], Optional[str]]) -> Optional[Dict[str, Any]]:
    """An ESPN scoreboard event joined to MLB's season schedule game (result, innings, announced weather)."""
    if hs.to_ts(event.get("start")) is None:
        return None
    state = event.get("state")
    home, away = event.get("home") or {}, event.get("away") or {}
    home_score, away_score = home.get("score"), away.get("score")
    info: Dict[str, Any] = {"status": event.get("status")}
    stats: List[Dict[str, Any]] = []
    weather: List[Dict[str, Any]] = []
    if mlb is not None:
        info.update({"mlb_pk": mlb.get("pk"), "game_number": mlb.get("game_number"),
                     "double_header": mlb.get("double_header"), "day_night": mlb.get("day_night"),
                     "venue": mlb.get("venue")})
        if mlb.get("state") == "final":
            home_score = (mlb.get("home") or {}).get("score", home_score)
            away_score = (mlb.get("away") or {}).get("score", away_score)
        for side in ("home", "away"):
            for index, runs in enumerate((mlb.get("innings") or {}).get(side) or [], start=1):
                if runs is not None:
                    stats.append({"team_side": side, "stat": f"period_{index}", "value": runs})
        conditions = mlb.get("weather") or {}
        temp_f, wind_mph = conditions.get("temp_f"), conditions.get("wind_mph")
        if temp_f is not None or wind_mph is not None:
            weather.append({"kind": "actual",
                            "temp_c": None if temp_f is None else round((float(temp_f) - 32) * 5 / 9, 1),
                            "wind_kmh": None if wind_mph is None else round(float(wind_mph) * 1.609344, 1),
                            "precip_mm": None})
            info["weather"] = {k: conditions.get(k) for k in ("condition", "wind_text", "roof_closed")}
    else:
        info["no_mlb_match"] = True
    game = {"league": league, "game_key": f"espn:{event['id']}", "espn_id": str(event["id"]),
            "start": event["start"], "home": home.get("name") or home.get("abbr") or "",
            "away": away.get("name") or away.get("abbr") or "",
            "home_score": home_score if state == "final" else None,
            "away_score": away_score if state == "final" else None,
            "status": "final" if state == "final" else "void", "season": season, "source": espn_hist.SOURCE,
            "home_key": keys[0], "away_key": keys[1], "info": {k: v for k, v in info.items() if v is not None}}
    return {"game": game, "lines": [], "stats": stats, "weather": weather}


class Backfill:
    """The nightly backfill (see the module's docstring). ``store`` is a data.store.Store (or a path, or an open
    connection), ``cache`` the shared TTLCache, ``now`` a clock (a function giving epoch seconds or a datetime) or a
    fixed time. ``http`` replaces the HTTP door (tests hand in a HistHttp over fixtures), ``sleep`` the 2-second
    wait, ``tz`` the server's time zone (default: its local time), ``per_night`` LEARN_BACKFILL_PER_NIGHT,
    ``sources``/``leagues`` limit the work."""

    def __init__(self, store: Any, cache: Any, now: Any = None, *, http: Optional[HistHttp] = None,
                 sleep: Optional[Callable[[float], None]] = None, tz: Optional[datetime.tzinfo] = None,
                 per_night: Optional[int] = None, sources: Optional[Sequence[str]] = None,
                 leagues: Optional[Sequence[str]] = None, seasons: int = SEASONS, chunk: int = CHUNK) -> None:
        self.store = store
        self.cache = cache
        self.clock = _clock(now)
        self.tz = tz
        self.per_night = per_night_budget() if per_night is None else max(0, int(per_night))
        self.sources = tuple(sources) if sources else SOURCES
        self.leagues = set(leagues) if leagues else None
        self.seasons = max(1, int(seasons))
        self.chunk = max(1, int(chunk))
        self.http = http if http is not None else HistHttp(pace=PACE, sleep=sleep)
        self.http.allow = self._allow
        self.http.on_request = self._counted
        self.fd = football_data.FootballData(self.http, cache)
        self.nfl = nflverse.NflVerse(self.http, cache)
        self.espn = espn_hist.EspnHistory(self.http, cache)
        self.run_requests = 0
        self.run_games = 0  # new games written this run (kept when the budget or the window stops it)
        self._db: Optional[sqlite3.Connection] = None
        self._night = ""
        self._budget = 0
        self._used = 0
        self._errors_in_row = 0
        self._by_source: Dict[str, int] = {}
        self._source = ""
        self.failed_units: set = set()  # files that failed this run: tried again next night

    # -- time and budget --

    def local(self, ts: float) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(ts, self.tz) if self.tz is not None else datetime.datetime.fromtimestamp(ts)

    def in_window(self, ts: Optional[float] = None) -> bool:
        """Whether ``ts`` (default now) is between 01:00 and 06:00 server time."""
        hour = self.local(self.clock() if ts is None else ts).hour
        return WINDOW_HOURS[0] <= hour < WINDOW_HOURS[1]

    def night(self, ts: Optional[float] = None) -> str:
        """The night a time belongs to: the server-local date of its 01:00-06:00 window."""
        return self.local(self.clock() if ts is None else ts).date().isoformat()

    def _allow(self) -> None:
        if self._db is None:
            raise Refused("the backfill is not running")
        if self._used >= self._budget:
            raise Refused("budget")
        if not self.in_window():
            raise Refused("window")

    def _counted(self, url: str) -> None:
        self.run_requests += 1
        self._by_source[self._source] = self._by_source.get(self._source, 0) + 1
        if self._db is not None:
            self._used = hs.add_budget(self._db, self._night, 1, self.clock())

    def ok(self) -> None:
        self._errors_in_row = 0

    def failed(self, exc: BaseException) -> None:
        self._errors_in_row += 1
        log.warning("history backfill: %s", exc)
        if self._errors_in_row >= STOP_AFTER_ERRORS:
            raise Refused("errors")

    # -- writing --

    def complete(self, db: sqlite3.Connection, league: str, game_key: str) -> bool:
        """Whether a game needs nothing more: void, or final with closing lines."""
        game = hs.get_game(db, league, game_key)
        if game is None:
            return False
        return game["status"] == "void" or hs.has_close_lines(db, game_key)

    def write(self, db: sqlite3.Connection, record: Mapping[str, Any], now: float, soccer: bool = False) -> int:
        """Write a record (1 if its game is new). A football-data game already stored under ESPN's id (by the live
        grader) is written under that key, and an earlier football-data row of it is moved there."""
        game = dict(record["game"])
        if soccer and not str(game["game_key"]).startswith("espn:"):
            match = hs.find_match(db, game["league"], game["start"], game.get("home_key"), game.get("away_key"),
                                  prefix="espn:")
            if match:
                hs.rekey(db, game["league"], game["game_key"], match)
                game["game_key"] = match
        new = hs.get_game(db, game["league"], game["game_key"]) is None
        hs.write_record(db, {**record, "game": game}, now)
        return 1 if new else 0

    # -- the work --

    def units(self, now: Optional[float] = None) -> List[_Unit]:
        """Every unit of work, in order: season tier (newest first), then cheap units, then ESPN's."""
        ts = self.clock() if now is None else now
        today = datetime.datetime.fromtimestamp(ts, EASTERN).date()
        found: List[_Unit] = []
        wanted = (lambda league: self.leagues is None or league in self.leagues)
        if "nflverse" in self.sources and wanted(nflverse.LEAGUE):
            found.append(_FileUnit("nflverse", "games.csv", 0, True, lambda: self.nfl.records(nflverse.FIRST_SEASON),
                                   soccer=False))
        if "football-data" in self.sources:
            current = today.year if today.month >= 7 else today.year - 1
            for div, league in football_data.DIVISIONS.items():
                if not wanted(league):
                    continue
                for season in range(current, football_data.FIRST_SEASON - 1, -1):
                    found.append(_FileUnit("football-data", f"{div}:{season}", current - season, season == current,
                                           (lambda d=div, s=season, c=(season == current):
                                            self.fd.season_records(d, s, c)), soccer=True))
            for country, (league, _name) in football_data.NEW_LEAGUES.items():
                if wanted(league):
                    found.append(_FileUnit("football-data", f"new:{country}", 0, True,
                                           (lambda c=country: self.fd.new_records(c, football_data.FIRST_SEASON)),
                                           soccer=True))
        if "espn" in self.sources:
            for league in espn_hist.LEAGUES:
                if not wanted(league):
                    continue
                current = espn_hist.current_season(league, today)
                for back in range(self.seasons):
                    found.append(_EspnUnit(league, current - back, back, back == 0))
        return sorted(found, key=lambda unit: (unit.tier, unit.cost))

    def run_night(self, budget: Optional[int] = None) -> Dict[str, Any]:
        """Spend up to ``budget`` requests tonight (default LEARN_BACKFILL_PER_NIGHT, counted across runs of the
        same night). Returns {"ran", "night", "budget", "used" (tonight, all runs), "requests" (this run), "games"
        (new games written), "stopped" ("done", "budget", "window", "errors" or why it didn't run),
        "sources": {source: requests this run}, "progress": progress()}."""
        now = self.clock()
        total = self.per_night if budget is None else max(0, int(budget))
        report: Dict[str, Any] = {"ran": False, "night": self.night(now), "budget": total, "used": 0,
                                  "requests": 0, "games": 0, "stopped": None, "sources": {}}
        if not self.in_window(now):
            report["stopped"] = "outside the 01:00-06:00 window"
            return report
        db, close = _connect(self.store)
        self._db, self._night, self._budget = db, self.night(now), total
        self._used = hs.budget_used(db, self._night)
        self.run_requests, self._errors_in_row, self._by_source, self.failed_units = 0, 0, {}, set()
        self.run_games = 0
        try:
            report["ran"] = True
            if self._used >= total:
                report["stopped"] = "budget"
                return report
            try:
                self._run(db)
                report["stopped"] = "done"
            except Refused as exc:
                report["stopped"] = str(exc) or "budget"
            report["progress"] = progress(db)
        finally:
            # Games written before a stop (the budget or the window) count too
            report.update({"used": self._used, "requests": self.run_requests, "games": self.run_games,
                           "sources": dict(self._by_source)})
            self._db = None
            if close:
                db.close()
        return report

    def _run(self, db: sqlite3.Connection) -> int:
        units = self.units()
        for tier in sorted({unit.tier for unit in units}):
            group = [unit for unit in units if unit.tier == tier]
            while True:
                moved = False
                for unit in group:
                    spent = self.run_requests
                    steps = 0
                    while self.run_requests - spent < self.chunk and steps < 10 * self.chunk:
                        now = self.clock()
                        if not unit.has_work(self, db, now):
                            break
                        if not self.in_window(now):
                            raise Refused("window")
                        if self._used >= self._budget:
                            raise Refused("budget")
                        self._source = unit.source
                        self.run_games += unit.step(self, db, now)
                        steps += 1
                        moved = True
                if not moved:
                    break
        return self.run_games


def progress(store: Any) -> Dict[str, Dict[str, Any]]:
    """Backfill progress per source for the progress page's Health block: {source: {"units", "done", "current",
    "active", "todo", "games", "requests", "errors", "updated_at", "last_night", "last_night_requests"}}."""
    db, close = _connect(store)
    try:
        found: Dict[str, Dict[str, Any]] = {}
        for row in hs.progress_rows(db):
            entry = found.setdefault(row["source"], {"units": 0, "done": 0, "current": 0, "active": 0, "todo": 0,
                                                     "games": 0, "requests": 0, "errors": 0, "updated_at": None})
            entry["units"] += 1
            entry[row["status"]] = entry.get(row["status"], 0) + 1
            for key in ("games", "requests", "errors"):
                entry[key] += int(row[key] or 0)
            if row["updated_at"] and (entry["updated_at"] is None or row["updated_at"] > entry["updated_at"]):
                entry["updated_at"] = row["updated_at"]
        last = db.execute("SELECT night, used FROM hist_budget ORDER BY night DESC LIMIT 1").fetchone()
        for entry in found.values():
            entry["last_night"], entry["last_night_requests"] = (last[0], last[1]) if last else (None, 0)
        return found
    finally:
        if close:
            db.close()


# --- the point-in-time reader -------------------------------------------------------------------------------------

class HistoryReader:
    """Reads the history tables. ``games(league, since, until)`` yields point-in-time records, oldest first:

        {"league", "game_key", "espn_id", "start" (ISO UTC), "start_ts", "season", "status", "source", "home", "away",
         "home_key", "away_key", "info",
         "lines": [{"kind", "book", "market", "side", "point", "price", "as_of", "source"}],  # as_of < start only
         "result": {"home", "away"} | None,                    # the outcome: for grading, never an input
         "stats": {"home": {stat: value}, "away": {...}},     # the outcome's box stats
         "weather": {"forecast_24h": {...}, "actual": {...}}}  # actual = the game-time observation

    ``since``/``until`` (start >= since, start < until) take ISO text, epoch seconds, dates or datetimes."""

    BATCH = 500

    def __init__(self, store: Any) -> None:
        self.store = store

    def games(self, league: str, since: Any = None, until: Any = None, *, status: Optional[str] = "final"
              ) -> Iterator[Dict[str, Any]]:
        db, close = _connect(self.store)
        try:
            low = hs.to_ts(since)
            high = hs.to_ts(until)
            rows = hs.games_between(db, league, None if low is None else hs.iso(low),
                                    None if high is None else hs.iso(high))
            if status:
                rows = [row for row in rows if row["status"] == status]
            for at in range(0, len(rows), self.BATCH):
                batch = rows[at:at + self.BATCH]
                keys = [row["game_key"] for row in batch]
                lines = hs.lines_for(db, keys)
                stats = hs.stats_for(db, keys)
                weather = hs.weather_for(db, keys)
                for row in batch:
                    yield self._record(row, lines.get(row["game_key"]) or [], stats.get(row["game_key"]),
                                       weather.get(row["game_key"]))
        finally:
            if close:
                db.close()

    def game(self, league: str, game_key: str) -> Optional[Dict[str, Any]]:
        db, close = _connect(self.store)
        try:
            row = hs.get_game(db, league, game_key)
            if row is None:
                return None
            keys = [game_key]
            return self._record(row, hs.lines_for(db, keys)[game_key], hs.stats_for(db, keys)[game_key],
                                hs.weather_for(db, keys)[game_key])
        finally:
            if close:
                db.close()

    def coverage(self) -> Dict[str, Dict[str, Any]]:
        """Per league: games, final, void, with closing lines, with stats, first and last start."""
        db, close = _connect(self.store)
        try:
            return hs.counts(db)
        finally:
            if close:
                db.close()

    @staticmethod
    def _record(row: Mapping[str, Any], lines: Sequence[Mapping[str, Any]], stats: Optional[Mapping[str, Any]],
                weather: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        start_ts = hs.to_ts(row["start"]) or 0.0
        before = [dict(line) for line in lines if line.get("as_of") is not None and float(line["as_of"]) < start_ts]
        result = None
        if row["status"] == "final" and row["home_score"] is not None and row["away_score"] is not None:
            result = {"home": row["home_score"], "away": row["away_score"]}
        return {"league": row["league"], "game_key": row["game_key"], "espn_id": row["espn_id"], "start": row["start"],
                "start_ts": start_ts, "season": row["season"], "status": row["status"], "source": row["source"],
                "home": row["home"], "away": row["away"], "home_key": row["home_key"], "away_key": row["away_key"],
                "info": dict(row["info"] or {}), "lines": before, "result": result,
                "stats": {"home": dict((stats or {}).get("home") or {}), "away": dict((stats or {}).get("away") or {})},
                "weather": dict(weather or {})}

    @staticmethod
    def sharp_line(record: Mapping[str, Any], market: str = "h2h", kind: str = "close",
                   order: Sequence[str] = SHARP_ORDER) -> Optional[Dict[str, Any]]:
        """The baseline price set of a record: the first book of ``order`` with a full set for ``market`` (h2h:
        home and away, plus the draw for soccer; spreads/totals: both sides at one point). Returns {"book",
        "point", "prices": {side: decimal odds}, "as_of"} or None. Pinnacle first; from its 2025-07-23 cut-off it
        isn't stored, so Betfair Exchange's close (then the market average) is the sharp baseline."""
        soccer = str(record.get("league") or "").startswith("soccer_")
        sides = {"h2h": ("home", "draw", "away") if soccer else ("home", "away"), "spreads": ("home", "away"),
                 "totals": ("over", "under")}.get(market)
        if sides is None:
            return None
        by_book: Dict[str, Dict[str, Mapping[str, Any]]] = {}
        for line in record.get("lines") or []:
            if line.get("market") == market and line.get("kind") == kind:
                by_book.setdefault(str(line.get("book")), {})[str(line.get("side"))] = line
        names = list(order) + sorted(set(by_book) - set(order))
        for book in names:
            found = by_book.get(book) or {}
            if not all(side in found for side in sides):
                continue
            if market == "totals" and found["over"].get("point") != found["under"].get("point"):
                continue
            if market == "spreads" and (found["home"].get("point") is None or found["away"].get("point") is None
                                        or abs(float(found["home"]["point"]) + float(found["away"]["point"])) > 1e-9):
                continue
            return {"book": book, "point": found[sides[0]].get("point"),
                    "prices": {side: float(found[side]["price"]) for side in sides},
                    "as_of": max(float(found[side]["as_of"]) for side in sides)}
        return None

    @staticmethod
    def no_vig(prices: Mapping[str, float]) -> Dict[str, float]:
        """Decimal prices as margin-free chances (multiplicative: each 1/price over their sum)."""
        inverse = {side: 1.0 / float(price) for side, price in prices.items() if price and float(price) > 1.0}
        total = sum(inverse.values())
        return {side: value / total for side, value in inverse.items()} if total > 0 else {}


# --- live capture (the grader) ------------------------------------------------------------------------------------

def _slug(name: Any) -> str:
    return "_".join(str(name or "").strip().lower().replace("-", " ").split()) or "unknown"


def _consensus_lines(row: Mapping[str, Any], start_ts: float) -> List[Dict[str, Any]]:
    """The consensus row's market sources (ok or thin) as fair prices 1/p (margin-free), close, at their as_of."""
    lines = []
    swapped = bool(row.get("swapped"))
    for source in row.get("sources") or []:
        if not isinstance(source, Mapping) or source.get("kind") not in LIVE_MARKET_KINDS:
            continue
        if source.get("status") not in ("ok", "thin"):
            continue
        as_of = hs.to_ts(source.get("as_of")) or hs.to_ts(row.get("logged_at"))
        if as_of is None or as_of >= start_ts:
            continue
        for side, chance in (source.get("p") or {}).items():
            p = hs.num(chance)
            if p is None or not 0 < p < 1 or side not in ("home", "away", "draw"):
                continue
            if swapped and side != "draw":
                side = "away" if side == "home" else "home"
            lines.append({"kind": "close", "book": _slug(source.get("name")), "market": "h2h", "side": side,
                          "point": None, "price": round(1.0 / p, 4), "as_of": as_of})
    return lines


def _context_lines(context: Optional[Mapping[str, Any]], start_ts: float) -> List[Dict[str, Any]]:
    """The context row's total line and the books' margin-free chance of the over, as fair prices."""
    if not context:
        return []
    line, p = hs.num(context.get("line")), hs.num(context.get("p_before"))
    as_of = hs.to_ts(context.get("logged_at"))
    if line is None or p is None or not 0 < p < 1 or as_of is None or as_of >= start_ts:
        return []
    return [{"kind": "close", "book": "books_fair", "market": "totals", "side": "over", "point": line,
             "price": round(1.0 / p, 4), "as_of": as_of},
            {"kind": "close", "book": "books_fair", "market": "totals", "side": "under", "point": line,
             "price": round(1.0 / (1.0 - p), 4), "as_of": as_of}]


def _weather_rows(rows: Sequence[Mapping[str, Any]], start_ts: float) -> List[Dict[str, Any]]:
    """weather_log rows of the game as hist_weather: the forecast as it stood a day out, and MLB's announcement."""
    forecast, announced = None, None
    for row in sorted(rows or [], key=lambda r: float(r.get("fetched_at") or 0)):
        fetched = hs.num(row.get("fetched_at"))
        conditions = row.get("conditions") or {}
        if fetched is None or fetched >= start_ts or not isinstance(conditions, Mapping):
            continue
        if fetched <= start_ts - 24 * 3600 and conditions.get("temp_c") is not None:
            forecast = conditions
        if isinstance(conditions.get("announced"), Mapping):
            announced = conditions["announced"]
    found = []
    if forecast is not None:
        found.append({"kind": "forecast_24h", "temp_c": forecast.get("temp_c"), "wind_kmh": forecast.get("wind_kmh"),
                      "precip_mm": forecast.get("precip_mm")})
    if announced is not None:
        temp_f, wind = hs.num(announced.get("temp_f")), hs.num(announced.get("wind_mph"))
        found.append({"kind": "actual", "temp_c": None if temp_f is None else round((temp_f - 32) * 5 / 9, 1),
                      "wind_kmh": None if wind is None else round(wind * 1.609344, 1), "precip_mm": None})
    return found


def _logged_weather(db: sqlite3.Connection, league: str, keys: Sequence[str]) -> List[Dict[str, Any]]:
    try:
        rows = []
        for key in keys:
            for fetched, conditions in db.execute("SELECT fetched_at, conditions FROM weather_log WHERE league = ? "
                                                  "AND game_key = ?", (league, key)):
                rows.append({"fetched_at": fetched, "conditions": hs.loads(conditions) or {}})
        return rows
    except sqlite3.Error:
        return []  # no weather_log table in this file


def _logged_context(db: sqlite3.Connection, league: str, game_key: str) -> Optional[Dict[str, Any]]:
    try:
        found = db.execute("SELECT line, p_before, logged_at FROM context_log WHERE league = ? AND game_key = ?",
                           (league, game_key)).fetchone()
    except sqlite3.Error:
        return None  # no context_log table in this file
    return {"line": found[0], "p_before": found[1], "logged_at": found[2]} if found else None


def capture_final(store: Any, league: str, row: Mapping[str, Any], final: Mapping[str, Any],
                  summary: Any = None, *, context: Optional[Mapping[str, Any]] = None,
                  weather: Optional[Sequence[Mapping[str, Any]]] = None,
                  now: Optional[float] = None) -> Dict[str, int]:
    """Write a graded game into the history tables (the grader calls this; the live record and the history then
    share one shape).

    ``row`` is the game's consensus_log row (league, game_key, espn_id, home, away, start, swapped, logged_at,
    sources) or any row with those names; ``final`` is ``EspnClient.game_result`` (state "final" or "void", home
    and away in ESPN's orientation, soccer's 90-minute score); ``summary`` the game's ESPN summary (the cached lean
    one: header, boxscore, pickcenter) for the box stats and the books' open and close; ``context`` the game's
    context_log row (its total line; looked up by the row's league and game_key when not given). ``weather``
    (weather_log rows) is looked up when not given. Rows are written in ESPN's orientation under ``espn:<id>`` (a football-data, nflverse or MLB row of the same game is moved there).
    Lines dated at or after the start are never written. Returns {"games", "lines", "stats", "weather"} written
    (zeros for a game not over)."""
    counts = {"games": 0, "lines": 0, "stats": 0, "weather": 0}
    state = (final or {}).get("state")
    if state not in ("final", "void"):
        return counts
    stamp = float(now if now is not None else time.time())
    espn_id = (final or {}).get("espn_id") or row.get("espn_id")
    key = f"espn:{espn_id}" if espn_id else str(row.get("game_key") or "")
    if not key:
        return counts
    parsed = espn_hist.parse_summary(summary, league) if isinstance(summary, Mapping) else None
    swapped = bool(row.get("swapped"))
    home, away = (row.get("away"), row.get("home")) if swapped else (row.get("home"), row.get("away"))
    if parsed and parsed.get("home") and parsed.get("away"):
        home = parsed["home"].get("name") or home
        away = parsed["away"].get("name") or away
    start = (parsed or {}).get("start") or row.get("start")
    start_ts = hs.to_ts(start)
    if start_ts is None:
        return counts
    hk, ak = _keys(league, home, away)
    if parsed and parsed.get("home") and parsed.get("away"):
        hk = hk or team_key(league, parsed["home"].get("abbr"))
        ak = ak or team_key(league, parsed["away"].get("abbr"))
    info = {"status": final.get("status"), "after_extra_time": final.get("after_extra_time"),
            "venue": (parsed or {}).get("venue"), "neutral": True if (parsed or {}).get("neutral") else None,
            "captured": "grader"}
    game = {"league": league, "game_key": key, "espn_id": str(espn_id) if espn_id else None, "start": start,
            "home": home or "", "away": away or "", "status": state, "season": season_for(league, start),
            "home_score": final.get("home") if state == "final" else None,
            "away_score": final.get("away") if state == "final" else None, "source": LIVE_SOURCE,
            "home_key": hk, "away_key": ak, "info": {k: v for k, v in info.items() if v is not None}}
    lines = _consensus_lines(row, start_ts)
    if isinstance(summary, Mapping):
        lines += [{**line, "as_of": start_ts - float(line.get("lead") or espn_hist.CLOSE_LEAD)}
                  for line in espn_hist.parse_pickcenter(summary.get("pickcenter"))]
    stats = espn_hist.summary_stats(parsed) if parsed and state == "final" else []
    db, close = _connect(store)
    try:
        if context is None and row.get("game_key"):
            context = _logged_context(db, league, str(row["game_key"]))
        lines += _context_lines(context, start_ts)
        if weather is None:
            keys = [k for k in (str(row.get("event_id") or ""), str(row.get("game_key") or ""), key) if k]
            weather = _logged_weather(db, league, list(dict.fromkeys(keys)))
        weather_rows = _weather_rows(weather or [], start_ts)
        with hs.transaction(db):
            for prefix in ("fd:", "nflverse:", "mlb:"):
                match = hs.find_match(db, league, start_ts, hk, ak, prefix=prefix, exclude=key)
                if match:
                    hs.rekey(db, league, match, key, espn_id=str(espn_id) if espn_id else None)
            new = hs.get_game(db, league, key) is None
            written = hs.write_record(db, {"game": game, "lines": lines, "stats": stats, "weather": weather_rows},
                                      stamp)
        counts.update(written)
        counts["games"] = 1 if new else 0
        return counts
    finally:
        if close:
            db.close()


__all__ = ["Backfill", "HistoryReader", "capture_final", "progress", "per_night_budget", "season_for",
           "PER_NIGHT_DEFAULT", "WINDOW_HOURS", "SHARP_ORDER"]
