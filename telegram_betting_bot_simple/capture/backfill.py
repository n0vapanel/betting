"""The backfill of older games into raw.db (v4 P1, the owner's "start the older games backfill now"): the full raw
ESPN record of past seasons, so P2 and later can recount and learn every event from history, not only the trimmed
results, box stats and lines the learn backfill keeps (learn/history.py).

``RawBackfill(raw, http, clock, *, leagues=None, seasons=3, per_night=6000, window=(1, 6), tz=None,
forward_due=None).run_night(budget=None) -> report``

**Which games.** For each league, the season still being played (up to the newest settled day: a US Eastern day
whose games finished SETTLE_HOURS ago, as the learn backfill; forward capture's sweep and final jobs hold the newer
days) and the last ``seasons`` completed seasons. Newest season first: every league's newest season in wave order
(plan section 1.3: NBA, NFL, the ten wave-1 soccer leagues, then NHL, UFC and MLB), then every league's season before,
and so on; within a season the newest week first.

**Season windows** (``WINDOWS``; a season is named by the calendar year it starts in):

- NBA and NHL 1 Oct - 30 Jun, MLB 15 Mar - 15 Nov: the learn backfill's (learn/sources/espn_hist.LEAGUES).
- NFL 1 Sep - 20 Feb: Week 1 in the week after Labor Day to the Super Bowl in early or mid February (no preseason).
- The nine European leagues 1 Aug - 31 May: opening weekends from early August (the Scottish Premiership's first),
  the last rounds, play-offs and relegation play-offs by the end of May.
- The Champions League 1 Sep - 15 Jun: the league phase from mid-September, the final in late May or early June.
- UFC the calendar year: cards most weeks, all year.

**Per 7-day window**, newest first: the window's one-day scoreboards, newest day first (ESPN's site API answers a
``dates`` range with HTTP 400, so a window is 7 boards), each kept raw under ``scoreboard_key(league, "YYYYMMDD")``
(the key the worker's sweep uses); then, for every game a board shows played to the end, the site ``summary``
untrimmed and, for the wave-1 soccer leagues, the core plays, every page. UFC reads its season's cards from one
board's ``leagues[0].calendar``, then for each past card its day's board and the card's ``fightcenter``
(``espn:<cardEventId>``). Each NFL season also gets nflverse's play-by-play file once
(``static:nflverse:pbp:<season>``). Never MLB StatsAPI (its terms are non-bulk) and never The Odds API.

**Budget, disk and politeness.** Every request goes through the shared PoliteHttp (its per-host token buckets)
inside ``http.guard``, whose check refuses a request outside the window (01:00-06:00 in ``tz``: the worker passes
``backfill_tz(config)``, the owner's TIMEZONE), while the learn backfill runs (learn/history.LEARN_BACKFILL_RUNNING:
the two never read ESPN at the same time), while a forward job is due (``forward_due``), while raw.db's disk is short
of room (``raw.room()``: the ~4 GB backfill never fills the server), while the request's host backs off (the backfill
never waits one out), past this call's ``budget``, or once ``raw.spend("backfill", night, 1, limit=per_night)`` finds
the night's ``per_night`` spent.

**Resume and repeat runs.** Each (league, season)'s cursor is saved in ``backfill_progress`` after every step (one
request at most: a board, a summary, one plays page, a fightcenter, a file), so a restart carries on at the same
board, game or page. A game whose final summary (UFC: fightcenter; soccer's plays: every page) is already in raw.db is
not fetched again, and a stored board whose games are all over is read back from raw.db, so a repeat run costs
nothing.

**Failures never leave a hole.** Only ESPN's definitive answers (a 4xx other than 429) count towards ITEM_TRIES, after
which a board or game is given up (listed in its cursor). A transient failure (a 5xx, a 429, an unreachable host, an
answer that can't be read) puts the board or game in the season's retry list, tried again after RETRY_WAIT (doubling,
up to RETRY_WAIT_CAP: a later night for a long outage) while the walk goes on, and never given up; a season with a
retry still waiting is not done. A host backing off leaves the item where it is.

**Forward capture first.** The worker calls ``run_night(budget=50)`` between its forward jobs in its capture thread;
the check before each request (and before each step) stops the call at once when a forward job is due (``stopped``
"forward"), and the next call resumes from the cursor.

**The learn backfill's raw-first hook** (learn/history.py) uses ``stored_summary`` and ``keep_summary``: a final
summary already in raw.db is read from there, and one it fetches is kept here too.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

from ..learn.history import LEARN_BACKFILL_RUNNING
from ..learn.sources import espn_hist
from . import plan
from .http import Backoff, Refused, Unreachable
from .raw_store import disk_is_full, utc_day
from .sources import RawFetch, espn_core, espn_fightcenter, espn_key, espn_site, nflverse, scoreboard_key

log = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")  # ESPN's days (data/espn_client.ESPN_DAY_TZ)
NFL = "americanfootball_nfl"
UFC = plan.UFC
UCL = "soccer_uefa_champs_league"
ORDER: Tuple[str, ...] = plan.WAVE1 + plan.WAVE2  # the wave order (plan section 1.3)
PLAYS_LEAGUES = frozenset(plan.WAVE1_SOCCER)      # core plays, every page
PER_NIGHT_DEFAULT = 6000   # CAPTURE_BACKFILL_PER_NIGHT
SEASONS_DEFAULT = 3        # CAPTURE_BACKFILL_SEASONS: completed seasons, besides the one being played
WINDOW_HOURS = (1, 6)      # 01:00 to 06:00 in the backfill's time zone (backfill_tz: TIMEZONE, New York by default)
WINDOW_DAYS = 7            # a window is 7 one-day boards
SETTLE_HOURS = 36          # a day is read once its games finished this long ago (learn/history.py's)
ITEM_TRIES = 3             # a board, game or page ESPN definitively refuses (a 4xx, not 429) this often is given up
RETRY_WAIT = 10 * 60.0     # a transient failure (5xx, 429, unreachable) waits this long before it is tried again,
RETRY_WAIT_CAP = 6 * 3600.0  # doubling each time up to this (a long outage: a later night); never given up
DEFAULT_TZ = "America/New_York"  # config's TIMEZONE default: the night window's time zone (backfill_tz)
STOP_AFTER_ERRORS = 10     # failures of one kind in a row (boards, summaries, plays, fightcenters) that stop that
                           # kind for the night: ESPN is down or has changed, and the rest waits for a later night
SKIPPED_KEPT = 100         # given-up items a cursor lists (the newest; "given_up" counts them all)
CALENDAR_REFRESH = 7 * 86400.0  # a season still being played reads UFC's calendar again after this
CURSOR_VERSION = 1
# A status name that means the game was not played to the end (its final record comes later, or never)
VOID_WORDS = ("POSTPONED", "CANCELED", "CANCELLED", "SUSPENDED", "ABANDONED", "DELAYED")

# league -> (first (month, day), last (month, day)) of a season named by the calendar year it starts in
WINDOWS: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    **{league: (first, last) for league, (_sport, _path, first, last) in espn_hist.LEAGUES.items()
       if league in ORDER},
    NFL: ((9, 1), (2, 20)),
    **{league: ((8, 1), (5, 31)) for league in plan.WAVE1_SOCCER if league != UCL},
    UCL: ((9, 1), (6, 15)),
    UFC: ((1, 1), (12, 31)),
}

# The projected total (logged once): games in a whole season (UFC: cards, the Contender Series' included), with the
# preseason and spring-training games inside the windows; compressed kilobytes per final game (plan section 9.4: the
# summary, soccer's core plays apart; P1 measures the real sizes), per board and per nflverse file
GAMES: Dict[str, int] = {
    "basketball_nba": 1390, NFL: 286, "soccer_epl": 380, "soccer_efl_champ": 557, "soccer_spain_la_liga": 380,
    "soccer_germany_bundesliga": 308, "soccer_italy_serie_a": 380, "soccer_france_ligue_one": 308,
    "soccer_spl": 228, "soccer_netherlands_eredivisie": 306, "soccer_portugal_primeira_liga": 306, UCL: 189,
    "icehockey_nhl": 1400, UFC: 52, "baseball_mlb": 2625}
GAME_KB: Dict[str, float] = {"basketball": 60, "football": 65, "soccer": 45, "hockey": 55, "mma": 40,
                             "baseball": 90}
PLAYS_PAGES = 5            # a match's core plays (1,481 items is 5 pages of 300)
PLAYS_KB = 180.0           # all of them
BOARD_KB: Dict[str, float] = {"basketball": 15, "football": 5, "soccer": 3, "hockey": 15, "mma": 8,
                              "baseball": 20}
PBP_KB = 20 * 1024.0       # play_by_play_<season>.csv.gz

_EVENT_REF = re.compile(r"/events/(\d+)")
_projected_lock = threading.Lock()
_projected = False  # the projected total is logged once a process


# --- seasons ---------------------------------------------------------------------------------------------------------

def season_window(league: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    try:
        return WINDOWS[league]
    except KeyError:
        raise ValueError(f"the backfill has no season window for {league!r}") from None


def season_dates(league: str, season: int) -> Tuple[datetime.date, datetime.date]:
    """The first and last day of ``season`` (the calendar year it starts in)."""
    (m1, d1), (m2, d2) = season_window(league)
    return datetime.date(season, m1, d1), datetime.date(season + (1 if (m2, d2) < (m1, d1) else 0), m2, d2)


def settled_day(now: float) -> datetime.date:
    """The newest day the backfill reads: the US Eastern day of ``now`` less SETTLE_HOURS."""
    return (datetime.datetime.fromtimestamp(now, EASTERN) - datetime.timedelta(hours=SETTLE_HOURS)).date()


def seasons_for(league: str, settled: datetime.date, completed: int
                ) -> List[Tuple[int, datetime.date, datetime.date, bool]]:
    """(season, first day, last day, still being played), newest first: the season being played up to ``settled``
    (when one is), then the last ``completed`` completed ones."""
    (m1, d1), _end = season_window(league)
    latest = settled.year if (settled.month, settled.day) >= (m1, d1) else settled.year - 1
    first, last = season_dates(league, latest)
    found: List[Tuple[int, datetime.date, datetime.date, bool]] = []
    if last > settled:
        found.append((latest, first, settled, True))
        latest -= 1
    for season in range(latest, latest - max(0, int(completed)), -1):
        a, b = season_dates(league, season)
        found.append((season, a, b, False))
    return found


@dataclass(frozen=True)
class Unit:
    """One league-season of the walk: its days ``first``..``last`` (``current``: still being played)."""

    league: str
    season: int
    first: datetime.date
    last: datetime.date
    current: bool
    tier: int  # 0: the league's newest season in scope
    rank: int  # the league's place in the wave order


# --- reading ESPN's answers ------------------------------------------------------------------------------------------

def completed(kind: Any) -> bool:
    """Whether an ESPN status type is a game played to the end (not postponed, cancelled, suspended, abandoned)."""
    if not isinstance(kind, Mapping):
        return False
    name = str(kind.get("name") or "").upper()
    return bool(kind.get("completed")) and not any(word in name for word in VOID_WORDS)


def _settled(kind: Any) -> bool:
    """Whether nothing more happens to the game that day: over, postponed or cancelled."""
    return isinstance(kind, Mapping) and (bool(kind.get("completed")) or kind.get("state") == "post")


def event_status(event: Mapping[str, Any], card: bool = False) -> Mapping[str, Any]:
    """A board event's status type: its first competition's (a team game), or the event's own first for an MMA card
    (``card``: a card's first bout can be over while the card is not)."""
    competitions = event.get("competitions")
    competition = competitions[0] if isinstance(competitions, list) and competitions else None
    holders = (event, competition) if card else (competition, event)
    for holder in holders:
        status = holder.get("status") if isinstance(holder, Mapping) else None
        kind = status.get("type") if isinstance(status, Mapping) else None
        if isinstance(kind, Mapping):
            return kind
    return {}


def board_events(data: Any) -> List[Mapping[str, Any]]:
    """A scoreboard's events that have an id."""
    events = data.get("events") if isinstance(data, Mapping) else None
    return [event for event in events if isinstance(event, Mapping) and event.get("id") not in (None, "")] \
        if isinstance(events, list) else []


def _when(text: Any) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None


def calendar_cards(data: Any, season: int) -> Optional[List[Tuple[str, str]]]:
    """A UFC board's calendar of ``season`` as [(card day "YYYY-MM-DD", US Eastern; card event id)], newest first;
    None when the board has no calendar of that season with event links."""
    leagues = data.get("leagues") if isinstance(data, Mapping) else None
    league = leagues[0] if isinstance(leagues, list) and leagues and isinstance(leagues[0], Mapping) else {}
    info = league.get("season")
    year = info.get("year") if isinstance(info, Mapping) else None
    calendar = league.get("calendar")
    if not isinstance(calendar, list) or str(year) != str(season):
        return None
    cards = set()
    for entry in calendar:
        if not isinstance(entry, Mapping):
            continue
        ref = entry.get("event")
        match = _EVENT_REF.search(str(ref.get("$ref") or "")) if isinstance(ref, Mapping) else None
        start = _when(entry.get("startDate"))
        if match and start is not None and start.tzinfo is not None:
            cards.add((start.astimezone(EASTERN).date().isoformat(), match.group(1)))
    return sorted(cards, reverse=True) if cards else None


def summary_state(body: Any) -> Tuple[str, Optional[str]]:
    """(state, horizon) a summary is kept with: ("final", "final") for a game played to the end, ("live", None) while
    it is on, ("pre", None) otherwise (not started, postponed, cancelled, unreadable), so that only a played game's
    summary counts as its final record (RawStore.has_final)."""
    try:
        data = json.loads(body) if isinstance(body, (bytes, bytearray, str)) else body
    except ValueError:
        return "pre", None
    header = data.get("header") if isinstance(data, Mapping) else None
    competitions = header.get("competitions") if isinstance(header, Mapping) else None
    competition = competitions[0] if isinstance(competitions, list) and competitions else None
    status = competition.get("status") if isinstance(competition, Mapping) else None
    kind = status.get("type") if isinstance(status, Mapping) else None
    if completed(kind):
        return "final", "final"
    if isinstance(kind, Mapping) and kind.get("state") == "in":
        return "live", None
    return "pre", None


# --- the night window's time zone ----------------------------------------------------------------------------------

_tz_warned: Set[str] = set()
_tz_lock = threading.Lock()


def backfill_tz(config: Any = None) -> Optional[datetime.tzinfo]:
    """The time zone the backfill's night window (01:00-06:00) is read in (v4 P1 decision 1): the owner's TIMEZONE
    (``config.timezone``, default America/New_York), the quietest night for US sports and European soccer alike
    whatever the server's own clock says (a UTC server's 01:00-06:00 is New York's evening slate). A name that isn't a
    time zone means server-local time (None), with one warning. The worker builds RawBackfill with it, and the health
    block keys "the backfill's last night" with it."""
    name = getattr(config, "timezone", None) if config is not None else None
    text = str(name).strip() if name is not None and str(name).strip() else DEFAULT_TZ
    try:
        return ZoneInfo(text)
    except Exception:  # noqa: BLE001 - ZoneInfoNotFoundError, ValueError (a path), IsADirectoryError ("America")
        with _tz_lock:
            first = text not in _tz_warned
            _tz_warned.add(text)
        if first:
            log.warning("TIMEZONE %r isn't a time zone name: the older-games backfill's night window keeps the "
                        "server's own time", text[:60])
        return None


def summary_start(data: Any) -> Optional[float]:
    """A summary's game start (header.competitions[0].date) in epoch seconds, or None."""
    header = data.get("header") if isinstance(data, Mapping) else None
    competitions = header.get("competitions") if isinstance(header, Mapping) else None
    competition = competitions[0] if isinstance(competitions, list) and competitions else None
    when = _when(competition.get("date")) if isinstance(competition, Mapping) else None
    return when.timestamp() if when is not None and when.tzinfo is not None else None


# --- the learn backfill's raw-first hook (learn/history.py) ---------------------------------------------------------

def stored_summary(raw: Any, event_id: Any) -> Optional[Any]:
    """A game's settled final summary from raw.db, parsed, or None: the newest 2xx copy, kept once the game was final
    and saying so, and last seen at least SETTLE_HOURS after the game's start (its ``seen_at``: a later fetch of the
    same body counts), as the learn backfill reads a game only once late corrections are in. An earlier copy (forward
    capture's, 30 minutes after the whistle) is fetched again, and a corrected body is kept as a new version."""
    payload = raw.latest(espn_key(event_id), "espn_site", "summary")
    if payload is None or payload.state != "final":
        return None
    try:
        data = payload.json()
    except ValueError:
        return None
    if summary_state(data)[0] != "final":
        return None
    start = summary_start(data)  # payload.seen_at: its newest sighting (RawStore keeps every dedup hit)
    return data if start is not None and float(payload.seen_at) >= start + SETTLE_HOURS * 3600.0 else None


def keep_summary(raw: Any, league: str, event_id: Any, body: bytes, *, fetched_at: Optional[float] = None,
                 status: int = 200) -> bool:
    """Keep a summary's raw body in raw.db under ``espn:<eventId>`` (its state from what it says); whether a row was
    added."""
    state, horizon = summary_state(body)
    _row, new = raw.insert(game_key=espn_key(event_id), league=league, source=espn_site.SOURCE, endpoint="summary",
                           params={}, body=body, state=state, http_status=status, horizon=horizon,
                           fetched_at=fetched_at)
    return new


# --- the backfill ----------------------------------------------------------------------------------------------------

class _Blocked(Exception):
    """The unit's host is backing off: the unit waits for a later call (the backfill never waits one out)."""


def _day(text: str) -> datetime.date:
    return datetime.date.fromisoformat(text)


def _ymd(day: datetime.date) -> str:
    return day.strftime("%Y%m%d")


class RawBackfill:
    """The backfill of older games into raw.db (see the module's docstring). ``raw`` is the RawStore, ``http`` the
    shared PoliteHttp, ``clock`` gives epoch seconds; ``leagues`` limits the work (wave order is kept), ``seasons``
    is how many completed seasons, ``per_night`` the night's request cap, ``window`` the (start, end) hours it runs
    in, read in ``tz`` (the worker passes backfill_tz(config): the owner's TIMEZONE, New York by default; None means
    the server's local time) and ``forward_due`` a no-argument callable that is true while a forward capture job is
    due (default: ``raw.next_due()`` against the clock)."""

    def __init__(self, raw: Any, http: Any, clock: Optional[Callable[[], float]] = None, *,
                 leagues: Optional[Sequence[str]] = None, seasons: int = SEASONS_DEFAULT,
                 per_night: int = PER_NIGHT_DEFAULT, window: Tuple[int, int] = WINDOW_HOURS,
                 tz: Optional[datetime.tzinfo] = None, forward_due: Optional[Callable[[], Any]] = None) -> None:
        self.raw = raw
        self.http = http
        self.clock = clock or time.time
        unknown = sorted(set(leagues or ()) - set(ORDER))
        if unknown:
            raise ValueError(f"the backfill covers the wave 1-2 leagues ({', '.join(ORDER)}), not {unknown}")
        self.leagues = tuple(ORDER) if leagues is None else tuple(league for league in ORDER if league in set(leagues))
        self.seasons = max(0, int(seasons))
        self.per_night = max(0, int(per_night))
        self.window = (int(window[0]), int(window[1]))
        self.tz = tz
        self.forward_due = forward_due if forward_due is not None else self._jobs_due
        self._streaks: Dict[str, int] = {}      # failures in a row per kind, kept across the night's calls
        self._halted: Set[str] = set()          # kinds STOP_AFTER_ERRORS stopped for the night
        self._streak_night = ""
        self._night = ""
        self._cap = 0
        self._league = ""
        self._reset_counts()
        self._log_projection()

    # -- time --

    def local(self, ts: float) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(ts, self.tz) if self.tz is not None else \
            datetime.datetime.fromtimestamp(ts)

    def in_window(self, ts: Optional[float] = None) -> bool:
        """Whether ``ts`` (default now) is inside the night window (01:00-06:00 by default, read in ``tz``)."""
        hour = self.local(self.clock() if ts is None else ts).hour
        start, end = self.window
        return start <= hour < end if start <= end else (hour >= start or hour < end)

    def night(self, ts: Optional[float] = None) -> str:
        """The night a time belongs to (its budget's day): the date, in ``tz``, of the window it falls in."""
        moment = self.local(self.clock() if ts is None else ts)
        start, end = self.window
        if start > end and moment.hour < end:  # a window across midnight belongs to the day it starts
            moment -= datetime.timedelta(days=1)
        return moment.date().isoformat()

    # -- the work --

    def units(self, now: Optional[float] = None) -> List[Unit]:
        """Every league-season in scope, in the walk's order: newest season first, leagues in wave order."""
        day = settled_day(self.clock() if now is None else now)
        found = [Unit(league, season, first, last, current, tier, ORDER.index(league))
                 for league in self.leagues
                 for tier, (season, first, last, current) in enumerate(seasons_for(league, day, self.seasons))]
        return sorted(found, key=lambda unit: (unit.tier, unit.rank))

    def projection(self, now: Optional[float] = None) -> Dict[str, Any]:
        """The projected requests and size of the whole backfill in scope: {"requests", "gb", "nights",
        "leagues": {league: {"seasons", "requests", "mb"}}} (from GAMES, GAME_KB, BOARD_KB...: estimates)."""
        leagues: Dict[str, Dict[str, Any]] = {}
        for unit in self.units(now):
            sport = plan.sport_of(unit.league) or ""
            whole_first, whole_last = season_dates(unit.league, unit.season)
            days = (unit.last - unit.first).days + 1
            games = GAMES[unit.league] * days / ((whole_last - whole_first).days + 1)
            if unit.league == UFC:
                boards = games + 1  # each card's day, and the board the calendar comes from
                requests = boards + games
                kb = boards * BOARD_KB[sport] + games * GAME_KB[sport]
            else:
                pages = PLAYS_PAGES if unit.league in PLAYS_LEAGUES else 0
                boards = days
                requests = boards + games * (1 + pages) + (1 if unit.league == NFL else 0)
                kb = (boards * BOARD_KB[sport] + games * (GAME_KB[sport] + (PLAYS_KB if pages else 0))
                      + (PBP_KB if unit.league == NFL else 0))
            entry = leagues.setdefault(unit.league, {"seasons": 0, "requests": 0.0, "mb": 0.0})
            entry["seasons"] += 1
            entry["requests"] += requests
            entry["mb"] += kb / 1024.0
        for entry in leagues.values():
            entry["requests"] = int(round(entry["requests"]))
            entry["mb"] = round(entry["mb"], 1)
        requests = sum(entry["requests"] for entry in leagues.values())
        gb = round(sum(entry["mb"] for entry in leagues.values()) / 1024.0, 2)
        nights = round(requests / self.per_night, 1) if self.per_night else None
        return {"requests": requests, "gb": gb, "nights": nights, "leagues": leagues}

    def _log_projection(self) -> None:
        global _projected
        with _projected_lock:
            if _projected:
                return
            _projected = True
        try:
            found = self.projection()
        except Exception:  # noqa: BLE001 - a projection never stops the backfill
            log.exception("Couldn't project the raw backfill's size")
            return
        each = "; ".join(f"{league} {entry['requests']:,} requests {entry['mb'] / 1024.0:.2f} GB"
                         for league, entry in found["leagues"].items())
        log.info("Raw backfill of older games: projected %s requests and %.1f GB (%s completed seasons and the one "
                 "being played, %d leagues), about %s nights at %s a night. %s", f"{found['requests']:,}",
                 found["gb"], self.seasons, len(found["leagues"]), found["nights"], f"{self.per_night:,}", each)

    def progress(self) -> List[Dict[str, Any]]:
        """backfill_progress for the leagues here: {"league", "season", "done", "games_seen", "games_stored",
        "updated_at", "day" (the board day last read), "days_read", "given_up", "errors"}."""
        found = []
        for row in self.raw.progress():
            if row["league"] not in self.leagues:
                continue
            cur = row.get("cursor") if isinstance(row.get("cursor"), Mapping) else {}
            found.append({key: row.get(key) for key in ("league", "season", "done", "games_seen", "games_stored",
                                                        "updated_at")}
                         | {"day": cur.get("day"), "days_read": cur.get("days_read", 0),
                            "given_up": cur.get("given_up", 0), "errors": cur.get("errors", 0)})
        return found

    # -- one night --

    def run_night(self, budget: Optional[int] = None) -> Dict[str, Any]:
        """Make at most ``budget`` requests now (None: the rest of tonight's ``per_night``; the night's cap holds
        across calls and restarts). Returns {"ran", "night", "budget", "per_night", "used" (tonight, every call),
        "requests" (this call), "stored" (new raw rows), "skipped" (games, pages and files found in raw.db),
        "failed" (failed requests), "given_up" (after definitive answers), "deferred" (transient failures put in the
        retry list), "stopped" ("done", "budget", "window", "forward", "learn" (the learn backfill is running),
        "disk" (raw.db's disk is short of room, or SQLite found it full), "backoff" (a host backing off, or retries
        still waiting) or
        "errors" (a kind of request failing STOP_AFTER_ERRORS times in a row tonight)), "leagues" ({league:
        requests this call}), "progress" (``progress()``, when it ran)}."""
        now = self.clock()
        night = self.night(now)
        cap = self.per_night if budget is None else max(0, int(budget))
        self._reset_counts()
        if night != self._streak_night:
            self._streaks, self._halted, self._streak_night = {}, set(), night
        report: Dict[str, Any] = {"ran": False, "night": night, "budget": cap, "per_night": self.per_night,
                                  "used": 0, "requests": 0, "stored": 0, "skipped": 0, "failed": 0, "given_up": 0,
                                  "deferred": 0, "stopped": None, "leagues": {}}
        if not self.in_window(now):
            report["stopped"] = "window"
        elif LEARN_BACKFILL_RUNNING.is_set():
            report["stopped"] = "learn"
        elif self._due():
            report["stopped"] = "forward"
        elif not self._room():
            report["stopped"] = "disk"
        else:
            report["ran"] = True
            self._night, self._cap = night, cap
            try:
                with self.http.guard(self._allow):
                    report["stopped"] = self._run()
            except Refused as exc:
                report["stopped"] = str(exc) or "budget"
            except sqlite3.OperationalError as exc:  # raw.db can't store more: RawStore.room() is off for FULL_PAUSE
                if not disk_is_full(exc):
                    raise
                report["stopped"] = "disk"  # the step's cursor wasn't moved: its item is fetched again with room
        report.update({"used": self.raw.spent("backfill", night), "requests": self._requests,
                       "stored": self._stored, "skipped": self._skipped, "failed": self._failed_requests,
                       "given_up": self._given_up, "deferred": self._deferred, "leagues": dict(self._by_league)})
        if report["ran"]:
            report["progress"] = self.progress()
        return report

    def _reset_counts(self) -> None:
        self._requests = self._stored = self._skipped = self._failed_requests = self._given_up = 0
        self._deferred = self._waiting = 0
        self._by_league: Dict[str, int] = {}

    def _jobs_due(self) -> bool:
        next_due = getattr(self.raw, "next_due", None)
        if not callable(next_due):
            return False
        due = next_due()
        return due is not None and float(due) <= self.clock()

    def _due(self) -> bool:
        """Whether a forward job is due (a failing check counts as due: forward capture always comes first)."""
        try:
            return bool(self.forward_due())
        except Exception:  # noqa: BLE001
            log.exception("The raw backfill couldn't tell whether a forward job is due; it stops for now")
            return True

    def _room(self) -> bool:
        """Whether raw.db's disk has room for more (RawStore.room, cached by the store; a failing check is no
        room)."""
        try:
            return bool(self.raw.room())
        except Exception:  # noqa: BLE001
            log.exception("The raw backfill couldn't check raw.db's disk; it stops for now")
            return False

    def _allow(self, url: str) -> None:
        """PoliteHttp's guard: called before every request the backfill makes."""
        if not self.in_window():
            raise Refused("window")
        if LEARN_BACKFILL_RUNNING.is_set():
            raise Refused("learn")
        if self._due():
            raise Refused("forward")
        if not self._room():
            raise Refused("disk")
        backing_off = getattr(self.http, "backing_off", None)
        if callable(backing_off) and backing_off(url):
            raise Refused("backoff")
        if self._requests >= self._cap:
            raise Refused("budget")
        if not self.raw.spend("backfill", self._night, 1, limit=self.per_night):
            raise Refused("budget")
        self._requests += 1
        self._by_league[self._league] = self._by_league.get(self._league, 0) + 1

    def _run(self) -> str:
        blocked = 0
        for unit in self.units():
            self._league = unit.league
            cur = self._cursor(unit)
            try:
                while True:
                    before = self._requests
                    if not self._step(unit, cur):
                        break
                    if self._requests == before:  # a step without a request: the guard didn't look
                        if not self.in_window():
                            raise Refused("window")
                        if LEARN_BACKFILL_RUNNING.is_set():
                            raise Refused("learn")
                        if self._due():
                            raise Refused("forward")
            except _Blocked:
                blocked += 1
        if self._halted:
            return "errors"
        return "backoff" if blocked or self._waiting else "done"

    # -- the cursor --

    def _cursor(self, unit: Unit) -> Dict[str, Any]:
        found = self.raw.get_cursor(unit.league, unit.season)
        cur: Dict[str, Any] = dict(found) if isinstance(found, Mapping) and found.get("v") == CURSOR_VERSION else {}
        cur.setdefault("v", CURSOR_VERSION)
        for key in ("top", "bottom", "window", "next", "day"):
            cur.setdefault(key, None)
        cur.setdefault("queue", [])
        cur.setdefault("retry", [])
        for key in ("days_read", "errors", "given_up", "tries"):
            cur.setdefault(key, 0)
        cur.setdefault("skipped", [])
        return cur

    def _save(self, unit: Unit, cur: Dict[str, Any], *, done: Optional[bool] = None, seen: int = 0,
              stored: int = 0) -> None:
        cur["span"] = [unit.first.isoformat(), unit.last.isoformat()]
        self.raw.set_cursor(unit.league, unit.season, cur, done=done, add_seen=seen, add_stored=stored,
                            now=self.clock())

    # -- one step (one request at most) --

    def _step(self, unit: Unit, cur: Dict[str, Any]) -> bool:
        """Do the unit's next piece of work (one request at most); False when it has none now."""
        if unit.league == NFL and cur.get("pbp") is not True and cur.get("pbp_night") != self._night:
            self._pbp(unit, cur)
            return True
        window = cur.get("window")
        if window and _day(cur["next"]) >= _day(window[0]):  # the window's boards first, the newest day first
            self._board(unit, cur, _day(cur["next"]), _day(window[0]))
            return True
        if cur["queue"]:  # then its games
            self._item(unit, cur)
            return True
        if window:
            self._close_window(unit, cur)
            return True
        due = self._due_retry(cur)  # then what a transient failure held up, once its wait is over
        if due is not None:
            self._retry(unit, cur, due)
            return True
        if unit.league == UFC and cur.get("cards") != "days":
            return self._ufc_step(unit, cur)
        found = self._next_window(unit, cur)
        if found is None:
            return self._finish(unit, cur)
        cur["window"], cur["next"] = [found[0].isoformat(), found[1].isoformat()], found[1].isoformat()
        self._save(unit, cur)
        return True

    def _next_window(self, unit: Unit, cur: Mapping[str, Any]) -> Optional[Tuple[datetime.date, datetime.date]]:
        """The next 7 days to read: the newest week first, back to the season's first day; for a season still being
        played, the days settled since come first (``top``: the newest day read; ``bottom``: the oldest)."""
        if unit.last < unit.first:
            return None
        if cur.get("top") is None:
            return max(unit.first, unit.last - datetime.timedelta(days=WINDOW_DAYS - 1)), unit.last
        top, bottom = _day(cur["top"]), _day(cur["bottom"])
        if top < unit.last:
            return top + datetime.timedelta(days=1), min(unit.last, top + datetime.timedelta(days=WINDOW_DAYS))
        if bottom > unit.first:
            return max(unit.first, bottom - datetime.timedelta(days=WINDOW_DAYS)), bottom - datetime.timedelta(days=1)
        return None

    def _close_window(self, unit: Unit, cur: Dict[str, Any]) -> None:
        first, last = cur["window"]
        if cur.get("top") is None or last > cur["top"]:
            cur["top"] = last
        if cur.get("bottom") is None or first < cur["bottom"]:
            cur["bottom"] = first
        cur["window"] = cur["next"] = None
        self._save(unit, cur)

    def _finish(self, unit: Unit, cur: Dict[str, Any]) -> bool:
        """The unit has nothing left now: a completed season is done, unless a retry still waits (a transient failure
        is never a hole); one still being played waits for new days."""
        if self._pending(unit, cur):
            self._waiting += 1
            return False
        done = not unit.current
        if cur.get("finished") != done:
            cur["finished"] = done
            self._save(unit, cur, done=done)
        return False

    # -- fetching, keeping, failing --

    def _fetch(self, call: Callable[[], RawFetch]) -> Tuple[Optional[RawFetch], Optional[str]]:
        """One request: (its answer, None), or (None, why) when the host can't be reached. A host backing off blocks
        the unit (the item stays where it is); the guard's other refusals (budget, window, a forward job due, the
        learn backfill running, the disk) stop the call."""
        try:
            return call(), None
        except Refused as exc:
            if str(exc) == "backoff":
                raise _Blocked(str(exc)) from exc
            raise
        except Backoff as exc:
            raise _Blocked(str(exc)) from exc
        except Unreachable as exc:
            self._count(False, self.clock())
            return None, f"unreachable ({type(exc).__name__})"

    def _count(self, ok: bool, ts: float) -> None:
        """Count a request's outcome under the backfill's own kinds (backfill_http_ok / backfill_http_error: v4 P1
        decision 3, forward capture keeps http_ok / http_error), unless the http door counts every request itself:
        the worker's shared PoliteHttp does (its on_result), and counting here too would count each request twice."""
        if getattr(self.http, "on_result", None) is None:
            self.raw.spend("backfill_http_ok" if ok else "backfill_http_error", utc_day(ts))

    def _keep(self, league: str, key: str, fetch: RawFetch) -> bool:
        """Store an answer (an error answer too: the health block lists them); whether a row was added."""
        state: str = "final"
        horizon: Optional[str] = "final"
        if fetch.source == nflverse.SOURCE:
            state, horizon = "static", None
        elif fetch.endpoint == "summary" and fetch.ok:
            state, horizon = summary_state(fetch.body)
        _row, new = self.raw.insert(game_key=key, league=league, source=fetch.source, endpoint=fetch.endpoint,
                                    params=fetch.params, body=fetch.body, state=state, http_status=fetch.status,
                                    horizon=horizon, fetched_at=fetch.fetched_at)
        self._count(fetch.ok, fetch.fetched_at)
        self._stored += int(new)
        return new

    def _ok(self, kind: str) -> None:
        self._streaks[kind] = 0

    @staticmethod
    def _definitive(fetch: Optional[RawFetch]) -> bool:
        """Whether a failure is ESPN's definitive answer for that URL: a 4xx other than 429. Everything else (a 5xx, a
        429, an unreachable host, an answer that can't be read) is transient."""
        return fetch is not None and 400 <= int(fetch.status) < 500 and int(fetch.status) != 429

    def _failure(self, cur: Dict[str, Any], kind: str, what: str, why: str) -> None:
        """Note a failed request (the cursor's errors, the kind's run of failures)."""
        self._failed_requests += 1
        self._streaks[kind] = self._streaks.get(kind, 0) + 1
        cur["errors"] = int(cur.get("errors") or 0) + 1
        cur["last_error"] = f"{what}: {why}"[:200]

    def _give_up(self, cur: Dict[str, Any], what: str, why: str, tries: int) -> bool:
        """After a definitive failure: whether this was the item's ITEM_TRIES-th (it is then given up, and listed in
        the cursor's "skipped")."""
        if tries + 1 < ITEM_TRIES:
            return False
        self._given_up += 1
        cur["given_up"] = int(cur.get("given_up") or 0) + 1
        cur["skipped"] = (list(cur.get("skipped") or []) + [[what, why]])[-SKIPPED_KEPT:]
        return True

    def _defer(self, cur: Dict[str, Any], kind: str, ident: str, tries: int, page: Optional[int],
               waits: int) -> None:
        """After a transient failure: the item waits in the season's retry list, never given up: RETRY_WAIT, doubling
        with each wait up to RETRY_WAIT_CAP (a long outage: a later night). The walk goes on meanwhile."""
        waits = int(waits or 0) + 1
        due = self.clock() + min(RETRY_WAIT_CAP, RETRY_WAIT * 2 ** (waits - 1))
        cur.setdefault("retry", []).append([kind, ident, int(tries or 0), page, due, waits])
        self._deferred += 1

    def _check_streak(self, kind: str) -> None:
        """After a step: a kind that has now failed STOP_AFTER_ERRORS times in a row stops for the night."""
        if self._streaks.get(kind, 0) < STOP_AFTER_ERRORS:
            return
        if kind not in self._halted:
            self._halted.add(kind)
            log.warning("The raw backfill stops its %s requests for tonight: %d failed in a row", kind,
                        self._streaks[kind])
        raise _Blocked(f"{kind} requests are failing")

    def _halt_check(self, kind: str) -> None:
        """Before a request of ``kind``: none while that kind is stopped for the night."""
        if kind in self._halted:
            raise _Blocked(f"{kind} requests are failing")

    @staticmethod
    def _why(fetch: Optional[RawFetch], error: Optional[str]) -> str:
        if fetch is not None and not fetch.ok:
            return f"HTTP {fetch.status}"
        return error or "unreadable"

    # -- the retry list (transient failures) --

    def _pending(self, unit: Unit, cur: Mapping[str, Any]) -> bool:
        """Whether the season still waits for something a transient failure held up: a retry, the NFL's file, UFC's
        calendar. Such a season is never marked done."""
        return (bool(cur.get("retry")) or (unit.league == NFL and cur.get("pbp") is not True)
                or (unit.league == UFC and cur.get("cards") is None and bool(cur.get("calendar_wait"))))

    def _due_retry(self, cur: Mapping[str, Any]) -> Optional[int]:
        """The first retry whose wait is over, or None."""
        now = self.clock()
        for index, entry in enumerate(cur.get("retry") or []):
            if len(entry) > 4 and float(entry[4] or 0) <= now:
                return index
        return None

    def _retry(self, unit: Unit, cur: Dict[str, Any], index: int) -> None:
        """A retry whose wait is over: a board is read again now; a game's item goes back to the head of the queue
        (the next step fetches it)."""
        entry = list(cur["retry"][index])
        kind, ident = str(entry[0]), str(entry[1])
        tries, page, waits = int(entry[2] or 0), entry[3], int(entry[5] or 0) if len(entry) > 5 else 0
        if kind != "board":
            del cur["retry"][index]
            cur["queue"].insert(0, [kind, ident, tries, page, waits])
            self._save(unit, cur)
            return
        day = _day(ident)
        data, status, why = self._read_board(unit, cur, day, day)
        del cur["retry"][index]
        seen = stored = 0
        if status == "ok":
            seen, stored = self._queue_finals(unit, cur, data)
        elif status == "transient":
            self._defer(cur, "board", ident, tries, None, waits)
        elif not self._give_up(cur, f"board {ident}", why, tries):
            cur["retry"].append(["board", ident, tries + 1, None, self.clock(), waits])  # due again at once
        self._save(unit, cur, seen=seen, stored=stored)
        self._check_streak("board")

    # -- boards --

    def _stored_board(self, key: str, card: bool) -> Optional[Any]:
        """A board already in raw.db whose games are all over (the sweep's or an earlier run's), or None."""
        payload = self.raw.latest(key, espn_site.SOURCE, "scoreboard")
        if payload is None:
            return None
        try:
            data = payload.json()
        except ValueError:
            return None
        if not isinstance(data, Mapping):
            return None
        return data if all(_settled(event_status(event, card)) for event in board_events(data)) else None

    def _read_board(self, unit: Unit, cur: Dict[str, Any], day: datetime.date,
                    first: datetime.date) -> Tuple[Optional[Any], str, str]:
        """(the day's board, "ok", ""), or (None, "definitive" (a 4xx) or "transient" (a 5xx, a 429, an unreachable
        host, an answer that can't be read), why). A board already stored costs no request."""
        key = scoreboard_key(unit.league, _ymd(day))
        data = self._stored_board(key, unit.league == UFC)
        if data is not None:
            self._skipped += 1
            return data, "ok", ""
        self._halt_check("board")
        boards = espn_site.scoreboard_days(self.http, unit.league, first, day, newest_first=True)
        try:
            fetch, error = self._fetch(lambda: next(boards))  # the window's newest board not read yet
        finally:
            boards.close()
        data = None
        if fetch is not None:
            self._keep(unit.league, key, fetch)
            if fetch.ok:
                try:
                    data = fetch.json()
                except ValueError:
                    data = None
        if isinstance(data, Mapping):
            self._ok("board")
            return data, "ok", ""
        why = self._why(fetch, error)
        self._failure(cur, "board", f"board {day.isoformat()}", why)
        return None, ("definitive" if self._definitive(fetch) else "transient"), why

    def _board(self, unit: Unit, cur: Dict[str, Any], day: datetime.date, first: datetime.date) -> None:
        """Read one board of the window and queue what its finished games still need. A board failing transiently
        waits in the retry list while the walk goes on; a definitive failure is tried again at once, and given up
        after ITEM_TRIES."""
        data, status, why = self._read_board(unit, cur, day, first)
        seen = stored = 0
        if status == "ok":
            seen, stored = self._queue_finals(unit, cur, data)
        elif status == "transient":
            self._defer(cur, "board", day.isoformat(), 0, None, 0)
        if status == "definitive":
            tries = int(cur.get("tries") or 0)
            moved = self._give_up(cur, f"board {day.isoformat()}", why, tries)
            cur["tries"] = 0 if moved else tries + 1
        else:
            moved, cur["tries"] = True, 0
        if moved:
            cur["next"] = (day - datetime.timedelta(days=1)).isoformat()
            cur["day"] = day.isoformat()
            cur["days_read"] = int(cur.get("days_read") or 0) + 1
        self._save(unit, cur, seen=seen, stored=stored)
        self._check_streak("board")

    def _queue_finals(self, unit: Unit, cur: Dict[str, Any], data: Any) -> Tuple[int, int]:
        """Queue what a board's finished games still need; (games seen, games already stored)."""
        seen = stored = 0
        card = unit.league == UFC  # UFC without a calendar ("days"), or a card day's board read again
        for event in board_events(data):
            if not completed(event_status(event, card)):
                continue
            seen += 1
            event_id = str(event["id"])
            key = espn_key(event_id)
            if card:
                if self.raw.has_final(key, espn_fightcenter.SOURCE, "fightcenter"):
                    stored += 1
                    self._skipped += 1
                else:
                    cur["queue"].append(["fightcenter", event_id, 0])
                continue
            if self.raw.has_final(key):
                stored += 1
                self._skipped += 1
            else:
                cur["queue"].append(["summary", event_id, 0])
            if unit.league in PLAYS_LEAGUES:
                if self._plays_stored(key):
                    self._skipped += 1
                else:
                    cur["queue"].append(["plays", event_id, 0, 1])
        return seen, stored

    def _plays_stored(self, key: str) -> bool:
        """Whether every page of the game's core plays is in raw.db (at most espn_core.MAX_PAGES)."""
        pages = self.raw.bundle(key).pages(espn_core.SOURCE, "core_plays")
        have = {p.params.get("page") for p in pages if p.params.get("limit") == espn_core.PLAYS_LIMIT}
        first = next((p for p in pages if p.params.get("page") == 1 and p.params.get("limit") ==
                      espn_core.PLAYS_LIMIT), None)
        if first is None:
            return False
        try:
            count = int(first.json()["pageCount"])
        except (ValueError, KeyError, TypeError):
            return False
        return all(page in have for page in range(1, min(count, espn_core.MAX_PAGES) + 1))

    # -- one game's item: a summary, a page of plays, a fightcenter --

    def _item(self, unit: Unit, cur: Dict[str, Any]) -> None:
        head = list(cur["queue"][0]) + [None] * 5
        kind, ident = str(head[0]), str(head[1])
        tries, waits = int(head[2] or 0), int(head[4] or 0)
        page = int(head[3]) if kind == "plays" and head[3] is not None else 1
        league, key = unit.league, espn_key(ident)
        if kind not in ("summary", "plays", "fightcenter"):  # an item this version doesn't know: dropped
            cur["queue"].pop(0)
            self._save(unit, cur)
            return
        # Already stored (by the worker, or before a restart that came between the store and the cursor's save)
        if kind == "summary" and self.raw.has_final(key):
            cur["queue"].pop(0)
            self._skipped += 1
            self._save(unit, cur, stored=1)
            return
        if kind == "fightcenter" and self.raw.has_final(key, espn_fightcenter.SOURCE, "fightcenter"):
            cur["queue"].pop(0)
            self._skipped += 1
            self._save(unit, cur, stored=1)
            return
        if kind == "plays":
            held = self.raw.latest(key, espn_core.SOURCE, "core_plays", {"page": page, "limit": espn_core.PLAYS_LIMIT})
            count = self._page_count(held)
            if count is not None:
                self._skipped += 1
                self._next_page(cur, ident, page, count)
                self._save(unit, cur)
                return
        self._halt_check(kind)
        if kind == "summary":
            fetch, error = self._fetch(lambda: espn_site.summary(self.http, league, ident))
        elif kind == "plays":
            fetch, error = self._fetch(lambda: espn_core.plays(self.http, league, ident, page=page,
                                                               limit=espn_core.PLAYS_LIMIT))
        else:
            fetch, error = self._fetch(lambda: espn_fightcenter.card(self.http, ident))
        new = self._keep(league, key, fetch) if fetch is not None else False
        stored = 0
        count = espn_core.page_count(fetch) if kind == "plays" and fetch is not None and fetch.ok else None
        if fetch is not None and fetch.ok and (kind != "plays" or count is not None):
            self._ok(kind)
            if kind == "plays":
                self._next_page(cur, ident, page, int(count))  # type: ignore[arg-type]
            else:
                cur["queue"].pop(0)
                final = summary_state(fetch.body)[0] == "final" if kind == "summary" else True
                stored = 1 if new and final else 0
        else:
            what = f"{kind} {ident}" + (f" page {page}" if kind == "plays" else "")
            why = self._why(fetch, error)
            self._failure(cur, kind, what, why)
            cur["queue"].pop(0)
            keep_page = page if kind == "plays" else None
            if not self._definitive(fetch):
                self._defer(cur, kind, ident, tries, keep_page, waits)  # an outage: tried again later, never lost
            elif not self._give_up(cur, what, why, tries):
                cur["queue"].append([kind, ident, tries + 1, keep_page, waits])
        self._save(unit, cur, stored=stored)
        self._check_streak(kind)

    @staticmethod
    def _page_count(payload: Any) -> Optional[int]:
        if payload is None:
            return None
        try:
            return int(payload.json()["pageCount"])
        except (ValueError, KeyError, TypeError):
            return None

    @staticmethod
    def _next_page(cur: Dict[str, Any], ident: str, page: int, count: int) -> None:
        """After a page of plays: the next one, or the game's plays are done (at most espn_core.MAX_PAGES)."""
        if page < min(count, espn_core.MAX_PAGES):
            cur["queue"][0] = ["plays", ident, 0, page + 1]
        else:
            cur["queue"].pop(0)

    # -- the NFL's play-by-play file --

    def _pbp(self, unit: Unit, cur: Dict[str, Any]) -> None:
        """The season's nflverse play-by-play, once (none when the worker or an earlier night stored it). A failure
        is tried again the next night (never given up: it is the NFL's richest record), and is not ESPN's: it
        neither counts towards nor waits for ESPN's failures."""
        key = nflverse.pbp_key(unit.season)
        if self.raw.latest(key, nflverse.SOURCE, "pbp_season", {"season": unit.season}) is not None:
            cur["pbp"] = True
            self._skipped += 1
            self._save(unit, cur)
            return
        try:
            fetch, error = self._fetch(lambda: nflverse.pbp_season(self.http, unit.season))
        except _Blocked:  # GitHub backing off: the file waits for the next night, the NFL's boards go on
            cur["pbp_night"] = self._night
            self._save(unit, cur)
            return
        if fetch is not None:
            self._keep(NFL, key, fetch)
        if fetch is not None and fetch.ok:
            cur["pbp"] = True
            cur.pop("pbp_night", None)
        else:
            self._failed_requests += 1
            cur["pbp_night"] = self._night
            cur["errors"] = int(cur.get("errors") or 0) + 1
            cur["last_error"] = f"nflverse pbp {unit.season}: {self._why(fetch, error)}"[:200]
        self._save(unit, cur)

    # -- UFC: the cards from the calendar --

    def _ufc_step(self, unit: Unit, cur: Dict[str, Any]) -> bool:
        wait = cur.get("calendar_wait")
        waiting = bool(wait) and self.clock() < float(wait[0])
        if cur.get("cards") is None and waiting:  # no calendar yet: it is read again once the wait is over
            return self._finish(unit, cur)
        stale = unit.current and self.clock() - float(cur.get("calendar_at") or 0) > CALENDAR_REFRESH
        if cur.get("cards") is None or (stale and not waiting and cur.get("probe_day") != unit.last.isoformat()):
            self._calendar(unit, cur)
            return True
        done = set(cur.get("done_cards") or [])
        card = next((item for item in cur["cards"] if unit.first.isoformat() <= item[0] <= unit.last.isoformat()
                     and str(item[1]) not in done), None)
        if card is None:
            return self._finish(unit, cur)
        self._card(unit, cur, _day(card[0]), str(card[1]))
        return True

    def _calendar(self, unit: Unit, cur: Dict[str, Any]) -> None:
        """Read the season's cards from a board's calendar: the newest day's board (30 December at the latest: the
        31st's can be the next season's), else 1 July's; with neither, every day of the season is read like a team
        league's ("days"). A board failing transiently is read again after a wait (``calendar_wait``)."""
        probes = [min(unit.last, datetime.date(unit.season, 12, 30))]
        middle = datetime.date(unit.season, 7, 1)
        if unit.first <= middle <= unit.last and middle not in probes:
            probes.append(middle)
        index = int(cur.get("probe") or 0)
        if index >= len(probes):
            if not isinstance(cur.get("cards"), list):
                cur["cards"] = "days"
            cur["calendar_at"], cur["probe_day"], cur["probe"] = self.clock(), unit.last.isoformat(), 0
            self._save(unit, cur)
            return
        day = probes[index]
        data, status, why = self._read_board(unit, cur, day, day)
        cards = calendar_cards(data, unit.season) if status == "ok" else None
        if cards is not None:
            known = {tuple(item) for item in (cur.get("cards") if isinstance(cur.get("cards"), list) else [])}
            cur["cards"] = [list(item) for item in sorted(known | set(cards), reverse=True)]
            cur["calendar_at"], cur["probe_day"], cur["probe"], cur["tries"] = (self.clock(), unit.last.isoformat(),
                                                                                0, 0)
            cur.pop("calendar_wait", None)
        elif status == "ok":  # no calendar of this season on that board: the next probe
            cur["probe"], cur["tries"] = index + 1, 0
        elif status == "transient":
            waits = int((cur.get("calendar_wait") or [0, 0])[1]) + 1
            cur["calendar_wait"] = [self.clock() + min(RETRY_WAIT_CAP, RETRY_WAIT * 2 ** (waits - 1)), waits]
            self._deferred += 1
        else:
            tries = int(cur.get("tries") or 0)
            if self._give_up(cur, f"board {day.isoformat()}", why, tries):
                cur["probe"], cur["tries"] = index + 1, 0
            else:
                cur["tries"] = tries + 1
        self._save(unit, cur)
        self._check_streak("board")

    def _card(self, unit: Unit, cur: Dict[str, Any], day: datetime.date, event_id: str) -> None:
        """A past card: its day's board (kept raw: each bout's details) and its fightcenter, unless the board shows
        it didn't happen (postponed, cancelled). A board failing transiently is read again later (the retry list);
        the card's fightcenter is asked for by its id meanwhile."""
        data, status, why = self._read_board(unit, cur, day, day)
        if status == "definitive":
            tries = int(cur.get("tries") or 0)
            if not self._give_up(cur, f"board {day.isoformat()}", why, tries):
                cur["tries"] = tries + 1  # the board is tried again on the next step
                self._save(unit, cur)
                self._check_streak("board")
                return
        elif status == "transient":
            self._defer(cur, "board", day.isoformat(), 0, None, 0)
        cur["tries"] = 0
        event = next((e for e in board_events(data) if str(e.get("id")) == event_id), None) if data else None
        seen = stored = 0
        if event is None or completed(event_status(event, card=True)):  # not on its board: asked for by its id
            seen = 1
            if self.raw.has_final(espn_key(event_id), espn_fightcenter.SOURCE, "fightcenter"):
                stored = 1
                self._skipped += 1
            else:
                cur["queue"].append(["fightcenter", event_id, 0])
        cur["done_cards"] = list(cur.get("done_cards") or []) + [event_id]
        cur["day"] = day.isoformat()
        cur["days_read"] = int(cur.get("days_read") or 0) + 1
        self._save(unit, cur, seen=seen, stored=stored)
        self._check_streak("board")

__all__ = ["RawBackfill", "Unit", "ORDER", "WINDOWS", "PLAYS_LEAGUES", "PER_NIGHT_DEFAULT", "SEASONS_DEFAULT",
           "WINDOW_HOURS", "SETTLE_HOURS", "season_window", "season_dates", "seasons_for", "settled_day",
           "completed", "event_status", "board_events", "calendar_cards", "summary_state", "summary_start",
           "stored_summary", "keep_summary", "backfill_tz", "DEFAULT_TZ", "RETRY_WAIT", "RETRY_WAIT_CAP"]
