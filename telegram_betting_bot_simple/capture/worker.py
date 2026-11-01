"""The capture worker (v4 P1, plan section 3.3): ``Capturer(store, raw, espn, http, clock)``.

**Jobs** (raw.db ``capture_jobs``, one per game, ``kinds_done`` naming what is in hand: "pre:t24:summary",
"final:core_plays"):
- ``schedule_from_slate(games)``: the slate's ESPN games (data/slate.py hands each round's over) become jobs, with
  their pre-start horizons. A game's key is ``espn:<eventId>``; a tennis match is ``espn-match:<competitionId>`` under
  its tour (tennis_atp, tennis_wta); an MMA bout belongs to its card, ``espn:<cardEventId>``, found on the day's lean
  UFC board in the cache (the slate's rows carry the bout's id only). The slate's status moves the final: "live"
  keeps it at least LIVE_HOLD (an hour) away, the first "final" round brings it to 30 minutes from then. An unchanged
  game costs one batched read and no write.
- ``sweep(day)``: the day's scoreboard of each of the 58 paths, one board per league (ESPN refuses ranges), through
  ``EspnClient.scoreboard_raw`` (the lean copy goes into the cache from the same answer), each kept raw under
  ``scoreboard_key(league, "YYYYMMDD")``; every final game nobody opened gets a job. The tennis boards are the tennis
  matches' record (their jobs are marked done from them), the UFC board a card's.
- ``run_due(budget)``: what is due, pre-start items first, at most ``budget`` requests (and CAPTURE_PER_DAY a UTC day,
  counted in raw.db). Pre items (plan.py) are fetched once per kind for every horizon already due and are missed once
  the game starts; a definitive answer (2xx, or a 4xx such as no props yet) serves the horizon, a 5xx, a 429 or an
  unreachable host is tried again PRE_RETRY later. The final, per sport (plan.KINDS): the site summary through ``EspnClient.summary_raw`` (one fetch
  for raw.db and the lean cache: the grader's next summary() is a hit), core plays (all pages) for
  CAPTURE_CORE_PLAYS_LEAGUES, MLB StatsAPI's live feed (the gamePk from data/mlb_stats.py's schedule, matched on the
  teams; one game at a time), a UFC card's fightcenter (with core status and fighter statistics for a bout it has no
  result for) and the day's board, a tennis match's day board. A summary that isn't final yet, or anything failing,
  is tried again at +2 h, +12 h and +36 h after the final; a job gives up 4 days after its start. A postponed or
  cancelled game is "void".
- Static history on its schedule (``run_static``): Greco1899 weekly, the Match Charting Project monthly, football-data
  twice a week, nflverse's play-by-play and officials every Tuesday.
- The older-games backfill (capture/backfill.py ``RawBackfill``, CAPTURE_BACKFILL) inside its night window, in chunks
  of ``BACKFILL_CHUNK`` requests, and only while no forward job can run (``has_runnable_due``, which it checks before
  each request: a job due on a backing-off host, or with today's budget spent, never holds it back).
- Nothing at all is fetched while raw.db's disk is short of room (``RawStore.room``: CAPTURE_MIN_FREE_GB and 10%).

**Budgets.** Every request is checked by the http guard: forward ones count "capture" against CAPTURE_PER_DAY (UTC
days; propBets and Kalshi count "prices" too), the backfill its own night (a host backing off costs neither). Every answer is counted "http_ok" or
"http_error" (an unreachable host too). Nothing here touches The Odds API.

**The thread.** ``capture_forever(capturer)`` is the dashboard's "capture" thread: a ``tick`` a minute under the store
lease "capture" (one process captures; a restarted one takes over within 15 minutes). Restarts resume: the jobs, the
budgets and the backfill's cursors are all in raw.db.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from ..data.espn_client import ESPN_DAY_TZ, competition_state, team_match_score
from . import plan
from .http import Backoff, PoliteHttp, Refused, Unreachable, set_shared, shared
from .raw_store import Job, RawStore, disk_is_full, resolve_raw_db_path, utc_day
from .sources import RawFetch, espn_key, match_key, scoreboard_key

log = logging.getLogger(__name__)

LEASE = "capture"
LEASE_SECONDS = 900.0  # renewed every tick and between backfill chunks
LEASE_RENEW = LEASE_SECONDS / 3  # and inside a tick's long run this often, so one tick never outlives the lease
TICK_SECONDS = 60.0
TICK_BUDGET = 300  # forward requests one tick makes at most (the rest wait for the next)
BACKFILL_CHUNK = 50  # requests one RawBackfill.run_night call makes at most
BACKFILL_TICK = 15 * 60.0  # one tick's backfill runs this long at most (then the tick ends and the loop goes on)
SWEEP_AT = 5  # yesterday's boards are swept from 05:00 US Eastern (ESPN's day): every game of it is over
TENNIS_AFTER = 3600.0  # a tennis match's own board fetch waits this long after its day's sweep
SWEEP_RETRY = 30 * 60.0  # boards a sweep couldn't read are tried again after this ...
SWEEP_TRIES = 3  # ... this many sweeps at most (the slate's jobs and tomorrow's sweep cover the rest)
PRE_RETRY = 10 * 60.0  # a pre-start fetch that failed is tried again after this (never after the start)
LIVE_HOLD = 3600.0  # while the slate shows a game live, its final is kept at least this far away (the first "final"
                    # round then brings it to 30 minutes from the whistle)
STATIC_RETRY = 3600.0  # a static source that failed is tried again after this
DISK_LOG_EVERY = 3600.0  # "the disk is nearly full" is logged at most this often
VOID_NAMES = ("POSTPONED", "CANCELED", "CANCELLED", "ABANDONED")
ERROR_TEXT = 300
MISSING = object()  # schedule_from_slate's "no job yet"


# --- small helpers ---------------------------------------------------------------------------------------------------

def espn_day(ts: float) -> datetime.date:
    """ESPN's day (US Eastern) of an epoch time."""
    return datetime.datetime.fromtimestamp(float(ts), ESPN_DAY_TZ).date()


def _yyyymmdd(day: datetime.date) -> str:
    return day.strftime("%Y%m%d")


def _parse(body: bytes) -> Any:
    try:
        return json.loads(bytes(body or b"").decode("utf-8-sig"))
    except (ValueError, UnicodeDecodeError):
        return None


def _iso_ts(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        when = datetime.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    return when.timestamp()


def summary_state(data: Any) -> Optional[str]:
    """A game summary's state: "final", "void" (postponed, cancelled, abandoned), "live" or "pre"; None if
    unreadable."""
    competitions = (data.get("header") or {}).get("competitions") if isinstance(data, dict) else None
    competition = competitions[0] if isinstance(competitions, list) and competitions else None
    status = (competition or {}).get("status") if isinstance(competition, dict) else None
    kind = (status or {}).get("type") if isinstance(status, dict) else None
    if not isinstance(kind, dict):
        return None
    name = str(kind.get("name") or "").upper()
    if any(word in name for word in VOID_NAMES):
        return "void"
    if kind.get("completed") or kind.get("state") == "post":
        return "final"
    if kind.get("state") == "in":
        return "live"
    if kind.get("state") == "pre":
        return "pre"
    return None


def _http_answer(exc: BaseException) -> Optional[Tuple[int, bytes]]:
    """(status, body) of an error that is an HTTP answer (data/espn_client.py's EspnHttpError), else None."""
    status = getattr(exc, "status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        body = getattr(exc, "body", b"")
        return int(status), bytes(body) if isinstance(body, (bytes, bytearray)) else b""
    return None


def answered(status: int) -> bool:
    """A definitive answer: 2xx, or a 4xx other than 429 (asking again soon won't change it)."""
    return 200 <= int(status) < 500 and int(status) != 429


def _tennis_finals(data: Any) -> Dict[str, float]:
    """{competition id: start} of a tennis board's singles matches that are over."""
    from ..data.slate import SINGLES
    found: Dict[str, float] = {}
    for event in (data.get("events") or []) if isinstance(data, dict) else []:
        for grouping in (event.get("groupings") or []) if isinstance(event, dict) else []:
            if not isinstance(grouping, dict) or (grouping.get("grouping") or {}).get("slug") not in SINGLES:
                continue
            for competition in grouping.get("competitions") or []:
                state = competition_state(competition) if isinstance(competition, dict) else None
                if state and state[0] == "final" and competition.get("id") not in (None, ""):
                    start = state[3] if state[3] is not None else _iso_ts(competition.get("date"))
                    found[str(competition["id"])] = float(start or 0.0)
    return found


def _events(data: Any) -> List[Dict[str, Any]]:
    events = data.get("events") if isinstance(data, dict) else None
    return [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []


@dataclass
class _Run:
    """What one job's run did."""

    picked: float  # the job's due time when it was picked (the retry schedule is anchored on it)
    fetched: int = 0
    stored: int = 0
    failed: int = 0
    done: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    pre_failed: Set[str] = field(default_factory=set)
    attempted: bool = False  # a final attempt was made
    void: bool = False
    stopped: Optional[str] = None  # "budget" / "day": the run stopped here
    wait_until: Optional[float] = None  # a host is backing off until then
    hard: bool = False  # something really failed (an error answer, an unreachable host, a game not over yet)
    became_done: bool = False
    statuses: List[int] = field(default_factory=list)  # every answer's status, in order


def options_from(config: Any, kalshi_series: Sequence[str] = ()) -> plan.Options:
    """plan.Options from the CAPTURE_* settings (None: the defaults)."""
    core = plan.league_list(getattr(config, "capture_core_plays_leagues", None), plan.CORE_PLAYS_LEAGUES)
    leagues = sorted({league for league in (_kalshi_league(t) for t in kalshi_series) if league})
    return plan.Options(core_plays_leagues=core, propbets=bool(getattr(config, "capture_propbets", True)),
                        kalshi_leagues=tuple(leagues))


def _kalshi_series(config: Any) -> Tuple[str, ...]:
    from .sources import kalshi_raw
    text = getattr(config, "capture_kalshi_series", None) if config is not None else None
    return tuple(kalshi_raw.parse_series(text))


def _kalshi_league(ticker: str) -> Optional[str]:
    from .sources import kalshi_raw
    return kalshi_raw.league_of(ticker)


def raw_db_file(config: Any, store: Any = None) -> str:
    """Where raw.db is (raw_store.resolve_raw_db_path): RAW_DB_PATH, else beside betting.db."""
    return resolve_raw_db_path(config, store)


# --- the worker ----------------------------------------------------------------------------------------------------------

class Capturer:
    """The capture worker (see the module's docstring). ``store`` is data/store.Store (its lease), ``raw`` the
    RawStore, ``espn`` the shared EspnClient (summary_raw / scoreboard_raw seed its lean cache), ``http`` the capture's
    PoliteHttp, ``clock`` epoch seconds. ``config`` holds the CAPTURE_* settings; tests hand in ``options``,
    ``per_day``, ``kalshi_series``, ``backfill`` (a RawBackfill or a stand-in), ``mlb`` (data/mlb_stats.MlbStats),
    ``tz`` (the server's time zone, for the static schedule) and ``lease``. ``share_door`` (the dashboard's capturer:
    build_capturer) makes ``http`` the process's door while this process holds the capture lease (see _lend_door)."""

    def __init__(self, store: Any, raw: RawStore, espn: Any, http: PoliteHttp,
                 clock: Optional[Callable[[], float]] = None, *, config: Any = None,
                 options: Optional[plan.Options] = None, per_day: Optional[int] = None,
                 kalshi_series: Optional[Sequence[str]] = None, backfill: Any = None, mlb: Any = None,
                 cache: Any = None, tz: Optional[datetime.tzinfo] = None,
                 lease: Optional[Callable[[], bool]] = None, share_door: bool = False) -> None:
        self.store = store
        self.raw = raw
        self.espn = espn
        self.http = http
        self.clock: Callable[[], float] = clock or time.time
        self.config = config
        self.tz = tz
        self.per_day = int(per_day if per_day is not None else getattr(config, "capture_per_day", 3000))
        self.kalshi_series = tuple(kalshi_series) if kalshi_series is not None else _kalshi_series(config)
        self.options = options or options_from(config, self.kalshi_series)
        self.cache = cache if cache is not None else getattr(espn, "cache", None)
        self._mlb = mlb
        self._backfill = backfill
        self._backfill_tried = backfill is not None
        self.backfill_enabled = backfill is not None or bool(getattr(config, "capture_backfill", True))
        if lease is not None:
            self.lease = lease
        elif store is not None and hasattr(store, "try_lease"):
            self.lease = lambda: bool(store.try_lease(LEASE, LEASE_SECONDS))
        else:
            self.lease = lambda: True
        self._lock = threading.RLock()  # one forward run at a time (ticks, run_due, sweep, static)
        self._call_left: Optional[int] = None
        self._price = False
        self._kalshi_at: Dict[str, float] = {}
        self.last: Dict[str, Any] = {}  # the latest tick's report (the health block may show it)
        self._disk_logged = -1e18
        self._lease_at: Optional[float] = None  # when this tick took or last renewed the lease (None: not in a tick)
        self.share_door = bool(share_door)
        if getattr(http, "on_result", "absent") is None:
            http.on_result = self._count_result

    # -- budgets --

    def _allow(self, url: str) -> None:
        """The forward guard: this call's budget, then the day's (CAPTURE_PER_DAY, UTC days)."""
        if self._call_left is not None and self._call_left <= 0:
            raise Refused("budget")
        day = utc_day(self.clock())
        if not self.raw.spend("capture", day, 1, limit=self.per_day):
            raise Refused("day")
        if self._price:
            self.raw.spend("prices", day, 1)
        if self._call_left is not None:
            self._call_left -= 1

    def _count_result(self, url: str, status: Optional[int]) -> None:
        """PoliteHttp's on_result: every request it sent, counted once per UTC day under its outcome. A request made
        inside another guard than the forward one (``_allow``: run_due, the sweep, the static files) is the older-games
        backfill's (RawBackfill.run_night's guard): it counts backfill_http_ok / backfill_http_error, so the forward
        alarm (http_ok / http_error) never drowns in the backfill's night of requests. The backfill counts nothing
        itself while its door counts (raw-backfill B1)."""
        ok = status is not None and 200 <= int(status) < 300
        active = getattr(self.http, "active_guard", None)
        guard = active() if callable(active) else None
        backfill = guard is not None and guard != self._allow
        kind = ("backfill_http_ok" if ok else "backfill_http_error") if backfill else ("http_ok" if ok else "http_error")
        self.raw.spend(kind, utc_day(self.clock()))

    @contextlib.contextmanager
    def _forward(self, budget: Optional[int]) -> Iterator[None]:
        before = self._call_left
        self._call_left = None if budget is None else max(0, int(budget))
        try:
            with self.http.guard(self._allow):
                yield
        finally:
            self._call_left = before

    @contextlib.contextmanager
    def _prices(self) -> Iterator[None]:
        before, self._price = self._price, True
        try:
            yield
        finally:
            self._price = before

    def room(self) -> bool:
        """Whether raw.db's disk has room (RawStore.room: CAPTURE_MIN_FREE_GB free and 10% of the disk). While it
        hasn't, nothing is fetched, forward or backfill (logged once an hour)."""
        room = getattr(self.raw, "room", None)
        if not callable(room) or room():
            return True
        now = self.clock()
        if now - self._disk_logged >= DISK_LOG_EVERY:
            self._disk_logged = now
            log.warning("Capture paused: raw.db's disk is nearly full (%s); nothing is fetched until there is room",
                        self.raw.path)
        return False

    def _lend_door(self, held: bool) -> None:
        """While this process holds the capture lease, ``http`` is the process's door (capture.http.set_shared): the
        learn backfill's HistHttp takes its tokens from the same per-host buckets, so forward capture and the learn
        backfill together keep to each host's rate (both run in the dashboard process). Withdrawn when the lease is
        another process's. Only the dashboard's capturer lends its door (``share_door``)."""
        if not self.share_door:
            return
        if held:
            set_shared(self.http)
        elif shared() is self.http:
            set_shared(None)

    def _hold_lease(self) -> bool:
        """Inside a tick: renew the capture lease once LEASE_RENEW has passed since it was taken, so a long run (slow
        answers) never outlives it and another process can't start capturing alongside; False when another process
        holds it now. Outside a tick (a direct call) there is no lease to keep."""
        if self._lease_at is None:
            return True
        now = self.clock()
        if now - self._lease_at < LEASE_RENEW:
            return True
        if not self.lease():
            self._lend_door(False)
            return False
        self._lease_at = now
        return True

    def day_left(self) -> int:
        """Forward requests left today (UTC)."""
        return max(0, self.per_day - self.raw.spent("capture", utc_day(self.clock())))

    def _due_kinds(self, job: Job, now: float) -> List[str]:
        """The kinds a due job would fetch if it ran now."""
        if now < job.start:
            return sorted({kind for kind, horizon in plan.pre_items(job.league, self.options)
                           if plan.pre_name(kind, horizon) not in job.kinds_done
                           and plan.fire_time(horizon, job.start, now) <= now})
        if job.due_at <= job.start:
            return []  # only its final's timing to plan: no request
        return [kind for kind in plan.kinds_for(job.league, "final", self.options)
                if plan.final_name(kind) not in job.kinds_done]

    def has_runnable_due(self) -> bool:
        """Whether run_due would make a request now: today's budget (CAPTURE_PER_DAY) has requests left and a due job
        has something to fetch from a host that isn't backing off. The backfill waits only for that (a job blocked on
        a spent budget or a backing-off host never holds it back all night)."""
        now = self.clock()
        due = self.raw.next_due()
        if due is None or due > now or self.day_left() <= 0:
            return False
        backing_off = getattr(self.http, "backing_off", None)
        for job in self.raw.due_jobs(now, limit=500):
            for kind in self._due_kinds(job, now):
                host = plan.HOST_OF.get(kind, "")
                if not callable(backing_off) or not host or not backing_off(host):
                    return True
        return False

    def forward_due(self) -> bool:
        """The backfill's check before each request (RawBackfill's ``forward_due``): has_runnable_due."""
        return self.has_runnable_due()

    # -- the slate's games --

    def schedule_from_slate(self, games: Iterable[Mapping[str, Any]]) -> int:
        """Make or update the jobs of the slate's ESPN games (see the module's docstring). Returns the jobs made."""
        now = self.clock()
        created = 0
        bouts: List[Tuple[str, float, str]] = []
        games_found: List[Tuple[str, str, str, float, str]] = []
        for row in games or []:
            try:
                found = self._slate_game(row)
                if found is None:
                    continue
                league, espn_id, start, status = found
                sport = plan.sport_of(league)
                if sport == "mma":
                    bouts.append((espn_id, start, status))
                    continue
                key = match_key(espn_id) if sport == "tennis" else espn_key(espn_id)
                games_found.append((key, league, espn_id, start, status))
            except Exception:  # noqa: BLE001 - one odd game costs only itself
                log.exception("Capture: couldn't read a slate game")
        # One read of what is known: only a game whose job changes takes the write lock
        known = self.raw.jobs_for(key for key, *_rest in games_found)
        for key, league, espn_id, start, status in games_found:
            try:
                created += self._upsert(key, league, espn_id, start, status, now, known=known.get(key, MISSING))
            except Exception:  # noqa: BLE001
                log.exception("Capture: couldn't schedule a slate game")
        if bouts:
            try:
                created += self._schedule_cards(bouts, now)
            except Exception:  # noqa: BLE001
                log.exception("Capture: couldn't schedule the UFC cards")
        return created

    @staticmethod
    def _slate_game(row: Mapping[str, Any]) -> Optional[Tuple[str, str, float, str]]:
        if not isinstance(row, Mapping):
            return None
        league = plan.espn_league(str(row.get("espn_league") or row.get("league") or ""))
        espn_id = str(row.get("espn_id") or "").strip()
        if league is None or not espn_id or not espn_id.isascii() or any(ch.isspace() or ch in "/:" for ch in espn_id):
            return None
        start = _iso_ts(row.get("_ts")) if row.get("_ts") is not None else None
        start = start if start is not None else _iso_ts(row.get("start"))
        if start is None:
            return None
        return league, espn_id, float(start), str(row.get("status") or "pre")

    def _schedule_cards(self, bouts: Sequence[Tuple[str, float, str]], now: float) -> int:
        """UFC bouts -> their cards' jobs (``espn:<cardEventId>``), through the day's lean UFC board in the cache."""
        by_day: Dict[datetime.date, List[Tuple[str, float, str]]] = {}
        for bout in bouts:
            by_day.setdefault(espn_day(bout[1]), []).append(bout)
        created = 0
        for day, items in sorted(by_day.items()):
            try:
                events = self.espn.scoreboard(plan.UFC, day)
            except Exception:  # noqa: BLE001 - the sweep finds the card after it
                log.info("Capture: no UFC board for %s in the cache; the sweep will find its card", day)
                continue
            card_of: Dict[str, str] = {}
            for event in events or []:
                if isinstance(event, dict) and event.get("id") not in (None, ""):
                    for competition in event.get("competitions") or []:
                        if isinstance(competition, dict) and competition.get("id") not in (None, ""):
                            card_of[str(competition["id"])] = str(event["id"])
            cards: Dict[str, Dict[str, Any]] = {}
            for bout_id, start, status in items:
                card = card_of.get(bout_id)
                if card is None:
                    continue
                entry = cards.setdefault(card, {"start": start, "statuses": []})
                entry["start"] = min(entry["start"], start)
                entry["statuses"].append(status)
            for card, entry in cards.items():
                statuses = entry["statuses"]
                status = ("final" if all(s in ("final", "postponed") for s in statuses) else
                          "live" if any(s in ("live", "final") for s in statuses) else "pre")
                created += self._upsert(espn_key(card), plan.UFC, card, entry["start"], status, now)
        return created

    def _upsert(self, key: str, league: str, espn_id: str, start: float, status: Optional[str], now: float,
                known: Any = None) -> int:
        """Make the game's job, or bring its start and the final's timing up to date (``status``: the slate's).
        ``known``: the job as just read (MISSING: none), so an unchanged game costs no write."""
        made: List[bool] = []

        def change(job: Optional[Job]) -> Optional[Job]:
            if job is None:
                made.append(True)
                return self._plan(Job(key, league, espn_id, float(start), 0.0), now, hint=status)
            if job.state in ("done", "gave_up"):
                return None
            before = (job.start, job.espn_id, job.due_at, job.state, tuple(job.kinds_done), job.attempts)
            moved = abs(job.start - float(start)) >= 60.0
            if job.state == "void":
                if not moved or status == "postponed":
                    return None
                # Rescheduled under the same id: capture it again on its new date
                job.state, job.attempts = "todo", 0
                job.kinds_done = [name for name in job.kinds_done if not name.startswith("final:")]
                job.due_at = float(start)
            elif moved and job.attempts == 0 and not any(n.startswith("final:") for n in job.kinds_done):
                job.due_at = min(job.due_at, float(start))  # plan its horizons and its final again
            job.start, job.espn_id = float(start), espn_id
            self._plan(job, now, hint=status)
            after = (job.start, job.espn_id, job.due_at, job.state, tuple(job.kinds_done), job.attempts)
            return None if after == before else job

        if known is not None:
            trial = None if known is MISSING else dataclasses.replace(known, kinds_done=list(known.kinds_done))
            if trial is not None and change(trial) is None:
                return 0  # nothing to write
            made.clear()
        self.raw.modify_job(key, change, now=now)  # under the lock, from the row as it is now
        return 1 if made else 0

    # -- planning --

    def _required(self, league: str) -> List[str]:
        return [plan.final_name(kind) for kind in plan.required_final(league, self.options)]

    def _first_final(self, job: Job) -> float:
        """When a game's final is first fetched: 30 minutes after it usually ends (a tennis match: after its day's
        sweep); always after the start."""
        if plan.sport_of(job.league) == "tennis":
            return max(self.sweep_time(espn_day(job.start)) + TENNIS_AFTER, job.start + 1.0)
        return max(plan.estimated_end(job.league, job.start) + plan.FINAL_DELAYS[0], job.start + 1.0)

    def _plan(self, job: Job, now: float, *, hint: Optional[str] = None, attempted: bool = False,
              picked: Optional[float] = None, retry_pre: Iterable[str] = (),
              wait_until: Optional[float] = None) -> Job:
        """Set the job's state and next due time from its start, what it has and how its last run went."""
        if job.state == "void":
            return job
        done = set(job.kinds_done)
        required = self._required(job.league)
        if required and all(name in done for name in required):
            job.state = "done"
            return job
        if now >= job.start + plan.GIVE_UP:
            job.state = "gave_up"
            return job
        job.state = "partial" if done else "todo"
        retry = set(retry_pre)
        if now < job.start:
            fires = []
            for kind, horizon in plan.pre_items(job.league, self.options):
                if plan.pre_name(kind, horizon) in done:
                    continue
                fire = plan.fire_time(horizon, job.start, now)
                fires.append(max(fire, now + PRE_RETRY) if kind in retry else fire)
            upcoming = [fire for fire in fires if fire < job.start]
            if upcoming:
                due = min(upcoming) if wait_until is None else max(min(upcoming), wait_until)
                if due < job.start:
                    job.due_at = due
                    return job
            if job.due_at > job.start and not job.attempts and not any(n.startswith("final:") for n in done):
                job.due_at = job.start  # a pre due time that slid past the start: plan the final afresh
        # The final
        final_started = job.attempts > 0 or any(name.startswith("final:") for name in done)
        if attempted:
            job.attempts += 1
            step = plan.next_final_delay(job.attempts)
            if step is None:
                job.state = "gave_up"
                return job
            job.due_at = max((picked if picked is not None else now) + step, now + plan.FINAL_DELAYS[0])
        elif job.due_at <= job.start:  # the final's first fetch isn't planned yet
            job.due_at = self._first_final(job)
        if not final_started and not attempted and plan.sport_of(job.league) != "tennis":
            if hint == "final":
                job.due_at = min(job.due_at, max(now + plan.FINAL_DELAYS[0], job.start + 1.0))
            elif hint == "live":
                job.due_at = max(job.due_at, now + LIVE_HOLD)
            elif hint == "postponed" and now >= job.start:
                job.due_at = min(job.due_at, max(now, job.start + 1.0))
        if wait_until is not None:
            job.due_at = max(job.due_at, wait_until)
        return job

    # -- running what is due --

    def run_due(self, budget: int) -> Dict[str, Any]:
        """Run the due jobs, pre-start items first, with at most ``budget`` requests (and today's CAPTURE_PER_DAY).
        Returns {"fetched", "stored", "failed", "jobs_done", "stopped"} (stopped: "budget", "day" or None)."""
        out: Dict[str, Any] = {"fetched": 0, "stored": 0, "failed": 0, "jobs_done": 0, "stopped": None}
        if int(budget) <= 0:
            out["stopped"] = "budget"
            return out
        if not self.room():
            out["stopped"] = "disk"
            return out
        with self._lock, self._forward(budget):
            now = self.clock()
            jobs = self.raw.due_jobs(now)
            jobs.sort(key=lambda job: (0 if now < job.start else 1, job.due_at, job.start, job.game_key))
            for job in jobs:
                if self._call_left is not None and self._call_left <= 0:
                    out["stopped"] = "budget"
                    break
                if not self.room():  # raw.db's disk filled up mid-run: stop before asking for more
                    out["stopped"] = "disk"
                    break
                if not self._hold_lease():
                    out["stopped"] = "lease"  # another process captures now
                    break
                run = self._run_job(job, self.clock())
                out["fetched"] += run.fetched
                out["stored"] += run.stored
                out["failed"] += run.failed
                if run.became_done:
                    out["jobs_done"] += 1
                if run.stopped:
                    out["stopped"] = run.stopped
                    break
        return out

    def _run_job(self, job: Job, now: float) -> _Run:
        run = _Run(picked=job.due_at)
        try:
            if now < job.start:
                self._run_pre(job, now, run)
            elif job.due_at > job.start:
                run.attempted = True
                self._run_final(job, now, run)
        except Refused as exc:
            run.stopped = str(exc) or "budget"
            run.attempted = False  # not an attempt: nothing failed
        except Exception as exc:  # noqa: BLE001 - a broken payload or a bug costs this job one attempt only
            if disk_is_full(exc):  # raw.db can't store now (RawStore.room is off): the job waits, untouched
                run.stopped, run.attempted = "disk", False
                return run
            log.exception("Capture: %s failed", job.game_key)
            run.errors.append(f"{type(exc).__name__}: {exc}")
            run.hard = True
        try:
            self._finish(job, run)
        except Exception as exc:  # noqa: BLE001
            if not disk_is_full(exc):
                raise
            run.stopped = "disk"
        return run

    def _finish(self, job: Job, run: _Run) -> None:
        def change(fresh: Optional[Job]) -> Optional[Job]:
            if fresh is None:
                return None
            fresh.kinds_done = sorted(set(fresh.kinds_done) | set(run.done))
            if run.errors:
                fresh.last_error = "; ".join(run.errors)[-ERROR_TEXT:]
            elif run.done:
                fresh.last_error = None
            if run.void:
                fresh.state = "void"
            # An attempt that only met a back-off (nothing sent) isn't counted: it waits for the host instead
            counted = run.attempted and not run.void and (run.hard or run.wait_until is None)
            return self._plan(fresh, self.clock(), attempted=counted, picked=run.picked, retry_pre=run.pre_failed,
                              wait_until=run.wait_until)

        before = job.state
        updated = self.raw.modify_job(job.game_key, change, now=self.clock())
        run.became_done = bool(updated is not None and updated.state == "done" and before != "done")

    def _keep(self, game_key: str, league: str, source: str, endpoint: str, params: Mapping[str, Any], body: bytes,
              state: str, status: int, horizon: Optional[str], run: _Run, fetched_at: Optional[float] = None) -> bool:
        """Store one answer (any status: an error is kept with its status) and count it; True for a 2xx."""
        ok = 200 <= int(status) < 300
        _id, new = self.raw.insert(game_key=game_key, league=league, source=source, endpoint=endpoint, params=params,
                                   body=body, state=state, http_status=int(status), horizon=horizon if ok else None,
                                   fetched_at=self.clock() if fetched_at is None else fetched_at)
        run.statuses.append(int(status))
        run.fetched += 1
        run.stored += 1 if new else 0
        if not ok:
            run.failed += 1
            run.hard = True
            run.errors.append(f"{source}/{endpoint}: HTTP {status}")
        return ok

    def _keep_fetch(self, fetch: RawFetch, game_key: str, league: str, state: str, horizon: Optional[str],
                    run: _Run) -> bool:
        return self._keep(game_key, league, fetch.source, fetch.endpoint, fetch.params, fetch.body, state,
                          fetch.status, horizon, run, fetched_at=fetch.fetched_at)

    def _attempt(self, run: _Run, what: str, step: Callable[[], bool]) -> bool:
        """Run one fetch step; an unreachable host or a back-off is a failure of this step only (Refused stops the
        run)."""
        try:
            return bool(step())
        except Backoff as exc:
            run.wait_until = max(run.wait_until or 0.0, exc.until)
            run.errors.append(f"{what}: {exc}")
        except Unreachable as exc:
            run.failed += 1
            run.hard = True
            run.errors.append(f"{what}: {exc}")
        return False

    # -- before the start --

    def _run_pre(self, job: Job, now: float, run: _Run) -> None:
        due: Dict[str, List[str]] = {}
        for kind, horizon in plan.pre_items(job.league, self.options):
            if plan.pre_name(kind, horizon) in job.kinds_done:
                continue
            if plan.fire_time(horizon, job.start, now) <= now:
                due.setdefault(kind, []).append(horizon)
        for kind, horizons in due.items():
            label = max(horizons, key=plan.HORIZON_ORDER.index)  # the window the snapshot falls in
            seen = len(run.statuses)
            ok = self._attempt(run, kind, lambda kind=kind, label=label: self._fetch_pre(job, kind, label, run))
            answers = run.statuses[seen:]
            # A definitive answer (2xx, or 4xx: e.g. no props posted yet) serves the horizon; a 5xx, a 429 or an
            # unreachable host is tried again PRE_RETRY later (never after the start)
            if ok or (answers and all(answered(status) for status in answers)):
                run.done.extend(plan.pre_name(kind, horizon) for horizon in horizons)
            else:
                run.pre_failed.add(kind)

    def _fetch_pre(self, job: Job, kind: str, horizon: str, run: _Run) -> bool:
        if kind == "summary":
            return self._summary(job, run, phase="pre", horizon=horizon) is not None
        if kind == "propbets":
            from .sources import espn_propbets
            with self._prices():  # each page kept as it comes (a budget stop loses none)
                kept = [self._keep_fetch(fetch, job.game_key, job.league, "pre", horizon, run)
                        for fetch in espn_propbets.propbets_all(self.http, job.league, job.espn_id)]
            return bool(kept) and all(kept)
        if kind == "fightcenter":
            from .sources import espn_fightcenter
            fetch = espn_fightcenter.card(self.http, job.espn_id)
            return self._keep_fetch(fetch, job.game_key, job.league, "pre", horizon, run)
        if kind == "kalshi":
            return self._kalshi(job, horizon, run)
        raise ValueError(f"unknown pre kind {kind!r}")

    def _kalshi(self, job: Job, horizon: str, run: _Run) -> bool:
        """Snapshot the Kalshi series of the game's league (one per series per KALSHI_EVERY seconds at most)."""
        from .sources import kalshi_raw
        ok = True
        for ticker in self.kalshi_series:
            if _kalshi_league(ticker) != job.league:
                continue
            if self.clock() - self._kalshi_at.get(ticker, -1e18) < plan.KALSHI_EVERY:
                continue  # a snapshot this recent serves this horizon too
            with self._prices():
                kept = [self._keep_fetch(fetch, kalshi_raw.game_key(ticker), job.league, "pre", horizon, run)
                        for fetch in kalshi_raw.series_all(self.http, ticker)]
            if kept and all(kept):
                self._kalshi_at[ticker] = self.clock()
            else:
                ok = False
        return ok

    # -- the final --

    def _run_final(self, job: Job, now: float, run: _Run) -> None:
        sport = plan.sport_of(job.league)
        if sport == "mma":
            self._final_card(job, run)
            return
        if sport == "tennis":
            self._final_tennis(job, run)
            return
        done = set(job.kinds_done)
        for kind in plan.kinds_for(job.league, "final", self.options):
            name = plan.final_name(kind)
            if name in done:
                continue
            if kind == "summary":
                found: Dict[str, Optional[str]] = {}

                def step() -> bool:
                    found["state"] = self._summary(job, run, phase="final", horizon="final")
                    return found["state"] is not None

                self._attempt(run, "summary", step)
                state = found.get("state")
                if state == "void":
                    run.void = True
                    run.done.append(name)
                    return
                if state != "final":
                    if state is not None:
                        run.hard = True
                        run.errors.append("summary: the game isn't over yet" if state in ("live", "pre") else
                                          "summary: unreadable")
                    return  # nothing else of a game whose summary isn't final
                run.done.append(name)
                done.add(name)
            elif kind == "core_plays":
                if self._attempt(run, kind, lambda: self._core_plays(job, run)):
                    run.done.append(name)
            elif kind == "feed_live":
                if self._attempt(run, kind, lambda: self._feed_live(job, run)):
                    run.done.append(name)

    def _summary(self, job: Job, run: _Run, *, phase: str, horizon: str) -> Optional[str]:
        """Fetch and keep the game's whole summary (EspnClient.summary_raw: the lean cache gets its copy from the same
        answer). Returns the game's state per the summary ("final", "void", "live", "pre", or "unknown" for a 2xx
        answer that can't be read), or None when the answer was an error."""
        try:
            if hasattr(self.espn, "summary_raw"):
                body, status = self.espn.summary_raw(job.league, job.espn_id, http=self.http), 200
            else:  # an EspnClient without the raw door (not this app's): the site summary alone
                from .sources import espn_site
                fetch = espn_site.summary(self.http, job.league, job.espn_id)
                body, status = fetch.body, fetch.status
        except (Refused, Unreachable):
            raise
        except Exception as exc:  # noqa: BLE001 - EspnHttpError: an answer with an error status
            answer = _http_answer(exc)
            if answer is None:
                raise
            status, body = answer
        ok = 200 <= status < 300
        state = (summary_state(_parse(body)) or "unknown") if ok else None
        stored = {"final": "final", "void": "final", "live": "live", "pre": "pre"}.get(state or "", phase)
        if phase == "final" and state == "unknown":
            stored = "live"  # unreadable: kept, but not taken for the final record
        label = horizon if ok and ((phase == "pre" and stored == "pre") or (phase == "final" and stored == "final")) \
            else None
        self._keep(job.game_key, job.league, "espn_site", "summary", {}, bytes(body or b""), stored, status, label,
                   run)
        return state

    def _core_plays(self, job: Job, run: _Run) -> bool:
        """Every page of the core plays, carrying on after the pages already kept (a restart, a budget stop)."""
        from .sources import espn_core
        have = {}
        for payload in self.raw.bundle(job.game_key).pages("espn_core", "core_plays"):
            page = payload.params.get("page")
            if isinstance(page, int) and payload.state == "final":
                have[page] = payload
        first, count = 1, None
        while first in have:
            first += 1
        if 1 in have:
            try:
                count = int((have[1].json() or {}).get("pageCount") or 0)
            except (ValueError, TypeError, AttributeError):
                count = None
        if count and first > count:
            return True  # every page is in hand already
        kept = [self._keep_fetch(fetch, job.game_key, job.league, "final", "final", run)
                for fetch in espn_core.plays_all(self.http, job.league, job.espn_id, first_page=first)]
        return bool(kept) and all(kept)

    @property
    def mlb(self) -> Any:
        if self._mlb is None:
            from ..data.mlb_stats import MlbStats
            self._mlb = MlbStats(self.cache)
        return self._mlb

    def game_pk(self, job: Job) -> Optional[int]:
        """The StatsAPI gamePk of an ESPN MLB game: the day's schedule (data/mlb_stats.py, cached), matched on both
        teams (from the game's stored summary), the start nearest the ESPN one (a double-header)."""
        summary = self.raw.latest(job.game_key, "espn_site", "summary")
        data = summary.json() if summary is not None else None
        competitions = (data.get("header") or {}).get("competitions") if isinstance(data, dict) else None
        competitors = (competitions[0].get("competitors") if competitions and isinstance(competitions[0], dict)
                       else None) or []
        sides = {c.get("homeAway"): (c.get("team") or {}) for c in competitors if isinstance(c, dict)}
        if "home" not in sides or "away" not in sides:
            return None
        best: Optional[Tuple[float, int]] = None
        for day in sorted({espn_day(job.start), espn_day(job.start - 6 * 3600.0)}):
            for game in self.mlb.games(day) or []:
                home, away = (game.get("home") or {}).get("name"), (game.get("away") or {}).get("name")
                if not home or not away or game.get("game_pk") is None:
                    continue
                if team_match_score(str(home), sides["home"]) < 0.5 or team_match_score(str(away), sides["away"]) < 0.5:
                    continue
                start = _iso_ts(game.get("start"))
                gap = abs((start if start is not None else job.start) - job.start)
                if best is None or gap < best[0]:
                    best = (gap, int(game["game_pk"]))
        return best[1] if best else None

    def _feed_live(self, job: Job, run: _Run) -> bool:
        from .sources import statsapi
        try:
            pk = self.game_pk(job)
        except Exception as exc:  # noqa: BLE001 - the schedule couldn't be read: try again later
            run.errors.append(f"feed_live: no StatsAPI schedule ({type(exc).__name__})")
            return False
        if pk is None:
            run.errors.append("feed_live: no StatsAPI game matches")
            return False
        return self._keep_fetch(statsapi.feed_live(self.http, pk), job.game_key, job.league, "final", "final", run)

    def _board(self, league: str, day: datetime.date, run: _Run, final: Optional[bool] = None) -> Optional[Any]:
        """Fetch and keep a league's board for one day raw (scoreboard_raw: the lean cache gets its copy from the same
        answer), under scoreboard_key(league, "YYYYMMDD"); returns the parsed board, None after an error. It is kept
        as "final" once its day is over (or ``final``: the card or match it is fetched for is)."""
        dates = _yyyymmdd(day)
        try:
            if hasattr(self.espn, "scoreboard_raw"):
                body, status = self.espn.scoreboard_raw(league, day, http=self.http), 200
            else:
                from .sources import espn_site
                fetch = espn_site.scoreboard(self.http, league, day)
                body, status = fetch.body, fetch.status
        except (Refused, Unreachable):
            raise
        except Exception as exc:  # noqa: BLE001
            answer = _http_answer(exc)
            if answer is None:
                raise
            status, body = answer
        over = day < espn_day(self.clock()) if final is None else bool(final)
        ok = self._keep(scoreboard_key(league, dates), league, "espn_site", "scoreboard", {"dates": dates},
                        bytes(body or b""), "final" if over else "live", status, "final" if over else None, run)
        return _parse(body) if ok else None

    def _final_card(self, job: Job, run: _Run) -> None:
        """A UFC card: its fightcenter once every bout is over (with core status and fighter statistics for a bout it
        has no result for), and its day's board."""
        from .sources import bout_key, espn_core, espn_fightcenter
        done = set(job.kinds_done)
        if plan.final_name("fightcenter") not in done:
            def card() -> bool:
                # A final fightcenter and bout answers already kept (before a budget stop or a restart) are not
                # fetched again: the card resumes where it stopped, as the core plays' pages do
                held = self.raw.latest(job.game_key, espn_fightcenter.SOURCE, "fightcenter")
                if held is not None and held.state == "final":
                    bouts = espn_fightcenter.bouts(held.json())
                else:
                    fetch = espn_fightcenter.card(self.http, job.espn_id)
                    payload = fetch.json() if fetch.ok else None
                    bouts = espn_fightcenter.bouts(payload) if payload is not None else []
                    over = bool(bouts) and all(bout["state"] == "final" for bout in bouts)
                    kept = self._keep_fetch(fetch, job.game_key, job.league, "final" if over else "live",
                                            "final" if over else None, run)
                    if not kept:
                        return False
                    if not over:
                        run.hard = True
                        run.errors.append("fightcenter: the card isn't over yet")
                        return False
                ok = True
                for bout in (b for b in bouts if not b.get("result")):
                    key = bout_key(bout["id"])
                    if not self.raw.has_final(key, espn_core.SOURCE, "core_status"):
                        ok &= self._keep_fetch(espn_core.status(self.http, job.league, job.espn_id, bout["id"]), key,
                                               job.league, "final", "final", run)
                    for competitor in bout.get("competitors") or []:
                        stats = self.raw.latest(key, espn_core.SOURCE, "core_competitor_stats",
                                                {"competitor": competitor})
                        if stats is None or stats.state != "final":
                            ok &= self._keep_fetch(espn_core.competitor_stats(self.http, job.league, job.espn_id,
                                                                              bout["id"], competitor),
                                                   key, job.league, "final", "final", run)
                return ok

            if self._attempt(run, "fightcenter", card):
                run.done.extend(plan.final_name(kind) for kind in ("fightcenter", "core_status",
                                                                   "core_competitor_stats"))
            else:
                return
        if plan.final_name("scoreboard") not in done:
            if self._attempt(run, "scoreboard",
                             lambda: self._board(job.league, espn_day(job.start), run, final=True) is not None):
                run.done.append(plan.final_name("scoreboard"))

    def _final_tennis(self, job: Job, run: _Run) -> None:
        """A tennis match: its day's board (marks every finished match on it done)."""
        def board() -> bool:
            data = self._board(job.league, espn_day(job.start), run)
            if data is None:
                return False
            finals = _tennis_finals(data)
            self._mark_tennis(job.league, finals, skip=job.game_key)
            if job.espn_id not in finals:
                run.hard = True
                run.errors.append("scoreboard: the match isn't over on its day's board")
                return False
            return True

        if self._attempt(run, "scoreboard", board):
            run.done.append(plan.final_name("scoreboard"))

    def _mark_tennis(self, league: str, finals: Mapping[str, float], skip: Optional[str] = None,
                     add: bool = False) -> int:
        """Mark the tennis jobs of finished matches done (a board was just kept); ``add`` makes a job, done, for a
        match nobody opened. Returns the jobs made."""
        made = 0
        now = self.clock()
        for competition_id, start in finals.items():
            key = match_key(competition_id)
            if key == skip:
                continue
            created: List[bool] = []

            def change(job: Optional[Job], competition_id: str = competition_id, start: float = start,
                       key: str = key) -> Optional[Job]:
                if job is None:
                    if not add:
                        return None
                    created.append(True)
                    return Job(key, league, competition_id, float(start or now), now, state="done",
                               kinds_done=[plan.final_name("scoreboard")])
                if job.state in ("done", "gave_up", "void"):
                    return None
                job.kinds_done = sorted(set(job.kinds_done) | {plan.final_name("scoreboard")})
                return self._plan(job, now)

            self.raw.modify_job(key, change, now=now)
            made += 1 if created else 0
        return made

    # -- the nightly sweep --

    def sweep_time(self, day: datetime.date) -> float:
        """When ``day``'s boards are swept: SWEEP_AT on the next ESPN day."""
        return datetime.datetime.combine(day + datetime.timedelta(days=1), datetime.time(SWEEP_AT),
                                         tzinfo=ESPN_DAY_TZ).timestamp()

    def sweep_day(self, now: Optional[float] = None) -> Optional[datetime.date]:
        """The day whose sweep is due at ``now`` (yesterday, once it is SWEEP_AT on ESPN's day; a sweep with failed
        boards tries them again SWEEP_RETRY later, SWEEP_TRIES times at most), or None."""
        now = self.clock() if now is None else now
        day = espn_day(now) - datetime.timedelta(days=1)
        if now < self.sweep_time(day) or self.raw.get_state("sweep:last") == day.isoformat():
            return None
        progress = self.raw.get_state(f"sweep:{day.isoformat()}") or {}
        return None if now < float(progress.get("next") or 0.0) else day

    def sweep(self, day: datetime.date) -> int:
        """Keep ``day``'s board of every one of the 58 paths raw and make a job for each final game nobody opened (a
        tennis match's and a card's board part done). Resumes: a league whose board is kept is not read again for the
        same day. Returns the jobs made."""
        if not self.room():
            self.last["sweep"] = {"day": day.isoformat(), "stopped": "disk"}
            return 0
        with self._lock, self._forward(None):
            mark = f"sweep:{day.isoformat()}"
            progress = dict(self.raw.get_state(mark) or {})
            done = set(progress.get("leagues") or [])
            made, failed, stopped = int(progress.get("made") or 0), [], False
            run = _Run(picked=self.clock())
            for league, sport, _path in plan.iter_paths():
                if league in done:
                    continue
                if not self._hold_lease():
                    stopped = True  # another process captures now (and carries on with the sweep)
                    break
                if not self.room():
                    stopped = "disk"  # raw.db can't take more: the rest waits (no try spent)
                    break
                try:
                    data = self._board(league, day, run)
                except Refused:
                    stopped = True  # the day's budget: the next tick carries on
                    break
                except Unreachable as exc:
                    failed.append(league)
                    log.info("Capture sweep: %s %s: %s", league, day, exc)
                    continue
                except Exception as exc:  # noqa: BLE001 - one odd board costs only itself (tried again with the failed)
                    if disk_is_full(exc):
                        stopped = "disk"  # the first board raw.db couldn't store stops the sweep: no more fetched
                        break
                    failed.append(league)
                    log.exception("Capture sweep: couldn't keep the %s board of %s", league, day)
                    continue
                if data is None:
                    failed.append(league)
                    continue
                try:
                    made += self._sweep_board(league, sport, data)
                except Exception as exc:  # noqa: BLE001 - one odd board costs only itself
                    if disk_is_full(exc):
                        stopped = "disk"
                        break
                    log.exception("Capture sweep: couldn't read the %s board of %s", league, day)
                done.add(league)
                progress.update({"leagues": sorted(done), "made": made})
                try:
                    self.raw.set_state(mark, progress)
                except Exception as exc:  # noqa: BLE001
                    if not disk_is_full(exc):
                        raise
                    stopped = "disk"
                    break
            if not stopped:
                tries = int(progress.get("tries") or 0) + 1
                progress.update({"leagues": sorted(done), "made": made, "tries": tries, "failed": failed,
                                 "next": self.clock() + SWEEP_RETRY})
                self.raw.set_state(mark, progress)
                if not failed or tries >= SWEEP_TRIES:
                    self.raw.set_state("sweep:last", day.isoformat())
            self.last["sweep"] = {"day": day.isoformat(), "boards": run.fetched, "stored": run.stored,
                                  "failed": failed, "made": made, "stopped": stopped}
        return made

    def _sweep_board(self, league: str, sport: str, data: Any) -> int:
        now = self.clock()
        made = 0
        if sport == "tennis":
            return self._mark_tennis(league, _tennis_finals(data), add=True)
        for event in _events(data):
            event_id = str(event.get("id") or "").strip()
            competitions = [c for c in event.get("competitions") or [] if isinstance(c, dict)]
            if not event_id or not competitions:
                continue
            states = [competition_state(c) for c in (competitions if sport == "mma" else competitions[:1])]
            if not states or any(state is None or state[0] != "final" for state in states):
                continue
            starts = [state[3] for state in states if state and state[3] is not None]
            start = min(starts) if starts else _iso_ts(event.get("date"))
            if start is None:
                continue
            made += self._upsert(espn_key(event_id), league, event_id, float(start), "final", now)
            if sport == "mma":  # this board is the card's record of the day
                self._mark_kind(espn_key(event_id), plan.final_name("scoreboard"))
        return made

    def _mark_kind(self, key: str, name: str) -> None:
        now = self.clock()

        def change(job: Optional[Job]) -> Optional[Job]:
            if job is None or job.state in ("done", "gave_up", "void") or name in job.kinds_done:
                return None
            job.kinds_done = sorted(set(job.kinds_done) | {name})
            return self._plan(job, now)

        self.raw.modify_job(key, change, now=now)

    # -- static history files --

    def _static_fetches(self, name: str, now: float) -> Iterator[RawFetch]:
        if name == "greco1899":
            from .sources import greco
            yield from greco.csvs(self.http)
        elif name == "mcp":
            from .sources import mcp
            yield from mcp.csvs(self.http)
        elif name == "football_data":
            from .sources import football_data_raw
            today = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).date()
            season = today.year if today.month >= 7 else today.year - 1
            for code in football_data_raw.CODES:
                yield football_data_raw.season_csv(self.http, code, season)
            for code in football_data_raw.NEW:
                yield football_data_raw.season_csv(self.http, code)
        elif name == "nflverse":
            from .sources import nflverse
            today = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).date()
            yield nflverse.pbp_season(self.http, today.year if today.month >= 9 else today.year - 1)
            yield nflverse.officials(self.http)
        else:
            raise ValueError(f"unknown static source {name!r}")

    def static_due(self, entry: plan.Static, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        last = self.raw.get_state(f"static:{entry.name}")
        if isinstance(last, Mapping) and last.get("ok") is False:
            return now - float(last.get("at") or 0.0) >= STATIC_RETRY
        return plan.static_due(entry, last if isinstance(last, Mapping) else None, now, self.tz)

    def run_static(self, names: Optional[Sequence[str]] = None, *, force: bool = False) -> Dict[str, Any]:
        """Fetch and keep the static sources that are due (or ``names``, with ``force`` whatever their schedule).
        Returns {name: {"fetched", "stored", "ok"}} for each one run."""
        from .sources import static_unit
        report: Dict[str, Any] = {}
        if not self.room():
            return {"stopped": "disk"}
        with self._lock, self._forward(None):
            for entry in plan.STATIC:
                if names is not None and entry.name not in names:
                    continue
                now = self.clock()
                if not force and not self.static_due(entry, now):
                    continue
                run = _Run(picked=now)
                ok, stopped = True, False
                try:
                    for fetch in self._static_fetches(entry.name, now):
                        key, league = static_unit(fetch)
                        ok &= self._keep_fetch(fetch, key, league, "static", None, run)
                except Refused:
                    stopped = True
                except Unreachable as exc:
                    ok = False
                    run.errors.append(str(exc))
                except Exception as exc:  # noqa: BLE001
                    if not disk_is_full(exc):
                        raise
                    report[entry.name] = {"fetched": run.fetched, "stored": run.stored, "ok": False}
                    report["stopped"] = "disk"  # raw.db can't store: the rest waits for room
                    break
                report[entry.name] = {"fetched": run.fetched, "stored": run.stored, "ok": ok and not stopped}
                if stopped:
                    break  # the day's budget: the next tick carries on
                self.raw.set_state(f"static:{entry.name}", {"at": now, "ok": ok, "period":
                                                           plan.static_period(entry, now, self.tz),
                                                           "errors": run.errors[-3:]})
        return report

    # -- the older-games backfill --

    @property
    def backfill(self) -> Any:
        """The RawBackfill (capture/backfill.py), built on first use; None when CAPTURE_BACKFILL is off or it can't
        start."""
        if self._backfill is None and not self._backfill_tried and self.backfill_enabled:
            self._backfill_tried = True
            try:
                from .backfill import RawBackfill
                cfg = self.config
                self._backfill = RawBackfill(
                    self.raw, self.http, self.clock, seasons=int(getattr(cfg, "capture_backfill_seasons", 3) or 3),
                    per_night=int(getattr(cfg, "capture_backfill_per_night", 6000)), tz=self._backfill_tz(),
                    forward_due=self.has_runnable_due)
            except Exception:  # noqa: BLE001 - forward capture works without it
                log.exception("Capture: the older-games backfill couldn't start")
                self._backfill = None
        return self._backfill

    def _backfill_tz(self) -> Optional[datetime.tzinfo]:
        """The backfill window's time zone: the one given (``tz``: tests), else the owner's TIMEZONE
        (capture.backfill.backfill_tz: 01:00-06:00 New York is the quiet window for US sports and European soccer
        alike; server-local time when the name is invalid)."""
        if self.tz is not None or self.config is None:
            return self.tz
        from .backfill import backfill_tz
        return backfill_tz(self.config)

    def run_backfill(self, chunk: int = BACKFILL_CHUNK, max_seconds: float = BACKFILL_TICK) -> Optional[Dict[str, Any]]:
        """The backfill inside its window, in chunks of ``chunk`` requests, forward jobs first between chunks (and
        before every request: RawBackfill checks forward_due). Returns a summary, or None when it didn't run."""
        backfill = self.backfill
        if backfill is None or not self.backfill_enabled or not backfill.in_window():
            return None
        started = self.clock()
        total: Dict[str, Any] = {"chunks": 0, "requests": 0, "stored": 0, "stopped": None}
        idle = 0
        while self.clock() - started < max_seconds:
            if not self.room():
                total["stopped"] = "disk"
                break
            if self.has_runnable_due():
                progress = self.run_due(TICK_BUDGET)
                if not progress["fetched"] and self.has_runnable_due():
                    total["stopped"] = "forward"  # forward work that can't run now holds the backfill back
                    break
            if not self.lease():
                total["stopped"] = "lease"
                break
            if self._lease_at is not None:
                self._lease_at = self.clock()
            report = dict(backfill.run_night(budget=chunk) or {})
            total["chunks"] += 1
            total["requests"] += int(report.get("requests") or 0)
            total["stored"] += int(report.get("stored") or 0)
            total["stopped"] = report.get("stopped")
            total["progress"] = report.get("progress")
            if report.get("stopped") in ("learn", "backoff"):
                # The learn backfill runs now (learn.history.LEARN_BACKFILL_RUNNING; both run in this dashboard
                # process), or only retries are left whose wait isn't over: this tick's chunks stop, the next tick
                # tries again
                break
            if report.get("stopped") == "forward":
                idle = idle + 1 if not report.get("requests") else 0
                if idle >= 3:
                    break
                continue
            if report.get("stopped") != "budget" or not report.get("requests"):
                break
        return total

    # -- the thread's step --

    def tick(self, now: Optional[float] = None) -> Dict[str, Any]:
        """One step of the capture thread: the lease, yesterday's sweep when due, the due jobs, then (nothing forward
        being due) the static sources and the backfill."""
        now = self.clock() if now is None else now
        report: Dict[str, Any] = {"at": now}
        if not self.lease():
            self._lend_door(False)
            report["skipped"] = "another process holds the capture lease"
            return report
        self._lend_door(True)
        self._lease_at = self.clock()
        try:
            self._tick(now, report)
        finally:
            self._lease_at = None
        self.last = {**self.last, "tick": report}
        return report

    def _tick(self, now: float, report: Dict[str, Any]) -> None:
        try:
            day = self.sweep_day(now)
            if day is not None:
                report["sweep"] = self.sweep(day)
        except Exception:  # noqa: BLE001 - the loop goes on
            log.exception("Capture: the sweep failed")
        try:
            report["due"] = self.run_due(TICK_BUDGET)
        except Exception:  # noqa: BLE001
            log.exception("Capture: running the due jobs failed")
        if not self.has_runnable_due():
            try:
                report["static"] = self.run_static()
            except Exception:  # noqa: BLE001
                log.exception("Capture: the static sources failed")
        if not self.has_runnable_due():
            try:
                report["backfill"] = self.run_backfill()
            except Exception:  # noqa: BLE001
                log.exception("Capture: the backfill failed")


def capture_forever(capturer: Capturer, every: float = TICK_SECONDS, rounds: Optional[int] = None,
                    sleep: Callable[[float], None] = time.sleep) -> None:
    """The dashboard's "capture" thread: a tick a minute (``rounds`` stops it early, for tests)."""
    done = 0
    while rounds is None or done < rounds:
        try:
            capturer.tick()
        except Exception:  # noqa: BLE001 - the loop never dies
            log.exception("The capture's tick failed")
        done += 1
        if rounds is None or done < rounds:
            sleep(every)


def build_capturer(config: Any, service: Any) -> Optional[Capturer]:
    """The dashboard's Capturer over the service's store, ESPN client and cache, with raw.db at RAW_DB_PATH (default
    beside betting.db); None when CAPTURE_ENABLED is off."""
    if not getattr(config, "capture_enabled", True):
        return None
    raw = RawStore(raw_db_file(config, getattr(service, "store", None)),
                   min_free_gb=getattr(config, "capture_min_free_gb", None))
    return Capturer(service.store, raw, service.espn, PoliteHttp(), time.time, config=config,
                    cache=getattr(service, "cache", None), share_door=True)


__all__ = ["Capturer", "capture_forever", "build_capturer", "raw_db_file", "options_from", "summary_state",
           "espn_day", "LEASE", "LEASE_SECONDS", "TICK_SECONDS", "TICK_BUDGET", "BACKFILL_CHUNK", "SWEEP_AT"]
