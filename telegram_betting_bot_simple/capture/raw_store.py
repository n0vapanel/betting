"""The raw capture store (v4 P1, plan section 3.3): a separate WAL SQLite file, so raw blobs never bloat betting.db.

``RawStore(path)`` (RAW_DB_PATH, default ``<STATE_DIR>/raw.db``) keeps:

- **``raw_payloads``**: every response capture fetched, as it came (``sha256`` and ``bytes_raw`` are of the response's
  own bytes), compressed with zlib level 6 (``codec`` "zlib6"), or kept as they are ("identity") when they already are
  a gzip or zip file (nflverse's ``.csv.gz``). ``put`` deduplicates on (game_key, source, endpoint, params, sha256):
  the same body again stores nothing and returns the first row's id, a corrected body (a new sha256) is a new row.
  ``params`` is the canonical JSON of the query (sorted keys, compact; the JSON types the fetchers fix: ``page`` an int,
  ``dates`` a string), so ``latest`` finds a row when asked with the same dict.
- **``capture_jobs``**: one per game (capture/worker.py): when it is next due, how far it got (``kinds_done``), its
  state (todo, partial, done, gave_up, void), attempts and last error.
- **``capture_budget``**: request counters per (kind, day): ``spend(kind, day, n, limit)`` counts atomically against a
  limit, ``spent(kind, day)`` reads one. Kinds: ``capture`` (forward requests, CAPTURE_PER_DAY), ``backfill``
  (CAPTURE_BACKFILL_PER_NIGHT), ``prices`` (propBets and Kalshi, inside CAPTURE_PER_DAY too), and the outcome counts
  ``http_ok`` / ``http_error`` (every capture request: a 2xx answer, or a non-2xx one or an unreachable host). The
  health block (learn/progress.py) reads this table read-only with SQL: its columns are a stable contract
  (``kind TEXT, day TEXT 'YYYY-MM-DD', used INTEGER, updated_at REAL``; capture, prices and the http counts use UTC
  days, the backfill its night).
- **``backfill_progress``**: the older-games backfill's cursor per (league, season) with its counts
  (capture/backfill.py): ``get_cursor``, ``set_cursor``, ``progress``.
- **``capture_state``**: small JSON values the worker keeps (when each static source last ran, the sweep's days).

Reads: ``latest`` (the newest payload of one kind), ``bundle`` (the newest payload per (source, endpoint, params) of a
game), ``iter_games``, ``has_final``; and for the health block ``size_bytes``, ``job_counts``, ``recent_errors``,
``http_counts`` (the last HEALTH_DAYS days by default, read by the fetch-time index ``raw_payloads_fetched``: the one
addition to section 3.3's schema, so the health reads never scan the table) and ``counts``. ``RawPayload.json()``
decompresses and parses.

**The disk floor.** ``room()`` (``disk_room``): raw.db's disk must keep CAPTURE_MIN_FREE_GB (default 2 GiB) and
DISK_FLOOR_SHARE (10%) free, or the capture and the backfill fetch nothing, so raw.db can never fill the disk betting.db
is on. ``resolve_raw_db_path`` is where raw.db is, the same for every reader.
"""

from __future__ import annotations

import datetime
import contextlib
import gzip
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import zlib
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_payloads (
    id INTEGER PRIMARY KEY,
    game_key TEXT NOT NULL,      -- espn:<eventId> (team games, as hist_games; a UFC card) | espn-bout:<competitionId> |
                                 -- espn-match:<competitionId> (tennis) | mlb:<gamePk> | static:<source>:<unit>
    league TEXT NOT NULL,        -- registry league key
    source TEXT NOT NULL,        -- espn_site | espn_core | espn_fightcenter | mlb_statsapi | nflverse | greco1899 |
                                 -- mcp | cricsheet | jolpica | espn_propbets | kalshi | football_data
    endpoint TEXT NOT NULL,      -- summary | core_plays | core_status | core_competitor_stats | fightcenter |
                                 -- scoreboard | feed_live | pbp_season | officials | propbets | events | csv
    params TEXT NOT NULL,        -- canonical JSON of the query (page, limit, competitor id); '{}' when none
    state TEXT NOT NULL,         -- pre | live | final | static
    horizon TEXT,                -- open | t24 | t90 | close | final | NULL
    fetched_at REAL NOT NULL,    -- epoch seconds
    http_status INTEGER NOT NULL,
    sha256 TEXT NOT NULL,        -- of the raw response bytes
    bytes_raw INTEGER NOT NULL,
    codec TEXT NOT NULL DEFAULT 'zlib6',
    body BLOB NOT NULL,
    UNIQUE (game_key, source, endpoint, params, sha256)
);
CREATE INDEX IF NOT EXISTS raw_by_game ON raw_payloads (game_key, source, endpoint, fetched_at);
-- Beyond section 3.3: the health reads (http_counts, recent_errors) by fetch time, from the index alone
CREATE INDEX IF NOT EXISTS raw_payloads_fetched ON raw_payloads (fetched_at, http_status);
CREATE TABLE IF NOT EXISTS capture_jobs (
    game_key TEXT PRIMARY KEY,
    league TEXT NOT NULL,
    espn_id TEXT,
    start REAL NOT NULL,
    due_at REAL NOT NULL,
    state TEXT NOT NULL DEFAULT 'todo',   -- todo | partial | done | gave_up | void
    attempts INTEGER NOT NULL DEFAULT 0,
    kinds_done TEXT NOT NULL DEFAULT '[]',
    last_error TEXT,
    updated_at REAL
);
CREATE INDEX IF NOT EXISTS capture_jobs_due ON capture_jobs (state, due_at);
-- Beyond section 3.3: the health block's job reads (coverage by start, the newest errors)
CREATE INDEX IF NOT EXISTS capture_jobs_start ON capture_jobs (start);
CREATE INDEX IF NOT EXISTS capture_jobs_errors ON capture_jobs (updated_at) WHERE last_error IS NOT NULL;
-- Beyond section 3.3 (additive): each time a body already kept came again (a dedup hit), when, at which horizon and in
-- which state; a payload's first sighting is its own row (fetched_at, horizon, state)
CREATE TABLE IF NOT EXISTS raw_seen (
    payload_id INTEGER NOT NULL,     -- raw_payloads.id
    seen_at REAL NOT NULL,           -- epoch seconds
    horizon TEXT,                    -- open | t24 | t90 | close | final | NULL
    state TEXT NOT NULL              -- pre | live | final | static
);
CREATE INDEX IF NOT EXISTS raw_seen_by_payload ON raw_seen (payload_id, seen_at);
CREATE TABLE IF NOT EXISTS capture_budget (
    kind TEXT NOT NULL,          -- capture | backfill | prices | http_ok | http_error | backfill_http_ok |
                                 -- backfill_http_error
    day TEXT NOT NULL,           -- YYYY-MM-DD: a UTC day (the backfill's budget: its night)
    used INTEGER NOT NULL DEFAULT 0,
    updated_at REAL,
    PRIMARY KEY (kind, day)
);
CREATE TABLE IF NOT EXISTS backfill_progress (
    league TEXT NOT NULL,
    season INTEGER NOT NULL,
    cursor TEXT,                 -- JSON: where the walk of this season is (the backfill's own shape); NULL: not begun
    done INTEGER NOT NULL DEFAULT 0,
    games_seen INTEGER NOT NULL DEFAULT 0,
    games_stored INTEGER NOT NULL DEFAULT 0,
    updated_at REAL,
    PRIMARY KEY (league, season)
);
CREATE TABLE IF NOT EXISTS capture_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,         -- JSON
    updated_at REAL
);
"""

STATES = ("pre", "live", "final", "static")
HORIZONS = ("open", "t24", "t90", "close", "final")
JOB_STATES = ("todo", "partial", "done", "gave_up", "void")
OPEN_JOB_STATES = ("todo", "partial")
# capture, prices: forward requests (CAPTURE_PER_DAY); backfill: the older-games backfill's (its night); http_ok,
# http_error: forward requests' outcomes; backfill_http_ok, backfill_http_error: the backfill's (UTC days)
BUDGET_KINDS = ("capture", "backfill", "prices", "http_ok", "http_error", "backfill_http_ok", "backfill_http_error")
BUDGET_KEPT_DAYS = 400  # counters older than this are dropped
DISK_FLOOR_SHARE = 0.10  # raw.db never fills its disk past this share of free space ...
MIN_FREE_GB = 2.0  # ... nor below this many GiB free (CAPTURE_MIN_FREE_GB)
ROOM_EVERY = 60.0  # RawStore.room() reads the disk at most this often
HEALTH_DAYS = 7  # http_counts and recent_errors look back this far by default
FULL_PAUSE = 300.0  # after SQLite says "database or disk is full", room() is False this long ...
FULL_LOG_EVERY = 3600.0  # ... and that is logged at most this often
# A payload's newest sighting: its own fetch, or a later dedup hit of the same body
SEEN_SQL = ("MAX(p.fetched_at, COALESCE((SELECT MAX(s.seen_at) FROM raw_seen s WHERE s.payload_id = p.id), "
            "p.fetched_at))")
GZIP_MAGIC = b"\x1f\x8b"
ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06")
ERROR_TEXT = 500  # a job's last_error is cut to this many characters

log = logging.getLogger(__name__)

PAYLOAD_COLUMNS = ("p.id, p.game_key, p.league, p.source, p.endpoint, p.params, p.state, p.horizon, p.fetched_at, "
                   "p.http_status, p.sha256, p.bytes_raw, p.codec, p.body")
JOB_COLUMNS = "game_key, league, espn_id, start, due_at, state, attempts, kinds_done, last_error, updated_at"


# --- encoding ----------------------------------------------------------------------------------------------------------

def canonical_params(params: Optional[Mapping[str, Any]]) -> str:
    """The canonical JSON of a query: sorted keys, compact, None values left out; '{}' when there are none. The JSON
    types are kept (``{"page": 1}`` and ``{"page": "1"}`` are different queries), as the fetchers fix them."""
    clean = {str(key): value for key, value in dict(params or {}).items() if value is not None}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def codec_for(body: bytes) -> str:
    """"identity" for a body that already is a gzip or zip file, else "zlib6"."""
    return "identity" if body[:2] == GZIP_MAGIC or body[:4] in ZIP_MAGICS else "zlib6"


def encode(body: bytes, codec: str) -> bytes:
    if codec == "zlib6":
        return zlib.compress(body, 6)
    if codec == "identity":
        return body
    raise ValueError(f"unknown codec {codec!r}")


def decode(blob: bytes, codec: str) -> bytes:
    if codec == "zlib6":
        return zlib.decompress(blob)
    if codec == "identity":
        return bytes(blob)
    raise ValueError(f"unknown codec {codec!r}")


def day_text(day: Union[str, datetime.date, datetime.datetime]) -> str:
    """A day as "YYYY-MM-DD" (a date, a datetime's date, or that text, checked)."""
    if isinstance(day, datetime.datetime):
        return day.date().isoformat()
    if isinstance(day, datetime.date):
        return day.isoformat()
    return datetime.date.fromisoformat(str(day)).isoformat()


def utc_day(ts: float) -> str:
    """The UTC day of an epoch time, "YYYY-MM-DD"."""
    return datetime.datetime.fromtimestamp(float(ts), datetime.timezone.utc).date().isoformat()


def disk_room(path: str, min_free_bytes: float) -> bool:
    """Whether the disk holding ``path`` (raw.db, or its folder) has room for more raw payloads: free space at least
    ``min_free_bytes`` and at least DISK_FLOOR_SHARE of the disk, so the capture and the backfill can never fill the
    disk betting.db lives on. True when the usage can't be read (it never raises)."""
    try:
        target = str(path)
        folder = target if os.path.isdir(target) else (os.path.dirname(os.path.abspath(target)) or ".")
        while not os.path.isdir(folder):
            parent = os.path.dirname(folder)
            if parent == folder:
                break
            folder = parent
        usage = shutil.disk_usage(folder)
        return float(usage.free) >= max(float(min_free_bytes), DISK_FLOOR_SHARE * float(usage.total))
    except Exception:  # noqa: BLE001 - an unreadable disk never stops the capture
        return True


def min_free_bytes(gb: Any = None) -> float:
    """CAPTURE_MIN_FREE_GB as bytes: ``gb`` when given, else the environment's, else MIN_FREE_GB (2 GiB); a bad value
    falls back to the default."""
    raw = os.environ.get("CAPTURE_MIN_FREE_GB", "") if gb is None else gb
    try:
        value = float(str(raw).strip()) if str(raw).strip() else MIN_FREE_GB
    except ValueError:
        value = MIN_FREE_GB
    return max(0.0, value if value == value else MIN_FREE_GB) * 1024 ** 3


_PATH_WARNED: Set[Tuple[str, str]] = set()  # (RAW_DB_PATH, why) already warned about in this process
SQLITE_MAGIC = b"SQLite format 3\x00"


def not_a_raw_store(path: str, betting_db: Optional[str] = None) -> Optional[str]:
    """Why ``path`` can't be raw.db (None when it can): it is betting.db, a folder, a file that isn't SQLite, or a
    SQLite database of something else (betting.db under another name). A path that doesn't exist yet, an empty file and
    an empty database are fine; a check that can't be made doesn't block (RawStore then fails, and capture is off)."""
    try:
        target = os.path.abspath(os.path.expanduser(str(path)))
        if betting_db:
            other = os.path.abspath(str(betting_db))
            if target == other or (os.path.exists(target) and os.path.exists(other) and os.path.samefile(target, other)):
                return "it is betting.db"
        if os.path.isdir(target):
            return "it is a folder"
        if not os.path.isfile(target) or os.path.getsize(target) == 0:
            return None
        with open(target, "rb") as handle:
            if handle.read(len(SQLITE_MAGIC)) != SQLITE_MAGIC:
                return "it isn't a SQLite file"
        with closing(sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=5)) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if tables and "raw_payloads" not in tables:
            return "it is another database, not a raw store"
    except Exception:  # noqa: BLE001 - can't tell: RawStore will try (and capture is off if it fails)
        return None
    return None


def resolve_raw_db_path(config: Any = None, state_dir_or_betting_db: Any = None) -> str:
    """Where raw.db is, the same for every reader (the capture thread, the learn backfill's raw-first hook, the
    health block): ``config.raw_db_path`` (RAW_DB_PATH) when set, else the RAW_DB_PATH environment variable, else
    ``raw.db`` in the state folder: beside betting.db. ``state_dir_or_betting_db`` is that folder, betting.db's path,
    or a data/store.Store (its ``path``); without it, the config's STATE_DIRECTORY (or the temp folder the services use
    without one). A RAW_DB_PATH that is betting.db, or any existing file that isn't a raw store, is ignored with one
    warning (raw blobs never go into betting.db)."""
    where = state_dir_or_betting_db
    if where is not None and not isinstance(where, (str, os.PathLike)):
        where = getattr(where, "path", None)
    if where is None:
        where = getattr(config, "state_dir", None) or os.path.join(tempfile.gettempdir(), "sportsbet-ai")
    text = os.fspath(where)
    folder = text if os.path.isdir(text) or not (os.path.isfile(text) or text.endswith(".db")) else \
        os.path.dirname(os.path.abspath(text))
    betting_db = os.path.abspath(text) if folder != text else os.path.join(folder, "betting.db")
    default = os.path.join(folder, "raw.db")
    path = getattr(config, "raw_db_path", None) if config is not None else None
    path = path or os.environ.get("RAW_DB_PATH") or None
    if not path or not str(path).strip():
        return default
    path = str(path).strip()
    why = not_a_raw_store(path, betting_db)
    if why is None:
        return path
    if (path, why) not in _PATH_WARNED:
        _PATH_WARNED.add((path, why))
        log.warning("RAW_DB_PATH is ignored (%s): raw.db goes to %s", why, default)
    return default


# --- rows ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RawPayload:
    id: int
    game_key: str
    league: str
    source: str
    endpoint: str
    params: Mapping[str, Any]
    state: str
    horizon: Optional[str]
    fetched_at: float
    http_status: int
    sha256: str
    bytes_raw: int = 0
    codec: str = "zlib6"
    blob: bytes = field(default=b"", repr=False, compare=False)
    seen_at: Optional[float] = None  # the newest sighting of this body (fetched_at when it was seen only once)

    def __post_init__(self) -> None:
        if self.seen_at is None:
            object.__setattr__(self, "seen_at", float(self.fetched_at))

    @property
    def ok(self) -> bool:
        return 200 <= int(self.http_status) < 300

    def raw_bytes(self) -> bytes:
        """The response's bytes, as fetched (a .csv.gz stays a gzip file)."""
        return decode(self.blob, self.codec)

    def text(self) -> str:
        """The body as text (a gzip file is unpacked; UTF-8 with or without a byte-order mark, else Windows-1252, as
        football-data's older files are)."""
        data = self.raw_bytes()
        if data[:2] == GZIP_MAGIC:
            data = gzip.decompress(data)
        elif data[:4] in ZIP_MAGICS:
            raise ValueError(f"payload {self.id} is a zip archive, not text")
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return data.decode("cp1252", errors="replace")

    def json(self) -> Any:
        """Decompress and parse (raises ValueError when the body is not JSON)."""
        return json.loads(self.text())


@dataclass(frozen=True)
class RawBundle:
    """A game's raw payloads: the newest one per (source, endpoint, params), sorted by those."""

    game_key: str
    payloads: Tuple[RawPayload, ...] = ()

    @property
    def league(self) -> Optional[str]:
        return self.payloads[0].league if self.payloads else None

    def __iter__(self) -> Iterator[RawPayload]:
        return iter(self.payloads)

    def __len__(self) -> int:
        return len(self.payloads)

    def __bool__(self) -> bool:
        return bool(self.payloads)

    def __contains__(self, item: Any) -> bool:
        """``(source, endpoint) in bundle``."""
        return any((p.source, p.endpoint) == tuple(item) for p in self.payloads)

    def all(self, source: Optional[str] = None, endpoint: Optional[str] = None) -> List[RawPayload]:
        return [p for p in self.payloads if (source is None or p.source == source)
                and (endpoint is None or p.endpoint == endpoint)]

    def get(self, source: str, endpoint: str, params: Optional[Mapping[str, Any]] = None) -> Optional[RawPayload]:
        """The payload of (source, endpoint) with these params; with params None, the newest of any params."""
        found = self.all(source, endpoint)
        if params is not None:
            wanted = canonical_params(params)
            found = [p for p in found if canonical_params(p.params) == wanted]
        return max(found, key=lambda p: (p.fetched_at, p.id)) if found else None

    def pages(self, source: str, endpoint: str) -> List[RawPayload]:
        """A pager's payloads (core plays, propBets) in page order."""
        def page(p: RawPayload) -> Tuple[int, str]:
            value = p.params.get("page")
            return (int(value) if isinstance(value, int) or str(value).isdigit() else 0, canonical_params(p.params))
        return sorted(self.all(source, endpoint), key=page)

    def endpoints(self) -> Set[Tuple[str, str]]:
        return {(p.source, p.endpoint) for p in self.payloads}


@dataclass
class Job:
    """One row of capture_jobs (``kinds_done`` as a list)."""

    game_key: str
    league: str
    espn_id: Optional[str]
    start: float
    due_at: float
    state: str = "todo"
    attempts: int = 0
    kinds_done: List[str] = field(default_factory=list)
    last_error: Optional[str] = None
    updated_at: Optional[float] = None

    @property
    def open(self) -> bool:
        return self.state in OPEN_JOB_STATES


def _payload(row: Sequence[Any], seen: Optional[float] = None) -> RawPayload:
    (pid, game_key, league, source, endpoint, params, state, horizon, fetched_at, status, digest, size, codec,
     body) = row
    try:
        parsed = json.loads(params) if params else {}
    except ValueError:
        parsed = {}
    return RawPayload(int(pid), str(game_key), str(league), str(source), str(endpoint),
                      parsed if isinstance(parsed, dict) else {}, str(state), horizon, float(fetched_at), int(status),
                      str(digest), int(size), str(codec), bytes(body),
                      seen_at=max(float(fetched_at), float(seen)) if seen is not None else None)


# Each raw.db's disk-full pause, by its absolute path: the health block reads raw.db through its own read-only
# connection and shows the pause from here. One dict for the process is enough: the capture thread (which sets it) and
# the /api/brain/progress route (which reads it) both run in the dashboard process
_FULL_UNTIL: Dict[str, float] = {}
_FULL_LOCK = threading.Lock()


def full_pause(path: str, now: Optional[float] = None) -> Optional[float]:
    """When the disk-full pause of the raw.db at ``path`` ends (epoch seconds), while it is still on; else None (no
    pause, or it is over). Never raises."""
    try:
        with _FULL_LOCK:
            until = _FULL_UNTIL.get(os.path.abspath(str(path)))
        moment = time.time() if now is None else float(now)
        return float(until) if until is not None and moment < until else None
    except Exception:  # noqa: BLE001 - a status read never breaks a page
        return None


def _rollback(db: sqlite3.Connection) -> None:
    """Roll back an open transaction (SQLite may already have rolled it back itself, after SQLITE_FULL say: then a
    ROLLBACK would raise and hide the real error)."""
    if db.in_transaction:
        db.execute("ROLLBACK")


def disk_is_full(exc: BaseException) -> bool:
    """Whether an error is SQLite's SQLITE_FULL ("database or disk is full")."""
    return isinstance(exc, sqlite3.OperationalError) and "disk is full" in str(exc).lower()


def _job(row: Sequence[Any]) -> Job:
    game_key, league, espn_id, start, due_at, state, attempts, kinds, error, updated = row
    try:
        done = json.loads(kinds or "[]")
    except ValueError:
        done = []
    return Job(str(game_key), str(league), None if espn_id is None else str(espn_id), float(start), float(due_at),
               str(state), int(attempts), [str(k) for k in done] if isinstance(done, list) else [], error,
               None if updated is None else float(updated))


# --- the store -------------------------------------------------------------------------------------------------------------

class RawStore:
    """The raw capture store (see the module's docstring). Every method opens its own short connection, so one RawStore
    can be shared by threads; writes that read first run under BEGIN IMMEDIATE. ``min_free_gb`` is
    CAPTURE_MIN_FREE_GB (None: the environment's, else 2) and ``clock`` the time ``room`` caches its answer on."""

    def __init__(self, path: str, *, min_free_gb: Any = None, clock: Optional[Callable[[], float]] = None) -> None:
        self.path = str(path)
        self.min_free = min_free_bytes(min_free_gb)
        self.clock: Callable[[], float] = clock or time.time
        self._room: Optional[Tuple[float, bool]] = None
        self._full_until = 0.0  # SQLite said "database or disk is full": no room until then
        self._full_logged = -1e18
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    def room(self) -> bool:
        """Whether raw.db's disk has room for more (disk_room: CAPTURE_MIN_FREE_GB free and DISK_FLOOR_SHARE of the
        disk), read at most every ROOM_EVERY seconds; and False for FULL_PAUSE after SQLite refused a write as "database
        or disk is full" (a full inode table, a quota: free blocks alone don't show it). The capture and the backfill
        fetch nothing while it is False."""
        now = self.clock()
        if now < self._full_until:
            return False
        held = self._room
        if held is not None and 0.0 <= now - held[0] < ROOM_EVERY:
            return held[1]
        answer = disk_room(self.path, self.min_free)
        if not answer and (held is None or held[1]):
            log.warning("raw.db's disk is nearly full (under %.1f GiB or %d%% free): the capture pauses",
                        self.min_free / 1024 ** 3, int(DISK_FLOOR_SHARE * 100))
        self._room = (now, answer)
        return answer

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.execute("PRAGMA synchronous=NORMAL")
        return db

    def _full(self) -> None:
        """SQLite refused a write: no room for FULL_PAUSE (logged at most once an hour)."""
        now = self.clock()
        self._full_until = now + FULL_PAUSE
        self._room = (now, False)
        with _FULL_LOCK:
            _FULL_UNTIL[os.path.abspath(self.path)] = self._full_until
        if now - self._full_logged >= FULL_LOG_EVERY:
            self._full_logged = now
            log.warning("raw.db can't store more (SQLite: database or disk is full, %s): the capture and the backfill "
                        "pause for %d minutes at a time until it can", self.path, int(FULL_PAUSE // 60))

    @contextlib.contextmanager
    def _writing(self) -> Iterator[sqlite3.Connection]:
        """A connection for a write; SQLite's "database or disk is full" turns room() off (and is raised on)."""
        try:
            with closing(self._connect()) as db:
                yield db
        except sqlite3.OperationalError as exc:
            if disk_is_full(exc):
                self._full()
            raise

    # -- payloads --

    def insert(self, *, game_key: str, league: str, source: str, endpoint: str, params: Optional[Mapping[str, Any]],
               body: bytes, state: str, http_status: int, horizon: Optional[str] = None,
               fetched_at: Optional[float] = None) -> Tuple[int, bool]:
        """``put``, also telling whether a row was added: (row id, new)."""
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}, not {state!r}")
        if horizon is not None and horizon not in HORIZONS:
            raise ValueError(f"horizon must be one of {HORIZONS} or None, not {horizon!r}")
        for name, value in (("game_key", game_key), ("league", league), ("source", source), ("endpoint", endpoint)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string, not {value!r}")
        data = bytes(body or b"")
        digest = hashlib.sha256(data).hexdigest()
        codec = codec_for(data)
        blob = encode(data, codec)
        canon = canonical_params(params)
        when = time.time() if fetched_at is None else float(fetched_at)
        with self._writing() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO raw_payloads (game_key, league, source, endpoint, params, state, horizon, "
                "fetched_at, http_status, sha256, bytes_raw, codec, body) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (game_key, league, source, endpoint, canon, state, horizon, when, int(http_status), digest, len(data),
                 codec, sqlite3.Binary(blob)))
            if cursor.rowcount == 1 and cursor.lastrowid:
                return int(cursor.lastrowid), True
            row = db.execute("SELECT id FROM raw_payloads WHERE game_key = ? AND source = ? AND endpoint = ? AND "
                             "params = ? AND sha256 = ?", (game_key, source, endpoint, canon, digest)).fetchone()
            # The same body again: no new row, but a sighting (the newest answer, every horizon it served)
            db.execute("INSERT INTO raw_seen (payload_id, seen_at, horizon, state) VALUES (?, ?, ?, ?)",
                       (int(row[0]), when, horizon, state))
        return int(row[0]), False

    def put(self, *, game_key: str, league: str, source: str, endpoint: str, params: Optional[Mapping[str, Any]],
            body: bytes, state: str, http_status: int, horizon: Optional[str] = None,
            fetched_at: Optional[float] = None) -> int:
        """Keep one response (deduplicated on its sha256 for this game, source, endpoint and params); returns the row
        id (the existing row's when the same body is already there: that is recorded as a sighting, raw_seen)."""
        return self.insert(game_key=game_key, league=league, source=source, endpoint=endpoint, params=params,
                           body=body, state=state, http_status=http_status, horizon=horizon,
                           fetched_at=fetched_at)[0]

    def latest(self, game_key: str, source: str, endpoint: str, params: Optional[Mapping[str, Any]] = None, *,
               ok_only: bool = True, horizon: Optional[str] = None) -> Optional[RawPayload]:
        """The newest payload of (game_key, source, endpoint) with these params (None: any params), by its newest
        sighting (a body that came back after another one is the newest again); only 2xx answers unless ``ok_only`` is
        False. With ``horizon``: the payload seen most recently at that horizon, even if the body was first kept under
        another one (a close snapshot identical to the t90 one). Its ``seen_at`` is its newest sighting of all."""
        where = "p.game_key = ? AND p.source = ? AND p.endpoint = ?"
        args: List[Any] = [game_key, source, endpoint]
        if params is not None:
            where += " AND p.params = ?"
            args.append(canonical_params(params))
        if ok_only:
            where += " AND p.http_status BETWEEN 200 AND 299"
        if horizon is None:
            order = SEEN_SQL
        else:
            where += (" AND (p.horizon = ? OR EXISTS (SELECT 1 FROM raw_seen s WHERE s.payload_id = p.id AND "
                      "s.horizon = ?))")
            args += [horizon, horizon]
            order = ("MAX(CASE WHEN p.horizon = ? THEN p.fetched_at ELSE -1e300 END, COALESCE((SELECT MAX(s.seen_at) "
                     "FROM raw_seen s WHERE s.payload_id = p.id AND s.horizon = ?), -1e300))")
            args += [horizon, horizon]
        with closing(self._connect()) as db:
            # The ids first (no blob read for the rows passed over), then the one payload
            found = db.execute(f"SELECT p.id FROM raw_payloads p WHERE {where} ORDER BY {order} DESC, p.id DESC "
                               "LIMIT 1", args).fetchone()
            if found is None:
                return None
            row = db.execute(f"SELECT {PAYLOAD_COLUMNS}, {SEEN_SQL} FROM raw_payloads p WHERE p.id = ?",
                             (int(found[0]),)).fetchone()
        return _payload(row[:-1], seen=row[-1]) if row else None

    def bundle(self, game_key: str, *, ok_only: bool = True) -> RawBundle:
        """The game's newest payload per (source, endpoint, params), by newest sighting (2xx answers only unless
        ``ok_only`` is False)."""
        sql = f"SELECT {PAYLOAD_COLUMNS}, {SEEN_SQL} FROM raw_payloads p WHERE p.game_key = ?"
        if ok_only:
            sql += " AND p.http_status BETWEEN 200 AND 299"
        with closing(self._connect()) as db:
            rows = db.execute(sql + " ORDER BY p.fetched_at, p.id", (game_key,)).fetchall()
        newest: Dict[Tuple[str, str, str], RawPayload] = {}
        for row in rows:
            payload = _payload(row[:-1], seen=row[-1])
            key = (payload.source, payload.endpoint, canonical_params(payload.params))
            held = newest.get(key)
            if held is None or (payload.seen_at, payload.id) >= (held.seen_at, held.id):
                newest[key] = payload
        return RawBundle(game_key, tuple(newest[key] for key in sorted(newest)))

    def sightings(self, payload_id: int) -> List[Dict[str, Any]]:
        """Every time a payload's body was seen, oldest first: its own fetch, then each dedup hit ({"seen_at",
        "horizon", "state"})."""
        with closing(self._connect()) as db:
            first = db.execute("SELECT fetched_at, horizon, state FROM raw_payloads WHERE id = ?",
                               (int(payload_id),)).fetchone()
            if first is None:
                return []
            again = db.execute("SELECT seen_at, horizon, state FROM raw_seen WHERE payload_id = ? ORDER BY seen_at, "
                               "rowid", (int(payload_id),)).fetchall()
        return [{"seen_at": float(at), "horizon": horizon, "state": state} for at, horizon, state in [first, *again]]

    def iter_games(self, league: Optional[str] = None, since: Optional[float] = None,
                   until: Optional[float] = None) -> Iterator[str]:
        """The game keys with a payload fetched in [since, until) (either end open when None), of one league or all,
        in the order their first such payload came."""
        sql, args = "SELECT game_key, MIN(fetched_at) AS first FROM raw_payloads WHERE 1 = 1", []  # type: ignore[var-annotated]
        if league is not None:
            sql += " AND league = ?"
            args.append(league)
        if since is not None:
            sql += " AND fetched_at >= ?"
            args.append(float(since))
        if until is not None:
            sql += " AND fetched_at < ?"
            args.append(float(until))
        with closing(self._connect()) as db:
            keys = [row[0] for row in db.execute(sql + " GROUP BY game_key ORDER BY first, game_key", args)]
        yield from keys

    def has_final(self, game_key: str, source: str = "espn_site", endpoint: str = "summary") -> bool:
        """Whether the game has a 2xx payload of (source, endpoint) taken (or seen again) once it was final."""
        with closing(self._connect()) as db:
            row = db.execute("SELECT 1 FROM raw_payloads p WHERE p.game_key = ? AND p.source = ? AND p.endpoint = ? AND "
                             "p.http_status BETWEEN 200 AND 299 AND (p.state = 'final' OR EXISTS (SELECT 1 FROM "
                             "raw_seen s WHERE s.payload_id = p.id AND s.state = 'final')) LIMIT 1",
                             (game_key, source, endpoint)).fetchone()
        return row is not None

    def payload_count(self, game_key: Optional[str] = None) -> int:
        with closing(self._connect()) as db:
            if game_key is None:
                return int(db.execute("SELECT COUNT(*) FROM raw_payloads").fetchone()[0])
            return int(db.execute("SELECT COUNT(*) FROM raw_payloads WHERE game_key = ?", (game_key,)).fetchone()[0])

    # -- jobs --

    def add_job(self, game_key: str, league: str, espn_id: Optional[str], start: float, due_at: float, *,
                kinds_done: Sequence[str] = (), state: str = "todo", now: Optional[float] = None) -> bool:
        """A new job (nothing happens when the game already has one); returns whether it was added."""
        if state not in JOB_STATES:
            raise ValueError(f"a job's state must be one of {JOB_STATES}, not {state!r}")
        when = time.time() if now is None else float(now)
        with self._writing() as db:
            cursor = db.execute(
                f"INSERT OR IGNORE INTO capture_jobs ({JOB_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, 0, ?, NULL, ?)",
                (game_key, league, None if espn_id is None else str(espn_id), float(start), float(due_at), state,
                 json.dumps(sorted(set(kinds_done))), when))
            return cursor.rowcount == 1

    def job(self, game_key: str) -> Optional[Job]:
        with closing(self._connect()) as db:
            row = db.execute(f"SELECT {JOB_COLUMNS} FROM capture_jobs WHERE game_key = ?", (game_key,)).fetchone()
        return _job(row) if row else None

    def jobs_for(self, keys: Iterable[str]) -> Dict[str, Job]:
        """{game_key: job} of the given keys that have a job (one read, in batches)."""
        wanted = list(dict.fromkeys(str(key) for key in keys))
        found: Dict[str, Job] = {}
        with closing(self._connect()) as db:
            for start in range(0, len(wanted), 500):
                batch = wanted[start:start + 500]
                for row in db.execute(f"SELECT {JOB_COLUMNS} FROM capture_jobs WHERE game_key IN "
                                      f"({', '.join('?' for _ in batch)})", batch):
                    job = _job(row)
                    found[job.game_key] = job
        return found

    def save_job(self, job: Job, *, now: Optional[float] = None) -> None:
        """Write a job's row as it is (insert or replace)."""
        if job.state not in JOB_STATES:
            raise ValueError(f"a job's state must be one of {JOB_STATES}, not {job.state!r}")
        job.updated_at = time.time() if now is None else float(now)
        with self._writing() as db:
            db.execute(f"INSERT OR REPLACE INTO capture_jobs ({JOB_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       _job_values(job))

    def modify_job(self, game_key: str, change: Callable[[Optional[Job]], Optional[Job]], *,
                   now: Optional[float] = None) -> Optional[Job]:
        """Read the job, ``change`` it and write the result, in one transaction (so the slate's hook and the worker
        never lose each other's update). ``change`` gets None when there is no job; returning None writes nothing."""
        with self._writing() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(f"SELECT {JOB_COLUMNS} FROM capture_jobs WHERE game_key = ?", (game_key,)).fetchone()
                job = change(_job(row) if row else None)
                if job is not None:
                    if job.state not in JOB_STATES:
                        raise ValueError(f"a job's state must be one of {JOB_STATES}, not {job.state!r}")
                    job.updated_at = time.time() if now is None else float(now)
                    db.execute(f"INSERT OR REPLACE INTO capture_jobs ({JOB_COLUMNS}) VALUES "
                               "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", _job_values(job))
            except BaseException:
                _rollback(db)
                raise
            db.execute("COMMIT")
        return job

    def due_jobs(self, now: float, limit: Optional[int] = None) -> List[Job]:
        """Open jobs (todo, partial) due at ``now``, the earliest due first."""
        sql = (f"SELECT {JOB_COLUMNS} FROM capture_jobs WHERE state IN ('todo', 'partial') AND due_at <= ? "
               "ORDER BY due_at, start, game_key")
        args: List[Any] = [float(now)]
        if limit is not None:
            sql += " LIMIT ?"
            args.append(int(limit))
        with closing(self._connect()) as db:
            return [_job(row) for row in db.execute(sql, args)]

    def next_due(self) -> Optional[float]:
        """The earliest due time of an open job (todo, partial), or None when there is none."""
        with closing(self._connect()) as db:
            row = db.execute("SELECT MIN(due_at) FROM capture_jobs WHERE state IN ('todo', 'partial')").fetchone()
        return None if not row or row[0] is None else float(row[0])

    def jobs(self, *, league: Optional[str] = None, states: Optional[Sequence[str]] = None,
             since: Optional[float] = None, until: Optional[float] = None,
             prefix: Optional[str] = None) -> List[Job]:
        """Jobs by league, state, start window [since, until) and game-key prefix, in start order."""
        sql, args = f"SELECT {JOB_COLUMNS} FROM capture_jobs WHERE 1 = 1", []  # type: ignore[var-annotated]
        if league is not None:
            sql += " AND league = ?"
            args.append(league)
        if states:
            sql += f" AND state IN ({', '.join('?' for _ in states)})"
            args.extend(states)
        if since is not None:
            sql += " AND start >= ?"
            args.append(float(since))
        if until is not None:
            sql += " AND start < ?"
            args.append(float(until))
        if prefix is not None:
            sql += " AND substr(game_key, 1, ?) = ?"
            args.extend([len(prefix), prefix])
        with closing(self._connect()) as db:
            return [_job(row) for row in db.execute(sql + " ORDER BY start, game_key", args)]

    def job_counts(self, league: Optional[str] = None) -> Dict[str, int]:
        """{state: jobs} (every state, 0 when none)."""
        sql, args = "SELECT state, COUNT(*) FROM capture_jobs", []  # type: ignore[var-annotated]
        if league is not None:
            sql += " WHERE league = ?"
            args.append(league)
        counts = {state: 0 for state in JOB_STATES}
        with closing(self._connect()) as db:
            for state, count in db.execute(sql + " GROUP BY state", args):
                counts[str(state)] = int(count)
        return counts

    # -- budgets and counters --

    def spend(self, kind: str, day: Union[str, datetime.date], n: int = 1, limit: Optional[int] = None) -> bool:
        """Count ``n`` more against (kind, day) unless that would pass ``limit`` (None: no limit); one transaction, so
        two callers never both take the last one. Returns whether it was counted."""
        if kind not in BUDGET_KINDS:
            raise ValueError(f"a budget kind must be one of {BUDGET_KINDS}, not {kind!r}")
        n = int(n)
        if n < 0:
            raise ValueError("n must not be negative")
        text, now = day_text(day), time.time()
        with self._writing() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("SELECT used FROM capture_budget WHERE kind = ? AND day = ?", (kind, text)).fetchone()
                used = int(row[0]) if row else 0
                if limit is not None and used + n > int(limit):
                    _rollback(db)
                    return False
                db.execute("INSERT INTO capture_budget (kind, day, used, updated_at) VALUES (?, ?, ?, ?) "
                           "ON CONFLICT(kind, day) DO UPDATE SET used = used + excluded.used, "
                           "updated_at = excluded.updated_at", (kind, text, n, now))
                if row is None:  # a new day: drop counters past their keep
                    old = (datetime.date.fromisoformat(text) - datetime.timedelta(days=BUDGET_KEPT_DAYS)).isoformat()
                    db.execute("DELETE FROM capture_budget WHERE day < ?", (old,))
            except BaseException:
                _rollback(db)
                raise
            db.execute("COMMIT")
        return True

    def spent(self, kind: str, day: Union[str, datetime.date]) -> int:
        """What (kind, day) has counted so far."""
        with closing(self._connect()) as db:
            row = db.execute("SELECT used FROM capture_budget WHERE kind = ? AND day = ?",
                             (kind, day_text(day))).fetchone()
        return int(row[0]) if row else 0

    def counts(self, kinds: Optional[Sequence[str]] = None, since_day: Optional[Union[str, datetime.date]] = None
               ) -> List[Dict[str, Any]]:
        """Counter rows {"kind", "day", "used"}, oldest day first."""
        sql, args = "SELECT kind, day, used FROM capture_budget WHERE 1 = 1", []  # type: ignore[var-annotated]
        if kinds:
            sql += f" AND kind IN ({', '.join('?' for _ in kinds)})"
            args.extend(kinds)
        if since_day is not None:
            sql += " AND day >= ?"
            args.append(day_text(since_day))
        with closing(self._connect()) as db:
            return [{"kind": kind, "day": day, "used": int(used)}
                    for kind, day, used in db.execute(sql + " ORDER BY day, kind", args)]

    # -- the backfill's progress --

    def get_cursor(self, league: str, season: int) -> Optional[Any]:
        """The saved cursor of (league, season), or None when its walk hasn't begun."""
        with closing(self._connect()) as db:
            row = db.execute("SELECT cursor FROM backfill_progress WHERE league = ? AND season = ?",
                             (league, int(season))).fetchone()
        if not row or row[0] is None:
            return None
        return json.loads(row[0])

    def set_cursor(self, league: str, season: int, cursor: Any, *, done: Optional[bool] = None,
                   games_seen: Optional[int] = None, games_stored: Optional[int] = None, add_seen: int = 0,
                   add_stored: int = 0, now: Optional[float] = None) -> None:
        """Save (league, season)'s cursor (any JSON value). ``done`` marks the season finished (None: as it was);
        ``games_seen``/``games_stored`` set the counts, ``add_seen``/``add_stored`` add to them."""
        when = time.time() if now is None else float(now)
        text = json.dumps(cursor, sort_keys=True, separators=(",", ":"))
        with self._writing() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("SELECT done, games_seen, games_stored FROM backfill_progress WHERE league = ? AND "
                                 "season = ?", (league, int(season))).fetchone()
                was_done, seen, stored = (bool(row[0]), int(row[1]), int(row[2])) if row else (False, 0, 0)
                seen = (seen if games_seen is None else int(games_seen)) + int(add_seen)
                stored = (stored if games_stored is None else int(games_stored)) + int(add_stored)
                flag = was_done if done is None else bool(done)
                db.execute("INSERT OR REPLACE INTO backfill_progress (league, season, cursor, done, games_seen, "
                           "games_stored, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (league, int(season), text, int(flag), seen, stored, when))
            except BaseException:
                _rollback(db)
                raise
            db.execute("COMMIT")

    def progress(self, league: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every (league, season) the backfill has begun: {"league", "season", "cursor", "done", "games_seen",
        "games_stored", "updated_at"}, by league then newest season first."""
        sql, args = ("SELECT league, season, cursor, done, games_seen, games_stored, updated_at FROM backfill_progress",
                     [])  # type: ignore[var-annotated]
        if league is not None:
            sql += " WHERE league = ?"
            args.append(league)
        with closing(self._connect()) as db:
            rows = db.execute(sql + " ORDER BY league, season DESC", args).fetchall()
        return [{"league": lg, "season": int(season), "cursor": json.loads(cursor) if cursor is not None else None,
                 "done": bool(done), "games_seen": int(seen), "games_stored": int(stored),
                 "updated_at": None if updated is None else float(updated)}
                for lg, season, cursor, done, seen, stored, updated in rows]

    # -- the worker's small state --

    def get_state(self, key: str, default: Any = None) -> Any:
        with closing(self._connect()) as db:
            row = db.execute("SELECT value FROM capture_state WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_state(self, key: str, value: Any, now: Optional[float] = None) -> None:
        with self._writing() as db:
            db.execute("INSERT INTO capture_state (key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE "
                       "SET value = excluded.value, updated_at = excluded.updated_at",
                       (key, json.dumps(value, sort_keys=True), time.time() if now is None else float(now)))

    # -- health --

    def size_bytes(self) -> int:
        """raw.db's size on disk, its write-ahead log included."""
        total = 0
        for suffix in ("", "-wal", "-shm"):
            try:
                total += os.path.getsize(self.path + suffix)
            except OSError:
                pass
        return total

    def _since(self, since: Optional[float]) -> float:
        return float(since) if since is not None else self.clock() - HEALTH_DAYS * 86400.0

    def http_counts(self, since: Optional[float] = None) -> Dict[str, int]:
        """Stored payloads by status class fetched since ``since`` (default: the last HEALTH_DAYS days): {"2xx",
        "3xx", "4xx", "5xx"}. An identical answer is one row; the request counts are the ``http_ok``/``http_error``
        counters. Read from the fetch-time index alone, never a scan of the table."""
        out = {"2xx": 0, "3xx": 0, "4xx": 0, "5xx": 0}
        with closing(self._connect()) as db:
            for klass, count in db.execute("SELECT http_status / 100, COUNT(*) FROM raw_payloads WHERE fetched_at >= ? "
                                           "GROUP BY http_status / 100", (self._since(since),)):
                name = f"{int(klass)}xx"
                out[name] = out.get(name, 0) + int(count)
        return out

    def recent_errors(self, limit: int = 10, since: Optional[float] = None) -> List[Dict[str, Any]]:
        """The newest problems since ``since`` (default: the last HEALTH_DAYS days), newest first: jobs with a last
        error ({"at", "game_key", "league", "state", "error"}) and non-2xx payloads ({"at", "game_key", "league",
        "source", "endpoint", "status"}). The payloads come by the fetch-time index, never a scan of the table."""
        start = self._since(since)
        with closing(self._connect()) as db:
            jobs = db.execute("SELECT updated_at, game_key, league, state, last_error FROM capture_jobs WHERE "
                              "last_error IS NOT NULL AND updated_at >= ? ORDER BY updated_at DESC LIMIT ?",
                              (start, int(limit))).fetchall()
            bad = db.execute("SELECT fetched_at, game_key, league, source, endpoint, http_status FROM raw_payloads "
                             "WHERE fetched_at >= ? AND http_status NOT BETWEEN 200 AND 299 "
                             "ORDER BY fetched_at DESC LIMIT ?", (start, int(limit))).fetchall()
        found = [{"at": float(at or 0), "game_key": key, "league": league, "state": state, "error": error}
                 for at, key, league, state, error in jobs]
        found += [{"at": float(at), "game_key": key, "league": league, "source": source, "endpoint": endpoint,
                   "status": int(status)} for at, key, league, source, endpoint, status in bad]
        return sorted(found, key=lambda item: -item["at"])[:int(limit)]


def _job_values(job: Job) -> Tuple[Any, ...]:
    error = job.last_error[:ERROR_TEXT] if isinstance(job.last_error, str) else job.last_error
    return (job.game_key, job.league, None if job.espn_id is None else str(job.espn_id), float(job.start),
            float(job.due_at), job.state, int(job.attempts), json.dumps(sorted(set(job.kinds_done))), error,
            job.updated_at)


__all__ = ["RawStore", "RawPayload", "RawBundle", "Job", "SCHEMA", "STATES", "HORIZONS", "JOB_STATES",
           "BUDGET_KINDS", "canonical_params", "codec_for", "encode", "decode", "day_text", "utc_day",
           "resolve_raw_db_path", "not_a_raw_store", "disk_room", "disk_is_full", "full_pause", "min_free_bytes",
           "DISK_FLOOR_SHARE", "MIN_FREE_GB", "FULL_PAUSE"]
