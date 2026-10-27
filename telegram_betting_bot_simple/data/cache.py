"""Short-term memory for everything the app fetches or computes, shared by every person in a process and, through
the store's SQLite file, by the bot and the dashboard.

A read, ``TTLCache.get(key, ttl, loader, stale=0, error_ttl=30, persist=None, kind=None)``:

- Fresh (younger than ``ttl``): the copy in memory.
- Within ``ttl + stale``: the copy in memory at once, and at most one background refresh for the key, on a small
  bounded pool of threads.
- Older, or missing: loaded under the key's lock, so any number of people asking at once cause one load. With a
  store, its ``shared_cache`` table is read first: a fresh copy saved by the other process is used as it is.
  Otherwise a short lease (``cache_leases``) makes one process the loader. A process that finds the lease taken
  polls the table for the holder's value (up to ``lease_wait`` seconds, every ``lease_poll``); if none comes it
  serves its last good copy, or loads itself when it has none.
- A loaded value is kept in memory and, with a store, saved to the table as gzip-compressed JSON with its
  ``fetched_at``, ``expires_at`` and ``stale_until``. A copy keeps the TTL it was fetched with (``life``, by default
  the read's ``ttl`` and ``stale``): a later reader's own TTL can shorten it but never lengthen it, so a copy
  fetched under a short policy (a pre-game summary, a game-day game log) is not served as fresh under a longer one
  once its data's state has moved on (the reader's stale window then runs from the copy's TTL). A computed result
  (``get_computed``) is fresh no longer than the inputs it was built from had left: one built from an input served
  past that input's TTL is fresh only briefly (PARTIAL_KEEP). ``persist=None`` saves values that come back from
  JSON unchanged (dicts with string keys, lists, strings, numbers, booleans, None); anything else stays in memory.
  ``persist=True`` saves any JSON-serialisable value (tuples come back as lists); ``persist=False`` keeps the
  value in this process only (no table, no lease).
- A failed load serves the last good copy, and until the failure's wait is over every reader gets that copy at
  once without trying again (a down source is not hammered, and nobody queues behind a failing fetch).
  With no copy, the error is remembered and re-raised instead of retried (negative caching). The wait is
  ``error_ttl`` seconds; with ``backoff_cap`` it doubles with each failure in a row up to that cap (the markets
  Keeper's rule), and background refreshes always keep to it. ``error_ttl=0`` remembers nothing (the caller paces
  its own retries); ``backoff=False`` makes the next reader past the TTL try again at once while a last good copy
  is held.
- A process holding a copy that waited out the lease once (the other process's load is slow, or it died holding
  the lease) serves that copy at once to the readers after it until the lease could have lapsed, instead of making
  each of them wait in turn; the first reader to find the lease free loads.
- Memory keeps at most ``max_entries`` copies, least recently used out first, and a key's lock goes with its
  copy. Memory and the table are pruned of copies a day past the end of their stale window, hourly or every
  PRUNE_WRITES writes.
- ``stats()`` counts every read per ``kind`` (see STAT_FIELDS); ``all_stats()`` adds up every cache in the process.

Keys are a value or a tuple of str, int, float, bool, None, dates and tuples. ``key_text`` writes them as stable
JSON (the table's key). They must never carry a secret: build them from what is asked for (league, event id,
player id), never from a URL, its query string or headers. Secrets registered with ``forbid_secrets`` are refused
with SecretInKey. A key with any other part (an object, a list) works in memory only.

How long each kind of data stays fresh is one table, ``ttl_for(kind, state)`` (POLICY), which callers use instead
of literals. AI chat answers are never cached here: chats are personal.
"""

from __future__ import annotations

import datetime
import gzip
import itertools
import json
import logging
import math
import os
import queue
import re
import threading
import time
import weakref
from collections import OrderedDict
from time import monotonic as _real_monotonic  # waits that must pass even under a test's frozen clock
from typing import Any, Callable, Dict, Hashable, Iterable, List, NamedTuple, Optional, Tuple, Union

log = logging.getLogger(__name__)

MINUTE = 60.0
HOUR = 3600.0
DAY = 86400.0
PARTIAL_KEEP = 10.0  # a computed result built while one of its sources failed lives this long (get_computed)


# --- the TTL policy ---------------------------------------------------------------------------------------

class Policy(NamedTuple):
    """How long a kind of data is fresh (``ttl``) and how long past that it is still served at once while one
    background refresh runs (``stale``), in seconds."""

    ttl: float
    stale: float = 0.0


_SUMMARY = {"live": Policy(20, 20), "pre": Policy(5 * MINUTE, 10 * MINUTE), "final": Policy(DAY, 0),
            None: Policy(5 * MINUTE, 10 * MINUTE)}
_PLAYER = {"gameday": Policy(HOUR, HOUR), "past": Policy(DAY, HOUR), None: Policy(6 * HOUR, HOUR)}
_INJURIES = {None: Policy(10 * MINUTE, 10 * MINUTE)}
_GAME_RESULT = {"live": Policy(20, 20), "pre": Policy(MINUTE, MINUTE), "final": Policy(HOUR, 0),
                None: Policy(MINUTE, MINUTE)}

# kind -> {state: Policy}; the None state is the kind's default. Starting values from the stage spec (v3 2b §2),
# with the research notes' update intervals (research_notes/Sports data sources and analysis UX/) behind them.
POLICY: Dict[str, Dict[Optional[str], Policy]] = {
    # ESPN game summary and box score, by the game's state: a final one doesn't change again
    "summary": _SUMMARY,
    "boxscore": _SUMMARY,
    # Scoreboards keep the slate's own intervals (config SLATE_LIVE/TODAY/IDLE/FUTURE_SECONDS defaults); a request
    # past that is served the board at once while one background refresh runs (the slate's own fetch never is)
    "scoreboard": {"live": Policy(20, 20), "today": Policy(MINUTE, MINUTE),
                   "idle": Policy(10 * MINUTE, 10 * MINUTE), "future": Policy(HOUR, HOUR),
                   "past": Policy(6 * HOUR, 6 * HOUR), None: Policy(5 * MINUTE, 5 * MINUTE)},
    # A player's overview and game log: 1 hour on his game days, 6 hours otherwise; a finished season's for a day
    "player": _PLAYER,
    "gamelog": _PLAYER,
    "splits": {None: Policy(6 * HOUR, 6 * HOUR)},
    "injuries": _INJURIES,
    "news": {None: Policy(10 * MINUTE, 10 * MINUTE)},
    "standings": {None: Policy(HOUR, 0), "past": Policy(DAY, 0)},
    "roster": {None: Policy(3 * HOUR, 0)},
    # Prediction markets (Polymarket, Kalshi): 5 minutes, 1 while a game of the league is on; their spreads and
    # totals every 15 minutes
    "markets": {None: Policy(5 * MINUTE, 5 * MINUTE), "live": Policy(MINUTE, 5 * MINUTE)},
    "market_lines": {None: Policy(15 * MINUTE, 15 * MINUTE)},
    "ratings": {None: Policy(12 * HOUR, 12 * HOUR)},
    "predictor": {None: Policy(HOUR, 0)},
    # Kinds found in the code beyond the spec's table, at the TTLs they used before
    "teams": {None: Policy(7 * DAY, DAY)},              # a league's team list (ratings' name matching)
    "search": {None: Policy(HOUR, HOUR)},               # player search
    "bio": _INJURIES,                                   # a player's bio: it carries his injury status
    "schedule": {None: Policy(HOUR, HOUR)},             # a team's schedule
    "career": {None: Policy(12 * HOUR, 12 * HOUR)},     # a player's per-season averages
    "props": {None: Policy(10 * MINUTE, 0)},            # DraftKings player props through ESPN
    "odds_sports": {None: Policy(6 * HOUR, HOUR)},      # The Odds API's sports listing (free: no credits)
    "odds_api": {None: Policy(2 * HOUR, 0)},            # The Odds API's boards via the legacy clients (credits)
    "http": {None: Policy(5 * MINUTE, 0)},              # any other GET (data/utils.py's fetch_json)
    "api_sports": {None: Policy(15 * MINUTE, 15 * MINUTE), "live": Policy(MINUTE, MINUTE)},
    # Computed results: the TTL of their freshest input (freshest() combines policies)
    "games": {None: Policy(MINUTE, 0)},                 # a league's game cards
    "game_card": _GAME_RESULT,                          # one game's detail (/api/game, inspector, bot card)
    "consensus": _GAME_RESULT,                          # the markets brain's block for one game
    "player_card": _PLAYER,
    "team_context": {None: Policy(30 * MINUTE, 30 * MINUTE)},
}

# Other spellings of the states (ESPN's status.type.state is "pre", "in" or "post")
STATE_NAMES = {"in": "live", "in_progress": "live", "inprogress": "live", "post": "final", "finished": "final",
               "done": "final", "scheduled": "pre", "upcoming": "pre", "game_day": "gameday", "today_game": "gameday"}


def ttl_for(kind: str, state: Optional[str] = None) -> Policy:
    """The Policy (ttl, stale) for ``kind`` in ``state`` (the kind's default when the state has no row of its
    own). An unknown kind is a KeyError: a typo must fail in tests, not cache for a guessed time."""
    table = POLICY[kind]
    if state is not None:
        name = str(state).strip().lower()
        policy = table.get(STATE_NAMES.get(name, name))
        if policy is not None:
            return policy
    return table[None]


def freshest(*policies: Policy) -> Policy:
    """The policy of a value computed from several inputs: the shortest TTL and the shortest stale window."""
    if not policies:
        raise ValueError("freshest() needs at least one policy")
    return Policy(min(p.ttl for p in policies), min(p.stale for p in policies))


# --- keys ---------------------------------------------------------------------------------------------------

class SecretInKey(ValueError):
    """A cache key carried a registered secret (an API key or token). The message never repeats it."""


class _Unshareable(TypeError):
    pass


_SECRETS: frozenset = frozenset()
_SECRETS_LOCK = threading.Lock()
MIN_SECRET = 8  # shorter strings would match ordinary key parts


def forbid_secrets(*values: Any) -> None:
    """Register secrets (API keys, tokens) no cache key may contain; they are also blanked from logged errors.
    Values shorter than 8 characters, and empty ones, are ignored."""
    global _SECRETS
    found = {value.strip() for value in values if isinstance(value, str) and len(value.strip()) >= MIN_SECRET}
    if found:
        with _SECRETS_LOCK:
            _SECRETS = _SECRETS | found


def _encode(part: Any) -> Any:
    if part is None or isinstance(part, (bool, str)):
        return part
    if isinstance(part, int):
        return int(part)
    if isinstance(part, float):
        if not math.isfinite(part):
            raise _Unshareable("float")
        return int(part) if part.is_integer() else part  # 1.0 == 1 in memory, so one row too
    if isinstance(part, datetime.datetime):
        return {"datetime": part.isoformat()}
    if isinstance(part, datetime.date):
        return {"date": part.isoformat()}
    if isinstance(part, tuple):
        return [_encode(item) for item in part]
    if isinstance(part, frozenset):
        return {"set": sorted(json.dumps(_encode(item), sort_keys=True) for item in part)}
    raise _Unshareable(type(part).__name__)


def _check_secrets(text: str) -> None:
    for secret in _SECRETS:
        if secret in text:
            raise SecretInKey("A cache key must not carry a secret (an API key or token): build it without one")


def key_text(key: Hashable) -> str:
    """The key as stable JSON: a tuple is an array (``["espn","summary","basketball_nba","401"]``), a date is
    ``{"date":"2026-10-27"}``; the same key gives the same text in every process. Raises SecretInKey for a key
    holding a registered secret, and TypeError for a part with no stable text."""
    text = json.dumps(_encode(key), separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    _check_secrets(text)
    return text


def _prefix_text(prefix: Tuple[Any, ...]) -> str:
    """The text every key starting with ``prefix`` (a tuple) starts with: its array without the closing bracket."""
    return key_text(tuple(prefix))[:-1]


def _default_kind(key: Hashable) -> str:
    if isinstance(key, tuple) and key and isinstance(key[0], str):
        return key[0]
    return "other"


# --- values -------------------------------------------------------------------------------------------------

def _plain(value: Any) -> bool:
    """True when ``value`` comes back from JSON unchanged: dicts with str keys, lists, str, int, float, bool,
    None (a tuple would come back as a list, an int key as a string, so those stay in memory)."""
    stack = [value]
    while stack:
        item = stack.pop()
        if item is None or isinstance(item, (str, int, float)):  # bool is an int
            continue
        if isinstance(item, dict):
            if not all(isinstance(name, str) for name in item):
                return False
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        else:
            return False
    return True


def _pack(value: Any, strict: bool) -> Optional[bytes]:
    """The value as gzip-compressed JSON, or None when it can't be (or, ``strict``, wouldn't come back the same)."""
    if strict and not _plain(value):
        return None
    try:
        text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError, RecursionError):
        return None
    return gzip.compress(text.encode("utf-8"), compresslevel=6, mtime=0)


def _unpack(blob: bytes) -> Any:
    return json.loads(gzip.decompress(blob).decode("utf-8"))


_QUERY = re.compile(r"\?\S*")


def _scrub(text: str) -> str:
    """An error's text fit for a log line: query strings (which can carry keys) and registered secrets blanked."""
    text = _QUERY.sub("?…", text)
    for secret in _SECRETS:
        text = text.replace(secret, "***")
    return text[:200]


# --- the cache ----------------------------------------------------------------------------------------------

STAT_FIELDS = (
    "hits",           # fresh from memory
    "stale_hits",     # served past its TTL: inside the stale window, while backing off a failure, or while the
                      # other process holds the lease
    "shared_hits",    # from the SQLite table (the other process's copy)
    "misses",         # loaded by the caller
    "negative_hits",  # a remembered failure re-raised without trying
    "loads",          # loader calls, foreground and background
    "load_errors",    # loader calls that raised
    "refreshes",      # background refreshes started
    "lease_waits",    # times this process waited for the other's load
    "evictions",      # copies dropped from memory to stay under max_entries
    "expired",        # copies dropped from memory a day past their stale window (the prune)
)
READ_FIELDS = ("hits", "stale_hits", "shared_hits", "misses", "negative_hits")  # each get() counts one of these

_INSTANCES: "weakref.WeakSet[TTLCache]" = weakref.WeakSet()
_SERIAL = itertools.count(1)

# (monotonic fetch time, wall-clock fetch time, value, kind, fresh for, usable for): the last two are the copy's own
# life in seconds after its fetch (its TTL, and its TTL plus stale window), which no reader can lengthen
Entry = Tuple[float, float, Any, str, float, float]
Life = Union[Tuple[float, float], Callable[[Any], Optional[Tuple[float, float]]], None]


class _Job(NamedTuple):
    """One read's key and rules (a background refresh carries them to its thread)."""

    key: Hashable
    ttl: float
    loader: Callable[[], Any]
    stale: float
    error_ttl: float
    persist: Optional[bool]
    kind: str
    backoff: bool
    backoff_cap: Optional[float] = None
    life: Life = None          # the loaded copy's life: (ttl, stale), a function of the value, or None (ttl, stale)
    computed: bool = False     # a computed result: its life is capped by the inputs read while building it


def _own_life(entry: Any) -> Tuple[float, float]:
    """An entry's own (fresh for, usable for); none (any reader's TTL) for one without them."""
    return (entry[4], entry[5]) if len(entry) > 5 else (math.inf, math.inf)


class _Row(NamedTuple):
    """The shared table's copy of a key."""

    fetched_at: float
    value: Any
    fresh_for: float   # expires_at - fetched_at
    usable_for: float  # stale_until - fetched_at


class TTLCache:
    """See the module's docstring. ``store`` is a data.store.Store (or anything with its ``cache_*`` methods);
    without one the cache is this process's memory only. ``clock`` (epoch seconds) replaces both the monotonic
    and the wall clock, and ``sleep`` the lease wait's sleep, for tests."""

    LEASE_SECONDS = 30.0      # a loader's claim on a key, in case its process dies mid-load
    LEASE_WAIT = 10.0         # how long a process waits for the other's load before serving or loading itself
    LEASE_POLL = 0.25         # how often it looks at the table meanwhile
    ERROR_CAP = 15 * MINUTE   # the longest a failing key waits between tries
    REFRESH_WORKERS = 2       # background refresh threads (started when first needed, gone when idle)
    REFRESH_QUEUE = 64        # refreshes waiting at most; past that a stale copy just waits for the next read
    WORKER_IDLE = 60.0        # an idle refresh thread ends after this
    PRUNE_EVERY = HOUR        # the table is pruned this often ...
    PRUNE_WRITES = 1000       # ... or after this many writes
    KEEP_AFTER_STALE = DAY    # rows are pruned this long after their stale_until
    MAX_SHARED_BYTES = 4 * 1024 * 1024  # a bigger compressed value stays in memory only
    MAX_TTL = 365 * DAY

    def __init__(self, store: Any = None, max_entries: int = 4000, name: str = "", *,
                 clock: Optional[Callable[[], float]] = None, sleep: Optional[Callable[[float], None]] = None,
                 lease_seconds: Optional[float] = None, lease_wait: Optional[float] = None,
                 lease_poll: Optional[float] = None, refresh_workers: Optional[int] = None,
                 error_detail: bool = True) -> None:
        self.store = store
        methods = ("cache_get", "cache_put", "cache_delete", "cache_delete_prefix", "cache_lease", "cache_release",
                   "cache_prune")
        self._shared = store if store is not None and all(hasattr(store, m) for m in methods) else None
        self.max_entries = max(1, int(max_entries))
        self.name = name or ""
        # Looked up on each call, so a test that swaps this module's ``time`` moves every cache's clock
        self._wall: Callable[[], float] = clock or (lambda: time.time())
        self._mono: Callable[[], float] = clock or (lambda: time.monotonic())
        self._sleep: Callable[[float], None] = sleep or (lambda seconds: time.sleep(seconds))
        self.lease_seconds = float(self.LEASE_SECONDS if lease_seconds is None else lease_seconds)
        self.lease_wait = max(0.0, float(self.LEASE_WAIT if lease_wait is None else lease_wait))
        self.lease_poll = max(0.001, float(self.LEASE_POLL if lease_poll is None else lease_poll))
        self.refresh_workers = max(1, int(self.REFRESH_WORKERS if refresh_workers is None else refresh_workers))
        self.error_detail = error_detail  # False: failures are logged by their type only
        # Who holds a lease: unique per cache, so two caches on one file (the bot and the dashboard) are two holders
        self.holder = f"{self.name or 'cache'}:{os.getpid()}:{next(_SERIAL)}"
        self._entries: "OrderedDict[Hashable, Any]" = OrderedDict()  # key -> Entry, least recently used first
        self._key_locks: Dict[Hashable, List[Any]] = {}  # key -> [Lock, threads using it]
        self._errors: "OrderedDict[Hashable, Tuple[int, float, BaseException, float]]" = OrderedDict()  # key ->
        # (failures in a row, retry at (monotonic), the last error, when it failed (wall clock))
        self._refreshing: set = set()
        self._busy: Dict[Hashable, float] = {}  # key -> until (monotonic): a lease this process waited out
        self._local = threading.local()  # per thread: the computed results being built (their inputs' life left)
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._queue: "queue.Queue[_Job]" = queue.Queue(maxsize=self.REFRESH_QUEUE)
        self._workers = 0
        self._stats: Dict[str, Dict[str, int]] = {}
        self._writes = 0
        self._pruned_at = -math.inf  # the first write prunes what an earlier run left
        self._store_warned = -math.inf
        _INSTANCES.add(self)

    # --- reading ------------------------------------------------------------------------------------------

    def get(self, key: Hashable, ttl_seconds: float, loader: Callable[[], Any], *, stale: float = 0.0,
            error_ttl: float = 30.0, persist: Optional[bool] = None, kind: Optional[str] = None,
            backoff: bool = True, backoff_cap: Optional[float] = None, life: Life = None) -> Any:
        """The value of ``key``: cached while fresh, else loaded by ``loader`` (see the module's docstring).
        ``backoff`` (default): after a failed refresh, serve the last good copy without retrying until the failure's
        wait is over; False: the next reader past the TTL tries again (one at a time, each waiting for the fetch).
        ``backoff_cap``: failures in a row double the wait up to this many seconds (default: every wait is
        ``error_ttl``). ``life``: how long a copy loaded now is good for, for every reader: (ttl, stale), or a
        function of the loaded value giving that (default: this read's ``ttl`` and ``stale``)."""
        return self._get(self._job(key, ttl_seconds, loader, stale, error_ttl, persist, kind, backoff, backoff_cap,
                                   life))[0]

    def get_with_time(self, key: Hashable, ttl_seconds: float, loader: Callable[[], Any], *, stale: float = 0.0,
                      error_ttl: float = 30.0, persist: Optional[bool] = None, kind: Optional[str] = None,
                      backoff: bool = True, backoff_cap: Optional[float] = None,
                      life: Life = None) -> Tuple[Any, float]:
        """As get, with when the value was fetched (epoch seconds; the fetching process's time for a shared copy)."""
        return self._get(self._job(key, ttl_seconds, loader, stale, error_ttl, persist, kind, backoff, backoff_cap,
                                   life))

    def get_for(self, kind: str, key: Hashable, loader: Callable[[], Any], state: Optional[str] = None,
                **options: Any) -> Any:
        """get with the TTL and stale window of ``ttl_for(kind, state)``, counted under ``kind``."""
        policy = ttl_for(kind, state)
        options.setdefault("stale", policy.stale)
        return self.get(key, policy.ttl, loader, kind=kind, **options)

    def get_computed(self, key: Hashable, ttl_seconds: float, loader: Callable[[], Any],
                     partial: Callable[[Any], bool], *, keep: float = PARTIAL_KEEP, **options: Any) -> Any:
        """get for a computed result (a game's card, a player's card). It lives no longer than the inputs read
        through this cache while building it (one built from an input served past that input's TTL is fresh for
        ``PARTIAL_KEEP`` seconds at most, so the next reader rebuilds it from the refreshed input). One built while
        a source it reads failed (``partial(value)`` is true) is kept only ``keep`` seconds: the people asking
        meanwhile, here and in the other process, share it instead of each rebuilding it, and the first reader after
        that rebuilds it, whole again once the source is back (its own back-off over). ``options`` as get's."""
        value = self._get(self._job(key, ttl_seconds, loader, computed=True, **options))[0]
        if partial(value):
            with self._lock:
                entry = self._entries.get(key)
            if entry is not None and self._mono() - entry[0] >= keep:
                # Only this copy goes: a newer one in the table (the other process's rebuild) is used, not deleted
                self._drop(key, upto=entry[1])
                value = self._get(self._job(key, ttl_seconds, loader, computed=True, **options))[0]
        return value

    @staticmethod
    def _job(key: Hashable, ttl: float, loader: Callable[[], Any], stale: float = 0.0, error_ttl: float = 30.0,
             persist: Optional[bool] = None, kind: Optional[str] = None, backoff: bool = True,
             backoff_cap: Optional[float] = None, life: Life = None, computed: bool = False) -> _Job:
        return _Job(key, float(ttl), loader, max(0.0, float(stale or 0.0)), max(0.0, float(error_ttl or 0.0)),
                    persist, kind or _default_kind(key), bool(backoff),
                    None if backoff_cap is None else max(0.0, float(backoff_cap)), life, bool(computed))

    def _get(self, job: _Job) -> Tuple[Any, float]:
        key = job.key
        with self._lock:
            served = self._serve_locked(job)
            record = self._hold_locked(key) if served is None else None
        if record is not None:
            try:
                with record[0]:
                    served = self._load_locked(job)
            finally:
                self._unhold(key, record)
        self._note_input(job)
        return served

    @staticmethod
    def _limits(job: _Job, entry: Any) -> Tuple[float, float]:
        """(fresh for, usable for) of a copy for this read, in seconds after its fetch: the read's TTL cut to the
        copy's own (a reader can shorten a copy's life, never lengthen it), then the read's stale window."""
        ttl = min(job.ttl, _own_life(entry)[0])
        return ttl, ttl + job.stale

    def _serve_locked(self, job: _Job) -> Optional[Tuple[Any, float]]:
        """The memory copy when it is fresh, or inside the stale window (starting a refresh); else None."""
        key, kind = job.key, job.kind
        entry = self._entries.get(key)
        if entry is None:
            return None
        age = self._mono() - entry[0]
        ttl, usable = self._limits(job, entry)
        if age < ttl:
            self._entries.move_to_end(key)
            self._count(kind, "hits")
            return entry[2], entry[1]
        if age < usable:
            self._entries.move_to_end(key)
            self._count(kind, "stale_hits")
            self._schedule_locked(job)
            return entry[2], entry[1]
        return None

    def _load_locked(self, job: _Job) -> Tuple[Any, float]:
        """The slow path, holding the key's lock: memory again, the shared table, a remembered failure, the lease,
        then the loader."""
        key, persist, kind = job.key, job.persist, job.kind
        with self._lock:
            served = self._serve_locked(job)  # another thread may have loaded it meanwhile
            if served is not None:
                return served
            entry = self._entries.get(key)
        text = self._text(key)
        shared = self._shared if text is not None and persist is not False else None
        seen = entry[1] if entry is not None else None
        if shared is not None:
            row = self._read_row(text, seen)  # only a copy newer than this process's is read and unpacked
            if row is not None:
                entry = self._adopt(key, row, kind)
                seen = entry[1]
                age = self._mono() - entry[0]
                ttl, usable = self._limits(job, entry)
                if age < usable:
                    with self._lock:
                        self._count(kind, "shared_hits")
                        if age >= ttl:
                            self._schedule_locked(job)
                    return entry[2], entry[1]
        error: Optional[BaseException] = None
        with self._lock:
            failure = self._backing_off_locked(key, entry) if job.error_ttl > 0 else None
            if failure is not None and entry is not None and job.backoff:
                self._count(kind, "stale_hits")
                return entry[2], entry[1]
            if failure is not None and entry is None:
                self._count(kind, "negative_hits")
                error = failure[2]
        if error is not None:
            raise error.with_traceback(None)
        leased = False
        if shared is not None:
            leased, row = self._claim(text, seen)
            if row is None and not leased:
                if entry is not None and self._busy_left(key) > 0:
                    # This process already waited out the holder's lease once: the last good copy at once
                    with self._lock:
                        self._count(kind, "stale_hits")
                    return entry[2], entry[1]
                with self._lock:
                    self._count(kind, "lease_waits")
                row, leased = self._wait_for_holder(text, seen)
                if row is None and not leased:
                    self._mark_busy(key)
            if row is not None:  # the other process's value, saved while this one looked or waited
                entry = self._adopt(key, row, kind)
                with self._lock:
                    self._count(kind, "shared_hits")
                return entry[2], entry[1]
            if not leased and entry is not None:
                with self._lock:
                    self._count(kind, "stale_hits")  # the holder is still on it: the last good copy meanwhile
                return entry[2], entry[1]
        with self._lock:
            self._count(kind, "misses")
        return self._call(job, text if shared is not None else None, entry, leased)

    def _call(self, job: _Job, text: Optional[str], entry: Optional[Any], leased: bool) -> Tuple[Any, float]:
        """Run the loader: keep and save its value, or remember its failure and serve ``entry`` (else raise)."""
        key, kind = job.key, job.kind
        with self._lock:
            self._count(kind, "loads")
        frames = self._frames() if job.computed else None
        inputs: Optional[List[float]] = None
        if frames is not None:
            frames.append([math.inf])  # the shortest fresh life its inputs had left, as the loader reads them
        try:
            try:
                value = job.loader()
            except Exception as exc:
                with self._lock:
                    self._count(kind, "load_errors")
                    if job.error_ttl > 0:
                        self._remember_locked(key, job.error_ttl, exc, job.backoff_cap)
                if entry is None:
                    raise
                log.warning("%s: refreshing %s failed (%s); serving the cached copy", self.name or "cache",
                            self._describe(key, text), self._error_text(exc))
                return entry[2], entry[1]
            finally:
                if frames is not None:
                    inputs = frames.pop()
            fresh_for, usable_for = self._life(job, value, inputs)
            fetched_at, now_m = self._wall(), self._mono()
            with self._lock:
                self._errors.pop(key, None)
                self._busy.pop(key, None)
                self._set_locked(key, (now_m, fetched_at, value, kind, fresh_for, usable_for))
            if text is not None:
                self._save(text, value, fetched_at, fresh_for, usable_for, job.persist, kind)
            self._wrote()
            return value, fetched_at
        finally:
            if leased and text is not None:
                self._store_call("cache_release", text, self.holder)

    @staticmethod
    def _life(job: _Job, value: Any, inputs: Optional[List[float]]) -> Tuple[float, float]:
        """(fresh for, usable for) of a copy just loaded: the job's ``life`` (or its ttl and stale); a computed
        result is fresh no longer than its inputs had left (but for PARTIAL_KEEP at least, so an input stuck past its
        TTL, its source failing, doesn't make every reader rebuild the result)."""
        life = job.life
        if callable(life):
            try:
                life = life(value)
            except Exception:  # a value the rule can't read: the read's own
                life = None
        fresh, stale = (job.ttl, job.stale) if life is None else (float(life[0]), max(0.0, float(life[1])))
        if inputs is not None:
            fresh = max(min(fresh, PARTIAL_KEEP), min(fresh, inputs[0]))
        fresh = max(0.0, fresh)
        return fresh, fresh + stale

    # --- computed results: what their inputs had left ----------------------------------------------------------

    def _frames(self) -> List[List[float]]:
        frames = getattr(self._local, "frames", None)
        if frames is None:
            frames = self._local.frames = []
        return frames

    def _note_input(self, job: _Job) -> None:
        """Inside a computed result's loader (this thread), note how long the copy just read has left."""
        frames = getattr(self._local, "frames", None)
        if not frames:
            return
        with self._lock:
            entry = self._entries.get(job.key)
        if entry is None:
            return
        frame = frames[-1]
        frame[0] = min(frame[0], self._limits(job, entry)[0] - (self._mono() - entry[0]))

    # --- a lease this process already waited out ----------------------------------------------------------------

    def _busy_left(self, key: Hashable) -> float:
        with self._lock:
            until = self._busy.get(key)
        return -1.0 if until is None else until - self._mono()

    def _mark_busy(self, key: Hashable) -> None:
        """The holder's lease outlived this process's wait: until it could have lapsed, readers with a copy don't
        wait again (they still try to take the lease, so a lapsed one is taken at once)."""
        now = self._mono()
        with self._lock:
            self._busy[key] = now + self.lease_seconds
            if len(self._busy) > self.max_entries:
                for old in [k for k, until in self._busy.items() if until <= now]:
                    del self._busy[old]
                while len(self._busy) > self.max_entries:
                    del self._busy[next(iter(self._busy))]

    # --- background refresh ---------------------------------------------------------------------------------

    def _schedule_locked(self, job: _Job) -> None:
        """Start one background refresh of the job's key, unless one is running, the key is backing off a failure
        or the queue is full."""
        key = job.key
        if key in self._refreshing or self._backing_off_locked(key, self._entries.get(key)) is not None:
            return
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            return
        self._refreshing.add(key)
        self._count(job.kind, "refreshes")
        if self._workers < self.refresh_workers:
            self._workers += 1
            threading.Thread(target=self._work, name=f"cache-refresh-{self.name or 'cache'}", daemon=True).start()

    def _work(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=self.WORKER_IDLE)
            except queue.Empty:
                with self._lock:
                    if self._queue.empty():
                        self._workers -= 1
                        return
                continue
            try:
                self._refresh(job)
            except Exception as exc:  # a refresh never takes its thread down
                log.debug("%s: background refresh of %s failed (%s)", self.name or "cache", job.key,
                          type(exc).__name__)
            finally:
                with self._lock:
                    self._refreshing.discard(job.key)
                    self._idle.notify_all()

    def _refresh(self, job: _Job) -> None:
        key, persist, kind = job.key, job.persist, job.kind
        with self._lock:
            record = self._hold_locked(key)
        try:
            # A reader that just scheduled this may still hold the key's lock for a moment; one that is loading the
            # key (a long fetch) makes the refresh pointless
            if not record[0].acquire(timeout=1.0):
                return
            try:
                with self._lock:
                    entry = self._entries.get(key)
                    if entry is not None and self._mono() - entry[0] < self._limits(job, entry)[0]:
                        return  # refreshed meanwhile
                    if self._backing_off_locked(key, entry) is not None:
                        return
                text = self._text(key)
                shared = self._shared if text is not None and persist is not False else None
                leased = False
                if shared is not None:
                    row = self._read_row(text, entry[1] if entry is not None else None)
                    if row is not None:
                        entry = self._adopt(key, row, kind)
                        if self._mono() - entry[0] < self._limits(job, entry)[0]:
                            return  # the other process refreshed it
                    leased, row = self._claim(text, entry[1] if entry is not None else None)
                    if row is not None:
                        self._adopt(key, row, kind)
                        return  # the other process refreshed it a moment ago
                    if not leased:
                        return  # the other process is refreshing it; its copy lands in the table
                try:
                    self._call(job, text if shared is not None else None, entry, leased)
                except Exception as exc:  # nothing to serve: remembered by _call, nobody waiting
                    log.info("%s: background load of %s failed (%s)", self.name or "cache",
                             self._describe(key, text), self._error_text(exc))
            finally:
                record[0].release()
        finally:
            self._unhold(key, record)

    def drain(self, timeout: Optional[float] = None) -> bool:
        """Wait until no background refresh is queued or running (tests, shutdown); False on timeout."""
        deadline = None if timeout is None else _real_monotonic() + timeout
        with self._lock:
            while self._refreshing:
                left = None if deadline is None else deadline - _real_monotonic()
                if left is not None and left <= 0:
                    return False
                self._idle.wait(left)
        return True

    # --- the shared table -----------------------------------------------------------------------------------

    def _text(self, key: Hashable) -> Optional[str]:
        """The key's table text, None when it has none (memory only). A key with a secret raises SecretInKey."""
        try:
            return key_text(key)
        except _Unshareable:
            return None

    def _read_row(self, text: str, newer_than: Optional[float] = None) -> Optional[_Row]:
        """The table's copy, or None; with ``newer_than``, only one fetched after that (an older or the same copy,
        this process's own, isn't even read from the file, let alone decompressed)."""
        if newer_than is None:
            row = self._store_call("cache_get", text)
        else:
            row = self._store_call("cache_get", text, newer_than)
        if not row or (newer_than is not None and float(row[0]) <= newer_than):
            return None
        try:
            fetched_at = float(row[0])
            return _Row(fetched_at, _unpack(row[3]), max(0.0, float(row[1]) - fetched_at),
                        max(0.0, float(row[2]) - fetched_at))
        except Exception as exc:  # a damaged row is no copy
            log.warning("%s: the shared copy of %s couldn't be read (%s)", self.name or "cache", text[:120],
                        type(exc).__name__)
            return None

    def _adopt(self, key: Hashable, row: _Row, kind: str) -> Any:
        """Keep the table's copy in memory (as old as it really is, with the life it was saved with), unless memory
        already has a newer one."""
        age = max(0.0, self._wall() - row.fetched_at)
        with self._lock:
            current = self._entries.get(key)
            if current is not None and current[1] >= row.fetched_at:
                return current
            entry = (self._mono() - age, row.fetched_at, row.value, kind, row.fresh_for, row.usable_for)
            self._set_locked(key, entry)
            return entry

    def _save(self, text: str, value: Any, fetched_at: float, fresh_for: float, usable_for: float,
              persist: Optional[bool], kind: str) -> None:
        if persist is False or self._shared is None:
            return
        blob = _pack(value, strict=persist is None)
        if blob is None:
            return  # not JSON (or not the same after it): this process's memory only
        if len(blob) > self.MAX_SHARED_BYTES:
            log.info("%s: %s is %d KB compressed; kept in memory only", self.name or "cache", text[:120],
                     len(blob) // 1024)
            return
        fresh_for = min(max(fresh_for, 0.0), self.MAX_TTL)
        usable_for = min(max(usable_for, fresh_for), self.MAX_TTL)
        self._store_call("cache_put", text, kind, blob, fetched_at, fetched_at + fresh_for, fetched_at + usable_for)

    def _wrote(self) -> None:
        """Count a copy kept; prune memory and the table hourly or every PRUNE_WRITES copies."""
        with self._lock:
            self._writes += 1
            due = self._writes >= self.PRUNE_WRITES or self._wall() - self._pruned_at >= self.PRUNE_EVERY
            if due:
                self._writes, self._pruned_at = 0, self._wall()
        if due:
            self.prune()

    def _take_lease(self, text: str) -> bool:
        return bool(self._store_call("cache_lease", text, self.holder, self.lease_seconds, self._wall(),
                                     default=True))

    def _claim(self, text: str, seen: Optional[float]) -> Tuple[bool, Optional[_Row]]:
        """Take the key's lease, then look at the table once more: the holder before may have saved its value and
        let go between this process's first look and its claim. Returns (True, None) when this process should
        load, (False, row) when that value is there (the lease is let go again), (False, None) when another
        process holds the lease."""
        if not self._take_lease(text):
            return False, None
        row = self._read_row(text, seen)
        if row is not None:
            self._store_call("cache_release", text, self.holder)
            return False, row
        return True, None

    def _wait_for_holder(self, text: str, seen: Optional[float]) -> Tuple[Optional[_Row], bool]:
        """Poll the table for the lease holder's value (newer than ``seen``). Returns (row, False) when it comes,
        (None, True) when the lease came free first (this process now holds it: the holder failed or died), and
        (None, False) when the wait ran out."""
        if self.lease_wait <= 0:
            return None, False
        deadline = self._mono() + self.lease_wait
        for _ in range(int(math.ceil(self.lease_wait / self.lease_poll)) + 1):
            if self._mono() >= deadline:
                break
            self._sleep(self.lease_poll)
            row = self._read_row(text, seen)
            if row is not None:
                return row, False
            leased, row = self._claim(text, seen)
            if row is not None or leased:
                return row, leased
        return None, False

    def _store_call(self, method: str, *args: Any, default: Any = None) -> Any:
        """A store method; a failing store makes the cache memory-only for that call (logged once a minute)."""
        if self._shared is None:
            return default
        try:
            return getattr(self._shared, method)(*args)
        except Exception as exc:
            now = _real_monotonic()
            if now - self._store_warned >= MINUTE:
                self._store_warned = now
                log.warning("%s: the shared cache table failed (%s %s); using memory only", self.name or "cache",
                            method, type(exc).__name__)
            return default

    def prune(self, now: Optional[float] = None) -> int:
        """Drop the copies a day past the end of their stale window, from memory and the table (and lapsed
        leases); returns the table's rows deleted."""
        now = self._wall() if now is None else now
        with self._lock:
            doomed = [key for key, entry in self._entries.items()
                      if now - entry[1] > _own_life(entry)[1] + self.KEEP_AFTER_STALE]
            for key in doomed:
                old = self._entries.pop(key)
                self._count(old[3] if len(old) > 3 and old[3] else _default_kind(key), "expired")
                record = self._key_locks.get(key)
                if record is not None and record[1] <= 0:
                    del self._key_locks[key]
            for key in [key for key, until in self._busy.items() if until <= self._mono()]:
                del self._busy[key]
        return int(self._store_call("cache_prune", now, self.KEEP_AFTER_STALE, default=0) or 0)

    # --- memory bookkeeping (callers hold self._lock where named _locked) -------------------------------------

    def _set_locked(self, key: Hashable, entry: Any) -> None:
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            old_key, old = self._entries.popitem(last=False)
            self._count(old[3] if len(old) > 3 and old[3] else _default_kind(old_key), "evictions")
            record = self._key_locks.get(old_key)
            if record is not None and record[1] <= 0:
                del self._key_locks[old_key]

    def _hold_locked(self, key: Hashable) -> List[Any]:
        record = self._key_locks.get(key)
        if record is None:
            record = self._key_locks[key] = [threading.Lock(), 0]
        record[1] += 1
        return record

    def _unhold(self, key: Hashable, record: List[Any]) -> None:
        with self._lock:
            record[1] -= 1
            if record[1] <= 0 and key not in self._entries and self._key_locks.get(key) is record:
                del self._key_locks[key]

    def _remember_locked(self, key: Hashable, error_ttl: float, exc: BaseException,
                         cap: Optional[float] = None) -> None:
        """Remember a failure for ``error_ttl`` seconds; with ``cap``, doubling with each failure in a row up to it."""
        now = self._mono()
        failures = (self._errors.get(key) or (0,))[0] + 1
        wait = error_ttl if cap is None else min(error_ttl * 2 ** min(failures - 1, 20), max(cap, error_ttl))
        self._errors[key] = (failures, now + wait, exc, self._wall())
        self._errors.move_to_end(key)
        while self._errors and (len(self._errors) > self.max_entries
                                or next(iter(self._errors.values()))[1] < now - self.ERROR_CAP):
            self._errors.popitem(last=False)

    def _backing_off_locked(self, key: Hashable, entry: Optional[Any]) -> Optional[Tuple[Any, ...]]:
        """The key's remembered failure while it is backing off, else None. A failure only holds back copies
        older than it: one that arrived since (the other process's, a put) is judged on its own."""
        failure = self._errors.get(key)
        if failure is None or self._mono() >= failure[1]:
            return None
        if entry is not None and entry[1] > failure[3]:
            return None
        return failure

    def _count(self, kind: str, field: str) -> None:
        row = self._stats.get(kind)
        if row is None:
            row = self._stats[kind] = dict.fromkeys(STAT_FIELDS, 0)
        row[field] += 1

    def _describe(self, key: Hashable, text: Optional[str]) -> str:
        if text is None:
            try:
                text = key_text(key)
            except (TypeError, ValueError):
                text = f"<{type(key).__name__} key>"
        return _scrub(text)[:160]

    def _error_text(self, exc: BaseException) -> str:
        name = type(exc).__name__
        if not self.error_detail:
            return name
        detail = _scrub(str(exc))
        return f"{name}: {detail}" if detail else name

    # --- the drop-in API, and more -----------------------------------------------------------------------------

    def loaded_at(self, key: Hashable) -> Optional[float]:
        """Wall-clock time (epoch seconds) the cached value was fetched, or None."""
        with self._lock:
            entry = self._entries.get(key)
        return entry[1] if entry else None

    def age(self, key: Hashable) -> Optional[float]:
        """Seconds since the cached value was fetched, or None when not cached."""
        with self._lock:
            entry = self._entries.get(key)
        return self._mono() - entry[0] if entry else None

    def invalidate(self, key: Hashable) -> None:
        """Forget ``key`` here and in the shared table (and any remembered failure), so the next read loads it.
        The other process's memory copy lives out its own TTL."""
        self._drop(key)

    def _drop(self, key: Hashable, upto: Optional[float] = None) -> None:
        """invalidate; with ``upto``, the table's copy goes only if it was fetched no later than that."""
        with self._lock:
            self._entries.pop(key, None)
            self._errors.pop(key, None)
            record = self._key_locks.get(key)
            if record is not None and record[1] <= 0:
                del self._key_locks[key]
        try:
            text = key_text(key)
        except (TypeError, ValueError):
            return
        if upto is None:
            self._store_call("cache_delete", text)
        else:
            self._store_call("cache_delete", text, upto)

    def invalidate_prefix(self, *prefixes: Any) -> int:
        """Forget every tuple key starting with one of ``prefixes`` (each a tuple, e.g. ("game", league, event_id)),
        here and in the shared table; returns how many memory copies went."""
        wanted = [tuple(p) if isinstance(p, tuple) else (p,) for p in prefixes]
        with self._lock:
            doomed = [key for key in self._entries
                      if isinstance(key, tuple) and any(key[:len(p)] == p for p in wanted)]
            for key in doomed:
                del self._entries[key]
                record = self._key_locks.get(key)
                if record is not None and record[1] <= 0:
                    del self._key_locks[key]
            for key in [key for key in self._errors
                        if isinstance(key, tuple) and any(key[:len(p)] == p for p in wanted)]:
                del self._errors[key]
        for prefix in wanted:
            try:
                text = _prefix_text(prefix)
            except (TypeError, ValueError):
                continue
            self._store_call("cache_delete_prefix", text)
        return len(doomed)

    def peek(self, key: Hashable, shared: bool = True) -> Optional[Tuple[float, Any]]:
        """(fetched_at, value) of the copy in memory, else (``shared``) of the shared table's (kept in memory then),
        without loading; None when there is neither."""
        with self._lock:
            entry = self._entries.get(key)
        if entry is not None:
            return entry[1], entry[2]
        text = self._text(key)
        if text is None or self._shared is None or not shared:
            return None
        row = self._read_row(text)
        if row is None:
            return None
        entry = self._adopt(key, row, _default_kind(key))
        return entry[1], entry[2]

    def seed(self, key: Hashable, value: Any, fetched_at: Optional[float] = None, kind: Optional[str] = None) -> None:
        """Start ``key`` (in memory) from a copy fetched at ``fetched_at`` elsewhere, unless one is held; its
        freshness is judged by each read's own TTL."""
        fetched_at = self._wall() if fetched_at is None else float(fetched_at)
        age = max(0.0, self._wall() - fetched_at)
        with self._lock:
            if key not in self._entries:
                self._set_locked(key, (self._mono() - age, fetched_at, value, kind or _default_kind(key), math.inf,
                                       math.inf))

    def put(self, key: Hashable, value: Any, ttl: float = DAY, stale: float = 0.0, persist: Optional[bool] = None,
            kind: Optional[str] = None) -> float:
        """Keep ``value`` as the copy of ``key`` fetched now (and save it to the table as get would); returns its
        time. ``ttl`` and ``stale`` are the copy's life (reads judge freshness by their own TTL, cut to it) and how
        long the table keeps the row."""
        kind = kind or _default_kind(key)
        text = self._text(key)
        fetched_at, now_m = self._wall(), self._mono()
        fresh_for = max(0.0, float(ttl))
        usable_for = fresh_for + max(0.0, float(stale))
        with self._lock:
            self._errors.pop(key, None)
            self._busy.pop(key, None)
            self._set_locked(key, (now_m, fetched_at, value, kind, fresh_for, usable_for))
        if text is not None:
            self._save(text, value, fetched_at, fresh_for, usable_for, persist, kind)
        self._wrote()
        return fetched_at

    def now(self) -> float:
        """This cache's wall clock (epoch seconds): the time its copies' fetched_at are on."""
        return self._wall()

    def failing(self, key: Hashable) -> Optional[Tuple[int, float]]:
        """(failures in a row, seconds until the next try) while ``key`` is backing off a failure, else None."""
        with self._lock:
            failure = self._errors.get(key)
        if failure is None:
            return None
        left = failure[1] - self._mono()
        return (failure[0], left) if left > 0 else None

    def clear(self) -> None:
        """Forget everything in this process's memory (the shared table is left alone)."""
        with self._lock:
            self._entries.clear()
            self._errors.clear()
            self._busy.clear()
            for key in [key for key, record in self._key_locks.items() if record[1] <= 0]:
                del self._key_locks[key]

    def stats(self, table: bool = False) -> Dict[str, Any]:
        """{"name", "entries", "max_entries", "shared", "refreshing", "failing", "kinds": {kind: {field: n}},
        "total": {field: n}} (fields: STAT_FIELDS), and with ``table`` the shared table's size ("table": {"rows",
        "bytes", "kinds"})."""
        with self._lock:
            kinds = {kind: dict(row) for kind, row in sorted(self._stats.items())}
            result: Dict[str, Any] = {"name": self.name, "entries": len(self._entries),
                                      "max_entries": self.max_entries, "shared": self._shared is not None,
                                      "refreshing": len(self._refreshing), "failing": len(self._errors)}
        result["kinds"] = kinds
        result["total"] = _add(kinds.values())
        if table:
            has_summary = hasattr(self._shared, "cache_summary")
            result["table"] = self._store_call("cache_summary", default=None) if has_summary else None
        return result


def _add(rows: Iterable[Dict[str, int]]) -> Dict[str, int]:
    total = dict.fromkeys(STAT_FIELDS, 0)
    for row in rows:
        for field in STAT_FIELDS:
            total[field] += row.get(field, 0)
    return total


def all_stats() -> Dict[str, Any]:
    """Every cache in this process: {"caches": [each stats()], "kinds": {kind: summed}, "total": summed}."""
    caches = sorted((cache.stats() for cache in list(_INSTANCES)), key=lambda s: (s["name"], -s["entries"]))
    kinds: Dict[str, Dict[str, int]] = {}
    for stats in caches:
        for kind, row in stats["kinds"].items():
            kinds[kind] = _add([kinds.get(kind, {}), row])
    return {"caches": caches, "kinds": dict(sorted(kinds.items())), "total": _add(kinds.values())}


def stats_line(stats: Optional[Dict[str, Any]] = None) -> str:
    """One line for the log: reads, how many memory/shared answered, loads and failures, and memory size."""
    stats = stats or all_stats()
    total = stats["total"]
    reads = sum(total[field] for field in READ_FIELDS)
    entries = sum(c["entries"] for c in stats["caches"]) if "caches" in stats else stats.get("entries", 0)

    def share(n: int) -> str:
        return f"{100 * n / reads:.0f}%" if reads else "0%"

    return (f"cache: {reads} reads, {share(total['hits'] + total['stale_hits'])} from memory "
            f"({total['stale_hits']} stale), {share(total['shared_hits'])} shared, {total['misses']} misses; "
            f"{total['loads']} loads ({total['load_errors']} failed, {total['negative_hits']} failures remembered), "
            f"{total['refreshes']} background refreshes, {total['evictions']} evicted; {entries} in memory")
