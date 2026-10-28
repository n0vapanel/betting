"""The learning loop's own tables: versioned parameters, the trials ledger and the weekly scores.

- ``brain_params``: one row per parameter version of one sport. ``status`` is ``champion`` (live), ``challenger``
  (scored in shadow, not live), ``retired`` (a former champion, replaced by a promotion) or ``rolled_back``;
  ``source`` is ``refit``, ``ai:<provider>`` or ``owner``; ``parent`` is the champion it replaced (0: the
  defaults). A version applies only to games logged after its ``effective_from``.
- ``brain_trials``: every candidate the gate judged, and every rollback, undo and safe-mode switch, with its
  numbers (n, mean change in log loss and its 95% range, p-values) and the decision in words.
- ``brain_scores``: per week, sport and target, the brain's scores next to the market's and a simple baseline's,
  and its calibration slope and intercept.
- ``brain_state``: a few keys: ``epoch`` (bumped on every change of what is live, so readers know when to reload),
  ``safe_mode`` and run marks.

``BRAIN_SCHEMA`` / ``init_brain_schema(db)`` create them on any connection to the shared SQLite file (data/store.py
hooks it in); every helper below takes an open ``sqlite3.Connection`` and works in autocommit or inside the
caller's transaction. ``BrainStore`` wraps a Store (or a path) for code that has no connection open.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import closing, contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

BRAIN_SCHEMA = """
CREATE TABLE IF NOT EXISTS brain_params (
    version INTEGER PRIMARY KEY AUTOINCREMENT,
    parent INTEGER NOT NULL DEFAULT 0,  -- the champion this one replaced or challenges (0: the defaults)
    sport TEXT NOT NULL,           -- learn/registry.py's sport key: nfl, ncaaf, nba, wnba, ncaab, nhl, mlb, soccer, other
    params_json TEXT NOT NULL,     -- JSON: every knob's value for the sport (a version is complete on its own)
    status TEXT NOT NULL,          -- champion, challenger, retired or rolled_back
    created_at REAL NOT NULL,
    effective_from REAL,           -- live (or scored in shadow) for games logged after it
    evidence_json TEXT,            -- JSON: the gate's numbers
    source TEXT NOT NULL,          -- refit, ai:<provider> or owner
    knob TEXT,                     -- the knob(s) this version changed, comma separated
    gate TEXT,                     -- nudge or structural
    ended_at REAL                  -- when it stopped being live or in shadow
);
CREATE INDEX IF NOT EXISTS brain_params_by_sport ON brain_params (sport, status);
CREATE TABLE IF NOT EXISTS brain_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    week TEXT NOT NULL,            -- ISO week of the decision, "2026-W44"
    sport TEXT NOT NULL,
    knob TEXT NOT NULL,
    old TEXT,                      -- JSON: the value before
    new TEXT,                      -- JSON: the value tried
    source TEXT NOT NULL,          -- refit, ai:<provider>, owner, tripwire
    train_n INTEGER,               -- games the candidate was fitted on
    eval_n INTEGER,                -- out-of-sample games it was judged on
    dll REAL,                      -- mean change in log loss per game (candidate minus champion; negative is better)
    lo REAL,                       -- its 95% range (day-block bootstrap)
    hi REAL,
    p REAL,                        -- one-sided Diebold-Mariano (HLN) p-value, structural candidates
    holm_p REAL,                   -- Holm-adjusted within the week's structural candidates
    decision TEXT NOT NULL,        -- promoted, held, rejected, shadow, rolled_back, undone, safe_mode, cleared
    reason TEXT,                   -- plain words, from code
    created_at REAL NOT NULL,
    gate TEXT,                     -- nudge, structural, rollback, owner, tripwire
    version INTEGER,               -- the brain_params version made or judged
    details_json TEXT              -- JSON: everything else the gate computed (Brier change, slope CI, shadow progress)
);
CREATE INDEX IF NOT EXISTS brain_trials_by_week ON brain_trials (week, sport);
CREATE TABLE IF NOT EXISTS brain_scores (
    week TEXT NOT NULL,            -- ISO week of the games' start, "2026-W44"
    sport TEXT NOT NULL,
    target TEXT NOT NULL,          -- result, total, goals, btts, corners, cards
    n INTEGER NOT NULL,            -- graded games scored
    brain_ll REAL,                 -- mean log loss (counts: mean log score) of the brain as served
    market_ll REAL,                -- the sharp (or closing) market's, on the games it priced
    baseline_ll REAL,              -- the base rate's (counts: climatology's)
    brier REAL,
    slope REAL,                    -- calibration slope over the trailing window (300 games), with its 95% range
    slope_lo REAL,
    slope_hi REAL,
    intercept REAL,
    intercept_lo REAL,
    intercept_hi REAL,
    gap REAL,                      -- mean brain minus market log loss on the same games, with its 95% range
    gap_lo REAL,
    gap_hi REAL,
    n_market INTEGER,              -- games the market priced too
    market_brier REAL,
    baseline_brier REAL,
    details_json TEXT,             -- JSON: reliability curve and band, PIT histogram, CRPS, per-version counts
    params_version INTEGER NOT NULL DEFAULT 0,  -- the version live for the week's latest game (0: the defaults)
    computed_at REAL NOT NULL,
    PRIMARY KEY (week, sport, target)
);
CREATE TABLE IF NOT EXISTS brain_state (
    key TEXT PRIMARY KEY,
    value TEXT,                    -- JSON
    updated_at REAL NOT NULL
);
"""

PARAM_STATUSES = ("champion", "challenger", "retired", "rolled_back")
PARAM_FIELDS = ("version", "parent", "sport", "params_json", "status", "created_at", "effective_from",
                "evidence_json", "source", "knob", "gate", "ended_at")
TRIAL_FIELDS = ("id", "week", "sport", "knob", "old", "new", "source", "train_n", "eval_n", "dll", "lo", "hi", "p",
                "holm_p", "decision", "reason", "created_at", "gate", "version", "details_json")
SCORE_FIELDS = ("week", "sport", "target", "n", "brain_ll", "market_ll", "baseline_ll", "brier", "slope", "slope_lo",
                "slope_hi", "intercept", "intercept_lo", "intercept_hi", "gap", "gap_lo", "gap_hi", "n_market",
                "market_brier", "baseline_brier", "details_json", "params_version", "computed_at")
DECISIONS = ("promoted", "held", "rejected", "shadow", "rolled_back", "undone", "safe_mode", "cleared")


def init_brain_schema(db: sqlite3.Connection) -> None:
    """Create the learning loop's tables on ``db`` (idempotent)."""
    db.executescript(BRAIN_SCHEMA)


# --- small helpers ------------------------------------------------------------------------------------------------

def _clean(value: Any) -> Any:
    """JSON-safe: non-finite floats become None, tuples lists, numpy scalars plain numbers."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and callable(value.item):  # numpy scalar
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if hasattr(value, "tolist"):
        return _clean(value.tolist())
    return str(value)


def dumps(value: Any) -> Optional[str]:
    return None if value is None else json.dumps(_clean(value), sort_keys=True, separators=(",", ":"))


def loads(text: Any) -> Any:
    if text is None or text == "":
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@contextmanager
def transaction(db: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT, unless the caller already has a transaction open (then it is theirs)."""
    if db.in_transaction:
        yield db
        return
    db.execute("BEGIN IMMEDIATE")
    try:
        yield db
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")


def _param_row(values: Sequence[Any]) -> Dict[str, Any]:
    row = dict(zip(PARAM_FIELDS, values))
    row["params"] = loads(row.pop("params_json")) or {}
    row["evidence"] = loads(row.pop("evidence_json"))
    return row


def _trial_row(values: Sequence[Any]) -> Dict[str, Any]:
    row = dict(zip(TRIAL_FIELDS, values))
    row["old"], row["new"] = loads(row["old"]), loads(row["new"])
    row["details"] = loads(row.pop("details_json"))
    return row


def _score_row(values: Sequence[Any]) -> Dict[str, Any]:
    row = dict(zip(SCORE_FIELDS, values))
    row["details"] = loads(row.pop("details_json"))
    return row


# --- state: epoch, safe mode, run marks ---------------------------------------------------------------------------

def get_state(db: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = db.execute("SELECT value FROM brain_state WHERE key = ?", (key,)).fetchone()
    return default if row is None else loads(row[0])


def set_state(db: sqlite3.Connection, key: str, value: Any, now: Optional[float] = None) -> None:
    db.execute("INSERT INTO brain_state (key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
               "value = excluded.value, updated_at = excluded.updated_at",
               (key, dumps(value), time.time() if now is None else now))


def epoch(db: sqlite3.Connection) -> int:
    """A counter bumped whenever what is live changes (a promotion, rollback, undo or safe-mode switch)."""
    value = get_state(db, "epoch", 0)
    return int(value) if isinstance(value, (int, float)) else 0


def bump_epoch(db: sqlite3.Connection, now: Optional[float] = None) -> int:
    with transaction(db):
        value = epoch(db) + 1
        set_state(db, "epoch", value, now)
    return value


def safe_mode(db: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    """{"since", "reason", "sport"?} while safe mode is on (the defaults are served), else None."""
    value = get_state(db, "safe_mode")
    return value if isinstance(value, dict) and value.get("on") else None


def set_safe_mode(db: sqlite3.Connection, on: bool, reason: str = "", now: Optional[float] = None,
                  details: Optional[Mapping[str, Any]] = None) -> bool:
    """Switch safe mode on or off; True when that changed it (and the epoch was bumped)."""
    now = time.time() if now is None else now
    with transaction(db):
        current = safe_mode(db)
        if bool(current) == bool(on):
            return False
        value = {"on": True, "since": now, "reason": reason, **dict(details or {})} if on else {"on": False,
                                                                                               "cleared_at": now}
        set_state(db, "safe_mode", value, now)
        set_state(db, "epoch", epoch(db) + 1, now)
    return True


# --- parameter versions -------------------------------------------------------------------------------------------

def insert_version(db: sqlite3.Connection, sport: str, params: Mapping[str, Any], *, status: str, parent: int = 0,
                   source: str = "refit", evidence: Any = None, effective_from: Optional[float] = None,
                   now: Optional[float] = None, knob: Optional[str] = None, gate: Optional[str] = None) -> int:
    """Add a version (a challenger, or a champion through ``promote``); returns its number."""
    if status not in PARAM_STATUSES:
        raise ValueError(f"Unknown parameter status {status!r}")
    now = time.time() if now is None else now
    cursor = db.execute(
        "INSERT INTO brain_params (parent, sport, params_json, status, created_at, effective_from, evidence_json, "
        "source, knob, gate) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (int(parent or 0), sport, dumps(dict(params)), status, now, effective_from, dumps(evidence), source, knob,
         gate))
    return int(cursor.lastrowid)


def get_version(db: sqlite3.Connection, version: int) -> Optional[Dict[str, Any]]:
    values = db.execute(f"SELECT {', '.join(PARAM_FIELDS)} FROM brain_params WHERE version = ?",
                        (int(version),)).fetchone()
    return _param_row(values) if values else None


def versions(db: sqlite3.Connection, sport: Optional[str] = None, status: Optional[str] = None,
             limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Versions, newest first."""
    where, args = [], []
    for clause, value in (("sport = ?", sport), ("status = ?", status)):
        if value is not None:
            where.append(clause)
            args.append(value)
    sql = f"SELECT {', '.join(PARAM_FIELDS)} FROM brain_params"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY version DESC LIMIT ?"
    return [_param_row(v) for v in db.execute(sql, (*args, -1 if limit is None else int(limit))).fetchall()]


def champion(db: sqlite3.Connection, sport: str) -> Optional[Dict[str, Any]]:
    """The live version of ``sport``, or None (the defaults are live)."""
    values = db.execute(f"SELECT {', '.join(PARAM_FIELDS)} FROM brain_params WHERE sport = ? AND status = 'champion' "
                        "ORDER BY version DESC LIMIT 1", (sport,)).fetchone()
    return _param_row(values) if values else None


def champions(db: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for values in db.execute(f"SELECT {', '.join(PARAM_FIELDS)} FROM brain_params WHERE status = 'champion' "
                             "ORDER BY version ASC").fetchall():
        row = _param_row(values)
        out[row["sport"]] = row
    return out


def promote(db: sqlite3.Connection, version: int, now: Optional[float] = None,
            evidence: Any = None) -> Dict[str, Any]:
    """Make ``version`` its sport's champion from ``now``: the current champion becomes ``retired`` and the new
    one's parent; the epoch is bumped. Returns the promoted row."""
    now = time.time() if now is None else now
    with transaction(db):
        row = get_version(db, version)
        if row is None:
            raise ValueError(f"No parameter version {version}")
        if row["status"] == "champion":
            return row
        current = champion(db, row["sport"])
        parent = current["version"] if current else 0
        if current:
            db.execute("UPDATE brain_params SET status = 'retired', ended_at = ? WHERE version = ?",
                       (now, current["version"]))
        if evidence is not None:
            db.execute("UPDATE brain_params SET evidence_json = ? WHERE version = ?", (dumps(evidence), version))
        db.execute("UPDATE brain_params SET status = 'champion', parent = ?, effective_from = ?, ended_at = NULL "
                   "WHERE version = ?", (parent, now, version))
        set_state(db, "epoch", epoch(db) + 1, now)
        return get_version(db, version) or row


def roll_back(db: sqlite3.Connection, version: int, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Retire a champion as ``rolled_back`` and make its parent live again (from ``now``; parent 0: the defaults).
    Returns the parent row (None for the defaults); ValueError when ``version`` is not the live champion."""
    now = time.time() if now is None else now
    with transaction(db):
        row = get_version(db, version)
        if row is None or row["status"] != "champion":
            raise ValueError(f"Version {version} is not a live champion")
        db.execute("UPDATE brain_params SET status = 'rolled_back', ended_at = ? WHERE version = ?", (now, version))
        parent = get_version(db, row["parent"]) if row["parent"] else None
        if parent is not None:
            db.execute("UPDATE brain_params SET status = 'champion', effective_from = ?, ended_at = NULL "
                       "WHERE version = ?", (now, parent["version"]))
        set_state(db, "epoch", epoch(db) + 1, now)
        return get_version(db, parent["version"]) if parent is not None else None


def set_params(db: sqlite3.Connection, version: int, params: Mapping[str, Any]) -> None:
    """Replace a not-yet-live version's values (a challenger rebased on the champion it will replace)."""
    db.execute("UPDATE brain_params SET params_json = ? WHERE version = ? AND status = 'challenger'",
               (dumps(dict(params)), int(version)))


def set_evidence(db: sqlite3.Connection, version: int, evidence: Any) -> None:
    """Replace a version's evidence (an AI challenger's planned looks are added to it as they happen)."""
    db.execute("UPDATE brain_params SET evidence_json = ? WHERE version = ?", (dumps(evidence), int(version)))


def end_challenger(db: sqlite3.Connection, version: int, status: str = "retired",
                   now: Optional[float] = None) -> None:
    """A challenger that failed (``retired``) or was withdrawn stops being scored."""
    if status not in ("retired", "rolled_back"):
        raise ValueError(f"A challenger ends retired or rolled_back, not {status!r}")
    db.execute("UPDATE brain_params SET status = ?, ended_at = ? WHERE version = ? AND status = 'challenger'",
               (status, time.time() if now is None else now, int(version)))


# --- the trials ledger --------------------------------------------------------------------------------------------

def add_trial(db: sqlite3.Connection, row: Mapping[str, Any], now: Optional[float] = None) -> int:
    """Append one row to the ledger. ``row`` holds TRIAL_FIELDS by name (``old``/``new``/``details`` as values)."""
    if row.get("decision") not in DECISIONS:
        raise ValueError(f"Unknown trial decision {row.get('decision')!r}")
    now = time.time() if now is None else now
    values = (row["week"], row["sport"], row["knob"], dumps(row.get("old")), dumps(row.get("new")),
              row.get("source") or "refit", row.get("train_n"), row.get("eval_n"), _num(row.get("dll")),
              _num(row.get("lo")), _num(row.get("hi")), _num(row.get("p")), _num(row.get("holm_p")), row["decision"],
              row.get("reason"), row.get("created_at") or now, row.get("gate"), row.get("version"),
              dumps(row.get("details")))
    cursor = db.execute(f"INSERT INTO brain_trials ({', '.join(TRIAL_FIELDS[1:])}) VALUES "
                        f"({', '.join('?' * (len(TRIAL_FIELDS) - 1))})", values)
    return int(cursor.lastrowid)


def trials(db: sqlite3.Connection, week: Optional[str] = None, sport: Optional[str] = None,
           since: Optional[float] = None, limit: Optional[int] = 500) -> List[Dict[str, Any]]:
    """Ledger rows, newest first."""
    where, args = [], []
    for clause, value in (("week = ?", week), ("sport = ?", sport), ("created_at >= ?", since)):
        if value is not None:
            where.append(clause)
            args.append(value)
    sql = f"SELECT {', '.join(TRIAL_FIELDS)} FROM brain_trials"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    return [_trial_row(v) for v in db.execute(sql, (*args, -1 if limit is None else int(limit))).fetchall()]


# --- weekly scores ------------------------------------------------------------------------------------------------

def put_score(db: sqlite3.Connection, row: Mapping[str, Any], now: Optional[float] = None) -> None:
    """Insert or replace one (week, sport, target) score row (nightly runs rewrite the week to date)."""
    now = time.time() if now is None else now
    values = []
    for name in SCORE_FIELDS:
        if name == "details_json":
            values.append(dumps(row.get("details")))
        elif name == "computed_at":
            values.append(row.get("computed_at") or now)
        elif name in ("week", "sport", "target"):
            values.append(str(row[name]))
        elif name in ("n", "n_market", "params_version"):
            value = row.get(name)
            values.append(int(value) if value is not None else (0 if name != "n_market" else None))
        else:
            values.append(_num(row.get(name)))
    db.execute(f"INSERT OR REPLACE INTO brain_scores ({', '.join(SCORE_FIELDS)}) VALUES "
               f"({', '.join('?' * len(SCORE_FIELDS))})", values)


def scores(db: sqlite3.Connection, sport: Optional[str] = None, target: Optional[str] = None,
           since_week: Optional[str] = None) -> List[Dict[str, Any]]:
    """Score rows, oldest week first."""
    where, args = [], []
    for clause, value in (("sport = ?", sport), ("target = ?", target), ("week >= ?", since_week)):
        if value is not None:
            where.append(clause)
            args.append(value)
    sql = f"SELECT {', '.join(SCORE_FIELDS)} FROM brain_scores"
    if where:
        sql += " WHERE " + " AND ".join(where)
    return [_score_row(v) for v in db.execute(sql + " ORDER BY week ASC, sport, target", args).fetchall()]


# --- a Store wrapper ----------------------------------------------------------------------------------------------

class BrainStore:
    """Opens connections on a data/store.Store (its ``_connect``) or a SQLite path, and makes sure the tables exist
    (once per instance). ``connect()`` is a context manager that closes the connection afterwards."""

    def __init__(self, store: Any) -> None:
        if isinstance(store, BrainStore):
            store = store.store
        self.store = store
        self.path = store if isinstance(store, str) else getattr(store, "path", None)
        self._ready = False

    def _open(self) -> sqlite3.Connection:
        if isinstance(self.store, str):
            return sqlite3.connect(self.store, timeout=10, isolation_level=None)
        return self.store._connect()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with closing(self._open()) as db:
            if not self._ready:
                init_brain_schema(db)
                self._ready = True
            yield db

    # Convenience pass-throughs, each on its own connection
    def champion(self, sport: str) -> Optional[Dict[str, Any]]:
        with self.connect() as db:
            return champion(db, sport)

    def champions(self) -> Dict[str, Dict[str, Any]]:
        with self.connect() as db:
            return champions(db)

    def version(self, version: int) -> Optional[Dict[str, Any]]:
        with self.connect() as db:
            return get_version(db, version)

    def versions(self, sport: Optional[str] = None, status: Optional[str] = None,
                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self.connect() as db:
            return versions(db, sport, status, limit)

    def epoch(self) -> int:
        with self.connect() as db:
            return epoch(db)

    def safe_mode(self) -> Optional[Dict[str, Any]]:
        with self.connect() as db:
            return safe_mode(db)

    def trials(self, week: Optional[str] = None, sport: Optional[str] = None, since: Optional[float] = None,
               limit: Optional[int] = 500) -> List[Dict[str, Any]]:
        with self.connect() as db:
            return trials(db, week, sport, since, limit)

    def scores(self, sport: Optional[str] = None, target: Optional[str] = None,
               since_week: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.connect() as db:
            return scores(db, sport, target, since_week)

    def state(self, key: str, default: Any = None) -> Any:
        with self.connect() as db:
            return get_state(db, key, default)

    def set_state(self, key: str, value: Any, now: Optional[float] = None) -> None:
        with self.connect() as db:
            set_state(db, key, value, now)
