"""A SQLite file shared by the bot and the dashboard on the same server.

Holds each league's latest odds snapshot (so the two processes never pay
twice for the same fetch and nothing is lost on restart), the last
all-sports slate Today is drawn from (served at once after a restart), a log of The
Odds API's quota headers, short leases so only one process refreshes a
league at a time, the predictions people make (and how they scored), and
the AI's own pre-game predictions for its track record, the AI
analysts' forecasts (Claude, DeepSeek) with what each call cost, and the
markets brain's consensus for every game (each source's opinion and the
blend, frozen at the start and scored after the final). WAL mode lets
readers and a writer work concurrently.

Version 2 adds per-person accounts, so a slip started in Telegram opens
unchanged in the browser: users (with the limits the AI cannot edit: age
confirmation, a weekly budget whose raises wait 24 hours, a cool-off),
saved slips whose legs keep the exact price they were built on, rollover
plans recorded one day at a time, chat threads, each person's daily AI
spend, and responsible-gambling flags for the owner to review. Money is in
whole cents and prices are exact decimal strings, never floats.

Pasted bet slips keep only what was read from them (and the pasted text),
never a screenshot, each in its owner's own rows: the newest 200 a person.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
import os
import secrets
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
from zoneinfo import ZoneInfo

from ..learn.brain_store import BRAIN_SCHEMA
from ..learn.history_store import HIST_SCHEMA

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    sport_key TEXT PRIMARY KEY,
    fetched_at REAL NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_usage (
    ts REAL NOT NULL,
    endpoint TEXT NOT NULL,
    cost INTEGER,
    remaining INTEGER,
    used INTEGER
);
CREATE TABLE IF NOT EXISTS leases (
    key TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    expires REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player TEXT NOT NULL,          -- who made it, e.g. "web:5f2c9a1e7b3d" (a random browser id) or "tg:12345"
    name TEXT NOT NULL,            -- name shown on the leaderboard
    league TEXT NOT NULL,
    event_id TEXT NOT NULL,        -- the odds feed's game id
    espn_id TEXT,                  -- ESPN's event id, to fetch the result
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    start TEXT NOT NULL,           -- ISO tip-off time
    created_at REAL NOT NULL,
    call TEXT NOT NULL,            -- JSON: what they predicted
    ai TEXT NOT NULL,              -- JSON: the AI's view of their call when they made it
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, graded or void
    result TEXT,                   -- JSON: the final score and how each part of the call did
    user_points REAL,
    ai_points REAL,
    graded_at REAL,
    UNIQUE (player, league, event_id)
);
CREATE TABLE IF NOT EXISTS ai_tickets (
    day TEXT NOT NULL,             -- the slate's date (US Eastern)
    league TEXT NOT NULL,
    tier TEXT NOT NULL,            -- safe, risky or crazy: the tier's top ticket that day
    logged_at REAL NOT NULL,       -- the last update before its first game started
    first_start TEXT NOT NULL,
    last_start TEXT NOT NULL,
    payload TEXT NOT NULL,         -- JSON: legs (with teams and start times), odds, chances
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, won, lost or void
    result TEXT,
    graded_at REAL,
    PRIMARY KEY (day, league, tier)
);
CREATE TABLE IF NOT EXISTS ai_predictions (
    league TEXT NOT NULL,
    event_id TEXT NOT NULL,
    espn_id TEXT,
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    start TEXT NOT NULL,
    logged_at REAL NOT NULL,       -- the last update before tip-off
    payload TEXT NOT NULL,         -- JSON: the AI's and the market's predictions
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    graded_at REAL,
    PRIMARY KEY (league, event_id)
);
CREATE TABLE IF NOT EXISTS analyses (
    analyst TEXT NOT NULL,         -- claude or deepseek
    league TEXT NOT NULL,
    event_id TEXT NOT NULL,
    espn_id TEXT,
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    start TEXT NOT NULL,
    created_at REAL NOT NULL,      -- its latest forecast before tip-off
    payload TEXT NOT NULL,         -- JSON: the forecast, its reasons, and the AI's and market's chances then
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, graded or void
    result TEXT,
    graded_at REAL,
    PRIMARY KEY (analyst, league, event_id)
);
CREATE TABLE IF NOT EXISTS slates (
    name TEXT PRIMARY KEY,         -- "today": the all-sports slate data/slate.py builds
    generated_at REAL NOT NULL,
    etag TEXT NOT NULL,
    body BLOB NOT NULL             -- the slate's JSON, exactly as served, so a restart serves it at once
);
CREATE TABLE IF NOT EXISTS analyst_calls (
    ts REAL NOT NULL,
    analyst TEXT NOT NULL,
    model TEXT NOT NULL,
    event_id TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost REAL NOT NULL,            -- estimated US dollars
    ok INTEGER NOT NULL            -- 0 when no forecast came back (it still counts toward the daily limit)
);
CREATE TABLE IF NOT EXISTS consensus_log (
    league TEXT NOT NULL,
    game_key TEXT NOT NULL,        -- "espn:<id>" when ESPN shows the game, else "odds:<id>": one row per game
    event_id TEXT,                 -- the odds feed's game id
    espn_id TEXT,                  -- ESPN's event id, to fetch the result
    swapped INTEGER NOT NULL DEFAULT 0,  -- 1 when ESPN lists home and away the other way round from the row
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    start TEXT NOT NULL,           -- ISO start time
    logged_at REAL NOT NULL,       -- the last refresh before the start: the row is frozen after it
    sources TEXT NOT NULL,         -- JSON: every source's opinion (engine/consensus.py's Source)
    blend TEXT NOT NULL,           -- JSON: the markets brain's blend (engine/consensus.py's Blend)
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, graded or void
    outcome TEXT,                  -- home, away or draw, once graded
    result TEXT,                   -- JSON: the final score, and each source's and the blend's Brier score and log loss
    graded_at REAL,
    PRIMARY KEY (league, game_key)
);
CREATE INDEX IF NOT EXISTS consensus_log_by_status ON consensus_log (status, start);
CREATE INDEX IF NOT EXISTS consensus_log_by_event ON consensus_log (league, event_id);
"""

CONSENSUS_FIELDS = ("league", "game_key", "event_id", "espn_id", "swapped", "home", "away", "start", "logged_at",
                    "sources", "blend", "status", "outcome", "result", "graded_at")
CONSENSUS_JSON = {"sources", "blend", "result"}
PREDICTION_FIELDS = ("id", "player", "name", "league", "event_id", "espn_id", "home", "away", "start",
                     "created_at", "call", "ai", "status", "result", "user_points", "ai_points", "graded_at")
AI_FIELDS = ("league", "event_id", "espn_id", "home", "away", "start", "logged_at", "payload", "status",
             "result", "graded_at")
TICKET_FIELDS = ("day", "league", "tier", "logged_at", "first_start", "last_start", "payload", "status", "result",
                 "graded_at")
ANALYSIS_FIELDS = ("analyst", "league", "event_id", "espn_id", "home", "away", "start", "created_at", "payload",
                   "status", "result", "graded_at")
JSON_FIELDS = {"call", "ai", "result", "payload"}


def _row(fields: Tuple[str, ...], values: Tuple[Any, ...]) -> Dict[str, Any]:
    row = dict(zip(fields, values))
    for key in JSON_FIELDS & row.keys():
        if row[key] is not None:
            row[key] = json.loads(row[key])
    return row


# --- version 2: accounts, slips, plans, chat, AI credits, responsible gambling ---------------------
# New tables only (CREATE IF NOT EXISTS), so an existing betting.db picks them up on the next start.

ACCOUNT_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,           -- "tg:<telegram id>" or "web:<random>": one identity for bot, Mini App and browser
    name TEXT,
    created_at REAL NOT NULL,
    age_confirmed_at REAL,         -- nothing betting-related shows before this is set
    odds_format TEXT NOT NULL DEFAULT 'decimal',   -- decimal, american or fractional
    currency TEXT NOT NULL DEFAULT 'USD',
    favourite_sports TEXT NOT NULL DEFAULT '[]',   -- JSON: league keys
    risk_tier TEXT,                -- default tier: safe, risky or crazy
    weekly_budget_cents INTEGER,   -- the limit in force (NULL until they set one)
    budget_raise_at REAL,          -- when a pending raise takes effect
    pending_budget_cents INTEGER,  -- a raise waiting out its 24 hours
    reality_check_minutes INTEGER, -- 15, 30 or 60; NULL is off
    cooloff_until REAL,            -- a time-out: picks, building and plans stay off until then
    settings TEXT NOT NULL DEFAULT '{}'            -- JSON: anything else (time zone, theme, ...)
);
CREATE TABLE IF NOT EXISTS slips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',  -- draft, saved, settled or void
    type TEXT NOT NULL,            -- single, multi, round_robin or system
    system TEXT,                   -- a named system bet: trixie, yankee, ...
    k INTEGER,                     -- a round robin's legs per bet
    stake_cents INTEGER NOT NULL,  -- the total staked across every bet on the ticket
    created_at REAL NOT NULL,      -- when it was saved as a bet (a draft: when it was started)
    updated_at REAL NOT NULL,
    share_code TEXT UNIQUE,
    payload TEXT,                  -- JSON: the priced ticket when it was saved
    result TEXT                    -- JSON: how it settled
);
CREATE INDEX IF NOT EXISTS slips_by_user ON slips (user, created_at);
CREATE INDEX IF NOT EXISTS slips_by_status ON slips (status);
CREATE TABLE IF NOT EXISTS slip_legs (
    slip_id INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    key TEXT,                      -- the pricing engine's stable leg id
    league TEXT NOT NULL,
    event_id TEXT NOT NULL,
    market TEXT NOT NULL,
    side TEXT NOT NULL,
    point REAL,
    price TEXT NOT NULL,           -- decimal odds exactly as quoted ("1.91"), never a float
    label TEXT NOT NULL,
    game TEXT,
    start TEXT,                    -- ISO 8601 UTC
    quoted_at REAL,                -- when that price was seen (epoch seconds)
    ai_prob REAL,
    market_prob REAL,
    push_prob REAL,
    book TEXT,
    result TEXT,                   -- JSON: won, lost, push, void, cancelled or a dead heat
    PRIMARY KEY (slip_id, idx)
);
CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT NOT NULL,
    created_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',  -- active, done, failed or stopped
    mode TEXT NOT NULL DEFAULT 'paper',     -- paper (no money) or tracked
    start_cents INTEGER NOT NULL,
    days INTEGER NOT NULL,
    min_odds TEXT NOT NULL,        -- exact decimal string
    legs_per_day INTEGER NOT NULL,
    strategy TEXT NOT NULL,        -- classic, bank_half, ladder or flat
    payload TEXT,                  -- JSON: the side-by-side comparison shown before they confirmed
    bankroll_cents INTEGER NOT NULL,
    day INTEGER NOT NULL DEFAULT 0 -- days recorded so far
);
CREATE INDEX IF NOT EXISTS plans_by_user ON plans (user, created_at);
CREATE TABLE IF NOT EXISTS plan_days (
    plan_id INTEGER NOT NULL,
    day INTEGER NOT NULL,
    slip_id INTEGER,
    stake_cents INTEGER NOT NULL,
    result TEXT,                   -- JSON: won, lost, push, void (NULL until known)
    banked_cents INTEGER NOT NULL DEFAULT 0,
    rolled_cents INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    PRIMARY KEY (plan_id, day)
);
CREATE TABLE IF NOT EXISTS chat_threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT NOT NULL,
    created_at REAL NOT NULL,
    title TEXT,
    summary TEXT                   -- the compacted story so far, so old turns needn't be resent
);
CREATE INDEX IF NOT EXISTS chat_threads_by_user ON chat_threads (user, created_at);
CREATE TABLE IF NOT EXISTS chat_messages (
    thread_id INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    role TEXT NOT NULL,            -- user or assistant
    content TEXT NOT NULL,         -- JSON: text or content blocks (tool calls and results by reference)
    ts REAL NOT NULL,
    tokens INTEGER,
    cost REAL,                     -- estimated US dollars, like analyst_calls
    PRIMARY KEY (thread_id, idx)
);
CREATE TABLE IF NOT EXISTS ai_credits (
    user TEXT NOT NULL,
    day TEXT NOT NULL,             -- YYYY-MM-DD
    used_cents REAL NOT NULL DEFAULT 0,  -- estimated US cents (fractions kept: a turn costs about a cent)
    PRIMARY KEY (user, day)
);
CREATE TABLE IF NOT EXISTS ai_holds (
    user TEXT PRIMARY KEY,         -- one running AI call per person (the bot and the dashboard alike)
    token TEXT NOT NULL,           -- which call holds it, so a late finish never frees someone else's
    cents REAL NOT NULL,           -- the most the call may still spend, held from the shared allowances
    pools TEXT NOT NULL,           -- JSON: the shared allowances it counts against ("*web", "*all")
    started_at REAL NOT NULL,
    expires_at REAL NOT NULL       -- a hold left by a crashed process stops counting then
);
CREATE TABLE IF NOT EXISTS rg_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT NOT NULL,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,            -- the warning sign: chasing, stake_jump, failed_rollovers, late_night, ...
    detail TEXT,                   -- JSON: what triggered it
    action TEXT,                   -- what code did about it, e.g. picks_off_24h, support_message
    reviewed_at REAL               -- when the owner looked at it (NCPG: within 48-72 hours)
);
CREATE INDEX IF NOT EXISTS rg_events_by_user ON rg_events (user, ts);
CREATE TABLE IF NOT EXISTS pasted_slips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- never reused, so an old id can't come to name a newer slip
    user TEXT NOT NULL,
    thread_id INTEGER,             -- the chat it was pasted into, if any
    created_at REAL NOT NULL,
    source TEXT NOT NULL,          -- text (the AI read pasted text), image (a screenshot) or basic (code read it)
    text TEXT,                     -- the pasted text, at most 6,000 characters (screenshots are never kept)
    extraction TEXT NOT NULL,      -- JSON: what was read, as engine/pasted.Extraction
    matches TEXT NOT NULL          -- JSON: each leg's game and price on the boards when it was read
);
CREATE INDEX IF NOT EXISTS pasted_slips_by_user ON pasted_slips (user, created_at);
"""

USER_FIELDS = ("id", "name", "created_at", "age_confirmed_at", "odds_format", "currency", "favourite_sports",
               "risk_tier", "weekly_budget_cents", "budget_raise_at", "pending_budget_cents", "reality_check_minutes",
               "cooloff_until", "settings")
SLIP_FIELDS = ("id", "user", "status", "type", "system", "k", "stake_cents", "created_at", "updated_at", "share_code",
               "payload", "result")
LEG_FIELDS = ("slip_id", "idx", "key", "league", "event_id", "market", "side", "point", "price", "label", "game",
              "start", "quoted_at", "ai_prob", "market_prob", "push_prob", "book", "result")
PLAN_FIELDS = ("id", "user", "created_at", "status", "mode", "start_cents", "days", "min_odds", "legs_per_day",
               "strategy", "payload", "bankroll_cents", "day")
PLAN_DAY_FIELDS = ("plan_id", "day", "slip_id", "stake_cents", "result", "banked_cents", "rolled_cents", "created_at")
THREAD_FIELDS = ("id", "user", "created_at", "title", "summary")
MESSAGE_FIELDS = ("thread_id", "idx", "role", "content", "ts", "tokens", "cost")
RG_FIELDS = ("id", "user", "ts", "kind", "detail", "action", "reviewed_at")
PASTED_FIELDS = ("id", "user", "thread_id", "created_at", "source", "text", "extraction", "matches")
ACCOUNT_JSON_FIELDS = {"favourite_sports", "settings", "payload", "result", "content", "detail", "extraction",
                       "matches"}
PASTED_SOURCES = ("text", "image", "basic")  # as engine.pasted.SOURCES
MAX_PASTED = 200  # pasted slips kept per person; older ones are deleted
MAX_PASTED_TEXT = 6000  # engine.pasted.MAX_PASTE_CHARS

BUDGET_RAISE_DELAY = 24 * 3600  # seconds: lowering a limit protects at once, raising one waits a day
COOLOFF_HOURS = (24, 24 * 92)   # a time-out lasts from 24 hours to about 3 months
ODDS_FORMATS = ("decimal", "american", "fractional")
RISK_TIERS = ("safe", "risky", "crazy")
REALITY_CHECK_MINUTES = (15, 30, 60)
SLIP_STATUSES = ("draft", "saved", "settled", "void")
TICKET_TYPES = ("single", "multi", "round_robin", "system")  # as engine.tickets.TICKET_TYPES
PLAN_STATUSES = ("active", "done", "failed", "stopped")
PLAN_MODES = ("paper", "tracked")
PLAN_STRATEGIES = ("classic", "bank_half", "ladder", "flat")  # as engine.rollover.STRATEGIES
PLAN_DAYS = (1, 60)            # the rollover planner's limits, so a stored plan can be re-run
PLAN_LEGS_PER_DAY = (1, 3)
PLAN_MIN_ODDS = Decimal("1.2")
MAX_CENTS = 10 ** 11           # $1 billion: far past any real stake, well inside SQLite's 64-bit integers
MAX_SLIP_LEGS = 12             # engine.tickets.MAX_LEGS
SHARE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"  # no 0/O, 1/I/L: easy to read out or type
SHARE_CODE_LENGTH = 8
CREDIT_DAY_TZ = ZoneInfo("America/New_York")  # the config's default; the service can pass its own day
REQUIRED_LEG_FIELDS = ("league", "event_id", "market", "side", "price", "label")
SQL_CHUNK = 500  # ids per "IN (...)" query, well under SQLite's variable limit
# Ids starting with "*" are the app's own rows, never a person's (people are "tg:<id>" or "web:<id>"): the
# AI allowances shared by many people in ai_credits ("*web", "*all"), and in users the gates every password
# browser shares (see shared_user)
RESERVED_PREFIX = "*"
AI_HOLD_SECONDS = 10 * 60  # a running AI call's hold lapses this long after it was taken or last renewed
# (renew_ai, before each request to the model), in case its process died mid-call


def _consensus(values: Sequence[Any]) -> Dict[str, Any]:
    row = dict(zip(CONSENSUS_FIELDS, values))
    for key in CONSENSUS_JSON:
        if row[key] is not None:
            row[key] = json.loads(row[key])
    row["swapped"] = bool(row["swapped"])
    return row


def _record(fields: Tuple[str, ...], values: Sequence[Any]) -> Dict[str, Any]:
    """One version-2 row as a dict, with its JSON columns decoded."""
    row = dict(zip(fields, values))
    for key in ACCOUNT_JSON_FIELDS & row.keys():
        if row[key] is not None:
            row[key] = json.loads(row[key])
    return row


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)  # exact: "1.91", never 1.9099999
    raise TypeError(f"Can't save {type(value).__name__} as JSON")


def _dumps(value: Any) -> Optional[str]:
    return None if value is None else json.dumps(value, default=_json_default)


def _now(now: Optional[float]) -> float:
    return time.time() if now is None else now


def _iso(now: float) -> str:
    """Epoch seconds as the ISO 8601 UTC form game start times use, so the two compare as text."""
    return datetime.datetime.fromtimestamp(now, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _limit(limit: Optional[int]) -> int:
    """A LIMIT for SQLite: None means no limit, which SQLite writes as -1."""
    return -1 if limit is None else limit


def _cents(value: Any, what: str) -> int:
    """A whole, non-negative number of cents (money never goes in as a float). The cap turns an absurd
    amount into a clear ValueError rather than SQLite's OverflowError (a 500 at the API)."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_CENTS:
        raise ValueError(f"{what} must be a whole number of cents from 0 to {MAX_CENTS}; got {value!r}")
    return value


def _count(value: Any, what: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{what} must be a whole number from {low} to {high}; got {value!r}")
    return value


def _finite(value: Any) -> Optional[float]:
    """A real, finite number as a float, or None for anything else (text, True, NaN, infinity, 10**400)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    try:
        number = float(value)  # Decimal("sNaN") and 10**400 raise here; Decimal("NaN") can't even be compared
    except (ValueError, OverflowError, InvalidOperation):
        return None
    return number if math.isfinite(number) else None


def _probability(value: Any, what: str) -> Optional[float]:
    """A chance from 0 to 1, or None. Anything else couldn't be rebuilt into a TicketLeg to re-price later."""
    if value is None:
        return None
    chance = _finite(value)
    if chance is None or not 0 <= chance <= 1:
        raise ValueError(f"{what} must be a chance from 0 to 1; got {value!r}")
    return chance


def _usage(value: Any, what: str) -> float:
    """An amount of 0 or more as a float (AI cost estimates keep fractions of a cent). Infinity would lock
    the person out of the AI for the day, and NaN would fail in SQLite rather than here."""
    amount = _finite(value)
    if amount is None or amount < 0:
        raise ValueError(f"{what} must be a number, 0 or more; got {value!r}")
    return amount


def _start_time(value: Any, what: str, required: bool) -> Optional[str]:
    """A game's start as ISO 8601 UTC in the one form ("2027-01-15T00:30:00Z") that the "has it started"
    checks compare as text: ESPN's "2027-01-15T00:30Z" or an offset would sort wrongly against it."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ValueError(f"{what} needs its game's start time")
        return None
    try:
        moment = datetime.datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=datetime.timezone.utc)
        return moment.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # as _iso
    except (ValueError, OverflowError):
        raise ValueError(f"{what}: start must be an ISO 8601 time; got {value!r}") from None


def _exact_price(value: Any) -> str:
    """Decimal odds as an exact string. A float goes through str() first (1.91 stays "1.91"), like TicketLeg."""
    if isinstance(value, bool):
        raise ValueError(f"Odds must be a number above 1; got {value!r}")
    try:
        price = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except InvalidOperation:
        raise ValueError(f"Odds must be a number above 1; got {value!r}") from None
    if not price.is_finite() or price <= 1:
        raise ValueError(f"Odds must be a number above 1; got {value!r}")
    return str(price)


def _statuses(status: Union[None, str, Iterable[str]]) -> Optional[Tuple[str, ...]]:
    if status is None:
        return None
    return (status,) if isinstance(status, str) else tuple(status)


def _in(column: str, values: Sequence[Any]) -> str:
    return f"{column} IN ({', '.join('?' * len(values))})"


@contextmanager
def _transaction(db: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Take the write lock up front, so a read-then-write (next day number, next message) can't race."""
    db.execute("BEGIN IMMEDIATE")
    try:
        yield db
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")


def _setting(field: str, value: Any) -> Any:
    """Check one setting a person may change themselves and return what to store."""
    if field == "name":
        if value is not None and not isinstance(value, str):
            raise ValueError("name must be text")
        cleaned = value.strip()[:64] if value else ""
        return cleaned or None  # never "": COALESCE keeps "", so a blank name could never be filled later
    if field == "odds_format" and value not in ODDS_FORMATS:
        raise ValueError(f"odds_format must be one of {', '.join(ODDS_FORMATS)}")
    if field == "currency":
        if not (isinstance(value, str) and len(value) == 3 and value.isalpha()):
            raise ValueError("currency must be a 3-letter code such as USD")
        return value.upper()
    if field == "favourite_sports":
        if not (isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value)):
            raise ValueError("favourite_sports must be a list of league keys")
        return json.dumps(list(dict.fromkeys(value)))  # no duplicates, order kept
    if field == "risk_tier" and value is not None and value not in RISK_TIERS:
        raise ValueError(f"risk_tier must be one of {', '.join(RISK_TIERS)}")
    if field == "reality_check_minutes" and value is not None and (
            isinstance(value, bool) or value not in REALITY_CHECK_MINUTES):
        raise ValueError("reality_check_minutes must be 15, 30, 60 or off")
    return value


# What update_user may change. Budget, age and cool-off have their own methods and rules.
EDITABLE_USER_FIELDS = ("name", "odds_format", "currency", "favourite_sports", "risk_tier", "reality_check_minutes")


def _prefixed(column: str, prefix: str) -> Tuple[str, Tuple[str, str]]:
    """A WHERE clause for ids starting with ``prefix`` (as a range, so the column's index serves it)."""
    if not prefix:
        raise ValueError("An id prefix is needed")
    return f"{column} >= ? AND {column} < ?", (prefix, prefix[:-1] + chr(ord(prefix[-1]) + 1))


def _add_usage(db: sqlite3.Connection, user: str, day: str, cents: float) -> None:
    db.execute(
        "INSERT INTO ai_credits (user, day, used_cents) VALUES (?, ?, ?) "
        "ON CONFLICT(user, day) DO UPDATE SET used_cents = ai_credits.used_cents + excluded.used_cents",
        (user, day, cents),
    )


def _apply_due_raise(db: sqlite3.Connection, user_id: str, now: float) -> None:
    """A budget raise whose 24 hours are up becomes the limit in force."""
    db.execute(
        "UPDATE users SET weekly_budget_cents = pending_budget_cents, pending_budget_cents = NULL, "
        "budget_raise_at = NULL WHERE id = ? AND budget_raise_at IS NOT NULL AND budget_raise_at <= ?",
        (user_id, now),
    )


def _leg_values(idx: int, leg: Any, quoted_at: Optional[float], status: str) -> Tuple[Any, ...]:
    """One slip leg (a dict such as a priced ticket's leg, or a TicketLeg) as slip_legs values.

    A saved bet needs every leg's start time: deleting a bet and the grader's queue both turn on
    whether its games have started. A draft may still be missing one.
    """
    data = dataclasses.asdict(leg) if dataclasses.is_dataclass(leg) and not isinstance(leg, type) else dict(leg)
    missing = [field for field in REQUIRED_LEG_FIELDS if data.get(field) in (None, "")]
    if missing:
        raise ValueError(f"Leg {idx + 1} is missing {', '.join(missing)}")
    what = f"Leg {idx + 1}"
    point = data.get("point")
    seen = data.get("quoted_at")
    chances = [_probability(data.get(name), f"{what} {name}") for name in ("ai_prob", "market_prob", "push_prob")]
    return (idx, data.get("key"), data["league"], str(data["event_id"]), data["market"], data["side"],
            None if point is None else float(point), _exact_price(data["price"]), data["label"], data.get("game"),
            _start_time(data.get("start"), what, required=status == "saved"), quoted_at if seen is None else seen,
            *chances, data.get("book"))


def _new_share_code(db: sqlite3.Connection) -> str:
    """A random code nobody can guess from the slip id (runs inside the write lock, so it stays unique)."""
    while True:
        code = "".join(secrets.choice(SHARE_ALPHABET) for _ in range(SHARE_CODE_LENGTH))
        if not db.execute("SELECT 1 FROM slips WHERE share_code = ?", (code,)).fetchone():
            return code


def _with_legs(db: sqlite3.Connection, rows: Sequence[Sequence[Any]]) -> List[Dict[str, Any]]:
    """Slip rows with their legs attached, in a query per few hundred slips rather than one per slip."""
    slips = [_record(SLIP_FIELDS, values) for values in rows]
    legs: Dict[int, List[Dict[str, Any]]] = {slip["id"]: [] for slip in slips}
    ids = list(legs)
    for i in range(0, len(ids), SQL_CHUNK):
        chunk = ids[i:i + SQL_CHUNK]
        for values in db.execute(
            f"SELECT {', '.join(LEG_FIELDS)} FROM slip_legs WHERE {_in('slip_id', chunk)} ORDER BY slip_id, idx", chunk
        ):
            leg = _record(LEG_FIELDS, values)
            legs[leg.pop("slip_id")].append(leg)
    for slip in slips:
        slip["legs"] = legs[slip["id"]]
    return slips


def _with_days(db: sqlite3.Connection, rows: Sequence[Sequence[Any]]) -> List[Dict[str, Any]]:
    """Plan rows with their recorded days attached, oldest day first."""
    plans = [_record(PLAN_FIELDS, values) for values in rows]
    days: Dict[int, List[Dict[str, Any]]] = {plan["id"]: [] for plan in plans}
    ids = list(days)
    for i in range(0, len(ids), SQL_CHUNK):
        chunk = ids[i:i + SQL_CHUNK]
        for values in db.execute(
            f"SELECT {', '.join(PLAN_DAY_FIELDS)} FROM plan_days WHERE {_in('plan_id', chunk)} ORDER BY plan_id, day",
            chunk,
        ):
            day = _record(PLAN_DAY_FIELDS, values)
            days[day["plan_id"]].append(day)
    for plan in plans:
        plan["days_played"] = days[plan["id"]]
    return plans


# The shared cache (data/cache.py's TTLCache): one copy of each fetched or computed value for the bot and the
# dashboard, and short leases so one process loads a key while the other waits for its copy
CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS shared_cache (
    key TEXT PRIMARY KEY,          -- data/cache.py's key_text: stable JSON of the key (never a secret)
    kind TEXT NOT NULL,            -- what it is, for stats ("summary", "player", "markets", ...)
    fetched_at REAL NOT NULL,      -- epoch seconds the value was fetched or computed
    expires_at REAL NOT NULL,      -- fetched_at + its TTL
    stale_until REAL NOT NULL,     -- expires_at + its stale window; the row is pruned a day after this
    value BLOB NOT NULL            -- gzip-compressed JSON
);
CREATE INDEX IF NOT EXISTS shared_cache_by_stale ON shared_cache (stale_until);
CREATE TABLE IF NOT EXISTS cache_leases (
    key TEXT PRIMARY KEY,
    holder TEXT NOT NULL,          -- the cache (process) loading the key
    until REAL NOT NULL            -- epoch seconds; a lapsed lease is free
);
"""

# v3 Stage 4, context (data/context.py): every forecast the context worker read for a game, with its fetch time, so
# the forecast as it stood when the odds were fetched can be found later; and the totals' chances before and after an
# adjustment, frozen at the start and graded after the final (Open-Meteo's daily call counter is data/weather.py's)
CONTEXT_SCHEMA = """
CREATE TABLE IF NOT EXISTS weather_log (
    league TEXT NOT NULL,
    game_key TEXT NOT NULL,        -- the context entry's game id: the odds feed's id, else "espn:<id>"
    fetched_at REAL NOT NULL,      -- when the forecast (or MLB's announcement) was fetched
    start TEXT NOT NULL,           -- the game's ISO start
    venue TEXT,                    -- the venue row's id
    source TEXT NOT NULL,          -- "Open-Meteo" or "MLB"
    effect REAL,                   -- the weather's effect on the total then, in the sport's unit (0 under a roof)
    conditions TEXT NOT NULL,      -- JSON: the hour at the start (with MLB's announcement), as the engine reads it
    sig TEXT NOT NULL,             -- a hash of the conditions: a row is added only when they change
    PRIMARY KEY (league, game_key, fetched_at)
);
CREATE TABLE IF NOT EXISTS context_log (
    league TEXT NOT NULL,
    game_key TEXT NOT NULL,        -- "espn:<id>" when ESPN shows the game, else "odds:<id>": one row per game
    event_id TEXT,                 -- the odds feed's game id
    espn_id TEXT,
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    start TEXT NOT NULL,
    logged_at REAL NOT NULL,       -- the last refresh before the start: frozen after it
    line REAL NOT NULL,            -- the total line
    p_before REAL NOT NULL,        -- the chance of the over before any adjustment (the books' margin-free chance)
    p_after REAL NOT NULL,         -- after it (the same when nothing was adjusted)
    applied REAL NOT NULL DEFAULT 0,  -- the adjustment applied to the expected total, in the sport's unit
    reason TEXT,
    adjustment TEXT,               -- JSON: the Adjustment, or null
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, graded or void
    outcome TEXT,                  -- over, under or push
    result TEXT,                   -- JSON: the final total and each chance's Brier score
    graded_at REAL,
    PRIMARY KEY (league, game_key)
);
CREATE INDEX IF NOT EXISTS context_log_by_status ON context_log (status, start);
"""

WEATHER_LOG_FIELDS = ("league", "game_key", "fetched_at", "start", "venue", "source", "effect", "conditions", "sig")
CONTEXT_LOG_FIELDS = ("league", "game_key", "event_id", "espn_id", "home", "away", "start", "logged_at", "line",
                      "p_before", "p_after", "applied", "reason", "adjustment", "status", "outcome", "result",
                      "graded_at")
CONTEXT_LOG_JSON = {"adjustment", "result"}


def _context_row(values: Sequence[Any]) -> Dict[str, Any]:
    row = dict(zip(CONTEXT_LOG_FIELDS, values))
    for key in CONTEXT_LOG_JSON:
        if row[key] is not None:
            row[key] = json.loads(row[key])
    return row


# v3 learning loop (telegram_betting_bot_simple/learn/): columns added to tables that already exist on a server, so
# an older betting.db is brought up to date on the next start (ALTER TABLE ... ADD COLUMN, once, under the write lock;
# a new file gets them the same way). Every existing row reads as the defaults: blended with version 0 (today's
# constants), and the last-before-start row (kind "last"; "t24" is the one-day-before snapshot of the same game).
MIGRATIONS: Tuple[Tuple[str, str, str], ...] = (
    ("consensus_log", "params_version", "INTEGER NOT NULL DEFAULT 0"),  # learn/registry.py version it was blended with
    ("consensus_log", "kind", "TEXT NOT NULL DEFAULT 'last'"),          # last (before the start) or t24
    ("context_log", "params_version", "INTEGER NOT NULL DEFAULT 0"),    # the version it was adjusted with
)
SNAPSHOT_SUFFIX = "#t24"  # a t24 row's game_key: the game's own key plus this
T24_WINDOW = (12 * 3600.0, 24 * 3600.0)  # the snapshot is taken on the first log this long before the start


def migrate(db: sqlite3.Connection) -> List[str]:
    """Add MIGRATIONS' missing columns (idempotent; safe when the bot and the dashboard start together: the check and
    the change run under one write lock, and a column another process added first is taken as done). Returns
    "table.column" for each column added."""
    added: List[str] = []
    db.execute("BEGIN IMMEDIATE")
    try:
        for table, column, declaration in MIGRATIONS:
            present = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
            if not present or column in present:
                continue
            try:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
                added.append(f"{table}.{column}")
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")
    return added


def _weather_row(values: Sequence[Any]) -> Dict[str, Any]:
    row = dict(zip(WEATHER_LOG_FIELDS, values))
    row["conditions"] = json.loads(row["conditions"]) if row["conditions"] else None
    return row


class _ThreadDb:
    """A thread's kept-open cache connection (Store._cache_db), closed when the thread's locals or the Store go."""

    __slots__ = ("db",)

    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:  # closing is best effort
            pass

    def __del__(self) -> None:
        self.close()


class Store:
    def __init__(self, path: str) -> None:
        self.path = path
        self.holder = f"{os.getpid()}"
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            db.executescript(ACCOUNT_SCHEMA)
            db.executescript(CACHE_SCHEMA)
            db.executescript(CONTEXT_SCHEMA)
            # v3 learning loop: the history tables (learn/history_store.py), the brain's versions, trials and scores
            # (learn/brain_store.py), and the columns the older tables gain
            db.executescript(HIST_SCHEMA)
            db.executescript(BRAIN_SCHEMA)
            migrate(db)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10, isolation_level=None)

    def odds_snapshot(self, sport_key: str) -> Optional[Tuple[float, List[Dict[str, Any]]]]:
        """(fetched_at epoch seconds, games) of the league's last fetch, or None."""
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT fetched_at, payload FROM odds_snapshots WHERE sport_key = ?", (sport_key,)
            ).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def odds_fetched_at(self) -> Dict[str, float]:
        """When each league's saved odds were fetched (epoch seconds), without reading the odds themselves."""
        with closing(self._connect()) as db:
            return {key: float(fetched) for key, fetched in db.execute("SELECT sport_key, fetched_at FROM odds_snapshots")}

    def save_odds_snapshot(self, sport_key: str, games: List[Dict[str, Any]], fetched_at: float) -> None:
        with closing(self._connect()) as db:
            db.execute(
                "INSERT INTO odds_snapshots (sport_key, fetched_at, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(sport_key) DO UPDATE SET fetched_at = excluded.fetched_at, payload = excluded.payload",
                (sport_key, fetched_at, json.dumps(games)),
            )

    def save_slate(self, name: str, body: bytes, etag: str, generated_at: float) -> None:
        """Keep the latest slate's bytes (data/slate.py), replacing the one before."""
        with closing(self._connect()) as db:
            db.execute(
                "INSERT INTO slates (name, generated_at, etag, body) VALUES (?, ?, ?, ?) ON CONFLICT(name) DO UPDATE "
                "SET generated_at = excluded.generated_at, etag = excluded.etag, body = excluded.body",
                (name, generated_at, etag, sqlite3.Binary(body)),
            )

    def slate(self, name: str) -> Optional[Tuple[float, str, bytes]]:
        """(generated_at, etag, body) of the saved slate, or None."""
        with closing(self._connect()) as db:
            row = db.execute("SELECT generated_at, etag, body FROM slates WHERE name = ?", (name,)).fetchone()
        return (float(row[0]), str(row[1]), bytes(row[2])) if row else None

    def record_usage(self, endpoint: str, cost: Optional[int], remaining: Optional[int], used: Optional[int]) -> None:
        with closing(self._connect()) as db:
            db.execute(
                "INSERT INTO api_usage (ts, endpoint, cost, remaining, used) VALUES (?, ?, ?, ?, ?)",
                (time.time(), endpoint, cost, remaining, used),
            )

    def latest_quota(self) -> Dict[str, Optional[int]]:
        """The most recent remaining/used counts either process saw."""
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT remaining, used FROM api_usage WHERE remaining IS NOT NULL ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        return {"remaining": row[0], "used": row[1]} if row else {"remaining": None, "used": None}

    def try_lease(self, key: str, seconds: float) -> bool:
        """Claim ``key`` for ``seconds`` unless another process holds an unexpired claim."""
        now = time.time()
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT holder, expires FROM leases WHERE key = ?", (key,)).fetchone()
            if row and row[0] != self.holder and row[1] > now:
                db.execute("ROLLBACK")
                return False
            db.execute(
                "INSERT INTO leases (key, holder, expires) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET holder = excluded.holder, expires = excluded.expires",
                (key, self.holder, now + seconds),
            )
            db.execute("COMMIT")
            return True

    # --- the shared cache (data/cache.py): values as gzip JSON, keyed by key_text --------------------------

    @contextmanager
    def _cache_db(self) -> Iterator[sqlite3.Connection]:
        """This thread's own connection for the shared cache's small reads and writes, kept open: a connection per
        call costs ~2 ms in opening, syncing and checkpointing the WAL on close, about 7 ms per cache miss (two
        reads, a lease, the value, the release); kept open, a write is ~0.05 ms. ``synchronous=NORMAL`` on this
        connection only (safe with WAL: a power cut may lose the last cache writes, never corrupts the file; a cache
        row or a lease is cheap to lose). A connection that errors is dropped and the next call opens a new one; one
        whose thread has ended is closed when the thread's locals go."""
        local = self.__dict__.setdefault("_cache_local", threading.local())
        held = getattr(local, "held", None)
        if held is None:
            db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("PRAGMA cache_size=-256")  # 256 KiB of pages: one connection per thread adds up
            held = local.held = _ThreadDb(db)
        try:
            yield held.db
        except sqlite3.Error:
            local.held = None
            held.close()
            raise

    def cache_get(self, key: str, newer_than: Optional[float] = None) -> Optional[Tuple[float, float, float, bytes]]:
        """(fetched_at, expires_at, stale_until, gzip JSON) of the shared copy of ``key``, or None; with
        ``newer_than``, None unless the copy was fetched after that (its value isn't read otherwise)."""
        with self._cache_db() as db:
            if newer_than is None:
                row = db.execute("SELECT fetched_at, expires_at, stale_until, value FROM shared_cache WHERE key = ?",
                                 (key,)).fetchone()
            else:
                row = db.execute("SELECT fetched_at, expires_at, stale_until, value FROM shared_cache "
                                 "WHERE key = ? AND fetched_at > ?", (key, float(newer_than))).fetchone()
        return (float(row[0]), float(row[1]), float(row[2]), bytes(row[3])) if row else None

    def cache_put(self, key: str, kind: str, value: bytes, fetched_at: float, expires_at: float,
                  stale_until: float) -> bool:
        """Save the shared copy of ``key`` unless the table holds a newer one; True when written."""
        with self._cache_db() as db:
            cursor = db.execute(
                "INSERT INTO shared_cache (key, kind, fetched_at, expires_at, stale_until, value) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET kind = excluded.kind, "
                "fetched_at = excluded.fetched_at, expires_at = excluded.expires_at, "
                "stale_until = excluded.stale_until, value = excluded.value "
                "WHERE excluded.fetched_at >= shared_cache.fetched_at",
                (key, kind, fetched_at, expires_at, stale_until, sqlite3.Binary(value)),
            )
        return cursor.rowcount > 0

    def cache_delete(self, key: str, upto: Optional[float] = None) -> int:
        """Delete the shared copy of ``key`` (with ``upto``, only if it was fetched no later than that)."""
        with self._cache_db() as db:
            if upto is None:
                return db.execute("DELETE FROM shared_cache WHERE key = ?", (key,)).rowcount
            return db.execute("DELETE FROM shared_cache WHERE key = ? AND fetched_at <= ?", (key, upto)).rowcount

    def cache_delete_prefix(self, prefix: str) -> int:
        """Delete every key that is the JSON array ``prefix`` + "]" or continues it (``prefix`` + ","): the rows
        of the tuple keys starting with a tuple (data/cache.py's invalidate_prefix)."""
        with self._cache_db() as db:
            # "," is 0x2C and "-" 0x2D, so the range is exactly the keys starting with prefix + ","
            return db.execute("DELETE FROM shared_cache WHERE key = ? OR (key >= ? AND key < ?)",
                              (prefix + "]", prefix + ",", prefix + "-")).rowcount

    def cache_lease(self, key: str, holder: str, seconds: float, now: Optional[float] = None) -> bool:
        """Claim ``key`` for ``holder`` for ``seconds`` unless another holder's claim is still running (a holder
        renews its own)."""
        now = _now(now)
        with self._cache_db() as db:
            cursor = db.execute(
                "INSERT INTO cache_leases (key, holder, until) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE "
                "SET holder = excluded.holder, until = excluded.until "
                "WHERE cache_leases.holder = excluded.holder OR cache_leases.until <= ?",
                (key, holder, now + seconds, now),
            )
        return cursor.rowcount > 0

    def cache_release(self, key: str, holder: str) -> None:
        """Drop ``holder``'s claim on ``key`` (another holder's is left alone)."""
        with self._cache_db() as db:
            db.execute("DELETE FROM cache_leases WHERE key = ? AND holder = ?", (key, holder))

    def cache_prune(self, now: Optional[float] = None, keep: float = 86400.0) -> int:
        """Delete shared copies more than ``keep`` seconds past their stale_until, and lapsed leases; returns the
        copies deleted."""
        now = _now(now)
        with self._cache_db() as db:
            deleted = db.execute("DELETE FROM shared_cache WHERE stale_until < ?", (now - keep,)).rowcount
            db.execute("DELETE FROM cache_leases WHERE until < ?", (now,))
        return deleted

    def cache_summary(self) -> Dict[str, Any]:
        """{"rows", "bytes" (compressed), "kinds": {kind: rows}} of the shared cache table."""
        with self._cache_db() as db:
            kinds = {str(kind): int(count) for kind, count in
                     db.execute("SELECT kind, COUNT(*) FROM shared_cache GROUP BY kind ORDER BY kind")}
            size = db.execute("SELECT COALESCE(SUM(LENGTH(value)), 0) FROM shared_cache").fetchone()[0]
        return {"rows": sum(kinds.values()), "bytes": int(size or 0), "kinds": kinds}

    # --- people's predictions ---------------------------------------------------

    def save_prediction(self, row: Dict[str, Any]) -> int:
        """Insert a call, or replace the player's earlier call on the same game; returns its id."""
        values = {**row, "call": json.dumps(row["call"]), "ai": json.dumps(row["ai"]), "created_at": time.time()}
        fields = ("player", "name", "league", "event_id", "espn_id", "home", "away", "start", "created_at", "call", "ai")
        with closing(self._connect()) as db:
            db.execute(
                f"INSERT INTO predictions ({', '.join(fields)}) VALUES ({', '.join('?' * len(fields))}) "
                "ON CONFLICT(player, league, event_id) DO UPDATE SET name = excluded.name, "
                "espn_id = excluded.espn_id, created_at = excluded.created_at, call = excluded.call, ai = excluded.ai",
                tuple(values.get(field) for field in fields),
            )
            row_id = db.execute(
                "SELECT id FROM predictions WHERE player = ? AND league = ? AND event_id = ?",
                (row["player"], row["league"], row["event_id"]),
            ).fetchone()[0]
        return row_id

    def delete_prediction(self, player: str, prediction_id: int) -> bool:
        """Remove a pending call (only its owner can)."""
        with closing(self._connect()) as db:
            cursor = db.execute(
                "DELETE FROM predictions WHERE id = ? AND player = ? AND status = 'pending'", (prediction_id, player)
            )
        return cursor.rowcount > 0

    def predictions(self, player: Optional[str] = None, status: Optional[str] = None,
                    started_before: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        """Calls, newest game first, optionally one player's, by status, or for games started before a time."""
        where, args = [], []
        for clause, value in (("player = ?", player), ("status = ?", status), ("start < ?", started_before)):
            if value is not None:
                where.append(clause)
                args.append(value)
        sql = f"SELECT {', '.join(PREDICTION_FIELDS)} FROM predictions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with closing(self._connect()) as db:
            rows = db.execute(sql + " ORDER BY start DESC, id DESC LIMIT ?", (*args, limit)).fetchall()
        return [_row(PREDICTION_FIELDS, values) for values in rows]

    def grade_prediction(self, prediction_id: int, status: str, result: Dict[str, Any],
                         user_points: Optional[float], ai_points: Optional[float]) -> None:
        with closing(self._connect()) as db:
            db.execute(
                "UPDATE predictions SET status = ?, result = ?, user_points = ?, ai_points = ?, graded_at = ? WHERE id = ?",
                (status, json.dumps(result), user_points, ai_points, time.time(), prediction_id),
            )

    def leaderboard(self) -> List[Dict[str, Any]]:
        """Each player's graded calls and points, "vs AI" (their points minus the AI's on the
        same calls) and "vs market" (their winner points minus the bookmakers')."""
        players: Dict[str, Dict[str, Any]] = {}
        for row in self.predictions(status="graded", limit=1_000_000):
            entry = players.setdefault(row["player"], {
                "player": row["player"], "name": row["name"], "calls": 0, "points": 0.0, "ai_points": 0.0,
                "vs_market": 0.0, "market_calls": 0, "beat_ai": 0,
            })
            entry["calls"] += 1
            entry["points"] += row["user_points"] or 0.0
            entry["ai_points"] += row["ai_points"] or 0.0
            entry["beat_ai"] += (row["user_points"] or 0.0) > (row["ai_points"] or 0.0)
            winner = next((p for p in (row["result"] or {}).get("parts", []) if p["part"] == "winner"), None)
            if winner and winner.get("market_points") is not None:
                entry["vs_market"] += winner["points"] - winner["market_points"]
                entry["market_calls"] += 1
        board = sorted(players.values(), key=lambda e: e["points"], reverse=True)
        for entry in board:
            entry["vs_ai"] = round(entry["points"] - entry["ai_points"], 1)
            entry["points"], entry["ai_points"] = round(entry["points"], 1), round(entry["ai_points"], 1)
            entry["vs_market"] = round(entry["vs_market"], 1)
        return board

    # --- the AI's own predictions -------------------------------------------------

    def log_ai_prediction(self, row: Dict[str, Any]) -> None:
        """Save the AI's latest pre-game prediction for a game (kept until it's graded)."""
        with closing(self._connect()) as db:
            db.execute(
                "INSERT INTO ai_predictions (league, event_id, espn_id, home, away, start, logged_at, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(league, event_id) DO UPDATE SET "
                "espn_id = COALESCE(excluded.espn_id, ai_predictions.espn_id), start = excluded.start, "
                "logged_at = excluded.logged_at, payload = excluded.payload WHERE ai_predictions.status = 'pending'",
                (row["league"], row["event_id"], row.get("espn_id"), row["home"], row["away"], row["start"],
                 time.time(), json.dumps(row["payload"])),
            )

    def ai_predictions(self, league: Optional[str] = None, status: Optional[str] = None,
                       started_before: Optional[str] = None, limit: int = 2000) -> List[Dict[str, Any]]:
        where, args = [], []
        for clause, value in (("league = ?", league), ("status = ?", status), ("start < ?", started_before)):
            if value is not None:
                where.append(clause)
                args.append(value)
        sql = f"SELECT {', '.join(AI_FIELDS)} FROM ai_predictions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with closing(self._connect()) as db:
            rows = db.execute(sql + " ORDER BY start DESC LIMIT ?", (*args, limit)).fetchall()
        return [_row(AI_FIELDS, values) for values in rows]

    # --- the AI's own bets, one per tier per day, for an honest per-tier record ---------------

    def log_ai_ticket(self, day: str, league: str, tier: str, first_start: str, last_start: str,
                      payload: Dict[str, Any], now_iso: str) -> None:
        """Save (or update) a tier's ticket for the day, until its first game has started."""
        with closing(self._connect()) as db:
            db.execute(
                "INSERT INTO ai_tickets (day, league, tier, logged_at, first_start, last_start, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(day, league, tier) DO UPDATE SET "
                "logged_at = excluded.logged_at, first_start = excluded.first_start, "
                "last_start = excluded.last_start, payload = excluded.payload "
                "WHERE ai_tickets.status = 'pending' AND ai_tickets.first_start > ?",
                (day, league, tier, time.time(), first_start, last_start, json.dumps(payload), now_iso),
            )

    def ai_tickets(self, league: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        where, args = [], []
        for clause, value in (("league = ?", league), ("status = ?", status)):
            if value is not None:
                where.append(clause)
                args.append(value)
        sql = f"SELECT {', '.join(TICKET_FIELDS)} FROM ai_tickets"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with closing(self._connect()) as db:
            rows = db.execute(sql + " ORDER BY day DESC", tuple(args)).fetchall()
        return [_row(TICKET_FIELDS, values) for values in rows]

    def grade_ai_ticket(self, day: str, league: str, tier: str, status: str, result: Dict[str, Any]) -> None:
        with closing(self._connect()) as db:
            db.execute(
                "UPDATE ai_tickets SET status = ?, result = ?, graded_at = ? WHERE day = ? AND league = ? AND tier = ?",
                (status, json.dumps(result), time.time(), day, league, tier),
            )

    def ai_prediction(self, league: str, event_id: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as db:
            values = db.execute(
                f"SELECT {', '.join(AI_FIELDS)} FROM ai_predictions WHERE league = ? AND event_id = ?", (league, event_id)
            ).fetchone()
        return _row(AI_FIELDS, values) if values else None

    def grade_ai_prediction(self, league: str, event_id: str, status: str, result: Dict[str, Any]) -> None:
        with closing(self._connect()) as db:
            db.execute(
                "UPDATE ai_predictions SET status = ?, result = ?, graded_at = ? WHERE league = ? AND event_id = ?",
                (status, json.dumps(result), time.time(), league, event_id),
            )

    # --- the markets brain's consensus, logged before each start and graded after the final ----------------

    def log_consensus(self, rows: Sequence[Mapping[str, Any]], now: Optional[float] = None) -> int:
        """Save each game's latest sources and blend (one transaction): a row is updated on every refresh until
        its game starts, then frozen; rows for games already started are ignored. Returns the rows written.

        ``params_version`` (learn/registry.py's version the blend was made with, 0 for the defaults) is kept with
        the row. A row with ``kind`` "t24" is the game's one-day-before snapshot: written once under the game's key
        plus SNAPSHOT_SUFFIX and never updated (the learning loop scores the brain by horizon with it)."""
        now = _now(now)
        now_iso = _iso(now)
        written = 0
        with closing(self._connect()) as db, _transaction(db):
            for row in rows:
                start = str(row["start"])
                if start <= now_iso:
                    continue
                version = int(row.get("params_version") or 0)
                if row.get("kind") == "t24":
                    key = str(row["game_key"])
                    key = key if key.endswith(SNAPSHOT_SUFFIX) else key + SNAPSHOT_SUFFIX
                    cursor = db.execute(
                        "INSERT INTO consensus_log (league, game_key, event_id, espn_id, swapped, home, away, start, "
                        "logged_at, sources, blend, params_version, kind) VALUES "
                        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 't24') ON CONFLICT(league, game_key) DO NOTHING",
                        (row["league"], key, row.get("event_id"), row.get("espn_id"), 1 if row.get("swapped") else 0,
                         row["home"], row["away"], start, now, json.dumps(row["sources"], default=_json_default),
                         json.dumps(row["blend"], default=_json_default), version),
                    )
                    written += cursor.rowcount
                    continue
                cursor = db.execute(
                    "INSERT INTO consensus_log (league, game_key, event_id, espn_id, swapped, home, away, start, "
                    "logged_at, sources, blend, params_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(league, game_key) DO UPDATE SET event_id = COALESCE(excluded.event_id, "
                    "consensus_log.event_id), espn_id = COALESCE(excluded.espn_id, consensus_log.espn_id), "
                    "swapped = excluded.swapped, home = excluded.home, away = excluded.away, start = excluded.start, "
                    "logged_at = excluded.logged_at, sources = excluded.sources, blend = excluded.blend, "
                    "params_version = excluded.params_version "
                    "WHERE consensus_log.status = 'pending' AND consensus_log.start > ?",
                    (row["league"], row["game_key"], row.get("event_id"), row.get("espn_id"),
                     1 if row.get("swapped") else 0, row["home"], row["away"], start, now,
                     json.dumps(row["sources"], default=_json_default), json.dumps(row["blend"], default=_json_default),
                     version, now_iso),
                )
                written += cursor.rowcount
        return written

    def consensus_rows(self, league: Optional[str] = None, status: Optional[str] = None,
                       started_before: Optional[str] = None, limit: int = 5000,
                       snapshots: bool = False) -> List[Dict[str, Any]]:
        """Logged consensus rows, latest start first: each game's last-before-start row (with ``snapshots`` also the
        one-day-before rows, which carry "kind": "t24" and "params_version"; the grader scores those too)."""
        where, args = [] if snapshots else ["kind = 'last'"], []
        for clause, value in (("league = ?", league), ("status = ?", status), ("start < ?", started_before)):
            if value is not None:
                where.append(clause)
                args.append(value)
        fields = CONSENSUS_FIELDS + (("kind", "params_version") if snapshots else ())
        sql = f"SELECT {', '.join(fields)} FROM consensus_log"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with closing(self._connect()) as db:
            rows = db.execute(sql + " ORDER BY start DESC LIMIT ?", (*args, _limit(limit))).fetchall()
        if not snapshots:
            return [_consensus(values) for values in rows]
        out = []
        for values in rows:
            row = _consensus(values[:len(CONSENSUS_FIELDS)])
            row["kind"], row["params_version"] = values[len(CONSENSUS_FIELDS)], int(values[-1] or 0)
            out.append(row)
        return out

    def consensus_graded(self) -> Tuple[Tuple[int, Optional[float]], List[Dict[str, Any]]]:
        """Every graded row's league, blend and result (not its sources: the record needs only the scores), with a
        mark that changes whenever a row is graded ((rows, latest graded_at)), so a reader can keep what it worked
        out until the mark moves. No limit: the record counts every graded game."""
        with closing(self._connect()) as db:
            mark = db.execute("SELECT COUNT(*), MAX(graded_at) FROM consensus_log WHERE status = 'graded' "
                              "AND kind = 'last'").fetchone()
            rows = db.execute("SELECT league, blend, result FROM consensus_log WHERE status = 'graded' "
                              "AND kind = 'last' ORDER BY start DESC").fetchall()
        return ((int(mark[0]), mark[1]),
                [{"league": league, "blend": json.loads(blend) if blend else None,
                  "result": json.loads(result) if result else None} for league, blend, result in rows])

    def consensus_graded_mark(self) -> Tuple[int, Optional[float]]:
        """(graded rows, latest graded_at): changes whenever a row is graded."""
        with closing(self._connect()) as db:
            count, latest = db.execute("SELECT COUNT(*), MAX(graded_at) FROM consensus_log WHERE status = 'graded' "
                                       "AND kind = 'last'").fetchone()
        return int(count), latest

    def consensus_row(self, league: str, event_id: Optional[str] = None, espn_id: Optional[str] = None,
                      game_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """One game's logged row, by its key, the odds feed's id or ESPN's id (the most recently logged match)."""
        for column, value in (("game_key", game_key), ("event_id", event_id), ("espn_id", espn_id)):
            if not value:
                continue
            with closing(self._connect()) as db:
                values = db.execute(
                    f"SELECT {', '.join(CONSENSUS_FIELDS)} FROM consensus_log WHERE league = ? AND {column} = ? "
                    "AND kind = 'last' ORDER BY logged_at DESC LIMIT 1", (league, str(value))).fetchone()
            if values:
                return _consensus(values)
        return None

    def grade_consensus(self, league: str, game_key: str, status: str, outcome: Optional[str],
                        result: Mapping[str, Any], now: Optional[float] = None) -> bool:
        """Record a game's outcome and every source's scores (status graded or void); a row is graded once."""
        if status not in ("graded", "void"):
            raise ValueError(f"Unknown consensus status {status!r}")
        with closing(self._connect()) as db:
            cursor = db.execute(
                "UPDATE consensus_log SET status = ?, outcome = ?, result = ?, graded_at = ? "
                "WHERE league = ? AND game_key = ? AND status = 'pending'",
                (status, outcome, json.dumps(result, default=_json_default), _now(now), league, game_key),
            )
        return cursor.rowcount > 0

    # --- context (v3 Stage 4): forecasts as they stood, the totals' chances, daily call counters ----------------

    def log_weather(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """Add forecast rows ({league, game_key, fetched_at, start, venue, source, effect, conditions, sig}): a row is
        written only when its conditions differ from the game's latest row (by ``sig``), so a forecast read again
        from the cache adds nothing. Returns the rows written."""
        written = 0
        with closing(self._connect()) as db, _transaction(db):
            for row in rows:
                latest = db.execute("SELECT sig, fetched_at FROM weather_log WHERE league = ? AND game_key = ? "
                                    "ORDER BY fetched_at DESC LIMIT 1", (row["league"], str(row["game_key"]))).fetchone()
                if latest is not None and (latest[0] == row["sig"] or latest[1] >= float(row["fetched_at"])):
                    continue
                db.execute(
                    f"INSERT OR IGNORE INTO weather_log ({', '.join(WEATHER_LOG_FIELDS)}) VALUES "
                    f"({', '.join('?' * len(WEATHER_LOG_FIELDS))})",
                    (row["league"], str(row["game_key"]), float(row["fetched_at"]), str(row["start"]),
                     row.get("venue"), str(row.get("source") or "Open-Meteo"),
                     None if row.get("effect") is None else float(row["effect"]),
                     json.dumps(row.get("conditions") or {}, default=_json_default), str(row["sig"])))
                written += 1
        return written

    def weather_rows(self, league: str, game_key: str, limit: int = 50) -> List[Dict[str, Any]]:
        """A game's logged forecasts, newest first."""
        with closing(self._connect()) as db:
            rows = db.execute(f"SELECT {', '.join(WEATHER_LOG_FIELDS)} FROM weather_log WHERE league = ? AND "
                              "game_key = ? ORDER BY fetched_at DESC LIMIT ?", (league, str(game_key), _limit(limit))
                              ).fetchall()
        return [_weather_row(values) for values in rows]

    def weather_at(self, league: str, game_key: str, at: float) -> Optional[Dict[str, Any]]:
        """The game's forecast as it stood at ``at``: the latest row fetched at or before it, or None."""
        with closing(self._connect()) as db:
            values = db.execute(f"SELECT {', '.join(WEATHER_LOG_FIELDS)} FROM weather_log WHERE league = ? AND "
                                "game_key = ? AND fetched_at <= ? ORDER BY fetched_at DESC LIMIT 1",
                                (league, str(game_key), float(at))).fetchone()
        return _weather_row(values) if values else None

    def prune_weather(self, before: float) -> int:
        """Drop forecast rows of games that started before ``before`` (epoch seconds)."""
        with closing(self._connect()) as db:
            cursor = db.execute("DELETE FROM weather_log WHERE start < ?", (_iso(before),))
        return cursor.rowcount

    def log_context(self, rows: Sequence[Mapping[str, Any]], now: Optional[float] = None) -> int:
        """Save each game's total line and chances of the over before and after any adjustment (one transaction):
        updated on every refresh until the game starts, then frozen; rows for games already started are ignored.
        ``params_version`` is learn/registry.py's version the adjustment was made with (0: the defaults)."""
        now = _now(now)
        now_iso = _iso(now)
        written = 0
        with closing(self._connect()) as db, _transaction(db):
            for row in rows:
                start = str(row["start"])
                if start <= now_iso:
                    continue
                cursor = db.execute(
                    "INSERT INTO context_log (league, game_key, event_id, espn_id, home, away, start, logged_at, line, "
                    "p_before, p_after, applied, reason, adjustment, params_version) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(league, game_key) DO UPDATE SET event_id = COALESCE(excluded.event_id, "
                    "context_log.event_id), espn_id = COALESCE(excluded.espn_id, context_log.espn_id), "
                    "home = excluded.home, away = excluded.away, start = excluded.start, logged_at = excluded.logged_at, "
                    "line = excluded.line, p_before = excluded.p_before, p_after = excluded.p_after, "
                    "applied = excluded.applied, reason = excluded.reason, adjustment = excluded.adjustment, "
                    "params_version = excluded.params_version "
                    "WHERE context_log.status = 'pending' AND context_log.start > ?",
                    (row["league"], row["game_key"], row.get("event_id"), row.get("espn_id"), row["home"], row["away"],
                     start, now, float(row["line"]), float(row["p_before"]), float(row["p_after"]),
                     float(row.get("applied") or 0.0), row.get("reason"), _dumps(row.get("adjustment")),
                     int(row.get("params_version") or 0), now_iso),
                )
                written += cursor.rowcount
        return written

    def context_rows(self, league: Optional[str] = None, status: Optional[str] = None,
                     started_before: Optional[str] = None, limit: Optional[int] = 5000) -> List[Dict[str, Any]]:
        """Logged context rows, latest start first."""
        where, args = [], []
        for clause, value in (("league = ?", league), ("status = ?", status), ("start < ?", started_before)):
            if value is not None:
                where.append(clause)
                args.append(value)
        sql = f"SELECT {', '.join(CONTEXT_LOG_FIELDS)} FROM context_log"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with closing(self._connect()) as db:
            rows = db.execute(sql + " ORDER BY start DESC LIMIT ?", (*args, _limit(limit))).fetchall()
        return [_context_row(values) for values in rows]

    def context_row(self, league: str, game_key: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as db:
            values = db.execute(f"SELECT {', '.join(CONTEXT_LOG_FIELDS)} FROM context_log WHERE league = ? AND "
                                "game_key = ?", (league, game_key)).fetchone()
        return _context_row(values) if values else None

    def grade_context(self, league: str, game_key: str, status: str, outcome: Optional[str],
                      result: Mapping[str, Any], now: Optional[float] = None) -> bool:
        """Record a game's total and each chance's Brier score (status graded or void); a row is graded once."""
        if status not in ("graded", "void"):
            raise ValueError(f"Unknown context status {status!r}")
        with closing(self._connect()) as db:
            cursor = db.execute(
                "UPDATE context_log SET status = ?, outcome = ?, result = ?, graded_at = ? "
                "WHERE league = ? AND game_key = ? AND status = 'pending'",
                (status, outcome, json.dumps(result, default=_json_default), _now(now), league, game_key),
            )
        return cursor.rowcount > 0

    def context_record(self) -> Dict[str, Any]:
        """{games, graded, brier_before, brier_after, adjusted_games}: every logged game with a total line, the graded
        ones, their mean Brier scores before and after the adjustment, and how many had one applied (with
        ``adjusted``: the same scores over those games only)."""
        with closing(self._connect()) as db:
            games = db.execute("SELECT COUNT(*) FROM context_log").fetchone()[0]
            adjusted_games = db.execute("SELECT COUNT(*) FROM context_log WHERE applied != 0").fetchone()[0]
            rows = db.execute("SELECT applied, result FROM context_log WHERE status = 'graded'").fetchall()
        scores = [(applied, json.loads(result)) for applied, result in rows if result]
        scores = [(applied, r) for applied, r in scores
                  if isinstance(r.get("brier_before"), (int, float)) and isinstance(r.get("brier_after"), (int, float))]

        def mean(values: List[float]) -> Optional[float]:
            return round(sum(values) / len(values), 4) if values else None

        moved = [r for applied, r in scores if applied]
        return {"games": int(games), "graded": len(scores),
                "brier_before": mean([r["brier_before"] for _, r in scores]),
                "brier_after": mean([r["brier_after"] for _, r in scores]),
                "adjusted_games": int(adjusted_games),
                "adjusted": {"graded": len(moved), "brier_before": mean([r["brier_before"] for r in moved]),
                             "brier_after": mean([r["brier_after"] for r in moved])}}

    # --- the AI analysts' forecasts and what they cost ---------------------------------

    def save_analysis(self, row: Dict[str, Any]) -> None:
        """Save an analyst's latest forecast for a game (replacing its earlier one until the game is graded)."""
        with closing(self._connect()) as db:
            db.execute(
                "INSERT INTO analyses (analyst, league, event_id, espn_id, home, away, start, created_at, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(analyst, league, event_id) DO UPDATE SET "
                "espn_id = COALESCE(excluded.espn_id, analyses.espn_id), start = excluded.start, "
                "created_at = excluded.created_at, payload = excluded.payload WHERE analyses.status = 'pending'",
                (row["analyst"], row["league"], row["event_id"], row.get("espn_id"), row["home"], row["away"],
                 row["start"], time.time(), json.dumps(row["payload"])),
            )

    def analysis(self, analyst: str, league: str, event_id: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as db:
            values = db.execute(
                f"SELECT {', '.join(ANALYSIS_FIELDS)} FROM analyses WHERE analyst = ? AND league = ? AND event_id = ?",
                (analyst, league, event_id),
            ).fetchone()
        return _row(ANALYSIS_FIELDS, values) if values else None

    def analyses(self, analyst: Optional[str] = None, league: Optional[str] = None, event_id: Optional[str] = None,
                 status: Optional[str] = None, started_before: Optional[str] = None,
                 limit: int = 5000) -> List[Dict[str, Any]]:
        where, args = [], []
        for clause, value in (("analyst = ?", analyst), ("league = ?", league), ("event_id = ?", event_id),
                              ("status = ?", status), ("start < ?", started_before)):
            if value is not None:
                where.append(clause)
                args.append(value)
        sql = f"SELECT {', '.join(ANALYSIS_FIELDS)} FROM analyses"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with closing(self._connect()) as db:
            rows = db.execute(sql + " ORDER BY start DESC LIMIT ?", (*args, limit)).fetchall()
        return [_row(ANALYSIS_FIELDS, values) for values in rows]

    def grade_analysis(self, analyst: str, league: str, event_id: str, status: str, result: Dict[str, Any]) -> None:
        with closing(self._connect()) as db:
            db.execute(
                "UPDATE analyses SET status = ?, result = ?, graded_at = ? WHERE analyst = ? AND league = ? "
                "AND event_id = ?",
                (status, json.dumps(result), time.time(), analyst, league, event_id),
            )

    def log_analyst_call(self, analyst: str, model: str, event_id: Optional[str], input_tokens: int,
                         output_tokens: int, cost: float, ok: bool, now: Optional[float] = None) -> None:
        with closing(self._connect()) as db:
            db.execute(
                "INSERT INTO analyst_calls (ts, analyst, model, event_id, input_tokens, output_tokens, cost, ok) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (_now(now), analyst, model, event_id, input_tokens, output_tokens, cost, int(ok)),
            )

    def analyst_spend(self, analyst: str, since: float = 0.0) -> Tuple[int, float]:
        """(calls, estimated dollars) for an analyst since a time (epoch seconds)."""
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(cost), 0) FROM analyst_calls WHERE analyst = ? AND ts >= ?",
                (analyst, since),
            ).fetchone()
        return int(row[0]), float(row[1])

    def last_analyst_call(self, analyst: str, event_id: str) -> Optional[Tuple[float, bool]]:
        """(when, whether a forecast came back) of the analyst's latest call on a game."""
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT ts, ok FROM analyst_calls WHERE analyst = ? AND event_id = ? ORDER BY ts DESC LIMIT 1",
                (analyst, event_id),
            ).fetchone()
        return (row[0], bool(row[1])) if row else None

    # --- users: one identity across the bot, the Mini App and the browser ------------------------

    def get_or_create_user(self, user_id: str, name: Optional[str] = None,
                           now: Optional[float] = None) -> Dict[str, Any]:
        """The person's account, made on first sight. A name only fills a blank one, so a name they
        chose in the app isn't overwritten by Telegram's on their next visit."""
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("A user id is needed")
        if user_id.startswith(RESERVED_PREFIX):
            raise ValueError("That id is reserved")
        with closing(self._connect()) as db:
            db.execute(
                "INSERT INTO users (id, name, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET name = COALESCE(users.name, excluded.name)",
                (user_id, _setting("name", name), _now(now)),
            )
        return self.user(user_id, now=now)

    def user(self, user_id: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """The account (with any budget raise that has waited its 24 hours applied), or None."""
        sql = f"SELECT {', '.join(USER_FIELDS)} FROM users WHERE id = ?"
        now = _now(now)
        with closing(self._connect()) as db:
            values = db.execute(sql, (user_id,)).fetchone()
            raise_at = values[USER_FIELDS.index("budget_raise_at")] if values else None
            if raise_at is not None and raise_at <= now:  # write only when a raise is due
                _apply_due_raise(db, user_id, now)
                values = db.execute(sql, (user_id,)).fetchone()
        return _record(USER_FIELDS, values) if values else None

    def update_user(self, user_id: str, changes: Mapping[str, Any],
                    now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Change the person's own settings: name, odds format, currency, favourite sports, default
        tier, reality-check interval, and ``settings`` (merged in; a None value removes a key).

        The budget, age confirmation and cool-off are refused here: each has its own method and rule.
        """
        unknown = set(changes) - set(EDITABLE_USER_FIELDS) - {"settings"}
        if unknown:
            raise ValueError(f"These can't be changed here: {', '.join(sorted(unknown))}")
        columns = {field: _setting(field, value) for field, value in changes.items() if field != "settings"}
        extra = changes.get("settings")
        if extra is not None and not isinstance(extra, Mapping):
            raise ValueError("settings must be an object")
        with closing(self._connect()) as db, _transaction(db):
            row = db.execute("SELECT settings FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                return None
            if extra:
                merged = {**json.loads(row[0]), **extra}
                columns["settings"] = _dumps({key: value for key, value in merged.items() if value is not None})
            if columns:
                assignments = ", ".join(f"{field} = ?" for field in columns)  # names come from the allow-list
                db.execute(f"UPDATE users SET {assignments} WHERE id = ?", (*columns.values(), user_id))
        return self.user(user_id, now=now)

    def set_weekly_budget(self, user_id: str, cents: int, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Set the weekly budget: a first budget or a lower one applies at once, a higher one waits 24 hours.

        A limit should protect straight away and loosen only after a cooling-off pause (UKGC, NCPG).
        Lowering (or re-confirming) the limit cancels a pending raise. Asking for more than a pending raise
        restarts its 24 hours; asking for less keeps its clock. Returns the account, or None if unknown.
        """
        cents = _cents(cents, "The weekly budget")
        now = _now(now)
        with closing(self._connect()) as db, _transaction(db):
            _apply_due_raise(db, user_id, now)
            row = db.execute(
                "SELECT weekly_budget_cents, pending_budget_cents, budget_raise_at FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if row is None:
                return None
            current, pending, raise_at = row
            if current is None or cents <= current:
                db.execute(
                    "UPDATE users SET weekly_budget_cents = ?, pending_budget_cents = NULL, budget_raise_at = NULL "
                    "WHERE id = ?", (cents, user_id),
                )
            else:
                if pending is None or cents > pending:
                    raise_at = now + BUDGET_RAISE_DELAY
                db.execute("UPDATE users SET pending_budget_cents = ?, budget_raise_at = ? WHERE id = ?",
                           (cents, raise_at, user_id))
        return self.user(user_id, now=now)

    def confirm_age(self, user_id: str, now: Optional[float] = None) -> bool:
        """Record that they confirmed they're old enough (the first confirmation's time is kept)."""
        with closing(self._connect()) as db:
            cursor = db.execute("UPDATE users SET age_confirmed_at = COALESCE(age_confirmed_at, ?) WHERE id = ?",
                                (_now(now), user_id))
        return cursor.rowcount > 0

    def start_cooloff(self, user_id: str, hours: float, now: Optional[float] = None) -> Optional[float]:
        """Start a time-out of 24 hours to about 3 months; returns when it ends (None if no such user).

        A time-out can be extended but never shortened or ended early: there is no automatic resume.
        """
        low, high = COOLOFF_HOURS
        if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not low <= hours <= high:
            raise ValueError(f"A cool-off lasts {low} hours to {high // 24} days")
        until = _now(now) + hours * 3600
        with closing(self._connect()) as db:
            db.execute("UPDATE users SET cooloff_until = MAX(COALESCE(cooloff_until, 0), ?) WHERE id = ?",
                       (until, user_id))
            row = db.execute("SELECT cooloff_until FROM users WHERE id = ?", (user_id,)).fetchone()
        return row[0] if row else None

    def cooloff_active(self, user_id: str, now: Optional[float] = None) -> bool:
        with closing(self._connect()) as db:
            row = db.execute("SELECT cooloff_until FROM users WHERE id = ?", (user_id,)).fetchone()
        return bool(row and row[0] is not None and row[0] > _now(now))

    def extend_setting(self, user_id: str, key: str, until: float) -> Optional[float]:
        """Set a time in the person's ``settings`` (such as the pause on picks) to ``until``, unless it already
        runs later: a pause is lengthened, never cut short. One write transaction, so the bot and the dashboard
        can't lose each other's. Returns the time now stored, or None if there is no such user."""
        until_value = _finite(until)
        if until_value is None:
            raise ValueError("A time is needed")
        with closing(self._connect()) as db, _transaction(db):
            row = db.execute("SELECT settings FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                return None
            settings = json.loads(row[0] or "{}")
            current = _finite(settings.get(key))
            if current is None or current < until_value:
                settings[key] = current = until_value
                db.execute("UPDATE users SET settings = ? WHERE id = ?", (_dumps(settings), user_id))
        return current

    def shared_user(self, shared_id: str, members: str, carry: Sequence[str] = (),
                    now: Optional[float] = None) -> Dict[str, Any]:
        """The reserved row ``shared_id`` (an id starting with "*") holding the gates shared by everyone whose
        id starts with ``members``: a break (cooloff_until), the weekly budget with its pending raise, and the
        ``carry`` settings (the pause on picks). set_weekly_budget, start_cooloff and extend_setting change it
        as they would a person's row, with the same rules.

        It is made on first use from the members' own rows, so no limit set before the gates were shared is
        lost: the latest break, the latest of each ``carry`` setting and the lowest weekly budget (a raise
        that has waited its 24 hours counts; one still waiting is dropped and must be asked for again).
        """
        if not shared_id.startswith(RESERVED_PREFIX):
            raise ValueError(f"A shared row's id starts with {RESERVED_PREFIX}")
        now = _now(now)
        row = self.user(shared_id, now=now)
        if row is not None:
            return row
        where, bounds = _prefixed("id", members)
        with closing(self._connect()) as db, _transaction(db):
            if db.execute("SELECT 1 FROM users WHERE id = ?", (shared_id,)).fetchone() is None:
                rows = db.execute(
                    "SELECT CASE WHEN budget_raise_at IS NOT NULL AND budget_raise_at <= ? THEN pending_budget_cents "
                    f"ELSE weekly_budget_cents END, cooloff_until, settings FROM users WHERE {where}", (now, *bounds),
                ).fetchall()
                settings: Dict[str, float] = {}
                for key in carry:
                    times = [_finite(json.loads(values[2] or "{}").get(key)) for values in rows]
                    if any(value is not None for value in times):
                        settings[key] = max(value for value in times if value is not None)
                db.execute(
                    "INSERT INTO users (id, created_at, weekly_budget_cents, cooloff_until, settings) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (shared_id, now, min((values[0] for values in rows if values[0] is not None), default=None),
                     max((values[1] for values in rows if values[1] is not None), default=None), _dumps(settings)),
                )
        return self.user(shared_id, now=now)

    # --- slips: tickets people save, each leg with the exact quote it was built on --------------

    def save_slip(self, user: str, ticket_type: str, legs: Sequence[Any], stake_cents: int,
                  payload: Optional[Dict[str, Any]] = None, system: Optional[str] = None, k: Optional[int] = None,
                  status: str = "saved", slip_id: Optional[int] = None, quoted_at: Optional[float] = None,
                  now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Save a draft or a bet with its legs; returns the slip (with a share code and its legs).

        ``legs`` are dicts (a priced ticket's legs work as they are) or TicketLegs; prices are kept as
        exact strings and start times as ISO 8601 UTC (a bet needs one on every leg; a draft may lack
        some). ``stake_cents`` is the total staked across every bet on the ticket. ``quoted_at``
        fills in the quote time of legs that don't carry their own. With ``slip_id`` the person's draft is
        replaced in place (same id and share code), e.g. as legs are added or when it is saved as a bet;
        None if that slip isn't one of their drafts. A draft saved as a bet takes the save time as its
        created_at, since that is when the money counts against the week's budget.
        """
        if status not in ("draft", "saved"):
            raise ValueError("A slip is saved as a draft or a bet; settling has its own method")
        if ticket_type not in TICKET_TYPES:
            raise ValueError(f"A slip's type is one of {', '.join(TICKET_TYPES)}")
        stake = _cents(stake_cents, "The stake")
        if status == "saved" and stake == 0:
            raise ValueError("A saved bet needs a stake")
        rows = [_leg_values(idx, leg, quoted_at, status) for idx, leg in enumerate(legs)]
        if not rows:
            raise ValueError("A slip needs at least one leg")
        if k is not None:
            _count(k, "A round robin's legs per bet", 1, MAX_SLIP_LEGS)
        now = _now(now)
        with closing(self._connect()) as db, _transaction(db):
            if slip_id is None:
                cursor = db.execute(
                    "INSERT INTO slips (user, status, type, system, k, stake_cents, created_at, updated_at, "
                    "share_code, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (user, status, ticket_type, system, k, stake, now, now, _new_share_code(db), _dumps(payload)),
                )
                slip_id = cursor.lastrowid
            else:
                cursor = db.execute(
                    "UPDATE slips SET status = ?, type = ?, system = ?, k = ?, stake_cents = ?, payload = ?, "
                    "updated_at = ?, created_at = CASE WHEN ? = 'saved' THEN ? ELSE created_at END "
                    "WHERE id = ? AND user = ? AND status = 'draft'",
                    (status, ticket_type, system, k, stake, _dumps(payload), now, status, now, slip_id, user),
                )
                if cursor.rowcount == 0:
                    return None
                db.execute("DELETE FROM slip_legs WHERE slip_id = ?", (slip_id,))
            db.executemany(
                "INSERT INTO slip_legs (slip_id, idx, key, league, event_id, market, side, point, price, label, game, "
                "start, quoted_at, ai_prob, market_prob, push_prob, book) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(slip_id, *row) for row in rows],
            )
        return self.slip(user, slip_id)

    def slip(self, user: str, slip_id: int) -> Optional[Dict[str, Any]]:
        """One of the person's slips with its legs; None if it isn't theirs."""
        with closing(self._connect()) as db:
            rows = db.execute(f"SELECT {', '.join(SLIP_FIELDS)} FROM slips WHERE id = ? AND user = ?",
                              (slip_id, user)).fetchall()
            slips = _with_legs(db, rows)
        return slips[0] if slips else None

    def shared_slip(self, share_code: str, viewer: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """A slip anyone with its share code may view. ``mine`` says whether the viewer owns it, and the
        owner's id (a Telegram id, say) is left out for everyone else."""
        code = (share_code or "").strip().upper()
        with closing(self._connect()) as db:
            rows = db.execute(f"SELECT {', '.join(SLIP_FIELDS)} FROM slips WHERE share_code = ?", (code,)).fetchall()
            slips = _with_legs(db, rows)
        if not slips:
            return None
        slip = slips[0]
        slip["mine"] = viewer is not None and viewer == slip["user"]
        if not slip["mine"]:
            slip["user"] = None
        return slip

    def slips(self, user: str, status: Union[None, str, Iterable[str]] = None,
              limit: Optional[int] = 100) -> List[Dict[str, Any]]:
        """The person's slips with their legs, newest first, optionally by status (one or several).
        ``limit=None`` returns every one."""
        where, args = ["user = ?"], [user]
        statuses = _statuses(status)
        if statuses:
            where.append(_in("status", statuses))
            args.extend(statuses)
        with closing(self._connect()) as db:
            rows = db.execute(
                f"SELECT {', '.join(SLIP_FIELDS)} FROM slips WHERE {' AND '.join(where)} "
                "ORDER BY created_at DESC, id DESC LIMIT ?", (*args, _limit(limit)),
            ).fetchall()
            return _with_legs(db, rows)

    def delete_slip(self, user: str, slip_id: int, now: Optional[float] = None) -> bool:
        """Remove one of the person's drafts, or a saved bet before any of its games has started.

        Settled slips, bets already under way and a plan's day stay: deleting them would rewrite the
        person's history and the week's budget.
        """
        now_iso = _iso(_now(now))
        with closing(self._connect()) as db, _transaction(db):
            row = db.execute("SELECT status FROM slips WHERE id = ? AND user = ?", (slip_id, user)).fetchone()
            if row is None or row[0] not in ("draft", "saved"):
                return False
            if db.execute("SELECT 1 FROM plan_days WHERE slip_id = ?", (slip_id,)).fetchone():
                return False
            # A leg with no start time counts as started: if unsure, keep the bet (fail closed)
            started = db.execute("SELECT 1 FROM slip_legs WHERE slip_id = ? AND (start IS NULL OR start <= ?)",
                                 (slip_id, now_iso)).fetchone()
            if row[0] == "saved" and started:
                return False
            db.execute("DELETE FROM slip_legs WHERE slip_id = ?", (slip_id,))
            db.execute("DELETE FROM slips WHERE id = ?", (slip_id,))
        return True

    def set_leg_results(self, slip_id: int, results: Sequence[Any]) -> None:
        """Record leg results as they come in, one per leg in order (None leaves a leg as it is)."""
        with closing(self._connect()) as db:
            db.executemany(
                "UPDATE slip_legs SET result = ? WHERE slip_id = ? AND idx = ?",
                [(_dumps(result), slip_id, idx) for idx, result in enumerate(results) if result is not None],
            )

    def settle_slip(self, slip_id: int, status: str, result: Dict[str, Any],
                    leg_results: Optional[Sequence[Any]] = None, now: Optional[float] = None) -> bool:
        """Mark a saved bet settled or void with its grade (and, optionally, each leg's result).

        A settled slip can be settled again (a corrected result); a draft was never a bet, so it can't.
        """
        if status not in ("settled", "void"):
            raise ValueError("A slip settles as 'settled' or 'void'")
        with closing(self._connect()) as db, _transaction(db):
            cursor = db.execute(
                "UPDATE slips SET status = ?, result = ?, updated_at = ? WHERE id = ? AND status != 'draft'",
                (status, _dumps(result), _now(now), slip_id),
            )
            if cursor.rowcount == 0:
                return False
            if leg_results is not None:
                db.executemany(
                    "UPDATE slip_legs SET result = ? WHERE slip_id = ? AND idx = ?",
                    [(_dumps(leg), slip_id, idx) for idx, leg in enumerate(leg_results) if leg is not None],
                )
        return True

    def open_slips(self, started_before: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        """Everyone's saved (unsettled) bets, oldest first: the grader's queue. With ``started_before``
        (ISO 8601 UTC) only bets with at least one game that started before then, or with no start time
        on record (saving a bet now needs one; this keeps any older bet from never being graded)."""
        sql = f"SELECT {', '.join(SLIP_FIELDS)} FROM slips WHERE status = 'saved'"
        args: List[Any] = []
        if started_before is not None:
            sql += " AND id IN (SELECT slip_id FROM slip_legs WHERE start IS NULL OR start < ?)"
            args.append(started_before)
        with closing(self._connect()) as db:
            rows = db.execute(sql + " ORDER BY created_at, id LIMIT ?", (*args, limit)).fetchall()
            return _with_legs(db, rows)

    def staked_since(self, user: str, since: float) -> int:
        """Cents the person has staked on saved and settled bets since a time (for the budget left).
        Drafts were never bets and void ones gave the stake back, so neither counts."""
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT COALESCE(SUM(stake_cents), 0) FROM slips WHERE user = ? AND status IN ('saved', 'settled') "
                "AND created_at >= ?", (user, since),
            ).fetchone()
        return int(row[0])

    def budget_staked_since(self, since: float, user: Optional[str] = None, members: Optional[str] = None) -> int:
        """Cents staked since a time that count against a weekly budget: one person's bets (``user``), or those
        of everyone whose id starts with ``members`` (every password browser, "web:"). As staked_since, but a
        paper plan's days are left out: they are play money."""
        if (user is None) == (members is None):
            raise ValueError("Give a user or an id prefix")
        where, args = ("user = ?", (user,)) if user is not None else _prefixed("user", members or "")
        with closing(self._connect()) as db:
            rows = db.execute(
                # Only a payload that mentions "paper" can be a paper plan's day: no other is read and decoded
                f"SELECT stake_cents, CASE WHEN payload LIKE '%paper%' THEN payload END FROM slips WHERE {where} "
                "AND status IN ('saved', 'settled') AND created_at >= ?", (*args, since),
            ).fetchall()
        total = 0
        for stake, payload in rows:
            plan = (json.loads(payload).get("plan") or {}) if payload else {}
            if not (isinstance(plan, dict) and plan.get("mode") == "paper"):
                total += int(stake)
        return total

    # --- rollover plans: recorded one day at a time, failures like successes ----------------------

    def create_plan(self, user: str, start_cents: int, days: int, min_odds: Any, legs_per_day: int, strategy: str,
                    payload: Optional[Dict[str, Any]] = None, mode: str = "paper",
                    now: Optional[float] = None) -> Dict[str, Any]:
        """Start a plan (paper by default) with the comparison the person saw; returns it.

        The limits are the rollover planner's (1 to 60 days, 1 to 3 legs a day, odds of 1.2 or more, one of
        its four strategies), so whatever is stored can be compared again later.
        """
        start = _cents(start_cents, "The starting amount")
        if start == 0:
            raise ValueError("A plan needs a starting amount")
        if mode not in PLAN_MODES:
            raise ValueError(f"mode must be one of {', '.join(PLAN_MODES)}")
        if strategy not in PLAN_STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(PLAN_STRATEGIES)}")
        days = _count(days, "days", *PLAN_DAYS)
        legs_per_day = _count(legs_per_day, "legs_per_day", *PLAN_LEGS_PER_DAY)
        odds = _exact_price(min_odds)
        if Decimal(odds) < PLAN_MIN_ODDS:
            raise ValueError(f"min_odds must be {PLAN_MIN_ODDS} or more; got {min_odds!r}")
        with closing(self._connect()) as db:
            cursor = db.execute(
                "INSERT INTO plans (user, created_at, mode, start_cents, days, min_odds, legs_per_day, strategy, "
                "payload, bankroll_cents) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user, _now(now), mode, start, days, odds, legs_per_day, strategy, _dumps(payload), start),
            )
        return self.plan(user, cursor.lastrowid)

    def plan(self, user: str, plan_id: int) -> Optional[Dict[str, Any]]:
        """One of the person's plans with ``days_played`` (its recorded days); None if it isn't theirs."""
        with closing(self._connect()) as db:
            rows = db.execute(f"SELECT {', '.join(PLAN_FIELDS)} FROM plans WHERE id = ? AND user = ?",
                              (plan_id, user)).fetchall()
            plans = _with_days(db, rows)
        return plans[0] if plans else None

    def plans(self, user: Optional[str] = None, status: Union[None, str, Iterable[str]] = None,
              limit: Optional[int] = 50, members: Optional[str] = None) -> List[Dict[str, Any]]:
        """The person's plans with their recorded days, newest first, optionally by status (``limit=None``:
        every one); or with ``members`` instead of ``user``, those of everyone whose id starts with it (every
        password browser, "web:", for the rules that cover them all)."""
        if (user is None) == (members is None):
            raise ValueError("Give a user or an id prefix")
        clause, bounds = ("user = ?", (user,)) if user is not None else _prefixed("user", members or "")
        where, args = [clause], list(bounds)
        statuses = _statuses(status)
        if statuses:
            where.append(_in("status", statuses))
            args.extend(statuses)
        with closing(self._connect()) as db:
            rows = db.execute(
                f"SELECT {', '.join(PLAN_FIELDS)} FROM plans WHERE {' AND '.join(where)} "
                "ORDER BY created_at DESC, id DESC LIMIT ?", (*args, _limit(limit)),
            ).fetchall()
            return _with_days(db, rows)

    def record_plan_day(self, user: str, plan_id: int, stake_cents: int, bankroll_cents: int,
                        result: Any = None, banked_cents: int = 0, rolled_cents: int = 0,
                        slip_id: Optional[int] = None, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Record the plan's next day (the person asked for it: nothing moves on by itself).

        The caller works out the money (the brain's strategy rules); this keeps the record. ``result``
        may wait (None) until the games are over: see settle_plan_day. Returns the day, or None if the
        plan isn't the person's active plan, every day is already recorded, or the slip isn't theirs.
        """
        values = [_cents(stake_cents, "The stake"), _cents(banked_cents, "The banked amount"),
                  _cents(rolled_cents, "The rolled amount"), _cents(bankroll_cents, "The bankroll")]
        now = _now(now)
        with closing(self._connect()) as db, _transaction(db):
            row = db.execute("SELECT status, days, day FROM plans WHERE id = ? AND user = ?",
                             (plan_id, user)).fetchone()
            if row is None or row[0] != "active" or row[2] >= row[1]:
                return None
            if slip_id is not None and not db.execute("SELECT 1 FROM slips WHERE id = ? AND user = ?",
                                                      (slip_id, user)).fetchone():
                return None
            day = row[2] + 1
            db.execute(
                "INSERT INTO plan_days (plan_id, day, slip_id, stake_cents, result, banked_cents, rolled_cents, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (plan_id, day, slip_id, values[0], _dumps(result), values[1], values[2], now),
            )
            db.execute("UPDATE plans SET day = ?, bankroll_cents = ? WHERE id = ?", (day, values[3], plan_id))
        return {"plan_id": plan_id, "day": day, "slip_id": slip_id, "stake_cents": values[0], "result": result,
                "banked_cents": values[1], "rolled_cents": values[2], "created_at": now}

    def settle_plan_day(self, user: str, plan_id: int, day: int, result: Any, bankroll_cents: int,
                        banked_cents: Optional[int] = None, rolled_cents: Optional[int] = None) -> bool:
        """Fill in a recorded day's result and the money after it, once its games are over. Works on a
        finished plan too, so a day still in play when the plan was stopped is recorded all the same.

        Banked and rolled amounts left out (None) keep what record_plan_day stored. The bankroll is the
        plan's current one only when this is its latest day: settling an earlier day late (the next day
        was already recorded) mustn't put an older bankroll back.
        """
        banked = None if banked_cents is None else _cents(banked_cents, "The banked amount")
        rolled = None if rolled_cents is None else _cents(rolled_cents, "The rolled amount")
        bankroll = _cents(bankroll_cents, "The bankroll")
        with closing(self._connect()) as db, _transaction(db):
            if not db.execute("SELECT 1 FROM plans WHERE id = ? AND user = ?", (plan_id, user)).fetchone():
                return False
            cursor = db.execute(
                "UPDATE plan_days SET result = ?, banked_cents = COALESCE(?, banked_cents), "
                "rolled_cents = COALESCE(?, rolled_cents) WHERE plan_id = ? AND day = ?",
                (_dumps(result), banked, rolled, plan_id, day),
            )
            if cursor.rowcount == 0:
                return False
            db.execute("UPDATE plans SET bankroll_cents = ? WHERE id = ? AND day = ?", (bankroll, plan_id, day))
        return True

    def finish_plan(self, user: str, plan_id: int, status: str) -> bool:
        """End an active plan as done (every day won), failed (a day lost) or stopped (the person stopped)."""
        if status not in PLAN_STATUSES or status == "active":
            raise ValueError("A plan finishes as done, failed or stopped")
        with closing(self._connect()) as db:
            cursor = db.execute("UPDATE plans SET status = ? WHERE id = ? AND user = ? AND status = 'active'",
                                (status, plan_id, user))
        return cursor.rowcount > 0

    # --- chat: one Claude conversation per topic ----------------------------------------------------

    def create_thread(self, user: str, title: Optional[str] = None, now: Optional[float] = None) -> Dict[str, Any]:
        with closing(self._connect()) as db:
            cursor = db.execute("INSERT INTO chat_threads (user, created_at, title) VALUES (?, ?, ?)",
                                (user, _now(now), title))
        return self.thread(user, cursor.lastrowid)

    def thread(self, user: str, thread_id: int) -> Optional[Dict[str, Any]]:
        """One of the person's threads; None if it isn't theirs."""
        with closing(self._connect()) as db:
            values = db.execute(f"SELECT {', '.join(THREAD_FIELDS)} FROM chat_threads WHERE id = ? AND user = ?",
                                (thread_id, user)).fetchone()
        return _record(THREAD_FIELDS, values) if values else None

    def threads(self, user: str, limit: int = 20) -> List[Dict[str, Any]]:
        """The person's threads, most recently active first, each with ``last_ts`` and ``messages`` (count)."""
        with closing(self._connect()) as db:
            rows = db.execute(
                f"SELECT {', '.join('t.' + field for field in THREAD_FIELDS)}, "
                "COALESCE(MAX(m.ts), t.created_at) AS last_ts, COUNT(m.idx) FROM chat_threads t "
                "LEFT JOIN chat_messages m ON m.thread_id = t.id WHERE t.user = ? "
                "GROUP BY t.id ORDER BY last_ts DESC, t.id DESC LIMIT ?", (user, limit),
            ).fetchall()
        return [{**_record(THREAD_FIELDS, values[:-2]), "last_ts": values[-2], "messages": values[-1]}
                for values in rows]

    def update_thread(self, user: str, thread_id: int, title: Optional[str] = None,
                      summary: Optional[str] = None) -> bool:
        """Rename a thread or store its compacted summary (whichever is given)."""
        columns = {name: value for name, value in (("title", title), ("summary", summary)) if value is not None}
        if not columns:
            return self.thread(user, thread_id) is not None
        with closing(self._connect()) as db:
            cursor = db.execute(
                f"UPDATE chat_threads SET {', '.join(f'{name} = ?' for name in columns)} WHERE id = ? AND user = ?",
                (*columns.values(), thread_id, user),
            )
        return cursor.rowcount > 0

    def add_message(self, user: str, thread_id: int, role: str, content: Any, tokens: Optional[int] = None,
                    cost: Optional[float] = None, now: Optional[float] = None) -> Optional[int]:
        """Append a message to one of the person's threads; returns its index (None if not their thread).
        ``content`` is text or content blocks; ``cost`` is estimated US dollars."""
        if not isinstance(role, str) or not role:
            raise ValueError("A message needs a role")
        if content is None:
            raise ValueError("A message needs content")
        if tokens is not None:
            _count(tokens, "tokens", 0, MAX_CENTS)  # any real count, and in range for SQLite
        if cost is not None:
            _usage(cost, "A message's cost")
        with closing(self._connect()) as db, _transaction(db):
            if not db.execute("SELECT 1 FROM chat_threads WHERE id = ? AND user = ?", (thread_id, user)).fetchone():
                return None
            idx = db.execute("SELECT COALESCE(MAX(idx) + 1, 0) FROM chat_messages WHERE thread_id = ?",
                             (thread_id,)).fetchone()[0]
            db.execute(
                "INSERT INTO chat_messages (thread_id, idx, role, content, ts, tokens, cost) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (thread_id, idx, role, _dumps(content), _now(now), tokens, cost),
            )
        return idx

    def messages(self, user: str, thread_id: int, last: Optional[int] = None) -> List[Dict[str, Any]]:
        """A thread's messages in order (only the last ``last`` if given); empty if it isn't the person's."""
        sql = (f"SELECT {', '.join('m.' + field for field in MESSAGE_FIELDS)} FROM chat_messages m "
               "JOIN chat_threads t ON t.id = m.thread_id WHERE m.thread_id = ? AND t.user = ? ORDER BY m.idx DESC")
        args: List[Any] = [thread_id, user]
        if last is not None:
            sql += " LIMIT ?"
            args.append(last)
        with closing(self._connect()) as db:
            rows = db.execute(sql, args).fetchall()
        return [_record(MESSAGE_FIELDS, values) for values in reversed(rows)]

    # --- AI credits: what each person's chat cost today -------------------------------------------

    @staticmethod
    def credit_day(now: Optional[float] = None) -> str:
        """Today's date (US Eastern, like the slate) for the daily AI allowance."""
        return datetime.datetime.fromtimestamp(_now(now), CREDIT_DAY_TZ).date().isoformat()

    def add_ai_usage(self, user: str, cents: float, day: Optional[str] = None, now: Optional[float] = None) -> float:
        """Add a turn's estimated cost (US cents, fractions kept) to the person's day; returns the day's total."""
        used = _usage(cents, "AI usage")
        day = day or self.credit_day(now)
        with closing(self._connect()) as db:
            _add_usage(db, user, day, used)
            return float(db.execute("SELECT used_cents FROM ai_credits WHERE user = ? AND day = ?",
                                    (user, day)).fetchone()[0])

    def ai_used_today(self, user: str, day: Optional[str] = None, now: Optional[float] = None) -> float:
        """US cents of AI the person has used on a day (today by default)."""
        with closing(self._connect()) as db:
            row = db.execute("SELECT used_cents FROM ai_credits WHERE user = ? AND day = ?",
                             (user, day or self.credit_day(now))).fetchone()
        return float(row[0]) if row else 0.0

    # --- AI holds: one running AI call per person, and what running calls hold from shared allowances ----
    # The bot and the dashboard are separate processes on this one file, so the hold lives here, not in
    # memory. An allowance is an ai_credits id with its daily cap in cents: the person's own id, and the
    # shared ones their spending also counts against ("*web" for every password browser, "*all").

    def _ai_left(self, db: sqlite3.Connection, user: str, caps: Mapping[str, float], now: float) -> float:
        """The least any of ``caps`` has left today after what it has used and what other people's running
        calls hold from it (the person's own running call isn't taken off)."""
        day = self.credit_day(now)
        held: Dict[str, float] = {}
        for cents, pools in db.execute("SELECT cents, pools FROM ai_holds WHERE user != ? AND expires_at > ?",
                                       (user, now)):
            for pool in json.loads(pools):
                held[pool] = held.get(pool, 0.0) + float(cents)
        left = math.inf
        for allowance, cap in caps.items():
            row = db.execute("SELECT used_cents FROM ai_credits WHERE user = ? AND day = ?",
                             (allowance, day)).fetchone()
            left = min(left, float(cap) - (float(row[0]) if row else 0.0) - held.get(allowance, 0.0))
        return left

    def ai_left(self, user: str, caps: Mapping[str, float], now: Optional[float] = None) -> float:
        """US cents the person may still spend on AI today: the least any of their allowances (``caps``:
        ai_credits id -> daily cap, their own id among them) has left, after other running calls' holds."""
        with closing(self._connect()) as db:
            return self._ai_left(db, user, caps, _now(now))

    def reserve_ai(self, user: str, caps: Mapping[str, float], most: float, least: float = 0.0,
                   now: Optional[float] = None, seconds: float = AI_HOLD_SECONDS) -> Optional[Tuple[str, float]]:
        """Start an AI call for a person, in one write transaction: (token, the most it may spend in US cents),
        or None while another call of theirs is running (in this process or another).

        The amount is at most ``most`` and never more than any of ``caps`` has left after what it has used
        and other running calls hold, so it is held from every allowance until settle_ai and release_ai.
        Under ``least`` there isn't enough for a call: nothing is held then, but the person still counts as
        busy until release_ai. A hold lapses after ``seconds``, in case the process holding it died.
        """
        now = _now(now)
        pools = [allowance for allowance in caps if allowance != user]
        with closing(self._connect()) as db, _transaction(db):
            db.execute("DELETE FROM ai_holds WHERE expires_at <= ?", (now,))
            if db.execute("SELECT 1 FROM ai_holds WHERE user = ?", (user,)).fetchone():
                return None
            cents = max(0.0, min(float(most), self._ai_left(db, user, caps, now)))
            token = secrets.token_hex(8)
            db.execute("INSERT INTO ai_holds (user, token, cents, pools, started_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                       (user, token, cents if cents >= least else 0.0, json.dumps(pools), now, now + seconds))
        return token, cents

    def settle_ai(self, user: str, token: str, cents: float, pools: Sequence[str] = (),
                  now: Optional[float] = None) -> None:
        """What a running call really cost (US cents), in one write: added to the person's day and to each of
        ``pools``, and the call's hold drops to nothing (the person stays busy until release_ai)."""
        used = _usage(cents, "AI usage")
        now = _now(now)
        day = self.credit_day(now)
        with closing(self._connect()) as db, _transaction(db):
            if used > 0:
                for allowance in dict.fromkeys([user, *pools]):
                    _add_usage(db, allowance, day, used)
            db.execute("UPDATE ai_holds SET cents = 0 WHERE user = ? AND token = ?", (user, token))

    def renew_ai(self, user: str, token: str, now: Optional[float] = None, seconds: float = AI_HOLD_SECONDS) -> bool:
        """Keep a running call's hold for ``seconds`` more (called before each request it sends to the model).
        False when it has already lapsed: other calls may have counted what it held as free since, so the call
        must stop rather than spend it."""
        now = _now(now)
        with closing(self._connect()) as db:
            cursor = db.execute("UPDATE ai_holds SET expires_at = ? WHERE user = ? AND token = ? AND expires_at > ?",
                                (now + seconds, user, token, now))
        return cursor.rowcount > 0

    def release_ai(self, user: str, token: str) -> None:
        """The call is over (or failed): the person may start another. Only that call's own hold is removed."""
        with closing(self._connect()) as db:
            db.execute("DELETE FROM ai_holds WHERE user = ? AND token = ?", (user, token))

    # --- pasted bet slips: what was read from each, and where its legs matched -------------------------

    def save_pasted(self, user: str, source: str, extraction: Mapping[str, Any], matches: Sequence[Any],
                    text: Optional[str] = None, thread_id: Optional[int] = None, now: Optional[float] = None) -> int:
        """Keep a pasted slip as read (``extraction``) and matched (``matches``, one entry per leg), with the
        pasted text (cut to 6,000 characters); returns its id. Only the person's newest MAX_PASTED are kept."""
        if not isinstance(user, str) or not user or user.startswith(RESERVED_PREFIX):
            raise ValueError("A pasted slip needs a person's id")
        if source not in PASTED_SOURCES:
            raise ValueError(f"source must be one of {', '.join(PASTED_SOURCES)}")
        if not isinstance(extraction, Mapping) or not isinstance(matches, (list, tuple)):
            raise ValueError("A pasted slip needs what was read and its matches")
        if thread_id is not None:
            _count(thread_id, "thread_id", 1, MAX_CENTS)
        if text is not None and not isinstance(text, str):
            raise ValueError("The pasted text must be text")
        with closing(self._connect()) as db, _transaction(db):
            cursor = db.execute(
                "INSERT INTO pasted_slips (user, thread_id, created_at, source, text, extraction, matches) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user, thread_id, _now(now), source, text[:MAX_PASTED_TEXT] if text else None,
                 _dumps(dict(extraction)), _dumps(list(matches))),
            )
            db.execute("DELETE FROM pasted_slips WHERE user = ? AND id NOT IN (SELECT id FROM pasted_slips "
                       "WHERE user = ? ORDER BY created_at DESC, id DESC LIMIT ?)", (user, user, MAX_PASTED))
        return int(cursor.lastrowid)

    def pasted(self, user: str, pasted_id: int) -> Optional[Dict[str, Any]]:
        """One of the person's pasted slips; None if there is none with that id or it isn't theirs."""
        with closing(self._connect()) as db:
            values = db.execute(f"SELECT {', '.join(PASTED_FIELDS)} FROM pasted_slips WHERE id = ? AND user = ?",
                                (pasted_id, user)).fetchone()
        return _record(PASTED_FIELDS, values) if values else None

    def pasted_for_thread(self, user: str, thread_id: int) -> List[Dict[str, Any]]:
        """The person's slips pasted into one chat, oldest first (newest last)."""
        with closing(self._connect()) as db:
            rows = db.execute(f"SELECT {', '.join(PASTED_FIELDS)} FROM pasted_slips WHERE user = ? AND thread_id = ? "
                              "ORDER BY created_at, id", (user, thread_id)).fetchall()
        return [_record(PASTED_FIELDS, values) for values in rows]

    # --- responsible gambling: warning signs code acted on, for the owner to review ----------------

    def add_rg_event(self, user: str, kind: str, detail: Optional[Dict[str, Any]] = None,
                     action: Optional[str] = None, now: Optional[float] = None) -> int:
        """Log a warning sign and what code did about it; returns the event's id."""
        if not isinstance(kind, str) or not kind:
            raise ValueError("A warning needs a kind")
        with closing(self._connect()) as db:
            cursor = db.execute("INSERT INTO rg_events (user, ts, kind, detail, action) VALUES (?, ?, ?, ?, ?)",
                                (user, _now(now), kind, _dumps(detail), action))
        return int(cursor.lastrowid)

    def rg_events(self, user: Optional[str] = None, kind: Optional[str] = None, reviewed: Optional[bool] = None,
                  since: Optional[float] = None, limit: int = 200,
                  members: Optional[str] = None) -> List[Dict[str, Any]]:
        """Warning signs, newest first: one person's, those of everyone whose id starts with ``members`` (every
        password browser, "web:") or everyone's, by kind, reviewed or not, since a time."""
        where: List[str] = []
        args: List[Any] = []
        if members is not None:
            clause, bounds = _prefixed("user", members)
            where.append(clause)
            args.extend(bounds)
        for clause, value in (("user = ?", user), ("kind = ?", kind), ("ts >= ?", since)):
            if value is not None:
                where.append(clause)
                args.append(value)
        if reviewed is not None:
            where.append("reviewed_at IS NOT NULL" if reviewed else "reviewed_at IS NULL")
        sql = f"SELECT {', '.join(RG_FIELDS)} FROM rg_events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with closing(self._connect()) as db:
            rows = db.execute(sql + " ORDER BY ts DESC, id DESC LIMIT ?", (*args, limit)).fetchall()
        return [_record(RG_FIELDS, values) for values in rows]

    def review_rg_event(self, event_id: int, now: Optional[float] = None) -> bool:
        """Mark a warning reviewed by the owner (the first review's time is kept)."""
        with closing(self._connect()) as db:
            cursor = db.execute("UPDATE rg_events SET reviewed_at = ? WHERE id = ? AND reviewed_at IS NULL",
                                (_now(now), event_id))
        return cursor.rowcount > 0
