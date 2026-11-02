"""The history tables (v3 learning loop, Contract 1): old games, their lines, match stats and weather, plus the
backfill's own progress and nightly request count.

- ``hist_games``: one row per game, unique on (league, game_key). ``game_key`` is ``espn:<id>`` whenever ESPN's
  event id is known (the same key the live logs use), else ``fd:<DIV>:<YYYYMMDD>:<home>:<away>`` (football-data),
  ``nflverse:<game_id>`` or ``mlb:<gamePk>``; a row is re-keyed to ``espn:<id>`` once the id is joined. Home and
  away are ESPN's orientation. ``start`` is ISO UTC (``2026-10-25T21:00:00Z``), ``status`` ``final`` or ``void``,
  ``season`` the calendar year the season starts in (the NBA's 2025-26 is 2025). ``home_key``/``away_key`` are
  ``data/markets/aliases.team_key`` of the names (joins across sources); ``info`` is JSON extras (referee, venue,
  roof, half-time score, the 90-minute rule's full score after extra time...).
- ``hist_lines``: prices known before the start. ``kind`` is ``open`` or ``close``; ``market`` ``h2h``, ``spreads``
  or ``totals``; ``side`` ``home``/``away``/``draw``/``over``/``under``; ``point`` the handicap from the side's
  own view (home -3.5) or the total; ``price`` decimal odds; ``as_of`` (epoch seconds) the latest time the price
  is known to have been on the board (an upper bound: the real time can be earlier). A line with ``as_of`` at or
  after the game's start is never written (``add_lines`` drops it), and HistoryReader filters again.
- ``hist_stats``: per game and side (``home``/``away``), a stat (corners, yellow, red, shots, shots_on_target,
  fouls, possession, xg, per-period scores, other sports' team box stats) and its value.
- ``hist_weather``: ``forecast_24h`` (what a forecast said a day out) or ``actual`` (the game-time observation),
  temperature in C, wind in km/h, precipitation in mm.
- ``hist_progress``: the backfill's state per (source, unit), resumable after a restart.
- ``hist_budget``: requests made per night (the server-local date the night's window opened on).

``HIST_SCHEMA`` / ``init_history_schema(db)`` create them on any connection to the shared SQLite file (data/store.py
hooks it in; the backfill and the reader also call it, idempotently). Every helper takes an open
``sqlite3.Connection`` and works in autocommit or inside the caller's transaction.
"""

from __future__ import annotations

import datetime
import json
import math
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

HIST_SCHEMA = """
CREATE TABLE IF NOT EXISTS hist_games (
    league TEXT NOT NULL,          -- the app's league key (soccer_epl, basketball_nba, ...)
    game_key TEXT NOT NULL,        -- espn:<id>, else fd:/nflverse:/mlb: keys (see learn/history_store.py)
    espn_id TEXT,
    start TEXT NOT NULL,           -- ISO UTC
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    home_score INTEGER,            -- soccer: the 90-minute score (the settlement rule)
    away_score INTEGER,
    status TEXT NOT NULL,          -- final or void
    season INTEGER,                -- the calendar year the season starts in
    source TEXT NOT NULL,          -- football-data, nflverse, espn, mlb-statsapi, live
    fetched_at REAL NOT NULL,
    home_key TEXT,                 -- aliases.team_key of the names, for joins across sources
    away_key TEXT,
    info TEXT,                     -- JSON extras
    PRIMARY KEY (league, game_key)
);
CREATE INDEX IF NOT EXISTS hist_games_by_start ON hist_games (league, start);
CREATE INDEX IF NOT EXISTS hist_games_by_key ON hist_games (game_key);
CREATE TABLE IF NOT EXISTS hist_lines (
    game_key TEXT NOT NULL,
    kind TEXT NOT NULL,            -- open or close
    book TEXT NOT NULL,            -- pinnacle, betfair_ex, avg, bet365, median, draftkings, espn_bet, nflverse, ...
    market TEXT NOT NULL,          -- h2h, spreads, totals
    side TEXT NOT NULL,            -- home, away, draw, over, under
    point REAL,                    -- the side's handicap, or the total
    price REAL,                    -- decimal odds
    as_of REAL NOT NULL,           -- epoch seconds, always before the start
    source TEXT NOT NULL,
    PRIMARY KEY (game_key, kind, market, book, side, source)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS hist_stats (
    game_key TEXT NOT NULL,
    team_side TEXT NOT NULL,       -- home or away
    stat TEXT NOT NULL,
    value REAL,
    source TEXT NOT NULL,
    PRIMARY KEY (game_key, team_side, stat, source)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS hist_weather (
    game_key TEXT NOT NULL,
    kind TEXT NOT NULL,            -- forecast_24h or actual
    temp_c REAL,
    wind_kmh REAL,
    precip_mm REAL,
    source TEXT NOT NULL,
    PRIMARY KEY (game_key, kind, source)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS hist_progress (
    source TEXT NOT NULL,          -- football-data, nflverse, espn
    unit TEXT NOT NULL,            -- e.g. E0:2025, games.csv, basketball_nba:2025
    status TEXT NOT NULL,          -- todo, active, current (caught up; more games will come), done
    state TEXT,                    -- JSON: the unit's cursor
    games INTEGER NOT NULL DEFAULT 0,
    requests INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    updated_at REAL,
    note TEXT,
    PRIMARY KEY (source, unit)
);
CREATE TABLE IF NOT EXISTS hist_budget (
    night TEXT PRIMARY KEY,        -- server-local date the night's window opened on
    used INTEGER NOT NULL DEFAULT 0,
    updated_at REAL
);
"""

GAME_FIELDS = ("league", "game_key", "espn_id", "start", "home", "away", "home_score", "away_score", "status",
               "season", "source", "fetched_at", "home_key", "away_key", "info")
LINE_FIELDS = ("game_key", "kind", "book", "market", "side", "point", "price", "as_of", "source")
KINDS = ("open", "close")
MARKETS = ("h2h", "spreads", "totals")
SIDES = ("home", "away", "draw", "over", "under")
WEATHER_KINDS = ("forecast_24h", "actual")
STATUSES = ("final", "void")


def init_history_schema(db: sqlite3.Connection) -> None:
    """Create the history tables on ``db`` (idempotent)."""
    db.executescript(HIST_SCHEMA)


# --- small helpers ------------------------------------------------------------------------------------------------

def num(value: Any) -> Optional[float]:
    """A finite float, or None (empty strings, text, NaN, booleans)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def to_ts(value: Any) -> Optional[float]:
    """Epoch seconds from a number, a datetime (naive = UTC), a date (midnight UTC) or an ISO string."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, datetime.datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
        return moment.timestamp()
    if isinstance(value, datetime.date):
        return datetime.datetime(value.year, value.month, value.day, tzinfo=datetime.timezone.utc).timestamp()
    if isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        if len(text) == 10:
            text += "T00:00:00+00:00"
        try:
            moment = datetime.datetime.fromisoformat(text)
        except ValueError:
            return None
        return to_ts(moment)
    return None


def iso(ts: float) -> str:
    """Epoch seconds as ISO UTC with a Z (the start format of every log table)."""
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def loads(text: Any) -> Any:
    if text is None or text == "":
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


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


# --- games --------------------------------------------------------------------------------------------------------

def get_game(db: sqlite3.Connection, league: str, game_key: str) -> Optional[Dict[str, Any]]:
    row = db.execute(f"SELECT {', '.join(GAME_FIELDS)} FROM hist_games WHERE league = ? AND game_key = ?",
                     (league, game_key)).fetchone()
    return _game_row(row) if row else None


def _game_row(values: Sequence[Any]) -> Dict[str, Any]:
    row = dict(zip(GAME_FIELDS, values))
    row["info"] = loads(row["info"]) or {}
    return row


def upsert_game(db: sqlite3.Connection, game: Mapping[str, Any], now: Optional[float] = None) -> str:
    """Insert or update one game (a mapping with GAME_FIELDS; ``start`` any time form). Known values are kept when
    the new row has none; ``info`` is merged; the first source stays the row's source. Returns the game_key."""
    start_ts = to_ts(game.get("start"))
    if start_ts is None:
        raise ValueError(f"a game needs a start: {game.get('game_key')}")
    status = game.get("status") or "final"
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    old = get_game(db, game["league"], game["game_key"])
    info = dict(old["info"]) if old else {}
    info.update({k: v for k, v in (game.get("info") or {}).items() if v is not None})
    values = {
        "league": game["league"], "game_key": game["game_key"], "espn_id": _text(game.get("espn_id")),
        "start": iso(start_ts), "home": str(game.get("home") or ""), "away": str(game.get("away") or ""),
        "home_score": _int(game.get("home_score")), "away_score": _int(game.get("away_score")),
        "status": status, "season": _int(game.get("season")), "source": str(game.get("source") or "unknown"),
        "fetched_at": float(now if now is not None else time.time()),
        "home_key": game.get("home_key"), "away_key": game.get("away_key"), "info": dumps(info or None)}
    if old:
        for key in ("espn_id", "home_score", "away_score", "season", "home_key", "away_key"):
            if values[key] is None:
                values[key] = old[key]
        rescheduled = old["status"] == "void" and status == "final"
        for key in ("home", "away", "start"):
            if old[key] and not (rescheduled and key == "start"):
                values[key] = old[key]  # the first source's names and start stay (ESPN's, when the grader was first)
        values["source"] = old["source"]
        if old["status"] == "final" and status == "void" and old["home_score"] is not None:
            values["status"] = "final"  # a later "void" read never undoes a played game
    db.execute(f"INSERT OR REPLACE INTO hist_games ({', '.join(GAME_FIELDS)}) VALUES "
               f"({', '.join('?' for _ in GAME_FIELDS)})", tuple(values[k] for k in GAME_FIELDS))
    return game["game_key"]


def _int(value: Any) -> Optional[int]:
    number = num(value)
    return None if number is None else int(round(number))


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def find_match(db: sqlite3.Connection, league: str, start: Any, home_key: Optional[str], away_key: Optional[str],
               *, hours: float = 36.0, prefix: Optional[str] = None,
               exclude: Optional[str] = None) -> Optional[str]:
    """The game_key of the same game already stored from another source: same league, a start within ``hours``,
    and the teams matching (both keys equal; else one equal and the other's words overlapping; else one equal and
    no other candidate has that team). ``prefix`` limits the keys looked at (e.g. "fd:"). None when unsure."""
    start_ts = to_ts(start)
    if start_ts is None or not (home_key or away_key):
        return None
    low, high = iso(start_ts - hours * 3600), iso(start_ts + hours * 3600)
    rows = db.execute("SELECT game_key, home_key, away_key, start FROM hist_games WHERE league = ? AND start >= ? "
                      "AND start <= ?", (league, low, high)).fetchall()
    candidates = [r for r in rows if (prefix is None or r[0].startswith(prefix)) and r[0] != exclude]
    best: List[Tuple[float, float, str]] = []
    for key, home, away, when in candidates:
        same_home, same_away = bool(home_key) and home == home_key, bool(away_key) and away == away_key
        if same_home and same_away:
            score = 3.0
        elif (same_home and _overlap(away, away_key)) or (same_away and _overlap(home, home_key)):
            score = 2.0
        elif same_home or same_away:
            team = home_key if same_home else away_key
            others = [r for r in candidates if r[0] != key and team in (r[1], r[2])]
            score = 1.0 if not others else 0.0
        else:
            score = 0.0
        if score:
            best.append((score, -abs((to_ts(when) or 0.0) - start_ts), key))
    if not best:
        return None
    best.sort(reverse=True)
    if len(best) > 1 and best[0][:2] == best[1][:2]:
        return None  # two equally good candidates: don't guess
    return best[0][2]


def _overlap(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    generic = {"united", "city", "fc", "real", "club", "sporting", "athletic", "de", "the", "and", "town"}
    return bool((set(a.split()) - generic) & (set(b.split()) - generic))


def rekey(db: sqlite3.Connection, league: str, old: str, new: str, espn_id: Optional[str] = None) -> bool:
    """Move a game and its lines, stats and weather from key ``old`` to ``new`` (rows already under ``new`` win).
    Returns True when something moved."""
    if old == new:
        return False
    with transaction(db):
        row = get_game(db, league, old)
        if row is None:
            return False
        if get_game(db, league, new) is None:
            db.execute("UPDATE hist_games SET game_key = ?, espn_id = COALESCE(?, espn_id) WHERE league = ? AND "
                       "game_key = ?", (new, espn_id, league, old))
        else:
            info = row["info"]
            info["merged_from"] = old
            upsert_game(db, {**row, "game_key": new, "espn_id": espn_id or row["espn_id"], "info": info,
                             "source": row["source"]}, row["fetched_at"])
            db.execute("DELETE FROM hist_games WHERE league = ? AND game_key = ?", (league, old))
        for table in ("hist_lines", "hist_stats", "hist_weather"):
            db.execute(f"UPDATE OR IGNORE {table} SET game_key = ? WHERE game_key = ?", (new, old))
            db.execute(f"DELETE FROM {table} WHERE game_key = ?", (old,))
    return True


# --- lines, stats, weather ----------------------------------------------------------------------------------------

def add_lines(db: sqlite3.Connection, game_key: str, start: Any, lines: Iterable[Mapping[str, Any]],
              source: str) -> int:
    """Write lines (dicts with kind, book, market, side, point, price, as_of). A line priced at or after the
    start, or with no as_of, an unknown kind/market/side or no usable price, is dropped. Returns the rows written."""
    start_ts = to_ts(start)
    if start_ts is None:
        return 0
    written = 0
    for line in lines:
        as_of = to_ts(line.get("as_of"))
        price = num(line.get("price"))
        if as_of is None or as_of >= start_ts:
            continue  # never a post-start price
        if line.get("kind") not in KINDS or line.get("market") not in MARKETS or line.get("side") not in SIDES:
            continue
        if price is None or price <= 1.0:
            continue
        db.execute("INSERT OR REPLACE INTO hist_lines (game_key, kind, book, market, side, point, price, as_of, "
                   "source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (game_key, line["kind"], str(line.get("book") or "unknown"), line["market"], line["side"],
                    num(line.get("point")), round(price, 4), as_of, source))
        written += 1
    return written


def add_stats(db: sqlite3.Connection, game_key: str, stats: Iterable[Mapping[str, Any]], source: str) -> int:
    """Write stats (dicts with team_side, stat, value). Returns the rows written."""
    written = 0
    for item in stats:
        value = num(item.get("value"))
        if item.get("team_side") not in ("home", "away") or not item.get("stat") or value is None:
            continue
        db.execute("INSERT OR REPLACE INTO hist_stats (game_key, team_side, stat, value, source) VALUES "
                   "(?, ?, ?, ?, ?)", (game_key, item["team_side"], str(item["stat"]), value, source))
        written += 1
    return written


def add_weather(db: sqlite3.Connection, game_key: str, rows: Iterable[Mapping[str, Any]], source: str) -> int:
    written = 0
    for item in rows:
        if item.get("kind") not in WEATHER_KINDS:
            continue
        values = [num(item.get(k)) for k in ("temp_c", "wind_kmh", "precip_mm")]
        if all(v is None for v in values):
            continue
        db.execute("INSERT OR REPLACE INTO hist_weather (game_key, kind, temp_c, wind_kmh, precip_mm, source) "
                   "VALUES (?, ?, ?, ?, ?, ?)", (game_key, item["kind"], *values, source))
        written += 1
    return written


def write_record(db: sqlite3.Connection, record: Mapping[str, Any], now: Optional[float] = None) -> Dict[str, int]:
    """Write one source record {"game": {...}, "lines": [...], "stats": [...], "weather": [...]} in a transaction.
    The game's ``source`` names the rows' source. Returns {"lines", "stats", "weather"} written."""
    game = record["game"]
    source = str(game.get("source") or "unknown")
    with transaction(db):
        key = upsert_game(db, game, now)
        start = get_game(db, game["league"], key)["start"]  # the stored start (a merged row keeps the first one's)
        return {"lines": add_lines(db, key, start, record.get("lines") or [], source),
                "stats": add_stats(db, key, record.get("stats") or [], source),
                "weather": add_weather(db, key, record.get("weather") or [], source)}


def has_close_lines(db: sqlite3.Connection, game_key: str, sources: Optional[Sequence[str]] = None) -> bool:
    sql = "SELECT 1 FROM hist_lines WHERE game_key = ? AND kind = 'close'"
    args: List[Any] = [game_key]
    if sources:
        sql += f" AND source IN ({', '.join('?' for _ in sources)})"
        args.extend(sources)
    return db.execute(sql + " LIMIT 1", args).fetchone() is not None


def lines_for(db: sqlite3.Connection, game_keys: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    found: Dict[str, List[Dict[str, Any]]] = {key: [] for key in game_keys}
    for chunk in _chunks(list(game_keys), 400):
        rows = db.execute(f"SELECT {', '.join(LINE_FIELDS)} FROM hist_lines WHERE game_key IN "
                          f"({', '.join('?' for _ in chunk)}) ORDER BY as_of, kind, market, book, side", chunk)
        for values in rows:
            line = dict(zip(LINE_FIELDS, values))
            found[line.pop("game_key")].append(line)
    return found


def stats_for(db: sqlite3.Connection, game_keys: Sequence[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{game_key: {"home": {stat: value}, "away": {...}}}; a stat from two sources keeps the live one, then any."""
    found: Dict[str, Dict[str, Dict[str, float]]] = {key: {"home": {}, "away": {}} for key in game_keys}
    for chunk in _chunks(list(game_keys), 400):
        rows = db.execute(f"SELECT game_key, team_side, stat, value, source FROM hist_stats WHERE game_key IN "
                          f"({', '.join('?' for _ in chunk)}) ORDER BY source = 'live'", chunk)
        for key, side, stat, value, _source in rows:
            found[key][side][stat] = value
    return found


def weather_for(db: sqlite3.Connection, game_keys: Sequence[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    found: Dict[str, Dict[str, Dict[str, Any]]] = {key: {} for key in game_keys}
    for chunk in _chunks(list(game_keys), 400):
        rows = db.execute(f"SELECT game_key, kind, temp_c, wind_kmh, precip_mm, source FROM hist_weather WHERE "
                          f"game_key IN ({', '.join('?' for _ in chunk)}) ORDER BY source", chunk)
        for key, kind, temp, wind, precip, source in rows:
            found[key].setdefault(kind, {"temp_c": temp, "wind_kmh": wind, "precip_mm": precip, "source": source})
    return found


def _chunks(items: List[Any], size: int) -> Iterator[List[Any]]:
    for at in range(0, len(items), size):
        yield items[at:at + size]


def games_between(db: sqlite3.Connection, league: str, since: Optional[str], until: Optional[str]
                  ) -> List[Dict[str, Any]]:
    """The league's games with ``since <= start < until`` (ISO UTC strings, either may be None), oldest first."""
    sql = f"SELECT {', '.join(GAME_FIELDS)} FROM hist_games WHERE league = ?"
    args: List[Any] = [league]
    if since:
        sql += " AND start >= ?"
        args.append(since)
    if until:
        sql += " AND start < ?"
        args.append(until)
    return [_game_row(r) for r in db.execute(sql + " ORDER BY start, game_key", args)]


def counts(db: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    """Per league: games, final, void, with close lines, with stats, first and last start."""
    found: Dict[str, Dict[str, Any]] = {}
    for league, status, n, first, last in db.execute(
            "SELECT league, status, COUNT(*), MIN(start), MAX(start) FROM hist_games GROUP BY league, status"):
        entry = found.setdefault(league, {"games": 0, "final": 0, "void": 0, "first": first, "last": last})
        entry["games"] += n
        entry[status] = n
        entry["first"] = min(entry["first"], first)
        entry["last"] = max(entry["last"], last)
    for league, n in db.execute("SELECT g.league, COUNT(DISTINCT g.game_key) FROM hist_games g JOIN hist_lines l ON "
                                "l.game_key = g.game_key AND l.kind = 'close' GROUP BY g.league"):
        found.setdefault(league, {"games": 0})["with_close"] = n
    for league, n in db.execute("SELECT g.league, COUNT(DISTINCT g.game_key) FROM hist_games g JOIN hist_stats s ON "
                                "s.game_key = g.game_key GROUP BY g.league"):
        found.setdefault(league, {"games": 0})["with_stats"] = n
    return found


# --- the backfill's progress and budget -------------------------------------------------------------------------

PROGRESS_FIELDS = ("source", "unit", "status", "state", "games", "requests", "errors", "updated_at", "note")


def get_progress(db: sqlite3.Connection, source: str, unit: str) -> Optional[Dict[str, Any]]:
    row = db.execute(f"SELECT {', '.join(PROGRESS_FIELDS)} FROM hist_progress WHERE source = ? AND unit = ?",
                     (source, unit)).fetchone()
    if not row:
        return None
    found = dict(zip(PROGRESS_FIELDS, row))
    found["state"] = loads(found["state"]) or {}
    return found


def set_progress(db: sqlite3.Connection, source: str, unit: str, *, status: Optional[str] = None,
                 state: Optional[Mapping[str, Any]] = None, games: int = 0, requests: int = 0, errors: int = 0,
                 note: Optional[str] = None, now: Optional[float] = None) -> None:
    """Create or update a unit's progress: ``status``/``state``/``note`` replace when given, the counts add up."""
    stamp = float(now if now is not None else time.time())
    with transaction(db):
        old = get_progress(db, source, unit)
        if old is None:
            db.execute("INSERT INTO hist_progress (source, unit, status, state, games, requests, errors, updated_at, "
                       "note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (source, unit, status or "todo", dumps(dict(state or {})), games, requests, errors, stamp, note))
            return
        db.execute("UPDATE hist_progress SET status = ?, state = ?, games = games + ?, requests = requests + ?, "
                   "errors = errors + ?, updated_at = ?, note = ? WHERE source = ? AND unit = ?",
                   (status or old["status"], dumps(dict(state)) if state is not None else dumps(old["state"]),
                    games, requests, errors, stamp, note if note is not None else old["note"], source, unit))


def progress_rows(db: sqlite3.Connection, source: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = f"SELECT {', '.join(PROGRESS_FIELDS)} FROM hist_progress"
    args: Tuple[Any, ...] = ()
    if source:
        sql += " WHERE source = ?"
        args = (source,)
    rows = []
    for values in db.execute(sql + " ORDER BY source, unit", args):
        row = dict(zip(PROGRESS_FIELDS, values))
        row["state"] = loads(row["state"]) or {}
        rows.append(row)
    return rows


def budget_used(db: sqlite3.Connection, night: str) -> int:
    row = db.execute("SELECT used FROM hist_budget WHERE night = ?", (night,)).fetchone()
    return int(row[0]) if row else 0


def add_budget(db: sqlite3.Connection, night: str, n: int = 1, now: Optional[float] = None) -> int:
    """Count ``n`` more requests for ``night``; returns the night's total."""
    stamp = float(now if now is not None else time.time())
    db.execute("INSERT INTO hist_budget (night, used, updated_at) VALUES (?, ?, ?) ON CONFLICT(night) DO UPDATE SET "
               "used = used + excluded.used, updated_at = excluded.updated_at", (night, int(n), stamp))
    return budget_used(db, night)
