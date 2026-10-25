"""A SQLite file shared by the bot and the dashboard on the same server.

Holds each league's latest odds snapshot (so the two processes never pay
twice for the same fetch and nothing is lost on restart), a log of The
Odds API's quota headers, short leases so only one process refreshes a
league at a time, the predictions people make (and how they scored), and
the AI's own pre-game predictions for its track record. WAL mode lets
readers and a writer work concurrently.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing
from typing import Any, Dict, List, Optional, Tuple

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
    player TEXT NOT NULL,          -- who made it, e.g. "web:Aidan" or "tg:12345"
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
"""

PREDICTION_FIELDS = ("id", "player", "name", "league", "event_id", "espn_id", "home", "away", "start",
                     "created_at", "call", "ai", "status", "result", "user_points", "ai_points", "graded_at")
AI_FIELDS = ("league", "event_id", "espn_id", "home", "away", "start", "logged_at", "payload", "status",
             "result", "graded_at")
TICKET_FIELDS = ("day", "league", "tier", "logged_at", "first_start", "last_start", "payload", "status", "result",
                 "graded_at")
JSON_FIELDS = {"call", "ai", "result", "payload"}


def _row(fields: Tuple[str, ...], values: Tuple[Any, ...]) -> Dict[str, Any]:
    row = dict(zip(fields, values))
    for key in JSON_FIELDS & row.keys():
        if row[key] is not None:
            row[key] = json.loads(row[key])
    return row


class Store:
    def __init__(self, path: str) -> None:
        self.path = path
        self.holder = f"{os.getpid()}"
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10, isolation_level=None)

    def odds_snapshot(self, sport_key: str) -> Optional[Tuple[float, List[Dict[str, Any]]]]:
        """(fetched_at epoch seconds, games) of the league's last fetch, or None."""
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT fetched_at, payload FROM odds_snapshots WHERE sport_key = ?", (sport_key,)
            ).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def save_odds_snapshot(self, sport_key: str, games: List[Dict[str, Any]], fetched_at: float) -> None:
        with closing(self._connect()) as db:
            db.execute(
                "INSERT INTO odds_snapshots (sport_key, fetched_at, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(sport_key) DO UPDATE SET fetched_at = excluded.fetched_at, payload = excluded.payload",
                (sport_key, fetched_at, json.dumps(games)),
            )

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
