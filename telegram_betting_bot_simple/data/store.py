"""A SQLite file shared by the bot and the dashboard on the same server.

Holds each league's latest odds snapshot (so the two processes never pay
twice for the same fetch and nothing is lost on restart), a log of The
Odds API's quota headers, and short leases so only one process refreshes
a league at a time. WAL mode lets readers and a writer work concurrently.
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
"""


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
