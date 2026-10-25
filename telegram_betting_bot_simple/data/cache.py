"""A small thread-safe time-to-live cache shared by the data clients.

Each key is loaded at most once per TTL even when many requests ask for
it at the same time, and if a refresh fails the last good value is
served instead of an error (stale data beats no data for odds and stats).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Hashable, Optional, Tuple


class TTLCache:
    def __init__(self) -> None:
        # key -> (monotonic load time, wall-clock load time, value)
        self._entries: Dict[Hashable, Tuple[float, float, Any]] = {}
        self._key_locks: Dict[Hashable, threading.Lock] = {}
        self._lock = threading.Lock()

    def _lookup(self, key: Hashable, ttl_seconds: float) -> Tuple[bool, Optional[Tuple[float, float, Any]]]:
        with self._lock:
            entry = self._entries.get(key)
        return bool(entry and time.monotonic() - entry[0] < ttl_seconds), entry

    def get(self, key: Hashable, ttl_seconds: float, loader: Callable[[], Any]) -> Any:
        """Return the cached value for ``key``, calling ``loader`` when it is missing or expired."""
        fresh, entry = self._lookup(key, ttl_seconds)
        if fresh:
            return entry[2]
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            # Another thread may have refreshed it while this one waited
            fresh, entry = self._lookup(key, ttl_seconds)
            if fresh:
                return entry[2]
            try:
                value = loader()
            except Exception as exc:
                if entry is None:
                    raise
                logging.warning("Refreshing %r failed (%s); serving the cached copy", key, exc)
                return entry[2]
            with self._lock:
                self._entries[key] = (time.monotonic(), time.time(), value)
            return value

    def loaded_at(self, key: Hashable) -> Optional[float]:
        """Wall-clock time (epoch seconds) the cached value was loaded, or None."""
        with self._lock:
            entry = self._entries.get(key)
        return entry[1] if entry else None

    def age(self, key: Hashable) -> Optional[float]:
        """Seconds since the cached value was loaded, or None when not cached."""
        with self._lock:
            entry = self._entries.get(key)
        return time.monotonic() - entry[0] if entry else None

    def invalidate(self, key: Hashable) -> None:
        with self._lock:
            self._entries.pop(key, None)
