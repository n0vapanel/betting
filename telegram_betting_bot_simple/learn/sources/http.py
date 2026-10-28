"""The backfill's one door to the network: a gentle GET, at least ``pace`` seconds apart, each counted.

``HistHttp.json(url, params)`` / ``HistHttp.text(url, params)`` are only ever called inside a TTLCache loader (see
``cached``): every source reads through the shared cache with a kind from data/cache.py's policy table
("hist_file", "hist_list", "hist_game"), so a value already fetched (by this process or the other one) costs no
request. Before each request ``allow()`` (the Backfill's budget and time-window check) may refuse it with
``Refused``; after it ``on_request(url)`` counts it. ``transport(url, params, timeout) -> (status, body bytes)``
is the actual GET (requests by default); tests hand in one that serves fixtures.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Hashable, Mapping, Optional, Tuple

import requests

from ...data.cache import TTLCache, ttl_for

USER_AGENT = "Mozilla/5.0 (personal sports analysis; history backfill)"
TIMEOUT = 30.0
PACE = 2.0  # seconds between two requests, at least

Transport = Callable[[str, Mapping[str, Any], float], Tuple[int, bytes]]


class Refused(Exception):
    """A request was not made: the night's budget is spent or the time window has closed."""


class HttpError(RuntimeError):
    """A source answered with an HTTP error (``status`` holds the code)."""

    def __init__(self, status: int, url: str) -> None:
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.url = url


def requests_transport(url: str, params: Mapping[str, Any], timeout: float) -> Tuple[int, bytes]:
    """One GET with requests (redirects followed: football-data's www. host answers 302)."""
    response = requests.get(url, params=dict(params or {}), timeout=timeout, headers={"User-Agent": USER_AGENT})
    return response.status_code, response.content or b""


class HistHttp:
    """GETs at least ``pace`` seconds apart (``clock``/``sleep`` replace the monotonic clock and the wait, for tests).
    ``allow()`` is called before each request (raise Refused to stop it) and ``on_request(url)`` after it."""

    def __init__(self, *, pace: float = PACE, clock: Optional[Callable[[], float]] = None,
                 sleep: Optional[Callable[[float], None]] = None, allow: Optional[Callable[[], None]] = None,
                 on_request: Optional[Callable[[str], None]] = None, timeout: float = TIMEOUT,
                 transport: Optional[Transport] = None) -> None:
        self.pace = max(0.0, float(pace))
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self.allow = allow
        self.on_request = on_request
        self.timeout = timeout
        self.transport: Transport = transport or requests_transport
        self.requests = 0
        self._last: Optional[float] = None
        self._lock = threading.Lock()

    def _get(self, url: str, params: Optional[Mapping[str, Any]]) -> bytes:
        with self._lock:
            if self.allow is not None:
                self.allow()
            if self._last is not None:
                wait = self.pace - (self.clock() - self._last)
                if wait > 0:
                    self.sleep(wait)
            try:
                status, body = self.transport(url, dict(params or {}), self.timeout)
            finally:
                self._last = self.clock()
                self.requests += 1
                if self.on_request is not None:
                    self.on_request(url)
        if not 200 <= int(status) < 300:
            raise HttpError(int(status), url)
        return body

    def json(self, url: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        return json.loads(self._get(url, params).decode("utf-8-sig"))

    def text(self, url: str, params: Optional[Mapping[str, Any]] = None) -> str:
        body = self._get(url, params)
        try:  # football-data's files are UTF-8, some with a byte-order mark; older ones Windows-1252
            return body.decode("utf-8-sig")
        except UnicodeDecodeError:
            return body.decode("cp1252", errors="replace")


def cached(cache: TTLCache, kind: str, key: Hashable, loader: Callable[[], Any], state: Optional[str] = None,
           life: Optional[Callable[[Any], Any]] = None) -> Any:
    """``loader()`` through the shared cache under ``kind`` (its policy for ``state``; ``life(value)`` may give a
    loaded value a shorter one, e.g. a game that turned out not to be over). No negative caching (the backfill paces
    its own retries); the parsed value is saved to the shared table."""
    policy = ttl_for(kind, state)
    options = {"life": life} if life is not None else {}
    return cache.get(key, policy.ttl, loader, stale=policy.stale, error_ttl=0, persist=True, kind=kind, **options)
