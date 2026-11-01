"""The capture's one door to the network (plan sections 3.3 and 9.3): ``PoliteHttp``.

``PoliteHttp(rates=None, *, clock=time.time, sleep=time.sleep, user_agent=None, timeout=20.0)``
    ``get(url, params=None, *, headers=None) -> HttpResponse(status, body, headers)``: one GET with urllib, through the
    proxy the environment names (``HTTPS_PROXY``/``HTTP_PROXY``/``NO_PROXY``) and its CA bundle (``SSL_CERT_FILE``,
    ``REQUESTS_CA_BUNDLE`` or ``CURL_CA_BUNDLE``). Any HTTP status is returned, never raised; a connection error, a
    timeout or a cut-off body raises ``Unreachable``. A gzip-encoded answer is decoded, so ``body`` is the payload's own
    bytes (a ``.csv.gz`` file stays as it is: that is its content, not its transfer encoding).

Politeness, per host:
- **A token bucket** (``RATES``): ``site.api.espn.com`` and ``sports.core.api.espn.com`` 1 request a second with a
  burst of 3, ``site.web.api.espn.com`` and ``statsapi.mlb.com`` 0.5 a second, the GitHub hosts (``github.com`` and
  ``*.githubusercontent.com``, one shared bucket: a release download redirects between them) 0.2 a second, any other
  host 0.5 a second. A request waits (``sleep``) for its token; the clock is injectable so tests run on a fake one.
- **A back-off** on 429 and 5xx answers and on unreachable hosts: exponential from ``BACKOFF_BASE`` seconds, capped at
  30 minutes (or longer when a 429's Retry-After says so, up to the cap), and a circuit breaker after ``BREAKER``
  failures in a row that keeps the host closed for the cap. A wait of up to ``MAX_WAIT`` seconds is slept; a longer one
  raises ``Backoff`` (an ``Unreachable``) at once, so the caller moves on and comes back later. Any other answer resets
  the host's count.
- **A stable User-Agent** (``USER_AGENT``; ``headers`` may override it for one request).
- **A deadline per GET** (``deadline``, default DEADLINE_TIMEOUTS x ``timeout`` = 60 s): urllib's timeout bounds each
  read only, so a host that trickles its answer raises ``Unreachable`` once the whole GET has taken longer, and counts
  as a failure for its back-off.

Budgets are the caller's: ``with http.guard(allow):`` calls ``allow(url)`` before every request made in that block
(this thread or task only), and ``allow`` raises ``Refused`` to stop it (the day's capture budget or the backfill's
night budget, counted in raw.db). It is called once per request, after the request's wait for its token, just before
the request goes out (so the backfill's "a forward job is due" check is never a token's wait old); a refused request
takes no token. The forward capture and the backfill share one PoliteHttp, so they share its buckets;
``active_guard()`` tells whose block a request runs in (the worker counts each request's outcome under its own kinds).

One door for the process: ``set_shared(http)`` makes a PoliteHttp the process's (the dashboard's capture registers its
own while it holds the capture lease; ``set_shared(None)`` clears it), and ``shared_acquire(url)`` waits for and takes
the URL's host token from it (``PoliteHttp.acquire``: no budget, no guard, nothing counted). The learn backfill's
HistHttp calls it before each of its own requests, so forward capture and the learn backfill together keep to each
host's rate; with no door set it does nothing.
"""

from __future__ import annotations

import contextlib
import contextvars
import email.utils
import gzip
import http.client
import logging
import os
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple, Union

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (personal sports analysis; raw capture)"
TIMEOUT = 20.0  # seconds a connection or one read may take ...
DEADLINE_TIMEOUTS = 3  # ... and a whole GET this many timeouts (60 s by default), however slowly the host trickles
GITHUB = "github"  # the shared bucket of github.com and *.githubusercontent.com
# host (or GITHUB) -> (requests a second, burst)
RATES: Dict[str, Tuple[float, int]] = {
    "site.api.espn.com": (1.0, 3),
    "sports.core.api.espn.com": (1.0, 3),
    "site.web.api.espn.com": (0.5, 1),
    "statsapi.mlb.com": (0.5, 1),
    GITHUB: (0.2, 1),
}
DEFAULT_RATE: Tuple[float, int] = (0.5, 1)
BACKOFF_BASE = 5.0          # seconds after the first failure in a row, doubling with each one after it
BACKOFF_CAP = 30 * 60.0     # the longest back-off, and how long the breaker keeps a host closed
BREAKER = 8                 # failures in a row that open the breaker
MAX_WAIT = 60.0             # waits up to this long are slept; a longer back-off raises Backoff
MAX_BODY = 256 * 1024 * 1024  # a body larger than this is cut off (Unreachable): no capture payload comes near it
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
TOKEN_EPSILON = 1e-9  # a token short by less than this (float rounding after a wait for it) is a whole one

Rate = Union[float, int, Tuple[float, int]]
Allow = Callable[[str], None]


class Unreachable(OSError):
    """The host could not be reached (connection error, timeout, cut-off body, or a back-off in force): there is no
    HTTP status to keep. The request may be tried again later."""


class Backoff(Unreachable):
    """The host is backing off after 429/5xx answers or failed connections (``until``: epoch seconds); nothing was
    sent."""

    def __init__(self, host: str, until: float, reason: str) -> None:
        super().__init__(f"{host} is backing off until {until:.0f} ({reason})")
        self.host = host
        self.until = until


class Refused(Exception):
    """A request was not made: the caller's budget is spent or its time window has closed (raised by a guard)."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)  # lower-case names

    @property
    def ok(self) -> bool:
        return 200 <= int(self.status) < 300


@dataclass
class _Bucket:
    rate: float
    burst: int
    tokens: float
    stamp: float
    failures: int = 0
    closed_until: float = 0.0


def is_github(host: str) -> bool:
    host = host.lower()
    return host in ("github.com", "githubusercontent.com") or host.endswith((".github.com", ".githubusercontent.com"))


def _rate(value: Rate) -> Tuple[float, int]:
    if isinstance(value, (tuple, list)):
        rate, burst = float(value[0]), int(value[1])
    else:
        rate, burst = float(value), 1
    if not rate > 0 or burst < 1:
        raise ValueError(f"a rate must be above 0 with a burst of 1 or more, not {value!r}")
    return rate, burst


def ca_bundle(env: Optional[Mapping[str, str]] = None) -> Tuple[Optional[str], Optional[str]]:
    """(CA file, CA directory) the environment names, or (None, None) for the system's own."""
    env = os.environ if env is None else env
    cafile = next((env[name] for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
                   if env.get(name) and os.path.isfile(env[name])), None)
    capath = env.get("SSL_CERT_DIR") if env.get("SSL_CERT_DIR") and os.path.isdir(env["SSL_CERT_DIR"]) else None
    return cafile, capath


def build_opener() -> urllib.request.OpenerDirector:
    """urllib's opener with the environment's proxies (ProxyHandler reads them) and CA bundle."""
    cafile, capath = ca_bundle()
    context = ssl.create_default_context(cafile=cafile, capath=capath)
    return urllib.request.build_opener(urllib.request.ProxyHandler(), urllib.request.HTTPSHandler(context=context))


def _decode(body: bytes, encoding: str) -> bytes:
    """The body without its transfer encoding (Content-Encoding gzip or deflate; ESPN gzips even its 400 answers,
    asked or not). A body that only *is* a gzip file (nflverse's .csv.gz: no Content-Encoding) is left as it is, and
    so is one that says gzip but isn't."""
    encoding = (encoding or "").strip().lower()
    try:
        if encoding in ("gzip", "x-gzip"):
            return gzip.decompress(body)
        if encoding == "deflate":
            try:
                return zlib.decompress(body)
            except zlib.error:
                return zlib.decompress(body, -zlib.MAX_WBITS)
    except (OSError, EOFError, zlib.error):
        return body
    return body


def _read(stream: Any, ends: Optional[float] = None, clock: Optional[Callable[[], float]] = None) -> bytes:
    """The whole body, at most MAX_BODY bytes, and done by ``ends`` on ``clock`` (a host that trickles its answer past
    the GET's deadline raises Unreachable: urllib's timeout only bounds each socket read). It reads with ``read1``
    (one socket read at a time, so the deadline is looked at between them: a plain ``read(n)`` keeps reading until it
    has n bytes, however slowly they come)."""
    read = getattr(stream, "read1", None) or stream.read
    chunks, size = [], 0
    while True:
        if ends is not None and clock is not None and clock() > ends:
            raise Unreachable("the answer took longer than its deadline")
        chunk = read(1 << 16)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > MAX_BODY:
            raise Unreachable(f"the answer is larger than {MAX_BODY} bytes")
        chunks.append(chunk)


def _retry_after(value: Optional[str], now: float) -> float:
    """Seconds a Retry-After header asks for (a number or an HTTP date), 0 when absent or unreadable."""
    if not value:
        return 0.0
    text = value.strip()
    if text.isdigit():
        return float(text)
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return 0.0
    return max(0.0, when.timestamp() - now) if when is not None else 0.0


class PoliteHttp:
    """GETs with a token bucket and a back-off per host (see the module's docstring). ``rates`` adds to or replaces
    RATES: host (or "github") -> requests a second, or (requests a second, burst). ``clock`` gives epoch seconds (the
    sources stamp ``fetched_at`` with it); ``sleep`` waits for a token."""

    def __init__(self, rates: Optional[Mapping[str, Rate]] = None, *, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, user_agent: Optional[str] = None,
                 timeout: float = TIMEOUT, deadline: Optional[float] = None) -> None:
        self.rates: Dict[str, Tuple[float, int]] = dict(RATES)
        for host, value in (rates or {}).items():
            self.rates[host.lower()] = _rate(value)
        self.clock = clock
        self.sleep = sleep
        self.user_agent = user_agent or USER_AGENT
        self.timeout = float(timeout)
        # A whole GET's deadline (default DEADLINE_TIMEOUTS timeouts), timed on the monotonic clock (a fake clock in
        # tests): a trickling host raises Unreachable and counts as a failure for the back-off
        self.deadline = float(deadline) if deadline is not None else DEADLINE_TIMEOUTS * self.timeout
        self._elapsed: Callable[[], float] = time.monotonic if clock is time.time else clock
        self.requests: Dict[str, int] = {}  # requests sent, per host
        # Called after every request sent with (url, HTTP status), or (url, None) when the host couldn't be reached
        # (the worker counts http_ok / http_error per UTC day in raw.db); its errors are logged, never raised
        self.on_result: Optional[Callable[[str, Optional[int]], None]] = None
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._opener: Optional[urllib.request.OpenerDirector] = None
        self._guard: contextvars.ContextVar[Optional[Allow]] = contextvars.ContextVar(f"capture_guard_{id(self)}",
                                                                                     default=None)

    # -- budgets --

    def active_guard(self) -> Optional[Allow]:
        """The guard of the block this request runs in (this thread or task), or None (the worker tells its forward
        requests from the backfill's by it, to count each under its own kinds)."""
        return self._guard.get()

    @contextlib.contextmanager
    def guard(self, allow: Allow) -> Iterator[None]:
        """Call ``allow(url)`` before every request made in the block, in this thread or task (it raises Refused to
        stop one). Guards nest: the inner one replaces the outer one for its block."""
        token = self._guard.set(allow)
        try:
            yield
        finally:
            self._guard.reset(token)

    # -- the buckets --

    def bucket_name(self, host: str) -> str:
        host = host.lower()
        if host in self.rates:
            return host
        return GITHUB if is_github(host) else host

    def _bucket(self, name: str, now: float) -> _Bucket:
        bucket = self._buckets.get(name)
        if bucket is None:
            rate, burst = self.rates.get(name, DEFAULT_RATE)
            bucket = self._buckets[name] = _Bucket(rate, burst, float(burst), now)
        return bucket

    def _open(self, name: str) -> None:
        """Raise Backoff when the host is backing off for longer than MAX_WAIT (before any budget is spent on it)."""
        with self._lock:
            bucket = self._buckets.get(name)
            if bucket is not None and bucket.closed_until - self.clock() > MAX_WAIT:
                raise Backoff(name, bucket.closed_until, f"{bucket.failures} failures in a row")

    def _refilled(self, name: str, now: float) -> _Bucket:
        """The host's bucket with the tokens it earned since it was last looked at (under the lock)."""
        bucket = self._bucket(name, now)
        bucket.tokens = min(float(bucket.burst), bucket.tokens + max(0.0, now - bucket.stamp) * bucket.rate)
        bucket.stamp = now
        return bucket

    def _reserve(self, name: str) -> float:
        """Take the host's next token; returns how long to wait for it. Raises Backoff when the host is backing off
        for longer than MAX_WAIT (no token is taken)."""
        with self._lock:
            now = self.clock()
            bucket = self._refilled(name, now)
            closed = max(0.0, bucket.closed_until - now)
            if closed > MAX_WAIT:
                raise Backoff(name, bucket.closed_until, f"{bucket.failures} failures in a row")
            bucket.tokens -= 1.0
            wait = 0.0 if bucket.tokens >= -TOKEN_EPSILON else -bucket.tokens / bucket.rate
            return max(wait, closed)

    def _ready_in(self, name: str) -> float:
        """How long until the host's next token is free (0 when one is), taking nothing; 0 while the host backs off
        (the guard is asked at once then: the backfill's refuses a host that is backing off rather than wait)."""
        with self._lock:
            now = self.clock()
            bucket = self._refilled(name, now)
            if bucket.closed_until > now or bucket.tokens >= 1.0 - TOKEN_EPSILON:
                return 0.0
            return (1.0 - bucket.tokens) / bucket.rate

    def acquire(self, url: str) -> None:
        """Wait for, and take, one token of ``url``'s host, as get() does before it sends (a short back-off is waited
        out; a longer one raises Backoff and takes no token). For a request another door of this process makes (the
        learn backfill's HistHttp, through shared_acquire): it spends no budget, asks no guard, and counts and
        settles nothing (that door's answers are its own)."""
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        if not host:
            raise ValueError(f"not a URL: {url!r}")
        wait = self._reserve(self.bucket_name(host))
        if wait > 0:
            self.sleep(wait)

    def _settle(self, name: str, failed: bool, retry_after: float = 0.0) -> None:
        with self._lock:
            bucket = self._bucket(name, self.clock())
            if not failed:
                bucket.failures, bucket.closed_until = 0, 0.0
                return
            bucket.failures += 1
            delay = BACKOFF_CAP if bucket.failures >= BREAKER else min(
                BACKOFF_CAP, BACKOFF_BASE * 2 ** (bucket.failures - 1))
            delay = min(BACKOFF_CAP, max(delay, retry_after))
            bucket.closed_until = max(bucket.closed_until, self.clock() + delay)

    def backoff_until(self, host: str) -> Optional[float]:
        """When the host's back-off ends (epoch seconds), or None when it isn't backing off."""
        with self._lock:
            bucket = self._buckets.get(self.bucket_name((host or "").lower()))
            if bucket is None or bucket.closed_until <= self.clock():
                return None
            return bucket.closed_until

    def backing_off(self, host: str) -> bool:
        """Whether the host (a name, or a URL) is backing off now, however briefly: a caller that must never wait
        (the backfill) stops instead."""
        if "://" in (host or ""):
            host = urllib.parse.urlsplit(host).hostname or ""
        return self.backoff_until(host) is not None

    # -- the request --

    def _send(self, request: urllib.request.Request) -> HttpResponse:
        """The actual GET (urllib); tests replace it. HTTP errors come back as responses."""
        if self._opener is None:
            self._opener = build_opener()
        ends = self._elapsed() + self.deadline
        try:
            with self._opener.open(request, timeout=self.timeout) as answer:
                body = _read(answer, ends, self._elapsed)
                status = int(getattr(answer, "status", None) or answer.getcode() or 0)
                headers = {key.lower(): value for key, value in answer.headers.items()}
        except urllib.error.HTTPError as exc:
            try:
                body = _read(exc, ends, self._elapsed) if exc.fp is not None else b""
            except Unreachable:
                raise
            except (OSError, http.client.HTTPException):
                body = b""
            headers = {key.lower(): value for key, value in (exc.headers or {}).items()}
            status = int(exc.code)
        return HttpResponse(status, _decode(body, headers.get("content-encoding", "")), headers)

    def get(self, url: str, params: Optional[Mapping[str, Any]] = None, *,
            headers: Optional[Mapping[str, str]] = None) -> HttpResponse:
        """One polite GET (see the module's docstring). Raises Unreachable (Backoff while the host backs off) or the
        guard's Refused; never an HTTP status."""
        if params:
            query = urllib.parse.urlencode([(key, str(value)) for key, value in params.items() if value is not None])
            url = f"{url}{'&' if '?' in url else '?'}{query}" if query else url
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        if not host:
            raise ValueError(f"not a URL: {url!r}")
        name = self.bucket_name(host)
        self._open(name)  # a host backing off costs no budget
        allow = self._guard.get()
        if allow is not None:
            # The token's wait comes first and the guard is asked after it, once (it spends the budget), just before
            # the request goes out: a forward job that falls due while a backfill request waits for its token has
            # that request refused, never sent ahead of it. A refused request takes no token.
            ready = self._ready_in(name)
            if ready > 0:
                self.sleep(ready)
            allow(url)
        wait = self._reserve(name)  # the token (another thread may have taken the one waited for: wait again)
        if wait > 0:
            self.sleep(wait)
        sent = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip", "Accept": "*/*"}
        sent.update(dict(headers or {}))
        request = urllib.request.Request(url, headers=sent, method="GET")
        with self._lock:
            self.requests[host] = self.requests.get(host, 0) + 1
        try:
            response = self._send(request)
        except Unreachable:
            self._settle(name, True)
            self._observe(url, None)
            raise
        except (OSError, http.client.HTTPException, ValueError, zlib.error, EOFError) as exc:
            # URLError, timeouts, resets, TLS failures, a truncated or undecodable body
            self._settle(name, True)
            self._observe(url, None)
            raise Unreachable(f"{host}: {type(exc).__name__}: {exc}") from exc
        failed = int(response.status) in RETRY_STATUSES or int(response.status) >= 500
        self._settle(name, failed, _retry_after(response.headers.get("retry-after"), self.clock())
                     if int(response.status) == 429 else 0.0)
        self._observe(url, int(response.status))
        return response

    def _observe(self, url: str, status: Optional[int]) -> None:
        if self.on_result is None:
            return
        try:
            self.on_result(url, status)
        except Exception:  # noqa: BLE001 - counting never stops a fetch
            log.exception("Couldn't count a capture request")


# -- one door for the process --
# The PoliteHttp whose per-host buckets every door of this process to the network waits on (the dashboard's capture
# sets its own while it holds the capture lease): the learn backfill's HistHttp takes a token here before each of its
# requests, so the capture thread, the slate's capture thread and the learner thread together keep to each host's rate.
# One per process (a module global, guarded by its lock; a token is taken under the PoliteHttp's own lock, and the wait
# for it is slept holding neither).
_SHARED: Optional[PoliteHttp] = None
_SHARED_LOCK = threading.Lock()


def set_shared(http: Optional[PoliteHttp]) -> None:
    """Make ``http`` the process's door (None clears it)."""
    global _SHARED
    with _SHARED_LOCK:
        _SHARED = http


def shared() -> Optional[PoliteHttp]:
    """The process's door, or None."""
    with _SHARED_LOCK:
        return _SHARED


def shared_acquire(url: str) -> None:
    """Wait for, and take, ``url``'s host token from the process's door (PoliteHttp.acquire), when one is set; nothing
    when none is. Never raises: when the door can't give a token (its host backs off for longer than MAX_WAIT, a URL
    without a host) the caller's request goes on as it would without the door."""
    door = shared()
    if door is None:
        return
    try:
        door.acquire(url)
    except Exception:  # noqa: BLE001 - another door's request is never stopped here
        log.debug("No shared token for %s", url, exc_info=True)


__all__ = ["PoliteHttp", "HttpResponse", "Unreachable", "Backoff", "Refused", "RATES", "DEFAULT_RATE", "USER_AGENT",
           "is_github", "ca_bundle", "build_opener", "set_shared", "shared", "shared_acquire"]
