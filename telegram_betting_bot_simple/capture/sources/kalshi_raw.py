"""Raw Kalshi series snapshots for the capture store: the confirmed corner, shot and save series, archive only (v4 P1,
plan sections 3.3 and 3.13). Normalising them into odds rows comes in P6. No API key: Kalshi's market data needs none.

    GET https://api.elections.kalshi.com/trade-api/v2/events?series_ticker={TICKER}&status=open
        &with_nested_markets=true&with_milestones=true&limit=200[&cursor=...]

This is the call data/markets/kalshi.py makes for game prices. It returns every open event of the series with its
markets (bid, ask, last, volume, strike) and milestones (Kalshi's game records, with the real start time and the
title P6 needs to tie an event to its game). Pages continue while the answer's ``cursor`` is non-empty, up to
``MAX_PAGES``.

The confirmed series (``DEFAULT_SERIES``; ``CAPTURE_KALSHI_SERIES`` overrides it, and an empty value turns Kalshi
capture off):
- EPL corners ``KXEPLCORNERS``;
- Champions League shots ``KXUCLSOT``;
- Champions League saves ``KXUCLSAVE``.

The plan marks these three "(confirmed)" in section 4.1 (the soccer.corners, soccer.shots and soccer.saves rows) and
names them in section 3.13 (from [R section 1]). The ticker strings come from research_notes/Sports data sources and
analysis UX/odds_and_predictions.md, from a live ``GET /series?category=Sports`` on 2026-10-26, and are unverified
beyond those notes (nothing here has called Kalshi since). That list also shows ``KXUCLCORNERS``, which the plan
doesn't mark confirmed, so it isn't a default.

Storage (the worker's): one snapshot per series per 10 minutes at most, when a game of its league (``league_of``)
reaches a pre horizon. Each page is stored under ``static:kalshi:<TICKER>`` (``game_key``) with state "pre", spending
the "prices" budget.

Rate. Anonymous reads are throttled after a few large calls (data/markets/kalshi.py). The slate's markets worker
reads Kalshi from the same dashboard process, paced at one request a second. So with the real clock, each request
here first waits its turn on that process-wide pacer, and the two together stay at one a second: capture can't get
the markets brain rate-limited. An ``http`` whose clock reads another time (a test's fake one) isn't paced here;
its own buckets pace it on that clock. ``pacer=`` overrides either.

Shape (capture/sources/__init__.py): ``series(http, ticker, *, page=1, cursor=None, pacer=None) -> RawFetch`` (one
page) and ``series_all(http, ticker, *, max_pages=MAX_PAGES, pacer=None) -> Iterator[RawFetch]`` (the pager).
Neither raises on an HTTP status; connection errors raise ``capture.http.Unreachable``. ``params`` is ``{"status":
"open", "limit": 200, "page": n}``: the canonical query. The cursor is opaque and changes between snapshots, so it is
sent but never kept in params. The ticker is in the game key.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple

from ...data.markets import kalshi as _kalshi
from . import RawFetch, fetch, static_key

log = logging.getLogger(__name__)

SOURCE = "kalshi"
ENDPOINT = "events"
BASE = _kalshi.BASE  # https://api.elections.kalshi.com/trade-api/v2
URL = f"{BASE}/events"
HOST = "api.elections.kalshi.com"
LIMIT = 200  # events a page: Kalshi's maximum
MAX_PAGES = 5  # as data/markets/kalshi.py
DEFAULT_SERIES: Tuple[str, ...] = ("KXEPLCORNERS", "KXUCLSOT", "KXUCLSAVE")
# The league whose games a series prices (the worker snapshots a series when one of them reaches a pre horizon).
# Tickers not listed here fall back to the longest winner-series prefix in data/markets/kalshi.py LEAGUES.
SERIES_LEAGUES: Dict[str, str] = {
    "KXEPLCORNERS": "soccer_epl",
    "KXUCLCORNERS": "soccer_uefa_champs_league",
    "KXUCLSOT": "soccer_uefa_champs_league",
    "KXUCLSAVE": "soccer_uefa_champs_league",
}
TICKER = re.compile(r"KX[A-Z0-9]{2,40}")
_WINNER = re.compile(r"(GAME|MATCH|FIGHT)$")


OFF_WORDS = frozenset({"none", "off", "0", "false", "no", "-"})


def parse_series(text: Optional[str], default: Iterable[str] = DEFAULT_SERIES) -> Tuple[str, ...]:
    """CAPTURE_KALSHI_SERIES as tickers: comma- (or space-) separated, upper-cased, de-duplicated, in order. None
    means ``default``; an empty value, "none" or "off" means none (Kalshi capture off). A word that isn't a Kalshi
    series ticker (``KX`` and 2-40 letters or digits) is dropped with a warning."""
    if text is None:
        return tuple(default)
    if str(text).strip().lower() in OFF_WORDS:
        return ()
    out = []
    for word in re.split(r"[\s,]+", str(text).strip()):
        ticker = word.strip().upper()
        if not ticker:
            continue
        if not TICKER.fullmatch(ticker):
            log.warning("CAPTURE_KALSHI_SERIES: %r isn't a Kalshi series ticker; left out", word[:60])
            continue
        if ticker not in out:
            out.append(ticker)
    return tuple(out)


def _ticker(ticker: Any) -> str:
    text = str(ticker if ticker is not None else "").strip().upper()
    if not TICKER.fullmatch(text):
        raise ValueError(f"not a Kalshi series ticker: {ticker!r}")
    return text


def game_key(ticker: Any) -> str:
    """The raw store's key for a series' snapshots: ``static:kalshi:<TICKER>``."""
    return static_key(SOURCE, _ticker(ticker))


def league_of(ticker: Any) -> Optional[str]:
    """The app league whose games the series prices: SERIES_LEAGUES, else the longest winner-series prefix of
    data/markets/kalshi.py LEAGUES ("KXEPLBTTS" -> soccer_epl from KXEPLGAME). None when nothing matches."""
    try:
        text = _ticker(ticker)
    except ValueError:
        return None
    if text in SERIES_LEAGUES:
        return SERIES_LEAGUES[text]
    best: Optional[Tuple[int, str]] = None
    for league, entry in _kalshi.LEAGUES.items():
        prefix = _WINNER.sub("", entry[0])
        if len(prefix) > 2 and text.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), league)
    return best[1] if best else None


REAL_CLOCK_SLACK = 300.0  # an http clock this close to the real time is the real one (a test's fake clock isn't)


def _real_clock(http: Any) -> bool:
    """Whether ``http`` runs on the real time: the dashboard's PoliteHttp, however its clock is passed."""
    clock = getattr(http, "clock", None)
    if not callable(clock):
        clock = getattr(http, "_clock", None)
    if not callable(clock) or clock is time.time:
        return True
    try:
        return abs(float(clock()) - time.time()) < REAL_CLOCK_SLACK
    except (TypeError, ValueError):
        return True


def _pacer(http: Any, pacer: Any) -> Any:
    if pacer is not None:
        return pacer
    return _kalshi._PROCESS_PACER if _real_clock(http) else None  # the slate's markets worker's own pacer


def series(http: Any, ticker: Any, *, page: int = 1, cursor: Optional[str] = None, pacer: Any = None) -> RawFetch:
    """One page of the series' open events, with their markets and milestones (see the module's docstring)."""
    text = _ticker(ticker)
    if int(page) < 1:
        raise ValueError("pages start at 1")
    query: Dict[str, Any] = {"series_ticker": text, "status": "open", "with_nested_markets": "true",
                             "with_milestones": "true", "limit": LIMIT}
    if cursor:
        query["cursor"] = str(cursor)
    wait = _pacer(http, pacer)
    if wait is not None:
        wait.wait()
    return fetch(http, URL, source=SOURCE, endpoint=ENDPOINT, params={"status": "open", "limit": LIMIT,
                                                                       "page": int(page)}, query=query)


def next_cursor(raw: RawFetch) -> Optional[str]:
    """The cursor of the next page when the answer is a 2xx page of events with one; None otherwise (the last page,
    an error, a body that isn't JSON or Kalshi's ``{"error": ...}``), which ends the pager."""
    if not raw.ok:
        return None
    try:
        data = raw.json()
    except ValueError:
        return None
    if not isinstance(data, Mapping) or data.get("error") or not isinstance(data.get("events"), list):
        return None
    cursor = data.get("cursor")
    return str(cursor) if isinstance(cursor, str) and cursor.strip() else None


def series_all(http: Any, ticker: Any, *, max_pages: int = MAX_PAGES, pacer: Any = None) -> Iterator[RawFetch]:
    """Every page of the series' open events: page 1, then one per cursor, at most ``max_pages``. A page that isn't
    a 2xx page of events is yielded and ends the walk (the caller sees its status)."""
    text = _ticker(ticker)  # a bad ticker fails before any request
    cursor: Optional[str] = None
    for page in range(1, max(1, int(max_pages)) + 1):
        raw = series(http, text, page=page, cursor=cursor, pacer=pacer)
        yield raw
        cursor = next_cursor(raw)
        if cursor is None:
            return


__all__ = ["SOURCE", "ENDPOINT", "URL", "HOST", "LIMIT", "MAX_PAGES", "DEFAULT_SERIES", "SERIES_LEAGUES",
           "parse_series", "game_key", "league_of", "series", "next_cursor", "series_all"]
