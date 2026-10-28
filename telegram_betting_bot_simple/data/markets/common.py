"""What the market sources share: the Quote shape and the thin rule, small parsing helpers, a paced HTTP getter,
and the Keeper that caches each source's copies (on data/cache.py's TTLCache, so the bot and the dashboard share
them), persists the last good one and never raises into callers.

Quote = {"p": float,                       # implied chance: bid/ask midpoint when both exist, else the last price
         "bid": float|None, "ask": float|None, "last": float|None,
         "liquidity": float|None, "volume": float|None,   # US dollars where the venue reports them
         "thin": bool}                     # spread > 0.06, or liquidity < $1,000 when liquidity is known

The thin thresholds are the quality filter in research_notes/Sports data sources and analysis UX/
odds_and_predictions.md §2 ("drop a market when bestAsk - bestBid > 0.06, liquidity < $1k"). A quote without a
bid or without an ask has no spread to judge and is thin too (only one side can be traded at that price).
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import time
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Tuple

import requests

from ..cache import TTLCache

log = logging.getLogger(__name__)

THIN_SPREAD = 0.06  # an ask more than 6 cents above the bid is a thin market ...
THIN_LIQUIDITY = 1000.0  # ... and so is one with less than $1,000 of liquidity (when the venue reports it)
USER_AGENT = "Mozilla/5.0 (private sports desk; cached, low volume)"
TIMEOUT = 15


# --- numbers and times ----------------------------------------------------------------------------------

def num(value: Any) -> Optional[float]:
    """A finite float from a number or numeric string ("0.4200"), else None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def price(value: Any) -> Optional[float]:
    """A price strictly between 0 and 1 (a venue's 0 or 1 means "no order there"), else None."""
    number = num(value)
    return number if number is not None and 0.0 < number < 1.0 else None


def iso(ts: Optional[float]) -> Optional[str]:
    """Epoch seconds as "2026-10-27T23:30:00Z"."""
    if ts is None:
        return None
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text: Any) -> Optional[float]:
    """Epoch seconds from the venues' time strings ("2026-10-27T23:30:00Z", "2026-10-27 23:30:00+00",
    "2026-10-27T23:30Z"), or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    value = text.strip().replace(" ", "T", 1).replace("Z", "+00:00")
    if value.endswith("+00"):
        value += ":00"
    try:
        moment = datetime.datetime.fromisoformat(value)
    except ValueError:
        try:
            moment = datetime.datetime.strptime(value[:16], "%Y-%m-%dT%H:%M")
        except ValueError:
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    return moment.timestamp()


def iso_of(text: Any) -> Optional[str]:
    """A venue's time string re-written as UTC ISO ("...Z"), or None."""
    return iso(parse_time(text))


# --- quotes ---------------------------------------------------------------------------------------------

def is_thin(bid: Optional[float], ask: Optional[float], liquidity: Optional[float]) -> bool:
    """The thin rule: spread > 0.06, no bid or no ask, or liquidity under $1,000 when it is known."""
    if bid is None or ask is None or ask - bid > THIN_SPREAD + 1e-9:
        return True
    return liquidity is not None and liquidity < THIN_LIQUIDITY


def make_quote(bid: Any, ask: Any, last: Any = None, liquidity: Any = None, volume: Any = None,
               fallback: Any = None) -> Optional[Dict[str, Any]]:
    """A Quote from a binary contract's best bid/ask (0-1), its last trade and size. ``p`` is the bid/ask
    midpoint when both exist, else the last price, else ``fallback`` (a venue's displayed price); None when
    none of them gives a chance. A crossed book (bid above ask) is treated as having no bid."""
    bid, ask, last = price(bid), price(ask), price(last)
    if bid is not None and ask is not None and bid > ask:
        bid = None
    liquidity, volume = num(liquidity), num(volume)
    if bid is not None and ask is not None:
        p = (bid + ask) / 2
    elif last is not None:
        p = last
    else:
        p = price(fallback)
    if p is None:
        return None
    return {"p": p, "bid": bid, "ask": ask, "last": last, "liquidity": liquidity, "volume": volume,
            "thin": is_thin(bid, ask, liquidity)}


def complement(quote: Mapping[str, Any]) -> Dict[str, Any]:
    """The other side of a binary contract: buying "No" at 1 - bid, selling it at 1 - ask."""
    flip = lambda value: None if value is None else 1.0 - value  # noqa: E731
    return {"p": 1.0 - quote["p"], "bid": flip(quote.get("ask")), "ask": flip(quote.get("bid")),
            "last": flip(quote.get("last")), "liquidity": quote.get("liquidity"), "volume": quote.get("volume"),
            "thin": bool(quote.get("thin"))}


def normalise(quotes: Mapping[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Mutually exclusive outcomes priced as separate contracts (Kalshi's two or three, Polymarket's soccer
    three) sum to a little over or under 1; scale each ``p`` so they sum to exactly 1. Bid, ask and last stay
    as the venue quotes them."""
    total = sum(quote["p"] for quote in quotes.values())
    if total <= 0:
        return {side: dict(quote) for side, quote in quotes.items()}
    return {side: {**quote, "p": quote["p"] / total} for side, quote in quotes.items()}


def rounded(quote: Mapping[str, Any]) -> Dict[str, Any]:
    """A quote with its numbers rounded for storage (chances to 4 places, money to cents)."""
    out = dict(quote)
    for field in ("p", "bid", "ask", "last"):
        if out.get(field) is not None:
            out[field] = round(out[field], 4)
    for field in ("liquidity", "volume"):
        if out.get(field) is not None:
            out[field] = round(out[field], 2)
    return out


def pick_line(candidates: Iterable[Tuple[float, Dict[str, Any], Dict[str, Any], Optional[float]]]
              ) -> Optional[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
    """The main line among (point, first-side quote, second-side quote, liquidity) candidates: the one whose
    chance is closest to 50/50, preferring lines that are not thin, then the most liquid."""
    best = None
    for point, first, second, liquidity in candidates:
        rank = (bool(first["thin"]), abs(first["p"] - 0.5), -(liquidity or 0.0))
        if best is None or rank < best[0]:
            best = (rank, point, first, second)
    return None if best is None else (best[1], best[2], best[3])


# --- HTTP ---------------------------------------------------------------------------------------------

class HttpError(RuntimeError):
    """A failed request, named by status only (never the URL's query string)."""

    def __init__(self, status: Optional[int], what: str) -> None:
        super().__init__(f"{what} answered HTTP {status}" if status else f"{what} could not be reached")
        self.status = status


Fetch = Callable[[str, Optional[Dict[str, Any]]], Tuple[int, Any]]


def http_get(url: str, params: Optional[Dict[str, Any]] = None, text: bool = False) -> Tuple[int, Any]:
    """GET a public URL: (status, parsed JSON or text; None when unreadable). Network errors raise HttpError
    naming only the error type."""
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        raise HttpError(None, type(exc).__name__) from None
    if text:
        return response.status_code, response.text
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, None


# --- the keeper: one source's copies, on the shared TTLCache's rules ------------------------------------------

class Keeper:
    """Copies of one source's results, each loaded at most once per TTL, on data/cache.py's TTLCache, so the
    markets sources follow the same rules as every other fetch:

    - A fresh copy (younger than the TTL) is served from memory.
    - A stale one is refreshed by calling the loader. Before that, the store's shared copy is read: another process
      (the bot and the dashboard share the store) may already have refreshed it, and after a restart it is the
      starting point, so a restart has data at once and doesn't refetch what is still fresh.
    - With a store, a short lease keeps two processes from fetching the same thing at once; the one that loses
      serves its last good copy (with none, it fetches too rather than show nothing).
    - When the loader fails, the last good copy is served (or ``empty`` when there has never been one), only the
      error's type is logged, nothing is raised, and the key isn't retried for a while (30 s, doubling to 15 min).
    - Every good copy is saved to the store's shared cache table, so a restart has data. Copies Stage 2 saved as
      named blobs in the slates table are read once, the first time a key has no other copy.
    """

    RETRY_BASE = 30.0
    RETRY_CAP = TTLCache.ERROR_CAP  # 15 minutes
    LEASE = 90.0
    KEEP = 7 * 24 * 3600.0  # how long put()'s copies stay in the shared table

    def __init__(self, source: str, store: Any = None, clock: Optional[Callable[[], float]] = None,
                 cache: Optional[TTLCache] = None) -> None:
        self.source = source
        self.store = store
        self.clock = clock or time.time
        # Its own cache (on the source's clock, which tests drive) unless one is given; lease_wait=0: a process
        # that finds the other loading serves its copy at once instead of holding up the slate's pass
        self.cache = cache or TTLCache(store, name=source, clock=clock, lease_seconds=self.LEASE, lease_wait=0,
                                       error_detail=False)
        self._upgraded: set = set()

    def _key(self, key: Hashable) -> Tuple[Any, ...]:
        return ("markets", self.source, *(key if isinstance(key, tuple) else (key,)))

    @staticmethod
    def _name(source: str, key: Hashable) -> str:
        """Stage 2's blob name for a copy in the slates table."""
        parts = key if isinstance(key, tuple) else (key,)
        return "markets:" + ":".join([source, *(str(part) for part in parts)])

    def _upgrade(self, key: Hashable) -> None:
        """The first time ``key`` is asked for with no copy in memory or the shared table, start it from the copy
        Stage 2 saved in the slates table, if there is one."""
        if key in self._upgraded:
            return
        self._upgraded.add(key)
        if self.store is None or not hasattr(self.store, "slate") or self.cache.peek(self._key(key)) is not None:
            return
        try:
            saved = self.store.slate(self._name(self.source, key))
            if saved:
                fetched_at, _etag, body = saved
                self.cache.seed(self._key(key), json.loads(body), float(fetched_at), kind=self.source)
        except Exception as exc:  # a broken store must not stop the source
            log.warning("%s: reading the saved copy failed (%s)", self.source, type(exc).__name__)

    def peek(self, key: Hashable) -> Optional[Tuple[float, Any]]:
        """(fetched_at, value) of the last good copy in memory or in the store, without fetching."""
        self._upgrade(key)
        try:
            return self.cache.peek(self._key(key))
        except Exception as exc:
            log.warning("%s: reading the saved copy failed (%s)", self.source, type(exc).__name__)
            return None

    def seed(self, key: Hashable, fetched_at: float, value: Any) -> None:
        """Start ``key`` from a copy saved elsewhere, unless a copy is already held."""
        self.cache.seed(self._key(key), value, float(fetched_at), kind=self.source)

    def put(self, key: Hashable, value: Any) -> float:
        """Keep ``value`` as the good copy of ``key`` now (and save it); returns its time."""
        return self.cache.put(self._key(key), value, ttl=self.KEEP, persist=True, kind=self.source)

    def fetched_at(self, key: Hashable) -> Optional[float]:
        copy = self.peek(key)
        return copy[0] if copy else None

    def get(self, key: Hashable, ttl: float, loader: Callable[[], Any], empty: Any = None, stale: float = 0.0) -> Any:
        """The copy of ``key``: fresh from memory or the store, else freshly loaded, else the last good one,
        else ``empty``. Never raises."""
        value, _ = self.get_with_time(key, ttl, loader, empty, stale)
        return value

    def get_with_time(self, key: Hashable, ttl: float, loader: Callable[[], Any], empty: Any = None,
                      stale: float = 0.0) -> Tuple[Any, Optional[float]]:
        """As get, with the copy's fetch time (None for ``empty``). ``stale`` serves a copy that much past its TTL
        at once while one background refresh runs."""
        self._upgrade(key)
        ran = []

        def load() -> Any:
            ran.append(True)
            return loader()

        try:
            return self.cache.get_with_time(self._key(key), ttl, load, stale=stale, error_ttl=self.RETRY_BASE,
                                            backoff_cap=self.RETRY_CAP, persist=True, kind=self.source, backoff=True)
        except Exception as exc:
            if ran:  # a remembered failure (backing off) is not logged again
                log.warning("%s %s: refresh failed (%s); nothing saved yet", self.source, key, type(exc).__name__)
            return empty, None


def season_now(sport: str, now: Optional[float] = None) -> int:
    """ESPN's season year for a sport on a date: basketball and hockey seasons are named for the year they end
    (2026-27 is 2027) from August; football seasons for the year they start (January and February games belong
    to the season before)."""
    today = datetime.datetime.fromtimestamp(now if now is not None else time.time(), datetime.timezone.utc)
    if sport in ("basketball", "hockey"):
        return today.year + 1 if today.month >= 8 else today.year
    return today.year if today.month >= 3 else today.year - 1


def text(value: Any, limit: int = 80) -> Optional[str]:
    """Outside text, tidied for storage: no control characters, single spaces, at most ``limit`` characters.
    The app still escapes it wherever it is shown."""
    if value is None or isinstance(value, (dict, list)):
        return None
    cleaned = "".join(ch if ch.isprintable() else " " for ch in str(value))
    cleaned = " ".join(cleaned.split())
    return cleaned[:limit] or None


def as_list(value: Any) -> List[Any]:
    """A JSON list, or a JSON string holding one (Polymarket's outcomes/outcomePrices), else []."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []
