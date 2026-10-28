"""Game odds from The Odds API for the dashboard and bot.

Fetches moneyline, spread and total odds in one request per league, reuses
them from a store shared by the bot and dashboard, tracks the remaining
quota from the response headers, and summarises each game's markets into
the best available price and the bookmakers' consensus fair probability
for every outcome.

The board asks for a hand-picked list of at most ten bookmakers
(``bookmakers=``, ODDS_BOOKMAKERS) instead of a region: "every group of 10
bookmakers is the equivalent of 1 region", so a board still costs one credit
per market, 3 in all. The list puts Pinnacle and two exchanges (SHARP_BOOKS)
beside the US books, and every book's prices are kept in the snapshot. The
books have two roles (BookRoles):

- sharp books (Pinnacle, Betfair, Matchbook) anchor the markets brain's
  consensus (``consensus_inputs``) and are never offered to anyone;
- price books (ODDS_PRICE_BOOKS: by default every listed book that is neither
  sharp nor an exchange) are the books a person can actually bet at. "Best
  price" here, and so every ticket the app prices or builds (the slip, the
  AI's Safe/Risky/Crazy bets, the target builder, a pasted slip's "better
  price"), uses only them.

Margin-free ("fair") chances still average every book at the consensus line,
sharp ones included: they are the market's view, not a price on offer.
"""

from __future__ import annotations

import datetime
import logging
import math
import re
import statistics
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import requests

from ..engine.odds_math import remove_vig
from .cache import TTLCache
from .store import Store
from .utils import policy

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
MARKETS = ("h2h", "spreads", "totals")
# Bookmakers update about once a minute, so forced refreshes closer together than this waste quota
MIN_REFRESH_SECONDS = 300
# The board's bookmakers (config.py's ODDS_BOOKMAKERS default; checked against The Odds API's bookmaker list on
# 27 October 2026). Ten books are one region: 3 credits a board, as regions=us was
DEFAULT_BOOKMAKERS: Tuple[str, ...] = ("pinnacle", "betfair_ex_eu", "matchbook", "draftkings", "fanduel", "betmgm",
                                       "williamhill_us", "espnbet", "betrivers", "bovada")
# The consensus anchor only (engine/consensus.py's SHARP_BOOKS): Pinnacle, whose public feed "may incur a delay",
# beside two exchanges. Nobody is offered their prices
DEFAULT_SHARP_BOOKS: Tuple[str, ...] = ("pinnacle", "betfair_ex_eu", "matchbook")
BOOKS_PER_REGION = 10
# Exchanges and prediction markets The Odds API lists as bookmakers: never a "best price" (most people can't bet
# there, and the prediction markets come into the brain through their own feeds)
EXCHANGE_BOOKS: FrozenSet[str] = frozenset({
    "betfair_ex_eu", "betfair_ex_uk", "betfair_ex_au", "matchbook", "smarkets", "kalshi", "polymarket",
    "polymarket_us", "novig", "prophetx", "betopenly", "limitless",
})
# The Odds API's titles for the listed books, so a price known only by its bookmaker's title (a ticket leg)
# can be told apart too. Titles seen in fetched boards are learned as well (normalize_event)
KNOWN_TITLES: Dict[str, str] = {
    "pinnacle": "pinnacle", "betfair": "betfair_ex_eu", "betfair exchange": "betfair_ex_eu",
    "matchbook": "matchbook", "smarkets": "smarkets", "draftkings": "draftkings", "fanduel": "fanduel",
    "betmgm": "betmgm", "caesars": "williamhill_us", "espn bet": "espnbet", "thescore bet": "espnbet",
    "betrivers": "betrivers", "bovada": "bovada", "kalshi": "kalshi", "polymarket": "polymarket", "novig": "novig",
    "prophetx": "prophetx",
}
_SEEN_TITLES: Dict[str, str] = {}  # title (lowercase) -> key, from fetched boards
_SEEN_LIMIT = 500


class OddsApiError(RuntimeError):
    """The Odds API refused a request; ``status`` is its HTTP status (the message never quotes the URL)."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


def _remember_title(key: str, title: Any) -> None:
    if isinstance(title, str) and title and len(_SEEN_TITLES) < _SEEN_LIMIT:
        _SEEN_TITLES.setdefault(title.strip().lower(), key)


def book_key(book: Union[str, Mapping[str, Any], None]) -> Optional[str]:
    """A bookmaker's Odds API key from a book dict ({"key", "title"}), a key or a title ("Pinnacle", "Caesars")."""
    if isinstance(book, Mapping):
        key = book.get("key")
        if isinstance(key, str) and key:
            return key.lower()
        book = book.get("title")
    if not isinstance(book, str) or not book.strip():
        return None
    text = book.strip()
    lowered = text.lower()
    if lowered in _SEEN_TITLES:
        return _SEEN_TITLES[lowered]
    if lowered in KNOWN_TITLES:
        return KNOWN_TITLES[lowered]
    if re.fullmatch(r"[a-z0-9_]+", text):
        return text  # already a key
    return re.sub(r"[^a-z0-9]", "", lowered) or None


def is_exchange(key: Optional[str]) -> bool:
    return bool(key) and (key in EXCHANGE_BOOKS or key.startswith("betfair_ex"))


@dataclass(frozen=True)
class BookRoles:
    """Which books anchor the consensus (``sharp``) and which a person can bet at (``price``).

    ``price`` is ODDS_PRICE_BOOKS when given (less any sharp book), else every book in ``bookmakers`` that is
    neither sharp nor an exchange. A book from outside the list (a board saved before the list was set, when the
    app asked for the whole US region) counts as a price book unless it is sharp or an exchange.
    """

    bookmakers: Tuple[str, ...] = DEFAULT_BOOKMAKERS
    sharp: FrozenSet[str] = frozenset(DEFAULT_SHARP_BOOKS)
    price: FrozenSet[str] = frozenset(k for k in DEFAULT_BOOKMAKERS if k not in DEFAULT_SHARP_BOOKS
                                      and not is_exchange(k))

    @classmethod
    def make(cls, bookmakers: Optional[Sequence[str]] = None, sharp: Optional[Iterable[str]] = None,
             price: Optional[Iterable[str]] = None) -> "BookRoles":
        listed = tuple(k.lower() for k in (DEFAULT_BOOKMAKERS if bookmakers is None else bookmakers))
        sharp_keys = frozenset(k.lower() for k in (DEFAULT_SHARP_BOOKS if sharp is None else sharp))
        chosen = [k.lower() for k in (price or ())]
        if chosen:
            price_keys = frozenset(chosen) - sharp_keys
        else:
            price_keys = frozenset(k for k in listed if k not in sharp_keys and not is_exchange(k))
        return cls(listed, sharp_keys, price_keys)

    def role(self, book: Union[str, Mapping[str, Any], None]) -> str:
        """"sharp" (consensus anchor), "price" (a person can bet there) or "other" (an exchange, or a listed book
        left out of ODDS_PRICE_BOOKS): only "price" books are ever offered."""
        key = book_key(book)
        if not key:
            return "other"
        if key in self.sharp:
            return "sharp"
        if key in self.price:
            return "price"
        if key in self.bookmakers or is_exchange(key):
            return "other"
        return "price"

    def is_price(self, book: Union[str, Mapping[str, Any], None]) -> bool:
        return self.role(book) == "price"


_ROLES = BookRoles()
_ROLES_LOCK = threading.Lock()


def configure_books(bookmakers: Optional[Sequence[str]] = None, sharp: Optional[Iterable[str]] = None,
                    price: Optional[Iterable[str]] = None) -> BookRoles:
    """Set the process's book roles from the settings (services.build_service does this once at startup)."""
    global _ROLES
    with _ROLES_LOCK:
        _ROLES = BookRoles.make(bookmakers, sharp, price)
        return _ROLES


def book_roles() -> BookRoles:
    return _ROLES


def is_price_book(book: Union[str, Mapping[str, Any], None]) -> bool:
    """Whether a person can be offered this book's prices (see BookRoles); takes a book dict, key or title."""
    return _ROLES.is_price(book)


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class OddsService:
    def __init__(
        self,
        api_key: str,
        cache: Optional[TTLCache] = None,
        odds_ttl_minutes: float = 120,
        regions: str = "us",
        store: Optional[Store] = None,
        reserve: int = 50,
        bookmakers: Optional[Sequence[str]] = None,
    ) -> None:
        self.api_key = api_key
        self.cache = cache if cache is not None else TTLCache()  # an empty TTLCache is falsy (len 0)
        self.odds_ttl = odds_ttl_minutes * 60
        # Used only without a bookmaker list (ODDS_BOOKMAKERS=none), or if The Odds API refuses the list
        self.regions = regions
        # The bookmakers each board asks for (at most 10 cost one region); empty asks for ``regions`` instead
        self.bookmakers: Tuple[str, ...] = tuple(DEFAULT_BOOKMAKERS if bookmakers is None else bookmakers)
        self._list_refused = False  # The Odds API answered 400/422 to the list: regions from then on
        self._last: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}  # league -> last board seen (no store)
        # Shared with the other process on the server; None keeps everything in memory
        self.store = store
        # Below this many requests left, serve saved odds instead of fetching
        self.reserve = reserve
        self._quota: Dict[str, Optional[int]] = {"remaining": None, "used": None}
        self._fetched_at: Dict[str, float] = {}

    @property
    def quota(self) -> Dict[str, Optional[int]]:
        """Requests remaining and used this month, as last reported by the API."""
        if self.store:
            stored = self.store.latest_quota()
            if stored["remaining"] is not None:
                return stored
        return dict(self._quota)

    def _get(self, path: str, params: Dict[str, Any]) -> Any:
        try:
            response = requests.get(
                f"{BASE_URL}{path}", params={"apiKey": self.api_key, **params}, timeout=15
            )
        except requests.RequestException as exc:
            # A connection error's text quotes the request URL, which carries the API key: name only its type
            # (and drop it as the cause, so a logged traceback doesn't print it either)
            raise RuntimeError(f"Couldn't reach The Odds API for {path} ({type(exc).__name__})") from None
        headers = {
            field: int(float(response.headers[f"x-requests-{field}"]))
            for field in ("remaining", "used", "last")
            if response.headers.get(f"x-requests-{field}") is not None
        }
        self._quota.update({k: v for k, v in headers.items() if k != "last"})
        if self.store and headers:
            self.store.record_usage(path, headers.get("last"), headers.get("remaining"), headers.get("used"))
        if not response.ok:
            # Report the status only: the request URL carries the API key
            raise OddsApiError(f"The Odds API returned HTTP {response.status_code} for {path}", response.status_code)
        try:
            return response.json()
        except ValueError:
            raise RuntimeError(f"The Odds API sent an unreadable answer for {path}") from None

    @property
    def region_units(self) -> int:
        """How many regions a board counts as: one per 10 listed bookmakers, else one per region asked for."""
        if self.bookmakers and not self._list_refused:
            return max(1, math.ceil(len(self.bookmakers) / BOOKS_PER_REGION))
        return max(1, len([r for r in str(self.regions or "").split(",") if r.strip()]))

    @property
    def board_cost(self) -> int:
        """Credits one league's board costs: one per market per region (3 for ten bookmakers)."""
        return len(MARKETS) * self.region_units

    def board_params(self) -> Dict[str, str]:
        """The board request's query, without the key: the bookmaker list, or the regions."""
        params = {"markets": ",".join(MARKETS), "oddsFormat": "decimal"}
        if self.bookmakers and not self._list_refused:
            params["bookmakers"] = ",".join(self.bookmakers)
        else:
            params["regions"] = self.regions
        return params

    def _board(self, sport_key: str) -> Any:
        path = f"/sports/{sport_key}/odds"
        try:
            return self._get(path, self.board_params())
        except OddsApiError as exc:
            # A list The Odds API won't take (a bookmaker it doesn't know) is no reason to show no odds at all:
            # ask for the regions instead, from now on. A refused request costs no credits
            if exc.status not in (400, 422) or not self.bookmakers or self._list_refused:
                raise
            self._list_refused = True
            log.warning("The Odds API refused the bookmaker list (HTTP %s); asking for regions=%s instead. "
                        "Check ODDS_BOOKMAKERS", exc.status, self.regions)
            return self._get(path, self.board_params())

    def sports(self) -> List[Dict[str, Any]]:
        """All sports and leagues (free: costs no quota), through the shared cache (kind "odds_sports", hours):
        the bot and the dashboard share one copy. The key is never part of the cache key or the value."""
        ttl, stale = policy("odds_sports")
        return self.cache.get(("odds", "sports"), ttl, lambda: self._get("/sports", {}), stale=stale,
                              kind="odds_sports")

    def odds_loaded_at(self, sport_key: str) -> Optional[float]:
        """When the league's odds were fetched from the API (epoch seconds), or None."""
        return self._fetched_at.get(sport_key)

    def league_games(self, sport_key: str, refresh: bool = False) -> List[Dict[str, Any]]:
        """Upcoming and live games of a league with every bookmaker's odds, soonest first.

        Each fetch costs one quota unit per market, so a league's odds are
        reused for ``odds_ttl_minutes`` (from the shared store when there is
        one, so the bot and dashboard share fetches). ``refresh`` fetches
        early, but not more than once every MIN_REFRESH_SECONDS; nothing is
        fetched while fewer than ``reserve`` requests are left and saved
        odds exist.
        """
        key = ("odds", sport_key)
        fetched = self._fetched_at.get(sport_key)
        force = refresh and fetched is not None and time.time() - fetched >= MIN_REFRESH_SECONDS
        if force:
            self.cache.invalidate(key)

        def load() -> List[Dict[str, Any]]:
            snapshot = self.store.odds_snapshot(sport_key) if self.store else None
            if snapshot:
                fetched_at, games = snapshot
                age = time.time() - fetched_at
                wanted = age >= self.odds_ttl or (force and age >= MIN_REFRESH_SECONDS)
                remaining = self.quota["remaining"]
                low = remaining is not None and remaining < self.reserve
                # Another process may be fetching this league right now: use its last copy
                if not wanted or low or not self.store.try_lease(f"odds:{sport_key}", 60):
                    self._fetched_at[sport_key] = fetched_at
                    return games
            try:
                events = self._board(sport_key)
            except Exception:
                if snapshot:
                    self._fetched_at[sport_key] = snapshot[0]
                    return snapshot[1]
                raise
            games = sorted((normalize_event(event) for event in events), key=lambda game: game["commence_time"])
            now = time.time()
            if self.store:
                self.store.save_odds_snapshot(sport_key, games, now)
            self._fetched_at[sport_key] = now
            self._last[sport_key] = (now, games)
            return games

        # With a shared store the in-memory copy is short-lived, so a refresh made by the
        # other process shows up here within a minute
        memory_ttl = min(60.0, self.odds_ttl) if self.store else self.odds_ttl
        # The snapshot is its own persistence (the store's odds_snapshots): the board stays out of the shared
        # cache's table, and its rules (TTL, reserve, lease) are the snapshot's
        return self.cache.get(key, memory_ttl, load, kind="odds_board", persist=False)

    def saved_games(self, sport_key: str) -> Optional[Tuple[float, List[Dict[str, Any]]]]:
        """(fetched_at, games) of the league's last board, from the shared store (or this process's memory),
        never fetching: what the markets brain and the request paths that must not spend credits read."""
        snapshot = None
        if self.store is not None:
            try:
                snapshot = self.store.odds_snapshot(sport_key)
            except Exception as exc:
                log.warning("Couldn't read the saved %s odds (%s)", sport_key, type(exc).__name__)
        return snapshot or self._last.get(sport_key)

    def consensus_inputs(self, league: str) -> Dict[str, Dict[str, Any]]:
        """Every book's moneyline for each game of the league's saved board, for the markets brain's consensus:
        {game id: {"books": {book key: {outcome: decimal price}}, "as_of": iso}}. Sharp books are included (they
        are the anchor). Never fetches, so it costs no credits."""
        saved = self.saved_games(league)
        if not saved:
            return {}
        fetched_at, games = saved
        as_of = _iso(fetched_at)
        out: Dict[str, Dict[str, Any]] = {}
        for game in games or []:
            if not isinstance(game, Mapping) or not game.get("id"):
                continue
            books = moneylines(game)
            if books:
                out[str(game["id"])] = {"books": books, "as_of": as_of}
        return out


def normalize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten an Odds API event into ``books`` keyed by market, with decimal prices."""
    home, away = event["home_team"], event["away_team"]
    side_of = {home: "home", away: "away", "Draw": "draw", "Over": "over", "Under": "under"}
    books = []
    for bookmaker in event.get("bookmakers", []):
        markets: Dict[str, Dict[str, Dict[str, float]]] = {}
        for market in bookmaker.get("markets", []):
            if market.get("key") not in MARKETS:
                continue
            outcomes = {}
            for outcome in market.get("outcomes", []):
                side = side_of.get(outcome.get("name"))
                if side and outcome.get("price"):
                    outcomes[side] = {"price": float(outcome["price"]), "point": outcome.get("point")}
            if outcomes:
                markets[market["key"]] = outcomes
        if markets:
            books.append({"key": bookmaker["key"], "title": bookmaker["title"], "markets": markets})
            _remember_title(bookmaker["key"], bookmaker.get("title"))
    return {
        "id": event["id"],
        "sport_key": event.get("sport_key"),
        "league": event.get("sport_title"),
        "commence_time": event["commence_time"],
        "home": home,
        "away": away,
        "books": books,
    }


def _consensus_point(points: List[float]) -> Optional[float]:
    """The line most bookmakers offer (median of the most common ones on a tie)."""
    if not points:
        return None
    counts = Counter(points)
    top = max(counts.values())
    return statistics.median(point for point, count in counts.items() if count == top)


def moneylines(game: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """{book key: {outcome: decimal price}} of one game's moneyline at every book that prices it (sharp books
    included): engine/consensus.py's book_consensus input."""
    books: Dict[str, Dict[str, float]] = {}
    for book in game.get("books") or []:
        if not isinstance(book, Mapping):
            continue
        key = book_key(book)
        outcomes = ((book.get("markets") or {}).get("h2h") or {})
        prices = {side: float(offer["price"]) for side, offer in outcomes.items()
                  if isinstance(offer, Mapping) and isinstance(offer.get("price"), (int, float)) and offer["price"] > 1}
        if key and len(prices) >= 2:
            books[key] = prices
    return books


def summarize_markets(game: Dict[str, Any], roles: Optional[BookRoles] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Best price and consensus fair probability for every outcome of every market.

    Spreads and totals use the consensus line (the point most bookmakers
    offer); prices at other lines are ignored so all books are compared on
    the same bet. Fair probabilities average each bookmaker's margin-free
    probabilities (sharp books included) and are None when no bookmaker
    prices every side. The best price, its book and the count of books
    quoting it come from price books only (BookRoles: never Pinnacle or an
    exchange); an outcome no price book offers is left out.
    """
    roles = roles or _ROLES
    summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for market in MARKETS:
        offers = [(book, book["markets"][market]) for book in game["books"] if market in book["markets"]]
        if not offers:
            continue
        sides = sorted({side for _, outcomes in offers for side in outcomes})
        anchor = "home" if market == "spreads" else "over" if market == "totals" else None
        line = None
        if anchor:
            line = _consensus_point(
                [outcomes[anchor]["point"] for _, outcomes in offers if anchor in outcomes]
            )

        def at_line(side: str, outcomes: Dict[str, Dict[str, Any]]) -> bool:
            if line is None or side not in outcomes:
                return side in outcomes
            point = outcomes[side]["point"]
            return point == (-line if market == "spreads" and side == "away" else line)

        fair_by_side: Dict[str, List[float]] = {side: [] for side in sides}
        for _, outcomes in offers:
            if all(at_line(side, outcomes) for side in sides):
                fair = remove_vig([outcomes[side]["price"] for side in sides])
                for side, probability in zip(sides, fair):
                    fair_by_side[side].append(probability)

        summary[market] = {}
        for side in sides:
            priced = [(outcomes[side], book) for book, outcomes in offers
                      if at_line(side, outcomes) and roles.is_price(book)]
            if not priced:
                continue
            best, book = max(priced, key=lambda pair: pair[0]["price"])
            fair = fair_by_side[side]
            summary[market][side] = {
                "price": best["price"],
                "point": best["point"],
                "book": book["title"],
                "books": len(priced),
                "fair": sum(fair) / len(fair) if fair else None,
            }
        if not summary[market]:
            del summary[market]  # only sharp books or exchanges price it: nothing a person can bet
    return summary
