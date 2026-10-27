"""Game odds from The Odds API for the dashboard and bot.

Fetches moneyline, spread and total odds from every bookmaker in a region
in one request per league (each costs one quota unit per market), reuses
them from a store shared by the bot and dashboard, tracks the remaining
quota from the response headers, and summarises each game's markets into
the best available price and the bookmakers' consensus fair probability
for every outcome.
"""

from __future__ import annotations

import statistics
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import requests

from ..engine.odds_math import remove_vig
from .cache import TTLCache
from .store import Store

BASE_URL = "https://api.the-odds-api.com/v4"
MARKETS = ("h2h", "spreads", "totals")
# Bookmakers update about once a minute, so forced refreshes closer together than this waste quota
MIN_REFRESH_SECONDS = 300


class OddsService:
    def __init__(
        self,
        api_key: str,
        cache: Optional[TTLCache] = None,
        odds_ttl_minutes: float = 120,
        regions: str = "us",
        store: Optional[Store] = None,
        reserve: int = 50,
    ) -> None:
        self.api_key = api_key
        self.cache = cache or TTLCache()
        self.odds_ttl = odds_ttl_minutes * 60
        self.regions = regions
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
            raise RuntimeError(f"The Odds API returned HTTP {response.status_code} for {path}")
        try:
            return response.json()
        except ValueError:
            raise RuntimeError(f"The Odds API sent an unreadable answer for {path}") from None

    def sports(self) -> List[Dict[str, Any]]:
        """All sports and leagues (free: costs no quota). Cached for 6 hours."""
        return self.cache.get(("odds", "sports"), 6 * 3600, lambda: self._get("/sports", {}))

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
                events = self._get(
                    f"/sports/{sport_key}/odds",
                    {"regions": self.regions, "markets": ",".join(MARKETS), "oddsFormat": "decimal"},
                )
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
            return games

        # With a shared store the in-memory copy is short-lived, so a refresh made by the
        # other process shows up here within a minute
        memory_ttl = min(60.0, self.odds_ttl) if self.store else self.odds_ttl
        return self.cache.get(key, memory_ttl, load)


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


def summarize_markets(game: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Best price and consensus fair probability for every outcome of every market.

    Spreads and totals use the consensus line (the point most bookmakers
    offer); prices at other lines are ignored so all books are compared on
    the same bet. Fair probabilities average each bookmaker's margin-free
    probabilities and are None when no bookmaker prices every side.
    """
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
            priced = [(outcomes[side], book) for book, outcomes in offers if at_line(side, outcomes)]
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
    return summary
