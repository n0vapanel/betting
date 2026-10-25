"""Game odds from The Odds API for the dashboard and bot.

Fetches moneyline, spread and total odds from every bookmaker in a region
in one request per league, caches them (each league request costs one
quota unit per market), records the remaining quota from the response
headers, and summarises each game's markets into the best available price
and the bookmakers' consensus fair probability for every outcome.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List, Optional

import requests

from ..engine.odds_math import remove_vig
from .cache import TTLCache

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
    ) -> None:
        self.api_key = api_key
        self.cache = cache or TTLCache()
        self.odds_ttl = odds_ttl_minutes * 60
        self.regions = regions
        self.quota: Dict[str, Optional[int]] = {"remaining": None, "used": None}

    def _get(self, path: str, params: Dict[str, Any]) -> Any:
        response = requests.get(
            f"{BASE_URL}{path}", params={"apiKey": self.api_key, **params}, timeout=15
        )
        for field in ("remaining", "used"):
            value = response.headers.get(f"x-requests-{field}")
            if value is not None:
                self.quota[field] = int(float(value))
        if not response.ok:
            # Report the status only: the request URL carries the API key
            raise RuntimeError(f"The Odds API returned HTTP {response.status_code} for {path}")
        return response.json()

    def sports(self) -> List[Dict[str, Any]]:
        """All sports and leagues (free: costs no quota). Cached for 6 hours."""
        return self.cache.get(("odds", "sports"), 6 * 3600, lambda: self._get("/sports", {}))

    def odds_loaded_at(self, sport_key: str) -> Optional[float]:
        """When the league's odds were fetched (epoch seconds), or None."""
        return self.cache.loaded_at(("odds", sport_key))

    def league_games(self, sport_key: str, refresh: bool = False) -> List[Dict[str, Any]]:
        """Upcoming and live games of a league with every bookmaker's odds, soonest first.

        Each fetch costs one quota unit per market, so odds are cached for
        ``odds_ttl_minutes``; ``refresh`` fetches early, but not more than
        once every MIN_REFRESH_SECONDS.
        """
        age = self.cache.age(("odds", sport_key))
        if refresh and age is not None and age >= MIN_REFRESH_SECONDS:
            self.cache.invalidate(("odds", sport_key))

        def load() -> List[Dict[str, Any]]:
            events = self._get(
                f"/sports/{sport_key}/odds",
                {"regions": self.regions, "markets": ",".join(MARKETS), "oddsFormat": "decimal"},
            )
            games = [normalize_event(event) for event in events]
            return sorted(games, key=lambda game: game["commence_time"])

        return self.cache.get(("odds", sport_key), self.odds_ttl, load)


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
