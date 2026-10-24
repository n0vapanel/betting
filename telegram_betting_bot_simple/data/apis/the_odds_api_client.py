"""Client for The Odds API (v4).

This module provides a wrapper around the official The Odds API to
retrieve betting odds for upcoming sports fixtures.  The API documents
are available at https://the-odds-api.com/.  A free plan exists with
a monthly quota; you may need to upgrade for heavier usage.

The client exposes methods to list supported sports and fetch odds
for a given sport and region.  It normalises the API's response into
a dictionary that is easier for the prediction engine to consume.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils import fetch_json


class TheOddsApiClient:
    """Retrieve betting odds from The Odds API."""

    DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        if not api_key:
            raise ValueError("The Odds API key must be provided")
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL

    def get_sports(self) -> List[Dict[str, Any]]:
        """Return the list of available sports from The Odds API."""
        url = f"{self.base_url}/sports"
        params = {"apiKey": self.api_key}
        try:
            data = fetch_json(url, params=params)
        except Exception as e:
            logging.error("Failed to fetch sports list: %s", e)
            return []
        return data if isinstance(data, list) else []

    def get_odds(
        self,
        sport: str,
        regions: str = "us",
        markets: str = "h2h",
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch odds for upcoming events for a given sport.

        Args:
            sport: The sport code (e.g. "basketball_nba", "soccer_epl").
            regions: Comma‑separated list of regions to pull odds from.
            markets: Comma‑separated list of market types (e.g. "h2h", "spreads").
            date_from: Filter events starting after this datetime.
            date_to: Filter events starting before this datetime.

        Returns:
            A list of event dictionaries, each containing keys:
            "id", "home_team", "away_team", "start_time" and "markets".
        """
        url = f"{self.base_url}/sports/{sport}/odds"
        params: Dict[str, Any] = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
        }
        if date_from:
            params["dateFormat"] = "iso"
            params["oddsFormat"] = "decimal"
            params["dateRangeFrom"] = date_from.isoformat()
        if date_to:
            params["dateRangeTo"] = date_to.isoformat()
        try:
            data = fetch_json(url, params=params)
        except Exception as e:
            logging.error("Failed to fetch odds: %s", e)
            return []
        events: List[Dict[str, Any]] = []
        for event in data if isinstance(data, list) else []:
            try:
                event_id = str(event.get("id") or event.get("key"))
                home_team = event.get("home_team")
                away_team = event.get("away_team")
                commence_time = event.get("commence_time")  # ISO8601
                markets_dict: Dict[str, Dict[str, float]] = {}
                # Use the first bookmaker's data as representative
                for bookmaker in event.get("bookmakers", [])[:1]:
                    for market in bookmaker.get("markets", []):
                        market_key = market.get("key")
                        outcomes = market.get("outcomes", [])
                        if not market_key or not outcomes:
                            continue
                        outcome_prices: Dict[str, float] = {}
                        for outcome in outcomes:
                            name = outcome.get("name")
                            price = outcome.get("price")
                            if name and price is not None:
                                outcome_prices[name] = float(price)
                        markets_dict[market_key] = outcome_prices
                events.append(
                    {
                        "id": event_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "start_time": commence_time,
                        "markets": markets_dict,
                    }
                )
            except Exception:
                continue
        return events
