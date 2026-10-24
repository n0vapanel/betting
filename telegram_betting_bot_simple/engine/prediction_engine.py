"""Coordinator for sport analyses and prediction generation."""

from __future__ import annotations

import logging
from datetime import date
from typing import List, Dict, Optional

from ..data.apis import ApiSportsClient, TheOddsApiClient, NewsApiClient
from .analytics import calculate_win_probabilities
from .tier_classifier import TierClassifier

class PredictionEngine:
    """Generate predictions for sports matches based on analytics."""

    def __init__(
        self,
        api_sports_key: Optional[str] = None,
        odds_api_key: Optional[str] = None,
        news_api_key: Optional[str] = None,
    ) -> None:
        self.api_sports_key = api_sports_key
        self.odds_api_key = odds_api_key
        self.news_api_key = news_api_key
        self._api_sports_client: Optional[ApiSportsClient] = None
        self._odds_client: Optional[TheOddsApiClient] = None
        self._news_client: Optional[NewsApiClient] = None
        self._tier_classifier = TierClassifier()

    @property
    def api_sports_client(self) -> ApiSportsClient:
        if self._api_sports_client is None:
            if not self.api_sports_key:
                raise RuntimeError("API‑SPORTS key is required for fixtures and stats")
            self._api_sports_client = ApiSportsClient(self.api_sports_key)
        return self._api_sports_client

    @property
    def odds_client(self) -> TheOddsApiClient:
        if self._odds_client is None:
            if not self.odds_api_key:
                raise RuntimeError("The Odds API key is required for odds data")
            self._odds_client = TheOddsApiClient(self.odds_api_key)
        return self._odds_client

    @property
    def news_client(self) -> Optional[NewsApiClient]:
        if self.news_api_key and self._news_client is None:
            self._news_client = NewsApiClient(self.news_api_key)
        return self._news_client

    def _map_sport_to_odds_code(self, sport: str) -> str:
        mapping = {
            "nba": "basketball_nba",
            "nhl": "icehockey_nhl",
            "soccer": "soccer_epl",
            "tennis": "tennis_atp",
            "baseball": "baseball_mlb",
            "football": "americanfootball_nfl",
            "racing": "racing_horse",
        }
        return mapping.get(sport.lower(), sport)

    def get_predictions(
        self,
        sport: str,
        match_date: date,
        prediction_type: str = "single",
    ) -> List[Dict[str, str]]:
        """
        Return predictions for a given sport and match date.

        Combines fixtures, team stats, odds and news data to compute
        win probabilities and assigns each recommendation to a tier.
        """
        try:
            fixtures = self.api_sports_client.get_fixtures(sport, match_date)
        except Exception as e:
            logging.error("Error fetching fixtures: %s", e)
            return []

        odds_code = self._map_sport_to_odds_code(sport)
        try:
            odds_data = self.odds_client.get_odds(odds_code, markets="h2h")
        except Exception as e:
            logging.error("Error fetching odds: %s", e)
            odds_data = []

        predictions: List[Dict[str, str]] = []
        for fixture in fixtures:
            # Extract team names (API‑SPORTS responses differ by sport)
            home_name = (
                fixture.get("teams", {}).get("home", {}).get("name")
                or fixture.get("home_team")
                or fixture.get("home")
            )
            away_name = (
                fixture.get("teams", {}).get("away", {}).get("name")
                or fixture.get("away_team")
                or fixture.get("away")
            )
            if not home_name or not away_name:
                continue

            # Fetch stats for both teams
            try:
                home_stats = self.api_sports_client.get_team_stats(sport, home_name)
                away_stats = self.api_sports_client.get_team_stats(sport, away_name)
            except Exception as e:
                logging.error("Error fetching team stats: %s", e)
                home_stats, away_stats = {}, {}

            # Find matching odds for this fixture
            event_odds: Dict[str, float] = {}
            for event in odds_data:
                if (
                    event.get("home_team") == home_name
                    and event.get("away_team") == away_name
                ) or (
                    event.get("home_team") == away_name
                    and event.get("away_team") == home_name
                ):
                    markets = event.get("markets", {})
                    event_odds = markets.get("h2h", {})
                    break
            if not event_odds:
                event_odds = {"home": 2.0, "away": 2.0}

            # Injury factors from news
            home_injury_factor = 1.0
            away_injury_factor = 1.0
            if self.news_client:
                try:
                    home_articles = self.news_client.get_recent_articles(f"{home_name} injury")
                    away_articles = self.news_client.get_recent_articles(f"{away_name} injury")
                    home_injury_factor = max(0.5, 1.0 - 0.1 * len(home_articles))
                    away_injury_factor = max(0.5, 1.0 - 0.1 * len(away_articles))
                except Exception:
                    pass

            # Compute probabilities
            probs = calculate_win_probabilities(
                home_name,
                away_name,
                home_stats,
                away_stats,
                event_odds,
                home_injury_factor,
                away_injury_factor,
            )
            # Classify outcome
            winner, tier = self._tier_classifier.classify(probs)
            predictions.append(
                {
                    "sport": sport,
                    "match_date": match_date.isoformat(),
                    "home_team": home_name,
                    "away_team": away_name,
                    "winner": winner,
                    "tier": tier,
                    "probabilities": probs,
                }
            )
        return predictions
