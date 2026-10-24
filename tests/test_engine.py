"""Tests for the analytics functions and prediction engine.

These tests use dummy API clients to avoid network requests.  They
ensure that probability calculations behave as expected and that
PredictionEngine.get_predictions returns structured data when the
underlying clients provide known outputs.
"""

from __future__ import annotations

from datetime import date

from telegram_betting_bot_simple.engine.analytics import (
    implied_probabilities_from_odds,
    calculate_win_probabilities,
)
from telegram_betting_bot_simple.engine.prediction_engine import PredictionEngine

def test_implied_probabilities() -> None:
    odds = {"home": 2.0, "away": 4.0}
    probs = implied_probabilities_from_odds(odds)
    inv_sum = (1.0 / 2.0) + (1.0 / 4.0)
    assert abs(probs["home"] - (1.0 / 2.0) / inv_sum) < 1e-9
    assert abs(probs["away"] - (1.0 / 4.0) / inv_sum) < 1e-9

def test_calculate_win_probabilities_even() -> None:
    probs = calculate_win_probabilities(
        home_team="A",
        away_team="B",
        home_stats={},
        away_stats={},
        odds={"home": 2.0, "away": 2.0},
        home_injury_factor=1.0,
        away_injury_factor=1.0,
    )
    assert abs(probs["A"] - 0.5) < 1e-6
    assert abs(probs["B"] - 0.5) < 1e-6

def test_prediction_engine_integration() -> None:
    """Ensure the prediction engine returns a list with proper keys."""
    engine = PredictionEngine(api_sports_key="dummy", odds_api_key="dummy", news_api_key="dummy")

    # Stub API clients to avoid network calls
    class DummyApiSportsClient:
        def get_fixtures(self, sport: str, match_date: date):
            return [{"home_team": "Team A", "away_team": "Team B"}]
        def get_team_stats(self, sport: str, team_id: str, season: str | None = None):
            return {"points_for": 100, "points_against": 90, "games_played": 10}

    class DummyOddsClient:
        def get_odds(self, sport_code: str, markets: str = "h2h"):
            return [{
                "home_team": "Team A",
                "away_team": "Team B",
                "markets": {"h2h": {"home": 1.8, "away": 2.2}},
            }]

    class DummyNewsClient:
        def get_recent_articles(self, search_term: str, from_days: int | None = None, max_results: int = 5):
            return []

    engine._api_sports_client = DummyApiSportsClient()  # type: ignore[assignment]
    engine._odds_client = DummyOddsClient()              # type: ignore[assignment]
    engine._news_client = DummyNewsClient()              # type: ignore[assignment]

    preds = engine.get_predictions("nba", date.today())
    assert isinstance(preds, list)
    assert preds
    entry = preds[0]
    assert "home_team" in entry
    assert "away_team" in entry
    assert "winner" in entry
    assert "tier" in entry
