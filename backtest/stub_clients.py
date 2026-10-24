"""Stub API clients that replay the historical NBA dataset through the real engine.

They implement the same duck-typed interface the `PredictionEngine` expects
(the same pattern as the `Dummy*` clients in `tests/test_engine.py`), so the
engine's production code path runs unmodified.
"""

from __future__ import annotations

from datetime import date
from typing import Callable, Dict, List


class NbaBacktestApiSportsClient:
    """Plays the role of `ApiSportsClient` using historical NBA data.

    `get_fixtures(sport, match_date)` returns the games actually played on
    that date.  `get_team_stats(...)` returns stats computed ONLY from games
    played before the date of the currently requested fixtures, so the engine
    can never see the game being predicted (or any later game).
    """

    def __init__(
        self,
        fixtures_by_date: Dict[str, List[Dict]],
        stats_provider: Callable[[date, str], Dict[str, float]],
    ) -> None:
        self._fixtures_by_date = fixtures_by_date
        self._stats_provider = stats_provider
        self._current_date: date | None = None

    def get_fixtures(self, sport: str, match_date: date) -> List[Dict]:
        self._current_date = match_date
        out: List[Dict] = []
        for game in self._fixtures_by_date.get(match_date.isoformat(), []):
            out.append({"home_team": game["home"], "away_team": game["away"]})
        return out

    def get_team_stats(self, sport: str, team_name: str, season: str | None = None) -> Dict[str, float]:
        if self._current_date is None:
            return {"points_for": 0.0, "points_against": 0.0, "games_played": 0}
        return self._stats_provider(self._current_date, team_name)


class NoOddsClient:
    """Returns no odds; the engine falls back to neutral h2h {home: 2.0, away: 2.0}."""

    def get_odds(self, sport_code: str, markets: str = "h2h") -> List:
        return []
