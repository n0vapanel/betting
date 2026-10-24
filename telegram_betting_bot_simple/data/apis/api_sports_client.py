"""Client wrapper for the API‑SPORTS service.

API‑SPORTS provides a unified REST interface for live scores, statistics,
fixtures and more across many sports.  This client exposes a few
convenience methods for retrieving fixtures and team/player data.

Refer to the API‑SPORTS documentation for specific parameters and
rate limits.  The default base URL targets the v3 Football API,
but you can override ``base_url`` when instantiating the client for
other sports (e.g. Basketball, Tennis).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

import requests


class ApiSportsClient:
    """Minimal wrapper around API‑SPORTS endpoints.

    Args:
        api_key: Your API‑SPORTS key.  Required for authenticated access.
        base_url: Base URL for the API.  Defaults to Football v3.  Override
            when using other sports (e.g. https://v1.basketball.api-sports.io).
    """

    def __init__(self, api_key: str, base_url: str = "https://v3.football.api-sports.io") -> None:
        if not api_key:
            raise ValueError("API‑SPORTS key must be provided")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        # API‑SPORTS expects the key in a header named 'x-apisports-key'
        self.session.headers.update({"x-apisports-key": self.api_key})

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Internal helper to perform GET requests and parse JSON."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self.session.get(url, params=params, timeout=15.0)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logging.exception("API‑SPORTS request failed: %s", exc)
            raise RuntimeError(f"API‑SPORTS request failed: {exc}") from exc

    def get_fixtures(self, sport: str, match_date: date) -> List[Dict[str, Any]]:
        """Retrieve fixtures for a given sport and date.

        The specific endpoint differs by sport.  For example, Football uses
        ``/fixtures``, while Basketball uses ``/games``.  This method
        implements a simple mapping and should be extended for other sports
        as needed.

        Args:
            sport: Short code identifying the sport (e.g. 'soccer', 'nba').
            match_date: Date for which to fetch fixtures.

        Returns:
            A list of fixture dictionaries as returned by the API.
        """
        endpoint_map = {
            "soccer": "fixtures",
            "football": "fixtures",  # NFL if supported
            "nba": "games",
            "basketball": "games",
            "tennis": "games",
            "nhl": "games",
            "baseball": "games",
        }
        path = endpoint_map.get(sport.lower(), "fixtures")
        params = {"date": match_date.isoformat()}
        data = self._get(path, params=params)
        fixtures = data.get("response") if isinstance(data, dict) else None
        return fixtures or []

    def get_team_stats(
        self, sport: str, team_id: str, season: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch aggregated statistics for a team."""
        if sport.lower() in {"soccer", "football"}:
            path = "teams/statistics"
            params = {"team": team_id}
            if season:
                params["season"] = season
        else:
            # Generic fallback for other sports
            path = "teams"
            params = {"id": team_id}
        data = self._get(path, params=params)
        return data.get("response", {}) if isinstance(data, dict) else {}

    def get_player_stats(
        self, sport: str, player_id: str, season: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch statistics for a specific player."""
        if sport.lower() in {"soccer", "football"}:
            path = "players"
            params = {"id": player_id}
            if season:
                params["season"] = season
        else:
            path = "players"
            params = {"id": player_id}
        data = self._get(path, params=params)
        return data.get("response", {}) if isinstance(data, dict) else {}

    def get_injuries(
        self,
        sport: str,
        league_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve a list of current injuries for a league or team."""
        if sport.lower() not in {"soccer", "football"}:
            return []
        path = "injuries"
        params: Dict[str, Any] = {}
        if league_id:
            params["league"] = league_id
        if team_id:
            params["team"] = team_id
        data = self._get(path, params=params)
        return data.get("response", []) if isinstance(data, dict) else []
