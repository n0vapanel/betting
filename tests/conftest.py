"""Shared test fixtures: a games/pricing service backed by fake odds and ESPN clients (no network)."""

from __future__ import annotations

import copy
from zoneinfo import ZoneInfo

import pytest

from telegram_betting_bot_simple.dashboard.service import DashboardService
from telegram_betting_bot_simple.data.odds_service import normalize_event

def make_event(event_id, home, away, start, books):
    bookmakers = []
    for title, (home_ml, away_ml, home_spread, over) in books.items():
        bookmakers.append({"key": title.lower(), "title": title, "markets": [
            {"key": "h2h", "outcomes": [{"name": home, "price": home_ml}, {"name": away, "price": away_ml}]},
            {"key": "spreads", "outcomes": [
                {"name": home, "price": 1.91, "point": home_spread}, {"name": away, "price": 1.91, "point": -home_spread},
            ]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": over, "point": 224.5}, {"name": "Under", "price": 1.91, "point": 224.5},
            ]},
        ]})
    return normalize_event({"id": event_id, "sport_key": "basketball_nba", "sport_title": "NBA",
                            "commence_time": start, "home_team": home, "away_team": away, "bookmakers": bookmakers})

GAMES = [
    make_event("g1", "Oklahoma City Thunder", "Los Angeles Clippers", "2030-01-01T00:00:00Z",
               {"DraftKings": (1.10, 7.25, -13.5, 1.91), "FanDuel": (1.11, 7.00, -13.5, 1.95)}),
    make_event("g2", "New York Knicks", "Orlando Magic", "2030-01-01T01:00:00Z",
               {"DraftKings": (1.33, 3.50, -6.5, 1.91), "FanDuel": (1.30, 3.60, -6.5, 1.87), "Bovada": (1.36, 3.40, -6.5, 1.91)}),
]

class FakeOdds:
    quota = {"remaining": 480, "used": 20}

    def sports(self):
        return [{"key": "basketball_nba", "group": "Basketball", "title": "NBA", "active": True, "has_outrights": False}]

    def league_games(self, league, refresh=False):
        return copy.deepcopy(GAMES)

    def odds_loaded_at(self, league):
        return None

class FakeEspn:
    def find_game(self, *args):
        return None  # no ESPN match: predictions come from the market

    def standings(self, league, season=None):
        return {}

@pytest.fixture()
def service():
    return DashboardService(FakeOdds(), FakeEspn(), ZoneInfo("America/New_York"))
