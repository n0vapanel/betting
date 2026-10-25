"""Tests for the dashboard API and service, with fake odds and ESPN clients (no network)."""

from __future__ import annotations

import copy
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.dashboard.service import DashboardService
from telegram_betting_bot_simple.data.odds_service import normalize_event

def _event(event_id, home, away, start, books):
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
    _event("g1", "Oklahoma City Thunder", "Los Angeles Clippers", "2030-01-01T00:00:00Z",
           {"DraftKings": (1.10, 7.25, -13.5, 1.91), "FanDuel": (1.11, 7.00, -13.5, 1.95)}),
    _event("g2", "New York Knicks", "Orlando Magic", "2030-01-01T01:00:00Z",
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

@pytest.fixture()
def client(service):
    return TestClient(create_app(service, "s3cret"))

AUTH = ("anyone", "s3cret")

def test_every_route_needs_the_password(client) -> None:
    for path in ("/", "/api/leagues", "/api/games?league=basketball_nba"):
        assert client.get(path).status_code == 401
        assert client.get(path, auth=("anyone", "wrong")).status_code == 401
    assert client.get("/", auth=AUTH).status_code == 200
    assert client.get("/api/leagues", auth=AUTH).json() == [
        {"group": "Basketball", "label": "🏀 Basketball", "leagues": [{"key": "basketball_nba", "title": "NBA", "ai": True}]}
    ]

def test_games_have_best_prices_and_model_probabilities(client) -> None:
    data = client.get("/api/games?league=basketball_nba", auth=AUTH).json()
    assert data["ai"] and data["quota"]["remaining"] == 480
    okc = data["games"][0]
    assert okc["markets"]["h2h"]["away"]["price"] == 7.25 and okc["markets"]["h2h"]["away"]["book"] == "DraftKings"
    assert okc["markets"]["spreads"]["home"]["label"] == "Oklahoma City Thunder -13.5"
    # Market-only prediction: the home team is favoured by the spread
    assert okc["prediction"]["margin"] == 13.5
    assert okc["prediction"]["pick"] == "Oklahoma City Thunder"
    assert client.get("/api/games?league=NBA!", auth=AUTH).status_code == 422

def test_multi_is_priced_at_one_bookmaker(client) -> None:
    picks = [
        {"league": "basketball_nba", "id": "g1", "market": "h2h", "side": "home"},
        {"league": "basketball_nba", "id": "g2", "market": "h2h", "side": "away"},
    ]
    slip = client.post("/api/slip", auth=AUTH, json={"mode": "multi", "stake": 10, "picks": picks}).json()
    # Bovada doesn't price game 1, so only DraftKings (1.10 x 3.50) and FanDuel (1.11 x 3.60) qualify
    assert [b["book"] for b in slip["books"]] == ["FanDuel", "DraftKings"]
    assert slip["book"] == "FanDuel"
    assert slip["odds"]["decimal"] == pytest.approx(round(1.11 * 3.60, 2))
    assert slip["returns"] == pytest.approx(round(10 * 1.11 * 3.60, 2))
    assert slip["valid"] and 0 < slip["book_edge"] < 0.2

    same_game = picks + [{"league": "basketball_nba", "id": "g1", "market": "totals", "side": "over"}]
    slip = client.post("/api/slip", auth=AUTH, json={"mode": "multi", "stake": 10, "picks": same_game}).json()
    assert not slip["valid"] and "same game" in slip["reasons"][0]

    gone = [{"league": "basketball_nba", "id": "nope", "market": "h2h", "side": "home"}]
    slip = client.post("/api/slip", auth=AUTH, json={"mode": "single", "stakes": {"0": 5}, "picks": gone}).json()
    assert slip["removed"] == gone and not slip["valid"]
    bad = client.post("/api/slip", auth=AUTH, json={"mode": "multi", "picks": [{**picks[0], "market": "props"}]})
    assert bad.status_code == 422

def test_without_an_odds_key_the_api_says_how_to_fix_it() -> None:
    client = TestClient(create_app(None, "s3cret"))
    response = client.get("/api/leagues", auth=AUTH)
    assert response.status_code == 503 and "THE_ODDS_API_KEY" in response.json()["detail"]
