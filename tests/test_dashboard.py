"""Tests for the dashboard API and service, with fake odds and ESPN clients (no network)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.dashboard.app import create_app

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
