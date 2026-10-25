"""Tests for the dashboard API and service, with fake odds and ESPN clients (no network)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.dashboard.app import SESSION_COOKIE, create_app, session_token, valid_session

@pytest.fixture()
def visitor(service):
    return TestClient(create_app(service, "s3cret", login_delay=0))

@pytest.fixture()
def client(visitor):
    assert visitor.post("/api/login", json={"password": "s3cret"}).status_code == 200
    return visitor

def test_the_api_needs_a_session_from_the_sign_in_form(visitor) -> None:
    page = visitor.get("/")
    assert page.status_code == 200  # the page holds no data and shows its own sign-in form
    # Re-checked on every load, so an update never leaves a browser running old code
    assert page.headers["cache-control"] == "no-cache"
    assert visitor.get("/static/app.js").headers["cache-control"] == "no-cache"
    for path in ("/api/leagues", "/api/games?league=basketball_nba"):
        response = visitor.get(path)
        assert response.status_code == 401
        assert "www-authenticate" not in response.headers  # which would open the browser's login pop-up
    assert visitor.post("/api/login", json={"password": "wrong"}).status_code == 401
    assert SESSION_COOKIE not in visitor.cookies
    response = visitor.post("/api/login", json={"password": "s3cret"})
    assert response.status_code == 200
    assert "httponly" in response.headers["set-cookie"].lower()
    assert "samesite=strict" in response.headers["set-cookie"].lower()
    assert visitor.get("/api/leagues").json() == [
        {"group": "Basketball", "label": "🏀 Basketball", "leagues": [{"key": "basketball_nba", "title": "NBA", "ai": True}]}
    ]
    visitor.post("/api/logout")
    assert visitor.get("/api/leagues").status_code == 401

def test_sessions_are_signed_and_expire() -> None:
    key = b"k" * 32
    token = session_token(key, 2_000)
    assert valid_session(key, token, now=1_000)
    assert not valid_session(key, token, now=3_000)  # expired
    assert not valid_session(b"x" * 32, token, now=1_000)  # the password changed
    assert not valid_session(key, token.replace("2000", "9000", 1), now=1_000)  # expiry tampered with
    assert not valid_session(key, "²000.abc", now=1_000)
    assert not valid_session(key, None, now=1_000)

def test_games_have_best_prices_and_model_probabilities(client) -> None:
    data = client.get("/api/games?league=basketball_nba").json()
    assert data["ai"] and data["quota"]["remaining"] == 480
    okc = data["games"][0]
    assert okc["markets"]["h2h"]["away"]["price"] == 7.25 and okc["markets"]["h2h"]["away"]["book"] == "DraftKings"
    assert okc["markets"]["spreads"]["home"]["label"] == "Oklahoma City Thunder -13.5"
    # Market-only prediction: the home team is favoured by the spread
    assert okc["prediction"]["margin"] == 13.5
    assert okc["prediction"]["pick"] == "Oklahoma City Thunder"
    assert client.get("/api/games?league=NBA!").status_code == 422

def test_multi_is_priced_at_one_bookmaker(client) -> None:
    picks = [
        {"league": "basketball_nba", "id": "g1", "market": "h2h", "side": "home"},
        {"league": "basketball_nba", "id": "g2", "market": "h2h", "side": "away"},
    ]
    slip = client.post("/api/slip", json={"mode": "multi", "stake": 10, "picks": picks}).json()
    # Bovada doesn't price game 1, so only DraftKings (1.10 x 3.50) and FanDuel (1.11 x 3.60) qualify
    assert [b["book"] for b in slip["books"]] == ["FanDuel", "DraftKings"]
    assert slip["book"] == "FanDuel"
    assert slip["odds"]["decimal"] == pytest.approx(round(1.11 * 3.60, 2))
    assert slip["returns"] == pytest.approx(round(10 * 1.11 * 3.60, 2))
    assert slip["valid"] and 0 < slip["book_edge"] < 0.2

    same_game = picks + [{"league": "basketball_nba", "id": "g1", "market": "totals", "side": "over"}]
    slip = client.post("/api/slip", json={"mode": "multi", "stake": 10, "picks": same_game}).json()
    assert not slip["valid"] and "same game" in slip["reasons"][0]

    gone = [{"league": "basketball_nba", "id": "nope", "market": "h2h", "side": "home"}]
    slip = client.post("/api/slip", json={"mode": "single", "stakes": {"0": 5}, "picks": gone}).json()
    assert slip["removed"] == gone and not slip["valid"]
    bad = client.post("/api/slip", json={"mode": "multi", "picks": [{**picks[0], "market": "props"}]})
    assert bad.status_code == 422

def test_without_an_odds_key_the_api_says_how_to_fix_it() -> None:
    client = TestClient(create_app(None, "s3cret"))
    client.post("/api/login", json={"password": "s3cret"})
    response = client.get("/api/leagues")
    assert response.status_code == 503 and "THE_ODDS_API_KEY" in response.json()["detail"]
