"""Basic tests for API client classes.

These tests verify that the client classes enforce API key requirements
and that certain methods return sensible defaults when invoked with
minimal input.  They deliberately avoid making real network calls.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import requests

from telegram_betting_bot_simple.data import utils
from telegram_betting_bot_simple.data.apis import ApiSportsClient, NewsApiClient, TheOddsApiClient
from telegram_betting_bot_simple.data.apis import the_odds_api_client

def test_api_sports_client_requires_key() -> None:
    with pytest.raises(ValueError):
        ApiSportsClient(api_key="")
    client = ApiSportsClient(api_key="dummy")
    assert client is not None

def test_news_api_client_returns_empty_on_blank_search() -> None:
    client = NewsApiClient(api_key="dummy")
    articles = client.get_recent_articles("")
    assert articles == []

def test_the_odds_api_client_requires_key() -> None:
    with pytest.raises(ValueError):
        TheOddsApiClient(api_key="")
    client = TheOddsApiClient(api_key="dummy")
    assert client is not None

def test_the_odds_api_client_get_odds_format_and_bookmaker(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_fetch_json(url, params=None, headers=None, timeout=10.0):
        calls.append((url, params))
        return [{
            "id": "e1",
            "commence_time": "2026-10-25T23:30:00Z",
            "home_team": "Los Angeles Lakers",
            "away_team": "Boston Celtics",
            "bookmakers": [{
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [{"key": "h2h", "outcomes": [
                    {"name": "Boston Celtics", "price": 130},
                    {"name": "Los Angeles Lakers", "price": -150},
                ]}],
            }],
        }]

    monkeypatch.setattr(the_odds_api_client, "fetch_json", fake_fetch_json)
    events = TheOddsApiClient(api_key="dummy").get_odds("basketball_nba", odds_format="american")
    assert calls[0][0].endswith("/sports/basketball_nba/odds")
    assert calls[0][1]["oddsFormat"] == "american"
    assert events == [{
        "id": "e1",
        "home_team": "Los Angeles Lakers",
        "away_team": "Boston Celtics",
        "start_time": "2026-10-25T23:30:00Z",
        "markets": {"h2h": {"Boston Celtics": 130.0, "Los Angeles Lakers": -150.0}},
        "bookmaker": "DraftKings",
    }]

def test_fetch_json_keeps_api_keys_out_of_logs_and_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def failing_get(url, params=None, headers=None, timeout=10.0):
        raise requests.HTTPError(
            "401 Client Error: Unauthorized for url: https://example.test/v4/sports?apiKey=SECRET123",
            response=SimpleNamespace(status_code=401),
        )

    monkeypatch.setattr(utils.requests, "get", failing_get)
    with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError) as excinfo:
        utils.fetch_json("https://example.test/v4/sports", params={"apiKey": "SECRET123"})
    assert "401" in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert "SECRET123" not in str(excinfo.value)
    assert "SECRET123" not in caplog.text
