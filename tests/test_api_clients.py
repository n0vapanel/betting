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


def test_soccer_after_extra_time_is_graded_on_the_ninety_minute_score() -> None:
    import copy
    import json
    from pathlib import Path

    from telegram_betting_bot_simple.data.espn_client import EspnClient
    from telegram_betting_bot_simple.engine.bets import leg_outcome

    # ESPN's summary for Chelsea at Real Madrid, 12 April 2022 (trimmed): 1-3 after 90 minutes, 2-3 after
    # extra time. Soccer moneylines, spreads and totals settle on the 90 minutes
    summary = json.loads((Path(__file__).parent / "fixtures" / "espn" / "soccer_summary_633241_aet.json").read_text())
    client = EspnClient()
    client.summary = lambda league, event_id: copy.deepcopy(summary)
    result = client.game_result("soccer_uefa_champs_league", "633241")
    assert result["state"] == "final" and (result["home"], result["away"]) == (1, 3)
    assert result["after_extra_time"] == {"home": 2, "away": 3}
    assert leg_outcome({"market": "totals", "side": "over", "point": 4.5}, result) == "lost"  # 4 goals in 90'
    assert leg_outcome({"market": "spreads", "side": "away", "point": -1.5}, result) == "won"  # Chelsea by 2
    # Other sports' overtime counts, as their bookmakers settle it
    assert client.game_result("basketball_nba", "633241")["home"] == 2
    # A soccer game without readable halves stays pending (in the end voided) rather than graded wrong
    summary["header"]["competitions"][0]["competitors"][0]["linescores"] = []
    assert client.game_result("soccer_uefa_champs_league", "633241")["state"] == "pending"
    # A game decided in 90 minutes keeps its score
    regular = copy.deepcopy(summary)
    competition = regular["header"]["competitions"][0]
    competition["status"]["type"]["name"] = "STATUS_FULL_TIME"
    for side in competition["competitors"]:
        side["linescores"] = side["linescores"][:2] if side["linescores"] else [{"displayValue": "1"}, {"displayValue": "1"}]
    summary = regular
    assert client.game_result("soccer_uefa_champs_league", "633241") == {"state": "final", "home": 2, "away": 3}
