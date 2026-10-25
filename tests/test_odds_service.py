"""Tests for the odds service: normalising events, summarising markets, quota and caching."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from telegram_betting_bot_simple.data import odds_service
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.odds_service import OddsService, normalize_event, summarize_markets

def _book(key, h2h, spread_home, spread_prices, total, total_prices):
    return {
        "key": key,
        "title": key.title(),
        "markets": [
            {"key": "h2h", "outcomes": [
                {"name": "Boston Celtics", "price": h2h[0]}, {"name": "Los Angeles Lakers", "price": h2h[1]},
            ]},
            {"key": "spreads", "outcomes": [
                {"name": "Boston Celtics", "price": spread_prices[0], "point": -spread_home},
                {"name": "Los Angeles Lakers", "price": spread_prices[1], "point": spread_home},
            ]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": total_prices[0], "point": total},
                {"name": "Under", "price": total_prices[1], "point": total},
            ]},
        ],
    }

EVENT = {
    "id": "e1",
    "sport_key": "basketball_nba",
    "sport_title": "NBA",
    "commence_time": "2026-10-25T23:30:00Z",
    "home_team": "Los Angeles Lakers",
    "away_team": "Boston Celtics",
    "bookmakers": [
        _book("draftkings", (2.30, 1.65), -3.5, (1.91, 1.91), 224.5, (1.91, 1.91)),
        _book("fanduel", (2.36, 1.62), -3.5, (1.95, 1.87), 224.5, (1.87, 1.95)),
        # An off-consensus line: its spread and total prices must be ignored
        _book("bovada", (2.25, 1.67), -4.0, (2.10, 1.75), 226.0, (2.20, 1.70)),
    ],
}

def test_normalize_event_keys_outcomes_by_side() -> None:
    game = normalize_event(EVENT)
    assert (game["home"], game["away"], game["league"]) == ("Los Angeles Lakers", "Boston Celtics", "NBA")
    draftkings = game["books"][0]["markets"]
    assert draftkings["h2h"] == {"away": {"price": 2.30, "point": None}, "home": {"price": 1.65, "point": None}}
    assert draftkings["spreads"]["home"] == {"price": 1.91, "point": -3.5}
    assert draftkings["totals"]["under"] == {"price": 1.91, "point": 224.5}

def test_summarize_markets_best_price_and_consensus_line() -> None:
    summary = summarize_markets(normalize_event(EVENT))
    assert summary["h2h"]["away"]["price"] == 2.36
    assert summary["h2h"]["away"]["book"] == "Fanduel"
    assert summary["h2h"]["home"]["price"] == 1.67
    assert summary["h2h"]["home"]["fair"] + summary["h2h"]["away"]["fair"] == pytest.approx(1.0)
    # Spread: consensus is home -3.5 (2 books); Bovada's -4 line is left out
    # Fair home cover chance: DraftKings 50.0%, FanDuel 51.0% (1.87 vs 1.95) -> 50.5%
    assert summary["spreads"]["home"] == {
        "price": 1.91, "point": -3.5, "book": "Draftkings", "books": 2, "fair": pytest.approx(0.5052, abs=1e-3),
    }
    assert summary["spreads"]["away"]["point"] == 3.5
    assert summary["spreads"]["away"]["price"] == 1.95
    assert summary["totals"]["over"]["point"] == 224.5
    assert summary["totals"]["over"]["books"] == 2
    assert summary["totals"]["under"]["price"] == 1.95

def test_service_reads_quota_and_keeps_key_out_of_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        if "/odds" in url:
            return SimpleNamespace(ok=False, status_code=401, headers={}, json=lambda: {})
        return SimpleNamespace(
            ok=True, status_code=200, headers={"x-requests-remaining": "497", "x-requests-used": "3"},
            json=lambda: [{"key": "basketball_nba"}],
        )

    monkeypatch.setattr(odds_service.requests, "get", fake_get)
    service = OddsService("SECRET123")
    assert service.sports() == [{"key": "basketball_nba"}]
    service.sports()  # cached: no second request
    assert len(calls) == 1
    assert service.quota == {"remaining": 497, "used": 3}
    with pytest.raises(RuntimeError) as excinfo:
        service.league_games("basketball_nba")
    assert "401" in str(excinfo.value) and "SECRET123" not in str(excinfo.value)

def test_ttl_cache_serves_stale_value_when_refresh_fails() -> None:
    cache = TTLCache()
    assert cache.get("k", 60, lambda: "fresh") == "fresh"
    assert cache.get("k", 60, lambda: "ignored") == "fresh"

    def broken():
        raise RuntimeError("feed down")

    assert cache.get("k", 0, broken) == "fresh"
    with pytest.raises(RuntimeError):
        cache.get("other", 60, broken)
