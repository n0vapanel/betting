"""The game page's matchup block (dashboard/service.py ``_card`` -> ``matchup``): only real columns — each team's
W-L record, the team model's points-for/against (NBA), season leaders when ESPN's feed returns them, and the
injury list — and never a fabricated offensive rating, standings or prediction chart."""

from __future__ import annotations

import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from telegram_betting_bot_simple.dashboard.service import DashboardService, matchup_view, player_projection_view
from telegram_betting_bot_simple.data.odds_service import normalize_event
from telegram_betting_bot_simple.engine import nba_model

NBA = "basketball_nba"


def game() -> Dict[str, Any]:
    markets = [
        {"key": "h2h", "outcomes": [{"name": "Boston Celtics", "price": 1.45},
                                    {"name": "New York Knicks", "price": 2.85}]},
        {"key": "spreads", "outcomes": [{"name": "Boston Celtics", "price": 1.91, "point": -6.5},
                                        {"name": "New York Knicks", "price": 1.91, "point": 6.5}]},
        {"key": "totals", "outcomes": [{"name": "Over", "price": 1.91, "point": 221.5},
                                       {"name": "Under", "price": 1.91, "point": 221.5}]},
    ]
    return normalize_event({"id": "n1", "sport_key": NBA, "sport_title": "NBA",
                            "commence_time": "2030-01-01T01:00:00Z",
                            "home_team": "Boston Celtics", "away_team": "New York Knicks",
                            "bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": markets}]})


class FakeEspn:
    def find_game(self, league, home, away, commence_time):
        return {"event_id": "400", "home_id": "2", "away_id": "6", "day": datetime.date(2030, 1, 1)}

    def summary(self, league, event_id):
        return {
            "header": {"competitions": [{"competitors": [
                {"homeAway": "home", "team": {"id": "2"},
                 "record": [{"type": "total", "summary": "4-1"}, {"type": "home", "summary": "2-0"}]},
                {"homeAway": "away", "team": {"id": "6"},
                 "record": [{"type": "total", "summary": "1-5"}, {"type": "road", "summary": "0-3"}]},
            ]}]},
            "predictor": {"homeTeam": {"id": "2", "gameProjection": "62.0"}},
            "injuries": [
                {"team": {"id": "2"}, "injuries": [
                    {"status": "Questionable", "athlete": {"id": "100", "displayName": "Jalen Brown"}}]},
            ],
            "leaders": [
                {"team": {"id": "2"}, "leaders": [
                    {"name": "points", "displayName": "Points",
                     "leaders": [{"athlete": {"id": "100", "displayName": "Jayson Tatum"},
                                  "displayValue": "28.5"}]}]},
                {"team": {"id": "6"}, "leaders": []},
            ],
        }

    def standings(self, league, season=None):
        return {"2": {"games": 5, "points_for": 118.0, "points_against": 108.0},
                "6": {"games": 5, "points_for": 105.0, "points_against": 114.0}}

    def played_on(self, league, day):
        return set()

    def player_seasons(self, league, player_id):
        raise AssertionError("player_seasons must not be called for a non-OUT injury")


class NoEspn(FakeEspn):
    def find_game(self, *args):
        return None


class FakeOdds:
    quota = {"remaining": 300, "used": 0}

    def __init__(self, board):
        self.board = board

    def sports(self):
        return [{"key": NBA, "group": "Basketball", "title": "NBA", "active": True, "has_outrights": False}]

    def league_games(self, league, refresh=False):
        return list(self.board)

    def odds_loaded_at(self, league):
        return 123456789.0


def context() -> Dict[str, Any]:
    return {
        "current": {"2": {"games": 5, "points_for": 118.0, "points_against": 108.0},
                    "6": {"games": 5, "points_for": 105.0, "points_against": 114.0}},
        "previous": {},
        "avg": 110.0,
        "averages": [110.0, 110.0, 110.0],
        "level": None,
    }


def test_matchup_has_real_columns_and_nothing_invented() -> None:
    service = DashboardService(FakeOdds([game()]), FakeEspn(), ZoneInfo("America/New_York"))
    card, info = service._card(NBA, game(), context())

    matchup = card["matchup"]
    assert set(matchup) == {"home", "away"}

    # W-L record straight from ESPN's summary header (the "total" record only)
    assert matchup["home"]["record"] == "4-1"
    assert matchup["away"]["record"] == "1-5"

    # The team model's points-for/against from real standings, not an invented rating
    assert matchup["home"]["rating"] == {"points_for": 118.0, "points_against": 108.0, "games": 5}
    assert matchup["away"]["rating"] == {"points_for": 105.0, "points_against": 114.0, "games": 5}

    # Season leaders only where the feed returned them (the away list was empty: column left out)
    assert matchup["home"]["leaders"] == [
        {"name": "points", "displayName": "Points",
         "leaders": [{"athlete": {"id": "100", "displayName": "Jayson Tatum"}, "displayValue": "28.5"}]}]
    assert "leaders" not in matchup["away"]

    # The existing injury list is still on the card, unchanged
    assert card["injuries"] == {"home": ["Jalen Brown (Questionable)"], "away": []}

    # Nothing invented: a matchup side holds only record/rating/leaders
    assert set(matchup["home"]) == {"record", "rating", "leaders"}
    assert set(matchup["away"]) == {"record", "rating"}
    for side in ("home", "away"):
        for invented in ("offensive_rating", "defensive_rating", "standings", "prediction_chart", "recent", "form",
                         "last10"):
            assert invented not in matchup[side]


def test_matchup_is_empty_without_espn() -> None:
    # find_game -> None: no record, rating or leaders are invented for a league ESPN doesn't match
    service = DashboardService(FakeOdds([game()]), NoEspn(), ZoneInfo("America/New_York"))
    card, info = service._card(NBA, game(), context())
    assert card["matchup"] == {}


def test_matchup_view_only_keeps_sourced_columns() -> None:
    info = {
        "records": {"home": "4-1", "away": None},
        "leaders": {"home": [{"name": "points", "leaders": []}], "away": []},
        "teams": {
            "home": nba_model.TeamInputs("Boston Celtics", nba_model.TeamRating(118.0, 108.0, 5.0)),
            "away": nba_model.TeamInputs("New York Knicks", None),
        },
    }
    assert matchup_view(info) == {
        "home": {"record": "4-1",
                 "rating": {"points_for": 118.0, "points_against": 108.0, "games": 5},
                 "leaders": [{"name": "points", "leaders": []}]},
    }
    # A side with no sourced column (record/rating/leaders all missing) is left out entirely
    assert matchup_view({}) == {}


def test_player_projection_view_keeps_real_espn_headshot() -> None:
    projection = SimpleNamespace(
        id="100", name="Jayson Tatum", position="SF", status="", minutes=35.25,
        pts=28.55, reb=8.15, ast=4.75, out=False,
    )
    row = player_projection_view(
        projection,
        {"100": {"headshot": "https://a.espncdn.com/i/headshots/nba/players/full/100.png"}},
    )

    assert row["headshot"] == "https://a.espncdn.com/i/headshots/nba/players/full/100.png"
    assert row["pts"] == 28.6
    assert row["minutes"] == 35.2


def test_game_frontend_renders_broadcast_player_cards() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "telegram_betting_bot_simple/dashboard/static/js/views/game.js"
    ).read_text()

    assert 'class="matchup-player"' in source
    assert 'class="matchup-headshot"' in source
    assert "Last 10 + full prediction" in source
    assert "pl.headshot" in source
