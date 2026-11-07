"""The matchup card's market prop lines and recent stat averages (dashboard/service.py market_player_view and
_market_players): only what ESPN actually returned, never a fabricated stat or line, and never a stat a sport
has no source for."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from telegram_betting_bot_simple.dashboard.service import DashboardService, market_player_view

NBA = "basketball_nba"
NFL = "americanfootball_nfl"


# --- the pure view: never fabricate ---------------------------------------------------------------

def test_market_player_view_only_keeps_the_cards_real_stats() -> None:
    view = market_player_view(
        {"id": "1", "name": "Jayson Tatum", "position": "F", "headshot": "https://x/t.png", "status": "Active",
         "last10": {"pts": 28.5, "reb": 8.1, "ast": 5.2, "fg3m": 3.0}},
        [{"stat": "pts", "name": "Total Points", "line": 27.5, "kind": "over_under",
          "over_price": Decimal("1.91"), "under_price": Decimal("1.91")}],
    )
    assert view["stats"] == {"pts": 28.5, "reb": 8.1, "ast": 5.2, "fg3m": 3.0}
    # Basketball stats only: no football key was invented
    assert "PASS YDS" not in view["stats"] and "REC YDS" not in view["stats"] and "HR" not in view["stats"]
    assert view["props"][0]["stat"] == "pts" and view["props"][0]["line"] == 27.5


def test_market_player_view_keeps_each_sports_own_stats() -> None:
    football = market_player_view(
        {"id": "2", "name": "Tyreek Hill", "last10": {"PASS YDS": 0.0, "RUSH YDS": 5.0, "REC YDS": 95.0,
                                                       "REC TD": 0.8}},
        [{"stat": "REC YDS", "name": "Receiving Yards", "line": 75.5, "kind": "over_under",
          "over_price": Decimal("1.87"), "under_price": Decimal("1.87")}],
    )
    assert football["stats"] == {"PASS YDS": 0.0, "RUSH YDS": 5.0, "REC YDS": 95.0, "REC TD": 0.8}
    # No basketball stat leaked into football
    assert "pts" not in football["stats"] and "reb" not in football["stats"] and "ast" not in football["stats"]


def test_market_player_view_is_empty_without_a_source() -> None:
    assert market_player_view({}, [])["stats"] == {} and market_player_view({}, [])["props"] == []
    assert market_player_view({"id": "9", "last10": None}, [])["stats"] == {}
    # A prop without a stat or a line is dropped, never fabricated
    assert market_player_view({"id": "9", "last10": {}}, [{"stat": None, "line": 5}])["props"] == []
    assert market_player_view({"id": "9", "last10": {}}, [{"stat": "pts", "line": None}])["props"] == []


def test_market_player_view_prices_are_real_decimal_odds() -> None:
    view = market_player_view({"id": "1", "last10": {"pts": 10}}, [
        {"stat": "pts", "line": 20, "over_price": Decimal("1.91"), "under_price": None}])
    assert view["props"][0]["over"]["decimal"] == 1.91 and isinstance(view["props"][0]["over"]["american"], str)
    assert view["props"][0]["under"] is None


# --- the matchup assembly: which sport gets what, and no source -> None ----------------------------

def _service(roster_by_team, props_by_player, cards_by_id):
    class Espn:
        def roster(self, league, team_id):
            return list(roster_by_team.get(team_id, []))

    class Players:
        def props(self, league, event_id):
            return {"event_id": event_id, "league": league, "book": "DraftKings", "players": props_by_player,
                    "count": sum(len(v) for v in props_by_player.values()), "as_of": None, "stale": False,
                    "reason": None}

        def card(self, league, athlete_id):
            return dict(cards_by_id[str(athlete_id)])

    service = DashboardService(SimpleNamespace(), Espn(), ZoneInfo("America/New_York"))
    service._player_data = Players()
    return service


def test_market_players_surfaces_football_props_and_recent_stats() -> None:
    roster_by_team = {"1": [{"id": "10", "name": "Tyreek Hill", "position": "WR", "headshot": "https://x/t.png",
                             "status": ""}]}
    props_by_player = {"10": [{"stat": "REC YDS", "name": "Receiving Yards", "line": 75.5, "kind": "over_under",
                               "over_price": Decimal("1.87"), "under_price": Decimal("1.87")}]}
    cards_by_id = {"10": {"id": "10", "name": "Tyreek Hill", "position": "WR", "headshot": "https://x/t.png",
                          "status": "", "last10": {"PASS YDS": 0.0, "RUSH YDS": 5.0, "REC YDS": 95.0,
                                                   "REC TD": 0.8}}}
    service = _service(roster_by_team, props_by_player, cards_by_id)
    out = service._market_players(NFL, {"match": {"event_id": "401", "home_id": "1", "away_id": "2"}})
    assert list(out) == ["home", "away"] and out["away"] == []
    [hill] = out["home"]
    assert hill["name"] == "Tyreek Hill" and hill["stats"] == {"PASS YDS": 0.0, "RUSH YDS": 5.0,
                                                               "REC YDS": 95.0, "REC TD": 0.8}
    assert "pts" not in hill["stats"] and "reb" not in hill["stats"]  # no basketball stat leaked in
    assert hill["props"][0]["stat"] == "REC YDS" and hill["props"][0]["line"] == 75.5


def test_market_players_returns_none_without_props_roster_or_match() -> None:
    service = _service({}, {}, {})
    assert service._market_players(NFL, {}) is None  # no match
    assert service._market_players(NFL, {"match": {"event_id": "401", "home_id": "1", "away_id": "2"}}) is None  # no props
