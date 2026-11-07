"""Tennis: The Odds API lists one league per tournament (tennis_atp_paris_masters) while ESPN shows the tours
(tennis_atp, tennis_wta). A tour has no odds market of its own, so the boards its games need are the tour's
in-season tournament boards: the preloader must ask for those.

Bug (reported 7 Nov 2026): tennis matches the bookmakers priced were not in the app (no chance, no price, and a
tournament's games ESPN has not listed yet were missing). The tour games were all outside The Odds API's listing,
and no tournament board was ever preloaded, so every tennis game went without odds: the only tennis boards ever
fetched were the ones a person opened by hand, ten days before this test.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from telegram_betting_bot_simple.data.slate import SlateBuilder
from tests.test_slate import NOW, TODAY, FakeEspn, FakeOdds, at, odds_event, store, wait_for

PARIS = "tennis_atp_paris_masters"
TOUR = "tennis_atp"
LISTING = [{"key": PARIS, "group": "Tennis", "title": "ATP Paris Masters", "active": True, "has_outrights": False},
           {"key": "basketball_nba", "group": "Basketball", "title": "NBA", "active": True, "has_outrights": False}]


def tennis_event(event_id: str, home: str, away: str, start: str) -> Dict[str, Any]:
    """ESPN's tennis scoreboard event: a tournament whose singles matches are the games."""
    return {"id": event_id, "date": start, "name": "Paris Masters",
            "groupings": [{"grouping": {"slug": "mens-singles"}, "competitions": [{
                "id": event_id, "date": start,
                "status": {"type": {"name": "STATUS_SCHEDULED", "state": "pre", "shortDetail": "Scheduled"}},
                "competitors": [
                    {"homeAway": "home", "id": "p1", "athlete": {"displayName": home, "shortName": home}},
                    {"homeAway": "away", "id": "p2", "athlete": {"displayName": away, "shortName": away}}]}]}]}


class TournamentOdds(FakeOdds):
    """The odds service, whose board for the tournament holds the final ESPN has not listed yet."""

    def league_games(self, league: str, refresh: bool = False) -> List[Dict[str, Any]]:
        self.fetches.append(league)
        if league == PARIS:
            self.store.save_odds_snapshot(league, [odds_event(league, "f1", "Ben Shelton", "Carlos Alcaraz", at(20),
                                                              home_ml=3.75, away_ml=1.27)], self.now())
        return []


@pytest.fixture()
def boards() -> Dict[Any, Any]:
    # ESPN shows the tour: a match in the next day. The Odds API lists only the tournament (the final).
    return {(TOUR, TODAY): [tennis_event("401", "Luca Van Assche", "Ben Shelton", at(6))]}


def slate_of(store, boards) -> SlateBuilder:
    return SlateBuilder(TournamentOdds(store, listing=LISTING), FakeEspn(boards), store, now=lambda: NOW, workers=2,
                        preload=True)


def games_of(builder: SlateBuilder) -> List[Dict[str, Any]]:
    return json.loads(builder.current().raw)["games"]


def test_a_tennis_tour_game_preloads_its_tournaments_board(store, boards) -> None:
    builder = slate_of(store, boards)
    builder.refresh()
    wait_for(builder)
    # The tour has no market of its own; the tournament's board is the one its games need, so it is preloaded
    assert builder.odds.fetches == [PARIS]


def test_a_tournaments_priced_game_reaches_the_slate_even_without_an_espn_game(store, boards) -> None:
    builder = slate_of(store, boards)
    builder.refresh()
    wait_for(builder)
    builder.refresh()
    wait_for(builder)
    final = [game for game in games_of(builder) if game["league"] == PARIS]
    assert [game["home"]["name"] for game in final] == ["Ben Shelton"]
    assert final[0]["chance"] and final[0]["chance"]["away"] > final[0]["chance"]["home"]
    assert final[0]["mkt"]["p"]["away"] > final[0]["mkt"]["p"]["home"]


def test_a_tennis_tour_without_prices_is_still_listed(store, boards) -> None:
    """No board for the tour's tournament yet: the tour's games stay on the slate, unpriced, never dropped."""
    builder = slate_of(store, boards)
    builder.refresh()
    wait_for(builder)
    tour = [game for game in games_of(builder) if game["league"] == TOUR]
    assert [game["home"]["name"] for game in tour] == ["Luca Van Assche"]
