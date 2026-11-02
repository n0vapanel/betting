"""Tests for the NBA game and player model."""

from __future__ import annotations

import datetime
from statistics import NormalDist

import pytest

from telegram_betting_bot_simple.engine.factors import check_sum
from telegram_betting_bot_simple.engine.nba_model import (
    GamePrediction,
    Params,
    TeamInputs,
    TeamRating,
    absence_impact,
    blend,
    player_profile,
    predict_game,
    project_players,
    season_label,
    spreads_for,
    team_rating,
)

MARKET = {
    "h2h": {"home": {"fair": 0.70}, "away": {"fair": 0.30}},
    "spreads": {"home": {"point": -6.5}, "away": {"point": 6.5}},
    "totals": {"over": {"point": 224.5}, "under": {"point": 224.5}},
}

def test_season_label() -> None:
    assert season_label(datetime.date(2026, 10, 25)) == "2026-27"
    assert season_label(datetime.date(2027, 3, 1)) == "2026-27"
    assert season_label(datetime.date(2027, 9, 1)) == "2027-28"

def test_blend_shrinks_small_samples_toward_last_season() -> None:
    assert blend(130.0, 2, 110.0, 10) == pytest.approx(113.33, abs=0.01)
    assert blend(None, 0, 110.0, 10) == 110.0
    assert blend(120.0, 5, None, 10) == 120.0
    rating = team_rating({"games": 2, "points_for": 130.0, "points_against": 100.0},
                         {"games": 82, "points_for": 110.0, "points_against": 110.0}, Params())
    assert (rating.points_for, rating.points_against) == (pytest.approx(113.33, abs=0.01), pytest.approx(108.33, abs=0.01))

def test_market_only_prediction_follows_the_spread_and_total() -> None:
    prediction = predict_game(TeamInputs("Home", None), TeamInputs("Away", None), None, MARKET, None)
    assert prediction.margin == pytest.approx(6.5)
    assert prediction.total == pytest.approx(224.5)
    assert prediction.home_score - prediction.away_score == pytest.approx(6.5)
    margin_sd, win_sd, total_sd = spreads_for(224.5, Params())
    assert (round(margin_sd, 1), round(win_sd, 1), round(total_sd, 1)) == (12.9, 12.0, 18.6)
    # The win chance is the moneyline's own margin-free chance when nothing else moves the margin
    assert prediction.home_win_prob == pytest.approx(0.70)
    # Covering exactly the market spread is a coin flip
    assert prediction.probability("spreads", "home", -6.5) == pytest.approx(0.5)
    assert prediction.probability("spreads", "away", 6.5) == pytest.approx(0.5)
    assert prediction.probability("totals", "over", 224.5) == pytest.approx(0.5)
    assert prediction.probability("h2h", "away") == pytest.approx(1 - prediction.home_win_prob)

def test_blended_prediction_uses_model_market_and_espn() -> None:
    home = TeamInputs("Home", TeamRating(118.0, 108.0, 10))
    away = TeamInputs("Away", TeamRating(112.0, 114.0, 10), injury_impact=3.0, injured=["Star"], back_to_back=True)
    params = Params()
    prediction = predict_game(home, away, 115.0, MARKET, 0.75, params)
    model = prediction.sources["model"]
    # 118 vs 114 allowed, home court, away team loses 3 (injury) + 1.25 (back-to-back)
    assert model["margin"] == pytest.approx((118 - 115 + 114 + 1) - (112 - 115 + 108 - 1) + 4.25, abs=1e-9)
    total = 0.9 * 224.5 + 0.1 * model["total"]
    assert prediction.total == pytest.approx(total)
    espn_margin = spreads_for(total, params)[1] * NormalDist().inv_cdf(0.75)
    # The market carries 90% when there is a line: a results-only model added nothing over it
    expected = model["margin"] * 0.05 + 6.5 * 0.90 + espn_margin * 0.05
    assert prediction.margin == pytest.approx(expected)
    # ...and the win chance is the moneyline's 70%, moved by how far the other sources move the margin
    win_sd = spreads_for(total, params)[1]
    assert prediction.home_win_prob == pytest.approx(NormalDist().cdf(NormalDist().inv_cdf(0.70) + (expected - 6.5) / win_sd))
    assert any("back-to-back" in factor for factor in prediction.factors)
    assert any("Star" in factor for factor in prediction.factors)
    # The text lines are rendered from typed factors (factor_text), word for word as before
    assert prediction.factors == ["Home court: Home +2.0", "Away on a back-to-back: -1.2", "Away missing Star: -3.0"]
    projections = prediction.projections()
    for key in ("margin", "total", "home_win", "model_margin", "model_total"):
        assert check_sum(projections[key]), key
    assert projections["margin"].value == prediction.margin
    assert projections["home_win"].value == pytest.approx(100 * prediction.home_win_prob)
    assert {f.key for f in projections["model_margin"].factors} >= {"home_court", "away_back_to_back", "away_injuries"}

def test_absence_impact_is_not_counted_twice() -> None:
    star = {"gp": 10, "min": 36.0, "pts": 28.0}
    assert absence_impact(star, 0, Params()) == pytest.approx(4.2)
    assert absence_impact(star, 10, Params()) == pytest.approx(4.2)  # played every game so far
    long_term = {"gp": 0, "min": 36.0, "pts": 28.0}  # missed all 30 games: mostly priced in already
    assert absence_impact(long_term, 30, Params()) == pytest.approx(4.2 * 10 / 40)

def test_player_profile_and_projection() -> None:
    seasons = [
        {"season": "2025-26", "gp": 70, "min": 34.0, "pts": 25.0, "reb": 5.0, "ast": 6.0, "fg3m": 2.0, "stl": 1.0, "blk": 0.5, "tov": 3.0},
        {"season": "2026-27", "gp": 2, "min": 36.0, "pts": 35.0, "reb": 7.0, "ast": 8.0, "fg3m": 4.0, "stl": 2.0, "blk": 1.0, "tov": 2.0},
    ]
    profile = player_profile(seasons, "2026-27", Params())
    # Minutes: 2 games this season count 2/(2+4); points per minute get 45 minutes of last season's rate
    minutes = (36 * 2 + 34 * 4) / 6
    rate = (35 * 2 + 45 * 25 / 34) / (36 * 2 + 45)
    assert profile["min"] == pytest.approx(minutes) and profile["pts"] == pytest.approx(rate * minutes)
    # Steals take far longer to stabilise: still mostly last season's rate
    assert profile["stl"] / minutes == pytest.approx((2 * 2 + 300 * 1 / 34) / (72 + 300))

    def player(pid, minutes, pts, out=False, position="G"):
        return {"id": pid, "name": pid, "position": position, "status": "Out" if out else "", "out": out,
                "profile": {"gp": 10, "min": minutes, "pts": pts, "reb": 4.0, "ast": 3.0, "fg3m": 1.0, "stl": 1.0, "blk": 0.5, "tov": 1.5}}

    players = [player("a", 34, 25), player("b", 32, 18), player("c", 30, 14, position="F"), player("d", 28, 12),
               player("e", 26, 10), player("f", 22, 8), player("g", 20, 7), player("h", 18, 6),
               player("i", 12, 4), player("j", 10, 3), player("star", 34, 28, out=True)]
    projections = project_players(players, team_points=None, pace_factor=1.0)
    playing = {p.name: p for p in projections if not p.out}
    assert sum(p.minutes for p in playing.values()) == pytest.approx(241.0, abs=0.5)
    assert all(p.minutes <= 38.0 for p in playing.values())
    # Everyone plays their usual minutes; of the 9 left over (241 - 232) only 40% go to the
    # 9-man rotation and the end of the bench gets the rest
    rotation = sum(p.minutes for name, p in playing.items() if name != "j")
    assert rotation - (34 + 32 + 30 + 28 + 26 + 22 + 20 + 18 + 12) == pytest.approx(0.4 * 9)
    assert playing["j"].minutes == pytest.approx(10 + 0.6 * 9)
    # The guards pick up more of the missing guard's scoring than the forward does
    guard = playing["a"].pts / (25 * playing["a"].minutes / 34)
    forward = playing["c"].pts / (14 * playing["c"].minutes / 30)
    assert guard > forward > 1.0
    assert projections[-1].name == "star" and projections[-1].out
    # The team's predicted score pulls points halfway (at most 10%)
    scaled = project_players(players, team_points=150.0, pace_factor=1.0)
    total = sum(p.pts for p in projections if not p.out)
    assert sum(p.pts for p in scaled if not p.out) == pytest.approx(total * 1.10, rel=0.01)

def test_crowded_roster_squeezes_the_bench_not_the_stars() -> None:
    def player(pid, minutes):
        return {"id": pid, "name": pid, "position": "G", "status": "", "out": False,
                "profile": {"gp": 10, "min": minutes, "pts": minutes / 2, "reb": 4.0, "ast": 3.0, "fg3m": 1.0,
                            "stl": 1.0, "blk": 0.5, "tov": 1.5}}

    # 13 players who averaged 300 minutes between them (new signings, rookies)
    usual = [36, 34, 32, 30, 28, 24, 20, 18, 16, 16, 16, 15, 15]
    projections = project_players([player(f"p{i}", m) for i, m in enumerate(usual)], team_points=None, pace_factor=1.0)
    minutes = {p.name: p.minutes for p in projections}
    assert sum(minutes.values()) == pytest.approx(241.0, abs=0.5)
    assert minutes["p0"] == 36 and minutes["p1"] == 34  # the stars keep their minutes
    assert "p12" not in minutes  # the end of the bench doesn't play

    # A 16-point favourite: its starters lose 0.77% of their minutes per point beyond 6, to the deep bench
    blowout = project_players([player(f"p{i}", m) for i, m in enumerate(usual)], None, 1.0, margin=16.0)
    minutes = {p.name: p.minutes for p in blowout}
    assert minutes["p0"] == pytest.approx(36 * (1 - 0.077))
    assert minutes["p10"] > 0 and minutes["p8"] == 16  # garbage time for the end of the bench only
    assert sum(minutes.values()) == pytest.approx(241.0, abs=0.5)
    # The 16-point underdog's starters lose much less
    underdog = project_players([player(f"p{i}", m) for i, m in enumerate(usual)], None, 1.0, margin=-16.0)
    assert {p.name: p.minutes for p in underdog}["p0"] == pytest.approx(36 * (1 - 0.0025 * 7))

def test_last_season_is_regressed_to_the_mean() -> None:
    # Last season +8 points above a 112 average: only 62% of that carries over
    previous = {"games": 82, "points_for": 120.0, "points_against": 112.0}
    rating = team_rating({"games": 0}, previous, Params(), averages=(None, 112.0, 115.0))
    assert rating.points_for == pytest.approx(115.0 + 0.62 * 8.0)
    assert rating.points_against == pytest.approx(115.0)
    # This season counts relative to its own average, blended in by games played
    rating = team_rating({"games": 10, "points_for": 118.0, "points_against": 110.0}, previous, Params(),
                         averages=(116.0, 112.0, 115.0))
    assert rating.points_for == pytest.approx(((118 - 116 + 115) * 10 + (115 + 0.62 * 8) * 10) / 20)

def test_whole_number_lines_can_push() -> None:
    game = GamePrediction(home_score=115, away_score=109, margin=6.0, total=224.0, home_win_prob=0.69,
                          margin_sd=13.0, total_sd=18.6)
    win, lose = game.probability("spreads", "home", -6.0), game.probability("spreads", "away", 6.0)
    push = game.push_probability("spreads", "home", -6.0)
    assert push == game.push_probability("spreads", "away", 6.0) == pytest.approx(0.0307, abs=0.001)
    assert win + lose + push == pytest.approx(1.0)
    assert win == pytest.approx(lose)  # the line sits on the prediction
    over, under = game.probability("totals", "over", 224.0), game.probability("totals", "under", 224.0)
    assert over + under + game.push_probability("totals", "over", 224.0) == pytest.approx(1.0)
    # Half-point lines can't push, and the two sides add up to 1
    assert game.push_probability("spreads", "home", -6.5) == 0
    assert game.probability("spreads", "home", -6.5) + game.probability("spreads", "away", 6.5) == pytest.approx(1.0)
    assert game.push_probability("spreads", "home", 0.0) == 0  # a pick'em can't end tied
