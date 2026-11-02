"""Typed factors (engine/factors.py) and the NBA model's explanations: every projection sums to its value."""

from __future__ import annotations

import datetime
import itertools
import math
from zoneinfo import ZoneInfo

import pytest

from telegram_betting_bot_simple.dashboard.service import DashboardService
from telegram_betting_bot_simple.engine import nba_model
from telegram_betting_bot_simple.engine.factors import (
    NOISY,
    Factor,
    Projection,
    allocate,
    check_sum,
    factor_lines,
    project,
    round_to_sum,
    split_by_share,
)
from telegram_betting_bot_simple.engine.nba_model import (
    GamePrediction,
    Params,
    TeamInputs,
    TeamRating,
    explain_players,
    factor_text,
    model_source,
    predict_game,
    project_players,
)
from tests.conftest import FakeEspn, FakeOdds

# --- allocate and the sum contract -------------------------------------------------------------------

WORKED = {
    "multipliers": {"opponent": ("Opponent", 1.02, "4% softer defence, shrunk by half", "ESPN"),
                    "pace": ("Pace", 1.025), "usage": ("Usage", 1.02, "teammate out"),
                    "back_to_back": ("Back-to-back", 0.985)},
    # 25.1 points in 34.0 minutes, projected for 35: + 25.1 / 34 = 0.738
    "additive": {"minutes": ("Minutes", 25.1 * (35 / 34 - 1), "35.0 min vs 34.0")},
}


def test_allocate_reproduces_the_reports_worked_example() -> None:
    projection = project("25.1 season", 25.1, unit="pts", **WORKED)
    deltas = {f.key: f.delta for f in projection.factors}
    # B' = 25.838235, prod m = 1.02 x 1.025 x 1.02 x 0.985 = 1.05041385, P = 27.140840, P - B' = 1.302605,
    # sum ln m = .0198026 + .0246926 + .0198026 - .0151136 = .0491842; c_i = 1.302605 x ln m_i / .0491842
    assert projection.value == pytest.approx(27.140840, abs=1e-6)
    assert deltas["minutes"] == pytest.approx(0.738235, abs=1e-6)
    assert deltas["opponent"] == pytest.approx(0.524457, abs=1e-6)
    assert deltas["pace"] == pytest.approx(0.653964, abs=1e-6)
    assert deltas["usage"] == pytest.approx(0.524457, abs=1e-6)
    assert deltas["back_to_back"] == pytest.approx(-0.400273, abs=1e-6)
    assert check_sum(projection, tol=1e-9)
    # Additive terms first, then the multipliers in the order given
    assert [f.key for f in projection.factors] == ["minutes", "opponent", "pace", "usage", "back_to_back"]
    # The report's sentence, sorted by size, every number to one decimal and summing exactly
    shown = projection.to_dict(1)
    parts = sorted(shown["factors"], key=lambda f: -abs(f["delta"]))
    names = {"minutes": "minutes", "pace": "pace", "opponent": "opponent", "usage": "usage",
             "back_to_back": "back-to-back"}
    text = f"{shown['value']} pts = {shown['baseline']} season" + "".join(
        f" {'+' if f['delta'] >= 0 else '−'} {abs(f['delta'])} {names[f['key']]}" for f in parts)
    assert text == "27.1 pts = 25.1 season + 0.7 minutes + 0.7 pace + 0.5 opponent + 0.5 usage − 0.4 back-to-back"
    assert shown["baseline"] + sum(f["delta"] for f in shown["factors"]) == pytest.approx(shown["value"], abs=1e-9)


def test_allocate_is_independent_of_order_and_typed() -> None:
    reference = {f.key: f.delta for f in allocate(25.1, **WORKED)}
    for order in itertools.permutations(WORKED["multipliers"]):
        shuffled = {key: WORKED["multipliers"][key] for key in order}
        assert {f.key: f.delta for f in allocate(25.1, shuffled, WORKED["additive"])} == pytest.approx(reference)
    factors = {f.key: f for f in allocate(25.1, unit="pts", **WORKED)}
    opponent = factors["opponent"]
    assert (opponent.label, opponent.unit, opponent.direction, opponent.evidence, opponent.source, opponent.confidence,
            opponent.as_of) == ("Opponent", "pts", "up", "4% softer defence, shrunk by half", "ESPN", "normal", None)
    assert factors["back_to_back"].direction == "down"
    noisy = allocate(10.0, {"ref": ("Referee", 1.01, "k = 30 games", "Model", NOISY, "2026-10-27T12:00:00Z")})[0]
    assert noisy.confidence == "small, noisy" and noisy.as_of == "2026-10-27T12:00:00Z"
    assert noisy.delta == pytest.approx(0.1)


def test_allocate_edge_cases() -> None:
    # Multipliers that cancel: no net change, but each still shows its own size (the limit B' ln m)
    factors = allocate(20.0, {"up": ("Up", 1.05), "down": ("Down", 1 / 1.05)})
    assert factors[0].delta == pytest.approx(20 * math.log(1.05)) and factors[1].delta == pytest.approx(-20 * math.log(1.05))
    assert sum(f.delta for f in factors) == pytest.approx(0.0, abs=1e-12)
    # A multiplier of 1 contributes nothing; a zero base leaves nothing to multiply
    assert allocate(20.0, {"pace": ("Pace", 1.0)})[0].delta == 0.0
    assert allocate(20.0, {"pace": ("Pace", 1.0)})[0].direction == "none"
    assert [f.delta for f in allocate(0.0, {"a": ("A", 1.2)}, {"m": ("Minutes", 0.0)})] == [0.0, 0.0]
    # Additive terms only, or multipliers only
    assert check_sum(project("base", 10.0, additive={"a": ("A", 2.5), "b": ("B", -1.0)}))
    single = project("base", 10.0, {"x": ("X", 1.3)})
    assert single.value == pytest.approx(13.0) and single.factors[0].delta == pytest.approx(3.0)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            allocate(10.0, {"bad": ("Bad", bad)})
    with pytest.raises(ValueError):
        allocate(10.0, additive={"bad": ("only a label",)})


def test_check_sum_and_rounding() -> None:
    projection = Projection("Market spread", 5.5, [Factor("a", "A", 0.24, "pts"), Factor("b", "B", 0.13, "pts")],
                            5.87, "pts")
    assert check_sum(projection) and projection.residual() == pytest.approx(0.0)
    assert not check_sum(Projection("x", 5.5, [Factor("a", "A", 0.24)], 5.87))
    assert check_sum(Projection("x", 5.5, [Factor("a", "A", 0.365)], 5.87))  # off by .005: inside 0.01
    assert not check_sum(Projection("x", 5.5, [Factor("a", "A", float("nan"))], 5.88))
    # Rounded to one decimal, naive rounding gives 5.5 + 0.2 + 0.1 = 5.8 against a shown 5.9; the largest
    # remainder keeps the sum exact
    shown = projection.to_dict(1)
    assert (shown["baseline"], shown["value"]) == (5.5, 5.9)
    assert [f["delta"] for f in shown["factors"]] == [0.3, 0.1]
    assert projection.to_dict()["factors"][0]["delta"] == 0.24  # full precision without digits
    assert round_to_sum([0.33, 0.33, 0.34], 1.0, 1) == [0.3, 0.3, 0.4]
    assert sorted(round_to_sum([-0.26, -0.26], -0.5, 1)) == [-0.3, -0.2]
    assert round_to_sum([], 0.0, 1) == []
    assert round_to_sum([float("nan"), 0.26], 1.0, 1)[1] == 0.3  # no crash on a bad number
    for values in ([1.234, -0.567, 2.111, 0.049], [0.01] * 9, [-3.3333, 1.6667, 1.6666]):
        rounded = round_to_sum(values, sum(values), 2)
        assert sum(rounded) == pytest.approx(round(sum(values), 2), abs=1e-9)
        assert all(abs(r - v) < 0.01 + 1e-9 for r, v in zip(rounded, values))


def test_factor_fields_and_helpers() -> None:
    factor = Factor("pace", "Pace", 0.65, "pts", "down", "fast game", "Team model", None, "normal")
    assert factor.direction == "up"  # always from the sign of delta
    assert Factor("x", "X", -0.2).direction == "down" and Factor("x", "X", 0).direction == "none"
    with pytest.raises(ValueError):
        Factor("x", "X", 1.0, confidence="certain")
    assert factor.to_dict() == {"key": "pace", "label": "Pace", "delta": 0.65, "unit": "pts", "direction": "up",
                                "evidence": "fast game", "source": "Team model", "as_of": None, "confidence": "normal"}
    assert factor_lines([factor, Factor("ref", "Referee", -0.04, "cards", confidence=NOISY)]) == [
        "Pace: +0.7 pts", "Referee: -0.0 cards (small, noisy)"]
    assert split_by_share(1.0, {"a": 3.0, "b": 1.0}, slope=0.0) == {"a": 0.75, "b": 0.25}
    assert split_by_share(0.0, {"a": 2.0, "b": -2.0}, slope=0.5) == {"a": 1.0, "b": -1.0}


# --- the NBA model: every projection sums to the number shown ----------------------------------------

MARKETS = {
    "full": {"h2h": {"home": {"fair": 0.70}, "away": {"fair": 0.30}},
             "spreads": {"home": {"point": -6.5}, "away": {"point": 6.5}},
             "totals": {"over": {"point": 224.5}, "under": {"point": 224.5}}},
    "moneyline only": {"h2h": {"home": {"fair": 0.35}, "away": {"fair": 0.65}}},
    "spread only": {"spreads": {"home": {"point": 3.5}, "away": {"point": -3.5}}},
    "total only": {"totals": {"over": {"point": 231.0}, "under": {"point": 231.0}}},
    "longshot": {"h2h": {"home": {"fair": 0.999}, "away": {"fair": 0.001}}},
    "none": {},
}


def assert_explained(prediction: GamePrediction) -> None:
    """Every projection of a game sums to its value, and the values are the prediction's own numbers."""
    projections = prediction.projections()
    assert {"margin", "total", "home_win"} <= set(projections)
    for key, projection in projections.items():
        assert check_sum(projection, tol=1e-9), (key, projection.residual())
        assert check_sum(projection), key
        assert projection.key == key and projection.label and projection.baseline_label
        for factor in projection.factors:
            assert factor.unit and factor.label and factor.direction in ("up", "down", "none")
            assert factor.confidence in ("normal", "small, noisy")
        assert len({f.key for f in projection.factors}) == len(projection.factors)
        shown = projection.to_dict(1)
        assert shown["baseline"] + sum(f["delta"] for f in shown["factors"]) == pytest.approx(shown["value"], abs=1e-9)
    assert projections["margin"].value == prediction.margin
    assert projections["total"].value == prediction.total
    assert projections["home_win"].value == pytest.approx(100 * prediction.home_win_prob, abs=1e-12)
    assert projections["margin"].unit == "pts" and projections["home_win"].unit == "%"
    if "model" in prediction.sources:
        assert projections["model_margin"].value == pytest.approx(prediction.sources["model"]["margin"], abs=1e-9)
        assert projections["model_total"].value == pytest.approx(prediction.sources["model"]["total"], abs=1e-9)


def team_variants():
    rated = (TeamRating(118.0, 108.0, 10), TeamRating(112.0, 114.0, 10))
    yield "no ratings", TeamInputs("Home", None), TeamInputs("Away", None)
    yield "plain", TeamInputs("Knicks", rated[0]), TeamInputs("Magic", rated[1])
    yield "away hurt", TeamInputs("Knicks", rated[0]), TeamInputs("Magic", rated[1], 3.0, ["Star"], True)
    yield "both hurt", TeamInputs("Knicks", rated[0], 4.2, ["A", "B"], True), TeamInputs("Magic", rated[1], 1.1, ["C"], True)
    yield "underdog home", TeamInputs("Wizards", TeamRating(104.0, 121.0, 3)), TeamInputs("Celtics", TeamRating(122.0, 106.0, 3))


@pytest.mark.parametrize("market", list(MARKETS))
@pytest.mark.parametrize("espn", [None, 0.75, 0.2])
@pytest.mark.parametrize("level", [None, 117.5])
def test_every_game_projection_sums(market, espn, level) -> None:
    for _name, home, away in team_variants():
        prediction = predict_game(home, away, 115.0, MARKETS[market], espn, Params(), level=level)
        if prediction is None:
            assert market in ("none", "total only") and espn is None and home.rating is None
            continue
        assert_explained(prediction)


def test_game_projections_start_from_the_market_and_keep_the_old_text() -> None:
    home = TeamInputs("Knicks", TeamRating(118.0, 108.0, 10))
    away = TeamInputs("Magic", TeamRating(112.0, 114.0, 10), injury_impact=3.0, injured=["Star"], back_to_back=True)
    prediction = predict_game(home, away, 115.0, MARKETS["full"], 0.75, Params())
    assert_explained(prediction)
    # The text lines older callers (the bot, the AI dossier, Lab notes) read are unchanged
    assert prediction.factors == ["Home court: Knicks +2.0", "Magic on a back-to-back: -1.2", "Magic missing Star: -3.0"]
    assert factor_text(prediction.projections()["model_margin"].factors) == [
        "Knicks strength: +10.0 pts", "Opponent: Magic (net -2.0): +2.0 pts", "Home court: Knicks +2.0",
        "Magic on a back-to-back: -1.2", "Magic missing Star: -3.0"]
    p = prediction.projections()
    # The margin starts from the market's spread; the team model and ESPN move it by their 5% weights
    assert (p["margin"].baseline_label, p["margin"].baseline) == ("Market spread: Knicks -6.5", 6.5)
    model, espn = (next(f for f in p["margin"].factors if f.key == key) for key in ("team_model", "espn_predictor"))
    assert model.delta == pytest.approx(0.05 * (prediction.sources["model"]["margin"] - 6.5))
    assert espn.delta == pytest.approx(0.05 * (prediction.sources["espn"]["margin"] - 6.5))
    assert model.confidence == "small, noisy" and espn.confidence == "normal"
    # The win chance starts from the moneyline's 70% and the same two factors, in percentage points
    assert (p["home_win"].baseline, p["home_win"].unit) == (pytest.approx(70.0), "%")
    assert [f.key for f in p["home_win"].factors] == ["team_model", "espn_predictor"]
    assert all(f.unit == "pp" for f in p["home_win"].factors)
    # The total starts from the market's 224.5; the team model has 10%
    assert p["total"].baseline == 224.5 and p["total"].factors[0].delta == pytest.approx(
        0.1 * (prediction.sources["model"]["total"] - 224.5))
    # The team model's own margin: even teams, then strength, home court, rest and injuries
    keys = [f.key for f in p["model_margin"].factors]
    assert keys == ["home_rating", "away_rating", "home_court", "away_back_to_back", "away_injuries"]
    assert [f.delta for f in p["model_margin"].factors] == pytest.approx([10.0, 2.0, 2.0, 1.25, 3.0])
    assert p["model_total"].baseline == 230.0
    assert [f.delta for f in p["model_total"].factors] == pytest.approx([-4.0, -4.0, -0.5, -1.2])


def test_without_a_market_the_margin_builds_up_from_an_even_game() -> None:
    home = TeamInputs("Knicks", TeamRating(118.0, 108.0, 10), back_to_back=True)
    away = TeamInputs("Magic", TeamRating(112.0, 114.0, 10))
    prediction = predict_game(home, away, 115.0, {}, 0.6, Params())
    p = prediction.projections()
    assert (p["margin"].baseline_label, p["margin"].baseline) == ("Even teams on a neutral court", 0.0)
    assert (p["home_win"].baseline, p["total"].baseline) == (50.0, 230.0)
    # The team model and ESPN split the weight evenly: each model part counts half
    deltas = {f.key: f.delta for f in p["margin"].factors}
    assert deltas["home_court"] == pytest.approx(1.0) and deltas["home_back_to_back"] == pytest.approx(-0.625)
    assert deltas["espn_predictor"] == pytest.approx(0.5 * prediction.sources["espn"]["margin"])
    assert_explained(prediction)


def test_later_adjustments_show_as_one_more_factor() -> None:
    """ai/desk.py moves the win chance and margin by the analysts' trusted forecasts after the model ran."""
    prediction = predict_game(TeamInputs("Knicks", None), TeamInputs("Magic", None), None, MARKETS["full"], None)
    before = prediction.projections()
    assert before["margin"].factors == [] and before["home_win"].factors == []
    prediction.sources["before_analysts"] = {"home_win_prob": prediction.home_win_prob, "margin": prediction.margin}
    prediction.home_win_prob += 0.02
    prediction.margin += 0.8
    after = prediction.projections()
    assert after["home_win"].factors[-1].key == "analysts"
    assert after["home_win"].factors[-1].delta == pytest.approx(2.0)
    assert after["margin"].factors[-1].delta == pytest.approx(0.8)
    assert after["total"].factors == []  # unchanged
    assert_explained(prediction)
    assert prediction.base_projections["margin"].factors == []  # the stored explanation isn't touched
    # Without the desk's marker the extra factor is a plain "later adjustment"
    other = predict_game(TeamInputs("Knicks", None), TeamInputs("Magic", None), None, MARKETS["spread only"], None)
    other.total += 1.0
    assert other.projections()["total"].factors[-1].key == "adjustment"
    assert_explained(other)
    # A prediction built by hand has no explanation
    assert GamePrediction(110, 105, 5.0, 215.0, 0.66).projections() == {}


def test_model_source_is_the_team_model_alone() -> None:
    home = TeamInputs("Knicks", TeamRating(118.0, 108.0, 10))
    away = TeamInputs("Magic", TeamRating(112.0, 114.0, 10))
    prediction = predict_game(home, away, 115.0, MARKETS["full"], None, Params())
    source = model_source(prediction, "2026-10-27T18:00:00Z")
    assert (source["name"], source["kind"], source["status"]) == ("Model", "team_model", "ok")
    assert source["p"]["home"] == pytest.approx(prediction.sources["model"]["home_win_prob"])
    assert source["p"]["home"] != pytest.approx(prediction.home_win_prob)  # not the market-led prediction
    assert model_source(predict_game(TeamInputs("A", None), TeamInputs("B", None), None, MARKETS["full"], None)) is None
    assert model_source(None) is None


def player(pid, minutes, pts, out=False, position="G", gp=10):
    return {"id": pid, "name": f"Player {pid}", "position": position, "status": "Out" if out else "", "out": out,
            "profile": {"gp": gp, "min": minutes, "pts": pts, "reb": 4.0 + minutes / 10, "ast": 3.0, "fg3m": 1.0,
                        "stl": 1.0, "blk": 0.5, "tov": 1.5}}


ROSTERS = {
    "injured star": [player("a", 34, 25), player("b", 32, 18), player("c", 30, 14, position="F"), player("d", 28, 12),
                     player("e", 26, 10), player("f", 22, 8), player("g", 20, 7), player("h", 18, 6),
                     player("i", 12, 4), player("j", 10, 3), player("star", 34, 28, out=True, gp=6),
                     player("big", 25, 11, out=True, position="C")],
    "crowded": [player(f"p{i}", m, m / 2) for i, m in enumerate([36, 34, 32, 30, 28, 24, 20, 18, 16, 16, 16, 15, 15])],
    "iron man": [player("iron", 40.5, 31)] + [player(f"q{i}", m, m / 3) for i, m in enumerate([30, 28, 26, 24, 20, 18, 14])],
    "short bench": [player(f"s{i}", m, m / 2.5, position="F") for i, m in enumerate([35, 33, 30, 28, 25, 3])],
}


@pytest.mark.parametrize("roster", list(ROSTERS))
@pytest.mark.parametrize("team_points,pace", [(None, 1.0), (150.0, 1.0), (80.0, 0.8), (118.0, 1.21)])
@pytest.mark.parametrize("margin,home", [(0.0, None), (16.0, True), (-16.0, False), (7.0, True)])
@pytest.mark.parametrize("team_games", [0, 10])
def test_every_player_projection_sums(roster, team_points, pace, margin, home, team_games) -> None:
    players = ROSTERS[roster]
    args = (players, team_points, pace, Params(), margin, home, team_games)
    projections = project_players(*args)
    explained = explain_players(*args)
    playing = [p for p in projections if not p.out]
    assert set(explained) == {p.id for p in playing}
    for p in playing:
        assert set(explained[p.id]) == set(nba_model.STATS)
        for stat, projection in explained[p.id].items():
            assert check_sum(projection, tol=1e-9), (p.id, stat, projection.residual())
            assert projection.value == getattr(p, stat)  # the number shown, to the last digit
            assert projection.unit == nba_model.STAT_UNITS[stat] and projection.key == stat
    # Explaining doesn't change the projections themselves
    assert [vars(p) for p in project_players(*args)] == [vars(p) for p in projections]


def test_player_factors_name_their_reasons() -> None:
    players = ROSTERS["injured star"]
    explained = explain_players(players, 121.0, 1.04, Params(), 12.0, True, 10)
    points = explained["a"]["pts"]
    assert points.baseline == pytest.approx(25.0) and points.baseline_label.startswith("Usual: 34.0 min")
    factors = {f.key: f for f in points.factors}
    assert {"minutes_open", "minutes_blowout", "usage_star", "usage_big", "scoring", "venue"} == set(factors)
    assert factors["minutes_blowout"].direction == "down" and factors["usage_star"].direction == "up"
    assert factors["venue"].confidence == "small, noisy" and factors["venue"].label == "Home floor"
    assert "played 60%" in factors["usage_star"].evidence  # 6 of the team's 10 games
    # Rebounds get pace, not the team's projected score; blocks get pace and the home split
    assert {f.key for f in explained["a"]["reb"].factors} >= {"pace"}
    assert "scoring" not in {f.key for f in explained["a"]["reb"].factors}
    assert {"pace", "venue"} <= {f.key for f in explained["a"]["blk"].factors}
    # Usage bumps reach only points, assists, threes and rebounds
    assert not any(f.key.startswith("usage") for f in explained["a"]["stl"].factors)


# --- every projection the model makes on the dashboard's test fixtures ---------------------------------

TEAM_IDS = {"Oklahoma City Thunder": "25", "Los Angeles Clippers": "12", "New York Knicks": "18", "Orlando Magic": "19"}
POSITIONS = ["PG", "SG", "SF", "PF", "C", "G", "F", "G", "C", "F", "G"]


class RichEspn:
    """ESPN stand-in with everything the NBA model reads: standings for both seasons, the predictor, injuries
    (one star out, one doubtful), a back-to-back, rosters and every player's seasons."""

    def find_game(self, league, home, away, start):
        return {"event_id": f"e{TEAM_IDS[home]}", "home_id": TEAM_IDS[home], "away_id": TEAM_IDS[away],
                "day": datetime.date(2029, 12, 31)}

    def summary(self, league, event_id):
        home_id = event_id[1:]
        injuries = [{"team": {"id": "12"}, "injuries": [{"athlete": {"id": "12-0", "displayName": "Clipper Star"},
                                                         "status": "Out"}]},
                    {"team": {"id": "19"}, "injuries": [{"athlete": {"id": "19-2", "displayName": "Magic Wing"},
                                                         "status": "Doubtful"}]}]
        predictor = {"homeTeam": {"id": home_id, "gameProjection": "71.4" if home_id == "25" else "58.0"}}
        return {"predictor": predictor, "injuries": injuries}

    def standings(self, league, season=None):
        table = {"25": (121.0, 107.0), "12": (113.0, 112.0), "18": (117.0, 111.0), "19": (110.0, 109.0),
                 "1": (114.0, 118.0), "2": (112.0, 115.0)}
        games = 82 if season else 6
        shift = -1.5 if season else 0.0
        return {team: {"games": games, "points_for": pf + shift, "points_against": pa + shift}
                for team, (pf, pa) in table.items()}

    def played_on(self, league, day):
        return {"12"}  # the Clippers played last night

    def roster(self, league, team_id):
        return [{"id": f"{team_id}-{i}", "name": f"Player {team_id}-{i}", "position": POSITIONS[i], "status": ""}
                for i in range(11)]

    def player_seasons(self, league, athlete_id):
        i = int(athlete_id.split("-")[1])
        minutes = 36.0 - 2.6 * i
        row = {"min": minutes, "pts": minutes * 0.7, "reb": minutes * 0.2, "ast": minutes * 0.15, "fg3m": minutes * 0.06,
               "stl": minutes * 0.03, "blk": minutes * 0.02, "tov": minutes * 0.05}
        return [{"season": "2025-26", "gp": 70, **row},
                {"season": nba_model.season_label(datetime.date.today()), "gp": 6,
                 **{k: v * 1.05 for k, v in row.items()}}]


def test_every_projection_on_the_dashboard_fixtures_sums(monkeypatch) -> None:
    games, players = [], []
    real_predict, real_project = nba_model.predict_game, nba_model.project_players

    def predict(*args, **kwargs):
        prediction = real_predict(*args, **kwargs)
        games.append(prediction)
        return prediction

    def project(*args, **kwargs):
        players.append((real_project(*args, **kwargs), explain_players(*args, **kwargs)))
        return players[-1][0]

    monkeypatch.setattr(nba_model, "predict_game", predict)
    monkeypatch.setattr(nba_model, "project_players", project)
    for espn in (RichEspn(), FakeEspn()):  # the full model, and the market-only one
        service = DashboardService(FakeOdds(), espn, ZoneInfo("America/New_York"))
        board = service.games("basketball_nba")
        for game_id in ("g1", "g2"):
            card = service.game("basketball_nba", game_id)
            # the text lines are still there for the page, the AI dossier and the bot
            assert all(isinstance(line, str) for line in card["prediction"]["factors"])
        assert len(board["games"]) == 2
    assert len(games) >= 8 and all(g is not None for g in games)
    with_model = [g for g in games if "model" in g.sources]
    assert with_model, "the rich fixture runs the team model"
    assert any("Clippers on a back-to-back" in line for g in with_model for line in g.factors)
    assert any("missing Clipper Star" in line for g in with_model for line in g.factors)
    for prediction in games:
        assert_explained(prediction)
    assert len(players) == 4  # both teams of both games
    count = 0
    for projections, explained in players:
        for p in projections:
            if p.out:
                assert p.id not in explained
                continue
            for stat in nba_model.STATS:
                assert check_sum(explained[p.id][stat], tol=1e-9)
                assert explained[p.id][stat].value == getattr(p, stat)
                count += 1
    assert count > 100
