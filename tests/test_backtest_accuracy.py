"""The backtest must not score a result it was given, and a fight grades on its winner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


streak = _load(ROOT / "backtest/multisport/mma/streak.py", "mma_streak")
odds = _load(ROOT / "backtest/multisport/soccer_v2/odds_columns.py", "soccer_odds_columns")


def test_a_loss_keeps_the_wins_that_came_before_it():
    # win, win, loss. The loss used to be forced to streak 0, which told the model the result.
    assert streak.prior_win_streak([1, 1, 0]) == [0, 1, 2]


def test_streak_does_not_change_when_the_current_result_is_flipped():
    before = streak.prior_win_streak([1, 1, 0])
    flipped = streak.prior_win_streak([1, 1, 1])
    assert before[2] == flipped[2] == 2


def test_an_unfinished_bout_does_not_reset_the_run():
    assert streak.prior_win_streak([1, 0, 1], known=[1, 0, 1]) == [0, 1, 1]


def test_streaks_do_not_cross_fighters():
    frame = pd.DataFrame({"fighter": ["A", "A", "B"], "won": [1, 0, 1]})
    streak.assign_prior_streak(frame)
    assert frame["streak_pre"].tolist() == [0, 1, 0]


def test_feature_scripts_use_the_prior_streak():
    paths = [
        ROOT / "backtest/multisport/mma/feature_utils.py",
        ROOT / "backtest/multisport/mma_v2/build_features_v2.py",
        ROOT / "backtest/multisport/mma_v2/build_features.py",
    ]
    for path in paths:
        text = path.read_text()
        assert "streak_incl" not in text, path.name
        assert "assign_prior_streak" in text, path.name


def test_closing_price_is_the_c_column():
    frame = pd.DataFrame({
        "AvgH": [2.0], "AvgD": [3.4], "AvgA": [4.0],
        "AvgCH": [1.8], "AvgCD": [3.6], "AvgCA": [4.5],
        "Avg>2.5": [1.90], "Avg<2.5": [1.90],
        "AvgC>2.5": [2.10], "AvgC<2.5": [1.75],
    })
    priced, sources = odds.assign_market_odds(frame)
    assert sources["close_1x2"] == "AvgC"
    assert priced["c_H"].iloc[0] == 1.8
    assert priced["c_Over"].iloc[0] == 2.10
    open_home = (1 / 2.0) / (1 / 2.0 + 1 / 3.4 + 1 / 4.0)
    assert priced["m_open_H"].iloc[0] == pytest.approx(open_home)
    assert priced["clv_H"].iloc[0] > 0  # the home price shortened into the close


def test_a_preclose_without_a_c_column_is_not_settled_as_the_close():
    frame = pd.DataFrame({
        "AvgH": [2.0], "AvgD": [3.2], "AvgA": [3.8],
        "Avg>2.5": [1.9], "Avg<2.5": [1.9],
    })
    priced, sources = odds.assign_market_odds(frame)
    assert sources["close_1x2"] is None
    assert sources["close_ou"] is None
    assert pd.isna(priced["c_H"].iloc[0])
    assert pd.isna(priced["c_Over"].iloc[0])


def test_fight_winner_is_the_grade():
    from telegram_betting_bot_simple.data.espn_client import EspnClient, competition_state, fight_points
    from telegram_betting_bot_simple.dashboard.service import can_grade

    fight = {
        "id": "401",
        "status": {"type": {"name": "STATUS_FINAL", "completed": True, "state": "post"}},
        "competitors": [
            {"order": 2, "winner": False, "athlete": {"displayName": "Away"}},
            {"order": 1, "winner": True, "athlete": {"displayName": "Home"}},
        ],
    }
    assert fight_points(fight) == (1, 0)
    assert competition_state(fight)[:3] == ("final", 1, 0)
    client = EspnClient()
    client.scoreboard = lambda league, day: [{"id": "600", "competitions": [fight]}]
    assert client.mma_result("401", start="2026-11-08T01:00:00Z") == {
        "state": "final", "home": 1, "away": 0, "status": "STATUS_FINAL",
    }
    assert can_grade("mma_mixed_martial_arts")
    assert can_grade("soccer_epl")
    assert not can_grade("tennis_atp")


def test_a_scheduled_fight_has_no_score_yet():
    from telegram_betting_bot_simple.data.espn_client import competition_state

    fight = {
        "status": {"type": {"name": "STATUS_SCHEDULED", "state": "pre", "completed": False}},
        "competitors": [
            {"order": 1, "winner": False},
            {"order": 2, "winner": False},
        ],
    }
    state, home, away, _start = competition_state(fight)
    assert state == "pre"
    assert home is None and away is None
