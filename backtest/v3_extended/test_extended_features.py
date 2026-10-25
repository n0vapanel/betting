"""Tests for the v3_extended new-feature modules (no network, no large files).

Run:
    /Users/x/.hermes/cache/scratch/msports-venv/bin/python -B -m pytest \
      -p no:cacheprovider backtest/v3_extended/test_extended_features.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backtest.v3_extended import features_extended as F
from backtest.v3_extended import evaluation_extended as E


# ------------------------------------------------------------------ official filing

def _inj_frame(rows):
    return pd.DataFrame(rows, columns=["snapshot_date", "snapshot_time", "game_date",
                                       "game_time", "matchup", "team", "player",
                                       "status", "reason"])


def test_official_filing_evening_uses_game_day_5pm():
    injuries = _inj_frame([
        ["2022-01-04", "17:00", "2022-01-05", "07:30 (ET)", "DET@CHA", "Charlotte Hornets",
         "Irving, Kyrie", "Out", "Injury"],
        ["2022-01-05", "17:00", "2022-01-05", "07:30 (ET)", "DET@CHA", "Charlotte Hornets",
         "Irving, Kyrie", "Questionable", "Injury"],
    ])
    rows, label, nyss = F.official_filing(injuries, "2022-01-05", "DET@CHA")
    assert label == "game-day 5PM official filing"
    assert nyss == 0
    assert (rows.status == "Questionable").all()


def test_official_filing_early_tip_uses_day_before_and_drops_nyss():
    injuries = _inj_frame([
        ["2022-01-16", "17:00", "2022-01-17", "12:00 (ET)", "BOS@CHA", "Charlotte Hornets",
         "Ball, LaMelo", "Questionable", "Injury"],
        ["2022-01-16", "17:00", "2022-01-17", "12:00 (ET)", "BOS@CHA", "Boston Celtics",
         None, None, "NOT YET SUBMITTED"],
        ["2022-01-17", "17:00", "2022-01-17", "12:00 (ET)", "BOS@CHA", "Charlotte Hornets",
         "Ball, LaMelo", "Out", "Injury"],
    ])
    rows, label, nyss = F.official_filing(injuries, "2022-01-17", "BOS@CHA")
    assert label == "day-before 5PM official filing (early tip)"
    assert nyss == 1
    assert len(rows) == 1
    assert rows.status.iloc[0] == "Questionable"  # game-day 5PM is post-tip, must not be used


def test_official_filing_missing_rows():
    injuries = _inj_frame([
        ["2022-01-04", "17:00", "2022-01-06", "07:30 (ET)", "DET@CHA", "Charlotte Hornets",
         "Irving, Kyrie", "Out", "Injury"],
    ])
    rows, label, nyss = F.official_filing(injuries, "2022-01-05", "DET@CHA")
    assert rows is None
    assert "no official 5PM filing rows" in label


# ------------------------------------------------------------------ rotation ewma

def test_rotation_ewma_hand_computed():
    by_team_date = {
        ("BOS", "2022-01-01"): pd.Series({"a": 30.0, "b": 20.0}),
        ("BOS", "2022-01-03"): pd.Series({"a": 40.0, "c": 30.0}),
    }
    team_dates = {"BOS": ["2022-01-01", "2022-01-03"]}
    rotation, n = F.rotation_ewma(by_team_date, team_dates, "BOS", "2022-01-05")
    assert n == 2
    q = 2 ** (-1 / 10)
    weights = np.array([q, 1.0])
    expected_a = (30.0 * q + 40.0) / weights.sum()
    expected_b = (20.0 * q + 0.0) / weights.sum()
    expected_c = (0.0 * q + 30.0) / weights.sum()
    assert rotation == {"a": pytest.approx(expected_a), "b": pytest.approx(expected_b),
                        "c": pytest.approx(expected_c)}
    empty, n0 = F.rotation_ewma(by_team_date, {"NOP": []}, "NOP", "2022-01-05")
    assert empty == {} and n0 == 0


# ------------------------------------------------------------------ minutes load

def _players():
    return pd.DataFrame({
        "game_id": ["g1", "g1", "g1", "g2", "g2", "g2"],
        "team_abbreviation": ["BOS", "BOS", "BOS", "BOS", "BOS", "BOS"],
        "game_date": ["2021-10-20", "2021-10-20", "2021-10-20",
                      "2021-10-22", "2021-10-22", "2021-10-22"],
        "player_id": ["1", "2", "3", "1", "2", "4"],
        "min": [30.0, 20.0, 10.0, 35.0, 25.0, 15.0],
    })


def test_compact_and_rolling_no_lookahead():
    compact = F.compact_team_minutes(_players())
    assert compact.iloc[0].team_min == 60.0
    assert compact.iloc[0].top8_min == 60.0
    assert set(compact.iloc[0].top5_ids.split("|")) == {"1", "2", "3"}
    roll = F.minutes_load_features(compact)
    assert pd.isna(roll.iloc[0].load7)  # cold start
    assert roll.iloc[1].load7 == 60.0  # 2 days later, within 7 days
    assert roll.iloc[1].top8share == 1.0
    assert pd.isna(roll.iloc[1].stability)  # only one prior game
    assert (roll.history_max_date.fillna("") < roll.date).all()


def test_minutes_load_seven_day_window_and_stability():
    compact = pd.DataFrame([
        {"game_id": "g0", "team": "BOS", "date": "2021-10-10", "team_min": 240.0,
         "top8_min": 200.0, "top5_ids": "a|b|c|d|e"},
        {"game_id": "g1", "team": "BOS", "date": "2021-10-13", "team_min": 240.0,
         "top8_min": 190.0, "top5_ids": "a|b|c|f|g"},
        {"game_id": "g2", "team": "BOS", "date": "2021-10-16", "team_min": 240.0,
         "top8_min": 200.0, "top5_ids": "a|b|c|d|e"},
        {"game_id": "g3", "team": "BOS", "date": "2021-10-24", "team_min": 240.0,
         "top8_min": 210.0, "top5_ids": "a|b|c|d|h"},
    ])
    roll = F.minutes_load_features(compact)
    row = roll.iloc[3]
    assert row.load7 == 0.0  # previous game 8 days earlier -> outside 7-day window
    assert row.top8share == pytest.approx((200 + 190 + 200) / 3 / 240.0)
    assert row.stability == pytest.approx(((3 / 5) + (3 / 5)) / 2)


def test_minutes_load_duplicate_players_are_aggregated():
    players = _players()
    doubled = pd.concat([players, players.iloc[[0]]], ignore_index=True)
    compact = F.compact_team_minutes(doubled)
    assert len(compact) == 2  # the duplicated player row is absorbed into the team-game
    assert not compact.duplicated(["game_id", "team"]).any()
    # the real guard lives at the player level and is asserted in build_features()
    assert doubled.duplicated(["game_id", "team_abbreviation", "player_id"]).any()


# ------------------------------------------------------------------ gates / eval

def _frame(n=10):
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, n).astype(float)
    market = np.full(n, 0.5)
    return pd.DataFrame({
        "date": [f"2022-03-{i + 1:02d}" for i in range(n)],
        "home_won": y, "p_market_home": market,
        "dec_home": np.full(n, 2.0), "dec_away": np.full(n, 2.0),
    })


def test_flat_bet_pnl_arithmetic():
    frame = _frame()
    p = np.full(len(frame), 0.6)
    bets = E.bet_frame(frame, p, "flat")
    assert (bets.bet == True).all()  # noqa: E712
    assert bets.pnl.sum() == pytest.approx(np.sum(np.where(bets.won, 1.0, -1.0)))
    metrics = E.evaluate(frame, p)
    assert metrics["n_games"] == len(frame)
    assert metrics["flat_roi"] == pytest.approx(metrics["flat_pnl"] / len(frame))
    assert 0 <= metrics["brier"] <= 1 and 0 <= metrics["market_brier"] <= 1


def test_edge_gate_zero_edge_never_bets():
    frame = _frame()
    p = frame.p_market_home.to_numpy().copy()
    for variant in ["lean", "winner"]:
        bets = E.bet_frame(frame, p, variant, eps=0.01)
        assert not bets.bet.any()


def test_bootstrap_date_clustered_shape():
    frame = _frame(10)
    p = np.full(len(frame), 0.55)
    ci = E.bootstrap_intervals(frame, p, "winner", 0.0, n_boot=50)
    assert ci["n_resamples"] == 50 and ci["n_dates"] == 10
    assert ci["accuracy"][0] <= ci["accuracy"][1]
    assert ci["flat_roi"][0] <= ci["flat_roi"][1]


def test_choose_gate_needs_minimum_bets():
    frame = _frame(8)
    frame["p_model_home"] = np.linspace(0.3, 0.7, 8)
    gate, table = E.choose_gate(frame)
    assert gate["minimum_bets"] == E.MINIMUM_VALIDATION_BETS
    assert all(row["n_bets"] <= 8 for row in table)
    assert gate["selection_period"] == E.VALIDATION_PERIOD
