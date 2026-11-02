"""Tests for bet slip evaluation (singles and multis)."""

from __future__ import annotations

import datetime

import pytest

from telegram_betting_bot_simple.engine.odds_math import american_to_decimal
from telegram_betting_bot_simple.engine.slip import MAX_LEGS, Leg, evaluate_slip

NOW = datetime.datetime(2026, 10, 25, 12, 0, tzinfo=datetime.timezone.utc)
LATER = "2026-10-25T23:00:00Z"

def leg(event, label, american, probability=None, start=LATER, market="h2h", side="home"):
    return Leg(event_id=event, market=market, side=side, price=american_to_decimal(american),
               label=label, game=f"Game {event}", start=start, probability=probability)

def test_three_leg_multi() -> None:
    legs = [leg("a", "Thunder", -150, 0.62), leg("b", "Celtics", 130, 0.45), leg("c", "Over 224.5", -110, 0.52)]
    slip = evaluate_slip(legs, "multi", stake=10, now=NOW)
    assert slip["valid"] and slip["reasons"] == []
    assert slip["odds"] == {"decimal": 7.32, "american": "+632"}
    assert slip["returns"] == 73.18
    assert slip["profit"] == 63.18
    assert slip["probability"] == pytest.approx(0.62 * 0.45 * 0.52)
    assert slip["ev"] == pytest.approx(0.62 * 0.45 * 0.52 * (1.6667 * 2.30 * 1.9091) - 1, abs=1e-3)

def test_multi_blocks_same_game_started_and_single_leg() -> None:
    assert evaluate_slip([leg("a", "Thunder", -150)], "multi", stake=10, now=NOW)["reasons"] == [
        "Add at least 2 picks from different games for a multi"
    ]
    slip = evaluate_slip(
        [leg("a", "Thunder", -150), leg("a", "Over 224.5", -110, market="totals", side="over"),
         leg("b", "Celtics", 130, start="2026-10-25T11:00:00Z")],
        "multi", stake=10, now=NOW,
    )
    assert not slip["valid"]
    assert slip["reasons"] == [
        "Thunder and Over 224.5 are from the same game; pick one (same-game multis aren't supported)",
        "Celtics has already started",
    ]
    assert slip["probability"] is None  # no model probabilities given
    too_many = [leg(str(i), f"Team {i}", -110) for i in range(MAX_LEGS + 1)]
    assert f"at most {MAX_LEGS} picks" in evaluate_slip(too_many, "multi", stake=1, now=NOW)["reasons"][0]

def test_singles_return_each_stake_times_its_odds() -> None:
    legs = [leg("a", "Thunder", -150, 0.62), leg("b", "Celtics", 130, 0.45)]
    slip = evaluate_slip(legs, "single", stakes={0: 15, 1: 10}, now=NOW)
    assert [item["returns"] for item in slip["legs"]] == [25.0, 23.0]
    assert slip["stake"] == 25.0 and slip["returns"] == 48.0
    assert slip["legs"][1]["ev"] == pytest.approx(0.45 * 2.3 - 1)
    assert slip["valid"]
