"""Tests for odds conversions and parlay maths."""

from __future__ import annotations

import pytest

from telegram_betting_bot_simple.engine.odds_math import (
    american_to_decimal,
    decimal_to_american,
    expected_value,
    format_american,
    parlay_decimal,
    parlay_probability,
    payout,
    remove_vig,
)

def test_american_decimal_round_trip() -> None:
    assert american_to_decimal(130) == pytest.approx(2.30)
    assert american_to_decimal(-150) == pytest.approx(1.6667, abs=1e-4)
    assert american_to_decimal(100) == pytest.approx(2.0)
    for price in (-1000, -150, -110, 100, 130, 625):
        assert decimal_to_american(american_to_decimal(price)) == pytest.approx(price)
    with pytest.raises(ValueError):
        american_to_decimal(50)
    with pytest.raises(ValueError):
        decimal_to_american(1.0)

def test_format_american() -> None:
    assert format_american(130.0) == "+130"
    assert format_american(-149.6) == "-150"

def test_remove_vig_makes_probabilities_sum_to_one() -> None:
    # -110 / -110: 52.4% implied each side, 50/50 fair
    fair = remove_vig([american_to_decimal(-110), american_to_decimal(-110)])
    assert fair == pytest.approx([0.5, 0.5])
    fair = remove_vig([american_to_decimal(-150), american_to_decimal(130)])
    assert sum(fair) == pytest.approx(1.0)
    assert fair[0] > fair[1]

def test_three_leg_parlay_with_favourites_and_underdog() -> None:
    legs = [american_to_decimal(-150), american_to_decimal(130), american_to_decimal(-110)]
    combined = parlay_decimal(legs)
    assert combined == pytest.approx(1.6667 * 2.30 * 1.9091, rel=1e-4)
    assert format_american(decimal_to_american(combined)) == "+632"
    assert payout(10, combined) == 73.18
    assert parlay_probability([0.6, 0.45, 0.5]) == pytest.approx(0.135)

def test_expected_value() -> None:
    assert expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert expected_value(0.55, 2.0) == pytest.approx(0.10)

def test_payout_rounds_half_up_to_cents() -> None:
    assert payout(1, 10.005) == 10.01
    assert payout(25, 1.9091) == 47.73


def test_shin_and_power_take_more_margin_off_long_shots() -> None:
    prices = [american_to_decimal(-400), american_to_decimal(320)]
    plain = remove_vig(prices)
    for method in ("shin", "power"):
        fair = remove_vig(prices, method)
        assert sum(fair) == pytest.approx(1.0)
        assert fair[0] > plain[0] and fair[1] < plain[1]  # the favourite keeps more of its chance
    # A balanced market is left at 50/50 by every method
    assert remove_vig([1.91, 1.91], "shin") == pytest.approx([0.5, 0.5])
    # Three-way markets work too, and an arbitrage-free market without margin is left alone
    assert sum(remove_vig([1.6, 5.0, 4.33], "shin")) == pytest.approx(1.0)
    assert remove_vig([2.0, 2.0], "power") == pytest.approx([0.5, 0.5])
