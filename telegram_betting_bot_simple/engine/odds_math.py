"""Odds conversions, bookmaker-margin removal and parlay maths.

Shared by the bot and the dashboard so both show the same numbers.
Decimal odds are the working format: a parlay's decimal odds are the
product of its legs' decimal odds, and a winning stake returns
``stake * decimal``.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable, List, Sequence


def american_to_decimal(price: float) -> float:
    """Convert American odds (e.g. -150, +130) to decimal odds (1.667, 2.30)."""
    if price >= 100:
        return 1 + price / 100
    if price <= -100:
        return 1 + 100 / -price
    raise ValueError(f"American odds must be <= -100 or >= +100, got {price}")


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds to American odds (+ for 2.0 and above, - below)."""
    if decimal <= 1:
        raise ValueError(f"Decimal odds must be greater than 1, got {decimal}")
    if decimal >= 2:
        return (decimal - 1) * 100
    return -100 / (decimal - 1)


def format_american(price: float) -> str:
    """American odds with an explicit sign and no decimals, e.g. "+130" or "-150"."""
    return f"{round(price):+d}"


def implied_probability(decimal: float) -> float:
    """The win probability a price implies, bookmaker margin included."""
    return 1 / decimal


def remove_vig(decimals: Sequence[float]) -> List[float]:
    """Fair probabilities for one market's outcomes, with the bookmaker margin removed.

    Uses proportional normalisation: each outcome's implied probability is
    divided by their sum (the "overround"), so the results add up to 1.
    """
    implied = [1 / d for d in decimals]
    total = sum(implied)
    return [p / total for p in implied]


def parlay_decimal(decimals: Iterable[float]) -> float:
    """Combined decimal odds of a parlay: the product of its legs' decimal odds."""
    return math.prod(decimals)


def parlay_probability(probabilities: Iterable[float]) -> float:
    """Chance every leg wins, treating legs as independent (true for different games)."""
    return math.prod(probabilities)


def expected_value(probability: float, decimal: float) -> float:
    """Expected profit per 1 unit staked, given a win probability and decimal odds."""
    return probability * decimal - 1


def round_money(amount: float) -> float:
    """Round to cents, halves up (e.g. 10.005 -> 10.01), as payouts are displayed."""
    return float(Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def payout(stake: float, decimal: float) -> float:
    """Total return of a winning bet (stake included), rounded to cents."""
    return round_money(stake * decimal)
