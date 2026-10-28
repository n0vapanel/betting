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


def remove_vig(decimals: Sequence[float], method: str = "multiplicative") -> List[float]:
    """Fair probabilities for one market's outcomes, with the bookmaker margin removed.

    "multiplicative" divides each implied probability by their sum (the
    overround). "shin" and "power" take more of the margin off long shots,
    where bookmakers put more of it (the favourite-longshot bias).
    """
    implied = [1 / d for d in decimals]
    total = sum(implied)
    if total <= 1 or method == "multiplicative":
        return [p / total for p in implied]
    if method == "shin":
        return shin_probabilities(implied)
    if method == "power":
        return power_probabilities(implied)
    raise ValueError(f"Unknown de-vig method {method!r}")


def _solve(f, low: float, high: float, steps: int = 200) -> float:
    """Root of a decreasing function on [low, high] by bisection."""
    for _ in range(steps):
        middle = (low + high) / 2
        if f(middle) > 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def shin_probabilities(implied: Sequence[float]) -> List[float]:
    """Shin's (1993) model: the margin comes from protecting against a share z of insider money.

    p_i = (sqrt(z^2 + 4 (1 - z) q_i^2 / B) - z) / (2 (1 - z)), with q_i the implied
    probabilities, B their sum and z chosen so the p_i add up to 1.
    """
    booksum = sum(implied)

    def fair(z: float) -> List[float]:
        return [(math.sqrt(z * z + 4 * (1 - z) * q * q / booksum) - z) / (2 * (1 - z)) for q in implied]

    z = _solve(lambda z: sum(fair(z)) - 1, 0.0, 0.5)
    probabilities = fair(z)
    total = sum(probabilities)
    return [p / total for p in probabilities]


def power_probabilities(implied: Sequence[float]) -> List[float]:
    """The power method: p_i = q_i^k with k chosen so the p_i add up to 1."""
    k = _solve(lambda k: sum(q ** k for q in implied) - 1, 1.0, 10.0)
    probabilities = [q ** k for q in implied]
    total = sum(probabilities)
    return [p / total for p in probabilities]


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
