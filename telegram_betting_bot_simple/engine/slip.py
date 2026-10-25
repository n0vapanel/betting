"""Bet slip evaluation shared by the bot and the dashboard.

One pure function turns a list of priced picks into what a sportsbook's
bet slip shows: for singles, each pick's return; for a multi (parlay),
the combined odds, payout, the model's chance of every leg winning, and
the reasons the multi can't be placed (fewer than 2 legs, two legs from
the same game, a game that already started). Money and odds use Decimal
and are rounded once, at the end.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional

from .odds_math import decimal_to_american, format_american

MAX_LEGS = 12
CENTS = Decimal("0.01")


@dataclass
class Leg:
    event_id: str
    market: str  # h2h, spreads or totals
    side: str  # home, away, draw, over or under
    price: float  # decimal odds
    label: str  # e.g. "Thunder -13.5"
    game: str  # e.g. "LA Clippers @ Oklahoma City Thunder"
    start: str  # ISO 8601 kickoff
    point: Optional[float] = None
    probability: Optional[float] = None  # model chance of winning


def _money(value: Decimal) -> float:
    return float(value.quantize(CENTS, rounding=ROUND_HALF_UP))


def _odds(decimal: Decimal) -> Dict[str, Any]:
    return {
        "decimal": float(decimal.quantize(CENTS, rounding=ROUND_HALF_UP)),
        "american": format_american(decimal_to_american(float(decimal))) if decimal > 1 else None,
    }


def evaluate_slip(
    legs: List[Leg],
    mode: str,
    stake: float = 0.0,
    stakes: Optional[Dict[int, float]] = None,
    now: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """Evaluate a slip as singles (``stakes`` per leg index) or as one multi (``stake``)."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    started = {
        i for i, leg in enumerate(legs)
        if datetime.datetime.fromisoformat(leg.start.replace("Z", "+00:00")) <= now
    }
    legs_out = []
    for i, leg in enumerate(legs):
        price = Decimal(str(leg.price))
        item: Dict[str, Any] = {
            "index": i,
            "label": leg.label,
            "game": leg.game,
            "odds": _odds(price),
            "probability": leg.probability,
            "started": i in started,
        }
        if leg.probability is not None:
            item["ev"] = leg.probability * leg.price - 1
        if mode == "single":
            single_stake = Decimal(str((stakes or {}).get(i, 0) or 0))
            item["stake"] = _money(single_stake)
            item["returns"] = _money(single_stake * price)
        legs_out.append(item)

    if mode == "single":
        total_stake = sum((Decimal(str(leg["stake"])) for leg in legs_out), Decimal(0))
        total_returns = sum((Decimal(str(leg["returns"])) for leg in legs_out), Decimal(0))
        reasons = [f"{legs[i].label} has already started" for i in sorted(started)]
        return {
            "mode": "single",
            "legs": legs_out,
            "stake": _money(total_stake),
            "returns": _money(total_returns),
            "valid": bool(legs) and not reasons,
            "reasons": reasons,
        }

    reasons = []
    if len(legs) < 2:
        reasons.append("Add at least 2 picks from different games for a multi")
    if len(legs) > MAX_LEGS:
        reasons.append(f"A multi can have at most {MAX_LEGS} picks")
    by_game: Dict[str, List[int]] = {}
    for i, leg in enumerate(legs):
        by_game.setdefault(leg.event_id, []).append(i)
    for indexes in by_game.values():
        if len(indexes) > 1:
            names = " and ".join(legs[i].label for i in indexes)
            reasons.append(f"{names} are from the same game; pick one (same-game multis aren't supported)")
    reasons += [f"{legs[i].label} has already started" for i in sorted(started)]

    combined = Decimal(1)
    for leg in legs:
        combined *= Decimal(str(leg.price))
    probability = None
    if legs and all(leg.probability is not None for leg in legs):
        probability = 1.0
        for leg in legs:
            probability *= leg.probability
    multi_stake = Decimal(str(stake or 0))
    payout = multi_stake * combined
    return {
        "mode": "multi",
        "legs": legs_out,
        "odds": _odds(combined) if legs else None,
        "stake": _money(multi_stake),
        "returns": _money(payout),
        "profit": _money(payout - multi_stake),
        "probability": probability,
        "ev": probability * float(combined) - 1 if probability is not None else None,
        "valid": not reasons,
        "reasons": reasons,
    }
