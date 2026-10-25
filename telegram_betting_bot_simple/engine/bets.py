"""AI-built bets at three risk levels: Safe, Risky and Crazy.

Every candidate leg is one outcome of one game (moneyline, spread or total)
with the AI's probability, the bookmakers' margin-free probability and each
bookmaker's price. A bet takes at most one leg per game (legs from different
games are close to independent, so the chance they all hit is the product of
their chances) and is priced at a single bookmaker that offers every leg, as
it would have to be placed. Each tier has its own rules for leg count, leg
probability and the combined chance/odds band, and its own ranking:

- Safe: 1-2 legs the AI rates 68%+, together 50%+: likely, small payout.
- Risky: 2-3 legs the AI rates 52%+, together 18-45%, paying about 2.2x-6x.
- Crazy: 3-6 legs, together paying 9x-150x with at least a 1% chance.

The expected value shown is honest: most multis lose money on average
because the bookmaker's margin compounds with every leg.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .odds_math import decimal_to_american, format_american


@dataclass
class Candidate:
    event_id: str
    game: str  # "Away @ Home"
    market: str
    side: str
    point: Optional[float]
    label: str
    probability: float  # the AI's
    fair: Optional[float]  # the bookmakers' margin-free
    prices: Dict[str, float]  # bookmaker -> decimal price at this line

    @property
    def best(self) -> float:
        return max(self.prices.values())

    @property
    def ev(self) -> float:
        return self.probability * self.best - 1


@dataclass(frozen=True)
class Tier:
    key: str
    title: str
    rule: str
    legs: Tuple[int, ...]
    min_leg_probability: float
    probability: Tuple[float, float]  # combined AI chance band
    odds: Tuple[float, float]  # combined decimal odds band
    options_per_game: int  # best outcomes per game considered (keeps the search small)


TIERS: Tuple[Tier, ...] = (
    Tier("safe", "Safe", "1-2 legs the AI rates 68%+, together 50%+.", (1, 2), 0.68, (0.50, 1.0), (1.15, 2.2), 2),
    Tier("risky", "Risky", "2-3 legs the AI rates 52%+, together 18-45%.", (2, 3), 0.52, (0.18, 0.45), (2.2, 6.0), 3),
    Tier("crazy", "Crazy", "3-6 legs paying 9x-150x, at least a 1% chance.", (3, 4, 5, 6), 0.30, (0.01, 0.12), (9.0, 150.0), 2),
)


@dataclass
class Bet:
    legs: List[Candidate]
    book: Optional[str]
    decimal: float
    probability: float
    fair_probability: Optional[float]
    leg_prices: List[float] = field(default_factory=list)

    @property
    def ev(self) -> float:
        return self.probability * self.decimal - 1


def price_at_one_book(legs: Sequence[Candidate]) -> Optional[Tuple[str, float, List[float]]]:
    """The best bookmaker offering every leg: (book, combined decimal odds, each leg's price)."""
    common = set(legs[0].prices)
    for leg in legs[1:]:
        common &= set(leg.prices)
    best = None
    for book in common:
        prices = [leg.prices[book] for leg in legs]
        combined = math.prod(prices)
        if best is None or combined > best[1]:
            best = (book, combined, prices)
    return best


def _leg_score(tier: Tier, leg: Candidate) -> float:
    """How good a leg is for this tier (used to keep each game's best few outcomes)."""
    if tier.key == "safe":
        return leg.probability
    if tier.key == "risky":
        return leg.ev + 0.25 * leg.probability
    return leg.ev + 0.05 * math.log(leg.best)


def _bet_score(tier: Tier, bet: Bet) -> float:
    if tier.key == "safe":
        return bet.probability + 0.2 * bet.ev
    if tier.key == "risky":
        return bet.ev + 0.3 * bet.probability
    return bet.ev + 0.02 * math.log(bet.decimal)


def build_tier(tier: Tier, candidates: Sequence[Candidate], count: int = 3) -> List[Bet]:
    """Up to ``count`` bets for one tier, best first, sharing as few legs as possible."""
    by_game: Dict[str, List[Candidate]] = {}
    for leg in candidates:
        if leg.probability >= tier.min_leg_probability and leg.prices:
            by_game.setdefault(leg.event_id, []).append(leg)
    for event_id, legs in by_game.items():
        legs.sort(key=lambda leg: _leg_score(tier, leg), reverse=True)
        by_game[event_id] = legs[: tier.options_per_game]
    games = list(by_game.values())

    found: List[Bet] = []
    for size in tier.legs:
        if size > len(games):
            break
        for group in itertools.combinations(games, size):
            for legs in itertools.product(*group):
                probability = math.prod(leg.probability for leg in legs)
                if not tier.probability[0] <= probability <= tier.probability[1]:
                    continue
                priced = price_at_one_book(legs)
                if priced is None:
                    continue
                book, decimal, prices = priced
                if not tier.odds[0] <= decimal <= tier.odds[1]:
                    continue
                fairs = [leg.fair for leg in legs]
                fair = math.prod(fairs) if all(f is not None for f in fairs) else None
                found.append(Bet(list(legs), book, decimal, probability, fair, prices))

    found.sort(key=lambda bet: _bet_score(tier, bet), reverse=True)
    chosen: List[Bet] = []
    for bet in found:
        keys = {(leg.event_id, leg.market, leg.side) for leg in bet.legs}
        # Skip bets that mostly repeat one already chosen
        if any(len(keys & {(l.event_id, l.market, l.side) for l in other.legs}) * 2 > len(keys) for other in chosen):
            continue
        chosen.append(bet)
        if len(chosen) == count:
            break
    return chosen


def describe(tier: Tier, bet: Bet) -> str:
    """One honest sentence on why this bet fits its tier."""
    chances = ", ".join(f"{leg.probability:.0%}" for leg in bet.legs)
    together = f"together {bet.probability:.0%}" if len(bet.legs) > 1 else ""
    edge = ""
    if bet.fair_probability:
        diff = bet.probability - bet.fair_probability
        edge = f"; the AI rates it {abs(diff):.0%} {'above' if diff >= 0 else 'below'} the bookmakers"
    pays = f"pays {bet.decimal:.2f}x"
    if len(bet.legs) == 1:
        return f"The AI rates this {chances}; {pays}{edge}."
    return f"Legs the AI rates {chances}, {together}; {pays}{edge}."


def bet_view(tier: Tier, bet: Bet) -> Dict[str, Any]:
    return {
        "legs": [
            {"event_id": leg.event_id, "game": leg.game, "market": leg.market, "side": leg.side, "point": leg.point,
             "label": leg.label, "probability": leg.probability, "fair": leg.fair,
             "odds": {"decimal": price, "american": format_american(decimal_to_american(price))}}
            for leg, price in zip(bet.legs, bet.leg_prices)
        ],
        "book": bet.book,
        "odds": {"decimal": round(bet.decimal, 4), "american": format_american(decimal_to_american(bet.decimal))},
        "probability": bet.probability,
        "fair_probability": bet.fair_probability,
        "ev": bet.ev,
        "why": describe(tier, bet),
    }


def build_all(candidates: Sequence[Candidate], count: int = 3) -> Dict[str, Dict[str, Any]]:
    out = {}
    for tier in TIERS:
        bets = build_tier(tier, candidates, count)
        out[tier.key] = {
            "title": tier.title,
            "rule": tier.rule,
            "bets": [bet_view(tier, bet) for bet in bets],
            "empty": "No mix of today's games fits this level." if not bets else None,
        }
    return out
