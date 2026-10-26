"""One pricing function for every ticket: singles, multis, round robins and system bets.

Every ticket reduces to one settlement factor per leg (Racing Post, US house rules):
a won leg contributes its decimal odds, a lost leg 0, a pushed, void or cancelled
leg 1 (it drops out), and a dead heat its odds times places paid over the number
tied. A bet returns its stake times the product of its legs' factors, and a round
robin or system bet is several bets: one per combination of legs, each at the unit
stake. Full-cover systems hold every combination of two or more legs (2^n - n - 1
bets: Trixie 4, Yankee 11, Heinz 57, Goliath 247); the "Lucky" and Patent versions
add the singles (2^n - 1).

Because every bet in these tickets is "all k-leg combinations" for one or more
sizes k, a ticket's return is a sum of elementary symmetric polynomials of the leg
factors. That gives three things cheaply and exactly:

- the max payout (every leg wins) and the expected return, since legs from
  different games are independent and the expectation of a product is the product
  of expectations (a leg's expected factor is win * odds + push);
- the bookmaker's compounded cut, 1 - Q * D for a multi, or one minus the expected
  return per unit staked at the market's margin-free chances (pushes left out) for
  anything else;
- an exact enumeration of every win / push / lose outcome (3^n, n <= 8; a multi
  stops at its first lost leg, so 12 legs is cheap, and singles that can't push
  are only 2^n), which gives the chance of any money back, of a profit, and the
  single most likely result.

The two views of a leg's chance mean different things when it can push. The AI's
is the chance of a win outright (a push is neither a win nor a loss). The market's
comes from de-vigging a two-way price, so both sides add up to 1: it is the chance
of a win when the leg doesn't push, and the push takes its share from both sides.
Reading it as an outright chance would count the push twice and make backing both
sides of a line look like a profit.

Money is Decimal throughout and is rounded to cents once, at the end. Chances are
floats. Same-game legs in a multi, round robin or system bet are refused: their
results are linked and multiplying their chances would misprice the ticket.
"""

from __future__ import annotations

import datetime
import itertools
import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .bets import whelan_max_legs
from .odds_math import decimal_to_american, format_american

TICKET_TYPES = ("single", "multi", "round_robin", "system")
MAX_LEGS = 12
MAX_COMBO_LEGS = 8  # round robins and system bets: also the limit for listing every outcome
CENTS = Decimal("0.01")
ZERO, ONE = Decimal(0), Decimal(1)
PRECISION = 60  # digits for the exact sums: 8 legs of 5-decimal odds need about 45
AI_GAP = 0.05  # the AI this far from the market on a leg: "check the injury news"

# Outcome buckets, from worst to best, for the "spread of outcomes" and the most likely one
LOSE_ALL, SOME_BACK, STAKE_BACK, PROFIT = "lose all", "some back", "stake back", "profit"
RESULTS = (LOSE_ALL, SOME_BACK, STAKE_BACK, PROFIT)

Split = Optional[Tuple[float, float, float]]  # one leg's (win, push, lose) chances, or None if unknown

EXTRA_LEG_WARNING = "Each extra leg adds the bookmaker's cut again."
SPREAD_WARNING = "The max payout needs every pick to win; most outcomes pay far less, so check the spread of outcomes."


@dataclass(frozen=True)
class System:
    title: str
    legs: int
    sizes: Tuple[int, ...]  # the bet sizes it covers: (2, 3) is every double and the treble

    @property
    def bets(self) -> int:
        return sum(math.comb(self.legs, size) for size in self.sizes)


def _full(legs: int) -> Tuple[int, ...]:
    return tuple(range(2, legs + 1))


def _lucky(legs: int) -> Tuple[int, ...]:
    return tuple(range(1, legs + 1))


SYSTEMS: Dict[str, System] = {
    "trixie": System("Trixie", 3, _full(3)),
    "patent": System("Patent", 3, _lucky(3)),
    "yankee": System("Yankee", 4, _full(4)),
    "lucky15": System("Lucky 15", 4, _lucky(4)),
    "canadian": System("Canadian", 5, _full(5)),
    "super_yankee": System("Super Yankee", 5, _full(5)),
    "lucky31": System("Lucky 31", 5, _lucky(5)),
    "heinz": System("Heinz", 6, _full(6)),
    "lucky63": System("Lucky 63", 6, _lucky(6)),
    "super_heinz": System("Super Heinz", 7, _full(7)),
    "goliath": System("Goliath", 8, _full(8)),
}
# "Lucky 15", "lucky_15" and "LUCKY15" all name the same system
_SYSTEM_NAMES = {re.sub(r"[^a-z0-9]", "", key): key for key in SYSTEMS}


@dataclass
class TicketLeg:
    key: str  # stable id, e.g. "basketball_nba:<event>:spreads:home:-3.5" or a prop key
    league: str
    event_id: str
    market: str  # h2h | spreads | totals | player_<stat>
    side: str  # home | away | draw | over | under
    point: Optional[float]
    price: Decimal  # decimal odds at the quote
    label: str  # "Thunder -3.5", "Over 224.5", "LeBron James over 27.5 points"
    game: str  # "Lakers @ Thunder"
    start: str  # ISO 8601 UTC
    ai_prob: Optional[float] = None  # the brain's chance this leg wins outright (a push is not a win)
    market_prob: Optional[float] = None  # the market's margin-free chance this leg wins, given no push
    push_prob: float = 0.0
    book: Optional[str] = None
    group: Optional[str] = None  # correlation group: legs sharing one are from the same game

    def __post_init__(self) -> None:
        # Floats arrive from JSON; _decimal's str() keeps 1.91 as exactly 1.91 rather than its binary neighbour
        self.price = _decimal(self.price)
        if not self.price > ONE:
            raise ValueError(f"{self.label}: decimal odds must be above 1, got {self.price}")
        # Callers pass None for a market that can't push (a moneyline): that is no chance of one
        self.push_prob = float(self.push_prob or 0.0)
        for name in ("ai_prob", "market_prob", "push_prob"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{self.label}: {name} must be between 0 and 1, got {value}")
        # The AI's win and the push are separate outcomes, so together they can't pass 1. The market's
        # chance is given no push, so any value up to 1 fits with any push and needs no such check.
        if self.ai_prob is not None and self.ai_prob + self.push_prob > 1 + 1e-9:
            raise ValueError(f"{self.label}: ai_prob {self.ai_prob} and push_prob {self.push_prob} "
                             "add up to more than 1")
        self.group = self.group or self.event_id


# --- small helpers ------------------------------------------------------------------------------

def chance_label(p: Optional[float]) -> Optional[str]:
    """Fixed words for a chance, so nobody (Claude included) has to pick them."""
    if p is None:
        return None
    if p >= 0.60:
        return "favourite"
    if p >= 0.40:
        return "coin flip"
    if p >= 0.20:
        return "underdog"
    if p >= 0.05:
        return "long shot"
    return "lottery ticket"


def ai_given_no_push(ai_prob: float, push_prob: float) -> float:
    """The AI's outright chance as a chance given no push, the market's footing, so the two can be compared."""
    no_push = 1.0 - (push_prob or 0.0)
    return ai_prob / no_push if no_push > 0 else ai_prob


def one_in(p: Optional[float]) -> Optional[int]:
    """A chance as "1 in X" (X at least 1), or None when there is no chance at all."""
    if p is None or p <= 0:
        return None
    return max(1, int(round(1 / p)))


def _decimal(value: Union[Decimal, float, int, str]) -> Decimal:
    """An exact number for money or odds. Anything that isn't a finite number (typed "abc",
    "NaN", "Infinity") raises ValueError, so an API layer can answer it as bad input."""
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"Not a number: {value!r}") from None
    if not number.is_finite():
        raise ValueError(f"Not a finite number: {value!r}")
    return number


def _money(value: Optional[Decimal]) -> Optional[str]:
    """Cents, half up, as a string; never "-0.00"."""
    if value is None:
        return None
    rounded = value.quantize(CENTS, rounding=ROUND_HALF_UP)
    return str(rounded if rounded != 0 else ZERO.quantize(CENTS))


def _chance(p: Optional[float]) -> Optional[float]:
    return None if p is None else round(p, 4)


def _odds(price: Decimal) -> Dict[str, Any]:
    return {
        "decimal": float(price.quantize(CENTS, rounding=ROUND_HALF_UP)),
        "american": format_american(decimal_to_american(float(price))),
    }


def _started(leg: TicketLeg, now: datetime.datetime) -> bool:
    start = datetime.datetime.fromisoformat(leg.start.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=datetime.timezone.utc)
    return start <= now


def _split(win: Optional[float], push: float) -> Split:
    """(win, push, lose) for one leg at an outright win chance (the AI's), or None without one.

    A push can't take the total past 1 (only float rounding could, as legs are checked)."""
    if win is None:
        return None
    push = min(push, 1.0 - win)
    return win, push, max(0.0, 1.0 - win - push)


def _market_split(q: Optional[float], push: float) -> Split:
    """(win, push, lose) for one leg at the market's chance q, or None without one.

    q comes from a de-vigged two-way price, so it is the chance of a win given no
    push: the push takes its share from the win and the loss alike.
    """
    if q is None:
        return None
    return q * (1.0 - push), push, (1.0 - q) * (1.0 - push)


def _symmetric(values: Sequence[Decimal], top: int) -> List[Decimal]:
    """e_0..e_top of the values: e_k is the sum, over every k-leg combination, of the product."""
    e = [ONE] + [ZERO] * top
    for value in values:
        e = _add_factor(e, value)
    return e


def _add_factor(e: List[Decimal], value: Decimal) -> List[Decimal]:
    """The e-vector after one more leg with this factor (a lost leg's 0 leaves it unchanged)."""
    return [e[0]] + [e[j] + value * e[j - 1] for j in range(1, len(e))]


def _sizes_total(e: Sequence[Decimal], sizes: Sequence[int]) -> Decimal:
    return sum((e[size] for size in sizes), ZERO)


# --- the shape of each ticket --------------------------------------------------------------------

def _system_key(system: Optional[str]) -> str:
    key = _SYSTEM_NAMES.get(re.sub(r"[^a-z0-9]", "", (system or "").lower()))
    if key is None:
        raise ValueError(f"Unknown system {system!r}; choose one of {', '.join(SYSTEMS)}")
    return key


def _check_type(ticket_type: str) -> str:
    if ticket_type not in TICKET_TYPES:
        raise ValueError(f"Unknown ticket type {ticket_type!r}; choose one of {', '.join(TICKET_TYPES)}")
    return ticket_type


def _sizes(ticket_type: str, n: int, k: Optional[int], system: Optional[str]) -> Tuple[int, ...]:
    """The bet sizes a ticket covers: singles (1,), a multi (n,), a round robin (k,), a system its own."""
    if ticket_type == "single":
        return (1,)
    if ticket_type == "multi":
        return (n,)
    if ticket_type == "round_robin":
        return (k or 2,)
    return SYSTEMS[_system_key(system)].sizes


def combinations(ticket_type: str, n: int, k: Optional[int] = None,
                 system: Optional[str] = None) -> List[Tuple[int, ...]]:
    """Every bet in a ticket of n legs, as tuples of leg indexes (smaller bets first).

    A round robin by k (default 2) holds every k-leg combination, for 2 <= k <= n - 1.
    A system bet needs exactly its own number of legs. Raises ValueError otherwise.
    """
    _check_type(ticket_type)
    if n < 1:
        raise ValueError("A ticket needs at least one leg")
    if ticket_type == "round_robin" and not 2 <= (k or 2) <= n - 1:
        raise ValueError(f"A round robin of {n} legs can be by 2s up to {n - 1}s, not {k}s")
    if ticket_type == "system" and SYSTEMS[_system_key(system)].legs != n:
        wanted = SYSTEMS[_system_key(system)]
        raise ValueError(f"A {wanted.title} needs exactly {wanted.legs} legs, not {n}")
    return [combo for size in _sizes(ticket_type, n, k, system)
            for combo in itertools.combinations(range(n), size)]


def _shape_reasons(ticket_type: str, n: int, k: Optional[int], system: Optional[str]) -> List[str]:
    """Why a ticket can't be priced at all (wrong number of legs for its type)."""
    if n == 0:
        return ["Add at least one pick"]
    if n > MAX_LEGS:
        return [f"A ticket can have at most {MAX_LEGS} picks"]
    if ticket_type in ("round_robin", "system") and n > MAX_COMBO_LEGS:
        return [f"Round robins and system bets can have at most {MAX_COMBO_LEGS} picks"]
    if ticket_type == "round_robin":
        if n < 3:
            return ["A round robin needs at least 3 picks"]
        if not 2 <= (k or 2) <= n - 1:
            return [f"A round robin of {n} picks can be by 2s up to {n - 1}s"]
    if ticket_type == "system" and SYSTEMS[system].legs != n:
        return [f"A {SYSTEMS[system].title} needs exactly {SYSTEMS[system].legs} picks"]
    return []


# --- settling ----------------------------------------------------------------------------------

_RESULT_NAMES = {
    "won": "won", "win": "won", "lost": "lost", "lose": "lost", "loss": "lost",
    "push": "push", "void": "void", "cancelled": "cancelled", "canceled": "cancelled",
    "dead_heat": "dead_heat", "deadheat": "dead_heat",
}


def _result_name(result: str) -> str:
    name = _RESULT_NAMES.get(re.sub(r"[\s-]+", "_", str(result).strip().lower()))
    if name is None:
        raise ValueError(f"Unknown leg result {result!r}")
    return name


def settle_factor(result: str, price: Decimal, places_paid: int = 1, tied: int = 1) -> Decimal:
    """What one leg multiplies its bets' stake by once it is settled.

    won -> its odds; lost -> 0; push, void or cancelled -> 1 (the leg drops out);
    dead heat -> odds * places paid / number tied (the stake is split between
    the tied picks, and only the paid share wins).
    """
    name, price = _result_name(result), _decimal(price)
    if name == "won":
        return price
    if name == "lost":
        return ZERO
    if name in ("push", "void", "cancelled"):
        return ONE
    if tied < 1 or places_paid < 1:
        raise ValueError("A dead heat needs at least one place paid and one pick tied")
    return price * Decimal(min(places_paid, tied)) / Decimal(tied)


# --- the spread of outcomes ----------------------------------------------------------------------

@dataclass
class _Bucket:
    ai: Optional[float]
    market: Optional[float]
    low: Decimal  # smallest and largest return in units, to show the spread
    high: Decimal


def _result_for(units: Decimal, bets: int) -> str:
    if units == 0:
        return LOSE_ALL
    if units < bets:
        return SOME_BACK
    return STAKE_BACK if units == bets else PROFIT


def _outcomes(prices: Sequence[Decimal], sizes: Sequence[int], bets: int,
              ai: Sequence[Split], market: Sequence[Split], pushes: Sequence[bool]) -> Dict[str, _Bucket]:
    """Every win / push / lose outcome of the legs, grouped by what the ticket returns.

    A branch stops early once every bet holds a lost leg (e.g. a multi's first lost
    leg): the rest of the legs can't change the return, and their chances add up to 1.
    """
    n, smallest, top = len(prices), min(sizes), max(sizes)
    use_ai, use_market = all(s is not None for s in ai), all(s is not None for s in market)
    buckets: Dict[str, _Bucket] = {}

    def record(units: Decimal, pa: float, pm: float) -> None:
        result = _result_for(units, bets)
        bucket = buckets.get(result)
        if bucket is None:
            buckets[result] = _Bucket(pa if use_ai else None, pm if use_market else None, units, units)
            return
        if use_ai:
            bucket.ai += pa
        if use_market:
            bucket.market += pm
        bucket.low, bucket.high = min(bucket.low, units), max(bucket.high, units)

    def visit(i: int, e: List[Decimal], alive: int, pa: float, pm: float) -> None:
        if alive + (n - i) < smallest:
            record(ZERO, pa, pm)  # every bet already has a lost leg
            return
        if i == n:
            record(_sizes_total(e, sizes), pa, pm)
            return
        a = ai[i] if use_ai else (1.0, 1.0, 1.0)
        m = market[i] if use_market else (1.0, 1.0, 1.0)
        visit(i + 1, _add_factor(e, prices[i]), alive + 1, pa * a[0], pm * m[0])
        if pushes[i]:
            visit(i + 1, _add_factor(e, ONE), alive + 1, pa * a[1], pm * m[1])
        visit(i + 1, e, alive, pa * a[2], pm * m[2])

    visit(0, [ONE] + [ZERO] * top, 0, 1.0, 1.0)
    return buckets


# --- pricing -----------------------------------------------------------------------------------

def _leg_view(index: int, leg: TicketLeg, started: bool) -> Dict[str, Any]:
    p = leg.ai_prob
    no_push = 1.0 - leg.push_prob
    return {
        "index": index,
        "key": leg.key,
        "label": leg.label,
        "game": leg.game,
        "league": leg.league,
        "event_id": leg.event_id,
        "market": leg.market,
        "side": leg.side,
        "point": leg.point,
        "start": leg.start,
        "price": str(leg.price),  # exact, for saving and grading
        "odds": _odds(leg.price),
        "ai_prob": _chance(p),
        "market_prob": _chance(leg.market_prob),
        "push_prob": _chance(leg.push_prob),
        # Fair odds and edge count a push as money back: with no push they are 1/p and p * odds - 1
        "fair_odds": round(no_push / p, 2) if p else None,
        "chance_label": chance_label(p if p is not None else leg.market_prob),
        "edge": _chance(p * float(leg.price) + leg.push_prob - 1) if p is not None else None,
        "book_cut": _chance(1 - leg.market_prob * float(leg.price)) if leg.market_prob is not None else None,
        "started": started,
        "book": leg.book,
        "group": leg.group,
    }


def _weakest_leg(legs: Sequence[TicketLeg]) -> Optional[int]:
    chances = [(leg.ai_prob if leg.ai_prob is not None else leg.market_prob, i) for i, leg in enumerate(legs)]
    known = [(p, i) for p, i in chances if p is not None]
    return min(known)[1] if known else None


def _same_game(legs: Sequence[TicketLeg]) -> List[List[int]]:
    by_group: Dict[str, List[int]] = {}
    for i, leg in enumerate(legs):
        by_group.setdefault(leg.group or leg.event_id, []).append(i)
    return [indexes for indexes in by_group.values() if len(indexes) > 1]


def _names(legs: Sequence[TicketLeg], indexes: Sequence[int]) -> str:
    return " and ".join(legs[i].label for i in indexes)


def _validity(legs: Sequence[TicketLeg], ticket_type: str, started: Sequence[bool],
              stake: Decimal) -> List[str]:
    reasons = []
    by_key: Dict[str, List[TicketLeg]] = {}
    for leg in legs:
        by_key.setdefault(leg.key, []).append(leg)
    reasons += [f"{same[0].label} is on the ticket more than once" for same in by_key.values() if len(same) > 1]
    if ticket_type != "single":
        reasons += [f"{_names(legs, group)} are from the same game, so their results are linked; pick one "
                    "(same-game tickets aren't supported)" for group in _same_game(legs)]
    reasons += [f"{leg.label} has already started" for leg, gone in zip(legs, started) if gone]
    if stake <= 0:
        reasons.append("Enter a stake above zero")
    return reasons


def _warnings(legs: Sequence[TicketLeg], ticket_type: str, sizes: Sequence[int],
              enumerated: bool) -> List[str]:
    warnings = []
    if max(sizes) >= 2:
        warnings.append(EXTRA_LEG_WARNING)
    if ticket_type == "multi":
        chances = [leg.ai_prob if leg.ai_prob is not None else leg.market_prob for leg in legs]
        if all(p is not None and p > 0 for p in chances):
            worth = whelan_max_legs(math.prod(chances) ** (1 / len(legs)))
            if len(legs) > worth:
                warnings.append(f"More legs than are worth combining at these chances ({worth} at most).")
        warnings += [f"A push on {leg.label} drops it from the ticket and lowers the payout."
                     for leg in legs if leg.push_prob > 0]
    if ticket_type in ("round_robin", "system") or (ticket_type == "single" and len(legs) > 1):
        warnings.append(SPREAD_WARNING)
    if ticket_type == "single":
        warnings += [f"{_names(legs, group)} are from the same game: the chances shown treat them as "
                     "unrelated, and they are not." for group in _same_game(legs)]
    if not enumerated:
        warnings.append("Too many picks to list every outcome: the chance shown is every pick winning, "
                        "not the chance of a profit.")
    # A view of the ticket needs a chance on every leg, so say which view (if any) is left
    market_view = all(leg.market_prob is not None for leg in legs)
    for leg in legs:
        if leg.ai_prob is None and leg.market_prob is None:
            warnings.append(f"No chance available for {leg.label}: the ticket's chances can't be shown.")
        elif leg.ai_prob is None:
            left = "only the market's view is shown." if market_view else "the AI's view of the ticket isn't shown."
            warnings.append(f"No AI chance for {leg.label}: {left}")
        elif (leg.market_prob is not None
              and abs(ai_given_no_push(leg.ai_prob, leg.push_prob) - leg.market_prob) > AI_GAP):
            warnings.append(f"The AI is far from the market on {leg.label}: check the injury news.")
    return warnings


def _empty_chance() -> Dict[str, Any]:
    pair = {"ai": None, "market": None}
    return {"headline": None, "ai": None, "market": None, "ai_one_in": None, "market_one_in": None,
            "label": None, "every_leg": dict(pair), "any_return": dict(pair), "profit": dict(pair)}


def _expected(prices: Sequence[Decimal], splits: Sequence[Split],
              sizes: Sequence[int], unit: Decimal, total: Decimal) -> Optional[Decimal]:
    """Expected profit in money: unit * sum e_k(win * odds + push) - total stake."""
    if any(s is None for s in splits):
        return None
    means = [Decimal(repr(s[0])) * price + Decimal(repr(s[1])) for s, price in zip(splits, prices)]
    return unit * _sizes_total(_symmetric(means, max(sizes)), sizes) - total


def _book_cut(legs: Sequence[TicketLeg], prices: Sequence[Decimal], sizes: Sequence[int], bets: int) -> Optional[float]:
    """What the bookmaker keeps on average per unit staked, at the market's chances (pushes left out)."""
    if any(leg.market_prob is None for leg in legs):
        return None
    fair = [Decimal(repr(leg.market_prob)) * price for leg, price in zip(legs, prices)]
    return _chance(float(1 - _sizes_total(_symmetric(fair, max(sizes)), sizes) / bets))


def _known(buckets: Optional[Dict[str, _Bucket]], basis: str) -> bool:
    """Whether the outcomes were listed with chances under this basis ("ai" or "market")."""
    return bool(buckets) and getattr(next(iter(buckets.values())), basis) is not None


def _chance_view(ticket_type: str, n: int, ai: Sequence[Split], market: Sequence[Split],
                 buckets: Optional[Dict[str, _Bucket]]) -> Dict[str, Any]:
    def every(splits: Sequence[Split]) -> Optional[float]:
        return math.prod(s[0] for s in splits) if all(s is not None for s in splits) else None

    def share(basis: str, results: Sequence[str]) -> Optional[float]:
        if not _known(buckets, basis):
            return None
        return sum(getattr(buckets[r], basis) for r in results if r in buckets)

    every_leg = {"ai": every(ai), "market": every(market)}
    any_return = {b: share(b, (SOME_BACK, STAKE_BACK, PROFIT)) for b in ("ai", "market")}
    profit = {b: share(b, (PROFIT,)) for b in ("ai", "market")}
    # A multi (or one single) wins only when every leg does; anything else is a spread, so: a profit.
    # Singles too many to list (9 to 12 that can push) fall back to every leg winning, so a
    # "1 in X" is still shown whenever every leg has a chance.
    spread = ticket_type in ("round_robin", "system") or (ticket_type == "single" and n > 1)
    headline = "profit" if spread and buckets is not None else "every leg wins"
    top = every_leg if headline == "every leg wins" else profit
    return {
        "headline": headline,
        "ai": _chance(top["ai"]),
        "market": _chance(top["market"]),
        "ai_one_in": one_in(top["ai"]),
        "market_one_in": one_in(top["market"]),
        "label": chance_label(top["ai"] if top["ai"] is not None else top["market"]),
        "every_leg": {b: _chance(v) for b, v in every_leg.items()},
        "any_return": {b: _chance(v) for b, v in any_return.items()},
        "profit": {b: _chance(v) for b, v in profit.items()},
    }


def _outcome_views(buckets: Optional[Dict[str, _Bucket]], unit: Decimal) -> List[Dict[str, Any]]:
    if buckets is None:
        return []
    views = []
    for result in RESULTS:
        bucket = buckets.get(result)
        chances = [p for p in (bucket.ai, bucket.market) if p is not None] if bucket else []
        if bucket is None or (chances and not any(chances)):
            continue  # a result no basis gives any chance to (e.g. a leg at 100%)
        views.append({"result": result, "ai": _chance(bucket.ai), "market": _chance(bucket.market),
                      "min_return": _money(unit * bucket.low), "max_return": _money(unit * bucket.high)})
    return views


def _most_likely(buckets: Optional[Dict[str, _Bucket]], unit: Decimal, total: Decimal) -> Optional[Dict[str, Any]]:
    """The single most likely result (by the AI's chances, else the market's), with its profit when fixed."""
    basis = "ai" if _known(buckets, "ai") else "market"
    if not _known(buckets, basis):
        return None
    # On a tie the worse result wins: RESULTS runs from worst to best and max keeps the first
    result = max(RESULTS, key=lambda r: getattr(buckets[r], basis) if r in buckets else -1.0)
    bucket = buckets[result]
    fixed = bucket.low == bucket.high
    return {"result": result, "probability": _chance(getattr(bucket, basis)), "basis": basis,
            "profit": _money(unit * bucket.low - total) if fixed else None}


def _unpriced(base: Dict[str, Any], stake: Decimal, reasons: List[str]) -> Dict[str, Any]:
    return {**base, "bets": 0, "unit_stake": _money(stake), "total_stake": _money(ZERO), "odds": None,
            "max_payout": None, "max_profit": None, "chance": _empty_chance(), "book_cut": None,
            "expected": {"ai": None, "market": None, "ai_roi": None, "market_roi": None},
            "most_likely": None, "outcomes": [], "warnings": [], "valid": False, "reasons": reasons}


def price_ticket(legs: Sequence[TicketLeg], ticket_type: str, stake: Union[Decimal, float, int, str],
                 k: Optional[int] = None, system: Optional[str] = None,
                 now: Optional[datetime.datetime] = None) -> Dict[str, Any]:
    """Price any ticket: odds, payouts, chances, the bookmaker's cut and the expected result.

    ``stake`` is per bet: each single, the one multi, each bet of a round robin
    or system (the unit stake); the total is stake x bets. A one-leg multi is
    priced as a single and a round robin defaults to 2s. Money comes back as
    strings in cents, chances as floats to 4 places. A ticket of the wrong shape
    (e.g. a Yankee with 3 picks) comes back unpriced with ``valid`` False and
    reasons; an unknown type or system name raises ValueError.
    """
    legs = list(legs)
    ticket_type = _check_type(ticket_type)
    if ticket_type == "multi" and len(legs) == 1:
        ticket_type = "single"
    k = (k or 2) if ticket_type == "round_robin" else None
    system = _system_key(system) if ticket_type == "system" else None
    stake = _decimal(stake)
    if stake < 0:
        raise ValueError(f"The stake can't be negative, got {stake}")
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)

    started = [_started(leg, now) for leg in legs]
    base = {"type": ticket_type, "system": system, "k": k,
            "legs": [_leg_view(i, leg, gone) for i, (leg, gone) in enumerate(zip(legs, started))],
            "weakest_leg": _weakest_leg(legs)}
    shape = _shape_reasons(ticket_type, len(legs), k, system)
    if shape:
        return _unpriced(base, stake, shape + _validity(legs, ticket_type, started, stake))

    with localcontext() as context:
        context.prec = PRECISION
        return {**base, **_price(legs, ticket_type, stake, k, system, started)}


def _price(legs: Sequence[TicketLeg], ticket_type: str, stake: Decimal, k: Optional[int],
           system: Optional[str], started: Sequence[bool]) -> Dict[str, Any]:
    n = len(legs)
    sizes = _sizes(ticket_type, n, k, system)
    bets = len(combinations(ticket_type, n, k, system))
    prices = [leg.price for leg in legs]
    total = stake * bets
    max_payout = stake * _sizes_total(_symmetric(prices, max(sizes)), sizes)

    ai = [_split(leg.ai_prob, leg.push_prob) for leg in legs]
    market = [_market_split(leg.market_prob, leg.push_prob) for leg in legs]
    pushes = [leg.push_prob > 0 for leg in legs]
    # Listing is cheap up to 8 legs, for a multi (it stops at the first lost leg) and for singles
    # that can't push (2^12 = 4,096 outcomes); 9 to 12 singles that can push would be 3^n
    enumerated = (n <= MAX_COMBO_LEGS or ticket_type == "multi"
                  or (ticket_type == "single" and not any(pushes)))
    buckets = _outcomes(prices, sizes, bets, ai, market, pushes) if enumerated else None
    expected = {"ai": _expected(prices, ai, sizes, stake, total),
                "market": _expected(prices, market, sizes, stake, total)}
    combined = math.prod(prices, start=ONE) if sizes == (n,) else None  # one bet holding every leg
    reasons = _validity(legs, ticket_type, started, stake)

    return {
        "bets": bets,
        "unit_stake": _money(stake),
        "total_stake": _money(total),
        "odds": _odds(combined) if combined is not None else None,
        "max_payout": _money(max_payout),
        "max_profit": _money(max_payout - total),
        "chance": _chance_view(ticket_type, n, ai, market, buckets),
        "book_cut": _book_cut(legs, prices, sizes, bets),
        "expected": {
            **{basis: _money(value) for basis, value in expected.items()},
            **{f"{basis}_roi": (_chance(float(value / total)) if value is not None and total > 0 else None)
               for basis, value in expected.items()},
        },
        "most_likely": _most_likely(buckets, stake, total),
        "outcomes": _outcome_views(buckets, stake),
        "warnings": _warnings(legs, ticket_type, sizes, enumerated),
        "valid": not reasons,
        "reasons": reasons,
    }


# --- grading -----------------------------------------------------------------------------------

def _leg_result(item: Any) -> Optional[Dict[str, Any]]:
    """A leg's result as {result, places_paid, tied}, or None while it is still to be decided."""
    if isinstance(item, dict):
        result, places, tied = item.get("result"), item.get("places_paid", 1), item.get("tied", 1)
    else:
        result, places, tied = item, 1, 1
    if result is None or str(result).strip().lower() == "pending":
        return None
    return {"result": _result_name(result), "places_paid": int(places or 1), "tied": int(tied or 1)}


def _leg_price(leg: Any) -> Decimal:
    if isinstance(leg, TicketLeg):
        return leg.price
    return _decimal(leg["price"] if leg.get("price") is not None else leg["odds"]["decimal"])


def grade(legs_results: Sequence[Any], ticket: Dict[str, Any]) -> Dict[str, Any]:
    """Settle a priced ticket from its legs' results, one per leg in order.

    Each result is "won", "lost", "push", "void", "cancelled" or "dead_heat" (a
    string, or a dict with ``result`` and, for a dead heat, ``places_paid`` and
    ``tied``); None or "pending" is a leg still to play. Status is "won" (no bet
    holds a lost leg), "lost" (nothing comes back), "void" (every leg voided or
    pushed: the stake comes back), "partial" (some bets paid, some lost) or
    "pending" (still open; a multi with a lost leg is already lost).
    """
    legs = ticket["legs"]
    if len(legs_results) != len(legs):
        raise ValueError(f"Expected {len(legs)} leg results, got {len(legs_results)}")
    if not ticket.get("bets"):
        raise ValueError("This ticket was never priced, so it can't be graded")
    combos = combinations(ticket["type"], len(legs), ticket.get("k"), ticket.get("system"))
    unit = _decimal(ticket["unit_stake"])
    total = unit * len(combos)
    results = [_leg_result(item) for item in legs_results]

    with localcontext() as context:
        context.prec = PRECISION
        factors = [None if r is None else settle_factor(r["result"], _leg_price(leg), r["places_paid"], r["tied"])
                   for r, leg in zip(results, legs)]
        lost = [any(factors[i] == 0 for i in combo) for combo in combos]
        open_bets = [combo for combo, dead in zip(combos, lost) if not dead and any(factors[i] is None for i in combo)]
        settled = [None if r is None else r["result"] for r in results]
        if open_bets:
            return {"status": "pending", "returns": None, "profit": None, "total_stake": _money(total),
                    "paid_bets": None, "bets": len(combos), "results": settled}
        paid = [ZERO if dead else math.prod((factors[i] for i in combo), start=ONE)
                for combo, dead in zip(combos, lost)]
        returns = unit * sum(paid, ZERO)

    if returns == 0:
        status = "lost"
    elif all(r is not None and r["result"] in ("push", "void", "cancelled") for r in results):
        status = "void"
    else:
        status = "partial" if any(lost) else "won"
    return {"status": status, "returns": _money(returns), "profit": _money(returns - total),
            "total_stake": _money(total), "paid_bets": sum(1 for value in paid if value > 0),
            "bets": len(combos), "results": settled}
