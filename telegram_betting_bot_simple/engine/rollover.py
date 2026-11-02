"""Rollover risk planner: four ways to play the same run of days, side by side.

A rollover (stake an amount, re-stake everything won each day at minimum odds)
is a parlay spread over time: with a daily win chance p at odds D it completes
N days with chance p**N, and its expected final value is (p*D)**N of the stake.
At 2.0 the market's own daily chance is about 0.47 once the bookmaker's cut is
removed, so a ten-day run finishes about once in 1,500 to 1,900 tries (see the
product blueprint, "A ten-day rollover at 2.0 finishes once in roughly 1,500 to
1,900 tries"). The planner therefore never shows the classic rollover alone: it
puts the real chances of four strategies next to each other before anyone
confirms a plan.

- classic: stake everything every day; one loss ends the run.
- bank_half: stake the start on day 1; after each win bank half of that day's
  profit and roll the rest; a loss ends the run and the banked money is kept.
- ladder ("pot and ladder"): take a pot of 10% of the start, roll it for at most
  two wins, bank it and take a new pot; stop once the bankroll is below half
  the start.
- flat: stake 5% of the start every day and keep the winnings.

Every strategy's end bankroll depends only on how many days were won and lost
(and, for the ladder, how many pots finished), never on anything else about the
path, so the planner enumerates every outcome exactly with a small dynamic
programme over those counts instead of sampling. The answers are exact,
deterministic and fast; simulate() replays the same rules with a seeded
random.Random as an independent check.

Probabilities are floats. Money is Decimal, computed exactly and rounded to
cents once, at the end. Nothing here asks a language model for a number.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Context, Decimal, InvalidOperation
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple

MIN_DAYS, MAX_DAYS = 1, 60
MIN_ODDS = Decimal("1.2")
MAX_ODDS = Decimal("1000")  # far beyond any sane daily target; keeps the maths finite
MAX_START = Decimal("1000000000")  # far beyond any real bankroll; keeps start * odds ** days within _EXACT
MIN_LEGS, MAX_LEGS = 1, 3
DEFAULT_CUT = 0.045  # a typical bookmaker margin per leg

STRATEGIES = ("classic", "bank_half", "ladder", "flat")
DEFAULT_STRATEGY = "ladder"  # the blueprint's default: rollover-shaped, never wipes out the bankroll

BANK_SHARE = 0.5  # bank_half: share of each day's profit that is banked
POT_SHARE = 0.10  # ladder: the pot, as a share of the start
POT_WINS = 2  # ladder: wins a pot rolls for before it is banked
LADDER_STOP = 0.5  # ladder: stop once the bankroll is below this share of the start
FLAT_SHARE = 0.05  # flat: the stake, as a share of the start

TITLES = {
    "classic": "Classic rollover (high risk)",
    "bank_half": "Roll and bank half",
    "ladder": "Pot and ladder",
    "flat": "Flat 5% stakes",
}
RULES = {
    "classic": "Stake everything every day. One loss ends the run and loses it all.",
    "bank_half": "Stake the start on day 1. After each win, bank half of that day's profit and roll the rest. "
                 "A loss ends the run; the banked money is kept.",
    "ladder": "Roll a pot of 10% of the start for at most 2 wins, bank it and start a new pot. "
              "Stop once the bankroll is below half the start.",
    "flat": "Stake 5% of the start every day and keep the winnings.",
}

CENT = Decimal("0.01")
# Enough digits that even 60 days at 1000.0 stay exact before the one rounding to cents.
_EXACT = Context(prec=400, rounding=ROUND_HALF_UP)
_EPS = 1e-9  # float slack for "at most half" / "in profit" / stop-rule comparisons
# Below this a "1 in X" number would have 300+ digits (and 1 / p can overflow a float), so it is not given.
TINY_CHANCE = 1e-300

State = Hashable
Step = Callable[[Any, bool], Tuple[Any, bool]]  # (state, won today) -> (new state, still running)
Value = Callable[[Any], float]  # state -> bankroll as a multiple of the start


# --- Plain formulas -------------------------------------------------------------------------------


def day_probability(min_odds: Decimal, legs_per_day: int = 1, cut: float = DEFAULT_CUT) -> float:
    """The market's chance of winning one day at min_odds built from legs_per_day legs.

    Each leg carries the bookmaker's cut, so the cut compounds within the day:
    ((1 - cut) ** legs) / min_odds. One leg at 2.0 with a 4% cut gives 0.48;
    two legs with a 4% cut give 0.4608.
    """
    odds = float(min_odds)
    if odds <= 1:
        raise ValueError("min_odds must be above 1")
    if legs_per_day < 1:
        raise ValueError("legs_per_day must be at least 1")
    if not 0 <= cut < 1:
        raise ValueError("cut must be between 0 and 1")
    return ((1 - cut) ** legs_per_day) / odds


def completion_chance(p: float, days: int) -> float:
    """Chance of winning every one of `days` days at daily chance p: p ** days."""
    return p ** days


def expected_multiple(p: float, odds: float, days: int) -> float:
    """Expected end value of a classic rollover as a multiple of the start: (p * odds) ** days."""
    return (p * odds) ** days


def survival(p: float, days: int) -> List[float]:
    """Classic rollover: P(still alive after day d) for d = 1..days, i.e. p ** d.

    The chance is multiplied out one day at a time, exactly as outcomes() carries
    it, so this curve, the day-by-day tables and the classic strategy's numbers
    agree to the last digit (p ** d itself can differ in the last bit, which
    shows up in a "1 in 10^20" figure).
    """
    curve, chance = [], 1.0
    for _ in range(days):
        chance *= p
        curve.append(chance)
    return curve


def required_daily_p(target_completion: float, days: int) -> float:
    """The daily win chance needed to complete `days` days with chance target_completion.

    A coin-flip chance of finishing 10 days needs 0.5 ** (1/10) = 0.933 every day.
    """
    if not 0 < target_completion <= 1:
        raise ValueError("target_completion must be in (0, 1]")
    if days < 1:
        raise ValueError("days must be at least 1")
    return target_completion ** (1 / days)


def overstatement(p_claimed: float, p_true: float, days: int) -> float:
    """How many times a claimed daily chance overstates the chance of completing `days` days.

    An AI that says 0.52 when the truth is 0.47 overstates a 10-day run 2.75-fold
    and a 30-day run 20.8-fold, which is why small overconfidence matters here.
    """
    if p_true <= 0:
        raise ValueError("p_true must be above 0")
    return (p_claimed / p_true) ** days


def kelly_fraction(p: float, odds: float) -> float:
    """Kelly stake as a share of the bankroll: (p*odds - 1) / (odds - 1), never below 0.

    Staking everything is 1 / kelly_fraction times the Kelly stake: at 2.0 with
    p = 0.52 that is 25 times.
    """
    return max((p * odds - 1) / (odds - 1), 0.0)


# --- Exact outcome distributions ------------------------------------------------------------------


@dataclass
class Outcomes:
    """One strategy's exact results: end bankrolls (multiples of the start) and when runs stop."""

    finals: List[Tuple[float, float]]  # (end multiple, probability), sorted by multiple
    ends: List[float]  # ends[d - 1]: chance the run stops on day d (busted or stopped)
    # alive[d - 1]: chance the run is still going after day d, summed from the runs still going.
    # Working it out as 1 - sum(ends) would cancel chances below about 1e-16 (a 60-day classic
    # run at 0.47 is 2.1e-20) down to rounding noise or 0.
    alive: List[float]
    best: float  # end multiple if every day wins

    @property
    def survival(self) -> List[float]:
        """P(the run is still going after day d) for d = 1..days."""
        return list(self.alive)

    @property
    def chance_complete(self) -> float:
        """Chance the run is still going after its last day."""
        return self.survival[-1]

    @property
    def mean(self) -> float:
        return sum(value * chance for value, chance in self.finals)

    @property
    def median(self) -> float:
        """The smallest end value that at least half of all runs reach or stay below."""
        total = 0.0
        for value, chance in self.finals:
            total += chance
            if total >= 0.5 - _EPS:
                return value
        return self.finals[-1][0]

    def share(self, test: Callable[[float], bool]) -> float:
        return sum(chance for value, chance in self.finals if test(value))

    def most_likely_end(self) -> Dict[str, Any]:
        """The single most likely way the run ends: on a given day, or by completing."""
        day, chance = max(enumerate(self.ends, start=1), key=lambda item: (item[1], -item[0]))
        complete = self.chance_complete
        if complete > chance:
            return {"day": None, "probability": _chance(complete), "completes": True}
        return {"day": day, "probability": _chance(chance), "completes": False}


def _classic(odds: float, pot_basis: str) -> Tuple[State, Step, Value]:
    # State: (days won, lost?). A loss is the end of the run.
    def step(state, won):
        wins, _ = state
        return ((wins + 1, False), True) if won else ((wins, True), False)

    def value(state):
        wins, lost = state
        return 0.0 if lost else odds ** wins

    return (0, False), step, value


def _bank_half(odds: float, pot_basis: str) -> Tuple[State, Step, Value]:
    # After a win the next stake is today's stake times `grow`; the banked part is kept for good.
    grow = 1 + (1 - BANK_SHARE) * (odds - 1)

    def banked(wins: int) -> float:
        return sum(BANK_SHARE * (odds - 1) * grow ** day for day in range(wins))

    def step(state, won):
        wins, _ = state
        return ((wins + 1, False), True) if won else ((wins, True), False)

    def value(state):
        wins, lost = state
        return banked(wins) + (0.0 if lost else grow ** wins)

    return (0, False), step, value


def _ladder(odds: float, pot_basis: str) -> Tuple[State, Step, Value]:
    # State: (pots banked, pots lost, wins of the open pot; 0 = no pot open).
    # pot_basis "start" (the spec) takes every pot as 10% of the start; "bankroll" takes 10% of the
    # bankroll at the time, the variant the blueprint's 30-day table was simulated with.
    if pot_basis == "start":
        def cash(banked: int, lost: int) -> float:
            return 1 + POT_SHARE * (banked * (odds ** POT_WINS - 1) - lost)

        def pot(banked: int, lost: int) -> float:
            return POT_SHARE
    elif pot_basis == "bankroll":
        def cash(banked: int, lost: int) -> float:
            return (1 - POT_SHARE + POT_SHARE * odds ** POT_WINS) ** banked * (1 - POT_SHARE) ** lost

        def pot(banked: int, lost: int) -> float:
            return POT_SHARE * cash(banked, lost)
    else:
        raise ValueError(f"Unknown pot_basis {pot_basis!r}")

    def step(state, won):
        banked, lost, wins = state
        if won:
            if wins + 1 == POT_WINS:
                return (banked + 1, lost, 0), True
            return (banked, lost, wins + 1), True
        return (banked, lost + 1, 0), cash(banked, lost + 1) >= LADDER_STOP - _EPS

    def value(state):
        banked, lost, wins = state
        if wins == 0:
            return cash(banked, lost)
        return cash(banked, lost) - pot(banked, lost) + pot(banked, lost) * odds ** wins

    return (0, 0, 0), step, value


def _flat(odds: float, pot_basis: str) -> Tuple[State, Step, Value]:
    # State: (days won, days lost). The run stops if the bankroll can no longer cover the stake.
    def bankroll(wins: int, losses: int) -> float:
        return 1 + FLAT_SHARE * (wins * (odds - 1) - losses)

    def step(state, won):
        wins, losses = state
        if won:
            return (wins + 1, losses), True
        return (wins, losses + 1), bankroll(wins, losses + 1) >= FLAT_SHARE - _EPS

    def value(state):
        return bankroll(*state)

    return (0, 0), step, value


_RULE_BUILDERS = {"classic": _classic, "bank_half": _bank_half, "ladder": _ladder, "flat": _flat}


def _rules(strategy: str, odds: float, pot_basis: str) -> Tuple[State, Step, Value]:
    if strategy not in _RULE_BUILDERS:
        raise ValueError(f"Unknown strategy {strategy!r}; use one of {', '.join(STRATEGIES)}")
    return _RULE_BUILDERS[strategy](odds, pot_basis)


def outcomes(strategy: str, days: int, odds: float, p: float, pot_basis: str = "start") -> Outcomes:
    """Every way `strategy` can end over `days` days at daily chance p, with exact probabilities.

    Runs that are still going are carried day to day as {state: probability};
    states are small tuples of counts, so equal situations merge and the work
    stays tiny (a few thousand states at 60 days).
    """
    if days < 1:
        raise ValueError("days must be at least 1")
    _check_probability("p", p)
    first, step, value = _rules(strategy, float(odds), pot_basis)
    running: Dict[State, float] = {first: 1.0}
    finished: Dict[State, float] = defaultdict(float)
    ends: List[float] = []
    alive: List[float] = []
    for _ in range(days):
        tomorrow: Dict[State, float] = defaultdict(float)
        stopped = 0.0
        for state, mass in running.items():
            for won, chance in ((True, p), (False, 1.0 - p)):
                new_state, still = step(state, won)
                if still:
                    tomorrow[new_state] += mass * chance
                else:
                    finished[new_state] += mass * chance
                    stopped += mass * chance
        ends.append(stopped)
        alive.append(sum(tomorrow.values()))
        running = tomorrow
    for state, mass in running.items():
        finished[state] += mass
    best = first
    for _ in range(days):
        best, _still = step(best, True)
    finals = sorted((value(state), mass) for state, mass in finished.items())
    return Outcomes(finals=finals, ends=ends, alive=alive, best=value(best))


def simulate(strategy: str, days: int, odds: float, p: float, runs: int = 20000, seed: int = 7,
             pot_basis: str = "start") -> List[float]:
    """End bankrolls (multiples of the start) of `runs` seeded random runs of the same rules.

    A cross-check on outcomes(): the same seed always gives the same runs.
    """
    rng = random.Random(seed)
    first, step, value = _rules(strategy, float(odds), pot_basis)
    results = []
    for _ in range(runs):
        state, still = first, True
        for _day in range(days):
            state, still = step(state, rng.random() < p)
            if not still:
                break
        results.append(value(state))
    return results


# --- Day-by-day tables ----------------------------------------------------------------------------


def _cents(amount: Decimal) -> Decimal:
    return amount.quantize(CENT, context=_EXACT)


def _money(start: Decimal, multiple: float) -> Decimal:
    """start * multiple in cents (the multiple comes from float probabilities, so it is not exact)."""
    return _cents(_EXACT.multiply(start, Decimal(repr(multiple))))


def _chance(p: float) -> float:
    """A chance rounded to 4 significant figures, so 1-in-2,000 chances keep their digits."""
    return float(f"{p:.4g}")


def _one_in(p: Optional[float]) -> Optional[int]:
    """"1 in X" for a chance p: None when there is no chance or it is below TINY_CHANCE."""
    if p is None or not p >= TINY_CHANCE:
        return None
    return round(1 / p)


def _row_chances(day: int, market: List[float], ai: Optional[List[float]]) -> Dict[str, Any]:
    """One table row's chances, read from survival() curves so they match the strategy numbers."""
    p_ai = None if ai is None else ai[day - 1]
    return {"market_chance": _chance(market[day - 1]), "market_one_in": _one_in(market[day - 1]),
            "ai_chance": None if p_ai is None else _chance(p_ai), "ai_one_in": _one_in(p_ai)}


def _curves(days: int, p_market: float, p_ai: Optional[float]) -> Tuple[List[float], Optional[List[float]]]:
    return survival(p_market, days), None if p_ai is None else survival(p_ai, days)


def classic_table(start: Decimal, days: int, min_odds: Decimal, p_market: float,
                  p_ai: Optional[float] = None) -> List[Dict[str, Any]]:
    """Classic rollover if every day wins: each day's stake and return, with the chance of getting there.

    The chance on day d is the chance of winning days 1..d, i.e. p ** d.
    """
    start, odds = _decimal(start), _decimal(min_odds)
    market, ai = _curves(days, p_market, p_ai)
    rows, stake = [], start
    for day in range(1, days + 1):
        returns = _EXACT.multiply(stake, odds)
        rows.append({"day": day, "stake": str(_cents(stake)), "returns": str(_cents(returns)),
                     **_row_chances(day, market, ai)})
        stake = returns
    return rows


def bank_half_table(start: Decimal, days: int, min_odds: Decimal, p_market: float,
                    p_ai: Optional[float] = None) -> List[Dict[str, Any]]:
    """Bank-as-you-go table: every day wins, half of each day's profit is banked, the rest rolls.

    `banked_total` is what a loss on the next day would leave; `roll` is the next day's stake.
    """
    start, odds = _decimal(start), _decimal(min_odds)
    share = Decimal(str(BANK_SHARE))
    market, ai = _curves(days, p_market, p_ai)
    rows, stake, banked = [], start, Decimal(0)
    for day in range(1, days + 1):
        returns = _EXACT.multiply(stake, odds)
        bank = _EXACT.multiply(_EXACT.subtract(returns, stake), share)
        banked = _EXACT.add(banked, bank)
        roll = _EXACT.subtract(returns, bank)
        rows.append({"day": day, "stake": str(_cents(stake)), "returns": str(_cents(returns)),
                     "bank": str(_cents(bank)), "banked_total": str(_cents(banked)), "roll": str(_cents(roll)),
                     **_row_chances(day, market, ai)})
        stake = roll
    return rows


# --- Side-by-side comparison ----------------------------------------------------------------------


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"Not a number: {value!r}") from error


def _is_count(value: Any) -> bool:
    """A plain int: 10.0 days or 1.5 legs is a mistake, and so is True (Python counts bools as ints)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _check_probability(name: str, p: Optional[float]) -> None:
    if p is not None and not 0 < p < 1:
        raise ValueError(f"{name} must be between 0 and 1 (exclusive), got {p}")


def _validate(start: Decimal, days: int, min_odds: Decimal, legs_per_day: int) -> None:
    # Finite first: comparing a Decimal NaN raises InvalidOperation, which callers would not expect.
    if not start.is_finite() or not 0 < start <= MAX_START:
        raise ValueError(f"start must be above 0 and at most {MAX_START:,}")
    if not _is_count(days) or not MIN_DAYS <= days <= MAX_DAYS:
        raise ValueError(f"days must be a whole number between {MIN_DAYS} and {MAX_DAYS}")
    if not min_odds.is_finite() or not MIN_ODDS <= min_odds <= MAX_ODDS:
        raise ValueError(f"min_odds must be between {MIN_ODDS} and {MAX_ODDS}")
    if not _is_count(legs_per_day) or not MIN_LEGS <= legs_per_day <= MAX_LEGS:
        raise ValueError(f"legs_per_day must be a whole number between {MIN_LEGS} and {MAX_LEGS}")


def _stats(result: Outcomes, start: Decimal, p: float) -> Dict[str, Any]:
    """The numbers one strategy shows under one daily probability."""
    curve = result.survival
    expected_end = _money(start, result.mean)
    complete = result.chance_complete
    likely = result.most_likely_end()
    return {
        "p_day": round(p, 4),
        "chance_complete": _chance(complete),
        "complete_one_in": _one_in(complete),
        "chance_profit": _chance(result.share(lambda value: value > 1 + _EPS)),
        "median_end": str(_money(start, result.median)),
        "chance_half_or_less": _chance(result.share(lambda value: value <= 0.5 + _EPS)),
        "expected_end": str(expected_end),
        "expected_cost": str(_cents(_EXACT.subtract(start, expected_end))),
        "best_end": str(_money(start, result.best)),
        "survival": [_chance(alive) for alive in curve],
        "most_likely_end_day": likely["day"],  # None when completing is the most likely outcome
        "most_likely_end": likely,  # {day, probability, completes}
        "over_by_day_2": _chance(1 - curve[min(2, len(curve)) - 1]),
        "average_days_survived": round(sum(curve), 2),
    }


def _warnings(days: int, min_odds: Decimal, legs_per_day: int, p_market: float, p_ai: Optional[float],
              estimated: bool) -> List[str]:
    notes = ["One loss on any day ends a classic rollover and loses everything staked so far."]
    if legs_per_day > 1:
        notes.append(f"Building each day from {legs_per_day} legs takes the bookmaker's cut {legs_per_day} times "
                     "a day; one leg near the minimum odds costs less.")
    if estimated:
        notes.append(f"The market's daily chance is estimated from a {DEFAULT_CUT:.1%} bookmaker cut per leg; "
                     "real prices may differ.")
    if p_ai is not None and p_ai > p_market:
        notes.append(f"The AI rates each day above the market. If the market is right, the AI overstates the "
                     f"chance of completing {overstatement(p_ai, p_market, days):.1f}-fold.")
    # Stands on its own: it can appear without the note above (no AI chance, or an AI at or below the market).
    source, p = ("the market's", p_market) if p_ai is None else ("the AI's", p_ai)
    kelly = kelly_fraction(p, float(min_odds))
    if kelly > 0:
        notes.append(f"At {source} daily chance of {_percent(p)} and odds of {_say_odds(min_odds)}, staking "
                     f"everything is about {1 / kelly:,.0f} times the Kelly stake (the stake that grows a "
                     "bankroll fastest; more only adds risk).")
    return notes


def compare(start: Decimal, days: int, min_odds: Decimal, legs_per_day: int = 1,
            p_market: Optional[float] = None, p_ai: Optional[float] = None, runs: int = 20000,
            seed: int = 7, currency: str = "$") -> Dict[str, Any]:
    """Classic, bank-half, pot-and-ladder and flat stakes side by side, under the market's and the AI's chances.

    p_market is the margin-free chance of winning one day; when it is None it
    is estimated with day_probability(min_odds, legs_per_day). For every
    strategy and probability the result carries: chance_complete (the run is
    still going after the last day: every day won for classic and bank-half,
    never stopped for the ladder, never out of money for flat; classic is
    exactly p ** days), chance_profit, median_end, chance_half_or_less,
    expected_end and expected_cost in money, best_end, the survival curve,
    most_likely_end_day (a day number, or None when completing is the most
    likely outcome) and most_likely_end ({day, probability, completes}). Money
    is returned as strings with 2 decimals; odds as text with 2 decimals.

    Every strategy is enumerated exactly, so the answer is deterministic and
    needs no sampling: `runs` and `seed` are accepted for callers written
    against a simulated planner and are not used (simulate() is the seeded
    cross-check).
    """
    start, min_odds = _decimal(start), _decimal(min_odds)
    _validate(start, days, min_odds, legs_per_day)
    estimated = p_market is None
    p_market = day_probability(min_odds, legs_per_day) if p_market is None else float(p_market)
    p_ai = None if p_ai is None else float(p_ai)
    _check_probability("p_market", p_market)
    _check_probability("p_ai", p_ai)
    odds = float(min_odds)
    probabilities = {"market": p_market, "ai": p_ai}

    strategies: Dict[str, Any] = {}
    for name in STRATEGIES:
        entry: Dict[str, Any] = {"title": TITLES[name], "rule": RULES[name], "default": name == DEFAULT_STRATEGY}
        for source, p in probabilities.items():
            entry[source] = None if p is None else _stats(outcomes(name, days, odds, p), start, p)
        strategies[name] = entry

    coin_flip = required_daily_p(0.5, days)
    return {
        "start": str(_cents(start)),
        "days": days,
        "min_odds": _say_odds(min_odds),
        "legs_per_day": legs_per_day,
        "target": str(_cents(_EXACT.multiply(start, _EXACT.power(min_odds, days)))),
        "p_market": round(p_market, 4),
        "p_market_estimated": estimated,
        "p_ai": None if p_ai is None else round(p_ai, 4),
        "strategies": strategies,
        "default": DEFAULT_STRATEGY,
        "ladder_table": classic_table(start, days, min_odds, p_market, p_ai),
        "bank_half_table": bank_half_table(start, days, min_odds, p_market, p_ai),
        "coin_flip": {"daily_p": round(coin_flip, 4), "edge": round(coin_flip * odds - 1, 4)},
        "overstatement": None if p_ai is None else round(overstatement(p_ai, p_market, days), 2),
        "summary": honest_summary(start, days, min_odds, p_market, p_ai, legs_per_day, currency),
        "requirement": requirement_sentence(days, min_odds, p_market),
        "warnings": _warnings(days, min_odds, legs_per_day, p_market, p_ai, estimated),
        "method": "exact",
    }


# --- Honest copy ----------------------------------------------------------------------------------


def _say_money(amount: Decimal, currency: str) -> str:
    """$10, $10,240 or $10.50: no cents when the amount is whole."""
    amount = _cents(amount)
    if amount == amount.to_integral_value():
        return f"{currency}{int(amount):,}"
    return f"{currency}{amount:,.2f}"


def _say_odds(odds: Any) -> str:
    """Decimal odds as every part of a plan shows them: 2 decimals, half up (1.625 -> "1.63"), like tickets.py."""
    return str(_cents(_decimal(odds)))


def _article(number: str) -> str:
    """"a" or "an" before a number below 1,000 as it is read aloud: an 8%, an 11%, an 18%, an 87%, a 93%."""
    whole = number.split(".")[0].rstrip("%")
    return "an" if whole.startswith("8") or whole in ("11", "18") else "a"


def _percent(p: float) -> str:
    pct = p * 100
    if pct >= 10 or pct == 0:
        return f"{pct:.0f}%"
    if pct >= 1:
        return f"{pct:.1f}%"
    return f"{pct:.2g}%"


def _about_one_in(p: float) -> str:
    """"about 1 in 1,900" (two significant figures above 100), or a percentage for likely events."""
    if p >= 0.5:
        return f"about {_article(_percent(p))} {_percent(p)} chance"
    if not p >= TINY_CHANCE:
        return "less than 1 in 10^300"
    n = round(1 / p)
    if n < 100:
        return f"about 1 in {n} ({_percent(p)})"
    rounded = round(n, -(len(str(n)) - 2))
    return f"about 1 in {rounded:,}"


def honest_summary(start: Decimal, days: int, min_odds: Decimal, p_market: float, p_ai: Optional[float] = None,
                   legs_per_day: int = 1, currency: str = "$") -> str:
    """The sentence every plan opens with, e.g.

    "To turn $10 into $10,240 in 10 days you need 10 wins in a row at 2.00. The
    market gives that about 1 in 1,900. Most likely the run ends on day 1 (53%)."
    """
    start, odds = _decimal(start), _decimal(min_odds)
    target = _EXACT.multiply(start, _EXACT.power(odds, days))
    span = "1 day" if days == 1 else f"{days} days"
    if legs_per_day == 1:
        need = "1 win" if days == 1 else f"{days} wins in a row"
    else:
        need = "1 winning day" if days == 1 else f"{days} winning days in a row"
    need += f" at {_say_odds(odds)}"
    if legs_per_day > 1:
        need += f", each built from {legs_per_day} legs"
    chance = _about_one_in(completion_chance(p_market, days))
    if p_ai is not None:
        chance += f" and the AI {_about_one_in(completion_chance(p_ai, days))}"
    likely = outcomes("classic", days, float(odds), p_market).most_likely_end()
    if likely["completes"]:
        ending = f"Most likely the run completes ({_percent(likely['probability'])})."
    else:
        ending = f"Most likely the run ends on day {likely['day']} ({_percent(likely['probability'])})."
    return (f"To turn {_say_money(start, currency)} into {_say_money(target, currency)} in {span} you need "
            f"{need}. The market gives that {chance}. {ending}")


def requirement_sentence(days: int, min_odds: Decimal, p_market: float, target: float = 0.5) -> str:
    """What daily chance a given chance of finishing would need, next to the market's own."""
    needed = required_daily_p(target, days)
    odds = _decimal(min_odds)
    edge = needed * float(odds) - 1
    wanted = _percent(target)
    goal = "A coin-flip chance" if target == 0.5 else f"{_article(wanted).capitalize()} {wanted} chance"
    span = "1 day" if days == 1 else f"{days} days"
    if edge > 0:
        price = f"an edge of {_percent(edge)} per bet at {_say_odds(odds)}"
    else:
        price = f"which odds of {_say_odds(odds)} cover without an edge"
    need = _percent(needed)
    return (f"{goal} of finishing {span} needs {_article(need)} {need} win chance every day, {price}. "
            f"The market gives each day {_percent(p_market)}.")


def cost_sentence(start: Decimal, expected_end: Decimal, currency: str = "$") -> str:
    """The expected result in money, e.g. "On average a run returns $5.39 of the $10 staked: ..."."""
    start, expected_end = _decimal(start), _decimal(expected_end)
    if expected_end < start:
        cost = _EXACT.subtract(start, expected_end)
        return (f"On average a run returns {_say_money(expected_end, currency)} of the "
                f"{_say_money(start, currency)} staked: an expected cost of {_say_money(cost, currency)}.")
    return (f"On average a run returns {_say_money(expected_end, currency)} for the {_say_money(start, currency)} "
            "staked, if the daily chance used is right.")
