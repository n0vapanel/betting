"""Stake-and-target builder: "stake $10, I want $100 back", with the honest ceiling first.

A target fixes the ticket's odds before any leg is picked: D* = target / stake.
The odds then cap the honest chance. With no real edge a leg wins about
(1 - h) / its odds, where h is the bookmaker's cut, so a ticket at D* wins
about (1 - h)^n / D* whichever n legs are chosen. There is also a harder limit:
if every bet on offer is fair or worse, no ticket or staking plan turns a stake
into M times itself more often than 1 in M (Markov's inequality; see the product
blueprint, "stake-and-target builder"). That ceiling is worked out and shown
before any ticket, so the user sees the real chance first.

The search then works only with numbers the brain already has:

- candidates that haven't started, carry an AI or a market chance, and have a
  market chance of 25-90% (long shots and near-certainties are where prices
  and models are least reliable);
- the 40 best of them by cautious expected return, where the cautious chance
  is the AI's pulled halfway back to the market's: p_c = market + 0.5 * (ai - market).
  Both are chances of a win outright: on a line that can push, the market's
  margin-free chance is a chance given no push (see tickets.py), so it is
  scaled by (1 - push) first, and the push itself counts once, as stake back.
  Each game gives at most its three best legs (as engine/bets.py does), so a
  game with dozens of player props can't crowd every other game out of a
  search that may only use one leg from it anyway;
- every combination of one to four legs from different games whose combined
  odds land between 10% below and 15% above D* (forty legs give about 100,000
  combinations, a fraction of a second), searched by the Builder
  (engine/builder.py, v3 Stage 3): one run per ticket size, with that window as
  its ticket-odds band, the cautious chance as each leg's chance and the
  cautious expected return as its only ranking (builder.RETURN_ONLY); each
  slot then takes the best of the builder's ranked tickets by this module's
  rank (cautious expected return, then cautious chance);
- legs priced only at books a person can bet at: a price from Pinnacle or an
  exchange (the markets brain's consensus anchor) is never used;
- the best single, the best two-leg ticket and the best three- or four-leg
  ticket by cautious expected return. At a fixed payout that is also the most
  likely ticket, because P = (1 + EV) / D.

Each ticket is priced by tickets.price_ticket, the one pricing function every
surface uses, at exactly the stake given: the builder never raises the stake
to reach a target. When no ticket has a positive cautious expected return the
verdict says so in money ("No edge found: entertainment with an expected cost
of $Y"). A ticket's loss chance counts a pushed leg as stake back, not as a
loss. Money is Decimal in whole cents (a stake or target in fractions of a cent
is refused rather than shown at a different value than the one priced),
rounded to cents once at the end; chances are floats.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import builder
from .bets import TIERS, Tier, price_book
from .odds_math import decimal_to_american, format_american
from .tickets import TicketLeg, chance_label, one_in, price_ticket

SEARCH_MAX_LEGS = 4  # more legs only add the bookmaker's cut again
POOL_SIZE = 40  # candidates kept for the search: C(40, 4) is about 91,000 combinations
PER_GAME = 3  # best legs a game may put in the pool (engine/bets.py keeps the same)
MARKET_LOW, MARKET_HIGH = 0.25, 0.90  # market chances a leg may have
WINDOW_LOW, WINDOW_HIGH = Decimal("0.90"), Decimal("1.15")  # combined odds allowed, as shares of D*
SHORT_ODDS = Decimal("1.15")  # below this a target is a single on a heavy favourite, or no bet
LOTTERY_ODDS = Decimal("50")  # above this a target is lottery-like
CUT = 0.045  # a typical bookmaker's cut per leg, for the no-edge ceiling
PULL = 0.5  # the cautious chance keeps half of the AI's gap to the market
CURRENCY = "$"
CENTS = Decimal("0.01")
MAX_AMOUNT = Decimal("1000000000")  # $1bn: past any real stake, and well inside exact cents arithmetic
MAX_MULTIPLE = Decimal("10000000")  # past 1 in 10,000,000 a ticket's numbers say nothing more than "no"
PRECISION = 40  # digits for the exact expected results
_EPS = 1e-9  # float slack at the edges of the odds window

# The three tickets returned: the best single, the best double and the best of three or four legs
SLOTS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("single", (1,)), ("double", (2,)), ("treble_or_more", (3, 4)),
)
TICKET_NAMES = {1: "Single", 2: "Double", 3: "Treble", 4: "Four-leg multi"}
_LEG_WORDS = {1: "one leg", 2: "two legs", 3: "three legs", 4: "four legs"}

Amount = Union[Decimal, float, int, str]


# --- the honest ceiling --------------------------------------------------------------------------

def no_edge_chance(multiple: float, legs: int = 1, cut: float = CUT) -> float:
    """The chance of a ticket at these odds with no real edge: (1 - cut)^legs / multiple, at most 1.

    At 10x a single wins about 9.6% (1 in 10) and four legs about 8.3% (1 in 12).
    """
    if multiple <= 0:
        raise ValueError(f"The odds multiple must be above zero, got {multiple}")
    return min(1.0, (1.0 - cut) ** legs / multiple)


def _market(leg: TicketLeg) -> float:
    """The market's margin-free chance given no push, or, when it is missing, the price less a typical cut.

    This is the chance the 25-90% filter reads; the search itself uses _market_outright."""
    if leg.market_prob is not None:
        return leg.market_prob
    return (1.0 - CUT) / float(leg.price)


def _market_outright(leg: TicketLeg) -> float:
    """The market's chance of a win outright, the AI's footing: a push takes its share of the no-push chance."""
    return _market(leg) * (1.0 - leg.push_prob)


def cautious_probability(leg: TicketLeg) -> Optional[float]:
    """The leg's cautious chance of a win outright: the AI's pulled halfway back to the market's.

    p_c = market + 0.5 * (ai - market), with the market's chance made outright first
    (q * (1 - push)) so a push is never counted in it and again as stake back.
    Without an AI chance it is the market's; without a market chance the price
    less a typical cut stands in for it. None when the leg has neither chance.
    """
    if leg.ai_prob is None and leg.market_prob is None:
        return None
    market = _market_outright(leg)
    if leg.ai_prob is None:
        return market
    return market + PULL * (leg.ai_prob - market)


# --- small helpers ------------------------------------------------------------------------------

def _amount(value: Amount, name: str) -> Decimal:
    """A stake or target as exact money: whole cents from $0.01 to $1,000,000,000, else ValueError.

    Fractions of a cent are refused rather than rounded, so the amount shown is
    always the amount priced; the upper limit keeps cents arithmetic exact.
    """
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"The {name} must be a number, got {value!r}") from None
    if not amount.is_finite() or amount <= 0:
        raise ValueError(f"The {name} must be above zero, got {value}")
    if amount < CENTS:
        raise ValueError(f"The {name} must be at least {_dollars(CENTS)}, got {value}")
    if amount > MAX_AMOUNT:
        raise ValueError(f"The {name} must be at most {_dollars(MAX_AMOUNT)}, got {value}")
    if amount != amount.quantize(CENTS):
        raise ValueError(f"The {name} must be in whole cents, got {value}")
    return amount


def _money(value: Decimal) -> str:
    """Cents, half up, as a string; never "-0.00"."""
    rounded = value.quantize(CENTS, rounding=ROUND_HALF_UP)
    return str(rounded if rounded != 0 else abs(rounded))


def _dollars(value: Decimal, sign: bool = False) -> str:
    """Money for a sentence: "$1,234.50", "-$0.45", or "+$0.08" when ``sign`` is set."""
    rounded = value.quantize(CENTS, rounding=ROUND_HALF_UP)
    text = f"{CURRENCY}{abs(rounded):,.2f}"
    if rounded < 0:
        return f"-{text}"
    return f"+{text}" if sign and rounded > 0 else text


def _now(now: Optional[datetime.datetime]) -> datetime.datetime:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=datetime.timezone.utc)


def _started(leg: TicketLeg, now: datetime.datetime) -> bool:
    """Whether the leg's game has begun; a start time that can't be read can't be checked, so it counts."""
    try:
        start = datetime.datetime.fromisoformat(str(leg.start).replace("Z", "+00:00"))
    except ValueError:
        return True
    if start.tzinfo is None:
        start = start.replace(tzinfo=datetime.timezone.utc)
    return start <= now


def _tier(tier: Union[None, str, Tier]) -> Optional[Tier]:
    """A tier from its key or title ("risky", "Risky"), or None for any chance."""
    if not tier or isinstance(tier, Tier):
        return tier or None
    name = str(tier).strip().lower()
    for known in TIERS:
        if name in (known.key, known.title.lower()):
            return known
    raise ValueError(f"Unknown tier {tier!r}; choose one of {', '.join(t.key for t in TIERS)}")


def _tier_of(chance: float) -> Optional[Tier]:
    return next((tier for tier in TIERS if tier.p_low <= chance < tier.p_high), None)


def _band(tier: Tier) -> str:
    return f"{tier.p_low:.0%}+" if tier.p_high > 1 else f"{tier.p_low:.0%}-{tier.p_high:.0%}"


def _slot(legs: int) -> str:
    return next(name for name, sizes in SLOTS if legs in sizes)


def _not_losing(chance: float, push: float) -> float:
    """The chance a leg doesn't lose: it wins, or it pushes and gives the stake back (at most 1)."""
    return chance + min(push, 1.0 - chance)


# --- candidates ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Option:
    """One usable leg with the numbers the search needs, worked out once."""
    leg: TicketLeg
    price: float
    cautious: float  # p_c
    brain: float  # the AI's chance, else the market's (outright): what sets the ticket's tier
    market: float  # the market's chance of a win outright, or the price less a typical cut
    push: float  # capped so win + push never passes 1

    @property
    def factor(self) -> float:
        """Expected return per unit staked at the cautious chance: a win pays the odds, a push the stake."""
        return self.cautious * self.price + self.push

    @property
    def game(self) -> str:
        """The leg's correlation group: legs that share it come from one game."""
        return self.leg.group or self.leg.event_id


def _option(leg: TicketLeg) -> _Option:
    cautious = cautious_probability(leg)
    market = _market_outright(leg)
    brain = leg.ai_prob if leg.ai_prob is not None else market
    return _Option(leg, float(leg.price), cautious, brain, market, min(leg.push_prob, 1.0 - cautious))


def _usable(candidates: Sequence[TicketLeg], now: datetime.datetime) -> Tuple[List[_Option], Dict[str, int]]:
    """The legs a ticket may use, and how many were left out for each reason."""
    dropped = {"started": 0, "no_chance": 0, "outside_range": 0}
    best: Dict[str, _Option] = {}
    for leg in candidates:
        if leg.book and not price_book(leg.book):
            continue  # Pinnacle or an exchange: the consensus's anchor, never a price to build a ticket at
        if leg.ai_prob is None and leg.market_prob is None:
            dropped["no_chance"] += 1
        elif _started(leg, now):
            dropped["started"] += 1
        elif not MARKET_LOW <= _market(leg) <= MARKET_HIGH:
            dropped["outside_range"] += 1
        else:
            option = _option(leg)
            kept = best.get(leg.key)
            if kept is None or option.price > kept.price:
                best[leg.key] = option  # the same pick at several books: keep the best price
    return list(best.values()), dropped


def _pool(options: Sequence[_Option], low: float, high: float, max_legs: int) -> Tuple[List[_Option], bool]:
    """The best legs by cautious expected return among those that could fit the odds window.

    A leg priced above the window can't be in any ticket that fits, and neither
    can one too short to reach it even next to the longest prices on the board.
    Each game keeps only its PER_GAME best legs before the POOL_SIZE cut: a
    ticket takes one leg per game, so a game with dozens of strong props would
    otherwise fill the pool with legs that can't go together. The pool comes
    back sorted by price, shortest first, for the search, with whether any leg
    that could fit was left out (so "no combination" is never said of the
    whole slate when only part of it was searched).
    """
    short_enough = [option for option in options if option.price <= high]
    if not short_enough:
        return [], False
    top = max(option.price for option in short_enough)
    fits = [option for option in short_enough if option.price * top ** (max_legs - 1) >= low]
    fits.sort(key=lambda option: (-option.factor, -option.price, option.leg.key))
    per_game: Dict[str, int] = {}
    kept: List[_Option] = []
    for option in fits:  # best first, so each game keeps its best legs
        if per_game.get(option.game, 0) < PER_GAME:
            per_game[option.game] = per_game.get(option.game, 0) + 1
            kept.append(option)
    pool = kept[:POOL_SIZE]
    return sorted(pool, key=lambda option: (option.price, option.leg.key)), len(pool) < len(fits)


# --- the search ---------------------------------------------------------------------------------

@dataclass
class _Found:
    options: Tuple[_Option, ...]
    odds: float
    rank: Tuple[float, ...]

    @property
    def cautious_return(self) -> float:
        return math.prod(option.factor for option in self.options)

    @property
    def cautious_chance(self) -> float:
        return math.prod(option.cautious for option in self.options)

    @property
    def brain_chance(self) -> float:
        return math.prod(option.brain for option in self.options)

    # A multi loses its stake only when some leg loses: a pushed leg drops out and the rest still count
    @property
    def cautious_loses(self) -> float:
        return 1.0 - math.prod(option.cautious + option.push for option in self.options)

    @property
    def brain_loses(self) -> float:
        return 1.0 - math.prod(_not_losing(option.brain, option.leg.push_prob) for option in self.options)

    @property
    def market_loses(self) -> float:
        return 1.0 - math.prod(_not_losing(option.market, option.leg.push_prob) for option in self.options)

    @property
    def name(self) -> str:
        return TICKET_NAMES[len(self.options)]


def _rank(options: Tuple[_Option, ...], odds: float, target: float) -> Tuple[float, ...]:
    """Cautious expected return first; then the cautious chance, closeness to the target, fewer legs."""
    cautious_return = math.prod(option.factor for option in options)
    cautious_chance = math.prod(option.cautious for option in options)
    return (round(cautious_return - 1, 9), round(cautious_chance, 9),
            -round(abs(math.log(odds / target)), 9), -len(options))


SEARCH_BOOK = "Bookmaker"  # the builder needs a book's name; a quote that names none gets this one


def _candidate(option: _Option) -> builder.Candidate:
    """One pool leg as the builder's Candidate: its chance is the cautious chance (the builder takes no caution off,
    so its cautious return per unit is p_c x odds + push, this module's), its game the leg's correlation group, and
    its ref unique per quote (two legs with one ref would be one leg to the builder)."""
    leg = option.leg
    away, _, home = leg.game.partition(" @ ")
    return builder.Candidate(
        ref={"league": leg.league, "id": leg.key, "market": leg.market, "side": leg.side, "point": leg.point},
        league=leg.league, game_key=option.game, teams=(away or leg.game, home or leg.game), start=leg.start,
        price=leg.price, book=leg.book or SEARCH_BOOK, chance=option.cautious, chance_source="books",
        label=leg.label, push=option.push)


def _controls(legs: int, low: float, high: float, tickets: int) -> builder.Controls:
    """The builder's controls for one ticket size: the odds window [low, high] as its ticket-odds band (no target,
    so no closeness term), one leg per game, no Whelan cap, no caution (the chances are already cautious), any board
    market and any price age. _search runs it with builder.RETURN_ONLY, so the builder ranks by the cautious
    expected return alone."""
    # Every price is above 1, so a window starting below 1 starts at 1 for the builder
    return builder.Controls(legs=legs, legs_max=legs, odds_min=max(low, 1.0), odds_max=high, caution_pp=0.0,
                            whelan=False,
                            max_per_game=1, max_per_team=builder.MAX_LEGS, markets=set(builder.MARKETS),
                            max_price_age_s=builder.MAX_AGE_S, tickets=tickets)


def _search(pool: Sequence[_Option], target: float, low: float, high: float, max_legs: int,
            accept: Callable[[float], bool], now: datetime.datetime) -> Tuple[Dict[str, _Found], int]:
    """The best ticket for each slot, and how many combinations landed in the odds window: the Builder's search
    (engine/builder.py), one ticket size at a time, over this pool.

    The builder ranks tickets by their cautious expected return alone (builder.RETURN_ONLY; here the cautious
    chance is this module's p_c) and hands back its best tickets (``Result.ranked``, up to builder.HEAP_KEEP of
    them: every combination in the window when there are fewer); each slot keeps the best of those the tier
    accepts by this module's rank (cautious expected return, then cautious chance).
    """
    by_key = {}
    candidates = []
    if low > builder.MAX_TARGET_ODDS:
        return {}, 0  # past the builder's odds: nothing on a board gets there
    high = min(high, builder.MAX_TARGET_ODDS)
    for option in pool:
        if option.leg.market not in builder.MARKETS:
            continue  # a market the builder doesn't know (a player prop) can't be searched
        candidate = _candidate(option)
        by_key[candidate.key] = option
        candidates.append(candidate)
    best: Dict[str, _Found] = {}
    in_window = 0
    if not candidates:
        return best, in_window

    for size in range(1, max_legs + 1):
        # One ticket asked for: the builder searches once, so its count of combinations is exact
        result = builder.build(candidates, _controls(size, low, high, 1), now, weights=builder.RETURN_ONLY)
        rejected = result.stats.get("rejected") or {}
        in_window += (int(result.stats.get("combos") or 0) - int(rejected.get("odds_low") or 0)
                      - int(rejected.get("odds_high") or 0))
        for _, keys in result.ranked:
            chosen = tuple(by_key[key] for key in keys)
            if not accept(math.prod(option.brain for option in chosen)):
                continue
            odds = math.prod(option.price for option in chosen)
            rank = _rank(chosen, odds, target)
            slot = _slot(len(chosen))
            if slot not in best or rank > best[slot].rank:
                ordered = tuple(sorted(chosen, key=lambda option: (option.price, option.leg.key)))
                best[slot] = _Found(ordered, odds, rank)
    return best, in_window


# --- the tickets --------------------------------------------------------------------------------

def _cautious_expected(options: Sequence[_Option], stake: Decimal) -> Decimal:
    """Expected profit in money at the cautious chances (a pushed leg drops out), exact until rounded."""
    with localcontext() as context:
        context.prec = PRECISION
        factor = math.prod((Decimal(repr(option.cautious)) * option.leg.price + Decimal(repr(option.push))
                            for option in options), start=Decimal(1))
        return stake * factor - stake


def _has_edge(expected: Decimal) -> bool:
    """A positive cautious expected result, counted in whole cents (no "+$0.00 edge")."""
    return expected.quantize(CENTS, rounding=ROUND_HALF_UP) > 0


def _ticket(found: _Found, rank: int, stake: Decimal, target: Decimal,
            now: datetime.datetime) -> Dict[str, Any]:
    """A priced ticket (price_ticket's own shape) plus what the builder knows about it."""
    legs = [option.leg for option in found.options]
    n = len(legs)
    priced = price_ticket(legs, "single" if n == 1 else "multi", stake, now=now)
    expected = _cautious_expected(found.options, stake)
    chance, brain = found.cautious_chance, found.brain_chance
    tier = _tier_of(brain)
    with localcontext() as context:
        context.prec = PRECISION
        payout = stake * math.prod((leg.price for leg in legs), start=Decimal(1))
    return {
        **priced,
        "rank": rank,
        "slot": _slot(n),
        "name": found.name,
        "cautious": {"chance": round(chance, 4), "one_in": one_in(chance), "label": chance_label(chance),
                     "ev": round(found.cautious_return - 1, 4), "expected": _money(expected)},
        "edge": _has_edge(expected),
        # The tier follows the AI's chance (the market's where the AI has none) and always shows the loss
        # chance by the same view; a push gives the stake back, so chance + loses falls short of 1 by the pushes
        "tier": {"key": tier.key if tier else None, "title": tier.title if tier else None,
                 "chance": round(brain, 4), "loses": round(found.brain_loses, 4)},
        "target_gap": _money(payout - target),  # what the ticket pays beyond (or short of) the target
    }


# --- what to say --------------------------------------------------------------------------------

def _percent(p: float) -> str:
    """A chance in words: whole percents from 10%, one decimal from 1%, and two significant figures below
    that, so a 1-in-10,000 chance reads 0.010% rather than 0.0%."""
    if p >= 0.1:
        return f"{p:.0%}"
    if p >= 0.01:
        return f"{p:.1%}"
    if p <= 0:
        return "0%"
    places = 1 - math.floor(math.log10(p * 100))  # 0.96%, 0.010%, 0.0095%
    return f"{p * 100:.{places}f}%"


def _chance_words(p: float) -> str:
    return f"{_percent(p)} (1 in {one_in(p):,})"


def _ceiling_note(stake: Decimal, target: Decimal, multiple: float, legs: int) -> str:
    text = (f"To turn {_dollars(stake)} into {_dollars(target)} you need odds of {multiple:,.2f}. "
            f"With no real edge nothing beats {_chance_words(1 / multiple)}; after a bookmaker's cut of "
            f"about {CUT:.1%} a leg it is about {_chance_words(no_edge_chance(multiple, 1))} with one leg")
    if legs > 1:
        text += f" and {_chance_words(no_edge_chance(multiple, legs))} with {_LEG_WORDS[legs]}"
    return text + "."


def _short_odds_note(multiple: float) -> str:
    loses = one_in(1 - no_edge_chance(multiple, 1))
    wins = max(1, round(1 / (multiple - 1)))
    return (f"Odds under {SHORT_ODDS}: a single on a heavy favourite or no bet at all. At {multiple:.2f} it "
            f"still loses about 1 in {loses}, and one loss costs as much as about {wins} wins.")


def _tier_clash(tier: Tier, multiple: float, legs: int) -> Optional[str]:
    """Why, with no edge, the target's odds sit outside the tier's chance range; None when they can meet."""
    ceiling = 1 / multiple
    if ceiling < tier.p_low:
        return (f"A {tier.title} ticket needs a {tier.p_low:.0%}+ chance, but at {multiple:,.2f} odds "
                f"the honest ceiling is {_chance_words(ceiling)}")
    if no_edge_chance(multiple, legs) >= tier.p_high:
        return (f"{tier.title} tickets win less than {tier.p_high:.0%} of the time, but {multiple:.2f} "
                f"odds sit near {_percent(no_edge_chance(multiple, 1))}")
    return None


def _tier_notes(tier: Optional[Tier], multiple: float, legs: int, found: Sequence[_Found]) -> List[str]:
    """What the tier means here, worded so it never contradicts the tickets returned beside it."""
    if tier is None:
        return []
    notes = []
    clash = _tier_clash(tier, multiple, legs)
    if clash and found:
        # A ticket made the tier anyway: only the AI's own chance put it there, and the market disagrees
        notes.append(f"{clash}: only the AI's chance, which the market doesn't share, puts these tickets "
                     "in the tier, and it isn't proven.")
    elif clash:
        notes.append(f"{clash}: this target and tier don't fit together.")
    if tier.key == "safe":
        cap = 1 - tier.p_low
        note = (f"Safe is a tier name, not a promise: by the AI's chance these tickets still lose up to "
                f"{cap:.0%} of the time")
        # The market's view is the least hopeful one; quote it when it reads worse than the tier's cap
        worst = max(found, key=lambda ticket: ticket.market_loses, default=None)
        if worst is not None and round(worst.market_loses, 2) > round(cap, 2):
            note += f"; by the market's chance the {worst.name} loses about {_percent(worst.market_loses)}"
        notes.append(note + ".")
    return notes


def _dropped_note(dropped: Dict[str, int]) -> Optional[str]:
    parts = [f"{dropped['started']} already started" if dropped["started"] else "",
             (f"{dropped['outside_range']} with a market chance outside {MARKET_LOW:.0%}-{MARKET_HIGH:.0%}"
              if dropped["outside_range"] else ""),
             f"{dropped['no_chance']} with no chance yet" if dropped["no_chance"] else ""]
    parts = [part for part in parts if part]
    if not parts:
        return None
    total = sum(dropped.values())
    return f"Left out {total} pick{'s' if total != 1 else ''}: {', '.join(parts)}."


def _empty_note(tier: Optional[Tier], in_window: int, low: float, high: float, searched: Optional[int]) -> str:
    """Why no ticket came back. ``searched`` is the pool size when the pool left out legs that could fit:
    then the note speaks only of the picks searched, never of the whole slate."""
    picks = ("the picks on offer" if searched is None
             else f"the {searched} best picks searched (up to {PER_GAME} a game)")
    if tier is not None and in_window:
        among = "" if searched is None else f" of {picks}"
        return (f"{in_window:,} combination{'s' if in_window != 1 else ''}{among} land near the target, but none "
                f"has a {tier.title} chance ({_band(tier)}).")
    return f"No combination of {picks} lands between {low:,.2f} and {high:,.2f} odds."


def _stand_in_notes(tickets: Sequence[Dict[str, Any]]) -> List[str]:
    labels = list(dict.fromkeys(leg["label"] for ticket in tickets for leg in ticket["legs"]
                                if leg["market_prob"] is None))
    return [f"No margin-free market chance for {label}: its price less a {CUT:.1%} cut stands in for it."
            for label in labels]


def _verdict(found: Optional[_Found], stake: Decimal, multiple: float) -> Tuple[str, bool, Optional[str]]:
    """The one-line verdict, whether the best ticket has a cautious edge, and the expected cost if not."""
    if found is None:
        return (f"Nothing on offer fits a {multiple:,.2f} target right now. With no edge nothing beats "
                f"1 in {one_in(1 / multiple):,}.", False, None)
    expected = _cautious_expected(found.options, stake)
    if _has_edge(expected):
        return (f"Edge on paper: at the cautious chance the top ticket's expected result is "
                f"{_dollars(expected, sign=True)}. The AI's edge isn't proven, and the ticket still loses "
                f"about {_percent(found.cautious_loses)} of the time.", True, None)
    return f"No edge found: entertainment with an expected cost of {_dollars(-expected)}.", False, _money(-expected)


# --- the builder --------------------------------------------------------------------------------

def _frame(stake: Decimal, target: Decimal, multiple: Decimal, legs: int, tier: Optional[Tier]) -> Dict[str, Any]:
    """The numbers that hold whatever legs are on offer: the target odds and the honest ceiling."""
    m = float(multiple)
    ceiling = min(1.0, 1 / m)
    return {
        "stake": _money(stake),
        "target_return": _money(target),
        "target_profit": _money(target - stake),
        "multiple": round(m, 2),
        "target_odds": {"decimal": round(m, 2),
                        "american": format_american(decimal_to_american(m)) if m > 1 else None,
                        "low": round(float(multiple * WINDOW_LOW), 2),
                        "high": round(float(multiple * WINDOW_HIGH), 2)},
        "ceiling_chance": round(ceiling, 4),
        "ceiling_one_in": one_in(ceiling),
        "no_edge": [{"legs": n, "chance": round(no_edge_chance(m, n), 4), "one_in": one_in(no_edge_chance(m, n))}
                    for n in range(1, legs + 1)],
        "lottery": multiple > LOTTERY_ODDS,
        "short_odds": multiple < SHORT_ODDS,
        "max_legs": legs,
        "tier": tier.key if tier else None,
    }


def build_for_target(candidates: Sequence[TicketLeg], stake: Amount, target_return: Amount,
                     max_legs: int = 3, tier: Union[None, str, Tier] = None,
                     now: Optional[datetime.datetime] = None) -> Dict[str, Any]:
    """Up to three ranked tickets that turn ``stake`` into about ``target_return``, ceiling first.

    ``target_return`` is the money back, stake included ($10 to get $100 back
    needs odds of 10). The search tries one to min(max_legs, 4) legs from
    different games whose combined odds land within -10% / +15% of that, keeps
    the best single, double and three- or four-leg ticket by cautious expected
    return, and prices each with price_ticket at exactly ``stake``. ``tier``
    ("safe", "risky" or "crazy") keeps only tickets whose AI chance is in that
    tier's range. Raises ValueError for a stake or target that isn't whole
    cents from $0.01 to $1,000,000,000, a target more than 10,000,000 times the
    stake, a ``max_legs`` below 1 or an unknown tier.
    """
    stake, target = _amount(stake, "stake"), _amount(target_return, "target")
    if max_legs < 1:
        raise ValueError(f"A ticket needs at least one leg, got max_legs={max_legs}")
    wanted = _tier(tier)
    now = _now(now)
    multiple = target / stake
    if multiple > MAX_MULTIPLE:
        raise ValueError(f"The target can be at most {MAX_MULTIPLE:,} times the stake, got {multiple:,.0f} times")
    m = float(multiple)
    legs = 1 if multiple < SHORT_ODDS else min(max_legs, SEARCH_MAX_LEGS)
    frame = _frame(stake, target, multiple, legs, wanted)

    if multiple <= 1:
        verdict = (f"No bet needed: {_dollars(target)} back from a {_dollars(stake)} stake is no more than "
                   "the stake itself.")
        return {**frame, "tickets": [], "edge": False, "expected_cost": None, "verdict": verdict,
                "notes": [], "searched": {"candidates": len(candidates), "usable": 0, "pool": 0, "combinations": 0}}

    options, dropped = _usable(candidates, now)
    low, high = float(multiple * WINDOW_LOW), float(multiple * WINDOW_HIGH)
    pool, cut = _pool(options, low * (1 - _EPS), high * (1 + _EPS), legs)
    accept = (lambda p: wanted.p_low <= p < wanted.p_high) if wanted else (lambda p: True)
    best, in_window = _search(pool, m, low, high, legs, accept, now)
    ranked = sorted(best.values(), key=lambda found: found.rank, reverse=True)
    tickets = [_ticket(found, i + 1, stake, target, now) for i, found in enumerate(ranked)]
    verdict, edge, cost = _verdict(ranked[0] if ranked else None, stake, m)

    notes = [_ceiling_note(stake, target, m, legs)]
    if multiple > LOTTERY_ODDS:
        notes.append(f"Lottery-like: {m:,.0f} times the stake is at best a 1 in {one_in(1 / m):,} shot, "
                     "whichever legs are picked.")
    if multiple < SHORT_ODDS:
        notes.append(_short_odds_note(m))
    elif max_legs > SEARCH_MAX_LEGS:
        notes.append(f"Searched up to {SEARCH_MAX_LEGS} legs: each extra leg adds the bookmaker's cut again.")
    notes += _tier_notes(wanted, m, legs, ranked)
    dropped_note = _dropped_note(dropped)
    if dropped_note:
        notes.append(dropped_note)
    if not tickets:
        notes.append(_empty_note(wanted, in_window, low, high, len(pool) if cut else None))
    notes += _stand_in_notes(tickets)

    return {**frame, "tickets": tickets, "edge": edge, "expected_cost": cost, "verdict": verdict, "notes": notes,
            "searched": {"candidates": len(candidates), "usable": len(options), "pool": len(pool),
                         "combinations": in_window}}
