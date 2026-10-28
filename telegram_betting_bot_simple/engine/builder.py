"""The Builder: tickets from every control a person can set (v3 Stage 3).

Safe, Risky and Crazy are presets (engine/presets.py) that fill these controls; any edit makes the run
"Custom". The builder fetches nothing: the caller hands it ``Candidate`` legs assembled from the slate, each
league's cached board, the markets brain's blend and its flags, and it returns ranked tickets with the
reasons, the near misses and, when nothing fits, which rules ruled everything out.

The numbers are the brain's, never the AI's:

- a leg's chance is the Stage 2 blend (engine/consensus.py), else the price books' margin-free chance, else
  the soccer goals model's (engine/soccer.py, ``chance_source`` "model"); its cautious chance is that minus
  ``caution_pp`` points (2 by default), the old tiers' sigma;
- prices come only from books a person can bet at: a SHARP_BOOKS book or an exchange is never used, whatever
  the caller passes;
- a ticket takes one leg per game (same-game legs are linked, and no book's same-game price is loaded), so its
  legs are independent and its joint chance is their product.

The search, for one ticket shape (legs 1-12 or Auto up to ``legs_max``):

1. Every candidate goes through the controls' rules. A leg out of scope (league, market, time window, kick-off
   window, started) is only counted; a leg that fails a quality rule (books, price age, odds range, chance,
   edge, the gap to the books in Safe, disagreement, thin market, low confidence, fair-odds pricing) is a near
   miss naming the rule. A leg priced at the goals model's fair odds has no edge (no bookmaker's price to
   measure one against), so a minimum edge leaves it out.
2. Tickets of up to 4 legs are searched exhaustively over K=40 legs, depth first over legs sorted by price,
   pruned on the odds window and the joint chance floor. With 40 legs or fewer this is every combination: tests
   check it against brute force. With more, the pool is the best 40 by score, or with a target half the best by
   value whatever their price (every ticket lands near the target's odds, so the joint chance's taste for short
   prices cancels out) and half the legs priced nearest a per-leg share of it, so the pool can reach the target;
   each game gives at most 3 before any gives a 4th. A search of a cut pool that finds nothing tries once more
   over the legs priced nearest that share.
3. Tickets of 5 to 12 legs come from a beam search (width 200) over 60 legs that may join a multi, chosen the
   same way. The beam keeps its partial tickets spread across 20 buckets of the odds they may still have at that
   depth (with a target, ranked whatever their price), so a long ticket that has to land near a target can
   still find its way there, and drops any partial ticket that can no longer reach the window.
4. A ticket's score is, with the preset's weights (presets.WEIGHTS):
       - w_target * distance of log(odds) from log(target), as a share of the tolerance on that side
       + w_return * log(cautious expected return)       (each leg: cautious chance * price + push; a leg
                                                          priced at the goals model's fair odds is valued with a
                                                          typical 5% bookmaker's margin off, MODEL_MARGIN)
       + w_chance * log(joint chance)
       - 0.001 per leg
5. Up to ``tickets`` tickets that share no leg are picked best first (searching again without the legs already
   used when the best tickets all share one), and each is priced exactly by tickets.price_ticket at the
   person's stake (10 by default).

Each ticket also carries its chance of reaching the target (exact, over every win/push outcome of its legs: a
pushed leg drops out at 1): ``reach_target.chance`` is the chance it pays at least the low end of the target
window (``reach_target.odds``), ``at_target`` the chance it pays the target itself or more, and with no target
both are the joint chance. ``joint`` holds the chance every leg wins, the chance it wins money (``profit``) and
loses the stake (``lose``), and its fair odds: each leg at (1 - push) / chance, where its expected result is nil.
Each leg has up to 3 swaps that keep every rule (and use no leg of another ticket), and the ticket has notes
such as "Priced at fair odds; check your bookmaker's price." for legs the goals model prices. Crazy, and any
ticket under a 1 in 4 chance of winning money, says "expect to lose most of these".

The preset's band (its ticket-odds window ``odds_min``-``odds_max`` and its window on the chance the ticket wins
money ``joint_min``-``joint_max``) describes a risk level when no target is set; a ``target_odds`` replaces that
band with target +/- tolerance, and every leg rule still applies. While a run is exactly a preset, the old
tier's rules on a ticket's value apply as well (presets.VALUE_RULES: the expected-result floor, the cap on the
books' cut, Safe's 3-leg floor and its gap to the books). The Whelan leg cap (engine/bets.py) limits Auto's
choice of legs; a leg count the person sets explicitly lifts it (price_ticket still warns).

When nothing fits, ``blocked`` names the rules that ruled everything out (a branch cut short by a cap or the
joint floor counts against it) and what to relax for that kind of block: more legs or a lower target when the
target is out of reach, fewer legs or a higher one when even the shortest legs pay too much, the rule itself
when one ruled out every combination that could land, a wider tolerance when none lands. Near misses are the
legs closest to passing (at most 2 a game, up to 12), with leagues that have no prices loaded on top.

Closest effort (``best_effort``, on by default; v3 "best effort"). The Builder always builds: asked for something it
can't meet exactly, it returns the closest tickets it can make, each saying how close it gets.

- What a run aims for: with ``target_odds`` only, odds inside the window as above; with ``target_chance`` only (the
  chance the whole ticket wins: every leg winning), the highest odds whose chance is at least the target (with the
  person's rules, never the number of legs, loosened round by round until a ticket reaches it), else the highest
  chance there is (with the rules those rounds loosened); with both, tickets that meet both (a normal run), else up to
  three labelled options: ``your_odds`` (inside the odds window, the highest chance), ``your_chance`` (the chance
  target at the highest odds, its rules loosened as for a target chance alone, else the highest chance there is)
  and ``closest`` (the smallest d = |ln(odds / target)| / ln(1 + tolerance) + |chance - target chance| / 0.10 of
  any ticket the rules never loosened allow: "Best balance"). Each option
  loosens the person's rules only as far as it needs. Options that pick the same ticket are merged under the
  first label. A target replaces the preset's own odds and chance band.
- When the person's rules give fewer tickets than asked (or none), the rules are loosened in rounds, one step of
  the most binding rule at a time (from the run's own counts: what each rule removed, why a round's bounds failed,
  and which rule ruled out the combinations a search landed in the window, which binds before more legs), in
  this order: the
  per-leg chance floors (5 points a step, down to none), the leg-odds band (25% wider a step, twice, then
  dropped), the minimum edge, the joint chance band (no target), the value rules (the Whelan cap, Safe's 3-leg
  rule, the expected-result floor, the cap on the bookmaker's cut), the odds window (tolerance 10, 20, 35, 60%;
  a preset's band 10, 20, 35, 60% wider) and last, for an exact number of legs, 1 then 2 legs either way. The
  rounds stop at the first one that finds tickets; a round whose bounds show it can't land a ticket (the longest
  legs can't reach the window, the likeliest can't reach the chance floor) is recorded without a search. If
  nothing is found at all while some leg passes the rules never loosened, a last round drops every rule closest
  effort may loosen and the window and takes the nearest odds there are: at the number of legs asked first, then 1
  and 2 either way, then any number up to it, so a run never ends empty while a leg is eligible.
- The work is bounded (RELAX_BUDGET combinations, deterministic, unlike a clock), shared out so one objective can't
  starve the next (with both targets: a third for tickets that meet both, then half of what is left for each
  option): once what is left is less than the last try cost, the next round that would search isn't searched and
  the rounds end. The tries that must run still do (each option's try at the person's own rules, the target chance's
  search at the rules its rounds reached, the last round), and once the budget is spent the searches that top their
  tickets up are narrow ones (a smaller pool and beam).
- Never loosened: started games, the price-age limit, price books only, one leg per game, the team and league
  caps, the markets chosen, the leagues, time and kick-off windows, the excluded flags (disagreement, thin, low
  confidence, fair odds) and Safe's limit on the gap to the books.
- Every ticket carries ``fit`` (its odds and chance against the targets; None without a target) and ``relaxed``
  ([{rule, from, to, words}]: only the rules this ticket needed loosened, at the tightest step that admits it,
  e.g. "per-leg chance lowered from 72% to 52%"; its swaps stay within those steps); the result carries
  ``options`` and, in ``stats["relaxation"]``, every step taken (Best balance's search too, objective
  "closest"). ``best_effort`` False (strict) runs exactly as before closest effort existed.

``top_picks`` ranks single legs by the brain's chance (the likeliest winners, one a game).

This module imports engine.bets and engine.tickets inside functions only, so engine/bets.py and
engine/target.py may import it at module level.
"""

from __future__ import annotations

import dataclasses
import datetime
import heapq
import json
import math
import re
import time
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .consensus import SHARP_BOOKS

# --- vocabulary ---------------------------------------------------------------------------------------

# Odds API market keys. h2h, spreads and totals come from the boards; the soccer ones from engine/soccer.py
MARKETS: Tuple[str, ...] = ("h2h", "spreads", "totals", "team_totals", "btts", "draw_no_bet", "double_chance")
MARKET_TITLES: Dict[str, str] = {
    "h2h": "Moneyline", "spreads": "Spread / handicap", "totals": "Totals", "team_totals": "Team totals",
    "btts": "Both teams to score", "draw_no_bet": "Draw no bet", "double_chance": "Double chance",
}
DEFAULT_MARKETS: Tuple[str, ...] = ("h2h", "spreads", "totals")
# Derived by the goals model from a soccer game's 1X2 and over/under 2.5 (no other sport has them yet)
SOCCER_MARKETS: Tuple[str, ...] = ("team_totals", "btts", "draw_no_bet", "double_chance")
UNAVAILABLE: Dict[str, Tuple[str, str]] = {
    "player_props": ("Player props", "Props, corners and cards arrive in Stage 6"),
    "corners": ("Corners", "Props, corners and cards arrive in Stage 6"),
    "cards": ("Cards", "Props, corners and cards arrive in Stage 6"),
}
FLAGS: Tuple[str, ...] = ("disagree", "thin", "low_conf", "stale_price", "live", "model_price")
# "adjusted": the books' margin-free chance of a totals leg moved by a forecast or roof change since the odds
# (v3 Stage 4, data/context.py): no longer the books' own number
CHANCE_SOURCES: Tuple[str, ...] = ("blend", "books", "model", "adjusted")

MAX_LEGS = 12  # tickets.MAX_LEGS
EXHAUSTIVE_MAX = 4  # up to this many legs every combination of the pool is tried
POOL_K = 40  # legs in the exhaustive search: C(40, 4) is about 91,000 four-leg tickets
BEAM_POOL = 60  # legs the beam may use
BEAM_WIDTH = 200
BEAM_BUCKETS = 20  # the beam keeps its partial tickets spread across this many odds buckets
PER_GAME_POOL = 3  # best legs a game may put in a pool
HEAP_KEEP = 400  # best tickets kept for picking diverse ones
# Closest effort's top-ups once its budget is spent search a narrower pool and beam (about a tenth of the work)
NARROW_POOL = 20
NARROW_BEAM_POOL = 30
NARROW_WIDTH = 50
MAX_TICKETS = 5
MAX_SWAPS = 3
MAX_NEAR_MISSES = 12  # legs; leagues with no prices loaded come on top
MAX_MISSES_PER_GAME = 2
MAX_IDEAS = 5
LEG_PENALTY = 0.001
# A leg priced at the goals model's fair odds has no bookmaker's cut in its price, so it would outrank every leg a
# person can actually bet at. For ranking only (its shown price stays the fair odds) it is valued as a bookmaker would
# price it, with a typical margin of 5% taken off its price.
MODEL_MARGIN = 0.05
DEFAULT_STAKE = Decimal("10")
MAX_TARGET_ODDS = 1_000_000.0
MAX_AGE_S = 7 * 24 * 3600
_EPS = 1e-9
_TINY = 1e-9

# Closest effort (see the module docstring)
OPTION_KEYS: Tuple[str, ...] = ("your_odds", "your_chance", "closest")
OPTION_LABELS: Dict[str, str] = {"your_odds": "Likeliest at your odds", "your_chance": "Closest to your chance",
                                 "closest": "Best balance"}
RELAX_ORDER: Tuple[str, ...] = ("floors", "leg_odds", "min_edge", "joint", "value", "window", "legs")
# The rule groups alone (1-5): what a search for the target chance loosens before it settles for the highest chance
RULE_GROUPS: Tuple[str, ...] = RELAX_ORDER[:5]
FLOOR_STEP = 0.05  # the per-leg chance floors drop 5 points a round
LEG_BAND_STEP = 1.25  # the leg-odds band widens 25% a round ...
LEG_BAND_STEPS = 2  # ... twice, then it is dropped
TOLERANCE_STEPS: Tuple[float, ...] = (10.0, 20.0, 35.0, 60.0)  # the odds window around a target, in %
BAND_STEPS: Tuple[float, ...] = (1.10, 1.20, 1.35, 1.60)  # a preset's ticket-odds band, widened alike
LEG_STEPS = 2  # an exact number of legs: 1, then 2 either way
CHANCE_UNIT = 0.10  # closest: 10 points of chance weigh as much as the tolerance on the odds
MIN_CLOSE_TOL = 0.01  # closest's odds term with a 0% tolerance: as if it were 1%
TIE = 0.001  # breaks ties in the odds and chance objectives
# The work closest effort may do in all, in combinations tried (deterministic, unlike a clock): a round that would
# search isn't searched when what is left is less than the last try cost; the tries that must run still do (narrow
# top-ups once it's spent)
RELAX_BUDGET = 300_000
# Closest effort's searches of 1 to 3 legs see a wider pool than POOL_K: as many legs as keep C(K, n) within about
# C(40, 4), the exhaustive search's own work at 4 legs (447 legs for pairs, 85 for trebles); the run at the person's
# own rules keeps POOL_K
WIDE_COMBOS = 100_000
MAX_ROUNDS = 64
VALUE_RULE_KEYS: Dict[str, str] = {"three_leg": "three_leg_min", "min_ev": "min_ev", "book_cut": "max_book_cut"}
# Why a round found too little (_feasible's reasons), by what can mend it: more legs, and longer ones (a lower chance
# floor or a higher leg-odds cap: the legs it adds are less likely, so longer), or shorter, likelier ones (a lower
# leg-odds minimum)
MORE_LEGS: Set[str] = {"legs", "games", "reach", "value"}
SHORTER_LEGS: Set[str] = {"legs", "games", "floor", "chance", "joint_min", "value"}
# Which way each stepped rule loosens (a ticket's relaxed "to"): +1 higher is looser, -1 lower is looser (odds_band:
# wider); the rest are only ever dropped
LOOSER: Dict[str, int] = {"min_chance": -1, "multi_leg_min": -1, "leg_odds_min": -1, "leg_odds_max": 1,
                          "tolerance_pct": 1, "odds_band": 0, "legs": 0}
# Ticket rules that rule out combinations already inside the window: when a search shows them, they bind before the
# rules that only let more legs in
INSIDE_RULES: Tuple[str, ...] = ("joint_min", "joint_max", "whelan", "three_leg", "min_ev", "book_cut")

_LEAGUE = re.compile(r"[a-z0-9_]{1,64}")
_HHMM = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")


# --- small helpers ------------------------------------------------------------------------------------

def _parse_time(value: Any) -> Optional[datetime.datetime]:
    """An aware datetime from an ISO string or a datetime; None when it can't be read."""
    if isinstance(value, datetime.datetime):
        moment = value
    elif isinstance(value, str) and value.strip():
        try:
            moment = datetime.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=datetime.timezone.utc)


def _iso(moment: datetime.datetime) -> str:
    return moment.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now(now: Optional[datetime.datetime]) -> datetime.datetime:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=datetime.timezone.utc)


def _decimal(value: Any, name: str) -> Decimal:
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not number.is_finite():
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def _number(value: Any, name: str, low: float, high: float, optional: bool = True) -> Optional[float]:
    if value is None or value == "":
        if optional:
            return None
        raise ValueError(f"{name} is required")
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}, got {value!r}")
    return number


def _integer(value: Any, name: str, low: int, high: int, optional: bool = True) -> Optional[int]:
    if value is None or value == "":
        if optional:
            return None
        raise ValueError(f"{name} is required")
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value.strip())
    if not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be a whole number from {low} to {high}, got {value!r}")
    return value


def _flag(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be true or false, got {value!r}")


def age_text(seconds: Union[int, float]) -> str:
    """A data age in words: "under 1 min", "40 min", "3 h", "3 h 40 min"."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return "under 1 min"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours} h" if rest == 0 else f"{hours} h {rest} min"


def _a_pct(p: float) -> str:
    """A chance with its article: "an 80%", "an 11%", "a 70%"."""
    text = _pct(p)
    return ("an " if re.match(r"8|1[18](?![0-9])", text) else "a ") + text


def _pct(p: float) -> str:
    """A chance in words: whole percents, one decimal below 10%, two significant figures below 1%."""
    if p >= 0.1 or p <= 0:
        return f"{p:.0%}"
    if p >= 0.01:
        return f"{p:.1%}"
    places = 1 - math.floor(math.log10(p * 100))
    return f"{p * 100:.{places}f}%"


def _odds_text(odds: float) -> str:
    return f"{odds:,.2f}" if odds < 1000 else f"{odds:,.0f}"


def target_odds_for(stake: Any, returns: Any) -> float:
    """The target odds for "stake X, win about Y back" (Y is the money back, stake included)."""
    stake, returns = _decimal(stake, "The stake"), _decimal(returns, "The amount back")
    if stake <= 0 or returns <= 0:
        raise ValueError("The stake and the amount back must be above zero")
    return float(returns / stake)


def leg_key(ref: Mapping[str, Any]) -> str:
    """The pricing engine's stable id for a pick ("basketball_nba:<id>:spreads:home:-3.5"), as dashboard/lab.py
    builds it."""
    key = f"{ref['league']}:{ref['id']}:{ref['market']}:{ref['side']}"
    point = ref.get("point")
    return key if point is None else f"{key}:{float(point):g}"


def _book_key(book: Any) -> Optional[str]:
    try:
        from ..data.odds_service import book_key  # the data layer imports the engine: import it here
    except Exception:  # pragma: no cover - the data layer is always there in the app
        return str(book).strip().lower() or None if book else None
    return book_key(book)


def _price_book(book: Any) -> bool:
    """A book a person can bet at: never a SHARP_BOOKS book or an exchange (data/odds_service.py's BookRoles)."""
    key = _book_key(book)
    if not key or key in SHARP_BOOKS:
        return False
    from .bets import price_book
    return price_book(book)


# --- the controls ------------------------------------------------------------------------------------

@dataclass
class Weights:
    """How a preset trades closeness to the target, the cautious expected return and the joint chance."""
    target: float = 0.15
    ret: float = 1.0
    chance: float = 0.2


# Ranking by the cautious expected return alone (engine/target.py and engine/bets.py: their own documented rank)
RETURN_ONLY = Weights(target=0.0, ret=1.0, chance=0.0)


@dataclass
class Controls:
    """Every setting of a build. Chances and edges are shares (0.55 for 55%); ``caution_pp`` is in points.

    The fields after ``caution_pp`` go beyond the stage contract: they carry the presets' old tier windows
    and rules (``legs_max`` for Auto, the band ``odds_min``-``odds_max`` / ``joint_min``-``joint_max``, the
    multi-leg chance floor, the Whelan cap) and the Safe-only exclusions, so a preset fills every control. The
    joint band is on the chance the ticket wins money (a pushed leg drops out at 1). The old tiers' rules on a
    ticket's value (presets.VALUE_RULES: the expected-value floor, the cap on the books' cut, Safe's 3-leg floor
    and its limit on the gap to the books) apply while a run is exactly a preset.

    ``target_chance`` (0.01-0.99) is the chance the whole ticket wins (every leg) that the person wants; with or
    without ``target_odds`` it changes what a run aims for (see the module docstring), and like a target odds it
    replaces the preset's band. ``best_effort`` (closest effort, on by default) returns the closest tickets when the
    rules give fewer than asked; False is strict: only exactly what was asked, as before closest effort. It doesn't
    change the risk of a ticket that meets the rules, so it keeps the preset's name (presets.SCOPE_FIELDS).
    """

    leagues: Optional[List[str]] = None  # None = every league
    window: Optional[Dict[str, str]] = None  # {"from": iso, "to": iso}; None = any start
    kickoff: Optional[Dict[str, Optional[str]]] = None  # {"earliest": "HH:MM", "latest": "HH:MM"} local time
    exclude_started: bool = True
    legs: Optional[int] = None  # None = Auto (1 to legs_max)
    target_odds: Optional[float] = None
    target_chance: Optional[float] = None  # the chance the whole ticket wins, e.g. 0.8
    tolerance_pct: float = 10.0
    best_effort: bool = True  # closest effort: the closest tickets when nothing fits exactly; False = strict
    leg_odds_min: Optional[float] = None
    leg_odds_max: Optional[float] = None
    min_chance: Optional[float] = None
    min_edge: Optional[float] = None  # chance - (1 - push) / price, e.g. -0.02
    max_per_game: int = 1
    max_per_team: int = 1
    max_per_league: Optional[int] = None
    markets: Set[str] = field(default_factory=lambda: set(DEFAULT_MARKETS))
    avoid_same_game: bool = True
    max_price_age_s: int = 3 * 3600
    exclude_disagree: bool = False
    exclude_thin: bool = False
    books: Optional[Set[str]] = None  # allowed books (keys or titles); None = every price book
    tickets: int = 3
    preset: Optional[str] = None  # the preset these controls started from ("safe", "risky", "crazy")
    caution_pp: float = 2.0
    legs_max: int = 4
    odds_min: Optional[float] = None
    odds_max: Optional[float] = None
    joint_min: Optional[float] = None
    joint_max: Optional[float] = None  # exclusive
    multi_leg_min: Optional[float] = None  # each leg of a multi needs at least this chance
    whelan: bool = True
    whelan_slack: int = 0
    exclude_low_conf: bool = False
    exclude_model_price: bool = False
    prefer_correlation: bool = False  # ideas only: same-game pairs are suggested, never built
    tz: Optional[str] = None  # IANA zone for the kick-off window; None = UTC

    def __post_init__(self) -> None:
        self._check()

    # validation ----------------------------------------------------------------------------------
    def _check(self) -> None:
        if self.leagues is not None:
            if isinstance(self.leagues, str) or not isinstance(self.leagues, Iterable):
                raise ValueError("leagues must be a list of league keys, or null for all")
            leagues = []
            for league in self.leagues:
                if not isinstance(league, str) or not _LEAGUE.fullmatch(league.strip().lower()):
                    raise ValueError(f"Unknown league key {league!r}")
                if league.strip().lower() not in leagues:
                    leagues.append(league.strip().lower())
            if len(leagues) > 80:
                raise ValueError("At most 80 leagues")
            self.leagues = leagues or None
        self.window = self._window(self.window)
        self.kickoff = self._kickoff(self.kickoff)
        self.exclude_started = _flag(self.exclude_started, "exclude_started")
        self.legs = _integer(self.legs, "legs", 1, MAX_LEGS)
        self.target_odds = _number(self.target_odds, "target_odds", 1.01, MAX_TARGET_ODDS)
        self.target_chance = _number(self.target_chance, "target_chance (a share: 0.8 for 80%)", 0.01, 0.99)
        self.tolerance_pct = _number(self.tolerance_pct, "tolerance_pct", 0.0, 90.0, optional=False)
        self.best_effort = _flag(self.best_effort, "best_effort")
        self.leg_odds_min = _number(self.leg_odds_min, "leg_odds_min", 1.0, 1000.0)
        self.leg_odds_max = _number(self.leg_odds_max, "leg_odds_max", 1.01, 1000.0)
        if self.leg_odds_min is not None and self.leg_odds_max is not None and self.leg_odds_min > self.leg_odds_max:
            raise ValueError("leg_odds_min can't be above leg_odds_max")
        self.min_chance = _number(self.min_chance, "min_chance (a share: 0.55 for 55%)", 0.0, 1.0)
        self.min_edge = _number(self.min_edge, "min_edge (a share: -0.02 for -2 points)", -1.0, 1.0)
        self.max_per_game = _integer(self.max_per_game, "max_per_game", 1, MAX_LEGS, optional=False)
        self.max_per_team = _integer(self.max_per_team, "max_per_team", 1, MAX_LEGS, optional=False)
        self.max_per_league = _integer(self.max_per_league, "max_per_league", 1, MAX_LEGS)
        if isinstance(self.markets, str) or not isinstance(self.markets, Iterable):
            raise ValueError("markets must be a list of market keys")
        markets = {str(m).strip().lower() for m in self.markets}
        unknown = sorted(markets - set(MARKETS))
        if unknown:
            reasons = [UNAVAILABLE[m][1] for m in unknown if m in UNAVAILABLE]
            raise ValueError(reasons[0] if reasons
                             else f"Unknown market {unknown[0]!r}; choose from {', '.join(MARKETS)}")
        if not markets:
            raise ValueError("Pick at least one market")
        self.markets = markets
        self.avoid_same_game = _flag(self.avoid_same_game, "avoid_same_game")
        self.max_price_age_s = _integer(self.max_price_age_s, "max_price_age_s", 60, MAX_AGE_S, optional=False)
        self.exclude_disagree = _flag(self.exclude_disagree, "exclude_disagree")
        self.exclude_thin = _flag(self.exclude_thin, "exclude_thin")
        if self.books is not None:
            if isinstance(self.books, str) or not isinstance(self.books, Iterable):
                raise ValueError("books must be a list of bookmakers, or null for all")
            books = {str(b).strip() for b in self.books if str(b).strip()}
            if len(books) > 40 or any(len(b) > 64 for b in books):
                raise ValueError("At most 40 books, each a short name")
            self.books = books or None
        self.tickets = _integer(self.tickets, "tickets", 1, MAX_TICKETS, optional=False)
        if self.preset is not None:
            name = str(self.preset).strip().lower()
            self.preset = None if name in ("", "custom") else name
            if self.preset is not None and self.preset not in ("safe", "risky", "crazy"):
                raise ValueError(f"Unknown preset {self.preset!r}; choose safe, risky or crazy")
        self.caution_pp = _number(self.caution_pp, "caution_pp", 0.0, 20.0, optional=False)
        self.legs_max = _integer(self.legs_max, "legs_max", 1, MAX_LEGS, optional=False)
        self.odds_min = _number(self.odds_min, "odds_min", 1.0, MAX_TARGET_ODDS)
        self.odds_max = _number(self.odds_max, "odds_max", 1.01, MAX_TARGET_ODDS)
        if self.odds_min is not None and self.odds_max is not None and self.odds_min > self.odds_max:
            raise ValueError("odds_min can't be above odds_max")
        self.joint_min = _number(self.joint_min, "joint_min", 0.0, 1.0)
        self.joint_max = _number(self.joint_max, "joint_max", 0.0, 1.01)
        if self.joint_min is not None and self.joint_max is not None and self.joint_min >= self.joint_max:
            raise ValueError("joint_min must be below joint_max")
        self.multi_leg_min = _number(self.multi_leg_min, "multi_leg_min", 0.0, 1.0)
        self.whelan = _flag(self.whelan, "whelan")
        self.whelan_slack = _integer(self.whelan_slack, "whelan_slack", 0, MAX_LEGS, optional=False)
        self.exclude_low_conf = _flag(self.exclude_low_conf, "exclude_low_conf")
        self.exclude_model_price = _flag(self.exclude_model_price, "exclude_model_price")
        self.prefer_correlation = _flag(self.prefer_correlation, "prefer_correlation")
        if self.tz is not None:
            self.tz = str(self.tz).strip() or None
            if self.tz is not None:
                _zone(self.tz)

    @staticmethod
    def _window(window: Any) -> Optional[Dict[str, str]]:
        if window is None:
            return None
        if not isinstance(window, Mapping):
            raise ValueError('window must be {"from": iso time, "to": iso time}')
        out: Dict[str, str] = {}
        for end in ("from", "to"):
            value = window.get(end)
            if value in (None, ""):
                continue
            moment = _parse_time(value)
            if moment is None:
                raise ValueError(f"window {end} must be an ISO time, got {value!r}")
            out[end] = _iso(moment)
        if "from" in out and "to" in out and out["from"] >= out["to"]:
            raise ValueError("The window must end after it starts")
        return out or None

    @staticmethod
    def _kickoff(kickoff: Any) -> Optional[Dict[str, Optional[str]]]:
        if kickoff is None:
            return None
        if not isinstance(kickoff, Mapping):
            raise ValueError('kickoff must be {"earliest": "HH:MM", "latest": "HH:MM"}')
        out: Dict[str, Optional[str]] = {}
        for end in ("earliest", "latest"):
            value = kickoff.get(end)
            if value in (None, ""):
                out[end] = None
                continue
            if not isinstance(value, str) or not _HHMM.fullmatch(value.strip()):
                raise ValueError(f"kickoff {end} must look like 18:30, got {value!r}")
            out[end] = value.strip()
        return out if (out["earliest"] or out["latest"]) else None

    # conversion ----------------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready: sets become sorted lists."""
        out = dataclasses.asdict(self)
        out["markets"] = sorted(self.markets, key=MARKETS.index)
        out["books"] = sorted(self.books) if self.books is not None else None
        return out

    def cache_key(self) -> str:
        """A stable string for caching a build: equal controls give equal keys."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def replace(self, **changes: Any) -> "Controls":
        return dataclasses.replace(self, **changes)

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]], base: Optional["Controls"] = None) -> "Controls":
        """Controls from JSON (a request, the person's saved settings, the AI's tool call).

        Missing fields come from the named ``preset`` when there is one, else from ``base``, else the
        defaults; unknown fields are ignored. Raises ValueError (with a plain reason) on a bad value.
        """
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ValueError("The controls must be an object")
        name = data.get("preset")
        start = base
        if isinstance(name, str) and name.strip().lower() not in ("", "custom"):
            from .presets import PRESETS
            key = name.strip().lower()
            if key not in PRESETS:
                raise ValueError(f"Unknown preset {name!r}; choose safe, risky or crazy")
            start = PRESETS[key]
        start = start or cls()
        values = start.to_dict()
        names = {f.name for f in dataclasses.fields(cls)}
        for key, value in data.items():
            if key in names:
                values[key] = value
        if "preset" in data and not (isinstance(name, str) and name.strip()):
            values["preset"] = None
        return cls(**values)


def _zone(name: Optional[str]) -> datetime.tzinfo:
    if not name:
        return datetime.timezone.utc
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        raise ValueError(f"Unknown time zone {name!r}") from None


# --- the candidates --------------------------------------------------------------------------------

@dataclass
class Candidate:
    """One leg the builder may use: an outcome of a game at a price, with the brain's chance and reasons.

    ``chance`` is the chance the leg wins outright (a push is not a win): the blend, else the price books'
    margin-free chance (times 1 - push on a line that can push), else the goals model's. ``fair`` is the books'
    margin-free chance given no push (tickets.TicketLeg.market_prob), when there is one. ``teams`` is (away,
    home). ``prices`` optionally holds every price book's price at this line, so "allowed books" can pick the
    best one the person bets at; otherwise ``price`` at ``book`` is the only one. ``edge`` defaults to
    chance - (1 - push) / price (chance - 1/price on a line that can't push).
    """

    ref: Dict[str, Any]  # {league, id, market, side, point}
    league: str
    game_key: str
    teams: Tuple[str, str]
    start: str
    price: Decimal
    book: str
    chance: float
    chance_source: str = "blend"
    edge: Optional[float] = None
    price_age_s: int = 0
    flags: Set[str] = field(default_factory=set)
    why: List[str] = field(default_factory=list)
    risk: Optional[str] = None
    label: str = ""
    fair: Optional[float] = None
    push: float = 0.0
    prices: Optional[Dict[str, Decimal]] = None
    # v3 Stage 4: a totals leg whose chance a forecast or roof change since the odds moved (data/context.py):
    # {"applied", "reason", ...}; None otherwise
    adjusted: Optional[Dict[str, Any]] = None
    # The Why lines in each unit system ({"metric": [...], "imperial": [...]}) when their words differ (a venue's
    # wind or temperature): ``why`` is the league's "auto" one; the app shows the person's
    why_units: Optional[Dict[str, List[str]]] = None

    def __post_init__(self) -> None:
        ref = dict(self.ref or {})
        ref.setdefault("league", self.league)
        if not ref.get("id") or not ref.get("market") or not ref.get("side"):
            raise ValueError("A candidate's ref needs id, market and side")
        point = ref.get("point")
        ref["point"] = None if point is None else float(point)
        ref["id"] = str(ref["id"])
        self.ref = {"league": ref["league"], "id": ref["id"], "market": ref["market"], "side": ref["side"],
                    "point": ref["point"]}
        self.price = _decimal(self.price, "price")
        if not self.price > 1:
            raise ValueError(f"{self.ref}: decimal odds must be above 1, got {self.price}")
        self.chance = float(self.chance)
        if not 0.0 <= self.chance <= 1.0:
            raise ValueError(f"{self.ref}: chance must be between 0 and 1, got {self.chance}")
        self.push = min(max(float(self.push or 0.0), 0.0), 1.0 - self.chance)
        if self.fair is not None:
            self.fair = float(self.fair)
            if not 0.0 <= self.fair <= 1.0:
                raise ValueError(f"{self.ref}: fair must be between 0 and 1, got {self.fair}")
        if self.chance_source not in CHANCE_SOURCES:
            raise ValueError(f"chance_source must be one of {', '.join(CHANCE_SOURCES)}")
        self.flags = {str(flag) for flag in (self.flags or ())}
        self.why = [str(line) for line in (self.why or ()) if line]
        if self.why_units:
            self.why_units = {str(name): [str(line) for line in lines or () if line]
                              for name, lines in self.why_units.items()} or None
        self.teams = (str(self.teams[0]), str(self.teams[1]))
        self.price_age_s = max(0, int(self.price_age_s or 0))
        if self.prices:
            prices = {}
            for book, price in self.prices.items():
                try:
                    value = _decimal(price, "price")
                except ValueError:
                    continue
                if value > 1:
                    prices[str(book)] = value
            self.prices = prices or None
        if self.edge is None:
            self.edge = edge_of(self.chance, self.push, self.price)
        if not self.label:
            self.label = _default_label(self.ref, self.teams)

    @property
    def market(self) -> str:
        return self.ref["market"]

    @property
    def side(self) -> str:
        return self.ref["side"]

    @property
    def point(self) -> Optional[float]:
        return self.ref["point"]

    @property
    def key(self) -> str:
        return leg_key(self.ref)

    @property
    def game(self) -> str:
        return f"{self.teams[0]} @ {self.teams[1]}"


def edge_of(chance: float, push: float, price: Union[Decimal, float]) -> float:
    """The leg's edge as a chance: chance - (1 - push) / price, so chance * price + push - 1 = price * edge."""
    return chance - (1.0 - push) / float(price)


def _default_label(ref: Mapping[str, Any], teams: Tuple[str, str]) -> str:
    away, home = teams
    market, side, point = ref["market"], ref["side"], ref.get("point")
    team = {"home": home, "away": away}.get(side)
    if market == "h2h":
        return "Draw" if side == "draw" else str(team or side)
    if market == "spreads" and team:
        return f"{team} {point:+g}" if point is not None else str(team)
    if market == "totals":
        return f"{side.title()} {point:g}" if point is not None else side.title()
    return f"{side} ({MARKET_TITLES.get(market, market)})"


# --- reach the target ---------------------------------------------------------------------------------

def reach_chance(legs: Sequence[Tuple[float, float, float]], threshold: Optional[float]) -> float:
    """The chance a multi pays at least ``threshold`` times the stake: every leg wins or pushes (a pushed leg
    drops out at 1) and the winning legs' prices multiply to the threshold or more. ``legs`` holds (chance of
    a win outright, chance of a push, decimal price). With no threshold it is the chance every leg wins."""
    win_all = math.prod(c for c, _, _ in legs)
    if threshold is None:
        return win_all
    fixed = [(c, d) for c, p, d in legs if p <= 0]
    varying = [(c, p, d) for c, p, d in legs if p > 0]
    base_p = math.prod(c for c, _ in fixed)
    base_d = math.prod(d for _, d in fixed)
    total = 0.0
    for mask in range(1 << len(varying)):
        chance, odds = base_p, base_d
        for bit, (c, p, d) in enumerate(varying):
            if mask >> bit & 1:
                chance, odds = chance * c, odds * d
            else:
                chance *= p
        if odds >= threshold * (1 - 1e-12):
            total += chance
    return total


# --- internal leg ------------------------------------------------------------------------------------

class _Leg:
    """A candidate that passed the rules, with the numbers the search needs worked out once."""

    __slots__ = ("cand", "key", "price", "book", "d", "ld", "c", "lc", "lw", "f", "lf", "le", "lb", "push", "edge",
                 "gbit", "teams", "league", "multi_ok", "q")

    def __init__(self, cand: Candidate, price: Decimal, book: str, caution: float, weights: Weights,
                 multi_leg_min: Optional[float]) -> None:
        self.cand, self.key, self.price, self.book = cand, cand.key, price, book
        self.d = float(price)
        self.ld = math.log(self.d)
        self.c = cand.chance
        self.lc = math.log(max(self.c, _TINY))
        self.push = cand.push
        self.lw = math.log(max(self.c + self.push, _TINY))  # the chance it doesn't lose (a push is stake back)
        model = "model_price" in cand.flags
        # A leg priced at the goals model's fair odds is valued as a bookmaker would price it (see MODEL_MARGIN)
        paid = self.d / (1.0 + MODEL_MARGIN) if model else self.d
        self.f = max(self.c - caution, 0.0) * paid + self.push  # cautious expected return per unit
        self.lf = math.log(max(self.f, _TINY))
        self.le = math.log(max(self.c * paid + self.push, _TINY))  # expected return per unit at the brain's chance
        # The books' margin-free return per unit at this price (tickets.py's book cut leaves pushes out); None
        # when the books give no chance (a goals-model leg)
        self.lb = math.log(max(cand.fair * self.d, _TINY)) if cand.fair is not None else None
        # No bookmaker prices a fair-odds leg, so there is no price to measure an edge against
        self.edge = None if model else edge_of(self.c, self.push, price)
        self.gbit = 0
        self.teams: Tuple[int, ...] = ()
        self.league = 0
        self.multi_ok = multi_leg_min is None or self.c >= multi_leg_min - _EPS
        self.q = weights.ret * self.lf + weights.chance * self.lc


@dataclass
class _Context:
    controls: Controls
    weights: Weights
    sizes: Set[int]
    log_lo: float
    log_hi: float
    log_target: Optional[float]
    log_jmin: float
    joint_max: Optional[float]
    whelan_on: bool
    whelan_slack: int
    team_cap: int
    league_cap: Optional[int]
    rejected: Dict[str, int]
    whelan_thresholds: Tuple[float, ...] = ()
    tried: int = 0
    pushes: bool = False  # some leg that passed can push
    min_ev: Optional[float] = None  # presets.VALUE_RULES while the run is exactly a preset
    max_book_cut: Optional[float] = None
    three_leg_min: Optional[float] = None
    # Closest effort: what the search aims for ("weights": the preset's score, as before; "odds": the highest odds;
    # "chance": the highest chance; "closest": the smallest distance to both targets; "near": the odds nearest the
    # target), a floor on the joint chance (every leg wins; -inf for none) and closest's scales
    objective: str = "weights"
    log_cmin: float = -math.inf
    target_chance: Optional[float] = None
    close_tol: float = math.log(1.1)
    wide: bool = False  # closest effort's searches: a wider exhaustive pool for 1 to 3 legs (WIDE_COMBOS)
    narrow: bool = False  # closest effort's top-ups once its budget is spent (NARROW_POOL, NARROW_WIDTH)

    def whelan_max(self, geometric_mean: float) -> int:
        """engine/bets.py's whelan_max_legs, from thresholds read once per build."""
        return 1 + sum(1 for threshold in self.whelan_thresholds if geometric_mean > threshold)

    def close(self, ld: float) -> float:
        """Distance from the target as a share of the tolerance on that side (0 at the target, 1 at the edge)."""
        if self.log_target is None:
            return 0.0
        gap = ld - self.log_target
        span = (self.log_target - self.log_lo) if gap < 0 else (self.log_hi - self.log_target)
        if span <= _TINY:
            return 0.0 if abs(gap) <= _TINY else 1.0
        return abs(gap) / span

    def score(self, n: int, ld: float, lc: float, lf: float) -> float:
        objective = self.objective
        if objective == "weights":
            w = self.weights
            return -w.target * self.close(ld) + w.ret * lf + w.chance * lc - LEG_PENALTY * n
        if objective == "odds":
            return ld + TIE * lc - LEG_PENALTY * n
        if objective == "chance":
            return lc + TIE * ld - LEG_PENALTY * n
        if objective == "near":
            return -abs(ld - (self.log_target or 0.0)) + TIE * lc - LEG_PENALTY * n
        return -self.distance(ld, lc) - LEG_PENALTY * n

    def distance(self, ld: float, lc: float) -> float:
        """closest's d: the odds' distance from the target in tolerances plus the chance's in 10-point units."""
        odds = abs(ld - self.log_target) / self.close_tol if self.log_target is not None else 0.0
        chance = abs(math.exp(lc) - self.target_chance) / CHANCE_UNIT if self.target_chance is not None else 0.0
        return odds + chance

    def wins_money(self, lc: float, legs: Sequence["_Leg"]) -> float:
        """The chance the ticket pays more than the stake: no leg loses and not every leg pushes (a pushed leg drops
        out at 1 and every price is above 1). With no push it is the chance every leg wins, exp(lc)."""
        if not self.pushes or not any(leg.push > 0 for leg in legs):
            return math.exp(lc)
        return math.prod(leg.c + leg.push for leg in legs) - math.prod(leg.push for leg in legs)

    def ticket_rule(self, n: int, ld: float, lc: float, legs: Sequence["_Leg"] = ()) -> Optional[str]:
        """The first ticket-level rule a complete ticket fails, or None. The joint band is on the chance the ticket
        wins money; ``legs`` (the ticket's legs) is needed for that when a leg can push, and for the rules on the
        whole ticket's value (min_ev, max_book_cut) and Safe's 3-leg floor."""
        if ld < self.log_lo - _EPS:
            return "odds_low"
        if ld > self.log_hi + _EPS:
            return "odds_high"
        if lc < self.log_cmin - _EPS:
            return "target_chance"
        if self.log_jmin > -math.inf or self.joint_max is not None:
            wins = self.wins_money(lc, legs)
            if self.log_jmin > -math.inf and math.log(max(wins, _TINY)) < self.log_jmin - _EPS:
                return "joint_min"
            if self.joint_max is not None and wins >= self.joint_max - 1e-12:
                return "joint_max"
        if self.whelan_on and n >= 2 and n > self.whelan_max(math.exp(lc / n)) + self.whelan_slack:
            return "whelan"
        if self.three_leg_min is not None and n >= 3 and any(leg.c < self.three_leg_min - _EPS for leg in legs):
            return "three_leg"
        if self.min_ev is not None and math.exp(sum(leg.le for leg in legs)) - 1 < self.min_ev - _EPS:
            return "min_ev"
        if self.max_book_cut is not None and legs and all(leg.lb is not None for leg in legs) \
                and 1 - math.exp(sum(leg.lb for leg in legs)) > self.max_book_cut + _EPS:
            return "book_cut"
        return None


# --- the rules ---------------------------------------------------------------------------------------

SCOPE_RULES = ("league", "market", "window", "kickoff", "started", "no_chance")
QUALITY_RULES = ("books", "max_price_age", "leg_odds_min", "leg_odds_max", "min_chance", "min_edge", "max_gap",
                 "disagree", "thin", "low_conf", "model_price")
TICKET_RULES = ("odds_low", "odds_high", "joint_min", "joint_max", "whelan", "caps", "three_leg", "min_ev", "book_cut",
                "target_chance")


def _local_minutes(moment: datetime.datetime, zone: datetime.tzinfo) -> int:
    local = moment.astimezone(zone)
    return local.hour * 60 + local.minute


def _hhmm(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    hours, minutes = text.split(":")
    return int(hours) * 60 + int(minutes)


class _Rules:
    """Every candidate rule of one set of controls, with the words for each."""

    def __init__(self, controls: Controls, label: str, now: datetime.datetime,
                 value_off: Iterable[str] = ()) -> None:
        self.controls, self.label, self.now = controls, label, now
        self.zone = _zone(controls.tz)
        self.leagues = set(controls.leagues) if controls.leagues else None
        self.window_from = _parse_time((controls.window or {}).get("from"))
        self.window_to = _parse_time((controls.window or {}).get("to"))
        self.earliest = _hhmm((controls.kickoff or {}).get("earliest"))
        self.latest = _hhmm((controls.kickoff or {}).get("latest"))
        allowed = controls.books
        self.allowed = None if allowed is None else {_book_key(b) or b.lower() for b in allowed}
        safe = label == "Safe"
        from .presets import value_rules
        self.value = value_rules(label)  # the old tier's rules on a ticket's value, while the run is that preset
        for rule in value_off:  # closest effort dropped these (never Safe's gap to the books)
            self.value.pop(VALUE_RULE_KEYS[rule], None)
        self.max_gap: Optional[float] = self.value.get("max_gap")
        self.exclude_low_conf = controls.exclude_low_conf or safe
        self.exclude_model_price = controls.exclude_model_price or safe
        self.exclude_disagree = controls.exclude_disagree or safe
        self._books: Dict[str, bool] = {}

    def book_ok(self, book: str) -> bool:
        known = self._books.get(book)
        if known is None:
            known = _price_book(book) and (self.allowed is None or (_book_key(book) or book.lower()) in self.allowed)
            self._books[book] = known
        return known

    def best_price(self, cand: Candidate) -> Optional[Tuple[Decimal, str]]:
        if "model_price" in cand.flags:
            return cand.price, cand.book  # priced at the goals model's fair odds: no book prices it
        offers = dict(cand.prices or {})
        offers.setdefault(cand.book, cand.price)
        usable = [(price, book) for book, price in offers.items() if self.book_ok(book)]
        if not usable:
            return None
        return max(usable, key=lambda pair: (pair[0], pair[1] == cand.book, pair[1]))

    def scope(self, cand: Candidate) -> Optional[str]:
        c = self.controls
        if self.leagues is not None and cand.league not in self.leagues:
            return "league"
        if cand.market not in c.markets:
            return "market"
        start = _parse_time(cand.start)
        if start is not None:
            if self.window_from is not None and start < self.window_from:
                return "window"
            if self.window_to is not None and start > self.window_to:
                return "window"
            if self.earliest is not None or self.latest is not None:
                minute = _local_minutes(start, self.zone)
                early = self.earliest if self.earliest is not None else 0
                late = self.latest if self.latest is not None else 24 * 60 - 1
                inside = early <= minute <= late if early <= late else (minute >= early or minute <= late)
                if not inside:
                    return "kickoff"
        # A game under way (or one whose start can't be read) has only pre-game prices: never built into a ticket
        if start is None or start <= self.now or "live" in cand.flags:
            return "started"
        if cand.chance <= 0.001:
            return "no_chance"
        return None

    def quality(self, cand: Candidate, price: Optional[Decimal]) -> List[str]:
        """Every quality rule the leg fails at its best allowed price (in QUALITY_RULES order)."""
        c = self.controls
        failed = []
        if price is None:
            failed.append("books")
            price = cand.price
        if cand.price_age_s > c.max_price_age_s:
            failed.append("max_price_age")
        d = float(price)
        if c.leg_odds_min is not None and d < c.leg_odds_min - _EPS:
            failed.append("leg_odds_min")
        if c.leg_odds_max is not None and d > c.leg_odds_max + _EPS:
            failed.append("leg_odds_max")
        if c.min_chance is not None and cand.chance < c.min_chance - _EPS:
            failed.append("min_chance")
        # A leg priced at the goals model's fair odds has no bookmaker's price to measure an edge against
        if c.min_edge is not None and ("model_price" in cand.flags
                                       or edge_of(cand.chance, cand.push, price) < c.min_edge - _EPS):
            failed.append("min_edge")
        gap = _gap(cand)
        if self.max_gap is not None and gap is not None and abs(gap) > self.max_gap + _EPS:
            failed.append("max_gap")
        if self.exclude_disagree and "disagree" in cand.flags:
            failed.append("disagree")
        if c.exclude_thin and "thin" in cand.flags:
            failed.append("thin")
        if self.exclude_low_conf and "low_conf" in cand.flags:
            failed.append("low_conf")
        if self.exclude_model_price and "model_price" in cand.flags:
            failed.append("model_price")
        return failed

    def distance(self, rule: str, cand: Candidate, price: Decimal) -> float:
        """How far a leg is from passing one rule, as a share of the rule's own threshold (0 at the edge); a yes/no
        rule (books, disagreement, thin market, low confidence, fair odds) counts 1."""
        c = self.controls
        d = float(price)
        if rule == "max_price_age":
            return (cand.price_age_s - c.max_price_age_s) / max(c.max_price_age_s, 1)
        if rule == "leg_odds_min":
            return (c.leg_odds_min - d) / c.leg_odds_min
        if rule == "leg_odds_max":
            return (d - c.leg_odds_max) / c.leg_odds_max
        if rule == "min_chance":
            return (c.min_chance - cand.chance) / max(c.min_chance, 0.01)
        if rule == "min_edge" and "model_price" not in cand.flags:
            return (c.min_edge - edge_of(cand.chance, cand.push, price)) / 0.05  # 5 points counts 1
        if rule == "max_gap":
            return (abs(_gap(cand) or 0.0) - self.max_gap) / max(self.max_gap, 0.01)
        return 1.0

    def reason(self, rule: str, cand: Candidate, price: Decimal) -> str:
        """Why one leg failed one rule, in plain words."""
        c = self.controls
        d = float(price)
        if rule == "books":
            return "not offered at your books"
        if rule == "max_price_age":
            return f"price {age_text(cand.price_age_s)} old (max {age_text(c.max_price_age_s)})"
        if rule == "leg_odds_min":
            return f"odds {d:.2f} below your minimum {c.leg_odds_min:.2f}"
        if rule == "leg_odds_max":
            return f"odds {d:.2f} above your maximum {c.leg_odds_max:.2f}"
        if rule == "min_chance":
            return f"chance {_pct(cand.chance)} below your minimum {_pct(c.min_chance)}"
        if rule == "min_edge":
            if "model_price" in cand.flags:
                return "priced at fair odds: no bookmaker's price to measure an edge against"
            edge = edge_of(cand.chance, cand.push, price)
            return f"edge {edge * 100:+.1f} pts below your minimum {c.min_edge * 100:+.1f}"
        if rule == "max_gap":
            gap = _gap(cand) or 0.0
            return (f"the brain's chance is {abs(gap) * 100:.0f} pts {'above' if gap > 0 else 'below'} the books' "
                    f"(max {self.max_gap * 100:.0f} in {self.label})")
        if rule == "disagree":
            return f"the sources disagree on this game ({'not in Safe' if self.label == 'Safe' else 'excluded'})"
        if rule == "thin":
            return "thin market only (excluded)"
        if rule == "low_conf":
            return "chance from ratings only, low confidence (" + (
                "not in Safe" if self.label == "Safe" else "excluded") + ")"
        if rule == "model_price":
            return "priced at fair odds by the goals model (" + (
                "not in Safe" if self.label == "Safe" else "excluded") + ")"
        return rule

    def rule_label(self, rule: str) -> str:
        """A rule's short name for "which rules ruled it out"."""
        c = self.controls
        labels = {
            "league": "leagues in scope", "market": "markets chosen", "window": "time window",
            "kickoff": "kick-off window", "started": "games already started", "no_chance": "no chance available",
            "books": "your books", "max_price_age": f"max price age {age_text(c.max_price_age_s)}",
            "leg_odds_min": f"leg odds {c.leg_odds_min or 0:.2f}+",
            "leg_odds_max": f"leg odds up to {c.leg_odds_max or 0:.2f}",
            "min_chance": f"min chance {_pct(c.min_chance or 0)}",
            "min_edge": f"min edge {(c.min_edge or 0) * 100:+.1f} pts",
            "max_gap": f"chance within {(self.max_gap or 0) * 100:.0f} pts of the books' ({self.label})",
            "disagree": "exclude disagreement", "thin": "exclude thin markets",
            "low_conf": "no low-confidence legs", "model_price": "no fair-odds legs",
            "joint_min": f"joint chance {_pct(c.joint_min or 0)}+",
            "joint_max": f"joint chance under {_pct(c.joint_max or 1)}",
            "whelan": "the Whelan leg cap",
            "caps": _caps_words(c),
            "three_leg": f"every leg {_pct(self.value.get('three_leg_min') or 0)}+ on a 3-leg ticket ({self.label})",
            "min_ev": f"expected result {(self.value.get('min_ev') or 0) * 100:+.0f}% or better ({self.label})",
            "book_cut": f"bookmaker's cut up to {_pct(self.value.get('max_book_cut') or 0)} ({self.label})",
            "target_chance": f"{_a_pct(c.target_chance or 0)}+ chance for the whole ticket",
        }
        if rule in ("odds_low", "odds_high"):
            return _window_words(c)
        return labels.get(rule, rule)


def _gap(cand: Candidate) -> Optional[float]:
    """How far the leg's chance is from the books' margin-free chance, like for like (both given no push)."""
    if cand.fair is None or cand.push >= 1.0:
        return None
    return cand.chance / (1.0 - cand.push) - cand.fair


def _caps_words(controls: Controls) -> str:
    parts = [f"{controls.max_per_team} per team"]
    if controls.max_per_league is not None:
        parts.append(f"{controls.max_per_league} per league")
    return f"legs per team and league (at most {' and '.join(parts)})"


def _window(controls: Controls) -> Tuple[float, float]:
    """The ticket's odds window: target +/- tolerance, else the preset's band, else anything. A target chance
    alone replaces the band too: it sets the ticket, at any odds."""
    if controls.target_odds is not None:
        tol = controls.tolerance_pct / 100.0
        return controls.target_odds * (1 - tol), controls.target_odds * (1 + tol)
    if controls.target_chance is not None:
        return 1.0, math.inf
    return (controls.odds_min or 1.0), (controls.odds_max or math.inf)


def _targeted(controls: Controls) -> bool:
    """Whether the person gave a target (odds or chance): then the preset's band doesn't apply."""
    return controls.target_odds is not None or controls.target_chance is not None


def _window_words(controls: Controls) -> str:
    low, high = _window(controls)
    if controls.target_odds is not None:
        return (f"odds {_odds_text(low)}-{_odds_text(high)} (target {controls.target_odds:,.1f}x "
                f"± {controls.tolerance_pct:g}%)")
    if math.isinf(high):
        return f"ticket odds {_odds_text(low)}+"
    return f"ticket odds {_odds_text(low)}-{_odds_text(high)}"


# --- the result -------------------------------------------------------------------------------------

@dataclass
class Result:
    """What build() returns; ``to_dict()`` is the JSON the routes send (money and prices as strings)."""

    label: str
    controls: Controls
    tickets: List[Dict[str, Any]] = field(default_factory=list)
    near_misses: List[Dict[str, Any]] = field(default_factory=list)
    blocked: Optional[Dict[str, Any]] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    ideas: List[Dict[str, Any]] = field(default_factory=list)
    target: Optional[Dict[str, Any]] = None
    stake: str = "10.00"
    # Closest effort's labelled options when both targets were given and no ticket meets both: [{key, label,
    # ticket_index}] (key "your_odds", "your_chance" or "closest"); empty otherwise
    options: List[Dict[str, Any]] = field(default_factory=list)
    # The first search's best tickets (score, leg keys), best first: for engine/target.py and engine/bets.py, which
    # pick by their own rules; not part of the JSON
    ranked: List[Tuple[float, Tuple[str, ...]]] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "preset": self.controls.preset, "controls": self.controls.to_dict(),
                "tickets": self.tickets, "near_misses": self.near_misses, "blocked": self.blocked,
                "stats": self.stats, "notes": self.notes, "ideas": self.ideas, "target": self.target,
                "stake": self.stake, "options": self.options}


# --- the search ---------------------------------------------------------------------------------------

def _ideals(ctx: _Context, sizes: Set[int]) -> List[float]:
    """The log price per leg that lands a ticket of each size in ``sizes`` at the target (else at the low end of
    the preset's band, when it has one above 1)."""
    aim = ctx.log_target if ctx.log_target is not None else (ctx.log_lo if ctx.log_lo > _EPS else None)
    if aim is None or aim <= 0:
        return []
    return [aim / n for n in sorted(sizes)]


def _take(ranked: Sequence[_Leg], size: int, out: List[_Leg]) -> List[_Leg]:
    """``out`` filled up to ``size`` legs from ``ranked`` (best first), at most PER_GAME_POOL a game while there is
    room for every game's best; a game's further legs only fill what is left at the end."""
    have = {leg.key for leg in out}
    per_game: Dict[int, int] = {}
    for leg in out:
        per_game[leg.gbit] = per_game.get(leg.gbit, 0) + 1
    spare: List[_Leg] = []
    for leg in ranked:
        if len(out) >= size:
            return out
        if leg.key in have:
            continue
        if per_game.get(leg.gbit, 0) >= PER_GAME_POOL:
            spare.append(leg)
            continue
        per_game[leg.gbit] = per_game.get(leg.gbit, 0) + 1
        have.add(leg.key)
        out.append(leg)
    return out + spare[: max(0, size - len(out))]


def _pool(legs: Sequence[_Leg], ctx: _Context, size: int, sizes: Set[int], near: bool = False
          ) -> Tuple[List[_Leg], bool]:
    """Up to ``size`` legs that could fit the window, for the search (every one of them when they fit), and whether
    some leg that could fit was left out.

    Without a target, the best by score. With one, every ticket lands near the same odds, so the joint chance's
    taste for short prices cancels out: half the pool is the best legs by value whatever their price (score plus
    w_chance x log price, a little closer to a per-leg share of the target being better), and the other half the
    legs priced nearest that share (log target / n for each leg count searched), so the pool's legs can multiply to
    the target. ``near`` fills the whole pool by that nearness (to the target, else the band's low end): the second
    try when the first finds too few tickets."""
    if not legs:
        return [], False
    n_max = max(sizes)
    top = max(leg.ld for leg in legs)
    fits = [leg for leg in legs if leg.ld <= ctx.log_hi + _EPS and leg.ld + top * (n_max - 1) >= ctx.log_lo - _EPS]
    if ctx.log_cmin > -math.inf:  # a chance floor: a leg less likely than the floor is in no ticket that meets it
        fits = [leg for leg in fits if leg.lc >= ctx.log_cmin - _EPS]
    if len(fits) <= size:
        return sorted(fits, key=lambda leg: (-leg.q, leg.key)), False  # everything fits: the search sees every leg
    if ctx.objective == "odds" and ctx.log_cmin > -math.inf:
        # The highest odds at a chance floor: half the pool the best value whatever the price, half the legs whose
        # chance is nearest a per-leg share of the floor (so the pool's legs can multiply to it)
        shares = [ctx.log_cmin / n for n in sorted(sizes)]

        def share_gap(leg: _Leg) -> float:
            return min(abs(leg.lc - share) / max(abs(share), _TINY) for share in shares)

        nearest = sorted(fits, key=lambda leg: (share_gap(leg), -leg.le, leg.key))
        if near:
            return _take(nearest, size, []), True
        out = _take(sorted(fits, key=lambda leg: (-leg.le, leg.key)), size - size // 2, [])
        return _take(nearest, size, out), True
    ideals = _ideals(ctx, sizes)

    def distance(leg: _Leg) -> float:
        return min(abs(leg.ld - ideal) / ideal for ideal in ideals) if ideals else 0.0

    if ctx.log_target is None and not near:
        return _take(sorted(fits, key=lambda leg: (-leg.q, leg.key)), size, []), True
    chance = ctx.weights.chance

    def value(leg: _Leg) -> float:
        return leg.q + chance * leg.ld - ctx.weights.target * distance(leg)

    closest = sorted(fits, key=lambda leg: (distance(leg), -value(leg), leg.key))
    if near:
        return _take(closest, size, []), True
    out = _take(sorted(fits, key=lambda leg: (-value(leg), leg.key)), size - size // 2, [])
    return _take(closest, size, out), True


class _Found:
    """The best tickets found so far (a bounded min-heap on score; equal scores are ordered by their legs' keys,
    so the result never depends on the order the search met them)."""

    def __init__(self, keep: int = HEAP_KEEP) -> None:
        self.keep = keep
        self.heap: List[Tuple[float, Tuple[str, ...], Tuple[_Leg, ...]]] = []
        self.seen: Set[Tuple[str, ...]] = set()

    def add(self, score: float, legs: Tuple[_Leg, ...]) -> None:
        full = len(self.heap) >= self.keep
        if full and score < self.heap[0][0]:
            return
        keys = tuple(sorted(leg.key for leg in legs))
        if keys in self.seen:
            return
        item = (score, keys, legs)
        if not full:
            heapq.heappush(self.heap, item)
            self.seen.add(keys)
        elif item[:2] > self.heap[0][:2]:
            dropped = heapq.heapreplace(self.heap, item)
            self.seen.discard(dropped[1])
            self.seen.add(keys)

    def worth(self, score: float) -> bool:
        """Whether a ticket with this score could enter (checked before its legs are gathered)."""
        return len(self.heap) < self.keep or score >= self.heap[0][0]

    def ranked(self) -> List[Tuple[float, Tuple[_Leg, ...]]]:
        return [(score, legs) for score, _, legs in sorted(self.heap, key=lambda item: item[:2], reverse=True)]


def _caps_ok(leg: _Leg, team_count: Dict[int, int], league_count: Dict[int, int], ctx: _Context) -> bool:
    if any(team_count.get(t, 0) >= ctx.team_cap for t in leg.teams):
        return False
    return ctx.league_cap is None or league_count.get(leg.league, 0) < ctx.league_cap


def _exhaustive(pool: List[_Leg], sizes: Set[int], ctx: _Context, found: _Found) -> None:
    """Every ticket of the pool's legs with a size in ``sizes`` (all at most EXHAUSTIVE_MAX), depth first over
    legs sorted by price: once the odds pass the window every longer leg does too.

    A branch cut by the joint-chance floor or a team/league cap counts against that rule whenever it could still
    have landed in the window (now, or with longer legs to come), so "which rule ruled it out" names it even when
    no complete ticket is ever reached."""
    if not pool or not sizes:
        return
    pool = sorted(pool, key=lambda leg: (leg.d, leg.key))
    count = len(pool)
    n_max = max(sizes)
    top = max(leg.ld for leg in pool)
    log_lo, log_hi, log_jmin, log_cmin = ctx.log_lo, ctx.log_hi, ctx.log_jmin, ctx.log_cmin
    team_count: Dict[int, int] = {}
    league_count: Dict[int, int] = {}
    chosen: List[_Leg] = []
    rejected = ctx.rejected

    def visit(start: int, ld: float, lc: float, lw: float, lf: float, games: int) -> None:
        depth = len(chosen)
        for j in range(start, count):
            leg = pool[j]
            nld = ld + leg.ld
            if nld > log_hi + _EPS:
                break
            if depth and not leg.multi_ok:
                continue
            if games & leg.gbit:
                continue
            n = depth + 1
            # Whether this branch could still land in the window: now, or with longer legs to come
            deeper = n < n_max and nld + top * (n_max - n) >= log_lo - _EPS
            could = deeper or (n in sizes and nld >= log_lo - _EPS)
            nlw = lw + leg.lw  # the chance of not losing bounds the chance of winning money from above
            if nlw < log_jmin - _EPS:
                rejected["joint_min"] += could
                continue
            nlc = lc + leg.lc
            if nlc < log_cmin - _EPS:  # more legs only lower the chance
                rejected["target_chance"] += could
                continue
            if not _caps_ok(leg, team_count, league_count, ctx):
                rejected["caps"] += could
                continue
            nlf = lf + leg.lf
            chosen.append(leg)
            for t in leg.teams:
                team_count[t] = team_count.get(t, 0) + 1
            league_count[leg.league] = league_count.get(leg.league, 0) + 1
            if n in sizes:
                ctx.tried += 1
                rule = ctx.ticket_rule(n, nld, nlc, chosen)
                if rule is None:
                    score = ctx.score(n, nld, nlc, nlf)
                    if found.worth(score):
                        found.add(score, tuple(chosen))
                else:
                    rejected[rule] += 1
            if deeper and leg.multi_ok:
                visit(j + 1, nld, nlc, nlw, nlf, games | leg.gbit)
            chosen.pop()
            for t in leg.teams:
                team_count[t] -= 1
            league_count[leg.league] -= 1

    visit(0, 0.0, 0.0, 0.0, 0.0, 0)


def _beam(pool: List[_Leg], sizes: Set[int], ctx: _Context, found: _Found) -> None:
    """Tickets of 5 to 12 legs: a beam over partial tickets, spread across odds buckets (see the module doc).

    One leg per game means a leg already in a partial ticket always has its game taken, so the game mask alone
    keeps a leg from being used twice. Extensions are kept as light records (quality, sums, parent, leg); only
    the ones the beam keeps become full states. Every extension dropped counts against its rule (the odds window,
    the joint-chance floor, the team and league caps), so a beam that ends empty can say why."""
    pool = [leg for leg in pool if leg.multi_ok]
    if not pool or not sizes:
        return
    count = len(pool)
    n_max = max(sizes)
    fixed = len(sizes) == 1
    LD = [leg.ld for leg in pool]
    LC = [leg.lc for leg in pool]
    LW = [leg.lw for leg in pool]
    LF = [leg.lf for leg in pool]
    GB = [leg.gbit for leg in pool]
    LG = [leg.league for leg in pool]
    TEAMS = [leg.teams for leg in pool]
    TB = [sum(1 << t for t in set(teams)) for teams in TEAMS]
    ld_min, ld_max = min(LD), max(LD)
    log_jmin = ctx.log_jmin - _EPS
    log_cmin = ctx.log_cmin - _EPS
    w_ret, w_chance = ctx.weights.ret, ctx.weights.chance
    # With a target every finished ticket lands near the same odds, so the joint chance's taste for short prices
    # cancels out there: partial tickets are ranked whatever their price (as _pool ranks legs), else the beam drifts
    # to the shortest legs and can no longer reach the target
    w_price = w_chance if ctx.log_target is not None else 0.0
    team_cap, league_cap = ctx.team_cap, ctx.league_cap
    single_team = team_cap == 1
    width = NARROW_WIDTH if ctx.narrow else BEAM_WIDTH
    per_bucket = max(1, width // BEAM_BUCKETS)
    # A state: (log odds, log chance, log chance of not losing, log cautious return, games, teams mask, team list,
    # league list, legs mask, leg indexes)
    states: List[Tuple[Any, ...]] = [(0.0, 0.0, 0.0, 0.0, 0, 0, (), (), 0, ())]
    tried = too_low = too_high = joint_cut = caps_cut = chance_cut = 0
    for depth in range(1, n_max + 1):
        remaining = n_max - depth
        high = ctx.log_hi + _EPS - (remaining * ld_min if fixed else 0.0)  # the shortest legs left still fit
        low = ctx.log_lo - _EPS - remaining * ld_max  # the longest legs left can still reach the window
        extended: Dict[int, Tuple[float, float, float, float, float, int, int]] = {}
        get = extended.get
        for si, (ld, lc, lw, lf, games, tmask, teams, leagues, mask, _) in enumerate(states):
            for j in range(count):
                if games & GB[j]:
                    continue
                nld = ld + LD[j]
                if nld > high:
                    too_high += 1
                    continue
                if nld < low:
                    too_low += 1
                    continue
                nlw = lw + LW[j]
                if nlw < log_jmin:
                    joint_cut += 1
                    continue
                nlc = lc + LC[j]
                if nlc < log_cmin:
                    chance_cut += 1
                    continue
                if single_team:
                    if tmask & TB[j]:
                        caps_cut += 1
                        continue
                elif any(teams.count(t) >= team_cap for t in TEAMS[j]):
                    caps_cut += 1
                    continue
                if league_cap is not None and leagues.count(LG[j]) >= league_cap:
                    caps_cut += 1
                    continue
                tried += 1
                nlf = lf + LF[j]
                nq = w_ret * nlf + w_chance * nlc + w_price * nld
                nmask = mask | (1 << j)
                previous = get(nmask)
                if previous is None or nq > previous[0]:
                    extended[nmask] = (nq, nld, nlc, nlw, nlf, si, j)
        if not extended:
            break
        if depth in sizes:
            for nq, nld, nlc, nlw, nlf, si, j in extended.values():
                legs = tuple(pool[i] for i in states[si][9] + (j,))
                rule = ctx.ticket_rule(depth, nld, nlc, legs)
                if rule is not None:
                    ctx.rejected[rule] += 1
                    continue
                score = ctx.score(depth, nld, nlc, nlf)
                if found.worth(score):
                    found.add(score, legs)
        if depth == n_max:
            break
        ranked = sorted(extended.items(), key=lambda item: (-item[1][0], item[0]))
        # Buckets span the odds a partial ticket of this depth may still have, so the kept ones stay spread across
        # every way to the window
        bottom = max(low, depth * ld_min)
        span = max(min(high, depth * ld_max) - bottom, _TINY)
        taken: Dict[int, int] = {}
        keep: List[Tuple[int, Tuple[float, ...]]] = []
        rest: List[Tuple[int, Tuple[float, ...]]] = []
        for item in ranked:
            bucket = min(BEAM_BUCKETS - 1, max(0, int((item[1][1] - bottom) / span * BEAM_BUCKETS)))
            if taken.get(bucket, 0) < per_bucket:
                taken[bucket] = taken.get(bucket, 0) + 1
                keep.append(item)
                if len(keep) == width:
                    break
            else:
                rest.append(item)
        keep += rest[: max(0, width - len(keep))]
        next_states = []
        for nmask, (nq, nld, nlc, nlw, nlf, si, j) in keep:
            ld, lc, lw, lf, games, tmask, teams, leagues, mask, legs = states[si]
            next_states.append((nld, nlc, nlw, nlf, games | GB[j], tmask | TB[j],
                                teams if single_team else teams + TEAMS[j],
                                leagues + (LG[j],) if league_cap is not None else leagues, nmask, legs + (j,)))
        states = next_states
    ctx.tried += tried
    for rule, cut in (("odds_low", too_low), ("odds_high", too_high), ("joint_min", joint_cut), ("caps", caps_cut),
                      ("target_chance", chance_cut)):
        ctx.rejected[rule] += cut


def _search(legs: Sequence[_Leg], ctx: _Context) -> List[Tuple[float, Tuple[_Leg, ...]]]:
    """The best tickets of the legs, best first. When the pool is a cut of the legs and the search finds no ticket,
    it searches once more over the legs priced nearest a per-leg share of the target (_pool's ``near``; build()
    searches again without the legs already used when it needs more tickets): the counts of what each rule ruled
    out keep the larger of the two tries."""
    found = _Found()
    small = {n for n in ctx.sizes if n <= EXHAUSTIVE_MAX}
    large = {n for n in ctx.sizes if n > EXHAUSTIVE_MAX}
    size = NARROW_POOL if ctx.narrow else _pool_size(max(small)) if ctx.wide and small else POOL_K
    beam_pool = NARROW_BEAM_POOL if ctx.narrow else BEAM_POOL
    multi = [leg for leg in legs if leg.multi_ok]
    cut = False
    for near in (False, True):
        if near:
            if not cut or found.heap or not (_ideals(ctx, ctx.sizes)
                                             or (ctx.objective == "odds" and ctx.log_cmin > -math.inf)):
                break
            first = dict(ctx.rejected)
            ctx.rejected = {rule: 0 for rule in first}
        if small:
            pool, left_out = _pool(legs, ctx, size, small, near)
            _exhaustive(pool, small, ctx, found)
            cut = cut or left_out
        if large:
            pool, left_out = _pool(multi, ctx, beam_pool, large, near)
            _beam(pool, large, ctx, found)
            cut = cut or left_out
        if near:
            ctx.rejected = {rule: max(count, first.get(rule, 0)) for rule, count in ctx.rejected.items()}
    return found.ranked()


def _pool_size(n: int) -> int:
    """The widest pool whose n-leg combinations stay within WIDE_COMBOS (never below POOL_K)."""
    k = POOL_K
    if n <= 1:
        return WIDE_COMBOS
    while math.comb(k + 1, n) <= WIDE_COMBOS:
        k += 1
    return k


def _disjoint(ranked: Sequence[Tuple[float, Tuple[_Leg, ...]]], used: Set[str], want: int
              ) -> List[Tuple[float, Tuple[_Leg, ...]]]:
    out = []
    for score, legs in ranked:
        keys = {leg.key for leg in legs}
        if keys & used:
            continue
        out.append((score, legs))
        used |= keys
        if len(out) == want:
            break
    return out


# --- the tickets ---------------------------------------------------------------------------------------

def _ticket_leg(leg: _Leg) -> Any:
    from .tickets import TicketLeg
    cand = leg.cand
    return TicketLeg(key=cand.key, league=cand.league, event_id=cand.ref["id"], market=cand.market,
                     side=cand.side, point=cand.point, price=leg.price, label=cand.label, game=cand.game,
                     start=cand.start, ai_prob=cand.chance, market_prob=cand.fair, push_prob=cand.push,
                     book=leg.book, group=cand.game_key, adjusted=cand.adjusted)


def _leg_risk(cand: Candidate) -> str:
    """The leg's main risk: the caller's, else the flags', else how often it loses."""
    if cand.risk:
        return cand.risk
    if "disagree" in cand.flags:
        return "The sources disagree on this game by 5 points or more."
    if "model_price" in cand.flags:
        return "Priced at fair odds by the goals model: check your bookmaker's price."
    if "low_conf" in cand.flags:
        return "Chance from ratings only: low confidence."
    if "thin" in cand.flags:
        return "Thin market: few prices behind it."
    if "stale_price" in cand.flags:
        return "The price is getting old."
    loses = 1.0 - cand.chance - cand.push
    return f"Loses about {max(0, round(loses * 10))} in 10."


def _round(value: Optional[float], places: int = 4) -> Optional[float]:
    return None if value is None else round(value, places)


def _swap_view(leg: _Leg, odds: float, joint: float, score: float, base_score: float) -> Dict[str, Any]:
    cand = leg.cand
    return {"ref": dict(cand.ref), "key": leg.key, "label": cand.label, "game": cand.game, "price": str(leg.price),
            "book": leg.book, "chance": round(cand.chance, 4), "chance_source": cand.chance_source,
            "edge": _round(leg.edge), "age": age_text(cand.price_age_s), "flags": sorted(cand.flags),
            "why": cand.why[:3], "risk": _leg_risk(cand), "odds": round(odds, 2), "joint": round(joint, 4),
            "score_change": round(score - base_score, 4), **({"adjusted": dict(cand.adjusted)} if cand.adjusted else {}),
            **_why_units(cand)}


def _why_units(cand: Candidate) -> Dict[str, Any]:
    """{"why_units": {"metric": [...], "imperial": [...]}} (the first 3 lines each) when the candidate has them."""
    if not cand.why_units:
        return {}
    return {"why_units": {name: lines[:3] for name, lines in cand.why_units.items()}}


def _swaps(legs: Tuple[_Leg, ...], index: int, passing: Sequence[_Leg], taken: Set[str], ctx: _Context,
           base_score: float, keep: Optional[Callable[[Sequence[_Leg]], bool]] = None) -> List[Dict[str, Any]]:
    """Up to MAX_SWAPS legs that can replace legs[index] and keep every rule (and no leg of another ticket); with
    ``keep``, only those whose ticket it keeps (closest effort's: no rule loosened further than the ticket says)."""
    n = len(legs)
    others = [leg for i, leg in enumerate(legs) if i != index]
    ld = sum(leg.ld for leg in others)
    lc = sum(leg.lc for leg in others)
    lf = sum(leg.lf for leg in others)
    games = 0
    team_count: Dict[int, int] = {}
    league_count: Dict[int, int] = {}
    for leg in others:
        games |= leg.gbit
        for t in leg.teams:
            team_count[t] = team_count.get(t, 0) + 1
        league_count[leg.league] = league_count.get(leg.league, 0) + 1
    options = []
    for leg in passing:
        if leg.key in taken or games & leg.gbit:
            continue
        if n >= 2 and not leg.multi_ok:
            continue
        if not _caps_ok(leg, team_count, league_count, ctx):
            continue
        nld, nlc, nlf = ld + leg.ld, lc + leg.lc, lf + leg.lf
        if ctx.ticket_rule(n, nld, nlc, others + [leg]) is not None:
            continue
        options.append((ctx.score(n, nld, nlc, nlf), leg, nld, nlc))
    options.sort(key=lambda item: (-item[0], item[1].key))
    if keep is not None:
        kept = []
        for item in options:
            if keep(others + [item[1]]):
                kept.append(item)
                if len(kept) == MAX_SWAPS:
                    break
        options = kept
    return [_swap_view(leg, math.exp(nld), math.exp(nlc), score, base_score)
            for score, leg, nld, nlc in options[:MAX_SWAPS]]


def _ticket(rank: int, score: float, legs: Tuple[_Leg, ...], ctx: _Context, label: str, stake: Decimal,
            now: datetime.datetime, passing: Sequence[_Leg], taken: Set[str],
            asked: Optional[Controls] = None, keep: Optional[Callable[[Sequence[_Leg]], bool]] = None
            ) -> Dict[str, Any]:
    """One finished ticket. ``ctx`` and ``passing`` are the search's (closest effort's loosened rules: its swaps keep
    them, and ``keep`` holds them to the ticket's own relaxed words); ``asked`` is the person's own controls (the
    target, the tolerance and the leg count they asked for)."""
    from .tickets import price_ticket
    controls = asked or ctx.controls
    legs = tuple(sorted(legs, key=lambda leg: (leg.cand.start, leg.key)))
    n = len(legs)
    priced = price_ticket([_ticket_leg(leg) for leg in legs], "single" if n == 1 else "multi", stake, now=now)
    joint = math.prod(leg.c for leg in legs)
    outcomes = [(leg.c, leg.push, leg.d) for leg in legs]
    chances = ticket_chances(outcomes, controls.target_odds, controls.tolerance_pct)
    cautious_return = math.prod(leg.f for leg in legs)
    cautious_chance = math.prod(max(leg.c - controls.caution_pp / 100.0, 0.0) for leg in legs)
    model_priced = [leg for leg in legs if "model_price" in leg.cand.flags]
    books = sorted({leg.book for leg in legs})
    profit, lose = _money_chances(outcomes)
    notes = ticket_notes([(leg.cand.label, leg.cand.flags, leg.book) for leg in legs], profit, label == "Crazy",
                         lose=lose)
    worth = ctx.whelan_max(joint ** (1 / n)) if n >= 2 and joint > 0 else n
    if controls.legs is not None and controls.whelan and n > worth and n == controls.legs:
        notes.append(f"You asked for {n} legs: more than are worth combining at these chances ({worth} at most), "
                     "so the Whelan cap was lifted for this run.")
    views = []
    for i, (view, leg) in enumerate(zip(priced["legs"], legs)):
        cand = leg.cand
        views.append({**view, "ev": view.get("edge"), "edge": _round(leg.edge), "ref": dict(cand.ref),
                      "chance": round(cand.chance, 4), "chance_source": cand.chance_source,
                      "price_age_s": cand.price_age_s, "age": age_text(cand.price_age_s),
                      "flags": sorted(cand.flags), "teams": list(cand.teams), "why": cand.why[:3],
                      "risk": _leg_risk(cand), "model_price": "model_price" in cand.flags,
                      "inputs": {"chance": cand.chance, "fair": cand.fair, "push": cand.push},
                      "swaps": _swaps(legs, i, passing, taken, ctx, score, keep),
                      **({"adjusted": dict(cand.adjusted)} if cand.adjusted else {}), **_why_units(cand)})
    return {
        **priced,
        "legs": views,
        "rank": rank,
        "label": label,
        "refs": [dict(leg.cand.ref) for leg in legs],
        "score": round(score, 6),
        **chances,
        "cautious": {"chance": round(cautious_chance, 4), "return": round(cautious_return, 4),
                     "ev": round(cautious_return - 1, 4)},
        "books": books,
        "model_priced": bool(model_priced),
        "notes": notes,
    }


def _money_chances(outcomes: Sequence[Tuple[float, float, Any]]) -> Tuple[float, float]:
    """(the chance a ticket wins money, the chance it loses the stake) from its legs' (chance of a win outright,
    chance of a push, price): a pushed leg drops out at 1, so it wins money when no leg loses and not every leg
    pushes, and loses when some leg loses."""
    not_lose = math.prod(float(c) + float(p or 0.0) for c, p, _ in outcomes)
    every_push = math.prod(float(p or 0.0) for _, p, _ in outcomes)
    return not_lose - every_push, 1.0 - not_lose


def ticket_notes(legs: Sequence[Tuple[str, Iterable[str], str]], profit: float, crazy: bool = False,
                 lose: Optional[float] = None) -> List[str]:
    """The notes on a ticket, from its legs' (label, flags, book), the chance it wins money (``profit``; with no
    push, the chance every leg wins) and the chance it loses the stake (``lose``; 1 - profit when not given): legs
    priced at the goals model's fair odds, legs priced at different books, a long shot ("expect to lose most of
    these": under a 1 in 4 chance of winning money while losing is the likely result; always for Crazy) and the
    legs the sources disagree on."""
    notes = []
    fair = [label for label, flags, _ in legs if "model_price" in set(flags)]
    if fair:
        notes.append(f"Priced at fair odds; check your bookmaker's price ({', '.join(fair)}). The expected result is "
                     "at these fair odds: a bookmaker's price is usually lower.")
    if len({book for _, _, book in legs}) > 1:
        notes.append("Legs are priced at different books: a multi is placed at one book, so check each leg's "
                     "price there.")
    lose = 1.0 - profit if lose is None else lose
    if crazy or (profit < 0.25 and lose > 0.5):
        notes.append("Long shot: expect to lose most of these.")
    disagree = [label for label, flags, _ in legs if "disagree" in set(flags)]
    if disagree:
        notes.append(f"The sources disagree on {', '.join(disagree)}: check the news before betting.")
    return notes


def ticket_chances(outcomes: Sequence[Tuple[float, float, Any]], target_odds: Optional[float] = None,
                   tolerance_pct: float = 10.0) -> Dict[str, Dict[str, Any]]:
    """A ticket's "joint" and "reach_target" blocks from its legs' (chance of a win outright, chance of a push,
    decimal price): build() puts them on every ticket, and the builder's price route on a ticket after a swap.

    ``joint``: the chance every leg wins, as "1 in N" and in words, the ticket's odds and its fair odds: the
    odds at which its expected result is nil, each leg at its own fair odds (1 - push) / chance, since a pushed leg
    drops out at 1 (with no push, 1 / the chance every leg wins). ``profit`` is the chance it wins money and
    ``lose`` the chance it loses the stake (with a push, every leg winning is not the only way to win money).
    ``reach_target``: the chance it pays at least the low end of the target window (target - tolerance; the odds
    named here; with no target, the chance every leg wins) and ``at_target``, the chance it pays the target or more.
    """
    from .tickets import chance_label, one_in
    legs = [(float(c), float(p or 0.0), float(d)) for c, p, d in outcomes]
    joint = math.prod(c for c, _, _ in legs)
    odds = math.prod(d for _, _, d in legs)
    fair = math.prod((1.0 - p) / c for c, p, _ in legs) if all(c > 0 for c, _, _ in legs) else None
    profit, lose = _money_chances(legs)
    threshold = target_odds * (1 - tolerance_pct / 100.0) if target_odds is not None else None
    reach = reach_chance(legs, threshold)
    at_target = reach_chance(legs, target_odds) if target_odds is not None else reach
    return {
        "joint": {"chance": round(joint, 4), "one_in": one_in(joint), "label": chance_label(joint),
                  "fair_odds": round(fair, 2) if fair else None, "odds": round(odds, 4),
                  "profit": round(profit, 4), "lose": round(lose, 4)},
        "reach_target": {"chance": round(reach, 4), "one_in": one_in(reach),
                         "odds": round(threshold if threshold is not None else odds, 4),
                         "target": target_odds, "at_target": round(at_target, 4), "basis": "blend"},
    }


def restake(result: Mapping[str, Any], stake: Any, now: Optional[datetime.datetime] = None) -> Dict[str, Any]:
    """A built result (Result.to_dict(), e.g. from the shared cache) priced again at the person's stake.

    Only price_ticket's money changes: the legs, chances, swaps and reasons are the same at any stake.
    """
    from .tickets import TicketLeg, price_ticket
    stake = _stake(stake)
    now = _now(now)
    out = dict(result)
    tickets = []
    for ticket in result.get("tickets") or []:
        legs = []
        for v in ticket["legs"]:
            exact = v.get("inputs") or {}  # the unrounded chances (the view's own are to 4 places)
            legs.append(TicketLeg(key=v["key"], league=v["league"], event_id=v["event_id"], market=v["market"],
                                  side=v["side"], point=v["point"], price=Decimal(str(v["price"])), label=v["label"],
                                  game=v["game"], start=v["start"], ai_prob=exact.get("chance", v["ai_prob"]),
                                  market_prob=exact.get("fair", v["market_prob"]),
                                  push_prob=exact.get("push", v["push_prob"]) or 0.0, book=v["book"],
                                  group=v["group"], adjusted=v.get("adjusted")))
        priced = price_ticket(legs, ticket["type"], stake, now=now)
        merged = {**ticket, **{k: v for k, v in priced.items() if k != "legs"}}
        merged["legs"] = [{**old, **{k: v for k, v in new.items() if k != "edge"}, "ev": new.get("edge")}
                          for old, new in zip(ticket["legs"], priced["legs"])]
        tickets.append(merged)
    out["tickets"] = tickets
    out["stake"] = str(stake.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    return out


def _stake(stake: Any) -> Decimal:
    if stake is None:
        return DEFAULT_STAKE
    value = _decimal(stake, "The stake")
    if value <= 0:
        raise ValueError(f"The stake must be above zero, got {stake}")
    return value


# --- near misses, blocked, ideas ---------------------------------------------------------------------

def _near_miss(cand: Candidate, price: Decimal, book: str, failed: Sequence[str], rules: _Rules) -> Dict[str, Any]:
    reason = rules.reason(failed[0], cand, price)
    return {"ref": dict(cand.ref), "key": cand.key, "label": cand.label, "game": cand.game, "price": str(price),
            "book": book, "chance": round(cand.chance, 4), "rule": failed[0], "rules": list(failed),
            "reason": reason, "text": f"{cand.label} @ {float(price):.2f}: {reason}", "game_key": cand.game_key}


def _blocked(rules: _Rules, removed: Dict[str, int], passing: Sequence[_Leg], ctx: _Context, seen: int,
             unpriced: Mapping[str, str], lone: Mapping[str, Tuple[float, float]]) -> Dict[str, Any]:
    """Why no ticket was built, and what to relax for this kind of block (``lone``: for each leg rule, the longest
    and the shortest price among the legs that failed only that rule, so a rule is suggested only when the legs it
    removed could change the outcome)."""
    controls = ctx.controls
    low, high = _window(controls)
    games = len({leg.gbit for leg in passing})
    near = (f" near {controls.target_odds:,.1f}x" if controls.target_odds is not None
            else "" if (low <= 1.0 and math.isinf(high)) else f" at {_window_words(controls)}")
    if controls.target_chance is not None:
        near += f" at {_a_pct(controls.target_chance)}+ chance"
    shape = f"{controls.legs}-leg ticket" if controls.legs else "ticket"
    leg_rules = sorted(((rule, count) for rule, count in removed.items() if count and rule in QUALITY_RULES),
                       key=lambda item: (-item[1], QUALITY_RULES.index(item[0])))
    scope_rules = sorted(((rule, count) for rule, count in removed.items() if count and rule in SCOPE_RULES),
                         key=lambda item: (-item[1], SCOPE_RULES.index(item[0])))
    ticket_rules = sorted(((rule, count) for rule, count in ctx.rejected.items() if count),
                          key=lambda item: (-item[1], TICKET_RULES.index(item[0])))
    inside_rules = [(rule, count) for rule, count in ticket_rules if rule not in ("odds_low", "odds_high")]
    top = leg_rules[0] if leg_rules else None
    top_words = f" '{rules.rule_label(top[0])}'" if top else ""
    need = controls.legs or min(ctx.sizes)
    target = controls.target_odds is not None
    # The controls that shape a ticket, in words for "Try relaxing"
    if controls.legs is not None:
        more_legs = f"more legs than {controls.legs}" if controls.legs < MAX_LEGS else None
    else:
        more_legs = f"more legs (Auto stops at {controls.legs_max})" if controls.legs_max < MAX_LEGS else None
    fewer_legs = f"fewer legs than {need}" if need > 1 else None
    lower = "a lower target" if target else "a lower ticket-odds band"
    higher = "a higher target" if target else "a higher ticket-odds band"
    wider = f"a wider tolerance than ±{controls.tolerance_pct:g}%" if target else None
    multi_words = f"each leg of a multi {_pct(controls.multi_leg_min or 0)}+"
    relax: List[Optional[str]] = []
    multi_games = len({leg.gbit for leg in passing if leg.multi_ok})
    if seen == 0:
        message = "No prices are loaded for the games in scope yet."
        if unpriced:
            message += " Asked for them in the background: try again in a few minutes."
    elif not passing:
        lead = leg_rules or scope_rules
        parts = [f"'{rules.rule_label(rule)}' ruled out {count}" for rule, count in lead[:2]]
        message = f"No {shape}{near}: no leg passes the rules ({'; '.join(parts)})."
        relax = [rules.rule_label(rule) for rule, _ in lead[:2]]
    elif games < need:
        message = (f"No {shape}{near}: only {games} game{'s' if games != 1 else ''} pass"
                   f"{'es' if games == 1 else ''}{top_words or ' the leg rules'}.")
        relax = [rules.rule_label(rule) for rule, _ in leg_rules[:2]] + [fewer_legs]
    elif need > 1 and controls.legs is not None and multi_games < need:
        message = (f"No {shape}{near}: only {multi_games} game{'s' if multi_games != 1 else ''} "
                   f"{'have' if multi_games != 1 else 'has'} a leg passing '{multi_words}'.")
        relax = [multi_words, fewer_legs]
    else:
        def per_game(legs: Iterable[_Leg]) -> List[float]:
            """Each game's first price in this order: a ticket takes one leg per game."""
            seen_games: Set[int] = set()
            prices: List[float] = []
            for leg in legs:
                if leg.gbit not in seen_games:
                    seen_games.add(leg.gbit)
                    prices.append(leg.d)
            return prices

        longest = per_game(sorted(passing, key=lambda leg: -leg.d))
        n_top = max(ctx.sizes)
        reach = math.prod(longest[:n_top])
        shortest_legs = per_game(sorted(passing, key=lambda leg: leg.d))
        shortest_leg = shortest_legs[0]
        shortest = math.prod(shortest_legs[:min(ctx.sizes)])
        if reach < low * (1 - 1e-9):
            count = min(n_top, len(longest))
            why = f" ({top_words.strip()} ruled out {top[1]})" if top else ""
            message = (f"No {shape}{near}: the longest {count} leg{'s' if count != 1 else ''} that pass reach only "
                       f"{_odds_text(reach)}x{why}.")
            # Only the leg rules that removed legs longer than the ones counted could raise the reach
            bar = longest[n_top - 1] if len(longest) >= n_top else 1.0
            relax = [more_legs, lower] + [rules.rule_label(rule) for rule, _ in leg_rules
                                          if rule in lone and lone[rule][0] > bar + _EPS][:2]
        elif shortest > high * (1 + 1e-9):
            several = min(ctx.sizes) > 1
            message = (f"No {shape}{near}: even the shortest leg{'s' if several else ''} that pass"
                       f"{'' if several else 'es'} pay{'' if several else 's'} {_odds_text(shortest)}x.")
            relax = [fewer_legs, higher] + [rules.rule_label(rule) for rule, _ in leg_rules
                                            if rule in lone and lone[rule][1] < shortest_leg - _EPS][:2]
        elif inside_rules:
            # Combinations outside the odds window say nothing about why: name the rules that ruled out the ones
            # that could land inside it
            names = [f"'{rules.rule_label(rule)}'" for rule, _ in inside_rules[:2]]
            message = f"No {shape}{near}: {' and '.join(names)} ruled out every combination that could land there."
            rule = inside_rules[0][0]
            if rule == "whelan":
                message += " More legs aren't worth combining at these chances; set the number of legs to lift it."
            elif rule == "joint_min":
                message += f" At these odds the honest chance is about {_pct(1 / max(low, 1.0001))} at best."
            relax = [rules.rule_label(rule) for rule, _ in inside_rules[:2]]
            if rule in ("joint_min", "whelan", "caps", "three_leg", "book_cut", "min_ev", "target_chance"):
                relax.append(fewer_legs if controls.legs is not None else None)
            if rule == "joint_min" and not target:
                relax.append(lower)
        elif ticket_rules:
            message = (f"No {shape}{near}: none of the {ctx.tried:,} combinations tried of the legs that pass "
                       f"lands at {_window_words(controls)}.")
            # With a target the window is too narrow; with a band, most combinations overshot it or fell short
            over = ctx.rejected.get("odds_high", 0) > ctx.rejected.get("odds_low", 0)
            if target:
                relax = [wider]
            else:
                relax = [fewer_legs if controls.legs is not None else None, higher] if over else [more_legs, lower]
        else:
            # Nothing was even counted: name the shape rules in force
            shapes = []
            if controls.max_per_league is not None or controls.max_per_team < MAX_LEGS:
                shapes.append(rules.rule_label("caps"))
            if not target and (controls.joint_min or controls.joint_max):
                shapes.append(rules.rule_label("joint_min" if controls.joint_min else "joint_max"))
            if target and controls.tolerance_pct < 1:
                shapes.append(_window_words(controls))
            message = (f"No {shape}{near}: no combination of the legs that pass fits "
                       + (" and ".join(f"'{words}'" for words in shapes) if shapes else "every rule") + ".")
            relax = shapes + [wider if target and controls.tolerance_pct < 1 else None]
    return {
        "message": message,
        "rules": [{"rule": rule, "label": rules.rule_label(rule), "removed": count} for rule, count in leg_rules],
        "scope": [{"rule": rule, "label": rules.rule_label(rule), "removed": count} for rule, count in scope_rules],
        "ticket_rules": [{"rule": rule, "label": rules.rule_label(rule), "rejected": count}
                         for rule, count in ticket_rules],
        "passed": {"legs": len(passing), "games": games},
        "relax": list(dict.fromkeys(words for words in relax if words)),
    }


_SCRIPT = {("h2h", "home"): "home", ("h2h", "away"): "away", ("spreads", "home"): "home",
           ("spreads", "away"): "away", ("totals", "over"): "over", ("totals", "under"): "under"}


def _ideas(passing: Sequence[_Leg]) -> List[Dict[str, Any]]:
    """Same-game pairs that tend to land together (the favourite and the over), as ideas only: no book's
    same-game price is loaded, so they are never built into a ticket or priced as one."""
    by_game: Dict[int, List[_Leg]] = {}
    for leg in passing:
        by_game.setdefault(leg.gbit, []).append(leg)
    ideas = []
    for legs in by_game.values():
        sides = {_SCRIPT.get((leg.cand.market, leg.cand.side)): leg for leg in legs}
        favourite = max((leg for leg in legs if leg.cand.market == "h2h" and leg.cand.side in ("home", "away")),
                        key=lambda leg: leg.c, default=None)
        over = sides.get("over")
        if favourite is None or over is None or favourite.c < 0.55:
            continue
        ideas.append({"game": favourite.cand.game, "legs": [dict(favourite.cand.ref), dict(over.cand.ref)],
                      "labels": [favourite.cand.label, over.cand.label],
                      "text": (f"{favourite.cand.label} and {over.cand.label} tend to land together. Idea only: "
                               "no book's same-game price is loaded, so it isn't built or priced."),
                      "score": favourite.q + over.q})
    ideas.sort(key=lambda idea: -idea["score"])
    for idea in ideas:
        idea.pop("score")
    return ideas[:MAX_IDEAS]


# --- one set of rules over the candidates -------------------------------------------------------------

@dataclass
class _Mode:
    """What one search aims for. The run at the person's own rules is "weights" (the preset's score, as before
    closest effort); closest effort's objectives change the score, the chance floor and the odds window (see the
    module docstring). ``kind`` is the rules a ticket of this search is held to for its relaxed words: "plain" (the
    odds window too), "chance" or "closest" (no window: they aim at the chance, or at both targets at once)."""

    objective: str = "weights"
    weights: Optional[Weights] = None  # the pool's and the beam's order; None: the run's own
    chance_floor: Optional[float] = None  # the joint chance (every leg wins) a ticket must reach
    window: bool = True  # False: any odds
    bounds: Optional[Tuple[float, float]] = None  # an odds window of its own (closest)
    kind: str = "plain"


# The objectives' orders for the pool and the beam (each objective scores tickets its own way)
ODDS_RANK = Weights(target=0.0, ret=1.0, chance=0.0)  # the highest odds at a chance floor: legs by value
CHANCE_RANK = Weights(target=0.0, ret=0.0, chance=1.0)  # the highest chance: the likeliest legs
NEAR_RANK = Weights(target=0.01, ret=0.0, chance=1.0)  # inside the odds window, the highest chance


@dataclass
class _Prep:
    """One set of rules applied to the candidates: the legs that pass and the search's context, with what each
    rule removed (``removed``: the first rule a leg failed; ``failing``: every quality rule it failed) and the legs
    that pass but sit below the multi-leg floor (``multi_short``). The near misses only for the person's own run."""

    rules: _Rules
    ctx: _Context
    passing: List[_Leg]
    removed: Dict[str, int]
    failing: Dict[str, int]
    multi_short: int
    misses: List[Tuple[int, float, float, str, Dict[str, Any]]]
    lone: Dict[str, Tuple[float, float]]


def _prepare(candidates: Sequence[Candidate], controls: Controls, label: str, now: datetime.datetime,
             weights: Weights, mode: Optional["_Mode"] = None, loose: Optional["_Loosen"] = None,
             misses: bool = False, books: Optional[Dict[str, bool]] = None) -> _Prep:
    """The candidates through ``controls``' rules (closest effort's loosened ones with ``loose``), and the
    search's context for ``mode``. ``misses`` keeps the near misses (the person's own run); ``books`` shares the
    bookmakers' check between runs (the allowed books are never loosened)."""
    from .bets import WHELAN
    mode = mode or _Mode()
    rules = _Rules(controls, label, now, loose.value_off if loose is not None else ())
    if books is not None:
        rules._books = books
    caution = controls.caution_pp / 100.0

    removed: Dict[str, int] = {rule: 0 for rule in SCOPE_RULES + QUALITY_RULES}
    failing: Dict[str, int] = {rule: 0 for rule in QUALITY_RULES}
    near: List[Tuple[int, float, float, str, Dict[str, Any]]] = []
    lone: Dict[str, Tuple[float, float]] = {}  # leg rule -> (longest, shortest) price of the legs only it removed
    best: Dict[str, _Leg] = {}
    games: Dict[str, int] = {}
    teams: Dict[str, int] = {}
    leagues: Dict[str, int] = {}
    for cand in candidates:
        rule = rules.scope(cand)
        if rule is not None:
            removed[rule] += 1
            continue
        offer = rules.best_price(cand)
        failed = rules.quality(cand, offer[0] if offer else None)
        price, book = offer if offer else (cand.price, cand.book)
        if failed:
            for name in failed[:1]:
                removed[name] += 1
            for name in failed:
                failing[name] += 1
            if misses:
                if len(failed) == 1:
                    far, short = lone.get(failed[0], (0.0, math.inf))
                    lone[failed[0]] = (max(far, float(price)), min(short, float(price)))
                leg = _Leg(cand, price, book, caution, weights, controls.multi_leg_min)
                distance = sum(rules.distance(name, cand, price) for name in failed)
                near.append((len(failed), distance, -leg.q, cand.key, _near_miss(cand, price, book, failed, rules)))
            continue
        leg = _Leg(cand, price, book, caution, weights, controls.multi_leg_min)
        kept = best.get(leg.key)
        if kept is None or (leg.d, leg.book) > (kept.d, kept.book):
            best[leg.key] = leg  # the same pick passed in twice (one per book): keep the best price
    passing = sorted(best.values(), key=lambda leg: leg.key)
    for leg in passing:
        leg.gbit = 1 << games.setdefault(leg.cand.game_key, len(games))
        leg.teams = tuple(teams.setdefault(team.strip().lower(), len(teams)) for team in leg.cand.teams)
        leg.league = leagues.setdefault(leg.cand.league, len(leagues))

    if mode.bounds is not None:
        low, high = mode.bounds
    elif not mode.window or (loose is not None and loose.window_off):
        low, high = 1.0, math.inf
    else:
        low, high = _window(controls)
    sizes = set(loose.sizes) if loose is not None and loose.sizes else (
        {controls.legs} if controls.legs else set(range(1, controls.legs_max + 1)))
    targeted = _targeted(controls)
    asked = loose.asked if loose is not None else controls
    floor = mode.chance_floor
    # The chance objectives aim at any odds: the target odds mustn't steer their pool or their beam to its price
    aimed = controls.target_odds is not None and mode.objective not in ("odds", "chance")
    ctx = _Context(
        controls=controls, weights=weights, sizes=sizes,
        log_lo=math.log(max(low, 1.0)), log_hi=math.log(high) if math.isfinite(high) else math.inf,
        log_target=math.log(controls.target_odds) if aimed else None,
        log_jmin=(math.log(controls.joint_min) if controls.joint_min and not targeted else -math.inf),
        joint_max=controls.joint_max if not targeted else None,
        whelan_on=controls.whelan and controls.legs is None, whelan_slack=controls.whelan_slack,
        team_cap=controls.max_per_team, league_cap=controls.max_per_league,
        rejected={rule: 0 for rule in TICKET_RULES}, whelan_thresholds=tuple(WHELAN),
        pushes=any(leg.push > 0 for leg in passing), min_ev=rules.value.get("min_ev"),
        max_book_cut=rules.value.get("max_book_cut"), three_leg_min=rules.value.get("three_leg_min"),
        objective=mode.objective, log_cmin=math.log(floor) if floor else -math.inf,
        target_chance=controls.target_chance,
        close_tol=math.log(1.0 + max(asked.tolerance_pct / 100.0, MIN_CLOSE_TOL)),
        wide=loose is not None,
    )
    multi_short = sum(1 for leg in passing if not leg.multi_ok)
    return _Prep(rules=rules, ctx=ctx, passing=passing, removed=removed, failing=failing, multi_short=multi_short,
                 misses=near, lone=lone)


def _choose(legs: Sequence[_Leg], ctx: _Context, want: int, used: Set[str], budget: Optional["_Budget"] = None
            ) -> Tuple[List[Tuple[float, Tuple[_Leg, ...]]], List[Tuple[float, Tuple[str, ...]]]]:
    """Up to ``want`` tickets that share no leg (with each other or ``used``, which gains theirs), best first, and
    the first search's best tickets as leg keys (Result.ranked). With ``budget`` (closest effort's) each search's
    combinations are spent from it, and once it is spent the searches that top the tickets up are narrow ones
    (_Context.narrow)."""
    chosen: List[Tuple[float, Tuple[_Leg, ...]]] = []
    first: List[Tuple[float, Tuple[str, ...]]] = []
    pool_legs = [leg for leg in legs if leg.key not in used]
    for _ in range(want):
        tried = ctx.tried
        ranked = _search(pool_legs, ctx)
        if budget is not None:
            budget.spend(ctx.tried - tried)
        if not chosen and not first:
            first = [(score, tuple(leg.key for leg in found)) for score, found in ranked]
        more = _disjoint(ranked, used, want - len(chosen))
        if not more:
            if ctx.narrow and budget is not None:
                budget.stop()  # a narrow search found no more: a full one might have
            break
        chosen += more
        if len(chosen) >= want:
            break
        if budget is not None and budget.out:
            ctx.narrow = True
        pool_legs = [leg for leg in pool_legs if leg.key not in used]

    # A search without the used legs can find a ticket better than one picked earlier (its pool had room for legs
    # the first pool left out), so rank them again
    chosen.sort(key=lambda item: (item[0], tuple(sorted(leg.key for leg in item[1]))), reverse=True)
    return chosen, first


def _feasible(legs: Sequence[_Leg], ctx: _Context) -> Tuple[bool, Set[str]]:
    """Whether some ticket of these legs could land in the window and reach the chance floors, by bounds alone (the
    longest legs one a game, the shortest, the likeliest): a necessary condition, so a search it lets through may
    still find nothing. With the reasons it failed: "legs", "games", "reach", "floor", "chance", "joint_min" or
    "value" (the window's low end and the chance floor together ask more than the legs' value can give)."""
    if not legs:
        return False, {"legs"}
    reasons: Set[str] = set()
    for n in sorted(ctx.sizes):
        rows: Dict[int, List[float]] = {}
        for leg in legs:
            if n > 1 and not leg.multi_ok:
                continue
            row = rows.get(leg.gbit)
            value = leg.ld + leg.lc
            if row is None:
                rows[leg.gbit] = [leg.ld, leg.ld, leg.lc, leg.lw, value]
            else:
                row[0] = max(row[0], leg.ld)
                row[1] = min(row[1], leg.ld)
                row[2] = max(row[2], leg.lc)
                row[3] = max(row[3], leg.lw)
                row[4] = max(row[4], value)
        if len(rows) < n:
            reasons.add("games")
            continue
        table = list(rows.values())

        def top(i: int) -> float:
            return sum(heapq.nlargest(n, (row[i] for row in table)))

        failed = set()
        if top(0) < ctx.log_lo - _EPS:
            failed.add("reach")
        if math.isfinite(ctx.log_hi) and sum(heapq.nsmallest(n, (row[1] for row in table))) > ctx.log_hi + _EPS:
            failed.add("floor")
        if ctx.log_cmin > -math.inf and top(2) < ctx.log_cmin - _EPS:
            failed.add("chance")
        if ctx.log_jmin > -math.inf and top(3) < ctx.log_jmin - _EPS:
            failed.add("joint_min")
        if ctx.log_cmin > -math.inf and ctx.log_lo > _EPS and top(4) < ctx.log_lo + ctx.log_cmin - _EPS:
            failed.add("value")
        if not failed:
            return True, set()
        reasons |= failed
    return False, reasons


# --- closest effort ----------------------------------------------------------------------------------------

class _Budget:
    """The combinations closest effort's searches may still try (RELAX_BUDGET), what the last try cost (``last``)
    and whether the budget stopped any work (``cut``: a round not searched, a ticket not topped up). ``part`` gives
    one objective a share of what is left, so the first can't starve the next; its spending counts here too."""

    def __init__(self, combos: int, parent: Optional["_Budget"] = None) -> None:
        self.left = combos
        self.last = 0
        self.cut = False
        self.parent = parent

    def spend(self, combos: int) -> None:
        self.left -= combos
        if self.parent is not None:
            self.parent.spend(combos)

    def stop(self) -> None:
        """The budget stopped some work (here and in the budgets this is a share of)."""
        self.cut = True
        if self.parent is not None:
            self.parent.stop()

    def part(self, share: float) -> "_Budget":
        return _Budget(int(max(self.left, 0) * share), parent=self)

    @property
    def out(self) -> bool:
        return self.left <= 0

    @property
    def low(self) -> bool:
        """No round starts: the budget is spent, or what is left is less than the last try cost (the next round
        would likely go past it). The tries that must run (``force``) still do."""
        return self.left <= 0 or self.left < self.last


@dataclass
class _Attempt:
    """One try at a set of rules: what it found, whether it searched (else its bounds showed it couldn't land a
    ticket: ``reasons``) and how many tickets it wanted."""

    prep: _Prep
    chosen: List[Tuple[float, Tuple[_Leg, ...]]]
    searched: bool
    reasons: Set[str]
    mode: _Mode
    want: int = 1

    @property
    def short(self) -> bool:
        """It searched and found fewer tickets than it wanted."""
        return self.searched and len(self.chosen) < self.want

    @property
    def inside(self) -> bool:
        """It searched and a joint or value rule ruled out combinations that landed in the window."""
        return self.searched and any(self.prep.ctx.rejected.get(rule) for rule in INSIDE_RULES)

    def mends(self, reasons: Set[str], inside: bool = False) -> bool:
        """Whether more legs of a kind could help: it found too few, or its bounds failed for one of ``reasons``.
        Not while the search shows combinations in the window ruled out by a joint or value rule (those bind
        first), unless the legs are of a kind that can mend them (``inside``: shorter, likelier legs)."""
        if self.inside and not inside:
            return False
        return self.short or bool(self.reasons & reasons)


@dataclass
class _Find:
    """A ticket found: its score and legs, the run that found it (its rules and context), the rules it is held to
    for its relaxed words (_Mode.kind) and whether it came from loosened rules (else its relaxed words are none)."""

    score: float
    legs: Tuple[_Leg, ...]
    prep: _Prep
    kind: str
    loosened: bool


def _lower(value: float) -> Optional[float]:
    """A chance floor one step lower (FLOOR_STEP); None once it reaches nothing."""
    new = round(value - FLOOR_STEP, 4)
    return None if new <= _EPS else new


def _band_words(low: Optional[float], high: Optional[float]) -> str:
    if low is None and high is None:
        return "any odds"
    if high is None:
        return f"{_odds_text(low)}x+"
    return f"{_odds_text(low or 1.0)}-{_odds_text(high)}x"


def _words(rule: str, old: Any, new: Any, label: str, step: bool = False) -> str:
    """A loosened rule in plain words: "per-leg chance lowered from 60% to 55%", "odds window widened to ±20%".
    ``step`` words a round's step (the log), where the multi-leg floor is named apart from the per-leg one."""
    multi = " in a multi" if step and rule == "multi_leg_min" else ""
    if rule in ("min_chance", "multi_leg_min"):
        if not new:
            return f"per-leg chance floor{multi} ({_pct(old)}) dropped"
        return f"per-leg chance{multi} lowered from {_pct(old)} to {_pct(new)}"
    if rule == "leg_odds_min":
        return (f"leg odds minimum ({old:.2f}) dropped" if new is None
                else f"leg odds minimum lowered from {old:.2f} to {new:.2f}")
    if rule == "leg_odds_max":
        return (f"leg odds maximum ({old:.2f}) dropped" if new is None
                else f"leg odds maximum raised from {old:.2f} to {new:.2f}")
    if rule == "min_edge":
        return f"minimum edge ({old * 100:+.1f} pts) dropped"
    if rule == "joint_min":
        return f"joint chance floor ({_pct(old)}+) dropped"
    if rule == "joint_max":
        return f"joint chance cap (under {_pct(old)}) dropped"
    if rule == "whelan":
        return "the Whelan leg cap lifted"
    if rule == "three_leg":
        return f"{label}'s 3-leg rule (every leg {_pct(old)}+) dropped"
    if rule == "min_ev":
        return f"{label}'s expected-result floor ({old * 100:+.0f}%) dropped"
    if rule == "book_cut":
        return f"{label}'s cap on the bookmaker's cut ({_pct(old)}) dropped"
    if rule == "tolerance_pct":
        return "odds window dropped: the nearest odds there are" if new is None else f"odds window widened to ±{new:g}%"
    if rule == "odds_band":
        if new is None or not any(value is not None for value in new):
            return "ticket odds band dropped"
        return f"ticket odds band widened to {_band_words(*new)}"
    if rule == "legs":
        shown = " or ".join(str(n) for n in new) if isinstance(new, (list, tuple)) else str(new)
        before = " or ".join(str(n) for n in old) if isinstance(old, (list, tuple)) else str(old)
        return f"legs changed from {before} to {shown}"
    return rule


class _Loosen:
    """The person's rules as closest effort has loosened them so far, one step at a time (RELAX_ORDER)."""

    def __init__(self, asked: Controls, label: str) -> None:
        from .presets import value_rules
        self.asked, self.label = asked, label
        self.controls = asked
        self.values = value_rules(label)
        self.value_off: Set[str] = set()
        self.sizes: Optional[Set[int]] = None
        self.window_off = False
        self.leg_steps = [0, 0]  # how often leg_odds_min and leg_odds_max were widened
        self.band_steps = 0
        self.legs_steps = 0
        self.stopped = False  # the budget ended the rounds before a round that would search (_effort)

    def base_sizes(self) -> Set[int]:
        asked = self.asked
        return {asked.legs} if asked.legs else set(range(1, asked.legs_max + 1))

    def current_sizes(self) -> Set[int]:
        return set(self.sizes) if self.sizes else self.base_sizes()

    @property
    def changed(self) -> bool:
        return (self.controls is not self.asked or bool(self.value_off) or self.sizes is not None
                or self.window_off)

    def _change(self, rule: str, old: Any, new: Any) -> Dict[str, Any]:
        return {"rule": rule, "from": old, "to": new, "words": _words(rule, old, new, self.label, step=True)}

    def _set(self, changes: Sequence[Dict[str, Any]]) -> None:
        fields = {}
        for change in changes:
            rule, new = change["rule"], change["to"]
            if rule == "odds_band":
                fields["odds_min"], fields["odds_max"] = (new or [None, None])
            elif rule == "whelan":
                fields["whelan"] = bool(new)
            elif rule in ("three_leg", "min_ev", "book_cut"):
                self.value_off.add(rule)
            elif rule == "legs":
                self.sizes = set(new) if isinstance(new, (list, tuple)) else {new}
            elif rule == "tolerance_pct" and new is None:
                self.window_off = True
            else:
                fields[rule] = new
        if fields:
            self.controls = self.controls.replace(**fields)

    # One step of each group (RELAX_ORDER), or [] when that group doesn't bind or is spent -------------------
    def _floors(self, attempt: _Attempt) -> List[Dict[str, Any]]:
        # A lower floor lets in less likely (so longer) legs: no help when the odds are too high or the chance low
        if not attempt.mends(MORE_LEGS):
            return []
        c = self.controls
        out = []
        if c.min_chance and attempt.prep.failing.get("min_chance"):
            out.append(self._change("min_chance", c.min_chance, _lower(c.min_chance)))
        if c.multi_leg_min and max(self.current_sizes()) >= 2 and attempt.prep.multi_short:
            out.append(self._change("multi_leg_min", c.multi_leg_min, _lower(c.multi_leg_min)))
        return out

    def _leg_odds(self, attempt: _Attempt) -> List[Dict[str, Any]]:
        c, asked = self.controls, self.asked
        out = []
        if c.leg_odds_min is not None and attempt.prep.failing.get("leg_odds_min") \
                and attempt.mends(SHORTER_LEGS, inside=True):
            self.leg_steps[0] += 1
            k = self.leg_steps[0]
            new = round(asked.leg_odds_min / LEG_BAND_STEP ** k, 4) if k <= LEG_BAND_STEPS else None
            out.append(self._change("leg_odds_min", c.leg_odds_min, None if new is None or new < 1.01 else new))
        if c.leg_odds_max is not None and attempt.prep.failing.get("leg_odds_max") and attempt.mends(MORE_LEGS):
            self.leg_steps[1] += 1
            k = self.leg_steps[1]
            new = round(asked.leg_odds_max * LEG_BAND_STEP ** k, 4) if k <= LEG_BAND_STEPS else None
            out.append(self._change("leg_odds_max", c.leg_odds_max, None if new is None or new > 1000 else new))
        return out

    def _min_edge(self, attempt: _Attempt) -> List[Dict[str, Any]]:
        c = self.controls
        if c.min_edge is not None and attempt.prep.failing.get("min_edge") and attempt.mends(MORE_LEGS | SHORTER_LEGS):
            return [self._change("min_edge", c.min_edge, None)]
        return []

    def _joint(self, attempt: _Attempt) -> List[Dict[str, Any]]:
        c = self.controls
        if _targeted(self.asked):
            return []  # a target replaces the band
        rejected = attempt.prep.ctx.rejected
        out = []
        if c.joint_min and (rejected.get("joint_min") or "joint_min" in attempt.reasons):
            out.append(self._change("joint_min", c.joint_min, None))
        if c.joint_max is not None and rejected.get("joint_max"):
            out.append(self._change("joint_max", c.joint_max, None))
        return out

    def _value(self, attempt: _Attempt) -> List[Dict[str, Any]]:
        c = self.controls
        rejected = attempt.prep.ctx.rejected
        out = []
        if c.whelan and c.legs is None and rejected.get("whelan"):
            out.append(self._change("whelan", True, False))
        for rule, key in VALUE_RULE_KEYS.items():
            if rule not in self.value_off and self.values.get(key) is not None and rejected.get(rule):
                out.append(self._change(rule, self.values[key], None))
        return out

    def _window(self, attempt: _Attempt) -> List[Dict[str, Any]]:
        if self.window_off:
            return []
        rejected = attempt.prep.ctx.rejected
        if not (attempt.reasons & {"reach", "floor", "value"} or rejected.get("odds_low") or rejected.get("odds_high")
                or attempt.short):
            return []
        c, asked = self.controls, self.asked
        if asked.target_odds is not None:
            step = next((t for t in TOLERANCE_STEPS if t > c.tolerance_pct + 1e-9), None)
            return [] if step is None else [self._change("tolerance_pct", c.tolerance_pct, step)]
        if _targeted(asked) or (asked.odds_min is None and asked.odds_max is None):
            return []
        if c.odds_min is None and c.odds_max is None:
            return []
        self.band_steps += 1
        k = self.band_steps
        if k > len(BAND_STEPS):
            return [self._change("odds_band", [c.odds_min, c.odds_max], [None, None])]
        factor = BAND_STEPS[k - 1]
        low = round(asked.odds_min / factor, 4) if asked.odds_min is not None else None
        high = round(min(asked.odds_max * factor, MAX_TARGET_ODDS), 4) if asked.odds_max is not None else None
        return [self._change("odds_band", [c.odds_min, c.odds_max], [None if low is not None and low <= 1.0 else low,
                                                                       high])]

    def _legs(self, attempt: _Attempt) -> List[Dict[str, Any]]:
        asked = self.asked
        if asked.legs is None or self.legs_steps >= LEG_STEPS:
            return []
        self.legs_steps += 1
        k = self.legs_steps
        sizes = sorted(n for n in (asked.legs - k, asked.legs + k) if 1 <= n <= MAX_LEGS)
        old = sorted(self.current_sizes())
        return [self._change("legs", old[0] if len(old) == 1 else old, sizes)]

    def step(self, attempt: _Attempt, groups: Sequence[str]) -> List[Dict[str, Any]]:
        """Loosen the most binding rule one step: the first group in RELAX_ORDER (of ``groups``) whose counts in
        ``attempt`` show it binding (it removed legs, or ruled out tickets); when none does, the odds window, then
        the number of legs. The changes made, or [] when nothing is left to loosen."""
        for group in RELAX_ORDER:
            if group not in groups or group in ("window", "legs"):
                continue
            changes = getattr(self, "_" + group)(attempt)
            if changes:
                self._set(changes)
                return changes
        for group in ("window", "legs"):
            if group in groups:
                changes = getattr(self, "_" + group)(attempt)
                if changes:
                    self._set(changes)
                    return changes
        return []

    def fallback(self, window: bool) -> List[Dict[str, Any]]:
        """The last round, when no round found a ticket: every rule closest effort may loosen goes, and the odds
        window too (``window``: the nearest odds there are). The leg counts stay as they are (``resize``)."""
        c = self.controls
        changes = []
        for rule in ("min_chance", "multi_leg_min", "leg_odds_min", "leg_odds_max", "min_edge"):
            if getattr(c, rule) is not None and (getattr(c, rule) or rule == "min_edge"):
                changes.append(self._change(rule, getattr(c, rule), None))
        if not _targeted(self.asked):
            for rule in ("joint_min", "joint_max"):
                if getattr(c, rule):
                    changes.append(self._change(rule, getattr(c, rule), None))
        if c.whelan and c.legs is None:
            changes.append(self._change("whelan", True, False))
        for rule, key in VALUE_RULE_KEYS.items():
            if rule not in self.value_off and self.values.get(key) is not None:
                changes.append(self._change(rule, self.values[key], None))
        if window and not self.window_off:
            if self.asked.target_odds is not None:
                changes.append(self._change("tolerance_pct", c.tolerance_pct, None))
            elif not _targeted(self.asked) and (c.odds_min is not None or c.odds_max is not None):
                changes.append(self._change("odds_band", [c.odds_min, c.odds_max], [None, None]))
        self._set(changes)
        return changes

    def resize(self, sizes: Set[int]) -> List[Dict[str, Any]]:
        """The leg counts set to ``sizes`` (the last round's: the asked ones first, then 1 and 2 either way, then
        any up to the asked number); the change made, or [] when they already are."""
        if sizes == self.current_sizes():
            return []
        old, new = sorted(self.current_sizes()), sorted(sizes)
        change = self._change("legs", old[0] if len(old) == 1 else old, new[0] if len(new) == 1 else new)
        self._set([change])
        if sizes == self.base_sizes():
            self.sizes = None
        return [change]

    def last_sizes(self) -> List[Set[int]]:
        """The leg counts the last round tries in turn: the asked ones, then (an exact number of legs) 1 and 2
        either way, then any number up to the asked one; each only if it adds a count not tried yet."""
        stages = [self.base_sizes()]
        n = self.asked.legs
        if n is not None:
            stages += [{m for m in (n - k, n + k) if 1 <= m <= MAX_LEGS} for k in range(1, LEG_STEPS + 1)]
            stages.append(set(range(1, n + 1)))
        out: List[Set[int]] = []
        tried: Set[int] = set()
        for sizes in stages:
            if sizes and not sizes <= tried:
                out.append(sizes)
                tried |= sizes
        return out

    def union(self, other: "_Loosen") -> "_Loosen":
        """The looser of two sets of loosened rules, rule by rule (closest searches where either option went)."""
        out = _Loosen(self.asked, self.label)
        a, b = self.controls, other.controls

        def low(x: Optional[float], y: Optional[float]) -> Optional[float]:
            return None if not x or not y else min(x, y)

        def high(x: Optional[float], y: Optional[float]) -> Optional[float]:
            return None if x is None or y is None else max(x, y)

        fields = {"min_chance": low(a.min_chance, b.min_chance), "multi_leg_min": low(a.multi_leg_min, b.multi_leg_min),
                  "leg_odds_min": None if a.leg_odds_min is None or b.leg_odds_min is None
                  else min(a.leg_odds_min, b.leg_odds_min),
                  "leg_odds_max": high(a.leg_odds_max, b.leg_odds_max),
                  "min_edge": None if a.min_edge is None or b.min_edge is None else min(a.min_edge, b.min_edge),
                  "joint_min": low(a.joint_min, b.joint_min), "joint_max": high(a.joint_max, b.joint_max),
                  "whelan": a.whelan and b.whelan, "tolerance_pct": max(a.tolerance_pct, b.tolerance_pct)}
        out.controls = self.asked.replace(**fields)
        out.value_off = self.value_off | other.value_off
        out.window_off = self.window_off or other.window_off
        sizes = self.current_sizes() | other.current_sizes()
        out.sizes = None if sizes == out.base_sizes() else sizes
        return out


def _attempt(candidates: Sequence[Candidate], loose: _Loosen, label: str, now: datetime.datetime, weights: Weights,
             mode: _Mode, want: int, used: Set[str], budget: _Budget, books: Optional[Dict[str, bool]],
             force: bool = False) -> _Attempt:
    """One search at the loosened rules, unless its bounds show it can't land a ticket (or the budget is too low for
    it, _Budget.low, and it isn't ``force``d: a try that must run, such as the one at the person's own rules or the
    last round)."""
    prep = _prepare(candidates, loose.controls, label, now, mode.weights or weights, mode, loose, books=books)
    legs = [leg for leg in prep.passing if leg.key not in used]
    ok, reasons = _feasible(legs, prep.ctx)
    if not ok or (budget.low and not force):
        if ok:
            budget.stop()
        return _Attempt(prep, [], False, reasons if not ok else {"budget"}, mode, want)
    chosen, _ = _choose(legs, prep.ctx, want, set(used), budget)
    budget.last = prep.ctx.tried
    return _Attempt(prep, chosen, True, reasons, mode, want)


def _try(candidates: Sequence[Candidate], loose: _Loosen, label: str, now: datetime.datetime, weights: Weights,
         modes: Sequence[_Mode], want: int, used: Set[str], budget: _Budget, books: Optional[Dict[str, bool]],
         force: bool = False) -> _Attempt:
    """Each mode in turn (e.g. the chance floor, then the highest chance there is) until one finds tickets."""
    attempt = None
    for mode in modes:
        attempt = _attempt(candidates, loose, label, now, weights, mode, want, used, budget, books, force)
        if attempt.chosen:
            return attempt
    return attempt  # type: ignore[return-value]


def _effort(candidates: Sequence[Candidate], asked: Controls, label: str, now: datetime.datetime, weights: Weights,
            modes: Sequence[_Mode], want: int, used: Set[str], budget: _Budget, books: Optional[Dict[str, bool]],
            log: List[Dict[str, Any]], name: str, groups: Sequence[str], first: _Attempt,
            fallback: Optional[Sequence[_Mode]] = None, window: bool = True
            ) -> Tuple[Optional[_Attempt], _Loosen]:
    """Closest effort for one objective: from ``first`` (the try at the person's own rules), loosen the most
    binding rule one step a round (``groups``, in RELAX_ORDER) until a round finds tickets. A round whose bounds
    show it can't land a ticket costs nothing; the first that would search once the budget is too low for it
    (_Budget.low) isn't searched and ends the rounds (``loose.stopped``). With ``fallback`` (the modes to try then), a
    last round with every loosenable rule off when none finds tickets: at the asked number of legs first, then 1 and
    2 either way, then any number up to it (_Loosen.last_sizes). Every round goes in ``log``."""
    loose = _Loosen(asked, label)
    attempt = first
    rounds = 0
    done = sum(1 for entry in log if entry["objective"] == name)  # an objective's second go numbers on from its first
    while rounds < MAX_ROUNDS:
        changes = loose.step(attempt, groups)
        if not changes:
            break
        rounds += 1
        attempt = _try(candidates, loose, label, now, weights, modes, want, used, budget, books)
        log.append({"objective": name, "round": done + rounds, "changes": changes, "searched": attempt.searched,
                    "found": len(attempt.chosen)})
        if attempt.chosen:
            return attempt, loose
        if not attempt.searched and "budget" in attempt.reasons:
            loose.stopped = True
            break
    if fallback:
        dropped = loose.fallback(window)
        for i, sizes in enumerate(loose.last_sizes()):
            changes = (dropped if i == 0 else []) + loose.resize(sizes)
            rounds += 1
            attempt = _try(candidates, loose, label, now, weights, fallback, want, used, budget, books, force=True)
            log.append({"objective": name, "round": done + rounds, "changes": changes, "searched": attempt.searched,
                        "found": len(attempt.chosen), "last": True})
            if attempt.chosen:
                return attempt, loose
    return None, loose


def _strict_mode(controls: Controls) -> _Mode:
    """What the run at the person's own rules aims for: as before closest effort without a target chance; with
    one alone, the highest odds at that chance; with both, the window and the chance together."""
    if controls.target_chance is None:
        return _Mode()
    if controls.target_odds is None:
        return _Mode(objective="odds", weights=ODDS_RANK, chance_floor=controls.target_chance, kind="chance")
    return _Mode(chance_floor=controls.target_chance)


def _closest_effort(candidates: Sequence[Candidate], asked: Controls, label: str, now: datetime.datetime,
                    weights: Weights, prep: _Prep, finds: List[_Find], log: List[Dict[str, Any]], budget: _Budget
                    ) -> Tuple[List[_Find], List[Dict[str, Any]]]:
    """The person's run gave fewer tickets than asked: the closest tickets there are (the module docstring), after
    the ones found, and the labelled options when both targets were given and no ticket meets both."""
    books = prep.rules._books
    want = asked.tickets - len(finds)
    used = {leg.key for find in finds for leg in find.legs}
    strict = _strict_mode(asked)
    # The person's own run, with what its bounds say about the legs left (why it found too few)
    reasons = _feasible([leg for leg in prep.passing if leg.key not in used], prep.ctx)[1]
    first = _Attempt(prep, [], True, reasons, strict, want)
    tc, to = asked.target_chance, asked.target_odds
    common = (candidates, asked, label, now, weights)

    def found(attempt: Optional[_Attempt], kind: str, loosened: bool = True) -> List[_Find]:
        if attempt is None:
            return []
        return [_Find(score, legs, attempt.prep, kind, loosened) for score, legs in attempt.chosen]

    if tc is not None and to is not None:
        # Both targets: tickets that meet both with rules 1-5 loosened (a normal run), else the labelled options.
        # When even every one of those rules off leaves no ticket that could meet both, no round is tried
        ceiling = _Loosen(asked, label)
        ceiling.fallback(window=False)
        top = _prepare(candidates, ceiling.controls, label, now, weights, strict, ceiling, books=books)
        if _feasible([leg for leg in top.passing if leg.key not in used], top.ctx)[0]:
            attempt, _ = _effort(*common, (strict,), want, used, budget.part(1 / 3), books, log, "both", RULE_GROUPS,
                                 first)
            finds = finds + found(attempt, "plain")
        if finds:
            return finds, []
        return _options(candidates, asked, label, now, weights, books, budget, log)
    if tc is not None:
        chance = _Mode(objective="chance", weights=CHANCE_RANK, kind="chance")
        groups = tuple(group for group in RELAX_ORDER if group != "window")
        if finds:  # some reach the chance: more that do, with rules loosened
            attempt, _ = _effort(*common, (strict,), want, used, budget, books, log, "your_chance", groups, first)
            return finds + found(attempt, "chance"), []
        # None reaches it: the rules loosened until one does, else the highest chance there is (_reach_chance)
        attempt, _, loosened = _reach_chance(candidates, asked, label, now, weights, strict, chance, want, used,
                                             budget, books, log, first)
        return found(attempt, "chance", loosened), []
    near = _Mode(objective="near") if to is not None else strict
    attempt, _ = _effort(*common, (strict,), want, used, budget, books, log, "plain", RELAX_ORDER, first,
                         fallback=None if finds else (near,))
    return finds + found(attempt, "plain"), []


def _reach_chance(candidates: Sequence[Candidate], asked: Controls, label: str, now: datetime.datetime,
                  weights: Weights, floor: _Mode, chance: _Mode, want: int, used: Set[str], budget: _Budget,
                  books: Optional[Dict[str, bool]], log: List[Dict[str, Any]], first: _Attempt
                  ) -> Tuple[Optional[_Attempt], _Loosen, bool]:
    """A target chance no ticket reaches at the person's rules (``first``: that try, in the ``floor`` mode): the rules
    (never the number of legs asked) loosened one step a round until a ticket reaches it; else the highest chance
    there is with the rules loosened as far as those rounds went (only the rules that kept likelier legs or tickets
    out loosen: the leg-odds band, the minimum edge, the value rules; each ticket's relaxed words name only what it
    needs); else, when no ticket of the asked size passes even those, the highest chance with the rules and then the
    number of legs loosened, and the last round. The attempt (None if nothing at all), its rules and whether they
    were loosened."""
    common = (candidates, asked, label, now, weights)
    attempt, loose = _effort(*common, (floor,), want, used, budget, books, log, "your_chance", RULE_GROUPS, first)
    if attempt is not None:
        return attempt, loose, True
    if loose.stopped:  # the budget stopped the rounds at a round that would search: that one search still runs
        attempt = _try(candidates, loose, label, now, weights, (floor,), want, used, budget, books, force=True)
        if attempt.chosen:
            return attempt, loose, True
    likeliest = _try(candidates, loose, label, now, weights, (chance,), want, used, budget, books, force=True)
    if likeliest.chosen:
        return likeliest, loose, loose.changed
    groups = tuple(group for group in RELAX_ORDER if group != "window")
    attempt, loose = _effort(*common, (chance,), want, used, budget, books, log, "your_chance", groups, likeliest,
                             (chance,), False)
    return attempt, loose, True


def _options(candidates: Sequence[Candidate], asked: Controls, label: str, now: datetime.datetime, weights: Weights,
             books: Optional[Dict[str, bool]], budget: _Budget, log: List[Dict[str, Any]]
             ) -> Tuple[List[_Find], List[Dict[str, Any]]]:
    """Both targets and no ticket meets both: up to three labelled options (OPTION_KEYS), each the best for its own
    aim with the rules loosened only as far as it needed, the same ticket picked twice kept once under its first
    label."""
    tc, to = asked.target_chance, asked.target_odds
    common = (candidates, asked, label, now, weights)
    picks: List[Tuple[str, _Find]] = []
    looses: List[_Loosen] = []
    # Each option's rounds have their own share of what is left of the budget (the first can't starve the second)
    odds_budget = budget.part(1 / 2)

    # Likeliest at your odds: inside the odds window (widened if need be), the highest chance. Its try at the person's
    # own rules always runs (one search), then its rounds
    near = _Mode(weights=NEAR_RANK)
    start = _try(candidates, _Loosen(asked, label), label, now, weights, (near,), 1, set(), odds_budget, books,
                 force=True)
    if start.chosen:
        picks.append(("your_odds", _Find(*start.chosen[0], start.prep, "plain", False)))
    else:
        attempt, loose = _effort(*common, (near,), 1, set(), odds_budget, books, log, "your_odds", RELAX_ORDER, start,
                                 (_Mode(objective="near", weights=NEAR_RANK),), True)
        if attempt is not None:
            picks.append(("your_odds", _Find(*attempt.chosen[0], attempt.prep, "plain", True)))
            looses.append(loose)
    # Closest to your chance: the chance at the highest odds, with the rules (never the number of legs) loosened as
    # far as that needs; else the highest chance there is (any odds: _reach_chance)
    floor = _Mode(objective="odds", weights=ODDS_RANK, chance_floor=tc, window=False, kind="chance")
    chance = _Mode(objective="chance", weights=CHANCE_RANK, window=False, kind="chance")
    chance_budget = budget.part(1)
    start = _try(candidates, _Loosen(asked, label), label, now, weights, (floor,), 1, set(), chance_budget, books,
                 force=True)
    if start.chosen:
        picks.append(("your_chance", _Find(*start.chosen[0], start.prep, "chance", False)))
    else:
        attempt, loose, loosened = _reach_chance(candidates, asked, label, now, weights, floor, chance, 1, set(),
                                                 chance_budget, books, log, start)
        if attempt is not None and attempt.chosen:
            picks.append(("your_chance", _Find(*attempt.chosen[0], attempt.prep, "chance", loosened)))
            if loosened:
                looses.append(loose)
    if picks:
        # The best balance: the smallest distance to both targets of any ticket there is, every rule closest effort
        # may loosen off (but the window and the leg counts the options used; its relaxed words name only what it
        # needed). No ticket further from the target odds than the options already found can be closer, which
        # bounds the window
        loose = _Loosen(asked, label)
        for other in looses:
            loose = loose.union(other)
        changes = loose.fallback(window=False)
        tol = max(asked.tolerance_pct / 100.0, MIN_CLOSE_TOL)

        def distance(find: _Find) -> float:
            """closest's d (as _Context.distance) for a ticket found by any objective."""
            ld, lc = sum(leg.ld for leg in find.legs), sum(leg.lc for leg in find.legs)
            return abs(ld - math.log(to)) / math.log(1.0 + tol) + abs(math.exp(lc) - tc) / CHANCE_UNIT

        reach = min(distance(find) for _, find in picks)
        span = math.exp(min(reach * math.log(1.0 + tol), math.log(MAX_TARGET_ODDS)))
        bounds = (max(to / span, 1.0), min(to * span, MAX_TARGET_ODDS))
        mode = _Mode(objective="closest", weights=NEAR_RANK, bounds=bounds, kind="closest")
        attempt = _try(candidates, loose, label, now, weights, (mode,), 1, set(), budget, books, force=True)
        log.append({"objective": "closest", "round": 1, "changes": changes, "searched": attempt.searched,
                    "found": len(attempt.chosen), "last": True})
        pool = [find for _, find in picks]
        if attempt.chosen:
            pool.append(_Find(*attempt.chosen[0], attempt.prep, "closest", loose.changed))
        best = min(enumerate(pool), key=lambda item: (round(distance(item[1]), 9), item[0]))[1]
        picks.append(("closest", best))

    finds: List[_Find] = []
    options: List[Dict[str, Any]] = []
    seen: List[frozenset] = []
    for key, find in picks:
        keys = frozenset(leg.key for leg in find.legs)
        if keys in seen:
            continue  # the same ticket as an earlier option: merged, keeping the first label
        seen.append(keys)
        finds.append(find)
        options.append({"key": key, "label": OPTION_LABELS[key], "ticket_index": len(finds) - 1})
    return finds, options


# --- what each ticket says about the ask ---------------------------------------------------------------

def ticket_fit(odds: Any, chance: float, target_odds: Optional[float] = None, target_chance: Optional[float] = None,
               tolerance_pct: float = 10.0) -> Optional[Dict[str, Any]]:
    """How close a ticket gets to what was asked: {odds, chance, target_odds, target_chance, odds_gap_pct,
    chance_gap_pp, meets: {odds, chance}}, or None when neither target was given.

    ``odds`` is the ticket's odds (a Decimal product of its prices gives price_ticket's own two places; a float is
    rounded half up), ``chance`` the chance every leg wins. ``odds_gap_pct`` is how far the odds are from the
    target in % (signed), ``chance_gap_pp`` the chance's in points (signed); ``meets.odds`` is whether the odds are
    inside the target +/- ``tolerance_pct``, ``meets.chance`` whether the chance is at least the target. A target not
    given has None in its places."""
    if target_odds is None and target_chance is None:
        return None
    exact = float(odds)
    shown = Decimal(str(odds)) if not isinstance(odds, Decimal) else odds
    chance = float(chance)
    tol = float(tolerance_pct) / 100.0
    return {
        "odds": float(shown.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
        "chance": round(chance, 4),
        "target_odds": target_odds,
        "target_chance": target_chance,
        "odds_gap_pct": None if target_odds is None else round((exact / target_odds - 1.0) * 100.0, 1),
        "chance_gap_pp": None if target_chance is None else round((chance - target_chance) * 100.0, 1),
        "meets": {
            "odds": None if target_odds is None
            else target_odds * (1 - tol) * (1 - 1e-9) <= exact <= target_odds * (1 + tol) * (1 + 1e-9),
            "chance": None if target_chance is None else chance >= target_chance - 1e-9,
        },
    }


def _fit(legs: Sequence[_Leg], asked: Controls) -> Optional[Dict[str, Any]]:
    return ticket_fit(math.prod((leg.price for leg in legs), start=Decimal(1)), math.prod(leg.c for leg in legs),
                      asked.target_odds, asked.target_chance, asked.tolerance_pct)


def _relaxed(legs: Sequence[_Leg], asked: Controls, label: str, kind: str) -> List[Dict[str, Any]]:
    """The rules this ticket breaks of the person's own, each at the tightest step of closest effort that admits it
    ([{rule, from, to, words}]): a ticket whose weakest leg is 53% under a 72% floor says "per-leg chance lowered
    from 72% to 52%" however far the round went. ``kind`` "plain" holds it to the odds window too (the chance and
    closest objectives aim at any odds)."""
    from .bets import WHELAN
    from .presets import value_rules
    out: List[Dict[str, Any]] = []

    def add(rule: str, old: Any, new: Any) -> None:
        out.append({"rule": rule, "from": old, "to": new, "words": _words(rule, old, new, label)})

    n = len(legs)
    chances = [leg.c for leg in legs]
    prices = [leg.d for leg in legs]
    # Per-leg chance: a multi's legs need the higher of the two floors
    base = asked.min_chance or 0.0
    multi = (asked.multi_leg_min or 0.0) if n >= 2 else 0.0
    floor = max(base, multi)
    weakest = min(chances)
    if floor > 0 and weakest < floor - _EPS:
        steps = math.ceil(round((floor - weakest) / FLOOR_STEP, 9))
        to = round(floor - steps * FLOOR_STEP, 4)
        add("multi_leg_min" if n >= 2 and multi >= base else "min_chance", floor, None if to <= _EPS else to)
    # The leg-odds band, side by side
    low = asked.leg_odds_min
    if low is not None and min(prices) < low - _EPS:
        to = next((step for step in (round(low / LEG_BAND_STEP ** k, 4) for k in range(1, LEG_BAND_STEPS + 1))
                   if step >= 1.01 and step <= min(prices) + _EPS), None)
        add("leg_odds_min", low, to)
    high = asked.leg_odds_max
    if high is not None and max(prices) > high + _EPS:
        to = next((step for step in (round(high * LEG_BAND_STEP ** k, 4) for k in range(1, LEG_BAND_STEPS + 1))
                   if step <= 1000 and step >= max(prices) - _EPS), None)
        add("leg_odds_max", high, to)
    if asked.min_edge is not None and any(leg.edge is None or leg.edge < asked.min_edge - _EPS for leg in legs):
        add("min_edge", asked.min_edge, None)
    joint = math.prod(chances)
    if not _targeted(asked):
        if any(leg.push > 0 for leg in legs):
            wins = math.prod(leg.c + leg.push for leg in legs) - math.prod(leg.push for leg in legs)
        else:
            wins = joint
        if asked.joint_min and wins < asked.joint_min - _EPS:
            add("joint_min", asked.joint_min, None)
        if asked.joint_max is not None and wins >= asked.joint_max - 1e-12:
            add("joint_max", asked.joint_max, None)
    # The value rules: the Whelan cap on Auto, and while the run is exactly a preset its old tier's rules
    if asked.whelan and asked.legs is None and n >= 2 and joint > 0:
        worth = 1 + sum(1 for threshold in WHELAN if joint ** (1 / n) > threshold)
        if n > worth + asked.whelan_slack:
            add("whelan", True, False)
    value = value_rules(label)
    three = value.get("three_leg_min")
    if three is not None and n >= 3 and any(c < three - _EPS for c in chances):
        add("three_leg", three, None)
    floor_ev = value.get("min_ev")
    if floor_ev is not None and math.exp(sum(leg.le for leg in legs)) - 1 < floor_ev - _EPS:
        add("min_ev", floor_ev, None)
    cut = value.get("max_book_cut")
    if cut is not None and all(leg.lb is not None for leg in legs) \
            and 1 - math.exp(sum(leg.lb for leg in legs)) > cut + _EPS:
        add("book_cut", cut, None)
    # The odds window: the target's tolerance, else the preset's band
    odds = math.prod(prices)
    if kind == "plain":
        target = asked.target_odds
        if target is not None:
            def inside(pct: float) -> bool:
                return target * (1 - pct / 100.0) * (1 - _EPS) <= odds <= target * (1 + pct / 100.0) * (1 + _EPS)

            if not inside(asked.tolerance_pct):
                add("tolerance_pct", asked.tolerance_pct,
                    next((pct for pct in TOLERANCE_STEPS if pct > asked.tolerance_pct + 1e-9 and inside(pct)), None))
        elif not _targeted(asked) and (asked.odds_min is not None or asked.odds_max is not None):
            def banded(factor: float) -> Tuple[Optional[float], Optional[float]]:
                lo = round(asked.odds_min / factor, 4) if asked.odds_min is not None else None
                hi = round(min(asked.odds_max * factor, MAX_TARGET_ODDS), 4) if asked.odds_max is not None else None
                return (None if lo is not None and lo <= 1.0 else lo), hi

            def within(band: Tuple[Optional[float], Optional[float]]) -> bool:
                return (band[0] is None or odds >= band[0] * (1 - _EPS)) and (band[1] is None
                                                                             or odds <= band[1] * (1 + _EPS))

            if not within((asked.odds_min, asked.odds_max)):
                band = next((banded(f) for f in BAND_STEPS if within(banded(f))), None)
                add("odds_band", [asked.odds_min, asked.odds_max], list(band) if band else [None, None])
    if asked.legs is not None and n != asked.legs:
        add("legs", asked.legs, n)
    return out


def _within(relaxed: Sequence[Mapping[str, Any]], words: Sequence[Mapping[str, Any]]) -> bool:
    """Whether a ticket relaxed as ``relaxed`` (_relaxed) breaks the person's rules no further than one relaxed as
    ``words``: every rule it loosens is one ``words`` loosens, at a step no looser."""
    named = {entry["rule"]: entry["to"] for entry in words}
    for entry in relaxed:
        rule, new = entry["rule"], entry["to"]
        if rule not in named:
            return False
        old = named[rule]
        if new == old or old is None or rule not in LOOSER:
            continue  # the same step, the rule dropped there, or a rule that is only ever dropped
        if new is None:
            return False
        if rule == "odds_band":
            if (old[0] is not None and (new[0] is None or new[0] < old[0])) or \
                    (old[1] is not None and (new[1] is None or new[1] > old[1])):
                return False
        elif LOOSER[rule] * (new - old) > _EPS:
            return False
    return True


def _keeper(asked: Controls, label: str, kind: str, words: List[Dict[str, Any]]
            ) -> Callable[[Sequence[_Leg]], bool]:
    """Whether a swapped ticket's legs break the person's rules no further than ``words`` (a ticket's relaxed)."""
    return lambda legs: _within(_relaxed(legs, asked, label, kind), words)


# --- build ---------------------------------------------------------------------------------------------

def build(candidates: Iterable[Candidate], controls: Controls, now: Optional[datetime.datetime] = None, *,
          stake: Any = None, unpriced: Optional[Mapping[str, str]] = None, weights: Optional[Weights] = None
          ) -> Result:
    """Tickets for these controls from these candidates (see the module docstring).

    ``stake`` prices the finalists (10 when None). ``unpriced`` maps a league in scope that has no prices
    loaded to the words for it ("No prices loaded for EPL; asked for them in the background"); each becomes a
    near miss (on top of the up to MAX_NEAR_MISSES legs). ``weights`` replaces the preset's ranking weights
    (engine/target.py and engine/bets.py rank by the cautious expected return alone: RETURN_ONLY); the result's
    ``ranked`` then holds the first search's best tickets as leg keys, best first, for a caller to pick from by
    its own rules. Raises ValueError only for a bad stake.

    With ``controls.best_effort`` (the default) a run whose rules give fewer tickets than asked adds the closest
    ones closest effort finds (the module docstring); the near misses, ``blocked``, ``ranked`` and the counts in
    ``stats`` stay those of the person's own rules (``stats["relaxation"]`` holds every round of loosening).
    """
    from .presets import label_for, weights_for
    started = time.perf_counter()
    stake = _stake(stake)
    now = _now(now)
    candidates = list(candidates)
    label = label_for(controls)
    weights = weights or weights_for(controls)
    mode = _strict_mode(controls)
    prep = _prepare(candidates, controls, label, now, mode.weights or weights, mode, misses=True)
    ctx, passing = prep.ctx, prep.passing
    chosen, first = _choose(passing, ctx, controls.tickets, set())
    finds = [_Find(score, legs, prep, mode.kind, False) for score, legs in chosen]
    options: List[Dict[str, Any]] = []
    log: List[Dict[str, Any]] = []
    budget = _Budget(RELAX_BUDGET)
    if controls.best_effort and len(finds) < controls.tickets:
        finds, options = _closest_effort(candidates, controls, label, now, weights, prep, finds, log, budget)

    taken = {leg.key for find in finds for leg in find.legs}
    tickets = []
    for i, find in enumerate(finds):
        relaxed = _relaxed(find.legs, controls, label, find.kind) if find.loosened else []
        # A loosened search's swaps keep its own rules, which may be looser than the ticket needed (Best balance and
        # the last round have every floor off): a swap stays within the steps the ticket's words name
        keep = _keeper(controls, label, find.kind, relaxed) if find.loosened else None
        ticket = _ticket(i + 1, find.score, find.legs, find.prep.ctx, label, stake, now, find.prep.passing, taken,
                         controls, keep)
        ticket["fit"] = _fit(find.legs, controls)
        ticket["relaxed"] = relaxed
        tickets.append(ticket)

    # Leagues with no prices loaded come first (a whole league is missing; they don't take the legs' places), then
    # up to MAX_NEAR_MISSES legs that nearly made it: the closest to passing each rule that ruled legs out (so every
    # such rule shows), then the rest by fewest rules failed, the closest to passing them (Rules.distance) and the
    # best by score; at most MAX_MISSES_PER_GAME a game
    near_misses: List[Dict[str, Any]] = [
        {"ref": None, "key": None, "label": league, "game": None, "price": None, "book": None, "chance": None,
         "rule": "no_prices", "rules": ["no_prices"], "reason": words, "text": words, "league": league}
        for league, words in sorted((unpriced or {}).items())]
    misses = prep.misses
    misses.sort(key=lambda item: item[:4])
    picked: List[int] = []
    listed: Set[str] = {leg.key for leg in passing}  # a pick that passed at another book is no near miss
    per_game: Dict[str, int] = {}
    for first_of_rule in (True, False):
        rules_shown: Set[str] = set()
        for i, (*_, miss) in enumerate(misses):
            if len(picked) >= MAX_NEAR_MISSES:
                break
            game = miss["game_key"]
            if miss["key"] in listed or per_game.get(game, 0) >= MAX_MISSES_PER_GAME:
                continue
            if first_of_rule and miss["rule"] in rules_shown:
                continue
            rules_shown.add(miss["rule"])
            listed.add(miss["key"])
            per_game[game] = per_game.get(game, 0) + 1
            picked.append(i)
    near_misses += [{k: v for k, v in misses[i][-1].items() if k != "game_key"} for i in sorted(picked)]

    notes = []
    if label == "Crazy":
        notes.append("Crazy is long shots: expect to lose most of these.")
    if not controls.avoid_same_game or controls.max_per_game > 1:
        notes.append("Each ticket still takes one leg per game: same-game legs are linked, and without a book's "
                     "own same-game price they can't be priced honestly.")
    if not controls.exclude_started:
        notes.append("Games already under way are left out: their prices are from before the start.")
    if controls.target_odds is not None and (controls.joint_min or controls.joint_max or controls.odds_min
                                             or controls.odds_max):
        notes.append("The target sets the ticket's odds; the preset's own odds and chance band don't apply.")
    elif controls.target_chance is not None and (controls.joint_min or controls.joint_max or controls.odds_min
                                                 or controls.odds_max):
        notes.append("The target chance sets the ticket; the preset's own odds and chance band don't apply.")
    if controls.prefer_correlation:
        notes.append("Positive correlation is for ideas only: no same-game ticket is built without a book's own "
                     "same-game price, which isn't loaded.")
    if options:
        notes.append("No one ticket meets both your odds and your chance: each option is the closest for its aim, "
                     "with its real odds and chance.")
    elif tickets and len(tickets) < controls.tickets and budget.cut:
        notes.append(f"Only {len(tickets)} ticket{'s' if len(tickets) != 1 else ''} found within the search's time "
                     f"limit (you asked for {controls.tickets}).")
    elif tickets and len(tickets) < controls.tickets:
        notes.append(f"Only {len(tickets)} ticket{'s' if len(tickets) != 1 else ''} sharing no leg fit"
                     f"{'s' if len(tickets) == 1 else ''} the rules (you asked for {controls.tickets}).")
    if not options and any(t["relaxed"] for t in tickets):
        notes.append("Nothing met every rule exactly, so the closest tickets are shown, each with the rules it "
                     "relaxed.")
    if controls.target_chance is not None and controls.target_odds is None and tickets \
            and not any(t["fit"]["meets"]["chance"] for t in tickets):
        notes.append(f"No ticket reaches {_a_pct(controls.target_chance)} chance: these are the likeliest there are.")
    if any(t["model_priced"] for t in tickets):
        notes.append("Some legs are priced at fair odds by the goals model; check your bookmaker's price.")

    target = None
    if controls.target_odds is not None:
        low, high = _window(controls)
        target = {"odds": controls.target_odds, "low": round(low, 4), "high": round(high, 4),
                  "tolerance_pct": controls.tolerance_pct}
    blocked = None if tickets else _blocked(prep.rules, prep.removed, passing, ctx, len(candidates), unpriced or {},
                                            prep.lone)
    stats = {"candidates": len(candidates), "passed": len(passing),
             "games": len({leg.gbit for leg in passing}), "combos": ctx.tried,
             "removed": {rule: count for rule, count in prep.removed.items() if count},
             "rejected": {rule: count for rule, count in ctx.rejected.items() if count},
             "ms": round((time.perf_counter() - started) * 1000, 1)}
    if log:
        stats["relaxation"] = log
    return Result(label=label, controls=controls, tickets=tickets, near_misses=near_misses, blocked=blocked,
                  stats=stats, notes=notes, ideas=_ideas(passing) if controls.prefer_correlation else [],
                  target=target, stake=str(stake.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)), ranked=first,
                  options=options)


# --- the likeliest winners -----------------------------------------------------------------------------

def top_picks(candidates: Iterable[Candidate], n: int = 10, markets: Iterable[str] = ("h2h",),
              min_price: Optional[float] = None, *, now: Optional[datetime.datetime] = None,
              max_price_age_s: Optional[int] = None) -> List[Dict[str, Any]]:
    """The likeliest winners: single legs in ``markets`` by the brain's chance, highest first (ties by the edge
    against the price), one a game, at most ``n``. Each pick is {ref, key, league, label, game, start, chance,
    chance_source, price, book, edge, why}, priced at the best price book (never a SHARP_BOOKS book or an
    exchange, never a leg the goals model prices at fair odds or a game under way); ``min_price`` leaves out
    shorter prices. With ``now`` a game that has started is left out, with ``max_price_age_s`` an older price (the
    caller's own checks may do both)."""
    wanted = {str(market).strip().lower() for market in (markets or ("h2h",))}
    when = _now(now) if now is not None else None
    rows = []
    books: Dict[str, bool] = {}
    for cand in candidates:
        if cand.market not in wanted or "live" in cand.flags or "model_price" in cand.flags or cand.chance <= 0.001:
            continue
        if max_price_age_s is not None and cand.price_age_s > max_price_age_s:
            continue
        if when is not None:
            start = _parse_time(cand.start)
            if start is None or start <= when:
                continue
        offers = dict(cand.prices or {})
        offers.setdefault(cand.book, cand.price)
        usable = []
        for book, price in offers.items():
            if book not in books:
                books[book] = _price_book(book)
            if books[book]:
                usable.append((price, book))
        if not usable:
            continue
        price, book = max(usable, key=lambda pair: (pair[0], pair[1] == cand.book, pair[1]))
        if min_price is not None and float(price) < float(min_price) - _EPS:
            continue
        rows.append((cand, price, book, edge_of(cand.chance, cand.push, price)))
    rows.sort(key=lambda row: (-row[0].chance, -row[3], row[0].key))
    picks: List[Dict[str, Any]] = []
    games: Set[str] = set()
    for cand, price, book, edge in rows:
        if len(picks) >= max(0, int(n)):
            break
        if cand.game_key in games:
            continue
        games.add(cand.game_key)
        picks.append({"ref": dict(cand.ref), "key": cand.key, "league": cand.league, "label": cand.label,
                      "game": cand.game, "start": cand.start, "chance": round(cand.chance, 4),
                      "chance_source": cand.chance_source, "price": str(price), "book": book, "edge": round(edge, 4),
                      "why": cand.why[:3], **_why_units(cand)})
    return picks


def market_catalog(soccer: bool = True) -> Dict[str, Dict[str, Any]]:
    """Every market the builder knows, with whether it can be picked and why not (for the presets route)."""
    out: Dict[str, Dict[str, Any]] = {}
    for market in MARKETS:
        reason = None
        available = True
        if market in SOCCER_MARKETS:
            reason = "Soccer only, from the goals model (priced at fair odds where no book prices it)"
            available = soccer
        out[market] = {"title": MARKET_TITLES[market], "available": available, "reason": reason}
    for market, (title, reason) in UNAVAILABLE.items():
        out[market] = {"title": title, "available": False, "reason": reason}
    return out
