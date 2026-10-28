"""AI-built bets at three risk levels: Safe, Risky and Crazy.

The tier is set by the AI's chance that the whole ticket wins:

- Safe: 60%+ (1-2 legs, or 3 if every leg is 85%+), paying 1.15x-1.80x.
- Risky: 25-60% (1-3 legs), paying 1.50x-4.50x.
- Crazy: 3-25% (1-5 legs; legs of a multi 34%+, or a single long shot of
  8-25%), paying 3.40x-36x.

Since v3 Stage 3 the search is the Builder's (engine/builder.py): each tier
is its preset (engine/presets.py: the same windows, leg floors, Whelan caps
and value rules as before), searched by the cautious expected return alone
(builder.RETURN_ONLY, not the preset's own weights), and these tickets are
the best the builder found under those controls, ranked here as below. What
stays here is what makes them the AI's daily bets:

Every candidate leg is one outcome of one game (moneyline, spread or total)
priced at a bookmaker whose margin on that market is 5% or less, and only
at books a person can bet at (never Pinnacle or an exchange: those anchor
the markets brain's consensus only). A ticket
takes at most one leg per game, so its legs are independent (checked on NBA
data: game-level correlation is about zero), and it is priced at a single
bookmaker that offers every leg, as it would have to be placed: the builder
searches at each leg's best price, and each ticket it finds is placed at the
bookmaker offering all its legs at the best combined price (a ticket no one
bookmaker offers is left out). Legs the AI
rates below the market are never used, and Safe skips legs where the AI is
more than 8 points from the market.

Tickets are ranked by expected value at a cautious probability (the AI's
minus 2 points per leg), then by chance to win, fewer legs and a lower
bookmaker edge, and the top three per tier are kept with no two sharing
more than one leg. Whelan (2026) caps how many legs are worth combining for
a given leg strength; Crazy may go two past it, flagged. The thresholds and
the method follow the risk research in research_notes (sections 5.2-5.7).

Nothing here is a tip: unless the AI has a proven record against the
closing line, every ticket is labelled "For fun", with its expected cost.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import builder
from .odds_math import decimal_to_american, format_american

MAX_HOLD = 0.05  # skip a bookmaker whose margin on the market is higher
MIN_LEG = 0.08  # the AI's chance below which a leg is never used (that bucket loses ~18%)
CHECK_NEWS_GAP = 0.05  # the AI this far from the market: "check the injury news"
SAFE_MAX_GAP = 0.08
# Geometric-mean leg chance above which one more leg is worth adding (Whelan 2026): 2 legs need
# legs above 1/3, 3 legs above 0.525 ...
WHELAN = (0.333, 0.525, 0.633, 0.701, 0.748, 0.782, 0.808, 0.829, 0.845)


@dataclass
class Candidate:
    event_id: str
    game: str  # "Away @ Home"
    market: str
    side: str
    point: Optional[float]
    label: str
    probability: float  # the AI's chance the leg wins (a push is not a win)
    fair: Optional[float]  # the bookmakers' margin-free chance
    prices: Dict[str, float]  # bookmaker -> decimal price at this line
    push: float = 0.0
    sigma: float = 0.02  # model uncertainty, taken off for the cautious chance
    start: Optional[str] = None  # ISO start time (the builder leaves out games under way); None: still to come

    @property
    def cautious(self) -> float:
        return max(self.probability - self.sigma, 0.0)

    @property
    def gap(self) -> Optional[float]:
        return None if self.fair is None else self.probability - self.fair

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.event_id, self.market, self.side)


@dataclass(frozen=True)
class Tier:
    key: str
    title: str
    subtitle: str
    p_low: float
    p_high: float
    max_legs: int
    single_min: float
    multi_leg_min: float
    ev_floor: float
    edge_cap: float  # the bookmaker's edge on the ticket
    odds_low: float
    odds_high: float
    whelan_slack: int
    per_game: int  # best legs per game kept for the search


TIERS: Tuple[Tier, ...] = (
    Tier("safe", "Safe", "More likely, small payout. It still loses some of the time.",
         0.60, 1.01, 3, 0.60, 0.72, -0.05, 0.09, 1.15, 1.80, 0, 3),
    Tier("risky", "Risky", "Balanced: a real chance and a real payout.",
         0.25, 0.60, 3, 0.25, 0.45, -0.10, 0.14, 1.50, 4.50, 0, 3),
    Tier("crazy", "Crazy", "Long shot, big payout: expect to lose most of these.",
         0.03, 0.25, 5, 0.08, 0.34, -0.15, 0.22, 3.40, 36.0, 2, 2),
)


def whelan_max_legs(geometric_mean: float) -> int:
    return 1 + sum(1 for threshold in WHELAN if geometric_mean > threshold)


@dataclass
class Ticket:
    legs: List[Candidate]
    book: str

    @property
    def prices(self) -> List[float]:
        return [leg.prices[self.book] for leg in self.legs]

    @property
    def decimal(self) -> float:
        return math.prod(self.prices)

    @property
    def probability(self) -> float:
        return math.prod(leg.probability for leg in self.legs)

    @property
    def fair_probability(self) -> Optional[float]:
        fairs = [leg.fair for leg in self.legs]
        return math.prod(fairs) if all(f is not None for f in fairs) else None

    def _expected_return(self, chances: Sequence[float]) -> float:
        # A pushed leg drops out of a multi (its price counts as 1)
        return math.prod(p * d + leg.push for p, d, leg in zip(chances, self.prices, self.legs))

    @property
    def ev(self) -> float:
        return self._expected_return([leg.probability for leg in self.legs]) - 1

    @property
    def cautious_ev(self) -> float:
        return self._expected_return([leg.cautious for leg in self.legs]) - 1

    @property
    def book_edge(self) -> Optional[float]:
        fair = self.fair_probability
        return None if fair is None else 1 - fair * self.decimal

    @property
    def max_gap(self) -> float:
        return max(abs(leg.gap or 0.0) for leg in self.legs)


def _leg_allowed(tier: Tier, leg: Candidate, legs: int) -> bool:
    if tier.key == "safe" and abs(leg.gap or 0.0) > SAFE_MAX_GAP:
        return False
    return leg.probability >= (tier.single_min if legs == 1 else tier.multi_leg_min)


def _fits(tier: Tier, ticket: Ticket) -> bool:
    n = len(ticket.legs)
    if not all(_leg_allowed(tier, leg, n) for leg in ticket.legs):
        return False
    if tier.key == "safe" and n == 3 and min(leg.probability for leg in ticket.legs) < 0.85:
        return False
    p = ticket.probability
    edge = ticket.book_edge
    return (tier.p_low <= p < tier.p_high
            and tier.odds_low <= ticket.decimal <= tier.odds_high
            and ticket.ev >= tier.ev_floor
            and (edge is None or edge <= tier.edge_cap)
            and n <= whelan_max_legs(p ** (1 / n)) + tier.whelan_slack)


def _rank(ticket: Ticket) -> Tuple[float, ...]:
    edge = ticket.book_edge
    return (round(ticket.cautious_ev, 6), ticket.probability, -len(ticket.legs),
            -(edge if edge is not None else 1.0), min(leg.probability for leg in ticket.legs), -ticket.max_gap)


BOARD_MARKETS = {"h2h", "spreads", "totals"}
SEARCH_TICKETS = builder.MAX_TICKETS  # tickets asked of the builder per bookmaker, before the tier's own checks


def tier_controls(tier: Tier, best_effort: bool = False) -> builder.Controls:
    """The builder's controls for a tier: its preset (engine/presets.py, the tier's windows, leg floors, Whelan cap
    and value rules; still labelled the tier), searching every board market at any start time for as many tickets
    as the builder gives. These legs carry no price age (the board is the latest), so the preset's price-age rule
    never removes one. Strict (the tier's own tickets, as the AI record logs them) unless ``best_effort``: closest
    effort, for a level nothing fits exactly (closest_tier)."""
    from .presets import preset
    return replace(preset(tier.key), markets=set(BOARD_MARKETS), tickets=SEARCH_TICKETS, window=None,
                   best_effort=best_effort)


def _as_candidate(leg: Candidate, now: datetime.datetime, only: Optional[str] = None) -> builder.Candidate:
    """One leg as the builder's Candidate: the AI's chance, the market's fair chance and every price book's price
    (the builder searches at each leg's best price), or only ``only``'s."""
    away, _, home = leg.game.partition(" @ ")
    start = leg.start or (now + datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    source = "books" if leg.fair is not None and abs(leg.probability - leg.fair) < 1e-12 else "model"
    prices = {book: Decimal(str(price)) for book, price in leg.prices.items()
              if price_book(book) and price > 1 and (only is None or book == only)}
    book = max(prices, key=lambda name: (prices[name], name))
    return builder.Candidate(
        ref={"league": "ai_bets", "id": leg.event_id, "market": leg.market, "side": leg.side, "point": leg.point},
        league="ai_bets", game_key=leg.event_id, teams=(away or leg.game, home or leg.game), start=start,
        price=prices[book], book=book, prices=prices, chance=leg.probability, chance_source=source,
        label=leg.label, fair=leg.fair, push=leg.push)


def _search(tier: Tier, legs: Sequence[Candidate], now: Optional[datetime.datetime] = None) -> List[Ticket]:
    """The builder's tickets under the tier's controls (tier_controls), each placed at the bookmaker that offers all
    its legs at the best combined price and checked against the tier there (_fits: the chance and odds windows, the
    leg floors, the expected value floor and the bookmaker's edge). When the builder's best tickets mix legs no one
    bookmaker offers together, it searches again at each of those legs' bookmakers alone."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    out, orphans = _search_at(tier, legs, now, None)
    for book in sorted(orphans):
        out += _search_at(tier, legs, now, book)[0]
    return out


def _search_at(tier: Tier, legs: Sequence[Candidate], now: datetime.datetime, only: Optional[str]
               ) -> Tuple[List[Ticket], set]:
    """One builder run (at every price book, or ``only`` one), ranked by the cautious expected return alone: of its
    best tickets (the builder's ranked list, not only the few sharing no leg), those that fit the tier at one
    bookmaker, and the bookmakers of tickets no one bookmaker offers."""
    by_key = {}
    candidates = []
    for leg in legs:
        if not any(price_book(book) and (only is None or book == only) for book in leg.prices):
            continue
        candidate = _as_candidate(leg, now, only)
        by_key[candidate.key] = leg
        candidates.append(candidate)
    if not candidates:
        return [], set()
    result = builder.build(candidates, tier_controls(tier), now, weights=builder.RETURN_ONLY)
    out, orphans = [], set()
    for _, keys in result.ranked:
        chosen = [by_key.get(key) for key in keys]
        if not chosen or any(leg is None for leg in chosen):
            continue
        books = set.intersection(*({b for b in leg.prices if price_book(b) and (only is None or b == only)}
                                   for leg in chosen))
        if not books:
            orphans |= {b for leg in chosen for b in leg.prices if price_book(b)}
            continue
        placed = sorted((Ticket(chosen, book) for book in books), key=lambda t: (t.decimal, t.book), reverse=True)
        fitting = next((ticket for ticket in placed if _fits(tier, ticket)), None)
        if fitting is not None:
            out.append(fitting)
    return out, orphans


def price_book(book: str) -> bool:
    """Whether a ticket may be priced at this bookmaker (by title or key): never a SHARP_BOOKS book or an
    exchange, only the books a person can actually bet at (data/odds_service.py's BookRoles, set from
    ODDS_PRICE_BOOKS)."""
    from ..data.odds_service import is_price_book  # imported here: the data layer imports the engine
    return is_price_book(book)


def _eligible(tier: Tier, candidates: Sequence[Candidate]) -> List[Candidate]:
    """The legs a tier may use: the AI's chance at least MIN_LEG and not below the market's, with prices, and for
    Safe no more than SAFE_MAX_GAP from the market."""
    return [leg for leg in candidates
            if leg.probability >= MIN_LEG and leg.prices and (leg.fair is None or leg.probability >= leg.fair - 0.01)
            and (tier.key != "safe" or abs(leg.gap or 0.0) <= SAFE_MAX_GAP)]


def build_tier(tier: Tier, candidates: Sequence[Candidate], count: int = 3,
               now: Optional[datetime.datetime] = None) -> List[Ticket]:
    """Up to ``count`` tickets for one tier, best first, sharing at most one leg with each other: the builder's
    tickets under the tier's preset, each at the one bookmaker that offers all its legs at the best price."""
    legs = _eligible(tier, candidates)
    # Only books a person can bet at (_search): Pinnacle and the exchanges anchor the consensus, never a ticket
    best: Dict[frozenset, Ticket] = {}
    for ticket in _search(tier, legs, now):
        keys = frozenset(leg.key for leg in ticket.legs)
        if keys not in best or ticket.decimal > best[keys].decimal:
            best[keys] = ticket
    chosen: List[Ticket] = []
    uses: Dict[Tuple[str, str, str], int] = {}
    for ticket in sorted(best.values(), key=_rank, reverse=True):
        keys = {leg.key for leg in ticket.legs}
        if any(len(keys & {leg.key for leg in other.legs}) >= 2 for other in chosen):
            continue
        if any(uses.get(key, 0) >= 2 for key in keys):
            continue
        chosen.append(ticket)
        for key in keys:
            uses[key] = uses.get(key, 0) + 1
        if len(chosen) == count:
            break
    return chosen


def closest_tier(tier: Tier, candidates: Sequence[Candidate], count: int = 3,
                 now: Optional[datetime.datetime] = None) -> List[Tuple[Ticket, List[str]]]:
    """Closest effort for a level nothing fits exactly: the builder under the tier's preset with best_effort on (it
    loosens the most binding rule a step at a time until a ticket comes out), from the same legs as the tier's own
    search, each ticket placed at the one bookmaker that offers all its legs at the best price, with the rules that
    were relaxed for it in the builder's words ("per-leg chance lowered from 72% to 67%"). Up to ``count``."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    by_key: Dict[str, Candidate] = {}
    candidates_ = []
    for leg in _eligible(tier, candidates):
        if any(price_book(book) for book in leg.prices):
            candidate = _as_candidate(leg, now)
            by_key[candidate.key] = leg
            candidates_.append(candidate)
    if not candidates_:
        return []
    controls = replace(tier_controls(tier, best_effort=True), tickets=max(1, min(count, builder.MAX_TICKETS)))
    out: List[Tuple[Ticket, List[str]]] = []
    for ticket in builder.build(candidates_, controls, now, weights=builder.RETURN_ONLY).tickets:
        chosen = [by_key.get(view.get("key")) for view in ticket.get("legs") or []]
        if not chosen or any(leg is None for leg in chosen):
            continue
        books = set.intersection(*({b for b in leg.prices if price_book(b)} for leg in chosen))
        if not books:
            continue
        placed = max((Ticket(chosen, book) for book in books), key=lambda t: (t.decimal, t.book))
        words = []
        for entry in ticket.get("relaxed") or []:
            text = entry.get("words") if isinstance(entry, dict) else None
            if text and text not in words:
                words.append(text)
        out.append((placed, words))
        if len(out) == count:
            break
    return out


def _american(decimal: float) -> str:
    return format_american(decimal_to_american(decimal)) if decimal > 1 else "–"


def describe(tier: Tier, ticket: Ticket) -> str:
    """One honest sentence on what this ticket is."""
    p = ticket.probability
    n = len(ticket.legs)
    whelan = whelan_max_legs(p ** (1 / n))
    if tier.key == "crazy":
        text = f"Hits about 1 in {max(2, round(1 / p))}"
        tries = math.log(0.5) / math.log(1 - p) if p < 1 else 1
        text += f"; it takes about {math.ceil(tries)} tries for an even chance of one win."
    else:
        text = f"Wins about {round(p * 10)} in 10, loses about {10 - round(p * 10)} in 10."
    if n > whelan:
        text += f" More legs than are worth combining at these chances ({whelan} max): it's here for fun."
    flagged = [leg.label for leg in ticket.legs if abs(leg.gap or 0.0) > CHECK_NEWS_GAP]
    if flagged:
        text += f" The AI is far from the bookmakers on {', '.join(flagged)}: check the injury news."
    return text


def ticket_view(tier: Tier, ticket: Ticket, value_proven: bool = False) -> Dict[str, Any]:
    p, fair, decimal = ticket.probability, ticket.fair_probability, ticket.decimal
    ev, cautious = ticket.ev, ticket.cautious_ev
    if value_proven and cautious > 0:
        badge = "Value"
    elif -0.02 <= ev <= 0:
        badge = "Close to fair"
    else:
        badge = "For fun"
    return {
        "legs": [
            {"event_id": leg.event_id, "game": leg.game, "market": leg.market, "side": leg.side, "point": leg.point,
             "label": leg.label, "probability": leg.probability, "fair": leg.fair,
             "check_news": abs(leg.gap or 0.0) > CHECK_NEWS_GAP,
             "odds": {"decimal": price, "american": _american(price)}}
            for leg, price in zip(ticket.legs, ticket.prices)
        ],
        "book": ticket.book,
        "odds": {"decimal": round(decimal, 4), "american": _american(decimal)},
        "probability": p,
        "fair_probability": fair,
        "ev": ev,
        "cautious_ev": cautious,
        "book_edge": ticket.book_edge,
        "fair_odds": {"ai": _american(1 / p), "market": _american(1 / fair) if fair else None},
        "one_in": max(1, round(1 / p)),
        "expected_cost_per_10": round(max(0.0, -ev) * 10, 2),
        "badge": badge,
        "why": describe(tier, ticket),
    }


VIG_NOTE = ("Each extra leg adds the bookmaker's cut again: on NBA moneylines about 4% for one leg, 7% for "
            "two, 11% for three and 17% for five.")


def build_all(candidates: Sequence[Candidate], count: int = 3, value_proven: bool = False,
              now: Optional[datetime.datetime] = None, best_effort: bool = True) -> Dict[str, Dict[str, Any]]:
    """Each tier's tickets (build_tier) as the app and the bot show them, and as the AI record logs them ("bets":
    only tickets that fit the tier). With ``best_effort`` a tier nothing fits gets its closest tickets instead
    ("closest", closest_tier: each view labelled "Closest to <tier>" with the rules relaxed, in words); they are
    never logged as the tier's own."""
    out = {}
    for tier in TIERS:
        tickets = build_tier(tier, candidates, count, now)
        closest = closest_tier(tier, candidates, count, now) if best_effort and not tickets else []
        out[tier.key] = {
            "title": tier.title,
            "rule": tier.subtitle,
            "detail": (f"AI chance {tier.p_low:.0%}{'+' if tier.p_high > 1 else f'-{tier.p_high:.0%}'}, "
                       f"up to {tier.max_legs} legs, paying {tier.odds_low:.2f}x-{tier.odds_high:g}x."),
            "bets": [ticket_view(tier, ticket, value_proven) for ticket in tickets],
            "closest": [{**ticket_view(tier, ticket, value_proven), "label": f"Closest to {tier.title}",
                         "relaxed": words} for ticket, words in closest],
            "empty": None if tickets else "Nothing fits this level right now.",
        }
    return out


# --- grading the AI's own tickets, for an honest per-tier record ----------------------------------

def leg_outcome(leg: Dict[str, Any], final: Dict[str, Any]) -> str:
    """"won", "lost" or "push" for one leg, from the final score."""
    margin, total = final["home"] - final["away"], final["home"] + final["away"]
    if leg["market"] == "h2h":
        if leg["side"] == "draw":
            return "won" if margin == 0 else "lost"
        if margin == 0:
            return "push"
        return "won" if (margin > 0) == (leg["side"] == "home") else "lost"
    if leg["market"] == "spreads":
        value = (margin if leg["side"] == "home" else -margin) + leg["point"]
    else:
        value = total - leg["point"] if leg["side"] == "over" else leg["point"] - total
    return "won" if value > 0 else "push" if value == 0 else "lost"


def grade_ticket(legs: Sequence[Dict[str, Any]], finals: Sequence[Optional[Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    """A ticket's status (won, lost or void) and profit per unit staked.

    ``finals`` holds each leg's final score, or None for a game that was
    called off (that leg is voided and drops out, like a push).
    """
    outcomes = [leg_outcome(leg, final) if final else "push" for leg, final in zip(legs, finals)]
    if "lost" in outcomes:
        return "lost", {"outcomes": outcomes, "profit": -1.0}
    if all(outcome == "push" for outcome in outcomes):
        return "void", {"outcomes": outcomes, "profit": 0.0}
    payout = math.prod(leg["odds"]["decimal"] for leg, outcome in zip(legs, outcomes) if outcome == "won")
    return "won", {"outcomes": outcomes, "profit": payout - 1}


def tier_record(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per tier: how the AI's daily tickets did, the predicted hit rate next to the real one, and ROI."""
    out = []
    for tier in TIERS:
        graded = [r for r in rows if r["tier"] == tier.key and r["status"] in ("won", "lost")]
        n = len(graded)
        won = sum(1 for r in graded if r["status"] == "won")
        profit = sum(r["result"]["profit"] for r in graded)
        out.append({
            "tier": tier.key,
            "title": tier.title,
            "tickets": n,
            "won": won,
            "void": sum(1 for r in rows if r["tier"] == tier.key and r["status"] == "void"),
            "predicted": sum(r["payload"]["probability"] for r in graded) / n if n else None,
            "actual": won / n if n else None,
            "roi": profit / n if n else None,
        })
    return out
