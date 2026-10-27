"""The brain's tools for the chat assistant: what Claude is shown, and the code that runs each call.

Claude decides which tools to call; code does every calculation. So no tool
takes a price or a probability: legs are named by reference (league, event
id, market, side) and priced from the cached odds on the server, and the
only amounts Claude passes are the ones the person gave (a stake, a target,
a rollover's start and minimum odds), checked here against what they said
so Claude can never pick or raise a stake on its own. A ticket's stake is
per bet, so the check is on the whole ticket: four singles at the person's
$10 would be $40, and are refused. slip_propose shows a
ticket with an "Add to slip" button and saves nothing: only the person adds
legs to their slip.

Results are compact JSON (a few hundred tokens, not the dashboard's full
cards), each with its source and "as of" time, and errors come back as
readable records with a hint Claude can act on ("stale_odds", "not_found",
"amount_not_given", ...). The full results stay in code, where the ticket,
plan and player cards are built from them, and each call leaves a one-line
summary (with the leg references) for the thread's memory.

Odds data is rationed: each league the Odds API loads costs 3 of the
month's ~500 credits. A turn may use the leagues the person asked about,
their favourites and any league already loaded; beyond those it gets one
more, and the next answers "quota" with a hint to ask which sport they mean.

The tool list is byte-for-byte the same on every request so the prompt cache
holds; while picks are paused for someone, the pricing and building tools
answer "picks_paused" instead of being removed.

A bet slip the person pasted (read by ai/slip_reader.py, matched and priced
by the Lab) is re-analysed with pasted_slip_analyse: by its id, and only one
pasted in this chat, with legs left out by number, another stake (checked
like every stake: the slip's own, or less) or another ticket type. Its
results carry the slip's numbers as the person's bookmaker wrote them.
"""

from __future__ import annotations

import datetime
import re
import inspect
import logging
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, ValidationError

from ..dashboard.service import CallRejected, NotFound
from ..data.leagues import POPULAR_LEAGUES
from ..engine import consensus
from ..engine.pasted import system_key
from ..engine.tickets import ai_given_no_push, chance_label, combinations, one_in

# The leagues the assistant may name and load odds for, chosen on purpose (not every scoreboard the slate shows):
# each key is spelled out in the router prompt and in about ten tool schemas, so every key costs input tokens on
# every round, and each league it loads costs 3 Odds API credits
ASSISTANT_LEAGUES = ("basketball_wnba",)
LEAGUE_KEYS: Tuple[str, ...] = tuple(sorted(set(POPULAR_LEAGUES) | set(ASSISTANT_LEAGUES)))
MARKETS = ("h2h", "spreads", "totals")
SIDES = ("home", "away", "draw", "over", "under")
TICKET_TYPES = ("single", "multi", "round_robin", "system")
TIERS = ("safe", "risky", "crazy")
MAX_LEAGUES = 4  # each league's odds fetch costs 3 of the month's ~500 Odds API credits
EXTRA_LEAGUES = 1  # leagues a turn may load beyond the asked-for, favourite and already-loaded ones
ODDS_TTL = 2 * 3600.0  # how long loaded odds are reused when the odds service doesn't say
MAX_EVENTS = 12
MAX_LEGS = 12
CENT = Decimal("0.005")  # amounts within half a cent of what the person said count as theirs
DEFAULT_MIN_ODDS = Decimal("2.00")
ODDS_SOURCE = "The Odds API (best price across bookmakers, margin-free market chances)"
BRAIN_SOURCE = "The brain: our model's chances next to the bookmakers' margin-free ones"
ESPN_SOURCE = "ESPN"
MARKETS_SOURCE = ("The markets brain: margin-free sharp and US book prices, Kalshi and Polymarket mid prices, ESPN's "
                  "predictor, power ratings and our model, blended in code")
MARKETS_NOTE = ("The blend is the app's best estimate; the market usually wins. Only sources with status ok are "
                "blended: thin, stale or missing ones are shown, not counted.")
PASTED_SOURCE = ("The user's pasted slip at their own odds, priced by the brain: chances from the market, the model "
                 "or estimated from their price minus a typical margin when the leg isn't on our boards")
MAX_PASTED_LEGS = 20
# Tools that price or build tickets: off while picks are paused after a warning sign
PICK_TOOLS = frozenset({"odds_get_prices", "brain_price_legs", "ticket_price", "ticket_build", "ticket_analyse",
                        "rollover_simulate", "slip_propose", "player_get_props", "pasted_slip_analyse"})


# --- what Claude sees -------------------------------------------------------------------------------

def _obj(properties: Dict[str, Any], optional: Sequence[str] = ()) -> Dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": [name for name in properties if name not in optional], "additionalProperties": False}


_STR = {"type": "string"}
_LEAGUE = {"type": "string", "enum": list(LEAGUE_KEYS), "description": "League key, e.g. basketball_nba"}
_LEAGUES = {"type": "array", "items": _LEAGUE, "description": f"1 to {MAX_LEAGUES} league keys"}
_REF = _obj({
    "league": _LEAGUE,
    "id": {"type": "string", "description": "event_id from sports_find_events or odds_get_prices"},
    "market": {"type": "string", "enum": list(MARKETS)},
    "side": {"type": "string", "enum": list(SIDES)},
})
_LEGS = {"type": "array", "items": _REF, "description": f"1 to {MAX_LEGS} legs, each from a different game "
                                                        "(copy the refs that odds_get_prices returned)"}
_TICKET = {
    "type": {"type": "string", "enum": list(TICKET_TYPES),
             "description": "single (each leg its own bet), multi (one parlay), round_robin or system"},
    "stake": {"type": "number", "description": "Stake per bet in dollars, as the user gave it. Singles, round "
                                               "robins and system bets hold several bets, and together they must "
                                               "fit in the user's stake: set split to share it across the bets"},
    "split": {"type": "boolean", "description": "true: stake is for the whole ticket and code divides it evenly "
                                                "across the bets (never use it on a stake the user gave per bet)"},
    "legs": _LEGS,
    "k": {"type": "integer", "description": "round_robin only: legs per bet (2 to 7)"},
    "system": {"type": "string", "description": "system only: trixie, patent, yankee, lucky15, canadian, "
                                                "lucky31, heinz, lucky63, super_heinz or goliath"},
}

# Not "strict": Claude compiles strict schemas into one grammar, and the tools with the league list in
# every leg made it too large ("The compiled grammar is too large", a 400 on every chat turn). Toolbox.run
# checks every input against its pydantic model instead, and a bad one comes back as an error with a hint.
TOOLS: List[Dict[str, Any]] = [
    {"name": "sports_find_events",
     "description": "List upcoming and live games with the AI's predicted winner and chance. Call this first "
                    "whenever the user mentions games, teams or a sport, to get current event ids; never "
                    "guess an id. query filters by team name (empty string for every game).",
     "input_schema": _obj({"leagues": _LEAGUES, "query": _STR,
                           "when": {"type": "string", "enum": ["today", "tomorrow", "week"]}})},
    {"name": "odds_get_prices",
     "description": "Every market for one game: best price, the AI's chance, the market's margin-free chance, "
                    "fair odds and a ref for each outcome. Call this before quoting any odds or chance for a game.",
     "input_schema": _obj({"league": _LEAGUE, "event_id": _STR})},
    {"name": "markets_consensus",
     "description": "What the markets say about one game: the blended chance per outcome (the app's best "
                    "estimate), each source's chance, as-of and status, and whether they disagree.",
     "input_schema": _obj({"league": _LEAGUE, "event_id": _STR})},
    {"name": "player_get_profile",
     "description": "A player's season and recent averages, last 10 games, home/away splits, injury status, "
                    "next game and dated news. Call this when the user asks about a player's form.",
     "input_schema": _obj({"name": {"type": "string", "description": "The player's name as the user wrote it"},
                           "league": _LEAGUE})},
    {"name": "player_get_props",
     "description": "A player's posted prop lines with hit rates (with sample sizes), the brain's projection "
                    "and its chance of the over. Call this before talking about any player prop.",
     "input_schema": _obj({"name": _STR, "league": _LEAGUE})},
    {"name": "news_get",
     "description": "Dated injury and lineup news with sources: for a player (player = name), one game "
                    "(event_id) or the league's games (both empty strings).",
     "input_schema": _obj({"league": _LEAGUE, "player": _STR, "event_id": _STR})},
    {"name": "brain_price_legs",
     "description": "The brain's view of each leg on its own: the AI's chance, the market's chance, fair odds, "
                    "edge and the bookmaker's cut. Use it to compare legs; it prices no ticket.",
     "input_schema": _obj({"legs": _LEGS})},
    {"name": "ticket_price",
     "description": "Price a ticket: combined odds, payout, chance as '1 in X' (AI and market), what the "
                    "bookmaker keeps, the expected result in money, the most likely outcome and warnings. "
                    "Call this before quoting any ticket's odds, payout or chance.",
     "input_schema": _obj(dict(_TICKET), optional=("k", "system", "split"))},
    {"name": "ticket_build",
     "description": "Build up to three tickets that turn the user's stake into about their target, with the "
                    "honest ceiling on the chance and a verdict. Use the user's own stake and target "
                    "(target = money back, stake included); ask if either is missing.",
     "input_schema": _obj({"leagues": _LEAGUES, "stake": {"type": "number"}, "target": {"type": "number"},
                           "max_legs": {"type": "integer", "description": "1 to 6"},
                           "tier": {"type": "string", "enum": list(TIERS)}}, optional=("tier",))},
    {"name": "ticket_analyse",
     "description": "Analyse the user's own ticket: priced, with notes per leg (injuries, AI vs market gap, "
                    "analysts' views) and alternatives the code worked out (drop the weakest leg, singles, "
                    "a round robin).",
     "input_schema": _obj(dict(_TICKET), optional=("k", "system", "split"))},
    {"name": "rollover_simulate",
     "description": "Compare rollover strategies side by side (classic all-in, bank half, pot-and-ladder, "
                    "flat stakes): chance to finish, chance of a profit, typical end, expected cost, the day a "
                    "run most likely ends. Use the user's own start amount and days.",
     "input_schema": _obj({"start": {"type": "number"}, "days": {"type": "integer", "description": "1 to 60"},
                           "min_odds": {"type": "number", "description": "Daily odds, 1.2 to 20; use 2.0 if "
                                                                         "the user gave none"},
                           "legs_per_day": {"type": "integer", "description": "1 to 3"}})},
    {"name": "slip_propose",
     "description": "Show the user a priced ticket card with an 'Add to slip' button. It saves nothing and "
                    "adds nothing: only the user can add it. Use it for the one ticket you suggest.",
     "input_schema": _obj({**_TICKET, "title": {"type": "string", "description": "A short card title"}},
                          optional=("k", "system", "split"))},
    {"name": "user_get_context",
     "description": "The user's settings: odds format, favourite sports, risk tier, weekly budget left, "
                    "cool-off and AI credits left (read only).",
     "input_schema": _obj({})},
    {"name": "pasted_slip_analyse",
     "description": "Analyse a bet slip the user pasted in this chat (pasted_id from the context) at their own "
                    "odds: chance as '1 in X', payout, what the bookmaker keeps, the expected result, notes per "
                    "leg (better prices on our boards, lines that differ, estimated chances) and cheaper "
                    "versions. To answer 'what if', leave legs out with drop (leg numbers from 1), or change the "
                    "stake or the ticket type. Call it before quoting any number about a changed slip.",
     "input_schema": _obj({
         "pasted_id": {"type": "integer"},
         "drop": {"type": "array", "items": {"type": "integer"}, "description": "Leg numbers to leave out (from 1)"},
         "stake": {"type": "number", "description": "Stake per bet in the user's currency, as the user gave it; "
                                                    "leave it out to keep the slip's own"},
         "split": {"type": "boolean", "description": "true: stake is for the whole ticket and code divides it "
                                                     "evenly across the bets"},
         "ticket_type": {"type": "string", "enum": list(TICKET_TYPES)},
         "k": {"type": "integer", "description": "round_robin only: legs per bet (2 to 7)"},
         "system": {"type": "string", "description": "system only, e.g. trixie, patent, yankee, lucky15"},
     }, optional=("drop", "stake", "split", "ticket_type", "k", "system"))},
]
TOOL_NAMES = tuple(tool["name"] for tool in TOOLS)


# --- validating what Claude sent ----------------------------------------------------------------------

LeagueKey = Literal[LEAGUE_KEYS]  # type: ignore[valid-type]


class Ref(BaseModel):
    league: LeagueKey  # type: ignore[valid-type]
    id: str = Field(min_length=1, max_length=64)
    market: Literal[MARKETS]  # type: ignore[valid-type]
    side: Literal[SIDES]  # type: ignore[valid-type]


class TicketInput(BaseModel):
    type: Literal[TICKET_TYPES]  # type: ignore[valid-type]
    stake: Decimal = Field(gt=0, le=100000, decimal_places=2)
    legs: List[Ref] = Field(min_length=1, max_length=MAX_LEGS)
    k: Optional[int] = Field(None, ge=2, le=7)
    system: Optional[str] = Field(None, max_length=20)
    split: Optional[bool] = None
    title: Optional[str] = Field(None, max_length=80)


class BuildInput(BaseModel):
    leagues: List[LeagueKey] = Field(min_length=1, max_length=MAX_LEAGUES)  # type: ignore[valid-type]
    stake: Decimal = Field(gt=0, le=100000, decimal_places=2)
    target: Decimal = Field(gt=0, le=10000000, decimal_places=2)
    max_legs: int = Field(ge=1, le=6)
    tier: Optional[Literal[TIERS]] = None  # type: ignore[valid-type]


class RolloverInput(BaseModel):
    start: Decimal = Field(ge=Decimal("0.01"), le=100000, decimal_places=2)
    days: int = Field(ge=1, le=60)
    min_odds: Decimal = Field(ge=Decimal("1.2"), le=20, decimal_places=2)
    legs_per_day: int = Field(ge=1, le=3)


class EventsInput(BaseModel):
    leagues: List[LeagueKey] = Field(min_length=1, max_length=MAX_LEAGUES)  # type: ignore[valid-type]
    query: str = Field("", max_length=60)
    when: Literal["today", "tomorrow", "week"] = "week"


class OddsInput(BaseModel):
    league: LeagueKey  # type: ignore[valid-type]
    event_id: str = Field(min_length=1, max_length=64)


class PlayerInput(BaseModel):
    name: str = Field(min_length=2, max_length=60)
    league: LeagueKey  # type: ignore[valid-type]


class NewsInput(BaseModel):
    league: LeagueKey  # type: ignore[valid-type]
    player: str = Field("", max_length=60)
    event_id: str = Field("", max_length=64)


class LegsInput(BaseModel):
    legs: List[Ref] = Field(min_length=1, max_length=MAX_LEGS)


class NoInput(BaseModel):
    pass


class PastedInput(BaseModel):
    pasted_id: int = Field(ge=1)
    drop: List[int] = Field(default_factory=list, max_length=MAX_PASTED_LEGS)
    stake: Optional[Decimal] = Field(None, gt=0, le=100000, decimal_places=2)
    split: Optional[bool] = None
    ticket_type: Optional[Literal[TICKET_TYPES]] = None  # type: ignore[valid-type]
    k: Optional[int] = Field(None, ge=2, le=7)
    system: Optional[str] = Field(None, max_length=20)


INPUTS: Dict[str, type] = {
    "sports_find_events": EventsInput, "odds_get_prices": OddsInput, "markets_consensus": OddsInput,
    "player_get_profile": PlayerInput,
    "player_get_props": PlayerInput, "news_get": NewsInput, "brain_price_legs": LegsInput,
    "ticket_price": TicketInput, "ticket_build": BuildInput, "ticket_analyse": TicketInput,
    "rollover_simulate": RolloverInput, "slip_propose": TicketInput, "user_get_context": NoInput,
    "pasted_slip_analyse": PastedInput,
}


def _cents(value: Decimal) -> str:
    """An amount as the Lab expects it: dollars with two decimals ("10.00")."""
    return str(value.quantize(Decimal("0.01")))


def bet_count(kind: Any, legs: int, k: Optional[int] = None, system: Optional[str] = None) -> int:
    """How many bets a ticket holds: each single, each round-robin or system bet (a multi is one).

    A shape that can't be priced counts as one bet: pricing then says what is wrong with it.
    """
    try:
        return max(1, len(combinations(str(kind), int(legs), k, system)))
    except (ValueError, KeyError, TypeError):
        return 1


class ToolError(Exception):
    """A readable failure: a short code, what went wrong and what Claude should do instead."""

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


@dataclass
class ToolRun:
    """One tool call: what Claude was shown, the full result for cards, and a line for the thread's memory."""
    name: str
    args: Dict[str, Any]
    ok: bool
    content: Dict[str, Any]
    summary: str
    cards: List[Dict[str, Any]]


# --- small formatting helpers ------------------------------------------------------------------------

def _p(value: Any) -> Optional[float]:
    return round(float(value), 4) if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool) else None


def _no_push(ai: Any, push: Any) -> Optional[float]:
    """The AI's outright chance on the market's footing (given no push), for a line that can push: market_chance
    is given no push, so this is the AI chance to set beside it."""
    ai, push = _p(ai), _p(push)
    return _p(ai_given_no_push(ai, push)) if ai is not None and push else None


def _chance(p: Any) -> Dict[str, Any]:
    """A chance in the forms Claude may quote: the number, "1 in X" and the fixed label."""
    value = _p(p)
    return {"chance": value, "one_in": one_in(value), "label": chance_label(value)}


def ref_of(leg: Mapping[str, Any]) -> Dict[str, Any]:
    """The pick a priced leg came from, as the app sends it ({league, id, market, side, point})."""
    return {"league": leg.get("league"), "id": leg.get("event_id") or leg.get("id"),
            "market": leg.get("market"), "side": leg.get("side"), "point": leg.get("point")}


def _ref_text(ref: Mapping[str, Any]) -> str:
    return f"{ref.get('league')}:{ref.get('id')}:{ref.get('market')}:{ref.get('side')}"


DROP_KEYS = {"headshot", "logo", "logos", "image", "images", "uid", "guid", "sources", "method", "group", "key"}


def trim(value: Any, items: int = 10, text: int = 240, depth: int = 0) -> Any:
    """A smaller copy of a lab result for Claude: short lists and strings, no images or internals."""
    if depth > 6:
        return None
    if isinstance(value, Mapping):
        return {str(k): trim(v, items, text, depth + 1) for k, v in value.items()
                if k not in DROP_KEYS and v not in (None, "", [], {})}
    if isinstance(value, (list, tuple)):
        return [trim(v, items, text, depth + 1) for v in list(value)[:items]]
    if isinstance(value, str):
        return value if len(value) <= text else value[: text - 1] + "…"
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, Decimal):
        return float(value)
    return value


def _game_name(card: Mapping[str, Any]) -> str:
    return f"{card.get('away')} @ {card.get('home')}"


def ticket_title(ticket: Mapping[str, Any]) -> str:
    """"Single: Knicks to win", "3-leg multi", "Trixie (3 legs)", "Round robin by 2s (4 legs)"."""
    legs = ticket.get("legs") or []
    n, kind = len(legs), ticket.get("type")
    if kind == "single":
        return f"Single: {legs[0].get('label')}" if n == 1 else f"{n} singles"
    if kind == "multi":
        return f"{n}-leg multi"
    if kind == "round_robin":
        return f"Round robin by {ticket.get('k') or 2}s ({n} legs)"
    if kind == "system":
        return f"{str(ticket.get('system') or 'System').replace('_', ' ').title()} ({n} legs)"
    return "Ticket"


def ticket_facts(ticket: Mapping[str, Any]) -> Dict[str, Any]:
    """The numbers of a priced ticket Claude may quote, without the full outcome tables."""
    chance = ticket.get("chance") or {}
    expected = ticket.get("expected") or {}
    lose_all = next((o for o in ticket.get("outcomes") or [] if o.get("result") == "lose all"), {})
    legs = []
    for leg in ticket.get("legs") or []:
        legs.append({
            "index": leg.get("index"), "label": leg.get("label"), "game": leg.get("game"),
            "odds": leg.get("odds"), "ai_chance": leg.get("ai_prob"), "market_chance": leg.get("market_prob"),
            "push_chance": leg.get("push_prob") or None,
            "ai_chance_given_no_push": _no_push(leg.get("ai_prob"), leg.get("push_prob")),
            "chance_label": leg.get("chance_label"),
            "fair_odds": leg.get("fair_odds"), "edge": leg.get("edge"), "book": leg.get("book"),
            "started": leg.get("started") or None, "ref": ref_of(leg),
        })
    facts = {
        "title": ticket_title(ticket), "type": ticket.get("type"), "system": ticket.get("system"),
        "k": ticket.get("k"), "bets": ticket.get("bets"), "stake_per_bet": ticket.get("unit_stake"),
        "total_stake": ticket.get("total_stake"), "odds": ticket.get("odds"), "max_payout": ticket.get("max_payout"),
        "max_profit": ticket.get("max_profit"),
        "chance": {key: chance.get(key) for key in ("headline", "ai", "market", "ai_one_in", "market_one_in", "label")},
        "lose_all_chance": {"ai": lose_all.get("ai"), "market": lose_all.get("market")},
        "book_keeps": ticket.get("book_cut"),
        "expected_result": {"ai": expected.get("ai"), "market": expected.get("market")},
        "most_likely": ticket.get("most_likely"),
        "weakest_leg": ticket.get("weakest_leg"),
        "legs": legs,
        "warnings": ticket.get("warnings") or [],
        "valid": ticket.get("valid"), "reasons": ticket.get("reasons") or [],
    }
    for name in ("tier", "cautious", "rank", "name"):  # what the target builder adds
        if ticket.get(name) is not None:
            facts[name] = ticket[name]
    return trim(facts, items=MAX_LEGS)


# --- cards: built in code from the full results -----------------------------------------------------

def ticket_card(ticket: Dict[str, Any], source: str, title: Optional[str] = None,
                extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    card = {"type": "ticket", "source": source, "title": title or ticket.get("name") or ticket_title(ticket),
            "ticket_type": ticket.get("type"), "k": ticket.get("k"), "system": ticket.get("system"),
            "stake": ticket.get("unit_stake"),
            "refs": ticket.get("refs") or [ref_of(leg) for leg in ticket.get("legs") or []],
            "ticket": ticket}
    card.update(extra or {})
    return card


def plan_card(body: Dict[str, Any], compare: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "plan", "source": "rollover_simulate",
            "title": f"{body['days']}-day rollover at {compare.get('min_odds') or body['min_odds']}",
            "body": body, "compare": compare}


def player_card(player: Dict[str, Any], source: str) -> Dict[str, Any]:
    keep = ("id", "league", "name", "team", "team_abbr", "position", "headshot", "status", "injury", "next_game",
            "season_label", "season", "last5", "last10", "sample", "as_of")
    return {"type": "player", "source": source, "title": player.get("name") or "Player",
            **{key: player.get(key) for key in keep if player.get(key) is not None}}


# --- a pasted slip ----------------------------------------------------------------------------------------

def amount_of(value: Any) -> Optional[Decimal]:
    """An amount from a pasted slip's view ("10.00"), or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).replace(",", ""))
    except ArithmeticError:
        return None
    return number if number.is_finite() and number > 0 else None


def pasted_label(leg: Mapping[str, Any]) -> str:
    """One pasted leg in words: "Boston Celtics -4.5", "Over 224.5 (Lakers @ Celtics)", "LeBron James over 27.5
    points", "Arsenal to win". The Lab's own label (PastedView legs[].label, what the card and the Lab's notes
    and warnings call the leg) when the view has one, with the game after a total; else built here."""
    market, line = leg.get("market"), leg.get("line")
    label = " ".join(str(leg.get("label") or "").split())
    if label:
        game = (leg.get("match") or {}).get("game")
        return f"{label} ({game})" if market == "total" and game and game not in label else label
    selection = " ".join(str(leg.get("selection") or "").split()) or "?"
    number = isinstance(line, (int, float)) and not isinstance(line, bool)
    side = str(leg.get("over_under") or "").lower()
    if market == "moneyline":
        return selection if selection.lower() == "draw" else f"{selection} to win"
    if market == "spread":
        return f"{selection} {line:+g}" if number else selection
    if market == "total":
        what = side.capitalize() if side in ("over", "under") else selection
        text = f"{what} {line:g}" if number else what
        game = (leg.get("match") or {}).get("game")
        return f"{text} ({game})" if game else text
    if market == "player_prop":
        parts = [selection] + ([side] if side in ("over", "under") else []) + ([f"{line:g}"] if number else [])
        parts += [str(leg["prop_stat"])] if leg.get("prop_stat") else []
        return " ".join(parts)
    return selection


def pasted_bets(view: Mapping[str, Any]) -> int:
    """How many bets the pasted slip holds as read (an unknown type counts as a multi, or one single)."""
    legs = len(view.get("legs") or [])
    kind = view.get("bet_type") if view.get("bet_type") in TICKET_TYPES else ("multi" if legs > 1 else "single")
    k = view.get("k") if isinstance(view.get("k"), int) and not isinstance(view.get("k"), bool) else None
    return bet_count(kind, legs, k, system_key(view.get("system")))


def pasted_stakes(view: Mapping[str, Any]) -> List[Decimal]:
    """The money the person put on the pasted slip: its stake per bet, and the whole slip (the stated total,
    or the stake times its bets). These are amounts they gave, so tools may use them (never more)."""
    stake, total = amount_of(view.get("stake")), amount_of(view.get("total_stake"))
    found = [amount for amount in (stake, total) if amount is not None]
    if stake is not None and total is None:
        found.append(stake * pasted_bets(view))
    return found


def slip_numbers(view: Mapping[str, Any]) -> Dict[str, Any]:
    """The numbers on the person's own slip, as written and as code read them (odds, lines, stake, the stated
    return): numbers a reply may quote back to them. Odds as written count only where code could read them as
    odds ("+150", "5/2"), so other text the reader put there isn't a number Claude may state."""
    legs = view.get("legs") or []
    return {"odds_as_written": [leg.get("odds_text") for leg in legs if leg.get("odds")],
            "odds": [(leg.get("odds") or {}) for leg in legs],
            "lines": [leg.get("line") for leg in legs],
            "stake": view.get("stake"), "total_stake": view.get("total_stake"),
            "stated_return": view.get("stated_return"), "computed_return": view.get("computed_return")}


def pasted_facts(view: Mapping[str, Any], chances: bool = True) -> Dict[str, Any]:
    """A pasted slip for Claude: what was read and how each leg matched; with ``chances``, the brain's
    chances and the best prices on the boards too. Everything pasted is short (the Lab caps it)."""
    legs = []
    for leg in view.get("legs") or []:
        match = leg.get("match") or {}
        entry: Dict[str, Any] = {
            "leg": leg.get("index"), "label": pasted_label(leg), "market": leg.get("market"),
            "odds_as_written": leg.get("odds_text"), "odds": (leg.get("odds") or {}).get("decimal"),
            "match": match.get("status"), "game": match.get("game"), "live": match.get("live") or None,
            "note": match.get("note")}
        if chances:
            entry.update(market_chance=_p(leg.get("market_chance")), ai_chance=_p(leg.get("ai_chance")),
                         chance_source=leg.get("chance_source"), best_price=leg.get("best_price"),
                         better_price=leg.get("better_price") or None)
        legs.append(entry)
    facts: Dict[str, Any] = {
        "pasted_id": view.get("id"), "read_from": view.get("source"), "bookmaker": view.get("bookmaker"),
        "bet_type": view.get("bet_type"), "k": view.get("k"), "system": view.get("system"),
        "currency": view.get("currency"), "stake_per_bet": view.get("stake"), "total_stake": view.get("total_stake"),
        "bookmaker_says_it_returns": view.get("stated_return"), "legs": legs, "warnings": view.get("warnings") or []}
    if chances:
        facts.update(code_return=view.get("computed_return"), return_check=view.get("return_check"))
    return trim(facts, items=MAX_PASTED_LEGS)


def pasted_ticket_facts(ticket: Mapping[str, Any]) -> Dict[str, Any]:
    """ticket_facts for a pasted slip's ticket, with every leg named by its number on the slip (1, 2, ...).

    engine/tickets numbers a ticket's legs from 0 among the legs it prices (weakest_leg too), while the slip,
    the leg notes and pasted_slip_analyse's drop count the slip's legs from 1, left-out legs included: the Lab
    gives each priced leg its slip_index, and Claude only ever sees that number."""
    facts = ticket_facts(ticket)
    numbers = [leg.get("slip_index") for leg in ticket.get("legs") or []]
    for leg, number in zip(facts.get("legs") or [], numbers):
        leg.pop("index", None)
        leg["leg"] = number
    weakest = ticket.get("weakest_leg")
    if isinstance(weakest, int) and not isinstance(weakest, bool) and 0 <= weakest < len(numbers):
        facts["weakest_leg"] = numbers[weakest]
    else:
        facts.pop("weakest_leg", None)
    return facts


def pasted_analysis_facts(analysis: Mapping[str, Any]) -> Dict[str, Any]:
    """The Lab's analysis of a pasted slip, in the compact form Claude is shown. Legs are numbered as on the
    slip everywhere (the ticket's, the weakest leg, the notes and each cheaper version's "drop", the legs to
    leave out with pasted_slip_analyse to get it)."""
    alternatives = [{"title": alt.get("title"), "summary": alt.get("summary"), "drop": alt.get("drop"),
                     "ticket": pasted_ticket_facts(alt["ticket"]) if alt.get("ticket") else None}
                    for alt in analysis.get("alternatives") or [] if isinstance(alt, Mapping)]
    return {"slip": pasted_facts(analysis.get("pasted") or {}), "left_out": list(analysis.get("drop") or []),
            "stake_per_bet": analysis.get("stake"), "ticket_type": analysis.get("ticket_type"),
            "ticket": pasted_ticket_facts(analysis.get("ticket") or {}),
            "leg_notes": trim(analysis.get("legs") or [], items=MAX_PASTED_LEGS),
            "alternatives": trim(alternatives, items=4), "source": PASTED_SOURCE}


# The slip's own words in a pasted slip's tool results: what the reader copied, never code's numbers
PASTED_WORDS = ("label", "selection", "raw", "event_text", "start_text", "sport_hint", "odds_text")
PASTED_WORD_KEYS = frozenset({"label", "bookmaker", "system", "odds_as_written", "currency"})


def pasted_quotable(content: Any, view: Mapping[str, Any]) -> Any:
    """A pasted slip's tool result with the slip's own words taken out, for the numbers check's sources.

    Labels, the bookmaker and the like come from what was pasted ("Knicks (tipster: 13% sure)"), and the leg
    notes, warnings and alternatives quote those labels: a number in them was never worked out by code, so a
    reply may not state it as a fact. Those fields are dropped, and the slip's words with a digit in them are
    cut out of every other string; the slip's own odds, lines and money are quotable through slip_numbers."""
    texts = {str(leg.get(key)) for leg in view.get("legs") or [] if isinstance(leg, Mapping)
             for key in PASTED_WORDS if leg.get(key)}
    texts |= {pasted_label(leg) for leg in view.get("legs") or [] if isinstance(leg, Mapping)}
    texts |= {str(view.get(key)) for key in ("bookmaker", "system", "notes", "text", "currency") if view.get(key)}
    # Only words holding a number matter; a bare number ("1", "-110") is the slip's own and quotable anyway
    cut = sorted((text for text in texts if re.search(r"\d", text) and not re.fullmatch(r"[\s+\-.,/\d]+", text)),
                 key=len, reverse=True)

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: clean(item) for key, item in value.items() if key not in PASTED_WORD_KEYS}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, str):
            for text in cut:
                value = value.replace(text, " ")
        return value

    return clean(content)


def pasted_card(view: Mapping[str, Any], analysis: Optional[Mapping[str, Any]], source: str,
                title: str = "Your slip") -> Dict[str, Any]:
    """The pasted slip's card: what was read, how it matched and the code's analysis (the app's card shows
    them with leg toggles; the bot writes them out)."""
    return {"type": "pasted", "source": source, "title": title, "pasted": dict(view),
            "analysis": dict(analysis) if analysis is not None else None}


# --- running the tools ----------------------------------------------------------------------------------

STEPS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "sports_find_events": lambda a: f"Finding {'/'.join(l.split('_')[-1].upper() for l in a.get('leagues') or [])} games…",
    "odds_get_prices": lambda a: "Checking the odds…",
    "markets_consensus": lambda a: "Checking what the markets say…",
    "player_get_profile": lambda a: f"Looking up {a.get('name') or 'the player'}…",
    "player_get_props": lambda a: f"Checking {a.get('name') or 'the player'}'s prop lines…",
    "news_get": lambda a: "Checking injury news…",
    "brain_price_legs": lambda a: f"Running the brain on {len(a.get('legs') or [])} legs…",
    "ticket_price": lambda a: f"Pricing {len(a.get('legs') or [])} legs…",
    "ticket_build": lambda a: "Building tickets…",
    "ticket_analyse": lambda a: "Analysing your ticket…",
    "rollover_simulate": lambda a: "Running the rollover numbers…",
    "slip_propose": lambda a: "Preparing a slip for you to check…",
    "user_get_context": lambda a: "Checking your settings…",
    "pasted_slip_analyse": lambda a: "Re-checking your slip…",
}


def step_text(name: str, args: Mapping[str, Any]) -> str:
    """A progress line for the typing indicator, e.g. "Pricing 3 legs…"."""
    try:
        return STEPS[name](dict(args or {}))
    except Exception:
        return "Working…"


class Toolbox:
    """Runs one person's tool calls for one chat turn.

    ``limits`` holds the amounts the person gave this thread, by kind: "stake"
    (what they'd put on a whole ticket; also a rollover's start), "target" and
    "min_odds". A ticket costing more than the largest stake they gave, or a
    target above theirs, is refused, so the assistant can lower one but never
    raise it; a minimum odds must be one they gave (or the 2.00 default).

    ``leagues`` are the ones the turn may load odds for freely (asked for, or
    the person's favourites); leagues already loaded are free too, and
    ``extra_leagues`` more may be loaded before "quota" errors start.

    ``pasted`` holds the slips pasted in this chat that pasted_slip_analyse may
    use: their views by id (None for one whose view is fetched when needed).
    Any other id is refused, so nothing pasted elsewhere can be reached.
    """

    def __init__(self, lab: Any, uid: str, tz: ZoneInfo, limits: Optional[Mapping[str, Iterable[Decimal]]] = None,
                 paused: bool = False, now: Optional[Callable[[], datetime.datetime]] = None,
                 leagues: Iterable[str] = (), extra_leagues: int = EXTRA_LEAGUES,
                 pasted: Optional[Mapping[int, Optional[Mapping[str, Any]]]] = None) -> None:
        self.lab = lab
        self.uid = uid
        self.tz = tz
        self.limits: Dict[str, Set[Decimal]] = {kind: set(values) for kind, values in (limits or {}).items()}
        self.paused = paused
        self.now = now or (lambda: datetime.datetime.now(datetime.timezone.utc))
        self.leagues: Set[str] = set(leagues)
        self.extra_leagues = extra_leagues
        self.extra: Set[str] = set()  # leagues loaded this turn beyond self.leagues
        self.pasted: Dict[int, Optional[Mapping[str, Any]]] = dict(pasted or {})
        self.runners: Dict[str, Callable[[Any], Tuple[Dict[str, Any], str, List[Dict[str, Any]]]]] = {
            "sports_find_events": self._events, "odds_get_prices": self._odds, "markets_consensus": self._markets,
            "player_get_profile": self._profile,
            "player_get_props": self._props, "news_get": self._news, "brain_price_legs": self._brain,
            "ticket_price": self._price, "ticket_build": self._build, "ticket_analyse": self._analyse,
            "rollover_simulate": self._rollover, "slip_propose": self._propose, "user_get_context": self._context,
            "pasted_slip_analyse": self._pasted,
        }

    def _as_of(self) -> str:
        return self.now().astimezone(self.tz).strftime("%a %b %-d, %-I:%M %p %Z")

    def run(self, name: str, args: Any) -> ToolRun:
        args = dict(args) if isinstance(args, Mapping) else {}
        try:
            if name not in self.runners:
                raise ToolError("unknown_tool", f"There is no tool called {name}", f"Use one of: {', '.join(TOOL_NAMES)}")
            if self.paused and name in PICK_TOOLS:
                raise ToolError("picks_paused", "Picks, prices and ticket building are paused for this person "
                                                "for 24 hours after a warning sign",
                                "Don't suggest bets; offer help, the budget and break settings, or how odds work")
            data = INPUTS[name].model_validate(args)
            content, summary, cards = self.runners[name](data)
            content.setdefault("as_of", self._as_of())
            return ToolRun(name, args, True, content, summary, cards)
        except ValidationError as exc:
            fields = ", ".join(sorted({".".join(str(p) for p in error["loc"]) for error in exc.errors()})) or "input"
            error = ToolError("invalid_input", f"These inputs aren't valid: {fields}",
                              "Check the ranges in the tool description and call again")
        except ToolError as exc:
            error = exc
        except NotFound as exc:
            error = ToolError("not_found", str(exc) or "Not found",
                              "Call sports_find_events (or the player search) for current ids, then try again")
        except (CallRejected, ValueError) as exc:
            error = ToolError("rejected", str(exc) or "The request was refused",
                              "Tell the user why in plain words; don't retry with values they didn't give")
        except Exception:
            logging.exception("Assistant tool %s failed", name)
            error = ToolError("unavailable", "That data source failed just now",
                              "Say the data isn't available right now; never fill the gap with a guess")
        content = {"error": error.code, "message": error.message, "hint": error.hint}
        return ToolRun(name, args, False, content, f"{name}: {error.code}", [])

    # --- amounts the person gave ---

    def _check_amount(self, kind: str, value: Decimal, what: str, bets: int = 1) -> None:
        """Refuse an amount the person didn't give: ``value`` per bet, over ``bets`` bets."""
        given = self.limits.get(kind) or set()
        if kind == "min_odds":
            if any(abs(value - known) <= CENT for known in given | {DEFAULT_MIN_ODDS}):
                return
            raise ToolError("amount_not_given", f"The user didn't ask for minimum odds of {value}",
                            f"Use the odds the user gave, or {DEFAULT_MIN_ODDS} if they gave none")
        if not given:
            raise ToolError("amount_not_given", f"The user hasn't said what {what} they want",
                            f"Ask the user for their {what}; never choose one for them")
        total = value * bets
        if total <= max(given) + CENT:
            return
        if bets > 1:
            raise ToolError("amount_not_given", f"This ticket is {bets} bets at {_cents(value)} each, "
                                                f"{_cents(total)} in all: more than the {_cents(max(given))} the user gave",
                            "Set split to true to share the user's stake across the bets, or ask the user how much "
                            "they want on the whole ticket; never suggest a bigger stake")
        raise ToolError("amount_not_given", f"{value} is more than the {what} the user gave",
                        f"Use the user's own {what}; never suggest a bigger one")

    # --- odds data the turn may load ---

    def _loaded(self, league: str) -> bool:
        """Whether a league's odds are loaded and fresh, so using them costs no quota."""
        odds = getattr(getattr(self.lab, "service", None), "odds", None)
        try:
            loaded = odds.odds_loaded_at(league) if odds is not None else None
            store = getattr(odds, "store", None)
            if loaded is None and store is not None:  # the bot or the dashboard may have loaded it
                snapshot = store.odds_snapshot(league)
                loaded = snapshot[0] if snapshot else None
        except Exception:
            logging.exception("Couldn't tell whether %s odds are loaded", league)
            return False
        ttl = getattr(odds, "odds_ttl", None) or ODDS_TTL
        return isinstance(loaded, (int, float)) and self.now().timestamp() - loaded < ttl

    def _use_leagues(self, leagues: Iterable[str]) -> None:
        """Allow loading these leagues' odds, or refuse when the turn's spare allowance is used up."""
        new = [league for league in dict.fromkeys(leagues)
               if league not in self.leagues and league not in self.extra and not self._loaded(league)]
        if len(self.extra) + len(new) > self.extra_leagues:
            usable = sorted(self.leagues | self.extra)
            raise ToolError("quota", f"Loading odds for {', '.join(new)} would use the app's limited odds data "
                                     "(each league costs part of a small monthly allowance)",
                            f"Stay with {', '.join(usable) or 'the leagues the user asked about'}, or ask the user "
                            "which sport they mean before loading another")
        self.extra.update(new)

    # --- the tools ---

    def _today(self, leagues: Sequence[str]) -> Dict[str, Any]:
        self._use_leagues(leagues)
        return self.lab.today(self.uid, list(leagues)) or {}

    def _card(self, league: str, event_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        data = self._today([league])
        card = next((c for c in data.get("games") or [] if c.get("id") == event_id), None)
        if card is None:
            error = (data.get("errors") or {}).get(league)
            if error:
                raise ToolError("odds_unavailable", str(error), "Say the odds for this league can't be loaded now")
            raise ToolError("not_found", f"No game {event_id} in {league} right now",
                            "Call sports_find_events for current event ids")
        return card, data

    def _in_window(self, card: Mapping[str, Any], when: str) -> bool:
        try:
            start = datetime.datetime.fromisoformat(str(card.get("start")).replace("Z", "+00:00"))
        except ValueError:
            return True
        today = self.now().astimezone(self.tz).date()
        day = start.astimezone(self.tz).date()
        if when == "today":
            return day <= today or bool(card.get("live"))
        if when == "tomorrow":
            return day == today + datetime.timedelta(days=1)
        return day <= today + datetime.timedelta(days=7)

    def _events(self, data: EventsInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        leagues = list(dict.fromkeys(data.leagues))
        today = self._today(leagues)
        words = [w for w in data.query.lower().split() if len(w) > 1]
        games = [c for c in today.get("games") or [] if self._in_window(c, data.when)
                 and all(w in _game_name(c).lower() for w in words)]
        events = []
        for card in games[:MAX_EVENTS]:
            event: Dict[str, Any] = {"league": card.get("league"), "event_id": card.get("id"), "game": _game_name(card),
                                     "when": card.get("when"), "live": bool(card.get("live"))}
            prediction = card.get("prediction") or {}
            if prediction:
                event["ai"] = {"pick": prediction.get("pick"), **_chance(prediction.get("pick_prob")),
                               "home_win": _p(prediction.get("home_win")),
                               "predicted_score": {"home": prediction.get("home_score"), "away": prediction.get("away_score")}}
            home = ((card.get("markets") or {}).get("h2h") or {}).get("home") or {}
            if home.get("fair") is not None:
                event["market_home_win"] = _p(home["fair"])
            out = [name for side in ("away", "home") for name in (card.get("injuries") or {}).get(side, [])
                   if "(Out)" in name or "(Doubtful)" in name]
            if out:
                event["out"] = out[:6]
            event["markets"] = sorted((card.get("markets") or {}).keys())
            events.append(event)
        content = {"events": events, "matching": len(games), "shown": len(events),
                   "errors": today.get("errors") or {}, "odds_as_of": today.get("odds_as_of"), "source": ODDS_SOURCE}
        if not events:
            content["hint"] = "No games match: try another day, another league or an empty query"
        summary = f"sports_find_events({', '.join(leagues)}, {data.when}): " + (
            "; ".join(f"{e['game']} [{e['league']}:{e['event_id']}]" for e in events[:8]) or "no games")
        return content, summary, []

    def _odds(self, data: OddsInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        card, today = self._card(data.league, data.event_id)
        outcomes = []
        for market in MARKETS:
            for side, offer in ((card.get("markets") or {}).get(market) or {}).items():
                ai, fair = _p(offer.get("ai")), _p(offer.get("fair"))
                outcomes.append({
                    "ref": {"league": data.league, "id": card["id"], "market": market, "side": side},
                    "label": offer.get("label"), "point": offer.get("point"),
                    "odds": {"decimal": offer.get("price"), "american": offer.get("american")},
                    "book": offer.get("book"), "books_quoting": offer.get("books"),
                    "ai_chance": ai, "ai_one_in": one_in(ai), "market_chance": fair, "market_one_in": one_in(fair),
                    "chance_label": chance_label(ai if ai is not None else fair),
                    "push_chance": _p(offer.get("push")) or None,
                    "ai_chance_given_no_push": _no_push(ai, offer.get("push")),
                    "fair_odds": round(1 / ai, 2) if ai else None, "edge": _p(offer.get("ev")),
                })
        # The markets brain's blend for the game-winner outcomes (computed in code from cached sources)
        blend = (self._market_detail(data.league, str(card["id"])) or {}).get("blend") or {}
        for outcome in outcomes:
            chance = (blend.get("p") or {}).get(outcome["ref"]["side"]) if outcome["ref"]["market"] == "h2h" else None
            if chance is not None:
                outcome["blend_chance"] = round(float(chance), 4)
        as_of = (today.get("odds_as_of") or {})
        content = {"game": _game_name(card), "when": card.get("when"), "live": bool(card.get("live")),
                   "outcomes": outcomes, "odds_as_of": as_of.get(data.league) if isinstance(as_of, dict) else as_of,
                   "source": ODDS_SOURCE}
        if blend:
            content["markets_say"] = {"confidence": blend.get("confidence"), "disagree": bool(blend.get("disagree")),
                                      "spread_points": blend.get("spread_pp"), "as_of": blend.get("as_of"),
                                      "note": "blend_chance is the app's best estimate; the market usually wins"}
        if card.get("live"):
            content["hint"] = "This game has started: its legs can't go on a ticket"
        summary = f"odds_get_prices({_game_name(card)}): " + "; ".join(
            f"{o['label']} [{_ref_text(o['ref'])}]" for o in outcomes)
        return content, summary, []

    def _market_detail(self, league: str, event_id: str) -> Optional[Dict[str, Any]]:
        """The game's "Markets say" from the service (cached sources only: never a fetch or a credit)."""
        service = getattr(self.lab, "service", None)
        detail = getattr(service, "markets_detail", None)
        if not callable(detail):
            return None
        try:
            return detail(league, event_id)
        except NotFound:
            return None
        except Exception:
            logging.exception("Couldn't read the markets for %s", event_id)
            return None

    def _game_names(self, league: str, event_id: str) -> Tuple[Optional[str], Optional[str]]:
        """(home, away) from the saved odds board or the logged consensus, never fetched."""
        service = getattr(self.lab, "service", None)
        saved = getattr(getattr(service, "odds", None), "saved_games", None)
        try:
            board = saved(league) if callable(saved) else None
            game = next((g for g in (board or (None, []))[1] if g.get("id") == event_id), None)
            if game:
                return game.get("home"), game.get("away")
            store = getattr(service, "store", None)
            row = store.consensus_row(league, event_id=event_id) if store is not None else None
            if row:
                return row.get("home"), row.get("away")
        except Exception:
            logging.exception("Couldn't name the teams of %s", event_id)
        return None, None

    def _markets(self, data: OddsInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        detail = self._market_detail(data.league, data.event_id)
        if not detail or not detail.get("sources"):
            raise ToolError("not_found", f"No market opinions for {data.event_id} in {data.league} yet",
                            "Check the event id with sports_find_events; if it's right, say the markets haven't "
                            "priced this game yet")
        home, away = self._game_names(data.league, data.event_id)
        blend = detail.get("blend") or {}
        rounded = lambda p: {o: round(float(v), 4) for o, v in (p or {}).items()}  # noqa: E731
        sources = [{"name": source.get("name"), "kind": source.get("kind"), "chance": rounded(source.get("p")),
                    "status": source.get("status"), "as_of": source.get("as_of"), "detail": source.get("detail")}
                   for source in detail.get("sources") or []]
        content: Dict[str, Any] = {
            "game": f"{away} @ {home}" if home and away else None, "home": home, "away": away,
            "blend": rounded(blend.get("p")) or None, "confidence": blend.get("confidence"),
            "disagree": bool(blend.get("disagree")), "spread_points": blend.get("spread_pp"),
            "weights": {name: round(float(w), 3) for name, w in (blend.get("weights") or {}).items()},
            "sources": sources, "blend_as_of": blend.get("as_of"), "source": MARKETS_SOURCE, "note": MARKETS_NOTE}
        if not blend:
            content["hint"] = "No source is usable right now (all thin, stale or missing): say there's no blend"
        elif blend.get("confidence") == "low":
            # A market price can be listed but not counted (thin or out of date): name it rather than deny it
            uncounted = [source.get("name") for source in detail.get("sources") or []
                         if source.get("kind") in consensus.MARKET_KINDS and source.get("status") in ("thin", "stale")]
            content["hint"] = (f"Low confidence: the only market prices ({', '.join(uncounted)}) are thin or out of "
                               "date and not counted; the chance rests on ratings; say so" if uncounted else
                               "Low confidence: no market prices this game, only ratings; say so")
        summary = f"markets_consensus({content['game'] or data.event_id} [{data.league}:{data.event_id}]): " + (
            ", ".join(f"{o} {p:.0%}" for o, p in (content["blend"] or {}).items()) or "no blend")
        return content, summary, []

    def _find_player(self, name: str, league: str) -> Dict[str, Any]:
        found = self.lab.player_search(name, league)
        players = found.get("players") if isinstance(found, Mapping) else found
        players = [p for p in players or [] if isinstance(p, Mapping) and p.get("id")]
        if not players:
            reason = found.get("reason") if isinstance(found, Mapping) else None
            raise ToolError("not_found", reason or f"No player called {name!r} in {league}",
                            "Check the spelling with the user, or try another league")
        exact = [p for p in players if str(p.get("name") or "").lower() == name.lower()]
        if len(exact) == 1 or len(players) == 1:
            return dict((exact or players)[0])
        candidates = [{"name": p.get("name"), "team": p.get("team"), "position": p.get("position")} for p in players[:5]]
        raise ToolError("ambiguous", f"Several players match {name!r}: " + "; ".join(
            f"{c['name']} ({c['team']})" for c in candidates), "Ask the user which one they mean")

    def _player(self, data: PlayerInput) -> Dict[str, Any]:
        self._use_leagues([data.league])  # a player's card prices their props from the league's odds
        hit = self._find_player(data.name, data.league)
        player = self.lab.player(hit.get("league") or data.league, hit["id"]) or {}
        if player.get("reason") and not player.get("name"):
            raise ToolError("not_found", str(player["reason"]), "Tell the user ESPN has nothing on this player now")
        return player

    def _profile(self, data: PlayerInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        player = self._player(data)
        keys = ("id", "league", "name", "team", "position", "status", "injury", "next_game", "note", "season_label",
                "season", "last5", "last10", "sample", "games", "splits", "news", "missing", "as_of")
        content = trim({key: player.get(key) for key in keys})
        content["source"] = ESPN_SOURCE
        return content, f"player_get_profile({player.get('name')}, {data.league})", [player_card(player, "player_get_profile")]

    def _props(self, data: PlayerInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        player = self._player(data)
        profile = {"games", "splits", "news", "season", "last5", "last10", "sample", "headshot", "season_label", "note"}
        content = trim({key: value for key, value in player.items() if key not in profile}, items=12)
        content.setdefault("source", "ESPN (DraftKings lines) and the brain's player model")
        return content, f"player_get_props({player.get('name')}, {data.league})", [player_card(player, "player_get_props")]

    def _news(self, data: NewsInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        content: Dict[str, Any] = {"source": ESPN_SOURCE}
        if data.player.strip():
            player = self._player(PlayerInput(name=data.player.strip(), league=data.league))
            content.update(player=player.get("name"), status=player.get("status"), injury=player.get("injury"),
                           items=trim(player.get("news") or [], items=5))
            return trim(content), f"news_get({player.get('name')})", []
        if data.event_id.strip():
            card, _ = self._card(data.league, data.event_id.strip())
            content.update(game=_game_name(card), injuries=card.get("injuries") or {})
            return trim(content, items=15), f"news_get({_game_name(card)})", []
        today = self._today([data.league])
        content["games"] = [{"game": _game_name(c), "event_id": c.get("id"), "injuries": c.get("injuries") or {}}
                            for c in (today.get("games") or [])[:MAX_EVENTS] if any((c.get("injuries") or {}).values())]
        return trim(content, items=15), f"news_get({data.league})", []

    def _brain(self, data: LegsInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        refs = [ref.model_dump() for ref in data.legs]
        self._use_leagues(ref["league"] for ref in refs)
        priced = self.lab.price(self.uid, {"type": "single", "stake": "1.00", "legs": refs}) or {}
        legs = []
        for leg in priced.get("legs") or []:
            ai, market = _p(leg.get("ai_prob")), _p(leg.get("market_prob"))
            legs.append({"index": leg.get("index"), "label": leg.get("label"), "game": leg.get("game"),
                         "odds": leg.get("odds"), "ai_chance": ai, "ai_one_in": one_in(ai), "market_chance": market,
                         "market_one_in": one_in(market), "push_chance": leg.get("push_prob") or None,
                         "ai_chance_given_no_push": _no_push(ai, leg.get("push_prob")),
                         "chance_label": leg.get("chance_label"), "fair_odds": leg.get("fair_odds"),
                         "edge": leg.get("edge"), "book_keeps": leg.get("book_cut"), "book": leg.get("book"),
                         "started": leg.get("started") or None, "ref": ref_of(leg)})
        content = {"legs": trim(legs, items=MAX_LEGS), "removed": priced.get("removed") or [],
                   "odds_as_of": priced.get("odds_as_of"), "source": BRAIN_SOURCE}
        return content, "brain_price_legs: " + "; ".join(f"{l['label']} [{_ref_text(l['ref'])}]" for l in legs), []

    def _ticket_body(self, data: TicketInput) -> Dict[str, Any]:
        """The Lab's ticket body, with the stake checked over the whole ticket (it is per bet)."""
        bets = bet_count(data.type, len(data.legs), data.k, data.system)
        stake = data.stake
        if data.split and bets > 1:
            # Code does the division, rounding down, so the bets never add up to more than the user's stake
            stake = (data.stake / bets).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            if stake <= 0:
                raise ToolError("rejected", f"{_cents(data.stake)} can't be shared across {bets} bets",
                                "Suggest fewer bets, such as a single or a multi")
        self._check_amount("stake", stake, "stake", bets)
        legs = [r.model_dump() for r in data.legs]
        self._use_leagues(ref["league"] for ref in legs)
        body: Dict[str, Any] = {"type": data.type, "stake": _cents(stake), "legs": legs}
        if data.k is not None:
            body["k"] = data.k
        if data.system:
            body["system"] = data.system
        return body

    def _priced(self, data: TicketInput, source: str, title: Optional[str] = None,
                extra: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        body = self._ticket_body(data)
        priced = self.lab.price(self.uid, body) or {}
        content = {"ticket": ticket_facts(priced), "removed": priced.get("removed") or [],
                   "moved": priced.get("moved") or [], "odds_as_of": priced.get("odds_as_of"), "source": BRAIN_SOURCE}
        if priced.get("removed"):
            content["hint"] = "Some legs are no longer offered; tell the user which"
        summary = f"{source}({ticket_title(priced)}, stake {body['stake']} per bet): " + "; ".join(
            f"{leg.get('label')} [{_ref_text(ref_of(leg))}]" for leg in priced.get("legs") or [])
        cards = [ticket_card(priced, source, title, extra)] if priced.get("legs") else []
        return content, summary, cards

    def _price(self, data: TicketInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        return self._priced(data, "ticket_price")

    def _propose(self, data: TicketInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        content, summary, cards = self._priced(data, "slip_propose", data.title, {"proposal": True})
        content["status"] = "Shown to the user as a card with an 'Add to slip' button; nothing is saved or added"
        return content, summary, cards

    def _build(self, data: BuildInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        self._check_amount("stake", data.stake, "stake")
        self._check_amount("target", data.target, "target")
        body = {"leagues": list(dict.fromkeys(data.leagues)), "stake": _cents(data.stake), "target": _cents(data.target),
                "max_legs": data.max_legs, "tier": data.tier}
        self._use_leagues(body["leagues"])
        built = self.lab.build(self.uid, body) or {}
        tickets = [t for t in built.get("tickets") or [] if isinstance(t, dict)]
        frame = ("stake", "target_return", "target_profit", "multiple", "target_odds", "ceiling_chance",
                 "ceiling_one_in", "no_edge", "lottery", "short_odds", "edge", "expected_cost", "verdict", "notes",
                 "errors", "odds_as_of")
        content = trim({key: built.get(key) for key in frame})
        content["tickets"] = [ticket_facts(ticket) for ticket in tickets]
        content["source"] = BRAIN_SOURCE
        if not tickets:
            content["hint"] = "No ticket fits: explain the verdict and notes; don't invent one"
        summary = (f"ticket_build({', '.join(body['leagues'])}, stake {body['stake']} to {body['target']}, max "
                   f"{data.max_legs} legs{', ' + data.tier if data.tier else ''}): " + " | ".join(
                       f"#{i + 1} " + "; ".join(f"{leg.get('label')} [{_ref_text(ref_of(leg))}]"
                                                for leg in ticket.get("legs") or [])
                       for i, ticket in enumerate(tickets)) or "no ticket")
        cards = [ticket_card(ticket, "ticket_build", extra={"verdict": built.get("verdict")}) for ticket in tickets]
        return content, summary, cards

    def _analyse(self, data: TicketInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        body = self._ticket_body(data)
        # The chat writes its own words, so the analysis is asked for without Claude's explanation
        quiet = "explain" in inspect.signature(self.lab.analyse).parameters
        result = (self.lab.analyse(self.uid, body, explain=False) if quiet else self.lab.analyse(self.uid, body)) or {}
        ticket = result.get("ticket") or {}
        alternatives = []
        for alt in result.get("alternatives") or []:
            priced = alt.get("ticket") or {}
            alternatives.append({"title": alt.get("title"), "refs": alt.get("refs"),
                                 "ticket": ticket_facts(priced) if priced else None})
        content = {"ticket": ticket_facts(ticket), "leg_notes": trim(result.get("legs") or [], items=MAX_LEGS),
                   "alternatives": trim(alternatives, items=4), "source": BRAIN_SOURCE}
        summary = f"ticket_analyse({ticket_title(ticket)}, stake {body['stake']} per bet): " + "; ".join(
            f"{leg.get('label')} [{_ref_text(ref_of(leg))}]" for leg in ticket.get("legs") or [])
        cards = [ticket_card(ticket, "ticket_analyse", extra={"notes": result.get("legs") or [],
                                                             "alternatives": result.get("alternatives") or []})
                 ] if ticket.get("legs") else []
        return content, summary, cards

    def _rollover(self, data: RolloverInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        self._check_amount("stake", data.start, "starting amount")
        self._check_amount("min_odds", data.min_odds, "minimum odds")
        body = {"start": _cents(data.start), "days": data.days, "min_odds": _cents(data.min_odds),
                "legs_per_day": data.legs_per_day}
        compare = self.lab.rollover_compare(body) or {}
        strategies = {}
        for name, entry in (compare.get("strategies") or {}).items():
            views = {}
            for basis in ("market", "ai"):
                stats = entry.get(basis)
                if stats:
                    views[basis] = {key: stats.get(key) for key in (
                        "p_day", "chance_complete", "complete_one_in", "chance_profit", "median_end",
                        "chance_half_or_less", "expected_end", "expected_cost", "most_likely_end", "over_by_day_2")}
            strategies[name] = {"title": entry.get("title"), "rule": entry.get("rule"), **views}
        content = trim({key: compare.get(key) for key in (
            "start", "days", "min_odds", "legs_per_day", "target", "p_market", "p_market_estimated", "p_ai",
            "summary", "requirement", "cost", "sentences", "warnings")})
        content["strategies"] = trim(strategies)
        content["source"] = "The brain's exact rollover maths"
        summary = (f"rollover_simulate(start {body['start']}, {data.days} days at {body['min_odds']}, "
                   f"{data.legs_per_day} leg(s) a day)")
        return content, summary, [plan_card(body, compare)]

    def _pasted(self, data: PastedInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        if data.pasted_id not in self.pasted:
            raise ToolError("not_found", f"No slip {data.pasted_id} was pasted in this chat",
                            "Use the pasted_id the context gives; for another slip, the user pastes it")
        view = self.pasted[data.pasted_id]
        if view is None:  # pasted earlier in this chat: its view is re-priced from the cached boards
            view = self.pasted[data.pasted_id] = self.lab.pasted_view(self.uid, data.pasted_id) or {}
        legs = len(view.get("legs") or [])
        drop = sorted(set(data.drop))
        if any(index < 1 or index > legs for index in drop):
            raise ToolError("invalid_input", f"The legs on this slip are numbered 1 to {legs}",
                            "Use the leg numbers from the slip")
        if len(drop) >= legs:
            raise ToolError("rejected", "That leaves no legs on the slip", "Leave at least one leg in")
        left = legs - len(drop)
        own_kind = view.get("bet_type") if view.get("bet_type") in TICKET_TYPES else None
        kind = data.ticket_type or own_kind or ("multi" if left > 1 else "single")
        k = data.k if data.k is not None else (view.get("k") if kind == "round_robin" else None)
        system = system_key(data.system or (view.get("system") if kind == "system" else None))
        bets = bet_count(kind, left, k, system)
        stake = data.stake
        if stake is not None and data.split and bets > 1:
            # Code does the division, rounding down, so the bets never add up to more than the user's stake
            stake = (stake / bets).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            if stake <= 0:
                raise ToolError("rejected", f"{_cents(data.stake)} can't be shared across {bets} bets",
                                "Suggest fewer bets, such as a single or a multi")
        own = amount_of(view.get("stake"))
        if stake is not None:
            self._check_amount("stake", stake, "stake", bets)
        elif own is not None:
            self._check_amount("stake", own, "stake", bets)  # the slip's stake on another shape may cost more
        elif data.ticket_type is not None and data.ticket_type != own_kind:
            raise ToolError("amount_not_given", "The slip shows no stake, so a new shape needs one from the user",
                            "Ask the user what stake they want; never choose one for them")
        options: Dict[str, Any] = {"drop": drop, "stake": _cents(stake) if stake is not None else None}
        if data.ticket_type is not None:
            options["ticket_type"] = data.ticket_type
        if data.k is not None:
            options["k"] = data.k
        if data.system:
            options["system"] = data.system
        analysis = self.lab.analyse_pasted(self.uid, data.pasted_id, **options) or {}
        content = pasted_analysis_facts(analysis)
        ticket = analysis.get("ticket") or {}
        summary = (f"pasted_slip_analyse(slip {data.pasted_id}{', without legs ' + ','.join(map(str, drop)) if drop else ''}"
                   f", {ticket_title(ticket)}, stake {analysis.get('stake')} per bet): "
                   + "; ".join(f"{leg.get('index')}. {pasted_label(leg)}"
                               for leg in (analysis.get("pasted") or view).get("legs") or []))
        cards = [pasted_card(analysis.get("pasted") or view, analysis, "pasted_slip_analyse")]
        return content, summary, cards

    def _context(self, data: NoInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        me = self.lab.me(self.uid) or {}
        keys = ("odds_format", "favourite_sports", "risk_tier", "budget", "reality_check_minutes", "cooloff_until",
                "credits", "age_confirmed")
        content = trim({key: me.get(key) for key in keys})
        content["source"] = "The user's settings (read only)"
        return content, "user_get_context", []
