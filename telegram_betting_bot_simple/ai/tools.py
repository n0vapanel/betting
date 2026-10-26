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
"""

from __future__ import annotations

import datetime
import inspect
import logging
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, ValidationError

from ..dashboard.service import CallRejected, NotFound
from ..data.espn_client import LEAGUES as ESPN_LEAGUES
from ..data.leagues import POPULAR_LEAGUES
from ..engine.tickets import chance_label, combinations, one_in

LEAGUE_KEYS: Tuple[str, ...] = tuple(sorted(set(ESPN_LEAGUES) | set(POPULAR_LEAGUES)))
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
# Tools that price or build tickets: off while picks are paused after a warning sign
PICK_TOOLS = frozenset({"odds_get_prices", "brain_price_legs", "ticket_price", "ticket_build", "ticket_analyse",
                        "rollover_simulate", "slip_propose", "player_get_props"})


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

TOOLS: List[Dict[str, Any]] = [
    {"name": "sports_find_events", "strict": True,
     "description": "List upcoming and live games with the AI's predicted winner and chance. Call this first "
                    "whenever the user mentions games, teams or a sport, to get current event ids; never "
                    "guess an id. query filters by team name (empty string for every game).",
     "input_schema": _obj({"leagues": _LEAGUES, "query": _STR,
                           "when": {"type": "string", "enum": ["today", "tomorrow", "week"]}})},
    {"name": "odds_get_prices", "strict": True,
     "description": "Every market for one game: best price, the AI's chance, the market's margin-free chance, "
                    "fair odds and a ref for each outcome. Call this before quoting any odds or chance for a game.",
     "input_schema": _obj({"league": _LEAGUE, "event_id": _STR})},
    {"name": "player_get_profile", "strict": True,
     "description": "A player's season and recent averages, last 10 games, home/away splits, injury status, "
                    "next game and dated news. Call this when the user asks about a player's form.",
     "input_schema": _obj({"name": {"type": "string", "description": "The player's name as the user wrote it"},
                           "league": _LEAGUE})},
    {"name": "player_get_props", "strict": True,
     "description": "A player's posted prop lines with hit rates (with sample sizes), the brain's projection "
                    "and its chance of the over. Call this before talking about any player prop.",
     "input_schema": _obj({"name": _STR, "league": _LEAGUE})},
    {"name": "news_get", "strict": True,
     "description": "Dated injury and lineup news with sources: for a player (player = name), one game "
                    "(event_id) or the league's games (both empty strings).",
     "input_schema": _obj({"league": _LEAGUE, "player": _STR, "event_id": _STR})},
    {"name": "brain_price_legs", "strict": True,
     "description": "The brain's view of each leg on its own: the AI's chance, the market's chance, fair odds, "
                    "edge and the bookmaker's cut. Use it to compare legs; it prices no ticket.",
     "input_schema": _obj({"legs": _LEGS})},
    {"name": "ticket_price", "strict": True,
     "description": "Price a ticket: combined odds, payout, chance as '1 in X' (AI and market), what the "
                    "bookmaker keeps, the expected result in money, the most likely outcome and warnings. "
                    "Call this before quoting any ticket's odds, payout or chance.",
     "input_schema": _obj(dict(_TICKET), optional=("k", "system", "split"))},
    {"name": "ticket_build", "strict": True,
     "description": "Build up to three tickets that turn the user's stake into about their target, with the "
                    "honest ceiling on the chance and a verdict. Use the user's own stake and target "
                    "(target = money back, stake included); ask if either is missing.",
     "input_schema": _obj({"leagues": _LEAGUES, "stake": {"type": "number"}, "target": {"type": "number"},
                           "max_legs": {"type": "integer", "description": "1 to 6"},
                           "tier": {"type": "string", "enum": list(TIERS)}}, optional=("tier",))},
    {"name": "ticket_analyse", "strict": True,
     "description": "Analyse the user's own ticket: priced, with notes per leg (injuries, AI vs market gap, "
                    "analysts' views) and alternatives the code worked out (drop the weakest leg, singles, "
                    "a round robin).",
     "input_schema": _obj(dict(_TICKET), optional=("k", "system", "split"))},
    {"name": "rollover_simulate", "strict": True,
     "description": "Compare rollover strategies side by side (classic all-in, bank half, pot-and-ladder, "
                    "flat stakes): chance to finish, chance of a profit, typical end, expected cost, the day a "
                    "run most likely ends. Use the user's own start amount and days.",
     "input_schema": _obj({"start": {"type": "number"}, "days": {"type": "integer", "description": "1 to 60"},
                           "min_odds": {"type": "number", "description": "Daily odds, 1.2 to 20; use 2.0 if "
                                                                         "the user gave none"},
                           "legs_per_day": {"type": "integer", "description": "1 to 3"}})},
    {"name": "slip_propose", "strict": True,
     "description": "Show the user a priced ticket card with an 'Add to slip' button. It saves nothing and "
                    "adds nothing: only the user can add it. Use it for the one ticket you suggest.",
     "input_schema": _obj({**_TICKET, "title": {"type": "string", "description": "A short card title"}},
                          optional=("k", "system", "split"))},
    {"name": "user_get_context", "strict": True,
     "description": "The user's settings: odds format, favourite sports, risk tier, weekly budget left, "
                    "cool-off and AI credits left (read only).",
     "input_schema": _obj({})},
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


INPUTS: Dict[str, type] = {
    "sports_find_events": EventsInput, "odds_get_prices": OddsInput, "player_get_profile": PlayerInput,
    "player_get_props": PlayerInput, "news_get": NewsInput, "brain_price_legs": LegsInput,
    "ticket_price": TicketInput, "ticket_build": BuildInput, "ticket_analyse": TicketInput,
    "rollover_simulate": RolloverInput, "slip_propose": TicketInput, "user_get_context": NoInput,
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
            "push_chance": leg.get("push_prob") or None, "chance_label": leg.get("chance_label"),
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


# --- running the tools ----------------------------------------------------------------------------------

STEPS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "sports_find_events": lambda a: f"Finding {'/'.join(l.split('_')[-1].upper() for l in a.get('leagues') or [])} games…",
    "odds_get_prices": lambda a: "Checking the odds…",
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
    """

    def __init__(self, lab: Any, uid: str, tz: ZoneInfo, limits: Optional[Mapping[str, Iterable[Decimal]]] = None,
                 paused: bool = False, now: Optional[Callable[[], datetime.datetime]] = None,
                 leagues: Iterable[str] = (), extra_leagues: int = EXTRA_LEAGUES) -> None:
        self.lab = lab
        self.uid = uid
        self.tz = tz
        self.limits: Dict[str, Set[Decimal]] = {kind: set(values) for kind, values in (limits or {}).items()}
        self.paused = paused
        self.now = now or (lambda: datetime.datetime.now(datetime.timezone.utc))
        self.leagues: Set[str] = set(leagues)
        self.extra_leagues = extra_leagues
        self.extra: Set[str] = set()  # leagues loaded this turn beyond self.leagues
        self.runners: Dict[str, Callable[[Any], Tuple[Dict[str, Any], str, List[Dict[str, Any]]]]] = {
            "sports_find_events": self._events, "odds_get_prices": self._odds, "player_get_profile": self._profile,
            "player_get_props": self._props, "news_get": self._news, "brain_price_legs": self._brain,
            "ticket_price": self._price, "ticket_build": self._build, "ticket_analyse": self._analyse,
            "rollover_simulate": self._rollover, "slip_propose": self._propose, "user_get_context": self._context,
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
                    "fair_odds": round(1 / ai, 2) if ai else None, "edge": _p(offer.get("ev")),
                })
        as_of = (today.get("odds_as_of") or {})
        content = {"game": _game_name(card), "when": card.get("when"), "live": bool(card.get("live")),
                   "outcomes": outcomes, "odds_as_of": as_of.get(data.league) if isinstance(as_of, dict) else as_of,
                   "source": ODDS_SOURCE}
        if card.get("live"):
            content["hint"] = "This game has started: its legs can't go on a ticket"
        summary = f"odds_get_prices({_game_name(card)}): " + "; ".join(
            f"{o['label']} [{_ref_text(o['ref'])}]" for o in outcomes)
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

    def _context(self, data: NoInput) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        me = self.lab.me(self.uid) or {}
        keys = ("odds_format", "favourite_sports", "risk_tier", "budget", "reality_check_minutes", "cooloff_until",
                "credits", "age_confirmed")
        content = trim({key: me.get(key) for key in keys})
        content["source"] = "The user's settings (read only)"
        return content, "user_get_context", []
