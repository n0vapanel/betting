"""The chat assistant: Claude decides which brain tools to call, code computes and checks everything.

Each chat turn runs the same steps, in this order:

1. Warning signs, screened by code rules first (free, and they work even when
   Claude is down): chasing losses, betting to pay bills or debts, distress.
   A strong sign (first-person money talk: "win my money back", "pay my
   rent") gets a fixed support message instead of picks, an rg_event for
   the owner to review and a 24-hour pause on picks. Code decides this,
   never the model. The small router model (step 3) can raise one too, and
   it is the judge of wording that sports talk shares ("chasing losses",
   "win it back"): such a soft sign never pauses picks on its own.
2. Gates: the person has confirmed their age, isn't on a break, and has AI
   credits left today. A failed gate gets a template answer and costs nothing.
3. The router (Claude Haiku, structured output) reads the message: what is
   asked (look up, build, analyse, plan, explain, help), the sports, stake,
   target, legs, risk tier, players, and any warning sign. Code fills in
   defaults and, if something required is still missing, asks one question
   with tappable chips.
4. The tool loop (Claude Opus, at most six rounds; effort low for look-ups,
   medium for building, analysing and plans) over the twelve tools in
   ai/tools.py. No tool takes a price or a probability.
5. Cards (tickets, plans, players) are built by code from the tool results.
6. The reply is checked: every number must come from this turn's tool results
   or the person's own message, and no banned phrase may appear. A failing
   reply is sent back once with the reasons, then replaced by a template.
7. The turn is stored (the last 12 messages are the next turn's context, with
   tool results kept as one-line summaries), and its cost is logged and taken
   from the person's daily credits.

The first reply in a thread carries the AI disclosure (Anthropic's usage
policy: tell people they are talking to an AI at the start of a session).
The system prompt and tool list never change between requests, so the prompt
cache holds; the date and the person's settings travel in their own message.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

from ..dashboard.lab import PAUSE_SETTING, paused_until
from ..dashboard.service import CallRejected, NotFound
from .explain import (
    AIError,
    ClaudeCaller,
    Explanation,
    explain_ticket,
    problems,
    response_text,
    source_numbers,
    template_explanation,
)
from .tools import LEAGUE_KEYS, TOOLS, Toolbox, ToolRun, bet_count, step_text

CHAT_MODEL = "claude-opus-5-5"
ROUTER_MODEL = "claude-haiku-5-5"
DAILY_CENTS = 100
MAX_ROUNDS = 6
RETRY_ROUNDS = 2  # a rejected reply gets one more go: a tool call if it needs one, then the answer
CONTEXT_MESSAGES = 12
MAX_MESSAGE = 2000  # characters a person can send in one message
MAX_CARDS = 4
MAX_CHIPS = 4
PAUSE_SECONDS = 24 * 3600  # picks stay off this long after a strong warning sign
LATE_HOURS = range(1, 5)  # 1:00-4:59 at night: tired decisions are worse ones
DEFAULT_LEAGUES = ["basketball_nba"]
DEFAULT_MAX_LEGS = 3
DEFAULT_LEGS_PER_DAY = 1
MAX_REQUEST_LEAGUES = 4  # each league's odds fetch costs quota

HELPLINE = "Gambling can be addictive · Help 24/7: 1-800-GAMBLER / 0808 8020 133"
DISCLOSURE = ("You're chatting with an AI analyst (Claude, made by Anthropic). Our code works out every number; "
              "the AI explains them and can still be wrong. Nothing here is a tip, and only you add bets to "
              "your slip.")
SUPPORT_MESSAGE = (
    "I'm going to stop here: I won't suggest any bets for the next 24 hours. Trying to get money back by "
    "betting, or betting to pay bills or debts, usually makes things worse, and the odds are set so that most "
    "people lose over time.\n\n"
    "If gambling is causing you stress, free and confidential help is there 24/7: call 1-800-GAMBLER (US) or "
    "0808 8020 133 (UK National Gambling Helpline). You can also set a weekly budget or take a longer break "
    "in Settings."
)
DISTRESS_EXTRA = ("If you're thinking about harming yourself, please call or text 988 (US) or call Samaritans "
                  "on 116 123 (UK) now.")
SUPPORT_CHIPS = ["How do I set a budget?", "How do I take a break?", "Where can I get help?"]
LATE_NOTE = "It's late where you are: decisions made tired tend to be worse ones."
# What the app's own chips send; they never suggest betting more
TICKET_CHIPS = ["Explain the risk", "Make it safer", "Swap the weakest leg"]
PLAN_CHIPS = ["Why do rollovers end early?", "Compare with flat stakes"]
LOOKUP_CHIPS = ["Who does the AI favour?", "Any injury news?"]
HELP_CHIPS = ["What can you do?", "How is this calculated?"]

INTENTS = ("lookup", "build", "analyse", "plan", "explain", "help", "account", "chat")
DEEP_INTENTS = {"build", "analyse", "plan"}  # medium effort; everything else is a look-up at low effort
PICK_INTENTS = {"build", "analyse", "plan"}  # refused outright while picks are paused
STRONG_FLAGS = ("chasing", "debt", "distress")
LEAGUE_NAMES = {"basketball_nba": "NBA", "basketball_wnba": "WNBA", "basketball_ncaab": "NCAAB",
                "americanfootball_nfl": "NFL", "americanfootball_ncaaf": "NCAAF", "icehockey_nhl": "NHL",
                "baseball_mlb": "MLB", "soccer_epl": "EPL", "soccer_spain_la_liga": "La Liga",
                "soccer_italy_serie_a": "Serie A", "soccer_germany_bundesliga": "Bundesliga",
                "soccer_france_ligue_one": "Ligue 1", "soccer_usa_mls": "MLS",
                "soccer_uefa_champs_league": "Champions League", "mma_mixed_martial_arts": "MMA"}

# Code's own screen for warning signs, in two strengths. A strong sign is first-person money talk ("win my
# money back", "lost $200, need to win it back", "pay my rent", "I'm in debt") and pauses picks for a day
# at once, before any model runs. A soft sign is the same kind of wording without the "I" or "my" ("chasing
# losses", "win it back", "pay the rent"), which sports talk can share: it only counts when the router reads
# the message as a warning sign too. Everyday sports questions ("win back-to-back titles", "cover the Bills
# spread", "recover from the loss to Detroit", "make up for the loss of Brown", "win their next game back
# home") match neither.
_WE = r"(?:I|we)(?:'ve|’ve|\s+have|\s+just|\s+already)?"
_MINE = r"(?:(?:some|all)\s+of\s+|all\s+)?(?:my|our)\s+(?:\w+\s+)?"
_WHAT_I_LOST = rf"(?:what|everything|all|(?:all\s+)?the\s+money|the\s+[$£€]?\d[\d,.]*k?)\s+(?:that\s+)?{_WE}\s+lost\b"
_GET_BACK = r"(?:win|winning|won|make|making|made|earn|earning|claw|clawing)"  # not "get": refunds come back too
_RECOVER = r"(?:chas(?:e|es|ed|ing)|recoup(?:s|ed|ing)?|recover(?:s|ed|ing)?|make up for|making up for)"
_BILLS = r"(?:rent|(?-i:bills?|BILLS?)|debts?|loans?|mortgage|credit\s+cards?|car\s+payments?)"  # "Bills": the team
_NOT_SPORTS = r"(?![-‐\s]+(?:to[-‐\s]+back|home|to\s+(?:the|a|an|their|his|her|its)|into)\b)"
_IT_BACK = re.compile(rf"\b(?:{_GET_BACK}|get|getting)\s+(?:it(?:\s+all)?|that|them|this)\s+back\b{_NOT_SPORTS}", re.I)
_OWN_LOSS = re.compile(rf"\b{_WE}\s+lost\b|\blost\s+(?:[$£€]\s?\d|\d[\d,.]*\s*(?:dollars|bucks|quid|k)\b)"
                       rf"|\b(?:my|our)\s+(?:\w+\s+)?loss(?:es)?\b", re.I)

RISK_RULES: Dict[str, "re.Pattern[str]"] = {
    "chasing": re.compile(
        rf"\b{_RECOVER}\s+{_MINE}(?:loss(?:es)?|money)\b"
        rf"|\b{_GET_BACK}\s+(?:back\s+{_MINE}(?:loss(?:es)?|money|cash|bankroll)\b"
        rf"|{_MINE}(?:loss(?:es)?|money|cash|bankroll)\s+back\b)"
        rf"|\b(?:get|getting|got)\s+(?:back\s+{_MINE}loss(?:es)?\b|{_MINE}loss(?:es)?\s+back\b)"
        rf"|\b(?:{_RECOVER}|{_GET_BACK}|get|getting)\s+(?:back\s+)?{_WHAT_I_LOST}"
        rf"|\blost\s+(?:all\s+(?:of\s+)?)?my\s+(?:rent|savings|paycheck|wages|pay|house|car)\b"
        rf"|\b{_WE}\s+lost\s+(?:everything|it all)\b", re.I),
    "debt": re.compile(
        rf"\b(?:pay|paying|cover|covering|clear|clearing|make)\s+(?:off\s+)?(?:my|our)\s+(?:\w+\s+)?{_BILLS}\b"
        rf"|\b(?:bet|betting|gamble|gambling|use|using|spend|spending|put|putting)\s+(?:all\s+(?:of\s+)?)?(?:my|our)\s+"
        rf"(?:rent|mortgage|(?-i:bills?|BILLS?)|savings|paycheck|wages|grocery|food)\b"
        rf"|\b(?:I|we)(?:'m|’m|\s+am|'re|’re|\s+are)\s+(?:\w+\s+){{0,2}}in\s+debt\b|\b(?:my|our)\s+debts?\b"
        rf"|\bloan\s+sharks?\b|\b(?:I|we)\s+(?:still\s+|really\s+)?owe\s+(?:\w+\s+){{0,3}}"
        rf"(?:money|[$£€]\s?\d|\d[\d,.]*\s*(?:dollars|bucks|quid|k)\b)", re.I),
    "distress": re.compile(
        r"\bsuicid|\bkill\s+myself\b|\bend\s+my\s+life\b|\bwant\s+to\s+die\b"
        r"|\b(?:I|I'm|I’m|I\s+am|I\s+feel|I've|I’ve)\b[^.!?]{0,30}\b(?:hopeless|no\s+way\s+out|can(?:'|’)?t\s+go\s+on"
        r"|end\s+it\s+all)\b"
        r"|\bcan(?:'|’)?t\s+stop\s+(?:betting|gambling)\b|\baddicted\s+to\s+(?:betting|gambling)\b"
        r"|\bgambling\s+(?:problem|addiction)\b", re.I),
}
SOFT_RULES: Dict[str, "re.Pattern[str]"] = {
    "chasing": re.compile(
        rf"\b(?:{_GET_BACK}|get|getting)\s+(?:it(?:\s+all)?|that|them|this|losses|money)\s+back\b{_NOT_SPORTS}"
        rf"|\b{_GET_BACK}\s+(?:[$£€]\s?\d[\d,.]*k?|\d[\d,.]*\s*(?:dollars|bucks|quid))\s+back\b{_NOT_SPORTS}"
        rf"|\b{_RECOVER}\s+(?:(?:the|those|these|some|all|all\s+the|all\s+those)\s+)?(?:loss(?:es)?|lost\s+money)\b"
        rf"(?!\s+of\b)|\bdouble\s+or\s+nothing\b|\blost\s+(?:everything|it all)\b", re.I),
    "debt": re.compile(
        rf"\b(?:pay|paying|cover|covering|clear|clearing)\s+(?:off\s+)?(?:the\s+)?{_BILLS}\b|\bin\s+debt\b"
        rf"|\bowe\s+(?:\w+\s+){{0,3}}(?:money|[$£€]\s?\d)", re.I),
    "distress": re.compile(r"\bhopeless\b|\bno\s+way\s+out\b|\bcan(?:'|’)?t\s+go\s+on\b|\bend\s+it\s+all\b|\bdesperate\b",
                           re.I),
}

SYSTEM_PROMPT = """\
You are the AI analyst inside SportsBet AI, a private app where a few friends look at games, odds and \
betting tickets. You explain what the app's brain worked out, in a chat in the app or in Telegram.

How the app works
- The brain is Python code. It computes every number: chances, odds, payouts, what the bookmaker keeps, \
expected results and rollover chances. You never do arithmetic and never estimate a number. If you need a \
number, call a tool; if no tool gives it, say you don't have it.
- Tools return JSON with a source and an "as of" time. Quote numbers as they appear in this turn's tool \
results (rounding is fine, and a chance of 0.5263 may be written 53%). Code checks every number in your \
reply against this turn's tool results and the user's message and rejects the reply if one doesn't match. \
Numbers from earlier turns don't count: call the tool again.
- Never pass a price or a probability to a tool. Legs are named by ref (league, id, market, side), copied \
from odds_get_prices, ticket results or the summaries of earlier turns. Use only the stakes and targets \
the user gave (the request in their message lists them); if one is missing, ask for it. A ticket's stake \
is per bet: singles, round robins and system bets hold several bets, and the whole ticket must fit in the \
user's stake (split shares it across the bets).
- Odds data is rationed: stay with the leagues the user asked about or follows. If a tool answers \
"quota", ask which sport they mean instead of trying other leagues.
- The app builds cards from your tool results (a priced ticket, a rollover plan, a player) and shows them \
under your text, so don't repeat every number: say what matters.

How to talk about a bet
- Give the honest picture first: the chance as "1 in X" by the AI and by the market, what the bookmaker \
keeps, and the expected result in money (usually a loss). Use the chance words the tools give: \
favourite, coin flip, underdog, long shot, lottery ticket.
- "Safe", "Risky" and "Crazy" are only names of risk tiers. Never call a ticket or a leg safe, and always \
mention its chance of losing.
- Never say or imply: lock, guaranteed, sure thing, risk-free, can't lose, win back, chasing losses. Never \
suggest a bigger stake, more legs or betting more often. Never add a leg for the user: suggest at most one \
ticket, with slip_propose, and only the user can add it to their slip.
- If a tool says odds are stale, a game has started or a leg was removed, say so plainly.
- If the user mentions losses they want back, debts, bills or distress, suggest no bets: point them to \
Settings (budget, break) and the helpline 1-800-GAMBLER / 0808 8020 133.

Style
- Short and plain: two to five sentences for most replies, at most about 150 words. No headings, tables \
or emojis; short dash lists are fine. The user is probably on a phone.
- You explain; you don't tip. Saying which leg is weakest, or where the AI and the market disagree and \
why, is fine.
- If the tools can't answer, say so and suggest what the user could ask instead.
"""

ROUTER_PROMPT = f"""\
You read one chat message sent to a sports betting assistant and describe it as JSON. You never answer it.

Fields:
- intent: lookup (games, odds, a team or player, news), build (make a ticket for a stake and target), \
analyse (judge a ticket the user has or describes), plan (a rollover or a plan over several days), \
explain (how something works: odds, parlays, the app's numbers), help (using the app), account (their \
budget, break, credits, settings) or chat (anything else, greetings, thanks).
- leagues: league keys the message is about, from this list only: {", ".join(LEAGUE_KEYS)}. \
"basketball" or "hoops" means basketball_nba unless another league is named; "football" means \
americanfootball_nfl unless the message is clearly about soccer. Empty if no sport is named.
- stake: the money they want to bet, in dollars, or null. target: the money they want back in total, or \
null ("$10 to win $50" is stake 10, target 50; "turn $5 into $100" is stake 5, target 100).
- max_legs: the most legs they want (1 to 6), or null. tier: safe, risky, crazy, or none.
- days, min_odds, legs_per_day: for a rollover or plan, or null.
- players: player names mentioned.
- risk_flags: warning signs of problem gambling in the latest message: chasing (wants to win back or \
recover losses, double or nothing after losing), debt (betting to pay bills, rent or debts, owing money), \
distress (hopeless, desperate, can't stop gambling, self-harm). Only the sender's own money, debts or \
feelings count. Empty when there are none: sports talk is not a warning sign, e.g. a team "losing", \
"chasing a title", "winning back-to-back", "recovering from a loss", "making up for the loss of" a \
player, "covering the Bills spread" or "winning back home".

If the assistant just asked a question and the latest message answers it, keep that question's intent and \
fill in the answer (e.g. "$50" after "How much do you want back?" is target 50).
"""

_NULLABLE_NUMBER = {"anyOf": [{"type": "number"}, {"type": "null"}]}
_NULLABLE_INT = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
ROUTER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "leagues": {"type": "array", "items": {"type": "string", "enum": list(LEAGUE_KEYS)}},
        "stake": _NULLABLE_NUMBER,
        "target": _NULLABLE_NUMBER,
        "max_legs": _NULLABLE_INT,
        "tier": {"type": "string", "enum": ["safe", "risky", "crazy", "none"]},
        "days": _NULLABLE_INT,
        "min_odds": _NULLABLE_NUMBER,
        "legs_per_day": _NULLABLE_INT,
        "players": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "array", "items": {"type": "string", "enum": list(STRONG_FLAGS)}},
    },
    "required": ["intent", "leagues", "stake", "target", "max_legs", "tier", "days", "min_odds", "legs_per_day",
                 "players", "risk_flags"],
    "additionalProperties": False,
}

# The request fields code fills, checks and passes on, with the ranges the API allows
FIELD_RANGES: Dict[str, Tuple[Any, Any]] = {
    "stake": (Decimal("0.01"), Decimal("100000")), "target": (Decimal("0.01"), Decimal("10000000")),
    "max_legs": (1, 6), "days": (1, 60), "min_odds": (Decimal("1.2"), Decimal("20")), "legs_per_day": (1, 3),
}
MONEY_FIELDS = ("stake", "target", "min_odds")


# --- small helpers -----------------------------------------------------------------------------------

def _amount(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).replace(",", "").replace("$", "").strip())
    except (InvalidOperation, ValueError):
        return None
    return number.quantize(Decimal("0.01")) if number.is_finite() else None


def _field(name: str, value: Any) -> Any:
    """A request field kept only when it is a number in the allowed range (None otherwise)."""
    low, high = FIELD_RANGES[name]
    if name in MONEY_FIELDS:
        number = _amount(value)
        return number if number is not None and low <= number <= high else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        return None
    return int(value) if low <= value <= high else None


def _dollars(value: Decimal) -> str:
    return f"${value:,.2f}".replace(".00", "")


def league_name(key: str) -> str:
    return LEAGUE_NAMES.get(key, key.split("_")[-1].upper())


def risk_flags(text: str) -> List[str]:
    """Code's strong warning signs in a message (chasing, debt, distress): first-person money talk."""
    text = text or ""
    flags = [flag for flag, pattern in RISK_RULES.items() if pattern.search(text)]
    # "win it back" is a strong sign next to a loss of their own ("lost $200, need to win it back")
    if "chasing" not in flags and _IT_BACK.search(text) and _OWN_LOSS.search(text):
        flags.insert(0, "chasing")
    return flags


def soft_flags(text: str) -> List[str]:
    """Wording that may be a warning sign or sports talk ("chasing losses", "win it back"), beyond the strong
    ones: it counts only when the router reads the message as a warning sign too."""
    strong = risk_flags(text)
    return [flag for flag, pattern in SOFT_RULES.items() if flag not in strong and pattern.search(text or "")]


@dataclass
class Route:
    """What the router (or the app's request bar) says the person wants; fields are None when not given."""
    intent: str = "chat"
    leagues: List[str] = field(default_factory=list)
    stake: Optional[Decimal] = None
    target: Optional[Decimal] = None
    max_legs: Optional[int] = None
    tier: Optional[str] = None
    days: Optional[int] = None
    min_odds: Optional[Decimal] = None
    legs_per_day: Optional[int] = None
    players: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    routed: bool = False  # whether the router answered (code falls back to rules when it didn't)

    def fields(self) -> Dict[str, Any]:
        out = {"intent": self.intent, "leagues": self.leagues}
        for name in FIELD_RANGES:
            value = getattr(self, name)
            if value is not None:
                out[name] = str(value) if isinstance(value, Decimal) else value
        if self.tier:
            out["tier"] = self.tier
        if self.players:
            out["players"] = self.players
        return out

    def merge(self, pending: Mapping[str, Any], text: str = "") -> None:
        """Fill fields this message left empty from a question the assistant asked before.

        Only when the message answers that question: the same intent, or small talk that gave an
        amount. Without the router, the first number in the message answers the question.
        """
        if not pending or pending.get("intent") not in INTENTS:
            return
        wanted = pending["intent"]
        if not self.routed:
            self._answer_from_text(wanted, pending, text)
        asked = pending.get("asks")
        if asked in ("target", "days") and getattr(self, asked) is None and self.stake is not None \
                and pending.get("stake") is not None:
            # "$40 back" answering "how much back?": a lone amount goes in the field that was asked for
            value, self.stake = self.stake, None
            setattr(self, asked, _field(asked, int(value) if asked == "days" else value))
        answered = any(getattr(self, name) is not None for name in FIELD_RANGES)
        if self.intent != wanted and not (self.intent == "chat" and answered):
            return
        self.intent = wanted
        self.leagues = self.leagues or [k for k in pending.get("leagues") or [] if k in LEAGUE_KEYS]
        for name in FIELD_RANGES:
            if getattr(self, name) is None and pending.get(name) is not None:
                setattr(self, name, _field(name, pending[name]))
        self.tier = self.tier or pending.get("tier")
        self.players = self.players or list(pending.get("players") or [])


    def _answer_from_text(self, wanted: str, pending: Mapping[str, Any], text: str) -> None:
        match = re.search(r"\d+(?:\.\d{1,2})?", text.replace(",", ""))
        if not match:
            return
        order = ("stake", "target") if wanted == "build" else ("stake", "days") if wanted == "plan" else ()
        missing = next((name for name in order if pending.get(name) is None), None)
        if missing:
            self.intent = wanted
            value = match.group(0)
            setattr(self, missing, _field(missing, int(float(value)) if missing == "days" else value))


def parse_route(data: Any) -> Route:
    """The router's JSON, checked field by field (a value out of range is dropped, not trusted)."""
    if not isinstance(data, Mapping):
        return Route()
    route = Route(routed=True)
    route.intent = data.get("intent") if data.get("intent") in INTENTS else "chat"
    route.leagues = list(dict.fromkeys(k for k in data.get("leagues") or [] if k in LEAGUE_KEYS))[:MAX_REQUEST_LEAGUES]
    for name in FIELD_RANGES:
        setattr(route, name, _field(name, data.get(name)))
    route.tier = data.get("tier") if data.get("tier") in ("safe", "risky", "crazy") else None
    route.players = [str(p).strip()[:60] for p in data.get("players") or [] if str(p).strip()][:5]
    route.flags = [flag for flag in data.get("risk_flags") or [] if flag in STRONG_FLAGS]
    return route


@dataclass
class Turn:
    """What one turn produced, before it is stored."""
    text: str
    kind: str  # answer, template, clarify, support, gate, paused, error
    cards: List[Dict[str, Any]] = field(default_factory=list)
    chips: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    pending: Optional[Dict[str, Any]] = None
    checked: Optional[bool] = None
    problems: List[str] = field(default_factory=list)


@dataclass
class Spend:
    """What a turn cost, per model (logged to analyst_calls and taken from the person's credits)."""
    by_model: Dict[str, List[float]] = field(default_factory=dict)  # model -> [dollars, tokens in, tokens out]

    def add(self, model: str, dollars: float, tokens_in: int = 0, tokens_out: int = 0) -> None:
        row = self.by_model.setdefault(model, [0.0, 0, 0])
        row[0] += dollars
        row[1] += tokens_in
        row[2] += tokens_out

    @property
    def dollars(self) -> float:
        return sum(row[0] for row in self.by_model.values())

    @property
    def tokens(self) -> int:
        return int(sum(row[1] + row[2] for row in self.by_model.values()))


# --- the assistant -------------------------------------------------------------------------------------

class Assistant:
    def __init__(self, client: Any, lab: Any, store: Any, chat_model: str = CHAT_MODEL,
                 router_model: str = ROUTER_MODEL, daily_cents: int = DAILY_CENTS,
                 tz: ZoneInfo = ZoneInfo("America/New_York"), clock: Callable[[], float] = time.time) -> None:
        self.caller = ClaudeCaller(client)
        self.lab = lab
        self.store = store
        self.chat_model = chat_model
        self.router_model = router_model
        self.daily_cents = daily_cents
        self.tz = tz
        self.clock = clock

    # --- credits and pauses ---

    def credits(self, uid: str) -> Dict[str, Any]:
        """Today's AI allowance in US cents: {used_cents, daily_cents, left_cents}."""
        used = self.store.ai_used_today(uid, now=self.clock())
        return {"used_cents": round(used, 2), "daily_cents": self.daily_cents,
                "left_cents": round(max(0.0, self.daily_cents - used), 2)}

    def picks_paused_until(self, uid: str) -> Optional[float]:
        """When picks come back on after a warning sign (epoch seconds), or None if they're on."""
        return paused_until(self.store.user(uid), self.clock())

    def _pause_picks(self, uid: str) -> float:
        until = max(self.clock() + PAUSE_SECONDS, self.picks_paused_until(uid) or 0.0)
        self.store.update_user(uid, {"settings": {PAUSE_SETTING: until}})
        return until

    def _record(self, uid: str, spend: Spend, ok: bool) -> None:
        for model, (dollars, tokens_in, tokens_out) in spend.by_model.items():
            if dollars <= 0 and not tokens_in and not tokens_out:
                continue
            self.store.log_analyst_call("assistant", model, None, int(tokens_in), int(tokens_out), dollars, ok)
        if spend.dollars > 0:
            self.store.add_ai_usage(uid, spend.dollars * 100, now=self.clock())

    def _local(self, epoch: float) -> str:
        moment = datetime.datetime.fromtimestamp(epoch, self.tz)
        return f"{moment:%a %b} {moment.day}, {moment:%-I:%M %p %Z}"

    # --- explaining a ticket (for the analyse screen) ---

    def explain(self, uid: str, fact_sheet: Dict[str, Any]) -> Dict[str, Any]:
        """Claude's verdict, per-leg notes and case against for a ticket, checked against the fact sheet.

        Returns {verdict, per_leg: [{index, text}], case_against, source ("claude" or "template"), model,
        cost}. Raises CallRejected, with the reason to show beside the code-only analysis, while picks are
        paused after a warning sign or when the person has no AI credits left today.
        """
        paused = self.picks_paused_until(uid)
        if paused:
            raise CallRejected(f"Picks are paused until {self._local(paused)} after an earlier message about losses "
                               f"or money worries, so the AI won't comment on tickets until then; the numbers above "
                               f"are complete. {HELPLINE}")
        if self.credits(uid)["left_cents"] <= 0:
            raise CallRejected("You've used today's AI allowance; it resets at midnight (US Eastern). "
                               "The numbers above are complete.")
        spend = Spend()
        try:
            result: Explanation = explain_ticket(fact_sheet, self.caller, self.chat_model)
        except AIError as exc:
            logging.warning("Claude couldn't explain a ticket: %s", exc)
            spend.add(self.chat_model, exc.cost)
            self._record(uid, spend, False)
            return {**template_explanation(fact_sheet), "source": "template", "model": None,
                    "cost": round(exc.cost, 5)}
        if result.notes:
            logging.info("Ticket explanation fell back or retried: %s", "; ".join(result.notes))
        spend.add(result.model or self.chat_model, result.cost, result.input_tokens, result.output_tokens)
        self._record(uid, spend, result.source == "claude")
        return result.as_dict()

    # --- a chat turn ---

    def reply(self, uid: str, thread_id: Optional[int] = None, message: Optional[str] = None,
              request: Optional[Mapping[str, Any]] = None, slip: Optional[Mapping[str, Any]] = None,
              on_step: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Answer one message (or a request-bar request, or a slip to analyse) in a thread.

        Returns {thread_id, message_idx, reply: {text, cards, chips}, credits, disclosure?}. ``on_step`` is
        called with progress lines ("Pricing 3 legs…") from this thread as the turn runs. Raises
        CallRejected for an empty or over-long message and NotFound for someone else's thread.
        """
        text = " ".join(str(message or "").split())
        if len(text) > MAX_MESSAGE:
            raise CallRejected(f"Keep messages under {MAX_MESSAGE} characters")
        if not text and not request and not slip:
            raise CallRejected("Type a message")
        step = on_step or (lambda line: None)
        user = self.store.get_or_create_user(uid)
        if thread_id is None:
            thread = self.store.create_thread(uid, title=None)
        else:
            thread = self.store.thread(uid, thread_id)
            if thread is None:
                raise NotFound("That chat doesn't exist")
        history = self.store.messages(uid, thread["id"], last=CONTEXT_MESSAGES)
        first = not history
        shown = text or self._describe(request, slip)
        if first:
            self.store.update_thread(uid, thread["id"], title=shown[:60])

        spend = Spend()
        route = Route()
        try:
            turn = self._turn(uid, user, thread["id"], text, request, slip, history, route, spend, step)
        except AIError as exc:
            logging.warning("The assistant couldn't reach Claude: %s", exc)
            spend.add(self.chat_model, exc.cost)
            turn = Turn(f"I couldn't reach the AI just now ({exc}). Try again in a minute.", "error",
                        chips=["Try again"] if text else [])
        except Exception:
            logging.exception("The assistant failed on a turn")
            turn = Turn("Something went wrong on my side. Try again in a minute.", "error")
        self._record(uid, spend, turn.kind == "answer")

        user_content: Dict[str, Any] = {"text": shown, "route": route.fields()}
        if request:
            user_content["request"] = dict(request)
        if slip:
            user_content["slip"] = dict(slip)
        self.store.add_message(uid, thread["id"], "user", user_content, now=self.clock())
        reply = {"text": turn.text, "cards": turn.cards[:MAX_CARDS], "chips": turn.chips[:MAX_CHIPS], "kind": turn.kind}
        stored = {**reply, "tools": turn.tools, "checked": turn.checked}
        if turn.pending:
            stored["pending"] = turn.pending
        if first:
            stored["disclosure"] = DISCLOSURE
        index = self.store.add_message(uid, thread["id"], "assistant", stored, tokens=spend.tokens,
                                       cost=round(spend.dollars, 6), now=self.clock())
        # message_idx names this reply in the thread, so a button under it reads this reply and no other
        out: Dict[str, Any] = {"thread_id": thread["id"], "message_idx": index, "reply": reply,
                               "credits": self.credits(uid)}
        if first:
            out["disclosure"] = DISCLOSURE
        return out

    def _turn(self, uid: str, user: Dict[str, Any], thread_id: int, text: str, request: Optional[Mapping[str, Any]],
              slip: Optional[Mapping[str, Any]], history: List[Dict[str, Any]], route: Route, spend: Spend,
              step: Callable[[str], None]) -> Turn:
        now = self.clock()
        # 1. Warning signs by code's rules: free, so they run before anything else
        flags = risk_flags(text)
        if flags:
            return self._support(uid, thread_id, text, flags, "rules")
        # 2. Gates
        gate = self._gate(uid, user)
        if gate:
            return gate
        # 3. The router (only for typed text: the request bar and the slip already say what they want). It
        # also judges the soft warning signs: wording sports talk can share counts only if it agrees
        soft = soft_flags(text)
        pending = self._pending(history)
        if text:
            step("Reading your message…")
            self._route(route, text, history, pending, spend)
            if route.flags:
                return self._support(uid, thread_id, text, route.flags, "router", soft)
        unconfirmed = bool(soft) and not route.routed  # the router was down: no pause, but the helpline shows
        if unconfirmed:
            self.store.add_rg_event(uid, f"possible_{soft[0]}", {"flags": soft, "source": "rules, router unavailable",
                                                                 "thread_id": thread_id, "excerpt": text[:200]},
                                    action="helpline_note", now=self.clock())
        if slip:
            route.intent = "analyse"
        elif request:
            self._apply_request(route, request)
        else:
            route.merge(pending or {}, text)
        paused = self.picks_paused_until(uid)
        if paused and route.intent in PICK_INTENTS:
            return Turn(f"Picks are paused until {self._local(paused)} after an earlier message about losses or "
                        f"money worries. I can still explain how odds and tickets work.\n\n{HELPLINE}", "paused",
                        chips=SUPPORT_CHIPS)
        self._defaults(route, user)
        question = self._clarify(route)
        if question:
            return question
        # 4-6. The tool loop, cards and the checks
        late = datetime.datetime.fromtimestamp(now, self.tz).hour in LATE_HOURS
        if late:
            self._late_night(uid, thread_id)
        turn = self._answer(uid, user, text, request, slip, history, route, bool(paused), spend, step)
        notes = [LATE_NOTE] if late else []
        if paused or unconfirmed or self._flagged_recently(uid):
            notes.append(HELPLINE)
        if notes:
            turn.text = "\n\n".join([turn.text] + notes)
        return turn

    # --- step 1: warning signs ---

    def _support(self, uid: str, thread_id: int, text: str, flags: Sequence[str], source: str,
                 soft: Sequence[str] = ()) -> Turn:
        until = self._pause_picks(uid)
        detail = {"flags": list(flags), "source": source, "thread_id": thread_id, "excerpt": text[:200],
                  "paused_until": until}
        if soft:
            detail["soft_rules"] = list(soft)  # code's rules saw it too, without the "I" or "my" to be sure
        self.store.add_rg_event(uid, flags[0], detail, action="support_message,picks_off_24h", now=self.clock())
        message = SUPPORT_MESSAGE + (f"\n\n{DISTRESS_EXTRA}" if "distress" in flags else "")
        return Turn(message, "support", chips=list(SUPPORT_CHIPS))

    def _flagged_recently(self, uid: str) -> bool:
        since = self.clock() - PAUSE_SECONDS
        return any(event["kind"] in STRONG_FLAGS for event in self.store.rg_events(uid, since=since, limit=20))

    def _late_night(self, uid: str, thread_id: int) -> None:
        """Log a late-night session once a night, for the owner's review (it doesn't pause anything)."""
        if not self.store.rg_events(uid, kind="late_night", since=self.clock() - 6 * 3600, limit=1):
            self.store.add_rg_event(uid, "late_night", {"thread_id": thread_id}, action="reminder", now=self.clock())

    # --- step 2: gates ---

    def _gate(self, uid: str, user: Dict[str, Any]) -> Optional[Turn]:
        if not user.get("age_confirmed_at"):
            return Turn("Before we chat, please confirm you're 18 or older (21+ in the US). You can do it in "
                        "the app's Settings.", "gate", chips=[])
        if self.store.cooloff_active(uid, now=self.clock()):
            return Turn(f"You're on a break until {self._local(float(user['cooloff_until']))}, so the AI chat "
                        f"is off until then.\n\n{HELPLINE}", "gate")
        if self.credits(uid)["left_cents"] <= 0:
            return Turn("You've used today's AI allowance. It resets at midnight (US Eastern); the app's games, "
                        "odds and ticket pricing still work without it.", "gate")
        return None

    # --- step 3: the router, defaults and clarifying questions ---

    def _pending(self, history: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        last = history[-1] if history else None
        if last and last["role"] == "assistant" and isinstance(last["content"], dict):
            return last["content"].get("pending")
        return None

    def _route(self, route: Route, text: str, history: Sequence[Dict[str, Any]],
               pending: Optional[Dict[str, Any]], spend: Spend) -> None:
        lines = [f"{row['role']}: {self._plain(row)[:300]}" for row in list(history)[-4:]]
        parts = []
        if lines:
            parts.append("Recent conversation (oldest first):\n" + "\n".join(lines))
        if pending:
            parts.append(f"The assistant is waiting for an answer to: {pending.get('question')}\n"
                         f"Known so far: {json.dumps({k: v for k, v in pending.items() if k not in ('question', 'asks')})}")
        parts.append(f"Latest message:\n{text}")
        try:
            response = self.caller.create(
                self.router_model,
                max_tokens=2000,
                system=[{"type": "text", "text": ROUTER_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": "\n\n".join(parts)}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": ROUTER_SCHEMA}},
            )
        except AIError as exc:
            spend.add(self.router_model, exc.cost)
            logging.warning("The router failed, carrying on without it: %s", exc)
            return
        dollars, tokens_in, tokens_out = self.caller.cost(self.router_model, response.usage)
        spend.add(self.router_model, dollars, tokens_in, tokens_out)
        if response.stop_reason in ("refusal", "max_tokens"):
            return
        try:
            parsed = parse_route(json.loads(response_text(response)))
        except ValueError:
            return
        for name, value in parsed.__dict__.items():
            setattr(route, name, value)

    def _apply_request(self, route: Route, request: Mapping[str, Any]) -> None:
        """The request bar's fields over whatever the message said (any of them may be missing)."""
        route.intent = "build"
        leagues = [k for k in request.get("leagues") or [] if k in LEAGUE_KEYS][:MAX_REQUEST_LEAGUES]
        route.leagues = leagues or route.leagues
        for name in ("stake", "target", "max_legs"):
            value = _field(name, request.get(name))
            if value is not None:
                setattr(route, name, value)
        if request.get("tier") in ("safe", "risky", "crazy"):
            route.tier = request["tier"]

    def _defaults(self, route: Route, user: Mapping[str, Any]) -> None:
        if not route.leagues:
            favourites = [k for k in user.get("favourite_sports") or [] if k in LEAGUE_KEYS]
            route.leagues = (favourites or list(DEFAULT_LEAGUES))[:MAX_REQUEST_LEAGUES]
        if route.intent == "build":
            route.max_legs = route.max_legs or DEFAULT_MAX_LEGS
            route.tier = route.tier or (user.get("risk_tier") if user.get("risk_tier") in ("safe", "risky", "crazy") else None)
        if route.intent == "plan":
            route.legs_per_day = route.legs_per_day or DEFAULT_LEGS_PER_DAY

    def _clarify(self, route: Route) -> Optional[Turn]:
        """One question with chips when a build or a plan lacks something only the person can say."""
        if route.intent == "build" and route.stake is None:
            asks, question, chips = "stake", "How much do you want to stake?", ["$5", "$10", "$25"]
        elif route.intent == "build" and route.target is None:
            stake = route.stake or Decimal(10)
            asks = "target"
            question = f"And how much would you like back in total if it wins ({_dollars(stake)} stake included)?"
            chips = [f"{_dollars(stake * m)} back" for m in (2, 5, 10)]
        elif route.intent == "plan" and route.stake is None:
            asks, question, chips = "stake", "How much do you want to start the rollover with?", ["$5", "$10", "$25"]
        elif route.intent == "plan" and route.days is None:
            asks, question, chips = "days", "For how many days?", ["3 days", "5 days", "10 days"]
        else:
            return None
        pending = {**route.fields(), "question": question, "asks": asks}
        return Turn(question, "clarify", chips=chips, pending=pending)

    # --- steps 4-6: the tool loop, cards and checks ---

    @staticmethod
    def _slip_total(slip: Optional[Mapping[str, Any]]) -> Optional[Decimal]:
        """What a slip from the app costs in all: its stake is per bet (each single, round-robin or system bet)."""
        if not isinstance(slip, Mapping):
            return None
        stake = _amount(slip.get("stake"))
        legs = slip.get("legs")
        if stake is None or not isinstance(legs, list):
            return stake
        k = slip.get("k") if isinstance(slip.get("k"), int) and not isinstance(slip.get("k"), bool) else None
        system = slip.get("system") if isinstance(slip.get("system"), str) else None
        return stake * bet_count(slip.get("type") or "multi", len(legs), k, system)

    def _limits(self, route: Route, request: Optional[Mapping[str, Any]], slip: Optional[Mapping[str, Any]],
                history: Sequence[Dict[str, Any]], text: str) -> Dict[str, Set[Decimal]]:
        """Amounts the person gave in this thread, by kind (the toolbox refuses bigger ones). A stake is what
        they'd put on a whole ticket, so a slip counts at its total (its stake times its bets)."""
        limits: Dict[str, Set[Decimal]] = {"stake": set(), "target": set(), "min_odds": set()}
        found: List[Mapping[str, Any]] = [route.fields(), dict(request or {})]
        if slip:
            found.append({"stake": self._slip_total(slip)})
        for row in history:
            content = row.get("content")
            if row["role"] == "user" and isinstance(content, dict):
                found += [content.get("route") or {}, content.get("request") or {},
                          {"stake": self._slip_total(content.get("slip"))}]
        for item in found:
            for kind in limits:
                value = _amount(item.get(kind))
                if value is not None and value > 0:
                    limits[kind].add(value)
        if not route.routed and not request and not slip:
            # No router: any amount in the message counts as one they gave
            for match in re.finditer(r"\$?\b\d+(?:\.\d{1,2})?\b", text):
                value = _amount(match.group(0))
                if value is not None and value > 0:
                    limits["stake"].add(value)
                    limits["target"].add(value)
        return limits

    def _context(self, user: Mapping[str, Any], uid: str, route: Route, request: Optional[Mapping[str, Any]],
                 slip: Optional[Mapping[str, Any]], paused: bool) -> str:
        now = datetime.datetime.fromtimestamp(self.clock(), self.tz)
        credits = self.credits(uid)
        lines = [
            f"Now: {now:%A %B} {now.day}, {now:%Y, %-I:%M %p %Z}.",
            f"User settings: odds format {user.get('odds_format') or 'decimal'}; favourite sports "
            f"{', '.join(user.get('favourite_sports') or []) or 'none set'}; risk tier {user.get('risk_tier') or 'none'}; "
            f"AI credits left today {credits['left_cents']} of {credits['daily_cents']} cents.",
            f"What code understood (use these values): {json.dumps(route.fields())}",
        ]
        if request:
            lines.append("They used the request bar, so build tickets for exactly this request.")
        if slip:
            lines.append(f"Their slip, to analyse with ticket_analyse: {json.dumps(dict(slip), default=str)}")
        if paused:
            lines.append("Picks are paused for this person after a warning sign: suggest no bets and price nothing.")
        return "<context>\n" + "\n".join(lines) + "\n</context>"

    def _history(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The last messages as plain user/assistant text, tool results as one-line summaries."""
        out: List[Dict[str, Any]] = []
        for row in rows:
            text = self._plain(row)
            content = row.get("content")
            if row["role"] == "assistant" and isinstance(content, dict) and content.get("tools"):
                text += ("\n[Tools used that turn, for refs only (call them again before quoting numbers): "
                         + " | ".join(content["tools"]) + "]")
            if not text:
                continue
            role = "user" if row["role"] == "user" else "assistant"
            if out and out[-1]["role"] == role:
                out[-1]["content"] += "\n\n" + text
            else:
                out.append({"role": role, "content": text})
        while out and out[0]["role"] != "user":
            out.pop(0)
        return out

    @staticmethod
    def _plain(row: Mapping[str, Any]) -> str:
        content = row.get("content")
        if isinstance(content, dict):
            return str(content.get("text") or "")
        return str(content or "")

    def _answer(self, uid: str, user: Dict[str, Any], text: str, request: Optional[Mapping[str, Any]],
                slip: Optional[Mapping[str, Any]], history: List[Dict[str, Any]], route: Route, paused: bool,
                spend: Spend, step: Callable[[str], None]) -> Turn:
        # Odds may be loaded freely for the leagues asked about (or the request's, the slip's) and the
        # person's favourites; the toolbox allows one more per turn and says "quota" after that
        leagues = set(route.leagues) | {k for k in user.get("favourite_sports") or [] if k in LEAGUE_KEYS}
        leagues |= {k for k in (request or {}).get("leagues") or [] if k in LEAGUE_KEYS}
        leagues |= {leg.get("league") for leg in (slip or {}).get("legs") or []
                    if isinstance(leg, Mapping) and leg.get("league") in LEAGUE_KEYS}
        toolbox = Toolbox(self.lab, uid, self.tz, self._limits(route, request, slip, history, text), paused=paused,
                          leagues=leagues)
        context = self._context(user, uid, route, request, slip, paused)
        said = text or self._describe(request, slip)
        messages = self._history(history)
        current = [{"type": "text", "text": context}, {"type": "text", "text": said}]
        if messages and messages[-1]["role"] == "user":  # an earlier turn that never got its answer stored
            current.insert(0, {"type": "text", "text": messages.pop()["content"]})
        messages.append({"role": "user", "content": current})
        effort = "medium" if route.intent in DEEP_INTENTS else "low"

        runs: List[ToolRun] = []
        reply, content = self._loop(messages, effort, toolbox, runs, spend, step, MAX_ROUNDS)
        sources = source_numbers([run.content for run in runs], said, context, dict(request or {}), dict(slip or {}))
        found = problems(reply, sources) if reply else ["no answer"]
        if found and reply and content is not None:
            step("Double-checking the numbers…")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": (
                "Your reply can't be shown because it has " + "; ".join(found) + ". Write it again: only numbers "
                "from this turn's tool results (call a tool if you need one you don't have), and none of the "
                "banned wording.")})
            first_problems = found
            reply, content = self._loop(messages, effort, toolbox, runs, spend, step, RETRY_ROUNDS)
            sources = source_numbers([run.content for run in runs], said, context, dict(request or {}), dict(slip or {}))
            found = problems(reply, sources) if reply else ["no answer"]
            if found:
                found = first_problems + found
        cards = self._cards(runs)
        chips = self._chips(route, cards)
        summaries = [run.summary for run in runs if run.ok][:12]
        if not found:
            return Turn(reply, "answer", cards, chips, summaries, checked=True)
        logging.info("Assistant reply replaced by a template: %s", "; ".join(found))
        return Turn(self._template(cards, runs), "template", cards, chips, summaries, checked=False, problems=found)

    def _loop(self, messages: List[Dict[str, Any]], effort: str, toolbox: Toolbox, runs: List[ToolRun], spend: Spend,
              step: Callable[[str], None], rounds: int) -> Tuple[str, Optional[List[Any]]]:
        """Claude and the tools, for at most ``rounds`` requests: (final text, its content blocks)."""
        for round_ in range(rounds):
            request: Dict[str, Any] = {
                "max_tokens": 16000,
                "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                "tools": TOOLS,
                "messages": messages,
                "output_config": {"effort": effort},
                "cache_control": {"type": "ephemeral"},  # the growing tail, so each round reads the last one
            }
            if round_ == rounds - 1:
                request["tool_choice"] = {"type": "none"}  # the last round must answer
            step("Thinking…" if round_ == 0 else "Putting it together…")
            response = self.caller.create(self.chat_model, **request)
            model = getattr(response, "model", None) or self.chat_model
            dollars, tokens_in, tokens_out = self.caller.cost(model, response.usage)
            spend.add(model, dollars, tokens_in, tokens_out)
            if response.stop_reason in ("refusal", "max_tokens"):
                # Declined, or cut off mid-sentence: the template answers instead. How often betting talk
                # trips the safety classifiers is worth knowing, so it is logged
                logging.info("Claude stopped early (%s): %s", response.stop_reason,
                             getattr(response, "stop_details", None))
                return "", None
            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue
            calls = [block for block in response.content if block.type == "tool_use"]
            if response.stop_reason == "tool_use" and calls:
                messages.append({"role": "assistant", "content": response.content})
                results = []
                for block in calls:
                    step(step_text(block.name, block.input))
                    run = toolbox.run(block.name, block.input)
                    runs.append(run)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": not run.ok,
                                    "content": json.dumps(run.content, default=str, separators=(",", ":"))})
                messages.append({"role": "user", "content": results})
                continue
            return response_text(response), response.content
        return "", None

    # --- cards, chips and the template ---

    def _cards(self, runs: Sequence[ToolRun]) -> List[Dict[str, Any]]:
        """Cards from this turn's results, latest first per ticket (a proposal replaces the same ticket priced)."""
        cards: List[Dict[str, Any]] = []
        seen: Dict[Any, int] = {}
        for run in runs:
            for card in run.cards:
                if card["type"] == "ticket":
                    key = ("ticket", card.get("ticket_type"), card.get("stake"),
                           tuple(json.dumps(ref, sort_keys=True) for ref in card.get("refs") or []))
                elif card["type"] == "player":
                    key = ("player", card.get("league"), card.get("id"))
                else:
                    key = ("plan", json.dumps(card.get("body"), sort_keys=True))
                if key in seen:
                    cards[seen[key]] = card
                else:
                    seen[key] = len(cards)
                    cards.append(card)
        return cards[:MAX_CARDS]

    def _chips(self, route: Route, cards: Sequence[Dict[str, Any]]) -> List[str]:
        """Follow-up suggestions in code: specific to what was shown, never "bet more"."""
        kinds = {card["type"] for card in cards}
        if "ticket" in kinds:
            tickets = [card for card in cards if card["type"] == "ticket"]
            chips = list(TICKET_CHIPS)
            if not any(len(card.get("refs") or []) > 1 for card in tickets):
                chips.remove("Swap the weakest leg")
            if len(tickets) > 1:
                chips.insert(0, "Compare these tickets")
            return chips[:MAX_CHIPS]
        if "plan" in kinds:
            return list(PLAN_CHIPS)
        if "player" in kinds:
            name = next(card.get("title") for card in cards if card["type"] == "player")
            return [f"{name}'s prop lines", f"{name}'s injury news"]
        if route.intent == "lookup":
            return list(LOOKUP_CHIPS)
        return list(HELP_CHIPS)

    @staticmethod
    def _template(cards: Sequence[Dict[str, Any]], runs: Sequence[ToolRun]) -> str:
        """The reply when Claude's text failed the checks twice: plain facts written by code."""
        for card in cards:
            if card["type"] == "ticket":
                facts = template_explanation({"ticket": card["ticket"]})
                return f"{card.get('title')}: {facts['verdict']} The card below has the details."
            if card["type"] == "plan":
                summary = (card.get("compare") or {}).get("summary")
                if summary:
                    return f"{summary} The card below compares the strategies side by side."
            if card["type"] == "player":
                return f"Here is {card.get('title')}'s card."
        if any(run.ok for run in runs):
            return "I couldn't put that into words reliably, so I've left it out. Try asking in a different way."
        return "I couldn't answer that reliably just now. Try asking in a different way."

    @staticmethod
    def _describe(request: Optional[Mapping[str, Any]], slip: Optional[Mapping[str, Any]]) -> str:
        """A request-bar request or a slip as readable text, so the thread shows what the AI was asked."""
        if slip:
            legs = len(slip.get("legs") or [])
            stake = _amount(slip.get("stake"))
            kind = str(slip.get("type") or "multi").replace("_", " ")
            return f"Analyse my slip: {legs}-leg {kind}" + (f", {_dollars(stake)} stake" if stake else "")
        if request:
            parts = [" + ".join(league_name(k) for k in request.get("leagues") or []) or "Any sport"]
            stake, target = _amount(request.get("stake")), _amount(request.get("target"))
            if stake and target:
                parts.append(f"{_dollars(stake)} to get about {_dollars(target)} back")
            elif stake:
                parts.append(f"{_dollars(stake)} stake")
            if request.get("max_legs"):
                parts.append(f"max {request['max_legs']} legs")
            if request.get("tier") in ("safe", "risky", "crazy"):
                parts.append(str(request["tier"]).title())
            return ", ".join(parts)
        return ""


def build_assistant(config: Any, lab: Any, store: Any, client: Optional[Any] = None) -> Optional[Assistant]:
    """The chat assistant, or None without ANTHROPIC_API_KEY (or without a store to keep threads in)."""
    if not getattr(config, "anthropic_api_key", None) or store is None or lab is None:
        return None
    if client is None:
        import anthropic  # the SDK is only needed with a key

        # A chat turn should answer in seconds; one retry on a 429 or 5xx keeps it from hanging long
        client = anthropic.Anthropic(api_key=config.anthropic_api_key, timeout=120.0, max_retries=1)
    return Assistant(
        client, lab, store,
        chat_model=getattr(config, "ai_chat_model", CHAT_MODEL),
        router_model=getattr(config, "ai_router_model", ROUTER_MODEL),
        daily_cents=getattr(config, "ai_chat_daily_cents", DAILY_CENTS),
        tz=ZoneInfo(getattr(config, "timezone", "America/New_York")),
    )
