"""Claude explains a ticket in words, and code checks every word it writes about money.

The brain (Python) computes each number on a ticket: chances, payouts, what
the bookmaker keeps, the expected result. Claude only explains them, so its
answer is checked before anyone sees it:

- numbers: every number in the text must be one the code gave it (rounding
  to the precision shown is fine, and so is a chance shown as a percentage);
  a number Claude worked out or made up is rejected;
- banned phrases: "lock", "guaranteed", "sure thing", "risk-free", "can't
  lose", "win back", calling a ticket safe, or nudging toward a bigger stake.

A rejected answer is retried once with the reasons; after that the reader
gets a template written by code from the same facts. The same checks guard
the chat assistant's replies (ai/assistant.py), and ``ClaudeCaller`` is the
one place both send requests to Claude: refusal fallbacks where the model
supports them, plain-English errors and a cost estimate, as in ai/claude.py.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import anthropic
from pydantic import BaseModel, ValidationError

from .claude import FALLBACK_BETA, FALLBACK_MODELS, PRICES

EXPLAIN_MODEL = "claude-opus-5-5"
EXPLAIN_EFFORT = "medium"
MAX_TEXT = 700  # characters per explanation field: a verdict is a sentence or two, not an essay


class AIError(Exception):
    """Claude couldn't answer; the message says why in plain English (never contains a key).

    ``cost`` is what the failed attempt still cost, in US dollars.
    """

    def __init__(self, message: str, cost: float = 0.0) -> None:
        super().__init__(message)
        self.cost = cost


# --- talking to Claude ------------------------------------------------------------------------

class ClaudeCaller:
    """Sends one request to Claude through the official SDK and prices the answer."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def create(self, model: str, **request: Any) -> Any:
        # A refusal on a supported model is retried server-side on a substitute ("default" picks it by reason)
        extra: Dict[str, Any] = {"betas": [FALLBACK_BETA], "fallbacks": "default"} if model in FALLBACK_MODELS else {}
        try:
            return self.client.beta.messages.create(model=model, **extra, **request)
        except anthropic.AuthenticationError:
            raise AIError("Claude rejected the API key (check ANTHROPIC_API_KEY)") from None
        except anthropic.PermissionDeniedError:
            raise AIError(f"This API key can't use {model}") from None
        except anthropic.NotFoundError:
            raise AIError(f"Unknown Claude model {model} (check AI_CHAT_MODEL and AI_ROUTER_MODEL)") from None
        except anthropic.RateLimitError:
            raise AIError("Claude is busy right now: try again in a minute") from None
        except anthropic.BadRequestError as exc:
            raise AIError(f"Claude didn't accept the request: {str(exc.message)[:200]}") from None
        except anthropic.APIStatusError as exc:
            raise AIError(f"Claude's API answered {exc.status_code}: try again later") from None
        except anthropic.APIConnectionError:
            raise AIError("Couldn't reach Claude's API") from None

    @staticmethod
    def cost(model: str, usage: Any) -> Tuple[float, int, int]:
        """(estimated dollars, input tokens, output tokens) of one response."""
        price_in, price_out, price_read = PRICES.get(model, PRICES[EXPLAIN_MODEL])
        fresh = getattr(usage, "input_tokens", 0) or 0
        written = getattr(usage, "cache_creation_input_tokens", 0) or 0
        read = getattr(usage, "cache_read_input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        dollars = (fresh * price_in + written * price_in * 1.25 + read * price_read + out * price_out) / 1e6
        return dollars, fresh + written + read, out


def response_text(response: Any) -> str:
    return "".join(getattr(block, "text", "") for block in response.content if block.type == "text").strip()


# --- the numbers check -------------------------------------------------------------------------

# A number as written: optional currency and sign, digits (with thousands commas), decimals, percent
_NUMBER = re.compile(
    r"(?P<cur>[$£€])?(?P<sign>[-+−])?(?P<cur2>[$£€])?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<pct>\s?%|\s?per ?cent\b)?"
)
_ONE_IN = re.compile(r"\b1(?:\s+|-)in(?:\s+|-)(?P<num>\d{1,3}(?:,\d{3})+|\d+)\b")
# Text that holds digits but no betting numbers: clock times, dates, helplines, age limits
_NOT_NUMBERS = [
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?m\.?\b)?", re.I),
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T[\d:.]+(?:Z|[+-]\d{2}:?\d{2})?)?"),
    re.compile(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
               r"|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?"
               r"(?:,?\s+\d{4})?\b", re.I),
    re.compile(r"1-800-GAMBLER|1-800-522-4700|0808 ?8020 ?133|\b988\b|\b116 ?123\b", re.I),
    re.compile(r"\b(?:18|21)\s?\+"),
]
# A number glued to letters is part of a name or a unit we don't check ("76ers", "3rd", "7pm")
_GLUED_AFTER = re.compile(r"(?:st|nd|rd|th|am|pm|ers|s)\b", re.I)
ROUNDING_SHARE = 0.05  # "about 1,900" may round a number by at most 5% of it
SMALL_COUNT = 12  # bare whole numbers up to this are counts and positions: "3 legs", "leg 2", "day 1"


@dataclass(frozen=True)
class Figure:
    """One number as written in a text."""
    text: str
    value: float  # its size (signs are ignored: "lose $2.60" and "-2.60" are the same figure)
    decimals: int
    percent: bool
    money: bool
    one_in: bool = False


def _figures(text: str, skip_trivial: bool) -> List[Figure]:
    figures: List[Figure] = []
    for match in _ONE_IN.finditer(text):
        number = match.group("num")
        figures.append(Figure(match.group(0), float(number.replace(",", "")), 0, False, False, one_in=True))
    # "1 in 12": the 12 was checked above and the 1 is just words
    text = _ONE_IN.sub(" ", text)
    for match in _NUMBER.finditer(text):
        start, end = match.span()
        before = text[start - 1] if start else " "
        if before.isalnum() or before in "_.":  # inside a word or a longer number ("fg3m", "v2.1")
            continue
        number = match.group("num")
        percent, money = bool(match.group("pct")), bool(match.group("cur") or match.group("cur2"))
        if not percent and not money and _GLUED_AFTER.match(text, end):
            continue
        if not percent and not money and end < len(text) and text[end].isalpha() and text[end] not in "xX":
            continue  # glued to a word: a team name or a unit ("5pts" is rare; "76ers" isn't)
        digits = number.replace(",", "")
        value = float(digits)
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        if skip_trivial and not percent and not money and decimals == 0 and not match.group("sign"):
            if value <= SMALL_COUNT or (len(digits) == 4 and 1900 <= value <= 2100):
                continue  # a count, a position or a year
        figures.append(Figure(match.group(0).strip(), value, decimals, percent, money))
    return figures


def numbers_in(text: str) -> List[Figure]:
    """The figures in a reply that must be traced back to the code: money, odds, chances, "1 in X"."""
    for pattern in _NOT_NUMBERS:
        text = pattern.sub(" ", text)
    return _figures(text, skip_trivial=True)


def source_numbers(*objects: Any) -> Set[float]:
    """Every number in tool results or a fact sheet (values and numbers inside strings), unsigned."""
    found: Set[float] = set()

    def walk(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float, Decimal)):
            number = float(value)
            if math.isfinite(number):
                found.add(abs(number))
        elif isinstance(value, str):
            found.update(figure.value for figure in _figures(value, skip_trivial=False))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item)

    for obj in objects:
        walk(obj)
    return found


def _traced(figure: Figure, sources: Iterable[float]) -> bool:
    """Whether a figure is a source number rounded to the precision shown (or a chance as a percentage)."""
    slack = 0.5 * 10 ** -figure.decimals + 1e-9
    whole = figure.decimals == 0 and not figure.percent and figure.value >= 100
    zeros = len(str(int(figure.value))) - len(str(int(figure.value)).rstrip("0")) if whole else 0
    for source in sources:
        if abs(source - figure.value) <= slack:
            return True
        if figure.percent and source <= 1 and abs(source * 100 - figure.value) <= slack:
            return True
        # "about 1,900" for 1,901: rounded to the tens, hundreds or thousands shown (but never far off)
        gap = abs(source - figure.value)
        if zeros and gap <= 0.5 * 10 ** min(zeros, 3) and gap <= ROUNDING_SHARE * source:
            return True
    return False


def invented_numbers(text: str, sources: Set[float]) -> List[str]:
    """The figures in ``text`` that no source number accounts for (as written)."""
    return [figure.text for figure in numbers_in(text) if not _traced(figure, sources)]


# --- banned phrases ----------------------------------------------------------------------------

# Phrases that oversell a bet or push someone to bet more (Massachusetts 205 CMR 256, CAP 16, NCPG).
# Lower-case "safe" is banned as a description; "Safe" with a capital is the tier's name.
_EARN = r"(?:win|wins|winning|won|make|makes|making|made|earn|earns|earning|earned|recoup|recoups|recouping|recouped)"
_LOST_MONEY = (r"(?:(?:all|some)\s+(?:of\s+)?)?(?:(?:your|my|our|their|the|those)\s+(?:\w+\s+)?)?"
               r"(?:money|loss(?:es)?|stake|cash|bankroll)|what\s+(?:you|I|we|they)(?:'ve|’ve|\s+have)?\s+lost")
BANNED: Sequence[Tuple[str, "re.Pattern[str]"]] = [
    ("lock", re.compile(r"\blocks?\b", re.I)),
    ("guaranteed", re.compile(r"\bguarantee(?:d|s|ing)?\b", re.I)),
    ("sure thing", re.compile(r"\bsure[- ](?:thing|bet|win|winner|shot|fire)\b|\bsurefire\b|\bdead cert", re.I)),
    ("risk-free", re.compile(r"\brisk[- ]?free\b|\b(?:no|zero)[- ]risk\b|\bwithout (?:any )?risk\b", re.I)),
    ("can't lose", re.compile(r"\b(?:can(?:'|’)?t|cannot|can not|won(?:'|’)?t) (?:lose|miss|go wrong)\b"
                              r"(?!\s+more\b)", re.I)),
    # Money won back, not sports: "win back-to-back titles", "won back home" and "make it back to the
    # playoffs" are fine
    ("win back", re.compile(
        rf"\b{_EARN}\s+(?:it(?:\s+all)?|that|them|{_LOST_MONEY})\s+back\b"
        r"(?![-‐\s]+(?:to[-‐\s]+back|home|to\s+(?:the|a|an|their|his|her|its)|into)\b)"
        rf"|\b{_EARN}\s+back\s+(?:{_LOST_MONEY}|what\b)", re.I)),
    ("chase losses", re.compile(r"\b(?:chas(?:e|es|ed|ing)|recoup(?:s|ed|ing)?|recover(?:s|ed|ing)?)\s+"
                                r"(?:(?:all|some)\s+(?:of\s+)?)?(?:(?:your|my|our|their|the|those|these)\s+(?:\w+\s+)?)?"
                                r"loss(?:es)?\b(?!\s+of\b)", re.I)),
    ("hype", re.compile(r"\bfree money\b|\beasy money\b|\bno[- ]brainer\b|\bnailed[- ]on\b|\bbanker\b"
                        r"|\bcan(?:'|’)?t[- ]miss\b|\bslam[- ]dunk\b|\bbet the house\b", re.I)),
    ("bigger stake", re.compile(r"\b(?:bet|stake|wager|risk|put down|put in)\s+(?:a lot\s+|much\s+)?"
                                r"(?:more(?!\s+than\b)|bigger|larger|higher|extra)\b"
                                r"|\b(?:increase|raise|up|double|triple|boost)\s+(?:your|the|that|this)\s+"
                                r"(?:stake|bet|wager|unit)s?\b"
                                r"|\b(?:bigger|larger|higher|max(?:imum)?)\s+(?:stake|bet|wager)s?\b", re.I)),
    ("safe", re.compile(r"\b(?:[Ii]s|[Ii]t(?:'|’)?s|[Ll]ooks|[Ff]eels|[Vv]ery|[Pp]retty|[Qq]uite|[Tt]otally"
                        r"|[Cc]ompletely|[Pp]erfectly|[Rr]eally|[Ss]uper|[Aa]bsolutely|[Ff]airly)\s+safe\b"
                        r"|\bsafe (?:bet|pick|play|ticket|parlay|multi|option|choice|money)s?\b")),
]


# These may appear in an honest warning ("nothing is guaranteed", "never raise your stake"); the rest
# oversell whatever surrounds them ("not a lock" still puts the idea in the reader's head)
NEGATABLE = {"guaranteed", "risk-free", "bigger stake"}
_NEGATION = re.compile(r"\b(?:never|not|no|nothing|nobody|don(?:'|’)?t|doesn(?:'|’)?t|isn(?:'|’)?t|aren(?:'|’)?t"
                       r"|won(?:'|’)?t|avoid|without)\b[^.!?]{0,30}$", re.I)


def banned_phrases(text: str) -> List[str]:
    """Which banned phrases the text uses (by name), in the order of the list above."""
    found = []
    for name, pattern in BANNED:
        for match in pattern.finditer(text):
            if name in NEGATABLE and _NEGATION.search(text[max(0, match.start() - 40):match.start()]):
                continue
            found.append(name)
            break
    return found


def problems(text: str, sources: Set[float]) -> List[str]:
    """Why a text can't be shown (empty when it can)."""
    found = []
    invented = invented_numbers(text, sources)
    if invented:
        found.append("numbers that are not in the facts you were given: " + ", ".join(dict.fromkeys(invented)))
    banned = banned_phrases(text)
    if banned:
        found.append("banned wording: " + ", ".join(banned))
    return found


# --- formatting helpers shared with the templates ---------------------------------------------

def pct(p: Any, decimals: int = 0) -> str:
    """0.0523 -> "5%" ("under 1%" for a chance that would round to nothing)."""
    try:
        value = float(p) * 100
    except (TypeError, ValueError):
        return "–"
    if 0 < value < 0.5 and decimals == 0:
        return "under 1%"
    return f"{value:.{decimals}f}%"


def money(amount: Any, currency: str = "$") -> str:
    """"12.4" -> "$12.40", "-2.6" -> "-$2.60"."""
    try:
        value = Decimal(str(amount))
    except Exception:
        return "–"
    sign = "-" if value < 0 else ""
    return f"{sign}{currency}{abs(value):,.2f}"


def one_in_text(n: Any) -> str:
    return f"1 in {int(n):,}" if isinstance(n, (int, float)) and n else "–"


# --- explaining a ticket -----------------------------------------------------------------------

EXPLAIN_PROMPT = """\
You explain a betting ticket to the person who built it. Our code has already worked out every number \
and gives you a JSON fact sheet: the priced ticket (each leg's odds, the AI's chance and the market's \
chance, the ticket's chance as "1 in X", what the bookmaker keeps, the expected result in money and the \
most likely outcome), notes on each leg (injuries, where the AI and the market disagree, analysts' \
views, rest days) and alternatives the code worked out.

Write three things:
- verdict: one or two plain sentences on what this ticket is: its chance as "1 in X" (with the \
percentage if you like), what the bookmaker keeps, and the expected result in money. Use the fixed \
chance words from the sheet (favourite, coin flip, underdog, long shot, lottery ticket).
- per_leg: one entry per leg, by its index from the sheet: one or two sentences on that leg using \
its notes.
- case_against: the strongest honest reason not to place the ticket, in one or two sentences (the \
weakest leg, the bookmaker's compounded cut, a note that worries you).

Rules:
- Every number you write must come from the fact sheet. You may round it or show a chance as a \
percentage; never add, subtract, multiply or divide numbers, and never estimate one yourself.
- Never use these words or ideas: lock, guaranteed, sure thing, risk-free, can't lose, win back, \
chasing losses. Never call a ticket or a leg safe ("Safe" is only the name of a risk tier). Never \
suggest a bigger stake or adding legs.
- You explain; you don't tip. Plain English for a casual fan, no markdown, no emojis.
"""

EXPLAIN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "per_leg": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"index": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["index", "text"],
                "additionalProperties": False,
            },
        },
        "case_against": {"type": "string"},
    },
    "required": ["verdict", "per_leg", "case_against"],
    "additionalProperties": False,
}


class _LegText(BaseModel):
    index: int
    text: str


class _Explanation(BaseModel):
    verdict: str
    per_leg: List[_LegText]
    case_against: str


@dataclass
class Explanation:
    """What explain_ticket returns: the texts and what producing them cost."""
    verdict: str
    per_leg: List[Dict[str, Any]]
    case_against: str
    source: str  # "claude", or "template" when Claude's answer failed the checks (or never came)
    model: Optional[str] = None
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    notes: List[str] = field(default_factory=list)  # why the template was used, for the log

    def as_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "per_leg": self.per_leg, "case_against": self.case_against,
                "source": self.source, "model": self.model, "cost": round(self.cost, 5)}


def _ticket(fact_sheet: Dict[str, Any]) -> Dict[str, Any]:
    ticket = fact_sheet.get("ticket")
    return ticket if isinstance(ticket, dict) else fact_sheet


def _legs(fact_sheet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The ticket's legs: inside the priced ticket, or beside it with their notes (the Lab's fact sheet)."""
    legs = _ticket(fact_sheet).get("legs") or fact_sheet.get("legs") or []
    return [leg for leg in legs if isinstance(leg, dict)]


def _leg_notes(fact_sheet: Dict[str, Any]) -> Dict[int, List[str]]:
    notes: Dict[int, List[str]] = {}
    for entry in fact_sheet.get("legs") or []:
        if isinstance(entry, dict) and isinstance(entry.get("index"), int):
            notes[entry["index"]] = [str(note) for note in entry.get("notes") or [] if note]
    return notes


def leg_indexes(fact_sheet: Dict[str, Any]) -> List[int]:
    indexes = [leg.get("index", i) for i, leg in enumerate(_legs(fact_sheet))]
    return indexes or sorted(_leg_notes(fact_sheet))


def template_explanation(fact_sheet: Dict[str, Any]) -> Dict[str, Any]:
    """The same three texts written by code from the fact sheet (used when Claude's fail the checks)."""
    ticket = _ticket(fact_sheet)
    chance = ticket.get("chance") or {}
    expected = ticket.get("expected") or {}
    stake = ticket.get("total_stake") or ticket.get("unit_stake")
    parts = []
    if chance.get("ai_one_in") or chance.get("market_one_in"):
        views = []
        if chance.get("ai_one_in"):
            views.append(f"about {one_in_text(chance['ai_one_in'])} ({pct(chance.get('ai'))}) by the AI")
        if chance.get("market_one_in"):
            views.append(f"{one_in_text(chance['market_one_in'])} ({pct(chance.get('market'))}) by the market")
        label = f"{str(chance['label']).capitalize()}: " if chance.get("label") else ""
        what = "a profit" if chance.get("headline") == "profit" else "every leg winning"
        parts.append(f"{label}the chance of {what} is " + " and ".join(views) + ".")
    cut = ticket.get("book_cut")
    if cut is not None and cut >= 0:
        parts.append(f"The bookmaker keeps about {pct(cut, 1)} of money staked on tickets like this.")
    elif cut is not None:
        parts.append("These best prices beat the bookmakers' consensus, so there is no cut on the market's view: "
                     "check they are still on offer.")
    basis = "ai" if expected.get("ai") is not None else "market"
    if expected.get(basis) is not None and stake:
        value = Decimal(str(expected[basis]))
        word = "lose about {} of" if value < 0 else "make about {} on"
        parts.append(f"On average you {word.format(money(abs(value)))} your {money(stake)} "
                     f"({'the AI' if basis == 'ai' else 'the market'}'s view).")
    verdict = " ".join(parts) or "This ticket couldn't be priced, so there is nothing to explain yet."

    notes = _leg_notes(fact_sheet)
    per_leg = []
    legs = _legs(fact_sheet)
    for i, leg in enumerate(legs):
        index = leg.get("index", i)
        odds = (leg.get("odds") or {}).get("decimal")
        bits = [f"{leg.get('label', 'This leg')}" + (f" at {odds:.2f}" if isinstance(odds, (int, float)) else "")]
        if leg.get("ai_prob") is not None:
            bits.append(f"AI {pct(leg['ai_prob'])}")
        if leg.get("market_prob") is not None:
            bits.append(f"market {pct(leg['market_prob'])}")
        text = ": ".join([bits[0], ", ".join(bits[1:])]) if len(bits) > 1 else bits[0]
        if leg.get("chance_label"):
            text += f" ({leg['chance_label']})"
        text += "."
        if notes.get(index):
            text += " " + notes[index][0].rstrip(".") + "."
        per_leg.append({"index": index, "text": text})

    against = []
    weakest = ticket.get("weakest_leg")
    if isinstance(weakest, int) and 0 <= weakest < len(legs):
        leg = legs[weakest]
        p = leg.get("ai_prob") if leg.get("ai_prob") is not None else leg.get("market_prob")
        against.append(f"The weakest leg is {leg.get('label', 'one leg')}" + (f" at {pct(p)}" if p is not None else "") + ".")
    if len(legs) > 1:
        against.append("Each extra leg adds the bookmaker's cut again.")
    most = ticket.get("most_likely") or {}
    if most.get("result") and most.get("probability") is not None:
        against.append(f"The most likely result is {most['result']} ({pct(most['probability'])}).")
    case_against = " ".join(against) or "Every bet carries the bookmaker's cut, so the expected result is a loss."
    return {"verdict": verdict, "per_leg": per_leg, "case_against": case_against}


def _texts(result: Dict[str, Any]) -> List[str]:
    return [result["verdict"], result["case_against"]] + [leg["text"] for leg in result["per_leg"]]


def _parse(text: str, fact_sheet: Dict[str, Any]) -> Dict[str, Any]:
    """Claude's JSON, trimmed, with one entry per leg of the sheet (gaps filled from the template)."""
    parsed = _Explanation.model_validate(json.loads(text))
    indexes = leg_indexes(fact_sheet)
    by_index = {leg.index: leg.text.strip()[:MAX_TEXT] for leg in parsed.per_leg if leg.text.strip()}
    template = {leg["index"]: leg["text"] for leg in template_explanation(fact_sheet)["per_leg"]}
    per_leg = [{"index": i, "text": by_index.get(i) or template.get(i, "")} for i in indexes]
    verdict, against = parsed.verdict.strip()[:MAX_TEXT], parsed.case_against.strip()[:MAX_TEXT]
    if not verdict or not against:
        raise ValueError("empty verdict or case against")
    return {"verdict": verdict, "per_leg": per_leg, "case_against": against}


def explain_ticket(fact_sheet: Dict[str, Any], caller: ClaudeCaller, model: str = EXPLAIN_MODEL,
                   effort: str = EXPLAIN_EFFORT) -> Explanation:
    """Claude's verdict, per-leg notes and case against for a ticket, checked against the fact sheet.

    A text with a number that isn't on the sheet, or with banned wording, is sent
    back once with the reasons; if the second try fails too (or Claude refuses,
    or its answer is cut off) the template is used. Raises AIError only when the
    API itself fails (its ``cost`` holds what was already spent).
    """
    sources = source_numbers(fact_sheet)
    messages: List[Dict[str, Any]] = [{"role": "user", "content": json.dumps(fact_sheet, default=str)}]
    out = Explanation("", [], "", "template")
    for attempt in range(2):
        try:
            response = caller.create(
                model,
                max_tokens=8000,
                system=[{"type": "text", "text": EXPLAIN_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": EXPLAIN_SCHEMA}},
            )
        except AIError as exc:
            raise AIError(str(exc), out.cost + exc.cost) from None
        cost, tokens_in, tokens_out = caller.cost(getattr(response, "model", model) or model, response.usage)
        out.cost += cost
        out.input_tokens += tokens_in
        out.output_tokens += tokens_out
        out.model = getattr(response, "model", None) or model
        if response.stop_reason in ("refusal", "max_tokens"):
            out.notes.append(f"Claude's answer stopped early ({response.stop_reason})")
            break
        try:
            result = _parse(response_text(response), fact_sheet)
        except (ValueError, ValidationError):
            out.notes.append("Claude's answer wasn't the expected JSON")
            break
        found = sorted({problem for text in _texts(result) for problem in problems(text, sources)})
        if not found:
            out.verdict, out.per_leg, out.case_against, out.source = (
                result["verdict"], result["per_leg"], result["case_against"], "claude")
            return out
        out.notes.append("; ".join(found))
        messages = messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": "Your answer can't be shown because it has " + "; ".join(found)
             + ". Write it again using only numbers that appear in the fact sheet, and none of the banned wording."},
        ]
    template = template_explanation(fact_sheet)
    out.verdict, out.per_leg, out.case_against, out.source = (
        template["verdict"], template["per_leg"], template["case_against"], "template")
    return out
