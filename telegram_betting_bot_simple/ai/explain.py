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
gets a template written by code from the same facts. The chat assistant also
sends back a reply that refuses or lectures while a tool could answer
(refusal_problems), in the same one retry, and its templates include the
player scan's (scan_template). The same checks guard
the chat assistant's replies (ai/assistant.py), and ``ClaudeCaller`` is the
one place both send requests to Claude: refusal fallbacks where the model
supports them, plain-English errors and a cost estimate, as in ai/claude.py.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import anthropic
from pydantic import BaseModel, ValidationError

from ..engine.soccer import MODEL_BOOK
from ..engine.tickets import ai_given_no_push
from .claude import FALLBACK_BETA, FALLBACK_MODELS, PRICES, UNKNOWN_PRICE, usage_cost

EXPLAIN_MODEL = "claude-opus-5-5"
EXPLAIN_EFFORT = "medium"
MAX_TEXT = 700  # characters per explanation field: a verdict is a sentence or two, not an essay
MIN_OUTPUT_TOKENS = 1024  # a request that can't afford this many output tokens isn't sent (thinking uses some)
# Anthropic answers an account with no prepaid API credits with a 400 ("Your credit balance is too low to access
# the Anthropic API..."): said plainly, with where the owner tops up (a claude.ai plan doesn't fund the API)
NO_CREDITS_SIGN = "credit balance is too low"
NO_CREDITS = ("The AI is paused: the app's Anthropic account has no API credits left. The owner can add credits "
              "at console.anthropic.com under Plans & Billing")


class AIError(Exception):
    """Claude couldn't answer; the message says why in plain English (never contains a key).

    ``cost`` is what the failed attempt still cost, in US dollars. ``temporary`` is False when waiting won't
    help (a bad key, an unknown model, a request Claude refuses), so the person isn't told to try again.
    ``detail`` is the API's own message for a refused request (a 400), for the server log only.
    """

    def __init__(self, message: str, cost: float = 0.0, temporary: bool = True, detail: Optional[str] = None) -> None:
        super().__init__(message)
        self.cost = cost
        self.temporary = temporary
        self.detail = detail


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
            raise AIError("Claude rejected the API key (check ANTHROPIC_API_KEY)", temporary=False) from None
        except anthropic.PermissionDeniedError:
            raise AIError(f"This API key can't use {model}", temporary=False) from None
        except anthropic.NotFoundError:
            raise AIError(f"Unknown Claude model {model} (check AI_CHAT_MODEL and AI_ROUTER_MODEL)",
                          temporary=False) from None
        except anthropic.RateLimitError:
            raise AIError("Claude is busy right now: try again in a minute") from None
        except anthropic.BadRequestError as exc:
            # The app's own request was refused: the detail is for the server log, not the person
            detail = str(exc.message)[:500]
            logging.warning("Claude refused a %s request: %s", model, detail)
            if NO_CREDITS_SIGN in detail.lower():  # a 400 that is about billing, not the request
                raise AIError(NO_CREDITS, temporary=False, detail=detail) from None
            raise AIError("Claude didn't accept the app's request (the details are in the server log)",
                          temporary=False, detail=detail) from None
        except anthropic.APIStatusError as exc:
            raise AIError(f"Claude's API answered {exc.status_code}: try again later") from None
        except anthropic.APIConnectionError:
            raise AIError("Couldn't reach Claude's API") from None

    @staticmethod
    def cost(model: str, usage: Any, served: Optional[str] = None) -> Tuple[float, int, int]:
        """(estimated dollars, input tokens, output tokens) of one response to a request for ``model``.

        ``served`` is the model that answered (``response.model``): a refusal fallback can hand the request
        to a dearer model, and the declined attempt is billed too (see ai/claude.py's usage_cost).
        """
        return usage_cost(model, usage, served)

    @staticmethod
    def prices(model: str) -> Tuple[float, float]:
        """The dearest US dollars per million tokens (input, output) a request for ``model`` can be billed at:
        a refusal fallback can rerun it on Opus 5 or 4.8, which cost more than Opus 5.5 (an attempt declined
        mid-answer is billed as well, which this leaves out: refusals are rare). Unknown models get the
        dearest known rates."""
        price_in, price_out, _ = PRICES.get(model, UNKNOWN_PRICE)
        if model in FALLBACK_MODELS:
            fallback = PRICES["claude-opus-5"]
            price_in, price_out = max(price_in, fallback[0]), max(price_out, fallback[1])
        return price_in, price_out

    def worst_cost(self, model: str, request: Dict[str, Any], max_tokens: int) -> float:
        """The most a request can cost (US dollars): every prompt token written to the cache (1.25x the input
        price; prompts counted at 3 characters a token, more than English or JSON use) and ``max_tokens``
        of output."""
        price_in, price_out = self.prices(model)
        prompt = len(json.dumps({k: v for k, v in request.items() if k != "max_tokens"}, default=str)) / 3
        return (prompt * price_in * 1.25 + max_tokens * price_out) / 1e6

    def max_tokens_within(self, model: str, budget: Optional[float], request: Dict[str, Any],
                          wanted: int) -> int:
        """The most output tokens a request may ask for (up to ``wanted``) so that even at its worst it costs
        no more than ``budget`` (US dollars; None for no limit). 0 when not even MIN_OUTPUT_TOKENS fit."""
        if budget is None:
            return wanted
        room = (budget - self.worst_cost(model, request, 0)) * 1e6 / self.prices(model)[1]
        tokens = int(min(wanted, room))
        return tokens if tokens >= MIN_OUTPUT_TOKENS else 0


def response_text(response: Any) -> str:
    return "".join(getattr(block, "text", "") for block in response.content if block.type == "text").strip()


# --- the numbers check -------------------------------------------------------------------------

# A number as written: optional currency and sign, digits (with thousands commas), decimals, percent. A pasted
# slip may be in rand, shillings or naira ("R50", "KSh 200", "₦1,000"): written that way, a number is money too
# (and checked: glued to letters, it would otherwise pass as part of a name)
_NUMBER = re.compile(
    r"(?P<cur>[$£€₦₹₵]|\b(?:R|KSh|Ksh)\s?(?=[-+−]?\d))?(?P<sign>[-+−])?(?P<cur2>[$£€₦₹₵])?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<pct>\s?%|\s?per ?cent\b)?"
)
_ONE_IN = re.compile(r"\b1(?:\s+|-)in(?:\s+|-)(?P<num>\d{1,3}(?:,\d{3})+|\d+)\b")
# Fractional odds ("6/4", "11/10", "4 / 6") are one figure, the decimal odds they stand for (6/4 is 2.50): its
# two halves alone would each pass for any small number nearby
_FRACTION = re.compile(r"(?<![\w/.,])(?P<top>\d{1,4})\s*/\s*(?P<bottom>\d{1,4})(?![\d/]|\.\d)")
# Text that holds digits but no betting numbers: clock times, dates, helplines, age limits
_NOT_NUMBERS = [
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?m\.?\b)?", re.I),
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T[\d:.]+(?:Z|[+-]\d{2}:?\d{2})?)?"),
    re.compile(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
               r"|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?"
               r"(?:,?\s+\d{4})?\b", re.I),
    re.compile(r"1-800-GAMBLER|1-800-522-4700|0808 ?8020 ?133|\b988\b|\b116 ?123\b|\b24\s*[/-]\s*7\b"
               r"|\btwenty[- ]four[- ]seven\b", re.I),
    re.compile(r"\b(?:18|21)\s?\+"),
]
# A number glued to letters is part of a name or a unit we don't check ("76ers", "3rd", "7pm")
_GLUED_AFTER = re.compile(r"(?:st|nd|rd|th|am|pm|ers|s)\b", re.I)
ROUNDING_SHARE = 0.05  # "about 1,900" may round a number by at most 5% of it
SMALL_COUNT = 12  # bare whole numbers up to this are counts and positions: "3 legs", "leg 2", "day 1"

# Numbers written as words are checked like digits ("one in three", "ten times your stake", "seventy percent")
_WORD_VALUES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_WORD = "|".join(list(_WORD_VALUES) + ["hundred", "thousand"])
# What kind of number word each is, for joining them: "twenty-four" is one number, "four seven" two
_WORD_KINDS = {word: "unit" if value < 10 else "teen" if value < 20 else "tens" for word, value in _WORD_VALUES.items()}
_WORD_RUN = re.compile(rf"\b(?:an?\s+(?=(?:hundred|thousand)\b))?(?:{_WORD})(?:[-\s]+(?:{_WORD}))*\b", re.I)
# Fractions and multiples only count where they state a chance or a payout
_WORD_FORMS = [
    (re.compile(r"\b(?:a|one)[-\s]+half\s+(?=of\s+the\s+time\b)|\bhalf\s+(?=the\s+time\b)", re.I), "50% "),
    (re.compile(r"\b(?:a|one)[-\s]+third\s+(?=of\s+the\s+time\b)", re.I), "33% "),
    (re.compile(r"\b(?:a|one)[-\s]+quarter\s+(?=of\s+the\s+time\b)", re.I), "25% "),
    (re.compile(r"\bthree[-\s]+quarters\s+(?=of\s+the\s+time\b)", re.I), "75% "),
    (re.compile(r"\btwo[-\s]+thirds\s+(?=of\s+the\s+time\b)", re.I), "67% "),
    (re.compile(r"\b(\d{1,3})\s*/\s*([1-9]\d{0,2})\s+(?=of\s+the\s+time\b)"),
     lambda m: f"{round(100 * int(m[1]) / int(m[2]))}% "),
    (re.compile(r"\b(?:doubles?|doubled|doubling|twice)\s+(?=(?:your|the|my|their)\s+(?:money|stake|bet|cash)\b)",
                re.I), "2 times "),
    (re.compile(r"\b(?:triples?|tripled|tripling)\s+(?=(?:your|the|my|their)\s+(?:money|stake|bet|cash)\b)", re.I),
     "3 times "),
]
# A small whole number is still a betting figure in these forms: "12x", "ten times your stake", "3 in 4",
# "3 out of 4", "pays 9 to 1", "9/1" (but "2 to 4 legs" is a range of counts)
_COUNT_NOUN = re.compile(
    r"\s+(?:legs?|days?|games?|bets?|picks?|points?|pts|rebounds?|assists?|wins?|losses|matches|nights?|weeks?"
    r"|hours?|minutes?|teams?|players?|tickets?|parlays?|selections?|goals?|runs?|seasons?|months?|years?)\b", re.I)
_CHECKED_AFTER = re.compile(
    r"\s*[xX×](?![A-Za-z])|[-\s]fold\b"
    r"|\s+times\s+(?:(?:your|the|my|their)\s+)?(?:stake|money|bet|wager|cash|payout)\b"
    r"|\s+(?:times\s+)?(?:in|out\s+of)\s+\d|/\d", re.I)
_CHECKED_BEFORE = re.compile(r"\d\s+(?:times\s+)?(?:in|out\s+of)\s+$|\d/$", re.I)
_TO_AFTER = re.compile(r"\s+to\s+\d+(?:\.\d+)?\b")
# Money in words after the number ("11 dollars", "eleven bucks", "26 cents"); "pounds" is left out (weights)
_MONEY_AFTER = re.compile(r"\s*(?:dollars?|bucks?|(?P<cents>cents?)|quid|euros?)\b", re.I)
_TO_BEFORE = re.compile(r"\d\s+to\s+$", re.I)
# What a figure is said to be: a gain ("a profit of $0.26", "make about $3") or a loss ("lose $2.60")
_FILLER = (r"(?:(?:about|around|roughly|nearly|almost|approximately|only|just|an?|the|of|on|average|expected"
           r"|estimated|some)\s+){0,4}")
_GAIN_BEFORE = re.compile(rf"\b(?:profits?|gains?|make|makes|making|made|win|wins|winning|won|earn|earns|earning"
                          rf"|earned|up|ahead|plus)\s+{_FILLER}$", re.I)
_LOSS_BEFORE = re.compile(rf"\b(?:lose|loses|losing|lost|loss(?:es)?|cost|costs|costing|down|behind|minus)\s+"
                          rf"{_FILLER}$", re.I)


@dataclass(frozen=True)
class Figure:
    """One number as written in a text."""
    text: str
    value: float  # its size
    decimals: int
    percent: bool
    money: bool
    one_in: bool = False
    sign: int = 0  # +1 or -1 when written with a sign ("+$0.26", "-2.60"), else 0
    gain: bool = False  # said to be a gain: "a profit of $0.26", "make about $3"
    loss: bool = False  # said to be a loss: "lose $2.60", "costs $3"
    fraction: bool = False  # fractional odds ("6/4"), as the decimal odds they stand for

    @property
    def signed(self) -> float:
        """The number with the sign the text gives it (a loss counts as negative)."""
        return -self.value if self.sign < 0 or (not self.sign and self.loss) else self.value


def _spell_out(text: str) -> str:
    """Numbers written as words, as digits: "one in three" -> "1 in 3", "seventy percent" -> "70 percent"."""
    for pattern, digits in _WORD_FORMS:
        text = pattern.sub(digits, text)

    def value(match: "re.Match[str]") -> str:
        # Words join into one number only as numbers are written ("twenty-four", "one hundred five", "two
        # thousand"); "twenty-four seven" and "four seven" are two numbers, never added up
        numbers: List[int] = []
        total = current = 0
        last: Optional[str] = None
        for word in re.split(r"[-\s]+", match.group(0).lower()):
            if word in ("a", "an"):
                continue
            kind = _WORD_KINDS.get(word, word)
            joins = ((kind == "unit" and last in ("tens", "hundred", "thousand"))
                     or (kind in ("teen", "tens") and last in ("hundred", "thousand"))
                     or (kind == "hundred" and last in ("unit", "teen"))
                     or (kind == "thousand" and last in ("unit", "teen", "tens", "hundred") and not total))
            if last is not None and not joins:
                numbers.append(total + current)
                total = current = 0
            if kind == "hundred":
                current = (current or 1) * 100
            elif kind == "thousand":
                total, current = (current or 1) * 1000, 0
            else:
                current += _WORD_VALUES[word]
            last = kind
        numbers.append(total + current)
        return " ".join(str(number) for number in numbers)

    return _WORD_RUN.sub(value, text)


def _checked_small(text: str, start: int, end: int) -> bool:
    """Whether a small whole number at text[start:end] is a betting figure (a multiple, ratio or odds)."""
    before = text[max(0, start - 20):start]
    if _CHECKED_AFTER.match(text, end) or _CHECKED_BEFORE.search(before):
        return True
    to = _TO_AFTER.match(text, end)
    if to:
        return not _COUNT_NOUN.match(text, to.end())
    return bool(_TO_BEFORE.search(before)) and not _COUNT_NOUN.match(text, end)


def _fraction_odds(match: "re.Match[str]") -> Optional[float]:
    """The decimal odds fractional odds stand for (5/2 is 3.5), or None for a zero."""
    top, bottom = int(match.group("top")), int(match.group("bottom"))
    return 1 + top / bottom if top and bottom else None


def _figures(text: str, skip_trivial: bool, fractions: bool = True) -> List[Figure]:
    """The figures in a text. With ``fractions``, fractional odds are read as one figure, their decimal odds
    to 2 places ("6/4" is 2.50), unless a count follows ("7/10 games" is a record, not odds)."""
    figures: List[Figure] = []
    for match in _ONE_IN.finditer(text):
        number = match.group("num")
        figures.append(Figure(match.group(0), float(number.replace(",", "")), 0, False, False, one_in=True))
    # "1 in 12": the 12 was checked above and the 1 is just words
    text = _ONE_IN.sub(" ", text)
    if fractions:
        def odds(match: "re.Match[str]") -> str:
            value = _fraction_odds(match)
            if value is None:
                return match.group(0)
            if not _COUNT_NOUN.match(text, match.end()):
                figures.append(Figure(match.group(0), value, 2, False, False, fraction=True))
            return " "
        text = _FRACTION.sub(odds, text)
    for match in _NUMBER.finditer(text):
        start, end = match.span()
        before = text[start - 1] if start else " "
        if before.isalnum() or before in "_.":  # inside a word or a longer number ("fg3m", "v2.1")
            continue
        number = match.group("num")
        percent, money = bool(match.group("pct")), bool(match.group("cur") or match.group("cur2"))
        in_words = None if percent or money else _MONEY_AFTER.match(text, end)
        money = money or in_words is not None
        if not percent and not money and _GLUED_AFTER.match(text, end):
            continue
        if not percent and not money and end < len(text) and text[end].isalpha() and text[end] not in "xX":
            continue  # glued to a word: a team name or a unit ("5pts" is rare; "76ers" isn't)
        digits = number.replace(",", "")
        value = float(digits)
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        if in_words is not None and in_words.group("cents"):
            value, decimals = value / 100, decimals + 2  # "26 cents" is $0.26
        if skip_trivial and not percent and not money and decimals == 0 and not match.group("sign"):
            if (value <= SMALL_COUNT and not _checked_small(text, start, end)) \
                    or (len(digits) == 4 and 1900 <= value <= 2100):
                continue  # a count, a position or a year
        sign = {"+": 1, "-": -1, "−": -1}.get(match.group("sign") or "", 0)
        said = text[max(0, start - 60):start]
        figures.append(Figure(match.group(0).strip(), value, decimals, percent, money, sign=sign,
                              gain=bool(_GAIN_BEFORE.search(said)), loss=bool(_LOSS_BEFORE.search(said))))
    return figures


def numbers_in(text: str, exempt: Optional["re.Pattern[str]"] = None) -> List[Figure]:
    """The figures in a reply that must be traced back to the code: money, odds, chances, "1 in X". Text
    matching ``exempt`` (after numbers in words are read as digits) isn't checked: the assistant passes the
    app's own rules there ("a break of 90 days")."""
    for pattern in _NOT_NUMBERS:  # before and after the words: "twenty-four seven", "24/7"
        text = pattern.sub(" ", text)
    text = _spell_out(text)
    for pattern in _NOT_NUMBERS + ([exempt] if exempt is not None else []):
        text = pattern.sub(" ", text)
    return _figures(text, skip_trivial=True)


class Sources(set):
    """The numbers a reply may quote (a set of floats), and apart from them the decimal odds of the fractions
    written in the sources ("5/2" is 3.5): only a fraction in the reply may be traced to those, so "11/10" on a
    slip never lets a bare "2" through."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.fractions: Set[float] = set()


def source_numbers(*objects: Any) -> Sources:
    """Every number in tool results or a fact sheet (values and numbers inside strings), with its sign: a
    number in a string counts as negative when written so ("-2.60") or said to be a loss ("cost $3"). A line's
    ``point`` counts with both signs: Boston +4.5 is the Lakers -4.5. Fractions in strings count by their halves
    as before, and their decimal odds go in ``fractions`` (see Sources). A tool's ``hint`` is how to answer, not a
    fact: its words never count (an example in it would let a reply copy the example's numbers)."""
    found = Sources()

    def walk(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float, Decimal)):
            number = float(value)
            if math.isfinite(number):
                found.add(number)
        elif isinstance(value, str):
            # Each half of "5/2" counts as before; the decimal odds it stands for (3.5) only for a fraction
            found.update(figure.signed for figure in _figures(value, skip_trivial=False, fractions=False))
            found.fractions.update(odds for odds in map(_fraction_odds, _FRACTION.finditer(value)) if odds is not None)
        elif isinstance(value, dict):
            point = value.get("point")
            if isinstance(point, (int, float, Decimal)) and not isinstance(point, bool):
                walk(-point)
            for key, item in value.items():
                if key != "hint":
                    walk(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item)

    for obj in objects:
        walk(obj)
    return found


def _sign_fits(figure: Figure, source: float) -> bool:
    """A figure written "+$0.26", or called a profit or a win, can't stand for a loss the code worked out;
    one written "-$0.26" can't stand for a gain. ("Lose $10" may be the $10 stake, so a loss word alone
    doesn't need a negative number.)"""
    if figure.sign > 0:
        return source > 0
    if figure.sign < 0:
        return source < 0
    if figure.gain and not figure.loss:
        return source >= 0
    return True


def _traced(figure: Figure, sources: Iterable[float]) -> bool:
    """Whether a figure is a source number rounded to the precision shown (or a chance as a percentage),
    with a sign that fits it."""
    slack = 0.5 * 10 ** -figure.decimals + 1e-9
    whole = figure.decimals == 0 and not figure.percent and figure.value >= 100
    zeros = len(str(int(figure.value))) - len(str(int(figure.value)).rstrip("0")) if whole else 0
    if figure.fraction:  # fractional odds: the code's decimal odds, or a fraction written in the sources
        sources = [*sources, *getattr(sources, "fractions", ())]
    for signed in sources:
        if not _sign_fits(figure, signed):
            continue
        source = abs(signed)
        if abs(source - figure.value) <= slack:
            return True
        if figure.percent and source <= 1 and abs(source * 100 - figure.value) <= slack:
            return True
        # "about 1,900" for 1,901: rounded to the tens, hundreds or thousands shown (but never far off)
        gap = abs(source - figure.value)
        if zeros and gap <= 0.5 * 10 ** min(zeros, 3) and gap <= ROUNDING_SHARE * source:
            return True
    return False


def invented_numbers(text: str, sources: Set[float], exempt: Optional["re.Pattern[str]"] = None) -> List[str]:
    """The figures in ``text`` that no source number accounts for (as written); see numbers_in for ``exempt``."""
    return [figure.text for figure in numbers_in(text, exempt) if not _traced(figure, sources)]


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
    ("can't lose", re.compile(r"\b(?:can(?:'|’)?t|cannot|can not|won(?:'|’)?t)[-\s](?:lose|miss|go wrong)\b"
                              r"(?!\s+more\b)", re.I)),
    # Money won back, not sports: "win back-to-back titles", "won back home" and "make it back to the
    # playoffs" are fine
    ("win back", re.compile(
        rf"\b{_EARN}\s+(?:it(?:\s+all)?|that|them|{_LOST_MONEY})\s+back\b"
        r"(?![-‐\s]+(?:to[-‐\s]+back|home|to\s+(?:the|a|an|their|his|her|its)|into)\b)"
        rf"|\b{_EARN}\s+back\s+(?:{_LOST_MONEY}|what\b)"
        # "get your money back" (a refund of a void bet is fine: see REFUND) and "back what you lost"
        r"|\bget(?:s|ting)?\s+(?:back\s+(?:your|my|our|their)\s+(?:money|losses|cash)\b"
        r"|(?:your|my|our|their)\s+(?:money|losses|cash)\s+back\b)"
        r"|\bback\s+what\s+(?:you|I|we|they)(?:'ve|’ve|\s+have)?\s+lost\b", re.I)),
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
    # More money or more legs in other words, said as a suggestion and about money or bets: sports talk such
    # as "they went all in on defence", "loaded up the paint", "put more pressure on the rim" or "put up 30
    # points" is fine, and so is the warning "a bigger parlay pays more but wins less often"
    ("bet more", re.compile(
        r"\bdouble[- ]down\b"
        r"|\bgo(?:ing)?\s+all[- ]in\b(?!\s+on\s+(?:defen[cs]e|offen[cs]e|youth|the\s+(?:rebuild|draft|deadline|trade)))"
        r"|\b(?:put|throw|throwing|putting)\s+more\s+(?:money|cash|on\s+(?:it|this|that|them)|into\s+(?:it|this|that))\b"
        r"|\bload(?:ing)?\s+up\s+on\b"
        r"|\b(?:add|throw\s+in|tack\s+on)\s+(?:another|more|one\s+more|an?\s+(?:extra|second|third|fourth|fifth|sixth))"
        r"\s+legs?\b"
        r"|\b(?:try|go\s+for|build|make|chase\s+it\s+with)\s+(?:an?\s+)?(?:bigger|larger)\s+"
        r"(?:parlay|multi|ticket|accumulator|acca)s?\b"
        r"|\bmax(?:imi[sz]e)?\s+out\s+(?:your|the)\s+(?:stake|bet|wager|bankroll)\b"
        r"|\b(?:put|bet|stake|risk|throw|wager)\s+(?:(?:your|the)\s+(?:whole|entire)|all\s+(?:of\s+)?your)\s+bankroll\b"
        r"|\bgo\s+big\s+on\b|(?:^|[.!?:;]\s*)go\s+big\b", re.I | re.M)),
    ("safe", re.compile(r"\b(?:[Ii]s|[Ii]t(?:'|’)?s|[Ll]ooks|[Ff]eels|[Vv]ery|[Pp]retty|[Qq]uite|[Tt]otally"
                        r"|[Cc]ompletely|[Pp]erfectly|[Rr]eally|[Ss]uper|[Aa]bsolutely|[Ff]airly)\s+safe\b"
                        r"|\bsafe (?:bet|pick|play|ticket|parlay|multi|option|choice|money)s?\b")),
]


# These may appear in an honest warning ("nothing is guaranteed", "never raise your stake"); the rest
# oversell whatever surrounds them ("not a lock" still puts the idea in the reader's head). Only a negation
# that governs the phrase counts: right before it, with at most "is", "ever", "a" and the like between, or
# as the subject of its sentence ("Nothing in betting is guaranteed", "None of these picks is guaranteed").
# "No doubt about it, this one is guaranteed", "No one can deny it's guaranteed" and "Don't hold back: bet
# more" are not negated
NEGATABLE = {"guaranteed", "risk-free", "bigger stake", "bet more"}
_NEGATOR = (r"(?:never|not|no|nothing|nothing(?:'|’)s|nobody|don(?:'|’)?t|do\s+not|doesn(?:'|’)?t|does\s+not"
            r"|isn(?:'|’)?t|aren(?:'|’)?t|won(?:'|’)?t|will\s+not|shouldn(?:'|’)?t|should\s+not"
            r"|wouldn(?:'|’)?t|would\s+not|avoid|without)")
# Words that can't sit in a negated subject: they start a clause of their own ("no doubt ...", "no one can deny")
_NOT_SUBJECT = (r"(?:can|cannot|could|will|would|should|must|may|might|do|does|did|is|are|was|were|be|been"
                r"|doubt|question|matter|deny|wonder|kidding|worries|lie|way|that|which|who|if|but|and|or|so"
                r"|because|when|than)")
_NEGATION = re.compile(
    rf"\b{_NEGATOR}\s+(?:(?:is|are|was|be|been|ever|really|actually|always|truly|a|an|any|so)\s+){{0,2}}$"
    r"|\bno\s+(?:bet|ticket|parlay|multi|pick|leg|result|outcome|game|win|wager|one)s?\s+(?:is|are)\s+"
    r"(?:(?:ever|really|truly|a|an)\s+){0,2}$"
    rf"|\b(?:nothing|none|no|not\s+even)(?:\s+(?!{_NOT_SUBJECT}\b)[\w'’-]+){{0,4}}\s+"
    r"(?:is|are|was|were|(?:can|could|will|would|should|may|might|must)(?:\s+(?:ever|really|truly))?\s+be)\s+"
    r"(?:(?:ever|really|actually|always|truly|a|an|any|so)\s+){0,2}$"
    r"|\bno\s+such\s+thing\s+as\s+(?:an?\s+)?$|\bno\s+need\s+to\s+$"
    r"|\b(?:never|not|isn(?:'|’)?t)\s+(?:a\s+)?(?:good|smart|wise|great)\s+(?:idea\s+)?to\s+$"
    rf"|\b{_NEGATOR}\s+(?:suggest|recommend|advise|tell\s+you\s+to)(?:ing)?\s+(?:an?\s+|that\s+you\s+)?$", re.I)
# "You get your money back if the game is postponed" is a refund, not chasing
REFUND = re.compile(r"\b(?:void(?:ed|s)?|postpon\w*|cancel\w*|refund\w*|push(?:es|ed)?|scratch(?:ed)?|abandon\w*)\b",
                    re.I)
_CLAUSE_BREAK = re.compile(r"[,;:–—]|\s-\s|\b(?:and|but)\b", re.I)
_CONDITION = re.compile(r"\s*(?:if|when|whenever|once|since|because|as)\b", re.I)


def _refund(text: str, start: int, end: int) -> bool:
    """Whether "get your money back" at text[start:end] is a refund: a void, push or postponement in its own
    clause ("you get your money back if the game is postponed") or in the condition just before it ("If a leg
    is voided, you get your money back"). "A win gets your money back, and a push returns the stake" isn't."""
    left = max(text.rfind(mark, 0, start) for mark in ".!?\n") + 1
    rights = [i for i in (text.find(mark, end) for mark in ".!?\n") if i >= 0]
    right = min(rights) if rights else len(text)
    bounds = [left] + [i for cut in _CLAUSE_BREAK.finditer(text, left, right) for i in (cut.start(), cut.end())]
    bounds.append(right)
    clauses = [text[bounds[i]:bounds[i + 1]] for i in range(0, len(bounds), 2)]  # the text between the cuts
    own = next(i for i in range(len(clauses)) if bounds[2 * i + 1] >= end or i == len(clauses) - 1)
    if REFUND.search(clauses[own]):
        return True
    before = clauses[own - 1] if own else ""
    return bool(_CONDITION.match(before) and REFUND.search(before))


def banned_phrases(text: str) -> List[str]:
    """Which banned phrases the text uses (by name), in the order of the list above."""
    found = []
    for name, pattern in BANNED:
        for match in pattern.finditer(text):
            if name in NEGATABLE and _NEGATION.search(text[max(0, match.start() - 60):match.start()]):
                continue
            said = match.group(0).lower()
            # Losses are never refunded: only "get your money back" can be a void or a push
            if name == "win back" and said.startswith("get") and "loss" not in said \
                    and _refund(text, match.start(), match.end()):
                continue
            found.append(name)
            break
    return found


def problems(text: str, sources: Set[float], exempt: Optional["re.Pattern[str]"] = None) -> List[str]:
    """Why a text can't be shown (empty when it can); see numbers_in for ``exempt``."""
    found = []
    invented = invented_numbers(text, sources, exempt)
    if invented:
        found.append("numbers that are not in the facts you were given: " + ", ".join(dict.fromkeys(invented)))
    banned = banned_phrases(text)
    if banned:
        found.append("banned wording: " + ", ".join(banned))
    return found


# --- refusals and lectures ---------------------------------------------------------------------

# A reply on a look-up, a build or an analysis that says no, or describes a tool's limits, while a tool could have
# answered ("I can't pull up a list ... the props tool only works one player at a time"), or that lectures about the
# downside nobody asked about ("a projection of 20 points doesn't mean he'll score 20", "close to a coin flip"). The
# assistant sends it back once with a correction naming the tool to use, sharing the numbers check's one retry; a
# second failure gets code's template from the tools already called.
_CANT = r"(?:can(?:'|’)?t|cannot|can\s+not)"
# What a tool does when a reply says it can only do it for one player ("I can only look up one player"): never what a
# result says ("I could only find one player projected for 30+", "I can only see one player clearing 15+": M1)
_TOOL_VERB = (r"(?:look\s+up|check|pull(?:\s+up)?|scan|search(?:\s+for)?|do|handle|take|process|run|give\s+you|show\s+you"
              r"|list|rank|cover|price|answer\s+for)")
_NUMBER_WORD = r"(?:\d+|two|three|four|five|six|seven|eight|nine|ten)"
REFUSALS: Sequence[Tuple[str, "re.Pattern[str]"]] = [
    ("I can't", re.compile(rf"\bI\s+{_CANT}\b", re.I)),
    ("I'm not able", re.compile(r"\bI(?:(?:'|’)m|\s+am)\s+not\s+able\b", re.I)),
    ("I'm unable", re.compile(r"\bI(?:(?:'|’)m|\s+am)\s+unable\b", re.I)),
    ("isn't possible", re.compile(r"\b(?:isn(?:'|’)?t|is\s+not|not)\s+possible\b", re.I)),
    ("only works one", re.compile(r"\bonly\s+works?\s+(?:for\s+|on\s+|with\s+)?one\b", re.I)),
    ("can't pull", re.compile(rf"\b{_CANT}\s+pull\b", re.I)),
    ("can't list", re.compile(rf"\b{_CANT}\s+list\b", re.I)),
    # The same "no" in other words (the second persona review's probes)
    ("only handles one", re.compile(
        r"\bonly\s+(?:handles?|takes?|covers?|does|checks?|scans?|searches|looks?\s+up|pulls?(?:\s+up)?)\s+"
        r"(?:(?:one|a\s+single)\s+(?:player|name|person)s?|one\s+at\s+a\s+time)\b", re.I)),
    ("works for one player", re.compile(r"\bworks?\s+(?:only\s+)?(?:for|on|with)\s+(?:only\s+)?(?:one|a\s+single)\s+"
                                        r"player\b", re.I)),
    ("I can only look up one", re.compile(
        rf"\bI\s+(?:can|could)\s+only\s+{_TOOL_VERB}\s+(?:(?:info|data|props?|lines?|numbers?|stats?)\s+(?:on|for)\s+)?"
        r"(?:(?:one|a\s+single)\s+(?:player|name|person|prop)s?|one\s+at\s+a\s+time)\b"
        r"|\bI\s+(?:can|could)\s+only\s+(?:[\w’']+\s+){1,3}(?:at\s+a\s+time|per\s+(?:request|message|ask|question))\b",
        re.I)),
    ("I don't have a tool", re.compile(r"\bI\s+(?:don(?:'|’)?t|do\s+not)\s+have\s+(?:a|any|the)\s+(?:tool|way|feature)s?\b"
                                       r"|\bthere(?:(?:'|’)s|\s+is)\s+no\s+tool\b", re.I)),
    # Only for a list, a scan or a ranking: "I won't be able to show his line until DraftKings posts it" is news (M1)
    ("won't be able to", re.compile(
        r"\b(?:I|we)\s+(?:won(?:'|’)?t|will\s+not|wouldn(?:'|’)?t|would\s+not)\s+be\s+able\s+to\s+"
        r"(?:list|scan|rank|search|pull\s+up\s+(?:a\s+list|every|all|the\s+(?:list|whole))"
        r"|give\s+you\s+(?:a\s+(?:full\s+)?list|the\s+(?:full\s+)?list|every|all)|look\s+up\s+(?:every|all|each)"
        r"|check\s+(?:every|all|each)|go\s+through\s+(?:every|all))\b", re.I)),
    ("no way for me to", re.compile(r"\bno\s+way\s+for\s+me\s+to\b", re.I)),
    ("not something I can do", re.compile(r"\bnot\s+something\s+I\s+(?:can|could)\b", re.I)),
    ("I'm not set up to", re.compile(r"\bI(?:(?:'|’)m|\s+am)\s+not\s+(?:set\s+up|built|designed|equipped)\s+(?:to|for)\b",
                                     re.I)),
    # Not for one player's own lines ("I'd need a player's name for his alt lines; for the list, Luka leads": M1)
    ("I'd need a player's name", re.compile(
        r"\bI(?:(?:'|’)d|\s+would)\s+need\s+(?:a|the|one)\s+player(?:(?:'|’)s)?\s+name\b(?!\s+for\s+(?:his|her|their)\b)",
        re.I)),
    # The commonest refusals the third persona review still saw (M2)
    ("I don't have access", re.compile(
        r"\bI\s+(?:don(?:'|’)?t|do\s+not)\s+(?:currently\s+|really\s+|actually\s+)?have\s+(?:access\s+to"
        r"|the\s+(?:ability|means|capability|tools?)\s+to|(?:a|any)\s+(?:way|means)\s+to"
        r"|(?:the\s+|any\s+)?(?:data|info(?:rmation)?|numbers|stats|projections?|lists?)\s+(?:on|for)\s+(?:every|all|each"
        r"|the\s+whole|a\s+full)|(?:real[-\s]time|live|full|complete)\s+(?:projections?|data|lists?|stats)\s+for)\b",
        re.I)),
    ("only lets me … one", re.compile(r"\bonly\s+(?:let|lets|allow|allows|permit|permits)\s+(?:me|us)\s+(?:to\s+)?"
                                      r"(?:[\w’']+\s+){1,3}(?:one|a\s+single)\b", re.I)),
    ("I'm limited to", re.compile(r"\b(?:I(?:(?:'|’)m|\s+am)|(?:my|the)\s+tools?\s+(?:are|is)|(?:the\s+)?(?:props?\s+)?"
                                  r"(?:lookup|tool|scan)\s+is)\s+(?:only\s+)?(?:limited|restricted)\s+to\b", re.I)),
    ("I'm not going to", re.compile(r"\bI(?:(?:'|’)m|\s+am)\s+not\s+(?:going\s+to|gonna)\s+(?:pick|give|list|rank|call|name"
                                    r"|make|build|provide|offer|do|hand)\b", re.I)),
    # "I won't list all ten here: the card has them" points to the card, so it isn't one
    ("I won't give", re.compile(r"\bI\s+(?:won(?:'|’)?t|will\s+not)\s+(?:be\s+)?(?:giv(?:e|ing)|pick(?:ing)?|list(?:ing)?"
                                r"|rank(?:ing)?|call(?:ing)?|nam(?:e|ing)|provid(?:e|ing)|hand(?:ing)?\s+out)\b"
                                r"(?![^.!?\n]*\b(?:card|below|finder)\b)", re.I)),
    ("isn't available here", re.compile(
        r"\b(?:isn(?:'|’)?t|aren(?:'|’)?t|is\s+not|are\s+not|not)\s+(?:available|supported|offered|possible)\s+(?:here"
        r"|in\s+(?:this\s+|the\s+)?(?:chat|app)|through\s+(?:me|this\s+chat|the\s+chat)|via\s+(?:chat|me))\b"
        r"|\b(?:isn(?:'|’)?t|is\s+not|not)\s+something\s+(?:the\s+(?:app|chat|scan|tools?)|this\s+chat|I|we)\s+"
        r"(?:supports?|offers?|does|covers?|handles?|can\s+do)\b", re.I)),
    ("no way to list", re.compile(r"\bno\s+way\s+(?:for\s+(?:me|us)\s+)?to\s+(?:list|scan|rank|pull|search|look\s+up"
                                  r"|check\s+(?:every|all|each)|get\s+(?:a|the)\s+(?:full\s+)?list|see\s+(?:every|all))\b",
                                  re.I)),
    ("beyond what I can do", re.compile(r"\b(?:beyond|outside(?:\s+of)?)\s+what\s+I\s+(?:can|could)\s+do\b", re.I)),
    ("check each player individually", re.compile(r"\b(?:check|look\s+up|search)\s+(?:each|every)\s+player\s+"
                                                  r"(?:individually|separately|one\s+(?:by|at\s+a)\s+(?:one|time))\b",
                                                  re.I)),
]
# Refusals that describe a tool's limits (one player at a time, no list here) rather than what the data doesn't hold:
# never an honest "can't", even on the retry (M4)
TOOL_LIMITS = frozenset({
    "only works one", "only handles one", "works for one player", "I can only look up one", "I don't have a tool",
    "won't be able to", "no way for me to", "not something I can do", "I'm not set up to", "I'd need a player's name",
    "I don't have access", "only lets me … one", "I'm limited to", "I'm not going to", "I won't give",
    "isn't available here", "no way to list", "beyond what I can do", "check each player individually"})
LECTURES: Sequence[Tuple[str, "re.Pattern[str]"]] = [
    ("doesn't mean he'll", re.compile(r"\bdoes(?:n(?:'|’)?t|\s+not)\s+mean\s+(?:he|she|they|it|that|this)"
                                      r"(?:(?:'|’)ll|\s+will)\b", re.I)),
    ("coin flip", re.compile(r"\bcoin[-\s]?flips?\b", re.I)),
    ("no guarantee", re.compile(r"\bno\s+guarantees?\b", re.I)),
    ("nothing is guaranteed", re.compile(r"\bnothing(?:(?:'|’)s|\s+is)\s+(?:ever\s+)?guaranteed\b", re.I)),
    ("the house always wins", re.compile(r"\bthe\s+house\s+always\s+wins\b", re.I)),
    ("the bookmaker always wins", re.compile(r"\bthe\s+(?:bookmakers?|bookies?|books?|sportsbooks?)\s+always\s+wins?\b",
                                             re.I)),
    ("chance of each one missing", re.compile(r"\bchance\s+of\s+each\s+(?:one\s+)?missing\b", re.I)),
    ("you'll lose more", re.compile(r"\byou(?:(?:'|’)ll|\s+will)\s+lose\s+more\b", re.I)),
    # The same lecture in other words (the second persona review's probes)
    ("doesn't guarantee", re.compile(rf"\b(?:does(?:n(?:'|’)?t|\s+not)|won(?:'|’)?t|will\s+not|{_CANT})\s+guarantee\b",
                                     re.I)),
    ("isn't a guarantee", re.compile(r"\b(?:isn(?:'|’)?t|is\s+not|not)\s+(?:a|any)\s+guarantee\b", re.I)),
    ("50/50", re.compile(r"(?<![\d.,/])50\s*[/-]\s*50(?!\d|[.,]\d|\s*%|/)|\bfifty[-\s]fifty\b", re.I)),
    ("coin toss", re.compile(r"\bcoin[-\s]?toss(?:es)?\b", re.I)),
    ("toss-up", re.compile(r"\btoss[-\s]?ups?\b", re.I)),
    ("has the edge", re.compile(r"\bthe\s+(?:house|casino|bookmakers?|bookies?|books?|sportsbooks?)\s+(?:always\s+)?"
                                r"(?:has|have|holds?|keeps?)\s+(?:the|an)\s+edge\b", re.I)),
    ("most bettors lose", re.compile(r"\bmost\s+(?:bettors|punters|gamblers|people)\s+(?:(?:will|still|end\s+up)\s+)?"
                                     r"los(?:e|ing)\b", re.I)),
    ("misses 1 in N", re.compile(r"\b(?:he|she|they|it)\s+(?:still\s+)?miss(?:es)?\s+(?:about\s+|around\s+|roughly\s+"
                                 r"|nearly\s+|almost\s+)?(?:1|one)\s+(?:game\s+|time\s+|night\s+)?in\s+(?:\d+|two|three"
                                 r"|four|five|six|seven|eight|nine|ten)\b", re.I)),
    # The commonest lectures the third persona review still saw (M2)
    ("anything can happen", re.compile(r"\banything\s+can\s+happen\b", re.I)),
    ("always … risk", re.compile(r"\bthere(?:(?:'|’)s|\s+is)\s+always\s+(?:some\s+|a\s+)?risk\b|\balways\s+(?:some\s+|a\s+)?"
                                 r"risk\b|\brisk\s+is\s+always\b", re.I)),
    ("loses 1 in N", re.compile(
        rf"\b(?:loses?|fails?|misses)\s+(?:about\s+|around\s+|roughly\s+|nearly\s+|almost\s+|still\s+)?(?:1|one)\s+"
        rf"(?:time\s+|game\s+|night\s+|out\s+of\s+)?in\s+{_NUMBER_WORD}\b"
        r"(?!\s+(?:free\s+throws?|shots?|attempts?|field\s+goals?|from\s+(?:the\s+line|three|deep)))", re.I)),
    ("designed to lose", re.compile(r"\b(?:are|is|were)\s+(?:designed|built|made|set\s+up|meant)\s+(?:for\s+you\s+|so\s+you\s+)?"
                                    r"to\s+lose\b|\bdesigned\s+so\s+(?:that\s+)?(?:you|most\s+people|bettors)\s+lose\b",
                                    re.I)),
    ("betting is unpredictable", re.compile(
        r"\b(?:sports\s+)?(?:betting|gambling|sports|props?|parlays?)\s+(?:is|are)\s+(?:so\s+|very\s+|really\s+|always\s+"
        r"|notoriously\s+)?(?:unpredictable|volatile|a\s+gamble)\b", re.I)),
    ("the odds are against you", re.compile(r"\bodds\s+are\s+(?:always\s+)?(?:stacked\s+)?against\s+you\b", re.I)),
    ("past results aren't predictive", re.compile(
        r"\bpast\s+(?:results|performance)\s+(?:(?:is|are)(?:n(?:'|’)?t|\s+not)|do(?:es)?(?:n(?:'|’)?t|\s+not))\b"
        r"|\bnot\s+(?:necessarily\s+)?predictive\b", re.I)),
    ("a real chance it misses", re.compile(r"\breal\s+chance\s+(?:it|he|she|they|this|that)\s+(?:misses|loses|fails)\b",
                                           re.I)),
    ("variance is high", re.compile(r"\bvariance\s+is\s+(?:high|huge|big|brutal)\b|\bsize\s+(?:your\s+bets\s+)?accordingly\b",
                                    re.I)),
]
# Lectures whose words also say ordinary things (M1): each counts only in a sentence where ``context`` holds. "The house
# always has the edge" and "the books have the edge on props" are the lecture; "the book has the edge on his line (edge
# -3.1%)" is the honest no-value read. "Points overs are close to 50/50" and "these props are a toss-up" are the lecture;
# "a toss-up between Luka and SGA for the top spot", "tied 50-50 at the half" and "won the coin toss" are the game.
_SPECIFIC_EDGE = re.compile(r"\bedge\s*[+\-−]?\s*\d|\d(?:\.\d+)?\s*%|\bhere\b|\b(?:this|that|his|her|their|each|both"
                            r"|either)\s+(?:\w+\s+){0,2}(?:line|lines|under|over|price|prop|side|sides|bet|leg|pick"
                            r"|number)\b", re.I)
_BET_TALK = re.compile(r"\b(?:props?|overs?|unders?|bets?|betting|picks?|parlays?|legs?|tickets?|lines?|odds|wagers?|ml"
                       r"|moneylines?|spreads?|totals?|chances?|whether|gets?\s+there|this\s+one|that\s+one|these|those"
                       r"|most\s+of\s+them)\b|\b(?:basically|essentially|pretty\s+much|more\s+or\s+less|close\s+to|nearly"
                       r"|practically|just)\b", re.I)
_GAME_TALK = re.compile(r"\bbetween\b|\bwho\s+(?:scores?|gets?|has|wins?|finishes|ends\s+up)\b|\bfor\s+(?:the\s+)?(?:top"
                        r"|first|last|lead|mvp|scoring|title)\b|\btied\b|\bat\s+the\s+(?:half|break)\b|\bsplit\b"
                        r"|\bthe\s+(?:opening\s+|pre-?game\s+)?coin[-\s]?toss\b|\brecord\b", re.I)


def _generic_edge(sentence: str) -> bool:
    """The house's edge as a truism ("always", or no line or number of this ask's in the sentence)."""
    found = dict(LECTURES)["has the edge"].search(sentence)
    return bool(found and re.search(r"\balways\b", found.group(0), re.I)) or not _SPECIFIC_EDGE.search(sentence)


def _bet_context(sentence: str) -> bool:
    """Whether a "toss-up", "50/50" or "coin toss" is about the bet ("points overs are close to 50/50") rather than the
    game ("tied 50-50 at the half", "a toss-up between Luka and SGA for the top spot")."""
    return bool(_BET_TALK.search(sentence)) and not _GAME_TALK.search(sentence)


LECTURE_CONTEXT: Dict[str, Callable[[str], bool]] = {
    "has the edge": _generic_edge, "50/50": _bet_context, "toss-up": _bet_context, "coin toss": _bet_context}
GUARDED_INTENTS = ("lookup", "build", "analyse")
# Quoted words in a reply are someone else's (the user's question read back, a tool's label), not the reply's own
_QUOTED = re.compile(r"\"[^\"\n]{0,300}\"|“[^”\n]{0,300}”|‘[^’\n]{0,300}’|«[^»\n]{0,300}»")
# A user asking about the downside gets it: no lecture check then. "Risky" is a question in "how risky", "too risky",
# "is it risky?", "riskier", "riskiest" and "am I risking too much" (M3); only the preset's name ("build me a risky
# parlay", "a Risky 6 leg ticket") isn't
_ASKS_RISK = re.compile(r"\brisk(?:s|ier|iest|ing)?\b|\bdownside\b|\bworst[-\s]case\b"
                        r"|\b(?:how|too|so|very|really|pretty|quite|more|less|most|least|is\s+(?:it|this|that|he|she)"
                        r"|isn(?:'|’)?t\s+(?:it|this|that)|are\s+(?:they|these|those)|it(?:'|’)s|that(?:'|’)s)\s+risky\b"
                        r"|\brisky\s*\?"
                        r"|\bchances?\s+of\s+(?:losing|missing|it\s+missing)\b"
                        r"|\bhow\s+(?:likely|often)\b[^.?!]{0,40}\b(?:lose|loses|losing|miss|misses|missing|fail)\b"
                        r"|\bcould\s+(?:it|he|she|they|this)\s+(?:lose|miss)\b|\bwhat\s+if\s+(?:it|he|she|they)\s+miss",
                        re.I)


# The app's own rules, which a reply may state with "I can't": only the user adds to the slip or places a bet, and
# nothing suggests a bigger stake or gets round a budget or a break. Read across the whole reply (B1): "I can't give you
# picks right now. Take a break." is the rule at work, whichever sentence says it
_RULE_TALK = re.compile(r"\b(?:slip|place|placing|placed|stakes?|budget|break|helpline|deposit|wager\s+for)\b", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

# Which tools answer a route (the interim route-aware guard; the full one is the resolver's, plan section 7.8): a
# list of players goes through players_scan, the likeliest winners through brain_top_picks, a ticket through
# ticket_build, a named player's prop through player_get_props and the person's own ticket or slip through the
# ticket tools. Any other ask (live odds, a 2012 box score, a team's alt-line ladder) has no tool here: saying so
# plainly is the honest answer, and the refusal check is off for it.
ANALYSE_TOOLS = ("ticket_analyse", "ticket_price", "pasted_slip_analyse")
# A named player's prop ask: his line, an over or under, a number to reach ("is Luka going over his points line")
_PROP_ASK = re.compile(r"\b(?:props?|lines?|over|under|o/u|alt|odds|hit\s+rates?)\b|\b[ou]\d|\d\s*\+(?!\w)", re.I)
# ... but not one about the past or the futures, which player_get_props (tonight's posted lines) can't answer: "did luka
# go over his points line last night", "what were lebron's odds to win mvp in 2012", "his season-long points prop" (M4)
_PAST_OR_FUTURES = re.compile(
    r"\b(?:last\s+(?:night|season|year|week(?:end)?|month)|yesterday|(?:19|20)\d\d|mvp|futures?|season[-\s]long"
    r"|(?:win|winning|won)\s+(?:the\s+)?(?:title|championship|finals|ring|cup|award|scoring\s+title|rookie\s+of\s+the\s+year)"
    r"|awards?|all[-\s]star|did\s+(?:\w+\s+){1,3}(?:go|hit|cover|score|clear|get|make|cash)|went\s+(?:over|under)"
    r"|was\s+(?:his|her|their|the)\s+(?:line|prop|over|under))\b", re.I)
# A ticket of player props ("make me a 3 leg parlay of player props"): ticket_build prices game picks only (moneyline,
# spread, total; the slip takes no props), so no tool builds one and an honest "can't" is the answer
_PROPS_TICKET = re.compile(r"\bprops?\b|\bplayer\s+(?:bets?|picks?|lines?|markets?)\b", re.I)


def _field_of(route: Any, name: str) -> Any:
    if isinstance(route, Mapping):
        return route.get(name)
    return getattr(route, name, None)


def answering_tools(route: Any, asked: str = "") -> Set[str]:
    """The tools whose purpose covers ``route`` (the assistant's Route, a mapping or an intent; ``asked`` the person's
    message): players_scan for a player scan, brain_top_picks for the likeliest winners, ticket_build for a build (of
    game picks: a ticket of player props has no tool), player_get_props for a named player's prop and the ticket tools
    for an analysis. Empty for anything else, which turns the refusal check off (a lecture is still checked)."""
    intent = _route_intent(route)
    tools: Set[str] = set()
    if _field_of(route, "player_scan"):
        tools.add("players_scan")
    if _field_of(route, "top_picks"):
        tools.add("brain_top_picks")
    if intent == "build" and not _PROPS_TICKET.search(asked or ""):
        tools.add("ticket_build")
    if intent == "analyse":
        tools.update(ANALYSE_TOOLS)
    if _field_of(route, "players") and _PROP_ASK.search(asked or "") and not _PAST_OR_FUTURES.search(asked or ""):
        tools.add("player_get_props")
    return tools


def _route_intent(route: Any) -> Optional[str]:
    """The intent of a route given as the assistant's Route, a mapping ({"intent": ...}) or the intent itself."""
    if isinstance(route, str):
        return route
    if isinstance(route, Mapping):
        return route.get("intent")
    return getattr(route, "intent", None)


def asks_about_risk(asked: str) -> bool:
    """Whether the person's message asks about the downside: the risk, the chance of missing, or the lecture's own
    words ("is it a coin flip?", "is it guaranteed?"). A reply may talk about it then."""
    text = asked or ""
    return bool(_ASKS_RISK.search(text)) or any(pattern.search(text) for _, pattern in LECTURES)


# "Ask again in a few minutes" and its paraphrases ("try again in a minute", "check back shortly", "give it a few
# minutes"): never the answer when a build or the top picks miss prices, since the Builder has already waited for the
# boards on their way (the tools' ``prices`` line says why they're missing; ai/assistant.py's guard holds a reply to it)
ASK_AGAIN = re.compile(
    r"\b(?:ask|try|check|come|circle)\s+(?:me\s+|it\s+)?(?:again|back)\b[^.!?\n]{0,40}?"
    r"(?:\bin\s+(?:a\s+)?(?:few|couple(?:\s+of)?|\d+)\s+(?:more\s+)?min(?:ute)?s?\b|\bin\s+a\s+(?:minute|moment|bit|sec)\b"
    r"|\bshortly\b|\bin\s+a\s+little\s+while\b)"
    r"|\bgive\s+it\s+(?:a\s+)?(?:few|couple(?:\s+of)?|\d+)\s+min(?:ute)?s?\b", re.IGNORECASE)
ASK_AGAIN_PROBLEM = "banned wording (\"ask again in a few minutes\": the prices line says why they're missing)"


def asks_again(text: str) -> bool:
    """Whether a reply tells the person to ask again in a few minutes (ASK_AGAIN)."""
    return bool(ASK_AGAIN.search(text or ""))


def refusal_problems(text: str, route: Any, tools_available: Any, asked: str = "", allowed: str = "") -> List[str]:
    """Why a reply refuses or lectures when the tools could answer (empty when it doesn't), on a look-up, build or
    analyse ``route`` (the assistant's Route, a mapping or an intent; anything else, a support reply included, is
    never checked).

    - A refusal ("I can't", "I'm not able", "I'm unable", "isn't possible", "only works one", "can't pull", "can't
      list" and their paraphrases: "only handles one player at a time", "I can only look up one player", "I don't
      have a tool that lists", "won't be able to", ...) counts only while ``tools_available`` (the tools that answer
      this route, see answering_tools, or True) is non-empty: with picks paused, or an ask no tool covers, there is
      nothing to call, and saying so is right.
    - A lecture ("doesn't mean he'll", "coin flip", "no guarantee", "nothing is guaranteed", "the house/bookmaker
      always wins", "chance of each one missing", "you'll lose more" and their paraphrases: "doesn't guarantee",
      "50/50", "coin toss", "toss-up", "isn't a guarantee", ...) counts unless the user asked about the risk.

    Only quoted words are left out (the user's question read back): a phrase the user typed is theirs inside quotes,
    not across the whole reply ("I can't decide" never lets the reply say "I can't give you a list"). The turn's tool
    results (``allowed``, e.g. a chance label "coin flip") are the tools' own words and may be repeated. A reply that
    states the app's own rules anywhere ("only you add to the slip", "a break") has no refusal (B1); a lecture whose
    words say ordinary things too counts only in its own context (LECTURE_CONTEXT: M1)."""
    if _route_intent(route) not in GUARDED_INTENTS or not text:
        return []
    own = _QUOTED.sub(" ", text)
    allowed_text = allowed or ""
    found: List[str] = []
    sentences = _SENTENCE.split(own)

    def said(patterns: Sequence[Tuple[str, "re.Pattern[str]"]]) -> List[str]:
        names = []
        for name, pattern in patterns:
            if pattern.search(allowed_text):
                continue
            context = LECTURE_CONTEXT.get(name)
            if any(pattern.search(sentence) and (context is None or context(sentence)) for sentence in sentences):
                names.append(name)
        return names

    # "I can't add it to your slip" or "I can't suggest a bigger stake" is the app's rule, not a refusal
    if tools_available and not _RULE_TALK.search(own):
        refused = said(REFUSALS)
        if refused:
            found.append("a refusal (" + ", ".join(f'"{name}"' for name in refused) + ") although a tool can answer")
    if not asks_about_risk(asked):
        lectured = said(LECTURES)
        if lectured:
            found.append("a lecture nobody asked for (" + ", ".join(f'"{name}"' for name in lectured) + ")")
    return found


def honest_cant(text: str) -> bool:
    """Whether a reply is an honest one-line "can't" (M4): at most two sentences, a reason given after the "no" ("I can't
    see MVP futures odds: the boards carry tonight's games and props only"), no tool's limits described (TOOL_LIMITS:
    "only one player at a time") and no "name a player instead"."""
    plain = " ".join(_QUOTED.sub(" ", text or "").split())
    if not plain or len(plain) > HONEST_MAX or len([s for s in _SENTENCE.split(plain) if s.strip()]) > 2:
        return False
    refused = [(name, match) for name, pattern in REFUSALS for match in [pattern.search(plain)] if match]
    if any(name in TOOL_LIMITS for name, _ in refused) or _ONE_PLAYER_ASK.search(plain):
        return False
    rest = plain[min((match.end() for _, match in refused), default=0):]
    mark = _REASON_MARK.search(rest)
    if mark and len(re.findall(r"\w+", rest[mark.end():])) >= 3:
        return True
    sentences = [s for s in _SENTENCE.split(plain) if s.strip()]
    return len(sentences) == 2 and bool(_DATA_SCOPE.search(sentences[1]))


HONEST_MAX = 300  # characters: one plain line, two at most
# A reason after the "no": "... odds: the boards carry tonight's games", "... because the logs start last season", or a
# second sentence about what the data holds ("... The boards only cover tonight's games"); never "Try again later" or
# "I can't give you the odds." (no reason at all)
_REASON_MARK = re.compile(r"[:;—–]|\s-\s|\b(?:because|since)\b", re.I)
_DATA_SCOPE = re.compile(r"\bthe\s+(?:boards?|logs?|game\s+logs?|app|scan|data|feed|lines?|odds|prices?|tools?)\b"
                         r"|\bthere(?:(?:'|’)s|\s+is|\s+are)\s+no\b|\b(?:isn(?:'|’)?t|aren(?:'|’)?t|is\s+not|are\s+not"
                         r"|hasn(?:'|’)?t|haven(?:'|’)?t)\s+(?:been\s+)?(?:posted|loaded|out|up|priced|on)\b"
                         r"|\bonly\s+(?:cover|carry|go|hold|have|has|price|start)s?\b", re.I)
# "Name a player and I'll check his line", "ask me about one player": the old refusal's redirect, never honest
_ONE_PLAYER_ASK = re.compile(r"\b(?:name|give\s+me|tell\s+me|pick|send\s+me|choose)\s+(?:a|one|the|any)\s+(?:single\s+)?"
                             r"(?:player|name)\b|\bask\s+(?:me\s+)?about\s+(?:a|one|any)\s+(?:single\s+)?player\b"
                             r"|\bone\s+player\s+at\s+a\s+time\b|\bplayer\s+by\s+player\b|\bname\s+one\b"
                             r"|\btell\s+me\s+who\b|\bwho\s+(?:do\s+)?you\s+want\s+me\s+to\s+(?:check|look\s+up)\b", re.I)


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
You are the app's betting analyst, analysing a ticket for the person who built it. Our code has already \
worked out every number and gives you a JSON fact sheet: the priced ticket (each leg's odds, the AI's chance \
and the market's chance, the ticket's odds and its chance as "1 in X", what the bookmaker keeps, the expected \
result in money and the most likely outcome), notes on each leg (injuries, where the AI and the market \
disagree, analysts' views, rest days) and alternatives the code worked out.

Write three things:
- verdict: one or two plain sentences that lead with the picks and the ticket's odds and chance, once \
("Knicks and Celtics at 3.52, wins about 1 in 3"), then the weakest leg and why, in a few words. Use the \
chance words from the sheet (favourite, coin flip, underdog, long shot, lottery ticket).
- per_leg: one entry per leg, by its index from the sheet: one or two sentences on that leg using its notes.
- case_against: one short line: the main risk (the weakest leg's note, the bookmaker's cut or a note that \
worries you).

Rules:
- Every number you write must come from the fact sheet. You may round it or show a chance as a \
percentage; never add, subtract, multiply or divide numbers, and never estimate one yourself.
- Never claim certainty (lock, guaranteed, sure thing, risk-free, can't lose), never say win back or \
chasing losses, never call a ticket or a leg safe ("Safe" is only the name of a preset) and never suggest \
a bigger stake or adding legs. "Most likely", "strongest" and "best pick" are fine.
- No lectures or moralising: one plain line of numbers is the honesty. Short, confident and direct, plain \
English for a casual fan, no markdown, no emojis.
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


MAX_PICKS_NAMED = 4  # picks named in a template's first sentence; the rest are counted ("and 3 more")


def picks_text(labels: Sequence[Any], most: int = MAX_PICKS_NAMED) -> str:
    """A ticket's picks in a few words: "Knicks to win, Over 221.5 and Heat +3.5" ("... and 3 more" past
    ``most``)."""
    names = [" ".join(str(label).split()) for label in labels if label]
    if not names:
        return ""
    shown, rest = names[:most], len(names) - most
    if rest > 0:
        return ", ".join(shown) + f" and {rest} more"
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + f" and {shown[-1]}"


_GAME_SIDES = re.compile(r"\s+(?:@|vs\.?|v)\s+", re.I)


def short_game(game: Any) -> str:
    """A game in a few words: "Utah Jazz @ Denver Nuggets" -> "Jazz @ Nuggets" (each side's last word; the full
    names when those would read the same, or the game as given when it isn't "away @ home")."""
    text = " ".join(str(game or "").split())
    sides = _GAME_SIDES.split(text, maxsplit=1)
    if len(sides) != 2 or not all(sides):
        return text
    joiner = _GAME_SIDES.search(text).group(0).strip()  # type: ignore[union-attr]
    short = [side.split()[-1] for side in sides]
    if short[0].lower() == short[1].lower():
        short = sides
    return f"{short[0]} {joiner} {short[1]}"


def leg_name(leg: Mapping[str, Any]) -> str:
    """A leg's label for a sentence, with its game in a few words when the label names neither team ("Under 230.5
    (Jazz @ Nuggets)", "Draw (Arsenal @ Chelsea)"), so two totals can be told apart."""
    label = " ".join(str(leg.get("label") or "").split())
    game = " ".join(str(leg.get("game") or "").split())
    sides = _GAME_SIDES.split(game, maxsplit=1)
    if not label or len(sides) != 2 or not all(sides):
        return label
    low = label.lower()
    if any(side.lower() in low or side.split()[-1].lower() in low for side in sides):
        return label
    return f"{label} ({short_game(game)})"


def first_reason(why: Any) -> Optional[str]:
    """A pick's reason in a few words: the first ``why`` entry that isn't its chance again ("Markets say 79% (Sharp
    79% · US 79%)" repeats the number beside it), else that one."""
    entries = [str(entry).strip() for entry in (why if isinstance(why, list) else [why] if why else [])
               if isinstance(entry, str) and entry.strip()]
    for entry in entries:
        if not entry.lower().startswith(("markets say", "books say")):
            return entry
    return entries[0] if entries else None


def _odds_of(value: Any) -> Optional[float]:
    """Decimal odds as a number (from 3.52, "3.52" or {"decimal": 3.52}), or None."""
    if isinstance(value, dict):
        value = value.get("decimal")
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 1 else None


def _chance_words(one_in: Any, p: Any) -> str:
    """A ticket's chance as "1 in 3 (31%)" when it is under even, else "66% of the time": decided by the chance
    itself, since "1 in X" is 1/p rounded (anything from about 40% to 66% is "1 in 2", which beside 66% reads as
    50%)."""
    number = isinstance(p, (int, float)) and not isinstance(p, bool)
    if number and p >= 0.5:
        return f"{pct(p)} of the time"
    if isinstance(one_in, (int, float)) and not isinstance(one_in, bool) and one_in >= 2:
        return f"{one_in_text(one_in)} ({pct(p)})" if number else one_in_text(one_in)
    return f"{pct(p)} of the time"


def template_explanation(fact_sheet: Dict[str, Any]) -> Dict[str, Any]:
    """The same three texts written by code from the fact sheet (used when Claude's fail the checks), as the
    analyst would put them: the verdict leads with the picks and the ticket's odds and chance, then the weakest
    leg; the case against is one short line (a price at fair odds to check, else the bookmaker's cut and the
    expected result)."""
    ticket = _ticket(fact_sheet)
    chance = ticket.get("chance") or {}
    expected = ticket.get("expected") or {}
    stake = ticket.get("total_stake") or ticket.get("unit_stake")
    legs = _legs(fact_sheet)
    # A leg priced at the goals model's fair odds (soccer btts, dnb, double chance, team totals): its chance is that
    # model's, and no bookmaker here offers its price
    fair = [leg for leg in legs if leg.get("book") == MODEL_BOOK]
    only_fair = bool(fair) and len(fair) == len(legs)
    who = "the goals model" if only_fair else "the AI and the goals model" if fair else "the AI"
    parts = []
    head = picks_text([leg_name(leg) for leg in legs])
    odds = _odds_of(ticket.get("odds"))
    if head and odds is not None:
        head += f" at {odds:.2f}"
    if chance.get("ai_one_in") or chance.get("market_one_in"):
        views = []
        if chance.get("ai_one_in"):
            views.append(f"about {_chance_words(chance['ai_one_in'], chance.get('ai'))} by {who}"
                         + (" at fair odds" if fair else ""))
        if chance.get("market_one_in"):
            views.append(f"{_chance_words(chance['market_one_in'], chance.get('market'))} by the market")
        what = "makes a profit" if chance.get("headline") == "profit" else "wins"
        label = f" ({chance['label']})" if chance.get("label") else ""
        parts.append((f"{head}: " if head else "") + f"{what} " + " and ".join(views) + f"{label}.")
    elif head:
        parts.append(f"{head}.")
    weakest = ticket.get("weakest_leg")
    if len(legs) > 1 and isinstance(weakest, int) and not isinstance(weakest, bool) and 0 <= weakest < len(legs):
        leg = legs[weakest]
        p = leg.get("ai_prob") if leg.get("ai_prob") is not None else leg.get("market_prob")
        parts.append(f"The weakest leg is {leg_name(leg) or 'one leg'}" + (f" at {pct(p)}" if p is not None else "")
                     + ".")
    verdict = " ".join(parts) or "This ticket couldn't be priced, so there is nothing to explain yet."

    notes = _leg_notes(fact_sheet)
    per_leg = []
    for i, leg in enumerate(legs):
        index = leg.get("index", i)
        odds = (leg.get("odds") or {}).get("decimal")
        bits = [f"{leg.get('label', 'This leg')}" + (f" at {odds:.2f}" if isinstance(odds, (int, float)) else "")]
        ai, market, push = leg.get("ai_prob"), leg.get("market_prob"), leg.get("push_prob") or 0.0
        both = ai is not None and market is not None and push > 0
        if ai is not None:
            # The AI's chance is outright and the market's given no push: side by side, both given no push
            name = "goals model" if leg.get("book") == MODEL_BOOK else "AI"
            bits.append(f"{name} {pct(ai_given_no_push(ai, push) if both else ai)}")
        if market is not None:
            bits.append(f"market {pct(market)}" + (" (both given no push)" if both else ""))
        text = ": ".join([bits[0], ", ".join(bits[1:])]) if len(bits) > 1 else bits[0]
        if leg.get("chance_label"):
            text += f" ({leg['chance_label']})"
        text += "."
        if notes.get(index):
            text += " " + notes[index][0].rstrip(".") + "."
        per_leg.append({"index": index, "text": text})

    # The case against, one short line: a price at fair odds to check, else what the bookmaker keeps and the
    # expected result in money
    outcome = None
    basis = "ai" if expected.get("ai") is not None else "market"
    if expected.get(basis) is not None and stake:
        value = Decimal(str(expected[basis]))
        word = "lose about {} of" if value < 0 else "make about {} on"
        view = f"{who}'s view" if basis == "ai" else "the market's view"
        outcome = f"on average you {word.format(money(abs(value)))} your {money(stake)} ({view})"
    cut = ticket.get("book_cut")
    if fair:
        names = picks_text([leg.get("label") or "a leg" for leg in fair])
        case_against = (f"{names} {'is' if len(fair) == 1 else 'are'} priced at fair odds: no bookmaker here offers "
                        "that price, so check yours.")
    elif cut is not None and cut >= 0:
        case_against = (f"The bookmaker keeps about {pct(cut, 1)} of money staked on tickets like this"
                        + (f"; {outcome}" if outcome else "") + ".")
    elif cut is not None:
        case_against = ("These best prices beat the bookmakers' consensus: check they are still on offer"
                        + (f"; {outcome}" if outcome else "") + ".")
    elif outcome:
        case_against = f"{outcome[0].upper()}{outcome[1:]}."
    else:
        case_against = "Every bet carries the bookmaker's cut."
    return {"verdict": verdict, "per_leg": per_leg, "case_against": case_against}


def relaxed_texts(entries: Any) -> List[str]:
    """The rules closest effort loosened (engine/builder.py's ``relaxed``: [{rule, from, to, words}], or the words
    alone), as their words, in order and once each."""
    words: List[str] = []
    for entry in entries or []:
        text = entry.get("words") if isinstance(entry, dict) else entry
        if isinstance(text, str) and text.strip() and text.strip() not in words:
            words.append(text.strip())
    return words


def builder_template(cards: Sequence[Dict[str, Any]], most: int = 3) -> str:
    """The chat's answer in code's own words for Builder tickets (ticket_build's cards), when Claude's text failed
    the checks: one line per ticket that leads with its option label ("Likeliest at your odds") or title, its picks,
    its odds and its chance (the ticket's ``fit`` when closest effort gave one, else its own numbers), and the rules
    the Builder relaxed for that ticket, quoted on its own line (each option loosens its own way). It never refuses:
    the tickets are what code built. A league asked about that still had no prices once the Builder waited for it is
    said first, in the Lab's one line (a card's ``prices_note``: "NBA prices are still on their way; here's the best I
    can do with NHL and NFL games now.")."""
    lines = [note for note in (card.get("prices_note") for card in cards) if note][:1]
    shown = list(cards)[:most]
    for card in shown:
        ticket = card.get("ticket") or {}
        fit = card.get("fit") if isinstance(card.get("fit"), dict) else {}
        option = card.get("option") if isinstance(card.get("option"), dict) else {}
        name = option.get("label") or card.get("title") or "Ticket"
        odds = _odds_of(fit.get("odds")) or _odds_of(ticket.get("odds"))
        chance = fit.get("chance")
        if chance is None:
            chance = (ticket.get("joint") or {}).get("chance")
        if chance is None:
            chance = (ticket.get("chance") or {}).get("ai")
        line = f"{name}: {picks_text([leg_name(leg) for leg in ticket.get('legs') or []]) or 'these picks'}"
        if odds is not None:
            line += f", {odds:.2f}x"
        if isinstance(chance, (int, float)) and not isinstance(chance, bool):
            line += f", wins about {pct(chance)} of the time"
        words = relaxed_texts(card.get("relaxed"))
        if words:
            line += f" (relaxed: {'; '.join(words)})"
        lines.append(line + ".")
    lines.append("The cards below have the details." if len(shown) > 1 else "The card below has the details.")
    return "\n".join(lines)


def picks_template(card: Dict[str, Any], most: int = 5) -> str:
    """The chat's answer in code's own words for the likeliest winners (brain_top_picks' card), strongest first:
    each pick's chance, price and book and why, in a few words (first_reason: not the chance again), after the Lab's
    one line on a league asked about that still has no prices (``prices_note``), when there is one."""
    picks = [pick for pick in card.get("picks") or [] if isinstance(pick, dict)]
    note = card.get("prices_note")
    if not picks:
        return note or "No game in scope has prices loaded right now: try another sport."
    lines = ([note] if note else []) + ["The most likely winners, strongest first:"]
    for i, pick in enumerate(picks[:most], 1):
        line = f"{pick.get('rank') or i}. {pick.get('label') or 'A pick'}"
        if isinstance(pick.get("chance"), (int, float)):
            line += f", {pct(pick['chance'])}"
        odds = _odds_of(pick.get("price"))
        if odds is not None:
            line += f" at {odds:.2f}" + (f" ({pick['book']})" if pick.get("book") else "")
        first = first_reason(pick.get("why"))
        if first:
            line += f": {first.rstrip('.')}"
        lines.append(line + ".")
    lines.append("Add any of them to your slip from the card below.")
    return "\n".join(lines)


FORM_WORDS = "from recent form"  # how a row from the form model (not the NBA game model) is labelled everywhere
FINDER_WORDS = "open the Player finder"  # where a scan's rows beyond the ones it returned are listed
SCAN_SORTS = ("chance", "projection", "l10", "edge", "hot", "cold")


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def scan_hits(hits: Any, window: str = "l10") -> Optional[str]:
    """A row's history in words: "9 of last 10" (``window`` "l5" or "l10"), "30 of 45 this season"; None without
    games. ``hits`` is the scan's {l5: [hits, games], l10: [...], season: [...]}."""
    pair = (hits or {}).get(window) if isinstance(hits, Mapping) else None
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    made, games = (_number(part) for part in pair)
    if made is None or not games:
        return None
    if window == "season":
        return f"{made:g} of {games:g} this season"
    return f"{made:g} of last {games:g}"


def scan_matchup(row: Mapping[str, Any]) -> str:
    """"DAL v PHX": the player's team v the opponent (abbreviations when the scan has them)."""
    team = row.get("team_abbr") or row.get("team")
    opponent = row.get("opponent_abbr") or row.get("opponent")
    if team and opponent:
        return f"{team} v {opponent}"
    return str(team or opponent or row.get("game") or "")


def scan_when(content: Mapping[str, Any]) -> str:
    return "tomorrow" if content.get("day") == "tomorrow" else "tonight"


def scan_ask(content: Mapping[str, Any]) -> str:
    """The ask in words ("20+ points"): the scan's own, else built from its fields."""
    ask = content.get("ask")
    if isinstance(ask, str) and ask.strip():
        return ask.strip()
    if isinstance(ask, Mapping) and ask.get("label"):
        return str(ask["label"])
    label = content.get("stat_label") or content.get("stat") or "the stat"
    threshold = _number(content.get("threshold"))
    if threshold is None:
        return f"{label} against the posted line"
    if content.get("side") == "under":
        return f"under {threshold - 0.5:g} {label}"
    return f"{threshold:g}+ {label}"


def names_text(names: Any) -> str:
    """"Stephen Curry", "Stephen Curry and Luka Dončić", "A, B and C"."""
    shown = [" ".join(str(name).split()) for name in names or [] if str(name).strip()] \
        if isinstance(names, (list, tuple)) else []
    if len(shown) <= 1:
        return "".join(shown)
    return ", ".join(shown[:-1]) + " and " + shown[-1]


def scan_mode(content: Mapping[str, Any]) -> str:
    """"edges" for the best edges across every stat's posted lines, else "scan"."""
    return "edges" if content.get("mode") == "edges" else "scan"


def scan_by_edge(content: Mapping[str, Any]) -> bool:
    """Whether a scan ranks by edge, so its rows show it: the best edges across stats, sort edge, or each player's own
    posted line (no number asked and no other order)."""
    sort = content.get("sort")
    return scan_mode(content) == "edges" or sort == "edge" or (content.get("threshold") is None and not sort)


def _yes_no_stat(league: Any, stat: Any) -> bool:
    """Whether the scan's table has ``stat`` as a yes/no event (anytime TD, anytime goal, double- and triple-double):
    in ``league``'s table, or with no league in any table that has it."""
    try:
        from ..engine.scan import STATS
    except Exception:  # a server without the scan's engine
        return False
    key = str(stat or "")
    if league in STATS:
        return bool((STATS[league].get(key) or {}).get("yes"))
    return any((table.get(key) or {}).get("yes") for table in STATS.values())


def scan_yes_no(row: Mapping[str, Any], content: Optional[Mapping[str, Any]] = None) -> bool:
    """Whether a scan row is a yes/no event (its chance is the answer; a "projection" of 1.6 double-double categories
    means nothing): its ``kind`` ("yes_no"), else its stat's own kind."""
    content = content or {}
    kind = row.get("kind")
    if kind:
        return kind == "yes_no"
    leagues = content.get("leagues") or []
    league = row.get("league") or (leagues[0] if leagues else None)
    return _yes_no_stat(league, row.get("stat") or content.get("stat"))


def edge_text(edge: Any) -> Optional[str]:
    """A row's edge (chance x posted price - 1) in words: "edge +6.8%"."""
    value = _number(edge)
    return None if value is None else f"edge {value * 100:+.1f}%"


def scan_trend_text(trend: Any, short: bool = True) -> Optional[str]:
    """A row's last 5 against its baseline, the same on the app's chip, the template and the bot (contract 2 of
    SPEC_P0_FIX): "L5 28.4 vs 24.1 season", or "L5 28.4 vs 24.1 last season" when the baseline is last season's
    (``vs`` "last_season": this season is still short of games). ``short`` is kept for older callers."""
    if not isinstance(trend, Mapping):
        return None
    l5, season = _number(trend.get("l5")), _number(trend.get("season"))
    if l5 is None or season is None:
        return None
    return f"L5 {l5:g} vs {season:g} {'last season' if trend.get('vs') == 'last_season' else 'season'}"


def scan_edge_label(row: Mapping[str, Any]) -> str:
    """One of the best edges' own bet in words: its stat, side and line ("points over 29.5", "points + rebounds +
    assists under 33.5"), or a yes/no event's side ("double-double (yes)": m2)."""
    label = str(row.get("stat_label") or row.get("stat") or "line")
    side = str(row.get("side") or "over")
    if scan_yes_no(row):
        return f"{label} ({'no' if side == 'under' else 'yes'})"
    line = row.get("line") if isinstance(row.get("line"), Mapping) else {}
    number = _number(line.get("line"))
    return f"{label} {side}" + (f" {number:g}" if number is not None else "")


def scan_chance_text(row: Mapping[str, Any], content: Mapping[str, Any], under: bool = False) -> Optional[str]:
    """A row's chance as the bot says it: "55%", "55% to stay under", and for hot and cold, whose rows each have their
    own number, what the chance is for: "55% for 30+" (m2). None without a chance."""
    chance = _number(row.get("chance"))
    if chance is None:
        return None
    if content.get("sort") in ("hot", "cold") and not scan_yes_no(row, content):
        threshold = _number(row.get("threshold"))
        if threshold is not None:
            return f"{pct(chance)} to stay under {threshold - 0.5:g}" if under else f"{pct(chance)} for {threshold:g}+"
    return pct(chance) + (" to stay under" if under else "")


def scan_heading(content: Mapping[str, Any]) -> str:
    """A scan's heading: "Projected for 20+ points" (or "under 9.5 rebounds") for a number, "Strongest anytime TD" or
    "Strongest PRA overs (posted lines)" for a yes/no ask or each player's posted line, "Best edges on the posted
    lines" (or overs, unders) across every stat, "Hottest on points (last 5 against the season)" (or "Coldest") by
    form, "Stephen Curry: 5+ threes" for the players named."""
    if scan_mode(content) == "edges":
        lines = {"over": "overs", "under": "unders"}.get(str(content.get("side") or ""), "lines")
        return f"Best edges on the posted {lines}"
    sort = content.get("sort")
    if sort in ("hot", "cold"):
        label = content.get("stat_label") or content.get("stat") or "the stat"
        return f"{'Hottest' if sort == 'hot' else 'Coldest'} on {label} (last 5 against the season)"
    ask = scan_ask(content)
    named = names_text(content.get("names"))
    if named:
        return f"{named}: {ask}"
    if "+" in ask.split(" ")[0] or ask.startswith("under "):
        return f"Projected for {ask}"
    return f"Strongest {ask}"


def scan_row_text(row: Mapping[str, Any], under: bool = False, content: Optional[Mapping[str, Any]] = None) -> str:
    """One scan row in code's words: "Luka Dončić (DAL v PHX) 31.2 projected, 82% to get there, 9 of last 10, line
    19.5 posted". A yes/no event gives only its chance (and history); a row ranked by edge adds "edge +6.8%"; a hot or
    cold one leads with his last 5 against the season; one of the best edges across stats names its own stat, side
    and line ("points over 29.5: ..."); a named player ruled out says so. ``content`` is the scan's card or facts."""
    content = content or {}
    name = " ".join(str(row.get("name") or "A player").split())
    matchup = scan_matchup(row)
    text = f"{name} ({matchup})" if matchup else name
    if row.get("status") and str(row["status"]).lower() not in ("active", "healthy", "available"):
        text += f" [{row['status']}]"
    if row.get("ruled_out"):
        return text + ": ruled out"
    edges = scan_mode(content) == "edges"
    side = str(row.get("side") or "over") if edges else ("under" if under else "over")
    line = row.get("line") if isinstance(row.get("line"), Mapping) else None
    yes_no = scan_yes_no(row, content)
    trended = content.get("sort") in ("hot", "cold")
    bits = []
    if trended:
        trend = scan_trend_text(row.get("trend"))
        if trend:
            bits.append(trend)
    projection = _number(row.get("projection"))
    if projection is not None and not yes_no:
        bits.append(f"{projection:.1f} projected")
    chance = _number(row.get("chance"))
    if chance is not None:
        if edges and yes_no:
            bits.append(pct(chance))  # "double-double (yes): 65%"
        elif edges:
            bits.append(f"{pct(chance)} to {'stay under' if side == 'under' else 'clear it'}")
        elif trended and not yes_no and _number(row.get("threshold")) is not None:
            bits.append(scan_chance_text(row, content, under) or pct(chance))  # "82% for 30+": each row's own number
        else:
            bits.append(f"{pct(chance)} to {'stay under' if under else 'get there'}")
    history = scan_hits(row.get("hits")) or scan_hits(row.get("hits"), "season")
    if history:
        bits.append(history)
    if edges:
        text += f" {scan_edge_label(row)}:"
    elif line and line.get("kind") == "milestone" and _number(line.get("line")) is not None:
        reached = line.get("label") or f"{_number(line.get('line')) + 0.5:g}+"  # a 19.5 milestone is "20+"
        bits.append(f"{reached} posted")
    elif line and _number(line.get("line")) is not None:
        bits.append(f"line {_number(line['line']):g} posted")
    edge = edge_text(row.get("edge")) if scan_by_edge(content) else None
    if edge:
        bits.append(edge)
    if bits:
        text += " " + ", ".join(bits)
    if row.get("model") == "form":
        text += f" ({FORM_WORDS})"
    return text


def scan_loading_text(loading: Any) -> Optional[str]:
    """"8 of 10 games loaded so far: ask again in a minute for the rest." while the board builds (only while fewer
    games are ready than there are), "2 games still missing data: ask again in a minute." for parts that failed and
    are still missing, else None."""
    if not isinstance(loading, Mapping):
        return None
    ready, total = _number(loading.get("games_ready")), _number(loading.get("games_total"))
    failed = _number(loading.get("failed")) or 0
    if not total:  # the day's games themselves are still being read
        return "The games are still loading: ask again in a minute."
    if ready is not None and ready < total:
        return f"{ready:g} of {total:g} games loaded so far: ask again in a minute for the rest."
    if failed > 0:
        return f"{failed:g} game{'' if failed == 1 else 's'} still missing data: ask again in a minute."
    return None


def _or_text(names: Sequence[Any]) -> str:
    """"A", "A or B", "A, B or C"."""
    shown = [" ".join(str(name).split()) for name in names if str(name).strip()]
    return shown[0] if len(shown) <= 1 else ", ".join(shown[:-1]) + " or " + shown[-1]


def scan_name_notes(content: Mapping[str, Any]) -> List[str]:
    """The names asked for that the scan couldn't rank, one line each, in the same words on every surface (contract 1
    of SPEC_P0_FIX; no pronouns for players): "I couldn't find Ja Morant in tonight's games." (``missing_names``),
    "More than one player matches Jalen: Jalen Brunson, Jalen Green or Jalen Williams. Which one?" (``ambiguous_names``)
    and "Leon Draisaitl's game (CGY @ EDM) has already started, so it isn't in the scan." (``started_names``)."""
    day = f"{scan_when(content)}'s"
    notes = [f"I couldn't find {' '.join(str(name).split())} in {day} games."
             for name in content.get("missing_names") or [] if str(name).strip()]
    for entry in content.get("ambiguous_names") or []:
        if isinstance(entry, Mapping) and str(entry.get("name") or "").strip():
            candidates = [name for name in entry.get("candidates") or [] if str(name).strip()]
            name = " ".join(str(entry["name"]).split())
            notes.append(f"More than one player matches {name}: {_or_text(candidates)}. Which one?" if candidates
                         else f"More than one player matches {name}. Which one?")
    for entry in content.get("started_names") or []:
        if isinstance(entry, Mapping) and str(entry.get("player") or entry.get("name") or "").strip():
            player = " ".join(str(entry.get("player") or entry.get("name")).split())
            game = " ".join(str(entry.get("game") or "").split())
            notes.append(f"{player}'s game ({game}) has already started, so it isn't in the scan." if game
                         else f"{player}'s game has already started, so it isn't in the scan.")
    return notes


def scan_started_note(content: Mapping[str, Any]) -> Optional[str]:
    """The games of the day that have started (contract 3: ``games`` is the games still to come, ``games_started`` the
    ones under way): "Every game tonight has started, so nobody is left to rank." when none is left, else "2 games
    tonight have already started: their players aren't listed."; None when none has."""
    started = int(_number(content.get("games_started")) or 0)
    if started <= 0:
        return None
    when = scan_when(content)
    if int(_number(content.get("games")) or 0) <= 0:
        return f"Every game {when} has started, so nobody is left to rank."
    return (f"{started} game{'' if started == 1 else 's'} {when} {'has' if started == 1 else 'have'} already started: "
            "their players aren't listed.")


def scan_notes(content: Mapping[str, Any]) -> List[str]:
    """What else a scan's reader needs in one line each: the names it couldn't rank (scan_name_notes), the scores not
    loading (for an empty scan) and the games already started (scan_started_note)."""
    notes = scan_name_notes(content)
    if not (content.get("players") or content.get("close")) and content.get("error"):
        notes.append("ESPN's scores didn't load just now: ask again in a minute.")
    started = scan_started_note(content)
    if started:
        notes.append(started)
    return notes


def scan_says_empty(content: Mapping[str, Any]) -> bool:
    """Whether an empty scan says "No player is projected for ... yet" (scan_empty_text): not when its notes say why
    already (the names asked for weren't found, or every game has started and no league was without games)."""
    if content.get("players") or content.get("close"):
        return False
    names = content.get("names") and (content.get("missing_names") or content.get("ambiguous_names")
                                      or content.get("started_names"))
    every_started = int(_number(content.get("games_started")) or 0) > 0 and not int(_number(content.get("games")) or 0) \
        and not content.get("leagues_without_games")
    return not (names or every_started)


def scan_more_text(content: Mapping[str, Any]) -> Optional[str]:
    """How many further rows the scan ranked beyond the ones it returned (``more``, the tool's own number: no surface
    counts it again, contract 4), worded by mode as on every surface: "229 more qualify: open the Player finder.", "229
    more players are ranked: ..." (hot and cold), "14 more posted lines have an edge: ..." (the best edges). None when
    there are none."""
    more = _number(content.get("more"))
    more = int(more) if more is not None else 0
    if more <= 0:
        return None
    if scan_mode(content) == "edges":
        what = f"{more} more posted line{' has' if more == 1 else 's have'} an edge"
    elif content.get("sort") in ("hot", "cold"):
        what = f"{more} more player{' is' if more == 1 else 's are'} ranked"
    else:
        what = f"{more} more qualif{'ies' if more == 1 else 'y'}"
    return f"{what}: {FINDER_WORDS}."


def scan_empty_text(content: Mapping[str, Any]) -> List[str]:
    """What an empty scan says: "No player is projected for anytime TD tonight yet (NFL: no games)." (the best edges
    and hot or cold in their own words), and "There are none tomorrow either." when tomorrow was tried too."""
    when = scan_when(content)
    where = ", ".join(str(league) for league in content.get("leagues_without_games") or [])
    if scan_mode(content) == "edges":
        first = f"No posted line shows an edge {when} yet"
    elif content.get("sort") in ("hot", "cold"):
        first = f"No hot or cold reads on {content.get('stat_label') or content.get('stat') or 'that stat'} {when} yet"
    else:
        first = f"No player is projected for {scan_ask(content)} {when} yet"
    lines = [first + (f" ({where}: no games)" if where else "") + "."]
    if where and content.get("checked_tomorrow"):
        lines.append("There are none tomorrow either.")
    return lines


def scan_template(content: Mapping[str, Any], most: int = 5) -> str:
    """The chat's answer in code's own words for a player scan (players_scan's card, or Lab.player_scan's result),
    strongest first: "Projected for 20+ points tonight:" then one line per player (up to ``most``) with the
    projection, the chance and the history (a yes/no event's chance only; the edge on rows ranked by it), then
    "Also: ..." for every other row it returned (as the app's card lists them) and "N more qualify: open the Player
    finder." for the rows beyond those (scan_more_text: the tool's ``more``), or "Also close: ..." when every
    qualifier is named, and, for a partial board, how many games are loaded. The best edges across stats, hot and
    cold, and the players named read the same way. It never refuses: when nobody qualifies yet it says so (and which
    league has no games, and that tomorrow was tried)."""
    edges = scan_mode(content) == "edges"
    under = content.get("side") == "under" and not edges
    ask, when, heading = scan_ask(content), scan_when(content), scan_heading(content)
    players = [row for row in content.get("players") or [] if isinstance(row, Mapping)]
    close = [row for row in content.get("close") or [] if isinstance(row, Mapping)]
    named = names_text(content.get("names")) if not edges and content.get("sort") not in ("hot", "cold") else ""
    lines = []
    moved = [str(league) for league in content.get("no_games_today") or []]
    if moved:
        lines.append(f"No {names_text(moved)} games today, so here's tomorrow.")
    if players:
        lines.append(f"{named} for {ask} {when}:" if named else f"{heading} {when}:")
        lines += [f"{row.get('rank') or i}. {scan_row_text(row, under, content)}"
                  for i, row in enumerate(players[:most], 1)]
        also = players[most:]
        if also:
            lines.append("Also: " + "; ".join(scan_row_text(row, under, content) for row in also) + ".")
        more = scan_more_text(content)
        if more:
            lines.append(more)
        elif close and not also:
            lines.append("Also close: " + "; ".join(scan_row_text(row, under, content) for row in close[:3]) + ".")
    elif close:
        lines.append(f"Nobody is projected for {ask} {when} yet; closest:")
        lines += [f"{i}. {scan_row_text(row, under, content)}" for i, row in enumerate(close[:most], 1)]
    elif scan_says_empty(content):
        lines += scan_empty_text(content)
    lines += scan_notes(content)
    loading = scan_loading_text(content.get("loading"))
    if loading:
        lines.append(loading)
    if players or close:
        lines.append("The card below has each player's lines and history.")
    return "\n".join(lines)


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
                   effort: str = EXPLAIN_EFFORT, budget: Optional[float] = None,
                   before: Optional[Callable[[], None]] = None) -> Explanation:
    """Claude's verdict, per-leg notes and case against for a ticket, checked against the fact sheet.

    A text with a number that isn't on the sheet, or with banned wording, is sent
    back once with the reasons; if the second try fails too (or Claude refuses,
    or its answer is cut off, or ``budget`` (US dollars) can't pay for a try) the
    template is used. ``before`` runs before each try (the assistant renews its
    AI hold there, and raises AIError once it has lapsed). Raises AIError only
    when the API itself fails or ``before`` does (its ``cost`` holds what was
    already spent).
    """
    sources = source_numbers(fact_sheet)
    messages: List[Dict[str, Any]] = [{"role": "user", "content": json.dumps(fact_sheet, default=str)}]
    out = Explanation("", [], "", "template")
    for attempt in range(2):
        request: Dict[str, Any] = {
            "max_tokens": 8000,
            "system": [{"type": "text", "text": EXPLAIN_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": EXPLAIN_SCHEMA}},
        }
        if budget is not None:
            request["max_tokens"] = caller.max_tokens_within(model, budget - out.cost, request, request["max_tokens"])
            if not request["max_tokens"]:
                out.notes.append("the AI allowance left can't pay for another try")
                break
        try:
            if before is not None:
                before()
            response = caller.create(model, **request)
        except AIError as exc:
            raise AIError(str(exc), out.cost + exc.cost) from None
        cost, tokens_in, tokens_out = caller.cost(model, response.usage, getattr(response, "model", None))
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
