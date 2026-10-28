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


class AIError(Exception):
    """Claude couldn't answer; the message says why in plain English (never contains a key).

    ``cost`` is what the failed attempt still cost, in US dollars. ``temporary`` is False when waiting won't
    help (a bad key, an unknown model, a request Claude refuses), so the person isn't told to try again.
    """

    def __init__(self, message: str, cost: float = 0.0, temporary: bool = True) -> None:
        super().__init__(message)
        self.cost = cost
        self.temporary = temporary


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
            logging.warning("Claude refused a %s request: %s", model, str(exc.message)[:500])
            raise AIError("Claude didn't accept the app's request (the details are in the server log)",
                          temporary=False) from None
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
    as before, and their decimal odds go in ``fractions`` (see Sources)."""
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
            for item in value.values():
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
    the tickets are what code built."""
    lines = []
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
    each pick's chance, price and book and why, in a few words (first_reason: not the chance again)."""
    picks = [pick for pick in card.get("picks") or [] if isinstance(pick, dict)]
    if not picks:
        return "No game in scope has prices loaded right now: try another sport."
    lines = ["The most likely winners, strongest first:"]
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
