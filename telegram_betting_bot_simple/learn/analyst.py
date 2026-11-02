"""Whichever AI is linked, as the learning loop's analyst (method.md section 6): it reads the evaluator's fact sheets,
writes short notes on the brain's misses, proposes bounded changes, explains the week's decisions in plain words and
writes a monthly memo. It never forecasts a game (old games' results may be in its training data), never sets a
value that ships (the tuner sizes every change by the LCB rule within the proposed direction and bound, and judges
it only on games played after it was made), and never writes a number the evaluator didn't give it.

Provider, chosen at run time (``choose``; LEARN_AI): ``auto`` takes Claude when ANTHROPIC_API_KEY is set, else
DeepSeek when DEEPSEEK_API_KEY is, else none; ``claude`` and ``deepseek`` take only that one; ``off`` none. With none
the loop runs fully on template text written by code.

- Claude (ai/claude.py's ClaudeAnalyst: its SDK client, refusal fallbacks and pricing): structured outputs
  (``output_config.format`` with a JSON schema; no forced tool use), and the Message Batches API (half price) for
  the nightly, weekly and monthly calls when the client has it: a batch is submitted and its answer collected on a
  later tick (``collect``). Batches take no refusal fallbacks, so a batch request leaves them out.
- DeepSeek (ai/deepseek.py's DeepSeekAnalyst: its HTTP session and pricing): ``response_format`` json_object with
  the word "json" and an example in the prompt; no schema is enforced, so every answer is validated with pydantic.

Every answer is validated (pydantic) and every text in it goes through ai/explain.py's ``problems()`` against the
fact sheet's numbers, plus a status-word check: "promoted", "improved"/"better" and "rolled back" may appear only
when the gate's decisions in the fact sheet say so. A failed answer is asked again once with the reasons; after
that the loop's template text is used and the rejection is counted.

Spend is logged in ``analyst_calls`` under ``brain:claude`` / ``brain:deepseek`` (never the analysts' names, so it
never counts toward AI_DAILY_LIMIT, and never people's chat credits) and capped per calendar month by
LEARN_AI_MONTHLY_CENTS: a call whose worst-case cost would pass the cap is not made (templates instead).
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field, StringConstraints, ValidationError, field_validator

from ..ai import explain
from . import registry

log = logging.getLogger(__name__)

AI_MODES = ("auto", "claude", "deepseek", "off")
KINDS = ("note", "proposals", "changelog", "memo")
BRAIN_ANALYSTS = {"claude": "brain:claude", "deepseek": "brain:deepseek"}  # analyst_calls names
MONTHLY_CENTS_DEFAULT = 500
MAX_TOKENS = 8000  # output (thinking included) per call
MAX_INPUT_CHARS = 120_000  # about 40k tokens: the fact sheet is trimmed to fit (and the trim is noted)
CHARS_PER_TOKEN = 3.0  # a cautious estimate for the cap check (JSON fact sheets tokenize densely)
BATCH_DISCOUNT = 0.5  # the Message Batches API bills half of every token
EFFORT = {"note": "low", "proposals": "medium", "changelog": "low", "memo": "high"}
# Calls per answer, the retry included: at most 2 a night (the note), 3 a week (proposals and the change texts) and
# 1 a month (the memo), as method.md 6.5 sets them
MAX_CALLS = {"note": 2, "proposals": 2, "changelog": 1, "memo": 1}
MAX_TEXT = {"note": 1500, "hypothesis": 400, "proposal": 400, "changelog": 600, "memo": 4000, "idea": 300}

# --- the answers' shapes --------------------------------------------------------------------------------------------


FactId = Annotated[str, StringConstraints(pattern=r"^[a-z_]+:[A-Za-z0-9:._/-]{1,100}$")]  # "miss:espn:401", "trial:12"


class Hypothesis(BaseModel):
    text: str = Field(max_length=MAX_TEXT["hypothesis"])
    evidence: List[FactId] = Field(default_factory=list, max_length=8)


class NoteAnswer(BaseModel):
    note: str = Field(max_length=MAX_TEXT["note"])
    hypotheses: List[Hypothesis] = Field(default_factory=list, max_length=3)


class Proposal(BaseModel):
    """One bounded proposal. Its knob and sport must be the registry's and its evidence fact ids, so no free text of
    the AI's ever reaches the page through them (a refusal names none of them)."""

    id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9:._-]{1,80}$")]
    knob: str = Field(max_length=80)
    sport: str = Field(max_length=40)
    direction: Literal["increase", "decrease"]
    size: float
    hypothesis: str = Field(max_length=MAX_TEXT["proposal"])
    evidence: List[FactId] = Field(default_factory=list, max_length=8)
    test: str = Field(default="", max_length=MAX_TEXT["proposal"])
    expect: str = Field(default="", max_length=MAX_TEXT["proposal"])

    @field_validator("knob")
    @classmethod
    def _knob(cls, value: str) -> str:
        if value not in registry.KNOBS:
            raise ValueError("not one of the registry's knobs")
        return value

    @field_validator("sport")
    @classmethod
    def _sport(cls, value: str) -> str:
        if registry.sport_key(value) == "other" and str(value).strip().lower() != "other":
            raise ValueError("not a sport the app follows")
        return value


class ProposalsAnswer(BaseModel):
    proposals: List[Proposal] = Field(default_factory=list, max_length=6)
    none_reason: Optional[str] = Field(default=None, max_length=MAX_TEXT["proposal"])


class ChangeText(BaseModel):
    id: str = Field(max_length=40)
    text: str = Field(max_length=MAX_TEXT["changelog"])


class ChangelogAnswer(BaseModel):
    texts: List[ChangeText] = Field(default_factory=list, max_length=20)


class MemoAnswer(BaseModel):
    memo: str = Field(max_length=MAX_TEXT["memo"])
    data_ideas: List[str] = Field(default_factory=list, max_length=5)


MODELS = {"note": NoteAnswer, "proposals": ProposalsAnswer, "changelog": ChangelogAnswer, "memo": MemoAnswer}


def _schema(properties: Dict[str, Any], required: Sequence[str]) -> Dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


_STRINGS = {"type": "array", "items": {"type": "string"}}
SCHEMAS: Dict[str, Dict[str, Any]] = {
    "note": _schema({"note": {"type": "string"},
                     "hypotheses": {"type": "array", "items": _schema({"text": {"type": "string"},
                                                                       "evidence": _STRINGS}, ["text", "evidence"])}},
                    ["note", "hypotheses"]),
    "proposals": _schema({"proposals": {"type": "array", "items": _schema({
        "id": {"type": "string"}, "knob": {"type": "string"}, "sport": {"type": "string"},
        "direction": {"type": "string", "enum": ["increase", "decrease"]}, "size": {"type": "number"},
        "hypothesis": {"type": "string"}, "evidence": _STRINGS, "test": {"type": "string"},
        "expect": {"type": "string"}}, ["id", "knob", "sport", "direction", "size", "hypothesis", "evidence", "test",
                                        "expect"])},
        "none_reason": {"anyOf": [{"type": "string"}, {"type": "null"}]}}, ["proposals", "none_reason"]),
    "changelog": _schema({"texts": {"type": "array", "items": _schema({"id": {"type": "string"},
                                                                      "text": {"type": "string"}}, ["id", "text"])}},
                         ["texts"]),
    "memo": _schema({"memo": {"type": "string"}, "data_ideas": _STRINGS}, ["memo", "data_ideas"]),
}

_RULES = """\
Rules for every text you write:
- Use only numbers that appear in the facts you are given, rounded as they are or less. Never work out a new \
number (no sums, differences or percentages of your own) and never invent one.
- Never forecast or rate a game, and never give betting advice: no tips, no "value", no stakes.
- The words "promoted", "improved", "better" and "rolled back" describe the gate's decisions: use them only for a \
change the facts list with that decision.
- Write plain English for a sports fan, not a statistician. Short sentences.
"""

SYSTEMS: Dict[str, str] = {
    "note": """\
You review a sports prediction app's "brain" (a blend that leans mostly on the betting market) after a night of \
graded games. The facts (JSON) hold the 20 worst misses of the last 7 days by log loss, each with every source's \
chance, the time before the start it was logged and the result; the calibration slope and intercept per sport with \
95% ranges; drift flags; and last week's decisions.

Write a short note (at most 200 words) on patterns in the misses and their likely causes, and 0 to 3 hypotheses a \
later test could check, each citing the ids of the facts behind it ("miss:...", "calibration:...", "trial:...").
Most misses are just the game's uncertainty: say so when nothing stands out.

Answer with one json object, for example:
{"note": "Two or three sentences.", "hypotheses": [{"text": "One sentence.", "evidence": ["miss:espn:401"]}]}
""" + _RULES,
    "proposals": """\
You propose at most 3 small changes to a sports prediction app's "brain" settings for its weekly test. The facts \
(JSON) hold this week's notes and hypotheses, the worst misses, the allowed settings ("registry": each knob's \
current value, bounds and largest step a week), the scores, and the trials ledger (what was already tried: do not \
propose those again).

Each proposal names a knob from the registry, a sport, a direction (increase or decrease) and a size no larger \
than that knob's step. You never set the final value: the app sizes the change from the data and judges it only \
on games played after today, so a proposal that doesn't help simply fails. Cite the ids of the facts behind each \
one. If nothing in the facts supports a change, return no proposals and say why in none_reason.

Answer with one json object, for example:
{"proposals": [{"id": "2026-11-02-nba-1", "knob": "consensus.near_start_seconds", "sport": "nba", \
"direction": "increase", "size": 300, "hypothesis": "One sentence.", "evidence": ["miss:espn:401"], \
"test": "paired log loss on near-start games", "expect": "lower log loss near the start"}], "none_reason": null}
""" + _RULES,
    "changelog": """\
You write the "What changed and why" lines of a sports prediction app's brain for this week. The facts (JSON) list \
each decision the gate made (its id, the setting, old and new values, the number of games, the change in log loss \
with its 95% range, the decision and the gate's reason). For each id, write one or two plain sentences: what was \
tested, what the gate decided and why. Most weeks nothing changes; say that plainly.

Answer with one json object, for example:
{"texts": [{"id": "12", "text": "One or two sentences."}]}
""" + _RULES,
    "memo": """\
You write a monthly memo for the owner of a sports prediction app on how its brain's self-tuning went: the month's \
scores against the market, the candidates tried and what passed, how the AI's own proposals fared, what the AI \
cost, and up to 5 ideas for new data sources or targets worth testing (ideas only: nothing changes on its own).

Answer with one json object, for example:
{"memo": "A few short paragraphs.", "data_ideas": ["One idea."]}
""" + _RULES,
}

# Status words only the gate may use (method.md 6.4): which decisions in the fact sheet allow each
STATUS_WORDS: Sequence[Tuple[str, "re.Pattern[str]"]] = (
    ("promoted", re.compile(r"\bpromot(?:e|ed|es|ing|ion|ions)\b", re.I)),
    ("improved", re.compile(r"\bimprov(?:e|ed|es|ing|ement|ements)\b", re.I)),
    ("better", re.compile(r"\b(?:is|was|got|gets|getting|became|become|becomes|now|are|were|been)\s+(?:\w+\s+)?"
                          r"better\b", re.I)),
    ("rolled back", re.compile(r"\broll(?:ed|s|ing)?[- ]?back\b|\brollbacks?\b|\breverted\b", re.I)),
)
ALLOWED_BY = {"promoted": {"promoted"}, "improved": {"improved"}, "better": {"improved"},
              "rolled back": {"rolled_back", "undone"}}


def gate_statuses(facts: Any) -> Set[str]:
    """The decisions the gate made in a fact sheet (every "decision" value in it), plus "improved" when a promoted
    change's 95% range of the change in log loss lies entirely below 0 (the gate showed it better)."""
    found: Set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            decision = value.get("decision")
            if isinstance(decision, str):
                found.add(decision)
                hi = value.get("hi")
                if decision == "promoted" and isinstance(hi, (int, float)) and hi < 0:
                    found.add("improved")
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(facts)
    return found


def text_problems(text: str, facts: Any, statuses: Optional[Iterable[str]] = None) -> List[str]:
    """Why an AI text can't be shown (empty when it can): ai/explain.py's numbers and banned-wording check against
    the fact sheet, and status words the gate didn't give."""
    found = explain.problems(text or "", explain.source_numbers(facts))
    allowed = set(gate_statuses(facts) if statuses is None else statuses)
    words = [name for name, pattern in STATUS_WORDS if pattern.search(text or "") and not ALLOWED_BY[name] & allowed]
    if words:
        found.append("status words the gate didn't give: " + ", ".join(words))
    return found


def answer_texts(kind: str, data: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """(where, text) for every text in a validated answer."""
    if kind == "note":
        return [("note", data.get("note") or "")] + [(f"hypothesis {i + 1}", h.get("text") or "")
                                                     for i, h in enumerate(data.get("hypotheses") or [])]
    if kind == "proposals":
        out = []
        for p in data.get("proposals") or []:
            for name in ("hypothesis", "test", "expect"):
                out.append((f"proposal {p.get('id')} {name}", p.get(name) or ""))
        if data.get("none_reason"):
            out.append(("none_reason", data["none_reason"]))
        return out
    if kind == "changelog":
        return [(f"change {t.get('id')}", t.get("text") or "") for t in data.get("texts") or []]
    if kind == "memo":
        return [("memo", data.get("memo") or "")] + [(f"idea {i + 1}", idea)
                                                     for i, idea in enumerate(data.get("data_ideas") or [])]
    return []


def validate(kind: str, data: Any) -> Dict[str, Any]:
    """The answer as a plain dict when it matches its shape; ValueError (with the fields at fault) otherwise."""
    try:
        return MODELS[kind].model_validate(data).model_dump()
    except ValidationError as exc:
        fields = ", ".join(sorted({".".join(str(p) for p in error["loc"]) for error in exc.errors()}))
        raise ValueError(f"the answer didn't match the {kind} format ({fields or 'shape'})") from None


# --- provider selection ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Choice:
    """Which AI the loop uses this run: ``provider`` "claude", "deepseek" or None (templates), with why."""

    mode: str
    provider: Optional[str]
    model: Optional[str]
    reason: str


def choose(mode: Any, anthropic_key: Any = None, deepseek_key: Any = None, *, claude_model: str = "claude-opus-5-5",
           deepseek_model: str = "deepseek-v4-pro") -> Choice:
    """LEARN_AI against the keys that are set (at run time: a key added or removed changes the next run)."""
    mode = str(mode or "auto").strip().lower()
    if mode not in AI_MODES:
        mode = "auto"
    has_claude, has_deepseek = bool(anthropic_key), bool(deepseek_key)
    if mode == "off":
        return Choice(mode, None, None, "AI notes: off (LEARN_AI=off)")
    if mode in ("auto", "claude") and has_claude:
        return Choice(mode, "claude", claude_model, "Claude (its key is set)" if mode == "auto" else "Claude")
    if mode in ("auto", "deepseek") and has_deepseek:
        return Choice(mode, "deepseek", deepseek_model,
                      "DeepSeek (no Claude key)" if mode == "auto" else "DeepSeek")
    if mode == "claude":
        return Choice(mode, None, None, "AI notes: off (LEARN_AI=claude but ANTHROPIC_API_KEY isn't set)")
    if mode == "deepseek":
        return Choice(mode, None, None, "AI notes: off (LEARN_AI=deepseek but DEEPSEEK_API_KEY isn't set)")
    return Choice(mode, None, None, "AI notes: off (no key)")


# --- the providers --------------------------------------------------------------------------------------------------

class ProviderError(Exception):
    """A call that failed; ``cost`` is what it still cost (US dollars)."""

    def __init__(self, message: str, cost: float = 0.0) -> None:
        super().__init__(message)
        self.cost = cost


@dataclass
class Raw:
    """One answer: its text (JSON), the model that wrote it, what it cost (US dollars) and its tokens."""

    text: str
    model: str
    cost: float
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None  # set when the call returned nothing usable


def _price(table: Mapping[str, Tuple[float, ...]], model: str, fallback: Tuple[float, ...]) -> Tuple[float, ...]:
    return table.get(model, fallback)


class ClaudeBrain:
    """Claude through ai/claude.py's ClaudeAnalyst (the same SDK client, refusal fallbacks and pricing as the app's
    analysts). ``batches``: use the Message Batches API when the client has it."""

    name = "claude"

    def __init__(self, analyst: Any, *, batches: bool = True) -> None:
        self.analyst = analyst
        self.model = getattr(analyst, "model", "claude-opus-5-5")
        client = getattr(analyst, "client", None)
        self.batches = bool(batches) and getattr(getattr(client, "messages", None), "batches", None) is not None

    def _request(self, kind: str, user: str) -> Dict[str, Any]:
        return {"max_tokens": MAX_TOKENS,
                "system": [{"type": "text", "text": SYSTEMS[kind], "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": user}],
                "output_config": {"effort": EFFORT[kind], "format": {"type": "json_schema", "schema": SCHEMAS[kind]}}}

    def estimate(self, chars: int, batch: bool) -> float:
        from ..ai.claude import PRICES, UNKNOWN_PRICE
        price_in, price_out, _ = _price(PRICES, self.model, UNKNOWN_PRICE)
        dollars = (chars / CHARS_PER_TOKEN * price_in + MAX_TOKENS * price_out) / 1e6
        return dollars * (BATCH_DISCOUNT if batch else 1.0)

    @staticmethod
    def _text(message: Any) -> str:
        return "".join(getattr(block, "text", "") for block in getattr(message, "content", None) or []
                       if getattr(block, "type", None) == "text")

    def call(self, kind: str, user: str) -> Raw:
        from ..ai.forecast import AnalystError
        try:
            response = self.analyst._create(**self._request(kind, user))
        except AnalystError as exc:
            raise ProviderError(str(exc), getattr(exc, "cost", 0.0)) from None
        cost, tokens_in, tokens_out = self.analyst._cost(response.usage, getattr(response, "model", None))
        model = str(getattr(response, "model", None) or self.model)
        if response.stop_reason in ("refusal", "max_tokens"):
            return Raw("", model, cost, tokens_in, tokens_out,
                       error="Claude declined" if response.stop_reason == "refusal" else "the answer was cut off")
        return Raw(self._text(response), model, cost, tokens_in, tokens_out)

    def submit(self, jobs: Sequence[Tuple[str, str, str]]) -> str:
        """Send (custom_id, kind, user) requests as one batch (no refusal fallbacks: Batches refuse them); returns
        the batch id."""
        requests = [{"custom_id": custom_id, "params": {"model": self.model, **self._request(kind, user)}}
                    for custom_id, kind, user in jobs]
        batch = self.analyst.client.messages.batches.create(requests=requests)
        return str(batch.id)

    def collect(self, batch_id: str) -> Optional[Dict[str, Raw]]:
        """{custom_id: Raw} once the batch has ended (None while it is still running)."""
        from ..ai.claude import usage_cost
        batches = self.analyst.client.messages.batches
        batch = batches.retrieve(batch_id)
        if getattr(batch, "processing_status", None) != "ended":
            return None
        out: Dict[str, Raw] = {}
        for item in batches.results(batch_id):
            result = item.result
            if getattr(result, "type", None) != "succeeded":
                out[str(item.custom_id)] = Raw("", self.model, 0.0, error=f"the batch request {result.type}")
                continue
            message = result.message
            dollars, tokens_in, tokens_out = usage_cost(self.model, message.usage, getattr(message, "model", None))
            model = str(getattr(message, "model", None) or self.model)
            if getattr(message, "stop_reason", None) in ("refusal", "max_tokens"):
                out[str(item.custom_id)] = Raw("", model, dollars * BATCH_DISCOUNT, tokens_in, tokens_out,
                                               error="no usable answer")
                continue
            out[str(item.custom_id)] = Raw(self._text(message), model, dollars * BATCH_DISCOUNT, tokens_in, tokens_out)
        return out


class DeepSeekBrain:
    """DeepSeek through ai/deepseek.py's DeepSeekAnalyst (its HTTP session and pricing): JSON output mode, an empty
    answer asked once more without thinking, as the app's analyst does."""

    name = "deepseek"
    batches = False

    def __init__(self, analyst: Any) -> None:
        self.analyst = analyst
        self.model = getattr(analyst, "model", "deepseek-v4-pro")

    def estimate(self, chars: int, batch: bool = False) -> float:
        from ..ai.deepseek import DEFAULT_MODEL, PRICES
        _, price_in, price_out = _price(PRICES, self.model, PRICES[DEFAULT_MODEL])
        return (chars / CHARS_PER_TOKEN * price_in + MAX_TOKENS * price_out) / 1e6

    def call(self, kind: str, user: str) -> Raw:
        from ..ai.forecast import AnalystError
        body: Dict[str, Any] = {"model": self.model,
                                "messages": [{"role": "system", "content": SYSTEMS[kind]},
                                             {"role": "user", "content": user}],
                                "response_format": {"type": "json_object"}, "max_tokens": MAX_TOKENS}
        try:
            payload = self.analyst._post(body, 0.0)
        except AnalystError as exc:
            raise ProviderError(str(exc), getattr(exc, "cost", 0.0)) from None
        cost, tokens_in, tokens_out = self.analyst._cost(payload.get("usage") or {})
        choice = (payload.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        model = str(payload.get("model") or self.model)
        if not text or choice.get("finish_reason") == "length":
            return Raw("", model, cost, tokens_in, tokens_out, error="DeepSeek returned an empty answer")
        return Raw(text, model, cost, tokens_in, tokens_out)


# --- the analyst ----------------------------------------------------------------------------------------------------

@dataclass
class Answer:
    """What one ask gave: ``status`` "done" (``data`` validated, texts checked), "pending" (a batch is out:
    ``batch``), "failed" (templates; ``reason``), "skipped" (no provider, paused or the cap; ``reason``).
    ``rejected`` lists texts dropped by the checks (their places get the template)."""

    kind: str
    status: str
    data: Optional[Dict[str, Any]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    cost: float = 0.0
    calls: int = 0
    reason: str = ""
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    batch: Optional[Dict[str, Any]] = None
    trimmed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "status": self.status, "data": self.data, "provider": self.provider,
                "model": self.model, "cost": round(self.cost, 6), "calls": self.calls, "reason": self.reason,
                "rejected": self.rejected, "batch": self.batch, "trimmed": self.trimmed}


def _dumps(facts: Any) -> str:
    return json.dumps(facts, sort_keys=True, separators=(",", ":"), default=str)


def trim(facts: Mapping[str, Any], limit: int = MAX_INPUT_CHARS) -> Tuple[Dict[str, Any], bool]:
    """The fact sheet within ``limit`` characters: its longest lists are shortened from the end (worst misses and
    ledger rows are listed most important first) until it fits. (sheet, whether anything was cut)."""
    sheet = json.loads(_dumps(facts))
    cut = False
    while len(_dumps(sheet)) > limit:
        lists = [(key, value) for key, value in sheet.items() if isinstance(value, list) and len(value) > 1]
        if not lists:
            break
        key, value = max(lists, key=lambda kv: len(_dumps(kv[1])))
        sheet[key] = value[:max(1, len(value) * 3 // 4)]
        cut = True
    return sheet, cut


def month_start(now: float, tz: Optional[datetime.tzinfo] = None) -> float:
    """The start of ``now``'s calendar month (server time, or ``tz``)."""
    local = datetime.datetime.fromtimestamp(now, tz) if tz is not None else datetime.datetime.fromtimestamp(now)
    first = local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return first.timestamp()


class BrainAnalyst:
    """The loop's AI. ``store`` is the data/store.Store (analyst_calls). ``mode`` is LEARN_AI; ``keys`` gives the
    API keys and models at run time (a callable returning {"anthropic_api_key", "deepseek_api_key",
    "anthropic_model", "deepseek_model"}, or that dict); ``factory(choice)`` builds the provider for a choice
    (default: ai/claude.py's ClaudeAnalyst or ai/deepseek.py's DeepSeekAnalyst, wrapped); ``monthly_cents`` is
    LEARN_AI_MONTHLY_CENTS; ``batches`` lets Claude use the Message Batches API."""

    def __init__(self, store: Any, *, mode: str = "auto", keys: Any = None, monthly_cents: int = MONTHLY_CENTS_DEFAULT,
                 factory: Optional[Callable[[Choice], Any]] = None, batches: bool = True,
                 clock: Callable[[], float] = time.time, tz: Optional[datetime.tzinfo] = None) -> None:
        self.store = store
        self.mode = mode
        self.keys = keys if keys is not None else {}
        self.monthly_cents = max(0, int(monthly_cents))
        self.factory = factory or self._default_factory
        self.use_batches = batches
        self.clock = clock
        self.tz = tz
        self._built: Dict[Tuple[Optional[str], Optional[str]], Any] = {}

    # -- which provider --

    def _keys(self) -> Mapping[str, Any]:
        keys = self.keys() if callable(self.keys) else self.keys
        return keys or {}

    def choice(self) -> Choice:
        keys = self._keys()
        return choose(self.mode, keys.get("anthropic_api_key"), keys.get("deepseek_api_key"),
                      claude_model=keys.get("anthropic_model") or "claude-opus-5-5",
                      deepseek_model=keys.get("deepseek_model") or "deepseek-v4-pro")

    def _default_factory(self, choice: Choice) -> Any:
        keys = self._keys()
        if choice.provider == "claude":
            from ..ai.claude import ClaudeAnalyst  # the SDK is only needed with a key
            return ClaudeBrain(ClaudeAnalyst(keys["anthropic_api_key"], choice.model or "claude-opus-5-5", "low"),
                               batches=self.use_batches)
        if choice.provider == "deepseek":
            from ..ai.deepseek import DeepSeekAnalyst
            return DeepSeekBrain(DeepSeekAnalyst(keys["deepseek_api_key"], choice.model or "deepseek-v4-pro"))
        return None

    def provider(self) -> Tuple[Choice, Any]:
        """(this run's choice, its provider or None)."""
        choice = self.choice()
        if choice.provider is None:
            return choice, None
        key = (choice.provider, choice.model)
        if key not in self._built:
            try:
                self._built[key] = self.factory(choice)
            except Exception as exc:  # noqa: BLE001 - a broken client means templates, never a crash
                log.warning("The learning loop's %s client couldn't start: %s", choice.provider, type(exc).__name__)
                self._built[key] = None
        return choice, self._built[key]

    # -- money --

    def spent(self, now: Optional[float] = None) -> float:
        """US dollars the loop's AI has cost this calendar month (both providers)."""
        now = self.clock() if now is None else now
        since = month_start(now, self.tz)
        total = 0.0
        for name in BRAIN_ANALYSTS.values():
            try:
                total += float(self.store.analyst_spend(name, since)[1])
            except Exception:  # noqa: BLE001 - a store without the table: nothing spent
                continue
        return total

    def status(self, now: Optional[float] = None, reserved: float = 0.0) -> Dict[str, Any]:
        """For the page and the tool: {mode, provider, model, reason, spent_cents, reserved_cents, cap_cents}."""
        choice = self.choice()
        return {"mode": choice.mode, "provider": choice.provider, "model": choice.model, "reason": choice.reason,
                "spent_cents": round(self.spent(now) * 100, 2), "reserved_cents": round(reserved * 100, 2),
                "cap_cents": self.monthly_cents}

    def _log(self, provider: str, raw: Raw, kind: str, ok: bool, now: float, job: str) -> None:
        try:
            self.store.log_analyst_call(BRAIN_ANALYSTS[provider], raw.model, f"{kind}:{job}", int(raw.tokens_in),
                                        int(raw.tokens_out), float(raw.cost), ok, now=now)
        except TypeError:  # a store whose log takes no clock
            self.store.log_analyst_call(BRAIN_ANALYSTS[provider], raw.model, f"{kind}:{job}", int(raw.tokens_in),
                                        int(raw.tokens_out), float(raw.cost), ok)

    def affordable(self, estimate: float, now: float, reserved: float = 0.0) -> bool:
        """Whether a call whose worst case costs ``estimate`` (dollars) stays within the monthly cap."""
        return (self.spent(now) + reserved + estimate) * 100 <= self.monthly_cents + 1e-9

    # -- asking --

    def ask(self, kind: str, facts: Mapping[str, Any], *, now: Optional[float] = None, reserved: float = 0.0,
            batch: bool = False, job: Optional[str] = None, max_calls: Optional[int] = None) -> Answer:
        """Ask the linked AI for a ``kind`` answer over ``facts`` (built by the evaluator). Synchronous unless
        ``batch`` and the provider has batches: then the request goes out as a batch and the answer is "pending"
        (``batch`` holds what ``finish`` needs). ``reserved``: dollars already promised to batches still out."""
        if kind not in KINDS:
            raise ValueError(f"Unknown kind {kind!r}")
        max_calls = MAX_CALLS[kind] if max_calls is None else max_calls
        now = self.clock() if now is None else now
        job = job or uuid.uuid4().hex[:12]
        choice, provider = self.provider()
        if choice.provider is None:
            return Answer(kind, "skipped", reason=choice.reason)
        if provider is None:
            return Answer(kind, "skipped", provider=choice.provider, reason=f"{choice.provider} couldn't start")
        sheet, trimmed = trim(facts)
        user = "Facts (json):\n" + _dumps(sheet)
        use_batch = bool(batch and getattr(provider, "batches", False))
        estimate = provider.estimate(len(user) + len(SYSTEMS[kind]), use_batch)
        if not self.affordable(estimate, now, reserved):
            return Answer(kind, "skipped", provider=choice.provider, model=choice.model, trimmed=trimmed,
                          reason=f"the monthly cap of {self.monthly_cents} cents is reached")
        if use_batch:
            try:
                batch_id = provider.submit([(job, kind, user)])
                return Answer(kind, "pending", provider=choice.provider, model=provider.model, trimmed=trimmed,
                              batch={"id": batch_id, "job": job, "estimate": estimate, "submitted_at": now,
                                     "provider": choice.provider, "user": user, "trimmed": trimmed})
            except Exception as exc:  # noqa: BLE001 - no batch today: one plain call instead
                log.warning("The learning loop's batch couldn't be sent (%s); asking directly", type(exc).__name__)
        return self._run(kind, sheet, user, provider, now, job, max_calls, trimmed, reserved=reserved)

    def _run(self, kind: str, sheet: Mapping[str, Any], user: str, provider: Any, now: float, job: str,
             max_calls: int, trimmed: bool, first: Optional[Raw] = None, reserved: float = 0.0) -> Answer:
        """Ask (or take ``first``, a batch's answer, as the first call) and check; one more call with the reasons
        when the answer can't be used, within ``max_calls`` calls and the cap (``reserved``: dollars promised to
        other batches still out, which will be billed too)."""
        answer = Answer(kind, "failed", provider=provider.name, model=provider.model, trimmed=trimmed,
                        calls=1 if first is not None else 0)
        prompt, raw = user, first
        while True:
            if raw is None:
                if answer.calls >= max_calls:
                    break
                if answer.calls and not self.affordable(provider.estimate(len(prompt) + len(SYSTEMS[kind]), False),
                                                        now, reserved):
                    answer.reason = f"the monthly cap of {self.monthly_cents} cents is reached"
                    break
                try:
                    raw = provider.call(kind, prompt)
                except ProviderError as exc:  # the key, the credit or an outage: asking again tonight won't help
                    self._log(provider.name, Raw("", provider.model, exc.cost, error=str(exc)), kind, False, now, job)
                    answer.cost += exc.cost
                    answer.calls += 1
                    answer.reason = str(exc)
                    break
                answer.calls += 1
            answer.cost += raw.cost
            answer.model = raw.model
            problems, data, rejected = check_answer(kind, raw, sheet)
            self._log(provider.name, raw, kind, not problems, now, job)
            if not problems:
                answer.status, answer.data, answer.rejected, answer.reason = "done", data, rejected, ""
                return answer
            answer.reason = "; ".join(problems)[:500]
            answer.rejected = rejected
            prompt = (user + "\n\nYour previous answer couldn't be used: " + answer.reason
                      + ". Answer again with one json object in the format asked, following every rule.")
            raw = None
        return answer

    def finish(self, pending: Mapping[str, Any], sheet: Mapping[str, Any], *, now: Optional[float] = None,
               kind: Optional[str] = None, reserved: float = 0.0, retry: bool = True) -> Optional[Answer]:
        """Collect a batch answer (``pending`` is an Answer's ``batch``): None while the batch runs, else the
        checked Answer (one plain call again when the batch's answer fails the checks, unless ``retry`` is off:
        learning paused). ``reserved``: dollars promised to the other batches still out, counted against the cap
        before that call."""
        now = self.clock() if now is None else now
        choice, provider = self.provider()
        kind = kind or str(pending.get("kind") or "note")
        if provider is None or provider.name != pending.get("provider") or not hasattr(provider, "collect"):
            return Answer(kind, "failed", provider=pending.get("provider"),
                          reason="the AI that took the batch isn't linked any more")
        try:
            results = provider.collect(str(pending["id"]))
        except Exception as exc:  # noqa: BLE001
            log.warning("Couldn't read the learning loop's batch: %s", type(exc).__name__)
            return None
        if results is None:
            return None
        raw = results.get(str(pending["job"])) or Raw("", provider.model, 0.0, error="the batch had no answer")
        return self._run(kind, sheet, str(pending.get("user") or ""), provider, now, str(pending["job"]),
                         max_calls=MAX_CALLS.get(kind, 1) if retry else 1, trimmed=bool(pending.get("trimmed")),
                         first=raw, reserved=reserved)


def has_content(kind: str, data: Optional[Mapping[str, Any]]) -> bool:
    """Whether a checked answer still says something (its main text, or for proposals a proposal or a reason)."""
    data = data or {}
    if kind == "proposals":
        return bool(data.get("proposals")) or bool(data.get("none_reason"))
    if kind == "changelog":
        return True  # nothing to add is an answer (the template lines stay)
    return bool(data.get("memo" if kind == "memo" else "note"))


def check_answer(kind: str, raw: Raw, sheet: Mapping[str, Any]
                 ) -> Tuple[List[str], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """(problems that call for asking again, the answer with failing texts removed, the texts removed with why).
    An answer is usable when it parses, matches its shape and still says something once the texts that failed
    the numbers and status-word checks are taken out."""
    if raw.error:
        return [raw.error], None, []
    try:
        data = validate(kind, json.loads(raw.text))
    except json.JSONDecodeError:
        return ["the answer wasn't valid json"], None, []
    except (TypeError, ValueError) as exc:
        return [str(exc)], None, []
    if kind == "changelog":
        rejected = changelog_problems(data, sheet)
    else:
        statuses = gate_statuses(sheet)
        rejected = [{"where": where, "problems": found} for where, text in answer_texts(kind, data)
                    if text and (found := text_problems(text, sheet, statuses))]
    kept = _drop_rejected(kind, data, {r["where"] for r in rejected})
    if not has_content(kind, kept):
        detail = [f"{r['where']}: {'; '.join(r['problems'])}" for r in rejected[:3]] or ["the answer was empty"]
        return detail, kept, rejected
    return [], kept, rejected


def changelog_problems(data: Mapping[str, Any], sheet: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The change texts that can't be shown, each checked against its own change only: its numbers must be that
    change's (or the sheet's shared fields'), and a status word needs that change's decision (one change promoted
    that week never lets another change's text say "promoted" or "improved"). A text for an id not in the sheet is
    dropped."""
    changes = {str(c.get("id") or "").split(":", 1)[-1]: c for c in sheet.get("changes") or []
               if isinstance(c, Mapping)}
    shared = {k: v for k, v in sheet.items() if k != "changes"}
    out = []
    for item in data.get("texts") or []:
        where = f"change {item.get('id')}"
        text = item.get("text") or ""
        change = changes.get(str(item.get("id") or "").split(":", 1)[-1])
        if change is None:
            out.append({"where": where, "problems": ["no change with that id in the facts"]})
            continue
        found = text_problems(text, {**shared, "change": change}, gate_statuses(change)) if text else []
        if found:
            out.append({"where": where, "problems": found})
    return out


def _drop_rejected(kind: str, data: Dict[str, Any], where: Set[str]) -> Dict[str, Any]:
    """The answer without the texts that failed the checks (a proposal whose text failed is dropped whole)."""
    if not where:
        return data
    data = json.loads(json.dumps(data))
    if kind == "note":
        data["hypotheses"] = [h for i, h in enumerate(data.get("hypotheses") or []) if f"hypothesis {i + 1}" not in where]
        if "note" in where:
            data["note"] = ""
    elif kind == "proposals":
        data["proposals"] = [p for p in data.get("proposals") or []
                             if not any(f"proposal {p.get('id')} {name}" in where for name in ("hypothesis", "test",
                                                                                               "expect"))]
        if "none_reason" in where:
            data["none_reason"] = None
    elif kind == "changelog":
        data["texts"] = [t for t in data.get("texts") or [] if f"change {t.get('id')}" not in where]
    elif kind == "memo":
        if "memo" in where:
            data["memo"] = ""
        data["data_ideas"] = [idea for i, idea in enumerate(data.get("data_ideas") or []) if f"idea {i + 1}" not in where]
    return data


def known_evidence(facts: Any) -> List[str]:
    """Every fact id in a fact sheet (the "id" values), which an AI proposal's evidence must come from."""
    found: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            ident = value.get("id")
            if isinstance(ident, str) and ":" in ident:
                found.append(ident)
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(facts)
    return list(dict.fromkeys(found))


def finite(value: Any, digits: int = 4) -> Optional[float]:
    """A number for a fact sheet: rounded, None when missing or not finite."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


__all__ = ["AI_MODES", "Answer", "BrainAnalyst", "Choice", "ClaudeBrain", "DeepSeekBrain", "KINDS", "Raw", "SCHEMAS",
           "SYSTEMS", "answer_texts", "changelog_problems", "choose", "finite", "gate_statuses", "known_evidence",
           "text_problems", "trim", "validate"]
