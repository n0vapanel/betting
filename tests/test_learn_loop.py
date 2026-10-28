"""The learning loop's scheduler, AI and wiring (v3, Contract 3): the learner on a fake clock with its store lease,
whichever AI is linked (Claude, DeepSeek or none) with validation, retry, the numbers check and the monthly cap, the
owner-only routes, Undo and Pause, the weekly Telegram summary, the grader's capture and the goals-model markets it
now settles, and the store's migration and the registry's wiring. No network: fake clients and fixtures only."""

from __future__ import annotations

import datetime
import json
import sqlite3
import types
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.ai.claude import ClaudeAnalyst
from telegram_betting_bot_simple.ai.deepseek import DeepSeekAnalyst
from telegram_betting_bot_simple.config import Config
from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.dashboard.lab import Lab, _outcome
from telegram_betting_bot_simple.dashboard.service import DashboardService, derived_outcome
from telegram_betting_bot_simple.data import slate as slate_mod
from telegram_betting_bot_simple.data.store import MIGRATIONS, SNAPSHOT_SUFFIX, Store, migrate
from telegram_betting_bot_simple.engine import consensus, tickets
from telegram_betting_bot_simple.learn import analyst as an
from telegram_betting_bot_simple.learn import brain_store as bs
from telegram_betting_bot_simple.learn import loop, progress, registry, tune
from tests import learn_sim as sim
from tests.test_auth import BOT_TOKEN, make_init_data

UTC = ZoneInfo("UTC")
NY = ZoneInfo("America/New_York")
MONDAY = datetime.datetime(2026, 10, 26, tzinfo=UTC)  # a Monday
FIXTURES = Path(__file__).parent / "fixtures" / "learn"


def ts(day: datetime.datetime, hours: float = 0.0) -> float:
    return day.timestamp() + hours * 3600


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def store(tmp_path) -> Store:
    return Store(str(tmp_path / "betting.db"))


@pytest.fixture(autouse=True)
def unbound_registry():
    registry.bind(None)
    yield
    registry.bind(None)


# --- fake AI clients (the real ClaudeAnalyst and DeepSeekAnalyst wrap them) ----------------------------------------

class Usage:
    def __init__(self, tokens_in: int = 20_000, tokens_out: int = 3_000) -> None:
        self.input_tokens, self.output_tokens = tokens_in, tokens_out
        self.cache_creation_input_tokens = self.cache_read_input_tokens = 0
        self.iterations = None
        self.server_tool_use = None


class Message:
    def __init__(self, text: str, stop: str = "end_turn", model: str = "claude-opus-5-5") -> None:
        self.content = [types.SimpleNamespace(type="text", text=text)]
        self.usage, self.model, self.stop_reason = Usage(), model, stop


class FakeBatches:
    def __init__(self) -> None:
        self.created: List[List[Dict[str, Any]]] = []
        self.answers: Dict[str, str] = {}
        self.status = "in_progress"

    def create(self, requests):
        self.created.append(list(requests))
        return types.SimpleNamespace(id=f"batch_{len(self.created)}")

    def retrieve(self, batch_id):
        return types.SimpleNamespace(processing_status=self.status)

    def results(self, batch_id):
        requests = self.created[int(batch_id.split("_")[1]) - 1]
        return [types.SimpleNamespace(custom_id=r["custom_id"], result=types.SimpleNamespace(
            type="succeeded", message=Message(self.answers.get(r["custom_id"], "{}")))) for r in requests]


class FakeAnthropic:
    """Stands in for anthropic.Anthropic: beta.messages.create (plain calls) and messages.batches."""

    def __init__(self, answers: List[str], batches: bool = False) -> None:
        self.answers = list(answers)
        self.requests: List[Dict[str, Any]] = []
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self._create))
        self.messages = types.SimpleNamespace(batches=FakeBatches()) if batches else types.SimpleNamespace()

    def _create(self, **request):
        self.requests.append(request)
        return Message(self.answers.pop(0))


class FakeResponse:
    status_code = 200

    def __init__(self, body: Dict[str, Any]) -> None:
        self.body = body

    def json(self) -> Dict[str, Any]:
        return self.body


class FakeDeepSeekSession:
    def __init__(self, answers: List[str]) -> None:
        self.answers = list(answers)
        self.bodies: List[Dict[str, Any]] = []

    def post(self, url, json=None, timeout=None, headers=None):
        self.bodies.append(json)
        return FakeResponse({"choices": [{"message": {"content": self.answers.pop(0)}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 10_000, "completion_tokens": 2_000}, "model": "deepseek-v4-pro"})


def claude_brain(store: Store, answers: List[str], *, batches: bool = False, cents: int = 500,
                 clock: Optional[Clock] = None) -> Any:
    client = FakeAnthropic(answers, batches=batches)
    brain = an.BrainAnalyst(store, mode="auto", keys={"anthropic_api_key": "sk-ant-test"}, monthly_cents=cents,
                            factory=lambda choice: an.ClaudeBrain(ClaudeAnalyst("sk-ant-test", client=client),
                                                                  batches=batches),
                            clock=clock or Clock(ts(MONDAY, 10)), tz=UTC)
    brain.client = client
    return brain


def deepseek_brain(store: Store, answers: List[str], cents: int = 500) -> Any:
    session = FakeDeepSeekSession(answers)
    brain = an.BrainAnalyst(store, mode="auto", keys={"deepseek_api_key": "sk-test"}, monthly_cents=cents,
                            factory=lambda choice: an.DeepSeekBrain(DeepSeekAnalyst("sk-test", session=session)),
                            clock=Clock(ts(MONDAY, 10)), tz=UTC)
    brain.session = session
    return brain


FACTS = {"misses": [{"id": "miss:espn:1", "brain_ll": 1.2371, "market_ll": 1.1899}], "changes": []}
GOOD_NOTE = json.dumps({"note": "The biggest miss had a log loss of 1.2371, against the market's 1.1899.",
                        "hypotheses": [{"text": "Nothing stands out yet.", "evidence": ["miss:espn:1"]}]})


# --- provider selection ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("keys, expected", [
    ({}, {"auto": None, "claude": None, "deepseek": None, "off": None}),
    ({"anthropic": "a"}, {"auto": "claude", "claude": "claude", "deepseek": None, "off": None}),
    ({"deepseek": "d"}, {"auto": "deepseek", "claude": None, "deepseek": "deepseek", "off": None}),
    ({"anthropic": "a", "deepseek": "d"}, {"auto": "claude", "claude": "claude", "deepseek": "deepseek", "off": None}),
])
def test_the_provider_is_claude_then_deepseek_then_templates_per_learn_ai(keys, expected) -> None:
    for mode, provider in expected.items():
        choice = an.choose(mode, keys.get("anthropic"), keys.get("deepseek"))
        assert choice.provider == provider, (mode, keys)
        if provider is None:
            assert choice.reason.startswith("AI notes: off")
    assert an.choose("auto").reason == "AI notes: off (no key)"
    assert an.choose("weird", "a").provider == "claude"  # an unknown setting is auto


def test_the_provider_is_chosen_at_run_time_from_the_keys_then(store) -> None:
    keys = {"anthropic_api_key": None, "deepseek_api_key": None}
    brain = an.BrainAnalyst(store, mode="auto", keys=lambda: keys, factory=lambda choice: None)
    assert brain.choice().provider is None
    keys["deepseek_api_key"] = "sk-test"  # added while the process runs: the next run uses it
    assert brain.choice().provider == "deepseek"
    keys["anthropic_api_key"] = "sk-ant-test"
    assert brain.choice().provider == "claude"
    assert an.BrainAnalyst(store, mode="off", keys=lambda: keys).ask("note", FACTS).status == "skipped"


# --- validation, retry, checks --------------------------------------------------------------------------------------

def test_claude_uses_structured_outputs_without_forced_tools_and_retries_once(store) -> None:
    brain = claude_brain(store, ["not json at all", GOOD_NOTE])
    answer = brain.ask("note", FACTS)
    assert answer.status == "done" and answer.calls == 2 and answer.provider == "claude"
    assert answer.data["note"].startswith("The biggest miss")
    first = brain.client.requests[0]
    assert first["output_config"]["format"]["type"] == "json_schema" and "tool_choice" not in first
    assert first["output_config"]["effort"] == "low" and first["fallbacks"] == "default"  # plain calls take fallbacks
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}  # the static prefix is cached
    assert "couldn't be used" in brain.client.requests[1]["messages"][0]["content"]  # the retry says why


def test_deepseek_answers_in_json_mode_and_is_validated_with_pydantic(store) -> None:
    wrong_shape = json.dumps({"note": 7})
    brain = deepseek_brain(store, [wrong_shape, GOOD_NOTE])
    answer = brain.ask("note", FACTS)
    assert answer.status == "done" and answer.calls == 2 and answer.provider == "deepseek"
    body = brain.session.bodies[0]
    assert body["response_format"] == {"type": "json_object"} and "json" in body["messages"][0]["content"]
    # Two bad answers: no third call, the template stays
    brain = deepseek_brain(store, [wrong_shape, "{"])
    answer = brain.ask("note", FACTS)
    assert answer.status == "failed" and answer.calls == 2 and "json" in answer.reason


def test_every_text_is_numbers_checked_and_status_words_need_the_gate(store) -> None:
    assert an.text_problems("The miss had a log loss of 1.2371.", FACTS) == []
    invented = an.text_problems("The brain was off by 7.5% on this game.", FACTS)
    assert invented and "numbers that are not in the facts" in invented[0]
    assert an.text_problems("The brain improved this week.", FACTS) == \
        ["status words the gate didn't give: improved"]
    assert an.text_problems("Blend weights were promoted.", FACTS)  # no promotion in the facts
    promoted = {"changes": [{"id": "trial:3", "decision": "promoted", "dll": -0.0012, "lo": -0.002, "hi": 0.0004}]}
    assert an.text_problems("Blend weights were promoted.", promoted) == []
    assert an.text_problems("The brain improved.", promoted)  # promoted as a nudge, not shown better
    better = {"changes": [{"id": "trial:3", "decision": "promoted", "dll": -0.0012, "lo": -0.002, "hi": -0.0004}]}
    assert an.text_problems("The brain is better at this now.", better) == []
    assert an.text_problems("It was rolled back.", {"changes": [{"decision": "rolled_back"}]}) == []
    assert an.text_problems("This is a lock, bet more.", FACTS)  # the app's banned wording too
    # An answer whose only text fails is asked again; the second, clean, is used
    bad = json.dumps({"note": "The brain improved by 12% this week.", "hypotheses": []})
    brain = claude_brain(store, [bad, GOOD_NOTE])
    answer = brain.ask("note", FACTS)
    assert answer.status == "done" and answer.calls == 2 and "improved" not in answer.data["note"]
    # A proposal whose text invents a number is dropped; the rest of the answer stands
    proposals = {"proposals": [
        {"id": "p1", "knob": "consensus.near_start_seconds", "sport": "nba", "direction": "increase", "size": 300,
         "hypothesis": "Late misses cluster near the start.", "evidence": ["miss:espn:1"], "test": "", "expect": ""},
        {"id": "p2", "knob": "soccer.rho", "sport": "soccer", "direction": "decrease", "size": 0.02,
         "hypothesis": "Draws run 4.2 points high.", "evidence": [], "test": "", "expect": ""}], "none_reason": None}
    answer = claude_brain(store, [json.dumps(proposals)]).ask("proposals", FACTS)
    assert answer.status == "done" and [p["id"] for p in answer.data["proposals"]] == ["p1"]
    assert answer.rejected and answer.rejected[0]["where"].startswith("proposal p2")


def test_a_proposal_carries_no_free_text_outside_its_checked_texts() -> None:
    """Evidence must be fact ids, the knob the registry's and the sport one the app follows: free text there fails
    validation (and is asked again) instead of reaching the page through a refusal."""
    good = {"id": "2026-11-02-nba-1", "knob": "consensus.weight.us_books", "sport": "nba", "direction": "increase",
            "size": 0.05, "hypothesis": "Late prices lag.", "evidence": ["miss:espn:401", "trial:12"], "test": "",
            "expect": ""}
    assert an.validate("proposals", {"proposals": [good], "none_reason": None})["proposals"][0]["sport"] == "nba"
    assert an.validate("proposals", {"proposals": [{**good, "sport": "basketball_nba"}], "none_reason": None})
    leak = "The brain improved 40% and was promoted: back the Lakers tonight"
    for bad in ({"evidence": [leak]}, {"knob": leak}, {"knob": "consensus.magic"}, {"sport": leak},
                {"id": leak}):
        with pytest.raises(ValueError) as caught:
            an.validate("proposals", {"proposals": [{**good, **bad}], "none_reason": None})
        assert "Lakers" not in str(caught.value)
    with pytest.raises(ValueError):
        an.validate("note", {"note": "x", "hypotheses": [{"text": "y", "evidence": [leak]}]})
    raw = an.Raw(json.dumps({"proposals": [{**good, "evidence": [leak]}], "none_reason": None}), "m", 0.0)
    problems, data, _ = an.check_answer("proposals", raw, FACTS)
    assert problems and data is None


def test_each_change_text_is_checked_against_its_own_change() -> None:
    """One change promoted (and shown better) in a week never lets another change's text say "promoted" or
    "improved", or quote that change's numbers."""
    facts = {"kind": "changelog", "week": "2026-W43", "changes": [
        {"id": "trial:1", "sport": "nba", "setting": "Blend weight of US books", "eval_n": 1440, "dll": -0.0006,
         "lo": -0.001, "hi": -0.0003, "decision": "promoted", "reason": "within the margin"},
        {"id": "trial:2", "sport": "nhl", "setting": "Near-start window", "eval_n": 410, "dll": 0.0002,
         "lo": -0.0004, "hi": 0.0009, "decision": "shadow", "reason": "two weeks in shadow first"}]}
    answer = {"texts": [{"id": "1", "text": "NBA: the US books weight was promoted after 1440 games; it improved."},
                        {"id": "2", "text": "NHL: the near-start window was promoted and improved the brain."},
                        {"id": "trial:2", "text": "NHL: tested on 1440 games."},
                        {"id": "9", "text": "A change that isn't in the facts."}]}
    problems, data, rejected = an.check_answer("changelog", an.Raw(json.dumps(answer), "claude-opus-5-5", 0.01),
                                               facts)
    assert problems == [] and [t["id"] for t in data["texts"]] == ["1"]
    why = {r["where"]: " ".join(r["problems"]) for r in rejected}
    assert "promoted, improved" in why["change 2"]
    assert "numbers" in why["change trial:2"]  # trial 1's game count, not this change's
    assert "no change with that id" in why["change 9"]


def test_spend_is_logged_under_brain_names_and_capped_monthly(store) -> None:
    brain = claude_brain(store, [GOOD_NOTE, GOOD_NOTE])
    assert brain.ask("note", FACTS, now=ts(MONDAY, 10)).status == "done"
    calls, dollars = store.analyst_spend("brain:claude")
    assert calls == 1 and dollars == pytest.approx((20_000 * 4 + 3_000 * 20) / 1e6)
    # Never the analysts' names (AI_DAILY_LIMIT) and never people's chat credits
    assert store.analyst_spend("claude") == (0, 0.0) and store.analyst_spend("assistant") == (0, 0.0)
    with closing(sqlite3.connect(store.path)) as db:
        assert db.execute("SELECT COUNT(*) FROM ai_credits").fetchone()[0] == 0
    # A worst case past the monthly cap is not asked at all: templates instead
    capped = claude_brain(store, [GOOD_NOTE], cents=20)
    answer = capped.ask("note", FACTS, now=ts(MONDAY, 11))
    assert answer.status == "skipped" and "monthly cap" in answer.reason and capped.client.requests == []
    assert capped.status(now=ts(MONDAY, 11))["spent_cents"] == pytest.approx(14.0)
    # A new month starts from nothing
    assert capped.spent(now=ts(MONDAY + datetime.timedelta(days=10))) == 0.0
    # DeepSeek's spend has its own name
    ds = deepseek_brain(store, [GOOD_NOTE])
    assert ds.ask("note", FACTS).status == "done" and store.analyst_spend("brain:deepseek")[0] == 1


def test_claude_batches_go_out_at_half_price_and_are_collected_later(store) -> None:
    brain = claude_brain(store, [], batches=True)
    pending = brain.ask("note", FACTS, batch=True, job="night-x")
    assert pending.status == "pending" and pending.batch["id"] == "batch_1"
    params = brain.client.messages.batches.created[0][0]["params"]
    assert "fallbacks" not in params and params["output_config"]["format"]["type"] == "json_schema"
    assert brain.finish(pending.batch, FACTS, kind="note") is None  # still running
    brain.client.messages.batches.status = "ended"
    brain.client.messages.batches.answers["night-x"] = GOOD_NOTE
    done = brain.finish(pending.batch, FACTS, kind="note")
    assert done.status == "done" and done.cost == pytest.approx((20_000 * 4 + 3_000 * 20) / 1e6 / 2)


BAD_NOTE = json.dumps({"note": "The biggest miss had a log loss of 9.8765.", "hypotheses": []})  # a number not given


def test_a_retry_counts_the_batches_still_out_against_the_cap(store) -> None:
    """Two batches out at once: when one's answer fails the checks, its full-price retry is made only if it fits
    under the cap together with what the other batch will still be billed."""
    clock = Clock(ts(MONDAY, 10))
    brain = claude_brain(store, [GOOD_NOTE], batches=True, cents=30, clock=clock)
    learn = learner(store, clock, analyst=brain)
    learn._ask("note", FACTS, clock.t, "night-x")
    learn._ask("memo", {"month": "2026-10", "decisions": {}}, clock.t, "month-x")
    assert [j["kind"] for j in learn._jobs()] == ["note", "memo"]
    batches = brain.client.messages.batches
    batches.status = "ended"
    batches.answers["night-x"] = BAD_NOTE
    clock.t += loop.POLL_SECONDS + 1
    learn._set("ai_jobs", [{**learn._jobs()[0]}, {**learn._jobs()[1], "polled_at": clock.t}], clock.t)  # memo waits
    done = learn.poll_ai(clock.t)
    assert [d["job"] for d in done] == ["night-x"] and done[0]["status"] == "failed"
    assert brain.client.requests == []  # 7 cents spent + 8 promised to the memo + a 16-cent retry would pass 30
    spent = brain.spent(clock.t) + learn.reserved()
    assert spent * 100 <= 30


def test_paused_learning_makes_no_ai_call_and_freezes_nothing(store) -> None:
    rows = sim.consensus_world(600, seed=5)
    with closing(store._connect()) as db:
        sim.write_consensus(db, rows)
    clock = Clock(max(r["start"] for r in rows) + 7200)
    brain = claude_brain(store, [GOOD_NOTE], batches=True, clock=clock)
    learn = learner(store, clock, analyst=brain)
    with closing(store._connect()) as db:
        facts = loop.week_facts(db, clock.t)
    learn._ask("note", FACTS, clock.t, "night-x")
    learn._ask("proposals", facts, clock.t, "week-x")
    learn.set_paused(True)
    batches = brain.client.messages.batches
    batches.status = "ended"
    batches.answers["night-x"] = BAD_NOTE  # would be asked again at full price if learning were on
    batches.answers["week-x"] = json.dumps({"proposals": [{
        "id": "p1", "knob": "consensus.weight.sharp", "sport": "nba", "direction": "increase", "size": 0.05,
        "hypothesis": "Sharp prices lead late.", "evidence": [facts["misses"][0]["id"]], "test": "", "expect": ""}],
        "none_reason": None})
    clock.t += loop.POLL_SECONDS + 1
    done = learn.poll_ai(clock.t)
    assert {d["job"]: d["status"] for d in done} == {"night-x": "failed", "week-x": "done"}
    assert brain.client.requests == [] and learn._jobs() == []
    with closing(store._connect()) as db:
        latest = bs.get_state(db, "proposals:latest")
        assert bs.trials(db) == [] and bs.versions(db) == []
    assert latest["proposals"][0]["outcome"] == {"rejected": "not frozen: learning is paused"}


def test_an_answer_that_cannot_be_used_is_collected_and_billed_once(store, monkeypatch) -> None:
    """A batch whose answer breaks while it is used leaves the queue first: it is never collected (or its cost
    logged) again, and the other answers of the same poll are still used once."""
    clock = Clock(ts(MONDAY, 10))
    brain = claude_brain(store, [], batches=True, clock=clock)
    learn = learner(store, clock, analyst=brain)
    learn._ask("proposals", {"misses": [], "registry": {}}, clock.t, "week-x")
    learn._ask("note", FACTS, clock.t, "night-x")
    batches = brain.client.messages.batches
    batches.status = "ended"
    batches.answers["week-x"] = json.dumps({"proposals": [{
        "id": "p1", "knob": "consensus.weight.model_rating", "sport": "nba", "direction": "increase", "size": 0.05,
        "hypothesis": "Ratings lead on back-to-backs.", "evidence": [], "test": "", "expect": ""}],
        "none_reason": None})
    batches.answers["night-x"] = GOOD_NOTE
    monkeypatch.setattr(tune, "freeze_proposals", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broken")))
    clock.t += loop.POLL_SECONDS + 1
    done = learn.poll_ai(clock.t)
    assert {d["job"]: d["status"] for d in done} == {"week-x": "failed", "night-x": "done"}
    assert learn._jobs() == [] and store.analyst_spend("brain:claude")[0] == 2
    for _ in range(3):
        clock.t += loop.POLL_SECONDS + 1
        assert learn.poll_ai(clock.t) == []
    assert store.analyst_spend("brain:claude")[0] == 2
    with closing(store._connect()) as db:
        assert bs.get_state(db, "note:latest")["source"] == "ai:claude"


# --- the scheduler -------------------------------------------------------------------------------------------------

def test_periods_follow_server_time() -> None:
    tuesday_1am = ts(MONDAY, 25)
    assert loop.period("night", tuesday_1am, UTC) == ("2026-10-26", ts(MONDAY, 2.5))
    assert loop.period("night", ts(MONDAY, 26.5), UTC)[0] == "2026-10-27"
    assert loop.period("week", tuesday_1am, UTC) == ("2026-W44", ts(MONDAY, 3.5))
    assert loop.period("week", ts(MONDAY, 3), UTC)[0] == "2026-W43"  # before Monday 03:30: last week's
    assert loop.period("month", tuesday_1am, UTC) == ("2026-10", datetime.datetime(2026, 10, 1, 4, 30,
                                                                                    tzinfo=UTC).timestamp())
    assert loop.period("night", ts(MONDAY, 7), NY)[0] == "2026-10-26"  # 03:00 in New York


class Grader:
    def __init__(self) -> None:
        self.calls = 0

    def grade_due(self) -> int:
        self.calls += 1
        return 0


def learner(store: Store, clock: Clock, **extra: Any) -> loop.Learner:
    sent = extra.pop("sent", [])
    options = {"service": Grader(), "clock": clock, "tz": UTC, "notify": lambda text: sent.append(text) or True,
               "backfill": lambda now: {"ran": False, "stopped": "test"}, "public_url": "https://bets.example.com",
               "analyst": an.BrainAnalyst(store, mode="off", clock=clock, tz=UTC)}
    options.update(extra)
    return loop.Learner(store, **options)


def test_the_learner_runs_on_a_fake_clock_once_per_period_under_its_lease(store, tmp_path) -> None:
    clock = Clock(ts(MONDAY, 25))  # Tuesday 01:00: Monday's night, this week's and this month's runs are due
    grader = Grader()
    learn = learner(store, clock, service=grader)
    assert learn.due() == [("night", "2026-10-26"), ("week", "2026-W44"), ("month", "2026-10")]
    report = learn.tick()
    assert [(r["kind"], r["key"], r.get("done")) for r in report["ran"]] == [
        ("night", "2026-10-26", True), ("week", "2026-W44", True), ("month", "2026-10", True)]
    assert list(report["ran"][0]["steps"]) == ["grade", "backfill", "scores", "note"]
    assert grader.calls == 1
    # Idempotent: nothing more until the next 02:30
    assert learn.due() == [] and learn.tick()["ran"] == [] and grader.calls == 1
    clock.t = ts(MONDAY, 26.6)  # Tuesday 02:36
    assert learn.due() == [("night", "2026-10-27")]
    learn.tick()
    assert grader.calls == 2
    # Another process holds the lease: it runs nothing
    other = Store(store.path)
    other.holder = "another-process"
    clock.t = ts(MONDAY, 50.6)
    assert learner(other, clock).tick()["skipped"].startswith("another process")
    assert learn.due() == [("night", "2026-10-28")]  # still due for the holder


def test_a_failed_step_is_resumed_on_the_next_tick_without_redoing_the_ones_before(store) -> None:
    clock = Clock(ts(MONDAY, 26.6))
    grader = Grader()
    attempts: List[float] = []

    def flaky(now: float) -> Dict[str, Any]:
        attempts.append(now)
        if len(attempts) == 1:
            raise RuntimeError("ESPN timed out")
        return {"ran": True, "stopped": "done", "requests": 3}

    learn = learner(store, clock, service=grader, backfill=flaky)
    first = learn.tick()["ran"][0]
    assert first["stopped"] == "backfill failed" and first["steps"]["grade"]["done"]
    assert learn.due()[0] == ("night", "2026-10-27")
    second = learn.tick()["ran"][0]
    assert second["done"] and grader.calls == 1 and len(attempts) == 2
    assert second["steps"]["backfill"]["result"]["requests"] == 3


def test_the_lease_is_renewed_while_a_long_step_runs(store) -> None:
    import time as real_time
    clock = Clock(ts(MONDAY, 26.6))
    renewals: List[float] = []

    def lease() -> bool:
        renewals.append(real_time.monotonic())
        return True

    learn = learner(store, clock, lease=lease, backfill=lambda now: real_time.sleep(0.35) or {"ran": True})
    learn.renew_every = 0.05
    assert learn.tick()["ran"][0]["done"]
    assert len(renewals) >= 1 + 4 + 3  # the tick, one before each step, and several during the slow backfill


def test_pause_freezes_values_but_scoring_goes_on(store, monkeypatch) -> None:
    clock = Clock(ts(MONDAY, 25))
    seen: Dict[str, Any] = {}
    real_week, real_night = tune.run_week, tune.run_night
    monkeypatch.setattr(tune, "run_week", lambda target, **kw: seen.setdefault("week", kw) and real_week(target, **kw))
    monkeypatch.setattr(tune, "run_night", lambda target, **kw: seen.setdefault("night", kw) and
                        real_night(target, **kw))
    asked: List[str] = []
    brain = claude_brain(store, [])
    brain.ask = lambda kind, facts, **kw: asked.append(kind)  # type: ignore[method-assign]
    sent: List[str] = []
    learn = learner(store, clock, analyst=brain, sent=sent,
                    backfill=lambda now: pytest.fail("no backfill while paused"))
    state = learn.set_paused(True)
    assert state["paused"] and not state["locked"] and sent[-1].startswith("Learning paused")
    report = learn.tick()
    assert all(r.get("done") for r in report["ran"])
    assert seen["week"]["paused"] is True and seen["night"]["paused"] is True and asked == []
    steps = report["ran"][0]["steps"]
    assert steps["backfill"]["result"] == {"skipped": "paused"} and steps["scores"]["done"]
    assert learn.set_paused(False)["paused"] is False and sent[-1].startswith("Learning switched back on")
    # LEARN_ENABLED=0 keeps it paused whatever the switch says
    locked = learner(store, clock, enabled=False)
    assert locked.set_paused(False) == {"paused": True, "by": "LEARN_ENABLED", "at": clock.t, "locked": True}


def test_a_night_rollback_is_told_even_when_a_later_part_of_the_step_fails(store, monkeypatch) -> None:
    before = sim.consensus_world(1500, seed=21)
    T = before[-1]["start"] + 86400
    bad = registry.defaults("nba")
    bad.update({"consensus.weight.sharp": 0.25, "consensus.weight.us_books": 0.25,
                "consensus.weight.prediction_market": 0.2, "consensus.weight.model_rating": 0.3,
                "consensus.weight.team_model": 0.0})
    with closing(store._connect()) as db:
        version = bs.insert_version(db, "nba", bad, status="challenger", now=T - 10)
        bs.promote(db, version, T)
    after = sim.consensus_world(400, seed=22, start=T + 3600, values=bad, params_version=version,
                                noise={"model_rating": 2.0}, game_prefix="h")
    with closing(store._connect()) as db:
        sim.write_consensus(db, before + after)
    night = datetime.datetime.fromtimestamp(after[-1]["start"] + 86400, UTC).replace(hour=2, minute=31, second=0)
    night += datetime.timedelta(days=1)
    clock = Clock(night.timestamp())
    real = loop.ev.write_week_scores
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real(*args, **kwargs)

    monkeypatch.setattr(loop.ev, "write_week_scores", flaky)
    sent: List[str] = []
    learn = learner(store, clock, sent=sent)
    key = loop.period("night", clock.t, UTC)[0]
    assert learn.run("night", key, clock.t)["stopped"] == "scores failed"
    told = [text for text in sent if "rolled back automatically" in text]
    assert len(told) == 1 and told[0].startswith("Brain: NBA rolled back automatically (Worse than the version")
    assert ").)" not in told[0]
    clock.t += 60
    assert learn.run("night", key, clock.t)["done"]
    assert len([text for text in sent if "rolled back automatically" in text]) == 1  # told once, not again
    with closing(store._connect()) as db:
        assert bs.get_version(db, version)["status"] == "rolled_back"


def plant_champion(store: Store, now: float, sport: str = "nba") -> int:
    values = registry.defaults(sport)
    values["consensus.weight.sharp"] = 0.55
    with closing(store._connect()) as db:
        version = bs.insert_version(db, sport, values, status="challenger", source="refit", now=now,
                                    knob="consensus.weight.sharp", gate="nudge")
        bs.promote(db, version, now)
        bs.add_trial(db, {"week": "2026-W43", "sport": sport, "knob": "consensus.weight.sharp", "old": {
            "consensus.weight.sharp": 0.5}, "new": {"consensus.weight.sharp": 0.55}, "source": "refit",
            "eval_n": 320, "dll": -0.0004, "lo": -0.0011, "hi": 0.0003, "decision": "promoted", "gate": "nudge",
            "version": version, "reason": "Non-inferior on 320 games."}, now)
    registry.invalidate()
    return version


def test_undo_takes_a_change_back_and_tells_the_owner(store) -> None:
    clock = Clock(ts(MONDAY, 10))
    version = plant_champion(store, clock.t - 3600)
    assert registry.current("basketball_nba", store).version == version
    sent: List[str] = []
    learn = learner(store, clock, sent=sent)
    result = learn.undo(version)
    assert result == {"undone": version, "now_live": 0}
    assert registry.current("basketball_nba", store).version == 0
    assert "Undone by the owner" in sent[-1] and sent[-1].endswith("https://bets.example.com/#/bets?tab=brain")
    with closing(store._connect()) as db:
        assert bs.trials(db, limit=1)[0]["decision"] == "undone"
        assert bs.get_version(db, version)["status"] == "rolled_back"
    assert learn.undo(version)["error"]  # not live any more
    # Withdrawing an experiment in shadow changes nothing live, and the message says so
    with closing(store._connect()) as db:
        shadow = bs.insert_version(db, "nba", registry.defaults("nba"), status="challenger", now=clock.t,
                                   knob="consensus.sharpness", gate="structural")
    out = learn.undo(shadow)
    assert out["withdrawn"] is True and out["undone"] == shadow
    assert sent[-1].startswith(f"Withdrawn by the owner: brain version {shadow} was an experiment in shadow; "
                               "nothing live changed.")
    assert "live again" not in sent[-1]


def test_the_weekly_summary_is_three_lines_from_the_gates_counts() -> None:
    headlines = {"nba": {"n": 200, "gap": 0.0004, "lo": -0.0011, "hi": 0.0019,
                         "word": "about level with the sharp market"}, "nhl": {"n": 40}}
    text = loop.weekly_summary("2026-W43", {"trials": 5, "promoted": [], "rolled_back": []}, headlines,
                               "https://bets.example.com/#/bets?tab=brain")
    lines = text.split("\n")
    assert len(lines) == 3
    assert lines[0] == "Brain, week 2026-W43: 5 candidates tested, no change this week (most weeks end this way)."
    assert lines[1] == ("NBA, last 200 games: about level with the sharp market (log loss +0.0004, 95% range -0.0011 "
                        "to +0.0019).")
    assert lines[2] == "Details: https://bets.example.com/#/bets?tab=brain"
    changed = loop.weekly_summary("2026-W43", {"trials": 5, "promoted": [{"sport": "nba"}], "rolled_back": []},
                                  headlines, None)
    assert "1 promoted, 0 rolled back" in changed and "undone" in changed
    assert "learning is paused" in loop.weekly_summary("2026-W43", {"trials": 2, "skipped": "paused"}, {}, None)


def test_a_week_whose_gate_never_ran_sends_no_old_counts(store, monkeypatch) -> None:
    monday = datetime.datetime(2026, 11, 2, 3, 31, tzinfo=UTC).timestamp()
    with closing(store._connect()) as db:
        bs.set_state(db, "week_report:latest", {"week": "2026-W43", "skipped": None, "trials": 5,
                                                "promoted": [{"sport": "nba", "version": 3}], "rolled_back": []},
                     monday - 7 * 86400)
        bs.set_state(db, "headlines", {"at": monday, "sports": {"nba": {
            "n": 200, "gap": 0.0004, "lo": -0.0011, "hi": 0.0019, "word": "about level with the sharp market"}}}, monday)
    monkeypatch.setattr(tune, "run_week", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("database is locked")))
    clock = Clock(monday)
    sent: List[str] = []
    learn = learner(store, clock, sent=sent)
    key = loop.period("week", clock.t, UTC)[0]
    for _ in range(loop.MAX_TRIES):
        learn.run("week", key, clock.t)
        clock.t += 60
    assert len(sent) == 1
    first = sent[0].split("\n")[0]
    assert first == "Brain, week 2026-W44: the weekly check couldn't run this time, so there is no decision to report."
    assert "promoted" not in sent[0] and "5 candidates" not in sent[0]


def test_a_handful_of_games_never_reads_as_ahead_or_behind(store) -> None:
    rows = sim.consensus_world(60, seed=8)
    with closing(store._connect()) as db:
        sim.write_consensus(db, rows)
        records = loop.ev.consensus_records(db)
    h = loop.sport_headline(records)
    assert h["n"] == 60 and h["word"] == "too few games to say yet"
    assert loop.headline_text("nba", h).startswith("NBA, last 60 games: too few games to say yet")
    assert loop.headline_text("nba", {"n": 1, "gap": 0.1, "lo": None, "hi": None, "word": "too few games to say yet"}) \
        == "NBA, last 1 game: too few games to say yet."


def test_the_weekly_run_sends_the_owner_the_summary_and_writes_template_texts(store) -> None:
    rows = sim.consensus_world(600, seed=3)
    with closing(store._connect()) as db:
        sim.write_consensus(db, rows)
    clock = Clock(max(r["start"] for r in rows) + 7200)
    sent: List[str] = []
    learn = learner(store, clock, sent=sent)
    report = learn.tick()
    week = next(r for r in report["ran"] if r["kind"] == "week")
    assert week["steps"]["telegram"]["result"]["sent"] is True
    assert len(sent) == 1 and len(sent[0].split("\n")) == 3 and sent[0].startswith("Brain, week ")
    with closing(store._connect()) as db:
        changelog = bs.get_state(db, "changelog")
        note = bs.get_state(db, "note:latest")
    assert changelog and all(entry["source"] == "template" for entry in changelog.values())
    assert note["source"] == "template" and "graded games in the last 7 days" in note["text"]
    # Without an owner chat nothing is sent, and the step says why
    quiet = loop.Learner(store, clock=clock, tz=UTC, use_notify=False, backfill=lambda now: {})
    assert quiet._week_telegram(clock.t, "x")["why"].startswith("no owner chat")


def test_ai_notes_come_from_the_linked_ai_through_a_batch_and_fall_back_to_templates(store) -> None:
    rows = sim.consensus_world(300, seed=4)
    with closing(store._connect()) as db:
        sim.write_consensus(db, rows)
    clock = Clock(max(r["start"] for r in rows) + 7200)
    brain = claude_brain(store, [], batches=True, clock=clock)
    learn = learner(store, clock, analyst=brain)
    learn.tick()
    jobs = learn._jobs()
    assert {j["kind"] for j in jobs} == {"note", "proposals", "changelog", "memo"}
    assert learn.ai_status()["reserved_cents"] > 0
    with closing(store._connect()) as db:
        facts = bs.get_state(db, f"ai_facts:{next(j['job'] for j in jobs if j['kind'] == 'note')}")
        worst = facts["misses"][0]
    note = json.dumps({"note": f"The worst miss had a log loss of {worst['brain_ll']}.", "hypotheses": []})
    batches = brain.client.messages.batches
    batches.status = "ended"
    for job in jobs:
        batches.answers[job["job"]] = note if job["kind"] == "note" else json.dumps(
            {"proposals": [], "none_reason": "Nothing in the facts supports a change."}
            if job["kind"] == "proposals" else {"texts": []} if job["kind"] == "changelog" else
            {"memo": "A quiet month.", "data_ideas": []})
    clock.t += loop.POLL_SECONDS + 1
    learn.tick()
    assert learn._jobs() == []
    with closing(store._connect()) as db:
        latest = bs.get_state(db, "note:latest")
        proposals = bs.get_state(db, "proposals:latest")
    assert latest["source"] == "ai:claude" and str(worst["brain_ll"]) in latest["text"]
    assert proposals["none_reason"].startswith("Nothing in the facts")
    assert store.analyst_spend("brain:claude")[0] == 4


def test_ai_proposals_are_frozen_as_challengers_judged_on_later_games(store) -> None:
    rows = sim.consensus_world(900, seed=5)
    with closing(store._connect()) as db:
        sim.write_consensus(db, rows)
    clock = Clock(max(r["start"] for r in rows) + 7200)
    with closing(store._connect()) as db:
        facts = loop.week_facts(db, clock.t)
    miss = facts["misses"][0]["id"]
    answer = an.Answer("proposals", "done", provider="claude", model="claude-opus-5-5", data={
        "proposals": [{"id": "p1", "knob": "consensus.weight.sharp", "sport": "nba", "direction": "increase",
                       "size": 0.05, "hypothesis": "Sharp prices lead late.", "evidence": [miss], "test": "",
                       "expect": ""},
                      {"id": "p2", "knob": "consensus.weight.sharp", "sport": "nba", "direction": "increase",
                       "size": 0.05, "hypothesis": "Again.", "evidence": ["miss:made-up"], "test": "", "expect": ""}],
        "none_reason": None})
    learn = learner(store, clock)
    learn._apply(answer, facts, clock.t, "week-x")
    with closing(store._connect()) as db:
        latest = bs.get_state(db, "proposals:latest")
        trials = bs.trials(db, limit=5)
    outcomes = [p["outcome"] for p in latest["proposals"]]
    assert "rejected" in outcomes[1] and "evidence" in outcomes[1]["rejected"]  # an id not in the fact sheet
    assert all(t["source"] == "ai:claude" for t in trials[:2])
    frozen = [t for t in trials if t["decision"] == "shadow"]
    for t in frozen:  # judged only on games after it was made: its version is live for later games only
        with closing(store._connect()) as db:
            assert bs.get_version(db, t["version"])["effective_from"] == clock.t
    assert loop.ai_record(sqlite3.connect(store.path))["made"] == 2


# --- routes ---------------------------------------------------------------------------------------------------------

def app_client(store: Store, learn: loop.Learner) -> TestClient:
    service = types.SimpleNamespace(store=store, lab=object(), slate=None)
    app = create_app(service, "s3cret", login_delay=0, bot_token=BOT_TOKEN, telegram_ids=[42, 7], learner=learn,
                     owner_id=42)
    return TestClient(app)


def sign_in_telegram(client: TestClient, uid: int) -> None:
    init = make_init_data({"id": uid, "first_name": "Sam"}, auth_date=int(datetime.datetime.now().timestamp()))
    assert client.post("/api/tg-login", json={"init_data": init}).status_code == 200


def test_the_progress_page_and_the_owner_only_switches(store) -> None:
    clock = Clock(ts(MONDAY, 10))
    version = plant_champion(store, clock.t - 3600)
    sent: List[str] = []
    learn = learner(store, clock, sent=sent)
    anyone = app_client(store, learn)
    assert anyone.get("/api/brain/progress").status_code == 401
    assert anyone.post("/api/brain/pause", json={"paused": True}).status_code == 401
    friend = app_client(store, learn)
    sign_in_telegram(friend, 7)  # allowed in the app, not the owner
    page = friend.get("/api/brain/progress").json()
    assert page["owner"] is False and page["changes"][0]["undo"] is False
    assert friend.post("/api/brain/undo", json={"version": version}).status_code == 403
    assert friend.post("/api/brain/pause", json={"paused": True}).status_code == 403
    owner = app_client(store, learn)
    sign_in_telegram(owner, 42)  # TELEGRAM_CHANNEL_ID's user
    page = owner.get("/api/brain/progress").json()
    change = page["changes"][0]
    assert page["owner"] is True and change["undo"] is True and change["decision"] == "promoted"
    assert change["label"] == "Blend weight of sharp books" and change["evidence"]["n"] == 320
    assert owner.post("/api/brain/pause", json={"paused": True}).json()["paused"] is True
    assert owner.get("/api/brain/progress").json()["learning"]["paused"] is True
    assert owner.post("/api/brain/undo", json={"version": version}).json() == {"undone": version, "now_live": 0}
    assert owner.post("/api/brain/undo", json={"version": version}).status_code == 409
    assert owner.post("/api/brain/undo", json={"version": 999}).status_code == 404
    after = owner.get("/api/brain/progress").json()
    promoted = next(c for c in after["changes"] if c["decision"] == "promoted")
    assert promoted["undo"] is False and promoted["rolled_back"]["decision"] == "undone"
    password = app_client(store, learn)
    assert password.post("/api/login", json={"password": "s3cret"}).status_code == 200
    assert password.post("/api/brain/pause", json={"paused": False}).json()["paused"] is False
    assert password.get("/api/brain/progress").json()["owner"] is True


def test_the_page_reads_real_numbers_from_the_store(store) -> None:
    rows = sim.consensus_world(800, seed=6)
    with closing(store._connect()) as db:
        sim.write_consensus(db, rows)
    clock = Clock(max(r["start"] for r in rows) + 7200)
    learn = learner(store, clock)
    learn.tick()
    page = progress.progress(store, learn, owner=True, now=clock.t)
    nba = page["sports"][0]
    assert nba["sport"] == "nba" and nba["headline"]["n"] == 200 and nba["headline"]["text"].startswith("NBA, last 200")
    assert nba["series"] and all(set(p) >= {"week", "n", "gap", "lo", "hi"} for p in nba["series"])
    assert nba["calibration"]["words"] and nba["version"]["number"] == 0
    assert page["held"]["tested"] >= 1 and page["held"]["words"].startswith("This week:")
    assert page["ai"]["reason"] == "AI notes: off (LEARN_AI=off)" and page["ai"]["note"]["source"] == "template"
    assert page["health"]["graded"]["nba"] == 800 and page["health"]["runs"]["night"]["last"]
    json.dumps(page)  # plain JSON
    lines = progress.brain_lines(store, now=clock.t)
    assert lines[0].startswith("NBA, last 200 games") and "version 0" in lines[0]
    facts = progress.tool_facts(store, learn, now=clock.t)
    assert facts["sports"][0]["sport"] == "NBA" and "Most weeks nothing changes" in facts["promise"]


def test_the_page_counts_only_what_the_gate_tested_and_dates_its_numbers(store) -> None:
    """Held steady and the cards count the gate's own candidates: an AI proposal frozen as a challenger is "queued",
    never "passed", and its later ruling isn't counted twice; rollbacks, undos and safe mode aren't candidates; the
    numbers are dated by the nightly run that worked them out, not by the request."""
    rows = sim.consensus_world(900, seed=5)
    with closing(store._connect()) as db:
        sim.write_consensus(db, rows)
    T = max(r["start"] for r in rows) + 7200
    clock = Clock(T)
    learn = learner(store, clock)
    learn.tick()  # a nightly and a weekly run: the gate's held candidates for the decided week
    week = tune.decision_week(T)
    with closing(store._connect()) as db:
        report = bs.get_state(db, f"week_report:{week}")
        gate_ids = [t["id"] for t in bs.trials(db, week=week)]
        # An AI proposal frozen in that week's key, then the gate's later ruling on the same version, then a
        # rollback and an owner's undo: none of them is a new candidate
        version = bs.insert_version(db, "nba", registry.defaults("nba"), status="challenger", source="ai:claude",
                                    effective_from=T, now=T, knob="consensus.weight.sharp", gate="structural")
        bs.add_trial(db, {"week": week, "sport": "nba", "knob": "consensus.weight.sharp", "source": "ai:claude",
                          "decision": "shadow", "gate": "structural", "version": version,
                          "reason": "Frozen as a challenger: judged only on games after today.",
                          "details": {"proposal": {}, "direction": "increase"}}, T)
        bs.add_trial(db, {"week": week, "sport": "nba", "knob": "-", "source": "refit", "decision": "rolled_back",
                          "gate": "rollback", "reason": "Worse.", "details": {}}, T)
        bs.add_trial(db, {"week": week, "sport": "nba", "knob": "-", "source": "owner", "decision": "undone",
                          "gate": "owner", "reason": "Undone by the owner.", "details": {}}, T)
    page = progress.progress(store, learn, owner=True, now=T + 3 * 86400)
    held = page["held"]
    assert held["week"] == week and held["tested"] == report["trials"] == len(gate_ids)
    assert held["passed"] == 0 and held["frozen"] == 1
    assert "1 AI proposal queued to be judged on later games" in held["words"] and "passed the gate" not in held["words"]
    assert page["health"]["trials"]["month"] == len(gate_ids)
    cards = [c for c in page["changes"] if c["week"] == week]
    assert cards and all(c["evidence"]["tried"] == sum(1 for t in gate_ids) for c in cards if c["sport"] == "nba")
    # Dated by the nightly run that scored them, not by the request three days later
    with closing(store._connect()) as db:
        scored = bs.get_state(db, "headlines")["at"]
    assert page["as_of"] == loop.iso(scored) and page["served_at"] == loop.iso(T + 3 * 86400)


def test_the_bot_brain_command_and_the_ai_tool(store) -> None:
    from telegram_betting_bot_simple.ai.tools import INPUTS, TOOLS, Toolbox
    from telegram_betting_bot_simple.bot.handlers import brain_text
    tool = next(t for t in TOOLS if t["name"] == "brain_progress")
    assert "strict" not in tool and tool["input_schema"]["properties"] == {} and "brain_progress" in INPUTS
    rows = sim.consensus_world(400, seed=7)
    with closing(store._connect()) as db:
        sim.write_consensus(db, rows)
    service = types.SimpleNamespace(store=store)
    text = brain_text(service)
    assert text.startswith("🧠 How the brain is doing") and "NBA, last 200 games" in text
    assert "No change yet" in text
    lab = types.SimpleNamespace(service=service, store=store)
    run = Toolbox(lab, "tg:42", NY).run("brain_progress", {})
    assert run.ok and run.content["sports"][0]["sport"] == "NBA" and run.content["source"].startswith("The learning")


# --- the grader: goals-model markets settle, and the history capture ---------------------------------------------

@pytest.mark.parametrize("market, side, point, home, away, expected", [
    ("btts", "yes", None, 2, 1, "won"), ("btts", "no", None, 2, 1, "lost"), ("btts", "no", None, 0, 0, "won"),
    ("draw_no_bet", "home", None, 2, 1, "won"), ("draw_no_bet", "away", None, 2, 1, "lost"),
    ("draw_no_bet", "home", None, 1, 1, "push"),
    ("double_chance", "1X", None, 1, 1, "won"), ("double_chance", "12", None, 1, 1, "lost"),
    ("double_chance", "X2", None, 0, 3, "won"), ("double_chance", "X2", None, 2, 1, "lost"),
    ("team_totals", "home_over", 1.5, 2, 1, "won"), ("team_totals", "home_under", 1.5, 2, 1, "lost"),
    ("team_totals", "away_over", 0.5, 2, 0, "lost"), ("team_totals", "away_under", 1.0, 2, 1, "push"),
])
def test_goals_model_markets_settle_from_the_ninety_minute_score(market, side, point, home, away, expected) -> None:
    assert derived_outcome(market, side, point, home, away) == expected
    assert _outcome({"market": market, "side": side, "point": point, "league": "soccer_epl"},
                    {"home": home, "away": away}, {}) == expected


def test_saved_goals_model_bets_are_settled_by_the_grader_not_voided() -> None:
    start = "2026-10-20T15:00:00Z"
    legs = [tickets.TicketLeg(key=f"soccer_epl:s1:{m}:{s}", league="soccer_epl", event_id="s1", market=m, side=s,
                              point=p, price=Decimal(price), label=f"{m} {s}", game="Chelsea @ Arsenal", start=start)
            for m, s, p, price in (("btts", "yes", None, "1.80"), ("draw_no_bet", "home", None, "1.50"),
                                   ("double_chance", "X2", None, "2.40"), ("team_totals", "home_over", 1.5, "2.10"),
                                   ("team_totals", "home_under", 2.0, "1.90"))]
    priced = tickets.price_ticket(legs, "single", Decimal("10"), now=datetime.datetime(2026, 10, 19,
                                                                                         tzinfo=datetime.timezone.utc))
    slip = {"id": 1, "user": "tg:42", "payload": {**priced, "events": {"s1": {"home": "Arsenal", "away": "Chelsea"}}},
            "legs": [{"league": "soccer_epl", "event_id": "s1", "market": leg.market, "side": leg.side,
                      "point": leg.point, "start": start, "result": None, "game": leg.game} for leg in legs]}

    class SlipStore:
        settled: Dict[str, Any] = {}

        def set_leg_results(self, slip_id, results):
            raise AssertionError("every leg is decided")

        def settle_slip(self, slip_id, status, result, leg_results=None, now=None):
            self.settled = {"status": status, "result": result, "legs": leg_results}

    class Finals:
        def final(self, row):
            return {"state": "final", "home": 2, "away": 1, "espn_id": "e1"}

    lab = object.__new__(Lab)
    lab.service = Finals()
    lab._now = lambda: 0.0  # type: ignore[assignment]
    target = SlipStore()
    assert lab._grade_slip(target, slip, "2026-10-21T00:00:00Z", "2026-10-17T00:00:00Z", {}) == 1
    assert [leg["result"] for leg in target.settled["legs"]] == ["won", "won", "lost", "won", "push"]
    assert target.settled["status"] == "settled"


class SummaryEspn:
    """ESPN with one finished EPL game: its result and lean summary (the captured fixture)."""

    def __init__(self) -> None:
        self.data = json.loads((FIXTURES / "espn_epl_summary.json").read_text())

    def game_result(self, league, event_id):
        return {"state": "final", "home": 1, "away": 1}

    def summary(self, league, event_id):
        return self.data


def test_the_grader_captures_finished_games_into_the_history_and_grades_snapshots(store) -> None:
    start = "2025-03-15T15:00:00Z"
    sharp = {"name": "Sharp", "kind": "sharp", "status": "ok", "as_of": "2025-03-15T14:50:00Z",
             "p": {"home": 0.45, "draw": 0.28, "away": 0.27}}
    blend = consensus.blend([sharp], "soccer_epl", now="2025-03-15T14:55:00Z", start=start)
    row = {"league": "soccer_epl", "game_key": "espn:704565", "event_id": "o1", "espn_id": "704565", "home": "Everton",
           "away": "West Ham United", "start": start, "sources": [sharp], "blend": blend}
    logged_at = datetime.datetime(2025, 3, 14, 20, tzinfo=datetime.timezone.utc).timestamp()  # 19 hours before
    assert store.log_consensus([row, {**row, "kind": "t24"}], now=logged_at) == 2
    service = DashboardService(None, SummaryEspn(), NY, store=store)
    assert service._grade_consensus("2025-03-16T00:00:00Z", "2025-03-01T00:00:00Z") == 1  # the snapshot isn't counted
    graded = store.consensus_rows(status="graded", snapshots=True)
    assert {r["game_key"] for r in graded} == {"espn:704565", "espn:704565" + SNAPSHOT_SUFFIX}
    assert [r["game_key"] for r in store.consensus_rows()] == ["espn:704565"]  # the record never counts it twice
    with closing(sqlite3.connect(store.path)) as db:
        game = db.execute("SELECT home, away, home_score, away_score, source FROM hist_games").fetchall()
        stats = dict(db.execute("SELECT team_side || ':' || stat, value FROM hist_stats").fetchall())
        lines = db.execute("SELECT COUNT(*) FROM hist_lines WHERE game_key = 'espn:704565'").fetchone()[0]
    assert game == [("Everton", "West Ham United", 1, 1, "live")]
    assert "home:corners" in stats and "away:yellow" in stats and lines > 0


# --- the store's migration and the registry's wiring ----------------------------------------------------------------

def test_an_older_database_gains_the_new_columns_once(tmp_path) -> None:
    path = str(tmp_path / "old.db")
    from telegram_betting_bot_simple.data import store as store_module
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        db.executescript(store_module.SCHEMA)
        db.executescript(store_module.CONTEXT_SCHEMA)
        db.execute("INSERT INTO consensus_log (league, game_key, home, away, start, logged_at, sources, blend, status) "
                   "VALUES ('basketball_nba', 'espn:1', 'A', 'B', '2026-01-01T00:00:00Z', 1, '[]', '{}', 'graded')")
    first, second = Store(path), Store(path)  # the bot and the dashboard starting together
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        columns = {table: {r[1] for r in db.execute(f"PRAGMA table_info({table})")} for table in
                   ("consensus_log", "context_log")}
        assert db.execute("SELECT params_version, kind FROM consensus_log").fetchall() == [(0, "last")]
        assert migrate(db) == []  # nothing left to add
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    for table, column, _ in MIGRATIONS:
        assert column in columns[table]
    assert {"hist_games", "hist_lines", "hist_stats", "brain_params", "brain_trials", "brain_scores"} <= tables
    assert first.consensus_graded()[0] == second.consensus_graded_mark() == (1, None)


def test_blends_adjustments_and_logged_rows_follow_the_live_version(store) -> None:
    now = ts(MONDAY, 10)
    start = now + 3 * 3600
    sources = [{"name": "Sharp", "kind": "sharp", "status": "ok", "as_of": consensus._iso(now - 60),
                "p": {"home": 0.6, "away": 0.4}},
               {"name": "Kalshi", "kind": "prediction_market", "status": "ok", "as_of": consensus._iso(now - 60),
                "p": {"home": 0.5, "away": 0.5}}]
    default = slate_mod.market_detail("basketball_nba", sources, start, now,
                                      params=slate_mod.brain_params("basketball_nba", store))
    assert default["blend"] == consensus.blend(sources, "basketball_nba", now=now, start=start)  # empty registry
    version = plant_champion(store, now - 60)
    params = slate_mod.brain_params("basketball_nba", store)
    tuned = slate_mod.market_detail("basketball_nba", sources, start, now, params=params)
    assert params.version == version and tuned["blend"]["p"]["home"] > default["blend"]["p"]["home"]
    row = slate_mod.SlateBuilder._consensus_row(
        {"league": "basketball_nba", "odds_id": "o1", "espn_id": "9", "home": {"name": "A"}, "away": {"name": "B"},
         "start": consensus._iso(start)}, tuned, params.version)
    assert store.log_consensus([row], now=now) == 1
    with closing(sqlite3.connect(store.path)) as db:
        assert db.execute("SELECT params_version FROM consensus_log").fetchone()[0] == version
    # The NBA model's weights and soccer's goals model read the same registry
    service = DashboardService(None, None, NY, store=store)
    assert service.params == service.base_params  # the champion moved a blend weight only
    assert slate_mod.brain_params("soccer_epl", store).soccer_kwargs() == {}


def test_the_t24_snapshot_is_sent_once_a_day_before_the_start(store) -> None:
    builder = object.__new__(slate_mod.SlateBuilder)
    builder.store, builder._logged_sig, builder._t24_logged = store, {}, set()
    builder._log_failure = lambda *a, **k: None
    now = ts(MONDAY, 10)
    row = {"league": "basketball_nba", "game_key": "espn:9", "event_id": "o9", "espn_id": "9", "swapped": False,
           "home": "A", "away": "B", "start": consensus._iso(now + 20 * 3600), "sources": [], "params_version": 0,
           "blend": {"p": {"home": 0.5, "away": 0.5}}}
    builder._log_consensus([row], now)
    builder._log_consensus([row], now + 600)  # unchanged: nothing new
    rows = store.consensus_rows(snapshots=True)
    assert sorted(r["game_key"] for r in rows) == ["espn:9", "espn:9#t24"]
    later = {**row, "game_key": "espn:10", "start": consensus._iso(now + 3 * 3600)}
    builder._log_consensus([later], now)  # first seen 3 hours before: no one-day snapshot for it
    assert "espn:10#t24" not in {r["game_key"] for r in store.consensus_rows(snapshots=True)}


def test_config_gains_the_learning_settings(monkeypatch) -> None:
    config = Config(TELEGRAM_BOT_TOKEN="123456:test", _env_file=None)
    assert (config.learn_enabled, config.learn_ai, config.learn_ai_monthly_cents,
            config.learn_backfill_per_night) == (True, "auto", 500, 1500)
    assert config.owner_telegram_id is None
    tuned = Config(TELEGRAM_BOT_TOKEN="123456:test", LEARN_ENABLED="0", LEARN_AI="DeepSeek",
                   LEARN_AI_MONTHLY_CENTS="", LEARN_BACKFILL_PER_NIGHT="300", TELEGRAM_CHANNEL_ID="12345678",
                   _env_file=None)
    assert (tuned.learn_enabled, tuned.learn_ai, tuned.learn_ai_monthly_cents, tuned.learn_backfill_per_night) == \
        (False, "deepseek", 500, 300)
    assert tuned.owner_telegram_id == 12345678
    assert Config(TELEGRAM_BOT_TOKEN="123456:test", TELEGRAM_CHANNEL_ID="-100123", _env_file=None).owner_telegram_id \
        is None  # a group is nobody's id
    with pytest.raises(ValueError):
        Config(TELEGRAM_BOT_TOKEN="123456:test", LEARN_AI="gpt", _env_file=None)
