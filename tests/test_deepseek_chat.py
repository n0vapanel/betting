"""Hardening tests for ai/deepseek_chat.py: OpenAI<->Anthropic translation edge cases, cost/budget
accounting, error mapping, provider selection and the text-only vision fallback. No network: every
HTTP path goes through FakeSession."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests

from telegram_betting_bot_simple.ai import assistant as assistant_module
from telegram_betting_bot_simple.ai.assistant import build_assistant
from telegram_betting_bot_simple.ai.deepseek_chat import DeepSeekCaller
from telegram_betting_bot_simple.ai.explain import AIError, MIN_OUTPUT_TOKENS, ClaudeCaller
from telegram_betting_bot_simple.ai.slip_reader import SlipReader
from telegram_betting_bot_simple.config import Config


# --- fakes ----------------------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


class BadJsonResponse(FakeResponse):
    def json(self):
        raise ValueError("not json")


class FakeSession:
    """Records one call; answers with a scripted payload or raises a scripted exception (no network)."""

    def __init__(self, payload=None, status=200, raise_exc=None, response=None):
        self.payload = payload or {}
        self.status = status
        self.raise_exc = raise_exc
        self.response = response
        self.requests = []

    def post(self, url, json=None, timeout=None, headers=None):
        self.requests.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response if self.response is not None else FakeResponse(self.status, self.payload)


def payload(finish_reason="stop", content="hello", tool_calls=None, usage=None, model="deepseek-v4-pro",
            refusal=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if refusal is not None:
        message["refusal"] = refusal
    return {"model": model,
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5}}


# --- (1) translation edge cases --------------------------------------------------------------------

def test_empty_content_drops_to_nothing() -> None:
    assert DeepSeekCaller._user_messages([]) == []
    # an empty text block still yields one user message with empty content (pinned: the list itself is
    # truthy, so "" is kept, not dropped)
    assert DeepSeekCaller._user_messages([{"type": "text", "text": ""}]) == \
        [{"role": "user", "content": ""}]
    assert DeepSeekCaller._assistant_message("") == {"role": "assistant", "content": ""}
    assert DeepSeekCaller._assistant_message([{"type": "text", "text": ""}]) == \
        {"role": "assistant", "content": None}
    # A tool_use beside empty text still carries the call, with content None
    assert DeepSeekCaller._assistant_message([{"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}}]) == \
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "f", "arguments": '{"a": 1}'}}]}


def test_multiple_tool_calls_and_tool_results_round_trip() -> None:
    assistant = DeepSeekCaller._assistant_message([
        {"type": "text", "text": "Two lookups"},
        {"type": "tool_use", "id": "c1", "name": "sports_find_events", "input": {"query": "knicks"}},
        {"type": "tool_use", "id": "c2", "name": "ticket_facts", "input": {"stake": 10}},
    ])
    assert assistant["content"] == "Two lookups"
    assert [call["id"] for call in assistant["tool_calls"]] == ["c1", "c2"]
    assert assistant["tool_calls"][1]["function"]["arguments"] == '{"stake": 10}'
    # Tool results return in order, one role=tool message each, with an empty id/content when missing
    user = DeepSeekCaller._user_messages([
        {"type": "tool_result", "tool_use_id": "c1", "content": "[]"},
        {"type": "tool_result", "tool_use_id": "c2", "content": "ok"},
        {"type": "tool_result"},
    ])
    assert user == [
        {"role": "tool", "tool_call_id": "c1", "content": "[]"},
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        {"role": "tool", "tool_call_id": "", "content": ""},
    ]


def test_messages_keeps_order_and_drops_unknown_roles() -> None:
    messages = [
        {"role": "system", "content": "ignored: system goes through the system key"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "f", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "ok"}]},
    ]
    out = DeepSeekCaller._messages(messages)
    assert [message["role"] for message in out] == ["user", "assistant", "tool"]
    assert out[2] == {"role": "tool", "tool_call_id": "c1", "content": "ok"}


def test_finish_reason_mapping() -> None:
    for finish_reason, stop_reason in (("stop", "end_turn"), ("tool_calls", "tool_use"),
                                       ("length", "max_tokens"), ("content_filter", "refusal"),
                                       ("something_new", "end_turn"), (None, "end_turn")):
        response = DeepSeekCaller._response(payload(finish_reason=finish_reason))
        assert response.stop_reason == stop_reason, finish_reason


def test_response_parses_list_parts_and_multiple_calls() -> None:
    response = DeepSeekCaller._response(payload(
        finish_reason="tool_calls",
        content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}},
            {"id": "c3", "type": "function", "function": {}},
        ],
    ))
    assert [block.type for block in response.content] == ["text", "tool_use", "tool_use"]
    assert response.content[0].text == "ab"
    assert response.content[1].name == "f" and response.content[1].input == {"x": 1}
    assert response.content[2].name == "" and response.content[2].input == {} and response.content[2].id == "c3"


def test_malformed_tool_arguments_fail_closed() -> None:
    # A malformed (or injected) tool_call must never run a brain tool with empty args: reject it, don't guess.
    with pytest.raises(AIError) as caught:
        DeepSeekCaller._response(payload(
            finish_reason="tool_calls",
            tool_calls=[{"id": "c2", "type": "function", "function": {"name": "g", "arguments": "not json"}}],
        ))
    assert "malformed" in str(caught.value) and caught.value.temporary is False


def test_tool_and_tool_choice_translation() -> None:
    assert DeepSeekCaller._tool({"name": "f"}) == {"type": "function",
                                                   "function": {"name": "f", "description": "",
                                                                "parameters": {"type": "object",
                                                                               "properties": {}}}}
    assert DeepSeekCaller._tool_choice(None) == "auto"
    assert DeepSeekCaller._tool_choice({"type": "none"}) == "none"
    assert DeepSeekCaller._tool_choice({"type": "tool", "name": "f"}) == \
        {"type": "function", "function": {"name": "f"}}
    assert DeepSeekCaller._tool_choice({"type": "tool"}) == "auto"  # no name: can't force
    assert DeepSeekCaller._tool_choice({"type": "any"}) == "auto"


def test_request_body_defaults_and_json_object_response_format() -> None:
    session = FakeSession(payload=payload())
    caller = DeepSeekCaller("sk-ds-test", session=session)
    caller.create("deepseek-v4-pro", system="You are terse.",
                  messages=[{"role": "user", "content": "hi"}],
                  output_config={"format": {"type": "json_schema", "schema": {}}})
    body = session.requests[0]["json"]
    assert body["max_tokens"] == 16000  # the default when not given
    assert body["messages"][0] == {"role": "system", "content": "You are terse."}
    assert body["response_format"] == {"type": "json_object"}
    assert "tools" not in body and "tool_choice" not in body
    # A non-json_schema output_config (and a missing one) leaves response_format out
    caller.create("deepseek-v4-pro", messages=[{"role": "user", "content": "hi"}],
                  output_config={"effort": "low"})
    assert "response_format" not in session.requests[1]["json"]


# --- (2) cost, budget and error mapping --------------------------------------------------------------

def test_cost_accounts_for_cache_miss_hit_unknown_and_object_usage() -> None:
    usage = {"prompt_tokens": 100, "prompt_cache_hit_tokens": 40, "prompt_cache_miss_tokens": 60,
             "completion_tokens": 20}
    dollars, tin, tout = DeepSeekCaller.cost("deepseek-v4-pro", usage)
    assert (tin, tout) == (100, 20)
    assert dollars == pytest.approx((40 * 0.044 + 60 * 1.32 + 20 * 3.96) / 1e6)
    # miss absent -> prompt minus hit
    dollars, tin, tout = DeepSeekCaller.cost("deepseek-v4-pro",
                                             {"prompt_tokens": 100, "prompt_cache_hit_tokens": 40,
                                              "completion_tokens": 20})
    assert (tin, tout) == (100, 20)
    assert dollars == pytest.approx((40 * 0.044 + 60 * 1.32 + 20 * 3.96) / 1e6)
    # hit absent -> the whole prompt at the uncached input rate
    dollars, tin, tout = DeepSeekCaller.cost("deepseek-v4-pro",
                                             {"prompt_tokens": 100, "completion_tokens": 20})
    assert (tin, tout) == (100, 20)
    assert dollars == pytest.approx((100 * 1.32 + 20 * 3.96) / 1e6)
    # flash prices
    dollars, _, _ = DeepSeekCaller.cost("deepseek-flash",
                                        {"prompt_tokens": 100, "completion_tokens": 20})
    assert dollars == pytest.approx((100 * 0.30 + 20 * 1.20) / 1e6)
    # unknown model falls back to deepseek-v4-pro prices
    dollars, _, _ = DeepSeekCaller.cost("deepseek-unknown",
                                        {"prompt_tokens": 100, "completion_tokens": 20})
    assert dollars == pytest.approx((100 * 1.32 + 20 * 3.96) / 1e6)
    # object-shaped usage (SimpleNamespace) prices the same as a dict
    obj = SimpleNamespace(prompt_tokens=100, prompt_cache_hit_tokens=40, prompt_cache_miss_tokens=60,
                          completion_tokens=20)
    dollars, tin, tout = DeepSeekCaller.cost("deepseek-v4-pro", obj)
    assert (tin, tout) == (100, 20)
    assert dollars == pytest.approx((40 * 0.044 + 60 * 1.32 + 20 * 3.96) / 1e6)
    # no usage -> free
    assert DeepSeekCaller.cost("deepseek-v4-pro", {}) == (0.0, 0, 0)


def test_cost_clamps_cache_hit_above_prompt_to_non_negative_miss() -> None:
    # A payload that reports cache hits but no prompt total and no miss (pathological; DeepSeek always
    # sends prompt_tokens) computes miss = prompt - hit = -40. Clamped to 0, so the 40 cache-hit tokens
    # bill at the hit rate and the spend can never go negative.
    usage = {"prompt_cache_hit_tokens": 40, "completion_tokens": 20}
    dollars, tin, tout = DeepSeekCaller.cost("deepseek-v4-pro", usage)
    assert tout == 20
    assert tin == 40
    assert dollars == pytest.approx((40 * 0.044 + 20 * 3.96) / 1e6)


def test_budget_boundaries_floor_to_min_output_tokens() -> None:
    caller = DeepSeekCaller("sk-ds-test", session=FakeSession())
    price_in, price_out = caller.prices("deepseek-v4-pro")
    assert (price_in, price_out) == (1.32, 3.96)
    assert caller.prices("deepseek-flash") == (0.30, 1.20)
    request = {"max_tokens": 16000, "messages": [{"role": "user", "content": "hi"}]}
    prompt_cost = caller.worst_cost("deepseek-v4-pro", request, 0)
    # no limit -> as wanted
    assert caller.max_tokens_within("deepseek-v4-pro", None, request, 16000) == 16000
    # zero budget -> 0
    assert caller.max_tokens_within("deepseek-v4-pro", 0.0, request, 16000) == 0
    # exactly the prompt cost leaves no room for output
    assert caller.max_tokens_within("deepseek-v4-pro", prompt_cost, request, 16000) == 0
    # just under MIN_OUTPUT_TOKENS -> 0 (never send a request that can't afford a usable answer)
    assert caller.max_tokens_within("deepseek-v4-pro", prompt_cost + 1023.5 * price_out / 1e6, request, 16000) == 0
    # just over -> MIN_OUTPUT_TOKENS
    assert caller.max_tokens_within("deepseek-v4-pro", prompt_cost + 1024.5 * price_out / 1e6, request, 16000) == \
        MIN_OUTPUT_TOKENS
    # roomier -> the floor of what fits
    assert caller.max_tokens_within("deepseek-v4-pro", prompt_cost + 5000.5 * price_out / 1e6, request, 16000) == 5000
    # worst cost: prompt at the uncached input rate plus output, no 1.25x cache-write premium
    prompt = len(json.dumps({k: v for k, v in request.items() if k != "max_tokens"}, default=str)) / 3
    assert caller.worst_cost("deepseek-v4-pro", request, 100) == \
        pytest.approx((prompt * 1.32 + 100 * 3.96) / 1e6)


def test_remaining_error_paths_and_non_json_answer() -> None:
    # 400/403/404/422 are permanent request errors (an unknown model, a refused request): temporary=False
    for status in (400, 403, 404, 422):
        caller = DeepSeekCaller("sk-ds-test", session=FakeSession(status=status, payload={}))
        with pytest.raises(AIError) as caught:
            caller.create("deepseek-v4-pro", messages=[{"role": "user", "content": "hi"}])
        assert caught.value.temporary is False and f"status {status}" in str(caught.value)
    # 429 and 5xx stay temporary
    for status in (429, 500, 503):
        caller = DeepSeekCaller("sk-ds-test", session=FakeSession(status=status, payload={}))
        with pytest.raises(AIError) as caught:
            caller.create("deepseek-v4-pro", messages=[{"role": "user", "content": "hi"}])
        assert caught.value.temporary is True
    # a transport error maps to a plain, temporary AIError
    caller = DeepSeekCaller("sk-ds-test", session=FakeSession(raise_exc=requests.ConnectionError("boom")))
    with pytest.raises(AIError) as caught:
        caller.create("deepseek-v4-pro", messages=[{"role": "user", "content": "hi"}])
    assert caught.value.temporary is True and "Couldn't reach DeepSeek" in str(caught.value)
    # a 200 with a non-JSON body is its own error, not a crash
    caller = DeepSeekCaller("sk-ds-test", session=FakeSession(response=BadJsonResponse(200, "html")))
    with pytest.raises(AIError) as caught:
        caller.create("deepseek-v4-pro", messages=[{"role": "user", "content": "hi"}])
    assert "wasn't JSON" in str(caught.value)


def test_a_refusal_only_message_yields_an_empty_refusal() -> None:
    # OpenAI-style refusals set message.refusal and an empty content; DeepSeek's finish_reason drives the
    # stop_reason. The refusal text is not copied into content today.
    response = DeepSeekCaller._response(payload(finish_reason="content_filter", content=None,
                                                refusal="I can't help with that"))
    assert response.stop_reason == "refusal" and response.content == []


# --- (3) provider selection matrix --------------------------------------------------------------------

def test_provider_selection_matrix() -> None:
    # neither key -> chat off
    assert build_assistant(Config(TELEGRAM_BOT_TOKEN="1:x", DEEPSEEK_API_KEY=None, ANTHROPIC_API_KEY=None),
                           object(), object()) is None
    # DeepSeek only -> DeepSeekCaller with the defaults
    built = build_assistant(Config(TELEGRAM_BOT_TOKEN="1:x", DEEPSEEK_API_KEY="sk-ds-test",
                                   ANTHROPIC_API_KEY=None), object(), object())
    assert isinstance(built.caller, DeepSeekCaller)
    assert (built.chat_model, built.router_model, built.reader.model) == \
        ("deepseek-v4-pro", "deepseek-flash", "deepseek-flash")
    assert built.reader.vision is False and built.disclosure == assistant_module.DEEPSEEK_DISCLOSURE
    # both keys -> DeepSeek wins
    both = build_assistant(Config(TELEGRAM_BOT_TOKEN="1:x", DEEPSEEK_API_KEY="sk-ds-test",
                                  ANTHROPIC_API_KEY="sk-ant-test"), object(), object())
    assert isinstance(both.caller, DeepSeekCaller)
    # Claude only -> the Claude path is intact (a fake client means no SDK import)
    claude = build_assistant(Config(TELEGRAM_BOT_TOKEN="1:x", DEEPSEEK_API_KEY=None,
                                    ANTHROPIC_API_KEY="sk-ant-test"), object(), object(), client=object())
    assert isinstance(claude.caller, ClaudeCaller)
    assert claude.disclosure == assistant_module.DISCLOSURE and claude.reader.vision is True


def test_services_chat_assistant_gate_off_and_deepseek() -> None:
    from telegram_betting_bot_simple.services import build_chat_assistant
    assert build_chat_assistant(Config(TELEGRAM_BOT_TOKEN="1:x", DEEPSEEK_API_KEY=None,
                                       ANTHROPIC_API_KEY=None), None, None) is None
    built = build_chat_assistant(Config(TELEGRAM_BOT_TOKEN="1:x", DEEPSEEK_API_KEY="sk-ds-test",
                                        ANTHROPIC_API_KEY=None), object(), object())
    assert built is not None and isinstance(built.caller, DeepSeekCaller)


# --- (4) text-only vision fallback -------------------------------------------------------------------

def test_a_text_only_caller_refuses_an_image_block_before_any_request() -> None:
    session = FakeSession(payload=payload())
    caller = DeepSeekCaller("sk-ds-test", session=session)
    with pytest.raises(AIError) as caught:
        caller.create("deepseek-v4-pro",
                      messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64",
                                                                                            "data": "x"}}]}])
    assert "text-only" in str(caught.value) and caught.value.temporary is False
    assert session.requests == []  # refused before it ever hit the wire


def test_the_read_step_skips_a_screenshot_with_a_text_only_reader() -> None:
    built = build_assistant(Config(TELEGRAM_BOT_TOKEN="1:x", DEEPSEEK_API_KEY="sk-ds-test"),
                            object(), object())
    assert built.reader.vision is False
    paste = assistant_module.Paste(image=b"\xff\xd8\xff" + b"0" * 32)  # a JPEG head, never sent
    spend = assistant_module.Spend()
    outcome = built._read("tg:42", paste, spend, lambda _step: None)
    assert outcome == (None, "the AI reader is text-only, so it can't read a screenshot")
    assert spend.by_model == {}  # nothing charged, nothing called


def test_the_slip_reader_mirrors_the_callers_vision_flag() -> None:
    assert SlipReader(ClaudeCaller(object()), "claude-haiku-5-5").vision is True
    assert SlipReader(DeepSeekCaller("sk-ds-test", session=FakeSession()), "deepseek-flash").vision is False


# --- (5) the tool loop round-trips a tool_use into a final answer -----------------------------------

def test_the_tool_loop_round_trips_a_tool_use_into_a_final_answer() -> None:
    from telegram_betting_bot_simple.ai.assistant import Assistant

    class QueueSession:
        """Answers each call with the next scripted payload (no network)."""

        def __init__(self, responses):
            self.responses = list(responses)
            self.requests = []

        def post(self, url, json=None, timeout=None, headers=None):
            self.requests.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
            return FakeResponse(200, self.responses.pop(0))

    round1 = payload(finish_reason="tool_calls", content="Let me check.",
                     tool_calls=[{"id": "call_1", "type": "function",
                                  "function": {"name": "sports_find_events",
                                               "arguments": '{"leagues": ["basketball_nba"], "query": "knicks"}'}}],
                     usage={"prompt_tokens": 100, "completion_tokens": 20})
    round2 = payload(finish_reason="stop", content="The Knicks play the Celtics tonight.",
                     usage={"prompt_tokens": 120, "completion_tokens": 8})
    session = QueueSession([round1, round2])
    caller = DeepSeekCaller("sk-ds-test", session=session)
    assistant = Assistant(None, None, None, caller=caller, chat_model="deepseek-v4-pro",
                          router_model="deepseek-flash")

    class Toolbox:
        def run(self, name, args):
            return SimpleNamespace(name=name, args=args, ok=True, content={"games": 3})

    messages = [{"role": "user", "content": [{"type": "text", "text": "Knicks games tonight"}]}]
    runs = []
    text, content = assistant._loop(messages, "low", Toolbox(), runs, assistant_module.Spend(),
                                    lambda _step: None, rounds=2)

    # The turn completes with the final answer ...
    assert text == "The Knicks play the Celtics tonight."
    assert len(session.requests) == 2
    # ... and the round-2 body carries the assistant's tool_calls and the tool's result (not a null-content
    # assistant beside an orphaned tool message, which DeepSeek rejects with a 400)
    second = session.requests[1]["json"]["messages"]
    assistant_msg = next(message for message in second if message["role"] == "assistant")
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "sports_find_events"
    assert assistant_msg["tool_calls"][0]["function"]["arguments"] == '{"leagues": ["basketball_nba"], "query": "knicks"}'
    tool_msg = next(message for message in second if message["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1" and tool_msg["content"] == '{"games":3}'
