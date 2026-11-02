"""Paste a slip end to end over the API: the real engine (engine/pasted.py, engine/tickets.py), the real Lab and
Store, the real Assistant and SlipReader, and the real FastAPI app, with a fake Anthropic client only.

The fake client answers the three kinds of request the app sends, by what each one is: the slip reader's
(structured output with the Extraction schema) with a fixed Extraction, the router's with a fixed intent, and
the chat's tool loop by reading the request as Claude would (the pasted_id in its context, the tool results it
gets back). The odds and ESPN are the shared fakes (tests/conftest.py). No network, no Anthropic calls, no Odds
API quota.
"""

from __future__ import annotations

import base64
import json
import re
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.ai.assistant import (
    CONFIRM_CHIP,
    HELPLINE,
    PASTED_CHIPS,
    ROUTER_PROMPT,
    Assistant,
)
from telegram_betting_bot_simple.ai.slip_reader import READER_PROMPT
from telegram_betting_bot_simple.dashboard import lab as lab_module
from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.engine import pasted as engine

from test_lab import Players
from test_slip_reader import NOW, SLIP_TEXT, extraction

NY = __import__("zoneinfo").ZoneInfo("America/New_York")
ME, THEM = {"X-Player": "browser-1234"}, {"X-Player": "someone-else-99"}
UID = "web:browser-1234"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
WEAKEST = "what if I drop the weakest leg?"


# --- a fake Anthropic client ---------------------------------------------------------------------------------

def _usage(fresh: int = 1000, out: int = 300) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=fresh, output_tokens=out, cache_creation_input_tokens=0,
                           cache_read_input_tokens=0, server_tool_use=None)


def _message(content: List[Any], model: str, stop: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop, usage=_usage(), model=model)


def _text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def _plain(value: Any) -> Any:
    """A request as JSON (the SDK's content blocks are objects; the app sends Claude's own blocks back)."""
    return json.loads(json.dumps(value, default=lambda o: getattr(o, "__dict__", str(o))))


class FakeAnthropic:
    """client.beta.messages.create(**request) as the Anthropic SDK has it, answering by the kind of request."""

    def __init__(self, extractions: Optional[List[Dict[str, Any]]] = None, intent: str = "analyse") -> None:
        self.extractions = list(extractions or [extraction()])
        self.intent = intent
        self.requests: List[Dict[str, Any]] = []
        self.calls = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def kinds(self) -> List[str]:
        return [self._kind(request) for request in self.requests]

    def of(self, kind: str) -> List[Dict[str, Any]]:
        return [request for request in self.requests if self._kind(request) == kind]

    @staticmethod
    def _kind(request: Dict[str, Any]) -> str:
        system = (request.get("system") or [{}])[0].get("text")
        return "reader" if system == READER_PROMPT else "router" if system == ROUTER_PROMPT else "chat"

    def _create(self, **request: Any) -> SimpleNamespace:
        request = _plain(request)
        self.requests.append(request)
        kind = self._kind(request)
        if kind == "reader":
            assert request["output_config"]["format"] == {"type": "json_schema", "schema": engine.extraction_schema()}
            data = self.extractions.pop(0) if len(self.extractions) > 1 else self.extractions[0]
            return _message([_text(json.dumps(data))], request["model"])
        if kind == "router":
            route = {"intent": self.intent, "leagues": [], "stake": None, "target": None, "max_legs": None,
                     "tier": "none", "days": None, "min_odds": None, "legs_per_day": None, "players": [],
                     "risk_flags": []}
            return _message([_text(json.dumps(route))], request["model"])
        return self._chat(request)

    # The chat's tool loop, as Claude would play it: numbers only from this turn's tool results
    def _chat(self, request: Dict[str, Any]) -> SimpleNamespace:
        model = request["model"]
        last = request["messages"][-1]["content"]
        blocks = last if isinstance(last, list) else [{"type": "text", "text": last}]
        texts = [block.get("text") or "" for block in blocks if block.get("type") == "text"]
        fresh = next((t for t in texts if t.startswith("<pasted_slip_analysis>")), None)
        if fresh is not None:  # a slip just read (or confirmed): code's analysis is in the request
            facts = json.loads(fresh[len("<pasted_slip_analysis>"):-len("</pasted_slip_analysis>")])
            return _message([_text(self._verdict(facts))], model)
        results = [block for block in blocks if block.get("type") == "tool_result"]
        if results:
            facts = json.loads(results[-1]["content"])
            if not facts.get("left_out"):  # the slip as pasted: now leave its weakest leg out
                self.calls += 1
                weakest = facts["ticket"]["weakest_leg"]
                return self._tool({"pasted_id": facts["slip"]["pasted_id"], "drop": [weakest]}, model)
            before = self._first_result(request)
            return _message([_text(f"Without leg {facts['left_out'][0]} the market's chance that it all wins goes "
                                   f"from 1 in {before['ticket']['chance']['market_one_in']} to 1 in "
                                   f"{facts['ticket']['chance']['market_one_in']}.")], model)
        context = texts[0] if texts else ""
        found = re.search(r"pasted_id (\d+)", context)
        if found and WEAKEST in texts[-1]:  # first look at the slip as pasted
            self.calls += 1
            return self._tool({"pasted_id": int(found.group(1))}, model)
        return _message([_text("The bookmaker builds a margin into every price.")], model)

    @staticmethod
    def _verdict(facts: Dict[str, Any]) -> str:
        ticket = facts["ticket"]
        weakest = next(leg for leg in ticket["legs"] if leg["leg"] == ticket["weakest_leg"])
        return (f"By the market this wins about 1 in {ticket['chance']['market_one_in']}. The weakest leg is leg "
                f"{weakest['leg']}, {weakest['label']}.")

    @staticmethod
    def _first_result(request: Dict[str, Any]) -> Dict[str, Any]:
        for message in request["messages"]:
            for block in message["content"] if isinstance(message["content"], list) else []:
                if block.get("type") == "tool_result":
                    return json.loads(block["content"])
        raise AssertionError("no tool result yet")

    def _tool(self, args: Dict[str, Any], model: str) -> SimpleNamespace:
        block = SimpleNamespace(type="tool_use", id=f"toolu_{self.calls}", name="pasted_slip_analyse", input=args)
        return _message([block], model, stop="tool_use")


# --- the app ---------------------------------------------------------------------------------------------------

def app_for(service: Any, tmp_path: Any, claude: FakeAnthropic) -> SimpleNamespace:
    store = Store(str(tmp_path / "betting.db"))
    lab = lab_module.Lab(service, store, players=Players(game_date="2030-01-01T00:00:00Z"), now=lambda: NOW)
    lab.assistant = Assistant(claude, lab, store, daily_cents=100, tz=NY, clock=lambda: NOW)
    client = TestClient(create_app(service, "s3cret", login_delay=0, lab=lab))
    assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
    client.headers.update(ME)
    for headers in (ME, THEM):
        assert client.put("/api/me", headers=headers, json={"confirm_age": True}).status_code == 200
    return SimpleNamespace(client=client, lab=lab, store=store, claude=claude)


@pytest.fixture()
def make(service, tmp_path) -> Callable[..., SimpleNamespace]:
    return lambda claude=None: app_for(service, tmp_path, claude or FakeAnthropic())


def pasted_card(body: Dict[str, Any]) -> Dict[str, Any]:
    [card] = [card for card in body["reply"]["cards"] if card["type"] == "pasted"]
    return card


# --- the flows -------------------------------------------------------------------------------------------------

def test_pasted_text_is_read_by_claude_then_matched_priced_and_explained_by_code(make) -> None:
    app = make()
    response = app.client.post("/api/chat/slip", json={"text": SLIP_TEXT})
    assert response.status_code == 200
    body = response.json()
    reply = body["reply"]
    assert reply["kind"] == "answer" and reply["chips"] == PASTED_CHIPS
    assert reply["text"].startswith("I read your slip as a 3-leg multi at DraftKings, $10.00 stake, a return of "
                                    "$33.21 shown:\n1. New York Knicks to win at 1.36")
    # Claude's words passed the numbers check: "1 in 4" is the code's own chance, leg 3 the slip's weakest
    assert reply["text"].endswith("By the market this wins about 1 in 4. The weakest leg is leg 3, LeBron James over "
                                  "27.5 points.")
    card = pasted_card(body)
    view, analysis = card["pasted"], card["analysis"]
    assert [leg["match"]["status"] for leg in view["legs"]] == ["matched", "matched", "prop"]
    assert view["return_check"] == "matches" and view["computed_return"] == "33.24" and not view["needs_confirmation"]
    assert analysis["ticket"]["valid"] and [leg["slip_index"] for leg in analysis["ticket"]["legs"]] == [1, 2, 3]
    # The cheap model read it, with the Extraction schema and the text quoted as data; then one chat answer
    assert app.claude.kinds() == ["reader", "chat"]
    [read] = app.claude.of("reader")
    assert read["model"] == "claude-haiku-5-5"
    assert "<slip>\nDraftKings\n3-leg parlay" in read["messages"][0]["content"][0]["text"]
    # The card's toggles re-price it for free, and the row is the person's own
    pasted_id = view["id"]
    assert app.client.get(f"/api/pasted/{pasted_id}").json()["legs"] == view["legs"]
    toggled = app.client.post(f"/api/pasted/{pasted_id}/analyse", json={"drop": [3]}).json()
    assert toggled["drop"] == [3] and [leg["slip_index"] for leg in toggled["ticket"]["legs"]] == [1, 2]
    assert app.store.pasted(UID, pasted_id)["thread_id"] == body["thread_id"]
    assert len(app.claude.requests) == 2  # the toggles cost no AI
    # Both calls were paid from the allowance in one hold
    assert app.client.get("/api/me").json()["credits"]["used_cents"] > 0


def test_a_screenshot_is_read_as_an_image_and_never_kept(make) -> None:
    app = make()
    image = {"media_type": "image/jpeg", "data": base64.b64encode(PNG).decode()}
    body = app.client.post("/api/chat/slip", json={"image": image}).json()
    assert body["reply"]["kind"] == "answer"
    [read] = app.claude.of("reader")
    blocks = read["messages"][0]["content"]
    assert blocks[0] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image["data"]}}
    view = pasted_card(body)["pasted"]
    assert view["source"] == "image" and all(leg["verified"] is False for leg in view["legs"])
    row = app.store.pasted(UID, view["id"])
    assert row["source"] == "image" and row["text"] is None
    assert image["data"] not in json.dumps(row) and image["data"] not in json.dumps(app.store.messages(UID, body["thread_id"]))
    assert app.store.messages(UID, body["thread_id"])[0]["content"]["text"] == "📋 My bet slip (screenshot)"


def test_a_follow_up_leaves_out_the_weakest_leg_by_its_number_on_the_slip(make) -> None:
    app = make()
    first = app.client.post("/api/chat/slip", json={"text": SLIP_TEXT}).json()
    thread = first["thread_id"]
    body = app.client.post("/api/chat", json={"thread_id": thread, "message": WEAKEST}).json()
    reply = body["reply"]
    # Claude looked at the slip, then left out the weakest leg the tool named: leg 3 on the slip (LeBron), which
    # the ticket itself counts as its leg 2 from 0
    calls = [block for request in app.claude.of("chat") for message in request["messages"] if message["role"] == "assistant"
             for block in (message["content"] if isinstance(message["content"], list) else []) if block.get("type") == "tool_use"]
    assert [call["input"] for call in calls][-2:] == [{"pasted_id": 1}, {"pasted_id": 1, "drop": [3]}]
    assert reply["kind"] == "answer" and reply["text"] == ("Without leg 3 the market's chance that it all wins goes "
                                                           "from 1 in 4 to 1 in 2.")
    cards = [card for card in reply["cards"] if card["type"] == "pasted"]
    assert [card["analysis"]["drop"] for card in cards] == [[], [3]]
    kept = [leg["label"] for leg in cards[-1]["analysis"]["ticket"]["legs"]]
    assert kept == ["New York Knicks to win", "Oklahoma City Thunder to win"]
    # The follow-up's context named the slip; the pasted text itself stayed out of Claude's view
    follow = app.claude.of("chat")[-1]["messages"]
    assert follow[0]["content"] == "[I pasted a bet slip; the reply below starts with what the app read from it]"
    assert "New York Knicks ML 1.36" not in json.dumps(follow)


def test_a_doubtful_read_asks_first_and_yes_gives_the_analysis(make) -> None:
    doubtful = extraction()
    doubtful["legs"][1]["odds_text"] = "-800"  # not what the pasted text says
    app = make(FakeAnthropic([doubtful]))
    first = app.client.post("/api/chat/slip", json={"text": SLIP_TEXT}).json()
    reply = first["reply"]
    assert reply["kind"] == "confirm" and reply["chips"] == [CONFIRM_CHIP, "Let me fix it"]
    assert "The odds read for leg 2 (-800) aren't in the text you pasted." in reply["text"]
    assert pasted_card(first)["pasted"]["needs_confirmation"] and app.claude.kinds() == ["reader"]  # no AI words yet
    second = app.client.post("/api/chat", json={"thread_id": first["thread_id"], "message": CONFIRM_CHIP}).json()
    assert second["reply"]["kind"] == "answer" and app.claude.kinds() == ["reader", "chat"]  # no router either
    view = pasted_card(second)["pasted"]
    assert view["confirmed"] and not view["needs_confirmation"] and view["id"] == pasted_card(first)["pasted"]["id"]


def test_a_slip_typed_into_the_chat_box_is_read_as_one(make) -> None:
    app = make()
    body = app.client.post("/api/chat", json={"message": SLIP_TEXT}).json()
    assert body["reply"]["text"].startswith("I read your slip as") and pasted_card(body)
    assert app.claude.kinds() == ["reader", "chat"]  # code spotted the slip: no router call


def test_the_free_reader_reads_and_prices_with_no_ai(make) -> None:
    app = make()
    body = app.client.post("/api/pasted", json={"text": SLIP_TEXT}).json()
    assert body["pasted"]["source"] == "basic" and body["analysis"]["ticket"]["legs"]
    assert not app.claude.requests


def test_in_the_pause_a_paste_gets_the_codes_numbers_and_no_ai_words(make) -> None:
    app = make()
    support = app.client.post("/api/chat", json={"message": "I lost my rent money and need to win it back"}).json()
    assert support["reply"]["kind"] == "support" and app.client.get("/api/me").json()["picks_paused_until"]
    before = len(app.claude.requests)
    body = app.client.post("/api/chat/slip", json={"text": SLIP_TEXT}).json()
    reply = body["reply"]
    assert reply["kind"] == "paused" and len(app.claude.requests) == before  # no Claude call at all
    assert reply["text"].startswith("I read your slip as") and "Picks are paused until" in reply["text"]
    assert reply["text"].endswith(HELPLINE)
    analysis = pasted_card(body)["analysis"]
    assert analysis["ticket"]["legs"] and analysis["alternatives"] == [] and analysis["paused_until"]
    assert pasted_card(body)["pasted"]["source"] == "basic"  # read by code: the AI doesn't read in the pause
    screenshot = app.client.post("/api/chat/slip", json={"image": {"data": base64.b64encode(PNG).decode()}}).json()
    assert screenshot["reply"]["kind"] == "paused" and len(app.claude.requests) == before


def test_on_a_break_pasting_is_refused_everywhere(make) -> None:
    app = make()
    pasted_id = app.client.post("/api/pasted", json={"text": SLIP_TEXT}).json()["pasted"]["id"]
    assert app.client.post("/api/me/cooloff", json={"hours": 24}).status_code == 200
    reply = app.client.post("/api/chat/slip", json={"text": SLIP_TEXT}).json()["reply"]
    assert reply["kind"] == "gate" and "on a break" in reply["text"] and not app.claude.requests
    for response in (app.client.post("/api/pasted", json={"text": SLIP_TEXT}),
                     app.client.get(f"/api/pasted/{pasted_id}"),
                     app.client.post(f"/api/pasted/{pasted_id}/analyse", json={})):
        assert response.status_code == 400 and "break" in response.json()["detail"]


def test_another_person_gets_a_404_for_a_pasted_slip_and_its_chat(make) -> None:
    app = make()
    body = app.client.post("/api/chat/slip", json={"text": SLIP_TEXT}).json()
    pasted_id, thread = pasted_card(body)["pasted"]["id"], body["thread_id"]
    for response in (app.client.get(f"/api/pasted/{pasted_id}", headers=THEM),
                     app.client.post(f"/api/pasted/{pasted_id}/analyse", headers=THEM, json={"drop": [1]}),
                     app.client.post("/api/chat/slip", headers=THEM, json={"text": SLIP_TEXT, "thread_id": thread}),
                     app.client.post("/api/pasted", headers=THEM, json={"text": SLIP_TEXT, "thread_id": thread})):
        assert response.status_code == 404
    assert app.client.get(f"/api/pasted/{pasted_id}").status_code == 200
