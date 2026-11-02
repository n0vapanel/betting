"""Closest effort across its three parts (v3 integration): the engine (engine/builder.py, presets.py), the wiring
(dashboard/lab.py, app.py, ai/tools.py, ai/assistant.py, ai/explain.py, bot/chat.py) and the app
(dashboard/static/js/views/build/builder/*, views/ask/cards.js, ui.js and the help), over the real modules.

The world is test_builder_wiring's: a real Store, the real SlateBuilder over fake ESPN boards, the real games
service, Lab and routes, the saved NBA and EPL boards. The only fakes are at the edges (the boards' feed, ESPN).
These tests hold the parts to one contract: the same labels everywhere, every field the app and the bot read is
one the server sends, Ask AI and the Builder show the same tickets with the same numbers, and the analyst's words
built from the tool's facts pass the real post-check. No network."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest
import requests

from telegram_betting_bot_simple.ai import tools as tools_module
from telegram_betting_bot_simple.ai.explain import banned_phrases, pct, problems, source_numbers
from telegram_betting_bot_simple.ai.tools import Toolbox
from telegram_betting_bot_simple.bot import chat as bot_chat
from telegram_betting_bot_simple.dashboard import app as app_module
from telegram_betting_bot_simple.dashboard import lab as lab_module
from telegram_betting_bot_simple.data.odds_service import configure_books
from telegram_betting_bot_simple.engine import builder as engine
from telegram_betting_bot_simple.engine import presets
from tests.test_builder_wiring import NBA, NY, PLAYER_A, World, make_world, run


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(requests.Session, "request", refuse)


@pytest.fixture(autouse=True)
def default_roles():
    configure_books()
    yield
    configure_books()


@pytest.fixture()
def world(tmp_path) -> World:
    w = make_world(tmp_path)
    w.client()  # PLAYER_A confirms their age
    return w


JS = Path(app_module.__file__).resolve().parent / "static" / "js"
UID = "web:" + PLAYER_A["X-Player"]
TRANSCRIPT = {"leagues": [NBA], "legs": 2, "target_odds": 3.5, "target_chance": 0.8}
SAFE_TRANSCRIPT = {**TRANSCRIPT, "preset": "safe"}


def read(name: str) -> str:
    return (JS / name).read_text(encoding="utf-8")


def js_object(source: str, name: str) -> Dict[str, Any]:
    """``const NAME = {...};`` of a JS file, read as JSON (keys quoted, no trailing commas)."""
    found = re.search(rf"const {name} = (\{{.*?\}});", source, re.S)
    assert found, name
    text = re.sub(r"([{,]\s*)([A-Za-z_]\w*)\s*:", r'\1"\2":', found.group(1))
    return json.loads(re.sub(r",\s*}", "}", text))


# --- one contract: the labels and the words -----------------------------------------------------------------------

def test_the_option_labels_are_the_engine_s_in_the_lab_the_tools_and_the_app() -> None:
    assert lab_module.OPTION_LABELS == engine.OPTION_LABELS == tools_module.OPTION_LABELS
    assert list(engine.OPTION_LABELS) == list(engine.OPTION_KEYS)
    assert js_object(read("views/build/builder/fit.js"), "OPTION_TITLES") == engine.OPTION_LABELS
    # The app's best_effort is a scope field, as the engine's presets say (the switch keeps the preset's name)
    assert "best_effort" in presets.SCOPE_FIELDS and "target_chance" not in presets.SCOPE_FIELDS
    assert '"best_effort"' in read("views/build/builder/model.js").split("export const SCOPE_FIELDS", 1)[1][:400]
    # Help has the two new terms, linked from the Builder's guide, with Safe's own floor in its example
    ui, content, help_page = read("ui.js"), read("views/help/content.js"), read("views/help.js")
    for term in ("target_chance", "closest_effort"):
        assert re.search(rf"^  {term}: {{", ui, re.M) and re.search(rf"^  {term}: {{", content, re.M), term
        assert f"#/help/{term}" in help_page, term
    floor = presets.PRESETS["safe"].multi_leg_min
    assert f"per-leg chance lowered from {floor:.0%}" in ui and f"{floor:.0%} or more" in content


# --- every field the app and the bot read is one the server sends ---------------------------------------------------

FIT_KEYS = {"odds", "chance", "target_odds", "target_chance", "odds_gap_pct", "chance_gap_pp", "meets"}


def test_the_route_sends_every_closest_effort_field_the_app_reads(world) -> None:
    client = world.client()
    closest = run(client, SAFE_TRANSCRIPT)
    strict = run(client, {**SAFE_TRANSCRIPT, "best_effort": False})
    # What the app reads (fit.js, results.js, builder.js), read from its source
    fit_js, results_js, builder_js = (read("views/build/builder/fit.js"), read("views/build/builder/results.js"),
                                      read("views/build/builder.js"))
    for key in ("odds", "chance", "target_odds", "target_chance", "meets"):
        assert f"fit.{key}" in fit_js, key
    assert "meets.odds" in fit_js and "meets.chance" in fit_js and "entry?.words" in fit_js
    assert "option.ticket_index" in fit_js and "result?.options" in fit_js and "ticket?.option" in fit_js
    assert "result.controls?.best_effort === true" in results_js and "result.controls?.best_effort === false" in results_js
    assert "run.data.controls?.best_effort" in builder_js and "ran.target_chance" in builder_js
    # ... and what the route sends
    assert closest["controls"]["best_effort"] is True and strict["controls"]["best_effort"] is False
    assert closest["controls"]["target_chance"] == 0.8 and closest["label"] == "Custom"
    assert [o["key"] for o in closest["options"]] == list(engine.OPTION_KEYS)
    assert all(set(o) == {"key", "label", "ticket_index"} for o in closest["options"])
    for i, ticket in enumerate(closest["tickets"]):
        assert set(ticket["fit"]) == FIT_KEYS and set(ticket["fit"]["meets"]) == {"odds", "chance"}
        assert ticket["option"] == {"key": closest["options"][i]["key"], "label": closest["options"][i]["label"]}
        assert all(set(entry) == {"rule", "from", "to", "words"} and entry["words"] for entry in ticket["relaxed"])
    # Strict is today's answer: nothing built, the blocked card's message, no closest-effort fields filled
    assert not strict["tickets"] and strict["options"] == [] and strict["blocked"]["message"].startswith(
        "No 2-leg ticket near 3.5x at an 80%+ chance")
    assert "relaxation" not in strict["stats"]
    # A swap re-priced by /api/builder/price carries the fit the app shows for it
    ticket = closest["tickets"][0]
    refs = [{key: leg[key] for key in ("league", "market", "side", "point")} | {"id": leg["event_id"]}
            for leg in ticket["legs"]]
    priced = client.post("/api/builder/price", json={"legs": refs, "target_odds": 3.5, "target_chance": 0.8})
    assert priced.status_code == 200 and priced.json()["fit"] == ticket["fit"]


def test_the_saved_switch_and_chance_come_back_to_the_app(world) -> None:
    client = world.client()
    saved = client.put("/api/me", json={"builder": {"preset": "safe", "target_chance": 0.7, "best_effort": False,
                                                    "legs": 2}})
    assert saved.status_code == 200, saved.text
    back = client.get("/api/builder/presets").json()["defaults"]
    assert back["saved"]["best_effort"] is False and back["saved"]["target_chance"] == 0.7
    assert back["controls"]["best_effort"] is True and back["controls"]["target_chance"] is None
    assert "best_effort" in back["scope_fields"]


PICK_KEYS = {"rank", "ref", "refs", "league", "label", "game", "start", "chance", "chance_source", "price", "book",
             "edge", "why"}


def test_the_picks_card_carries_every_field_the_app_and_the_bot_read(world) -> None:
    done = Toolbox(world.lab, UID, NY, leagues={NBA}).run("brain_top_picks", {"leagues": [NBA], "n": 5})
    assert done.ok, done.content
    [card] = done.cards
    cards_js = read("views/ask/cards.js")
    for key in PICK_KEYS - {"refs", "league"}:
        assert f"pick.{key}" in cards_js or f"pick?.{key}" in cards_js, key
    sources = js_object(cards_js, "PICK_SOURCES")
    for pick in card["picks"]:
        assert PICK_KEYS <= set(pick), set(pick)
        assert pick["chance_source"] in sources  # the app names every source the engine gives
        assert pick["refs"] == [pick["ref"]] and pick["ref"]["market"] in ("h2h", "spreads", "totals")
        assert {"league", "id", "market", "side"} <= set(pick["ref"])  # pickLeg's and the bot's add_pick's needs
        assert isinstance(pick["price"], str) and float(pick["price"]) > 1
    # The facts Claude reads are the card's numbers
    assert [(f["rank"], f["pick"], f["odds"], f["book"]) for f in done.content["picks"]] == [
        (p["rank"], p["label"], p["price"], p["book"]) for p in card["picks"]]
    # The bot: the same card as text, and an "Add pick" button its own handler's pattern takes
    text = bot_chat.card_text(card)
    first = card["picks"][0]
    assert text.startswith(f"🏆 <b>{card['title']}</b>") and f"1. {first['label']} · {pct(first['chance'])}" in text
    markup = bot_chat.reply_markup({"thread_id": 7, "message_idx": 2, "reply": {"cards": [card], "chips": []}}, None,
                                   False)
    data = [button.callback_data for row in markup.inline_keyboard for button in row]
    pattern = re.compile(r"^ai:pick:\d{1,12}:\d{1,6}:\d:\d$")  # bot/chat.py setup_chat's add_pick pattern
    assert data and all(pattern.match(item) for item in data)
    assert r'pattern=r"^ai:pick:\d{1,12}:\d{1,6}:\d:\d$"' in Path(bot_chat.__file__).read_text(encoding="utf-8")


# --- Ask AI and the Builder show the same tickets, and the analyst's words pass the post-check ----------------------

def test_ask_ai_and_the_builder_show_the_same_closest_tickets_with_the_same_numbers(world) -> None:
    route = run(world.client(), SAFE_TRANSCRIPT)
    done = Toolbox(world.lab, UID, NY, leagues={NBA}).run("ticket_build", SAFE_TRANSCRIPT)
    assert done.ok, done.content
    facts = done.content
    assert [card["title"] for card in done.cards] == [o["label"] for o in route["options"]]
    for card, ticket in zip(done.cards, route["tickets"]):
        assert card["fit"] == ticket["fit"] and card["relaxed"] == ticket["relaxed"]
        assert card["option"] == ticket["option"] and card["refs"] == ticket["refs"]
        assert card["ticket"]["odds"] == ticket["odds"]
    # The facts Claude may quote: each option's fit and its own relaxed words (the options loosen the same rule to
    # different steps, so there is no list of every ticket's words to misattribute)
    assert [(o["option"], o["ticket"], o["fit"], o.get("relaxed", [])) for o in facts["options"]] == [
        (o["label"], o["ticket_index"] + 1, tools_module.fit_facts(route["tickets"][o["ticket_index"]]["fit"]),
         [e["words"] for e in route["tickets"][o["ticket_index"]]["relaxed"]]) for o in route["options"]]
    assert "relaxed" not in facts and "each option's own relaxed" in facts["hint"]
    for option in facts["options"]:
        assert facts["tickets"][option["ticket"] - 1].get("relaxed") == option.get("relaxed")
    # The bot shows each card's fit line and relaxed words as the app does
    for card in done.cards:
        text = bot_chat.card_text(card)
        assert f"🎯 {card['fit']['odds']:.2f}x · {pct(card['fit']['chance'])} (you asked 3.5x · 80%)" in text
        for entry in card["relaxed"]:
            assert entry["words"] in text


def test_the_analyst_s_answer_from_the_facts_passes_the_real_post_check(world) -> None:
    box = Toolbox(world.lab, UID, NY, leagues={NBA})
    built = box.run("ticket_build", SAFE_TRANSCRIPT)
    picks = box.run("brain_top_picks", {"leagues": [NBA], "n": 3})
    facts = built.content
    lines = []
    for option in facts["options"]:
        ticket = facts["tickets"][option["ticket"] - 1]
        names = " and ".join(leg["label"] for leg in ticket["legs"])
        lines.append(f"{option['option']}: {names}, {option['fit']['odds']:.2f}x, wins about "
                     f"{option['fit']['chance'] * 100:.0f}% of the time"
                     + (f" ({'; '.join(option['relaxed'])})." if option.get("relaxed") else "."))
    top = picks.content["picks"][0]
    lines.append(f"The strongest pick tonight is {top['pick']}, the most likely winner at {top['chance'] * 100:.0f}% "
                 f"({float(top['odds']):.2f} at {top['book']}).")
    answer = " ".join(lines)
    sources = source_numbers(facts, picks.content)
    assert problems(answer, sources) == [], answer
    assert not banned_phrases(answer)
    # The number asked for but never built is not one the analyst may claim for a ticket
    assert problems("Likeliest at your odds wins about 80% of the time at 3.18x.", source_numbers(picks.content))
