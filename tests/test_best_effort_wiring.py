"""Closest effort and the analyst persona, wiring: POST /api/builder/run's options, fit and relaxed, ticket_build's
target_chance and best_effort and its facts, the brain_top_picks tool and its card, the router's target_chance, legs
and top-picks look-up, _clarify never asking a stake for odds or a chance, the rewritten system prompt, the templates
and the post-check, and the bot's AI chat and AI bets.

The transcript that started it: "2 leg NBA ticket around 3.5 odds with 80% chance" was refused. Here a fake Claude
runs it through the real assistant: it routes to build (target odds 3.5, target chance 0.8, legs 2), calls
ticket_build and answers with ticket cards and the closest options, never a refusal template.

The world is tests/test_builder_wiring.py's (a real Store, slate, games service and Lab; saved NBA and EPL boards)
over the real engine (engine/builder.py's closest effort and top_picks). No network (requests' Session.request fails
loudly)."""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import requests

from telegram_betting_bot_simple.ai import assistant as assistant_module
from telegram_betting_bot_simple.ai.assistant import Assistant, parse_route
from telegram_betting_bot_simple.ai.explain import (
    EXPLAIN_PROMPT,
    banned_phrases,
    builder_template,
    first_reason,
    leg_name,
    picks_template,
    problems,
    source_numbers,
    template_explanation,
)
from telegram_betting_bot_simple.ai.tools import TOOLS, Toolbox, builder_ticket_facts
from telegram_betting_bot_simple.dashboard.lab import Lab, best_effort_view
from telegram_betting_bot_simple.data.odds_service import configure_books
from tests.test_builder_wiring import NBA, NHL, NOW, NY, PLAYER_A, make_world, run

TRANSCRIPT = "2 leg NBA ticket around 3.5 odds with 80% chance"
LABELS = {"your_odds": "Likeliest at your odds", "your_chance": "Closest to your chance", "closest": "Best balance"}
UID = "web:" + PLAYER_A["X-Player"]


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
def world(tmp_path):
    w = make_world(tmp_path)
    w.client()  # PLAYER_A confirms their age
    return w


def transcript_args() -> Dict[str, Any]:
    return {"leagues": [NBA], "legs": 2, "target_odds": 3.5, "target_chance": 0.8}


# Three Safe legs in this world: Safe's 72% floor for a multi's legs and its 60% band can't be met, so closest effort
# loosens them and says so
LOOSENED = {"leagues": [NBA], "preset": "safe", "legs": 3}


# --- the Lab and the route: options, fit and relaxed ------------------------------------------------------------

def test_best_effort_view_labels_options_and_fills_fit_and_relaxed() -> None:
    result = {"tickets": [{"rank": 1, "fit": {"odds": 3.4}}, {"rank": 2, "relaxed": [{"rule": "min_chance",
                                                                                     "words": "per-leg chance lowered "
                                                                                              "from 60% to 55%"}]}],
              "options": [{"key": "your_odds", "ticket_index": 0}, {"key": "your_chance", "label": "Mine",
                                                                    "ticket_index": 1},
                          {"key": "closest", "ticket_index": 0}, {"key": "closest", "ticket_index": 7}, "junk"]}
    out = best_effort_view(result)
    assert [(o["key"], o["label"], o["ticket_index"]) for o in out["options"]] == [
        ("your_odds", "Likeliest at your odds", 0), ("your_chance", "Mine", 1), ("closest", "Best balance", 0)]
    first, second = out["tickets"]
    assert first["option"] == {"key": "your_odds", "label": "Likeliest at your odds"}  # the first label is kept
    assert second["option"] == {"key": "your_chance", "label": "Mine"}
    assert first["relaxed"] == [] and second["fit"] is None and second["relaxed"][0]["rule"] == "min_chance"
    plain = best_effort_view({"tickets": [{"rank": 1}]})
    assert plain["options"] == [] and plain["tickets"][0] == {"rank": 1, "fit": None, "relaxed": [], "option": None}


def test_the_builder_route_takes_target_chance_and_best_effort_and_sends_the_shapes(world) -> None:
    client = world.client()
    out = run(client, {"preset": "risky", "legs": 2, "target_odds": 3.5, "target_chance": 0.8, "best_effort": True})
    assert isinstance(out["options"], list)
    assert out["tickets"] and all({"fit", "relaxed", "option"} <= set(t) for t in out["tickets"])
    strict = run(client, {"preset": "risky", "best_effort": False})
    assert strict["options"] == [] and all(t["relaxed"] == [] and t["option"] is None for t in strict["tickets"])
    for bad in ({"target_chance": 1.5}, {"target_chance": 0}, {"best_effort": "maybe"}):
        assert client.post("/api/builder/run", json={"controls": bad}).status_code == 422
    # The person's saved controls keep both
    saved = client.put("/api/me", json={"builder": {"preset": "risky", "target_chance": 0.7, "best_effort": False}})
    assert saved.status_code == 200, saved.text


def test_the_transcript_runs_on_the_route_with_three_labelled_options(world) -> None:
    out = run(world.client(), {"leagues": [NBA], "legs": 2, "target_odds": 3.5, "target_chance": 0.8})
    assert [(option["key"], option["label"]) for option in out["options"]] == list(LABELS.items())
    assert sorted(option["ticket_index"] for option in out["options"]) == [0, 1, 2]
    for option in out["options"]:
        ticket = out["tickets"][option["ticket_index"]]
        fit = ticket["fit"]
        assert (fit["target_odds"], fit["target_chance"]) == (3.5, 0.8)
        assert fit["chance"] == pytest.approx(ticket["joint"]["chance"], abs=1e-4)  # the engine's own numbers
        assert fit["odds"] == ticket["odds"]["decimal"] and ticket["option"] == {"key": option["key"],
                                                                                "label": option["label"]}
    strict = run(world.client(), {"leagues": [NBA], "legs": 2, "target_odds": 3.5, "target_chance": 0.8,
                                  "best_effort": False})
    assert strict["options"] == [] and all(not t["relaxed"] for t in strict["tickets"])
    loosened = run(world.client(), LOOSENED)
    assert loosened["tickets"] and all(t["relaxed"] and t["fit"] is None for t in loosened["tickets"])
    assert all(entry["words"] for t in loosened["tickets"] for entry in t["relaxed"])


# --- ticket_build: the new inputs and the facts -------------------------------------------------------------------

def test_ticket_build_takes_target_chance_and_best_effort_and_its_facts_carry_the_options(world) -> None:
    schema = next(tool for tool in TOOLS if tool["name"] == "ticket_build")
    assert "strict" not in schema
    props = schema["input_schema"]["properties"]
    assert {"target_chance", "best_effort"} <= set(props) and not {"target_chance", "best_effort"} & set(
        schema["input_schema"]["required"])
    box = Toolbox(world.lab, UID, NY, leagues={NBA})
    done = box.run("ticket_build", transcript_args())
    assert done.ok, done.content
    facts = done.content
    assert [o["option"] for o in facts["options"]] == list(LABELS.values())  # the three options, in order
    for option in facts["options"]:
        ticket = facts["tickets"][option["ticket"] - 1]
        assert ticket["option"] in LABELS.values() and option["fit"] == ticket["fit"]
        assert {"odds", "chance", "target_odds", "target_chance", "meets"} <= set(option["fit"])
    assert "closest options" in facts["hint"] and "Don't refuse" in facts["hint"]
    assert "per $10" in facts["hint"]  # no stake was given
    assert "80% chance" in done.summary and "about 3.5x" in done.summary
    # The cards: titled by their option, with the fit and the relaxed rules beside the ticket
    titles = [card["title"] for card in done.cards]
    assert set(titles) <= set(LABELS.values()) and titles
    assert all(card["builder"] and card["option"]["label"] == card["title"] and card["fit"] for card in done.cards)
    # Each option's facts carry that ticket's own relaxed words, never one list of every ticket's
    assert "relaxed" not in facts and "each option's own relaxed" in facts["hint"]
    for option in facts["options"]:
        assert option.get("relaxed") == facts["tickets"][option["ticket"] - 1].get("relaxed")
    # Loosened rules: the closest tickets, each with the rules relaxed for it, quoted with that ticket
    loosened = box.run("ticket_build", LOOSENED)
    assert loosened.ok and "relaxed" not in loosened.content
    assert "each ticket's own relaxed with that ticket only" in loosened.content["hint"]
    assert all(ticket["relaxed"] and all(isinstance(words, str) for words in ticket["relaxed"])
               for ticket in loosened.content["tickets"])
    assert all(card["relaxed"] and card["option"] is None for card in loosened.cards)
    # A percentage is read as the share it means; best_effort false asks for exactly that or nothing
    assert box.run("ticket_build", {**transcript_args(), "target_chance": 80}).ok
    strict = box.run("ticket_build", {**transcript_args(), "best_effort": False})
    assert strict.ok and "options" not in strict.content
    assert box.run("ticket_build", {**transcript_args(), "target_chance": 150}).content["error"] == "invalid_input"


def test_a_stake_and_target_nothing_fits_go_to_the_builder_s_closest_tickets(world, monkeypatch) -> None:
    seen: List[Dict[str, Any]] = []
    monkeypatch.setattr(Lab, "build", lambda self, uid, body: {"tickets": [], "verdict": "Nothing fits.", "notes": []})
    real = Lab.builder_run

    def spy(self, uid, body):
        seen.append(body)
        return real(self, uid, body)

    monkeypatch.setattr(Lab, "builder_run", spy)
    box = Toolbox(world.lab, UID, NY, {"stake": {Decimal("10")}, "target": {Decimal("35")}}, leagues={NBA})
    done = box.run("ticket_build", {"leagues": [NBA], "stake": 10, "target": 35})
    assert done.ok and seen and seen[0]["controls"]["target_odds"] == pytest.approx(3.5)
    assert seen[0]["controls"]["best_effort"] is True and seen[0]["stake"] == "10.00"
    assert done.cards and all(card["builder"] for card in done.cards)
    # Strict: the target builder's own answer, nothing more
    seen.clear()
    exact = box.run("ticket_build", {"leagues": [NBA], "stake": 10, "target": 35, "best_effort": False})
    assert exact.ok and not seen and exact.content["tickets"] == [] and exact.content["verdict"] == "Nothing fits."


def test_builder_ticket_facts_carry_the_option_fit_and_relaxed_words() -> None:
    ticket = {"label": "Custom", "legs": [], "option": {"key": "your_chance", "label": "Closest to your chance"},
              "fit": {"odds": 1.82, "chance": 0.55, "target_odds": 3.5, "target_chance": 0.8, "odds_gap_pct": -48.0,
                      "chance_gap_pp": -25.0, "meets": {"odds": False, "chance": False}, "extra": 1},
              "relaxed": [{"rule": "min_chance", "from": 0.6, "to": 0.55, "words": "per-leg chance lowered from 60% "
                                                                                   "to 55%"},
                          {"rule": "min_chance", "words": "per-leg chance lowered from 60% to 55%"}]}
    facts = builder_ticket_facts(ticket)
    assert facts["option"] == "Closest to your chance"
    assert facts["fit"] == {"odds": 1.82, "chance": 0.55, "target_odds": 3.5, "target_chance": 0.8,
                            "odds_gap_pct": -48.0, "chance_gap_pp": -25.0, "meets": {"odds": False, "chance": False}}
    assert facts["relaxed"] == ["per-leg chance lowered from 60% to 55%"]  # once each


# --- brain_top_picks ----------------------------------------------------------------------------------------------

def test_brain_top_picks_ranks_the_likeliest_winners_one_per_game_with_a_card(world) -> None:
    schema = next(tool for tool in TOOLS if tool["name"] == "brain_top_picks")
    assert "strict" not in schema and schema["input_schema"]["required"] == ["leagues"]
    box = Toolbox(world.lab, UID, NY, leagues={NBA})
    done = box.run("brain_top_picks", {"leagues": [NBA], "n": 3})
    assert done.ok, done.content
    picks = done.content["picks"]
    assert 1 <= len(picks) <= 3 and [p["rank"] for p in picks] == list(range(1, len(picks) + 1))
    chances = [p["chance"] for p in picks]
    assert chances == sorted(chances, reverse=True)
    assert len({p["ref"]["id"] for p in picks}) == len(picks)  # one pick a game
    assert all(p["ref"]["market"] == "h2h" and p["book"] in ("DraftKings", "FanDuel") for p in picks)
    assert all(pick["league"] == NBA == pick["ref"]["league"] for pick in done.cards[0]["picks"])
    assert "strongest first" in done.content["hint"]
    [card] = done.cards
    assert card["type"] == "picks" and card["source"] == "brain_top_picks" and card["title"].endswith("NBA")
    assert [p["refs"] for p in card["picks"]] == [[p["ref"]] for p in card["picks"]]  # what "Add to slip" takes
    assert all(isinstance(p["price"], str) and p["why"] is not None for p in card["picks"])
    assert done.summary.startswith("brain_top_picks(basketball_nba): ")
    # Price safety: a game under way and a stale price are never a pick
    later = Lab(world.service, world.store, now=lambda: NOW + 3 * 3600)
    late = Toolbox(later, UID, NY, leagues={NBA}).run("brain_top_picks", {"leagues": [NBA]})
    assert late.ok and {p["ref"]["id"] for p in late.content.get("picks") or []} <= {"n4"}
    # A league with games and no prices: no picks, named, and its odds asked for in the background
    nothing = box.run("brain_top_picks", {"leagues": [NHL]})
    assert nothing.ok and nothing.content["picks"] == [] and nothing.cards == []
    assert nothing.content["unpriced"] and "no prices loaded" in nothing.content["unpriced"][0]
    assert NHL in world.queued
    # Off while picks are paused, and behind the gates
    assert Toolbox(world.lab, UID, NY, paused=True).run("brain_top_picks", {"leagues": [NBA]}).content["error"] == \
        "picks_paused"


# --- the router, _clarify and the context -----------------------------------------------------------------------

ROUTE = {"intent": "build", "leagues": [NBA], "stake": None, "target": None, "max_legs": None, "tier": "none",
         "target_odds": 3.5, "target_chance": 0.8, "legs": 2, "top_picks": False, "days": None, "min_odds": None,
         "legs_per_day": None, "players": [], "risk_flags": []}


def test_the_router_reads_target_chance_legs_and_top_picks() -> None:
    schema = assistant_module.ROUTER_SCHEMA
    assert {"target_chance", "legs", "top_picks"} <= set(schema["required"])
    prompt = assistant_module.ROUTER_PROMPT
    assert '"80% chance" or "4 in 5" is 0.8' in prompt and '"a safe one" is null' in prompt
    assert "most likely winners" in prompt and "who will win" in prompt
    route = parse_route(ROUTE)
    assert (route.intent, route.target_odds, route.target_chance, route.legs) == ("build", Decimal("3.50"), 0.8, 2)
    assert route.fields()["target_chance"] == 0.8 and route.fields()["legs"] == 2
    assert parse_route({**ROUTE, "target_chance": 80}).target_chance == 0.8  # a percentage is read as a share
    assert parse_route({**ROUTE, "target_chance": "75%"}).target_chance == 0.75
    for bad in (0, 0.001, 1, 100.5, True, "lots", float("nan")):
        assert parse_route({**ROUTE, "target_chance": bad}).target_chance is None, bad
    assert parse_route({**ROUTE, "legs": 13}).legs is None
    # "Who will win tonight?" is a look-up of the likeliest winners, whatever the router called it
    for intent in ("chat", "explain", "lookup"):
        looked = parse_route({**ROUTE, "intent": intent, "target_odds": None, "target_chance": None, "legs": None,
                              "top_picks": True})
        assert looked.intent == "lookup" and looked.top_picks and looked.fields()["top_picks"] is True
    assert parse_route({**ROUTE, "top_picks": "yes"}).top_picks is False


def test_clarify_never_asks_a_stake_for_odds_a_chance_or_legs() -> None:
    for fields in ({"target_odds": 3.5}, {"target_chance": 0.8}, {"legs": 2},
                   {"target_odds": 3.5, "target_chance": 0.8, "legs": 2}, {"top_picks": True}):
        route = parse_route({**ROUTE, "target_odds": None, "target_chance": None, "legs": None, **fields})
        assert Assistant._clarify(None, route) is None, fields
    plain = parse_route({**ROUTE, "target_odds": None, "target_chance": None, "legs": None})
    asked = Assistant._clarify(None, plain)
    assert asked is not None and asked.text == "How much do you want to stake?"  # a plain "build me one" still asks


# --- the system prompt and the post-check -------------------------------------------------------------------------

def test_the_system_prompt_is_the_analyst_persona_and_keeps_the_rules() -> None:
    prompt = assistant_module.SYSTEM_PROMPT
    assert "You explain; you don't tip" not in prompt and "always mention its chance of losing" not in prompt
    assert "Give the honest picture first" not in prompt
    for kept in ("How the app works", "Never pass a price or a probability to a tool", "Odds data is rationed",
                 "you explain it, you don't override it", "Code checks every number in your reply",
                 "1-800-GAMBLER / 0808 8020 133", "suggest no bets", "never suggest a bigger stake"):
        assert kept in prompt, kept
    for persona in ("You are the app's betting analyst", "call ticket_build or brain_top_picks", "strongest first",
                    "wins about 31% of the time", "don't refuse", "likeliest at your odds", "closest to your chance",
                    "best balance", "quoted from its own relaxed", "No lectures or moralising", "nothing is guaranteed",
                    "Don't repeat the downside unless asked", "Never claim certainty", "Short, confident and direct"):
        assert persona in prompt, persona
    # The app's rules part is free of banned wording (it is shown verbatim nowhere, but the checks hold it too)
    assert not banned_phrases(prompt.split("How to talk")[0])
    assert "You explain; you don't tip" not in EXPLAIN_PROMPT
    assert "lead with the picks and the ticket's odds and chance" in EXPLAIN_PROMPT


PERSONA = [
    "Likeliest at your odds: Celtics to win and Suns to win, 3.52x, wins about 31% of the time.",
    "Closest to your chance: 1.77x, about 56%.",
    "Best balance: 2.60x at 38%.",
    "To get there the Builder relaxed one rule: per-leg chance lowered from 60% to 55%.",
    "The strongest pick is the Nuggets, the most likely winner tonight at 78%.",
    "Best pick: Denver, favourite to win at 1.22 with DraftKings.",
    "These are the best of tonight's options, ranked strongest first.",
    "Most likely winners: Nuggets, Celtics and Suns.",
]


def test_the_persona_s_typical_sentences_pass_and_certainty_still_fails() -> None:
    sources = source_numbers({"odds": 3.52, "chance": 0.31, "others": [1.77, 0.56, 2.60, 0.38, 0.78, 1.22],
                              "relaxed": ["per-leg chance lowered from 60% to 55%"]})
    for sentence in PERSONA:
        assert problems(sentence, sources) == [], sentence
    for claim, name in (("The Nuggets are a lock tonight.", "lock"), ("This one is guaranteed.", "guaranteed"),
                        ("A sure thing at 78%.", "sure thing"), ("Risk-free money.", "risk-free"),
                        ("The Celtics can't lose.", "can't lose"), ("Bet more on this one.", "bigger stake"),
                        ("Go all in on the Nuggets.", "bet more"), ("It's a safe bet.", "safe")):
        assert name in banned_phrases(claim), claim
    # A number no tool gave is still sent back: "80%" was asked, not built
    assert problems("This ticket wins about 80% of the time.", sources)


# --- the templates ---------------------------------------------------------------------------------------------

def test_the_templates_lead_with_the_picks_odds_and_chance_and_never_refuse(world) -> None:
    box = Toolbox(world.lab, UID, NY, leagues={NBA})
    done = box.run("ticket_build", transcript_args())
    words = builder_template(done.cards[:2] + box.run("ticket_build", LOOSENED).cards[:1])
    lines = words.split("\n")
    first = done.cards[0]
    assert lines[0].startswith(f"{first['title']}: ") and "x, wins about " in lines[0]
    assert f"{first['fit']['odds']:.2f}x" in lines[0]
    # Each ticket's relaxed words on its own line (each option loosens its own way), never one merged list
    assert not any(line.startswith("Relaxed to get there") for line in lines)
    for line, card in zip(lines, done.cards[:2] + box.run("ticket_build", LOOSENED).cards[:1]):
        named = [entry["words"] for entry in card["relaxed"]]
        assert line.endswith(f" (relaxed: {'; '.join(named)}).") if named else "relaxed" not in line, line
    assert any("(relaxed: " in line for line in lines)
    assert lines[-1] in ("The cards below have the details.", "The card below has the details.")
    for refusal in ("couldn't build", "No ticket can", "nothing in betting is guaranteed", "lose more often"):
        assert refusal.lower() not in words.lower()
    assert not banned_phrases(words)
    # Analyse: the picks, the odds and the chance first, then the weakest leg; the case against is one line
    ticket = first["ticket"]
    explained = template_explanation({"ticket": ticket})
    labels = [leg["label"] for leg in ticket["legs"]]
    assert explained["verdict"].startswith(labels[0]) and f"at {ticket['odds']['decimal']:.2f}" in explained["verdict"]
    assert "The weakest leg is" in explained["verdict"]
    assert explained["case_against"].count(". ") == 0 and not banned_phrases(explained["case_against"])
    # The likeliest winners in code's words
    card = {"type": "picks", "picks": [{"rank": 1, "label": "Denver Nuggets to win", "chance": 0.7812,
                                        "price": "1.22", "book": "DraftKings", "why": ["Best record at home."]},
                                       {"rank": 2, "label": "Boston Celtics to win", "chance": 0.66, "price": "1.45",
                                        "book": "FanDuel", "why": []}]}
    text = picks_template(card)
    assert text.split("\n")[1] == "1. Denver Nuggets to win, 78% at 1.22 (DraftKings): Best record at home."
    assert text.split("\n")[2] == "2. Boston Celtics to win, 66% at 1.45 (FanDuel)."
    assert picks_template({"picks": []}).startswith("No game in scope has prices loaded")


def test_the_picks_say_why_not_the_chance_again() -> None:
    # A pick's first "why" is its "Markets say" line, the chance beside it again: the template and the bot give the
    # reason after it
    why = ["Markets say 79% (Sharp 79% · US 79%)", "Charlotte Hornets on a back-to-back: the line already has about "
                                                    "+1.3 pts for it"]
    assert first_reason(why) == why[1]
    assert first_reason(["Markets say 79%"]) == "Markets say 79%" and first_reason([]) is None
    card = {"type": "picks", "title": "Most likely winners",
            "picks": [{"rank": 1, "label": "Denver Nuggets to win", "chance": 0.79, "price": "1.22",
                       "book": "DraftKings", "why": why}]}
    assert picks_template(card).split("\n")[1] == ("1. Denver Nuggets to win, 79% at 1.22 (DraftKings): Charlotte "
                                                   "Hornets on a back-to-back: the line already has about +1.3 pts "
                                                   "for it.")
    from telegram_betting_bot_simple.bot import chat
    text = chat.card_text(card)
    assert "back-to-back" in text and "Markets say" not in text


def test_the_templates_name_a_total_s_game_and_say_a_chance_over_even_as_a_share() -> None:
    legs = [{"index": 0, "label": "Under 230.5", "game": "Utah Jazz @ Denver Nuggets", "market": "totals",
             "odds": {"decimal": 1.91}, "ai_prob": 0.5, "market_prob": 0.5},
            {"index": 1, "label": "Orlando Magic to win", "game": "Orlando Magic @ Miami Heat", "market": "h2h",
             "odds": {"decimal": 2.1}, "ai_prob": 0.48, "market_prob": 0.47},
            {"index": 2, "label": "Under 225.5", "game": "Boston Celtics @ New York Knicks", "market": "totals",
             "odds": {"decimal": 1.81}, "ai_prob": 0.52, "market_prob": 0.5}]
    ticket = {"legs": legs, "odds": {"decimal": 7.26}, "weakest_leg": 1,
              "chance": {"ai": 0.1248, "ai_one_in": 8, "market": 0.1175, "market_one_in": 9}}
    verdict = template_explanation({"ticket": ticket})["verdict"]
    assert verdict.startswith("Under 230.5 (Jazz @ Nuggets), Orlando Magic to win and Under 225.5 (Celtics @ "
                              "Knicks) at 7.26: wins about 1 in 8 (12%)")
    assert leg_name(legs[1]) == "Orlando Magic to win"  # a label that names a team needs no game
    assert leg_name({"label": "Draw", "game": "Arsenal @ Chelsea"}) == "Draw (Arsenal @ Chelsea)"
    builder = builder_template([{"title": "Custom ticket #2", "ticket": {**ticket, "joint": {"chance": 0.1248}}}])
    assert builder.startswith("Custom ticket #2: Under 230.5 (Jazz @ Nuggets), Orlando Magic to win and Under "
                              "225.5 (Celtics @ Knicks), 7.26x, wins about 12% of the time.")
    # A single at 66%: "66% of the time", never "1 in 2 (66%)" (1/0.66 rounds to 2, which reads as 50%)
    single = {"legs": [{"index": 0, "label": "Boston Celtics to win", "game": "Celtics @ Nets",
                        "odds": {"decimal": 1.45}, "ai_prob": 0.66, "market_prob": 0.66}],
              "odds": {"decimal": 1.45}, "chance": {"ai": 0.66, "ai_one_in": 2, "market": 0.6, "market_one_in": 2}}
    words = template_explanation({"ticket": single})["verdict"]
    assert "wins about 66% of the time by the AI and 60% of the time by the market" in words and "1 in 2" not in words
    # The bot's ticket card says it the same way: "AI 66%", and "1 in 8 (12%)" under even
    from telegram_betting_bot_simple.bot import chat
    card = chat.card_text({"type": "ticket", "title": "Single", "ticket": {**single, "type": "single"}})
    assert "Chance it wins: AI 66% · market 60%" in card and "1 in 2" not in card
    card = chat.card_text({"type": "ticket", "title": "Treble", "ticket": {**ticket, "type": "multi"}})
    assert "Chance it wins: AI 1 in 8 (12%) · market 1 in 9 (12%)" in card


def test_the_ai_notice_doesnt_say_nothing_here_is_a_tip() -> None:
    notice = assistant_module.DISCLOSURE
    assert "tip" not in notice.lower() and not banned_phrases(notice)
    for kept in ("AI analyst (Claude, made by Anthropic)", "Our code works out every number", "can still be wrong",
                 "only you add bets to your slip"):
        assert kept in notice, kept


# --- the transcript through the real assistant, with a fake Claude --------------------------------------------------

def usage():
    return SimpleNamespace(input_tokens=900, output_tokens=200, cache_creation_input_tokens=0,
                           cache_read_input_tokens=0, server_tool_use=None)


def reply(content, stop="end_turn", model="claude-opus-5-5"):
    return SimpleNamespace(content=content, stop_reason=stop, usage=usage(), model=model)


def say(value):
    return SimpleNamespace(type="text", text=value)


class TranscriptClaude:
    """The router reads the transcript's message as Claude Haiku would; the chat model calls ticket_build with what
    code understood, then writes its answer from the tool's facts (each option in one sentence, numbers as given)."""

    def __init__(self, answer=None):
        self.requests: List[Dict[str, Any]] = []
        self.answer = answer
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(json.loads(json.dumps(request, default=lambda o: getattr(o, "__dict__", str(o)))))
        if request["model"] == "claude-haiku-5-5":
            return reply([say(json.dumps(ROUTE))], model="claude-haiku-5-5")
        results = [block for message in request["messages"] if isinstance(message["content"], list)
                   for block in message["content"] if isinstance(block, dict) and block.get("type") == "tool_result"]
        if not results:
            context = request["messages"][-1]["content"][0]["text"]
            understood = json.loads(context.split("What code understood (use these values): ", 1)[1].split("\n", 1)[0])
            args = {key: understood[key] for key in ("leagues", "legs", "target_chance") if key in understood}
            args["target_odds"] = float(understood["target_odds"])
            return reply([SimpleNamespace(type="tool_use", id="toolu_1", name="ticket_build", input=args)],
                         stop="tool_use")
        facts = json.loads(results[-1]["content"])
        if self.answer is not None:
            return reply([say(self.answer)])
        lines = []
        for option in facts.get("options") or []:
            fit = option["fit"]
            relaxed = f" (to get there: {option['relaxed'][0]})" if option.get("relaxed") else ""
            lines.append(f"{option['option']}: {fit['odds']:.2f}x, wins about {fit['chance'] * 100:.0f}% of the "
                         f"time{relaxed}.")
        return reply([say(" ".join(lines))])


def test_the_transcript_builds_ticket_cards_with_the_closest_options_and_no_refusal(world) -> None:
    claude = TranscriptClaude()
    assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    world.lab.assistant = assistant
    out = world.lab.chat(UID, {"message": TRANSCRIPT})
    answer = out["reply"]
    # Routed to build with the transcript's targets, no question about a stake
    context = claude.requests[1]["messages"][-1]["content"][0]["text"]
    understood = json.loads(context.split("What code understood (use these values): ", 1)[1].split("\n", 1)[0])
    assert (understood["intent"], understood["target_odds"], understood["target_chance"], understood["legs"]) == \
        ("build", "3.50", 0.8, 2)
    assert "Build exactly this with ticket_build" in context
    chat = [r for r in claude.requests if r["model"] != "claude-haiku-5-5"]
    calls = [block for r in chat for m in r["messages"] if isinstance(m["content"], list) for block in m["content"]
             if isinstance(block, dict) and block.get("type") == "tool_use"]
    assert calls and calls[0]["name"] == "ticket_build" and calls[0]["input"]["target_chance"] == 0.8
    assert calls[0]["input"]["legs"] == 2 and calls[0]["input"]["target_odds"] == 3.5
    # The answer: ticket cards with the options, the reply the persona's lines (checked: every number traced)
    assert answer["kind"] == "answer", answer
    assert answer["text"].startswith("Likeliest at your odds: ") and "wins about" in answer["text"]
    cards = answer["cards"]
    assert cards and all(card["type"] == "ticket" and card["source"] == "ticket_build" for card in cards)
    assert [card["title"] for card in cards] == [card["option"]["label"] for card in cards]
    assert all(card["fit"]["target_chance"] == 0.8 for card in cards)
    assert "Compare these tickets" in answer["chips"] or len(cards) == 1
    for refusal in ("couldn't build", "No ticket can", "nothing in betting is guaranteed"):
        assert refusal.lower() not in answer["text"].lower()


def test_a_bad_answer_to_the_transcript_falls_back_to_the_tickets_never_a_refusal(world) -> None:
    claude = TranscriptClaude(answer="I couldn't build that: no ticket can have an 80% chance at 3.5 odds, it's "
                                     "guaranteed to lose more often than it wins at 97%.")
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    answer = world.lab.chat(UID, {"message": TRANSCRIPT})["reply"]
    assert answer["kind"] == "template" and answer["cards"]
    assert answer["text"].startswith(answer["cards"][0]["title"] + ": ")
    assert "couldn't" not in answer["text"] and not banned_phrases(answer["text"])


PICKS_ROUTE = {**ROUTE, "intent": "chat", "target_odds": None, "target_chance": None, "legs": None, "top_picks": True}


class PicksClaude(TranscriptClaude):
    """"Who will win tonight?": the router reads a look-up of the likeliest winners; the chat model calls
    brain_top_picks and names the strongest pick with its chance, as the tool gave it."""

    def _create(self, **request):
        self.requests.append(json.loads(json.dumps(request, default=lambda o: getattr(o, "__dict__", str(o)))))
        if request["model"] == "claude-haiku-5-5":
            return reply([say(json.dumps(PICKS_ROUTE))], model="claude-haiku-5-5")
        results = [block for m in request["messages"] if isinstance(m["content"], list) for block in m["content"]
                   if isinstance(block, dict) and block.get("type") == "tool_result"]
        if not results:
            return reply([SimpleNamespace(type="tool_use", id="toolu_1", name="brain_top_picks",
                                          input={"leagues": [NBA], "n": 3})], stop="tool_use")
        top = json.loads(results[-1]["content"])["picks"][0]
        return reply([say(f"The strongest pick is {top['pick']}, the most likely winner tonight at "
                          f"{top['chance'] * 100:.0f}%.")])


def test_who_will_win_tonight_calls_brain_top_picks_and_shows_its_card(world) -> None:
    claude = PicksClaude()
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    answer = world.lab.chat(UID, {"message": "who will win tonight in the NBA?"})["reply"]
    context = claude.requests[1]["messages"][-1]["content"][0]["text"]
    assert "call brain_top_picks" in context and '"intent": "lookup"' in context
    assert answer["kind"] == "answer" and answer["text"].startswith("The strongest pick is ")
    assert [card["type"] for card in answer["cards"]] == ["picks"]
    assert answer["chips"] == ["Why these picks?", "Any injury news?"]  # never "bet more"


# --- the bot ------------------------------------------------------------------------------------------------------

def test_the_bot_shows_the_option_the_fit_line_the_relaxed_rules_and_the_picks(world) -> None:
    from telegram_betting_bot_simple.bot import chat
    box = Toolbox(world.lab, UID, NY, leagues={NBA})
    done = box.run("ticket_build", transcript_args())
    card = done.cards[0]
    text = chat.card_text(card)
    fit = card["fit"]
    assert text.startswith(f"🧾 <b>{card['title']}</b>")
    assert f"🎯 {fit['odds']:.2f}x · " in text and "(you asked 3.5x · 80%)" in text
    with_relaxed = chat.card_text(box.run("ticket_build", LOOSENED).cards[0])
    assert "<i>Relaxed: per-leg chance lowered from 72% to " in with_relaxed and "🎯" not in with_relaxed
    picks = {"type": "picks", "title": "Most likely winners · NBA",
             "picks": [{"rank": 1, "label": "Nuggets <to> win", "chance": 0.78, "price": "1.22", "book": "DraftKings",
                        "why": ["Home & rested"], "ref": {"league": NBA, "id": "n2", "market": "h2h", "side": "home"}}]}
    shown = chat.card_text(picks)
    assert shown.startswith("🏆 <b>Most likely winners · NBA</b>")
    assert "1. Nuggets &lt;to&gt; win · 78% · 1.22 at DraftKings" in shown and "Home &amp; rested" in shown
    assert "· 1.70 at FanDuel" in chat.card_text({**picks, "picks": [{**picks["picks"][0], "price": "1.7",
                                                                       "book": "FanDuel"}]})
    markup = chat.reply_markup({"thread_id": 3, "message_idx": 1, "reply": {"cards": [picks], "chips": []}}, None,
                               False)
    assert [(b.text, b.callback_data) for row in markup.inline_keyboard for b in row] == [("➕ Add pick 1",
                                                                                          "ai:pick:3:1:0:0")]


def test_the_bot_s_ai_bets_show_a_level_s_closest_ticket_with_its_label_and_relaxed_words() -> None:
    from telegram_betting_bot_simple.bot.handlers import bets_text
    bet = {"legs": [{"label": "Nuggets to win", "odds": {"american": "-455", "decimal": 1.22}, "probability": 0.78,
                     "fair": 0.79}],
           "book": "DraftKings", "odds": {"decimal": 1.22, "american": "-455"}, "badge": "For fun",
           "probability": 0.78, "fair_probability": 0.79, "ev": -0.05, "why": "Wins about 8 in 10.",
           "label": "Closest to Safe", "relaxed": ["per-leg chance lowered from 60% to 55%"]}
    tiers = {key: {"title": key.title(), "rule": "rule", "detail": "detail", "bets": [], "closest": [],
                   "empty": "Nothing fits this level right now."} for key in ("safe", "risky", "crazy")}
    tiers["safe"]["closest"] = [bet]
    text = bets_text({"note": "", "tiers": tiers}, None, "NBA")
    assert "🎯 <b>Closest to Safe</b>" in text and "<i>Relaxed: per-leg chance lowered from 60% to 55%</i>" in text
    assert "Nothing fits this level right now." in text  # Risky and Crazy have nothing at all


def bot_turn(world, claude, words):
    """One message to the bot's AI chat (bot/chat.py) from tg:42, over the world's Lab: (the answer sent, context)."""
    import asyncio
    from telegram.ext import Application
    from telegram_betting_bot_simple.bot import chat, handlers
    from tests.test_bot_chat import CONFIG, FakeBot, typed
    world.store.get_or_create_user("tg:42", "Sam", now=NOW)
    world.store.confirm_age("tg:42", now=NOW)
    application = Application.builder().token(CONFIG.telegram_token).build()
    handlers.setup_handlers(application, CONFIG)
    application.bot_data["assistant"] = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    context = SimpleNamespace(application=application, user_data={}, bot=FakeBot())
    asyncio.run(chat.chat_message(typed(words)[0], context))
    return context.bot.sent[-1], context


def test_the_transcript_in_the_bot_gives_the_labelled_tickets_with_their_real_odds_and_chances(world) -> None:
    from tests.test_bot_chat import buttons
    sent, context = bot_turn(world, TranscriptClaude(), TRANSCRIPT)
    first = sent.text.split("🧾", 1)[0]
    assert "Likeliest at your odds: " in first and "wins about" in first  # the analyst's words, checked
    for label in LABELS.values():
        assert f"🧾 <b>{label}</b>" in sent.text, label
    assert sent.text.count("(you asked 3.5x · 80%)") == 3
    assert any(label.startswith("➕ Add ticket 1") for label, _, _ in buttons(sent.markup))


def test_the_likeliest_winners_in_the_bot_with_an_add_pick_button_per_pick(world) -> None:
    import asyncio
    from telegram_betting_bot_simple.bot import chat
    from tests.test_bot_chat import buttons, tapped
    sent, context = bot_turn(world, PicksClaude(), "who will win tonight in the NBA?")
    assert "🏆 <b>Most likely winners · NBA</b>" in sent.text and "1. " in sent.text
    picks = [(label, data) for label, data, _ in buttons(sent.markup) if label.startswith("➕ Add pick")]
    assert picks and all(data.startswith(f"ai:pick:{context.user_data['ai_thread']}:1:0:") for _, data in picks)
    assert not context.user_data.get("picks")  # the AI never adds a pick by itself
    update, query = tapped(picks[0][1])
    asyncio.run(chat.add_pick(update, context))
    assert len(context.user_data["picks"]) == 1 and context.user_data["picks"][0]["market"] == "h2h"
    assert query.answers[0][0].startswith("Added to your slip")
    update, query = tapped(picks[0][1][:-1] + "9")  # no such pick
    asyncio.run(chat.add_pick(update, context))
    assert "isn't available" in query.answers[0][0] and len(context.user_data["picks"]) == 1


def test_the_ai_bets_give_a_level_nothing_fits_its_closest_tickets_never_as_its_own() -> None:
    from telegram_betting_bot_simple.bot.handlers import bets_text
    from telegram_betting_bot_simple.engine import bets
    from tests.test_bets import leg
    coin_flips = [leg(f"g{i}", 0.52 + i * 0.01, 1.85) for i in range(6)]  # nothing wins 60%+: no Safe ticket
    out = bets.build_all(coin_flips)
    safe = out["safe"]
    assert safe["bets"] == [] and safe["closest"]  # the record (bets) stays the tier's own tickets
    for view in safe["closest"]:
        assert view["label"] == "Closest to Safe" and view["book"] in ("Book A", "Book B")
        assert view["relaxed"] and all(isinstance(words, str) for words in view["relaxed"])
    assert out["risky"]["bets"] and out["risky"]["closest"] == []  # a level with its own tickets needs none
    assert bets.tier_controls(bets.TIERS[0]).best_effort is False  # the tier's own search is strict
    assert bets.build_all(coin_flips, best_effort=False)["safe"]["closest"] == []
    text = bets_text({"note": "", "tiers": out}, "safe", "NBA")
    assert "🎯 <b>Closest to Safe</b>" in text and f"<i>Relaxed: {'; '.join(safe['closest'][0]['relaxed'])}</i>" in text


def test_a_swapped_ticket_is_priced_with_its_fit_to_the_run_s_targets(world) -> None:
    client = world.client()
    out = run(client, {"leagues": [NBA], **{k: v for k, v in transcript_args().items() if k != "leagues"}})
    ticket = out["tickets"][0]
    refs = [{key: leg[key] for key in ("league", "market", "side", "point")} | {"id": leg["event_id"]}
            for leg in ticket["legs"]]
    priced = client.post("/api/builder/price", json={"legs": refs, "target_odds": 3.5, "target_chance": 0.8})
    assert priced.status_code == 200, priced.text
    assert priced.json()["fit"] == ticket["fit"]  # the same legs: the same fit as the run gave
    plain = client.post("/api/builder/price", json={"legs": refs}).json()
    assert plain["fit"] is None
    assert client.post("/api/builder/price", json={"legs": refs, "target_chance": 2}).status_code == 422
