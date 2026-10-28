"""The Builder across its three parts (v3 Stage 3 integration): the engine (engine/builder.py, presets.py,
soccer.py), the wiring (dashboard/lab.py, service.py, app.py, ai/tools.py, ai/assistant.py) and the app
(dashboard/static/js/views/build/builder/*), over the real modules.

The world is test_builder_wiring's: a real Store, the real SlateBuilder over fake ESPN boards, the real games
service, Lab and routes, the saved NBA and EPL boards. The only fakes are at the edges (the boards' feed, ESPN,
the Anthropic client). No network."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest
import requests

from telegram_betting_bot_simple.dashboard import app as app_module
from telegram_betting_bot_simple.dashboard import lab as lab_module
from telegram_betting_bot_simple.data.odds_service import configure_books
from telegram_betting_bot_simple.engine import builder as engine
from telegram_betting_bot_simple.engine import presets
from tests.test_builder_wiring import EPL, NBA, NHL, NOW, NY, PLAYER_B, World, make_world, run


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
    return make_world(tmp_path)


STATIC = Path(app_module.__file__).resolve().parent / "static" / "js" / "views" / "build" / "builder"
ALL_MARKETS = list(engine.MARKETS)


def js_const(name: str) -> Any:
    """A constant of the app's builder model (model.js), read as JSON."""
    source = (STATIC / "model.js").read_text(encoding="utf-8")
    found = re.search(rf"export const {name} = (\[.*?\]|\{{.*?\}}|[\d.]+);", source, re.S)
    assert found, name
    text = re.sub(r"//[^\n]*", "", found.group(1))
    text = re.sub(r"([{,]\s*)([A-Za-z_]\w*)\s*:", r'\1"\2":', text)  # quote the keys
    text = re.sub(r",\s*([\]}])", r"\1", text)  # no trailing commas
    return json.loads(text)


# --- the app's model and the engine's controls agree -------------------------------------------------------------

def test_the_app_sends_exactly_the_engine_s_controls_with_its_defaults() -> None:
    fields = js_const("FIELDS")
    defaults = engine.Controls().to_dict()
    assert sorted(fields) == sorted(defaults)  # every control, both ways
    assert set(app_module.BuilderControls.model_fields) == set(fields) | {"ui"}  # the API takes them all
    assert js_const("DEFAULTS") == json.loads(json.dumps(defaults))
    assert js_const("SCOPE_FIELDS") == list(presets.SCOPE_FIELDS)  # what keeps the preset's name
    assert set(engine.market_catalog()) <= set(js_const("MARKET_ORDER"))
    assert js_const("MAX_WINDOW_DAYS") == lab_module.MAX_WINDOW_DAYS
    assert js_const("MAX_LEGS") == engine.MAX_LEGS and js_const("MAX_TICKETS") == engine.MAX_TICKETS


def test_every_preset_round_trips_through_the_route_and_the_engine(world) -> None:
    client = world.client()
    data = client.get("/api/builder/presets").json()
    for name, sent in data["presets"].items():
        # The app sends a preset's controls back as they came (plus a window and a time zone): still that preset
        body = {**sent, "window": None, "tz": "America/New_York"}
        controls = world.lab.builder_controls(body)
        assert presets.label_for(controls) == presets.TITLES[name]
        result = run(client, body)
        assert result["label"] == presets.TITLES[name] and result["controls"]["preset"] == name


# --- what the app reads from a run --------------------------------------------------------------------------------

TICKET_KEYS = {"legs", "refs", "label", "joint", "reach_target", "book_cut", "expected", "notes", "odds", "type",
               "total_stake", "max_payout", "unit_stake", "rank", "books", "model_priced", "cautious"}
LEG_KEYS = {"label", "odds", "price", "book", "game", "league", "start", "chance", "chance_source", "age", "flags",
            "edge", "why", "risk", "swaps", "key", "ref", "event_id", "market", "side", "point", "model_price"}
SWAP_KEYS = {"key", "ref", "label", "price", "game", "book", "chance", "chance_source", "edge", "age", "flags", "why",
             "risk", "odds", "joint"}
MISS_KEYS = {"rule", "rules", "label", "price", "reason", "text", "game", "ref", "book", "chance"}


def test_a_run_carries_every_field_the_app_reads(world) -> None:
    client = world.client()
    result = run(client, {"preset": "crazy", "markets": ALL_MARKETS, "legs": 3, "target_odds": 6,
                          "tolerance_pct": 30, "min_chance": 0.3, "prefer_correlation": True}, stake="5")
    assert {"label", "stake", "stake_given", "notes", "near_misses", "blocked", "ideas", "stats", "target",
            "unpriced", "controls", "as_of", "tickets"} <= set(result)
    assert result["stake"] == "5.00" and result["stake_given"] is True
    assert {"candidates", "passed", "games", "combos", "ms"} <= set(result["stats"])
    assert result["target"] == {"odds": 6.0, "low": 4.2, "high": 7.8, "tolerance_pct": 30.0}
    assert result["tickets"], result["blocked"]
    for ticket in result["tickets"]:
        assert TICKET_KEYS <= set(ticket), TICKET_KEYS - set(ticket)
        assert {"chance", "one_in", "fair_odds"} <= set(ticket["joint"])
        assert {"chance", "one_in", "odds", "at_target"} <= set(ticket["reach_target"])
        assert ticket["reach_target"]["odds"] == pytest.approx(4.2)
        assert {"ai", "market"} <= set(ticket["expected"]) and isinstance(ticket["odds"], dict)
        assert ticket["refs"] == [leg["ref"] for leg in ticket["legs"]]
        for leg in ticket["legs"]:
            assert LEG_KEYS <= set(leg), LEG_KEYS - set(leg)
            assert 1 <= len(leg["why"]) <= 3 and leg["risk"]
            for swap in leg["swaps"]:
                assert SWAP_KEYS <= set(swap), SWAP_KEYS - set(swap)
    # NHL has ESPN games and no saved board: a league without prices, named (the app lists result.unpriced)
    nhl = next(entry for entry in result["unpriced"] if entry["league"] == NHL)
    assert nhl["title"] and nhl["reason"].startswith(f"{nhl['title']}: no prices loaded (2 games)")
    assert result["near_misses"][0]["rule"] == "no_prices"
    misses = [miss for miss in result["near_misses"] if miss["rule"] != "no_prices"]
    assert misses and all(MISS_KEYS <= set(miss) for miss in misses)
    assert all(miss["text"] == f"{miss['label']} @ {float(miss['price']):.2f}: {miss['reason']}" for miss in misses)


def test_a_blocked_run_says_which_rules_and_how_many(world) -> None:
    client = world.client()
    # Strict (closest effort off): nothing fits, and blocked says why
    result = run(client, {"preset": "risky", "legs": 6, "target_odds": 20, "min_chance": 0.55, "best_effort": False})
    assert not result["tickets"]
    blocked = result["blocked"]
    assert {"message", "rules", "scope", "ticket_rules", "passed", "relax"} <= set(blocked)
    assert blocked["message"].startswith("No 6-leg ticket near 20.0x: only ")
    assert "'min chance 55%'" in blocked["message"]
    assert blocked["rules"][0] == {"rule": "min_chance", "label": "min chance 55%",
                                   "removed": result["stats"]["removed"]["min_chance"]}
    assert blocked["passed"]["games"] < 6 and "min chance 55%" in blocked["relax"]
    # Closest effort (the default): the closest ticket, with the rule it loosened in words, and nothing blocked
    closest = run(client, {"preset": "risky", "legs": 6, "target_odds": 20, "min_chance": 0.55})
    assert closest["tickets"] and closest["blocked"] is None
    assert all(t["relaxed"] and all(entry["words"] for entry in t["relaxed"]) for t in closest["tickets"])
    assert any(entry["rule"] == "min_chance" for t in closest["tickets"] for entry in t["relaxed"])


# --- a swap: the price route and the engine agree -----------------------------------------------------------------

def test_the_price_route_gives_a_ticket_the_run_s_own_numbers(world) -> None:
    client = world.client(PLAYER_B)
    result = run(client, {"preset": "crazy", "legs": 3, "target_odds": 6, "tolerance_pct": 30, "min_chance": 0.3},
                 stake="4")
    ticket = result["tickets"][0]
    body = {"legs": ticket["refs"], "stake": "4", "target_odds": 6, "tolerance_pct": 30}
    priced = client.post("/api/builder/price", json=body).json()
    assert priced["odds"] == ticket["odds"] and priced["max_payout"] == ticket["max_payout"]
    assert priced["joint"] == ticket["joint"] and priced["reach_target"] == ticket["reach_target"]
    assert priced["notes"] == [n for n in ticket["notes"] if not n.startswith("You asked for")]
    assert priced["removed"] == [] and priced["model_priced"] is False


def test_a_swap_priced_by_the_route_is_the_swap_the_engine_offered(world) -> None:
    client = world.client()
    result = run(client, {"preset": "crazy", "legs": 3, "target_odds": 6, "tolerance_pct": 30, "min_chance": 0.3,
                          "markets": ALL_MARKETS})
    ticket, index, swap = next((t, i, leg["swaps"][0]) for t in result["tickets"]
                               for i, leg in enumerate(t["legs"]) if leg["swaps"])
    refs = [swap["ref"] if i == index else ref for i, ref in enumerate(ticket["refs"])]
    priced = client.post("/api/builder/price", json={"legs": refs, "target_odds": 6, "tolerance_pct": 30}).json()
    assert priced["odds"]["decimal"] == pytest.approx(swap["odds"], abs=0.01)
    assert priced["joint"]["chance"] == pytest.approx(swap["joint"], abs=1e-4)
    assert priced["reach_target"]["odds"] == pytest.approx(4.2)
    assert priced["refs"][index] == {k: swap["ref"][k] for k in ("league", "id", "market", "side", "point")}
    # A field the route doesn't know is refused (it never guesses what was meant)
    assert client.post("/api/builder/price", json={"legs": refs, "target": 6}).status_code == 422


# --- the reasons: the team model only where it ran -----------------------------------------------------------------

def test_a_leg_names_the_team_model_only_where_the_model_ran(world) -> None:
    # The games board as the NBA page caches it: n1's card has the team model's own numbers, n2's only the market's
    cards = [{"id": "n1", "home": "Boston Celtics", "away": "New York Knicks",
              "prediction": {"margin": 5.1, "total": 222.0, "factors": ["Home court: Boston Celtics +2.0"],
                             "sources": {"model": {"margin": 4.24, "total": 223.4, "home_win_prob": 0.64},
                                         "market": {"margin": 6.5, "home_win_prob": 0.66, "total": 221.5}}}},
             {"id": "n2", "home": "Denver Nuggets", "away": "Utah Jazz",
              "prediction": {"margin": 9.5, "total": 230.5, "factors": [],
                             "sources": {"market": {"margin": 9.5, "home_win_prob": 0.79, "total": 230.5}}}}]
    world.service.cache.put(("games", NBA), {"games": cards}, ttl=600, persist=False)
    by_key = world.service.builder_candidates(now=NOW)["by_key"]
    assert by_key[f"{NBA}:n1:h2h:home"].why[1] == ("Team model: Boston Celtics by 4.2, total 223; "
                                                    "Home court: Boston Celtics +2.0")
    assert not any("model" in line.lower() for line in by_key[f"{NBA}:n2:h2h:home"].why)


# --- Ask AI: "build me 5 legs across NBA and EPL at about 10x, Risky" ---------------------------------------------

def _message(content, stop="end_turn", model="claude-opus-5-5"):
    usage = SimpleNamespace(input_tokens=900, output_tokens=200, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0, server_tool_use=None)
    return SimpleNamespace(content=content, stop_reason=stop, usage=usage, model=model)


def _text(value: str):
    return SimpleNamespace(type="text", text=value)


class ScriptedClaude:
    """The Anthropic client's surface: the router (Haiku) and the chat (Opus) from two scripted queues."""

    def __init__(self, router: List[Any], chat: List[Any]) -> None:
        self.router, self.chat, self.requests = list(router), list(chat), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> Any:
        self.requests.append(json.loads(json.dumps(request, default=lambda o: getattr(o, "__dict__", str(o)))))
        queue = self.router if request["model"] == "claude-haiku-5-5" else self.chat
        return queue.pop(0)


def test_ask_ai_builds_with_the_controls_through_the_real_assistant(world) -> None:
    from telegram_betting_bot_simple.ai.assistant import Assistant
    uid = "tg:4242"
    world.store.get_or_create_user(uid, "Ava", now=NOW)
    world.store.confirm_age(uid, now=NOW)
    route = {"intent": "build", "leagues": [NBA, EPL], "stake": None, "target": None, "max_legs": 5, "tier": "risky",
             "target_odds": 10, "days": None, "min_odds": None, "legs_per_day": None, "players": [], "risk_flags": []}
    call = SimpleNamespace(type="tool_use", id="toolu_b1", name="ticket_build",
                           input={"leagues": [NBA, EPL], "legs": 5, "target_odds": 10, "preset": "risky"})
    claude = ScriptedClaude(
        router=[_message([_text(json.dumps(route))], model="claude-haiku-5-5")],
        chat=[_message([call], stop="tool_use"),
              _message([_text("I built these with the Builder from the saved prices; long tickets lose most of the "
                               "time, and each leg's reasons are on the card.")])])
    assistant = Assistant(claude, world.lab, world.store, daily_cents=100, tz=NY, clock=lambda: NOW)
    out = assistant.reply(uid, message="build me 5 legs across NBA and EPL at about 10x, Risky")
    reply = out["reply"]
    assert reply["kind"] == "answer", reply  # no "how much do you want to stake?" first
    context = claude.requests[1]["messages"][-1]["content"][0]["text"]
    assert '"target_odds": "10.00"' in context and '"tier": "risky"' in context
    result = json.loads(claude.requests[2]["messages"][-1]["content"][0]["content"])
    assert result["label"] == "Custom" and result["source"].startswith("The Builder")
    if result["tickets"]:
        cards = reply["cards"]
        assert cards and all(card["builder"] and card["label"] == "Custom" for card in cards)
        # Five legs, unless closest effort changed the count and says so
        assert all(len(card["ticket"]["legs"]) == 5 or any(entry["rule"] == "legs" for entry in card["relaxed"])
                   for card in cards)
        assert all({ref["league"] for ref in card["refs"]} <= {NBA, EPL} for card in cards)
        # The ticket's own notes ride in the ticket; a card's "notes" would be per-leg notes to the app
        assert all("notes" not in card and isinstance(card["ticket"]["notes"], list) for card in cards)
    else:
        assert result["blocked"].startswith("No 5-leg ticket near 10.0x")
