"""The AI chat assistant: gates, warning signs, the router, the tool loop, the checks and credits (no network).

Claude is a fake client that hands back scripted responses (router JSON, tool_use blocks, final text),
and the Lab is a small fake over the shared fake games board that prices with the real engine
(tickets, target builder, rollover planner), so the numbers the checks compare against are real.
"""

from __future__ import annotations

import datetime
import json
import time
from decimal import Decimal
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from telegram_betting_bot_simple.ai import assistant as assistant_module
from telegram_betting_bot_simple.ai.assistant import (
    DISCLOSURE,
    HELPLINE,
    SUPPORT_MESSAGE,
    Assistant,
    build_assistant,
    parse_route,
    risk_flags,
    soft_flags,
)
from telegram_betting_bot_simple.ai.explain import (
    banned_phrases,
    explain_ticket,
    invented_numbers,
    numbers_in,
    source_numbers,
    template_explanation,
    ClaudeCaller,
)
from telegram_betting_bot_simple.ai.tools import PICK_TOOLS, TOOLS, Toolbox, bet_count
from telegram_betting_bot_simple.config import Config
from telegram_betting_bot_simple.dashboard.service import CallRejected, NotFound
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.engine import rollover, target, tickets
from telegram_betting_bot_simple.engine.tickets import TicketLeg

# 18:00 UTC on 15 January 2027: 1 PM in New York, so no late-night note
NOW = datetime.datetime(2027, 1, 15, 18, 0, tzinfo=datetime.timezone.utc).timestamp()
UID = "tg:42"
NY = __import__("zoneinfo").ZoneInfo("America/New_York")


# --- fakes ---------------------------------------------------------------------------------------

class FakeLab:
    """The Lab's methods the tools call, over the shared fake board, priced by the real engine."""

    def __init__(self, service):
        self.service = service
        self.calls = []

    def today(self, uid, leagues, day=None):
        self.calls.append(("today", tuple(leagues)))
        games, as_of = [], {}
        for league in leagues:
            board = self.service.games(league)
            games += [{**card, "league_title": board["title"]} for card in board["games"]]
            as_of[league] = board["odds_as_of"]
        return {"games": games, "errors": {}, "quota": {}, "odds_as_of": as_of}

    def _leg(self, ref):
        card = next(c for c in self.service.games(ref["league"])["games"] if c["id"] == ref["id"])
        offer = card["markets"][ref["market"]][ref["side"]]
        return TicketLeg(key=f"{ref['league']}:{ref['id']}:{ref['market']}:{ref['side']}", league=ref["league"],
                         event_id=ref["id"], market=ref["market"], side=ref["side"], point=offer["point"],
                         price=Decimal(str(offer["price"])), label=offer["label"],
                         game=f"{card['away']} @ {card['home']}", start=card["start"], ai_prob=offer["ai"],
                         market_prob=offer["fair"], push_prob=offer["push"], book=offer["book"])

    def price(self, uid, body):
        self.calls.append(("price", body["type"], body["stake"]))
        legs = [self._leg(ref) for ref in body["legs"]]
        return {**tickets.price_ticket(legs, body["type"], Decimal(body["stake"]), k=body.get("k"),
                                       system=body.get("system")), "removed": []}

    def build(self, uid, body):
        self.calls.append(("build", body["stake"], body["target"]))
        candidates = []
        for league in body["leagues"]:
            for card in self.service.games(league)["games"]:
                for market, sides in card["markets"].items():
                    for side in sides:
                        candidates.append(self._leg({"league": league, "id": card["id"], "market": market,
                                                     "side": side}))
        return target.build_for_target(candidates, body["stake"], body["target"], body["max_legs"], body.get("tier"))

    def analyse(self, uid, body, explain=True):
        self.calls.append(("analyse", explain))
        ticket = self.price(uid, body)
        return {"ticket": ticket, "legs": [{"index": leg["index"], "notes": ["No injuries reported"]}
                                           for leg in ticket["legs"]],
                "explanation": None, "alternatives": []}

    def rollover_compare(self, body):
        self.calls.append(("rollover", body["start"]))
        return rollover.compare(Decimal(body["start"]), body["days"], Decimal(body["min_odds"]), body["legs_per_day"])

    def player_search(self, q, league=None):
        players = [{"id": "1966", "name": "LeBron James", "team": "Los Angeles Lakers", "league": "basketball_nba"},
                   {"id": "4433134", "name": "Bronny James", "team": "Los Angeles Lakers", "league": "basketball_nba"}]
        return [p for p in players if q.lower() in p["name"].lower()]

    def player(self, league, athlete_id):
        return {"id": athlete_id, "league": league, "name": "LeBron James", "team": "Los Angeles Lakers",
                "position": "F", "headshot": "https://a.espncdn.com/x.png", "status": "Active",
                "last10": {"pts": 24.6, "reb": 7.9}, "sample": {"last10": 10}, "news": [],
                "props": [{"stat": "pts", "line": 24.5, "hit_rates": {"last10": "6/10"}, "ai_over": 0.52}],
                "as_of": "2027-01-15T17:00:00Z"}

    def me(self, uid):
        return {"odds_format": "decimal", "favourite_sports": ["basketball_nba"], "budget": None,
                "credits": {"left_cents": 90}}


def usage(fresh=1000, out=500, read=0):
    return SimpleNamespace(input_tokens=fresh, output_tokens=out, cache_creation_input_tokens=0,
                           cache_read_input_tokens=read, server_tool_use=None)


def message(content, stop="end_turn", model="claude-opus-5-5"):
    return SimpleNamespace(content=content, stop_reason=stop, usage=usage(), model=model)


def text(value):
    return SimpleNamespace(type="text", text=value)


def tool_use(name, args, use_id="toolu_1"):
    return SimpleNamespace(type="tool_use", id=use_id, name=name, input=args)


def route(intent="chat", **fields):
    data = {"intent": intent, "leagues": [], "stake": None, "target": None, "max_legs": None, "tier": "none",
            "days": None, "min_odds": None, "legs_per_day": None, "players": [], "risk_flags": []}
    data.update(fields)
    return message([text(json.dumps(data))], model="claude-haiku-5-5")


class FakeClaude:
    """Scripted answers: router calls (Haiku) and chat calls (Opus) come from separate queues."""

    def __init__(self, router=(), chat=()):
        self.router, self.chat = list(router), list(chat)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(json.loads(json.dumps(request, default=lambda o: getattr(o, "__dict__", str(o)))))
        queue = self.router if request["model"] == "claude-haiku-5-5" else self.chat
        if not queue:
            raise AssertionError(f"Unexpected call to {request['model']}")
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def chat_requests(self):
        return [r for r in self.requests if r["model"] != "claude-haiku-5-5"]


@pytest.fixture()
def store(tmp_path):
    store = Store(str(tmp_path / "betting.db"))
    store.get_or_create_user(UID, "Sam", now=NOW)
    store.confirm_age(UID, now=NOW)
    return store


@pytest.fixture()
def lab(service):
    return FakeLab(service)


def make(store, lab, claude, clock=NOW, daily=100):
    return Assistant(claude, lab, store, daily_cents=daily, tz=NY, clock=lambda: clock)


def build_route(**extra):
    return route("build", leagues=["basketball_nba"], stake=10, target=40, max_legs=3, **extra)


# --- the tool loop -----------------------------------------------------------------------------------

def test_a_build_runs_the_tool_loop_and_cards_come_from_the_tool_results(store, lab) -> None:
    built = lab.build(UID, {"leagues": ["basketball_nba"], "stake": "10.00", "target": "40.00", "max_legs": 3})
    best = built["tickets"][0]
    answer = (f"The closest fit is {best['name']}: about 1 in {best['chance']['ai_one_in']} by the AI and "
              f"1 in {best['chance']['market_one_in']} by the market, for your $10 to get about $40 back. "
              f"The bookmaker keeps about {best['book_cut'] * 100:.0f}% of money staked on tickets like this.")
    claude = FakeClaude(
        router=[build_route()],
        chat=[message([text("Let me build that."),
                       tool_use("ticket_build", {"leagues": ["basketball_nba"], "stake": 10, "target": 40,
                                                 "max_legs": 3})], stop="tool_use"),
              message([SimpleNamespace(type="thinking", thinking=""), text(answer)])],
    )
    steps = []
    out = make(store, lab, claude).reply(UID, message="NBA tonight, $10 to get about $40 back, max 3 legs",
                                         on_step=steps.append)

    assert out["reply"]["text"] == answer and out["reply"]["kind"] == "answer"
    cards = out["reply"]["cards"]
    assert cards and all(card["type"] == "ticket" and card["source"] == "ticket_build" for card in cards)
    assert cards[0]["ticket"]["legs"] == best["legs"]  # the card is the engine's ticket, not Claude's text
    assert cards[0]["refs"] == [{"league": leg["league"], "id": leg["event_id"], "market": leg["market"],
                                 "side": leg["side"], "point": leg["point"]} for leg in best["legs"]]
    assert "Explain the risk" in out["reply"]["chips"] and "Make it safer" in out["reply"]["chips"]
    assert out["disclosure"] == DISCLOSURE  # the first message of a thread
    assert "Building tickets…" in steps and steps[0] == "Reading your message…"

    first, second = claude.chat_requests()
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default"
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["output_config"] == {"effort": "medium"}  # building is analysis, not a look-up
    assert "tool_choice" not in first  # forced tool use is a 400 on Opus 5.5; auto is the default
    assert all(tool["strict"] and tool["input_schema"]["additionalProperties"] is False for tool in first["tools"])
    assert len(first["tools"]) == 12
    result = second["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_1" and not result["is_error"]
    assert "ceiling_one_in" in result["content"]
    context = first["messages"][-1]["content"][0]["text"]
    assert context.startswith("<context>") and '"stake": "10.00"' in context

    # Stored: both messages, the tools used (as summaries with refs) and the cost, taken from today's credits
    rows = store.messages(UID, out["thread_id"])
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert out["message_idx"] == rows[1]["idx"] == 1  # what buttons under this answer point at
    assert rows[1]["content"]["tools"][0].startswith("ticket_build(basketball_nba, stake 10.00 to 40.00")
    assert "[basketball_nba:" in rows[1]["content"]["tools"][0]
    assert rows[1]["cost"] > 0 and out["credits"]["used_cents"] > 0
    calls = store.analyst_spend("assistant")
    assert calls[0] == 2 and calls[1] == pytest.approx(rows[1]["cost"])  # one row per model: router + chat
    assert ("build", "10.00", "40.00") in lab.calls


def test_the_system_prompt_and_tools_stay_byte_identical_between_turns(store, lab) -> None:
    claude = FakeClaude(router=[route("explain"), route("explain")],
                        chat=[message([text("Odds show what a bet pays.")]), message([text("A parlay joins legs.")])])
    assistant = make(store, lab, claude)
    first = assistant.reply(UID, message="what do odds mean?")
    assistant.reply(UID, thread_id=first["thread_id"], message="and a parlay?")
    a, b = claude.chat_requests()
    assert a["system"] == b["system"] and a["tools"] == b["tools"]
    assert a["output_config"] == {"effort": "low"}  # explaining is a look-up
    assert [m["role"] for m in b["messages"]] == ["user", "assistant", "user"]  # the thread is the context
    router = [r for r in claude.requests if r["model"] == "claude-haiku-5-5"][0]
    assert "fallbacks" not in router  # Haiku 5.5 has no server-side fallback
    assert router["output_config"]["format"]["type"] == "json_schema"


def test_the_context_is_the_last_12_messages_with_tool_results_summarised(store, lab) -> None:
    thread = store.create_thread(UID, now=NOW)
    for i in range(10):
        store.add_message(UID, thread["id"], "user", {"text": f"question {i}"}, now=NOW)
        store.add_message(UID, thread["id"], "assistant",
                          {"text": f"answer {i}", "tools": [f"odds_get_prices(game {i}): Over 224.5 [ref {i}]"]}, now=NOW)
    claude = FakeClaude(router=[route("chat")], chat=[message([text("Hello again.")])])
    make(store, lab, claude).reply(UID, thread_id=thread["id"], message="hi")
    messages = claude.chat_requests()[0]["messages"]
    assert len(messages) == 13  # 12 earlier messages and this one
    assert messages[0]["content"] == "question 4"
    assert "odds_get_prices(game 9)" in messages[-2]["content"] and "call them again" in messages[-2]["content"]


# --- the checks on Claude's text -------------------------------------------------------------------------

def test_an_invented_number_is_sent_back_once_then_replaced_by_a_template(store, lab) -> None:
    ref = {"league": "basketball_nba", "id": "g2", "market": "h2h", "side": "home"}
    claude = FakeClaude(
        router=[route("analyse", leagues=["basketball_nba"], stake=10)],
        chat=[message([tool_use("ticket_price", {"type": "single", "stake": 10, "legs": [ref]})], stop="tool_use"),
              message([text("This pays $47.50 if the Knicks win.")]),   # not what the engine says
              message([text("Still $47.50, a 1 in 9 shot.")])],
    )
    out = make(store, lab, claude).reply(UID, message="price $10 on the Knicks")
    assert out["reply"]["kind"] == "template"
    retry = claude.chat_requests()[2]["messages"]
    assert retry[-1]["role"] == "user" and "$47.50" in retry[-1]["content"]
    assert retry[-2]["role"] == "assistant"
    priced = lab.price(UID, {"type": "single", "stake": "10", "legs": [ref]})
    assert out["reply"]["text"].startswith("Single: New York Knicks to win:")
    assert f"1 in {priced['chance']['ai_one_in']}" in out["reply"]["text"]
    assert "$47.50" not in out["reply"]["text"]
    assert out["reply"]["cards"][0]["ticket"]["max_payout"] == priced["max_payout"]  # the card is still shown
    assert store.messages(UID, out["thread_id"])[-1]["content"]["checked"] is False


def test_a_fixed_retry_is_shown(store, lab) -> None:
    ref = {"league": "basketball_nba", "id": "g2", "market": "h2h", "side": "home"}
    priced = lab.price(UID, {"type": "single", "stake": "10", "legs": [ref]})
    good = f"A $10 single on the Knicks pays {priced['max_payout']} back if they win: a favourite."
    claude = FakeClaude(
        router=[route("analyse", leagues=["basketball_nba"], stake=10)],
        chat=[message([tool_use("ticket_price", {"type": "single", "stake": 10, "legs": [ref]})], stop="tool_use"),
              message([text("It pays $99 back.")]), message([text(good)])],
    )
    out = make(store, lab, claude).reply(UID, message="price $10 on the Knicks")
    assert out["reply"]["text"] == good and out["reply"]["kind"] == "answer"


def test_banned_wording_is_never_shown(store, lab) -> None:
    claude = FakeClaude(router=[route("lookup", leagues=["basketball_nba"])],
                        chat=[message([text("The Thunder are a lock tonight.")]),
                              message([text("Honestly it's a sure thing, risk-free.")])])
    out = make(store, lab, claude).reply(UID, message="who wins Thunder Clippers?")
    assert out["reply"]["kind"] == "template"
    assert not banned_phrases(out["reply"]["text"])
    assert "banned wording: lock" in claude.chat_requests()[1]["messages"][-1]["content"]


def test_the_user_s_own_numbers_may_be_repeated(store, lab) -> None:
    claude = FakeClaude(router=[route("chat")], chat=[message([text("Sure: $25 stake and 3 legs noted.")])])
    out = make(store, lab, claude).reply(UID, message="I usually bet $25 on 3 legs")
    assert out["reply"]["kind"] == "answer"


# --- warning signs, gates and credits ------------------------------------------------------------------

def test_chasing_losses_gets_the_support_message_a_review_flag_and_no_picks_for_a_day(store, lab) -> None:
    claude = FakeClaude(router=[build_route()])
    assistant = make(store, lab, claude)
    out = assistant.reply(UID, message="lost $200 today, need to win it back tonight. give me a big parlay")
    assert out["reply"]["text"] == SUPPORT_MESSAGE and out["reply"]["kind"] == "support"
    assert not claude.requests  # code's own rules caught it: no model call, no cost
    event = store.rg_events(UID)[0]
    assert event["kind"] == "chasing" and event["action"] == "support_message,picks_off_24h"
    assert event["detail"]["thread_id"] == out["thread_id"]
    assert assistant.picks_paused_until(UID) == pytest.approx(NOW + 24 * 3600)

    # Later that day a build request is refused by code, whatever the model would say
    later = make(store, lab, claude, clock=NOW + 3600).reply(UID, thread_id=out["thread_id"],
                                                             message="ok just build me $10 to $40 NBA")
    assert later["reply"]["kind"] == "paused" and HELPLINE in later["reply"]["text"]
    assert not claude.chat_requests()
    # A day later picks are back on
    assert make(store, lab, claude, clock=NOW + 25 * 3600).picks_paused_until(UID) is None


def test_the_router_can_raise_a_warning_sign_code_missed(store, lab) -> None:
    claude = FakeClaude(router=[route("build", risk_flags=["debt"])])
    out = make(store, lab, claude).reply(UID, message="rent is due friday and I'm short, what's a good bet")
    assert out["reply"]["kind"] == "support"
    assert store.rg_events(UID)[0]["detail"]["source"] == "router"


def test_distress_adds_crisis_lines() -> None:
    assert risk_flags("I feel hopeless, I can't stop gambling") == ["distress"]
    assert risk_flags("rent money") == []
    assert risk_flags("The Celtics won back-to-back titles") == []
    assert risk_flags("can you get back to me on the Knicks?") == []
    assert risk_flags("I need to pay off my credit card") == ["debt"]
    assert risk_flags("chasing my losses") == ["chasing"]


# Everyday sports questions that share words with warning signs: none may pause anyone's picks
SPORTS_TALK = [
    "Can the Chiefs cover the Bills spread?",
    "Can the Thunder win back to back titles?",
    "Will the Bears recover from the loss to Detroit?",
    "Can Tatum make up for the loss of Brown?",
    "how do the Nuggets recover after losses on the road",
    "Will Denver win their next game back home?",
    "NBA, $10 to get about $50 back, max 3 legs",
    "If the game is postponed do I get my money back?",
    "What are my losses this month?",
]


@pytest.mark.parametrize("question", SPORTS_TALK)
def test_sports_talk_is_not_a_warning_sign(question) -> None:
    assert risk_flags(question) == [] and soft_flags(question) == []


@pytest.mark.parametrize("message, flag", [
    ("lost $200 today, need to win it back tonight", "chasing"), ("I want to win back what I lost", "chasing"),
    ("need to make my money back", "chasing"), ("recoup some of my losses", "chasing"),
    ("I lost my rent on parlays", "chasing"), ("I'm in debt and need a win", "debt"),
    ("I owe my bookie $500", "debt"), ("betting my rent money", "debt"), ("pay my bills with a parlay", "debt"),
    ("I have a gambling problem", "distress"),
])
def test_first_person_money_talk_is_a_strong_sign(message, flag) -> None:
    assert risk_flags(message) == [flag]


def test_wording_sports_talk_shares_is_only_a_soft_sign() -> None:
    for message, flag in (("chasing losses", "chasing"), ("Can the Chiefs win it back next year?", "chasing"),
                          ("double or nothing", "chasing"), ("pay the rent", "debt"), ("The Jets are hopeless", "distress")):
        assert risk_flags(message) == [] and soft_flags(message) == [flag], message
    assert soft_flags("chasing my losses") == []  # already strong: not counted twice


def test_sports_talk_gets_an_answer_and_no_pause(store, lab) -> None:
    claude = FakeClaude(router=[route("lookup", leagues=["americanfootball_nfl"])],
                        chat=[message([text("The Chiefs are favourites; the spread asks them to win by a few points.")])])
    assistant = make(store, lab, claude)
    out = assistant.reply(UID, message="Can the Chiefs cover the Bills spread?")
    assert out["reply"]["kind"] == "answer" and HELPLINE not in out["reply"]["text"]
    assert not store.rg_events(UID) and assistant.picks_paused_until(UID) is None


def test_a_soft_sign_counts_only_when_the_router_agrees(store, lab) -> None:
    # The router reads "win it back" as sports talk: an ordinary answer, nothing paused or logged
    claude = FakeClaude(router=[route("lookup", leagues=["americanfootball_nfl"]),
                                route("build", risk_flags=["chasing"])],
                        chat=[message([text("They lost the final last year.")])])
    assistant = make(store, lab, claude)
    first = assistant.reply(UID, message="Can the Chiefs win it back next year?")
    assert first["reply"]["kind"] == "answer" and not store.rg_events(UID)
    # The router reads "chasing losses" as a warning sign: the support message, with the rule noted
    second = assistant.reply(UID, message="been chasing losses all week, one more parlay")
    assert second["reply"]["kind"] == "support" and assistant.picks_paused_until(UID)
    event = store.rg_events(UID)[0]
    assert event["detail"]["source"] == "router" and event["detail"]["soft_rules"] == ["chasing"]


def test_a_soft_sign_with_the_router_down_shows_the_helpline_without_a_pause(store, lab) -> None:
    down = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    claude = FakeClaude(router=[down], chat=[message([text("Here's how a parlay works.")])])
    assistant = make(store, lab, claude)
    out = assistant.reply(UID, message="is chasing losses with a parlay smart?")
    assert out["reply"]["kind"] == "answer" and HELPLINE in out["reply"]["text"]
    assert assistant.picks_paused_until(UID) is None
    [event] = store.rg_events(UID)
    assert event["kind"] == "possible_chasing" and event["action"] == "helpline_note"


def test_while_paused_lookups_still_work_without_pricing_tools(store, lab) -> None:
    assistant = make(store, lab, FakeClaude())
    assistant._pause_picks(UID)
    claude = FakeClaude(router=[route("lookup", leagues=["basketball_nba"])],
                        chat=[message([tool_use("odds_get_prices", {"league": "basketball_nba", "event_id": "g1"})],
                                      stop="tool_use"),
                              message([text("I can't show prices right now, but here's how odds work.")])])
    out = make(store, lab, claude).reply(UID, message="odds for Thunder Clippers?")
    result = claude.chat_requests()[1]["messages"][-1]["content"][0]
    assert result["is_error"] and "picks_paused" in result["content"]
    assert "picks are paused" in claude.chat_requests()[0]["messages"][-1]["content"][0]["text"].lower()
    assert HELPLINE in out["reply"]["text"]
    assert "odds_get_prices" in PICK_TOOLS


def test_no_ai_until_the_age_is_confirmed(tmp_path, lab) -> None:
    store = Store(str(tmp_path / "fresh.db"))
    claude = FakeClaude()
    out = make(store, lab, claude).reply("tg:7", message="hi")
    assert out["reply"]["kind"] == "gate" and "18" in out["reply"]["text"]
    assert not claude.requests


def test_a_cool_off_switches_the_chat_off(store, lab) -> None:
    store.start_cooloff(UID, 72, now=NOW)
    out = make(store, lab, FakeClaude()).reply(UID, message="hi")
    assert out["reply"]["kind"] == "gate" and "on a break until" in out["reply"]["text"]


def test_when_the_credits_are_used_up_no_model_is_called(store, lab) -> None:
    store.add_ai_usage(UID, 100, now=NOW)
    claude = FakeClaude()
    assistant = make(store, lab, claude)
    out = assistant.reply(UID, message="who wins tonight?")
    assert out["reply"]["kind"] == "gate" and "today's AI allowance" in out["reply"]["text"]
    assert out["credits"] == {"used_cents": 100.0, "daily_cents": 100, "left_cents": 0.0}
    assert not claude.requests
    # The analyse screen shows the code-only analysis with this reason (the Lab shows it as a note)
    with pytest.raises(CallRejected, match="today's AI allowance"):
        assistant.explain(UID, {"ticket": {}})
    assert not claude.requests


def test_while_picks_are_paused_the_ai_won_t_explain_tickets(store, lab) -> None:
    claude = FakeClaude()
    assistant = make(store, lab, claude)
    assistant._pause_picks(UID)
    with pytest.raises(CallRejected, match="Picks are paused until") as rejected:
        assistant.explain(UID, _fact_sheet(lab))
    assert HELPLINE in str(rejected.value) and not banned_phrases(str(rejected.value))
    assert not claude.requests
    assert assistant_module.paused_until(store.user(UID), NOW) == pytest.approx(NOW + 24 * 3600)
    assert assistant_module.paused_until(store.user(UID), NOW + 25 * 3600) is None


def test_the_disclosure_comes_once_per_thread(store, lab) -> None:
    claude = FakeClaude(router=[route("chat"), route("chat")],
                        chat=[message([text("Hi! Ask me about games.")]), message([text("Glad to help.")])])
    assistant = make(store, lab, claude)
    first = assistant.reply(UID, message="hello")
    second = assistant.reply(UID, thread_id=first["thread_id"], message="thanks")
    assert first["disclosure"] == DISCLOSURE and "disclosure" not in second
    assert store.messages(UID, first["thread_id"])[1]["content"]["disclosure"] == DISCLOSURE


def test_someone_else_s_thread_and_empty_messages_are_refused(store, lab) -> None:
    other = store.create_thread("tg:99", now=NOW)
    with pytest.raises(NotFound):
        make(store, lab, FakeClaude()).reply(UID, thread_id=other["id"], message="hi")
    with pytest.raises(CallRejected):
        make(store, lab, FakeClaude()).reply(UID, message="   ")


# --- clarifying questions and the amounts Claude may use ------------------------------------------------

def test_a_missing_stake_and_target_are_asked_for_one_at_a_time(store, lab) -> None:
    claude = FakeClaude(
        router=[route("build", leagues=["basketball_nba"]), route("build", stake=10), route("chat", target=40)],
        chat=[message([tool_use("ticket_build", {"leagues": ["basketball_nba"], "stake": 10, "target": 40,
                                                 "max_legs": 3})], stop="tool_use"),
              message([text("Here's what fits.")])],
    )
    assistant = make(store, lab, claude)
    first = assistant.reply(UID, message="build me an NBA ticket")
    assert first["reply"]["text"] == "How much do you want to stake?"
    assert first["reply"]["chips"] == ["$5", "$10", "$25"] and first["reply"]["kind"] == "clarify"
    second = assistant.reply(UID, thread_id=first["thread_id"], message="$10")
    assert second["reply"]["text"].startswith("And how much would you like back in total")
    assert second["reply"]["chips"] == ["$20 back", "$50 back", "$100 back"]
    third = assistant.reply(UID, thread_id=first["thread_id"], message="$40 back")
    assert third["reply"]["kind"] == "answer" and third["reply"]["cards"]
    assert ("build", "10.00", "40.00") in lab.calls  # the stake from two messages ago was accepted
    assert '"intent": "build"' in claude.chat_requests()[0]["messages"][-1]["content"][0]["text"]


def test_a_lone_amount_fills_the_field_that_was_asked(store, lab) -> None:
    # The router reads "$40 back" as a stake: code knows the question was about the target
    claude = FakeClaude(router=[route("build", leagues=["basketball_nba"], stake=10), route("build", stake=40)],
                        chat=[message([text("Here's what fits.")])])
    assistant = make(store, lab, claude)
    first = assistant.reply(UID, message="NBA ticket for $10")
    assert store.messages(UID, first["thread_id"])[-1]["content"]["pending"]["asks"] == "target"
    second = assistant.reply(UID, thread_id=first["thread_id"], message="$40 back")
    assert second["reply"]["kind"] == "answer"
    assert '"stake": "10.00", "target": "40.00"' in claude.chat_requests()[0]["messages"][-1]["content"][0]["text"]


def test_the_question_is_answered_from_the_text_when_the_router_is_down(store, lab) -> None:
    down = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    claude = FakeClaude(router=[route("build", leagues=["basketball_nba"]), down])
    assistant = make(store, lab, claude)
    first = assistant.reply(UID, message="build me an NBA ticket")
    second = assistant.reply(UID, thread_id=first["thread_id"], message="10 dollars")
    assert second["reply"]["text"].startswith("And how much would you like back")


def test_claude_can_t_raise_the_stake(store, lab) -> None:
    claude = FakeClaude(
        router=[build_route()],
        chat=[message([tool_use("ticket_build", {"leagues": ["basketball_nba"], "stake": 100, "target": 400,
                                                 "max_legs": 3})], stop="tool_use"),
              message([text("Tell me if you want me to build it for your own stake.")])],
    )
    make(store, lab, claude).reply(UID, message="NBA, $10 to get $40 back")
    result = claude.chat_requests()[1]["messages"][-1]["content"][0]
    assert result["is_error"] and "amount_not_given" in result["content"]
    assert not [call for call in lab.calls if call[0] == "build"]


NBA_G1 = {"league": "basketball_nba", "id": "g1", "market": "h2h", "side": "home"}
NBA_G2 = {"league": "basketball_nba", "id": "g2", "market": "h2h", "side": "home"}


def test_a_ticket_s_stake_is_checked_over_all_its_bets() -> None:
    assert [bet_count("single", 3), bet_count("multi", 3), bet_count("round_robin", 4, 2),
            bet_count("system", 3, system="trixie"), bet_count("system", 4, system="yankee"),
            bet_count("system", 4, system="lucky15")] == [3, 1, 6, 4, 11, 15]
    assert bet_count("system", 3, system="yankee") == 1  # a shape pricing will refuse anyway


def test_claude_can_t_turn_a_10_dollar_stake_into_more_bets_at_10(lab) -> None:
    box = Toolbox(lab, UID, NY, {"stake": {Decimal("10")}}, leagues={"basketball_nba", "icehockey_nhl"})
    singles = box.run("slip_propose", {"type": "single", "stake": 10, "legs": [NBA_G1, NBA_G2], "title": "Singles"})
    assert not singles.ok and singles.content["error"] == "amount_not_given"
    assert "2 bets at 10.00 each, 20.00 in all" in singles.content["message"] and "split" in singles.content["hint"]
    four = [NBA_G1, NBA_G2, {**NBA_G1, "league": "icehockey_nhl"}, {**NBA_G2, "league": "icehockey_nhl"}]
    for system in ("yankee", "lucky15"):
        refused = box.run("slip_propose", {"type": "system", "system": system, "stake": 10, "legs": four, "title": "x"})
        assert refused.content["error"] == "amount_not_given", system
    assert not [call for call in lab.calls if call[0] == "price"]  # refused before anything was priced
    # split: code shares the person's $10 across the bets (Claude never divides)
    shared = box.run("slip_propose", {"type": "single", "stake": 10, "split": True, "legs": [NBA_G1, NBA_G2],
                                      "title": "Singles"})
    assert shared.ok and shared.cards[0]["stake"] == "5.00" and shared.cards[0]["ticket"]["total_stake"] == "10.00"
    assert "stake 5.00 per bet" in shared.summary
    thirds = box.run("ticket_price", {"type": "system", "system": "trixie", "stake": 10, "split": True,
                                      "legs": [NBA_G1, NBA_G2, {**NBA_G1, "league": "icehockey_nhl"}]})
    assert thirds.ok and Decimal(thirds.content["ticket"]["total_stake"]) <= 10  # 4 bets at 2.50
    assert box.run("ticket_price", {"type": "multi", "stake": 10, "legs": [NBA_G1, NBA_G2]}).ok  # one bet at $10


def test_a_slip_counts_at_its_total_so_its_own_bets_can_be_analysed(store, lab) -> None:
    slip = {"type": "single", "stake": 10, "legs": [NBA_G1, NBA_G2]}  # two $10 singles: $20 in all
    claude = FakeClaude(chat=[
        message([tool_use("ticket_analyse", dict(slip)),
                 tool_use("slip_propose", {"type": "multi", "stake": 25, "legs": [NBA_G1, NBA_G2], "title": "x"},
                          use_id="toolu_2")], stop="tool_use"),
        message([text("Your two singles, analysed.")])])
    make(store, lab, claude).reply(UID, slip=slip)
    analysed, bigger = claude.chat_requests()[1]["messages"][-1]["content"]
    assert not analysed["is_error"] and bigger["is_error"] and "amount_not_given" in bigger["content"]
    assert Assistant._slip_total({"type": "system", "system": "yankee", "stake": "1", "legs": [{}] * 4}) == 11


def test_a_turn_loads_odds_for_the_leagues_asked_about_plus_one_more(lab) -> None:
    box = Toolbox(lab, UID, NY, {"stake": {Decimal("10")}}, leagues={"basketball_nba"})
    assert box.run("sports_find_events", {"leagues": ["basketball_nba", "icehockey_nhl"], "query": "",
                                          "when": "week"}).ok  # the one extra league
    third = box.run("odds_get_prices", {"league": "americanfootball_nfl", "event_id": "g1"})
    assert not third.ok and third.content["error"] == "quota"
    assert "basketball_nba" in third.content["hint"] and "ask the user" in third.content["hint"]
    priced = box.run("ticket_price", {"type": "multi", "stake": 10, "legs": [{**NBA_G1, "league": "soccer_epl"}]})
    assert priced.content["error"] == "quota"
    assert not [call for call in lab.calls if call[0] == "today" and "americanfootball_nfl" in call[1]]
    assert box.run("odds_get_prices", {"league": "icehockey_nhl", "event_id": "g1"}).ok  # loaded this turn
    # A league whose odds are already loaded (and fresh) costs nothing, so it is always allowed
    lab.service.odds.odds_loaded_at = lambda league: (time.time() - 60 if league == "americanfootball_nfl" else None)
    assert box.run("odds_get_prices", {"league": "americanfootball_nfl", "event_id": "g1"}).ok
    lab.service.odds.odds_loaded_at = lambda league: time.time() - 3 * 3600  # stale: loading it would cost
    assert box.run("odds_get_prices", {"league": "baseball_mlb", "event_id": "g1"}).content["error"] == "quota"


def test_the_assistant_lets_claude_load_the_asked_for_and_favourite_leagues(store, lab) -> None:
    store.update_user(UID, {"favourite_sports": ["icehockey_nhl"]})
    claude = FakeClaude(router=[route("lookup", leagues=["basketball_nba"])], chat=[
        message([tool_use("sports_find_events", {"leagues": ["basketball_nba", "icehockey_nhl", "baseball_mlb"],
                                                 "query": "", "when": "today"}),
                 tool_use("sports_find_events", {"leagues": ["soccer_epl"], "query": "", "when": "today"},
                          use_id="toolu_2")], stop="tool_use"),
        message([text("Here's tonight's slate.")])])
    make(store, lab, claude).reply(UID, message="what's on tonight?")
    first, second = claude.chat_requests()[1]["messages"][-1]["content"]
    assert not first["is_error"]  # asked for, a favourite, and one more
    assert second["is_error"] and '"quota"' in second["content"]


def test_the_request_bar_and_a_slip_skip_the_router(store, lab) -> None:
    ref = {"league": "basketball_nba", "id": "g1", "market": "h2h", "side": "home"}
    claude = FakeClaude(chat=[
        message([tool_use("ticket_build", {"leagues": ["basketball_nba"], "stake": 5, "target": 20, "max_legs": 2,
                                           "tier": "risky"})], stop="tool_use"),
        message([text("Two options.")]),
        message([tool_use("ticket_analyse", {"type": "single", "stake": 5, "legs": [ref]})], stop="tool_use"),
        message([text("Your single, analysed.")]),
    ])
    assistant = make(store, lab, claude)
    built = assistant.reply(UID, request={"leagues": ["basketball_nba"], "stake": 5, "target": 20, "max_legs": 2,
                                          "tier": "risky"})
    assert store.messages(UID, built["thread_id"])[0]["content"]["text"] == "NBA, $5 to get about $20 back, max 2 legs, Risky"
    analysed = assistant.reply(UID, slip={"type": "single", "stake": 5, "legs": [ref]})
    assert analysed["reply"]["cards"][0]["source"] == "ticket_analyse"
    assert ("analyse", False) in lab.calls  # the chat writes its own words: no second explanation
    assert not [r for r in claude.requests if r["model"] == "claude-haiku-5-5"]


def test_a_failed_api_call_gets_a_plain_message_and_its_cost_is_kept(store, lab) -> None:
    busy = anthropic.RateLimitError("busy", response=httpx2.Response(
        429, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")), body=None)
    claude = FakeClaude(router=[route("chat")], chat=[busy])
    out = make(store, lab, claude).reply(UID, message="hi")
    assert out["reply"]["kind"] == "error" and "try again in a minute" in out["reply"]["text"].lower()
    assert store.ai_used_today(UID, now=NOW) > 0  # the router call still cost something


def test_late_at_night_the_reply_carries_a_note_and_the_owner_sees_it(store, lab) -> None:
    night = datetime.datetime(2027, 1, 15, 7, 30, tzinfo=datetime.timezone.utc).timestamp()  # 2:30 AM ET
    claude = FakeClaude(router=[route("chat")], chat=[message([text("Hi there.")])])
    out = make(store, lab, claude, clock=night).reply(UID, message="hi")
    assert "It's late where you are" in out["reply"]["text"]
    assert store.rg_events(UID, kind="late_night")


# --- the tools ----------------------------------------------------------------------------------------

def _inputs(schema):
    """Every input name in a tool's schema, nested ones included."""
    names = set(schema.get("properties") or {})
    for value in (schema.get("properties") or {}).values():
        names |= _inputs(value.get("items") or value)
    return names


def test_tools_take_refs_never_prices_or_probabilities() -> None:
    for tool in TOOLS:
        names = _inputs(tool["input_schema"]) - {"min_odds"}  # a rollover's daily odds, as the user gave them
        assert not [n for n in names if any(w in n for w in ("price", "prob", "odds", "chance", "payout"))], tool["name"]
        schema = json.dumps(tool["input_schema"])
        assert "minimum" not in schema and "maximum" not in schema  # ranges are checked in code


def test_a_slip_proposal_shows_a_card_and_saves_nothing(lab) -> None:
    box = Toolbox(lab, UID, NY, {"stake": {Decimal("10")}})
    ref = {"league": "basketball_nba", "id": "g2", "market": "totals", "side": "over"}
    run = box.run("slip_propose", {"type": "single", "stake": 10, "legs": [ref], "title": "Over in New York"})
    assert run.ok and run.cards[0]["proposal"] is True and run.cards[0]["title"] == "Over in New York"
    assert "nothing is saved" in run.content["status"]
    assert [call[0] for call in lab.calls] == ["price"]  # priced, never saved


def test_tool_errors_are_readable_with_a_hint(lab) -> None:
    box = Toolbox(lab, UID, NY, {"stake": {Decimal("10")}})
    gone = box.run("odds_get_prices", {"league": "basketball_nba", "event_id": "nope"})
    assert not gone.ok and gone.content["error"] == "not_found" and "sports_find_events" in gone.content["hint"]
    bad = box.run("ticket_price", {"type": "multi", "stake": -5, "legs": []})
    assert bad.content["error"] == "invalid_input" and "stake" in bad.content["message"]
    assert box.run("place_bet", {}).content["error"] == "unknown_tool"
    unsure = box.run("player_get_profile", {"name": "James", "league": "basketball_nba"})
    assert unsure.content["error"] == "ambiguous" and "Bronny James" in unsure.content["message"]
    no_stake = Toolbox(lab, UID, NY).run("ticket_price", {"type": "single", "stake": 10, "legs": [
        {"league": "basketball_nba", "id": "g1", "market": "h2h", "side": "home"}]})
    assert no_stake.content["error"] == "amount_not_given" and "Ask the user" in no_stake.content["hint"]


def test_lookups_return_compact_results_with_source_and_time(lab) -> None:
    box = Toolbox(lab, UID, NY, now=lambda: datetime.datetime(2029, 12, 31, 18, 0, tzinfo=datetime.timezone.utc))
    events = box.run("sports_find_events", {"leagues": ["basketball_nba"], "query": "knicks", "when": "week"})
    assert [e["game"] for e in events.content["events"]] == ["Orlando Magic @ New York Knicks"]
    assert events.content["source"] and events.content["as_of"]
    assert "[basketball_nba:g2]" in events.summary
    odds = box.run("odds_get_prices", {"league": "basketball_nba", "event_id": "g2"})
    outcome = next(o for o in odds.content["outcomes"] if o["ref"]["side"] == "home" and o["ref"]["market"] == "h2h")
    assert outcome["odds"]["decimal"] == 1.36 and outcome["ai_one_in"] and outcome["chance_label"]
    profile = box.run("player_get_profile", {"name": "LeBron James", "league": "basketball_nba"})
    assert profile.ok and "headshot" not in profile.content and profile.cards[0]["type"] == "player"
    plan = Toolbox(lab, UID, NY, {"stake": {Decimal("10")}}).run(
        "rollover_simulate", {"start": 10, "days": 5, "min_odds": 2.0, "legs_per_day": 1})
    assert plan.ok and plan.cards[0]["type"] == "plan" and plan.content["strategies"]["classic"]["market"]


# --- the numbers check and banned phrases --------------------------------------------------------------

def test_numbers_must_trace_to_the_sources() -> None:
    sources = source_numbers({"ai": 0.0806, "ai_one_in": 12, "book_cut": 0.1302, "expected": "-2.60",
                              "stake": "20.00", "summary": "about 1 in 1,901 tries", "american": "+597"})
    ok = ("About 1 in 12 (8%). The bookmaker keeps 13%. You lose about $2.60 of your $20 (+597). "
          "Tip-off 7:30 PM on Oct 26, 2026; 3 legs; 76ers; 18+; 1-800-GAMBLER / 0808 8020 133; about 1 in 1,900.")
    assert invented_numbers(ok, sources) == []
    assert invented_numbers("A 1 in 9 shot paying $45.20, 31%, odds 2.10", sources) == ["1 in 9", "$45.20", "31%", "2.10"]
    assert invented_numbers("about $100", source_numbers({"payout": 140})) == ["$100"]  # rounding stays close
    assert [f.text for f in numbers_in("leg 2 of 3, day 10")] == []
    assert invented_numbers("a 1-in-9 shot", sources) == ["1-in-9"]


def test_banned_phrases_and_every_fixed_message_pass_the_filter() -> None:
    for bad in ("a lock", "Guaranteed", "sure thing", "risk-free", "can't lose", "win it back", "win back your losses",
                "make your money back", "chasing losses", "it's safe",
                "a safe bet", "bet more", "raise your stake", "bigger stake"):
        assert banned_phrases(bad), bad
    for fine in ("the Safe tier", "won back-to-back games", "can win back to back titles", "Denver won back home",
                 "they can make it back to the playoffs", "Denver recovered after losses on the road",
                 "Tatum has to make up for the loss of Brown", "you get your stake back if it's void",
                 "make sure", "blocks", "Nothing is guaranteed.",
                 "Never bet more than you can afford to lose.", "You can't lose more than your $10 stake.",
                 "Don't raise your stake to catch up.", "There's no such thing as a risk-free bet."):
        assert not banned_phrases(fine), fine
    for still_bad in ("Not a lock, but close.", "This is guaranteed. Not kidding.", "Don't worry, it can't lose."):
        assert banned_phrases(still_bad), still_bad
    fixed = [SUPPORT_MESSAGE, assistant_module.DISTRESS_EXTRA, DISCLOSURE, HELPLINE, assistant_module.LATE_NOTE,
             assistant_module.SYSTEM_PROMPT.split("How to talk")[0]]
    fixed += assistant_module.TICKET_CHIPS + assistant_module.PLAN_CHIPS + assistant_module.SUPPORT_CHIPS
    assert all(not banned_phrases(message) for message in fixed)


def test_router_answers_are_checked_field_by_field() -> None:
    parsed = parse_route({"intent": "build", "leagues": ["basketball_nba", "made_up_league"], "stake": -5,
                          "target": 50, "max_legs": 9, "tier": "none", "days": None, "min_odds": 1.1,
                          "legs_per_day": None, "players": [], "risk_flags": ["chasing", "bogus"]})
    assert parsed.leagues == ["basketball_nba"] and parsed.stake is None and parsed.target == Decimal("50.00")
    assert parsed.max_legs is None and parsed.min_odds is None and parsed.flags == ["chasing"]
    assert parse_route("not json").intent == "chat"


# --- explaining a ticket -----------------------------------------------------------------------------------

def _fact_sheet(lab):
    refs = [{"league": "basketball_nba", "id": "g1", "market": "h2h", "side": "home"},
            {"league": "basketball_nba", "id": "g2", "market": "totals", "side": "over"}]
    ticket = lab.price(UID, {"type": "multi", "stake": "10", "legs": refs})
    return {"ticket": ticket, "legs": [{"index": 0, "notes": ["Rested"]}, {"index": 1, "notes": []}]}


def test_explain_checks_claude_s_words_against_the_fact_sheet(store, lab) -> None:
    sheet = _fact_sheet(lab)
    ticket = sheet["ticket"]
    good = {"verdict": f"A long shot: about 1 in {ticket['chance']['ai_one_in']} by the AI.",
            "per_leg": [{"index": 0, "text": "The Thunder are rested."}, {"index": 1, "text": "The total is close."}],
            "case_against": f"The bookmaker keeps about {ticket['book_cut'] * 100:.0f}%."}
    claude = FakeClaude(chat=[message([text(json.dumps(good))])])
    out = make(store, lab, claude).explain(UID, sheet)
    assert out["source"] == "claude" and out["verdict"] == good["verdict"] and len(out["per_leg"]) == 2
    request = claude.chat_requests()[0]
    assert request["output_config"]["effort"] == "medium"
    assert request["output_config"]["format"]["schema"]["required"] == ["verdict", "per_leg", "case_against"]
    assert store.analyst_spend("assistant")[0] == 1 and store.ai_used_today(UID, now=NOW) > 0


def test_explain_falls_back_to_the_template_after_two_bad_answers(store, lab) -> None:
    sheet = _fact_sheet(lab)
    bad = {"verdict": "A lock at 1 in 2.", "per_leg": [], "case_against": "None."}
    claude = FakeClaude(chat=[message([text(json.dumps(bad))]), message([text(json.dumps(bad))])])
    out = make(store, lab, claude).explain(UID, sheet)
    assert out["source"] == "template"
    assert out == {**template_explanation(sheet), "source": "template", "model": "claude-opus-5-5",
                   "cost": out["cost"]}
    assert f"1 in {sheet['ticket']['chance']['ai_one_in']}" in out["verdict"]
    assert not invented_numbers(" ".join([out["verdict"], out["case_against"]] + [l["text"] for l in out["per_leg"]]),
                                source_numbers(sheet))


def test_explain_uses_the_template_when_claude_is_unreachable(store, lab) -> None:
    down = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    out = make(store, lab, FakeClaude(chat=[down])).explain(UID, _fact_sheet(lab))
    assert out["source"] == "template" and out["model"] is None


def test_explain_ticket_fills_legs_claude_left_out(lab) -> None:
    sheet = _fact_sheet(lab)
    partial = {"verdict": "Two legs.", "per_leg": [{"index": 1, "text": "Close total."}], "case_against": "Two cuts."}
    result = explain_ticket(sheet, ClaudeCaller(FakeClaude(chat=[message([text(json.dumps(partial))])])))
    assert [leg["index"] for leg in result.per_leg] == [0, 1] and result.per_leg[1]["text"] == "Close total."
    assert result.per_leg[0]["text"].startswith("Oklahoma City Thunder to win")


# --- wiring -------------------------------------------------------------------------------------------------

def test_build_assistant_needs_a_key_and_reads_the_config(store, lab) -> None:
    assert build_assistant(Config(TELEGRAM_BOT_TOKEN="1:x"), lab, store) is None
    config = Config(TELEGRAM_BOT_TOKEN="1:x", ANTHROPIC_API_KEY="sk-ant-test", AI_CHAT_DAILY_CENTS=50,
                    AI_CHAT_MODEL="claude-sonnet-5-5", AI_ROUTER_MODEL="claude-haiku-5-5")
    built = build_assistant(config, lab, store, client=FakeClaude())
    assert built.daily_cents == 50 and built.chat_model == "claude-sonnet-5-5"
    assert built.credits(UID)["daily_cents"] == 50
    defaults = Config(TELEGRAM_BOT_TOKEN="1:x")
    assert (defaults.ai_chat_daily_cents, defaults.ai_chat_model, defaults.ai_router_model) == (
        100, "claude-opus-5-5", "claude-haiku-5-5")


# --- with the real Lab ------------------------------------------------------------------------------------

def test_with_the_real_lab_the_chat_prices_and_proposes_and_the_analysis_is_explained(service, store) -> None:
    from telegram_betting_bot_simple.dashboard.lab import Lab

    service.store = store
    real = Lab(service, store, now=lambda: NOW)
    ref = {"league": "basketball_nba", "id": "g2", "market": "h2h", "side": "home"}
    priced = real.price(UID, {"type": "single", "stake": "10", "legs": [ref]})
    answer = f"The Knicks single pays {priced['max_payout']} back: a favourite at about 1 in {priced['chance']['ai_one_in']}."
    claude = FakeClaude(
        router=[route("analyse", leagues=["basketball_nba"], stake=10)],
        chat=[message([tool_use("slip_propose", {"type": "single", "stake": 10, "legs": [ref],
                                                 "title": "Knicks to win"})], stop="tool_use"),
              message([text(answer)])],
    )
    assistant = Assistant(claude, real, store, tz=NY, clock=lambda: NOW)
    real.assistant = assistant
    out = real.chat(UID, {"message": "price $10 on the Knicks to win"})
    assert out["reply"]["text"] == answer and out["reply"]["cards"][0]["proposal"] is True
    assert out["reply"]["cards"][0]["refs"] == priced["refs"]
    assert not store.slips(UID)  # a proposal saves nothing

    # The analyse screen: the Lab's fact sheet goes to explain(); a bad answer twice gives the template
    two = [ref, {"league": "basketball_nba", "id": "g1", "market": "totals", "side": "over"}]
    bad = {"verdict": "A lock.", "per_leg": [], "case_against": "None."}
    claude.chat += [message([text(json.dumps(bad))]), message([text(json.dumps(bad))])]
    analysis = real.analyse(UID, {"type": "multi", "stake": "10", "legs": two})
    explanation = analysis["explanation"]
    assert explanation["source"] == "template" and [leg["index"] for leg in explanation["per_leg"]] == [0, 1]
    assert explanation["per_leg"][0]["text"].startswith("New York Knicks to win at 1.36")
    assert "The weakest leg is" in explanation["case_against"]


def test_the_template_never_reports_a_negative_cut() -> None:
    sheet = {"ticket": {"book_cut": -0.03, "chance": {"ai": 0.3, "ai_one_in": 3}, "expected": {"ai": "0.50"},
                        "total_stake": "10.00", "legs": []}}
    verdict = template_explanation(sheet)["verdict"]
    assert "keeps" not in verdict and "beat the bookmakers' consensus" in verdict
    assert "On average you make about $0.50 on your $10.00" in verdict


def test_the_real_lab_shows_why_there_is_no_ai_explanation(service, store) -> None:
    from telegram_betting_bot_simple.dashboard.lab import Lab

    service.store = store
    real = Lab(service, store, now=lambda: NOW)
    claude = FakeClaude()
    assistant = Assistant(claude, real, store, tz=NY, clock=lambda: NOW)
    real.assistant = assistant
    body = {"type": "multi", "stake": "10", "legs": [{"league": "basketball_nba", "id": "g2", "market": "h2h",
                                                      "side": "home"},
                                                     {"league": "basketball_nba", "id": "g1", "market": "totals",
                                                      "side": "over"}]}
    store.add_ai_usage(UID, 100, now=NOW)
    used_up = real.analyse(UID, body)
    assert used_up["explanation"] is None and "today's AI allowance" in used_up["note"]
    assert used_up["ticket"]["legs"]  # the code's numbers are all still there
    assistant._pause_picks(UID)
    paused = real.analyse(UID, body)
    assert paused["explanation"] is None and paused["note"].startswith("Picks are paused until")
    assert not claude.requests
