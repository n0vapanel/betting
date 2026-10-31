"""The P0 fix phase's AI and bot findings (scratchpad/build/SPEC_P0_FIX.md, fixer fix:ai), each as a regression test
that failed on the reviewed tree (scratchpad/p0/prefix_tree) and holds the fix. The cases are the reviewers' own:

- the persona review (scratchpad/p0/review-persona/FINDINGS.md): B1 and M9 (responsible gambling: a decline after a
  rough night stands, and the router-down reading pushes no picks after a warning sign), M1-M3 (the guard's
  precision, from guard_corpus.py and flows/probe_units_test.py), M4 (an honest "can't" about a player's past or
  futures), M5 (the started-games note), M6 (an explicit day wins over a weekday), M7 ("besides luka" stays a list),
  M8 (the bot's best-edges "Also:" rows), m1 (one "more", worded by mode, with a Player finder button in the bot),
  m2 (yes/no and hot rows), m3 (the router's clock), m4 ("who's hot" lets the Lab pick the league) and m5 (the
  router-down misreadings);
- the data review (scratchpad/p0/review-data/FINDINGS.md): F6 (= M5, through the real Lab) and F3's and F2's wording
  (contracts 1 and 2: the names the scan couldn't rank, and the trend against this season or last).

Every new guard or warning-sign pattern has a positive and a lookalike that isn't one. A scripted fake Claude plays the
router and the analyst through the real assistant, Lab and store (tests/test_player_scan_wiring.py's world); no
network."""

from __future__ import annotations

import datetime
import re
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import requests

from telegram_betting_bot_simple.ai import assistant as assistant_module
from telegram_betting_bot_simple.ai import explain as explain_module
from telegram_betting_bot_simple.ai.assistant import Assistant, Route, risk_flags, scan_fallback, soft_flags
from telegram_betting_bot_simple.ai.explain import answering_tools, refusal_problems, scan_notes, scan_template
from telegram_betting_bot_simple.ai.tools import TOOLS, Toolbox, scan_card, scan_facts
from telegram_betting_bot_simple.bot import chat as bot_chat
from telegram_betting_bot_simple.data.odds_service import configure_books
from tests.test_builder_wiring import make_world
from tests.test_player_scan_wiring import DK, NBA, NFL, NOW, NY, OLD_REFUSAL, UID, FakeBoard, FakeBoards, \
    nba_boards, routed, row, scan
from tests.test_review2_persona import ContractScan, Scripted, asked_scan, call, found, run, text

SATURDAY = datetime.datetime(2026, 10, 31, 18, 5, tzinfo=datetime.timezone.utc).timestamp()  # 2:05 PM in New York


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
    w.boards = nba_boards()
    w.lab.player_board = w.boards
    return w


def chat_in(world, thread_id, route, script, message: str):
    """One turn in a given thread (None: a new one), as tests/test_review2_persona.py's run: (reply, claude, thread)."""
    claude = Scripted(route, script)
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    out = world.lab.chat(UID, {"message": message, "thread_id": thread_id})
    return out["reply"], claude, out["thread_id"]


def context_of(claude) -> str:
    """The <context> block code sent the chat model on the turn's first request."""
    chat = claude.chat()
    return chat[0]["messages"][-1]["content"][0]["text"] if chat else ""


def _qualifiers(count: int) -> List[Dict[str, Any]]:
    return [row(i, f"Player {i}", "BOS", "NYK", 30.0 - i * 0.4, round(0.86 - i * 0.01, 4), [9, 10], [40, 45],
                athlete=str(9000 + i)) for i in range(1, count + 1)]


# --- B1 (blocker): the model's own decline after a rough night is never turned into code's picks ------------------

DECLINE = "I can't give you picks right now. Take a breather tonight and come back fresh."
HOLD_OFF = "I'd hold off on picks tonight. I can't recommend more picks after that run. Take a break."


def test_b1_a_cautious_decline_after_a_rough_night_stands_without_helpline_or_settings_words(world) -> None:
    # flows/probe_flows_test.py::test_f1: an earlier turn in the thread says "rough night, lost 3 parlays in a row"
    # (no strong rule fires on it); "give me 3 locks" gets the model's own decline, with no helpline or Settings words
    _, _, thread = chat_in(world, None, routed(intent="chat"), [text("Sorry to hear that.")],
                           "rough night, lost 3 parlays in a row")
    out, claude, _ = chat_in(world, thread, routed(top_picks=True, leagues=[NBA]), [text(DECLINE)], "give me 3 locks")
    assert not re.search(r"GAMBLER|8020|helpline|Settings", DECLINE)
    assert out["kind"] == "answer" and out["text"].startswith(DECLINE), out
    assert not out["cards"] and claude.chat_calls == 1 and not claude.corrections()  # no retry, no code-run picks


def test_b1_take_a_break_in_its_own_sentence_stands(world) -> None:
    # flows/probe_flows_test.py::test_f1b: no earlier turn; the wind-down words alone are the support rule at work
    out, claude = run(world, routed(top_picks=True, leagues=[NBA]), [text(HOLD_OFF)], "give me 3 locks")
    assert out["kind"] == "answer" and out["text"].startswith(HOLD_OFF), out
    assert not out["cards"] and claude.chat_calls == 1 and not claude.corrections()


def test_b1_loss_talk_in_the_last_three_turns_keeps_a_plain_decline_and_still_guards_without_it(world) -> None:
    plain = "I can't give you picks right now."  # no wind-down words, no helpline: only the thread says why
    _, _, thread = chat_in(world, None, routed(intent="chat"), [text("Sorry to hear that.")], "down bad this week")
    chat_in(world, thread, routed(intent="chat"), [text("Sure.")], "anyway")
    chat_in(world, thread, routed(intent="chat"), [text("Hi again.")], "hey")  # three turns on, it still counts
    out, claude, _ = chat_in(world, thread, routed(top_picks=True, leagues=[NBA]), [text(plain)], "give me 3 locks")
    assert out["kind"] == "answer" and out["text"].startswith(plain) and not out["cards"], out
    assert claude.chat_calls == 1 and not claude.corrections()
    # A thread without the loss talk: the same refusal is pushed back, and code's picks answer (m2 still works)
    out, claude, _ = chat_in(world, None, routed(top_picks=True, leagues=[NBA]), [text(plain)], "give me 3 locks")
    assert out["kind"] == "template" and [card["type"] for card in out["cards"]] == ["picks"]


WIND_DOWN_REPLIES = [
    "Take a break tonight.", "Take a breather and come back fresh.", "Step away for tonight.", "I'd sit this one out.",
    "Maybe sit tonight out.", "I'd hold off on picks tonight.", "Call it a night.", "Come back fresh tomorrow.",
    "Cool off for a bit.", "No picks tonight.", "No more bets tonight, honestly.", "Sleep on it.",
    "Take the night off.", "I'd stop for the night.", "Let's leave it there for tonight.", "I'd pass on this one.",
]
NOT_WIND_DOWN = [  # analyst replies that share the words
    "Luka is 82% for 20+ tonight: 9 of his last 10.", "His step-back three is falling: 31.2 projected.",
    "The Celtics held off a late run last time they met.", "He's cooling off: L5 21.0 vs 27.4 season.",
    "Get the rest of the list in the Player finder.", "No NBA games today, so here's tomorrow's list.",
    "Brunson broke out for 40 last night.", "The board is still loading: ask again in a minute.",
]


@pytest.mark.parametrize("reply", WIND_DOWN_REPLIES)
def test_b1_wind_down_wording_counts_as_support(reply) -> None:
    assert Assistant._guard("I can't give you picks right now. " + reply, Route(intent="lookup", top_picks=True),
                            False, "give me 3 locks", []) == [], reply
    assert assistant_module.support_reply(reply), reply


@pytest.mark.parametrize("reply", NOT_WIND_DOWN)
def test_b1_analyst_replies_that_share_the_words_are_not_wind_down(reply) -> None:
    assert not assistant_module.WIND_DOWN.search(reply) and not assistant_module.support_reply(reply), reply


LOSS_TALK = [  # (message, a lookalike that isn't the person's own loss)
    ("rough night, lost 3 parlays in a row", "the Lakers have lost 3 in a row"),
    ("lost 4 straight, need a lock", "Boston lost 4 straight at home"),
    ("I've lost 5 in a row", "They've lost 5 straight"),
    ("lost two bets tonight", "The Jets lost two picks in the trade"),
    ("lost a couple parlays", "the Knicks lost 3 games in a row"),
    ("down bad this week", "The Celtics are down bad at the half"),
    ("I'm down bad", "who's scoring 20+ tonight"),
    ("rough night", "rough night for the Lakers"),
    ("had a rough week", "Luka had a rough night from three"),
    ("what a rough night", "Tatum had a brutal night shooting"),
    ("it's been a rough week", "Luka had a bad game"),
]


@pytest.mark.parametrize("message, lookalike", LOSS_TALK)
def test_b1_loss_talk_is_a_soft_sign_and_its_lookalike_is_not(message, lookalike) -> None:
    assert risk_flags(message) == [] and soft_flags(message) == ["losses"], message
    assert soft_flags(lookalike) == [] and risk_flags(lookalike) == [], lookalike


MONEY_BACK = [  # (message, flags, a lookalike: a refund, or sports talk)
    ("gimme 2 locks, gotta get that money back", ["chasing"], "If the game is postponed do I get my money back?"),
    ("need to get my money back tonight", ["chasing"], "do I get the money back if it's voided?"),
    ("win it all back tonight", ["chasing"], "Can the Lakers make it back to the playoffs?"),
    ("make it back", ["chasing"], "The Thunder can win back-to-back titles"),
    ("chasing losses", ["chasing"], "What are my losses this month?"),
]


@pytest.mark.parametrize("message, flags, lookalike", MONEY_BACK)
def test_b1_money_wanted_back_is_soft_chasing_and_a_refund_is_not(message, flags, lookalike) -> None:
    assert soft_flags(message) == flags and risk_flags(message) == [], message
    assert soft_flags(lookalike) == [] and risk_flags(lookalike) == [], lookalike
    assert risk_flags("chasing my losses") == ["chasing"] and soft_flags("chasing my losses") == []  # strong already


def test_b1_the_app_s_rules_count_across_the_whole_reply() -> None:
    # "I can't ..." in one sentence and the break in the next: the rule at work, not a refusal
    for reply in ("I can't give you picks right now. Take a break.", "I can't add that. Only you add picks to your slip."):
        assert refusal_problems(reply, "lookup", {"brain_top_picks"}, asked="give me 3 locks") == [], reply
    # Without any rule talk the refusal is still caught
    assert refusal_problems("I can't give you picks right now.", "lookup", {"brain_top_picks"}, asked="locks")


# --- M9: with the router down, a warning sign gets no code-pushed picks or scan ----------------------------------

@pytest.mark.parametrize("message", ["need 3 locks to win it back tonight", "double or nothing, give me a lock",
                                     "gimme 2 locks, gotta get that money back"])
def test_m9_router_down_after_a_soft_sign_pushes_no_picks_and_says_suggest_no_bets(world, message) -> None:
    # flows/probe_rg_routerdown_test.py
    assert soft_flags(message) and not risk_flags(message)
    out, claude = run(world, None, [text("Not tonight: I'd step away from the picks for now.")], message)
    context = context_of(claude)
    assert "call brain_top_picks" not in context and "players_scan" not in context, context
    assert "suggest no bets and price nothing" in context and '"top_picks"' not in context
    assert out["kind"] == "answer" and not out["cards"] and "1-800-GAMBLER" in out["text"]


def test_m9_router_down_after_loss_talk_earlier_in_the_thread_pushes_no_picks(world) -> None:
    _, _, thread = chat_in(world, None, None, [text("Sorry to hear that.")], "rough night, lost 3 parlays in a row")
    out, claude, _ = chat_in(world, thread, None, [text("I'd sit tonight out.")], "give me 3 locks")
    context = context_of(claude)
    assert "call brain_top_picks" not in context and "suggest no bets" in context
    # Without a sign the router-down reading still gives the strongest picks (m11)
    out, claude, _ = chat_in(world, None, None, [call("brain_top_picks", {"leagues": [NBA], "n": 3}),
                                                 text("On the card.")], "give me 3 locks")
    assert "call brain_top_picks" in context_of(claude) and "suggest no bets" not in context_of(claude)


# --- M1: ordinary analyst replies the guard took for refusals or lectures ----------------------------------------

NORMAL_REPLIES = [  # guard_corpus.py's NORMAL rows the new patterns caught (each passed on c5deb8f)
    "I could only find one player projected for 30+ tonight: Luka Doncic at 31.2 projected and 58%.",
    "I can only see one player clearing 15+ boards tonight: Jokic at 13.9 projected, 41%.",
    "The book has the edge on the under here: edge -4.2%, so the over is the only side with value.",
    "On his points line the book has the edge (edge -3.1%): his 49% doesn't beat the price.",
    "The sportsbook has an edge on both sides of that line, so there's nothing worth a look.",
    "DraftKings hasn't posted PRA lines yet, so we won't be able to show edges until they do; by projection Luka "
    "leads at 48.2.",
    "I won't be able to show his line until DraftKings posts it; his projection is 31.2 and 82% for 20+.",
    "It's a toss-up between Luka and SGA for the top spot: 31.2 and 30.9 projected.",
    "Luka and SGA are close to 50-50 for who scores more tonight, 31.2 against 30.9.",
    "The Lakers and Celtics were tied 50-50 at the half last time they met.",
    "The Chiefs won the coin toss in both meetings and deferred.",
    "He's had a 50/50 split of home and road games so far.",
    "I'd need a player's name for his alt lines; for the list, Luka leads at 82%.",
]
STILL_CAUGHT = [  # (the same words as the lecture or refusal they still are, its name)
    ("I can only look up one player at a time. Who do you want me to check?", "I can only look up one"),
    ("I can only pull up one player.", "I can only look up one"),
    ("Remember that the sportsbook always has the edge.", "has the edge"),
    ("The books have the edge on props, so size small.", "has the edge"),
    ("I won't be able to list every player projected for 20 points.", "won't be able to"),
    ("These props are basically a toss-up.", "toss-up"),
    ("Points overs are close to 50/50.", "50/50"),
    ("It's a coin toss whether he gets there.", "coin toss"),
    ("I'd need a player's name: the props lookup works for a single player.", "I'd need a player's name"),
]


@pytest.mark.parametrize("reply", NORMAL_REPLIES)
def test_m1_ordinary_analyst_replies_are_not_refusals_or_lectures(reply) -> None:
    assert refusal_problems(reply, "lookup", {"players_scan"}, asked="who's scoring 20+ tonight") == [], reply


@pytest.mark.parametrize("reply, name", STILL_CAUGHT)
def test_m1_the_narrowed_patterns_still_catch_the_real_thing(reply, name) -> None:
    found_ = refusal_problems(reply, "lookup", {"players_scan"}, asked="who's scoring 20+ tonight")
    assert found_ and f'"{name}"' in " ".join(found_), (reply, found_)


def test_m1_a_one_qualifier_answer_and_a_negative_edge_read_reach_the_person(world) -> None:
    # flows/probe_flows_test.py::test_f3 and test_f3b: each ended as the template after three chat calls
    one = {"players": [row(1, "Luka Dončić", "DAL", "PHX", 31.2, 0.58, [5, 10], [20, 45], threshold=30,
                           athlete="3945274")], "close": []}
    world.lab.player_board = FakeBoards({(NBA, "today"): FakeBoard(NBA, "today", {"pts": one}, games_total=10)})
    said = "I could only find one player projected for 30+ tonight: Luka Dončić, 31.2 projected and 58%."
    out, claude = run(world, routed(player_scan=scan("pts", 30)), [asked_scan(), text(said), text(said)],
                      "who's scoring 30+ tonight")
    assert out["kind"] == "answer" and claude.chat_calls == 2 and not claude.corrections(), out
    line = {"kind": "over_under", "line": 31.5, "over": {"decimal": 1.87}, "under": {"decimal": 1.95},
            "book": "DraftKings"}
    lined = {"players": [row(1, "Luka Dončić", "DAL", "PHX", 31.2, 0.49, [5, 10], [20, 45], threshold=32,
                             athlete="3945274", line=line, edge=-0.0837)], "close": []}
    world.lab.player_board = FakeBoards({(NBA, "today"): FakeBoard(NBA, "today", {"pts": lined}, games_total=10)})
    said = "Luka's points line is 31.5: he's 49% to clear it, so the book has the edge here (edge -8.4%)."
    out, claude = run(world, routed(player_scan={**scan("pts", None), "player": "Luka Dončić"},
                                    players=["Luka Dončić"]),
                      [asked_scan(), text(said), text(said)], "is luka going over his points line")
    assert out["kind"] == "answer" and claude.chat_calls == 2 and not claude.corrections(), out


# --- M2: the commonest refusals and lectures it missed --------------------------------------------------------------

NEW_REFUSALS = [  # (the refusal, a lookalike that isn't one, the phrase's name)
    ("I don't have access to a full list of players projected for 20+ tonight.",
     "I don't have a favourite: the numbers lean Luka at 82%.", "I don't have access"),
    ("I don't have the ability to scan every player, but name one and I'll check his line.",
     "I don't have his line yet: DraftKings hasn't posted it. He's 82% for 20+.", "I don't have access"),
    ("The tools here only let me look up one player at a time.",
     "The Celtics only let one player shoot over 40% from three: Luka is 82% for 20+.", "only lets me … one"),
    ("There's no way to list every player projected for 20+ from here.",
     "There's no way he sits tonight: no injury tag, 35.5 projected minutes.", "no way to list"),
    ("I'm not going to pick locks for you.", "I'm not going to overthink it: Luka at 82% is the strongest.",
     "I'm not going to"),
    ("I won't give locks: nothing is certain.", "I won't list all ten here: the card below has every row.",
     "I won't give"),
    ("Player lists aren't available here. Name a player and I'll check him.",
     "His line isn't available yet: DraftKings posts it closer to tip-off.", "isn't available here"),
    ("I'm limited to one player at a time.", "His minutes are limited to 24 tonight: 18.6 projected.",
     "I'm limited to"),
    ("My tools are limited to single-player lookups, so give me a name.",
     "Luka is limited to 30 minutes: 27.9 projected.", "I'm limited to"),
    ("Sorry, a league-wide scan isn't something the app supports.", "That's something I can check: Luka is 82%.",
     "isn't available here"),
    ("That's beyond what I can do here: I check props player by player.",
     "That's beyond what Luka usually does: 31.2 projected.", "beyond what I can do"),
    ("You'll need to check each player individually in the Player finder.",
     "Check each player's card for his line: Luka is 82%.", "check each player individually"),
]
NEW_LECTURES = [  # (the lecture, a lookalike that isn't one, the phrase's name)
    ("Keep in mind props are volatile and anything can happen.", "He can do anything with the ball: 31.2 projected.",
     "anything can happen"),
    ("None of these are sure: there's always risk.", "His minutes risk is low: 35.5 projected minutes.",
     "always … risk"),
    ("Even an 82% shot loses about 1 time in 5.", "The Knicks lost 1 in 5 at home in October.", "loses 1 in N"),
    ("Parlays are designed to lose.", "The Celtics' defense is designed to force turnovers.", "designed to lose"),
    ("Remember, sports betting is unpredictable.", "His minutes are unpredictable on a back-to-back: 31.2 projected.",
     "betting is unpredictable"),
    ("The odds are always against you over time.", "The odds are against the Knicks tonight: 34%.",
     "the odds are against you"),
    ("Past results aren't predictive of future performance.", "His past 5 games average 28.4.",
     "past results aren't predictive"),
    ("There's a real chance it misses.", "He has a real chance at 40 tonight: 31.2 projected.",
     "a real chance it misses"),
    ("Variance is high on player props, so size accordingly.", "His usage variance is low: 9 of last 10.",
     "variance is high"),
]


@pytest.mark.parametrize("said, lookalike, name", NEW_REFUSALS)
def test_m2_each_new_refusal_is_caught_and_its_lookalike_is_not(said, lookalike, name) -> None:
    found_ = refusal_problems(said, "lookup", {"players_scan"}, asked="who's scoring 20+ tonight")
    assert found_ and found_[0].startswith("a refusal") and f'"{name}"' in found_[0], (said, found_)
    assert refusal_problems(said, "lookup", set(), asked="who's scoring 20+ tonight") == []  # nothing to call
    assert refusal_problems(lookalike, "lookup", {"players_scan"}, asked="who's scoring 20+ tonight") == [], lookalike


@pytest.mark.parametrize("said, lookalike, name", NEW_LECTURES)
def test_m2_each_new_lecture_is_caught_and_its_lookalike_is_not(said, lookalike, name) -> None:
    found_ = refusal_problems(said, "lookup", {"players_scan"}, asked="who's scoring 20+ tonight")
    assert found_ and found_[-1].startswith("a lecture") and f'"{name}"' in found_[-1], (said, found_)
    assert refusal_problems(said, "lookup", {"players_scan"}, asked="what's the risk he misses?") == []  # they asked
    assert refusal_problems(lookalike, "lookup", {"players_scan"}, asked="who's scoring 20+ tonight") == [], lookalike


# --- M3: "how risky is this?" asks about the risk; only the preset's name doesn't ----------------------------------

@pytest.mark.parametrize("asked, said", [
    ("how risky is this parlay?", "Honestly it's close to a coin flip: 52% for both legs."),
    ("which leg is the riskiest?", "The riskiest leg is the Knicks at 55%: close to a coin flip."),
    ("is this too risky?", "It's no guarantee: 6 legs win about 1 time in 40."),
    ("is luka 20+ riskier than jokic 10+ boards?", "Luka's 82% is stronger; Jokic at 55% is close to a coin flip."),
    ("am i risking too much on this 5 leg?", "Five legs at 70% each win about 1 time in 6: there's no guarantee."),
    ("is it risky?", "It's no guarantee: 52%."),
])
def test_m3_risk_questions_still_ask_about_the_risk(asked, said) -> None:
    assert refusal_problems(said, "analyse", {"ticket_analyse"}, asked=asked) == [], asked


@pytest.mark.parametrize("asked", ["build me a risky 6 leg parlay", "build me a Risky 6 leg parlay",
                                   "give me something risky tonight"])
def test_m3_the_risky_preset_still_gets_the_lecture_check(asked) -> None:
    found_ = refusal_problems("Remember nothing is guaranteed: six legs is a long shot.", "build", {"ticket_build"},
                              asked=asked)
    assert found_ and '"nothing is guaranteed"' in found_[0], asked


# --- M4: an honest "can't" about a player's past or futures stands ---------------------------------------------------

@pytest.mark.parametrize("player, message, honest", [
    ("LeBron James", "what were lebron's odds to win mvp in 2012",
     "I can't see MVP futures odds: the boards carry tonight's games and props only."),
    ("Luka Dončić", "did luka go over his points line last night",
     "I can't see last night's prop results: the boards carry tonight's lines."),
])
def test_m4_an_honest_can_t_about_a_named_player_s_past_or_futures_stands(world, player, message, honest) -> None:
    # flows/probe_flows_test.py::test_f2 and test_f2b: both ended "I couldn't answer that reliably just now"
    out, claude = run(world, routed(leagues=[NBA], players=[player]), [text(honest)], message)
    assert out["kind"] == "answer" and out["text"].startswith(honest), out
    assert claude.chat_calls == 1 and not claude.corrections()


def test_m4_player_props_answer_tonight_s_lines_not_the_past_or_futures() -> None:
    luka = Route(intent="lookup", players=["Luka Dončić"])
    for asked in ("did luka go over his points line last night", "luka's odds to win mvp", "luka over 29.5 in 2012",
                  "luka's season-long points prop", "was his line 29.5 yesterday?"):
        assert answering_tools(luka, asked) == set(), asked
    for asked in ("is luka going over his points line", "luka over 29.5 points?", "what are luka's odds tonight",
                  "luka 30+ tonight?"):
        assert answering_tools(luka, asked) == {"player_get_props"}, asked


@pytest.mark.parametrize("reply, honest", [
    ("I can't see MVP futures odds: the boards carry tonight's games and props only.", True),
    ("I can't rank tonight's players yet: the board is still reading tonight's games.", True),
    ("I can't see last night's results. The boards only cover tonight's games.", True),
    ("I can't do that.", False),  # no reason given
    ("I can't give you the odds.", False),
    ("I can't check that right now. Try again later.", False),
    ("I can't pull that up, so name one and I'll look.", False),  # the old redirect
    ("Unfortunately, I'm not able to do that.", False),
    ("Sorry, I'm unable to list players. Ask me about one player.", False),  # the old redirect
    ("I can only look up one player at a time: name one.", False),  # a tool's limits
    (OLD_REFUSAL, False),  # four sentences
])
def test_m4_the_retry_s_honest_one_line_can_t(reply, honest) -> None:
    assert explain_module.honest_cant(reply) is honest, reply


# --- M5 = F6: "games" is the games still to come ---------------------------------------------------------------------

BASE = {"ask": "45+ points", "stat": "pts", "stat_label": "points", "threshold": 45, "side": "over", "day": "today",
        "leagues": [NBA], "players": [], "close": [], "more": 0, "mode": "scan", "names": [], "missing_names": []}


def test_m5_the_started_note_reads_games_as_the_games_still_to_come() -> None:
    # flows/probe_started_test.py: 6 games under way and 4 still to come, nobody reaches 45+
    some = scan_card(dict(BASE, games=4, games_started=6))
    for words in (scan_template(some), bot_chat.card_text(some)):
        assert "Every game" not in words and "nobody is left" not in words
        assert "6 games tonight have already started: their players aren" in words
    assert scan_template(scan_card(dict(BASE, games=4, games_started=1))).endswith(
        "1 game tonight has already started: their players aren't listed.")
    # Every game under way (games 0): said, and the hint says so instead of "another stat, number or day"
    every = scan_card(dict(BASE, games=0, games_started=10))
    assert scan_template(every) == "Every game tonight has started, so nobody is left to rank."
    assert "<i>Every game tonight has started, so nobody is left to rank.</i>" in bot_chat.card_text(every)
    facts = scan_facts(dict(BASE, games=0, games_started=10))
    assert facts["games"] == 0 and facts["games_started"] == 10 and "Every game that day has already started" in \
        facts["hint"] and "another stat" not in facts["hint"]
    assert not any("started" in note for note in scan_notes(dict(BASE, games=8, games_started=0)))


def test_f6_the_started_note_through_the_real_lab_after_tip_off(tmp_path, monkeypatch) -> None:
    # review-data py/test_probe_missing_started.py::test_probe_after_last_tip: the real NHL board, Lab, card and template
    from telegram_betting_bot_simple.data.player_board import parse_events
    from telegram_betting_bot_simple.engine import scan as engine
    from tests.test_player_scan import Web, fixture_routes, load
    from tests.test_review2_data import Clock, _lab, nhl_board
    Web(monkeypatch, fixture_routes)
    clock = Clock()
    board = nhl_board(clock)
    try:
        assert board.wait(engine.NHL, "today", 30)
        lab = _lab(tmp_path, board)
        starts = sorted(event["start"] for event in parse_events(load("nhl_scoreboard_20261028")["events"]))
        clock.now = starts[1] + 60  # two of the four games under way
        out = lab.player_scan("web:review2-0001", engine.NHL, "sog", 9)  # a number nobody reaches
        assert out["games"] == 2 and out["games_started"] == 2
        words = scan_template(scan_card(out))
        assert "2 games tonight have already started: their players aren't listed." in words and "Every game" not in words
        clock.now = starts[-1] + 60  # all four under way
        out = lab.player_scan("web:review2-0001", engine.NHL, "sog", 9)
        assert out["games"] == 0 and out["games_started"] == 4
        assert scan_template(scan_card(out)) == "Every game tonight has started, so nobody is left to rank."
    finally:
        board.close()


# --- M6: an explicit tonight or tomorrow wins over a weekday in passing ----------------------------------------------

def _assistant() -> Assistant:
    a = Assistant.__new__(Assistant)
    a.tz, a.clock = NY, lambda: SATURDAY
    return a


@pytest.mark.parametrize("message, router_day, want", [
    ("20+ points tomorrow or wednesday", "tomorrow", "tomorrow"),
    ("who is scoring 20+ tonight, not sunday", "today", "today"),
    ("PRA overs tonight. sunday was brutal", "today", "today"),
    ("ks tonight, the sunday slate later", "today", "today"),
    ("who's scoring 20+, sunday was brutal", "today", "today"),
    ("anytime td scorers but not sunday", "today", "today"),
    # A weekday that names the ask's day still maps, on a Saturday
    ("who should i bet on to score a touchdown sunday", "today", "tomorrow"),
])
def test_m6_a_weekday_maps_only_when_it_names_the_ask_s_day(message, router_day, want) -> None:
    route = Route(intent="lookup", player_scan={"stat": "pts", "threshold": 20, "side": "over", "day": router_day})
    _assistant()._weekday(route, message)
    assert route.player_scan["day"] == want and "wide" not in route.player_scan, (message, route.player_scan)


# --- M7: leaving a player out keeps the list ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", ["20+ points tonight besides luka", "PRA overs tonight, skip luka",
                                     "any 30 point games tonight other than luka?", "20+ points excluding Dončić",
                                     "PRA overs without luka"])
def test_m7_a_list_that_leaves_the_named_player_out_stays_a_list(message) -> None:
    route = Route(intent="lookup", players=["Luka Dončić"],
                  player_scan={"stat": "pts", "threshold": 20, "side": "over", "day": "today"})
    Assistant._one_player(route, message)
    assert "player" not in route.player_scan, (message, route.player_scan)


@pytest.mark.parametrize("message", ["will luka score 30 without kyrie", "is luka going over his points line",
                                     "will steph hit 5 threes"])
def test_m7_a_question_about_the_named_player_is_still_his_own_row(message) -> None:
    name = "Stephen Curry" if "steph" in message else "Luka Dončić"
    route = Route(intent="lookup", players=[name],
                  player_scan={"stat": "pts", "threshold": 30, "side": "over", "day": "today"})
    Assistant._one_player(route, message)
    assert route.player_scan.get("player") == name, (message, route.player_scan)


# --- M8: the bot's best-edges "Also:" rows name their own bet; hot and cold ones their trend ----------------------

def _edge_row(rank: int, name: str, stat: str, label: str, side: str, line: float, edge: float,
              chance: float = 0.5) -> Dict[str, Any]:
    return {"rank": rank, "athlete_id": str(9000 + rank), "name": name, "team": "MIN", "opponent": "CHA",
            "league": NBA, "projection": 35.1, "chance": chance, "kind": "count", "stat": stat, "stat_label": label,
            "side": side, "threshold": int(line) + 1, "edge": edge,
            "line": {"kind": "over_under", "line": line, "over": {"decimal": 2.6}, "under": {"decimal": 2.6},
                     "book": "DraftKings"}, "hits": {"l10": [6, 10], "season": [3, 4]}}


def test_m8_the_bot_s_best_edges_also_rows_keep_stat_side_and_line() -> None:
    rows = [_edge_row(i, f"P{i}", "pts", "points", "over", 20.5, 0.30 - i / 100) for i in range(1, 6)]
    rows.append(_edge_row(6, "Bryce Nash", "pra", "points + rebounds + assists", "under", 33.5, 0.252, 0.48))
    card = scan_card({"mode": "edges", "players": rows, "close": [], "leagues": [NBA], "day": "today", "sort": "edge",
                      "side": None, "more": 0})
    also = [line for line in bot_chat.card_text(card).split("\n") if line.startswith("Also:")][0]
    assert also == "Also: Bryce Nash points + rebounds + assists under 33.5 (35.1, 48%, edge +25.2%)", also
    assert "Bryce Nash (MIN v CHA) points + rebounds + assists under 33.5: 35.1 projected, 48% to stay under" in \
        scan_template(card)


def test_m8_the_bot_s_hot_also_rows_carry_their_trend() -> None:
    rows = [dict(row(i, f"Player {i}", "BOS", "NYK", 25.0 - i, 0.6 - i / 100, [6, 10], [9, 16], threshold=24 - i),
                 trend={"l5": 30.0 - i, "season": 22.0, "diff": 8.0 - i, "vs": "season"}) for i in range(1, 7)]
    card = scan_card(found(rows, sort="hot", threshold=None))
    also = [line for line in bot_chat.card_text(card).split("\n") if line.startswith("Also:")][0]
    assert also == "Also: Player 6 (L5 24 vs 22 season, 19.0, 54% for 18+)", also


# --- m1: one "more", the tool's own, worded by mode; the bot's finder button -------------------------------------

def test_minor_m1_the_chat_and_the_bot_say_the_card_s_own_more_worded_by_mode() -> None:
    # "who's hot" said "231 more qualify" in the chat and the bot (2 of the card's 10 rows counted in again) while the
    # app's card said "229 more players are ranked" for the same result: one number, the tool's
    hot = [dict(r, trend={"l5": 25.0, "season": 20.0, "diff": 5.0, "vs": "season"}) for r in _qualifiers(10)]
    card = scan_card(found(hot, sort="hot", more=229))
    template, bot = scan_template(card), bot_chat.card_text(card)
    assert "229 more players are ranked: open the Player finder." in template
    assert "📱 229 more players are ranked: open the Player finder." in bot
    assert "231" not in template + bot and "qualify" not in template + bot
    assert "; Player 10 (BOS v NYK)" in template and "; Player 10 (" in bot  # every row returned is named
    edges = [dict(r, stat="pts", stat_label="points", side="over", line=dict(DK, line=19.5), edge=0.05)
             for r in _qualifiers(6)]
    edge_card = scan_card(found(edges, mode="edges", stat=None, threshold=None, more=14, sort="edge"))
    assert "14 more posted lines have an edge: open the Player finder." in scan_template(edge_card)
    assert "📱 14 more posted lines have an edge: open the Player finder." in bot_chat.card_text(edge_card)
    plain = scan_card(found(_qualifiers(10), more=1))
    assert "1 more qualifies: open the Player finder." in scan_template(plain)
    assert "more players are ranked" in scan_facts(found(hot, sort="hot", more=229))["hint"]
    assert "more posted lines have an edge" in scan_facts(found(edges, mode="edges", more=14))["hint"]


def test_minor_m1_the_bot_s_more_line_gets_a_button_that_opens_the_player_finder_on_the_same_scan() -> None:
    config = SimpleNamespace(mini_app_url="https://app.example")
    card = scan_card(found(_qualifiers(10), more=4, names=[]))
    result = {"thread_id": 7, "message_idx": 1, "reply": {"cards": [card], "chips": []}}
    buttons = [button for line in bot_chat.reply_markup(result, config, True).inline_keyboard for button in line]
    finder = [button for button in buttons if button.text == bot_chat.FINDER_BUTTON]
    assert len(finder) == 1
    assert finder[0].web_app.url == ("https://app.example/#/players?league=basketball_nba&stat=pts&at=20&day=today")
    # The same way as the bot's other Mini App button, which still follows it
    assert buttons[-1].text == "📱 Open in app" and buttons[-1].web_app.url == "https://app.example/#ask"
    hot = scan_card(found(_qualifiers(10), sort="hot", threshold=None, more=229, names=["Ja Morant"], day="tomorrow"))
    assert bot_chat.finder_hash(hot) == ("#/players?league=basketball_nba&stat=pts&at=line&day=tomorrow&sort=hot"
                                         "&player=Ja+Morant")
    # No "more" line: no button; outside a private chat Telegram refuses Mini App buttons
    none = {"thread_id": 7, "message_idx": 1, "reply": {"cards": [scan_card(found(_qualifiers(3)))], "chips": []}}
    assert not any(button.text == bot_chat.FINDER_BUTTON for line in bot_chat.reply_markup(none, config, True)
                   .inline_keyboard for button in line)
    assert bot_chat.reply_markup(result, config, False) is None or not any(
        button.text == bot_chat.FINDER_BUTTON for line in bot_chat.reply_markup(result, config, False).inline_keyboard
        for button in line)


# --- m2: best-edges yes/no rows and hot rows' chances --------------------------------------------------------------

def test_minor_m2_a_yes_no_best_edge_reads_yes_and_a_hot_row_says_what_its_chance_is_for() -> None:
    dd = dict(row(1, "Evan Quinn", "CHA", "MIN", 1.6, 0.65, [4, 10], [1, 4], threshold=2,
                  line={"kind": "yes_no", "yes": {"decimal": 1.8}, "book": "DraftKings"}, edge=0.17),
              stat="dd", stat_label="double-double", side="over", kind="yes_no")
    card = scan_card(found([dd], mode="edges", stat=None, threshold=None, sort="edge"))
    template, bot = scan_template(card), bot_chat.card_text(card)
    assert "Evan Quinn (CHA v MIN) double-double (yes): 65%" in template and "double-double over" not in template
    assert "   double-double (yes) · 65%" in bot and "double-double over" not in bot
    hot = [dict(row(1, "Jalen Duren", "DET", "WSH", 12.1, 0.55, [6, 10], [9, 16], threshold=25),
                trend={"l5": 14.2, "season": 11.0, "diff": 3.2, "vs": "season"})]
    hot_card = scan_card(found(hot, sort="hot", threshold=None))
    assert "L5 14.2 vs 11 season · 12.1 proj · 55% for 25+" in bot_chat.card_text(hot_card)
    assert "55% for 25+" in scan_template(hot_card)


# --- m3: the router's clock says whose it is --------------------------------------------------------------------------

def test_minor_m3_the_now_line_says_the_app_s_clock_unless_the_person_s_zone_is_stored() -> None:
    a = _assistant()
    assert a._now_line({}) == "Now: Saturday 31 October 2026, 2:05 PM EDT (the app's clock: America/New_York)"
    assert a._now_line({"settings": {"timezone": "Not/AZone"}}).endswith("(the app's clock: America/New_York)")
    assert a._now_line({"settings": {"timezone": "Europe/London"}}) == \
        "Now: Saturday 31 October 2026, 6:05 PM GMT (the person's zone: Europe/London)"
    assert "the app's clock when their zone isn't known" in " ".join(assistant_module.ROUTER_PROMPT.split())


# --- m4: "who's hot" with no league lets the Lab pick the league with games, and its stat --------------------------

def test_minor_m4_whos_hot_on_an_nfl_only_day_finds_the_nfl(world) -> None:
    # flows/probe_hot_nfl_test.py: the tool sent points, which no NFL board has
    hot = dict(row(1, "Owen Fields", "DET", "WSH", 80.0, 0.6, [6, 10], [9, 16], model="form", threshold=70),
               trend={"l5": 95.0, "season": 70.0, "diff": 25.0, "vs": "season"})
    world.lab.player_board = FakeBoards({(NFL, "today"): FakeBoard(NFL, "today", {"rush_rec_yds": {"players": [hot]}},
                                                                    games_total=4)})
    done = Toolbox(world.lab, UID, NY).run("players_scan", {"sort": "hot"})
    assert done.ok and done.content["leagues"] == [NFL] and done.content["stat"] == "rush_rec_yds", done.content
    assert scan_template(done.cards[0]).startswith("Hottest on rushing + receiving yards (last 5 against the season) "
                                                   "tonight:\n1. Owen Fields (DET v WSH) L5 95 vs 70 season")
    # With a league named it still reads that league's own stat
    lab_scan = ContractScan(found(_qualifiers(2)))
    world.lab.player_scan = lab_scan
    Toolbox(world.lab, UID, NY).run("players_scan", {"sort": "cold", "league": NBA})
    Toolbox(world.lab, UID, NY).run("players_scan", {"sort": "hot"})
    assert [(c["league"], c["stat"], c["sort"]) for c in lab_scan.calls] == [(NBA, "pts", "cold"), (None, None, "hot")]


# --- m5: the router-down reading of other senses of "lock", one player's props, and the weather -------------------

@pytest.mark.parametrize("message", ["will the hammers beat chelsea today", "is the over 220.5 a lock",
                                     "are the celtics a lock", "can i lock in this price",
                                     "lock screen keeps logging me out", "best props for luka",
                                     "best overs for the lakers game", "who plays in the cold tonight"])
def test_minor_m5_router_down_other_senses_are_not_picks_edges_or_form(message) -> None:
    route = Route()
    scan_fallback(route, message)
    assert route.top_picks is False and (route.player_scan or {}).get("sort") not in ("edge", "cold"), \
        (message, route.top_picks, route.player_scan)


@pytest.mark.parametrize("message, wanted", [
    ("give me 3 locks", "picks"), ("lock of the night", "picks"), ("what's your hammer tonight", "picks"),
    ("any nukes on the slate", "picks"), ("who's the lock tonight", "picks"), ("locks?", "picks"),
    ("best player props tonight", "edge"), ("any good overs tonight?", "edge"), ("best props for tonight", "edge"),
    ("who's cold right now, who should i fade", "cold"),
])
def test_minor_m5_router_down_asks_for_picks_edges_and_form_still_route(message, wanted) -> None:
    route = Route()
    scan_fallback(route, message)
    if wanted == "picks":
        assert route.top_picks is True and route.player_scan is None, message
    else:
        assert route.player_scan is not None and route.player_scan["sort"] == wanted, (message, route.player_scan)


# --- F3 (contract 1): the names the scan couldn't rank, in the same words everywhere ------------------------------

NAMED = {"ask": "20+ points", "stat": "pts", "stat_label": "points", "threshold": 20, "side": "over", "day": "today",
         "leagues": [NBA], "players": [], "close": [], "more": 0, "mode": "scan", "games": 6, "games_started": 0}


def test_f3_missing_ambiguous_and_started_names_read_the_same_in_the_template_bot_and_hint() -> None:
    result = dict(NAMED, names=["Ja Morant", "Jalen", "Leon Draisaitl"], missing_names=["Ja Morant"],
                  ambiguous_names=[{"name": "Jalen", "candidates": ["Jalen Brunson", "Jalen Green", "Jalen Williams"]}],
                  started_names=[{"name": "Leon Draisaitl", "player": "Leon Draisaitl", "game": "CGY @ EDM"}])
    card = scan_card(result)
    assert card["ambiguous_names"] == result["ambiguous_names"] and card["started_names"] == result["started_names"]
    wanted = ["I couldn't find Ja Morant in tonight's games.",
              "More than one player matches Jalen: Jalen Brunson, Jalen Green or Jalen Williams. Which one?",
              "Leon Draisaitl's game (CGY @ EDM) has already started, so it isn't in the scan."]
    assert scan_template(card) == "\n".join(wanted)  # nothing else: no "No player is projected" above it
    bot = bot_chat.card_text(card)
    for line in wanted:
        assert line.replace("'", "&#x27;") in bot, line
    facts = scan_facts(result)
    assert facts["ambiguous_names"] == result["ambiguous_names"] and facts["started_names"] == result["started_names"]
    assert "couldn't rank the names" in facts["hint"] and "ask which one" in facts["hint"]
    assert not re.search(r"\d", facts["hint"])
    assert not re.search(r"\b(?:he|his|him|she|her|they|their)\b", " ".join(wanted), re.I)  # no pronouns for players
    tomorrow = scan_card(dict(result, day="tomorrow"))
    assert "I couldn't find Ja Morant in tomorrow's games." in scan_template(tomorrow)
    two = scan_notes({"ambiguous_names": [{"name": "Curry", "candidates": ["Seth Curry", "Stephen Curry"]}]})
    assert two == ["More than one player matches Curry: Seth Curry or Stephen Curry. Which one?"]


def test_f3_the_tool_carries_the_lab_s_lists_under_the_same_keys(world) -> None:
    lab_scan = ContractScan(found([], names=["Jalen"], missing_names=[],
                                  ambiguous_names=[{"name": "Jalen", "candidates": ["Jalen Brunson", "Jalen Green"]}],
                                  started_names=[]))
    world.lab.player_scan = lab_scan
    done = Toolbox(world.lab, UID, NY).run("players_scan", {"stat": "pts", "threshold": 20, "player": "Jalen"})
    card = done.cards[0]
    assert card["ambiguous_names"] == [{"name": "Jalen", "candidates": ["Jalen Brunson", "Jalen Green"]}]
    assert card["started_names"] == [] and card["missing_names"] == []
    assert done.content["ambiguous_names"] == card["ambiguous_names"] and "started_names" not in done.content
    assert scan_template(card) == "More than one player matches Jalen: Jalen Brunson or Jalen Green. Which one?"


def test_f3_a_named_player_whose_game_has_started_through_the_real_lab(tmp_path, monkeypatch) -> None:
    # review-data py/test_probe_missing_started.py::test_probe: Draisaitl ten minutes into his game read "Leon
    # Draisaitl isn't in tonight's games."
    from telegram_betting_bot_simple.data.player_board import parse_events
    from telegram_betting_bot_simple.engine import scan as engine
    from tests.test_player_scan import Web, fixture_routes, load
    from tests.test_review2_data import EDM_GAME, Clock, _lab, nhl_board
    Web(monkeypatch, fixture_routes)
    clock = Clock()
    board = nhl_board(clock)
    try:
        assert board.wait(engine.NHL, "today", 30)
        lab = _lab(tmp_path, board)
        start = next(e for e in parse_events(load("nhl_scoreboard_20261028")["events"]) if e["event_id"] == EDM_GAME)
        clock.now = start["start"] + 600
        out = lab.player_scan("web:review2-0001", engine.NHL, "sog", 3, names=["Leon Draisaitl"])
        [entry] = out["started_names"]
        words = scan_template(scan_card(out))
        assert f"Leon Draisaitl's game ({entry['game']}) has already started, so it isn't in the scan." in words
        assert "couldn't find" not in words and "isn't in tonight's games" not in words
    finally:
        board.close()


# --- F2 (contract 2): the trend against this season or last, in the same words everywhere -----------------------

@pytest.mark.parametrize("vs, words", [("season", "L5 28.4 vs 24.1 season"), ("last_season", "L5 28.4 vs 24.1 last season")])
def test_f2_the_trend_reads_against_this_season_or_last(vs, words) -> None:
    hot = [dict(row(1, "Jalen Brunson", "NYK", "BOS", 27.0, 0.62, [7, 10], [3, 4], threshold=25),
                trend={"l5": 28.4, "season": 24.1, "diff": 4.3, "vs": vs})]
    result = found(hot, sort="hot", threshold=None)
    card = scan_card(result)
    assert scan_template(card).split("\n")[1].startswith(f"1. Jalen Brunson (NYK v BOS) {words}, 27.0 projected")
    assert f"   {words} · 27.0 proj" in bot_chat.card_text(card)
    assert scan_facts(result)["players"][0]["trend"] == {"l5": 28.4, "season": 24.1, "diff": 4.3, "vs": vs}
    if vs == "season":
        assert "last season" not in scan_template(card)


# --- the tool budget ------------------------------------------------------------------------------------------------

def test_the_tools_json_stays_inside_its_budget() -> None:
    import json
    assert len(json.dumps(TOOLS)) < 21_000


# --- the router's 400s (optional in the spec): logged once at ERROR with the API's message ------------------------

def test_router_400s_are_logged_once_at_error_and_nothing_changes_for_the_person(world, caplog) -> None:
    import logging

    import anthropic
    import httpx2

    refused = anthropic.BadRequestError(
        "The compiled grammar is too large, which would cause performance issues.", response=httpx2.Response(
            400, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")), body=None)

    class Refusing(Scripted):
        def _create(self, **request):
            if request["model"] == "claude-haiku-5-5":
                self.requests.append({"model": request["model"]})
                raise refused
            return super()._create(**request)

    assistant_module.ROUTER_REFUSALS_LOGGED.clear()
    with caplog.at_level(logging.ERROR):
        for _ in range(2):
            claude = Refusing(None, [text("Hi there.")])
            world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: SATURDAY)
            out = world.lab.chat(UID, {"message": "hi"})["reply"]
            assert out["kind"] == "answer" and out["text"] == "Hi there."  # the turn runs without the router
    errors = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert errors == ["The router's requests are refused, so turns run without it: The compiled grammar is too "
                      "large, which would cause performance issues."], errors


# --- V1 (the final check, also on c5deb8f): a calm turn's retry invites no tool and runs none -------------------
# verify/rgrun/dbg_wind2_test.py: a wind-down decline with a banned word in it ("No locks tonight. Take a break and come
# back fresh.") got the numbers check's retry, which said "call a tool if you need one you don't have": after loss talk a
# model that obeyed brought the picks card, and one that repeated itself got "I couldn't answer that reliably"

BANNED_WIND_DOWN = ["No locks tonight. Take a break and come back fresh.",
                    "Nothing is a sure thing, so let's call it a night."]


def _calm_correction(claude) -> str:
    [correction] = [m["content"] for m in claude.chat()[1]["messages"] if m["role"] == "user"
                    and isinstance(m["content"], str) and "Your reply can't be shown" in m["content"]]
    return correction


@pytest.mark.parametrize("reply", BANNED_WIND_DOWN)
def test_v1_a_calm_retry_has_no_tools_and_a_clean_one_stands(world, reply) -> None:
    clean = "Take a break tonight and come back fresh."
    out, claude, _ = chat_in(world, None, routed(top_picks=True, leagues=[NBA]), [text(reply), text(clean)],
                             "give me 3 locks")
    retry = claude.chat()[1]
    assert retry["tool_choice"] == {"type": "none"}, retry.get("tool_choice")
    correction = _calm_correction(claude)
    assert correction.startswith("Your reply can't be shown because it has banned wording: ")
    assert correction.endswith("Write it again with only numbers from this turn's tool results and none of the banned "
                               "wording. Call no tools and suggest no bets.")
    assert "call a tool" not in correction.lower()
    assert out["kind"] == "answer" and out["text"].startswith(clean) and not out["cards"], out
    assert claude.chat_calls == 2


@pytest.mark.parametrize("reply", BANNED_WIND_DOWN)
def test_v1_a_calm_retry_that_repeats_itself_ends_with_code_s_wind_down_line(world, reply) -> None:
    out, claude, _ = chat_in(world, None, routed(top_picks=True, leagues=[NBA]), [text(reply), text(reply)],
                             "give me 3 locks")
    assert "couldn't answer" not in out["text"] and "different way" not in out["text"] and not out["cards"], out
    assert out["text"] == assistant_module.WIND_DOWN_LINE
    assert out["chips"] == list(assistant_module.SUPPORT_CHIPS)
    assert claude.chat_calls == 2 and claude.chat()[1]["tool_choice"] == {"type": "none"}
    assert assistant_module.support_reply(assistant_module.WIND_DOWN_LINE)  # the line is a wind-down reply itself


@pytest.mark.parametrize("reply", BANNED_WIND_DOWN)
def test_v1_after_loss_talk_a_model_that_wants_picks_on_the_retry_can_t_reach_them(world, reply) -> None:
    _, _, thread = chat_in(world, None, routed(intent="chat"), [text("Sorry to hear that.")],
                           "rough night, lost 3 parlays in a row")
    out, claude, _ = chat_in(world, thread, routed(top_picks=True, leagues=[NBA]),
                             [text(reply), call("brain_top_picks", {"leagues": [NBA], "n": 3}),
                              text("Here are the strongest picks.")], "give me 3 locks")
    assert claude.chat()[1]["tool_choice"] == {"type": "none"} and "call a tool" not in _calm_correction(claude).lower()
    assert not [card for card in out["cards"] if card["type"] in ("picks", "player_scan", "ticket")], out["cards"]
    assert out["text"] == assistant_module.WIND_DOWN_LINE and claude.chat_calls == 2, out
    assert not claude.tool_calls()  # the tool the retry asked for never ran


def test_v1_disclosure_and_helpline_stay_as_they_are_on_a_calm_turn(world) -> None:
    # A new thread still gets the disclosure; with the router down a soft sign still adds the helpline under code's line
    claude = Scripted(None, [text(BANNED_WIND_DOWN[0]), text(BANNED_WIND_DOWN[0])])
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    result = world.lab.chat(UID, {"message": "double or nothing, give me a lock"})
    assert result["disclosure"] == assistant_module.DISCLOSURE
    reply = result["reply"]
    assert reply["text"].endswith("\n\n" + assistant_module.HELPLINE) and not reply["cards"], reply
    assert "couldn't answer" not in reply["text"]
    assert reply["text"] == assistant_module.WIND_DOWN_LINE + "\n\n" + assistant_module.HELPLINE


# --- V4 (the final check): the bot's finder names follow the finder's own cleanNames, character for character ----

@pytest.mark.parametrize("value, names", [
    ("Marvin Bagley 3rd", ["Marvin Bagley 3rd"]),
    ("Michael Porter, Jr.", ["Michael Porter Jr."]),
    ("Marcus Sutton (MIA)", ["Marcus Sutton (MIA)"]),
    ("Robert Williams, III, Kelly Oubre Jr", ["Robert Williams III", "Kelly Oubre Jr"]),
    ('"><svg/onload=alert(1)>', ["svg onload alert(1)"]),
    ('<img src=x onerror="window.__xss=1">', ["img src x onerror window. xss 1"]),
    ("Jr., Luka", ["Jr.", "Luka"]),
    ("Luka Dončić, luka dončić", ["Luka Dončić"]),
])
def test_v4_finder_names_follow_the_finder_s_rule(value, names) -> None:
    assert bot_chat._finder_names([value]) == names
    assert bot_chat._finder_names(names) == names  # the rule is idempotent


def test_v4_finder_names_join_across_the_list_cut_to_sixty_and_keep_five() -> None:
    assert bot_chat._finder_names(["Michael Porter", "Jr."]) == ["Michael Porter Jr."]
    long = bot_chat._finder_names(["x" * 58, "Jr."])
    assert long == ["x" * 58 + " J"] and len(long[0]) == 60  # never past the route's 60
    assert bot_chat._finder_names(["y" * 59, "Jr."]) == ["y" * 59]  # cut at the space, then trimmed (the finder's cutText)
    assert bot_chat._finder_names(["Tab\there", "under_score", "Ünïcödé Ñame", "A", "B", "C"]) == \
        ["Tab here", "under score", "Ünïcödé Ñame", "A", "B"]
    card = scan_card(found(_qualifiers(10), more=4, names=["Marvin Bagley 3rd", "Michael Porter, Jr."]))
    assert bot_chat.finder_hash(card).endswith("&player=Marvin+Bagley+3rd&player=Michael+Porter+Jr.")
