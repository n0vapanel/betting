"""The second persona review of the player scan (scratchpad/scan/review2-persona: FINDINGS.md, probes/*.py and their
logs, plain_reading.log, template_probe.log), its probes kept as regression tests. Each test failed on c5deb8f (the
findings' commit) and holds the fix:

- M2: an honest "I can't" where no tool answers (live odds, a 2012 box score, a team's alt-line ladder) is no longer
  turned into "I couldn't answer that reliably": the guard is route-aware, and a tool that came back empty leaves an
  honest one-line answer; the retry allows one, with no invented number.
- M3: the guard's paraphrases (refusals and lectures), each with a positive and a negative.
- m1-m11: the scan card first, top picks run by code, an empty scan's card and tomorrow, "Also:" and "N more qualify",
  yes/no rows, the edge on posted lines, the router's date and weekday names, a player's own row, quoted spans only,
  "risky" isn't a risk question, and the router-down reading.
- The plain-language gaps: "who's hot" and "who's cold" (sort by the last 5 against the season), "best player props
  tonight" and "any good overs" (the best edges on every posted line), "will steph hit 5 threes" (his own row).
- RD1 (the data review): the players_scan hint has no example numbers, and no hint is a source of numbers.

The world is tests/test_player_scan_wiring.py's (the real Lab, Store and assistant over canned scan rows); a scripted
fake Claude plays the router and the analyst. A recording stand-in for Lab.player_scan's new parameters (names, sort,
edges mode, contract A of SPEC_P0) checks what the tool asks for. No network."""

from __future__ import annotations

import datetime
import json
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pytest
import requests

from telegram_betting_bot_simple.ai import assistant as assistant_module
from telegram_betting_bot_simple.ai.assistant import Assistant, Route, parse_route, scan_args, scan_fallback
from telegram_betting_bot_simple.ai.explain import (
    problems,
    refusal_problems,
    scan_template,
    source_numbers,
)
from telegram_betting_bot_simple.ai.tools import TOOLS, Toolbox, scan_card, scan_facts
from telegram_betting_bot_simple.data.odds_service import configure_books
from tests.test_builder_wiring import make_world
from tests.test_player_scan_wiring import (
    DK,
    NBA,
    NFL,
    NOW,
    NY,
    OLD_REFUSAL,
    SCREENSHOT,
    UID,
    FakeBoard,
    FakeBoards,
    nba_boards,
    reply,
    results_of,
    routed,
    row,
    say,
    scan,
    tool_use,
    understood,
)

# --- the wiring tests' world (tests/test_player_scan_wiring.py's fixtures, made the same way) ----------------------

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


# Saturday 31 October 2026, 2:05 PM in New York: "touchdown sunday" is tomorrow
SATURDAY = datetime.datetime(2026, 10, 31, 18, 5, tzinfo=datetime.timezone.utc).timestamp()


# --- a scripted fake Claude through the real assistant ----------------------------------------------------------------

class Scripted:
    """The router answers ``route`` (None: the router is down); the chat model plays ``script``, one step per chat
    request (the last step repeats). A step takes the request's tool results and the request."""

    def __init__(self, route: Optional[Dict[str, Any]], script: List[Callable[..., Any]]) -> None:
        self.route, self.script, self.requests, self.chat_calls = route, script, [], 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(json.loads(json.dumps(request, default=lambda o: getattr(o, "__dict__", str(o)))))
        if request["model"] == "claude-haiku-5-5":
            if self.route is None:
                return reply([], stop="refusal", model="claude-haiku-5-5")
            return reply([say(json.dumps(self.route))], model="claude-haiku-5-5")
        step = self.script[min(self.chat_calls, len(self.script) - 1)]
        self.chat_calls += 1
        return step(results_of(self.requests[-1]), self.requests[-1])

    def chat(self) -> List[Dict[str, Any]]:
        return [r for r in self.requests if r["model"] != "claude-haiku-5-5"]

    def router(self) -> List[Dict[str, Any]]:
        return [r for r in self.requests if r["model"] == "claude-haiku-5-5"]

    def tool_calls(self) -> List[Dict[str, Any]]:
        seen, out = set(), []
        for request in self.chat():
            for message in request["messages"]:
                for block in message["content"] if isinstance(message["content"], list) else []:
                    if isinstance(block, dict) and block.get("type") == "tool_use" and block["id"] not in seen:
                        seen.add(block["id"])
                        out.append(block)
        return out

    def corrections(self) -> List[str]:
        """The retry's corrections sent to the chat model (each once, though later requests carry it too)."""
        return list(dict.fromkeys(m["content"] for r in self.chat() for m in r["messages"] if m["role"] == "user"
                                  and isinstance(m["content"], str) and "Your reply can't be shown" in m["content"]))


def text(words: str) -> Callable[..., Any]:
    return lambda results, request: reply([say(words)])


def call(name: str, args: Dict[str, Any], n: int = 1) -> Callable[..., Any]:
    return lambda results, request: tool_use(name, args, n)


def asked_scan(n: int = 1) -> Callable[..., Any]:
    """Call players_scan with exactly what code understood (the context's player_scan)."""
    def step(results, request):
        return tool_use("players_scan", scan_args(SimpleNamespace(player_scan=understood(request)["player_scan"])), n)
    return step


def run(world, route: Optional[Dict[str, Any]], script: List[Callable[..., Any]], message: str,
        clock: float = NOW) -> Any:
    claude = Scripted(route, script)
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: clock)
    out = world.lab.chat(UID, {"message": message})["reply"]
    return out, claude


class ContractScan:
    """Lab.player_scan with contract A's parameters (names, sort; stat None with sort edge is edges mode), as a
    recording stand-in: each call's arguments, and a canned result per call (the last one repeats)."""

    def __init__(self, *results: Dict[str, Any]) -> None:
        self.results, self.calls = list(results), []

    def __call__(self, uid, league=None, stat="pts", threshold=None, side="over", day="today", n=10, *, names=None,
                 sort=None):
        self.calls.append({"league": league, "stat": stat, "threshold": threshold, "side": side, "day": day, "n": n,
                           "names": names, "sort": sort})
        return json.loads(json.dumps(self.results[min(len(self.calls), len(self.results)) - 1]))


def found(players: List[Dict[str, Any]], close: Optional[List[Dict[str, Any]]] = None, **fields: Any) -> Dict[str, Any]:
    """Lab.player_scan's result for canned rows (each row with its league and addable, as the Lab adds them)."""
    base = {"ask": "20+ points", "stat": "pts", "stat_label": "points", "threshold": 20, "side": "over",
            "day": "today", "leagues": [NBA], "more": 0, "games": 8, "games_started": 0, "as_of": "2026-10-26T21:45:00Z",
            "players": [dict(r, league=NBA, addable=False) for r in players],
            "close": [dict(r, league=NBA, addable=False) for r in close or []], "mode": "scan", "missing_names": []}
    base.update(fields)
    return base


# --- M3: the guard's paraphrases, each caught, each with a sentence that isn't one --------------------------------

REFUSAL_PARAPHRASES = [  # (the probe's refusal, the same words that aren't a refusal, the phrase's name)
    ("The props tool only handles one player at a time, so name a player and I'll check his line.",
     "The Celtics only handle one big man well, and Luka is 82% for 20+.", "only handles one"),
    ("I can only look up one player at a time. Who do you want me to check?",
     "I can only see one NBA game on tonight's board: Luka is 82% for 20+.", "I can only look up one"),
    ("I don't have a tool that lists every player projected for 20+. Name a player and I'll pull his line.",
     "I don't have a favourite: the numbers lean Luka at 82%.", "I don't have a tool"),
    ("There's no way for me to scan every player tonight. Give me a name and I'll check his props.",
     "No way he sits tonight: Luka is 82% for 20+.", "no way for me to"),
    ("Unfortunately that's not something I can do: I check props player by player.",
     "That's something I can check: Luka is 82% for 20+.", "not something I can do"),
    ("I'm not set up to rank every player, but name one and I'll check his line.",
     "The Knicks are not set up to switch everything, so Luka is 82% for 20+.", "I'm not set up to"),
    ("I won't be able to list every player projected for 20 points. Pick a player and I'll look.",
     "Tatum won't be able to play tonight (ankle), so Luka is 82% for 20+.", "won't be able to"),
    ("I'd need a player's name: the props lookup works for a single player.",
     "The pick-and-roll works for one reason: spacing. Luka is 82% for 20+.", "I'd need a player's name"),
    ("The props lookup works for a single player, so tell me who.",
     "I'd need a minute more for the last game's lines; Luka is 82% for 20+.", "works for one player"),
]
LECTURE_PARAPHRASES = [  # (the probe's lecture, the same words that aren't one, the phrase's name)
    ("A projection of 20 doesn't guarantee he'll get there.", "Doncic doesn't need a big night: 31.2 projected.",
     "doesn't guarantee"),
    ("Past performance doesn't guarantee future results.", "His usage doesn't change with Irving back.",
     "doesn't guarantee"),
    ("Remember a projection isn't a guarantee.", "His minutes aren't a question tonight: 35.5 projected.",
     "isn't a guarantee"),
    ("Points overs are close to 50/50.", "He's shooting 50% from three over his last 5.", "50/50"),
    ("Most of these are a coin toss.", "He tossed in 31 last night.", "coin toss"),
    ("These props are basically a toss-up.", "He tossed up a floater at the buzzer.", "toss-up"),
    ("A projection doesn't mean it'll happen.", "A rest day doesn't mean much for Luka: 31.2 projected.",
     "doesn't mean he'll"),
    ("Keep in mind the sportsbook always has the edge.", "Luka has the edge in this matchup: 82% to get 20+.",
     "has the edge"),
    ("The books have the edge on props, so most bettors lose over time.",
     "Most players lose minutes on a back-to-back; Luka is at 35.5.", "most bettors lose"),
    ("Even at 82%, he misses about 1 game in 5.", "He missed 1 game in October with a sprain.", "misses 1 in N"),
]


@pytest.mark.parametrize("said, fine, name", REFUSAL_PARAPHRASES)
def test_each_refusal_paraphrase_is_caught_and_its_lookalike_is_not(said, fine, name) -> None:
    found_ = refusal_problems(said, "lookup", True)
    assert found_ and found_[0].startswith("a refusal") and f'"{name}"' in found_[0], (said, found_)
    assert refusal_problems(said, "lookup", False) == []  # nothing to call (picks paused): saying so is right
    assert refusal_problems(fine, "lookup", True) == [], fine


@pytest.mark.parametrize("said, fine, name", LECTURE_PARAPHRASES)
def test_each_lecture_paraphrase_is_caught_and_its_lookalike_is_not(said, fine, name) -> None:
    found_ = refusal_problems(said, "lookup", True)
    assert found_ and found_[-1].startswith("a lecture") and f'"{name}"' in found_[-1], (said, found_)
    assert refusal_problems(said, "lookup", True, asked="what's the risk he misses?") == []  # they asked
    assert refusal_problems(fine, "lookup", True) == [], fine


# --- m9: only quoted words are exempt; m10: "risky" is the preset, not a risk question -----------------------------

def test_a_phrase_the_user_typed_is_exempt_only_inside_quotes_in_the_reply() -> None:
    # The user's own "I can't" no longer lets the reply refuse
    for asked, said in (("I can't decide, give me players for 20+ points tonight",
                         "I can't give you a full list; name one player and I'll check his line."),
                        ("I can't find good props, who's hitting 3+ threes", "I can't rank everyone, but tell me a player.")):
        found_ = refusal_problems(said, "lookup", True, asked=asked)
        assert found_ and '"I can\'t"' in found_[0], (asked, said)
    # Quoted, the words are theirs; asking whether it's a coin flip is asking about the risk
    assert refusal_problems("He said “I can't miss tonight”: he's 82% for 20+.", "lookup", True,
                            asked="he said “I can't miss tonight”") == []
    assert refusal_problems('Your buddy\'s right that it\'s "nothing is guaranteed", but 82% is strong.', "lookup",
                            True, asked="my buddy says 'nothing is guaranteed'") == []
    assert refusal_problems("It's not a coin flip: 82% to get there.", "lookup", True, asked="is it a coin flip?") == []


def test_risky_is_the_preset_s_name_so_it_doesn_t_turn_the_lecture_check_off() -> None:
    lecture = "Remember nothing is guaranteed: six legs is a long shot."
    found_ = refusal_problems(lecture, "build", True, asked="build me a Risky 6 leg parlay")
    assert found_ and '"nothing is guaranteed"' in found_[0]
    assert refusal_problems(lecture, "build", True, asked="what's the risk on a 6 leg parlay?") == []
    assert refusal_problems(lecture, "build", True, asked="what are the risks here?") == []
    # ... and "risky" asked as a question is still a question about the risk (fix:ai M3: this test only checked the
    # preset's name, while "how risky is this?" had stopped counting)
    for asked in ("how risky is this parlay?", "which leg is the riskiest?", "is this too risky?",
                  "is luka 20+ riskier than jokic 10+ boards?", "am i risking too much on this 5 leg?"):
        assert refusal_problems(lecture, "analyse", True, asked=asked) == [], asked


# --- M2: route-aware tools, honest answers where no tool answers -------------------------------------------------------

HONEST = [  # (the route, the message, the honest answer: no tool covers it)
    (Route(intent="lookup", leagues=[NBA]), "what are the live odds in the lakers game right now",
     "I can't see live in-game odds: the boards are pre-game prices. Ask me for tonight's likeliest winners instead."),
    (Route(intent="lookup", leagues=[NBA]), "who wins the 2027 finals",
     "Futures aren't something the app prices: it's not possible to quote a 2027 champion yet."),
    (Route(intent="lookup", leagues=[NBA], players=["LeBron James"]), "how many points did lebron score in the 2012 finals",
     "I can't look that far back: the game logs cover this season and last. Ask about his last 10 instead."),
    (Route(intent="lookup", leagues=[NBA]), "alt line ladder for the celtics tonight",
     "I can't pull alt lines for the Celtics: the boards carry the main spread and total."),
]


def test_the_guard_counts_a_refusal_only_where_a_tool_answers_the_route() -> None:
    for route, asked, said in HONEST:
        assert Assistant._guard(said, route, False, asked, []) == [], said
    # Where a tool answers, the old refusal is still caught: a scan, the likeliest winners, a build, a player's line
    for route, asked in ((Route(intent="lookup", player_scan=scan("pts", 20)), SCREENSHOT),
                         (Route(intent="lookup", top_picks=True, leagues=[NBA]), "give me 3 locks"),
                         (Route(intent="build", leagues=[NBA]), "build me a parlay"),
                         (Route(intent="lookup", players=["Luka Dončić"]), "is luka going over his points line")):
        found_ = Assistant._guard(OLD_REFUSAL, route, False, asked, [])
        assert found_ and found_[0].startswith("a refusal"), asked
    # The static map, route by route (anything else: no tool, no refusal check)
    from telegram_betting_bot_simple.ai.explain import answering_tools
    assert answering_tools(Route(intent="lookup", player_scan=scan("pts", 20))) == {"players_scan"}
    assert answering_tools(Route(intent="lookup", top_picks=True)) == {"brain_top_picks"}
    assert answering_tools(Route(intent="build")) == {"ticket_build"}
    assert answering_tools(Route(intent="analyse")) == {"ticket_analyse", "ticket_price", "pasted_slip_analyse"}
    assert answering_tools(Route(intent="lookup", players=["Luka Dončić"]), "luka over 29.5 points?") == \
        {"player_get_props"}
    assert answering_tools(Route(intent="lookup", players=["LeBron James"]), "lebron in the 2012 finals") == set()
    assert answering_tools(Route(intent="lookup", leagues=[NBA])) == set()


@pytest.mark.parametrize("route, message, said", [
    (routed(leagues=[NBA]), HONEST[0][1], HONEST[0][2]),
    (routed(leagues=[NBA], players=["LeBron James"]), HONEST[2][1], HONEST[2][2]),
    (routed(leagues=[NBA]), HONEST[3][1], HONEST[3][2]),
])
def test_an_honest_answer_where_no_tool_answers_reaches_the_person(world, route, message, said) -> None:
    out, claude = run(world, route, [text(said)], message)
    assert out["kind"] == "answer" and out["text"].startswith(said), out
    assert claude.chat_calls == 1 and not claude.corrections()  # no retry, never "I couldn't answer that reliably"


def test_after_the_scan_came_back_empty_an_honest_one_line_answer_stands(world) -> None:
    world.lab.player_board = FakeBoards({})  # no NFL game today or tomorrow
    honest = "I can't find an NFL game today or tomorrow, so there are no anytime TD scorers to rank yet."
    out, claude = run(world, routed(player_scan=scan("anytime_td", 1), leagues=[NFL]), [asked_scan(), text(honest)],
                      "anytime TD scorers")
    assert out["kind"] == "answer" and out["text"].startswith(honest), out
    assert [c["name"] for c in claude.tool_calls()] == ["players_scan"] and not claude.corrections()


def test_the_retry_allows_an_honest_one_line_can_t_with_no_invented_number(world) -> None:
    # The retry sends the honest one-line "can't" the correction allows, and it stands (fix:ai M4: this test only
    # checked the correction's words, while the retry ran the same guard and turned that "can't" into the template)
    honest = "I can't rank tonight's players yet: the board is still reading tonight's games."
    out, claude = run(world, routed(player_scan=scan("pts", 20)), [text(OLD_REFUSAL), text(honest)], SCREENSHOT)
    [correction] = claude.corrections()
    assert "Don't refuse and don't add disclaimers: call players_scan with" in correction
    assert "If the tools can't answer it after all, say so in one plain line, with no number the tools didn't give." \
        in correction
    assert out["kind"] == "answer" and out["text"].startswith(honest), out
    assert claude.chat_calls == 2 and not out["cards"]  # no code-run scan under an honest "can't"
    # Not with a number the tools didn't give, and not once the scan answered with rows: then the template answers
    invented = "I can't rank tonight's players yet: the board has only Luka, at 41% for 20+."
    out, claude = run(world, routed(player_scan=scan("pts", 20)), [text(OLD_REFUSAL), text(invented)], SCREENSHOT)
    assert out["kind"] == "template" and out["cards"][0]["type"] == "player_scan"
    out, claude = run(world, routed(player_scan=scan("pts", 20)), [text(OLD_REFUSAL), asked_scan(2), text(honest)],
                      SCREENSHOT)
    assert out["kind"] == "template" and out["text"].startswith("Projected for 20+ points tonight:\n1. ")


# --- m1: the scan card first; m2: top picks run by code; m3: an empty scan's card, and tomorrow ------------------

def test_a_scan_asked_for_leads_the_cards_and_the_template_after_a_player_lookup_and_two_refusals(world,
                                                                                                    monkeypatch) -> None:
    from telegram_betting_bot_simple.ai import tools as tools_module
    player = {"id": "3945274", "league": NBA, "name": "Luka Dončić", "team": "DAL", "props": [], "source": "x"}
    monkeypatch.setattr(tools_module.Toolbox, "_player", lambda self, data: dict(player))
    out, claude = run(world, routed(player_scan=scan("pts", 20)), [
        call("player_get_props", {"name": "Luka Doncic", "league": NBA}), text(OLD_REFUSAL),
        text("Sorry, I'm unable to list players. Ask me about one player.")], SCREENSHOT)
    assert out["kind"] == "template"
    assert out["text"].startswith("Projected for 20+ points tonight:\n1. Luka Dončić (DAL v PHX)")
    assert "Here is" not in out["text"]
    assert [card["type"] for card in out["cards"]] == ["player_scan", "player"]


def test_a_top_pick_ask_refused_twice_gets_code_s_strongest_picks(world) -> None:
    out, claude = run(world, routed(top_picks=True, leagues=[NBA]),
                      [text("I can't give you locks: nothing is guaranteed in betting.")], "give me 3 locks")
    assert out["kind"] == "template"
    assert out["text"].startswith("The most likely winners, strongest first:\n1. ")
    assert [card["type"] for card in out["cards"]] == ["picks"] and len(out["cards"][0]["picks"]) == 3
    assert "lock" not in out["text"].lower() and "couldn't" not in out["text"]


def test_an_empty_scan_still_makes_a_card_that_says_which_league_has_no_games(world) -> None:
    from telegram_betting_bot_simple.bot import chat
    world.lab.player_board = FakeBoards({})  # no NFL game today or tomorrow
    bad = "No NFL games today, but there's a 99% chance of some tomorrow."
    out, claude = run(world, routed(player_scan=scan("anytime_td", 1), leagues=[NFL]),
                      [asked_scan(), text(bad), text(bad)], "anytime TD scorers")
    assert out["kind"] == "template"
    assert out["text"] == ("No player is projected for anytime TD tonight yet (NFL: no games).\n"
                           "There are none tomorrow either.")
    [card] = out["cards"]
    assert card["type"] == "player_scan" and card["players"] == [] and card["leagues_without_games"] == ["NFL"]
    assert "No player is projected for anytime TD tonight yet (NFL: no games)." in chat.card_text(card)
    assert "Nobody qualifies yet." not in chat.card_text(card)


def test_a_league_with_no_games_today_is_scanned_for_tomorrow_and_says_so(world) -> None:
    from telegram_betting_bot_simple.bot import chat
    td = {"players": [row(1, "Owen Fields", "DET", "WSH", 0.9, 0.5878, [9, 10], [6, 7], model="form", threshold=1,
                          why=["a TD in 9 of last 10"])]}
    world.lab.player_board = FakeBoards({(NFL, "tomorrow"): FakeBoard(NFL, "tomorrow", {"anytime_td": td})})
    done = Toolbox(world.lab, UID, NY).run("players_scan", {"stat": "anytime_td", "threshold": 1, "league": NFL})
    assert done.ok and done.content["day"] == "tomorrow" and done.content["no_games_today"] == ["NFL"]
    assert "these are tomorrow's: say so in one line" in done.content["hint"]
    [card] = done.cards
    assert card["title"] == "Strongest anytime TD · NFL · tomorrow" and card["no_games_today"] == ["NFL"]
    words = scan_template(card)
    assert words.startswith("No NFL games today, so here's tomorrow.\nStrongest anytime TD tomorrow:\n1. Owen Fields")
    assert "<i>No NFL games today, so here's tomorrow.</i>" in chat.card_text(card)


# --- m4: "Also:" and "N more qualify"; m5: yes/no rows; m6: the edge on posted lines --------------------------------

def _qualifiers(count: int) -> List[Dict[str, Any]]:
    return [row(i, f"Player {i}", "BOS", "NYK", 30.0 - i * 0.4, round(0.86 - i * 0.01, 4), [9, 10], [40, 45],
                athlete=str(9000 + i)) for i in range(1, count + 1)]


def test_qualifiers_past_the_fifth_are_also_and_the_rest_are_counted_in_the_template_bot_and_card(world) -> None:
    from telegram_betting_bot_simple.bot import chat
    lows = [row(11, "Low One", "MIA", "GS", 6.7, 0.17, [1, 10], [3, 40], athlete="9100")]
    card = scan_card(found(_qualifiers(10), lows, more=4))
    assert card["more"] == 4
    lines = scan_template(card).split("\n")
    # Every row the scan returned past the fifth is named under "Also:" (as the app's card lists them), and "more" is
    # the tool's own number, never counted again (fix:ai m1, contract 4 of SPEC_P0_FIX: it said 6 here, rows 9 and 10
    # counted twice against the card's 4)
    assert lines[6].startswith("Also: Player 6 (BOS v NYK) 27.6 projected, 80% to get there")
    assert lines[6].endswith("; Player 10 (BOS v NYK) 26.0 projected, 76% to get there, 9 of last 10.")
    assert lines[7] == "4 more qualify: open the Player finder."
    assert not any(line.startswith("Also close") for line in lines)  # 17% rows never sit above hidden 62% ones
    bot = chat.card_text(card)
    assert ("Also: Player 6 (27.6, 80%); Player 7 (27.2, 79%); Player 8 (26.8, 78%); Player 9 (26.4, 77%); "
            "Player 10 (26.0, 76%)") in bot
    assert "📱 4 more qualify: open the Player finder." in bot and "Also close" not in bot and "Low One" not in bot
    # With every qualifier named, who is close still shows
    few = scan_card(found(_qualifiers(4), lows))
    assert "Also close: Low One (MIA v GS) 6.7 projected, 17% to get there" in scan_template(few)
    assert "Also close: Low One (6.7, 17%)" in chat.card_text(few)


def test_a_yes_no_event_shows_its_chance_and_never_a_projection(world) -> None:
    from telegram_betting_bot_simple.bot import chat
    dd = [row(1, "Evan Quinn", "CHA", "MIN", 1.6, 0.65, [4, 10], [1, 4], threshold=2, why=["a double-double in 4 of "
                                                                                          "last 10"])]
    result = found(dd, ask="double-double", stat="dd", stat_label="double-double", threshold=2)
    card = scan_card(result)
    assert scan_template(card).split("\n")[1] == "1. Evan Quinn (CHA v MIN) 65% to get there, 4 of last 10"
    bot = chat.card_text(card)
    assert "1.6 proj" not in bot and "   65% · 4/10 L10 · 1/4 season" in bot
    facts = scan_facts(result)
    assert "projection" not in facts["players"][0] and facts["players"][0]["kind"] == "yes_no"
    assert "never a projection" in facts["hint"]


def test_rows_ranked_by_edge_show_their_edge(world) -> None:
    from telegram_betting_bot_simple.bot import chat
    line = dict(DK, line=37.5, label="Points + Rebounds + Assists", market="player_points_rebounds_assists")
    pra = [row(1, "Darius Tate", "SA", "TOR", 38.1, 0.4862, [4, 10], [1, 4], line=line, edge=0.0162, threshold=38),
           row(2, "Caleb Shaw", "OKC", "LAL", 39.8, 0.5797, [7, 10], [2, 4], line=dict(line, line=36.5), edge=-0.0233,
               threshold=37)]
    card = scan_card(found(pra, ask="PRA overs (posted lines)", stat="pra", threshold=None))
    lines = scan_template(card).split("\n")
    assert lines[1].endswith("line 37.5 posted, edge +1.6%") and lines[2].endswith("edge -2.3%")
    assert "edge +1.6%" in chat.card_text(card)
    # A number asked for ranks by chance: no edge there
    assert "edge" not in scan_template(scan_card(found(_qualifiers(2))))


# --- m7: the router's date, and weekday names --------------------------------------------------------------------------

def test_the_router_reads_the_date_and_a_weekday_is_mapped_to_the_scan_s_day(world) -> None:
    td = {"players": [row(1, "Owen Fields", "DET", "WSH", 0.9, 0.5878, [9, 10], [6, 7], model="form", threshold=1)]}
    world.lab.player_board = FakeBoards({(NFL, "today"): FakeBoard(NFL, "today", {"anytime_td": td}),
                                         (NFL, "tomorrow"): FakeBoard(NFL, "tomorrow", {"anytime_td": td})})
    # The router (without the date, as Haiku answered in the review) says today; code reads "sunday" on a Saturday
    out, claude = run(world, routed(player_scan=scan("anytime_td", 1), leagues=[NFL]),
                      [asked_scan(), text("Owen Fields leads at 59%.")],
                      "who should i bet on to score a touchdown sunday", clock=SATURDAY)
    first = claude.router()[0]["messages"][0]["content"].split("\n")[0]
    # No zone stored for them: the app's clock, and it says so (fix:ai m3; this said "the person's zone")
    assert first == "Now: Saturday 31 October 2026, 2:05 PM EDT (the app's clock: America/New_York)"
    assert "weekday names" in assistant_module.ROUTER_PROMPT and '"touchdown sunday" on a Saturday is tomorrow' in \
        " ".join(assistant_module.ROUTER_PROMPT.split())
    assert claude.tool_calls()[0]["input"]["day"] == "tomorrow"
    # A weekday further off is scanned for today, and the reply says scans cover today and tomorrow
    route = Route(intent="lookup", player_scan={"stat": "pts", "threshold": 20, "side": "over", "day": "today"})
    Assistant(Scripted(None, []), world.lab, world.store, tz=NY, clock=lambda: SATURDAY)._weekday(
        route, "20+ points on wednesday")
    assert route.player_scan["day"] == "today" and route.player_scan["wide"] is True
    # Without the router, code's own reading maps it too
    fallback = Route()
    scan_fallback(fallback, "who's scoring a touchdown sunday", now=SATURDAY, tz=NY)
    assert fallback.player_scan["stat"] == "anytime_td" and fallback.player_scan["day"] == "tomorrow"


# --- m8 and the plain-language gaps: a player's own row, hot and cold, the best edges ------------------------------

def test_will_steph_hit_5_threes_answers_about_steph(world) -> None:
    steph = row(1, "Stephen Curry", "GS", "LAL", 3.4, 0.2412, [2, 10], [9, 40], threshold=5, athlete="3975",
                why=["2 of last 10 at 5+"])
    lab_scan = ContractScan(found([dict(steph, qualifies=False)], ask="5+ threes", stat="fg3m", stat_label="threes",
                                  threshold=5))
    world.lab.player_scan = lab_scan
    prompt = " ".join(assistant_module.ROUTER_PROMPT.split())
    assert '"will steph hit 5 threes" is {fg3m, 5, over, today, player Stephen Curry}' in prompt
    route = routed(player_scan={**scan("fg3m", 5), "player": "Stephen Curry"}, players=["Stephen Curry"])
    out, claude = run(world, route, [asked_scan(), text("Stephen Curry is 24% for 5+ threes tonight.")],
                      "will steph hit 5 threes")
    context = claude.chat()[0]["messages"][-1]["content"][0]["text"]
    assert "They asked about Stephen Curry: call players_scan with" in context
    assert claude.tool_calls()[0]["input"]["player"] == "Stephen Curry"
    assert lab_scan.calls[0]["names"] == ["Stephen Curry"] and lab_scan.calls[0]["stat"] == "fg3m"
    card = out["cards"][0]
    assert card["names"] == ["Stephen Curry"] and card["title"] == "Stephen Curry: 5+ threes · NBA · tonight"
    assert scan_template(card).startswith("Stephen Curry for 5+ threes tonight:\n1. Stephen Curry (GS v LAL) 3.4 "
                                          "projected, 24% to get there, 2 of last 10")
    assert "doesn't qualify isn't projected to get there" in Toolbox(world.lab, UID, NY).run(
        "players_scan", {"stat": "fg3m", "threshold": 5, "player": "Stephen Curry"}).content["hint"]
    # Not in tonight's games: the scan says so
    gone = scan_card(found([], ask="5+ threes", stat="fg3m", threshold=5, names=["Ja Morant"],
                           missing_names=["Ja Morant"]))
    assert scan_template(gone) == "I couldn't find Ja Morant in tonight's games."  # contract 1's words


@pytest.mark.parametrize("message, reading, sort, side", [
    ("who's hot right now", {"sort": "hot", "day": "today"}, "hot", None),
    ("who's cold right now, who should i fade", {"sort": "cold", "day": "today"}, "cold", None),
    ("best player props tonight", {"sort": "edge", "day": "today"}, "edge", None),
    ("any good overs tonight?", {"sort": "edge", "side": "over", "day": "today"}, "edge", "over"),
])
def test_plain_language_asks_route_to_the_scan_s_new_orders(world, message, reading, sort, side) -> None:
    route = parse_route(routed(player_scan=reading, intent="chat"))
    assert route.intent == "lookup" and route.player_scan["stat"] is None and route.player_scan["sort"] == sort
    args = scan_args(route)
    assert args.get("sort") == sort and "stat" not in args and args.get("side") == side
    # The tool: hot and cold with no league named send no stat, so the Lab picks the first league with games and its
    # stat (fix:ai m4, contract 5: it sent points, which finds nothing on an NFL-only day); edge with no stat is edges
    # mode
    lab_scan = ContractScan(found(_qualifiers(3), mode="edges" if sort == "edge" else "scan"))
    world.lab.player_scan = lab_scan
    done = Toolbox(world.lab, UID, NY).run("players_scan", args)
    assert done.ok, done.content
    call_ = lab_scan.calls[0]
    assert call_["sort"] == sort and call_["stat"] is None
    assert call_["side"] == (side if sort == "edge" else "over")
    assert done.cards[0]["sort"] == sort and done.content["mode"] == ("edges" if sort == "edge" else "scan")
    # The router's prompt teaches each of them
    prompt = " ".join(assistant_module.ROUTER_PROMPT.split())
    assert {"hot": '"who\'s hot" is {sort hot, today}', "cold": '"who\'s cold, who should I fade" {sort cold, today}',
            "edge": '"best player props tonight" is {sort edge, today}'}[sort] in prompt
    if side:
        assert '"any good overs" {sort edge, side over, today}' in prompt


def test_the_best_edges_and_hot_or_cold_read_naturally_in_the_template_and_the_bot(world) -> None:
    from telegram_betting_bot_simple.bot import chat
    line = dict(DK, line=29.5)
    edge_rows = [dict(row(1, "Luka Dončić", "DAL", "PHX", 31.2, 0.58, [7, 10], [30, 45], line=line, edge=0.062,
                          threshold=30), stat="pts", stat_label="points", side="over"),
                 dict(row(2, "Evan Quinn", "CHA", "MIN", 8.8, 0.61, [6, 10], [3, 4], line=dict(line, line=9.5),
                          edge=0.041, threshold=9), stat="reb", stat_label="rebounds", side="under")]
    card = scan_card(dict(found(edge_rows, mode="edges", ask=None, stat=None, threshold=None), side=None, sort="edge"))
    assert card["title"] == "Best edges on the posted lines · NBA · tonight" and card["mode"] == "edges"
    lines = scan_template(card).split("\n")
    assert lines[0] == "Best edges on the posted lines tonight:"
    assert lines[1] == ("1. Luka Dončić (DAL v PHX) points over 29.5: 31.2 projected, 58% to clear it, 7 of last 10, "
                        "edge +6.2%")
    assert lines[2].startswith("2. Evan Quinn (CHA v MIN) rebounds under 9.5: 8.8 projected, 61% to stay under")
    bot = chat.card_text(card)
    assert "   points over 29.5 · 31.2 proj · 58% · 7/10 L10" in bot and "edge +6.2%" in bot
    assert "rebounds under 9.5 · 8.8 proj · 61% to stay under" in bot
    hot = [dict(row(1, "Luka Dončić", "DAL", "PHX", 31.2, 0.82, [9, 10], [41, 45]),
                trend={"l5": 34.2, "season": 28.1, "diff": 6.1, "vs": "season"})]
    hot_card = scan_card(found(hot, sort="hot"))
    assert hot_card["title"] == "Hottest on points (last 5 against the season) · NBA · tonight"
    # The trend in contract 2's words, the same as the app's chip and the bot (fix:ai F2: the template said "34.2 over
    # the last 5 against 28.1 this season (+6.1)"), and the chance with its own number (m2)
    assert scan_template(hot_card).split("\n")[1] == ("1. Luka Dončić (DAL v PHX) L5 34.2 vs 28.1 season, 31.2 projected, "
                                                      "82% for 20+, 9 of last 10")
    assert "L5 34.2 vs 28.1 season · 31.2 proj · 82% for 20+" in chat.card_text(hot_card)


# --- m11: the router-down reading --------------------------------------------------------------------------------------

@pytest.mark.parametrize("message, wanted", [
    ("10+ boards", {"stat": "reb", "threshold": 10}), ("SOG 3+", {"stat": "sog", "threshold": 3}),
    ("PRA overs", {"stat": "pra", "threshold": None}), ("PRA overs tonight", {"stat": "pra", "threshold": None}),
    ("Ks tonight", {"stat": "k", "threshold": None}),
    ("who's hot right now", {"stat": None, "sort": "hot"}),
    ("who's cold right now, who should i fade", {"stat": None, "sort": "cold"}),
    ("best player props tonight", {"stat": None, "sort": "edge"}),
    ("any good overs tonight?", {"stat": None, "sort": "edge", "side": "over"}),
])
def test_with_the_router_down_bettor_shorthand_still_scans(message, wanted) -> None:
    route = Route()
    scan_fallback(route, message)
    assert route.player_scan is not None and route.intent == "lookup", message
    assert {key: route.player_scan.get(key) for key in wanted} == wanted, (message, route.player_scan)


@pytest.mark.parametrize("message", ["give me 3 locks", "lock of the night", "what's your hammer tonight",
                                     "any nukes on the slate"])
def test_with_the_router_down_locks_hammers_and_nukes_are_the_strongest_picks(message) -> None:
    route = Route()
    scan_fallback(route, message)
    assert route.top_picks is True and route.player_scan is None and route.intent == "lookup", message


def test_with_the_router_down_game_lines_tickets_and_questions_stay_out_of_the_scan(world) -> None:
    for message in ("how many points did Luka score last night?", "$10 to win $50", "what does PRA mean?",
                    "what's the spread on lakers celtics", "make me a 3 leg parlay of player props",
                    "explain how overs work"):
        route = Route()
        scan_fallback(route, message)
        assert route.player_scan is None and route.top_picks is False, message
    # Through the assistant: "give me 3 locks" with the router down still gets the strongest picks
    out, claude = run(world, None, [call("brain_top_picks", {"leagues": [NBA], "n": 3}),
                                    text("Your 3 strongest picks tonight are on the card.")], "give me 3 locks")
    context = claude.chat()[0]["messages"][-1]["content"][0]["text"]
    assert "call brain_top_picks" in context and [card["type"] for card in out["cards"]] == ["picks"]


# --- RD1: no example numbers in the hint, and no hint is a source ------------------------------------------------------

def test_the_scan_hint_has_no_example_numbers_and_a_copied_example_fails_the_post_check(world) -> None:
    from telegram_betting_bot_simple.ai.explain import numbers_in
    facts = Toolbox(world.lab, UID, NY).run("players_scan", {"stat": "pts", "threshold": 20}).content
    assert numbers_in(facts["hint"]) == [] and "31.2" not in facts["hint"] and "82%" not in facts["hint"]
    assert 31.2 not in source_numbers({"hint": "31.2 projected, 82% to get 20+"})  # a hint is never a source
    # A board without Luka: the old hint's example ("31.2 projected, 82% to get 20+") copied for another player
    wes = {"players": [row(1, "Wes Porter", "SAC", "PHX", 28.3, 0.62, [7, 10], [30, 45], athlete="9001")]}
    world.lab.player_board = FakeBoards({(NBA, "today"): FakeBoard(NBA, "today", {"pts": wes}, games_total=8)})
    copied = "Wes Porter (SAC v PHX): 31.2 projected, 82% to get 20+, 9 of last 10."
    out, claude = run(world, routed(player_scan=scan("pts", 20)), [asked_scan(), text(copied)], SCREENSHOT)
    assert out["kind"] == "template" and "31.2" not in out["text"], out
    assert "numbers that are not in the facts you were given: 31.2, 82%" in claude.corrections()[0]
    # Whatever a hint says, its numbers aren't the turn's facts
    wes_facts = Toolbox(world.lab, UID, NY).run("players_scan", {"stat": "pts", "threshold": 20}).content
    run_ = SimpleNamespace(name="players_scan", ok=True, content=dict(wes_facts, hint="31.2 projected, 82%"), cards=[])
    assert "hint" not in Assistant._quotable(run_)
    assert problems(copied, source_numbers(Assistant._quotable(run_)))


# --- the card's contract with the app (D's), and the tool's input ------------------------------------------------------

def test_the_scan_card_carries_what_the_app_needs_to_show_and_re_run_it(world) -> None:
    done = Toolbox(world.lab, UID, NY).run("players_scan", {"stat": "pts", "threshold": 20})
    card = done.cards[0]
    for key, value in (("mode", "scan"), ("missing_names", []), ("games", 10), ("games_started", 0), ("error", None),
                       ("sort", None), ("names", [])):
        assert key in card and card[key] == value, (key, card.get(key))
    assert done.content["mode"] == "scan"
    world.lab.player_scan = ContractScan(found(_qualifiers(2), missing_names=["Ja Morant"], games_started=2,
                                               error="ESPN scoreboard timed out"))
    named = Toolbox(world.lab, UID, NY).run("players_scan", {"stat": "pts", "threshold": 20, "player": "Ja Morant",
                                                             "sort": "l10"})
    card = named.cards[0]
    assert card["names"] == ["Ja Morant"] and card["missing_names"] == ["Ja Morant"] and card["sort"] is None
    assert card["games_started"] == 2 and card["error"] == "ESPN scoreboard timed out"
    hot = Toolbox(world.lab, UID, NY).run("players_scan", {"stat": None, "sort": "hot"})
    assert hot.ok and hot.cards[0]["sort"] == "hot"


def test_the_tool_takes_a_player_and_a_sort_and_checks_them() -> None:
    tool = next(t for t in TOOLS if t["name"] == "players_scan")
    properties = tool["input_schema"]["properties"]
    assert properties["player"] == {"type": "string"}
    assert properties["sort"]["enum"] == ["chance", "projection", "l10", "edge", "hot", "cold"]
    assert properties["stat"]["type"] == ["string", "null"]
    assert len(json.dumps(TOOLS)) < 21_000  # every byte is input tokens on every round
    lab = SimpleNamespace(player_scan=ContractScan(found(_qualifiers(1))))
    box = Toolbox(lab, UID, NY)
    assert box.run("players_scan", {"stat": "pts", "player": "x" * 61}).content["error"] == "invalid_input"
    assert box.run("players_scan", {"stat": "pts", "sort": "vibes"}).content["error"] == "invalid_input"
    assert box.run("players_scan", {"stat": None}).content["error"] == "invalid_input"  # no stat, no order
    assert box.run("players_scan", {"stat": "pts", "player": "  Stephen   Curry "}).ok
    assert lab.player_scan.calls[-1]["names"] == ["Stephen Curry"]


# --- over the real Lab and engine (contract A as built): a player's row, hot and cold, the best edges ----------------

def test_the_tool_s_new_asks_run_through_the_real_lab_and_engine(world) -> None:
    from telegram_betting_bot_simple.bot import chat
    from tests.test_player_scan_wiring import real_boards
    boards = real_boards()  # tonight's NBA board as a snapshot, ranked by the real engine
    world.lab.player_board = boards
    box = Toolbox(world.lab, UID, NY)
    # A player's own row, whether he qualifies or not; one ruled out says so; one not on the board is named
    banchero = box.run("players_scan", {"stat": "pts", "threshold": 20, "player": "Paolo Banchero"})
    assert banchero.ok and boards.asks[-1]["names"] == ["Paolo Banchero"]
    [row_] = banchero.content["players"]
    assert row_["name"] == "Paolo Banchero" and row_["qualifies"] is False and banchero.cards[0]["names"] == \
        ["Paolo Banchero"]
    assert scan_template(banchero.cards[0]).startswith("Paolo Banchero for 20+ points tonight:\n1. Paolo Banchero "
                                                       "(ORL v MIA) ")
    lebron = box.run("players_scan", {"stat": "pts", "threshold": 20, "player": "LeBron James"})
    assert "1. LeBron James (LAL v MIN) [Out]: ruled out" in scan_template(lebron.cards[0])
    steph = box.run("players_scan", {"stat": "pts", "threshold": 20, "player": "Stephen Curry"})
    assert steph.content["missing_names"] == ["Stephen Curry"] and scan_template(steph.cards[0]) == \
        "I couldn't find Stephen Curry in tonight's games."
    # Hot and cold: the NBA's points by the last 5 against the season, highest or lowest first
    hot = box.run("players_scan", {"stat": None, "sort": "hot"})
    assert hot.ok and boards.asks[-1]["sort"] == "hot" and boards.asks[-1]["stat"] == "pts"
    diffs = [r["trend"]["diff"] for r in hot.content["players"]]
    assert diffs == sorted(diffs, reverse=True) and hot.cards[0]["sort"] == "hot"
    assert scan_template(hot.cards[0]).startswith("Hottest on points (last 5 against the season) tonight:\n1. ")
    assert "L5 " in chat.card_text(hot.cards[0])
    cold = box.run("players_scan", {"stat": None, "sort": "cold"})
    diffs = [r["trend"]["diff"] for r in cold.content["players"]]
    assert diffs == sorted(diffs) and "a fade looks at" in cold.content["hint"]
    # The best edges on every posted line: either side unless one is asked
    edges = box.run("players_scan", {"stat": None, "sort": "edge"})
    assert edges.ok and boards.asks[-1]["mode"] == "edges" and boards.asks[-1]["side"] is None
    card = edges.cards[0]
    assert card["mode"] == "edges" and card["players"] and all(r["edge"] > 0 for r in card["players"])
    first = card["players"][0]
    assert scan_template(card).split("\n")[1].startswith(
        f"1. {first['name']} ({first['team']} v {first['opponent']}) {first['stat_label']} {first['side']} ")
    overs = box.run("players_scan", {"stat": None, "sort": "edge", "side": "over"})
    assert boards.asks[-1]["side"] == "over" and all(r["side"] == "over" for r in overs.content["players"])


# --- the responsible-gambling hole the probes still found: a decline in either word order -------------------------

def test_a_decline_pointing_to_the_budget_and_break_in_settings_is_never_overridden_by_a_scan(world) -> None:
    decline = ("Betting more to make up a bad week usually means you'll lose more. Your budget and a break are in "
               "Settings.")
    out, claude = run(world, routed(player_scan=scan("pts", 20)), [text(decline)],
                      "rough week, down a lot. who's scoring 20+ tonight")
    assert out["kind"] == "answer" and out["text"].startswith(decline), out
    assert not out["cards"] and not claude.tool_calls() and claude.chat_calls == 1  # no retry, no code-run scan
    assert Assistant._guard(decline, Route(intent="lookup", player_scan=scan("pts", 20)), False, "rough week", []) == []


def test_a_scan_about_the_one_player_named_is_his_own_row(world) -> None:
    lab_scan = ContractScan(found([row(1, "Stephen Curry", "GS", "LAL", 3.4, 0.2412, [2, 10], [9, 40], threshold=5)],
                                  ask="5+ threes", stat="fg3m", stat_label="threes", threshold=5))
    world.lab.player_scan = lab_scan
    # The router named him among the players but left the scan's player out: code fills it in
    out, claude = run(world, routed(player_scan=scan("fg3m", 5), players=["Stephen Curry"]),
                      [asked_scan(), text("Stephen Curry is 24% for 5+ threes tonight.")], "will steph hit 5 threes")
    assert claude.tool_calls()[0]["input"]["player"] == "Stephen Curry" and lab_scan.calls[0]["names"] == \
        ["Stephen Curry"]
    # A list ask that names someone stays a list
    out, claude = run(world, routed(player_scan=scan("pts", 20), players=["Luka Dončić"]),
                      [asked_scan(), text("Here's the list.")], "who's scoring 20+ tonight besides luka")
    assert "player" not in claude.tool_calls()[0]["input"]


# --- the board's state in words: parts still missing, the scores down, every game started (RD5, RD7) -----------

def test_a_scan_says_what_s_missing_why_it_s_empty_and_never_calls_it_no_games(world) -> None:
    from telegram_betting_bot_simple.bot import chat
    # Every game loaded, one still missing a part that failed: a note, not "still building"
    failed = scan_card(found(_qualifiers(2), loading={"games_ready": 8, "games_total": 8, "failed": 2}))
    assert "2 games still missing data: ask again in a minute." in scan_template(failed)
    assert "<i>⏳ 2 games still missing data: ask again in a minute.</i>" in chat.card_text(failed)
    facts = scan_facts(found(_qualifiers(2), loading={"games_ready": 8, "games_total": 8, "failed": 2}))
    assert facts["loading"]["failed"] == 2 and "still missing data" in facts["hint"]
    assert "still loading" not in facts["hint"]
    # The scoreboard didn't load: say so, never "no games"
    down = scan_card(found([], leagues=[NBA], error="ESPN scoreboard timed out"))
    assert scan_template(down) == ("No player is projected for 20+ points tonight yet.\n"
                                   "ESPN's scores didn't load just now: ask again in a minute.")
    assert "the scores service failed just now" in scan_facts(found([], error="timeout"))["hint"]
    # Every game has started: nobody is left to rank. ``games`` is the games still to come, so that is games 0 (fix:ai
    # M5, contract 3: this built games=8, games_started=8, a card the code never makes, and read games as the total)
    started = scan_card(found([], games=0, games_started=8))
    assert scan_template(started) == "Every game tonight has started, so nobody is left to rank."
    assert "<i>Every game tonight has started, so nobody is left to rank.</i>" in chat.card_text(started)
    some = scan_card(found([], games=4, games_started=6))
    assert "Every game" not in scan_template(some)
    assert "6 games tonight have already started: their players aren't listed." in scan_template(some)


def test_no_scan_hint_carries_a_number_whatever_the_scan_found() -> None:
    import re
    line = dict(DK, line=29.5)
    variants = [
        found(_qualifiers(10), more=4), found([], _qualifiers(2)), found([]),
        found(_qualifiers(2), loading={"games_ready": 4, "games_total": 8, "failed": 1}),
        found(_qualifiers(2), sort="hot"), found(_qualifiers(2), sort="cold"),
        found([dict(row(1, "Luka Dončić", "DAL", "PHX", 31.2, 0.58, [7, 10], [30, 45], line=line, edge=0.062),
                    stat="pts", stat_label="points", side="over")], mode="edges", stat=None, threshold=None),
        found(_qualifiers(1), names=["Player 1", "Ja Morant"], missing_names=["Ja Morant"]),
        found(_qualifiers(1), ask="PRA overs (posted lines)", stat="pra", threshold=None, no_games_today=["NFL"]),
        found([], error="timeout", checked_tomorrow=True),
    ]
    for result in variants:
        hint = scan_facts(result)["hint"]
        assert not re.search(r"\d", hint), hint
