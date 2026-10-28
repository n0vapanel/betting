"""The player scan end to end over the real modules (v3 integration).

The owner's complaint: Ask AI answered "give me players that are projected to score at least 20 points today" with
"I can't pull up a list ... the props tool only works one player at a time" and a lecture. Here that message runs
through every real layer at once, with only ESPN's HTTP and Claude faked:

    ESPN's captures (tests/fixtures/scan, replayed by tests/test_player_scan.py's NbaWorld: ten NBA games of 28 October
    2026, rosters, season averages, game logs, injuries and DraftKings props; the NHL and NFL captures for the form
    model) -> the real EspnClient, PlayerData and shared TTLCache -> the real DashboardService (the NBA model's
    projections, over a saved odds board that costs no credits) -> the real PlayerBoard (data/player_board.py), made
    by the Lab (Lab.player_board -> board_for) and prewarmed by the real SlateBuilder -> the real engine
    (engine/scan.py) -> Lab.player_scan, GET /api/players/scan, the players_scan tool, the real Assistant (router
    reading, persona context, post-check, refusal guard, template) -> the bot's text.

Every number on the card is checked against its source: the chance is calls.stat_probabilities on the NBA model's own
mean and minutes (the player card's maths), the hits are hand counts of the player's ESPN game log. No request leaves
ESPN's hosts, never more than four at once, and The Odds API is never asked.
"""

from __future__ import annotations

import datetime
import json
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.ai.assistant import Assistant, scan_args
from telegram_betting_bot_simple.ai.explain import banned_phrases, refusal_problems, scan_template
from telegram_betting_bot_simple.ai.tools import Toolbox
from telegram_betting_bot_simple.bot import chat as bot_chat
from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.dashboard.lab import Lab
from telegram_betting_bot_simple.dashboard.service import DashboardService
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.espn_client import EspnClient
from telegram_betting_bot_simple.data.espn_players import PlayerData
from telegram_betting_bot_simple.data.player_board import PlayerBoard, board_for
from telegram_betting_bot_simple.data.slate import SlateBuilder
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.engine import calls
from telegram_betting_bot_simple.engine import scan as engine
from tests.test_player_scan import ESPN_HOSTS, NOW, NbaWorld, Web, WorldOdds, fixture_routes
from tests.test_player_scan_wiring import (
    LECTURE_AND_REFUSAL,
    OLD_REFUSAL,
    SCREENSHOT,
    reply,
    results_of,
    routed,
    say,
    tool_use,
    understood,
)
from tests.test_player_scan_wiring import scan as reading

NBA, NFL, NHL = engine.NBA, engine.NFL, engine.NHL
NY = ZoneInfo("America/New_York")
UID = "web:browser-scan0001"
STATIC = Path(__file__).resolve().parents[1] / "telegram_betting_bot_simple" / "dashboard" / "static"


# --- the stack -----------------------------------------------------------------------------------------------------

class Stack:
    """The real stack over ESPN's captures: NBA from NbaWorld (the real NBA model projects it), everything else from the
    scan fixtures (the NHL's and NFL's boards: the form model). ``gate`` holds one NBA game's DraftKings props until it
    is set (a partial board on demand)."""

    def __init__(self, tmp_path, monkeypatch: pytest.MonkeyPatch, delay: float = 0.0) -> None:
        self.nba = NbaWorld()
        self.held: Optional[str] = None
        self.gate = threading.Event()
        self.gate.set()
        self.web = Web(monkeypatch, self.routes, delay=delay)
        self.cache = TTLCache(name="scan-integration")
        self.espn = EspnClient(self.cache, clock=lambda: NOW)
        self.store = Store(str(tmp_path / "betting.db"))
        self.odds = WorldOdds(self.nba.odds_games())
        self.service = DashboardService(self.odds, self.espn, NY, store=self.store, cache=self.cache)
        self.players = PlayerData(self.espn, now=lambda: datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc))
        self.lab = Lab(self.service, self.store, self.players, now=lambda: NOW)
        self.lab.user(UID)
        self.store.confirm_age(UID, now=NOW)

    def routes(self, url: str, params: Dict[str, Any]) -> Optional[Any]:
        if "/basketball/" in url:
            if self.held and f"/events/{self.held}/" in url and url.endswith("/propBets"):
                assert self.gate.wait(30), "the held game was never let go"
            return self.nba.routes(url, params)
        return fixture_routes(url, params)

    def hold(self, event_id: str) -> None:
        self.held = event_id
        self.gate.clear()

    def board(self) -> PlayerBoard:
        return self.lab.player_board

    def client(self) -> TestClient:
        client = TestClient(create_app(self.service, "s3cret", login_delay=0, lab=self.lab))
        assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
        client.headers.update({"X-Player": UID.split(":", 1)[1]})
        return client

    def close(self) -> None:
        board = getattr(self.service, "player_board", None)
        if isinstance(board, PlayerBoard):
            board.close()


@pytest.fixture()
def stack(tmp_path, monkeypatch):
    built = Stack(tmp_path, monkeypatch)
    yield built
    built.gate.set()
    built.close()


# --- a fake Claude through the real assistant ----------------------------------------------------------------------

def facts_answer(facts: Dict[str, Any]) -> str:
    """The analyst's answer from players_scan's facts: the top three, each with the projection, the chance and the L10
    history exactly as the tool gave them, and the loading line when the board is partial."""
    lines = []
    for player in (facts.get("players") or facts.get("close") or [])[:3]:
        history = next((h for h in player.get("history") or [] if "last 10" in h), None) \
            or next((h for h in player.get("history") or [] if "last" in h), None)
        bits = [f"{player['projection']} projected", f"{player['chance'] * 100:.0f}% to get {facts['threshold']}+"]
        if history:
            bits.append(history)
        lines.append(f"{player['name']} ({player['team']} v {player['opponent']}): " + ", ".join(bits) + ".")
    loading = facts.get("loading") or {}
    if loading.get("games_total"):
        lines.append(f"{loading['games_ready']} of {loading['games_total']} games loaded so far.")
    return " ".join(lines) or f"Nobody is projected for {facts['ask']} yet."


class Claude:
    """The router reads each message as its prompt's examples teach (``route``); the chat model calls players_scan with
    what code understood and answers from its facts. ``first`` is an answer given before any tool (the old refusal)."""

    def __init__(self, route: Dict[str, Any], first: Optional[str] = None) -> None:
        self.route, self.first = route, first
        self.requests: List[Dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def chat(self) -> List[Dict[str, Any]]:
        return [r for r in self.requests if r["model"] != "claude-haiku-5-5"]

    def tool_calls(self) -> List[Dict[str, Any]]:
        seen, out = set(), []
        for request in self.chat():
            for message in request["messages"]:
                for block in message["content"] if isinstance(message["content"], list) else []:
                    if isinstance(block, dict) and block.get("type") == "tool_use" and block["id"] not in seen:
                        seen.add(block["id"])
                        out.append(block)
        return out

    def _create(self, **request):
        self.requests.append(json.loads(json.dumps(request, default=lambda o: getattr(o, "__dict__", str(o)))))
        if request["model"] == "claude-haiku-5-5":
            return reply([say(json.dumps(self.route))], model="claude-haiku-5-5")
        last = self.requests[-1]
        results = results_of(last)
        corrected = any("Your reply can't be shown" in str(m.get("content")) for m in last["messages"])
        if self.first is not None and not corrected:
            return reply([say(self.first)])
        if not results:
            asked = understood(last)["player_scan"]
            return tool_use("players_scan", scan_args(SimpleNamespace(player_scan=asked)), n=len(self.requests))
        return reply([say(facts_answer(json.loads(results[-1]["content"])))])


def ask(stack: Stack, claude: Claude, message: str) -> Dict[str, Any]:
    stack.lab.assistant = Assistant(claude, stack.lab, stack.store, tz=NY, clock=lambda: NOW)
    return stack.lab.chat(UID, {"message": message})["reply"]


def odds_id(event_id: str) -> str:
    return f"odds-{event_id}"


# --- the owner's message, end to end -------------------------------------------------------------------------------

def test_the_owners_message_ranks_tonights_nba_players_from_the_real_board_model_and_logs(stack) -> None:
    assert stack.board().wait(NBA, "today", 60)  # the Lab's own board (board_for), built through the shared cache
    claude = Claude(routed(player_scan=reading("pts", 20)))
    answer = ask(stack, claude, SCREENSHOT)
    # Routed to the scan, players_scan called once, answered first time: every number traced, no refusal or lecture
    assert answer["kind"] == "answer", answer
    assert [call["name"] for call in claude.tool_calls()] == ["players_scan"]
    assert claude.tool_calls()[0]["input"] == {"stat": "pts", "threshold": 20, "side": "over", "day": "today"}
    assert len(claude.chat()) == 2
    for phrase in LECTURE_AND_REFUSAL:
        assert phrase.lower() not in answer["text"].lower(), phrase
    assert not banned_phrases(answer["text"])
    assert refusal_problems(answer["text"], "lookup", True, asked=SCREENSHOT) == []
    [card] = answer["cards"]
    assert card["type"] == "player_scan" and card["leagues"] == [NBA] and card["ask"] == "20+ points"
    assert card["title"] == "Projected for 20+ points · NBA · tonight"
    rows = card["players"]
    assert len(rows) >= 5 and [row["rank"] for row in rows] == list(range(1, len(rows) + 1))
    assert [row["chance"] for row in rows] == sorted((row["chance"] for row in rows), reverse=True)
    assert answer["text"].startswith(f"{rows[0]['name']} ({rows[0]['team']} v {rows[0]['opponent']}): "
                                     f"{rows[0]['projection']} projected")
    events = {row["event_id"] for row in rows + card["close"]}
    assert len(events) > 3
    details = {event: stack.service.detail(NBA, odds_id(event))[2] for event in events}
    for row in rows + card["close"]:
        # The NBA model's own projection for the game, and the player card's own maths on it
        model_row = details[row["event_id"]][row["athlete_id"]]
        assert row["model"] == "nba" and row["chance_source"] == "nba" and row["model_label"] == "NBA model"
        assert row["projection"] == round(model_row["pts"], 1) and row["minutes"] == round(model_row["minutes"], 1)
        assert row["chance"] == round(calls.stat_probabilities("pts", model_row["pts"], 19.5, model_row["minutes"])[0], 4)
        # The history is his ESPN game log, counted by hand
        log = stack.players.gamelog(NBA, row["athlete_id"], last=100)["rows"]
        points = [game["stats"]["pts"] for game in log if game["stats"].get("pts") is not None]
        assert row["hits"]["l5"] == [sum(p >= 20 for p in points[:5]), len(points[:5])]
        assert row["hits"]["l10"] == [sum(p >= 20 for p in points[:10]), len(points[:10])]
        assert row["avg"]["l5"] == round(sum(points[:5]) / len(points[:5]), 1)
        assert set(row["hits"]) == {"l5", "l10", "season"} and row["addable"] is False and row["league"] == NBA
        assert 1 <= len(row["why"]) <= 3 and not banned_phrases(" ".join(row["why"]))
        assert not any(re.search(r"[+\-−]\d{3,}|posted at", reason) for reason in row["why"])  # no price in words
    # DraftKings' lines through ESPN (the captured game's, re-pointed at each game's leaders)
    lined = [row for row in rows + card["close"] if row.get("line")]
    assert lined and all(row["line"]["book"] == "DraftKings" for row in lined)
    # The app's route gives the same rows from the same board
    api = stack.client().get("/api/players/scan", params={"stat": "pts", "threshold": 20})
    assert api.status_code == 200, api.text
    assert [(r["athlete_id"], r["chance"], r["hits"]) for r in api.json()["players"]] == \
        [(r["athlete_id"], r["chance"], r["hits"]) for r in rows]
    # The bot's numbered list of the same card
    text = bot_chat.card_text(card)
    assert text.startswith("📊 <b>Projected for 20+ points · NBA · tonight</b>")
    assert f"1. <b>{rows[0]['name']}</b>" in text and f"{rows[0]['projection']:.1f} proj" in text
    # Only ESPN was asked, never more than four requests at once; The Odds API never
    assert stack.web.hosts() <= ESPN_HOSTS and stack.web.peak <= 4
    assert not any("the-odds-api" in call["url"] for call in stack.web.calls)
    assert all(call["kind"] for call in stack.web.calls)  # every request inside a shared-cache entry


def test_the_old_refusal_is_caught_retried_and_ends_with_the_real_boards_numbers(stack) -> None:
    assert stack.board().wait(NBA, "today", 60)
    claude = Claude(routed(player_scan=reading("pts", 20)), first=OLD_REFUSAL)
    answer = ask(stack, claude, SCREENSHOT)
    assert answer["kind"] == "answer", answer
    first_correction = next(m for r in claude.chat() for m in r["messages"]
                            if "Your reply can't be shown" in str(m.get("content")))
    assert "a refusal" in str(first_correction["content"]) and "players_scan" in str(first_correction["content"])
    assert [call["name"] for call in claude.tool_calls()] == ["players_scan"]
    assert answer["cards"][0]["type"] == "player_scan" and answer["cards"][0]["players"]
    for phrase in LECTURE_AND_REFUSAL:
        assert phrase.lower() not in answer["text"].lower(), phrase


def test_a_cold_board_answers_at_once_with_what_is_loaded_then_in_full(tmp_path, monkeypatch) -> None:
    built = Stack(tmp_path, monkeypatch)
    try:
        last_game = built.nba.events[-1]["id"]
        built.hold(last_game)
        # The process's board, shared with the Lab (board_for): a scan never waits for it here
        board = board_for(built.service, built.players, now=lambda: NOW, wait=0.0)
        assert built.lab.player_board is board
        first = built.lab.player_scan(UID, NBA, "pts", 20)
        assert "loading" in first  # the day's games, or some of them, still being read
        deadline = time.time() + 30
        while time.time() < deadline and board.snapshot(NBA, "today")["games_ready"] < 9:
            time.sleep(0.05)
        partial = built.lab.player_scan(UID, NBA, "pts", 20)
        assert partial["loading"] == {"games_ready": 9, "games_total": 10}
        assert partial["players"] and all(row["event_id"] != last_game for row in partial["players"])
        # The tool tells the analyst to give what's ready and say what's loading; code's template says it plainly
        run = Toolbox(built.lab, UID, NY).run("players_scan", {"league": NBA, "stat": "pts", "threshold": 20})
        assert run.ok and run.content["loading"] == {"games_ready": 9, "games_total": 10}
        assert "still loading" in run.content["hint"]
        assert run.cards[0]["loading"] == {"games_ready": 9, "games_total": 10}
        assert "9 of 10 games loaded so far: ask again in a minute for the rest." in scan_template(run.cards[0])
        built.gate.set()
        assert board.wait(NBA, "today", 30)
        full = built.lab.player_scan(UID, NBA, "pts", 20)
        assert "loading" not in full and full["games"] == 10
        assert any(row["event_id"] == last_game for row in full["players"] + full["close"])
    finally:
        built.gate.set()
        built.close()


def test_the_slate_worker_prewarms_the_board_the_lab_scans(stack) -> None:
    slate = SlateBuilder(odds=None, espn=None, service=stack.service, now=lambda: NOW)
    first_tip = stack.nba.events[0]["date"]
    slate._prewarm_players([{"league": NBA, "start": first_tip, "status": "pre"},
                            {"league": NHL, "start": first_tip, "status": "pre"}], NOW)
    # The slate asked the service's Lab, which made the process's one board and started tonight's NBA build
    board = stack.service.player_board
    assert isinstance(board, PlayerBoard) and stack.lab.player_board is board
    assert board.wait(NBA, "today", 60)
    assert [entry["league"] for entry in board.status()] == [NBA]  # one cold build at a time: the NHL comes next round
    out = stack.lab.player_scan(UID, None, "points", 20)
    assert out["leagues"] == [NBA] and "loading" not in out and out["players"]


# --- the form model, other leagues and the lingo -------------------------------------------------------------------

def test_shots_on_goal_scan_the_nhl_from_recent_form_through_the_lab_tool_and_bot(stack) -> None:
    out = stack.lab.player_scan(UID, None, "shots", 3)
    assert stack.board().wait(NHL, "today", 60)
    out = stack.lab.player_scan(UID, None, "shots", 3)
    assert out["leagues"] == [NHL] and out["stat"] == "sog" and out["ask"] == "3+ shots on goal"
    rows = out["players"] + out["close"]
    assert rows and all(row["model"] == "form" and row["model_label"] == "from recent form" for row in rows)
    assert all(set(row["hits"]) == {"l5", "l10", "season"} for row in rows)
    assert [entry["league"] for entry in stack.board().status()] == [NHL]  # "shots" never touched the NBA's board
    run = Toolbox(stack.lab, UID, NY).run("players_scan", {"stat": "SOG", "threshold": 3})
    assert run.ok and run.content["leagues"] == [NHL]
    assert all(row["model_label"] == "from recent form" for row in run.content["players"] + run.content["close"])
    if run.cards:
        assert "from recent form" in bot_chat.card_text(run.cards[0])


def test_anytime_td_without_nfl_games_today_says_so_in_one_line(stack) -> None:
    out = stack.lab.player_scan(UID, None, "anytime TD", None)
    assert out["players"] == [] and out["leagues_without_games"] == ["NFL"]
    assert scan_template(out) == "No player is projected for anytime TD tonight yet (NFL: no games)."
    assert not any("/basketball/" in call["url"] for call in stack.web.calls)


@pytest.mark.parametrize("phrase, leagues", [
    ("points", [NBA, engine.WNBA, NHL]), ("assists", [NBA, engine.WNBA, NHL]), ("TD", [NFL]),
    ("anytime TD scorers", [NFL]), ("shots", [NHL]), ("SOG", [NHL]), ("boards", [NBA, engine.WNBA]),
    ("Ks", [engine.MLB]), ("PRA", [NBA, engine.WNBA]), ("goals", [NHL]),
])
def test_a_stat_without_a_league_goes_to_the_spec_s_leagues_in_order(phrase, leagues) -> None:
    assert [league for league, _ in Lab._scan_candidates(engine, phrase)] == leagues
    assert list(engine.leagues_for(phrase)) == leagues


def test_the_router_s_triple_double_goes_to_the_nba_not_to_touchdowns(stack) -> None:
    assert stack.board().wait(NBA, "today", 60)
    claude = Claude(routed(player_scan=reading("td", 3)))
    answer = ask(stack, claude, "who's getting a triple-double tonight")
    assert claude.tool_calls()[0]["input"]["league"] == NBA and claude.tool_calls()[0]["input"]["stat"] == "td"
    assert answer["kind"] == "answer" and answer["text"] == "Nobody is projected for triple-double yet."
    # The tool ran the NBA's triple-double over tonight's board (nobody in this world gets to 30%: no card)
    content = json.loads(results_of(claude.chat()[-1])[-1]["content"])
    assert content["ask"] == "triple-double" and content["leagues"] == [NBA] and content["stat"] == "td"
    # A person's own "TD" without a league is a touchdown
    assert engine.leagues_for("TD") == {NFL: "anytime_td"}


# --- the frontend's stat menu is the engine's ----------------------------------------------------------------------

def _js_object(source: str, name: str) -> Any:
    start = source.index(f"const {name} = ") + len(f"const {name} = ")
    depth, end = 0, start
    for end in range(start, len(source)):
        depth += source[end] in "{[" and 1 or source[end] in "}]" and -1 or 0
        if depth == 0 and end > start:
            break
    return source[start:end + 1]


def test_the_player_finders_stat_menu_mirrors_the_engines_table() -> None:
    source = (STATIC / "js" / "views" / "players" / "scan.js").read_text(encoding="utf-8")
    tables = json.loads(_js_object(source, "SCAN_TABLES"))
    leagues = re.findall(r'\{ key: "([a-z_]+)", title: "[^"]+", table: "([a-z]+)"', _js_object(source, "SCAN_LEAGUES"))
    assert [league for league, _ in leagues] == [league for league in engine.LEAGUE_ORDER if league in engine.STATS]
    assert set(engine.STATS) == {league for league, _ in leagues}
    for league, table in leagues:
        menu = tables[table]
        assert [stat["key"] for stat in menu] == list(engine.STATS[league]), league
        for stat in menu:
            spec = engine.STATS[league][stat["key"]]
            assert stat["yes"] == spec["yes"], (league, stat["key"])
            if spec["threshold"] is not None:
                assert stat["start"] == spec["threshold"], (league, stat["key"])
            engine.make_ask(league, stat["key"], stat["start"])  # every menu entry makes an ask
