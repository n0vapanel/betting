"""v4 P0 end to end over the real modules (the integrator's tests): every engineer's part at once, with only ESPN's HTTP
and Claude faked.

    ESPN's captures (tests/fixtures/scan, replayed by tests/test_player_scan.py's NbaWorld for ten NBA games of 28
    October 2026; the NHL and NFL captures for the form model) -> the real EspnClient, PlayerData and shared TTLCache
    -> the real DashboardService (the NBA model, over a saved odds board that costs no credits) -> the real
    PlayerBoard (made by the Lab) -> the real engine (engine/scan.py: names, sorts, trend, kind, edges mode) ->
    Lab.player_scan and GET /api/players/scan (contract A) -> the players_scan tool, the real Assistant (router
    reading, the Now line, the route-aware guard, the post-check, the template: contract B) -> the bot's text.

The asks are the owner's and the persona review's: "give me players that are projected to score at least 20 points
today", "who's hot", "best player props tonight", "will Trae Young score 20+", an under at a milestone's number, a
board whose roster read failed once, a decline pointing to the budget and break, and the picks pause after a warning
sign. The router reads each message as its prompt's examples teach (a fixed JSON per test); the analyst calls
players_scan with what code understood and answers from the tool's facts only, so every number it writes is the
tool's. The Odds API is never asked, and only ESPN's hosts are.

The world keeps one clock for the Lab, its board and the shared cache (``Clock``), so a refresh half an hour on finds
the cache's copies as old as they would be on the server.
"""

from __future__ import annotations

import datetime
import json
import re
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.ai.assistant import Assistant, scan_args
from telegram_betting_bot_simple.ai.explain import banned_phrases, refusal_problems, scan_template
from telegram_betting_bot_simple.ai.tools import Toolbox
from telegram_betting_bot_simple.bot import chat as bot_chat
from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.dashboard.lab import Lab
from telegram_betting_bot_simple.dashboard.service import CallRejected, DashboardService
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.espn_client import EspnClient
from telegram_betting_bot_simple.data.espn_players import PlayerData
from telegram_betting_bot_simple.data.player_board import PlayerBoard
from telegram_betting_bot_simple.data.store import Store
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
from tests.test_review2_data import Clock, settle

NBA, NFL, NHL = engine.NBA, engine.NFL, engine.NHL
NY = ZoneInfo("America/New_York")
UID = "web:browser-p0int01"
EDMONTON = "6"  # the NHL fixture's Edmonton Oilers (Calgary at Edmonton has DraftKings props)
# Two of the NBA world's template players get real names (their ids, logs and props are unchanged): a name filter reads
# a person's words, accents and all
NAMED = {"ATL Player 3": "Trae Young", "CHI Player 3": "Nikola Vučević"}
EDGE_WORDS = re.compile(r"edge ([+\-−]?\d+(?:\.\d+)?)%")


# --- the stack -----------------------------------------------------------------------------------------------------

class World:
    """The real stack over ESPN's captures, on one clock: NBA from NbaWorld (the real NBA model projects it, two players
    renamed), everything else from the scan fixtures (the NHL's and NFL's boards: the form model). ``fail_roster(team)``
    makes that NHL team's next roster read fail once (an ESPN blip at build time)."""

    def __init__(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.clock = Clock()
        self.nba = NbaWorld()
        for roster in self.nba.rosters.values():
            for athlete in roster:
                athlete["displayName"] = NAMED.get(athlete.get("displayName"), athlete.get("displayName"))
        self.failing: List[str] = []
        self.failed: List[str] = []
        self.web = Web(monkeypatch, self.routes)
        self.cache = TTLCache(name="p0-integration", clock=self.clock)
        self.espn = EspnClient(self.cache, clock=lambda: NOW)
        self.store = Store(str(tmp_path / "betting.db"))
        self.service = DashboardService(WorldOdds(self.nba.odds_games()), self.espn, NY, store=self.store,
                                        cache=self.cache)
        self.players = PlayerData(self.espn, now=lambda: datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc))
        self.lab = Lab(self.service, self.store, self.players, now=self.clock)
        self.lab.user(UID)
        self.store.confirm_age(UID, now=NOW)

    def routes(self, url: str, params: Dict[str, Any]) -> Optional[Any]:
        if "/basketball/" in url:
            return self.nba.routes(url, params)
        for team in list(self.failing):
            if "/hockey/" in url and url.endswith(f"/teams/{team}/roster"):
                self.failing.remove(team)
                self.failed.append(url)
                raise ConnectionError("ESPN blip")
        return fixture_routes(url, params)

    def fail_roster(self, team: str) -> None:
        self.failing.append(team)

    def board(self) -> PlayerBoard:
        return self.lab.player_board

    def client(self) -> TestClient:
        client = TestClient(create_app(self.service, "s3cret", login_delay=0, lab=self.lab))
        assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
        client.headers.update({"X-Player": UID.split(":", 1)[1]})
        return client

    def snapshot_players(self, league: str) -> Dict[str, Dict[str, Any]]:
        return {player["athlete_id"]: player for player in self.board().snapshot(league)["players"]}

    def espn_only(self) -> None:
        """No request left ESPN's hosts (The Odds API never), never more than four at once."""
        hosts = {re.sub(r"^https?://([^/]+).*$", r"\1", call["url"]) for call in self.web.calls}
        assert hosts <= ESPN_HOSTS, hosts
        assert not any("the-odds-api" in call["url"] for call in self.web.calls)
        assert self.web.peak <= 4

    def close(self) -> None:
        board = getattr(self.service, "player_board", None)
        if isinstance(board, PlayerBoard):
            board.close()


@pytest.fixture()
def world(tmp_path, monkeypatch):
    built = World(tmp_path, monkeypatch)
    yield built
    built.close()


# --- a fake Claude through the real assistant ----------------------------------------------------------------------

def _pct(share: Any) -> str:
    return f"{float(share) * 100:.0f}%"


def _history(row: Dict[str, Any]) -> Optional[str]:
    return next((h for h in row.get("history") or [] if "last 10" in h), None) \
        or next((h for h in row.get("history") or [] if "last" in h), None)


def _row_words(row: Dict[str, Any], facts: Dict[str, Any]) -> str:
    """One row as the analyst says it, from the row's facts only."""
    name = f"{row['name']} ({row['team']} v {row['opponent']})"
    if row.get("ruled_out"):
        return f"{name} is ruled out"
    bits = []
    if row.get("projection") is not None:  # a yes/no row has none: its chance is the answer
        bits.append(f"{row['projection']:g} projected")
    if facts.get("side") == "under":
        bits.append(f"{_pct(row['chance'])} to stay under {row['threshold'] - 0.5:g}")
    else:
        bits.append(f"{_pct(row['chance'])} to get {row['threshold']}+")
    history = _history(row)
    if history:
        bits.append(history)
    words = f"{name}: " + ", ".join(bits)
    if row.get("qualifies") is False:
        words += " (not projected to get there)"
    return words


def analyst_answer(facts: Dict[str, Any]) -> str:
    """How the analyst answers players_scan's facts, the way its hint says, every number the tool's: the best edges with
    each one's stat, side, line, chance and edge; hot and cold with the last 5 against the season; named players one by
    one (and the names not in that day's games); else the top three, and the board's state in one line."""
    if facts.get("error") and not facts.get("players"):
        return "Picks are paused right now, so I can't list players until then." if "paused" in str(facts["error"]) \
            else "ESPN's scores didn't load just now: ask again in a minute."
    rows = facts.get("players") or []
    when = "tomorrow" if facts.get("day") == "tomorrow" else "tonight"
    if facts.get("mode") == "edges":
        words = f"Best edges on {when}'s posted lines: " + "; ".join(
            f"{r['name']} {r['stat_label']} {r['side']} {r['line']['line']:g}: {_pct(r['chance'])}, edge "
            f"{float(r['edge']) * 100:+.1f}%" for r in rows[:3]) + "." if rows else "No posted line shows an edge yet."
    elif facts.get("sort") in ("hot", "cold"):
        words = f"{'Hottest' if facts['sort'] == 'hot' else 'Coldest'} on {facts.get('stat')} {when}: " + "; ".join(
            f"{r['name']} ({r['trend']['l5']:g} over the last 5 against {r['trend']['season']:g})"
            for r in rows[:3]) + "."
    elif facts.get("names"):
        words = " ".join([_row_words(r, facts) + "." for r in rows[:2]] +
                         [f"{name} isn't in {when}'s games." for name in facts.get("missing_names") or []])
    else:
        shown = rows or facts.get("close") or []
        words = " ".join(_row_words(r, facts) + "." for r in shown[:3]) \
            or f"Nobody is projected for {facts['ask']} yet."
    loading = facts.get("loading") or {}
    if loading.get("games_total") and (loading.get("games_ready") or 0) < loading["games_total"]:
        words += f" {loading['games_ready']} of {loading['games_total']} games loaded so far."
    elif loading.get("failed"):
        words += f" {loading['failed']} game{'' if loading['failed'] == 1 else 's'} still missing data."
    return words


class Analyst:
    """The router reads the message as its prompt teaches (``route``, the JSON Claude Haiku returns); the chat model
    calls players_scan with what code understood (the context's values; ``more`` adds inputs of its own, such as n)
    and answers from its facts. ``first`` is a reply given before any tool (a refusal, a decline, an honest "can't")."""

    def __init__(self, route: Dict[str, Any], first: Optional[str] = None,
                 answer: Callable[[Dict[str, Any]], str] = analyst_answer,
                 more: Optional[Dict[str, Any]] = None) -> None:
        self.route, self.first, self.answer, self.more = route, first, answer, dict(more or {})
        self.requests: List[Dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def chat(self) -> List[Dict[str, Any]]:
        return [r for r in self.requests if r["model"] != "claude-haiku-5-5"]

    def router(self) -> List[Dict[str, Any]]:
        return [r for r in self.requests if r["model"] == "claude-haiku-5-5"]

    def corrections(self) -> List[str]:
        return [str(m.get("content")) for r in self.chat() for m in r["messages"]
                if "Your reply can't be shown" in str(m.get("content"))]

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
        if self.first is not None and not corrected and not results:
            return reply([say(self.first)])
        if not results:
            asked = understood(last).get("player_scan")
            if asked is None:  # no scan to run: it says the same again (an honest "can't" stays honest)
                return reply([say(self.first or "Ask me about tonight's players.")])
            return tool_use("players_scan", {**scan_args(SimpleNamespace(player_scan=asked)), **self.more},
                            n=len(self.requests))
        return reply([say(self.answer(json.loads(results[-1]["content"])))])


def chat(world: World, analyst: Analyst, message: str) -> Dict[str, Any]:
    world.lab.assistant = Assistant(analyst, world.lab, world.store, tz=NY, clock=world.clock)
    return world.lab.chat(UID, {"message": message})["reply"]


def facts_of(analyst: Analyst) -> Dict[str, Any]:
    """The last players_scan facts the analyst was given."""
    results = [r for request in analyst.chat() for r in results_of(request)]
    return json.loads(results[-1]["content"])


def clean(answer: Dict[str, Any]) -> None:
    """Answered first time by the analyst: no refusal, no lecture, no banned wording."""
    assert answer["kind"] == "answer", answer
    for phrase in LECTURE_AND_REFUSAL:
        assert phrase.lower() not in answer["text"].lower(), phrase
    assert not banned_phrases(answer["text"])
    assert refusal_problems(answer["text"], "lookup", True) == []


# --- the owner's question ------------------------------------------------------------------------------------------

def test_the_owners_20_plus_points_question_answers_with_the_p0_card_template_and_bot(world) -> None:
    assert world.board().wait(NBA, "today", 60)
    analyst = Analyst(routed(player_scan=reading("pts", 20)))
    answer = chat(world, analyst, SCREENSHOT)
    clean(answer)
    assert [call["input"] for call in analyst.tool_calls()] == [{"stat": "pts", "threshold": 20, "side": "over",
                                                                  "day": "today"}]
    # The router's first line is the person's own date and time (m7): Wednesday 28 October 2026, noon in New York
    router_text = json.dumps(analyst.router()[-1]["messages"])
    # No zone stored for this person: the app's clock, said so (P0 fix m3)
    assert "Now: Wednesday 28 October 2026, 12:00 PM EDT (the app's clock: America/New_York)" in router_text
    [card] = answer["cards"]
    # The card's P0 contract (B, for the app's trend chip, Refresh and "Open in Player finder")
    assert card["type"] == "player_scan" and card["mode"] == "scan" and card["sort"] is None and card["names"] == []
    assert card["missing_names"] == [] and card["games"] == 10 and card["games_started"] == 0 and card["error"] is None
    assert card["title"] == "Projected for 20+ points · NBA · tonight" and card["more"] > 0
    rows = card["players"]
    assert len(rows) == 10 and [row["chance"] for row in rows] == sorted((row["chance"] for row in rows), reverse=True)
    for row in rows + card["close"]:
        # Every row: its kind and whether it qualifies (contract A); no trend: the captured world's 4 games this season
        # are short of the 5 a trend needs (P0 fix contract 2, F2: the hot and cold test moves the world on)
        assert row["kind"] == "count" and "trend" not in row
        assert row["model"] == "nba" and row["league"] == NBA and row["addable"] is False
    assert all(row["qualifies"] is True for row in rows) and all(row["qualifies"] is False for row in card["close"])
    # m4: five, then "Also:" with every other row returned, then how many more the scan ranked: the card's own ``more``,
    # never counted again (P0 fix m1, contract 4: this counted rows 9 and 10 into it)
    lines = scan_template(card).split("\n")
    assert lines[0] == "Projected for 20+ points tonight:" and [line[:3] for line in lines[1:6]] == \
        ["1. ", "2. ", "3. ", "4. ", "5. "]
    assert lines[6].startswith(f"Also: {rows[5]['name']} ") and f"; {rows[9]['name']} " in lines[6]
    assert lines[7] == f"{card['more']} more qualify: open the Player finder."
    text = bot_chat.card_text(card)
    assert f"\nAlso: {rows[5]['name']} ({rows[5]['projection']:.1f}, {_pct(rows[5]['chance'])})" in text
    assert f"📱 {card['more']} more qualify: open the Player finder." in text and "Also close" not in text
    # RD1: the tool's hint carries no number, so a reply can't copy one from it
    facts = facts_of(analyst)
    assert facts["mode"] == "scan" and not re.search(r"\d", facts["hint"])
    assert all(player["kind"] == "count" and "trend" not in player for player in facts["players"])
    # The app's route gives the same rows from the same board, with the same contract fields
    body = world.client().get("/api/players/scan", params={"stat": "pts", "threshold": 20}).json()
    assert [(r["athlete_id"], r["chance"], r.get("trend")) for r in body["players"]] == \
        [(r["athlete_id"], r["chance"], r.get("trend")) for r in rows]
    assert body["mode"] == "scan" and body["sort"] is None and body["names"] == [] and body["no_games"] is False
    world.espn_only()


# --- "who's hot" -------------------------------------------------------------------------------------------------

def test_whos_hot_and_whos_cold_rank_tonights_nba_by_the_last_5_against_the_season(world) -> None:
    # A trend needs 5 games this season (P0 fix contract 2, F2): the captured 28 October has 4, so the world moves on to
    # 12 games, the last 5 against this season's other 7
    from tests.test_p0_fix_data import move_on
    move_on(world.nba, 12)
    assert world.board().wait(NBA, "today", 60)
    analyst = Analyst(routed(player_scan={"sort": "hot", "day": "today"}))
    answer = chat(world, analyst, "who's hot")
    clean(answer)
    # No stat named: the league's own (points for the NBA), no stat or side sent
    assert [call["input"] for call in analyst.tool_calls()] == [{"day": "today", "sort": "hot"}]
    [card] = answer["cards"]
    assert card["sort"] == "hot" and card["mode"] == "scan" and card["stat"] == "pts" and card["leagues"] == [NBA]
    rows = card["players"]
    assert rows and card["close"] == []
    diffs = [row["trend"]["diff"] for row in rows]
    assert diffs == sorted(diffs, reverse=True)
    snapshot = world.snapshot_players(NBA)
    for row in rows:
        played = [r for r in snapshot[row["athlete_id"]]["rows"] if (r.get("stats") or {}).get("pts") is not None]
        assert len(played) >= engine.TREND_GAMES and not engine.ruled_out(row.get("status"))
        # The trend is his game log's, by hand: this season's last 5 games' average against its other games
        season = [r["stats"]["pts"] for r in played if r.get("season") == played[0].get("season")]
        assert len(season) == 12 and row["trend"]["vs"] == "season"
        assert row["trend"]["l5"] == round(sum(season[:5]) / 5, 1)
        assert row["trend"]["season"] == round(sum(season[5:]) / len(season[5:]), 1)
    first = rows[0]
    assert f"{first['name']} ({first['trend']['l5']:g} over the last 5 against {first['trend']['season']:g})" in \
        answer["text"]
    assert scan_template(card).startswith("Hottest on points (last 5 against the season) tonight:\n1. ")
    assert f"L5 {first['trend']['l5']:g} vs {first['trend']['season']:g} season" in bot_chat.card_text(card)
    # Cold: lowest first, and the hint says it's what a fade looks at
    analyst = Analyst(routed(player_scan={"sort": "cold", "day": "today"}))
    answer = chat(world, analyst, "who's cold, who should I fade")
    clean(answer)
    [card] = answer["cards"]
    diffs = [row["trend"]["diff"] for row in card["players"]]
    assert card["sort"] == "cold" and diffs == sorted(diffs) and "a fade looks at" in facts_of(analyst)["hint"]
    world.espn_only()


# --- "best player props tonight" --------------------------------------------------------------------------------

def test_best_player_props_tonight_ranks_every_stats_lines_and_never_shows_a_suspect_edge(world) -> None:
    assert world.board().wait(NBA, "today", 60)
    # The same board holds edges the numbers can't stand behind: a 25+ rung at +179 for a player 64% to get there
    # reads +78%, the model's tail against a long price (contract A: edge_suspect, no number, never ranked)
    suspects = [row for row in world.lab.player_scan(UID, NBA, "pts", 25, n=25)["players"] if row.get("edge_suspect")]
    assert suspects and all("edge" not in row and row["line"]["label"] == "25+" for row in suspects)
    would_be = {f"{(row['chance'] * row['line']['over']['decimal'] - 1) * 100:+.1f}%" for row in suspects}
    assert all(float(value.rstrip("%")) > engine.EDGE_CAP * 100 for value in would_be)
    analyst = Analyst(routed(player_scan={"sort": "edge", "day": "today"}))
    answer = chat(world, analyst, "best player props tonight")
    clean(answer)
    assert [call["input"] for call in analyst.tool_calls()] == [{"day": "today", "sort": "edge"}]
    [card] = answer["cards"]
    assert card["mode"] == "edges" and card["sort"] == "edge" and card.get("stat") is None and card.get("side") is None
    rows = card["players"]
    assert rows and [row["edge"] for row in rows] == sorted((row["edge"] for row in rows), reverse=True)
    for row in rows:
        assert 0 < row["edge"] <= engine.EDGE_CAP and "edge_suspect" not in row
        assert row["stat"] in engine.STATS[NBA] and row["side"] in ("over", "under") and row["line"]["matches"]
        assert row["stat_label"] == engine.STATS[NBA][row["stat"]]["label"]
    assert not {(row["athlete_id"], "pts", 25) for row in suspects} & {(r["athlete_id"], r["stat"], r["threshold"])
                                                                        for r in rows}
    # No suspect value anywhere the person reads: the reply, code's template, the bot
    template, text = scan_template(card), bot_chat.card_text(card)
    for words in (answer["text"], template, text):
        assert "edge_suspect" not in words and not any(value in words for value in would_be)
        assert all(float(found.replace("−", "-")) <= engine.EDGE_CAP * 100 for found in EDGE_WORDS.findall(words))
    assert EDGE_WORDS.findall(answer["text"]) and template.startswith("Best edges on the posted lines tonight:\n1. ")
    first = rows[0]
    assert template.split("\n")[1].startswith(f"1. {first['name']} ({first['team']} v {first['opponent']}) "
                                              f"{first['stat_label']} {first['side']} ")
    # "any good overs": the over side only
    analyst = Analyst(routed(player_scan={"sort": "edge", "side": "over", "day": "today"}))
    answer = chat(world, analyst, "any good overs")
    [card] = answer["cards"]
    assert card["mode"] == "edges" and card["side"] == "over" and all(r["side"] == "over" for r in card["players"])
    world.espn_only()


def test_a_scan_whose_rows_have_suspect_edges_never_quotes_one(world) -> None:
    assert world.board().wait(NBA, "today", 60)
    analyst = Analyst(routed(player_scan=reading("pts", 25)), more={"n": 25})  # the 25+ rungs at +179 rank 11th on
    answer = chat(world, analyst, "who's getting 25+ points tonight")
    clean(answer)
    [card] = answer["cards"]
    suspects = [row for row in card["players"] if row.get("edge_suspect")]
    assert suspects and all("edge" not in row for row in suspects)
    facts = facts_of(analyst)
    named = {row["name"] for row in suspects}
    assert all("edge" not in row for row in facts["players"] if row["name"] in named)
    for words in (answer["text"], scan_template(card), bot_chat.card_text(card)):
        assert not EDGE_WORDS.findall(words) and "edge_suspect" not in words


# --- "will <a named player> score 20+" ---------------------------------------------------------------------------

def test_will_a_named_player_score_20_plus_is_his_own_row_whether_he_qualifies_or_not(world) -> None:
    assert world.board().wait(NBA, "today", 60)
    route = routed(players=["Trae Young"], player_scan={**reading("pts", 20), "player": "Trae Young"})
    analyst = Analyst(route)
    answer = chat(world, analyst, "will trae young score 20+ tonight")
    clean(answer)
    assert [call["input"] for call in analyst.tool_calls()] == [{"stat": "pts", "threshold": 20, "side": "over",
                                                                  "day": "today", "player": "Trae Young"}]
    [card] = answer["cards"]
    assert card["names"] == ["Trae Young"] and card["missing_names"] == [] and card["mode"] == "scan"
    [row] = card["players"]
    assert row["name"] == "Trae Young" and row["team"] == "ATL" and row["qualifies"] is True and card["close"] == []
    assert answer["text"].startswith(f"Trae Young (ATL v {row['opponent']}): {row['projection']:g} projected, "
                                     f"{_pct(row['chance'])} to get 20+")
    assert scan_template(card).startswith(
        f"Trae Young for 20+ points tonight:\n1. Trae Young (ATL v {row['opponent']}) ")
    # The router named him only among the players: code still asks for his own row (_one_player)
    analyst = Analyst(routed(players=["Trae Young"], player_scan=reading("pts", 20)))
    chat(world, analyst, "will trae young score 20+ tonight")
    assert analyst.tool_calls()[0]["input"]["player"] == "Trae Young"
    # A player who isn't projected for it: his row anyway, marked; accents don't count ("Vucevic" is Vučević)
    analyst = Analyst(routed(player_scan={**reading("pts", 40), "player": "Nikola Vucevic"}))
    answer = chat(world, analyst, "can vucevic drop 40 tonight")
    clean(answer)
    [row] = answer["cards"][0]["players"]
    assert row["name"] == "Nikola Vučević" and row["qualifies"] is False
    assert "(not projected to get there)" in answer["text"] and "qualifies" not in answer["text"]
    # Someone not in tonight's games: said plainly, nobody else listed
    analyst = Analyst(routed(player_scan={**reading("pts", 20), "player": "Stephen Curry"}))
    answer = chat(world, analyst, "will steph score 20+ tonight")
    clean(answer)
    [card] = answer["cards"]
    assert card["players"] == [] and card["close"] == [] and card["missing_names"] == ["Stephen Curry"]
    assert answer["text"] == "Stephen Curry isn't in tonight's games."
    assert scan_template(card) == "I couldn't find Stephen Curry in tonight's games."  # P0 fix contract 1's words
    world.espn_only()


# --- an under at a milestone's number --------------------------------------------------------------------------

def test_an_under_at_a_milestones_number_never_takes_the_over_only_milestone_as_its_line(world) -> None:
    assert world.board().wait(NBA, "today", 60)
    analyst = Analyst(routed(player_scan=reading("reb", 6, side="under")))
    answer = chat(world, analyst, "who stays under 5.5 rebounds tonight")
    clean(answer)
    [card] = answer["cards"]
    assert card["ask"] == "under 5.5 rebounds" and card["side"] == "under"
    snapshot = world.snapshot_players(NBA)
    milestone = [row for row in card["players"] + card["close"]
                 if any(p.get("market") == "player_rebounds" and p.get("kind") == "milestone" and p.get("line") == 5.5
                        for p in snapshot[row["athlete_id"]]["props"])]
    assert milestone, "the world posts a 6+ rebounds milestone for these players"
    for row in milestone:
        # The 6+ milestone pays on 6 or more: never an under's line, its price never the under's, no edge from it
        assert (row.get("line") or {}).get("kind") != "milestone" and "edge" not in row
        assert not any("6+" in reason for reason in row["why"])
        assert row["chance"] == round(1 - engine.count_chance(row["projection"], 6, 1.0), 4) or row["model"] == "nba"
    facts = facts_of(analyst)
    by_name = {row["name"]: row for row in facts["players"] + facts["close"]}
    for row in milestone:
        assert (by_name[row["name"]].get("line") or {}).get("kind") != "milestone"
    for words in (answer["text"], scan_template(card), bot_chat.card_text(card)):
        assert "6+" not in words
    assert "to stay under 5.5" in answer["text"]
    body = world.client().get("/api/players/scan", params={"stat": "reb", "threshold": 6, "side": "under"}).json()
    assert [(r["athlete_id"], r.get("line")) for r in body["players"]] == \
        [(r["athlete_id"], r.get("line")) for r in card["players"]]


# --- a board with one failed roster ------------------------------------------------------------------------------

def test_a_board_whose_roster_failed_says_so_everywhere_and_recovers_on_refresh(world) -> None:
    world.fail_roster(EDMONTON)
    board = world.board()
    assert board.wait(NHL, "today", 30) and world.failed
    out = world.lab.player_scan(UID, NHL, "sog", 3)
    # Built, but partial: the scan says a game is missing a part (contract A: loading.failed), not "no games"
    assert out["loading"] == {"games_ready": 4, "games_total": 4, "failed": 1} and out["no_games"] is False
    assert not [row for row in out["players"] + out["close"] if row["team"] == "EDM"]
    body = world.client().get("/api/players/scan", params={"league": NHL, "stat": "sog", "threshold": 3}).json()
    assert body["loading"] == {"games_ready": 4, "games_total": 4, "failed": 1}
    # The tool, the analyst, code's template and the bot each say it in one line (built, so not "still loading")
    run = Toolbox(world.lab, UID, NY).run("players_scan", {"league": NHL, "stat": "sog", "threshold": 3})
    assert run.ok and run.content["loading"] == {"games_ready": 4, "games_total": 4, "failed": 1}
    assert "still missing data (loading failed)" in run.content["hint"] and "still loading" not in run.content["hint"]
    [card] = run.cards
    assert card["loading"]["failed"] == 1
    assert "1 game still missing data: ask again in a minute." in scan_template(card)
    assert "1 game still missing data: ask again in a minute." in bot_chat.card_text(card)
    analyst = Analyst(routed(player_scan=reading("sog", 3), leagues=[NHL]))
    answer = chat(world, analyst, "SOG 3+")
    clean(answer)
    assert answer["text"].endswith("1 game still missing data.")
    # Half an hour on, the board reads the game again: Edmonton's roster comes in, and the note goes
    world.clock.advance(31 * 60)
    board.snapshot(NHL)
    settle(board)
    later = world.lab.player_scan(UID, NHL, "sog", 3)
    assert "loading" not in later and [row for row in later["players"] + later["close"] if row["team"] == "EDM"]
    assert board.snapshot(NHL)["games_failed"] == 0
    run = Toolbox(world.lab, UID, NY).run("players_scan", {"league": NHL, "stat": "sog", "threshold": 3})
    assert "missing data" not in run.content["hint"] and "missing data" not in scan_template(run.cards[0])
    world.espn_only()


# --- a support decline -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("route, message, decline", [
    (routed(player_scan=reading("pts", 20)), "rough week, down a lot. who's scoring 20+ tonight",
     "Betting more to make up a bad week usually means you'll lose more. Your budget and a break are in Settings."),
    (routed(player_scan={"sort": "edge", "day": "today"}), "down 300 this week, need a big night. best props tonight?",
     "I can't suggest bets to make up a rough week. Settings has a weekly budget and a break, and 1-800-GAMBLER is "
     "free and confidential."),
    (routed(player_scan={"sort": "hot", "day": "today"}), "down bad this month, who's hot tonight",
     "I won't pick bets to make up a bad month. Take a break in Settings, or call 1-800-GAMBLER: it's free."),
])
def test_a_support_decline_is_never_overridden_by_a_scan(world, route, message, decline) -> None:
    assert world.board().wait(NBA, "today", 60)
    analyst = Analyst(route, first=decline)
    answer = chat(world, analyst, message)
    # The decline stands as written: no retry, no scan run by code, no card under it
    assert answer["kind"] == "answer" and answer["text"].startswith(decline), answer
    assert not answer["cards"] and not analyst.tool_calls() and len(analyst.chat()) == 1 and not analyst.corrections()


# --- the picks pause ---------------------------------------------------------------------------------------------

def test_the_picks_pause_after_a_warning_sign_covers_every_scan(world) -> None:
    assert world.board().wait(NBA, "today", 60)
    sign = chat(world, Analyst(routed(risk_flags=["chasing"])), "I need to win back what I lost")
    assert sign["kind"] == "support" and not sign["cards"]
    # The Lab: a plain scan, a player, hot and cold, and the best edges
    for args, options in (((NBA, "pts", 20), {}), ((NBA, "pts", 20), {"names": ["Trae Young"]}),
                          ((NBA, None, None), {"sort": "hot"}), ((NBA, None, None), {"sort": "cold"}),
                          ((NBA, None, None, None, "today", 10), {"sort": "edge"})):
        with pytest.raises(CallRejected, match="Picks are paused"):
            world.lab.player_scan(UID, *args, **options)
    # The route
    client = world.client()
    for params in ({"stat": "pts", "threshold": 20}, {"sort": "edge"}, {"sort": "hot"},
                   [("stat", "pts"), ("threshold", 20), ("player", "Trae Young")]):
        found = client.get("/api/players/scan", params=params)
        assert found.status_code == 400 and "Picks are paused" in found.json()["detail"], params
    # The tool
    run = Toolbox(world.lab, UID, NY).run("players_scan", {"stat": None, "sort": "edge"})
    assert not run.ok and not run.cards
    # The chat: no scan card for any P0 ask, and code never runs one (its template doesn't either)
    for route, message in ((routed(player_scan={"sort": "edge", "day": "today"}), "best player props tonight"),
                           (routed(player_scan={"sort": "hot", "day": "today"}), "who's hot"),
                           (routed(player_scan={**reading("pts", 20), "player": "Trae Young"}),
                            "will trae young score 20+"),
                           (routed(player_scan=reading("pts", 20)), SCREENSHOT)):
        analyst = Analyst(route, first=OLD_REFUSAL)
        answer = chat(world, analyst, message)
        assert not answer["cards"], (message, answer["cards"])
        assert "paused" in answer["text"].lower() or "1-800-GAMBLER" in answer["text"], answer["text"]


# --- an honest answer where no tool answers -----------------------------------------------------------------------

def test_a_parlay_of_player_props_gets_the_honest_can_t_not_a_template(world) -> None:
    """The persona review's "make me a 3 leg parlay of player props": the router reads a build, but ticket_build prices
    game picks only, so no tool builds it. The honest one-line "can't" reaches the person (before the integrator's
    fix it was sent back as a refusal and ended as "I couldn't answer that reliably")."""
    honest = ("I can't build a parlay of player props: the Builder's tickets take game picks (moneyline, spread, "
              "total). Ask for the best player props tonight and I'll rank them.")
    analyst = Analyst(routed(intent="build", leagues=[NBA], legs=3), first=honest)
    answer = chat(world, analyst, "make me a 3 leg parlay of player props")
    assert answer["kind"] == "answer" and answer["text"].startswith(honest), answer
    assert len(analyst.chat()) == 1 and not analyst.corrections()
    # A build of game picks is still ticket_build's: the old refusal is sent back
    analyst = Analyst(routed(intent="build", leagues=[NBA], legs=3), first=OLD_REFUSAL)
    chat(world, analyst, "make me a 3 leg parlay")
    assert analyst.corrections() and "ticket_build" in analyst.corrections()[0]
