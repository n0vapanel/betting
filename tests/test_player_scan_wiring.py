"""The player scan's wiring, AI and bot (v3): Lab.player_scan and GET /api/players/scan, the players_scan tool (its facts
carry every number the reply may quote, its card the ranked rows), the router's player_scan field and the bettor lingo,
the analyst persona, the refusal and lecture guard (one retry shared with the numbers check, then code's template), the
template and the bot's numbered list with "View player".

The owner's screenshot: Ask AI answered "give me players that are projected to score at least 20 points today" with
"I can't pull up a list ... the props tool only works one player at a time" and "a projection of 20 points doesn't mean
he'll score 20 ... close to a coin flip". Here a fake Claude runs that message through the real assistant: it routes
to player_scan {pts, 20}, calls players_scan and answers with the ranked players, never a refusal or a lecture; a
fake Claude whose first answer is the old refusal is caught, retried and ends with the tool's numbers.

The world is tests/test_builder_wiring.py's (a real Store, slate, games service and Lab with saved NBA and EPL boards)
over the real engine (engine/scan.py: its stat tables, make_ask and scan). The player board (data/player_board.py,
which reads ESPN) stands in as a small fake with its scan(league, stat, threshold, side, day, n) answering canned rows,
or the real engine's over a hand-built snapshot. No network."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import pytest
import requests

from telegram_betting_bot_simple.ai import assistant as assistant_module
from telegram_betting_bot_simple.ai.assistant import (
    GUARD_PROBLEMS,
    SYSTEM_PROMPT,
    Assistant,
    Route,
    parse_route,
    scan_args,
)
from telegram_betting_bot_simple.ai.explain import (
    banned_phrases,
    problems,
    refusal_problems,
    scan_template,
    source_numbers,
)
from telegram_betting_bot_simple.ai.tools import (
    PICK_TOOLS,
    SCAN_SLIP_NOTE,
    SCAN_STAT_KEYS,
    TOOLS,
    Toolbox,
)
from telegram_betting_bot_simple.dashboard import lab as lab_module
from telegram_betting_bot_simple.data.odds_service import configure_books
from telegram_betting_bot_simple.engine import scan as scan_engine
from tests.test_builder_wiring import NBA, NOW, NY, PLAYER_A, make_world

SCREENSHOT = "give me players that are projected to score at least 20 points today"
OLD_REFUSAL = ("I can't pull up a list of every player projected for 20+ points today. The props tool only works one "
               "player at a time, so name a player and I'll check his line. A projection of 20 points doesn't mean "
               "he'll score 20. Most points overs are close to a coin flip.")
UID = "web:" + PLAYER_A["X-Player"]
WNBA, NFL, NHL, MLB = "basketball_wnba", "americanfootball_nfl", "icehockey_nhl", "baseball_mlb"
LECTURE_AND_REFUSAL = ("I can't", "I'm not able", "I'm unable", "isn't possible", "only works one", "can't pull",
                       "can't list", "doesn't mean he'll", "coin flip", "no guarantee", "nothing is guaranteed",
                       "always wins", "chance of each one missing", "you'll lose more")


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


# --- the scan's board: canned engine rows, or a hand-built snapshot through the real engine ----------------------------

AS_OF = "2026-10-26T21:45:00Z"


def row(rank: int, name: str, team: str, opponent: str, projection: float, chance: float, l10: List[int],
        season: List[int], l5: Optional[List[int]] = None, line: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None, model: str = "nba", why: Optional[List[str]] = None, athlete: str = "1000",
        edge: Optional[float] = None, threshold: int = 20) -> Dict[str, Any]:
    """One row as engine/scan.py's scan() makes it."""
    out = {"rank": rank, "athlete_id": athlete, "name": name, "team": team, "opponent": opponent, "home": True,
           "position": "G", "game": f"{opponent} @ {team}", "event_id": "401700001", "start": "2026-10-26T23:30Z",
           "status": status, "status_tag": scan_engine.status_tag(status), "threshold": threshold,
           "projection": projection, "chance": chance, "chance_source": model, "model": model,
           "model_label": scan_engine.MODEL_LABELS[model],
           "hits": {"l5": l5 or [min(5, l10[0]), 5], "l10": l10, "season": season},
           "avg": {"l5": round(projection + 1.1, 1), "l10": round(projection - 0.4, 1),
                   "season": round(projection - 1.3, 1)},
           "why": why or [f"{l10[0]} of last 10 at {threshold}+"]}
    if model == "nba":
        out["minutes"] = 35.5
    if line is not None:
        out["line"] = line
    if edge is not None:
        out["edge"] = edge
    return out


DK = {"kind": "over_under", "line": 19.5, "label": "Points", "market": "player_points",
      "over": {"decimal": 1.3, "american": "-334"}, "under": {"decimal": 3.4, "american": "+240"}, "yes": None,
      "book": "DraftKings", "as_of": "2026-10-26T21:40:00Z", "stale": False, "matches": True}
NBA_PTS = {
    "players": [
        row(1, "Luka Dončić", "DAL", "PHX", 31.2, 0.8214, [9, 10], [41, 45], line=DK, edge=0.0678,
            athlete="3945274", why=["9 of last 10 at 20+", "averaging 32.3 points over the last 5"]),
        row(2, "Shai Gilgeous-Alexander", "OKC", "DEN", 29.6, 0.7932, [10, 10], [44, 46], athlete="4278073"),
        row(3, "Jayson Tatum", "BOS", "NYK", 26.4, 0.7105, [8, 10], [36, 47], status="Questionable", athlete="4065648",
            why=["8 of last 10 at 20+", "Brown out"]),
        row(4, "Anthony Edwards", "MIN", "LAL", 25.1, 0.6648, [7, 10], [33, 44], athlete="4594268"),
    ],
    "close": [row(5, "Paolo Banchero", "ORL", "MIA", 18.9, 0.4417, [5, 10], [21, 40], athlete="4432573")],
}


class FakeBoard:
    """One (league, day): the engine's rows by stat (``by_stat``), or a snapshot the real engine scans; how many of
    its games are loaded."""

    def __init__(self, league: str, day: str, by_stat: Optional[Mapping[str, Mapping[str, Any]]] = None,
                 games_total: int = 8, games_ready: Optional[int] = None, snapshot: Optional[Dict[str, Any]] = None,
                 reading: bool = False) -> None:
        self.league, self.day, self.by_stat, self.snapshot = league, day, by_stat or {}, snapshot
        self.games_total = games_total
        self.games_ready = games_total if games_ready is None else games_ready
        self.reading = reading  # the day's games are still being read


class FakeBoards:
    """data/player_board.py's PlayerBoard as the Lab uses it: scan(league, stat, threshold, side, day, n) is the
    engine's result for make_ask(...) (the real make_ask, and the real scan() over a snapshot) plus {league, day,
    title, no_games}. Each league and day asked for is recorded, with its ask."""

    def __init__(self, boards: Optional[Dict[tuple, FakeBoard]] = None) -> None:
        self.boards = boards or {}
        self.asked: List[tuple] = []
        self.asks: List[Dict[str, Any]] = []

    def scan(self, league: str, stat: Any, threshold: Any = None, side: str = "over", day: str = "today",
             n: int = 10, wait: Optional[float] = None) -> Dict[str, Any]:
        ask = scan_engine.make_ask(league, stat, threshold, side, day)
        self.asked.append((league, day))
        self.asks.append({**ask, "n": n})
        board = self.boards.get((league, day)) or FakeBoard(league, day, games_total=0)
        if board.snapshot is not None:
            result = scan_engine.scan(board.snapshot, ask, n)
        else:
            found = board.by_stat.get(ask["stat"]) or {}
            result = {"ask": ask, "players": [dict(r) for r in found.get("players") or []][:n],
                      "close": [dict(r) for r in found.get("close") or []][:5], "more": 0, "as_of": AS_OF,
                      "games": board.games_total, "games_started": 0}
            if board.games_ready < board.games_total:
                result["loading"] = {"games_ready": board.games_ready, "games_total": board.games_total}
        if board.reading:
            result["loading"] = {"games_ready": 0, "games_total": 0}
        result.update(league=league, day=day, title=scan_engine.TITLES.get(league, league),
                      no_games=not board.reading and board.games_total == 0)
        return result


def nba_boards(**extra: FakeBoard) -> FakeBoards:
    boards = {(NBA, "today"): FakeBoard(NBA, "today", {"pts": NBA_PTS}, games_total=10)}
    boards.update({tuple(key.split("|")): value for key, value in extra.items()})
    return FakeBoards(boards)


def log(points: List[float], season: str = "2027") -> List[Dict[str, Any]]:
    """Game-log rows, newest first, as PlayerData.gamelog gives them."""
    return [{"season": season, "stats": {"pts": p, "reb": 6.0, "ast": 5.0, "fg3m": 2.0, "stl": 1.0, "blk": 0.0}}
            for p in points]


def nba_player(athlete: str, name: str, team: str, opponent: str, means: Dict[str, float], points: List[float],
               status: Optional[str] = None, props: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {"athlete_id": athlete, "name": name, "team": team, "opponent": opponent, "home": True, "position": "G",
            "event_id": "401700001", "status": status, "projection": {"means": means, "minutes": 35.0},
            "rows": log(points), "props": props or []}


def nba_snapshot() -> Dict[str, Any]:
    """Tonight's NBA board as data/player_board.py's snapshot gives it, for the real engine to rank: the NBA model's
    means and minutes, game logs and DraftKings' props (a 20+ milestone for one)."""
    means = lambda pts: {"pts": pts, "reb": 7.0, "ast": 6.0, "fg3m": 2.5, "stl": 1.1, "blk": 0.4, "tov": 3.0,
                         "pra": pts + 13.0}
    milestone = [{"market": "player_points", "kind": "milestone", "line": 19.5, "label": "20+ Points",
                  "over_price": 1.12},
                 {"market": "player_points", "kind": "over_under", "line": 29.5, "label": "Points",
                  "over_price": 1.87, "under_price": 1.95}]
    players = [
        nba_player("3945274", "Luka Dončić", "DAL", "PHX", means(31.2),
                   [34, 28, 41, 25, 33, 18, 30, 27, 36, 29, 31, 26, 22, 35], props=milestone),
        nba_player("4278073", "Shai Gilgeous-Alexander", "OKC", "DEN", means(29.6),
                   [31, 27, 33, 24, 30, 29, 26, 35, 28, 32]),
        nba_player("4065648", "Jayson Tatum", "BOS", "NYK", means(26.4),
                   [24, 31, 19, 27, 22, 28, 25, 30, 17, 26], status="Questionable"),
        nba_player("4594268", "Anthony Edwards", "MIN", "LAL", means(24.9),
                   [22, 27, 25, 18, 29, 21, 26, 24, 19, 28]),
        nba_player("4432573", "Paolo Banchero", "ORL", "MIA", means(18.6),
                   [17, 21, 15, 19, 23, 14, 18, 20, 16, 22]),
        nba_player("1966", "LeBron James", "LAL", "MIN", means(23.0), [25, 22, 27, 20, 24, 21, 26, 23, 19, 28],
                   status="Out"),  # ruled out: never ranked
    ]
    game = {"event_id": "401700001", "label": "PHX @ DAL", "start": "2026-10-26T23:30Z", "outs": {"LAL": ["LeBron James"]},
            "props": {"book": "DraftKings", "as_of": "2026-10-26T21:40:00Z", "stale": False}}
    return {"league": NBA, "day": "2026-10-26", "as_of": AS_OF, "games": {"401700001": game}, "players": players,
            "games_ready": 1, "games_total": 1, "games_started": 0}


def real_boards() -> FakeBoards:
    return FakeBoards({(NBA, "today"): FakeBoard(NBA, "today", snapshot=nba_snapshot(), games_total=1)})


@pytest.fixture()
def world(tmp_path):
    w = make_world(tmp_path)
    w.client()  # PLAYER_A confirms their age
    w.boards = nba_boards()
    w.lab.player_board = w.boards
    return w


# --- Lab.player_scan and GET /api/players/scan ------------------------------------------------------------------------

def test_the_lab_scan_ranks_nba_players_for_20_points_with_their_history_projection_chance_and_line(world) -> None:
    out = world.lab.player_scan(UID, None, "points", 20)
    assert out["ask"] == "20+ points" and out["stat"] == "pts" and out["threshold"] == 20 and out["side"] == "over"
    assert out["day"] == "today" and out["leagues"] == [NBA] and world.boards.asked == [(NBA, "today")]
    assert world.boards.asks[0]["threshold"] == 20 and world.boards.asks[0]["n"] == 10
    names = [r["name"] for r in out["players"]]
    assert names == ["Luka Dončić", "Shai Gilgeous-Alexander", "Jayson Tatum", "Anthony Edwards"]
    first = out["players"][0]
    assert first["rank"] == 1 and first["league"] == NBA and first["addable"] is False  # the slip takes game picks
    assert first["hits"] == {"l5": [5, 5], "l10": [9, 10], "season": [41, 45]} and first["line"]["line"] == 19.5
    assert [r["name"] for r in out["close"]] == ["Paolo Banchero"] and out["close"][0]["addable"] is False
    assert out["as_of"] == AS_OF and out["games"] == 10 and "loading" not in out and "leagues_without_games" not in out


def test_the_real_engine_s_rows_come_through_the_lab_ranked_by_chance(world) -> None:
    world.lab.player_board = real_boards()
    out = world.lab.player_scan(UID, None, "pts", 20)
    rows = out["players"] + out["close"]
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert [r["chance"] for r in out["players"]] == sorted((r["chance"] for r in out["players"]), reverse=True)
    assert out["players"][0]["name"] == "Luka Dončić" and "LeBron James" not in [r["name"] for r in rows]
    luka = out["players"][0]
    # The NBA model's chance is the player card's (calls.stat_probabilities at 19.5 on the model's mean)
    from telegram_betting_bot_simple.engine.calls import stat_probabilities
    assert luka["chance"] == pytest.approx(round(stat_probabilities("pts", 31.2, 19.5, 35.0)[0], 4))
    assert luka["model"] == "nba" and luka["projection"] == 31.2 and luka["hits"]["l10"] == [9, 10]
    assert luka["line"]["kind"] == "milestone" and luka["line"]["matches"] and "edge" in luka
    assert all(r["addable"] is False and r["league"] == NBA for r in rows)
    tatum = next(r for r in rows if r["name"] == "Jayson Tatum")
    assert tatum["status_tag"] == "Q"


def test_a_phrase_a_half_line_tomorrow_and_n_are_read_as_the_scan_means_them(world) -> None:
    boards = nba_boards(**{f"{NBA}|tomorrow": FakeBoard(NBA, "tomorrow", {"reb": NBA_PTS})})
    world.lab.player_board = boards
    out = world.lab.player_scan(UID, NBA, "Boards", 9.5, "under", "tomorrow", 99)
    ask = boards.asks[-1]
    assert (ask["stat"], ask["threshold"], ask["side"], ask["day"], ask["n"]) == ("reb", 10, "under", "tomorrow", 25)
    assert out["ask"] == "under 9.5 rebounds" and out["day"] == "tomorrow" and out["stat_label"] == "rebounds"
    # "tonight" is today; 19.5 over is 20; no number is each player's posted line
    world.lab.player_scan(UID, NBA, "PTS", 19.5, "over", "tonight", 3)
    assert (boards.asks[-1]["threshold"], boards.asks[-1]["day"], boards.asks[-1]["n"]) == (20, "today", 3)
    lined = world.lab.player_scan(UID, NBA, "pra", None)
    assert boards.asks[-1]["threshold"] is None and lined["ask"] == "PRA overs (posted lines)"
    attd = world.lab.player_scan(UID, NFL, "ATTD", 5)
    assert attd["ask"] == "anytime TD" and attd["threshold"] == 1  # a yes/no ask has its own number
    assert attd["leagues"] == [] and attd["leagues_without_games"] == ["NFL"]


@pytest.mark.parametrize("words, leagues", [
    ("points", [NBA, WNBA, NHL]), ("pts", [NBA, WNBA, NHL]), ("rebounds", [NBA, WNBA]), ("boards", [NBA, WNBA]),
    ("assists", [NBA, WNBA, NHL]), ("threes", [NBA, WNBA]), ("PRA", [NBA, WNBA]), ("yards", [NFL]),
    ("TD", [NFL]), ("anytime TD scorers", [NFL]), ("SOG", [NHL]), ("shots", [NHL]), ("goals", [NHL]),
    ("Ks", [MLB]), ("strikeouts", [MLB]), ("dingers", [MLB]),
])
def test_without_a_league_the_stat_picks_the_leagues_nba_first_for_points_nfl_for_tds_nhl_for_shots(
        world, words, leagues) -> None:
    boards = FakeBoards()  # no games anywhere: every league the stat names is looked at, in order (at most 3)
    world.lab.player_board = boards
    out = world.lab.player_scan(UID, None, words, 3)
    assert [league for league, _ in boards.asked] == leagues[:3], words
    assert out["leagues"] == [] and out["players"] == []
    assert out["leagues_without_games"] == [scan_engine.TITLES.get(league, league) for league in leagues[:3]]


def test_without_a_league_the_first_league_with_games_is_scanned_and_the_others_are_named(world) -> None:
    # The NBA has no games tomorrow; the WNBA does: points go to the WNBA, and the NBA is listed
    wnba = FakeBoard(WNBA, "tomorrow", {"pts": {"players": [row(1, "A'ja Wilson", "LV", "SEA", 24.8, 0.71, [8, 10],
                                                               [30, 38], model="form")]}})
    boards = nba_boards(**{f"{WNBA}|tomorrow": wnba})
    world.lab.player_board = boards
    out = world.lab.player_scan(UID, None, "points", 20, day="tomorrow")
    assert boards.asked == [(NBA, "tomorrow"), (WNBA, "tomorrow")]
    assert out["leagues"] == [WNBA] and out["leagues_without_games"] == ["NBA"]
    assert out["players"][0]["model"] == "form" and out["players"][0]["league"] == WNBA
    # Games but nobody reaches it: that league, empty (never the next league)
    nobody = world.lab.player_scan(UID, None, "rebounds", 10, day="tomorrow")
    assert nobody["leagues"] == [WNBA] and nobody["players"] == [] and nobody["leagues_without_games"] == ["NBA"]


def test_a_partial_board_answers_at_once_with_what_s_ready_and_loading(world) -> None:
    world.lab.player_board = FakeBoards({(NBA, "today"): FakeBoard(NBA, "today", {"pts": NBA_PTS}, games_total=10,
                                                                   games_ready=4)})
    out = world.lab.player_scan(UID, NBA, "pts", 20)
    assert out["loading"] == {"games_ready": 4, "games_total": 10} and out["players"]
    # The day's games still being read is not "no games"
    world.lab.player_board = FakeBoards({(NBA, "today"): FakeBoard(NBA, "today", games_total=0, reading=True)})
    waiting = world.lab.player_scan(UID, None, "pts", 20)
    assert waiting["leagues"] == [NBA] and waiting["loading"] == {"games_ready": 0, "games_total": 0}
    assert "leagues_without_games" not in waiting


def test_bad_asks_are_refused_in_plain_words(world) -> None:
    from telegram_betting_bot_simple.dashboard.service import CallRejected
    for args, words in (((None, "vibes", 20), "doesn't know 'vibes'"), ((NBA, "sog", 3), "doesn't know 'sog' for NBA"),
                        (("cricket_odi", "pts", 20), "covers the NBA, WNBA, NFL, NHL, MLB and soccer"),
                        ((NBA, "pts", 20, "sideways"), "over or under"),
                        ((NBA, "pts", 20, "over", "next week"), "today and tomorrow"),
                        ((NBA, "pts", -3), "above 0"), ((NBA, "pts", 1000), "at most 999"),
                        ((NBA, "pts", "lots"), "must be a number"), ((NBA, "", 20), "which stat")):
        with pytest.raises(CallRejected) as raised:
            world.lab.player_scan(UID, *args)
        assert words in str(raised.value), (args, str(raised.value))


def test_the_lab_makes_one_shared_player_board_on_the_lab_s_clock(world, monkeypatch) -> None:
    from telegram_betting_bot_simple.data import player_board
    world.lab.player_board = None
    world.lab._players = SimpleNamespace()  # the world's games service has no ESPN client
    made = world.lab.player_board
    assert isinstance(made, player_board.PlayerBoard) and world.service.player_board is made
    assert made._now() == NOW  # the Lab's clock
    assert world.lab.player_board is made and player_board.board_for(world.service) is made
    made.close()
    # A server without the scan's engine says so instead of failing
    monkeypatch.setattr(lab_module, "_engine_scan", lambda: (_ for _ in ()).throw(ImportError("no engine")))
    from telegram_betting_bot_simple.dashboard.service import CallRejected
    with pytest.raises(CallRejected, match="isn't available on this server"):
        world.lab.player_scan(UID, NBA, "pts", 20)


def test_the_route_scans_behind_the_player_pages_gates(world) -> None:
    client = world.client()
    response = client.get("/api/players/scan", params={"stat": "points", "threshold": 20})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ask"] == "20+ points" and [r["rank"] for r in body["players"]] == [1, 2, 3, 4]
    assert body["players"][0]["hits"]["l10"] == [9, 10] and body["players"][0]["addable"] is False
    lined = client.get("/api/players/scan", params={"league": NBA, "stat": "pra", "side": "over", "day": "today",
                                                    "n": 5})
    assert lined.status_code == 200 and lined.json()["threshold"] is None
    for bad in ({"stat": "pts", "threshold": 0}, {"stat": "pts", "day": "friday"}, {"stat": "pts", "side": "both"},
                {"stat": "pts", "n": 26}, {"stat": "<script>"}, {"stat": "pts", "league": "NBA!"}, {}):
        assert client.get("/api/players/scan", params=bad).status_code == 422, bad
    assert client.get("/api/players/scan", params={"stat": "vibes"}).status_code == 400
    # Age not confirmed: the same gate as the player pages
    fresh = world.client({"X-Player": "browser-cccc3333"}, confirm_age=False)
    refused = fresh.get("/api/players/scan", params={"stat": "pts", "threshold": 20})
    assert refused.status_code == 400 and "18+" in refused.json()["detail"]
    # On a break: closed too
    assert client.post("/api/me/cooloff", json={"hours": 24}).status_code == 200
    assert client.get("/api/players/scan", params={"stat": "pts"}).status_code == 400


# --- the players_scan tool ---------------------------------------------------------------------------------------------

def test_players_scan_is_a_loose_tool_whose_facts_carry_every_number_and_whose_card_has_the_rows(world) -> None:
    tool = next(t for t in TOOLS if t["name"] == "players_scan")
    assert "strict" not in tool and tool["input_schema"]["additionalProperties"] is False
    assert set(tool["input_schema"]["required"]) == {"stat", "threshold"}
    assert tool["input_schema"]["properties"]["threshold"]["type"] == ["number", "null"]
    assert "players_scan" in PICK_TOOLS  # off while picks are paused
    box = Toolbox(world.lab, UID, NY, leagues={NBA})
    done = box.run("players_scan", {"stat": "pts", "threshold": 20})
    assert done.ok, done.content
    facts = done.content
    assert facts["ask"] == "20+ points" and facts["threshold"] == 20 and facts["leagues"] == [NBA]
    first = facts["players"][0]
    assert first["name"] == "Luka Dončić" and first["projection"] == 31.2 and first["chance"] == 0.8214
    assert first["hits"] == {"l5": "5/5", "l10": "9/10", "season": "41/45"}
    assert first["history"] == ["5 of last 5", "9 of last 10", "41 of 45 this season"]
    assert first["line"]["line"] == 19.5 and first["edge"] == 0.0678 and first["avg"]["l5"] == 32.3
    assert "Lead with the players" in facts["hint"] and facts["source"].startswith("The player scan")
    assert [r["name"] for r in facts["close"]] == ["Paolo Banchero"]
    assert done.summary.startswith("players_scan(20+ points, basketball_nba, today): Luka Dončić 31.2 (0.8214)")
    card = done.cards[0]
    assert card["type"] == "player_scan" and card["title"] == "Projected for 20+ points · NBA · tonight"
    assert [r["rank"] for r in card["players"]] == [1, 2, 3, 4] and card["slip_note"] == SCAN_SLIP_NOTE
    assert all(r["addable"] is False for r in card["players"] + card["close"])
    # A phrase, "tonight", an empty league; a bad input comes back as a readable error
    assert box.run("players_scan", {"stat": "points", "threshold": 19.5, "day": "Tonight", "league": ""}).ok
    assert box.run("players_scan", {"stat": "pts", "threshold": 20, "n": 99}).content["error"] == "invalid_input"
    assert box.run("players_scan", {"stat": "vibes", "threshold": 20}).content["error"] == "rejected"
    # Paused after a warning sign: no scan
    paused = Toolbox(world.lab, UID, NY, paused=True).run("players_scan", {"stat": "pts", "threshold": 20})
    assert paused.content["error"] == "picks_paused"
    # It never spends Odds API quota: the scan loads no league's odds
    assert not box.extra and world.queued == []


def test_the_post_check_accepts_the_scan_s_numbers_and_still_rejects_invented_ones_and_certainty(world) -> None:
    facts = Toolbox(world.lab, UID, NY).run("players_scan", {"stat": "pts", "threshold": 20}).content
    sources = source_numbers(facts, SCREENSHOT)
    for sentence in ("Luka Dončić (DAL v PHX): 31.2 projected, 82% to get 20+, 9 of last 10, 41 of 45 this season.",
                     "Luka is 9/10 L10 and 41/45 on the season, averaging 32.3 in his last 5.",
                     "The line is 19.5 at 1.30 (-334) with DraftKings, an edge of +6.8%.",
                     "Shai Gilgeous-Alexander 29.6 projected, 79% to get there, 10 of last 10.",
                     "Strongest of tonight's 20+ points options: Luka, then SGA at 79%."):
        assert problems(sentence, sources) == [], sentence
        assert refusal_problems(sentence, "lookup", True, asked=SCREENSHOT) == [], sentence
    assert problems("Luka is 95% to get 20+.", sources)  # no tool gave 95%
    assert problems("Luka projects for 34.7 tonight.", sources)
    assert problems("Luka is 6/10 L10.", sources)  # his record is 9/10
    assert banned_phrases("Luka over 19.5 is a lock.") == ["lock"]
    assert banned_phrases("Luka 20+ is guaranteed.") == ["guaranteed"]


# --- the refusal and lecture guard -------------------------------------------------------------------------------------

def test_the_guard_catches_the_old_refusal_and_its_lecture() -> None:
    found = refusal_problems(OLD_REFUSAL, "lookup", True, asked=SCREENSHOT)
    assert len(found) == 2 and found[0].startswith("a refusal") and found[1].startswith("a lecture")
    for phrase in ('"I can\'t"', '"only works one"', '"can\'t pull"'):
        assert phrase in found[0]
    assert '"doesn\'t mean he\'ll"' in found[1] and '"coin flip"' in found[1]
    assert all(problem.startswith(GUARD_PROBLEMS) for problem in found)


@pytest.mark.parametrize("text, name", [
    ("I can't list every player.", "I can't"), ("I cannot do that from here.", "I can't"),
    ("I’m not able to pull that up.", "I'm not able"), ("I am unable to scan the league.", "I'm unable"),
    ("A list like that isn't possible.", "isn't possible"), ("The tool only works for one player.", "only works one"),
    ("Sadly the app can't pull lists.", "can't pull"), ("The tools cannot list players.", "can't list"),
])
def test_each_refusal_phrase_is_caught_while_a_tool_can_answer(text, name) -> None:
    found = refusal_problems(text, "lookup", True)
    assert found and f'"{name}"' in found[0], (text, found)
    assert refusal_problems(text, "lookup", False) == []  # picks paused: nothing to call, and saying so is right


@pytest.mark.parametrize("text, name", [
    ("A 31.2 projection doesn't mean he'll score 20.", "doesn't mean he'll"),
    ("It does not mean she will get there.", "doesn't mean he'll"), ("It's close to a coin-flip.", "coin flip"),
    ("There's no guarantee here.", "no guarantee"), ("Nothing is guaranteed in sports.", "nothing is guaranteed"),
    ("Remember the house always wins.", "the house always wins"),
    ("The bookmakers always win over time.", "the bookmaker always wins"),
    ("Think about the chance of each one missing.", "chance of each one missing"),
    ("With more legs you'll lose more often.", "you'll lose more"),
])
def test_each_lecture_phrase_is_caught_unless_they_asked_about_risk(text, name) -> None:
    found = refusal_problems(text, "build", True)
    assert found and f'"{name}"' in found[-1], (text, found)
    assert refusal_problems(text, "build", True, asked="what's the risk on this one?") == []
    assert refusal_problems(text, "build", True, asked="how likely is it to miss?") == []


def test_the_guard_leaves_alone_other_routes_quotes_rules_labels_and_the_user_s_own_words() -> None:
    assert refusal_problems(OLD_REFUSAL, "explain", True) == []  # "how do parlays work" may say "coin flip"
    assert refusal_problems(OLD_REFUSAL, "chat", True) == []
    assert refusal_problems(OLD_REFUSAL, "support", True) == []
    assert refusal_problems(OLD_REFUSAL, Route(intent="help"), True) == []
    assert refusal_problems(OLD_REFUSAL, {"intent": "lookup"}, True)  # a mapping or a Route work as well as a str
    assert refusal_problems(OLD_REFUSAL, Route(intent="analyse"), True)
    # The app's rules said with "I can't": only the user adds to the slip, and no bigger stake
    for rule in ("I can't add it to your slip for you: tap Add to slip.", "I can't suggest a bigger stake.",
                 "I can't place bets."):
        assert refusal_problems(rule, "lookup", True) == [], rule
    # Quoted words, the user's own words and a tool's chance label aren't the reply's
    assert refusal_problems('You asked "can\'t you list them all?" Here they are.', "lookup", True) == []
    assert refusal_problems("It isn't a coin flip: 82% to get there.", "lookup", True,
                            asked="is luka 20+ a coin flip?") == []
    assert refusal_problems("Nuggets at 52%, a coin flip.", "lookup", True,
                            allowed=json.dumps([{"picks": [{"label": "Coin flip"}]}])) == []


# --- the router, the lingo and the persona -----------------------------------------------------------------------------

def routed(**fields: Any) -> Dict[str, Any]:
    """The router's JSON with every field, as Claude Haiku returns it."""
    base = {"intent": "lookup", "leagues": [], "stake": None, "target": None, "max_legs": None, "tier": "none",
            "target_odds": None, "target_chance": None, "legs": None, "top_picks": False, "days": None,
            "min_odds": None, "legs_per_day": None, "players": [], "player_scan": None, "risk_flags": []}
    base.update(fields)
    return base


def scan(stat_key: str, threshold: Optional[float], day: str = "today", side: str = "over") -> Dict[str, Any]:
    return {"stat": stat_key, "threshold": threshold, "side": side, "day": day}


LINGO = [  # (message, the router's reading as its prompt's examples teach it, the leagues it names)
    (SCREENSHOT, scan("pts", 20), []),
    ("who's going for 10+ boards tonight", scan("reb", 10), []),
    ("PRA overs tonight", scan("pra", None), []),
    ("10+ boards", scan("reb", 10), []),
    ("who's hitting 3+ threes", scan("fg3m", 3), []),
    ("anytime TD scorers this week", scan("anytime_td", 1, "week"), [NFL]),
    ("anytime TD", scan("anytime_td", 1), [NFL]),
    ("SOG 3+", scan("sog", 3), [NHL]),
    ("Ks tonight", scan("k", None), [MLB]),
]


def test_the_router_schema_and_prompt_carry_player_scan_and_the_lingo() -> None:
    schema = assistant_module.ROUTER_SCHEMA
    assert "player_scan" in schema["required"]
    shape = schema["properties"]["player_scan"]["anyOf"][0]
    assert set(shape["required"]) == {"stat", "threshold", "side", "day"} and shape["additionalProperties"] is False
    assert tuple(shape["properties"]["stat"]["enum"]) == SCAN_STAT_KEYS
    assert shape["properties"]["day"]["enum"] == ["today", "tomorrow", "week"]
    prompt = assistant_module.ROUTER_PROMPT
    for term in ("ML (moneyline)", "spread/ATS", "o/u", "alt line", "ladder", "juice/vig", "plus money", "chalk", "dog",
                 "pick'em", "SGP", "parlay", "teaser", "round robin", "1H/1Q", "team total", "PRA", "P+R", "3PM", "SOG",
                 "Ks", "TB", "H+R+RBI", "anytime TD", "first basket", "double-double", "triple-double", "L5/L10/L20",
                 "hit rate", "hot/cold", "fade", "tail", "units", "hedge", "cash out", "lock, hammer, nuke mean their "
                 "strongest picks"):
        assert term in prompt, term
    for message in ("who's going for 10+ boards tonight", "who's hitting 3+ threes", "PRA overs tonight",
                    "anytime TD scorers this week", "SOG 3+", "Ks tonight", "give me 3 locks"):
        assert f'"{message}"' in prompt, message
    assert '"players projected to score at least 20 points today" is lookup, player_scan {pts, 20, over, today}' \
        in prompt
    assert '"give me 3 locks" is lookup with top_picks true and player_scan null' in prompt


@pytest.mark.parametrize("message, reading, leagues", LINGO)
def test_the_lingo_routes_to_the_scan_with_the_right_ask(message, reading, leagues) -> None:
    route = parse_route(routed(player_scan=reading, leagues=leagues, intent="chat"))
    assert route.intent == "lookup"  # a list of players is a look-up, whatever the router called it
    expected = {key: reading[key] for key in ("stat", "threshold", "side")}
    assert {key: route.player_scan[key] for key in expected} == expected
    assert route.player_scan["day"] == "today"  # "this week" is scanned for today ...
    assert route.player_scan.get("wide") is (True if reading["day"] == "week" else None)  # ... and the reply says so
    assert route.fields()["player_scan"]["stat"] == reading["stat"]
    # The league they named goes to the scan; none named: the leagues with games, never their favourites
    Assistant._defaults(None, route, {"favourite_sports": ["soccer_epl"]})
    assert route.player_scan.get("league") == (leagues[0] if leagues else None)
    args = scan_args(route)
    assert args["stat"] == reading["stat"] and ("league" in args) is bool(leagues)


def test_the_router_s_scan_is_checked_field_by_field() -> None:
    assert parse_route(routed(player_scan=scan("vibes", 20))).player_scan is None
    assert parse_route(routed(player_scan="pts")).player_scan is None
    over = parse_route(routed(player_scan=scan("pts", 19.5))).player_scan
    assert over["threshold"] == 20  # "over 19.5" reaches 20
    under = parse_route(routed(player_scan=scan("reb", 9.5, side="under"))).player_scan
    assert (under["threshold"], under["side"]) == (10, "under")
    for bad in (0, -4, 10_000, True, "lots", float("nan")):
        assert parse_route(routed(player_scan=scan("pts", bad))).player_scan["threshold"] is None, bad
    odd = parse_route(routed(player_scan={"stat": "PTS", "threshold": 20, "side": "sideways", "day": "someday"}))
    assert odd.player_scan == {"stat": "pts", "threshold": 20, "side": "over", "day": "today"}
    # A scan kept while the assistant asks something else, then merged back
    route = Route(intent="lookup", routed=True)
    route.merge({"intent": "lookup", "player_scan": scan("reb", 10)}, "tonight")
    assert route.player_scan["stat"] == "reb"


def test_the_persona_speaks_bettor_uses_the_tools_and_adds_no_disclaimers() -> None:
    prompt = SYSTEM_PROMPT
    for kept in ("How the app works", "Never pass a price or a probability to a tool", "Code checks every number",
                 "1-800-GAMBLER / 0808 8020 133", "suggest no bets", "never suggest a bigger stake",
                 "Never claim certainty (lock, guaranteed, sure thing, risk-free, can't lose)",
                 "only the user adds to the slip", "No lectures or moralising"):
        assert kept in prompt, kept
    for added in ("Speak bettor", "never correct their", '"Lock", "hammer" and "nuke" mean their strongest picks',
                  '"strongest" or "most likely"', "Use the tools before saying no", "check every tool",
                  "a list of players goes", "through players_scan", "a list of picks through brain_top_picks",
                  "a ticket through ticket_build", "Never describe a", '("only works one player at a time")',
                  "give what's ready and say what's still", "No disclaimers",
                  '("31.2 projected, 82% to get 20+, 9 of last 10")', '"close to a coin flip"',
                  '"the chance of each one missing"', "unless the user asks about risk"):
        assert added in " ".join(prompt.split()), added
    assert "players_scan ranks today's or tomorrow's players" in prompt  # How the app works knows the tool
    assert not banned_phrases(prompt.split("How to talk")[0])


# --- the template ------------------------------------------------------------------------------------------------------

def test_the_scan_template_lists_the_players_then_who_is_close_and_what_is_loading(world) -> None:
    card = Toolbox(world.lab, UID, NY).run("players_scan", {"stat": "pts", "threshold": 20}).cards[0]
    text = scan_template(card)
    lines = text.split("\n")
    assert lines[0] == "Projected for 20+ points tonight:"
    assert lines[1] == ("1. Luka Dončić (DAL v PHX) 31.2 projected, 82% to get there, 9 of last 10, line 19.5 posted")
    assert lines[3].startswith("3. Jayson Tatum (BOS v NYK) [Questionable] 26.4 projected, 71% to get there")
    assert lines[5] == "Also close: Paolo Banchero (ORL v MIA) 18.9 projected, 44% to get there, 5 of last 10."
    assert not refusal_problems(text, "lookup", True) and not banned_phrases(text)
    # Up to five players; a partial board says how many games are in; the form model is labelled
    many = dict(card, players=[dict(r, rank=i) for i, r in enumerate(card["players"] * 2, 1)],
                loading={"games_ready": 8, "games_total": 10})
    lines = scan_template(many).split("\n")
    assert [line[:2] for line in lines[1:6]] == ["1.", "2.", "3.", "4.", "5."] and lines[6].startswith("Also close: ")
    assert "8 of 10 games loaded so far" in scan_template(many)
    form = dict(card, players=[dict(card["players"][0], model="form")], close=[])
    assert scan_template(form).split("\n")[1].endswith("(from recent form)")
    nobody = scan_template({"ask": "60+ points", "players": [], "close": [], "day": "tomorrow",
                            "leagues_without_games": ["NBA"]})
    assert nobody == "No player is projected for 60+ points tomorrow yet (NBA: no games)."


# --- the screenshot's message through the real assistant, with a fake Claude ---------------------------------------------

def usage():
    return SimpleNamespace(input_tokens=900, output_tokens=200, cache_creation_input_tokens=0,
                           cache_read_input_tokens=0, server_tool_use=None)


def reply(content, stop="end_turn", model="claude-opus-5-5"):
    return SimpleNamespace(content=content, stop_reason=stop, usage=usage(), model=model)


def say(value):
    return SimpleNamespace(type="text", text=value)


def tool_use(name, args, n=1):
    return reply([SimpleNamespace(type="tool_use", id=f"toolu_{n}", name=name, input=args)], stop="tool_use")


def results_of(request) -> List[Dict[str, Any]]:
    return [block for message in request["messages"] if isinstance(message["content"], list)
            for block in message["content"] if isinstance(block, dict) and block.get("type") == "tool_result"]


def understood(request) -> Dict[str, Any]:
    first = next(m for m in request["messages"] if m["role"] == "user" and isinstance(m["content"], list))
    context = first["content"][0]["text"]
    return json.loads(context.split("What code understood (use these values): ", 1)[1].split("\n", 1)[0])


def analyst_answer(facts: Mapping[str, Any]) -> str:
    """How the analyst answers from players_scan's facts: the top three, each with the projection, the chance and the
    history, as the tool gave them (and, when the tool said no, why)."""
    if facts.get("error"):
        return "Picks are paused right now, so I can't list players until then."
    lines = []
    for player in facts["players"][:3]:
        lines.append(f"{player['name']} ({player['team']} v {player['opponent']}): {player['projection']} projected, "
                     f"{player['chance'] * 100:.0f}% to get {facts['threshold']}+, {player['history'][1]}.")
    return " ".join(lines)


class ScanClaude:
    """The router reads the screenshot's message as Claude Haiku would; the chat model calls players_scan with what
    code understood (its first answer may be ``first``, e.g. the old refusal) and answers from the tool's facts."""

    def __init__(self, route: Optional[Dict[str, Any]] = None, first: Optional[str] = None,
                 again: Optional[str] = None, router_down: bool = False) -> None:
        self.router_down = router_down
        self.requests: List[Dict[str, Any]] = []
        self.route = route or routed(player_scan=scan("pts", 20))
        self.first, self.again = first, again
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def chat(self) -> List[Dict[str, Any]]:
        return [r for r in self.requests if r["model"] != "claude-haiku-5-5"]

    def _create(self, **request):
        self.requests.append(json.loads(json.dumps(request, default=lambda o: getattr(o, "__dict__", str(o)))))
        if request["model"] == "claude-haiku-5-5":
            if self.router_down:
                return reply([], stop="refusal", model="claude-haiku-5-5")
            return reply([say(json.dumps(self.route))], model="claude-haiku-5-5")
        results = results_of(self.requests[-1])
        corrected = any("Your reply can't be shown" in str(m.get("content")) for m in self.requests[-1]["messages"])
        if self.first is not None and not corrected:
            return reply([say(self.first)])  # the old answer: no tool call at all
        if corrected and self.again is not None:
            return reply([say(self.again)])
        if not results:
            return tool_use("players_scan", scan_args(SimpleNamespace(player_scan=understood(self.requests[-1])
                                                                      ["player_scan"])))
        return reply([say(analyst_answer(json.loads(results[-1]["content"])))])


def test_the_screenshot_s_message_gives_ranked_players_with_no_refusal_or_lecture(world) -> None:
    claude = ScanClaude()
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    out = world.lab.chat(UID, {"message": SCREENSHOT})
    answer = out["reply"]
    # Routed to the scan {pts, 20}: code tells the analyst to call players_scan, no league (the leagues with games)
    chat = claude.chat()
    seen = understood(chat[0])
    assert seen["intent"] == "lookup" and seen["player_scan"] == {"stat": "pts", "threshold": 20, "side": "over",
                                                                  "day": "today"}
    context = chat[0]["messages"][-1]["content"][0]["text"]
    assert 'call players_scan with {"stat": "pts", "threshold": 20, "side": "over", "day": "today"}' in context
    assert "it scans the leagues with games that day" in context
    calls = [b for r in chat for m in r["messages"] if isinstance(m["content"], list) for b in m["content"]
             if isinstance(b, dict) and b.get("type") == "tool_use"]
    assert [c["name"] for c in calls] == ["players_scan"]
    assert calls[0]["input"] == {"stat": "pts", "threshold": 20, "side": "over", "day": "today"}
    # The answer: the analyst's ranked players, checked (every number traced), with the player_scan card
    assert answer["kind"] == "answer", answer
    assert answer["text"].startswith("Luka Dončić (DAL v PHX): 31.2 projected, 82% to get 20+, 9 of last 10.")
    assert "Shai Gilgeous-Alexander" in answer["text"] and "Jayson Tatum" in answer["text"]
    for phrase in LECTURE_AND_REFUSAL:
        assert phrase.lower() not in answer["text"].lower(), phrase
    cards = answer["cards"]
    assert [card["type"] for card in cards] == ["player_scan"]
    assert [r["rank"] for r in cards[0]["players"]] == [1, 2, 3, 4] and cards[0]["players"][0]["projection"] == 31.2
    assert all({"hits", "projection", "chance"} <= set(r) for r in cards[0]["players"])
    assert answer["chips"] == ["Why these players?", "Any injury news?"]
    assert len(chat) == 2  # one tool round and the answer: no retry
    # The thread remembers the scan for follow-ups
    stored = world.store.messages(UID, out["thread_id"])
    assert stored[-1]["content"]["tools"][0].startswith("players_scan(20+ points, basketball_nba, today)")


def test_the_screenshot_s_message_over_the_real_engine_s_numbers_passes_the_post_check(world) -> None:
    world.lab.player_board = real_boards()
    claude = ScanClaude()
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    answer = world.lab.chat(UID, {"message": SCREENSHOT})["reply"]
    assert answer["kind"] == "answer", answer
    card = answer["cards"][0]
    first = card["players"][0]
    assert answer["text"].startswith(f"Luka Dončić (DAL v PHX): {first['projection']} projected, "
                                     f"{first['chance'] * 100:.0f}% to get 20+, 9 of last 10.")
    assert [r["name"] for r in card["players"]][:2] == ["Luka Dončić", "Shai Gilgeous-Alexander"]
    assert len(claude.chat()) == 2  # no retry: every number traced to the engine's


def test_with_the_router_down_code_reads_the_scan_ask_itself(world) -> None:
    from telegram_betting_bot_simple.ai.assistant import scan_fallback
    claude = ScanClaude(router_down=True)
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    answer = world.lab.chat(UID, {"message": SCREENSHOT})["reply"]
    seen = understood(claude.chat()[0])
    assert seen["intent"] == "lookup" and seen["player_scan"] == {"stat": "pts", "threshold": 20, "side": "over",
                                                                  "day": "today"}
    assert answer["kind"] == "answer" and [card["type"] for card in answer["cards"]] == ["player_scan"]
    # Only a list ask with a stat and a number: "how many points did Luka score?" stays chat
    for text, wanted in (("who's getting 10+ boards tomorrow in the nba", {"stat": "reb", "threshold": 10,
                                                                          "side": "over", "day": "tomorrow",
                                                                          "league": NBA}),
                         ("how many points did Luka score last night?", None), ("$10 to win $50", None),
                         ("players with under 9.5 rebounds", {"stat": "reb", "threshold": 10, "side": "under",
                                                              "day": "today"})):
        route = Route()
        scan_fallback(route, text)
        assert route.player_scan == wanted, text


def test_the_old_refusal_is_caught_retried_with_the_tool_named_and_ends_with_the_tool_s_numbers(world) -> None:
    claude = ScanClaude(first=OLD_REFUSAL)
    steps: List[str] = []
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    answer = world.lab.assistant.reply(UID, message=SCREENSHOT, on_step=steps.append)["reply"]
    chat = claude.chat()
    correction = chat[1]["messages"][-1]["content"]
    assert correction.startswith("Your reply can't be shown because it has a refusal (")
    assert "a lecture nobody asked for (" in correction
    assert "Don't refuse and don't add disclaimers: call players_scan with " \
           '{"stat": "pts", "threshold": 20, "side": "over", "day": "today"}' in correction
    assert "Double-checking the answer…" in steps
    # The retry called the scan and answered with its numbers: checked, the card under it
    assert answer["kind"] == "answer" and answer["text"].startswith("Luka Dončić (DAL v PHX): 31.2 projected")
    assert [card["type"] for card in answer["cards"]] == ["player_scan"]
    assert len(chat) == 3  # the refusal, the retry's tool call, the answer: one retry, never a loop


def test_a_second_refusal_gets_code_s_template_with_the_scan_code_ran_itself(world) -> None:
    claude = ScanClaude(first=OLD_REFUSAL, again="Sorry, I'm unable to list players. Ask me about one player.")
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    answer = world.lab.chat(UID, {"message": SCREENSHOT})["reply"]
    assert answer["kind"] == "template"
    assert answer["text"].startswith("Projected for 20+ points tonight:\n1. Luka Dončić (DAL v PHX) 31.2 projected, "
                                     "82% to get there, 9 of last 10")
    assert "unable" not in answer["text"] and "can't" not in answer["text"]
    assert [card["type"] for card in answer["cards"]] == ["player_scan"]  # code ran the scan Claude never called
    assert len(claude.chat()) == 2  # the refusal and the retry: the retry budget is shared, nothing more


def test_a_reply_with_an_invented_number_and_a_lecture_shares_the_one_retry(world) -> None:
    lecture = ("Luka Dončić is 95% to get 20+. Remember a projection doesn't mean he'll score 20: it's close to a coin "
               "flip.")
    claude = ScanClaude(again=lecture)
    calls = {"n": 0}
    real = claude._create

    def first_bad(**request):
        out = real(**request)
        if request["model"] != "claude-haiku-5-5" and out.stop_reason == "end_turn" and calls["n"] == 0:
            calls["n"] += 1
            return reply([say(lecture)])
        return out

    claude.beta = SimpleNamespace(messages=SimpleNamespace(create=first_bad))
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    answer = world.lab.chat(UID, {"message": SCREENSHOT})["reply"]
    correction = claude.chat()[2]["messages"][-1]["content"]
    assert "numbers that are not in the facts you were given: 95%" in correction and "a lecture" in correction
    assert answer["kind"] == "template" and answer["text"].startswith("Projected for 20+ points tonight:")
    assert len(claude.chat()) == 3  # tool call, the bad answer, one retry: then the template


def test_while_picks_are_paused_the_scan_is_off_and_saying_so_isn_t_a_refusal(world) -> None:
    world.lab.assistant = assistant = Assistant(ScanClaude(), world.lab, world.store, tz=NY, clock=lambda: NOW)
    assistant._pause_picks(UID)
    answer = world.lab.chat(UID, {"message": SCREENSHOT})["reply"]
    # A look-up runs (picks paused only stops pick intents), but the scan tool answers picks_paused and code runs
    # nothing itself; the guard doesn't call "I can't suggest picks now" a refusal then
    assert not [card for card in answer["cards"] if card["type"] == "player_scan"]
    route = Route(intent="lookup", player_scan=scan("pts", 20))
    assert Assistant._guard("I can't list picks while they're paused.", route, True, SCREENSHOT, []) == []


class LocksClaude(ScanClaude):
    """"give me 3 locks": the router reads a look-up of the strongest picks (top_picks, no scan); the chat model calls
    brain_top_picks and names them "strongest", never "lock"."""

    def _create(self, **request):
        self.requests.append(json.loads(json.dumps(request, default=lambda o: getattr(o, "__dict__", str(o)))))
        if request["model"] == "claude-haiku-5-5":
            return reply([say(json.dumps(routed(top_picks=True, leagues=[NBA])))], model="claude-haiku-5-5")
        results = results_of(self.requests[-1])
        if not results:
            return tool_use("brain_top_picks", {"leagues": [NBA], "n": 3})
        picks = json.loads(results[-1]["content"])["picks"]
        named = ", ".join(f"{pick['pick']} ({pick['chance'] * 100:.0f}%)" for pick in picks)
        return reply([say(f"Your 3 strongest picks tonight: {named}.")])


def test_give_me_3_locks_gets_the_strongest_picks_in_the_analyst_s_own_words(world) -> None:
    claude = LocksClaude()
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    answer = world.lab.chat(UID, {"message": "give me 3 locks"})["reply"]
    context = claude.chat()[0]["messages"][-1]["content"][0]["text"]
    assert "call brain_top_picks" in context and '"intent": "lookup"' in context
    assert answer["kind"] == "answer" and "strongest" in answer["text"]
    assert not banned_phrases(answer["text"]) and "lock" not in answer["text"].lower()
    assert [card["type"] for card in answer["cards"]] == ["picks"]


def test_this_week_is_scanned_for_today_and_the_context_says_so_in_one_line(world) -> None:
    claude = ScanClaude(route=routed(player_scan=scan("anytime_td", 1, "week"), leagues=[NFL]))
    world.lab.player_board = FakeBoards({(NFL, "today"): FakeBoard(NFL, "today", {"anytime_td": {"players": [
        row(1, "Christian McCaffrey", "SF", "SEA", 0.9, 0.58, [6, 10], [9, 14], model="form", threshold=1,
            why=["a TD in 6 of last 10"])]}})})
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    answer = world.lab.chat(UID, {"message": "anytime TD scorers this week"})["reply"]
    context = claude.chat()[0]["messages"][-1]["content"][0]["text"]
    assert '"league": "americanfootball_nfl"' in context and "scans cover today and tomorrow only" in context
    card = answer["cards"][0]
    assert card["ask"] == "anytime TD" and card["players"][0]["model"] == "form"
    assert card["title"] == "Strongest anytime TD · NFL · tonight"
    assert scan_template(card).startswith("Strongest anytime TD tonight:\n1. Christian McCaffrey (SF v SEA) 0.9 "
                                          "projected, 58% to get there, 6 of last 10 (from recent form)")


# --- the bot -----------------------------------------------------------------------------------------------------------

def test_the_bot_writes_the_scan_as_a_numbered_list_with_view_player_for_the_top_three(world) -> None:
    from telegram_betting_bot_simple.bot import chat
    card = Toolbox(world.lab, UID, NY).run("players_scan", {"stat": "pts", "threshold": 20}).cards[0]
    card["players"][1]["model"] = "form"
    text = chat.card_text(card)
    lines = text.split("\n")
    assert lines[0] == "📊 <b>Projected for 20+ points · NBA · tonight</b>"
    assert lines[1] == "1. <b>Luka Dončić</b> (DAL v PHX)"
    assert lines[2] == "   31.2 proj · 82% · 9/10 L10 · 41/45 season · o19.5 1.30 · u19.5 3.40 (DraftKings)"
    assert lines[3] == "   <i>9 of last 10 at 20+</i>"
    assert "<i>10 of last 10 at 20+ · from recent form</i>" in text  # the form model's row says so
    assert "3. <b>Jayson Tatum</b> (BOS v NYK) · Questionable" in text
    assert "Also close: Paolo Banchero (18.9, 44%)" in text
    partial = chat.card_text(dict(card, loading={"games_ready": 8, "games_total": 10}))
    assert "<i>⏳ 8 of 10 games loaded so far: ask again in a minute for the rest.</i>" in partial
    escaped = chat.card_text(dict(card, players=[dict(card["players"][0], name="<b>Bad</b> & co")]))
    assert "&lt;b&gt;Bad&lt;/b&gt; &amp; co" in escaped
    markup = chat.reply_markup({"thread_id": 3, "message_idx": 1, "reply": {"cards": [card], "chips": []}}, None, False)
    shown = [(b.text, b.callback_data) for row_ in markup.inline_keyboard for b in row_]
    assert shown == [("👤 View Luka Dončić", "ai:pl:3:1:0:0"), ("👤 View Shai Gilgeous-Alexander", "ai:pl:3:1:0:1"),
                     ("👤 View Jayson Tatum", "ai:pl:3:1:0:2")]
    # The helpline shows under a scan, as under picks and tickets
    full = chat.reply_text({"reply": {"text": "Luka leads.", "cards": [card]}, "credits": {}})
    assert "1-800-GAMBLER" in full


def bot_turn(world, claude, words):
    """One message to the bot's AI chat (bot/chat.py) from tg:42, over the world's Lab."""
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


def test_the_screenshot_s_message_in_the_bot_and_view_player(world, monkeypatch) -> None:
    from telegram_betting_bot_simple.bot import chat
    from tests.test_bot_chat import buttons, tapped
    sent, context = bot_turn(world, ScanClaude(), SCREENSHOT)
    words = sent.text.split("📊", 1)[0]
    assert "Luka Dončić (DAL v PHX): 31.2 projected, 82% to get 20+, 9 of last 10." in words
    assert "📊 <b>Projected for 20+ points · NBA · tonight</b>" in sent.text and "\n1. <b>Luka Dončić</b>" in sent.text
    views = [(label, data) for label, data, _ in buttons(sent.markup) if label.startswith("👤 View")]
    thread = context.user_data["ai_thread"]
    assert [data for _, data in views] == [f"ai:pl:{thread}:1:0:{n}" for n in range(3)]
    # Tapping one sends that player's page (the Lab's player page, behind the same gates)
    pages: List[tuple] = []

    def page(league, athlete_id):
        pages.append((league, athlete_id))
        return {"name": "Luka Dončić", "team": "Dallas Mavericks", "position": "G", "status": "Active",
                "props": [{"label": "Points", "line": 29.5, "over": {"decimal": 1.87}, "under": {"decimal": 1.95},
                           "hit_rates": {"last10": "7/10"}, "ai": {"over": 0.6123}}],
                "projection": {"stats": {"pts": {"mean": 31.2}, "reb": {"mean": 8.4}}}}

    monkeypatch.setattr(world.lab, "player", page)
    update, query = tapped(views[0][1])
    asyncio.run(chat.view_player(update, context))
    assert pages == [(NBA, "3945274")] and query.answers == [(None, False)]
    shown = update.effective_message.replies[-1]
    assert shown.startswith("👤 <b>Luka Dončić</b> (Dallas Mavericks, G)") and "Projected: pts 31.2, reb 8.4" in shown
    assert "• Points · o29.5 1.87 · u29.5 1.95 · L10 7/10 · AI 61% over" in shown
    assert "the slip takes game picks" in shown
    # A row that isn't there any more
    update, query = tapped(views[0][1][:-1] + "9")
    asyncio.run(chat.view_player(update, context))
    assert query.answers == [(chat.SCAN_PLAYER_GONE, True)]


def test_the_bot_registers_view_player_before_the_chips() -> None:
    from telegram.ext import Application, CallbackQueryHandler
    from telegram_betting_bot_simple.bot import chat
    from tests.test_bot_chat import CONFIG
    application = Application.builder().token(CONFIG.telegram_token).build()
    chat.setup_chat(application, CONFIG)
    handlers = [h for group in application.handlers.values() for h in group if isinstance(h, CallbackQueryHandler)]
    viewing = next(h for h in handlers if h.callback is chat.view_player)
    assert viewing.pattern.match("ai:pl:12:3:0:2") and not viewing.pattern.match("ai:pl:12:3:0")
    assert not any(h.pattern.match("ai:pl:12:3:0:2") for h in handlers if h.callback is chat.chip_tapped)


# --- over the real player board, from the recorded ESPN fixtures ----------------------------------------------------

def test_sog_3_plus_runs_through_the_lab_the_tool_and_the_bot_over_the_real_board(world, monkeypatch) -> None:
    """data/player_board.py's real PlayerBoard over tests/test_player_scan.py's recorded ESPN answers (the NHL slate of
    28 October): the Lab, players_scan, the assistant with a fake Claude and the bot's list all take its rows, and
    nothing but ESPN is asked."""
    from telegram_betting_bot_simple.bot import chat
    from tests.test_player_scan import ESPN_HOSTS, Web, fixture_routes, make_board
    web = Web(monkeypatch, fixture_routes)
    board = make_board()
    assert board.wait(NHL, "today", 30)
    world.lab.player_board = board
    try:
        out = world.lab.player_scan(UID, None, "SOG", 3)
        assert out["leagues"] == [NHL] and out["ask"] == "3+ shots on goal" and out["stat"] == "sog", out
        top = out["players"][0]
        assert top["name"] == "Leon Draisaitl" and top["model"] == "form" and top["addable"] is False
        # The tool's facts and card; a fake Claude's answer from them passes the post-check
        done = Toolbox(world.lab, UID, NY).run("players_scan", {"stat": "sog", "threshold": 3, "league": NHL})
        assert done.ok and done.content["players"][0]["model_label"] == "from recent form"
        claude = ScanClaude(route=routed(player_scan=scan("sog", 3), leagues=[NHL]))
        world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
        answer = world.lab.chat(UID, {"message": "SOG 3+"})["reply"]
        assert answer["kind"] == "answer" and answer["text"].startswith("Leon Draisaitl (")
        assert [card["type"] for card in answer["cards"]] == ["player_scan"]
        text = chat.card_text(answer["cards"][0])
        assert "1. <b>Leon Draisaitl</b>" in text and "from recent form" in text
        # ESPN only, through the cache: never The Odds API
        assert web.hosts() <= ESPN_HOSTS and world.queued == []
    finally:
        board.close()


# --- responsible-gambling holes found in review: the pause covers the scan, and the guard never overrides support --


def test_the_player_scan_route_honours_the_picks_pause_like_the_builder(world) -> None:
    from telegram_betting_bot_simple.dashboard.service import CallRejected
    client = world.client()
    assert client.get("/api/players/scan", params={"stat": "pts", "threshold": 20}).status_code == 200
    world.lab.pause_picks(UID, NOW + 24 * 3600)
    paused = client.get("/api/players/scan", params={"stat": "pts", "threshold": 20})
    assert paused.status_code == 400, paused.text
    with pytest.raises(CallRejected):
        world.lab.player_scan(UID, NBA, "pts", 20)


def test_a_reply_that_points_to_help_is_never_called_a_refusal_and_no_scan_is_run_for_it(world) -> None:
    decline = ("I'd rather not pull picks while you're trying to win back losses. You can set a budget or take a "
               "break in Settings, and the helpline 1-800-GAMBLER is free 24/7.")
    claude = ScanClaude(first=decline, again=decline)
    world.lab.assistant = Assistant(claude, world.lab, world.store, tz=NY, clock=lambda: NOW)
    out = world.lab.chat(UID, {"message": "lost a lot this week, give me players for 20+ points to make it back"})
    text = out["reply"]["text"]
    assert "players_scan" not in json.dumps([c.get("type") for c in out["reply"].get("cards") or []])
    assert not any(card.get("type") == "player_scan" for card in out["reply"].get("cards") or [])
    assert "Luka" not in text and "projected" not in text
    # the guard leaves a help reply alone even on a plain look-up
    assert Assistant._guard(decline, SimpleNamespace(intent="lookup"), False, "give me players for 20+", []) == []
