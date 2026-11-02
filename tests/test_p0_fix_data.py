"""v4 P0 fix phase, the data half (scratchpad/build/SPEC_P0_FIX.md; the review: scratchpad/p0/review-data/FINDINGS.md,
F1-F7): one regression test per finding of fix:data's (F1, F2, F3's engine, board, Lab and route side, F4, F5, F7;
F6 is fix:ai's), each the reviewer's repro (review-data/py/) turned into a test. Every test here failed on the tree the
reviewers read (scratchpad/p0/prefix_tree) and passes with the fix.

Nothing touches the network: ESPN's answers are tests/test_player_scan.py's fixtures (and its NbaWorld, the real NBA
model over ten captured games) through its patched HTTP layer, and The Odds API is never asked.
"""

from __future__ import annotations

import copy
import datetime
import re
from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient

from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.dashboard.service import DashboardService
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.espn_client import EspnClient
from telegram_betting_bot_simple.data.espn_players import PlayerData
from telegram_betting_bot_simple.data.player_board import PlayerBoard, parse_events
from telegram_betting_bot_simple.engine import scan as engine
from tests.test_player_scan import NOW, NbaWorld, Web, WorldOdds, fixture_routes, load
from tests.test_review2_data import (
    EDM_GAME,
    NY,
    Clock,
    _lab,
    nhl_board,
    one_game_view,
    player,
    refresh,
    rows_of,
    settle,
)

NBA, WNBA, NFL, NHL, MLB = engine.NBA, engine.WNBA, engine.NFL, engine.NHL, engine.MLB
UID = "web:review2-0001"  # tests/test_review2_data.py's _lab person
PROPS = re.compile(r"/events/(\d+)/competitions/\d+/odds/100/propBets$")
WAIT = 10.0  # a scan's wait for a board building or stale props read again (SCAN_WAIT's 2 s, with room on a busy box)


# --- helpers -------------------------------------------------------------------------------------------------------

def move_on(world: NbaWorld, games: int) -> None:
    """The NBA world later in the season (contract 2: a trend needs 5 games this season, and a baseline: this
    season's other games from 10 on, else last season's average): this season's log holds ``games`` games, the 4
    captured and last season's newest moved into it (relabelled as this season's), and last season's log what is
    left. Call it before the board is built."""
    current, previous = copy.deepcopy(world.current_log), copy.deepcopy(world.previous_log)
    regular = next(kind for kind in current["seasonTypes"] if "Regular" in kind["displayName"])
    have = sum(len(category.get("events") or []) for category in regular["categories"])
    older = sorted(((previous["events"][entry["eventId"]]["gameDate"], entry, category)
                    for kind in previous["seasonTypes"] for category in kind["categories"]
                    for entry in category.get("events") or []), key=lambda item: item[0], reverse=True)
    for _, entry, category in older[:max(0, games - have)]:
        category["events"].remove(entry)
        regular["categories"][0]["events"].append(entry)
        current["events"][entry["eventId"]] = previous["events"].pop(entry["eventId"])
    world.current_log, world.previous_log = current, previous


def season_rows(values: List[float], season: str = "2026-27", key: str = "pts") -> List[Dict[str, Any]]:
    return [{"season": season, "stats": {key: value}} for value in values]


def nba_board(world: NbaWorld, clock: Clock) -> PlayerBoard:
    """A board over the NBA world, the real NBA model projecting it (its odds board saved ten minutes ago: free),
    the cache on the board's clock."""
    cache = TTLCache(name="p0-fix-data", clock=clock)
    espn = EspnClient(cache, clock=lambda: NOW)
    service = DashboardService(WorldOdds(world.odds_games()), espn, NY, cache=cache)
    players = PlayerData(espn, now=lambda: datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc))
    return PlayerBoard(service, players, cache, now=clock)


def client_for(lab: Any) -> TestClient:
    client = TestClient(create_app(lab.service, "s3cret", login_delay=0, lab=lab))
    assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
    client.headers.update({"X-Player": UID.split(":", 1)[1]})
    return client


def props_reads(web: Web) -> Dict[str, int]:
    """ESPN's DraftKings props requests by game."""
    counts: Dict[str, int] = {}
    for call in web.calls:
        found = PROPS.search(call["url"])
        if found:
            counts[found.group(1)] = counts.get(found.group(1), 0) + 1
    return counts


def edm_start() -> float:
    return next(event for event in parse_events(load("nhl_scoreboard_20261028")["events"])
                if event["event_id"] == EDM_GAME)["start"]


# --- F1: best edges through the props' time to live ----------------------------------------------------------------

def _edges(board: PlayerBoard) -> List[tuple]:
    return [(row["name"], row["stat"], row["side"], row["edge"]) for row in board.scan_edges(NHL)["players"]]


def test_f1_best_edges_stay_through_the_props_time_to_live(tmp_path, monkeypatch) -> None:
    """py/test_probe_stale_window.py: a game's props went stale after their 10-minute time to live, and the board read
    them again only every 30 minutes, so best-edges mode came back empty for 20 of every 30 ("No posted line shows an
    edge tonight yet", 0 rows at 11, 25, 29 and 42 minutes against 3 at 0, 9 and 31). A scan now reads stale props
    again on their own (ESPN's, never The Odds API) and waits for that read, so the same edges are there at every
    point of the reviewer's timeline, the posted lines unstale, and the Lab's edges mode says them."""
    web = Web(monkeypatch, fixture_routes)
    clock = Clock()
    board = nhl_board(clock, wait=WAIT)
    try:
        assert board.wait(NHL, "today", 30)
        first = _edges(board)
        assert len(first) == 3 and all(edge > 0 for *_, edge in first)
        for minutes in (9, 11, 25, 29):
            clock.now = NOW + minutes * 60
            assert _edges(board) == first, minutes
            posted = rows_of(board.scan(NHL, "sog", None))  # "SOG overs": each player's posted line
            assert posted and not any(row["line"]["stale"] for row in posted), minutes
            settle(board)
        clock.now = NOW + 29 * 60
        refresh(board, clock, NHL, minutes=2)  # 31 minutes: the board's own refresh of props and injuries
        assert _edges(board) == first
        clock.now = NOW + 42 * 60
        assert _edges(board) == first
        # What the person is told: the Lab's edges mode, its props past their time to live again
        lab = _lab(tmp_path, board)
        clock.now = NOW + 55 * 60
        out = lab.player_scan(UID, NHL, None, None, sort="edge")
        assert [(row["name"], row["stat"], row["side"], row["edge"]) for row in out["players"]] == first
        assert not any("the-odds-api" in call["url"] for call in web.calls)
    finally:
        board.close()


def test_f1_a_props_read_over_before_the_scan_looks_again_still_counts(monkeypatch) -> None:
    """A read of stale props can be over before the scan that started it looks at it (fast ESPN, a busy scan thread):
    the scan answers with the fresh lines then too, not the stale view it took first. Here every read is over by the
    time the scan looks (each started refresh is waited for before the scan goes on)."""
    real = PlayerBoard._refresh_due

    def over_at_once(self, board, games, now, stale=()):
        started = real(self, board, games, now, stale)
        for game in games:
            assert game.idle.wait(10)
        return started

    monkeypatch.setattr(PlayerBoard, "_refresh_due", over_at_once)
    Web(monkeypatch, fixture_routes)
    clock = Clock()
    board = nhl_board(clock, wait=WAIT)
    try:
        assert board.wait(NHL, "today", 30)
        first = _edges(board)
        for minutes in (11, 25, 42):
            clock.now = NOW + minutes * 60
            assert _edges(board) == first, minutes
    finally:
        board.close()


def test_f1_reading_stale_props_again_costs_a_few_espn_requests_and_only_while_someone_scans(monkeypatch) -> None:
    """F1's cost. A game's props are read again on their own only for a scan, and at most once per props time to live
    (their cached copy can't be renewed sooner): a scan every minute for an hour reads each game's props 6 times (the
    build, then 10, 20, 30 with the board's own 30-minute refresh, 40 and 50), where the board's refresh alone reads
    them twice (the build and 30). So a 10-game slate costs at most 40 more props requests an hour (one page each up to
    500 props) while someone is scanning, and none while nobody is: the slate's 5-minute prewarm rounds don't read
    them again."""
    web = Web(monkeypatch, fixture_routes)
    clock = Clock()
    board = nhl_board(clock)
    try:
        assert board.wait(NHL, "today", 30)
        games = sorted(board.snapshot(NHL)["games"])
        assert len(games) == 4 and props_reads(web) == {game: 1 for game in games}
        for minutes in range(5, 60, 5):  # nobody scans: the prewarm's rounds only
            clock.now = NOW + minutes * 60
            board.prewarm(leagues=[NHL])
            settle(board)
        assert props_reads(web) == {game: 2 for game in games}
    finally:
        board.close()
    web = Web(monkeypatch, fixture_routes)
    clock = Clock()
    board = nhl_board(clock, wait=WAIT)
    try:
        assert board.wait(NHL, "today", 30)
        for minutes in range(1, 60):  # a scan every minute
            clock.now = NOW + minutes * 60
            assert board.scan_edges(NHL)["players"], minutes
            settle(board)
        assert props_reads(web) == {game: 6 for game in games}
    finally:
        board.close()


def test_f1_an_espn_props_outage_costs_one_read_per_game_per_props_time_to_live(monkeypatch) -> None:
    """The final check's V2: with ESPN's props failing, a scan every minute read each game's props every minute
    (PROPS_RETRY): 30 in 30 minutes, about 600 requests an hour for a 10-game slate. A read that brings no newer copy
    now holds every read of that game's props off for the props' time to live, the board's 30-minute refresh too: at
    most 3 in those 30 minutes, the documented once per 10 minutes (60 an hour for 10 games). The lines stay stale
    meanwhile (no edge: RD8), and once ESPN is back the first read after the hold renews them and the edges return."""
    down = {"props": False}

    def routes(url: str, params: Dict[str, Any]) -> Any:
        if down["props"] and url.endswith("/propBets"):
            raise ConnectionError("ESPN's props down")
        return fixture_routes(url, params)

    web = Web(monkeypatch, routes)
    clock = Clock()
    board = nhl_board(clock, wait=WAIT)
    try:
        assert board.wait(NHL, "today", 30)
        first = _edges(board)
        games = sorted(board.snapshot(NHL)["games"])
        before = props_reads(web)
        down["props"] = True
        for minutes in range(11, 41):  # a scan every minute for 30 minutes, the props past their time to live
            clock.now = NOW + minutes * 60
            assert _edges(board) == [], minutes
            settle(board)
        tried = {game: props_reads(web).get(game, 0) - before.get(game, 0) for game in games}
        assert 1 <= min(tried.values()) and max(tried.values()) <= 3, tried  # was 30 each
        down["props"] = False
        clock.now = NOW + 41 * 60  # the hold since the read at 31 minutes is over
        assert _edges(board) == first
    finally:
        board.close()


# --- F2: the trend is this season's last 5 against a baseline ------------------------------------------------------

def test_f2_the_trend_is_this_seasons_last_5_against_a_baseline() -> None:
    """py/test_probe_trend.py: the trend's last 5 reached into last season while its season average was this season's
    alone, so early in a season one game of last season set the whole ranking (every NBA player on 28 October 2026),
    and a player scoring 30, 32 and 28 this season after last season's 15s ranked as the coldest ("L5 24.2 vs 30.0
    season"). Contract 2: ``l5`` is this season's last 5 (5 needed, never last season's), against this season's
    other games from 10 on (``vs`` "season"), else last season's average with 10 games (``vs`` "last_season"), else
    no trend at all; hot and cold rank only rows with one."""
    back = player("1", "Back Strong", season_rows([30, 32, 28]) + season_rows([15, 16, 14, 15, 15, 17, 16], "2025-26"))
    steady = player("2", "Steady Twenty", season_rows([20] * 10))
    slump = player("3", "Real Slump", season_rows([14, 15, 16, 15, 14] + [24] * 5))
    view = one_game_view(NBA, [back, steady, slump])
    ask = engine.make_ask(NBA, "pts", None)
    cold = engine.scan(view, ask, sort="cold")
    assert [row["name"] for row in cold["players"]] == ["Real Slump", "Steady Twenty"]  # Back Strong: no trend
    assert cold["players"][0]["trend"] == {"l5": 14.8, "season": 24.0, "diff": -9.2, "vs": "season"}
    assert [row["name"] for row in engine.scan(view, ask, sort="hot")["players"]] == ["Steady Twenty", "Real Slump"]
    plain = rows_of(engine.scan(view, engine.make_ask(NBA, "pts", 20)))
    assert "trend" not in next(row for row in plain if row["name"] == "Back Strong")  # no chip either
    spec = engine.STATS[NBA]["pts"]
    # Mixed seasons: 6 games this season, and last season's whole log (the board's last_season_rows)
    this = season_rows([30, 29, 31, 28, 30, 10])
    last = season_rows([20] * 12, "2025-26")
    mixed = {"l5": 29.6, "season": 20.0, "diff": 9.6, "vs": "last_season"}
    assert engine.trend(this + last[:4], spec, last) == mixed
    assert engine.trend(this + last, spec) == mixed  # without the whole log: the rows' own tail, with 10 games
    assert engine.trend(this + last[:4], spec) is None  # 4 games of last season: no baseline
    assert engine.trend(this[1:5] + last, spec, last) is None  # 4 games this season: the last 5 never reach back
    assert engine.trend(last[:10], spec, last) is None  # no game this season yet: his newest is last season's
    # 10 games this season: the last 5 against the other 5, whatever last season was
    ten = season_rows([30, 29, 31, 28, 30, 20, 21, 19, 20, 20])
    assert engine.trend(ten + last, spec, last) == {"l5": 29.6, "season": 20.0, "diff": 9.6, "vs": "season"}
    # The engine reads the board's entry: a player with 6 games this season ranks hot against last season
    early = player("4", "Early Heater", this + last[:4])
    early["last_season_rows"] = last
    hot = engine.scan(one_game_view(NBA, [early, steady]), ask, sort="hot")
    assert [(row["name"], row["trend"]["vs"]) for row in hot["players"]] == [("Early Heater", "last_season"),
                                                                             ("Steady Twenty", "season")]


def test_f2_the_board_keeps_last_seasons_log_while_this_season_is_short_and_asks_for_nothing_more(monkeypatch) -> None:
    """The board's side of contract 2: while a player's season is short of 10 games his entry carries last season's
    whole log (``last_season_rows``), from the copy his last 10 games already reached into: each game log is still
    asked for once. On the captured 28 October (4 games this season) nobody has a trend, so hot ranks nobody, where
    it ranked all 260 by one game of last season; 6 games on, the last 5 rank against last season's average; 12 on,
    against this season's other 7."""
    for games, vs in ((4, None), (6, "last_season"), (12, "season")):
        world = NbaWorld()
        world.events = world.events[:2]
        move_on(world, games)
        web = Web(monkeypatch, world.routes)
        board = nba_board(world, Clock())
        try:
            assert board.wait(NBA, "today", 120)
            hot = board.scan(NBA, "pts", None, n=25, sort="hot")
            if vs is None:  # it ranked every player by the 5th-newest game, last season's
                assert hot["players"] == [] and hot["more"] == 0
                assert not any("trend" in row for row in rows_of(board.scan(NBA, "pts", 20, n=25)))
            logs = [(call["url"], call["params"].get("season")) for call in web.calls if call["url"].endswith("/gamelog")]
            assert logs and len(logs) == len(set(logs)), games  # every game log once
            entries = board.snapshot(NBA)["players"]
            assert entries and all(len(entry["rows"]) >= min(games, 10) for entry in entries)
            if games < 10:  # contract 2's baseline: this season's other games from 10 on
                assert all(len(entry["last_season_rows"]) == 28 - games for entry in entries), games
                assert all(row["season"] == "2025-26" for entry in entries for row in entry["last_season_rows"])
            else:
                assert not any("last_season_rows" in entry for entry in entries)
            if vs is None:
                continue
            assert hot["players"] and {row["trend"]["vs"] for row in hot["players"]} == {vs}
            by_id = {entry["athlete_id"]: entry for entry in entries}
            for row in hot["players"]:
                rows = by_id[row["athlete_id"]]["rows"]
                this = [r["stats"]["pts"] for r in rows if r["season"] == "2026-27"][:games]
                assert len(this) == games and row["trend"]["l5"] == round(sum(this[:5]) / 5, 1)
                base = this[5:] if vs == "season" else [r["stats"]["pts"] for r in
                                                        by_id[row["athlete_id"]]["last_season_rows"]]
                assert row["trend"]["season"] == round(sum(base) / len(base), 1)
        finally:
            board.close()


# --- F3: the names a scan couldn't rank ------------------------------------------------------------------------------

def test_f3_a_named_player_whose_game_has_started_is_started_not_missing(tmp_path, monkeypatch) -> None:
    """py/test_probe_missing_started.py: ten minutes into Edmonton's game the note read "Leon Draisaitl isn't in
    tonight's games": every name off the board went into missing_names, and the board holds only the games still to
    come. Contract 1: the board keeps the started games' players' names (``started_players``), and the engine, the
    Lab and the route put him in ``started_names`` ({name, player, game}: the game's label), in the scan and in edges
    mode, with no league named too (the league that knows him is chosen over the NBA, where nobody has his name)."""
    Web(monkeypatch, fixture_routes)
    clock = Clock()
    board = nhl_board(clock)
    try:
        assert board.wait(NHL, "today", 30)
        lab = _lab(tmp_path, board)
        before = lab.player_scan(UID, NHL, "sog", 3, names=["Leon Draisaitl"])
        assert [row["name"] for row in before["players"]] == ["Leon Draisaitl"] and before["missing_names"] == []
        clock.now = edm_start() + 600
        started = {"name": "Leon Draisaitl", "player": "Leon Draisaitl", "game": "CGY @ EDM"}
        during = lab.player_scan(UID, NHL, "sog", 3, names=["Leon Draisaitl"])
        assert during["players"] == [] and during["close"] == [] and during["games_started"] == 2
        assert during["missing_names"] == []  # was ["Leon Draisaitl"]: "isn't in tonight's games"
        assert (during["ambiguous_names"], during["started_names"]) == ([], [started])
        assert (before["ambiguous_names"], before["started_names"]) == ([], [])
        edges = lab.player_scan(UID, NHL, None, None, sort="edge", names=["draisaitl"])
        assert edges["mode"] == "edges" and edges["players"] == [] and edges["missing_names"] == []
        assert edges["started_names"] == [{**started, "name": "draisaitl"}]
        anywhere = lab.player_scan(UID, None, "points", 1, names=["Leon Draisaitl"])
        assert anywhere["leagues"] == [NHL] and anywhere["started_names"] == [started]
        assert anywhere["missing_names"] == []
        body = client_for(lab).get("/api/players/scan", params=[("league", NHL), ("stat", "sog"), ("threshold", 3),
                                                                ("player", "Leon Draisaitl")]).json()
        assert (body["missing_names"], body["ambiguous_names"], body["started_names"]) == ([], [], [started])
        # Someone in none of the day's games is still missing; a scan without names has none of the lists
        nobody = lab.player_scan(UID, NHL, "sog", 3, names=["Wayne Gretzky"])
        assert (nobody["missing_names"], nobody["started_names"]) == (["Wayne Gretzky"], [])
        assert not {"missing_names", "ambiguous_names", "started_names"} & set(board.scan(NHL, "sog", 3))
        assert not {"missing_names", "ambiguous_names", "started_names"} & set(board.scan_edges(NHL))
    finally:
        board.close()


def test_f3_a_board_built_after_tip_off_still_knows_the_names_in_the_games_under_way(monkeypatch) -> None:
    """A board built once a game is under way (the first scan of the day, or a restart, after the first tip) never had
    that game's players: it now reads each such game's two rosters, after the games to come and for names only, so a
    named player there is started, not missing. Before the first tip it reads no roster more."""
    web = Web(monkeypatch, fixture_routes)
    board = nhl_board(Clock(edm_start() + 600))
    try:
        assert board.wait(NHL, "today", 30)
        out = board.scan(NHL, "sog", 3, names=["Leon Draisaitl", "Wayne Gretzky"])
        assert out["missing_names"] == ["Wayne Gretzky"] and out["players"] == []  # Draisaitl was missing too
        assert out["started_names"] == [{"name": "Leon Draisaitl", "player": "Leon Draisaitl", "game": "CGY @ EDM"}]
        view = board.snapshot(NHL)
        assert EDM_GAME not in view["games"] and (view["games_total"], view["games_started"]) == (2, 2)
        assert ("3114727", "Leon Draisaitl", "CGY @ EDM") in {(p["athlete_id"], p["name"], p["game"])
                                                              for p in view["started_players"]}
        rosters = [call["url"] for call in web.calls if call["url"].endswith("/roster")]
        assert len(rosters) == len(set(rosters)) == 2 * (view["games_total"] + view["games_started"])
    finally:
        board.close()
    web = Web(monkeypatch, fixture_routes)
    board = nhl_board(Clock())
    try:
        assert board.wait(NHL, "today", 30)
        view = board.snapshot(NHL)
        assert view["started_players"] == [] and view["games_started"] == 0
        rosters = [call["url"] for call in web.calls if call["url"].endswith("/roster")]
        assert len(rosters) == 2 * view["games_total"]
    finally:
        board.close()


def test_f3_an_ambiguous_name_lists_its_candidates_instead_of_finding_nobody() -> None:
    """An ambiguous first name matched nobody, so the note read "Jalen isn't in tonight's games". Contract 1: each name
    asked is in exactly one place, and one more than one player answers to is in ``ambiguous_names`` with up to 5 full
    names in alphabetical order, and none of their rows; players in a game under way count."""
    names = ["Jalen Williams", "Jalen Brunson", "Jalen Green", "Jalen Duren", "Jalen Johnson", "Jalen Suggs",
             "Luka Dončić", "Nikola Vučević", "Nikola Jokić", "Stephen Curry"]
    view = one_game_view(NBA, [player(str(i), name, season_rows([20] * 10)) for i, name in enumerate(names)])
    view["started_players"] = [{"athlete_id": "99", "name": "Seth Curry", "team": "CHA", "event_id": "2",
                                "game": "CHA @ BOS"}]
    out = engine.scan(view, engine.make_ask(NBA, "pts", 20), names=["Jalen", "Nikola", "Luka", "Curry", "Nobody"])
    assert [row["name"] for row in out["players"]] == ["Luka Dončić"]
    assert out["ambiguous_names"] == [
        {"name": "Jalen", "candidates": ["Jalen Brunson", "Jalen Duren", "Jalen Green", "Jalen Johnson", "Jalen Suggs"]},
        {"name": "Nikola", "candidates": ["Nikola Jokić", "Nikola Vučević"]},
        {"name": "Curry", "candidates": ["Seth Curry", "Stephen Curry"]}]
    assert out["missing_names"] == ["Nobody"] and out["started_names"] == []
    edges = engine.scan_edges(view, NBA, names=["Nikola", "Seth Curry"])
    assert edges["players"] == [] and edges["missing_names"] == []
    assert edges["ambiguous_names"] == [{"name": "Nikola", "candidates": ["Nikola Jokić", "Nikola Vučević"]}]
    assert edges["started_names"] == [{"name": "Seth Curry", "player": "Seth Curry", "game": "CHA @ BOS"}]


def test_f3_two_players_with_one_name_are_told_apart_by_team_and_a_doubleheader_is_one_player(tmp_path,
                                                                                               monkeypatch) -> None:
    """The e2e world drew "Marcus Sutton" for two teams (Houston's 8100001008, Miami's 8100001400: two people), so the
    note read "More than one player matches Marcus Sutton: Marcus Sutton or Marcus Sutton. Which one?". Candidates who
    share a full name now carry their team ("Marcus Sutton (MIA)"), and that form, or the team after the name, picks
    the one meant, on the route too. One player in two games of the day (an MLB doubleheader) is one player: his row
    for each game, not ambiguous."""
    def sutton(athlete_id: str, team: str) -> Dict[str, Any]:
        return {**player(athlete_id, "Marcus Sutton", season_rows([22] * 10)), "team": team}

    view = one_game_view(NBA, [sutton("8100001008", "HOU"), sutton("8100001400", "MIA"),
                               player("3", "Darius Sutton", season_rows([12] * 10))])
    ask = engine.make_ask(NBA, "pts", 20)
    out = engine.scan(view, ask, names=["Marcus Sutton", "Sutton"])
    assert out["players"] == [] and out["missing_names"] == []
    assert out["ambiguous_names"] == [
        {"name": "Marcus Sutton", "candidates": ["Marcus Sutton (HOU)", "Marcus Sutton (MIA)"]},
        {"name": "Sutton", "candidates": ["Darius Sutton", "Marcus Sutton (HOU)", "Marcus Sutton (MIA)"]}]
    for asked in ("Marcus Sutton (MIA)", "Marcus Sutton MIA", "marcus sutton (mia)"):
        picked = engine.scan(view, ask, names=[asked])
        assert [(row["athlete_id"], row["team"]) for row in picked["players"]] == [("8100001400", "MIA")], asked
        assert (picked["missing_names"], picked["ambiguous_names"]) == ([], []), asked
    assert engine.scan(view, ask, names=["Marcus Sutton (NYK)"])["missing_names"] == ["Marcus Sutton (NYK)"]
    # A doubleheader: one batter in both games of the day
    hits = [{"season": "2026", "stats": {"H": value, "AB": 4}} for value in [1, 2, 0, 1, 1, 2, 1, 0, 1, 1]]
    batter = {**player("77", "Pete Alonso", hits), "team": "NYM"}
    double = one_game_view(MLB, [batter, {**batter, "event_id": "2"}])
    double["games"]["2"] = {**double["games"]["1"], "event_id": "2", "label": "BBB @ AAA (game 2)"}
    both = engine.scan(double, engine.make_ask(MLB, "hits", 1), names=["Pete Alonso"])
    assert [(row["athlete_id"], row["game"]) for row in both["players"]] == [("77", "BBB @ AAA"),
                                                                             ("77", "BBB @ AAA (game 2)")]
    assert (both["missing_names"], both["ambiguous_names"], both["started_names"]) == ([], [], [])
    # The route takes the form the candidates are written in
    Web(monkeypatch, fixture_routes)
    board = nhl_board(Clock())
    try:
        assert board.wait(NHL, "today", 30)
        body = client_for(_lab(tmp_path, board)).get("/api/players/scan", params=[
            ("league", NHL), ("stat", "sog"), ("threshold", 3), ("player", "Leon Draisaitl (EDM)")])
        assert body.status_code == 200, body.text
        assert [row["name"] for row in body.json()["players"]] == ["Leon Draisaitl"]
    finally:
        board.close()


# --- F4: the injury report on games the NBA model projects -----------------------------------------------------------

def test_f4_on_nba_model_games_the_fresh_report_is_the_authority(monkeypatch) -> None:
    """py/test_probe_rd6_nba.py: on a game the NBA model projects, a Questionable and an Out rotation player both
    cleared kept the Q chip, and the Out one never joined, until the roster's 3-hour copy expired: the model's row
    reads the roster's status when the report doesn't list a player, and the board took it. The report read on the
    refresh is the authority now: the chip goes, and the cleared player joins (on the form model, the model still
    leaving him out on the roster's copy) whether or not the odds board is free to re-run the model; once the model
    has him he moves onto it, and ruled out again he drops out."""
    for odds_free in (True, False):
        world = NbaWorld()
        world.events = world.events[:2]
        game = world.events[0]
        roster = world.rosters[game["competitions"][0]["competitors"][0]["team"]["id"]]
        healthy = [athlete for athlete in roster if not athlete.get("injuries")]
        doubt, out = healthy[1], healthy[2]  # two rotation players
        doubt["injuries"] = [{"status": "Questionable"}]
        out["injuries"] = [{"status": "Out"}]
        Web(monkeypatch, world.routes)
        clock = Clock()
        board = nba_board(world, clock)
        try:
            assert board.wait(NBA, "today", 120)

            def find(athlete: Dict[str, Any]) -> Optional[tuple]:
                rows = rows_of(board.scan(NBA, "pts", 5, n=25, names=[athlete["displayName"]]))
                return (rows[0]["status"], rows[0]["model"], rows[0].get("ruled_out")) if rows else None

            assert (find(doubt), find(out)) == (("Questionable", "nba", None), None)
            doubt.pop("injuries")
            out.pop("injuries")  # both cleared: off the injury report
            if not odds_free:
                board._odds_ready = lambda league: False
            refresh(board, clock, NBA)
            assert (find(doubt), find(out)) == ((None, "nba", None), (None, "form", None)), odds_free
            if odds_free:
                board.cache.invalidate_prefix(("espn", "roster"))  # the roster's copy expires: the model has him
                refresh(board, clock, NBA)
                assert find(out) == (None, "nba", None)
                out["injuries"] = [{"status": "Out"}]
                refresh(board, clock, NBA)
                assert find(out)[0] == "Out" and find(out)[2] is True
                assert out["displayName"] not in {row["name"] for row in rows_of(board.scan(NBA, "pts", 5, n=25))}
        finally:
            board.close()


# --- F5: the modes without a stat look at every league ---------------------------------------------------------------

class OnlyBoard:
    """A board where only ``live`` leagues have games; it records the leagues asked, and says what it knows of the day
    (``known``: games_on's answer by league) the way PlayerBoard.games_on does."""

    def __init__(self, live: Any, known: Optional[Dict[str, int]] = None) -> None:
        self.live = {live} if isinstance(live, str) else set(live)
        self.known = dict(known or {})
        self.asked: List[str] = []
        self.players = None

    def day_of(self, day: Any = "today") -> datetime.date:
        return datetime.date(2026, 10, 28)

    def games_on(self, league: str, day: Any = "today") -> Optional[int]:
        return self.known.get(league)

    def _out(self, league: str, mode: str) -> Dict[str, Any]:
        if league not in self.live:
            return {"mode": mode, "players": [], "close": [], "more": 0, "games": 0, "games_started": 0,
                    "no_games": True, "league": league, "day": "2026-10-28", "as_of": "x"}
        row = {"rank": 1, "athlete_id": "1", "name": "Somebody", "team": "AAA", "opponent": "BBB", "threshold": 3,
               "chance": 0.6, "edge": 0.1, "stat": engine.default_stat(league), "side": "over", "kind": "count",
               "hits": {"l5": [3, 5], "l10": [6, 10], "season": [6, 10]}, "avg": {}, "why": []}
        return {"mode": mode, "players": [row], "close": [], "more": 0, "games": 3, "games_started": 0,
                "no_games": False, "league": league, "day": "2026-10-28", "as_of": "x"}

    def scan_edges(self, league, side=None, day="today", n=10, wait=None, *, names=None):
        self.asked.append(league)
        return self._out(league, "edges")

    def scan(self, league, stat, threshold=None, side="over", day="today", n=10, wait=None, *, names=None,
             sort=None):
        self.asked.append(league)
        out = self._out(league, "scan")
        out["ask"] = engine.make_ask(league, stat, threshold, side)
        return out


class HeldSlate:
    """The slate's scoreboards held in memory (SlateBuilder.scoreboards with fetch False)."""

    def __init__(self, games: Dict[str, int]) -> None:
        self.games = games
        self.asked: List[str] = []

    def scoreboards(self, league, days, fetch=False):
        self.asked.append(league)
        return {day: [{"league": league}] * self.games[league] for day in days if league in self.games}


def test_f5_edges_mode_and_hot_or_cold_without_a_stat_look_at_the_nhl_and_mlb(tmp_path) -> None:
    """py/test_probe_lab_leagues.py: both modes listed five leagues and cut the list to MAX_SCAN_CHECKS = 3, so on a day
    only the NHL plays, or only MLB, they asked the NBA, WNBA and NFL and answered no_games. Contract 5: every league
    but soccer is looked at, with no cap; hot and cold stay one league, on its own default stat."""
    for live in (NHL, MLB):
        for sort in ("edge", "hot", "cold"):
            board = OnlyBoard(live)
            lab = _lab(tmp_path, board)
            out = lab.player_scan(UID, None, None, None, sort=sort)
            assert out["leagues"] == [live] and out["no_games"] is False and out["players"], (live, sort)
            order = [NBA, WNBA, NFL, NHL, MLB]
            assert board.asked == order[:order.index(live) + 1], (live, sort)  # every league in turn: no cap of 3
            if sort != "edge":
                assert out["stat"] == engine.default_stat(live) and out["mode"] == "scan"


def test_f5_leagues_with_games_that_day_go_first_and_hot_never_mixes_two(tmp_path) -> None:
    """Contract 5: the leagues known to have games that day go first: the player board's built boards (games_on), else
    the slate's scoreboards held in memory, never a fetch. So an NHL day reads no other board, and hot or cold reads
    the first league with games only, on its stat: never two stats in one ranking."""
    board = OnlyBoard(NHL, known={NBA: 0, WNBA: 0, NHL: 4})
    lab = _lab(tmp_path, board)
    out = lab.player_scan(UID, None, None, None, sort="edge")
    assert board.asked == [NHL] and out["leagues"] == [NHL] and "leagues_without_games" not in out
    board = OnlyBoard(MLB)
    lab = _lab(tmp_path, board)
    lab.service.slate = HeldSlate({NBA: 0, MLB: 7})
    out = lab.player_scan(UID, None, None, None, sort="hot")
    assert board.asked == [MLB] and out["leagues"] == [MLB] and out["stat"] == "tb"
    board = OnlyBoard({NBA, NHL})
    lab = _lab(tmp_path, board)
    out = lab.player_scan(UID, None, None, None, sort="cold")
    assert board.asked == [NBA] and out["leagues"] == [NBA] and out["stat"] == "pts"
    assert {row["stat"] for row in out["players"]} == {"pts"}
    # A league named is the only one looked at
    board = OnlyBoard(NHL)
    lab = _lab(tmp_path, board)
    lab.player_scan(UID, NFL, None, None, sort="edge")
    assert board.asked == [NFL]


def test_f5_an_nhl_day_and_an_mlb_day_through_the_real_board(tmp_path, monkeypatch) -> None:
    """The same through the real board and Lab: a day with only the NHL's games (its board already built, so known to
    have games) and one with only MLB's (nothing known: every league's scoreboard read in order until MLB's)."""
    for live, path in ((NHL, "/hockey/"), (MLB, "/baseball/")):
        def routes(url: str, params: Dict[str, Any], path: str = path) -> Any:
            if url.endswith("/scoreboard") and path not in url:
                return {"events": []}
            return fixture_routes(url, params)

        Web(monkeypatch, routes)
        board = nhl_board(Clock(), wait=WAIT)
        try:
            if live == NHL:
                assert board.wait(NHL, "today", 30)
            lab = _lab(tmp_path, board)
            edges = lab.player_scan(UID, None, None, None, sort="edge")
            assert edges["leagues"] == [live] and edges["no_games"] is False and edges["games"] > 0
            hot = lab.player_scan(UID, None, None, None, sort="hot")
            assert hot["leagues"] == [live] and hot["stat"] == engine.default_stat(live) and hot["no_games"] is False
            if live == NHL:
                assert edges["players"] and "leagues_without_games" not in edges
                assert [row["name"] for row in hot["players"]] == ["Leon Draisaitl"]  # 12 games this season
            else:
                assert edges["leagues_without_games"] == ["NBA", "WNBA", "NFL", "NHL"]
        finally:
            board.close()


# --- F7: name forms --------------------------------------------------------------------------------------------------

NAMES = ["Luka Dončić", "Shai Gilgeous-Alexander", "Jaren Jackson Jr.", "Isaiah Jackson", "Jalen Williams",
         "Jaylin Williams", "Michael Porter Jr.", "Gary Payton II", "Marvin Bagley III", "Nikola Jokić",
         "Stephen Curry"]


def test_f7_a_suffix_an_initial_and_last_first_match() -> None:
    """py/test_probe_names_real.py: "Payton II", "Porter Jr.", "L. Doncic", "Doncic, Luka" and "Marvin Bagley 3rd"
    came back missing: the typed name's suffix wasn't left off before the last-name comparison, and initials weren't
    read. Now a typed suffix (Jr., Sr., II, III, IV, 2nd, 3rd) is left off, and "F. Last" and "Last, First" match;
    one that fits several players is ambiguous (contract 1), not a guess."""
    players = [{"athlete_id": str(i), "name": name} for i, name in enumerate(NAMES)]
    expect = {"Payton II": "Gary Payton II", "Porter Jr.": "Michael Porter Jr.", "L. Doncic": "Luka Dončić",
              "Doncic, Luka": "Luka Dončić", "Marvin Bagley 3rd": "Marvin Bagley III", "N Jokic": "Nikola Jokić",
              "Porter Jr., Michael": "Michael Porter Jr.", "Michael Porter, Jr.": "Michael Porter Jr.",
              "Jackson Jr., Jaren": "Jaren Jackson Jr.", "Gilgeous-Alexander, Shai": "Shai Gilgeous-Alexander",
              "S. Curry": "Stephen Curry", "Bagley III": "Marvin Bagley III",
              # what matched before still does
              "Jaren Jackson Jr.": "Jaren Jackson Jr.", "Jaren Jackson": "Jaren Jackson Jr.", "luka": "Luka Dončić",
              "Steph Curry": "Stephen Curry", "Gary Payton": "Gary Payton II"}
    found, missing = engine.match_names(players, list(expect) + ["J. Williams", "Nobody, Here"])
    for asked, name in expect.items():
        assert [NAMES[int(i)] for i in found.get(asked, [])] == [name], asked
    assert missing == ["Nobody, Here"]
    view = one_game_view(NBA, [player(str(i), name, season_rows([20] * 10)) for i, name in enumerate(NAMES)])
    out = engine.scan(view, engine.make_ask(NBA, "pts", 20), names=["J. Williams", "Payton II", "Doncic, Luka"])
    assert [row["name"] for row in out["players"]] == ["Gary Payton II", "Luka Dončić"]  # same chance: by name
    assert out["ambiguous_names"] == [{"name": "J. Williams", "candidates": ["Jalen Williams", "Jaylin Williams"]}]
    assert out["missing_names"] == []
