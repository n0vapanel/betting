"""The second data review of the player scan (scratchpad/scan/review2-data, findings RD1-RD12), each finding's repro
turned into a regression test (v4 P0). Every finding's test here failed on c5deb8f and passes with the fix. The last
test is the review's own hand checks of the numbers it found correct (py/hand.py): not a finding, so it passed on
c5deb8f too, and is kept as a guard.

Nothing touches the network: ESPN's answers are tests/test_player_scan.py's fixtures through its patched HTTP layer,
and The Odds API is never asked (its one entry point here records and refuses).

The boards here keep one clock for the board and its cache (``Clock``): a refresh half an hour on finds the cache's
copies as old as they would be on the server, where both read time.time().
"""

from __future__ import annotations

import datetime
import json
import re
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest

from telegram_betting_bot_simple.dashboard.lab import Lab
from telegram_betting_bot_simple.dashboard.service import DashboardService
from telegram_betting_bot_simple.data import espn_players as ep
from telegram_betting_bot_simple.data import odds_service as odds_mod
from telegram_betting_bot_simple.data import player_board as pb_mod
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.espn_client import EspnClient
from telegram_betting_bot_simple.data.espn_players import PlayerData
from telegram_betting_bot_simple.data.odds_service import OddsService
from telegram_betting_bot_simple.data.player_board import PlayerBoard, parse_events
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.engine import scan as engine
from tests.test_player_scan import FIX, NOW, NbaWorld, Service, Web, WorldOdds, fixture_routes, load

NBA, NFL, NHL = engine.NBA, engine.NFL, engine.NHL
NY = ZoneInfo("America/New_York")
DRAISAITL = "3114727"
EDM_GAME = "401892597"  # Calgary at Edmonton, the NHL fixture game with props and Edmonton's roster


class Clock:
    """A test's clock (epoch seconds) that a test moves by hand."""

    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def nhl_board(clock: Clock, service: Any = None, **options: Any) -> PlayerBoard:
    """A board over the fixtures whose cache keeps the board's own clock."""
    cache = TTLCache(name="review2", clock=clock)
    espn = EspnClient(cache, clock=lambda: NOW)
    players = PlayerData(espn, now=lambda: datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc))
    return PlayerBoard(service or Service(espn, cache), players, cache, now=clock, **options)


def settle(board: PlayerBoard, timeout: float = 20.0) -> None:
    """Wait for every refresh the last snapshot started to finish."""
    time.sleep(0.05)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with board._lock:
            games = [game for built in board._boards.values() for game in built.games.values()]
        if not any(game.refreshing for game in games):
            return
        time.sleep(0.02)
    raise AssertionError("a refresh never finished")


def refresh(board: PlayerBoard, clock: Clock, league: str = NHL, minutes: float = 31) -> None:
    """Move past the board's refresh interval, start the refresh (a snapshot does) and wait for it."""
    clock.advance(minutes * 60)
    board.snapshot(league)
    settle(board)


def rows_of(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(out.get("players") or []) + list(out.get("close") or [])


def summary_with(statuses: Dict[str, str]) -> Dict[str, Any]:
    """An ESPN summary whose injury report lists Edmonton's players with these statuses (and nobody else)."""
    return {"injuries": [{"team": {"id": "6"}, "injuries": [
        {"status": status, "athlete": {"id": pid, "displayName": pid}} for pid, status in statuses.items()]}]}


def form_rows(values: List[float], key: str = "pts", season: str = "2026-27") -> List[Dict[str, Any]]:
    return [{"season": season, "stats": {key: value}} for value in values]


def one_game_view(league: str, players: List[Dict[str, Any]], props_meta: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
    return {"league": league, "games": {"1": {"event_id": "1", "label": "BBB @ AAA", "start": "x", "outs": {},
                                              "props": props_meta or {"book": "DraftKings"}}},
            "games_ready": 1, "games_total": 1, "players": players}


def player(athlete_id: str, name: str, rows: List[Dict[str, Any]], props: Optional[List[Dict[str, Any]]] = None,
           projection: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"athlete_id": athlete_id, "name": name, "team": "AAA", "opponent": "BBB", "home": True, "event_id": "1",
            "status": None, "projection": projection, "rows": rows, "props": props or []}


# --- RD1: the hint's example numbers ----------------------------------------------------------------------------

def test_rd1_a_reply_copying_the_hints_example_numbers_fails_the_post_check() -> None:
    """py/hint_numbers.py: players_scan's hint carried "31.2 projected, 82%, 9 of last 10", and the post-check took
    hint strings as tool numbers, so a reply copying the example for another player passed."""
    from telegram_betting_bot_simple.ai.explain import problems, source_numbers
    from telegram_betting_bot_simple.ai.tools import scan_facts

    found = {"ask": "20+ points", "stat": "pts", "threshold": 20, "side": "over", "day": "today",
             "leagues": [NBA], "games": 3,
             "players": [{"rank": 1, "name": "Real Player", "team": "DAL", "opponent": "OKC", "threshold": 20,
                          "projection": 24.4, "chance": 0.6713, "model": "nba",
                          "hits": {"l5": [3, 5], "l10": [6, 10], "season": [3, 4]},
                          "avg": {"l5": 22.6, "l10": 21.9, "season": 23.0}, "why": ["6 of last 10 at 20+"]}],
             "close": []}
    content = scan_facts(found)
    sources = source_numbers([content])
    copied = "Real Player: 31.2 projected, 82% to get 20+, 9 of last 10."
    assert problems(copied, sources), "the hint's example numbers count as the tool's numbers"
    assert not re.search(r"\b31\.2\b|\b82%", json.dumps(content.get("hint") or ""))
    # The row's own numbers still pass
    assert not problems("Real Player: 24.4 projected, 67% to get 20+, 6 of last 10.", sources)


# --- RD2: double- and triple-doubles without the NBA model ------------------------------------------------------

def double_player(athlete_id: str, name: str, games: List[tuple]) -> Dict[str, Any]:
    rows = [{"season": "2026-27", "stats": {"pts": p, "reb": r, "ast": a, "stl": 1, "blk": 0}} for p, r, a in games]
    return player(athlete_id, name, rows)


def test_rd2_double_doubles_from_the_log_are_a_shrunk_share_of_games() -> None:
    """py/dd.py: without a model projection a double-double was a negative binomial on a 0-5 count of categories:
    10 of 10 gave 59.4%, 1 of 10 gave 31.9% (and qualified), 0 of 10 gave 26.4%."""
    view = one_game_view(NBA, [
        double_player("1", "Scorer NoDD", [(22, 5, 4)] * 10),
        double_player("2", "One DD", [(18, 11, 3)] + [(20, 6, 3)] * 9),
        double_player("3", "Always DD", [(8, 12, 10)] * 10),
    ])
    out = engine.scan(view, engine.make_ask(NBA, "dd"))
    by_name = {row["name"]: row for row in rows_of(out)}
    assert [row["name"] for row in out["players"]] == ["Always DD"]
    assert by_name["Always DD"]["chance"] > 0.8
    assert "One DD" not in [row["name"] for row in out["players"]] and by_name["One DD"]["chance"] < engine.YES_BAR
    # 1 of 10, the newest game: (10 x the recency-weighted share + 5 x the season share 0.1) / 15
    weights = [0.5 ** (i / 5) for i in range(10)]
    assert by_name["One DD"]["chance"] == pytest.approx(round((10 * weights[0] / sum(weights) + 5 * 0.1) / 15, 4))
    spec = engine.STATS[NBA]["dd"]
    never = engine.form_projection(double_player("1", "x", [(22, 5, 4)] * 10)["rows"], spec)
    assert engine.form_chance(never, spec, 2) < 0.05  # 0 of 10 is near nothing
    # A triple-double the same way
    td = engine.scan(one_game_view(NBA, [double_player("4", "Always TD", [(12, 12, 10)] * 10)]),
                     engine.make_ask(NBA, "td"))
    assert td["players"][0]["chance"] > 0.8


# --- RD3: an under's line ---------------------------------------------------------------------------------------

def test_rd3_an_under_never_takes_a_milestone_as_its_line() -> None:
    """py/under_line.py: "under 11.5 points" took the 12+ milestone, an over-only bet at -1060, as its matching line
    (in the app, the bot and the AI's facts), with "12+ posted" as a reason."""
    items = load("nba_props_401909891")["items"]
    props = ep._props_from_json(ep._props_to_json(ep._parse_props("basketball", items)))
    plist = props["6442"]  # Kyrie Irving: a points line at 20.5 and rungs from 12+
    spec = engine.STATS[NBA]["pts"]
    choice = engine.pick_line(plist, spec, 12, "under")
    assert choice is not None and choice["prop"]["kind"] == "over_under" and choice["matches"] is False
    assert choice["prop"]["line"] == 20.5
    rows = form_rows([9.0] * 10)
    view = one_game_view(NBA, [player("6442", "Kyrie Irving", rows, plist,
                                      {"means": {"pts": 10.0, "reb": 5, "ast": 5, "fg3m": 1, "stl": 1, "blk": 1},
                                       "minutes": 32})])
    out = engine.scan(view, engine.make_ask(NBA, "pts", 11.5, "under"))
    row = rows_of(out)[0]
    assert row["line"]["kind"] == "over_under" and row["line"]["matches"] is False and row["line"]["over"] is None
    assert "edge" not in row and not any(re.search(r"\b\d+\+ posted", reason) for reason in row["why"])
    # An over at the same rung still takes the milestone and its price
    over = engine.pick_line(plist, spec, 12, "over")
    assert over["prop"]["kind"] == "milestone" and over["matches"] is True


# --- RD4: no Odds API credits during a build --------------------------------------------------------------------

def _cold_build_with_real_odds(tmp_path, monkeypatch, age: float, jump: float = 0.0) -> Dict[str, Any]:
    """t/test_odds_spend.py: a cold 10-game NBA build over the real OddsService and DashboardService, its saved board
    ``age`` seconds old at the start, each ESPN request moving a shared fake clock 0.17 s (about 0.7 s of latency at 4
    wide), and the first season line's request ``jump`` seconds more: after the build's odds check at its start, before
    any game's service.detail (each game reads its rotation's season lines first), whatever the threads' order. The
    Odds API's only entry point (OddsService._get) records every call and refuses it."""
    world = NbaWorld()
    t0 = time.time()
    clock = {"now": t0, "jump": jump}
    lock = threading.Lock()

    def fake_time() -> float:
        with lock:
            return clock["now"]

    def routes(url: str, params: Dict[str, Any]) -> Any:
        with lock:
            clock["now"] += 0.17
            if clock["jump"] and url.endswith("/stats"):
                clock["now"] += clock["jump"]
                clock["jump"] = 0.0
        return world.routes(url, params)

    fake = SimpleNamespace(time=fake_time, monotonic=time.monotonic, sleep=time.sleep)
    monkeypatch.setattr(odds_mod, "time", fake)
    monkeypatch.setattr(pb_mod, "time", fake)
    web = Web(monkeypatch, routes)
    store = Store(str(tmp_path / f"odds-{age:.0f}.db"))
    store.save_odds_snapshot(NBA, world.odds_games(), t0 - age)
    spent: List[str] = []

    def no_api(self, path, params):
        spent.append(path)
        raise RuntimeError("would have spent Odds API credits")

    monkeypatch.setattr(OddsService, "_get", no_api)
    odds = OddsService("key", cache=TTLCache(name="odds", clock=fake_time), store=store)
    cache = TTLCache(name="espn")
    espn = EspnClient(cache, clock=lambda: NOW)
    service = DashboardService(odds, espn, NY, store=store, cache=cache)
    details: List[float] = []
    real_detail = service.detail

    def detail(league, event_id):
        details.append(round(fake_time() - t0, 1))
        return real_detail(league, event_id)

    service.detail = detail
    players = PlayerData(espn, now=lambda: datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc))
    board = PlayerBoard(service, players, cache, now=lambda: NOW)
    try:
        assert board.wait(NBA, "today", 120)
        result = board.scan(NBA, "pts", 20)
    finally:
        board.close()
    odds_paths = [path for path in spent if re.fullmatch(r"/sports/[^/]+/odds", path)]
    hosts = {re.sub(r"^https?://([^/]+).*$", r"\1", call["url"]) for call in web.calls}
    return {"spent": spent, "odds_paths": odds_paths, "details": details, "hosts": hosts, "result": result}


def test_rd4_no_odds_api_request_during_a_board_build(tmp_path, monkeypatch) -> None:
    """The coverage test: a build 157 fake seconds long whose saved odds board passed the readiness check at the start
    (7079 s old against 7200 - 120) called service.detail 148 s in, after the board's time to live had run out, and
    the real OddsService fetched /sports/basketball_nba/odds."""
    found = _cold_build_with_real_odds(tmp_path, monkeypatch, age=7079)
    assert found["odds_paths"] == [] and found["spent"] == [], found["spent"]
    assert "api.the-odds-api.com" not in found["hosts"]
    assert pb_mod.ODDS_MARGIN >= 900  # odds within 15 minutes of their time to live count as stale
    # The guard costs nothing when the saved board is fresh: every game gets the NBA model, still with no request
    fresh = _cold_build_with_real_odds(tmp_path, monkeypatch, age=600)
    assert fresh["spent"] == [] and len(fresh["details"]) == 10
    assert rows_of(fresh["result"]) and {row["model"] for row in rows_of(fresh["result"])} == {"nba"}


def test_rd4_the_odds_check_runs_inside_each_projection_task(tmp_path, monkeypatch) -> None:
    """With the old 120 s margin the board must still never fetch: the check that the odds board is free runs again
    inside each pool task, just before service.detail, not once at the start of the build.

    v4 P0 fix (the final check's V3): the saved board now ages past its time to live at a fixed point, the first
    season line's request (200 s on), so the old code's service.detail reads it stale and fetches every run; before,
    that depended on the threads' order against the fake clock (3 of 6 runs on c5deb8f)."""
    monkeypatch.setattr(pb_mod, "ODDS_MARGIN", 120.0)
    found = _cold_build_with_real_odds(tmp_path, monkeypatch, age=7079, jump=200.0)
    assert found["odds_paths"] == [] and found["spent"] == [], found["spent"]
    assert found["details"] == []  # every game left to the form model: none of its odds could be read for free


# --- RD5: a failed read leaves a hole all day ------------------------------------------------------------------------

def _roster_blip(monkeypatch) -> List[str]:
    """Edmonton's roster read fails once (at build time)."""
    failed: List[str] = []

    def routes(url: str, params: Dict[str, Any]) -> Any:
        if url.endswith("/teams/6/roster") and not failed:
            failed.append(url)
            raise ConnectionError("ESPN blip")
        return fixture_routes(url, params)

    Web(monkeypatch, routes)
    return failed


def test_rd5_a_failed_roster_is_read_again_on_refresh(monkeypatch) -> None:
    """t/test_board_gaps.py: one failed roster read at build time dropped Edmonton for the rest of the day."""
    failed = _roster_blip(monkeypatch)
    clock = Clock()
    board = nhl_board(clock)
    assert board.wait(NHL, "today", 30)
    assert failed and not [row for row in rows_of(board.scan(NHL, "sog", 3)) if row["team"] == "EDM"]
    refresh(board, clock)
    later = board.scan(NHL, "sog", 3)
    assert [row for row in rows_of(later) if row["team"] == "EDM"], "Edmonton still missing after the refresh"


def test_rd5_the_scan_says_a_game_is_missing_a_part_until_it_comes_in(monkeypatch) -> None:
    """t/test_board_gaps.py: with Edmonton missing, the scan never said it was partial ("loading" absent)."""
    _roster_blip(monkeypatch)
    clock = Clock()
    board = nhl_board(clock)
    assert board.wait(NHL, "today", 30)
    first = board.scan(NHL, "sog", 3)
    assert first.get("loading") == {"games_ready": 4, "games_total": 4, "failed": 1}  # built, but partial
    assert board.snapshot(NHL).get("games_failed") == 1
    refresh(board, clock)
    assert "loading" not in board.scan(NHL, "sog", 3) and board.snapshot(NHL)["games_failed"] == 0


def test_rd5_a_failed_nba_projection_is_retried_on_refresh(monkeypatch) -> None:
    """t/test_detail_once.py: one failed service.detail at build time left that NBA game on the form model all day
    (the refresh only updated players who already had a model projection)."""
    world = NbaWorld()
    Web(monkeypatch, world.routes)
    clock = Clock()
    cache = TTLCache(name="detail-once", clock=clock)
    espn = EspnClient(cache, clock=lambda: NOW)
    service = DashboardService(WorldOdds(world.odds_games()), espn, NY, cache=cache)
    real = service.detail
    failed: List[str] = []

    def detail(league, event_id):
        if not failed:
            failed.append(event_id)
            raise TimeoutError("ESPN slow")
        return real(league, event_id)

    service.detail = detail
    players = PlayerData(espn, now=lambda: datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc))
    board = PlayerBoard(service, players, cache, now=clock)
    try:
        assert board.wait(NBA, "today", 120)
        game = next(g.event_id for b in board._boards.values() for g in b.games.values() if g.odds_id == failed[0])

        def models() -> set:
            return {row["model"] for row in rows_of(board.scan(NBA, "pts", 15, n=25)) if row["event_id"] == game}

        assert models() == {"form"}
        partial = board.snapshot(NBA).get("games_failed")
        refresh(board, clock, NBA)
        assert models() == {"nba"}
        assert (partial, board.snapshot(NBA)["games_failed"]) == (1, 0)  # missing (and said so) until it came in
    finally:
        board.close()


def test_rd5_a_failed_game_log_is_read_again_on_refresh(monkeypatch) -> None:
    """A game log that failed at build time is read again on the refresh (Draisaitl's here)."""
    failed: List[str] = []

    def routes(url: str, params: Dict[str, Any]) -> Any:
        if f"/athletes/{DRAISAITL}/gamelog" in url and not failed:
            failed.append(url)
            raise ConnectionError("ESPN blip")
        return fixture_routes(url, params)

    Web(monkeypatch, routes)
    clock = Clock()
    board = nhl_board(clock)
    assert board.wait(NHL, "today", 30)
    names = lambda: [row["name"] for row in rows_of(board.scan(NHL, "sog", 3))]
    assert failed and "Leon Draisaitl" not in names()
    partial = board.snapshot(NHL).get("games_failed")
    refresh(board, clock)
    assert "Leon Draisaitl" in names()
    assert (partial, board.snapshot(NHL)["games_failed"]) == (1, 0)


# --- RD6: injury changes after the build -------------------------------------------------------------------------------

def _report_routes(report: Dict[str, Dict[str, str]]):
    def routes(url: str, params: Dict[str, Any]) -> Any:
        if url.endswith("/summary") and params.get("event") == EDM_GAME:
            return summary_with(report["now"])
        return fixture_routes(url, params)
    return routes


def test_rd6_a_player_cleared_after_the_build_joins_the_scan(monkeypatch) -> None:
    """t/test_status_refresh.py: a player doubtful at build time and cleared later never came back that day."""
    report = {"now": {DRAISAITL: "Doubtful"}}
    Web(monkeypatch, _report_routes(report))
    clock = Clock()
    board = nhl_board(clock)
    assert board.wait(NHL, "today", 30)
    assert "Leon Draisaitl" not in [row["name"] for row in rows_of(board.scan(NHL, "sog", 3))]
    report["now"] = {}  # cleared to play: off the injury report
    refresh(board, clock)
    after = rows_of(board.scan(NHL, "sog", 3))
    leon = next(row for row in after if row["name"] == "Leon Draisaitl")
    assert leon["status"] is None and leon["status_tag"] is None


def test_rd6_a_questionable_chip_goes_when_he_leaves_the_report(monkeypatch) -> None:
    """t/test_status_refresh.py: a questionable player cleared later kept his Q chip all day."""
    report = {"now": {DRAISAITL: "Questionable"}}
    Web(monkeypatch, _report_routes(report))
    clock = Clock()
    board = nhl_board(clock)
    assert board.wait(NHL, "today", 30)

    def leon() -> Dict[str, Any]:
        return next(row for row in rows_of(board.scan(NHL, "sog", 3)) if row["name"] == "Leon Draisaitl")

    assert (leon()["status"], leon()["status_tag"]) == ("Questionable", "Q")
    report["now"] = {}
    refresh(board, clock)
    assert (leon()["status"], leon()["status_tag"]) == (None, None)
    # And a player ruled out after the build drops out of the scan
    report["now"] = {DRAISAITL: "Out"}
    refresh(board, clock)
    assert "Leon Draisaitl" not in [row["name"] for row in rows_of(board.scan(NHL, "sog", 3))]


# --- RD7: "no games" after the last tip-off, or with ESPN down ---------------------------------------------------------

def test_rd7_after_the_last_tip_off_the_day_still_had_games(monkeypatch) -> None:
    """t/test_board_gaps.py: after the last tip-off the board said no_games, and the Lab moved on to other leagues."""
    Web(monkeypatch, fixture_routes)
    clock = Clock()
    board = nhl_board(clock)
    assert board.wait(NHL, "today", 30)
    clock.now = parse_events(load("nhl_scoreboard_20261028")["events"])[-1]["start"] + 60
    out = board.scan(NHL, "sog", 3)
    assert out["games"] == 0 and out["games_started"] == 4
    assert out["no_games"] is False and Lab._no_games(out) is False


def test_rd7_espn_down_is_not_a_day_without_games(monkeypatch) -> None:
    """t/test_espn_down2.py: ESPN's scoreboard failing read as "no games" to the Lab, and its error was dropped."""
    def routes(url: str, params: Dict[str, Any]) -> Any:
        if url.endswith("/scoreboard"):
            raise ConnectionError("ESPN down")
        return fixture_routes(url, params)

    Web(monkeypatch, routes)
    board = nhl_board(Clock())
    assert board.wait(NHL, "today", 30)
    out = board.scan(NHL, "sog", 3)
    assert out["error"] and out["no_games"] is False and not Lab._no_games(out)


def _lab(tmp_path, board: PlayerBoard) -> Lab:
    """A real Lab over the board, its person's age confirmed."""
    cache = TTLCache(name="review2-lab")
    service = DashboardService(WorldOdds([]), EspnClient(cache, clock=lambda: NOW), NY,
                               store=Store(str(tmp_path / "lab.db")), cache=cache)
    lab = Lab(service, service.store, board.players, now=lambda: NOW)
    lab.player_board = board
    lab.user("web:review2-0001")
    service.store.confirm_age("web:review2-0001", now=NOW)
    return lab


def test_rd7_the_lab_passes_the_error_and_the_started_games_on(tmp_path, monkeypatch) -> None:
    """The Lab's scan carries the board's error and started count (for the facts), and stays on the league instead
    of reporting it as without games."""
    def routes(url: str, params: Dict[str, Any]) -> Any:
        if url.endswith("/scoreboard"):
            raise ConnectionError("ESPN down")
        return fixture_routes(url, params)

    Web(monkeypatch, routes)
    board = nhl_board(Clock())
    assert board.wait(NHL, "today", 30)
    lab = _lab(tmp_path, board)
    down = lab.player_scan("web:review2-0001", None, "sog", 3)
    assert down["leagues"] == [NHL] and down["error"] and down["no_games"] is False
    assert "leagues_without_games" not in down


def test_rd7_the_lab_reports_started_games(tmp_path, monkeypatch) -> None:
    Web(monkeypatch, fixture_routes)
    clock = Clock()
    board = nhl_board(clock)
    assert board.wait(NHL, "today", 30)
    lab = _lab(tmp_path, board)
    clock.now = parse_events(load("nhl_scoreboard_20261028")["events"])[-1]["start"] + 60
    late = lab.player_scan("web:review2-0001", None, "sog", 3)
    assert late["leagues"] == [NHL] and late["games_started"] == 4 and late["no_games"] is False
    assert late["players"] == [] and "leagues_without_games" not in late


# --- RD8: stale lines ----------------------------------------------------------------------------------------------

def test_rd8_lines_past_the_props_ttl_are_stale_and_get_no_edge(monkeypatch) -> None:
    """t/test_board_gaps.py: the stale flag was set only when props were fetched, so 10-30 minute old lines (the board
    reads them every 30) showed as fresh and were priced for an edge.

    v4 P0 fix (F1): a scan now reads stale props again before it answers (tests/test_p0_fix_data.py has the edges
    back at 11 and 25 minutes), so this test, which read the fixtures' props fine, asserted the 20-minute gap F1 is
    about. A line stays stale now only when that read can't renew its copy: ESPN's props failing here."""
    down = {"props": False}

    def routes(url: str, params: Dict[str, Any]) -> Any:
        if down["props"] and url.endswith("/propBets"):
            raise ConnectionError("ESPN down")
        return fixture_routes(url, params)

    Web(monkeypatch, routes)
    clock = Clock()
    board = nhl_board(clock)
    assert board.wait(NHL, "today", 30)
    fresh = board.scan(NHL, "sog", 3)["players"][0]
    assert fresh["line"]["stale"] is False and "edge" in fresh
    down["props"] = True
    for minutes in (11, 25):  # past the props' 10-minute time to live, before the 30-minute refresh
        clock.now = NOW + minutes * 60
        top = board.scan(NHL, "sog", 3)["players"][0]
        assert top["line"]["stale"] is True, minutes
        assert "edge" not in top, minutes


# --- RD9: at most four requests in flight ------------------------------------------------------------------------------

def test_rd9_at_most_four_requests_in_flight_over_stale_game_logs(monkeypatch) -> None:
    """t/test_peak3.py: a board built over a shared cache whose game logs were past their time to live (inside the
    stale window) had them refreshed on the cache's own threads, outside the board's limit: six in flight."""
    lock = threading.Lock()
    flying: Dict[int, str] = {}
    over: List[List[str]] = []

    def routes(url: str, params: Dict[str, Any]) -> Any:
        with lock:
            flying[threading.get_ident()] = threading.current_thread().name
            if len(flying) > 4:
                over.append(sorted(flying.values()))
        try:
            time.sleep(0.03)
            return fixture_routes(url, params)
        finally:
            with lock:
                flying.pop(threading.get_ident(), None)

    web = Web(monkeypatch, routes)
    clock = Clock()
    cache = TTLCache(name="peak3", clock=clock)
    espn = EspnClient(cache, clock=lambda: NOW)
    players = PlayerData(espn, now=lambda: datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc))
    first = PlayerBoard(Service(espn, cache), players, cache, now=clock)
    assert first.wait(NHL, "today", 60)
    web.peak = 0
    clock.advance(70 * 60)  # game logs: 1 hour fresh + 1 hour stale on game day
    second = PlayerBoard(Service(espn, cache), players, cache, now=clock)
    assert second.wait(NHL, "today", 60)
    time.sleep(0.5)
    assert web.peak <= 4, over[:2]


def test_rd9_matching_odds_on_a_refresh_runs_on_the_boards_pool(monkeypatch) -> None:
    """The refresh's odds match (an NBA game without an odds id yet) ran on the refresh thread, outside the limit."""
    Web(monkeypatch, fixture_routes)
    clock = Clock()
    board = nhl_board(clock)
    threads: List[str] = []
    real = board._match_odds

    def match(league, games):
        threads.append(threading.current_thread().name)
        return real(league, games)

    board._match_odds = match
    assert board.wait(NBA, "today", 30)
    built = len(threads)
    refresh(board, clock, NBA)
    refreshed = threads[built:]
    assert refreshed and all(name.startswith("player-board_") for name in refreshed), refreshed


# --- RD10-RD12: the engine's words and numbers ---------------------------------------------------------------------

def test_rd10_under_half_a_goal_reads_as_no_goal() -> None:
    """py/under_why.py: "under 0.5 goals" for a player who scored in 2 of 10 read "a goal in 8 of last 10"."""
    rows = [{"season": "2026-27", "stats": {"G": g, "A": 0, "PTS": g, "SOG": 3}} for g in [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    view = one_game_view(NHL, [player("1", "Two Goals In Ten", rows)])
    out = engine.scan(view, engine.make_ask(NHL, "goals", 0.5, "under"))
    row = rows_of(out)[0]
    assert row["hits"]["l10"] == [8, 10]
    assert row["why"][0] == "no goal in 8 of last 10"
    assert not any(reason.startswith("a goal in") for reason in row["why"])


def test_rd11_a_one_game_season_no_longer_outweighs_the_last_ten() -> None:
    """py/shrink.py: 40 then nine 15s (last season's) gave 26.2 projected and 74% for 20+: the season's one game
    weighed as five."""
    rows = [{"season": "2026-27", "stats": {"pts": 40}}] + [{"season": "2025-26", "stats": {"pts": 15}}] * 9
    spec = engine.STATS[NBA]["pts"]
    form = engine.form_projection(rows, spec)
    weights = [0.5 ** (i / 5) for i in range(10)]
    recent = sum(w * v for w, v in zip(weights, [40] + [15] * 9)) / sum(weights)
    assert form["mean"] == pytest.approx((10 * recent + 1 * 40) / 11)  # the season weighs its one game
    assert round(form["mean"], 1) != 26.2 and round(form["mean"], 1) == 21.2
    chance = engine.form_chance(form, spec, 20)
    assert chance < 0.6 and round(chance, 2) != 0.74
    # Five or more games this season weigh five, as before
    assert engine.form_mean([30] * 10, [30] * 10 + [10] * 10) == pytest.approx(400 / 15)


def test_rd12_two_plus_tds_and_goals_are_never_answered_as_anytime() -> None:
    """engine/scan.py's make_ask: "2+ TDs" and "2+ goals" were silently the anytime (1+) ask."""
    two = engine.make_ask(NFL, "TDs", 2)
    assert (two["threshold"], two["label"]) == (2, "2+ TDs")
    assert engine.make_ask(NFL, "anytime_td", 3)["label"] == "3+ TDs"
    assert engine.make_ask(NHL, "anytime goal", 2)["label"] == "2+ goals"
    assert engine.make_ask(NFL, "ATTD")["label"] == "anytime TD" and engine.make_ask(NFL, "ATTD", 1)["threshold"] == 1
    assert engine.parse_ask("2+ TDs")["threshold"] == 2 and engine.parse_ask("anytime TD scorers")["threshold"] == 1
    # The same player: 2+ is its own (smaller) chance, never the anytime price
    tds = [{"season": "2026", "stats": {"RUSH TD": v}} for v in [1, 2, 1, 1, 1, 0, 2, 1, 0, 1]]
    yes = [{"market": "player_anytime_td", "kind": "yes", "line": None, "yes_price": "2.10", "over_price": None,
            "under_price": None, "label": None}]
    view = one_game_view(NFL, [player("9", "Goal Line Back", tds, yes)])
    one = rows_of(engine.scan(view, engine.make_ask(NFL, "anytime_td")))[0]
    twice = rows_of(engine.scan(view, engine.make_ask(NFL, "anytime_td", 2)))[0]
    assert twice["chance"] < one["chance"]
    assert one["line"]["kind"] == "yes" and "line" not in twice and "edge" not in twice
    assert twice["why"][0] == "2 of last 10 at 2+"


# --- the review's hand checks (numbers it found correct), kept as a guard --------------------------------------------

def test_the_reviews_hand_checks_still_hold() -> None:
    """py/hand.py: hit rates and averages, the form average, the dispersion clamp and the NB chance, anytime as 1 -
    P(0), the yards spread floors. Not a finding: this passed on c5deb8f too and must keep passing."""
    import math
    from statistics import NormalDist

    def nb_pmf(k, mean, ratio):
        if ratio <= 1.0001:
            return math.exp(-mean) * mean ** k / math.factorial(k)
        p = 1 / ratio
        r = mean * p / (1 - p)
        return math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)) * p ** r * (1 - p) ** k

    pts = [25, 18, 30, 20, 19, 22, 31, 15, 24, 20, 28, 12]
    rows = [{"season": "2026-27" if i < 7 else "2025-26", "stats": {"pts": v}} for i, v in enumerate(pts)]
    spec = engine.STATS[NBA]["pts"]
    hits, avg, _ = engine.history(rows, spec, 20, "over")
    assert hits == {"l5": [sum(v >= 20 for v in pts[:5]), 5], "l10": [sum(v >= 20 for v in pts[:10]), 10],
                    "season": [sum(v >= 20 for v in pts[:7]), 7]}
    assert avg == {"l5": round(sum(pts[:5]) / 5, 1), "l10": round(sum(pts[:10]) / 10, 1),
                   "season": round(sum(pts[:7]) / 7, 1)}
    under, _, _ = engine.history(rows, spec, 20, "under")
    assert under == {key: [games - hit, games] for key, (hit, games) in hits.items()}
    weights = [0.5 ** (i / 5) for i in range(10)]
    weighted = sum(w * v for w, v in zip(weights, pts[:10])) / sum(weights)
    mean = (10 * weighted + 5 * sum(pts[:7]) / 7) / 15
    assert engine.form_mean(pts, pts[:7]) == pytest.approx(mean, abs=1e-9)
    m = sum(pts) / len(pts)
    ratio = min(max(sum((v - m) ** 2 for v in pts) / (len(pts) - 1) / m, 1), 4)
    assert engine.dispersion(pts) == pytest.approx(ratio)
    assert engine.count_chance(mean, 20, ratio) == pytest.approx(1 - sum(nb_pmf(k, mean, ratio) for k in range(20)),
                                                                 abs=1e-9)
    assert engine.count_chance(0.45, 1, 1.3) == pytest.approx(1 - nb_pmf(0, 0.45, 1.3))
    yds = [250, 310, 190, 275, 228]
    sd = math.sqrt(sum((v - sum(yds) / 5) ** 2 for v in yds) / 4)
    assert engine.yards_sd(yds, 240.0) == pytest.approx(max(sd, 0.15 * 240, 10))
    assert engine.yards_sd([100, 100, 100], 100) == 15 and engine.yards_sd([20, 20, 20], 20) == 10
    assert engine.yards_chance(240.0, 250, engine.yards_sd(yds, 240.0)) == pytest.approx(
        1 - NormalDist().cdf((249.5 - 240.0) / engine.yards_sd(yds, 240.0)))
    assert FIX.exists()
