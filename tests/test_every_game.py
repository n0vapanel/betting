"""The brain's every-game record (v4 P3-lite, learn/every_game.py): every game on ESPN's boards gets a logged prediction
before its start (from the Odds API board the slate holds, else ESPN's own line, else the brain's ratings), graded after
the final by ESPN's id, the last 14 days back-tested walk-forward without leakage, the NFL and the WNBA in the history,
and the Brain page's and /brain's per-sport lines. Fake ESPN answers, a real Store and raw.db in a temporary folder:
no network, and a counting test shows no Odds API request and no extra ESPN request."""

from __future__ import annotations

import copy
import datetime
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pytest

from telegram_betting_bot_simple.bot.handlers import TELEGRAM_LIMIT, brain_text
from telegram_betting_bot_simple.capture import raw_store
from telegram_betting_bot_simple.capture.raw_store import RawStore
from telegram_betting_bot_simple.dashboard.service import DashboardService
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.espn_client import ESPN_DAY_TZ, EspnClient, espn_path
from telegram_betting_bot_simple.data.slate import SlateBuilder
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.learn import brain_store as bs, every_game, history, registry
from telegram_betting_bot_simple.learn.history import Backfill, HistoryReader
from telegram_betting_bot_simple.learn.progress import progress
from telegram_betting_bot_simple.learn.sources import espn_hist
from telegram_betting_bot_simple.learn.sources.http import HistHttp
from tests.conftest import FakeOdds as ServiceOdds
from tests import test_capture_health
from tests.test_slate import NBA, NHL, NOW, FakeOdds, at, espn_game, odds_event

NFL, WNBA, EPL = "americanfootball_nfl", "basketball_wnba", "soccer_epl"
draw = test_capture_health.draw  # the Node fixture that draws the Brain page
NY = ZoneInfo("America/New_York")
DAY = "20261026"  # NOW's ESPN day


def american_block(close: Optional[str], opened: Optional[str] = None) -> Dict[str, Any]:
    block: Dict[str, Any] = {}
    if close:
        block["close"] = {"odds": close}
    if opened:
        block["open"] = {"odds": opened}
    return block


def pick(home: str = "-150", away: str = "+130", spread: str = "-3.0", total: str = "44.5", tie: Optional[str] = None,
         home_open: Optional[str] = None, away_open: Optional[str] = None, book: str = "ESPN BET") -> Dict[str, Any]:
    """A pickcenter (or a board competition's odds) entry in ESPN's open/close shape."""
    moneyline = {"home": american_block(home, home_open), "away": american_block(away, away_open)}
    if tie:
        moneyline["draw"] = american_block(tie)
    return {"provider": {"id": "58", "name": book}, "moneyline": moneyline,
            "pointSpread": {"home": {"close": {"line": spread, "odds": "-110"}},
                            "away": {"close": {"line": str(-float(spread)), "odds": "-110"}}},
            "total": {"over": {"close": {"line": f"o{total}", "odds": "-110"}},
                      "under": {"close": {"line": f"u{total}", "odds": "-110"}}}}


def with_odds(event: Dict[str, Any], entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if entry is not None:
        event["competitions"][0]["odds"] = [entry]
    return event


class Boards:
    """EspnClient._get: ESPN's scoreboards from fixed events {(league, "YYYYMMDD"): [events]}; every request counted
    by its endpoint (a summary or anything else is an error here: nothing should ask for one)."""

    def __init__(self) -> None:
        self.events: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self.calls: Counter = Counter()
        self.paths = {f"/{sport}/{path}/": league for league, (sport, path) in
                      ((key, espn_path(key)) for key in (NBA, NHL, NFL, WNBA, EPL))}

    def __call__(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        endpoint = url.rsplit("/", 1)[-1]
        self.calls[endpoint] += 1
        if endpoint != "scoreboard":
            raise AssertionError(f"unexpected ESPN request: {url}")
        league = next((league for part, league in self.paths.items() if part in url), None)
        return {"events": copy.deepcopy(self.events.get((league, (params or {}).get("dates")), []))}


class Ratings:
    """The power ratings, read from what they hold (fetch=False); the markets worker's fetches find nothing."""

    def __init__(self, table: Dict[str, Dict[str, float]]) -> None:
        self.table = table

    def power(self, league: str) -> Dict[str, Any]:
        return {}

    def espn_predictor(self, league: str, espn_event_id: Any) -> None:
        return None

    def game_prob(self, league, home, away, neutral=False, surface=None, *, fetch=True):
        found = self.table.get(league)
        return {**found, "source": "Elo", "as_of": at(-2)} if found else None


class World:
    """A slate over a real EspnClient (its _get faked and counted), a real Store and the test_slate odds stand-in."""

    def __init__(self, tmp_path: Path, name: str = "w", every: bool = True, priced: bool = False) -> None:
        self.clock = {"t": NOW}
        now = lambda: self.clock["t"]  # noqa: E731
        self.store = Store(str(tmp_path / f"{name}.db"))
        self.boards = Boards()
        self.espn = EspnClient(TTLCache(clock=now), clock=now)
        self.espn._get = self.boards  # type: ignore[method-assign]
        self.odds = FakeOdds(self.store, now=now)
        self.store.save_odds_snapshot(NBA, [odds_event(NBA, "oddsA", "Boston Celtics", "New York Knicks", at(0.5),
                                                       home_ml=1.6, away_ml=2.5)], NOW - 600)
        if priced:  # the priced path's row for that game (dashboard/service.py _log_ai_predictions)
            self.priced = {"model": "v1", "home_win": 0.7, "margin": 4.0, "total": 221.0, "market_home_win": 0.61,
                           "market_margin": 2.5, "market_total": 220.5, "espn_home_win": None, "model_home_win": 0.66}
            self.store.log_ai_prediction({"league": NBA, "event_id": "oddsA", "espn_id": "401", "home": "Boston Celtics",
                                          "away": "New York Knicks", "start": at(0.5), "payload": self.priced})
        self.events(nfl_home="-150")
        self.slate = SlateBuilder(self.odds, self.espn, self.store, now=now, workers=2, every_game=every,
                                  ratings=Ratings({NHL: {"home": 0.56, "away": 0.44}}))

    def events(self, nfl_home: str) -> None:
        self.boards.events = {
            (NBA, DAY): [espn_game("401", "Boston Celtics", "New York Knicks", at(0.5))],
            (NFL, DAY): [with_odds(espn_game("501", "Kansas City Chiefs", "Denver Broncos", at(0.75), sport="nfl",
                                             ids=("12", "7")), pick(home=nfl_home)),
                         with_odds(espn_game("502", "Buffalo Bills", "Miami Dolphins", at(3), sport="nfl",
                                             ids=("2", "15")), pick())],
            (NHL, DAY): [espn_game("601", "Boston Bruins", "Toronto Maple Leafs", at(0.33), sport="nhl")],
            (WNBA, DAY): [espn_game("701", "Las Vegas Aces", "New York Liberty", at(0.5), sport="wnba")],
        }

    def round(self, minutes: float = 0.0) -> None:
        self.clock["t"] += minutes * 60
        self.slate.refresh()
        self.slate.wait_markets()

    def rows(self) -> Dict[str, Dict[str, Any]]:
        return {row["event_id"]: row for row in self.store.ai_predictions()}


@pytest.fixture(autouse=True)
def room(monkeypatch: pytest.MonkeyPatch) -> None:
    """This container's disk is under raw.db's floor: the capture tests' way round it."""
    monkeypatch.setattr(raw_store.shutil, "disk_usage", lambda path: SimpleNamespace(total=100e9, used=10e9, free=90e9))


# --- 1 · every game gets a logged prediction before it starts -------------------------------------------------------

def test_every_game_is_logged_before_its_start_from_the_first_market_input_there_is(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.round()
    rows = world.rows()
    # NBA: the Odds API board the slate holds; NFL: ESPN's own line from the board; NHL: the ratings alone; the WNBA
    # game has none of the three (nothing logged), the Bills game is 3 hours off (not yet)
    assert set(rows) == {"oddsA", "501", "601"}
    assert {k: rows[k]["payload"]["source"] for k in rows} == {"oddsA": "odds_api", "501": "espn_line", "601": "model"}
    assert all(row["payload"]["retro"] is False and row["status"] == "pending" for row in rows.values())
    assert rows["oddsA"]["espn_id"] == "401" and rows["501"]["espn_id"] == "501"  # odds id, else ESPN's
    nba, nfl, nhl = rows["oddsA"]["payload"], rows["501"]["payload"], rows["601"]["payload"]
    assert nba["market_home_win"] == pytest.approx((1 / 1.6) / (1 / 1.6 + 1 / 2.5), abs=1e-6)
    assert (nba["market_margin"], nba["market_total"]) == (2.5, 220.5)
    assert nfl["market_home_win"] == pytest.approx((1 / (1 + 100 / 150)) / (1 / (1 + 100 / 150) + 1 / 2.3), abs=1e-5)
    assert (nfl["margin"], nfl["total"], nfl["line"]) == (3.0, 44.5, "close")
    assert nhl["market_p"] is None and 0.5 < nhl["home_win"] < 0.56  # the ratings, shrunk: never a made-up 50%
    counts = world.slate.brain_counts
    assert (counts["odds_api"], counts["espn_line"], counts["model"], counts["no_input"], counts["logged"]) == \
        (1, 1, 1, 1, 3)
    # A pending row is updated until its start ...
    world.events(nfl_home="-170")
    world.round(minutes=10)
    moved = world.rows()["501"]["payload"]["market_home_win"]
    assert moved > nfl["market_home_win"]
    # ... and never changed after it (ESPN still says "pre": a late start)
    world.events(nfl_home="-300")
    world.round(minutes=36)
    assert world.rows()["501"]["payload"]["market_home_win"] == moved
    late = {**world.rows()["501"], "payload": {**nfl, "market_home_win": 0.9}}
    assert world.store.log_ai_prediction(late, now=world.clock["t"]) == 0
    assert world.rows()["501"]["payload"]["market_home_win"] == moved


def test_a_priced_prediction_is_never_overwritten_and_the_slate_shows_only_the_nba_model(tmp_path: Path) -> None:
    world = World(tmp_path, priced=True)
    world.round()
    rows = world.rows()
    assert rows["oddsA"]["payload"] == world.priced  # byte for byte the priced path's
    assert [r for r in world.store.ai_predictions() if r["espn_id"] == "401"] == [rows["oddsA"]]
    shown = {g["id"]: g for g in json.loads(world.slate.current().raw)["games"]}
    assert shown[f"{NFL}:espn:501"].get("ai") is None  # a learning record is never shown as the model's chance
    assert shown[f"{NBA}:oddsA"]["ai"] == {"home": 0.7, "away": 0.3}


def test_the_record_adds_no_odds_api_request_and_no_espn_request(tmp_path: Path) -> None:
    on, off = World(tmp_path, "on"), World(tmp_path, "off", every=False)
    for minutes in (0, 10, 10, 20):
        on.round(minutes)
        off.round(minutes)
    assert on.rows() and not off.rows()
    assert on.odds.fetches == [] and off.odds.fetches == []  # zero Odds API requests
    assert on.boards.calls == off.boards.calls and on.boards.calls["scoreboard"] > 0  # not one extra ESPN request
    assert set(on.boards.calls) == {"scoreboard"}  # no summary asked for


def test_a_bad_board_never_breaks_the_slate(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.boards.events[(NFL, DAY)][0]["competitions"][0]["odds"] = [{"provider": "??", "moneyline": [1, 2]}, 7, None]
    world.round()
    assert {g["league"] for g in json.loads(world.slate.current().raw)["games"]} >= {NBA, NFL, NHL}
    assert "501" not in world.rows() and world.slate.brain_counts["failed"] == 0


# --- 2 · grading every logged prediction --------------------------------------------------------------------------

class Finals:
    """ESPN for the grader: each game's result by ESPN id ({id: (home, away)} or "void"), what the boards last said
    (event_state), and every result asked for."""

    def __init__(self, scores: Dict[str, Any], states: Optional[Dict[str, str]] = None) -> None:
        self.scores, self.states, self.asked = scores, states or {}, []

    def find_game(self, *args: Any) -> None:
        return None

    def event_state(self, league: str, espn_id: Any) -> Optional[Dict[str, Any]]:
        state = self.states.get(str(espn_id))
        return {"state": state, "home": None, "away": None} if state else None

    def game_result(self, league: str, espn_id: str) -> Dict[str, Any]:
        self.asked.append(str(espn_id))
        found = self.scores.get(str(espn_id))
        if found == "void":
            return {"state": "void", "status": "STATUS_POSTPONED"}
        if found is None:
            return {"state": "pending"}
        return {"state": "final", "home": found[0], "away": found[1]}

    def closing_line(self, league: str, espn_id: str) -> None:
        return None

    def box_score(self, league: str, espn_id: str) -> Dict[str, Any]:
        return {}

    def standings(self, league: str, season: Any = None) -> Dict[str, Any]:
        return {}


def started(store: Store, hours: float) -> None:
    moment = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours))
    with store._connect() as db:
        db.execute("UPDATE ai_predictions SET start = ?", (moment.strftime("%Y-%m-%dT%H:%M:%SZ"),))


def test_every_logged_prediction_is_graded_by_espn_id_with_hit_and_brier_for_brain_and_line(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.round()
    started(world.store, 3)
    service = DashboardService(ServiceOdds(), Finals({"401": (110, 100), "501": (17, 24), "601": (3, 2)},
                                                     states={"501": "live"}), NY)
    service.store = world.store
    service.grade_due()
    rows = world.rows()
    assert "501" not in service.espn.asked and rows["501"]["status"] == "pending"  # ESPN's board: still on
    nba = rows["oddsA"]
    assert nba["status"] == "graded" and nba["result"]["espn_id"] == "401"  # matched by ESPN's id, not the odds id
    brain = nba["result"]["brain"]
    assert brain["outcome"] == "home" and brain["brain"]["hit"] is True and brain["market"]["hit"] is True
    p_home = nba["payload"]["p"]["home"]
    assert brain["brain"]["brier"] == pytest.approx((1 - p_home) ** 2, abs=1e-6)
    m_home = nba["payload"]["market_home_win"]
    assert brain["market"]["brier"] == pytest.approx((1 - m_home) ** 2, abs=1e-6)
    assert rows["601"]["result"]["brain"]["market"] is None  # the ratings alone: no market line to score
    service.espn.states = {}
    service.grade_due()
    nfl = world.rows()["501"]
    assert nfl["status"] == "graded" and nfl["result"]["brain"]["brain"]["hit"] is False  # the away side won
    assert nfl["result"]["brain"]["brain"]["log_loss"] > 0


def test_no_result_by_the_give_up_time_is_void(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.round()
    started(world.store, 24 * 5)
    service = DashboardService(ServiceOdds(), Finals({"401": "void"}), NY)
    service.store = world.store
    service.grade_due()
    assert {r["status"] for r in world.rows().values()} == {"void"}


def test_the_priced_picks_are_logged_and_graded_byte_identically(tmp_path: Path) -> None:
    def priced(path: Path, extra: bool) -> Tuple[List[Tuple[Any, ...]], Dict[str, Any]]:
        store = Store(str(path))
        service = DashboardService(ServiceOdds(), Finals({}), NY)
        service.store = store
        service.games(NBA)
        if extra:  # the every-game record's rows beside them
            for gid, espn_id in (("g9", "409"), ("409b", "410")):
                payload = every_game.view(NBA, "espn_line", [every_game.line_source(
                    {"p": {"home": 0.6, "away": 0.4}, "as_of": "2020-01-01T00:00:00Z", "book": "espn_bet",
                     "kind": "close"})], start=None, now=0.0,
                    line={"p": {"home": 0.6, "away": 0.4}, "spread": -4.0, "total": 220.0, "kind": "close"})
                assert payload is not None
                store.log_ai_prediction({"league": NBA, "event_id": gid, "espn_id": espn_id, "home": "A", "away": "B",
                                         "start": "2030-01-01T00:00:00Z", "payload": payload}, now=0.0)
        with store._connect() as db:
            db.execute("UPDATE ai_predictions SET start = '2020-01-01T00:00:00Z'")
        service.espn = type("Over", (Finals,), {"find_game": lambda self, *a: {"event_id": "401"}})(
            {"401": (110, 100), "409": (90, 99), "410": (100, 90)})
        service.grade_due()
        rows = sorted((r["event_id"], json.dumps(r["payload"], sort_keys=True), r["status"],
                       json.dumps(r["result"], sort_keys=True)) for r in store.ai_predictions()
                      if not every_game.is_every_game(r["payload"]))
        return rows, service.track_record()

    alone, record_alone = priced(tmp_path / "a.db", extra=False)
    beside, record_beside = priced(tmp_path / "b.db", extra=True)
    assert [r[0] for r in alone] == ["g1", "g2"] and alone == beside
    assert record_alone == record_beside and record_alone["games"] == 2


# --- 3 · the last 14 days, back-tested walk-forward ---------------------------------------------------------------

NOW_B = datetime.datetime(2026, 11, 2, 12, 0, tzinfo=ESPN_DAY_TZ).timestamp()


def summary(espn_id: str, start: float, home: str, away: str, score: Tuple[int, int], *, entry: Optional[Dict] = None,
            status: str = "STATUS_FINAL") -> Dict[str, Any]:
    done = status == "STATUS_FINAL"
    side = lambda where, name, points, tid: {"homeAway": where, "score": str(points),  # noqa: E731
                                             "team": {"id": tid, "displayName": name, "abbreviation": name[:3].upper()}}
    return {"header": {"id": espn_id, "competitions": [{
        "id": espn_id, "date": datetime.datetime.fromtimestamp(start, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "status": {"type": {"name": status, "state": "post" if done else "pre", "completed": done}},
        "competitors": [side("home", home, score[0], "1"), side("away", away, score[1], "2")]}]},
        "pickcenter": [entry] if entry else []}


def fourteen_days(store: Store, raw: RawStore, cache: TTLCache, *, flip: Optional[str] = None,
                  later: bool = False) -> Dict[str, Any]:
    """A fake 14-day world in raw.db (capture jobs and final summaries) and the consensus log, returning the special
    games' ids. ``flip``: that game's own result turned round; ``later``: every game after it a different score and
    line, and a brain version promoted after it."""
    ids: Dict[str, Any] = {"games": []}
    seq = [0]
    x_start = (datetime.datetime.fromtimestamp(NOW_B, ESPN_DAY_TZ).replace(hour=19, minute=0)
               - datetime.timedelta(days=10)).timestamp()  # game X: the NBA's first game 10 days back

    def game(league: str, day: int, hour: int, *, entry: Optional[Dict] = "default", where: str = "raw",  # type: ignore
             status: str = "STATUS_FINAL", sources: Optional[List[Dict]] = None, score=(3, 1)) -> str:
        seq[0] += 1
        espn_id = f"9{seq[0]:04d}"
        local = datetime.datetime.fromtimestamp(NOW_B, ESPN_DAY_TZ).replace(hour=hour, minute=0) - \
            datetime.timedelta(days=day)
        start = local.timestamp()
        three_way = league == EPL
        if entry == "default":
            entry = pick(home="-150", away="+130", home_open="-120", away_open="+100",
                         tie="+260" if three_way else None)
            if three_way:
                entry["moneyline"]["home"]["close"]["odds"], entry["moneyline"]["away"]["close"]["odds"] = "+120", "+240"
        if later and start > x_start:
            score = (score[1], score[0] + 5)
            entry = pick(home="+200", away="-240") if entry else entry
        if flip == espn_id:
            score = (score[1], score[0])
        body = summary(espn_id, start, f"Home {espn_id}", f"Away {espn_id}", score, entry=entry, status=status)
        if where == "raw":
            raw.put(game_key=f"espn:{espn_id}", league=league, source="espn_site", endpoint="summary",
                    params={"event": espn_id}, body=json.dumps(body).encode(), state="final", http_status=200,
                    horizon="final", fetched_at=start + 4 * 3600)
        elif where == "cache":
            cache.put(("espn", "summary", league, espn_id), body, 86400)
        raw.add_job(f"espn:{espn_id}", league, espn_id, start, start + 3 * 3600, state="done", now=start)
        if sources:
            store.log_consensus([{"league": league, "game_key": f"espn:{espn_id}", "event_id": None, "espn_id": espn_id,
                                  "swapped": False, "home": f"Home {espn_id}", "away": f"Away {espn_id}",
                                  "start": every_game._iso(start), "sources": sources,
                                  "blend": {"p": sources[0]["p"]}, "params_version": 0}], start - 600)
        ids["games"].append((league, espn_id, start))
        return espn_id

    rating = lambda h: {"name": "Elo", "kind": "model_rating", "p": {"home": h, "away": round(1 - h, 6)},  # noqa: E731
                        "as_of": "2026-10-01T00:00:00Z", "status": "ok"}
    sharp = {"name": "Sharp", "kind": "sharp", "p": {"home": 0.64, "away": 0.36}, "as_of": "2026-10-01T00:00:00Z",
             "status": "ok"}
    for day in range(1, 15):
        for hour in (19, 21):
            espn_id = game(NBA, day, hour, sources=[rating(0.7)] if (day, hour) == (10, 19) else None)
            if (day, hour) == (10, 19):
                ids["x"], ids["x_start"] = espn_id, ids["games"][-1][2]
                assert ids["x_start"] == x_start
        game(NHL, day, 19)
        weekday = (datetime.datetime.fromtimestamp(NOW_B, ESPN_DAY_TZ) - datetime.timedelta(days=day)).weekday()
        if weekday == 6:
            for hour in (13, 13, 16, 20):
                game(NFL, day, hour)
        if weekday in (0, 3):
            game(NFL, day, 20)
        if weekday in (5, 6):
            game(EPL, day, 10)
            game(EPL, day, 12)
    ids["cached"] = game(NBA, 3, 22, where="cache")              # its summary only in the lean cache
    ids["missing"] = game(NBA, 4, 22, where="none")              # no summary anywhere: waits for the capture
    ids["postponed"] = game(NHL, 5, 21, status="STATUS_POSTPONED")
    ids["model"] = game(NHL, 6, 21, entry=None, sources=[rating(0.58)])
    ids["sharp"] = game(NFL, 2, 21, entry=None, sources=[sharp])
    ids["nothing"] = game(EPL, 7, 15, entry=None)
    game(NBA, 16, 19)                                            # outside the 14 days
    game(NBA, 0, 11)                                             # an hour ago: may not be over
    if later:
        with closing_db(store) as db:
            values = {**registry.defaults("nba"), "consensus.weight.model_rating": 0.15,
                      "consensus.weight.us_books": 0.10}
            version = bs.insert_version(db, "nba", values, status="challenger", now=x_start)
            bs.promote(db, version, now=x_start + 86400)
        ids["version"] = version
    return ids


class closing_db:
    def __init__(self, store: Store) -> None:
        self.db = store._connect()

    def __enter__(self):
        return self.db

    def __exit__(self, *exc: Any) -> None:
        self.db.close()


def backtest_world(tmp_path: Path, name: str, **options: Any) -> Tuple[Store, RawStore, TTLCache, Dict[str, Any]]:
    store, raw, cache = Store(str(tmp_path / f"{name}.db")), RawStore(str(tmp_path / f"{name}-raw.db")), TTLCache()
    return store, raw, cache, fourteen_days(store, raw, cache, **options)


def test_the_last_14_days_are_back_tested_logged_retro_and_graded_at_once(tmp_path: Path) -> None:
    store, raw, cache, ids = backtest_world(tmp_path, "b")
    report = every_game.backtest(store, raw, cache, now=NOW_B)
    by_league = {league: v["logged"] for league, v in report["by_league"].items()}
    assert by_league == {NBA: 29, NHL: 16, NFL: 13, EPL: 8}  # the fake world's per-sport count
    assert (report["no_summary"], report["no_input"], report["void"], report["failed"]) == (1, 1, 1, 0)
    rows = {row["espn_id"]: row for row in store.ai_predictions()}
    assert all(row["payload"]["retro"] is True for row in rows.values())
    assert {row["status"] for row in rows.values()} == {"graded", "void"} and rows[ids["postponed"]]["status"] == "void"
    assert ids["missing"] not in rows and ids["nothing"] not in rows and ids["cached"] in rows
    assert rows[ids["model"]]["payload"]["source"] == "model" and rows[ids["sharp"]]["payload"]["source"] == "odds_api"
    first = rows[ids["games"][0][1]]
    line = first["payload"]
    assert (line["source"], line["line"]) == ("espn_line", "close")  # the close, dated by its lead rule
    assert line["as_of"] == every_game._iso(ids["games"][0][2] - espn_hist.CLOSE_LEAD)
    assert line["market_home_win"] == pytest.approx(0.579832, abs=1e-5)
    assert first["result"]["brain"]["brain"]["hit"] is True  # the home side won 3-1
    soccer = next(r for r in rows.values() if r["league"] == EPL)["payload"]
    assert set(soccer["p"]) == {"home", "away", "draw"}
    # A re-run adds nothing
    again = every_game.backtest(store, raw, cache, now=NOW_B)
    assert again["logged"] == 0 and len(store.ai_predictions()) == len(rows)


def test_the_back_test_never_sees_a_game_s_own_result_or_anything_later(tmp_path: Path) -> None:
    base_store, base_raw, base_cache, ids = backtest_world(tmp_path, "base")
    every_game.backtest(base_store, base_raw, base_cache, now=NOW_B)
    x = ids["x"]
    changed_store, changed_raw, changed_cache, changed = backtest_world(tmp_path, "changed", flip=x, later=True)
    every_game.backtest(changed_store, changed_raw, changed_cache, now=NOW_B)
    before = next(r for r in base_store.ai_predictions() if r["espn_id"] == x)
    after = next(r for r in changed_store.ai_predictions() if r["espn_id"] == x)
    assert before["payload"] == after["payload"]  # same prediction ...
    assert before["result"]["brain"]["outcome"] != after["result"]["brain"]["outcome"]  # ... graded on its new result
    assert after["payload"]["params_version"] == 0
    # The version promoted after it does serve the games after it (the test can see a leak)
    later = [r for r in changed_store.ai_predictions() if r["league"] == NBA and
             every_game._ts(r["start"]) > ids["x_start"] + 2 * 86400]
    assert later and {r["payload"]["params_version"] for r in later} == {changed["version"]}


def test_the_grader_runs_the_back_test_where_raw_db_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, raw, cache, ids = backtest_world(tmp_path, "g")
    real, runs = every_game.backtest, []

    def at_now_b(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        runs.append(kwargs.get("now"))
        return real(*args, **{**kwargs, "now": NOW_B})  # the fake world's clock

    monkeypatch.setattr(every_game, "backtest", at_now_b)
    service = DashboardService(ServiceOdds(), Finals({}), NY)
    service.store = store
    service.grade_due()
    assert not runs and not store.ai_predictions()  # no raw.db in this process: not run
    service.raw = raw
    service.espn = Finals({})
    service.espn.cache = cache
    assert service.grade_due() == 65  # the back-test's graded rows count as graded
    assert len(runs) == 1 and len(store.ai_predictions()) == 66
    service.grade_due()
    assert len(runs) == 1  # at most every RETRO_EVERY seconds


# --- 4 · the NFL and the WNBA in the brain's history ---------------------------------------------------------------

OLD_LEAGUES = {
    "basketball_nba": ("basketball", "nba", (10, 1), (6, 30)),
    "icehockey_nhl": ("hockey", "nhl", (10, 1), (6, 30)),
    "basketball_ncaab": ("basketball", "mens-college-basketball", (11, 1), (4, 15)),
    "americanfootball_ncaaf": ("football", "college-football", (8, 20), (1, 25)),
    "baseball_mlb": ("baseball", "mlb", (3, 15), (11, 15)),
}


class HistTransport:
    """The learn backfill's ESPN: one NFL and one WNBA game a window (their summaries and odds), nothing else."""

    def __init__(self, clock: Any) -> None:
        self.clock, self.calls = clock, []

    def __call__(self, url: str, params: Dict[str, Any], timeout: float) -> Tuple[int, bytes]:
        self.calls.append(url)
        if url.endswith("/events"):
            league = "nfl" if "/football/leagues/nfl/" in url else "wnba" if "/basketball/leagues/wnba/" in url else None
            if league is None:
                return 200, b'{"items": [], "pageCount": 1}'
            ref = f"http://sports.core.api.espn.com/v2/sports/x/leagues/{league}/events/{'7001' if league == 'nfl' else '7002'}"
            return 200, json.dumps({"items": [{"$ref": ref}], "pageCount": 1}).encode()
        if url.endswith("/summary"):
            espn_id = params.get("event")
            data = summary(espn_id, NOW_B - 9 * 86400, "Kansas City Chiefs" if espn_id == "7001" else "Las Vegas Aces",
                           "Denver Broncos" if espn_id == "7001" else "New York Liberty", (27, 20))
            data["boxscore"] = {"teams": [{"homeAway": "home", "statistics": [
                {"name": "firstDowns" if espn_id == "7001" else "totalRebounds", "displayValue": "22"}]}]}
            return 200, json.dumps(data).encode()
        if url.endswith("/odds"):
            return 200, b'{"items": []}'
        return 404, b"{}"


def test_the_nfl_and_the_wnba_join_the_history_with_football_and_basketball_parsing(tmp_path: Path) -> None:
    assert espn_hist.LEAGUES[NFL] == ("football", "nfl", (9, 1), (2, 20))
    assert espn_hist.LEAGUES[WNBA] == ("basketball", "wnba", (5, 1), (10, 25))
    assert {k: v for k, v in espn_hist.LEAGUES.items() if k not in (NFL, WNBA)} == OLD_LEAGUES  # the rest as before
    assert espn_hist.season_dates(NFL, 2025) == (datetime.date(2025, 9, 1), datetime.date(2026, 2, 20))
    assert espn_hist.current_season(NFL, datetime.date(2026, 2, 8)) == 2025
    assert history.season_for(NFL, "2026-08-10T23:00:00Z") == 2026  # a preseason game: still that year's
    store = Store(str(tmp_path / "h.db"))
    clock = SimpleNamespace(ts=datetime.datetime(2026, 10, 28, 2, tzinfo=datetime.timezone.utc).timestamp())
    clock_fn = lambda: clock.ts  # noqa: E731
    transport = HistTransport(clock_fn)
    http = HistHttp(clock=clock_fn, sleep=lambda s: setattr(clock, "ts", clock.ts + s), transport=transport)
    report = Backfill(store, TTLCache(), clock_fn, http=http, tz=datetime.timezone.utc, per_night=200, seasons=1,
                      sources=["espn"], leagues=[NFL, WNBA]).run_night()
    assert report["games"] == 2
    nfl, wnba = list(HistoryReader(store).games(NFL)), list(HistoryReader(store).games(WNBA))
    assert len(nfl) == 1 and nfl[0]["result"] == {"home": 27, "away": 20}
    assert nfl[0]["stats"]["home"]["first_downs"] == 22 and wnba[0]["stats"]["home"]["rebounds"] == 22
    assert any("/football/nfl/summary" in url for url in transport.calls)
    assert any("/basketball/wnba/summary" in url for url in transport.calls)


# --- 5 · the Brain page and /brain ---------------------------------------------------------------------------------

def graded_store(tmp_path: Path) -> Store:
    store, raw, cache, _ids = backtest_world(tmp_path, "p")
    every_game.backtest(store, raw, cache, now=NOW_B)
    world = World(tmp_path, "live")
    world.round()
    for row in world.store.ai_predictions():  # the live rows beside the back-test (pending)
        store.log_ai_prediction(row, now=NOW)
    store.log_ai_prediction({"league": NBA, "event_id": "pricedX", "espn_id": None, "home": "A", "away": "B",
                             "start": "2026-10-20T23:00:00Z",
                             "payload": {"model": "v1", "home_win": 0.6, "market_home_win": 0.55}})
    store.grade_ai_prediction(NBA, "pricedX", "graded", {"state": "final", "home": 101, "away": 99})
    return store


def test_the_record_per_sport_keeps_live_back_tested_and_priced_apart(tmp_path: Path) -> None:
    record = every_game.summary(graded_store(tmp_path))
    nba = next(s for s in record["sports"] if s["sport"] == "nba")
    assert nba["retro"]["predicted"] == 29 and nba["retro"]["graded"] == 29 and nba["retro"]["pending"] == 0
    assert nba["live"]["predicted"] == 1 and nba["live"]["pending"] == 1
    assert nba["priced"]["predicted"] == 1 and nba["priced"]["hit_rate"] == 1.0
    assert nba["retro"]["hit_rate"] is not None and nba["retro"]["market_brier"] is not None
    assert {s["sport"] for s in record["sports"]} >= {"nba", "nhl", "nfl", "soccer"}
    assert progress(Store(str(tmp_path / "p.db")))["every_game"]["sports"]


def test_the_brain_page_shows_the_record_per_sport_escaped(tmp_path: Path, draw) -> None:
    payload = progress(graded_store(tmp_path))
    payload["every_game"]["sports"][0]["label"] = "<img src=x onerror=alert(1)>"
    [page] = draw({"kind": "page", "payload": payload})
    card = next(el for el in page.find("section") if el.attrs.get("id") == "bp-every-game")
    words = card.text()
    assert "Every game, live" in words and "Every game, last 14 days (back-tested)" in words and "Priced picks" in words
    assert "predicted" in words and "hit rate" in words and "vs the line's" in words
    assert not card.find("img") and "<img src=x onerror=alert(1)>" in words  # escaped, shown as text


def test_brain_stays_within_telegram_s_limit_and_names_the_record(tmp_path: Path) -> None:
    store = graded_store(tmp_path)
    text = brain_text(SimpleNamespace(store=store))
    assert len(text) <= TELEGRAM_LIMIT and "Every game, last 14 days (back-tested)" in text and "Priced picks" in text
    # However many sports, the message stays within the limit
    with store._connect() as db:
        for n in range(400):
            db.execute("INSERT INTO ai_predictions (league, event_id, espn_id, home, away, start, logged_at, payload) "
                       "VALUES (?, ?, ?, 'A', 'B', '2026-10-30T00:00:00Z', 0, ?)",
                       (f"soccer_x{n}" if n % 2 else f"tennis_y{n}", f"e{n}", f"{n}",
                        json.dumps({"source": "espn_line", "retro": False, "p": {"home": 0.5, "away": 0.5}})))
    assert len(brain_text(SimpleNamespace(store=store))) <= TELEGRAM_LIMIT


def test_the_every_game_card_fits_a_360_px_phone(tmp_path: Path) -> None:
    joined = test_capture_health.node_path()
    if not joined:
        pytest.skip("Playwright for Node isn't installed: the 360 px layout check needs Chromium")
    payload = json.loads(json.dumps(progress(graded_store(tmp_path))))
    payload["every_game"]["sports"][0]["label"] = "College basketball, women's tournament and conference play-offs"
    script = tmp_path / "layout.cjs"
    script.write_text(test_capture_health.LAYOUT.replace('"bp-capture"', '"bp-every-game"'), encoding="utf-8")
    data = json.dumps({"static": str(test_capture_health.STATIC), "payload": payload,
                       "html": test_capture_health.PAGE % json.dumps(test_capture_health.TITLES),
                       "appStub": test_capture_health.app_stub()})
    done = subprocess.run([test_capture_health.NODE, str(script)], input=data, capture_output=True, text=True,
                          encoding="utf-8", timeout=180, cwd=tmp_path,
                          env={**os.environ, "NODE_PATH": joined, "NODE_OPTIONS": ""})
    assert done.returncode == 0, done.stderr[-3000:]
    out = json.loads(done.stdout)
    if out.get("skip"):
        pytest.skip(out["skip"])
    assert "error" not in out, out.get("error")
    assert [r["scheme"] for r in out["results"]] == ["light", "dark"]
    for result in out["results"]:
        assert not result.get("failed"), result
        assert result["problems"] == [] and result["overflow"] == [], result
        assert result["width"] == 360 and result["page"] <= 360, result
