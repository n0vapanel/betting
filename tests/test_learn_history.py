"""The learning loop's history (v3, Contract 1): football-data, nflverse and ESPN backfills into the hist_* tables,
the point-in-time reader and the grader's live capture. Nothing here touches the network: every request goes to a
fake transport serving trimmed captures in tests/fixtures/learn/ (football-data.co.uk E0 2025/26 and 2026/27, D1
2019/20, the USA file; nflverse games.csv; ESPN's NBA events, summary and core odds, an NFL odds answer with the
live-odds provider, an EPL summary, MLB's 2025-07-03 scoreboard and odds; MLB StatsAPI's 2025 schedule, trimmed),
captured on 2026-10-27/28."""

from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.markets.aliases import team_key
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.learn import history_store as hs
from telegram_betting_bot_simple.learn.history import (PER_NIGHT_DEFAULT, Backfill, HistoryReader, capture_final,
                                                       per_night_budget, progress)
from telegram_betting_bot_simple.learn.sources import espn_hist, football_data as fd, nflverse
from telegram_betting_bot_simple.learn.sources.http import HistHttp

FIX = Path(__file__).parent / "fixtures" / "learn"
UTC = datetime.timezone.utc


def fixture(name: str) -> bytes:
    return (FIX / name).read_bytes()


def at(*parts: int) -> float:
    return datetime.datetime(*parts, tzinfo=UTC).timestamp()


class Clock:
    """A fake clock (epoch seconds): sleeping moves it on."""

    def __init__(self, ts: float) -> None:
        self.ts = ts
        self.slept: List[float] = []

    def __call__(self) -> float:
        return self.ts

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.ts += seconds


EMPTY_LIST = json.dumps({"count": 0, "pageIndex": 1, "pageCount": 1, "items": []}).encode()


class Transport:
    """Routes the backfill's URLs to the fixtures (404 for files not captured); records (url, params, time)."""

    def __init__(self, clock: Clock, per_request: float = 0.0, fail: Tuple[str, ...] = ()) -> None:
        self.clock = clock
        self.per_request = per_request
        self.fail = list(fail)
        self.calls: List[Tuple[str, Dict[str, Any], float]] = []

    def __call__(self, url: str, params: Dict[str, Any], timeout: float) -> Tuple[int, bytes]:
        self.calls.append((url, dict(params), self.clock()))
        self.clock.ts += self.per_request
        for pattern in list(self.fail):
            if pattern in url:
                self.fail.remove(pattern)
                raise ConnectionError("connection reset")
        files = {"mmz4281/2526/E0.csv": "fd_E0_2526.csv", "mmz4281/1920/D1.csv": "fd_D1_1920.csv",
                 "mmz4281/2627/E0.csv": "fd_E0_2627.csv", "/new/USA.csv": "fd_new_USA.csv",
                 "nfldata/master/data/games.csv": "nfl_games.csv"}
        for pattern, name in files.items():
            if url.endswith(pattern):
                return 200, fixture(name)
        if "football-data.co.uk" in url:
            return 404, b"Not Found"
        if "sports.core.api.espn.com" in url and url.endswith("/events"):
            return 200, fixture("espn_nba_events.json") if "/nba/" in url else EMPTY_LIST
        if url.endswith("/odds"):
            return 200, fixture("espn_mlb_odds.json") if "/mlb/" in url else fixture("espn_nba_odds.json")
        if url.endswith("/summary"):
            return 200, fixture("espn_nba_summary.json")
        if url.endswith("/scoreboard"):
            if "/mlb/" in url and params.get("dates") == "20250703":
                return 200, fixture("espn_mlb_scoreboard_20250703.json")
            return 200, b'{"events": []}'
        if "statsapi.mlb.com" in url:
            return 200, fixture("mlb_season_2025.json") if str(params.get("season")) == "2025" else b'{"dates": []}'
        return 404, b"{}"

    def urls(self, part: str = "") -> List[str]:
        return [url for url, _params, _ts in self.calls if part in url]


def backfill(store: Store, clock: Clock, transport: Transport, **options: Any) -> Backfill:
    http = HistHttp(clock=clock, sleep=clock.sleep, transport=transport)
    return Backfill(store, TTLCache(), clock, http=http, tz=UTC, **options)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(str(tmp_path / "betting.db"))


def db_of(store: Store) -> sqlite3.Connection:
    db = store._connect()
    hs.init_history_schema(db)
    return db


# --- the schema on a real Store ---------------------------------------------------------------------------------

def test_history_schema_works_on_a_real_store_connection(store: Store) -> None:
    db = store._connect()
    hs.init_history_schema(db)
    hs.init_history_schema(db)  # idempotent
    names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"hist_games", "hist_lines", "hist_stats", "hist_weather", "hist_progress", "hist_budget"} <= names
    assert {"consensus_log", "shared_cache"} <= names  # the Store's own tables are untouched
    with pytest.raises(sqlite3.IntegrityError):  # unique on (league, game_key)
        db.execute("INSERT INTO hist_games (league, game_key, start, home, away, status, source, fetched_at) "
                   "VALUES ('l', 'k', 's', 'h', 'a', 'final', 'x', 0), ('l', 'k', 's', 'h', 'a', 'final', 'x', 0)")
    key = hs.upsert_game(db, {"league": "basketball_nba", "game_key": "espn:1", "start": "2026-01-01T00:00:00Z",
                              "home": "Boston Celtics", "away": "New York Knicks", "home_score": 100,
                              "away_score": 90, "status": "final", "source": "espn"}, now=1.0)
    assert hs.get_game(db, "basketball_nba", key)["home_score"] == 100
    db.close()


# --- football-data: the closing columns, Pinnacle, aliases ---------------------------------------------------------

def test_football_data_c_columns_are_the_close_and_plain_columns_the_early_market() -> None:
    mapping = {column: (kind, book, market, side, point) for column, kind, book, market, side, point in fd.COLUMNS}
    # The pinned mapping: a C after the bookmaker is the close (the old backtest had it the wrong way round)
    assert mapping["AvgCH"] == ("close", "avg", "h2h", "home", None)
    assert mapping["AvgH"] == ("open", "avg", "h2h", "home", None)
    assert mapping["PSCD"] == ("close", "pinnacle", "h2h", "draw", None)
    assert mapping["PSD"] == ("open", "pinnacle", "h2h", "draw", None)
    assert mapping["BFECA"] == ("close", "betfair_ex", "h2h", "away", None)
    assert mapping["B365C>2.5"] == ("close", "bet365", "totals", "over", "2.5")
    assert mapping["MaxCH"] == ("close", "max", "h2h", "home", None)
    assert mapping["Avg<2.5"] == ("open", "avg", "totals", "under", "2.5")
    assert mapping["AvgC<2.5"] == ("close", "avg", "totals", "under", "2.5")
    assert mapping["BFE>2.5"] == ("open", "betfair_ex", "totals", "over", "2.5")
    assert "B365H" not in mapping and "MaxH" not in mapping  # early-market prices: Pinnacle, average, exchange
    assert mapping["PCAHH"] == ("close", "pinnacle", "spreads", "home", "AHCh")
    assert mapping["PAHH"] == ("open", "pinnacle", "spreads", "home", "AHh")
    assert mapping["BFECAHA"] == ("close", "betfair_ex", "spreads", "away", "AHCh")
    # Every priced column of a real header that is mapped: closing exactly when the book has a C after it
    header = fd.parse_csv(fixture("fd_E0_2526.csv").decode("utf-8-sig"))[0].keys()
    for column in header:
        if column in mapping:
            closing = any(column.startswith(prefix + "C") for prefix in ("B365", "PS", "P", "Avg", "Max", "BFE")) \
                or column == "AHCh"
            assert (mapping[column][0] == "close") == closing, column

    rows = fd.parse_csv(fixture("fd_E0_2526.csv").decode("utf-8-sig"))
    item = fd.record(rows[0], "soccer_epl", 2025, "E0")
    start = hs.to_ts(item["game"]["start"])
    lines = {(line["kind"], line["book"], line["market"], line["side"]): line for line in item["lines"]}
    assert lines[("close", "avg", "h2h", "home")]["price"] == float(rows[0]["AvgCH"])
    assert lines[("open", "avg", "h2h", "home")]["price"] == float(rows[0]["AvgH"])
    assert lines[("close", "bet365", "totals", "over")]["price"] == float(rows[0]["B365C>2.5"])
    assert lines[("open", "avg", "totals", "over")]["price"] == float(rows[0]["Avg>2.5"])
    assert lines[("close", "max", "h2h", "away")]["price"] == float(rows[0]["MaxCA"])
    assert lines[("close", "avg", "spreads", "home")]["point"] == float(rows[0]["AHCh"])
    assert lines[("close", "avg", "spreads", "away")]["point"] == -float(rows[0]["AHCh"])
    assert lines[("open", "avg", "spreads", "home")]["point"] == float(rows[0]["AHh"])
    # Close: a minute before kick-off; the early market: the Friday 18:00 UK snapshot before a Friday 20:00 game
    assert lines[("close", "avg", "h2h", "home")]["as_of"] == start - 60
    opened = datetime.datetime.fromtimestamp(lines[("open", "avg", "h2h", "home")]["as_of"], fd.UK)
    assert (opened.weekday(), opened.hour) == (4, 18) and opened.timestamp() < start
    assert item["game"]["start"] == "2025-08-15T19:00:00Z"  # 20:00 UK summer time
    stats = {(s["team_side"], s["stat"]): s["value"] for s in item["stats"]}
    assert stats[("home", "corners")] == float(rows[0]["HC"]) and stats[("away", "yellow")] == float(rows[0]["AY"])


def test_open_snapshot_time_rule() -> None:
    def opened(day: Tuple[int, int, int], hour: int) -> Tuple[int, int]:
        start = datetime.datetime(*day, hour, 0, tzinfo=fd.UK).timestamp()
        moment = datetime.datetime.fromtimestamp(fd.open_as_of(start), fd.UK)
        return moment.weekday(), moment.day

    assert opened((2025, 8, 16), 15) == (4, 15)    # Saturday -> Friday
    assert opened((2025, 8, 18), 20) == (4, 15)    # Monday -> the Friday before
    assert opened((2025, 8, 26), 19) == (1, 26)    # Tuesday evening -> that Tuesday 18:00
    assert opened((2025, 8, 27), 19) == (1, 26)    # Wednesday -> Tuesday
    assert opened((2025, 8, 26), 17) == (4, 22)    # Tuesday 17:00 -> the Friday before (18:00 is too late)


def test_pinnacle_is_dropped_from_its_cut_off_and_the_sharp_baseline_moves_to_betfair(store: Store) -> None:
    old = fd.record(fd.parse_csv(fixture("fd_D1_1920.csv").decode())[0], "soccer_germany_bundesliga", 2019, "D1")
    assert {line["kind"] for line in old["lines"] if line["book"] == "pinnacle"} == {"open", "close"}
    rows = fd.parse_csv(fixture("fd_E0_2526.csv").decode("utf-8-sig"))
    assert rows[0].get("PSCH")  # the file still has Pinnacle's closing price ...
    new = fd.record(rows[0], "soccer_epl", 2025, "E0")
    assert not [line for line in new["lines"] if line["book"] == "pinnacle"]  # ... but it is not stored
    usa = [fd.record(row, "soccer_usa_mls", int(row["Season"]), "USA")
           for row in fd.parse_csv(fixture("fd_new_USA.csv").decode("utf-8-sig"))]
    by_day = {item["game"]["start"][:10]: item for item in usa if item}
    assert any(line["book"] == "pinnacle" for line in by_day["2025-05-03"]["lines"])  # before 2025-07-23
    assert not any(line["book"] == "pinnacle" for line in by_day["2025-09-01"]["lines"])  # after it
    assert fd.PINNACLE_UNRELIABLE_FROM == datetime.date(2025, 7, 23)

    db = db_of(store)
    for item in (old, new):
        hs.write_record(db, item, 1.0)
    reader = HistoryReader(store)
    bundesliga = next(reader.games("soccer_germany_bundesliga"))
    assert HistoryReader.sharp_line(bundesliga)["book"] == "pinnacle"
    epl = next(reader.games("soccer_epl"))
    sharp = HistoryReader.sharp_line(epl)
    assert sharp["book"] == "betfair_ex" and set(sharp["prices"]) == {"home", "draw", "away"}
    assert sharp["prices"]["home"] == float(rows[0]["BFECH"])
    assert abs(sum(HistoryReader.no_vig(sharp["prices"]).values()) - 1) < 1e-9
    totals = HistoryReader.sharp_line(epl, "totals")
    assert totals["point"] == 2.5 and set(totals["prices"]) == {"over", "under"}
    db.close()


def test_team_names_join_through_team_key_and_the_alias_table() -> None:
    assert fd.fd_team_key("soccer_epl", "Nott'm Forest") == team_key("soccer_epl", "Nottingham Forest")
    assert fd.fd_team_key("soccer_germany_bundesliga", "M'gladbach") == \
        team_key("soccer_germany_bundesliga", "Borussia Mönchengladbach")
    assert fd.fd_team_key("soccer_spain_la_liga", "Ath Madrid") == team_key("soccer_spain_la_liga", "Atlético Madrid")
    assert fd.fd_team_key("soccer_epl", "West Ham") == team_key("soccer_epl", "West Ham United")
    assert fd.fd_team_key("soccer_epl", "Man United") == team_key("soccer_epl", "Manchester United")
    assert nflverse.nfl_key("OAK") == nflverse.nfl_key("LV") == "LV" and nflverse.nfl_key("LA") == "LAR"
    assert team_key("baseball_mlb", "Minnesota Twins") == team_key("baseball_mlb", "MIN")


EVERTON_CSV = ("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HC,AC,AvgH,AvgD,AvgA,AvgCH,AvgCD,AvgCA\r\n"
               "E0,15/03/2025,15:00,Everton,West Ham,1,1,D,6,5,2.0,3.4,4.0,2.1,3.3,3.9\r\n")


def _epl_capture(store: Store) -> Dict[str, int]:
    summary = json.loads(fixture("espn_epl_summary.json"))
    row = {"league": "soccer_epl", "game_key": "espn:704565", "espn_id": "704565", "home": "Everton",
           "away": "West Ham United", "start": "2025-03-15T15:00:00Z", "swapped": False, "logged_at": at(2025, 3, 15, 14),
           "sources": [{"name": "Sharp", "kind": "sharp", "status": "ok", "as_of": "2025-03-15T14:50:00Z",
                        "p": {"home": 0.45, "draw": 0.28, "away": 0.27}}]}
    return capture_final(store, "soccer_epl", row, {"state": "final", "home": 1, "away": 1}, summary, now=at(2025, 3, 16))


@pytest.mark.parametrize("grader_first", [True, False])
def test_a_football_data_game_and_the_graders_capture_become_one_row(store: Store, grader_first: bool) -> None:
    clock = Clock(at(2026, 10, 28, 2))
    bf = backfill(store, clock, Transport(clock))
    record = fd.record(fd.parse_csv(EVERTON_CSV)[0], "soccer_epl", 2024, "E0")
    db = db_of(store)
    if grader_first:
        _epl_capture(store)
        assert bf.write(db, record, clock(), soccer=True) == 0  # written under the grader's key, not a new game
    else:
        assert bf.write(db, record, clock(), soccer=True) == 1
        _epl_capture(store)
    games = db.execute("SELECT game_key, home, away, home_score, source FROM hist_games").fetchall()
    assert len(games) == 1 and games[0][0] == "espn:704565"
    assert games[0][1] == ("Everton" if grader_first else "Everton")
    sources = {row[0] for row in db.execute("SELECT DISTINCT source FROM hist_lines WHERE game_key = 'espn:704565'")}
    assert sources == {"football-data", "live"}
    assert not db.execute("SELECT 1 FROM hist_lines WHERE game_key LIKE 'fd:%'").fetchall()
    stats = dict(db.execute("SELECT stat || ':' || source, value FROM hist_stats WHERE team_side = 'home'").fetchall())
    assert stats["corners:football-data"] == 6 and stats["corners:live"] == 6
    db.close()


# --- ESPN: the live-odds providers, current prices ---------------------------------------------------------------

def test_espn_live_odds_providers_and_current_prices_are_never_used() -> None:
    nfl = espn_hist.parse_odds(json.loads(fixture("espn_nfl_odds_live.json")))
    assert nfl["books"] == ["espn_bet"] and nfl["dropped"] == ["ESPN Bet - Live Odds"]
    raw = json.loads(fixture("espn_nfl_odds_live.json"))
    live = [item for item in raw["items"] if item["provider"]["id"] == "59"][0]
    live_total = float(live["current"]["total"]["american"])
    assert all(line["point"] != live_total for line in nfl["lines"] if line["market"] == "totals")
    nba = espn_hist.parse_odds(json.loads(fixture("espn_nba_odds.json")))
    assert "accuscore" in nba["dropped"] and set(nba["books"]) >= {"espn_bet", "draftkings"}
    raw = json.loads(fixture("espn_nba_odds.json"))
    espn_bet = [item for item in raw["items"] if item["provider"]["id"] == "58"][0]
    current_total = float(espn_bet["current"]["total"]["american"])
    close_total = float(espn_bet["close"]["total"]["american"])
    assert current_total != close_total  # ESPN BET's "current" (217.5) is noise; its close is 234.5
    totals = {line["point"] for line in nba["lines"] if line["market"] == "totals" and line["book"] == "espn_bet"}
    assert close_total in totals and current_total not in totals
    assert {line["kind"] for line in nba["lines"]} == {"open", "close"}
    assert all(line["lead"] > 0 for line in nba["lines"])
    assert espn_hist.book_name({"id": "999", "name": "Some Book Live Odds"}) is None
    assert espn_hist.book_name({"id": "46", "name": "Caesars"}) is None
    picks = [{"provider": {"id": "59", "name": "ESPN BET - Live Odds"},
              "moneyline": {"home": {"close": {"odds": "-3000"}}}}]
    assert espn_hist.parse_pickcenter(picks) == []


# --- no post-start line, ever ----------------------------------------------------------------------------------

def test_no_line_dated_at_or_after_the_start_is_ever_returned(store: Store) -> None:
    db = db_of(store)
    start = "2026-01-10T00:00:00Z"
    hs.upsert_game(db, {"league": "basketball_nba", "game_key": "espn:9", "start": start, "home": "BOS",
                        "away": "NY", "home_score": 100, "away_score": 99, "status": "final", "source": "espn"})
    ts = hs.to_ts(start)
    lines = [{"kind": "close", "book": "x", "market": "h2h", "side": "home", "price": 1.8, "as_of": ts - 60},
             {"kind": "close", "book": "late", "market": "h2h", "side": "home", "price": 1.1, "as_of": ts},
             {"kind": "close", "book": "inplay", "market": "h2h", "side": "home", "price": 1.01, "as_of": ts + 900},
             {"kind": "close", "book": "nodate", "market": "h2h", "side": "home", "price": 1.5, "as_of": None}]
    assert hs.add_lines(db, "espn:9", start, lines, "test") == 1
    # Even a row slipped in behind the helpers' back is never returned
    db.execute("INSERT INTO hist_lines VALUES ('espn:9', 'close', 'sneaky', 'h2h', 'away', NULL, 9.0, ?, 'x')",
               (ts + 1,))
    record = next(HistoryReader(store).games("basketball_nba"))
    assert [line["book"] for line in record["lines"]] == ["x"]
    assert all(line["as_of"] < record["start_ts"] for line in record["lines"])
    db.close()


def test_a_full_backfill_returns_only_pre_start_lines(store: Store) -> None:
    clock = Clock(at(2026, 10, 28, 2))
    report = backfill(store, clock, Transport(clock), per_night=400, seasons=1).run_night()
    assert report["stopped"] == "done", report
    reader = HistoryReader(store)
    seen = 0
    for league in ("soccer_epl", "soccer_usa_mls", "americanfootball_nfl", "basketball_nba"):
        for record in reader.games(league):
            for line in record["lines"]:
                assert line["as_of"] < record["start_ts"], (league, record["game_key"], line)
                seen += 1
    assert seen > 100


# --- budget, pace and the time window ------------------------------------------------------------------------

def test_the_nightly_budget_is_kept_across_runs_and_requests_are_paced(store: Store) -> None:
    clock = Clock(at(2026, 10, 28, 2))
    transport = Transport(clock)
    report = backfill(store, clock, transport, per_night=5).run_night()
    assert (report["ran"], report["requests"], report["used"], report["stopped"]) == (True, 5, 5, "budget")
    times = [ts for _url, _params, ts in transport.calls]
    assert len(times) == 5 and all(b - a >= 2.0 for a, b in zip(times, times[1:]))
    # A restart the same night spends nothing more
    again = backfill(store, clock, transport, per_night=5).run_night()
    assert (again["requests"], again["stopped"]) == (0, "budget") and len(transport.calls) == 5
    # A smaller budget passed in is the night's cap too
    assert backfill(store, clock, transport, per_night=50).run_night(budget=5)["requests"] == 0
    assert backfill(store, clock, transport, per_night=50).run_night(budget=8)["requests"] == 3
    # The next night has its own budget
    clock.ts = at(2026, 10, 29, 1, 30)
    nxt = backfill(store, clock, transport, per_night=5).run_night()
    assert nxt["requests"] == 5 and nxt["night"] == "2026-10-29"
    db = db_of(store)
    assert hs.budget_used(db, "2026-10-28") == 8 and hs.budget_used(db, "2026-10-29") == 5
    db.close()


def test_the_budget_setting(monkeypatch: pytest.MonkeyPatch, store: Store) -> None:
    assert per_night_budget({}) == PER_NIGHT_DEFAULT == 1500
    assert per_night_budget({"LEARN_BACKFILL_PER_NIGHT": "200"}) == 200
    assert per_night_budget({"LEARN_BACKFILL_PER_NIGHT": "lots"}) == 1500
    assert per_night_budget({"LEARN_BACKFILL_PER_NIGHT": "-3"}) == 1500
    monkeypatch.setenv("LEARN_BACKFILL_PER_NIGHT", "7")
    assert Backfill(store, TTLCache(), at(2026, 10, 28, 2)).per_night == 7


@pytest.mark.parametrize("hour,minute", [(0, 59), (6, 0), (12, 0), (23, 30)])
def test_nothing_is_fetched_outside_01_to_06(store: Store, hour: int, minute: int) -> None:
    clock = Clock(at(2026, 10, 28, hour, minute))
    transport = Transport(clock)
    report = backfill(store, clock, transport, per_night=100).run_night()
    assert report["ran"] is False and report["requests"] == 0 and transport.calls == []
    assert "01:00-06:00" in report["stopped"]


def test_the_run_stops_when_the_window_closes(store: Store) -> None:
    clock = Clock(at(2026, 10, 28, 5, 58))
    transport = Transport(clock, per_request=30.0)
    report = backfill(store, clock, transport, per_night=1000).run_night()
    assert report["stopped"] == "window"
    assert 1 <= report["requests"] <= 5
    assert all(datetime.datetime.fromtimestamp(ts, UTC).hour < 6 for _u, _p, ts in transport.calls)


def test_the_server_time_zone_decides_the_window(store: Store) -> None:
    clock = Clock(at(2026, 10, 28, 2))  # 02:00 UTC is 22:00 the day before in New York
    http = HistHttp(clock=clock, sleep=clock.sleep, transport=Transport(clock))
    eastern = Backfill(store, TTLCache(), clock, http=http, tz=datetime.timezone(datetime.timedelta(hours=-4)))
    assert eastern.in_window() is False and eastern.run_night()["ran"] is False


# --- order, seasons and resumability --------------------------------------------------------------------------

def test_newest_seasons_first_and_the_last_three(store: Store) -> None:
    clock = Clock(at(2026, 10, 28, 2))
    bf = backfill(store, clock, Transport(clock))
    units = bf.units()
    tiers = [unit.tier for unit in units]
    assert tiers == sorted(tiers)
    espn = {unit.name for unit in units if unit.source == "espn"}
    assert espn == {f"{league}:{season}" for league, seasons in {
        "basketball_nba": (2026, 2025, 2024), "icehockey_nhl": (2026, 2025, 2024),
        "basketball_ncaab": (2025, 2024, 2023), "americanfootball_ncaaf": (2026, 2025, 2024),
        "baseball_mlb": (2026, 2025, 2024)}.items() for season in seasons}
    soccer = sorted(unit.name for unit in units if unit.source == "football-data" and unit.name.startswith("E0:"))
    assert soccer == [f"E0:{season}" for season in range(2019, 2027)]  # football-data from 2019/20
    first = [unit.name for unit in units if unit.tier == 0]
    assert "games.csv" in first and "E0:2026" in first and "basketball_nba:2026" in first
    assert all(unit.name != "E0:2019" for unit in units[:len(first)])


def test_a_night_cut_short_carries_on_where_it_stopped(store: Store, tmp_path: Path) -> None:
    # One uninterrupted run ...
    whole = Store(str(tmp_path / "whole.db"))
    clock = Clock(at(2026, 10, 28, 2))
    once = Transport(clock)
    first = backfill(whole, clock, once, per_night=10_000, seasons=1).run_night()
    assert first["stopped"] == "done"
    # ... against many short ones: a fresh Backfill (a restart) after every 7 requests, a failed request on the way
    clock2 = Clock(at(2026, 10, 28, 2))
    pieces = Transport(clock2, fail=("/summary",))
    runs = 0
    while True:
        runs += 1
        report = backfill(store, clock2, pieces, per_night=10_000, seasons=1).run_night(budget=7 * runs)
        if report["stopped"] == "done":
            break
        assert report["stopped"] == "budget" and report["requests"] == 7 and report["used"] == 7 * runs
        assert runs < 500
    # The same work: one more request for the one that failed, nothing fetched twice otherwise
    assert len(pieces.calls) == len(once.calls) + 1
    assert sorted(pieces.urls()) == sorted(once.urls() + [pieces.urls("/summary")[0]])

    def summary(target: Store) -> Tuple[int, int, int]:
        db = db_of(target)
        found = tuple(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                      for table in ("hist_games", "hist_lines", "hist_stats"))
        db.close()
        return found  # type: ignore[return-value]

    assert summary(store) == summary(whole)
    # Progress per source survives the restarts
    found = progress(store)
    assert set(found) == {"nflverse", "football-data", "espn"}
    assert found["espn"]["games"] >= 4 and found["football-data"]["requests"] > 0
    assert found["espn"]["current"] >= 1  # the season being played is caught up, not done


def test_espn_games_get_results_box_stats_and_open_and_close_lines(store: Store) -> None:
    clock = Clock(at(2026, 10, 28, 2))
    transport = Transport(clock)
    report = backfill(store, clock, transport, per_night=500, seasons=1, sources=["espn"],
                      leagues=["basketball_nba"]).run_night()
    assert report["stopped"] == "done"
    # The 4 games came back in every window: each one's summary and odds were read once
    assert len(transport.urls("/summary")) == 4 and len(transport.urls("/odds")) == 4
    windows = [params["dates"] for url, params, _ts in transport.calls if url.endswith("/events")]
    assert windows[0] == "20261020-20261026" and windows[-1].startswith("20261001")
    records = list(HistoryReader(store).games("basketball_nba"))
    assert len(records) == 4
    record = records[0]
    assert record["result"] == {"home": 123, "away": 132} and record["stats"]["home"]["rebounds"] == 52
    assert record["stats"]["away"]["period_2"] == 45
    books = {line["book"] for line in record["lines"]}
    assert books == {"espn_bet", "draftkings", "median"}
    assert {line["kind"] for line in record["lines"]} == {"open", "close"}
    assert HistoryReader.sharp_line(record)["book"] == "median"
    assert record["season"] == 2026 and record["source"] == "espn"


def test_mlb_uses_one_schedule_request_a_season_and_no_box_scores(store: Store) -> None:
    clock = Clock(at(2025, 7, 5, 2))
    transport = Transport(clock)
    report = backfill(store, clock, transport, per_night=1000, seasons=1, sources=["espn"],
                      leagues=["baseball_mlb"]).run_night()
    assert report["stopped"] == "done"
    assert len(transport.urls("statsapi.mlb.com")) == 1
    assert transport.urls("/summary") == [] and transport.urls("boxscore") == []
    assert transport.urls("/scoreboard")[0].endswith("/mlb/scoreboard")
    assert transport.calls[0][1]["dates"] == "20250703"  # newest day first
    records = {r["espn_id"]: r for r in HistoryReader(store).games("baseball_mlb")}
    assert set(records) == {"401696211", "401696214", "401696209"}
    marlins = records["401696211"]
    assert marlins["result"] == {"home": 4, "away": 1} and marlins["info"]["mlb_pk"] == "777263"
    assert marlins["weather"]["actual"]["temp_c"] == round((72 - 32) * 5 / 9, 1)
    assert marlins["stats"]["home"]["period_1"] == 3
    assert {line["book"] for line in marlins["lines"]} == {"espn_bet", "median"}  # the live provider is dropped
    assert len(transport.urls("/odds")) == 3


# --- live capture ------------------------------------------------------------------------------------------------

def _nba_row(**changes: Any) -> Dict[str, Any]:
    start = "2026-10-25T21:00:00Z"
    row = {"league": "basketball_nba", "game_key": "espn:401909861", "espn_id": "401909861", "event_id": "odds1",
           "home": "Utah Jazz", "away": "Los Angeles Lakers", "start": start, "swapped": False,
           "logged_at": hs.to_ts(start) - 120,
           "sources": [
               {"name": "Sharp", "kind": "sharp", "status": "ok", "as_of": "2026-10-25T20:55:00Z",
                "p": {"home": 0.3, "away": 0.7}},
               {"name": "Polymarket", "kind": "prediction_market", "status": "thin", "as_of": "2026-10-25T20:50:00Z",
                "p": {"home": 0.32, "away": 0.68}},
               {"name": "Kalshi", "kind": "prediction_market", "status": "stale", "as_of": "2026-10-25T10:00:00Z",
                "p": {"home": 0.4, "away": 0.6}},
               {"name": "Model", "kind": "team_model", "status": "ok", "as_of": "2026-10-25T20:00:00Z",
                "p": {"home": 0.35, "away": 0.65}},
               {"name": "US books", "kind": "us_books", "status": "ok", "as_of": "2026-10-25T21:30:00Z",
                "p": {"home": 0.2, "away": 0.8}}]}
    row.update(changes)
    return row


def test_capture_writes_the_result_box_stats_and_last_pre_start_lines(store: Store) -> None:
    summary = json.loads(fixture("espn_nba_summary.json"))
    context = {"line": 237.5, "p_before": 0.52, "logged_at": hs.to_ts("2026-10-25T20:58:00Z")}
    written = capture_final(store, "basketball_nba", _nba_row(), {"state": "final", "home": 123, "away": 132},
                            summary, context=context, now=at(2026, 10, 26))
    assert written["games"] == 1 and written["lines"] > 0 and written["stats"] > 0
    record = HistoryReader(store).game("basketball_nba", "espn:401909861")
    assert record["result"] == {"home": 123, "away": 132} and record["source"] == "live"
    assert record["home"] == "Utah Jazz" and record["stats"]["home"]["rebounds"] == 52
    lines = {(line["book"], line["kind"], line["market"], line["side"]): line for line in record["lines"]}
    assert lines[("sharp", "close", "h2h", "home")]["price"] == round(1 / 0.3, 4)
    assert ("polymarket", "close", "h2h", "home") in lines  # thin sources are graded too
    assert ("kalshi", "close", "h2h", "home") not in lines and ("model", "close", "h2h", "home") not in lines
    assert ("us_books", "close", "h2h", "home") not in lines  # its as_of was after the start
    assert lines[("books_fair", "close", "totals", "over")]["point"] == 237.5
    assert lines[("draftkings", "close", "h2h", "home")]["price"] == 3.05
    assert lines[("draftkings", "open", "totals", "over")]["point"] == 236.5
    assert all(line["as_of"] < record["start_ts"] for line in record["lines"])
    # Again (the grader retried): nothing new
    again = capture_final(store, "basketball_nba", _nba_row(), {"state": "final", "home": 123, "away": 132}, summary,
                          context=context)
    assert again["games"] == 0
    db = db_of(store)
    assert db.execute("SELECT COUNT(*) FROM hist_games").fetchone()[0] == 1
    db.close()


def test_capture_turns_a_swapped_row_round_and_skips_games_not_over(store: Store) -> None:
    row = _nba_row(swapped=True, home="Los Angeles Lakers", away="Utah Jazz")
    capture_final(store, "basketball_nba", row, {"state": "final", "home": 123, "away": 132})
    record = HistoryReader(store).game("basketball_nba", "espn:401909861")
    assert record["home"] == "Utah Jazz"  # ESPN's orientation
    sharp = HistoryReader.sharp_line(record, order=("sharp",))
    assert sharp["prices"]["away"] == round(1 / 0.3, 4)  # the row's "home" chance was ESPN's away team's
    assert capture_final(store, "basketball_nba", _nba_row(espn_id="5"), {"state": "pending"}) == \
        {"games": 0, "lines": 0, "stats": 0, "weather": 0}
    void = capture_final(store, "basketball_nba", _nba_row(espn_id="6", game_key="espn:6"), {"state": "void"})
    assert void["games"] == 1
    assert list(HistoryReader(store).games("basketball_nba", status="void"))[0]["result"] is None


def test_capture_finds_the_logged_total_line(store: Store) -> None:
    start = hs.to_ts("2026-10-25T21:00:00Z")
    store.log_context([{"league": "basketball_nba", "game_key": "espn:401909861", "event_id": "odds1",
                        "espn_id": "401909861", "home": "Utah Jazz", "away": "Los Angeles Lakers",
                        "start": "2026-10-25T21:00:00Z", "line": 236.5, "p_before": 0.5, "p_after": 0.5,
                        "applied": 0.0, "reason": None, "adjustment": None}], now=start - 600)
    capture_final(store, "basketball_nba", _nba_row(), {"state": "final", "home": 123, "away": 132})
    record = HistoryReader(store).game("basketball_nba", "espn:401909861")
    totals = HistoryReader.sharp_line(record, "totals", order=("books_fair",))
    assert totals["point"] == 236.5 and totals["prices"] == {"over": 2.0, "under": 2.0}
    assert totals["as_of"] == start - 600


def test_capture_reads_the_logged_weather(store: Store) -> None:
    start = hs.to_ts("2026-10-25T21:00:00Z")
    store.log_weather([
        {"league": "basketball_nba", "game_key": "odds1", "fetched_at": start - 30 * 3600, "start": "x",
         "source": "Open-Meteo", "conditions": {"temp_c": 12.0, "wind_kmh": 20.0, "precip_mm": 0.4}, "sig": "a"},
        {"league": "basketball_nba", "game_key": "odds1", "fetched_at": start - 3600, "start": "x",
         "source": "MLB", "conditions": {"temp_c": 15.0, "announced": {"temp_f": 68, "wind_mph": 10}}, "sig": "b"}])
    capture_final(store, "basketball_nba", _nba_row(), {"state": "final", "home": 123, "away": 132})
    weather = HistoryReader(store).game("basketball_nba", "espn:401909861")["weather"]
    assert weather["forecast_24h"]["temp_c"] == 12.0 and weather["forecast_24h"]["precip_mm"] == 0.4
    assert weather["actual"]["temp_c"] == 20.0 and weather["actual"]["wind_kmh"] == round(10 * 1.609344, 1)


def test_nflverse_records_join_espn_and_carry_conditions() -> None:
    rows = nflverse.parse_csv(fixture("nfl_games.csv").decode())
    assert {row["season"] for row in rows} == {"2025", "2026"}  # 2019 on (2018 left out)
    records = [nflverse.record(row) for row in rows]
    assert records[-1] is None  # not played yet
    eagles = records[0]
    assert eagles["game"]["game_key"] == "espn:401772510" and eagles["game"]["home"] == "PHI"
    lines = {(line["market"], line["side"]): line for line in eagles["lines"]}
    assert lines[("spreads", "home")]["point"] == -8.5  # spread_line 8.5: the home team favoured by 8.5
    assert lines[("h2h", "home")]["price"] == round(1 + 100 / 425, 4)
    assert all(line["kind"] == "close" and line["book"] == "nflverse" for line in eagles["lines"])
    assert eagles["weather"][0]["kind"] == "actual" and eagles["weather"][0]["temp_c"] == round((75 - 32) * 5 / 9, 1)
    assert records[1]["weather"] == []  # a dome: no temperature


def test_a_failing_file_is_left_for_the_next_night_and_the_rest_carries_on(store: Store) -> None:
    clock = Clock(at(2026, 10, 28, 2))
    transport = Transport(clock, fail=("games.csv",))
    report = backfill(store, clock, transport, per_night=400, seasons=1).run_night()
    assert report["stopped"] == "done" and len(transport.urls("games.csv")) == 1  # not hammered
    assert transport.urls("football-data.co.uk") and transport.urls("/events")
    row = [r for r in hs.progress_rows(db_of(store), "nflverse")][0]
    assert row["errors"] == 1 and row["note"].startswith("failed")
    assert list(HistoryReader(store).games("americanfootball_nfl")) == []
    clock.ts = at(2026, 10, 29, 2)
    again = backfill(store, clock, transport, per_night=400, seasons=1).run_night()
    assert again["stopped"] == "done"
    assert len(list(HistoryReader(store).games("americanfootball_nfl"))) == 3
    assert [r for r in hs.progress_rows(db_of(store), "nflverse")][0]["note"] == ""


def test_ten_failures_in_a_row_stop_the_night(store: Store) -> None:
    clock = Clock(at(2026, 10, 28, 2))
    transport = Transport(clock, fail=tuple(["/events"] * 30))
    report = backfill(store, clock, transport, per_night=400, seasons=1, sources=["espn"]).run_night()
    assert report["stopped"] == "errors" and report["requests"] == 10
