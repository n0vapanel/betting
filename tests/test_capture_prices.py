"""v4 P1, contract C (p1:prices): free prices archived raw, and the slate's hook into the raw capture.

- capture/sources/espn_propbets.py: ESPN core propBets (DraftKings) for NBA, WNBA, NFL, NHL and MLB, every page as it
  came, in the RawFetch shape (capture/sources/__init__.py), stored by the worker at open, t24, t90 and close.
- capture/sources/kalshi_raw.py: Kalshi's confirmed corner, shot and save series (no key), every page as it came, on
  the process's Kalshi pacer with the real clock.
- data/slate.py: each round hands the slate's ESPN games to Capturer.schedule_from_slate on a thread of its own; it
  never blocks the slate and an error never costs it anything; without a capturer nothing changes.

No network: a fake http (PoliteHttp's get), or the real PoliteHttp with its _send replaced; fixtures from
tests/fixtures/espn (DraftKings props) and tests/fixtures/markets (Kalshi), read as bytes.
"""

from __future__ import annotations

import copy
import datetime
import json
import logging
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from telegram_betting_bot_simple.capture import raw_store
from telegram_betting_bot_simple.capture.http import HttpResponse, PoliteHttp, Refused, Unreachable
from telegram_betting_bot_simple.capture.raw_store import RawStore
from telegram_betting_bot_simple.capture.sources import RawFetch, espn_propbets, kalshi_raw
from telegram_betting_bot_simple.data.espn_client import ESPN_DAY_TZ, LEAGUES
from telegram_betting_bot_simple.data.markets import kalshi as markets_kalshi
from telegram_betting_bot_simple.data.odds_service import normalize_event
from telegram_betting_bot_simple.data.slate import CAPTURE_EVERY, CAPTURE_KEYS, CAPTURE_RETRY, SlateBuilder
from telegram_betting_bot_simple.data.store import Store

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROPS = FIXTURES / "espn"
MARKETS = FIXTURES / "markets"
T0 = 1_790_000_000.0  # a fake clock far from the real time
NBA, NFL, NHL, MLB, WNBA, EPL = ("basketball_nba", "americanfootball_nfl", "icehockey_nhl", "baseball_mlb",
                                 "basketball_wnba", "soccer_epl")


@pytest.fixture(autouse=True)
def roomy_disk(monkeypatch):
    """Plenty of room on raw.db's disk (capture/raw_store.py disk_room reads it), whatever this machine's is."""
    usage = (int(100e9), int(10e9), int(90e9))
    monkeypatch.setattr(raw_store.shutil, "disk_usage",
                        lambda path: type("Usage", (), {"total": usage[0], "used": usage[1], "free": usage[2]})())


class FakeHttp:
    """PoliteHttp's get() over ``answer(url, params) -> HttpResponse | exception``; every call is kept."""

    def __init__(self, answer: Callable[[str, Dict[str, str]], Any], clock: Callable[[], float] = lambda: T0) -> None:
        self.answer = answer
        self.calls: List[Dict[str, Any]] = []
        self.clock = clock
        self.sleep = lambda seconds: None

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, *, headers: Optional[Dict[str, str]] = None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        out = self.answer(url, dict(params or {}))
        if isinstance(out, BaseException):
            raise out
        return out


def ok(body: bytes, status: int = 200) -> HttpResponse:
    return HttpResponse(status, body, {"content-type": "application/json"})


def page_body(page: int, pages: int, items: int = 2) -> bytes:
    return json.dumps({"count": pages * items, "pageIndex": page, "pageSize": items, "pageCount": pages,
                       "items": [{"type": {"id": str(i), "name": "Total Points"}} for i in range(items)]}).encode()


def hosts(http: FakeHttp) -> set:
    return {urllib.parse.urlsplit(call["url"]).hostname for call in http.calls}


# --- ESPN propBets ------------------------------------------------------------------------------------------------------

def test_propbets_cover_the_five_leagues_with_their_espn_paths() -> None:
    assert espn_propbets.LEAGUES == (NBA, WNBA, NFL, NHL, MLB)
    for league in espn_propbets.LEAGUES:
        sport, path = LEAGUES[league]
        assert espn_propbets.url_for(league, 401) == (
            f"https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{path}/events/401/competitions/401/odds/100/propBets")
    assert espn_propbets.HORIZONS == ("open", "t24", "t90", "close")
    assert set(espn_propbets.HORIZONS) <= set(raw_store.HORIZONS)  # the store takes every one of them


@pytest.mark.parametrize("league, event_id", [(EPL, 401), ("basketball_ncaab", 401), (NBA, "401/../402"), (NBA, "abc"),
                                              (NBA, ""), (NBA, None), (NBA, "４０１")])
def test_a_league_without_props_or_an_id_that_isnt_a_number_is_refused_before_any_request(league, event_id) -> None:
    http = FakeHttp(lambda url, params: ok(page_body(1, 1)))
    with pytest.raises(ValueError):
        espn_propbets.propbets(http, league, event_id)
    with pytest.raises(ValueError):
        list(espn_propbets.propbets_all(http, league, event_id))
    assert http.calls == []


def test_one_page_is_a_raw_fetch_of_the_bytes_as_they_came() -> None:
    body = (PROPS / "props_nba_401909866.json").read_bytes()
    http = FakeHttp(lambda url, params: ok(body))
    raw = espn_propbets.propbets(http, NBA, "401909866")
    assert isinstance(raw, RawFetch)
    assert (raw.source, raw.endpoint, raw.status, raw.fetched_at) == ("espn_propbets", "propbets", 200, T0)
    assert raw.body == body  # untouched: archive only
    assert raw.params == {"limit": 500, "page": 1} and all(type(v) is int for v in raw.params.values())
    assert raw.url == espn_propbets.url_for(NBA, 401909866) + "?limit=500&page=1"
    assert http.calls == [{"url": espn_propbets.url_for(NBA, 401909866), "params": {"limit": "500", "page": "1"},
                           "headers": {}}]
    assert "401909866" not in json.dumps(raw.params)  # the event id is in the game key, never in params
    assert espn_propbets.game_key("401909866") == "espn:401909866"


def test_the_pager_reads_every_page_up_to_page_count_and_no_more() -> None:
    pages = {1: (PROPS / "props_mlb_401908044_page1.json").read_bytes(),
             2: (PROPS / "props_mlb_401908044_page2.json").read_bytes()}
    http = FakeHttp(lambda url, params: ok(pages[int(params["page"])]))
    fetched = list(espn_propbets.propbets_all(http, MLB, 401908044))
    assert [f.params for f in fetched] == [{"limit": 500, "page": 1}, {"limit": 500, "page": 2}]
    assert [f.body for f in fetched] == [pages[1], pages[2]]
    assert len(http.calls) == 2  # pageCount is 2: no third request
    single = FakeHttp(lambda url, params: ok((PROPS / "props_nba_401909866.json").read_bytes()))
    assert len(list(espn_propbets.propbets_all(single, NBA, 401909866))) == 1 and len(single.calls) == 1


def test_the_pager_stops_at_max_pages() -> None:
    http = FakeHttp(lambda url, params: ok(page_body(int(params["page"]), 9)))
    assert [f.params["page"] for f in espn_propbets.propbets_all(http, NFL, 401)] == [1, 2, 3, 4, 5]
    http = FakeHttp(lambda url, params: ok(page_body(int(params["page"]), 9)))
    assert len(list(espn_propbets.propbets_all(http, NFL, 401, max_pages=2))) == 2


@pytest.mark.parametrize("first, second, seen", [
    (ok(b'{"error": {"code": 404}}', 404), None, [404]),        # no props posted (yet): kept, nothing more asked
    (ok(b"<html>moved</html>"), None, [200]),                   # not JSON: kept for the record, the walk ends
    (ok(page_body(1, 3)), ok(b"oops", 500), [200, 500]),        # a failing page ends the walk; never raised
])
def test_an_http_error_or_odd_answer_is_kept_and_ends_the_walk(first, second, seen) -> None:
    answers = {1: first, 2: second}
    http = FakeHttp(lambda url, params: answers[int(params["page"])])
    fetched = list(espn_propbets.propbets_all(http, NHL, 401))
    assert [f.status for f in fetched] == seen and len(http.calls) == len(seen)


def test_a_connection_error_raises_unreachable() -> None:
    http = FakeHttp(lambda url, params: Unreachable("reset"))
    with pytest.raises(Unreachable):
        espn_propbets.propbets(http, NBA, 401)


def test_through_polite_http_the_price_budget_guard_stops_the_pager(monkeypatch) -> None:
    clock = {"t": T0}
    sleeps: List[float] = []
    http = PoliteHttp(clock=lambda: clock["t"], sleep=lambda s: (sleeps.append(s), clock.__setitem__("t", clock["t"] + s)))
    sent: List[Any] = []
    monkeypatch.setattr(http, "_send", lambda request: (sent.append(request), ok(page_body(len(sent), 4)))[1])
    asked: List[str] = []

    def allow(url: str) -> None:
        asked.append(url)
        if len(asked) > 2:
            raise Refused("the day's price budget is spent")

    with http.guard(allow), pytest.raises(Refused):
        list(espn_propbets.propbets_all(http, NBA, 401))
    assert len(sent) == 2 and len(asked) == 3  # the third page was refused before it was sent
    assert all(r.get_header("User-agent") for r in sent) and not any("odds-api" in r.full_url for r in sent)
    assert sent[0].full_url.endswith("/odds/100/propBets?limit=500&page=1")


def test_pages_are_stored_raw_and_found_again_with_the_same_params(tmp_path) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    pages = {1: (PROPS / "props_mlb_401908044_page1.json").read_bytes(),
             2: (PROPS / "props_mlb_401908044_page2.json").read_bytes()}
    http = FakeHttp(lambda url, params: ok(pages[int(params["page"])]))
    key = espn_propbets.game_key(401908044)
    ids = [raw.put(game_key=key, league=MLB, source=f.source, endpoint=f.endpoint, params=f.params, body=f.body,
                   state="pre", http_status=f.status, horizon="open", fetched_at=f.fetched_at)
           for f in espn_propbets.propbets_all(http, MLB, 401908044)]
    assert raw.latest(key, "espn_propbets", "propbets", {"limit": 500, "page": 2}).raw_bytes() == pages[2]
    # The same answer at the next horizon adds nothing; a changed one is a new row
    again = [raw.put(game_key=key, league=MLB, source=f.source, endpoint=f.endpoint, params=f.params, body=f.body,
                     state="pre", http_status=f.status, horizon="t24", fetched_at=f.fetched_at + 3600)
             for f in espn_propbets.propbets_all(http, MLB, 401908044)]
    assert again == ids
    moved = pages[1].replace(b'"value": 0.5', b'"value": 1.5') if b'"value": 0.5' in pages[1] else pages[1] + b" "
    new = raw.put(game_key=key, league=MLB, source="espn_propbets", endpoint="propbets", params={"limit": 500, "page": 1},
                  body=moved, state="pre", http_status=200, horizon="close", fetched_at=T0 + 7200)
    assert new not in ids and raw.latest(key, "espn_propbets", "propbets", {"limit": 500, "page": 1}).raw_bytes() == moved


# --- Kalshi ---------------------------------------------------------------------------------------------------------

def test_the_confirmed_series_are_the_default_and_each_names_a_league_espn_covers() -> None:
    assert kalshi_raw.DEFAULT_SERIES == ("KXEPLCORNERS", "KXUCLSOT", "KXUCLSAVE")
    assert {ticker: kalshi_raw.league_of(ticker) for ticker in kalshi_raw.DEFAULT_SERIES} == {
        "KXEPLCORNERS": EPL, "KXUCLSOT": "soccer_uefa_champs_league", "KXUCLSAVE": "soccer_uefa_champs_league"}
    assert all(kalshi_raw.league_of(t) in LEAGUES for t in kalshi_raw.DEFAULT_SERIES)
    # A ticker the owner adds finds its league from the winner series' prefix; an unknown one has none
    assert kalshi_raw.league_of("KXEPLBTTS") == EPL and kalshi_raw.league_of("KXUCLCORNERS") == "soccer_uefa_champs_league"
    assert kalshi_raw.league_of("KXNOSUCHTHING") is None and kalshi_raw.league_of("not a ticker") is None
    assert kalshi_raw.game_key("kxeplcorners") == "static:kalshi:KXEPLCORNERS"


def test_the_configs_default_is_the_confirmed_series() -> None:
    from telegram_betting_bot_simple import config
    assert kalshi_raw.parse_series(config.KALSHI_SERIES_DEFAULT) == kalshi_raw.DEFAULT_SERIES


def test_capture_kalshi_series_parses_and_an_empty_value_turns_kalshi_off(caplog) -> None:
    assert kalshi_raw.parse_series(None) == kalshi_raw.DEFAULT_SERIES
    assert kalshi_raw.parse_series("") == () and kalshi_raw.parse_series("  ,, ") == ()
    assert kalshi_raw.parse_series("none") == () == kalshi_raw.parse_series(" OFF ")
    with caplog.at_level(logging.WARNING):
        assert kalshi_raw.parse_series("kxeplcorners, KXUCLSOT KXEPLCORNERS,../etc,KX") == ("KXEPLCORNERS", "KXUCLSOT")
    assert "../etc" in caplog.text


def test_one_series_page_asks_for_every_open_event_with_markets_and_milestones_and_no_key() -> None:
    body = (MARKETS / "kalshi_epl_total.json").read_bytes()
    http = FakeHttp(lambda url, params: ok(body))
    raw = kalshi_raw.series(http, "KXEPLCORNERS")
    assert (raw.source, raw.endpoint, raw.status, raw.body, raw.fetched_at) == ("kalshi", "events", 200, body, T0)
    assert raw.params == {"status": "open", "limit": 200, "page": 1}
    [call] = http.calls
    assert call["url"] == "https://api.elections.kalshi.com/trade-api/v2/events"
    assert call["params"] == {"series_ticker": "KXEPLCORNERS", "status": "open", "with_nested_markets": "true",
                              "with_milestones": "true", "limit": "200"}
    assert call["headers"] == {}  # no key, no Authorization
    assert "key" not in raw.url.lower() and "KXEPLCORNERS" in raw.url


def test_the_series_pager_follows_the_cursor_and_keeps_it_out_of_params() -> None:
    def answer(url, params):
        cursor = params.get("cursor")
        nxt = {None: "c2", "c2": "c3", "c3": ""}[cursor]
        return ok(json.dumps({"events": [{"event_ticker": f"KXUCLSAVE-{cursor}"}], "milestones": [],
                              "cursor": nxt}).encode())

    http = FakeHttp(answer)
    fetched = list(kalshi_raw.series_all(http, "KXUCLSAVE"))
    assert [f.params for f in fetched] == [{"status": "open", "limit": 200, "page": n} for n in (1, 2, 3)]
    assert [c["params"].get("cursor") for c in http.calls] == [None, "c2", "c3"]
    http = FakeHttp(answer)
    assert len(list(kalshi_raw.series_all(http, "KXUCLSAVE", max_pages=2))) == 2


@pytest.mark.parametrize("answer, status", [
    (ok((MARKETS / "kalshi_too_many_requests.json").read_bytes(), 429), 429),
    (ok(b'{"error": {"code": "too_many_requests"}}'), 200),
    (ok(b"<html>busy</html>"), 200),
])
def test_a_throttled_or_odd_answer_is_kept_and_ends_the_walk(answer, status) -> None:
    http = FakeHttp(lambda url, params: answer)
    fetched = list(kalshi_raw.series_all(http, "KXEPLCORNERS"))
    assert [f.status for f in fetched] == [status] and len(http.calls) == 1


def test_a_bad_ticker_is_refused_before_any_request() -> None:
    http = FakeHttp(lambda url, params: ok(b"{}"))
    for ticker in ("", None, "EPLCORNERS", "KX/../../x", "KXEPL CORNERS"):
        with pytest.raises(ValueError):
            kalshi_raw.series(http, ticker)
        with pytest.raises(ValueError):
            list(kalshi_raw.series_all(http, ticker))
    assert http.calls == []


class RecordingPacer:
    def __init__(self) -> None:
        self.waits = 0

    def wait(self) -> None:
        self.waits += 1


def test_with_the_real_clock_kalshi_waits_on_the_markets_workers_own_pacer(monkeypatch) -> None:
    shared = RecordingPacer()
    monkeypatch.setattr(markets_kalshi, "_PROCESS_PACER", shared)
    body = (MARKETS / "kalshi_epl_total.json").read_bytes()
    real = FakeHttp(lambda url, params: ok(body), clock=time.time)
    kalshi_raw.series(real, "KXEPLCORNERS")
    wrapped = FakeHttp(lambda url, params: ok(body), clock=lambda: time.time())  # however the clock is passed
    list(kalshi_raw.series_all(wrapped, "KXUCLSOT"))
    assert shared.waits == 2
    # A test's fake clock isn't paced here (its own buckets are); an explicit pacer wins either way
    kalshi_raw.series(FakeHttp(lambda url, params: ok(body)), "KXEPLCORNERS")
    assert shared.waits == 2
    mine = RecordingPacer()
    kalshi_raw.series(real, "KXEPLCORNERS", pacer=mine)
    assert mine.waits == 1 and shared.waits == 2


def test_through_polite_http_kalshi_goes_at_half_a_request_a_second(monkeypatch) -> None:
    clock = {"t": T0}
    sleeps: List[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["t"] += seconds

    http = PoliteHttp(clock=lambda: clock["t"], sleep=sleep)
    pages = iter([b'{"events": [], "cursor": "x"}', b'{"events": [], "cursor": "y"}', b'{"events": [], "cursor": ""}'])
    monkeypatch.setattr(http, "_send", lambda request: ok(next(pages)))
    fetched = list(kalshi_raw.series_all(http, "KXEPLCORNERS"))
    assert [f.status for f in fetched] == [200, 200, 200] and [f.fetched_at for f in fetched] == [T0, T0 + 2, T0 + 4]
    assert sleeps == [2.0, 2.0] and http.requests == {kalshi_raw.HOST: 3}


def test_a_series_snapshot_is_stored_under_its_static_key(tmp_path) -> None:
    raw = RawStore(str(tmp_path / "raw.db"))
    body = (MARKETS / "kalshi_epl_total.json").read_bytes()
    [f] = list(kalshi_raw.series_all(FakeHttp(lambda url, params: ok(body)), "KXEPLCORNERS"))
    key = kalshi_raw.game_key("KXEPLCORNERS")
    raw.put(game_key=key, league=kalshi_raw.league_of("KXEPLCORNERS"), source=f.source, endpoint=f.endpoint,
            params=f.params, body=f.body, state="pre", http_status=f.status, horizon="t24", fetched_at=f.fetched_at)
    found = raw.latest(key, "kalshi", "events", {"status": "open", "limit": 200, "page": 1})
    assert found is not None and found.raw_bytes() == body and found.league == EPL


def test_no_price_fetch_touches_the_odds_api() -> None:
    def answer(url, params):
        if "propBets" in url:
            return ok(page_body(int(params["page"]), 2))
        return ok(b'{"events": [], "cursor": ""}')

    http = FakeHttp(answer)
    for league in espn_propbets.LEAGUES:
        list(espn_propbets.propbets_all(http, league, 401))
    for ticker in kalshi_raw.DEFAULT_SERIES:
        list(kalshi_raw.series_all(http, ticker))
    assert hosts(http) == {"sports.core.api.espn.com", "api.elections.kalshi.com"}


# --- the slate's hook -----------------------------------------------------------------------------------------------

NOW = datetime.datetime(2026, 10, 26, 18, 0, tzinfo=ESPN_DAY_TZ).timestamp()
TODAY = datetime.date(2026, 10, 26)


def at(hours: float) -> str:
    return datetime.datetime.fromtimestamp(NOW + hours * 3600, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def espn_game(event_id: str, home: str, away: str, start: str, state: str = "pre",
              name: str = "STATUS_SCHEDULED") -> Dict[str, Any]:
    def side(where: str, label: str, team_id: str) -> Dict[str, Any]:
        return {"homeAway": where, "score": "0", "team": {"id": team_id, "displayName": label,
                                                         "shortDisplayName": label.split()[-1],
                                                         "abbreviation": label[:3].upper()}}
    return {"id": event_id, "date": start, "competitions": [{
        "id": event_id, "date": start, "venue": {"fullName": "Arena", "address": {"city": "Boston"}},
        "status": {"period": 0, "displayClock": "0.0", "type": {"name": name, "state": state,
                                                                "completed": state == "post", "shortDetail": "7 PM"}},
        "competitors": [side("home", home, "1"), side("away", away, "2")]}]}


class FakeEspn:
    def __init__(self, boards: Dict[Any, Any]) -> None:
        self.boards = boards

    def fetch_scoreboard(self, league: str, day: datetime.date) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.boards.get((league, day), []))


class FakeOdds:
    odds_ttl = 7200
    reserve = 50

    def __init__(self, store: Store) -> None:
        self.store = store
        self.quota = {"remaining": 300, "used": 200}
        self.fetches: List[str] = []

    def sports(self) -> List[Dict[str, Any]]:
        return [{"key": NBA, "group": "Basketball", "title": "NBA", "active": True, "has_outrights": False},
                {"key": NHL, "group": "Ice Hockey", "title": "NHL", "active": True, "has_outrights": False},
                {"key": "cricket_odi", "group": "Cricket", "title": "ODI", "active": True, "has_outrights": False}]

    def league_games(self, league: str, refresh: bool = False) -> List[Dict[str, Any]]:
        self.fetches.append(league)
        return []


class RecordingCapturer:
    """Capturer.schedule_from_slate as the hook sees it: every call kept; ``hold`` blocks a call; ``fail`` raises."""

    def __init__(self) -> None:
        self.calls: List[List[Dict[str, Any]]] = []
        self.threads: List[str] = []
        self.hold: Optional[threading.Event] = None
        self.entered = threading.Event()
        self.fail: Optional[BaseException] = None

    def schedule_from_slate(self, games: List[Dict[str, Any]]) -> int:
        self.calls.append(copy.deepcopy(games))
        self.threads.append(threading.current_thread().name)
        self.entered.set()
        if self.hold is not None:
            assert self.hold.wait(20), "the test never let the hand-off go"
        if self.fail is not None:
            raise self.fail
        return len(games)


@pytest.fixture()
def store(tmp_path) -> Store:
    return Store(str(tmp_path / "betting.db"))


def boards(nba_state: str = "pre") -> FakeEspn:
    name = {"pre": "STATUS_SCHEDULED", "in": "STATUS_IN_PROGRESS", "post": "STATUS_FINAL"}[nba_state]
    return FakeEspn({(NBA, TODAY): [espn_game("401", "Boston Celtics", "New York Knicks", at(1), nba_state, name)],
                     (NHL, TODAY): [espn_game("402", "Boston Bruins", "Toronto Maple Leafs", at(2))]})


def builder(store: Store, espn: FakeEspn, clock: Dict[str, float], capturer: Any = None, odds: Any = None,
            **kwargs: Any) -> SlateBuilder:
    extra = {} if capturer is None else {"capturer": capturer}
    return SlateBuilder(odds or FakeOdds(store), espn, store, now=lambda: clock["t"], workers=2, **extra, **kwargs)


def save_odds(store: Store) -> None:
    """An odds-only cricket game (no ESPN id): it is on the slate, not handed to the capture."""
    event = normalize_event({"id": "odi1", "sport_key": "cricket_odi", "sport_title": "ODI", "commence_time": at(3),
                             "home_team": "India", "away_team": "Australia",
                             "bookmakers": [{"key": "dk", "title": "DraftKings", "markets": [
                                 {"key": "h2h", "outcomes": [{"name": "India", "price": 1.8},
                                                             {"name": "Australia", "price": 2.1}]}]}]})
    store.save_odds_snapshot("cricket_odi", [event], NOW)


def test_without_a_capturer_nothing_changes(store) -> None:
    clock = {"t": NOW}
    plain = builder(store, boards(), clock)
    assert plain.capturer is None
    plain.refresh()
    assert plain._capture_thread is None and not any(t.name == "slate-capture" for t in threading.enumerate())


def test_each_round_hands_the_slates_espn_games_to_the_capture_and_the_slate_is_unchanged(store, tmp_path) -> None:
    save_odds(store)
    clock = {"t": NOW}
    capturer = RecordingCapturer()
    hooked = builder(store, boards(), clock, capturer)
    hooked.refresh()
    hooked.wait_capture()
    plain = builder(Store(str(tmp_path / "other.db")), boards(), clock)
    save_odds(plain.store)
    plain.refresh()
    assert hooked.current().raw == plain.current().raw  # the hook changes nothing the slate serves
    [rows] = capturer.calls
    assert capturer.threads == ["slate-capture"]
    games = {g["id"]: g for g in json.loads(hooked.current().raw)["games"]}
    assert "cricket_odi:odi1" in games  # odds-only: on the slate, but no ESPN id to capture
    assert sorted(row["espn_id"] for row in rows) == ["401", "402"]
    for row in rows:
        assert set(row) <= set(CAPTURE_KEYS) | {"espn_league"}
        assert {"id", "league", "espn_id", "start", "status", "_ts", "espn_league"} <= set(row)
        assert row["espn_league"] == row["league"] and row["status"] == "pre"
        assert row["_ts"] == pytest.approx(NOW + (1 if row["espn_id"] == "401" else 2) * 3600)
        assert row["start"] == at(1 if row["espn_id"] == "401" else 2).replace("Z", ":00Z")
        assert isinstance(row["home"], dict) and row["home"]["name"]


def test_a_tennis_tournaments_games_name_their_espn_tour() -> None:
    game = {"id": "tennis_atp_paris:espn:187149", "league": "tennis_atp_paris", "espn_id": "187149", "status": "pre",
            "start": "2026-10-27T10:00:00Z", "_ts": NOW, "_time_ok": False, "_books": {"big": "x"}, "badges": ["live"]}
    row = SlateBuilder._capture_input(game)
    assert row["espn_league"] == "tennis_atp" and row["_time_ok"] is False
    assert "_books" not in row and "badges" not in row
    assert SlateBuilder._capture_input({"league": "mma_mixed_martial_arts", "espn_id": "1"})["espn_league"] == \
        "mma_mixed_martial_arts"


def test_a_capture_that_hangs_never_holds_up_the_slate(store) -> None:
    clock = {"t": NOW}
    capturer = RecordingCapturer()
    capturer.hold = threading.Event()
    slate = builder(store, boards(), clock, capturer)
    try:
        started = time.monotonic()
        slate.refresh()
        assert capturer.entered.wait(10)
        assert slate.current() is not None and time.monotonic() - started < 5
        # A round while the hand-off hangs: still at once, no second thread; the game's new status waits its turn
        espn = slate.espn
        espn.boards[(NBA, TODAY)] = [espn_game("401", "Boston Celtics", "New York Knicks", at(1), "in",
                                               "STATUS_IN_PROGRESS")]
        clock["t"] = NOW + 60
        started = time.monotonic()
        slate.refresh()
        assert time.monotonic() - started < 5 and len(capturer.calls) == 1
        assert sum(t.name == "slate-capture" for t in threading.enumerate()) == 1
    finally:
        capturer.hold.set()
    slate.wait_capture()
    assert len(capturer.calls) == 2
    assert {row["espn_id"]: row["status"] for row in capturer.calls[1]} == {"401": "live", "402": "pre"}


def test_a_failing_capture_costs_the_slate_nothing_and_is_tried_again_after_a_pause(store, caplog) -> None:
    clock = {"t": NOW}
    capturer = RecordingCapturer()
    capturer.fail = RuntimeError("raw.db is locked")
    slate = builder(store, boards(), clock, capturer)
    with caplog.at_level(logging.WARNING):
        slate.refresh()
        slate.wait_capture()
    assert slate.current() is not None and len(capturer.calls) == 1
    assert "raw capture couldn't take the slate's games" in caplog.text
    clock["t"] = NOW + CAPTURE_RETRY / 2
    slate.refresh()
    slate.wait_capture()
    assert len(capturer.calls) == 1  # inside the pause
    capturer.fail = None
    clock["t"] = NOW + CAPTURE_RETRY + 1
    slate.refresh()
    slate.wait_capture()
    assert len(capturer.calls) == 2


@pytest.mark.parametrize("capturer", [object(), type("Broken", (), {"schedule_from_slate": None})()])
def test_a_capturer_that_cant_take_games_is_logged_not_fatal(store, capturer, caplog) -> None:
    clock = {"t": NOW}
    slate = builder(store, boards(), clock, capturer)
    with caplog.at_level(logging.WARNING):
        slate.refresh()
        slate.wait_capture()
    assert slate.current() is not None and "raw capture" in caplog.text


def test_rows_that_cant_be_made_cost_the_slate_nothing(store, monkeypatch, caplog) -> None:
    clock = {"t": NOW}
    capturer = RecordingCapturer()
    slate = builder(store, boards(), clock, capturer)
    monkeypatch.setattr(SlateBuilder, "_capture_input", staticmethod(lambda game: 1 / 0))
    with caplog.at_level(logging.ERROR):
        slate.refresh()
    assert slate.current() is not None and capturer.calls == [] and slate._capture_thread is None
    assert "handing the games to the raw capture failed" in caplog.text


def test_unchanged_games_go_again_only_every_ten_minutes_and_a_change_at_once(store) -> None:
    clock = {"t": NOW}
    capturer = RecordingCapturer()
    slate = builder(store, boards(), clock, capturer)
    slate.refresh()
    slate.wait_capture()
    clock["t"] = NOW + 30
    slate.refresh()
    slate.wait_capture()
    assert len(capturer.calls) == 1  # nothing changed
    slate.espn.boards[(NBA, TODAY)] = [espn_game("401", "Boston Celtics", "New York Knicks", at(1), "post",
                                                 "STATUS_FINAL")]
    clock["t"] = NOW + 600  # the board comes due again
    slate.refresh()
    slate.wait_capture()
    assert len(capturer.calls) == 2 and {r["espn_id"]: r["status"] for r in capturer.calls[1]}["401"] == "final"
    clock["t"] = NOW + 600 + CAPTURE_EVERY + 1
    slate.refresh()
    slate.wait_capture()
    assert len(capturer.calls) == 3 and capturer.calls[2] == capturer.calls[1]


def test_the_hook_never_asks_the_odds_feed_for_anything(store) -> None:
    clock = {"t": NOW}
    odds = FakeOdds(store)
    capturer = RecordingCapturer()
    slate = builder(store, boards(), clock, capturer, odds=odds)
    slate.refresh()
    slate.wait_capture()
    plain_odds = FakeOdds(store)
    builder(store, boards(), clock, odds=plain_odds).refresh()
    assert odds.fetches == plain_odds.fetches == []
