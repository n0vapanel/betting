"""v4 P1, contract B: the raw capture's fetchers (capture/sources/*) and EspnClient.summary_raw/scoreboard_raw.

Every fetcher is driven against a fake http (PoliteHttp's ``get(url, params=None, *, headers=None) ->
HttpResponse``) serving the trimmed audit payloads in tests/fixtures/capture/ (manifest.json lists each one's live
URL and status) and small synthetic files. Nothing here touches the network: requests and PoliteHttp's send are
refused unless a test serves them.
"""

from __future__ import annotations

import dataclasses
import datetime
import gzip
import io
import json
import re
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import requests

from telegram_betting_bot_simple.capture import http as capture_http
from telegram_betting_bot_simple.capture import sources
from telegram_betting_bot_simple.capture.http import Backoff, HttpResponse, PoliteHttp, Refused, Unreachable
from telegram_betting_bot_simple.capture.sources import (RawFetch, bout_key, espn_core, espn_fightcenter, espn_key,
                                                          espn_site, football_data_raw, full_url, greco,
                                                          match_key, mcp, mlb_key, nflverse, parse_game_key,
                                                          scoreboard_key, static_key, static_unit, statsapi)
from telegram_betting_bot_simple.data import espn_client
from telegram_betting_bot_simple.data.cache import TTLCache, key_text, ttl_for
from telegram_betting_bot_simple.data.espn_client import (LEAN_DROP, LEAGUES, SCOREBOARD_ONLY, EspnClient,
                                                         EspnHttpError, lean_events, lean_summary)
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.learn.sources import football_data as learn_football_data
from telegram_betting_bot_simple.learn.sources.http import HttpError as LearnHttpError
from telegram_betting_bot_simple.learn.sources.http import Refused as LearnRefused

PACKAGE = Path(espn_client.__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "capture"
MANIFEST = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))
ENTRIES = {item["name"]: item for item in MANIFEST["fixtures"]}
NOW = datetime.datetime(2026, 11, 1, 12, 0, tzinfo=datetime.timezone.utc).timestamp()
LEAGUE_OF = {path: key for key, path in {**LEAGUES, **SCOREBOARD_ONLY}.items()}  # ESPN (sport, path) -> league key
REAL_SEND = PoliteHttp._send  # the autouse fixture refuses it; one test puts it back over a fake opener


def fixture(name: str) -> bytes:
    return (FIX / ENTRIES[name]["file"]).read_bytes()


class Clock:
    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeHttp:
    """PoliteHttp's interface over canned answers keyed by the full URL (the query as PoliteHttp appends it). An
    answer is (status, body[, headers]), an exception to raise, or a function of (url, params) giving either."""

    def __init__(self, answers: Optional[Dict[str, Any]] = None, *, default: Any = (404, b"not served"),
                 clock: Optional[Clock] = None) -> None:
        self.answers = dict(answers or {})
        self.default = default
        self.clock = clock or Clock()
        self.calls: List[Tuple[str, Dict[str, Any], Optional[Dict[str, str]]]] = []

    def get(self, url: str, params: Any = None, *, headers: Any = None) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers) if headers else None))
        answer = self.answers.get(full_url(url, params), self.default)
        if callable(answer):
            answer = answer(url, params)
        if isinstance(answer, BaseException):
            raise answer
        status, body = answer[0], answer[1]
        return HttpResponse(int(status), body, dict(answer[2]) if len(answer) > 2 else {})

    @property
    def urls(self) -> List[str]:
        return [full_url(url, params) for url, params, _ in self.calls]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a test of the capture sources reached for the network")

    monkeypatch.setattr(requests.sessions.Session, "request", refuse)
    monkeypatch.setattr(PoliteHttp, "_send", refuse)


# --- the contract ------------------------------------------------------------------------------------------------


def test_rawfetch_is_the_contract() -> None:
    assert [field.name for field in dataclasses.fields(RawFetch)] == [
        "source", "endpoint", "params", "url", "status", "body", "fetched_at"]
    raw = RawFetch("espn_site", "summary", {}, "https://x.test/summary?event=1", 200, b'{"a": [1]}', 5.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        raw.status = 500  # type: ignore[misc]
    assert raw.ok and raw.json() == {"a": [1]}
    assert not RawFetch("espn_site", "summary", {}, "u", 404, b"", 5.0).ok
    with pytest.raises(ValueError):
        RawFetch("espn_site", "summary", {}, "u", 200, b"<html>", 5.0).json()
    assert set(sources.SOURCES) == {"espn_site", "espn_core", "espn_fightcenter", "mlb_statsapi", "nflverse",
                                    "greco1899", "mcp", "football_data", "espn_propbets", "kalshi"}
    assert set(sources.ENDPOINTS) == {"summary", "scoreboard", "core_plays", "core_status", "core_competitor_stats",
                                      "core_events", "fightcenter", "feed_live", "pbp_season", "officials", "csv",
                                      "propbets", "events"}


def test_game_keys_follow_section_3_3() -> None:
    assert espn_key(401909278) == "espn:401909278"
    assert bout_key("401911632") == "espn-bout:401911632"
    assert match_key(187149) == "espn-match:187149"
    assert mlb_key(777079) == "mlb:777079"
    assert static_key("greco1899", "ufc_fight_results.csv") == "static:greco1899:ufc_fight_results.csv"
    assert scoreboard_key("tennis_atp", "20261031") == "static:espn_site:scoreboard:tennis_atp:20261031"
    assert espn_fightcenter.card_key(600061267) == "espn:600061267"  # a UFC card is its ESPN event
    for key, parsed in (("espn:401", ("espn", "401")), ("espn-bout:9", ("espn-bout", "9")),
                        ("espn-match:7", ("espn-match", "7")), ("mlb:777079", ("mlb", "777079")),
                        ("static:nflverse:pbp:2024", ("static", "nflverse:pbp:2024"))):
        assert parse_game_key(key) == parsed
    for bad in ("", "odds:abc", "espn", "espn:", "nonsense:1"):
        with pytest.raises(ValueError):
            parse_game_key(bad)
    for bad in ("", None, "4 01", " "):
        with pytest.raises(ValueError):
            espn_key(bad)


def test_static_units_are_the_stores_table() -> None:
    http = FakeHttp(default=(200, b"x"))
    expected = [
        (nflverse.pbp_season(http, 2024), "static:nflverse:pbp:2024", "americanfootball_nfl"),
        (nflverse.officials(http), "static:nflverse:officials", "americanfootball_nfl"),
        (greco.csv_file(http, "ufc_fight_results.csv"), "static:greco1899:ufc_fight_results.csv",
         "mma_mixed_martial_arts"),
        (mcp.csv_file(http, "charting-m-matches.csv"), "static:mcp:charting-m-matches.csv", "tennis_atp"),
        (mcp.csv_file(http, "charting-w-matches.csv"), "static:mcp:charting-w-matches.csv", "tennis_wta"),
        (football_data_raw.season_csv(http, "E0", 2024), "static:football_data:E0:2425", "soccer_epl"),
        (football_data_raw.season_csv(http, "SC0", "2526"), "static:football_data:SC0:2526", "soccer_spl"),
        (football_data_raw.season_csv(http, "USA"), "static:football_data:USA", "soccer_usa_mls"),
    ]
    for raw, key, league in expected:
        assert static_unit(raw) == (key, league)
    # each module's own helpers agree
    assert nflverse.pbp_key(2024) == "static:nflverse:pbp:2024" and nflverse.OFFICIALS_KEY.endswith(":officials")
    assert greco.game_key("ufc_fight_stats.csv") == "static:greco1899:ufc_fight_stats.csv"
    assert mcp.game_key("charting-m-stats-Overview.csv") == "static:mcp:charting-m-stats-Overview.csv"
    assert football_data_raw.game_key("E0", 2024) == football_data_raw.game_key("E0", "2425")
    assert football_data_raw.league_of("N1") == "soccer_netherlands_eredivisie"
    with pytest.raises(ValueError):
        static_unit(espn_site.summary(http, "basketball_nba", 1))


# --- one GET: statuses are kept, connection errors are Unreachable, capture's own errors pass through -------------


def test_a_fetch_never_raises_on_an_http_status() -> None:
    for status in (200, 304, 400, 404, 429, 500, 503):
        http = FakeHttp(default=(status, b'{"code": %d}' % status))
        raw = espn_site.summary(http, "basketball_nba", "401")
        assert raw.status == status and raw.body == b'{"code": %d}' % status
    # an http that raises urllib's HTTPError (rather than answering) still gives its status and body
    def error(url: str, params: Any) -> Any:
        return urllib.error.HTTPError(url, 503, "busy", {}, io.BytesIO(b"try later"))  # type: ignore[arg-type]

    raw = espn_site.summary(FakeHttp(default=error), "basketball_nba", "401")
    assert (raw.status, raw.body) == (503, b"try later")


def test_connection_errors_and_timeouts_raise_unreachable() -> None:
    for exc in (ConnectionResetError("reset"), TimeoutError("timed out"), urllib.error.URLError("no route"),
                OSError("network down")):
        with pytest.raises(Unreachable) as caught:
            espn_core.status(FakeHttp(default=exc), "mma_mixed_martial_arts", 600061267, 401911632)
        assert caught.value.__cause__ is exc


def _streams(http: FakeHttp) -> List[Any]:
    """Every fetcher and pager, as a thunk that drives it to the end."""
    return [
        lambda: espn_site.summary(http, "basketball_nba", "401"),
        lambda: espn_site.scoreboard(http, "basketball_nba", datetime.date(2026, 10, 30)),
        lambda: list(espn_site.scoreboard_days(http, "soccer_epl", datetime.date(2026, 10, 24),
                                               datetime.date(2026, 10, 30))),
        lambda: espn_core.plays(http, "soccer_epl", "401878765", page=1),
        lambda: list(espn_core.plays_all(http, "soccer_epl", "401878765")),
        lambda: espn_core.status(http, "mma_mixed_martial_arts", 600061267, 401911632),
        lambda: espn_core.competitor_stats(http, "mma_mixed_martial_arts", 600061267, 401911632, 3949584),
        lambda: list(espn_core.events_all(http, "basketball_nba", dates="20240101-20240107")),
        lambda: espn_fightcenter.card(http, 600061267),
        lambda: statsapi.feed_live(http, 777079),
        lambda: nflverse.pbp_season(http, 2024),
        lambda: nflverse.officials(http),
        lambda: list(greco.csvs(http)),
        lambda: list(mcp.csvs(http)),
        lambda: football_data_raw.season_csv(http, "E0", 2024),
    ]


@pytest.mark.parametrize("make", [
    lambda: Unreachable("site.api.espn.com: connection refused"),
    lambda: Backoff("site.api.espn.com", NOW + 600, "3 failures in a row"),
    lambda: Refused("the day's capture budget is spent"),
])
def test_capture_http_errors_pass_through_every_fetcher_unchanged(make: Any) -> None:
    for index in range(len(_streams(FakeHttp()))):
        exc = make()
        http = FakeHttp(default=exc)
        with pytest.raises(type(exc)) as caught:
            _streams(http)[index]()
        assert caught.value is exc  # the same object: a Backoff stays a Backoff, a Refused is never a status
        assert type(caught.value) is type(exc)


def test_a_pager_stops_at_a_refusal_after_keeping_what_it_had() -> None:
    page = json.loads(fixture("core_epl_plays"))
    refusal = Refused("the night's budget is spent")
    base = "https://sports.core.api.espn.com/v2/sports/soccer/leagues/eng.1/events/401878765/competitions/401878765"
    http = FakeHttp({f"{base}/plays?limit=300&page=1": (200, json.dumps(dict(page, pageIndex=1)).encode()),
                     f"{base}/plays?limit=300&page=2": refusal})
    got = []
    with pytest.raises(Refused) as caught:
        for raw in espn_core.plays_all(http, "soccer_epl", "401878765"):
            got.append(raw)
    assert caught.value is refusal and [raw.params["page"] for raw in got] == [1]


def test_gzip_transfer_encoding_is_decoded_but_a_gz_file_is_kept() -> None:
    plain = b'{"code":400,"message":"Failed to get events endpoint."}'
    http = FakeHttp(default=(400, gzip.compress(plain), {"content-encoding": "gzip"}))
    raw = espn_site.scoreboard(http, "basketball_nba", dates="20240101-20240107")
    assert raw.body == plain and raw.json()["code"] == 400
    pbp = fixture("syn_play_by_play_2024.csv")
    raw = nflverse.pbp_season(FakeHttp(default=(200, pbp, {"content-type": "application/octet-stream"})), 2024)
    assert raw.body == pbp and raw.body[:2] == b"\x1f\x8b"  # a .csv.gz stays a gzip file


def test_fetched_at_is_the_https_clock() -> None:
    http = FakeHttp(default=(200, b"{}"), clock=Clock(NOW + 42))
    assert espn_site.summary(http, "basketball_nba", "1").fetched_at == NOW + 42

    class Private:
        _clock = staticmethod(lambda: 7.0)

        def get(self, url: str, params: Any = None, *, headers: Any = None) -> HttpResponse:
            return HttpResponse(200, b"{}", {})

    assert espn_site.summary(Private(), "basketball_nba", "1").fetched_at == 7.0


class FakeOpener:
    """urllib's opener as PoliteHttp uses it (``open(request, timeout)``), answering by full URL."""

    class Answer:
        def __init__(self, status: int, body: bytes, headers: Dict[str, str]) -> None:
            self.status, self.headers, self._body = status, headers, io.BytesIO(body)

        def read(self, size: int = -1) -> bytes:
            return self._body.read(size)

        def getcode(self) -> int:
            return self.status

        def __enter__(self) -> "FakeOpener.Answer":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

    def __init__(self, answers: Dict[str, Tuple[int, bytes, Dict[str, str]]]) -> None:
        self.answers = answers
        self.sent: List[str] = []

    def open(self, request: Any, timeout: Any = None) -> "FakeOpener.Answer":
        self.sent.append(request.full_url)
        status, body, headers = self.answers[request.full_url]
        if status >= 400:
            raise urllib.error.HTTPError(request.full_url, status, "error", headers,  # type: ignore[arg-type]
                                         io.BytesIO(body))
        return self.Answer(status, body, headers)


def test_the_fetchers_work_through_the_real_polite_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through p1:store's PoliteHttp (only urllib's opener faked): the URL it sends is the one the RawFetch records,
    fetched_at is its clock's, a gzip answer arrives decoded and an HTTP error comes back as a status."""
    clock = Clock()
    nba, ufc = fixture("sum_nba"), fixture("sum_ufc")
    opener = FakeOpener({ENTRIES["sum_nba"]["url"]: (200, gzip.compress(nba), {"Content-Encoding": "gzip"}),
                         ENTRIES["sum_ufc"]["url"]: (404, ufc, {"Content-Type": "text/plain"})})
    monkeypatch.setattr(PoliteHttp, "_send", REAL_SEND)
    monkeypatch.setattr(capture_http, "build_opener", lambda: opener)
    polite = PoliteHttp(clock=clock, sleep=lambda seconds: setattr(clock, "now", clock.now + seconds))
    raw = espn_site.summary(polite, "basketball_nba", "401909278")
    assert opener.sent == [raw.url] == [ENTRIES["sum_nba"]["url"]]
    assert (raw.status, raw.body, raw.fetched_at) == (200, nba, clock.now)
    raw = espn_site.summary(polite, "mma_mixed_martial_arts", "600061267")
    assert (raw.status, raw.body, raw.url) == (404, ufc, ENTRIES["sum_ufc"]["url"])


# --- ESPN site: summaries and scoreboards ------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(n for n in ENTRIES if n.startswith("sum_")))
def test_summary_asks_the_audits_url_and_keeps_the_body_whole(name: str) -> None:
    entry = ENTRIES[name]
    sport, path, event = re.search(r"/sports/([^/]+)/([^/]+)/summary\?event=([\w-]+)$", entry["url"]).groups()
    http = FakeHttp({entry["url"]: (entry["status"], fixture(name))})
    raw = espn_site.summary(http, LEAGUE_OF[(sport, path)], event)
    assert raw.url == entry["url"]
    assert (raw.source, raw.endpoint, raw.params) == ("espn_site", "summary", {})
    assert (raw.status, raw.body) == (entry["status"], fixture(name))  # MMA 404 and tennis 400 come back as such
    assert http.calls == [(f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{path}/summary",
                           {"event": event}, None)]
    if raw.ok:  # untrimmed: everything lean_summary would drop is still there
        assert set(raw.json()) - set(lean_summary(raw.json())), "the raw summary has more than the lean parts"


@pytest.mark.parametrize("name", sorted(n for n in ENTRIES if n.startswith("sb_") and n != "sb_range_400"))
def test_scoreboard_sends_the_lean_clients_query_and_keeps_every_key(name: str) -> None:
    entry = ENTRIES[name]
    sport, path = re.search(r"/sports/([^/]+)/([^/]+)/scoreboard", entry["url"]).groups()
    league = LEAGUE_OF[(sport, path)]
    found = re.search(r"dates=(\d{8})", entry["url"])
    base = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{path}/scoreboard"
    if found:
        url = f"{base}?dates={found.group(1)}&limit=300"
        raw = espn_site.scoreboard(FakeHttp({url: (200, fixture(name))}), league,
                                   datetime.datetime.strptime(found.group(1), "%Y%m%d").date())
        assert raw.params == {"dates": found.group(1)}
    else:  # the audit read tennis's default board (no dates)
        url = f"{base}?limit=300"
        raw = espn_site.scoreboard(FakeHttp({url: (200, fixture(name))}), league)
        assert raw.params == {}
    assert (raw.url, raw.source, raw.endpoint, raw.status) == (url, "espn_site", "scoreboard", 200)
    assert raw.body == fixture(name)
    assert lean_events(raw.json()["events"]) != raw.json()["events"]  # the LEAN_DROP keys are in the raw board


def test_the_ufc_board_keeps_each_bouts_details_raw() -> None:
    raw = espn_site.scoreboard(FakeHttp(default=(200, fixture("sb_ufc"))), "mma_mixed_martial_arts",
                               datetime.date(2026, 10, 24))
    bouts = raw.json()["events"][0]["competitions"]
    assert "details" in LEAN_DROP and all("details" in bout for bout in bouts)
    assert raw.json()["leagues"][0]["calendar"], "the season's cards (the backfill reads UFC from the calendar)"


def test_scoreboard_options() -> None:
    http = FakeHttp(default=(200, b'{"events": []}'))
    raw = espn_site.scoreboard(http, "basketball_ncaab", "2026-11-03")
    assert raw.url.endswith("/basketball/mens-college-basketball/scoreboard?dates=20261103&limit=300&groups=50")
    assert raw.params == {"dates": "20261103"}  # the constant query parts never tell two boards apart
    raw = espn_site.scoreboard(http, "basketball_nba", dates="20261030", params={"seasontype": "2"})
    assert raw.params == {"dates": "20261030", "seasontype": "2"} and raw.url.endswith("&seasontype=2")
    raw = espn_site.scoreboard(http, ("basketball", "nba"), datetime.datetime(2026, 10, 30, 23, 0))
    assert raw.params == {"dates": "20261030"}
    for bad in (dict(day=datetime.date(2026, 10, 30), dates="20261030"), dict(dates="2026-10-30"),
                dict(dates="20261007-20261001"), dict(day="20261332")):
        with pytest.raises(ValueError):
            espn_site.scoreboard(http, "basketball_nba", **bad)
    with pytest.raises(KeyError):
        espn_site.scoreboard(http, "soccer_nowhere_league")


def test_a_date_range_is_asked_as_given_and_espns_400_comes_back() -> None:
    entry = ENTRIES["sb_range_400"]
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=20240101-20240107&limit=300"
    raw = espn_site.scoreboard(FakeHttp({url: (400, fixture("sb_range_400"))}), "basketball_nba",
                               dates="20240101-20240107")
    assert (raw.status, raw.params) == (400, {"dates": "20240101-20240107"}) and entry["status"] == 400
    assert raw.json() == {"code": 400, "message": "Failed to get events endpoint."}


def test_scoreboard_days_reads_a_window_one_board_a_day_newest_first() -> None:
    http = FakeHttp(default=(200, b'{"events": []}'))
    first, last = datetime.date(2023, 10, 1), datetime.date(2023, 10, 7)
    boards = list(espn_site.scoreboard_days(http, "soccer_epl", first, last))
    assert [raw.params["dates"] for raw in boards] == [f"202310{d:02d}" for d in range(7, 0, -1)]
    assert len(http.calls) == 7 and all("-" not in raw.params["dates"] for raw in boards)
    assert [scoreboard_key("soccer_epl", raw.params["dates"]) for raw in boards][0] == \
        "static:espn_site:scoreboard:soccer_epl:20231007"
    oldest = list(espn_site.scoreboard_days(http, "soccer_epl", first, last, newest_first=False))
    assert [raw.params["dates"] for raw in oldest] == sorted(raw.params["dates"] for raw in boards)
    assert list(espn_site.scoreboard_days(http, "soccer_epl", last, first)) == []
    lazy = espn_site.scoreboard_days(http, "soccer_epl", first, last)
    next(lazy)
    assert len(http.calls) == 7 + 7 + 1  # a request per board taken, no more


# --- ESPN core: the plays pager, status, competitor statistics, event lists -------------------------------------

CORE_EPL = ("https://sports.core.api.espn.com/v2/sports/soccer/leagues/eng.1/events/401878765/competitions/"
            "401878765/plays")


def _plays_world(pages: int, **overrides: Any) -> FakeHttp:
    page = json.loads(fixture("core_epl_plays"))
    answers: Dict[str, Any] = {}
    for number in range(1, pages + 1):
        body = dict(page, pageIndex=number, pageCount=pages, items=page["items"][:2])
        answers[f"{CORE_EPL}?limit=300&page={number}"] = (200, json.dumps(body).encode())
    for number, answer in overrides.items():
        answers[f"{CORE_EPL}?limit=300&page={number[1:]}"] = answer
    return FakeHttp(answers)


def test_core_plays_page_url_and_params() -> None:
    entry = ENTRIES["core_epl_plays"]
    http = FakeHttp({f"{entry['url']}&page=1": (200, fixture("core_epl_plays"))})
    raw = espn_core.plays(http, "soccer_epl", "401878765", page=1)
    assert raw.url == f"{entry['url']}&page=1"  # the audit's URL (?limit=300), page 1 asked for explicitly
    assert (raw.source, raw.endpoint, raw.params) == ("espn_core", "core_plays", {"page": 1, "limit": 300})
    assert espn_core.page_count(raw) == 5 and raw.json()["count"] == 1481  # 1,481 items is 5 pages [R §1]
    with pytest.raises(ValueError):
        espn_core.plays(http, "soccer_epl", "401878765", page=0)


def test_plays_all_walks_every_page_and_stops_at_the_last() -> None:
    http = _plays_world(5)
    pages = list(espn_core.plays_all(http, "soccer_epl", "401878765"))
    assert [raw.params for raw in pages] == [{"page": n, "limit": 300} for n in range(1, 6)]
    assert len(http.calls) == 5 and all(raw.ok for raw in pages)


def test_plays_all_stops_at_max_pages_a_failure_or_an_unreadable_page() -> None:
    assert len(list(espn_core.plays_all(_plays_world(5), "soccer_epl", "401878765", max_pages=3))) == 3
    failed = list(espn_core.plays_all(_plays_world(5, p3=(500, b"oops")), "soccer_epl", "401878765"))
    assert [raw.status for raw in failed] == [200, 200, 500]  # the failure is yielded: the caller sees it
    odd = list(espn_core.plays_all(_plays_world(5, p2=(200, b"<html>")), "soccer_epl", "401878765"))
    assert len(odd) == 2
    empty = json.dumps({"pageIndex": 2, "pageCount": 5, "items": []}).encode()
    assert len(list(espn_core.plays_all(_plays_world(5, p2=(200, empty)), "soccer_epl", "401878765"))) == 2
    resumed = list(espn_core.plays_all(_plays_world(5), "soccer_epl", "401878765", first_page=4))
    assert [raw.params["page"] for raw in resumed] == [4, 5]
    with pytest.raises(ValueError):
        list(espn_core.plays_all(_plays_world(1), "soccer_epl", "401878765", max_pages=0))


@pytest.mark.parametrize("name, call", [
    ("core_ufc_status", lambda http: espn_core.status(http, "mma_mixed_martial_arts", "600061267", "401911632")),
    ("core_ufc_stats", lambda http: espn_core.competitor_stats(http, "mma_mixed_martial_arts", 600061267, 401911632,
                                                                3949584)),
    ("core_atp_qf_stats", lambda http: espn_core.competitor_stats(http, "tennis_atp", "10-2026", 185201, 2383)),
])
def test_core_status_and_competitor_stats_ask_the_audits_urls(name: str, call: Any) -> None:
    entry = ENTRIES[name]
    raw = call(FakeHttp({entry["url"]: (entry["status"], fixture(name))}))
    assert raw.url == entry["url"] and raw.status == entry["status"] and raw.body == fixture(name)
    assert raw.source == "espn_core"
    if name == "core_ufc_status":
        assert (raw.endpoint, raw.params) == ("core_status", {})
        assert raw.json()["result"]["name"] == "kotko"  # the method
    else:
        assert raw.endpoint == "core_competitor_stats" and raw.params == {"competitor": entry["url"].split("/")[-2]}


def test_core_events_lists_a_range_of_event_ids() -> None:
    url = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events"
    ref = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{}?lang=en&region=us"
    pages = {1: ["401585089", "401585090"], 2: ["401585091", "401585089"]}
    answers = {f"{url}?dates=20240101-20240107&limit=2&page={n}": (200, json.dumps(
        {"count": 4, "pageIndex": n, "pageCount": 2, "items": [{"$ref": ref.format(i)} for i in ids]}).encode())
        for n, ids in pages.items()}
    http = FakeHttp(answers)
    got = list(espn_core.events_all(http, "basketball_nba", dates="20240101-20240107", limit=2))
    assert [raw.params for raw in got] == [{"dates": "20240101-20240107", "page": n, "limit": 2} for n in (1, 2)]
    assert {raw.endpoint for raw in got} == {"core_events"}
    assert espn_core.event_ids(got) == ["401585089", "401585090", "401585091"]
    # the UFC board's calendar refs (on ESPN's internal host) read the same way: the backfill's card ids
    cards = json.loads(fixture("sb_ufc"))["leagues"][0]["calendar"]
    page = RawFetch("espn_site", "scoreboard", {}, "u", 200,
                    json.dumps({"items": [card["event"] for card in cards]}).encode(), NOW)
    ids = espn_core.event_ids([page])
    assert ids[0] == "600057024" and len(ids) == len({card["event"]["$ref"] for card in cards}) == 52
    assert "600061267" in ids  # UFC 333, the audit's card


# --- fightcenter, StatsAPI ---------------------------------------------------------------------------------------


def test_fightcenter_card_url_and_bouts() -> None:
    url = "https://site.web.api.espn.com/apis/common/v3/sports/mma/ufc/fightcenter/600061267"  # the notes' URL
    raw = espn_fightcenter.card(FakeHttp({url: (200, fixture("syn_fightcenter"))}), 600061267)
    assert (raw.url, raw.source, raw.endpoint, raw.params, raw.status) == (url, "espn_fightcenter", "fightcenter",
                                                                          {}, 200)
    bouts = espn_fightcenter.bouts(raw.json())
    assert [(b["id"], b["segment"], b["state"], b["result"]) for b in bouts] == [
        ("401911632", "main", "final", "kotko"), ("401911640", "prelims1", "final", None),
        ("401911645", "prelims2", "pre", None)]
    assert bouts[0]["competitors"] == ["3949584", "4024000"]
    # a bout fightcenter has no result for is the one the worker asks core for (status, competitor statistics)
    assert [b["id"] for b in bouts if b["state"] == "final" and not b["result"]] == ["401911640"]
    assert espn_fightcenter.bouts({"cards": "nope"}) == [] and espn_fightcenter.bouts(None) == []


def test_statsapi_feed_is_one_game_keyed_on_its_game_pk() -> None:
    url = "https://statsapi.mlb.com/api/v1.1/game/777079/feed/live"  # the notes' URL
    http = FakeHttp({url: (200, fixture("syn_feed_live"))})
    for pk in (777079, "777079"):
        raw = statsapi.feed_live(http, pk)
        assert (raw.url, raw.source, raw.endpoint, raw.params) == (url, "mlb_statsapi", "feed_live",
                                                                   {"game_pk": 777079})
        assert type(raw.params["game_pk"]) is int
    assert http.calls[0] == ("https://statsapi.mlb.com/api/v1.1/game/777079/feed/live", {}, None)
    for bad in ([777079, 777080], "777079,777080", True, 0, -5, "abc", None, 7.5):
        with pytest.raises(ValueError):
            statsapi.feed_live(http, bad)


def test_the_backfill_never_uses_statsapi() -> None:
    """StatsAPI's terms are non-bulk: the backfill of older games (capture/backfill.py, the learn backfill's hook)
    never imports the live-feed fetcher."""
    pattern = re.compile(r"^\s*(?:from|import)\b.*\bstatsapi\b|\bfeed_live\s*\(")  # an import of it, or a call
    for name in ("capture/backfill.py", "learn/history.py"):
        path = PACKAGE / name
        if path.exists():
            assert not [line for line in path.read_text(encoding="utf-8").splitlines() if pattern.search(line)], name


# --- static history files ----------------------------------------------------------------------------------------


def test_nflverse_release_files() -> None:
    pbp_url = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2024.csv.gz"
    off_url = "https://github.com/nflverse/nflverse-data/releases/download/officials/officials.csv"
    http = FakeHttp({pbp_url: (200, fixture("syn_play_by_play_2024.csv")), off_url: (200, fixture("syn_officials"))})
    raw = nflverse.pbp_season(http, 2024)
    assert (raw.url, raw.source, raw.endpoint, raw.params) == (pbp_url, "nflverse", "pbp_season", {"season": 2024})
    assert b"False Start" in gzip.decompress(raw.body)
    raw = nflverse.officials(http)
    assert (raw.url, raw.endpoint, raw.params) == (off_url, "officials", {}) and b"Referee" in raw.body
    assert nflverse.pbp_season(http, "2023").params == {"season": 2023}
    for bad in (1998, "x", True, None):
        with pytest.raises(ValueError):
            nflverse.pbp_season(http, bad)


def test_greco_six_csvs_in_order() -> None:
    base = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
    answers = {f"{base}{name[4:]}.csv": (200, fixture(name)) for name in (
        "syn_ufc_event_details", "syn_ufc_fight_results", "syn_ufc_fight_stats", "syn_ufc_fighter_tott")}
    http = FakeHttp(answers, default=(200, b"HEADER\n"))
    got = list(greco.csvs(http))
    assert [raw.params["file"] for raw in got] == list(greco.FILES) and len(got) == 6
    assert greco.FILES[0] == "ufc_event_details.csv"  # the file the research downloaded (not "ufc_events.csv")
    assert all((raw.source, raw.endpoint) == ("greco1899", "csv") and raw.url == base + raw.params["file"]
               for raw in got)
    assert b"KO/TKO" in got[2].body
    assert [raw.params["file"] for raw in greco.csvs(http, ["ufc_fight_results.csv"])] == ["ufc_fight_results.csv"]
    for bad in ("../secrets.csv", "x/y.csv", "results.txt", ""):
        with pytest.raises(ValueError):
            greco.csv_file(http, bad)


def test_mcp_charting_files() -> None:
    base = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/"
    http = FakeHttp({f"{base}{name[4:]}.csv": (200, fixture(name)) for name in (
        "syn_charting-m-matches", "syn_charting-m-points-2020s", "syn_charting-m-stats-Overview")})
    got = list(mcp.csvs(http))
    assert [raw.params["file"] for raw in got] == list(mcp.FILES)
    assert all(raw.status == 200 and raw.url == base + raw.params["file"] and raw.source == "mcp" for raw in got)
    assert b"aces" in got[2].body
    assert mcp.league_of("charting-w-matches.csv") == "tennis_wta" and mcp.league_of(got[0]) == "tennis_atp"


def test_football_data_season_files() -> None:
    url = "https://football-data.co.uk/mmz4281/2425/E0.csv"
    http = FakeHttp({url: (200, fixture("syn_E0_2425"))})
    for season in (2024, "2425"):
        raw = football_data_raw.season_csv(http, "E0", season)
        assert (raw.url, raw.source, raw.endpoint, raw.params) == (url, "football_data", "csv", {"season": "2425"})
        assert raw.body.startswith(b"Div,Date,Time,HomeTeam")
    raw = football_data_raw.season_csv(http, "USA", 2025)  # one country file: every season
    assert (raw.url, raw.params) == ("https://football-data.co.uk/new/USA.csv", {})
    assert football_data_raw.season_code(1999) == "9900" and football_data_raw.season_code("2021") == "2021"
    for bad in (("XX", 2024), ("E0", "2024"), ("E0", True), ("E0", None), ("E0", 1990)):
        with pytest.raises(ValueError):
            football_data_raw.season_csv(http, *bad)
    assert set(football_data_raw.CODES) == {"E0", "E1", "SP1", "I1", "D1", "F1", "SC0", "N1", "P1"}


def test_football_data_raw_agrees_with_the_learn_source() -> None:
    """The raw copy and learn/sources/football_data.py read the same files."""
    assert football_data_raw.BASE == learn_football_data.BASE
    assert {code: league for code, league in football_data_raw.CODES.items()
            if code in learn_football_data.DIVISIONS} == learn_football_data.DIVISIONS
    assert {code: league for code, (league, _) in learn_football_data.NEW_LEAGUES.items()} == football_data_raw.NEW
    for season in range(2019, 2027):
        assert football_data_raw.season_code(season) == learn_football_data.season_code(season)
        assert football_data_raw.file_url("E0", season) == learn_football_data.file_url("E0", season)
    assert football_data_raw.file_url("USA") == learn_football_data.new_url("USA")
    leagues = {**LEAGUES}
    assert all(league in leagues for league in football_data_raw.CODES.values())


# --- the hosts capture reaches: never The Odds API -------------------------------------------------------------


def test_capture_sources_reach_only_their_hosts_and_never_the_odds_api() -> None:
    http = FakeHttp(default=(200, b"{}"))
    for drive in _streams(http):
        drive()
    hosts = {re.sub(r"^https?://([^/]+).*$", r"\1", url) for url in http.urls}
    assert hosts == {"site.api.espn.com", "sports.core.api.espn.com", "site.web.api.espn.com", "statsapi.mlb.com",
                     "github.com", "raw.githubusercontent.com", "football-data.co.uk"}
    assert not [url for url in http.urls if "the-odds-api" in url]
    text = " ".join(path.read_text(encoding="utf-8") for path in (PACKAGE / "capture" / "sources").glob("*.py")
                    if path.name not in ("espn_propbets.py", "kalshi_raw.py"))
    assert "the-odds-api" not in text and "apiKey" not in text and "api_key" not in text


# --- EspnClient.summary_raw / scoreboard_raw: one fetch serves both stores ------------------------------------


class RequestsFake:
    """requests.get as espn_client calls it, serving bodies by URL; counts every call."""

    def __init__(self, bodies: Optional[Dict[str, bytes]] = None) -> None:
        self.bodies = bodies or {}
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def __call__(self, url: str, params: Any = None, timeout: Any = None, headers: Any = None) -> requests.Response:
        self.calls.append((url, dict(params or {})))
        if url not in self.bodies:
            raise AssertionError(f"an extra request: {url}")
        response = requests.Response()
        response.status_code = 200
        response._content = self.bodies[url]
        response.headers["Content-Type"] = "application/json;charset=UTF-8"
        response.url = url
        return response


def _world(tmp_path: Path, name: str) -> Tuple[EspnClient, TTLCache, Store, Clock]:
    clock = Clock()
    store = Store(str(tmp_path / f"{name}.db"))
    cache = TTLCache(store, name=name, clock=clock)
    return EspnClient(cache, clock=clock), cache, store, clock


def _copy(cache: TTLCache, store: Store, key: Tuple[Any, ...]) -> Tuple[str, Any, float, float, bytes, float, float]:
    """(value as JSON text, kind, fresh for, usable for, the shared row's blob, its fresh and usable offsets)."""
    entry = cache._entries[key]
    row = store.cache_get(key_text(key))
    assert row is not None, "the copy is in the shared table too"
    return (json.dumps(entry[2]), entry[3], entry[4], entry[5], row[3], row[1] - row[0], row[2] - row[0])


SUMMARIES = ["sum_nba", "sum_nba2", "sum_nba3", "sum_nfl", "sum_nhl", "sum_mlb", "sum_epl"]
# A game's summary at other moments than final (v4 P1 review F7): its status, and its start against NOW. Each copy must
# be seeded with the life a load of it gets (the policy of its own state: a pre-game copy minutes, a live one seconds)
MOMENTS: Dict[str, Tuple[Dict[str, Any], float, str]] = {
    "pre": ({"id": "1", "name": "STATUS_SCHEDULED", "state": "pre", "completed": False, "description": "Scheduled"},
            86400.0, "pre"),
    "live": ({"id": "2", "name": "STATUS_IN_PROGRESS", "state": "in", "completed": False,
              "description": "In Progress"}, -3600.0, "live"),
    # a pre-game copy whose start has passed reads as live
    "late": ({"id": "1", "name": "STATUS_SCHEDULED", "state": "pre", "completed": False, "description": "Scheduled"},
             -600.0, "live"),
}
SUMMARY_CASES = [(name, "final") for name in SUMMARIES] + [
    (name, moment) for name in ("sum_nba", "sum_nhl", "sum_mlb", "sum_epl") for moment in MOMENTS]


def _iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _summary_at(name: str, moment: str) -> bytes:
    """A fixture summary as it read at ``moment`` ("final": the fixture as it is)."""
    if moment == "final":
        return fixture(name)
    data = json.loads(fixture(name))
    status, offset, _state = MOMENTS[moment]
    competition = data["header"]["competitions"][0]
    competition["status"]["type"] = dict(status)
    competition["date"] = _iso(NOW + offset)
    return json.dumps(data).encode()


@pytest.mark.parametrize("name, moment", SUMMARY_CASES)
def test_summary_raw_seeds_the_lean_cache_exactly_as_summary_loads_it(tmp_path: Path, name: str, moment: str,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    entry = ENTRIES[name]
    sport, path, event = re.search(r"/sports/([^/]+)/([^/]+)/summary\?event=(\d+)$", entry["url"]).groups()
    league, base = LEAGUE_OF[(sport, path)], entry["url"].split("?")[0]
    key = ("espn", "summary", league, event)
    served = _summary_at(name, moment)

    # Today: the grader's summary() loads the game through the cache
    before, cache_a, store_a, _ = _world(tmp_path, "before")
    monkeypatch.setattr(espn_client.requests, "get", RequestsFake({base: served}))
    lean = before.summary(league, event)
    graded = [before.game_result(league, event), before.box_score(league, event), before.closing_line(league, event)]

    # With capture: one fetch through the capture's http, then the grader
    after, cache_b, store_b, _ = _world(tmp_path, "after")
    http = FakeHttp({entry["url"]: (200, served)})
    body = after.summary_raw(league, event, http=http)
    assert body == served and http.calls == [(base, {"event": event}, None)]
    # The copy's life is its own state's policy, as a load gives it
    life = ttl_for("summary", "final" if moment == "final" else MOMENTS[moment][2])
    assert _copy(cache_b, store_b, key)[2:4] == (life.ttl, life.ttl + life.stale)
    assert after.event_state(league, event) == before.event_state(league, event)  # known at once, as a load does
    nothing = RequestsFake()
    monkeypatch.setattr(espn_client.requests, "get", nothing)
    assert after.summary(league, event) == lean
    assert [after.game_result(league, event), after.box_score(league, event),
            after.closing_line(league, event)] == graded
    assert json.dumps(graded) == json.dumps([after.game_result(league, event), after.box_score(league, event),
                                             after.closing_line(league, event)])
    assert nothing.calls == [], "the grader read the seeded copy: no extra request"
    assert _copy(cache_a, store_a, key) == _copy(cache_b, store_b, key)  # the lean cache is byte-identical
    assert after.event_state(league, event) == before.event_state(league, event)
    assert json.loads(body) != lean  # the raw store gets the whole summary


def test_summary_raw_without_an_http_uses_requests_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, cache, _, _ = _world(tmp_path, "plain")
    base = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
    fake = RequestsFake({base: fixture("sum_nba")})
    monkeypatch.setattr(espn_client.requests, "get", fake)
    assert client.summary_raw("basketball_nba", 401909278) == fixture("sum_nba")
    assert fake.calls == [(base, {"event": "401909278"})]  # only the event, as the learn backfill sends it
    assert client.summary("basketball_nba", "401909278") == lean_summary(json.loads(fixture("sum_nba")))
    assert len(fake.calls) == 1 and cache.peek(("espn", "summary", "basketball_nba", "401909278")) is not None


def test_seed_false_fetches_the_same_and_writes_nothing_to_the_lean_cache(tmp_path: Path) -> None:
    """The learn backfill's old games (v4 P1 review M1): the same request and the same bytes, but no lean copy in
    memory or in the shared table, and no state noted."""
    client, cache, store, _ = _world(tmp_path, "unseeded")
    entry = ENTRIES["sum_nba"]
    event = re.search(r"event=(\d+)$", entry["url"]).group(1)
    http = FakeHttp({entry["url"]: (200, fixture("sum_nba"))})
    assert client.summary_raw("basketball_nba", event, http=http, seed=False) == fixture("sum_nba")
    assert http.calls == [(entry["url"].split("?")[0], {"event": event}, None)]
    day = datetime.date(2026, 10, 30)
    board_url = f"{espn_client.SITE}/basketball/nba/scoreboard?dates=20261030&limit=300"
    boards = FakeHttp({board_url: (200, fixture("sb_nba"))})
    assert client.scoreboard_raw("basketball_nba", day, http=boards, seed=False) == fixture("sb_nba")
    assert len(boards.calls) == 1
    for key in (("espn", "summary", "basketball_nba", event), ("espn", "scoreboard", "basketball_nba", day.isoformat())):
        assert cache.peek(key) is None and store.cache_get(key_text(key)) is None
    assert client.event_state("basketball_nba", event) is None
    assert all(client.event_state("basketball_nba", e["id"]) is None for e in json.loads(fixture("sb_nba"))["events"])
    # seed=True (the default) seeds, as before
    assert client.summary_raw("basketball_nba", event, http=http) == fixture("sum_nba")
    assert cache.peek(("espn", "summary", "basketball_nba", event)) is not None


class DoorClosed(Exception):
    """An exception type nothing in the package knows (a caller's own door)."""


@pytest.mark.parametrize("make", [
    lambda: DoorClosed("the learn backfill's door is shut"),
    lambda: LearnRefused("the learn backfill's budget is spent"),
    lambda: LearnHttpError(404, "https://site.api.espn.com/x"),
    lambda: Unreachable("site.api.espn.com: timed out"),
    lambda: Backoff("site.api.espn.com", NOW + 600, "429"),
    lambda: Refused("the capture budget is spent"),
])
def test_summary_raw_and_scoreboard_raw_let_whatever_http_raises_through_unchanged(tmp_path: Path, make: Any) -> None:
    client, cache, _, _ = _world(tmp_path, "door")
    day = datetime.date(2026, 10, 30)
    for call, key in ((lambda http: client.summary_raw("basketball_nba", "401", http=http),
                       ("espn", "summary", "basketball_nba", "401")),
                      (lambda http: client.scoreboard_raw("basketball_nba", day, http=http),
                       ("espn", "scoreboard", "basketball_nba", day.isoformat()))):
        exc = make()
        with pytest.raises(type(exc)) as caught:
            call(FakeHttp(default=exc))
        assert caught.value is exc and type(caught.value) is type(exc)
        assert cache.peek(key) is None  # a failure seeds nothing
    assert client.event_state("basketball_nba", "401") is None


def test_an_http_error_answer_raises_and_seeds_nothing(tmp_path: Path) -> None:
    client, cache, _, _ = _world(tmp_path, "error")
    with pytest.raises(EspnHttpError) as caught:
        client.summary_raw("basketball_nba", "401", http=FakeHttp(default=(404, b"HTTP Error 404: Not Found")))
    assert caught.value.status == 404 and caught.value.body == b"HTTP Error 404: Not Found"
    assert isinstance(caught.value, RuntimeError) and "ESPN returned HTTP 404" in str(caught.value)  # as _get's
    with pytest.raises(EspnHttpError):
        client.scoreboard_raw("basketball_nba", datetime.date(2026, 10, 30), http=FakeHttp(default=(503, b"")))
    assert cache.peek(("espn", "summary", "basketball_nba", "401")) is None
    with pytest.raises(KeyError):
        client.summary_raw("tennis_atp", "187149", http=FakeHttp())  # no site summary: fightcenter, scoreboard
    # a door that hands back the body itself (it raises on HTTP errors on its own) works too
    class Door:
        def get(self, url: str, params: Any = None, *, headers: Any = None) -> bytes:
            return fixture("sum_nba")

    assert client.summary_raw("basketball_nba", "401909278", http=Door()) == fixture("sum_nba")


BOARDS = [("sb_nba", "basketball_nba"), ("sb_nfl", "americanfootball_nfl"), ("sb_epl", "soccer_epl"),
          ("sb_ufc", "mma_mixed_martial_arts"), ("sb_nhl", "icehockey_nhl"), ("sb_mlb", "baseball_mlb")]
# A board at other moments than as captured (a past day of finals): its day against NOW and its games' statuses, and
# the board state (data/espn_client.board_state) whose life a load gives it (v4 P1 review F7)
BOARD_MOMENTS = {"captured": "past", "live": "live", "later": "today", "idle": "idle", "future": "future"}
BOARD_CASES = [(name, league, "captured") for name, league in BOARDS] + [
    (name, league, moment) for name, league in BOARDS[:3] for moment in ("live", "later", "idle", "future")]
TODAY = datetime.datetime.fromtimestamp(NOW, espn_client.ESPN_DAY_TZ).date()


def _board_at(name: str, moment: str) -> Tuple[bytes, datetime.date]:
    """A fixture board as it read at ``moment``, and its day ("captured": the fixture's own)."""
    if moment == "captured":
        url = ENTRIES[name]["url"]
        return fixture(name), datetime.datetime.strptime(re.search(r"dates=(\d{8})", url).group(1), "%Y%m%d").date()
    data = json.loads(fixture(name))
    day = TODAY + datetime.timedelta(days=1 if moment == "future" else 0)
    for n, event in enumerate(data["events"]):
        competition = event["competitions"][0]
        if moment == "idle":
            start = NOW - 6 * 3600.0  # over earlier today
        elif moment == "live" and n == 0:
            start = NOW - 3600.0
            status = {"id": "2", "name": "STATUS_IN_PROGRESS", "state": "in", "completed": False}
        else:
            start = NOW + (86400.0 if moment == "future" else 3 * 3600.0)
            status = {"id": "1", "name": "STATUS_SCHEDULED", "state": "pre", "completed": False}
        for holder in (event, competition):
            holder["date"] = _iso(start)
            if moment != "idle" and isinstance(holder.get("status"), dict):
                holder["status"]["type"] = dict(status)
    return json.dumps(data).encode(), day


@pytest.mark.parametrize("name, league, moment", BOARD_CASES)
def test_scoreboard_raw_seeds_the_lean_board_exactly_as_scoreboard_loads_it(tmp_path: Path, name: str, league: str,
                                                                            moment: str,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    served, day = _board_at(name, moment)
    base = ENTRIES[name]["url"].split("?")[0]
    key = ("espn", "scoreboard", league, day.isoformat())

    before, cache_a, store_a, _ = _world(tmp_path, "before")
    monkeypatch.setattr(espn_client.requests, "get", RequestsFake({base: served}))
    events = before.scoreboard(league, day)
    assert before.board_state(day, events) == BOARD_MOMENTS[moment]

    after, cache_b, store_b, _ = _world(tmp_path, "after")
    query = f"{base}?dates={day:%Y%m%d}&limit=300"
    http = FakeHttp({query: (200, served)})
    body = after.scoreboard_raw(league, day, http=http)
    assert body == served
    life = ttl_for("scoreboard", BOARD_MOMENTS[moment])  # the board's own state's policy, as a load gives it
    assert _copy(cache_b, store_b, key)[2:4] == (life.ttl, life.ttl + life.stale)
    for event in json.loads(body)["events"]:  # the games' states are known at once, as a load leaves them
        assert after.event_state(league, event["id"]) == before.event_state(league, event["id"])
    assert http.calls == [(base, {"dates": f"{day:%Y%m%d}", "limit": espn_client.SCOREBOARD_LIMIT}, None)]
    nothing = RequestsFake()
    monkeypatch.setattr(espn_client.requests, "get", nothing)
    assert after.scoreboard(league, day) == events and nothing.calls == []
    assert _copy(cache_a, store_a, key) == _copy(cache_b, store_b, key)
    for event in json.loads(body)["events"]:
        assert after.event_state(league, event["id"]) == before.event_state(league, event["id"])
    raw_keys = json.dumps(json.loads(body))
    assert any(f'"{drop}"' in raw_keys for drop in LEAN_DROP) and not any(
        f'"{drop}":' in json.dumps(events) for drop in LEAN_DROP)  # raw keeps what the lean board drops


def test_scoreboard_raw_sends_the_lean_clients_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _, _ = _world(tmp_path, "query")
    http = FakeHttp(default=(200, b'{"events": []}'))
    client.scoreboard_raw("basketball_ncaab", datetime.date(2026, 11, 3), http=http)
    assert http.calls == [("https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/"
                           "scoreboard", {"dates": "20261103", "limit": 300, "groups": "50"}, None)]
    fake = RequestsFake({"https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard": fixture("sb_atp")})
    monkeypatch.setattr(espn_client.requests, "get", fake)
    assert client.scoreboard_raw("tennis_atp", datetime.date(2026, 10, 31)) == fixture("sb_atp")
    assert fake.calls == [("https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard",
                           {"dates": "20261031", "limit": 300})]
    with pytest.raises(KeyError):
        client.scoreboard_raw("soccer_nowhere_league", datetime.date(2026, 10, 31), http=http)


def test_the_lean_rules_are_unchanged() -> None:
    """SUMMARY_KEEP and LEAN_DROP stay as they are (section 3.3): the cache keeps the lean shape."""
    assert espn_client.SUMMARY_KEEP == ("header", "predictor", "injuries", "boxscore", "pickcenter")
    assert LEAN_DROP == frozenset({"links", "leaders", "statistics", "headlines", "highlights", "odds", "details",
                                   "featuredAthletes", "roster", "broadcasts", "broadcast", "geoBroadcasts", "tickets",
                                   "situation", "probables", "notes"})
    assert ttl_for("summary", "final").ttl > ttl_for("summary", "live").ttl


# --- the fixtures ------------------------------------------------------------------------------------------------


def test_the_fixtures_are_listed_small_and_readable() -> None:
    files = {path.name for path in FIX.iterdir() if path.is_file()} - {"manifest.json"}
    assert files == {item["file"] for item in MANIFEST["fixtures"]}
    assert sum((FIX / name).stat().st_size for name in files) < 700_000, "keep the capture fixtures small"
    audit = [item for item in MANIFEST["fixtures"] if not item.get("synthetic")]
    assert len(audit) == 25  # the audit's 24 payloads and ESPN's 400 for a date range
    for item in audit:
        assert item["url"].startswith("https://") and item["status"] in (200, 400, 404)
        if item["file"].endswith(".json"):
            json.loads((FIX / item["file"]).read_bytes())
    assert all(item["file"].startswith("syn_") for item in MANIFEST["fixtures"] if item.get("synthetic"))
