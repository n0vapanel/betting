"""The coverage audit (v3 Stage 2b): every outbound HTTP call in the package goes through the shared TTLCache with a
kind of data (and, for the data clients, the policy table's time to live), except the AI clients (chats are personal,
never cached across people), the odds snapshot loader (the store's odds snapshots are its own persistence) and the raw
capture (v4 P1: raw.db is its own store, see ``allowlisted``).

How: the HTTP layer is monkeypatched (requests' Session.request, which requests.get and every Session go through;
httpx, urllib and http.client fail loudly), TTLCache's read path is wrapped so a loader marks the thread it runs
on, and every fetch path of every module that talks to the network is driven with canned answers. A call made
outside a loader is a violation unless its call site is on the allowlist. Nothing here touches the network.

Two more checks keep it honest: a module that starts importing an HTTP library must be added to AUDITED (and its
fetch paths driven here), and a new public method on an audited client must be classified (FETCHES or NO_FETCH).
"""

from __future__ import annotations

import inspect
import json
import re
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import requests

from telegram_betting_bot_simple.data import cache as cache_module
from telegram_betting_bot_simple.data import espn_client, utils
from telegram_betting_bot_simple.data.cache import POLICY, TTLCache
from telegram_betting_bot_simple.data.espn_client import EspnClient

PACKAGE = Path(cache_module.__file__).resolve().parents[1]

# Modules that import an HTTP client library, and how their calls are covered
AUDITED = {
    "data/espn_client.py": "every ESPN endpoint through the cache",
    "data/odds_service.py": "the sports listing through the cache; the boards are the odds snapshot loader",
    "data/apis/api_sports_client.py": "API-Sports through the cache",
    "data/utils.py": "fetch_json through the cache (The Odds API's and NewsAPI's legacy clients use it)",
    "data/markets/common.py": "Polymarket, Kalshi and the rating sites through the Keeper (a TTLCache)",
    "ai/deepseek.py": "AI: allowlisted (chats and forecasts are never cached across people)",
    "data/weather.py": "Open-Meteo's forecast and geocoder through the cache (kinds weather, geocode)",
    "data/mlb_stats.py": "MLB StatsAPI's schedule through the cache (kind mlb_schedule)",
    "learn/sources/http.py": "the history backfill (football-data, nflverse, ESPN, MLB StatsAPI) through the cache "
                             "(kinds hist_file, hist_list, hist_game)",
    "capture/http.py": "the raw capture's door (PoliteHttp, urllib): allowlisted by design (raw.db keeps every answer "
                       "untrimmed; per-host token buckets and raw.db's budgets pace it), see allowlisted()",
}
AI_FILES = {"ai/claude.py", "ai/explain.py", "ai/assistant.py", "ai/deepseek.py"}  # the anthropic SDK and DeepSeek
TRANSPORT = {"main.py"}  # python-telegram-bot: the bot's own messaging, not a data fetch (also bot/*)
HTTP_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(requests|httpx|urllib3|aiohttp|http\.client|urllib\.request|socket|pycurl|websocket)\b"
    r"|^\s*import\s+urllib\s*$", re.M)
SDK_IMPORT = re.compile(r"^\s*(?:import|from)\s+anthropic\b", re.M)
TELEGRAM_IMPORT = re.compile(r"^\s*(?:import|from)\s+telegram\b", re.M)


def _sources() -> Dict[str, str]:
    return {str(path.relative_to(PACKAGE)): path.read_text(encoding="utf-8") for path in PACKAGE.rglob("*.py")}


def test_every_module_that_can_reach_the_network_is_audited() -> None:
    found = {name for name, text in _sources().items() if HTTP_IMPORT.search(text)}
    assert found - set(AUDITED) == set(), "a new module talks HTTP: route it through TTLCache and add it to AUDITED"
    assert set(AUDITED) - found == set(), "AUDITED lists a module that no longer imports an HTTP library"
    sdk = {name for name, text in _sources().items() if SDK_IMPORT.search(text)}
    assert sdk <= AI_FILES, "only the AI clients may use the anthropic SDK (never cached across people)"
    telegram = {name for name, text in _sources().items() if TELEGRAM_IMPORT.search(text)}
    assert all(name in TRANSPORT or name.startswith("bot/") for name in telegram)


# --- the recorder -----------------------------------------------------------------------------------------

class Call:
    def __init__(self, url: str, kind: Optional[str], ttl: Optional[float], site: str, stack: List[str]) -> None:
        self.url, self.kind, self.ttl, self.site, self.stack = url, kind, ttl, site, stack

    def __repr__(self) -> str:
        return f"<{self.site} {self.url} kind={self.kind} ttl={self.ttl}>"


def _site() -> tuple:
    """(the innermost package frame outside data/cache.py as "file:function", every package frame's function)."""
    frames = []
    for frame in reversed(traceback.extract_stack()):
        path = Path(frame.filename).resolve()
        if PACKAGE in path.parents and path.name != "cache.py":
            frames.append(f"{path.relative_to(PACKAGE)}:{frame.name}")
    return (frames[0] if frames else "outside the package"), frames


ANSWERS = [
    # (URL pattern, JSON answer); the first match wins, else {}
    (r"api\.the-odds-api\.com/v4/sports/?$", []),
    (r"api\.the-odds-api\.com/v4/sports/[^/]+/odds", []),
    (r"/scoreboard$", {"events": []}),
    (r"gamma-api\.polymarket\.com", []),
    (r"barttorvik\.com", []),
    (r"newsapi\.org", {"articles": []}),
    (r"api\.open-meteo\.com/v1/forecast", {"latitude": 39.74, "longitude": -104.99, "elevation": 1600.0, "hourly": {
        "time": ["2026-10-27T00:00", "2026-10-27T01:00"], "temperature_2m": [10.0, 9.5],
        "precipitation_probability": [0, 5], "precipitation": [0.0, 0.0], "wind_speed_10m": [12.0, 14.0],
        "wind_direction_10m": [200, 210], "wind_gusts_10m": [25.0, 27.0]}}),
    (r"geocoding-api\.open-meteo\.com", {"results": [{"name": "Denver", "latitude": 39.74, "longitude": -104.98,
                                                       "country_code": "US", "elevation": 1609.0}]}),
    (r"statsapi\.mlb\.com", {"dates": []}),
]


def _answer(url: str) -> requests.Response:
    body: Any = {}
    for pattern, answer in ANSWERS:
        if re.search(pattern, url):
            body = answer
            break
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(body).encode()
    response.headers["Content-Type"] = "application/json"
    response.url = url
    return response


class Recorder:
    """Patches the HTTP layer and TTLCache's read path; ``calls`` are every request made, with the kind and TTL of
    the cache loader it ran inside (None outside one)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: List[Call] = []
        self.local = threading.local()
        recorder = self

        def request(session, method, url, *args, **kwargs):
            site, stack = _site()
            frames = getattr(recorder.local, "loaders", [])
            kind, ttl = frames[-1] if frames else (None, None)
            recorder.calls.append(Call(str(url), kind, ttl, site, stack))
            return _answer(str(url))

        monkeypatch.setattr(requests.sessions.Session, "request", request)

        # The raw capture's door (capture/http.py PoliteHttp) sends with urllib: its one send is recorded the same way
        # (urllib itself stays refused below, so nothing else can use it unseen)
        from telegram_betting_bot_simple.capture import http as capture_http

        def polite_send(polite, req):
            site, stack = _site()
            frames = getattr(recorder.local, "loaders", [])
            kind, ttl = frames[-1] if frames else (None, None)
            recorder.calls.append(Call(str(req.full_url), kind, ttl, site, stack))
            answer = _answer(str(req.full_url))
            return capture_http.HttpResponse(answer.status_code, answer.content, {"content-type": "application/json"})

        monkeypatch.setattr(capture_http.PoliteHttp, "_send", polite_send)

        def refuse(*args, **kwargs):
            raise AssertionError("an HTTP call outside requests: audit it (and route it through TTLCache)")

        import http.client
        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", refuse)
        monkeypatch.setattr(http.client.HTTPConnection, "request", refuse)
        try:
            import httpx
            monkeypatch.setattr(httpx.Client, "send", refuse)
        except ImportError:  # pragma: no cover
            pass

        def wrap(real):
            def read(cache, key, ttl, loader, **options):
                kind = options.get("kind")

                def marked():
                    frames = recorder.local.__dict__.setdefault("loaders", [])
                    frames.append((kind, float(ttl)))
                    try:
                        return loader()
                    finally:
                        frames.pop()
                return real(cache, key, ttl, marked, **options)
            return read

        # Both public reads (get_for goes through get); a loader marks whichever thread runs it (background too)
        monkeypatch.setattr(TTLCache, "get", wrap(TTLCache.get))
        monkeypatch.setattr(TTLCache, "get_with_time", wrap(TTLCache.get_with_time))

    def outside(self) -> List[Call]:
        return [call for call in self.calls if call.kind is None and not allowlisted(call)]


# The raw capture (v4 P1) fetches outside the lean cache by design: raw.db (capture/raw_store.py) is its own store, every
# answer kept untrimmed and deduplicated on its sha256, so a cache copy would be a second store of the same bytes; the
# capture's door (capture/http.py PoliteHttp) paces each host with a token bucket and raw.db counts the day's and the
# night's request budgets. EspnClient.summary_raw/scoreboard_raw are the one fetch that serves both stores: the raw body
# goes to raw.db and the lean copy into the cache from the same response, so the grader's next summary() is a hit.
CAPTURE_FRAMES = ("capture/http.py:get", "data/espn_client.py:summary_raw", "data/espn_client.py:scoreboard_raw")


def allowlisted(call: Call) -> bool:
    """The AI clients, the odds snapshot loader (OddsService.league_games' board, saved as a snapshot), the odds
    quota's re-check (OddsService.recheck_quota reads the free /sports list for its usage headers, so it must reach
    the API, not a cached copy: at most hourly, one process, and only while the saved count is under the reserve),
    and the raw capture (CAPTURE_FRAMES: raw.db is its own store, see above)."""
    return call.site.split(":")[0] in AI_FILES or any(
        frame in ("data/odds_service.py:_board", "data/odds_service.py:recheck_quota") + CAPTURE_FRAMES
        for frame in call.stack)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    monkeypatch.setattr(utils, "_cache", TTLCache(name="coverage"))  # fetch_json's process cache, fresh
    return Recorder(monkeypatch)


# --- the audit fails when a module fetches outside the cache ----------------------------------------------

def test_the_audit_catches_a_fetch_outside_the_cache(recorder: Recorder) -> None:
    EspnClient()._get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary", {"event": "1"})
    requests.get("https://example.test/feed")
    assert len(recorder.outside()) == 2  # both seen, neither inside a loader
    assert recorder.outside()[0].site == "data/espn_client.py:_get"
    # ... and the same request through the cache passes
    EspnClient(TTLCache()).summary("basketball_nba", "1")
    assert [call.kind for call in recorder.calls] == [None, None, "summary"]


# --- every fetch path, driven ---------------------------------------------------------------------------

DAY = espn_client.datetime.date(2026, 10, 27)

# Public methods of each audited client: the ones that fetch (driven below) and the ones that never do
FETCHES = {
    # summary_raw/scoreboard_raw: the raw capture's one fetch that serves both stores (allowlisted, see CAPTURE_FRAMES)
    "EspnClient": {"scoreboard", "fetch_scoreboard", "summary", "standings", "roster", "player_seasons", "find_game",
                   "game_result", "box_score", "closing_line", "played_on", "summary_raw", "scoreboard_raw"},
    "PlayerData": {"search", "overview", "gamelog", "splits", "news", "props", "card"},
    "OddsService": {"sports", "league_games", "recheck_quota"},
    "ApiSportsClient": {"get_fixtures", "get_team_stats", "get_player_stats", "get_injuries"},
    "Polymarket": {"events", "fetch_moneylines", "fetch_lines"},
    "Kalshi": {"events", "fetch_games", "fetch_lines", "fetch_series", "request"},
    "Ratings": {"power", "espn_predictor", "game_prob"},
    "WeatherClient": {"forecast", "forecasts", "geocode"},
    "MlbStats": {"games"},
    "FootballData": {"season_records", "new_records"},
    "NflVerse": {"rows", "records"},
    "EspnHistory": {"events", "summary", "odds", "scoreboard", "mlb_season"},
}
NO_FETCH = {
    "EspnClient": {"event_state", "game_state", "board_state"},
    "PlayerData": set(),
    # budgeted only marks the slate's already-counted board for the thread (a fetch inside it is league_games')
    "OddsService": {"odds_loaded_at", "saved_games", "consensus_inputs", "board_params", "quota", "region_units",
                    "board_cost", "budgeted"},
    "ApiSportsClient": set(),
    "Polymarket": {"leagues", "covers", "cached_events"},
    "Kalshi": {"leagues", "covers", "cached_events", "resting"},
    "Ratings": {"cached_power"},
    "WeatherClient": {"at", "attribution", "calls_today"},
    "MlbStats": {"day_state"},
    "FootballData": set(),
    "NflVerse": set(),
    "EspnHistory": set(),
}


def _clients():
    from telegram_betting_bot_simple.data.apis import ApiSportsClient
    from telegram_betting_bot_simple.data.espn_players import PlayerData
    from telegram_betting_bot_simple.data.markets import Kalshi, Polymarket, Ratings
    from telegram_betting_bot_simple.data.mlb_stats import MlbStats
    from telegram_betting_bot_simple.data.odds_service import OddsService
    from telegram_betting_bot_simple.data.weather import WeatherClient
    from telegram_betting_bot_simple.learn.sources import EspnHistory, FootballData, NflVerse
    return {"EspnClient": EspnClient, "PlayerData": PlayerData, "OddsService": OddsService,
            "ApiSportsClient": ApiSportsClient, "Polymarket": Polymarket, "Kalshi": Kalshi, "Ratings": Ratings,
            "WeatherClient": WeatherClient, "MlbStats": MlbStats, "FootballData": FootballData, "NflVerse": NflVerse,
            "EspnHistory": EspnHistory}


def test_every_public_method_of_an_audited_client_is_classified() -> None:
    for name, cls in _clients().items():
        public = {attr for attr, value in inspect.getmembers(cls) if not attr.startswith("_")
                  and (callable(value) or isinstance(value, property))}
        assert public == FETCHES[name] | NO_FETCH[name], (
            f"{name}: classify {sorted(public - FETCHES[name] - NO_FETCH[name])} (and drive the fetching ones in "
            f"test_every_fetch_goes_through_the_cache); drop {sorted((FETCHES[name] | NO_FETCH[name]) - public)}")


def _drive_everything(cache: TTLCache) -> None:
    from telegram_betting_bot_simple.ai.deepseek import DeepSeekAnalyst
    from telegram_betting_bot_simple.data.apis import ApiSportsClient, NewsApiClient, TheOddsApiClient
    from telegram_betting_bot_simple.data.espn_players import PlayerData
    from telegram_betting_bot_simple.data.markets import Kalshi, Polymarket, Ratings
    from telegram_betting_bot_simple.data.markets.kalshi import Pacer
    from telegram_betting_bot_simple.data.odds_service import OddsService

    nba = "basketball_nba"
    espn = EspnClient(cache)
    espn.scoreboard(nba, DAY)
    espn.fetch_scoreboard("tennis_atp", DAY)
    espn.summary(nba, "401")
    espn.standings(nba)
    espn.standings(nba, 2026)
    espn.roster(nba, "13")
    espn.player_seasons(nba, "1966")
    espn.find_game(nba, "Boston Celtics", "New York Knicks", "2026-10-28T23:30:00Z")
    espn.game_result(nba, "402")
    espn.box_score(nba, "403")
    espn.closing_line(nba, "404")
    espn.played_on(nba, DAY - espn_client.datetime.timedelta(days=1))
    # The raw capture's one fetch for both stores (allowlisted: CAPTURE_FRAMES), without and with the capture's door
    espn.summary_raw(nba, "407")
    espn.scoreboard_raw(nba, DAY - espn_client.datetime.timedelta(days=1))

    players = PlayerData(espn)
    players.search("LeBron James", nba)
    players.overview(nba, "1966")
    players.gamelog(nba, "1966")
    players.splits(nba, "1966")
    players.news(nba, "1966")
    players.props(nba, "401")
    players.card(nba, "3112335")

    odds = OddsService("odds-key-not-real-123", cache)
    odds.sports()
    odds.league_games(nba)  # the snapshot loader (allowlisted; it runs inside the cache too)
    odds._quota = {"remaining": 1, "used": 499}  # under the reserve, so the re-check reads /sports (allowlisted)
    assert odds.recheck_quota()

    sports = ApiSportsClient("api-sports-key-not-real", cache=cache)
    sports.get_fixtures("nba", DAY)
    sports.get_team_stats("soccer", "33", "2026")
    sports.get_player_stats("soccer", "276", "2026")
    sports.get_injuries("soccer", league_id="39")

    utils.fetch_json("https://example.test/v1/feed", params={"apiKey": "never-in-a-key", "q": "x"})
    TheOddsApiClient("legacy-odds-not-real").get_sports()
    TheOddsApiClient("legacy-odds-not-real").get_odds(nba)
    NewsApiClient("news-key-not-real-1").get_recent_articles("Lakers")

    Polymarket(cache).events(nba)
    Kalshi(cache, pacer=Pacer(0.0, sleep=lambda seconds: None)).events(nba)
    ratings = Ratings(espn, cache)
    for league in (nba, "basketball_ncaab", "tennis_atp_paris", "soccer_fifa_world_cup_qualifiers_europe"):
        ratings.power(league)
    ratings.espn_predictor(nba, "405")

    # Stage 4's context sources: Open-Meteo (a forecast, the same with its fetch times, a geocode) and MLB StatsAPI
    from telegram_betting_bot_simple.data.mlb_stats import MlbStats
    from telegram_betting_bot_simple.data.weather import WeatherClient
    weather = WeatherClient(cache)
    soon = espn_client.datetime.datetime.now(espn_client.datetime.timezone.utc)
    weather.forecast([("coors-field", 39.7560, -104.9941)], soon, soon + espn_client.datetime.timedelta(hours=4))
    weather.forecasts([("ball-arena", 39.7487, -105.0077)], soon, soon + espn_client.datetime.timedelta(hours=4))
    weather.geocode("Denver", "US")
    MlbStats(cache).games(DAY)

    # The learning loop's history backfill (learn/sources): football-data.co.uk, nflverse, ESPN's past games and
    # MLB StatsAPI's season schedule, through the real HTTP door (no budget hook: it is the Backfill's)
    from telegram_betting_bot_simple.learn.sources import EspnHistory, FootballData, HistHttp, NflVerse
    door = HistHttp(pace=0.0)
    soccer = FootballData(door, cache)
    soccer.season_records("E0", 2025)
    soccer.season_records("E0", 2026, current=True)
    soccer.new_records("USA")
    NflVerse(door, cache).records()
    past = EspnHistory(door, cache)
    past.events(nba, DAY - espn_client.datetime.timedelta(days=7), DAY)
    past.summary(nba, "406")
    past.odds(nba, "406")
    past.scoreboard("baseball_mlb", DAY)
    past.mlb_season(2025)

    try:
        DeepSeekAnalyst("deepseek-key-not-real").analyze("a dossier")
    except Exception:
        pass  # the canned answer isn't a forecast; the call is what counts

    _drive_capture(cache)


def _drive_capture(cache: TTLCache) -> None:
    """The raw capture's fetchers (v4 P1, capture/sources), each through the capture's own door (PoliteHttp, its sleep
    a no-op), and the one fetch that serves both stores (summary_raw / scoreboard_raw through it)."""
    from telegram_betting_bot_simple.capture.http import PoliteHttp
    from telegram_betting_bot_simple.capture.sources import (espn_core, espn_fightcenter, espn_propbets, espn_site,
                                                             football_data_raw, greco, kalshi_raw, mcp, nflverse,
                                                             statsapi)
    from telegram_betting_bot_simple.data.markets.kalshi import Pacer
    http = PoliteHttp(sleep=lambda seconds: None)
    espn_site.summary(http, "basketball_nba", "401")
    espn_site.scoreboard(http, "basketball_nba", DAY)
    list(espn_core.plays_all(http, "soccer_epl", "402", max_pages=1))
    espn_core.status(http, "mma_mixed_martial_arts", "600061267", "401911632")
    espn_core.competitor_stats(http, "mma_mixed_martial_arts", "600061267", "401911632", "3949584")
    espn_fightcenter.card(http, "600061267")
    statsapi.feed_live(http, 777079)
    nflverse.pbp_season(http, 2025)
    nflverse.officials(http)
    list(greco.csvs(http))
    list(mcp.csvs(http))
    football_data_raw.season_csv(http, "E0", 2025)
    list(espn_propbets.propbets_all(http, "basketball_nba", "401"))
    list(kalshi_raw.series_all(http, "KXEPLCORNERS", pacer=Pacer(0.0, sleep=lambda seconds: None)))
    espn = EspnClient(cache)
    espn.summary_raw("basketball_nba", "408", http=http)
    espn.scoreboard_raw("soccer_epl", DAY, http=http)


def test_every_fetch_goes_through_the_cache(recorder: Recorder) -> None:
    _drive_everything(TTLCache(name="coverage"))
    hosts = {re.sub(r"^https?://([^/]+).*$", r"\1", call.url) for call in recorder.calls}
    # Every source was reached (so the audit saw each one's calls)
    assert {"site.api.espn.com", "site.web.api.espn.com", "sports.core.api.espn.com", "api.the-odds-api.com",
            "v3.football.api-sports.io", "example.test", "newsapi.org", "gamma-api.polymarket.com",
            "api.elections.kalshi.com", "barttorvik.com", "www.tennisabstract.com", "www.eloratings.net",
            "api.deepseek.com", "api.open-meteo.com", "geocoding-api.open-meteo.com",
            "statsapi.mlb.com", "football-data.co.uk", "raw.githubusercontent.com", "github.com"} <= hosts, hosts
    assert recorder.outside() == [], "fetched outside the cache"
    ai = [call for call in recorder.calls if call.site.split(":")[0] in AI_FILES]
    assert ai and all(call.kind is None for call in ai)  # the AI is never cached
    board = [call for call in recorder.calls if call.site.startswith("data/odds_service.py") and "/odds" in call.url]
    assert board and all(allowlisted(call) and call.kind == "odds_board" for call in board)  # in the cache anyway


def test_the_raw_capture_reaches_its_new_hosts_through_its_own_door(recorder: Recorder) -> None:
    """v4 P1: the capture's fetches go through capture/http.py (PoliteHttp: per-host token buckets, raw.db's budgets)
    rather than the lean cache, by design (CAPTURE_FRAMES, allowlisted), and reach the new hosts: MLB StatsAPI's live
    feed, GitHub's raw and release hosts (nflverse, Greco1899, the Match Charting Project), site.web.api.espn.com
    (fightcenter); never The Odds API."""
    _drive_capture(TTLCache(name="coverage"))
    capture = [call for call in recorder.calls if "capture/http.py:get" in call.stack]
    assert capture and all(allowlisted(call) and call.kind is None for call in capture)
    assert recorder.outside() == [] and len(capture) == len(recorder.calls)  # every one through the door
    hosts = {re.sub(r"^https?://([^/]+).*$", r"\1", call.url) for call in capture}
    assert {"site.api.espn.com", "sports.core.api.espn.com", "site.web.api.espn.com", "statsapi.mlb.com",
            "github.com", "raw.githubusercontent.com", "football-data.co.uk", "api.elections.kalshi.com"} <= hosts
    assert any(re.search(r"^https://statsapi\.mlb\.com/api/v1\.1/game/\d+/feed/live$", call.url) for call in capture)
    assert any(re.search(r"^https://github\.com/nflverse/nflverse-data/releases/download/", call.url) for call in capture)
    assert any(re.search(r"^https://site\.web\.api\.espn\.com/apis/common/v3/sports/mma/ufc/fightcenter/", call.url)
               for call in capture)
    assert not any("the-odds-api" in call.url for call in capture)
    raw_fetches = [call for call in capture if any(f in call.stack for f in CAPTURE_FRAMES[1:])]
    assert len(raw_fetches) == 2  # summary_raw and scoreboard_raw, through the same door


# The markets Keeper counts each source under its own name; its TTLs are these policy kinds'
KEEPER_KINDS = {"polymarket": ("markets", "market_lines"), "kalshi": ("markets", "market_lines"),
                "ratings": ("ratings", "teams", "predictor")}


def test_every_data_fetch_has_a_policy_kind_and_ttl(recorder: Recorder) -> None:
    """The data clients' fetches carry a kind from data/cache.py's policy table and one of its TTLs (the markets
    Keeper's source names map to their policy kinds, whose TTLs the sources read from the table).
    Exceptions, each by design: the slate's own scoreboard fetch (SLATE_REUSE: the slate keeps its own intervals)
    and the odds snapshot loader (the snapshot's rules)."""
    _drive_everything(TTLCache(name="coverage"))
    wrong = []
    for call in recorder.calls:
        if call.kind is None or allowlisted(call):
            continue
        if call.kind == "scoreboard" and call.ttl == espn_client.SLATE_REUSE:
            continue
        kinds = KEEPER_KINDS.get(call.kind, (call.kind,))
        ok = all(kind in POLICY for kind in kinds) and call.ttl in {
            policy.ttl for kind in kinds for policy in POLICY[kind].values()}
        if not ok:
            wrong.append(call)
    assert wrong == []


def test_no_secret_reaches_a_cache_key_or_the_shared_table(tmp_path, monkeypatch, recorder: Recorder) -> None:
    from telegram_betting_bot_simple.data.store import Store
    store = Store(str(tmp_path / "betting.db"))
    keys = ("odds-key-not-real-123", "api-sports-key-not-real", "never-in-a-key", "legacy-odds-not-real",
            "news-key-not-real-1", "deepseek-key-not-real")
    monkeypatch.setattr(cache_module, "_SECRETS", cache_module._SECRETS)  # put back after the test
    cache_module.forbid_secrets(*keys)  # a key carrying one would raise SecretInKey
    _drive_everything(TTLCache(store=store, name="coverage"))
    with store._connect() as db:
        rows = db.execute("SELECT key, value FROM shared_cache").fetchall()
    assert rows  # the run did fill the table
    import gzip
    text = " ".join(key + gzip.decompress(value).decode("utf-8", "replace") for key, value in rows)
    assert not any(secret in text for secret in keys)
