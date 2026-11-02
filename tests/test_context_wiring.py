"""Stage 4 wiring: data/context.py's ContextService over the real venue table, Open-Meteo client, MLB StatsAPI client,
context inputs and engine (each HTTP layer faked: no network), the slate worker and its badges, /api/game's context
block, the totals' adjustment in every place the brain's chance is used (the game page's cards, the Builder's legs,
odds_get_prices and the slip), the Builder's Why line, context_log and GET /api/record/context, the units setting,
the game_context tool, the bot's line, the size budget and the cache coverage of the new fetches."""

from __future__ import annotations

import datetime
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional
from zoneinfo import ZoneInfo

import pytest
import requests
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.ai.tools import TOOLS, Toolbox
from telegram_betting_bot_simple.bot.handlers import context_line, game_card_text
from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.dashboard.lab import Lab
from telegram_betting_bot_simple.dashboard.service import DashboardService
from telegram_betting_bot_simple.data import context as context_mod
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.context import ContextService, projection_of
from telegram_betting_bot_simple.data.mlb_stats import MlbStats
from telegram_betting_bot_simple.data.odds_service import configure_books, normalize_event
from telegram_betting_bot_simple.data.slate import SlateBuilder
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.data.venues import load_venues
from telegram_betting_bot_simple.data.weather import WeatherClient
from telegram_betting_bot_simple.engine import context as engine
from telegram_betting_bot_simple.engine.factors import check_sum
from tests.test_slate import team

UTC = datetime.timezone.utc
NY = ZoneInfo("America/New_York")
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "context"
# Saturday 18 July 2026, 14:00 UTC (10 am in New York): the day of the captured MLB schedule with announced weather
NOW = datetime.datetime(2026, 7, 18, 14, 0, tzinfo=UTC).timestamp()
T0 = NOW - 3600  # the first context round, before the odds were fetched
ODDS_AT = NOW - 600  # when every saved board was fetched
NFL, MLB, NBA, NHL, EPL = ("americanfootball_nfl", "baseball_mlb", "basketball_nba", "icehockey_nhl", "soccer_epl")
TODAY = datetime.date(2026, 7, 18)
PLAYER = {"X-Player": "browser-cccc3333"}
VENUES = load_venues()


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


def at(hours: float, base: float = NOW) -> str:
    return datetime.datetime.fromtimestamp(base + hours * 3600, UTC).strftime("%Y-%m-%dT%H:%MZ")


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- the HTTP layers, faked ---------------------------------------------------------------------------------------

class Response:
    def __init__(self, body: Any, status: int = 200) -> None:
        self.body, self.status_code, self.ok = body, status, status < 400

    def json(self) -> Any:
        return self.body


def spot(row: Mapping[str, Any]) -> str:
    return f"{row['lat']:.2f},{row['lon']:.2f}"


class Meteo:
    """Open-Meteo's forecast for the comma-separated points asked for: each venue's conditions (by "lat,lon"), the
    same every hour; every call is recorded."""

    def __init__(self, clock, conditions: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        self.clock = clock
        self.conditions = conditions or {}
        self.default = {"temp_c": 20.0, "wind_kmh": 5.0, "wind_dir": 180, "gust_kmh": 9.0, "precip_prob": 0,
                        "precip_mm": 0.0}
        self.calls: List[Dict[str, Any]] = []

    def get(self, url, params=None, timeout=None, headers=None):
        params = dict(params or {})
        self.calls.append({"url": url, **params})
        if "geocoding" in url:
            return Response({"results": []})
        lats, lons = str(params["latitude"]).split(","), str(params["longitude"]).split(",")
        first = datetime.datetime.fromtimestamp(self.clock(), UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        hours = 24 * int(params.get("forecast_days") or 7)
        times = [(first + datetime.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(hours)]
        items = []
        for lat, lon in zip(lats, lons):
            c = {**self.default, **self.conditions.get(f"{lat},{lon}", {})}
            items.append({"latitude": float(lat), "longitude": float(lon), "elevation": 100.0, "hourly": {
                "time": times, "temperature_2m": [c["temp_c"]] * hours,
                "precipitation_probability": [c["precip_prob"]] * hours, "precipitation": [c["precip_mm"]] * hours,
                "wind_speed_10m": [c["wind_kmh"]] * hours, "wind_direction_10m": [c["wind_dir"]] * hours,
                "wind_gusts_10m": [c["gust_kmh"]] * hours}})
        return Response(items if len(items) > 1 else items[0])


class MlbSchedule:
    """MLB StatsAPI's schedule: the captured 18 July 2026 day once ``announced``; before that the same day with only
    Toronto's weather out, the roof open (a known state at the odds' time: the closed roof since is newer news)."""

    EARLY = {"condition": "Sunny", "temp": "84", "wind": "12 mph, Out To CF"}

    def __init__(self) -> None:
        self.announced = False
        self.calls = 0

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls += 1
        if (params or {}).get("date") != "2026-07-18":
            return Response({"dates": []})
        day = json.loads((FIXTURES / "mlb_schedule_2026-07-18.json").read_text())
        if not self.announced:
            for game in day["dates"][0]["games"]:
                weather = dict(self.EARLY) if game["venue"]["name"] == "Rogers Centre" else {}
                game["weather"] = weather
        return Response(day)


class Standings:
    """EspnClient.standings' shape (with each team's home record), counted."""

    TABLES = {NFL: {"3": {"home": "5-3"}, "9": {"home": "6-2"}}, NBA: {"7": {"home": "30-11"}, "13": {"home": "26-15"}},
              NHL: {"1": {"home": "25-12-4"}}, MLB: {"10": {"home": "30-20"}, "14": {"home": "28-22"}}}

    def __init__(self) -> None:
        self.calls: List[str] = []

    def standings(self, league, season=None):
        self.calls.append(league)
        return self.TABLES.get(league, {})


class Espn:
    """The slate's ESPN: fetch_scoreboard from fixed boards (counted)."""

    def __init__(self, boards: Dict[Any, List[Dict[str, Any]]]) -> None:
        self.boards = boards
        self.calls: List[Any] = []

    def fetch_scoreboard(self, league, day):
        self.calls.append((league, day))
        return json.loads(json.dumps(self.boards.get((league, day), [])))


def espn_event(event_id: str, home: str, away: str, start: str, ids, venue_id: str, venue: str, city: str,
               sport: str, indoor: bool = False, final: bool = False, country: str = "USA") -> Dict[str, Any]:
    state, name = ("post", "STATUS_FINAL") if final else ("pre", "STATUS_SCHEDULED")

    def side(where, label, team_id, score):
        return {"homeAway": where, "score": str(score), "team": team(team_id, label, sport)}
    return {"id": event_id, "date": start, "competitions": [{
        "id": event_id, "date": start, "neutralSite": False,
        "venue": {"id": venue_id, "fullName": venue, "address": {"city": city, "country": country}, "indoor": indoor},
        "status": {"period": 0, "displayClock": "0:00",
                   "type": {"name": name, "state": state, "completed": final, "shortDetail": "x"}},
        "competitors": [side("home", home, ids[0], 101 if final else 0), side("away", away, ids[1], 99 if final else 0)],
    }]}


def book(key: str, title: str, home: str, away: str, ml, spread=None, total=None) -> Dict[str, Any]:
    h2h = [{"name": home, "price": ml[0]}, {"name": away, "price": ml[1]}]
    if len(ml) > 2:
        h2h.append({"name": "Draw", "price": ml[2]})
    markets = [{"key": "h2h", "outcomes": h2h}]
    if spread:
        markets.append({"key": "spreads", "outcomes": [{"name": home, "price": spread[1], "point": spread[0]},
                                                       {"name": away, "price": spread[2], "point": -spread[0]}]})
    if total:
        markets.append({"key": "totals", "outcomes": [{"name": "Over", "price": total[1], "point": total[0]},
                                                      {"name": "Under", "price": total[2], "point": total[0]}]})
    return {"key": key, "title": title, "markets": markets}


def odds_event(league, event_id, home, away, start, ml, spread=None, total=None) -> Dict[str, Any]:
    books = [book("draftkings", "DraftKings", home, away, ml, spread, total),
             book("fanduel", "FanDuel", home, away, tuple(p + 0.02 for p in ml), spread, total)]
    return normalize_event({"id": event_id, "sport_key": league, "sport_title": league, "commence_time": start,
                            "home_team": home, "away_team": away, "bookmakers": books})


# The games: (league, odds id, ESPN id, home, away, start, (home id, away id), (venue id, name, city), indoor, sport,
# moneyline, spread, total)
GAMES = [
    (NFL, "nfl1", "501", "Chicago Bears", "Green Bay Packers", at(7), ("3", "9"), ("3933", "Soldier Field", "Chicago"),
     False, "nfl", (1.80, 2.10), (-3.5, 1.91, 1.91), (41.5, 1.91, 1.91)),
    (NFL, "nfl2", "502", "Buffalo Bills", "Kansas City Chiefs", at(7), ("2", "12"),
     ("11938", "Highmark Stadium", "Orchard Park"), False, "nfl", (2.0, 1.9), (1.5, 1.91, 1.91), (47.5, 1.91, 1.91)),
    (MLB, "mlb1", "601", "New York Yankees", "Los Angeles Dodgers", "2026-07-19T00:08Z", ("10", "19"),
     ("3313", "Yankee Stadium", "Bronx"), False, "mlb", (1.85, 2.05), (-1.5, 2.4, 1.6), (8.5, 1.91, 1.91)),
    (MLB, "mlb2", "602", "Toronto Blue Jays", "Chicago White Sox", "2026-07-18T19:07Z", ("14", "4"),
     ("14", "Rogers Centre", "Toronto"), False, "mlb", (1.6, 2.4), (-1.5, 2.1, 1.75), (8.5, 1.91, 1.91)),
    (NBA, "nba1", "701", "Denver Nuggets", "Los Angeles Lakers", at(9), ("7", "13"), ("892", "Ball Arena", "Denver"),
     True, "nba", (1.5, 2.7), (-6.5, 1.91, 1.91), (228.5, 1.91, 1.91)),
    (NHL, "nhl1", "801", "Boston Bruins", "New York Rangers", at(8), ("1", "13"), ("1824", "TD Garden", "Boston"),
     True, "nhl", (1.9, 2.0), (-1.5, 2.6, 1.5), (6.5, 1.91, 1.91)),
    (EPL, "epl1", "901", "Arsenal", "Everton", at(5), ("359", "368"), ("2267", "Emirates Stadium", "London"),
     False, "soccer", (1.55, 6.0, 4.2), None, (2.5, 1.85, 2.0)),
]
# Yesterday: the Lakers at home (a back-to-back today), the Rangers at home; two days ago the Nuggets at home
PAST = {
    (NBA, TODAY - datetime.timedelta(days=1)): [espn_event("700", "Los Angeles Lakers", "Phoenix Suns",
                                                           "2026-07-17T23:00Z", ("13", "21"), "1841",
                                                           "Crypto.com Arena", "Los Angeles", "basketball", True,
                                                           final=True)],
    (NBA, TODAY - datetime.timedelta(days=2)): [espn_event("699", "Denver Nuggets", "Utah Jazz", "2026-07-16T23:00Z",
                                                           ("7", "26"), "892", "Ball Arena", "Denver", "basketball",
                                                           True, final=True)],
    (NHL, TODAY - datetime.timedelta(days=1)): [espn_event("800", "New York Rangers", "Buffalo Sabres",
                                                           "2026-07-17T23:00Z", ("13", "2"), "1830",
                                                           "Madison Square Garden", "New York", "hockey", True,
                                                           final=True)],
}
LISTING = [{"key": key, "group": group, "title": title, "active": True, "has_outrights": False}
           for key, group, title in ((NFL, "American Football", "NFL"), (MLB, "Baseball", "MLB"),
                                     (NBA, "Basketball", "NBA"), (NHL, "Ice Hockey", "NHL"), (EPL, "Soccer", "EPL"))]


def espn_boards() -> Dict[Any, List[Dict[str, Any]]]:
    boards: Dict[Any, List[Dict[str, Any]]] = dict(PAST)
    for league, _, espn_id, home, away, start, ids, (vid, venue, city), indoor, sport, *_ in GAMES:
        sport_path = {"nfl": "football", "mlb": "baseball", "nba": "basketball", "nhl": "hockey"}.get(sport, sport)
        day = datetime.datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(NY).date()
        boards.setdefault((league, day), []).append(
            espn_event(espn_id, home, away, start, ids, vid, venue, city, sport_path, indoor,
                       country="England" if league == EPL else ("Canada" if city == "Toronto" else "USA")))
    return boards


class BoardOdds:
    """The odds service over the saved boards: never a request (league_games(refresh=True) fails the test)."""

    odds_ttl = 7200
    reserve = 50
    quota = {"remaining": 300, "used": 200}

    def __init__(self, store: Store) -> None:
        self.store = store

    def sports(self):
        return [dict(entry) for entry in LISTING]

    def saved_games(self, league):
        return self.store.odds_snapshot(league)

    def odds_loaded_at(self, league):
        saved = self.store.odds_snapshot(league)
        return saved[0] if saved else None

    def league_games(self, league, refresh=False):
        if refresh:
            raise AssertionError(f"the test loaded the {league} board")
        saved = self.store.odds_snapshot(league)
        return list(saved[1]) if saved else []


class NoEspn:
    def find_game(self, *args):
        return None


# Each venue's weather: before the odds (T0) and after (NOW)
SOLDIER = spot(VENUES.by_id("soldier-field"))
EMIRATES = spot(VENUES.by_id("emirates-stadium"))
YANKEE = spot(VENUES.by_id("yankee-stadium"))
ROGERS = spot(VENUES.by_id("rogers-centre"))
BEFORE = {SOLDIER: {"wind_kmh": 10.0, "temp_c": -5.0}, EMIRATES: {"temp_c": 15.0}, YANKEE: {"temp_c": 24.0},
          ROGERS: {"temp_c": 28.0}}
AFTER = {SOLDIER: {"wind_kmh": 30.0, "temp_c": -5.0}, EMIRATES: {"temp_c": 25.0}, YANKEE: {"temp_c": 24.0},
         ROGERS: {"temp_c": 28.0}}


class World:
    def __init__(self, tmp_path) -> None:
        self.clock = {"t": T0}
        now = lambda: self.clock["t"]  # noqa: E731
        self.store = Store(str(tmp_path / "betting.db"))
        for league in {g[0] for g in GAMES}:
            board = [odds_event(g[0], g[1], g[3], g[4], g[5], g[10], g[11], g[12]) for g in GAMES if g[0] == league]
            self.store.save_odds_snapshot(league, board, ODDS_AT)
        self.cache = TTLCache(store=self.store, clock=now, name="test")
        self.meteo = Meteo(now, dict(BEFORE))
        self.mlb = MlbSchedule()
        self.standings = Standings()
        self.weather = WeatherClient(self.cache, self.store, session=self.meteo)
        self.context = ContextService(VENUES, self.weather, MlbStats(self.cache, session=self.mlb), self.store,
                                      self.cache, now=now, espn=self.standings)
        self.odds = BoardOdds(self.store)
        self.espn = Espn(espn_boards())
        self.slate = SlateBuilder(self.odds, self.espn, self.store, now=now, workers=2, context=self.context)
        self.context.slate = self.slate
        self.service = DashboardService(self.odds, NoEspn(), NY, store=self.store, cache=self.cache)
        self.service.slate = self.slate
        self.service.context = self.context
        self.lab = Lab(self.service, self.store, now=now)
        # The slate's background odds loader is never needed here (every board is saved): run its listing inline
        self.slate._odds_pool.submit = lambda fn, *args: fn(*args) if not args else None  # type: ignore[method-assign]

    def round(self) -> None:
        """A slate refresh, the context round it starts, and the rebuild that round asks for."""
        self.slate.refresh()
        self.slate.wait_context(20)
        self.slate.refresh()

    def later(self) -> None:
        """An hour on: the odds were fetched 10 minutes ago, then the forecast changed and MLB announced its
        weather (a closed roof in Toronto)."""
        self.clock["t"] = NOW
        self.meteo.conditions = dict(AFTER)
        self.mlb.announced = True
        self.round()

    def block(self, league: str, game_id: str) -> Dict[str, Any]:
        found = self.context.for_game(league, game_id)
        assert found["status"] == "ready", found
        return found

    def client(self) -> TestClient:
        client = TestClient(create_app(self.service, "s3cret", login_delay=0, lab=self.lab, slate=self.slate))
        assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
        client.headers.update(PLAYER)
        assert client.put("/api/me", json={"confirm_age": True}).status_code == 200
        return client


@pytest.fixture()
def world(tmp_path) -> World:
    w = World(tmp_path)
    w.round()
    w.later()
    return w


def sums(shown: Mapping[str, Any]) -> bool:
    """The projection as sent adds up exactly (to 0.01) to the number shown."""
    projection = projection_of(shown)
    return check_sum(projection) and abs(projection.baseline + sum(f.delta for f in projection.factors)
                                         - projection.value) < 1e-6


# --- the blocks -----------------------------------------------------------------------------------------------

def test_nfl_open_air_block_with_wind_and_an_exact_attribution(world) -> None:
    block = world.block(NFL, "nfl1")
    venue = block["venue"]
    assert venue["id"] == "soldier-field" and venue["roof"] == "open" and venue["roof_state"] == "open"
    assert venue["map"].startswith("https://www.openstreetmap.org/?mlat=") and venue["approx"] is False
    weather = block["weather"]
    assert weather["source"] == "Open-Meteo" and weather["attribution"]["url"] == "https://open-meteo.com/"
    assert weather["start"]["wind_kmh"] == 30.0 and 4 <= len(weather["hours"]) <= 5
    assert weather["effect"]["unit"] == "pts" and weather["effect"]["total"] < -6
    # The margin: the board's point spread (home -3.5), home field and the remainder
    margin = block["attribution"]["margin"]
    assert margin["value"] == 3.5 and margin["baseline"] == 0 and sums(margin)
    assert [f["key"] for f in margin["factors"]][-1] == "other"
    home_field = next(f for f in margin["factors"] if f["key"] == "home_field")
    assert "ESPN standings" in home_field["source"]  # the league's home records (the Bears and Packers here)
    # The total: the league average, the weather at the odds' time (calm, but freezing) and the rest
    total = block["attribution"]["total"]
    assert total["value"] == 41.5 and sums(total)
    assert {f["key"] for f in total["factors"]} >= {"cold", "other"} and "wind" not in {
        f["key"] for f in total["factors"] if f["delta"]}
    assert block["badges"][:2] == ["adj", "wind:30"] and "cold:-5" in block["badges"]
    assert block["line"]["imperial"].startswith("19 mph wind · 23°F") and "30 km/h" in block["line"]["metric"]


def test_mlb_announced_wind_and_a_roof_closed_after_the_odds(world) -> None:
    yankees = world.block(MLB, "mlb1")
    assert yankees["weather"]["announced"] == {"condition": "Sunny", "temp_f": 76, "wind_text": "15 mph, Out To RF"}
    assert yankees["venue"]["park"]["runs"] and sums(yankees["attribution"]["total"])
    assert "out to RF" in yankees["line"]["imperial"] and "76°F" not in yankees["line"]["metric"]
    # MLB's run line is fixed at 1.5: the margin comes from the blend's chance, not the run line
    assert yankees["attribution"]["margin"]["source"].startswith("The margin the markets brain") \
        and sums(yankees["attribution"]["margin"])
    toronto = world.block(MLB, "mlb2")
    assert toronto["venue"]["roof"] == "retractable" and toronto["venue"]["roof_state"] == "closed"
    assert "roof" in toronto["badges"] and "adj" in toronto["badges"]
    assert toronto["weather"]["effect"]["total"] == 0 and toronto["line"]["metric"] == "Roof closed"
    [moved] = toronto["adjustments"]
    # The forecast at the odds' time had a warm open-air game; the roof closed since: the change is 0 - that effect
    assert moved["effect_now"] == 0 and moved["effect_at_odds"] > 0 and moved["raw"] == -moved["effect_at_odds"]
    assert moved["applied"] == pytest.approx(round(0.5 * moved["raw"], 3), abs=1e-3)
    assert moved["reason"].startswith("The roof was announced closed after the odds were fetched at ")
    assert moved["p_over"]["after"] < moved["p_over"]["before"] == 0.5


def test_nba_denver_altitude_and_a_back_to_back(world) -> None:
    block = world.block(NBA, "nba1")
    assert block["venue"]["id"] == "ball-arena" and block["weather"] is None and block["line"]["metric"] is None
    lakers = block["teams"]["away"]
    assert lakers["rest_days"] == 0 and lakers["back_to_back"] is True
    assert lakers["last"]["venue"] == "Crypto.com Arena" and lakers["last"]["opponent"] == "Phoenix Suns"
    assert 1300 <= lakers["travel_km"] <= 1400 and lakers["tz_shift_h"] == 1.0  # Los Angeles to Denver, one hour east
    assert block["teams"]["home"]["rest_days"] == 1  # the Nuggets played two days ago
    margin = block["attribution"]["margin"]
    keys = {f["key"]: f["delta"] for f in margin["factors"]}
    assert keys["altitude"] == pytest.approx(2.6, abs=0.05) and keys["away_rest"] == pytest.approx(1.25, abs=0.06)
    assert margin["value"] == 6.5 and sums(margin) and sums(block["attribution"]["total"])
    assert block["badges"] == ["alt:1578"] and block["adjustments"] == []
    # The slate's row: its own back-to-back badge, then the context's
    row = next(g for g in json.loads(world.slate.current().raw)["games"] if g["id"] == f"{NBA}:nba1")
    assert row["badges"] == ["b2b-away", "alt:1578"]


def test_nhl_and_epl_blocks(world) -> None:
    nhl = world.block(NHL, "nhl1")
    assert nhl["teams"]["away"]["back_to_back"] is True
    assert any(f["key"] == "away_rest" for f in nhl["attribution"]["margin"]["factors"])
    assert sums(nhl["attribution"]["margin"]) and sums(nhl["attribution"]["total"]) and nhl["unit"] == "goals"
    epl = world.block(EPL, "epl1")
    assert epl["venue"]["id"] == "emirates-stadium" and epl["weather"]["start"]["temp_c"] == 25.0
    assert sums(epl["attribution"]["margin"]) and sums(epl["attribution"]["total"])
    [moved] = epl["adjustments"]  # 15 -> 25 C since the odds: about 6.8% more goals, half of it applied
    assert moved["kind"] == "total" and 0 < moved["applied"] <= 0.30 and moved["unit"] == "goals"
    assert moved["p_over"]["after"] == pytest.approx(
        engine.adjusted_over("soccer", 2.5, moved["p_over"]["before"], moved["applied"]), abs=1e-4)
    assert "15 → 25°C" in moved["reason"]


def test_no_adjustment_when_the_forecast_did_not_change(world) -> None:
    bills = world.block(NFL, "nfl2")
    assert bills["adjustments"] == [] and "adj" not in bills["badges"]
    assert world.block(MLB, "mlb1")["weather"]["at_odds"]["known"] is True


def test_the_forecast_at_the_odds_time_unknown_means_attribution_only(tmp_path) -> None:
    w = World(tmp_path)
    w.clock["t"] = NOW  # the first round comes after the odds: no forecast from their time
    w.round()
    block = w.block(NFL, "nfl1")
    assert block["adjustments"] == [] and block["weather"]["at_odds"]["known"] is False
    other = next(f for f in block["attribution"]["total"]["factors"] if f["key"] == "other")
    assert "unknown" in other["evidence"] and sums(block["attribution"]["total"])


# --- the request path never fetches ----------------------------------------------------------------------------

def test_for_game_reads_the_cache_only_and_queues_the_game(tmp_path, monkeypatch) -> None:
    w = World(tmp_path)
    w.slate.refresh()  # the slate's own round (its context round runs beside it)
    w.slate.wait_context(20)
    calls = len(w.meteo.calls), w.mlb.calls, len(w.standings.calls), len(w.espn.calls)
    fresh = ContextService(VENUES, w.weather, None, w.store, TTLCache(), now=lambda: w.clock["t"])
    assert fresh.for_game(NFL, "nfl1") == {"status": "loading"}  # nothing warmed in this cache
    assert (NFL, "nfl1") in fresh._queued
    assert w.context.for_game(NFL, "nfl1")["status"] == "ready"
    assert w.context.for_game("tennis_atp", "x") == {"status": "none"}
    assert w.service.context_for("tennis_atp", "x") is None
    # Reading changed nothing upstream
    assert (len(w.meteo.calls), w.mlb.calls, len(w.standings.calls), len(w.espn.calls)) == calls


def test_a_game_asked_for_is_warmed_whatever_its_start(tmp_path) -> None:
    w = World(tmp_path)
    w.context.window = 3600  # only games within the hour are warmed by themselves
    w.round()
    assert w.context.for_game(NFL, "nfl1") == {"status": "loading"}
    w.clock["t"] += 120
    w.round()
    assert w.context.for_game(NFL, "nfl1")["status"] == "ready"


# --- Open-Meteo: one call per 50 venues, the daily cap, a simulated day ------------------------------------------

def fake_venue(n: int) -> Dict[str, Any]:
    return {"id": f"v{n}", "name": f"Ground {n}", "city": "X", "country": "US", "lat": 30 + n * 0.1,
            "lon": -100 - n * 0.1, "elevation_m": 100, "tz": "America/Chicago", "roof": "open", "surface": "grass",
            "capacity": None, "park": None, "approx": False, "source": "test"}


def picked(n: int, start: float, league: str = NFL):
    games = {(league, f"g{i}"): {"league": league, "odds_id": f"g{i}", "_ts": start + (i % 12) * 3600,
                                 "start": iso(start + (i % 12) * 3600), "status": "pre"} for i in range(n)}
    venues = {key: fake_venue(i) for i, key in enumerate(games)}
    return games, venues, set(games)


def test_warm_batches_weather_into_one_call_per_50_venues(tmp_path) -> None:
    store = Store(str(tmp_path / "b.db"))
    clock = {"t": NOW}
    cache = TTLCache(store=store, clock=lambda: clock["t"])
    meteo = Meteo(lambda: clock["t"])
    ctx = ContextService(VENUES, WeatherClient(cache, store, session=meteo), None, store, cache,
                         now=lambda: clock["t"])
    games, venues, rich = picked(60, NOW + 3600)
    found = ctx._forecasts(games, venues, rich, NOW)
    assert len(found) == 60 and len(meteo.calls) == 2  # 50 + 10 locations
    assert sorted(len(call["latitude"].split(",")) for call in meteo.calls) == [10, 50]
    assert all(3 <= call["forecast_days"] <= 7 and call["timezone"] == "GMT" for call in meteo.calls)
    ctx._forecasts(games, venues, rich, NOW)
    assert len(meteo.calls) == 2  # fresh copies: no call


def test_the_daily_cap_holds_and_the_last_forecast_is_served(tmp_path) -> None:
    store = Store(str(tmp_path / "b.db"))
    clock = {"t": NOW}
    cache = TTLCache(store=store, clock=lambda: clock["t"])
    meteo = Meteo(lambda: clock["t"])
    ctx = ContextService(VENUES, WeatherClient(cache, store, session=meteo, daily_cap=2), None, store, cache,
                         now=lambda: clock["t"])
    games, venues, rich = picked(5, NOW + 9 * 3600)
    for step in range(3):
        clock["t"] = NOW + step * 3600
        found = ctx._forecasts(games, venues, rich, clock["t"])
        # Past the cap the last copy is served (marked stale) while it is within its stale window
        assert len(found) == 5 and all(f["stale"] is (step == 2) for f in found.values())
    clock["t"] = NOW + 4 * 3600
    assert ctx._forecasts(games, venues, rich, clock["t"]) == {}  # too old to show: no weather rather than old weather
    assert len(meteo.calls) == 2 and WeatherClient(cache, store).calls_today()["calls"] == 2


def test_a_simulated_day_stays_well_under_a_thousand_calls(tmp_path) -> None:
    store = Store(str(tmp_path / "b.db"))
    clock = {"t": NOW}
    cache = TTLCache(store=store, clock=lambda: clock["t"])
    meteo = Meteo(lambda: clock["t"])
    ctx = ContextService(VENUES, WeatherClient(cache, store, session=meteo), None, store, cache,
                         now=lambda: clock["t"])
    # 80 venues with a game over the next two days (one every 36 minutes), warmed every 2 minutes for 24 hours (the
    # forecast is refreshed every 15 minutes at the soonest, so a round a minute makes the same calls)
    games = {(NFL, f"g{i}"): {"league": NFL, "odds_id": f"g{i}", "_ts": NOW + i * 2160, "status": "pre",
                              "start": iso(NOW + i * 2160)} for i in range(80)}
    venues = {key: fake_venue(i) for i, key in enumerate(games)}
    for step in range(24 * 30):
        clock["t"] = NOW + step * 120
        due = {key: game for key, game in games.items() if game["_ts"] + 3 * 3600 > clock["t"]}
        ctx._forecasts(due, venues, set(due), clock["t"])
    # About 360: hourly for every venue, every 15 minutes within 6 hours of a start, a call per 50 venues
    assert len(meteo.calls) < 500, len(meteo.calls)
    assert WeatherClient(cache, store).calls_today()["calls"] <= 1000


# --- the totals' chance moves by exactly adjusted_over, everywhere ----------------------------------------------

def nfl_adjustment(world) -> Dict[str, Any]:
    block = world.block(NFL, "nfl1")
    [moved] = block["adjustments"]
    return moved


def test_the_nfl_adjustment(world) -> None:
    moved = nfl_adjustment(world)
    # At the odds' time: 6 mph and freezing (-1 pt); now 19 mph, freezing and windy (-3.8 - 1 - 2): raw -5.8, half of
    # it -2.9, capped at -2.0
    assert moved["raw"] == pytest.approx(-5.8, abs=0.05) and moved["applied"] == -2.0 and moved["capped"] is True
    assert moved["weight"] == 0.5 and moved["cap"] == 2.0 and moved["full"] is False
    assert moved["reasons"]["imperial"].endswith("wind 6 → 19 mph.") and "km/h" in moved["reasons"]["metric"]
    assert moved["p_over"] == {"line": 41.5, "before": 0.5,
                               "after": round(engine.adjusted_over("nfl", 41.5, 0.5, -2.0), 4)}


def test_the_game_page_cards_carry_the_adjusted_totals(world) -> None:
    after = engine.adjusted_over("nfl", 41.5, 0.5, -2.0)
    card = next(c for c in world.service.games(NFL)["games"] if c["id"] == "nfl1")
    over, under = card["markets"]["totals"]["over"], card["markets"]["totals"]["under"]
    assert over["fair"] == 0.5 and over["ai"] == pytest.approx(after, abs=1e-12)
    assert under["ai"] == pytest.approx(1 - after, abs=1e-12)
    assert over["adjusted"]["applied"] == -2.0 and over["adjusted"]["reason"].startswith("The forecast changed")
    assert over["adjusted"]["before"] == 0.5 and over["ev"] == pytest.approx(after * over["price"] - 1)
    # Sides are unchanged in Stage 4
    assert "adjusted" not in card["markets"]["spreads"]["home"] and "adjusted" not in card["markets"]["h2h"]["home"]
    # The game without a change keeps the books' chance
    bills = next(c for c in world.service.games(NFL)["games"] if c["id"] == "nfl2")
    assert bills["markets"]["totals"]["over"]["ai"] == bills["markets"]["totals"]["over"]["fair"]


def test_the_game_route_carries_the_context_block(world) -> None:
    client = world.client()
    game = client.get("/api/game", params={"league": NFL, "id": "nfl1"}).json()
    assert game["context"]["status"] == "ready" and game["context"]["venue"]["name"] == "Soldier Field"
    assert game["context"]["adjustments"][0]["applied"] == -2.0
    assert game["markets"]["totals"]["over"]["adjusted"]["applied"] == -2.0
    # No context for a sport without it; the gates apply (a browser that hasn't confirmed its age gets nothing)
    other = TestClient(create_app(world.service, "s3cret", login_delay=0, lab=world.lab, slate=world.slate))
    assert other.post("/api/login", json={"password": "s3cret"}).status_code == 200
    other.headers.update({"X-Player": "browser-dddd4444"})
    assert other.get("/api/game", params={"league": NFL, "id": "nfl1"}).status_code == 400


def test_the_builder_legs_move_by_adjusted_over_and_say_why(world) -> None:
    after = engine.adjusted_over("nfl", 41.5, 0.5, -2.0)
    by_key = world.service.builder_candidates(now=NOW)["by_key"]
    over = by_key[f"{NFL}:nfl1:totals:over:41.5"]
    under = by_key[f"{NFL}:nfl1:totals:under:41.5"]
    assert over.chance == pytest.approx(after, abs=1e-12) and under.chance == pytest.approx(1 - after, abs=1e-12)
    assert over.adjusted["applied"] == -2.0 and over.adjusted["reason"].startswith("The forecast changed")
    assert over.why[0].startswith("Books' margin-free chance 50%, adjusted to ")
    # The Why's context line: the attribution's top factor for the leg's market
    assert over.why[1] == "Freezing at kick-off (23°F): the line already has about −1.0 pts for it"
    nba = by_key[f"{NBA}:nba1:h2h:home"]
    assert any(line.startswith("Altitude: the line already has about +2.6 pts for it") for line in nba.why)
    assert by_key[f"{NFL}:nfl2:totals:over:47.5"].adjusted is None
    # A Builder run's ticket legs carry it too
    client = world.client()
    result = client.post("/api/builder/run", json={"controls": {"preset": "risky", "leagues": [NFL],
                                                                "markets": ["totals"]}}).json()
    legs = [leg for ticket in result.get("tickets") or [] for leg in ticket["legs"]]
    adjusted = [leg for leg in legs if leg["ref"]["id"] == "nfl1"]
    assert all(leg["adjusted"]["applied"] == -2.0 for leg in adjusted)


def test_the_slip_and_odds_get_prices_use_the_adjusted_chance(world) -> None:
    after = engine.adjusted_over("nfl", 41.5, 0.5, -2.0)
    uid = "tg:4242"
    world.store.get_or_create_user(uid, now=NOW)
    world.store.confirm_age(uid, now=NOW)
    priced = world.lab.price(uid, {"type": "single", "stake": "10",
                                   "legs": [{"league": NFL, "id": "nfl1", "market": "totals", "side": "under"}]})
    [leg] = priced["legs"]
    assert leg["ai_prob"] == pytest.approx(1 - after, abs=1e-4) and leg["market_prob"] == 0.5
    assert leg["adjusted"]["applied"] == -2.0
    analysed = world.lab.analyse(uid, {"type": "single", "stake": "10",
                                       "legs": [{"league": NFL, "id": "nfl1", "market": "totals", "side": "under"}]},
                                 explain=False)
    notes = analysed["legs"][0]["notes"]
    assert any(note.startswith("Adjusted: The forecast changed") for note in notes)
    assert not any("injury news" in note for note in notes)
    run = Toolbox(world.lab, uid, NY, now=lambda: datetime.datetime.fromtimestamp(NOW, UTC)).run(
        "odds_get_prices", {"league": NFL, "event_id": "nfl1"})
    assert run.ok, run.content
    over = next(o for o in run.content["outcomes"] if o["ref"]["market"] == "totals" and o["ref"]["side"] == "over")
    assert over["ai_chance"] == round(after, 4) and over["market_chance"] == 0.5
    assert over["adjusted"]["applied"] == -2.0 and over["adjusted"]["reason"].startswith("The forecast changed")


# --- context_log -----------------------------------------------------------------------------------------------

def test_context_log_freezes_at_the_start_and_grades(world, monkeypatch) -> None:
    row = world.store.context_row(NFL, "espn:501")
    after = round(engine.adjusted_over("nfl", 41.5, 0.5, -2.0), 4)
    assert row["line"] == 41.5 and row["p_before"] == 0.5 and row["p_after"] == after and row["applied"] == -2.0
    assert row["reason"].startswith("The forecast changed") and row["status"] == "pending"
    bills = world.store.context_row(NFL, "espn:502")
    assert bills["applied"] == 0 and bills["p_after"] == bills["p_before"]
    assert world.store.context_row(NBA, "espn:701")["p_before"] == 0.5  # every game with a total line
    # Frozen at the start: a later write changes nothing
    changed = dict(row, p_after=0.1, start=row["start"])
    world.store.log_context([changed], now=datetime.datetime.fromisoformat(row["start"].replace("Z", "+00:00"))
                            .timestamp() + 60)
    assert world.store.context_row(NFL, "espn:501")["p_after"] == after
    # Graded after the final: 38 points, under 41.5
    finals = {"501": {"state": "final", "home": 20, "away": 18}, "502": {"state": "final", "home": 30, "away": 21}}
    monkeypatch.setattr(world.service, "final", lambda r: {**finals.get(str(r.get("espn_id")), {"state": "pending"}),
                                                           "espn_id": r.get("espn_id")})
    graded = world.service._grade_context("2026-07-20T00:00:00Z", "2026-07-01T00:00:00Z")
    assert graded == 2
    row = world.store.context_row(NFL, "espn:501")
    assert row["status"] == "graded" and row["outcome"] == "under"
    assert row["result"]["brier_before"] == 0.25 and row["result"]["brier_after"] == round(after ** 2, 6)
    record = world.client().get("/api/record/context").json()
    assert record["graded"] == 2 and record["adjusted_games"] >= 3 and record["games"] == 7
    assert record["brier_before"] == 0.25 and record["adjusted"]["graded"] == 1
    assert record["adjusted"]["brier_after"] == round(after ** 2, 4)
    # Signed in only
    anonymous = TestClient(create_app(world.service, "s3cret", login_delay=0, lab=world.lab, slate=world.slate))
    assert anonymous.get("/api/record/context").status_code == 401


# --- settings, the AI tool, the bot ----------------------------------------------------------------------------

def test_the_units_setting(world) -> None:
    client = world.client()
    assert client.get("/api/me").json()["units"] == "auto"
    assert client.put("/api/me", json={"units": "metric"}).json()["units"] == "metric"
    assert client.get("/api/me").json()["units"] == "metric"
    assert client.put("/api/me", json={"units": "imperial"}).json()["units"] == "imperial"
    assert client.put("/api/me", json={"units": "auto"}).json()["units"] == "auto"
    assert client.put("/api/me", json={"units": "kelvin"}).status_code == 422
    assert context_mod.units_for(NFL) == "imperial" and context_mod.units_for(EPL) == "metric"
    assert context_mod.units_for(EPL, "imperial") == "imperial"


def test_the_game_context_tool_through_toolbox_run(world) -> None:
    tool = next(t for t in TOOLS if t["name"] == "game_context")
    assert "strict" not in tool and tool["input_schema"]["required"] == ["league", "event_id"]
    box = Toolbox(world.lab, "tg:1", NY)
    run = box.run("game_context", {"league": NFL, "event_id": "nfl1"})
    assert run.ok, run.content
    content = run.content
    assert content["status"] == "ready" and content["venue"]["name"] == "Soldier Field"
    assert content["conditions"]["at_start"]["wind_mph"] == 19 and content["conditions"]["at_start"]["wind_kmh"] == 30
    assert content["adjustments"][0]["applied"] == -2.0 and content["teams"]["home"]["name"] == "Chicago Bears"
    for projection in content["attribution"].values():
        assert len(projection["factors"]) <= 7
        assert round(projection["baseline"] + sum(f["delta"] for f in projection["factors"]), 2) == projection["value"]
    assert "small and noisy" in content["note"] and "game_context(" in run.summary
    assert box.run("game_context", {"league": NFL, "event_id": "nope"}).content["status"] == "loading"
    bad = box.run("game_context", {"league": NFL})
    assert not bad.ok and bad.content["error"] == "invalid_input"
    from telegram_betting_bot_simple.ai.assistant import SYSTEM_PROMPT
    assert ("Context explains the line; only an adjustment moves a number; quote sizes only from game_context; "
            "weather and travel effects other than NFL wind are small and noisy.") in SYSTEM_PROMPT


def test_the_bot_card_has_one_context_line(world) -> None:
    card = world.service.game(MLB, "mlb1")
    line = context_line(card)
    assert line.startswith("🏟 Yankee Stadium · ") and "mph out to RF" in line and "°F" in line
    assert line in game_card_text(card, None)
    denver = world.service.game(NBA, "nba1")
    assert context_line(denver) == "🏟 Ball Arena · 5,177 ft"
    epl = world.service.game(EPL, "epl1")
    assert context_line(epl).startswith("🏟 Emirates Stadium · ") and "°C" in context_line(epl) \
        and "· totals adjusted · " in context_line(epl)
    # Open-Meteo's forecast is shown, so it is credited (CC BY 4.0), linked; MLB's own announcement and an indoor
    # venue need no credit
    credit = ' · weather: <a href="https://open-meteo.com/">Open-Meteo</a>'
    assert context_line(epl).endswith(credit) and line.endswith(credit)
    assert "Open-Meteo" not in context_line(denver)
    assert context_line({"league": NFL, "context": {"status": "loading"}}) is None


# --- the slate: badges and the size budget ---------------------------------------------------------------------

def test_slate_rows_carry_the_context_badges(world) -> None:
    rows = {g["id"]: g for g in json.loads(world.slate.current().raw)["games"]}
    assert rows[f"{NFL}:nfl1"]["badges"][:2] == ["adj", "wind:30"]
    assert "roof" in rows[f"{MLB}:mlb2"]["badges"]
    assert "badges" not in rows[f"{NFL}:nfl2"]  # nothing to flag
    assert not any(key.startswith("_") for row in rows.values() for key in row)


def test_the_fixture_slate_with_context_stays_within_its_budget(tmp_path) -> None:
    """Stage 2's 60-game fixture slate (tests/test_markets_wiring.py), with the context worker on: the badges may add
    at most 0.2 KB to its 3.4 KB."""
    from tests import test_markets_wiring as mw
    from tests.test_slate import NOW as SLATE_NOW, TODAY as SLATE_TODAY, odds_event as slate_odds_event

    def build(with_context: bool) -> int:
        store = Store(str(tmp_path / f"s{int(with_context)}.db"))
        rng = random.Random(3)
        boards: Dict[Any, Any] = {}
        snapshot, kalshi, poly = [], [], []
        for n in range(60):
            day = SLATE_TODAY + datetime.timedelta(days=n % 7)
            home, away = mw.TEAMS[n % 16], mw.TEAMS[(n * 7 + 3) % 16]
            start = mw.at(1 + (n % 7) * 24 + (n % 5) * 0.5)
            boards.setdefault((NBA, day), []).append(mw.espn_game(f"4019{n:05d}", home, away, start,
                                                                  ids=(str(n % 30), str(n % 29))))
            if n % 2 == 0:
                snapshot.append(slate_odds_event(NBA, f"{n:032x}", home, away, start))
                kalshi.append(mw.venue_event("kalshi", NBA, home, away, start, round(0.53 + rng.uniform(-0.02, 0.02), 2)))
                poly.append(mw.venue_event("polymarket", NBA, home, away, start,
                                           round(0.53 + rng.uniform(-0.02, 0.02), 3)))
        store.save_odds_snapshot(NBA, snapshot, SLATE_NOW - 60)
        cache = TTLCache(store=store, clock=lambda: SLATE_NOW)
        context = ContextService(VENUES, None, None, store, cache, now=lambda: SLATE_NOW) if with_context else None
        builder = mw.builder_with(store, mw.FakeEspn(boards), kalshi=mw.FakeVenue("kalshi", {NBA: kalshi}),
                                  polymarket=mw.FakeVenue("polymarket", {NBA: poly}),
                                  ratings=mw.FakeRatings({team: rng.uniform(-8, 8) for team in mw.TEAMS}),
                                  context=context)
        if context is not None:
            context.slate = builder
        mw.run_markets_round(builder)
        builder.wait_context(20)
        builder.refresh()
        games = json.loads(builder.current().raw)["games"]
        assert len(games) == 60
        if with_context:
            assert sum(any(":" in badge for badge in g.get("badges") or []) for g in games) >= 2  # Denver's altitude
        return len(builder.current().gzip)

    before, after = build(False), build(True)
    assert before <= 3400 and after - before <= 200 and after <= 3600, (before, after)


# --- the coverage of the new fetches -----------------------------------------------------------------------------

def test_the_context_round_fetches_only_through_the_cache(tmp_path, monkeypatch) -> None:
    from tests.test_cache_coverage import Recorder
    from telegram_betting_bot_simple.data import utils
    from telegram_betting_bot_simple.data.espn_client import EspnClient
    monkeypatch.setattr(utils, "_cache", TTLCache(name="coverage"))
    recorder = Recorder(monkeypatch)
    store = Store(str(tmp_path / "b.db"))
    cache = TTLCache(store=store, name="coverage")
    espn = EspnClient(cache)
    ctx = ContextService(VENUES, WeatherClient(cache, store), MlbStats(cache), store, cache, espn=espn)
    slate = SlateBuilder(BoardOdds(store), espn, store, workers=1, context=ctx)
    ctx.slate = slate
    start = datetime.datetime.now(UTC) + datetime.timedelta(hours=3)
    game = {"id": f"{MLB}:m1", "league": MLB, "odds_id": "m1", "espn_id": "1", "_ts": start.timestamp(),
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "status": "pre", "home": {"name": "Colorado Rockies"},
            "away": {"name": "San Diego Padres"}, "venue": {"name": "Coors Field", "city": "Denver"},
            "_ids": ("27", "25"), "_place": {"id": "27"}, "odds_as_of": iso(start.timestamp() - 7200),
            "chance": {"home": 0.5, "away": 0.5}, "_lines": {"total": 11.5, "p_over": 0.5}}
    unknown = dict(game, id=f"{EPL}:e1", league=EPL, odds_id="e1", espn_id="2", _ids=("1", "2"), _place={"id": "99"},
                   venue={"name": "Somewhere Park", "city": "Reading"})
    ctx.warm([game, unknown])
    hosts = {call.url.split("/")[2] for call in recorder.calls}
    assert {"api.open-meteo.com", "statsapi.mlb.com", "site.api.espn.com",
            "geocoding-api.open-meteo.com"} <= hosts, hosts
    assert recorder.outside() == [], "fetched outside the cache"


# --- settings: WEATHER_ENABLED, WEATHER_DAILY_CAP, the setup script, the services -------------------------------

@pytest.mark.parametrize("text, enabled, cap", [("", True, 1000), ("WEATHER_ENABLED=0\nWEATHER_DAILY_CAP=250\n", False, 250),
                                                ("WEATHER_ENABLED=\nWEATHER_DAILY_CAP=\n", True, 1000)])
def test_the_weather_settings(tmp_path, monkeypatch, text, enabled, cap) -> None:
    from telegram_betting_bot_simple.config import Config
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=123456:test\n" + text)
    for name in ("TELEGRAM_BOT_TOKEN", "WEATHER_ENABLED", "WEATHER_DAILY_CAP"):
        monkeypatch.delenv(name, raising=False)
    config = Config(_env_file=env_file)
    assert config.weather_enabled is enabled and config.weather_daily_cap == cap


def test_the_setup_script_writes_the_optional_weather_settings() -> None:
    text = (Path(__file__).resolve().parent.parent / "deploy" / "setup_vps.sh").read_text()
    assert "  [WEATHER_ENABLED]='^(0|1|true|false)$'" in text and "  [WEATHER_DAILY_CAP]='^[0-9]{1,5}$'" in text
    assert "#   WEATHER_ENABLED=0 " in text and "#   WEATHER_DAILY_CAP=1000 " in text  # documented in the header
    # Written with their defaults once (an existing value is kept; one given on the command line replaces it)
    assert "if ! grep -q '^WEATHER_ENABLED=' \"$ENV_FILE\"; then set_env WEATHER_ENABLED 1; fi" in text
    assert "if ! grep -q '^WEATHER_DAILY_CAP=' \"$ENV_FILE\"; then set_env WEATHER_DAILY_CAP 1000; fi" in text


def test_the_services_build_the_context_and_hand_it_the_slate(tmp_path, monkeypatch) -> None:
    from telegram_betting_bot_simple import services
    store = Store(str(tmp_path / "b.db"))
    cache = TTLCache(store=store)
    on = SimpleNamespace(weather_enabled=True, weather_daily_cap=321)
    ctx = services.build_context(on, store, cache, NoEspn())
    assert isinstance(ctx, ContextService) and ctx.weather.daily_cap == 321 and ctx.mlb is not None
    assert services.build_context(SimpleNamespace(weather_enabled=False), store, cache, NoEspn()).weather is None
    service = DashboardService(BoardOdds(store), NoEspn(), NY, store=store, cache=cache)
    service.context = ctx
    config = SimpleNamespace(markets_enabled=False, slate_live_seconds=20, slate_today_seconds=60,
                             slate_idle_seconds=600, slate_future_seconds=3600, slate_odds_per_day=12)
    slate = services.build_slate(config, service)
    assert slate.context is ctx and ctx.slate is slate
