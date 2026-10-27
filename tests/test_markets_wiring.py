"""Stage 2 wiring: the odds board's bookmaker list, the price-book rule, the slate's "Markets say", the game
route's markets block and typed factors, consensus logging and grading, GET /api/record/markets, the
markets_consensus tool, the bot's line and the settings. Fakes and a real Store in a temporary folder: no network
(requests.get fails loudly)."""

from __future__ import annotations

import copy
import datetime
import gzip
import json
import math
import random
import threading
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.ai.tools import PICK_TOOLS, TOOLS, Toolbox
from telegram_betting_bot_simple.bot.handlers import game_card_text, markets_line
from telegram_betting_bot_simple.config import DEFAULT_BOOKMAKERS, DEFAULT_SHARP_BOOKS, Config
from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.dashboard.service import DashboardService, consensus_score
from telegram_betting_bot_simple.data import odds_service
from telegram_betting_bot_simple.data.odds_service import (BookRoles, OddsService, book_roles, configure_books,
                                                           normalize_event, summarize_markets)
from telegram_betting_bot_simple.data.slate import (SlateBuilder, book_sources, chance_of, compact, ids_of,
                                                    market_detail, trim_game)
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.engine import bets, consensus, target
from telegram_betting_bot_simple.engine.tickets import TicketLeg
from tests.test_slate import NBA, NHL, NOW, TODAY, FakeEspn, FakeOdds, at, espn_game

PLAYER = {"X-Player": "browser-1234"}
NY = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(odds_service.requests, "get", refuse)


@pytest.fixture(autouse=True)
def default_roles():
    """Each test starts from the default book roles (services.build_service sets them from the settings)."""
    configure_books()
    yield
    configure_books()


@pytest.fixture()
def store(tmp_path):
    return Store(str(tmp_path / "betting.db"))


# --- fakes ------------------------------------------------------------------------------------------------

def bookmaker(key: str, title: str, home: str, away: str, home_ml: float, away_ml: float,
              spread: Optional[float] = -2.5, total: Optional[float] = 220.5, draw: Optional[float] = None
              ) -> Dict[str, Any]:
    h2h = [{"name": home, "price": home_ml}, {"name": away, "price": away_ml}]
    if draw:
        h2h.append({"name": "Draw", "price": draw})
    markets = [{"key": "h2h", "outcomes": h2h}]
    if spread is not None:
        markets.append({"key": "spreads", "outcomes": [{"name": home, "price": 1.91, "point": spread},
                                                       {"name": away, "price": 1.91, "point": -spread}]})
    if total is not None:
        markets.append({"key": "totals", "outcomes": [{"name": "Over", "price": 1.91, "point": total},
                                                      {"name": "Under", "price": 1.91, "point": total}]})
    return {"key": key, "title": title, "markets": markets}


def event(league: str, event_id: str, home: str, away: str, start: str, books: List[Dict[str, Any]]) -> Dict[str, Any]:
    return normalize_event({"id": event_id, "sport_key": league, "sport_title": league, "commence_time": start,
                            "home_team": home, "away_team": away, "bookmakers": books})


def sharp_and_us(league: str, event_id: str, home: str, away: str, start: str, sharp=(1.60, 2.55),
                 us=((1.55, 2.50), (1.57, 2.45))) -> Dict[str, Any]:
    """Pinnacle (the best moneyline price on the board, on purpose) beside two US books."""
    books = [bookmaker("pinnacle", "Pinnacle", home, away, *sharp, spread=-3.5)]
    books.append(bookmaker("betfair_ex_eu", "Betfair", home, away, sharp[0] + 0.02, sharp[1] + 0.03, spread=None,
                           total=None))
    for (key, title), prices in zip((("draftkings", "DraftKings"), ("fanduel", "FanDuel")), us):
        books.append(bookmaker(key, title, home, away, *prices))
    return event(league, event_id, home, away, start, books)


def quote(p: float, spread: float = 0.02, liquidity: float = 25000.0) -> Dict[str, Any]:
    bid, ask = round(p - spread / 2, 3), round(p + spread / 2, 3)
    return {"p": p, "bid": bid, "ask": ask, "last": p, "liquidity": liquidity, "volume": 50000.0,
            "thin": ask - bid > 0.06 or liquidity < 1000}


def venue_event(source: str, league: str, home: str, away: str, start: str, p_home: float, thin: bool = False,
                as_of: Optional[str] = None) -> Dict[str, Any]:
    spread = 0.10 if thin else 0.02
    return {"source": source, "league": league, "home": home, "away": away, "start": start,
            "url": f"https://{source}.com/event/{home.split()[-1].lower()}-{away.split()[-1].lower()}",
            "as_of": as_of or at(-0.05).replace("Z", ":00Z"), "id": f"{source}-{home}-{away}",
            "markets": {"h2h": {"home": quote(p_home, spread), "away": quote(round(1 - p_home, 3), spread)}}}


class FakeVenue:
    """Kalshi or Polymarket: events() is the worker's fetch (counted); cached_events() never fetches."""

    def __init__(self, source: str, data: Dict[str, List[Dict[str, Any]]]) -> None:
        self.source = source
        self.data = data
        self.cached: Dict[str, List[Dict[str, Any]]] = {}
        self.fetches: List[str] = []
        self.threads: List[str] = []

    def covers(self, league: str) -> bool:
        return league in self.data

    def events(self, league: str) -> List[Dict[str, Any]]:
        self.fetches.append(league)
        self.threads.append(threading.current_thread().name)
        self.cached[league] = copy.deepcopy(self.data.get(league, []))
        return self.cached[league]

    def cached_events(self, league: str) -> List[Dict[str, Any]]:
        return self.cached.get(league, [])


class FakeRatings:
    """Ratings: power() and espn_predictor() fetch (counted); game_prob(fetch=False) reads the cached table."""

    def __init__(self, table: Dict[str, float], predictor: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.table = table  # team name -> BPI-like expected margin against an average team
        self.predictor = predictor or {}
        self.loaded: Dict[str, Dict[str, float]] = {}
        self.fetches: List[Any] = []

    def power(self, league: str) -> Dict[str, Any]:
        self.fetches.append(("power", league))
        if league == NBA:
            self.loaded[league] = dict(self.table)
        return self.loaded.get(league, {})

    def espn_predictor(self, league: str, espn_event_id: Any) -> Optional[Dict[str, Any]]:
        self.fetches.append(("predictor", league, espn_event_id))
        return self.predictor.get(str(espn_event_id))

    def game_prob(self, league, home, away, neutral=False, surface=None, *, fetch=True):
        if fetch:
            self.fetches.append(("game_prob", league))
            self.power(league)
        table = self.loaded.get(league) or {}
        name = lambda side: side.get("name") if isinstance(side, dict) else side  # noqa: E731
        if name(home) not in table or name(away) not in table:
            return None
        margin = table[name(home)] - table[name(away)] + (0 if neutral else 2.0)
        p = consensus_phi(margin / 12.0)
        return {"home": p, "away": 1 - p, "source": "BPI", "as_of": "2026-10-26T06:00:00Z",
                "detail": f"BPI margin {margin:+.1f}, sd 12"}


def consensus_phi(z: float) -> float:
    from statistics import NormalDist
    return NormalDist().cdf(z)


def builder_with(store, espn, odds=None, **kwargs) -> SlateBuilder:
    return SlateBuilder(odds or FakeOdds(store), espn, store, now=kwargs.pop("now", lambda: NOW), workers=2, **kwargs)


def run_markets_round(builder: SlateBuilder) -> None:
    """One slate refresh, the markets worker's round it starts, then the rebuild that round asks for."""
    builder.refresh()
    builder.wait_markets(10)
    builder.refresh()


def games_by_id(builder: SlateBuilder) -> Dict[str, Dict[str, Any]]:
    return {g["id"]: g for g in json.loads(builder.current().raw)["games"]}


# --- the odds board: a bookmaker list at the same credit cost --------------------------------------------

def test_the_board_asks_for_ten_bookmakers_and_still_costs_three_credits(monkeypatch) -> None:
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params)))
        return SimpleNamespace(ok=True, status_code=200, json=lambda: [],
                               headers={"x-requests-remaining": "300", "x-requests-used": "200", "x-requests-last": "3"})

    monkeypatch.setattr(odds_service.requests, "get", fake_get)
    service = OddsService("k")
    service.league_games(NBA)
    [(url, params)] = calls
    assert url.endswith("/sports/basketball_nba/odds")
    assert params["bookmakers"] == DEFAULT_BOOKMAKERS and "regions" not in params
    assert len(params["bookmakers"].split(",")) == 10  # every 10 bookmakers count as one region
    assert params["markets"] == "h2h,spreads,totals" and params["oddsFormat"] == "decimal"
    assert service.region_units == 1 and service.board_cost == 3  # what regions=us cost
    # Eleven would be a second region: the settings refuse it rather than double the price
    assert OddsService("k", bookmakers=[f"book{n}" for n in range(11)]).board_cost == 6
    assert OddsService("k", bookmakers=(), regions="us").board_params()["regions"] == "us"


def test_a_refused_bookmaker_list_falls_back_to_the_region(monkeypatch, caplog) -> None:
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params))
        if "bookmakers" in params:
            return SimpleNamespace(ok=False, status_code=422, headers={}, json=lambda: {})
        return SimpleNamespace(ok=True, status_code=200, json=lambda: [], headers={})

    monkeypatch.setattr(odds_service.requests, "get", fake_get)
    service = OddsService("SECRET-KEY-123")
    assert service.league_games(NBA) == []
    assert "bookmakers" in calls[0] and calls[1]["regions"] == "us" and "bookmakers" not in calls[1]
    assert "SECRET-KEY-123" not in caplog.text and "422" in caplog.text
    assert service.board_params()["regions"] == "us"  # from then on, no second refusal per board


def test_the_settings_defaults_agree_and_bad_lists_fail_at_startup(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:x")
    for name in ("ODDS_BOOKMAKERS", "SHARP_BOOKS", "ODDS_PRICE_BOOKS", "MARKETS_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    config = Config(_env_file=None)
    assert ",".join(config.bookmakers) == DEFAULT_BOOKMAKERS == ",".join(odds_service.DEFAULT_BOOKMAKERS)
    assert config.sharp_book_keys == odds_service.DEFAULT_SHARP_BOOKS == tuple(consensus.SHARP_BOOKS)
    assert ",".join(config.sharp_book_keys) == DEFAULT_SHARP_BOOKS and config.price_book_keys == ()
    assert config.markets_enabled is True
    monkeypatch.setenv("ODDS_BOOKMAKERS", " Pinnacle , DraftKings,,")
    monkeypatch.setenv("MARKETS_ENABLED", "")
    config = Config(_env_file=None)
    assert config.bookmakers == ("pinnacle", "draftkings") and config.markets_enabled is True
    monkeypatch.setenv("ODDS_BOOKMAKERS", ",".join(f"book{n}" for n in range(11)))
    with pytest.raises(ValueError, match="at most 10"):
        Config(_env_file=None)
    monkeypatch.setenv("ODDS_BOOKMAKERS", "pinnacle;rm -rf")
    with pytest.raises(ValueError) as error:
        Config(_env_file=None)
    assert "rm -rf" not in str(error.value).split("input_value")[0]  # the bad entry isn't repeated in the message
    monkeypatch.setenv("ODDS_BOOKMAKERS", "none")
    assert Config(_env_file=None).bookmakers == ()


def test_the_setup_script_passes_the_new_settings_through() -> None:
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / "deploy" / "setup_vps.sh").read_text()
    for name in ("ODDS_BOOKMAKERS", "SHARP_BOOKS", "ODDS_PRICE_BOOKS", "MARKETS_ENABLED"):
        assert f"  [{name}]='^" in text, name
        assert f"#   {name}=" in text, name  # documented in the header


def test_consensus_inputs_read_the_saved_board_and_never_fetch(store) -> None:
    store.save_odds_snapshot(NBA, [sharp_and_us(NBA, "o1", "Boston Celtics", "New York Knicks", at(2))], NOW - 60)
    service = OddsService("k", store=store)
    inputs = service.consensus_inputs(NBA)
    assert set(inputs) == {"o1"}
    assert inputs["o1"]["books"]["pinnacle"] == {"home": 1.6, "away": 2.55}  # sharp books included: the anchor
    assert set(inputs["o1"]["books"]) == {"pinnacle", "betfair_ex_eu", "draftkings", "fanduel"}
    assert inputs["o1"]["as_of"] == "2026-10-26T21:59:00Z"
    assert service.consensus_inputs(NHL) == {}  # nothing saved: nothing, and no request


# --- the price-book rule: never a SHARP_BOOKS price -------------------------------------------------------

def test_best_price_comes_from_price_books_only() -> None:
    game = sharp_and_us(NBA, "o1", "Boston Celtics", "New York Knicks", at(2))
    summary = summarize_markets(game)
    home = summary["h2h"]["home"]
    # Pinnacle's 1.60 and Betfair's 1.62 are the best on the board, but nobody is offered them
    assert home["price"] == 1.57 and home["book"] == "FanDuel" and home["books"] == 2
    assert summary["spreads"]["home"]["book"] in ("DraftKings", "FanDuel")
    # The fair chance still averages every book, the sharp ones included
    us_only = summarize_markets({**game, "books": [b for b in game["books"] if b["key"] in ("draftkings", "fanduel")]})
    assert home["fair"] != pytest.approx(us_only["h2h"]["home"]["fair"])
    # A market only sharp books price is left out altogether
    sharp_only = summarize_markets({**game, "books": [b for b in game["books"] if b["key"] == "pinnacle"]})
    assert sharp_only == {}
    roles = book_roles()
    assert roles.role("pinnacle") == roles.role("Pinnacle") == roles.role({"key": "matchbook"}) == "sharp"
    assert roles.role("Betfair") == "sharp" and roles.role("Caesars") == roles.role("theScore Bet") == "price"
    assert roles.role("smarkets") == "other" and roles.role("kalshi") == "other"  # exchanges: never a price
    assert roles.role("lowvig") == "price"  # a US book from a board saved before the list: bettable


def test_odds_price_books_narrows_every_price_to_those_books() -> None:
    configure_books(price=["fanduel"])
    game = sharp_and_us(NBA, "o1", "Boston Celtics", "New York Knicks", at(2), us=((1.59, 2.50), (1.50, 2.70)))
    summary = summarize_markets(game)
    assert summary["h2h"]["home"]["book"] == "FanDuel" and summary["h2h"]["home"]["price"] == 1.5
    assert BookRoles.make(price=["pinnacle", "draftkings"]).price == frozenset({"draftkings"})  # sharp wins


class BoardOdds:
    """The odds service as the games service uses it, over fixed games (never fetches)."""

    quota = {"remaining": 300, "used": 200}
    odds_ttl = 7200
    reserve = 50

    def __init__(self, games: List[Dict[str, Any]], store: Optional[Store] = None, fetched_at: float = NOW - 60):
        self.games = games
        self.store = store
        self.fetched_at = fetched_at

    def sports(self):
        return [{"key": NBA, "group": "Basketball", "title": "NBA", "active": True, "has_outrights": False}]

    def league_games(self, league, refresh=False):
        return copy.deepcopy(self.games)

    def odds_loaded_at(self, league):
        return self.fetched_at

    def saved_games(self, league):
        return (self.fetched_at, copy.deepcopy(self.games))


class NoEspn:
    def find_game(self, *args):
        return None

    def standings(self, league, season=None):
        return {}


def two_games() -> List[Dict[str, Any]]:
    return [sharp_and_us(NBA, "g1", "Boston Celtics", "New York Knicks", "2030-01-01T00:00:00Z"),
            sharp_and_us(NBA, "g2", "Denver Nuggets", "Utah Jazz", "2030-01-01T01:00:00Z", sharp=(1.30, 4.10),
                         us=((1.28, 3.80), (1.27, 3.90)))]


def test_slip_pricing_and_the_ai_bets_never_use_a_sharp_price() -> None:
    service = DashboardService(BoardOdds(two_games()), NoEspn(), NY)
    assert set(service._book_prices(NBA, "g1", "h2h", "home", None)) == {"DraftKings", "FanDuel"}
    picks = [{"league": NBA, "id": "g1", "market": "h2h", "side": "home"},
             {"league": NBA, "id": "g2", "market": "h2h", "side": "home"}]
    multi = service.price_slip(picks, "multi", 10.0, {})
    assert {entry["book"] for entry in multi["books"]} == {"DraftKings", "FanDuel"}
    single = service.price_slip(picks[:1], "single", 10.0, {})
    assert single["legs"][0]["odds"]["decimal"] == pytest.approx(1.57)  # FanDuel's, not Pinnacle's 1.60
    tiers = service.ai_bets(NBA)["tiers"]
    books = {ticket["book"] for tier in tiers.values() for ticket in tier["bets"]}
    assert books and books <= {"DraftKings", "FanDuel"}
    # The engine itself refuses a sharp book, whoever builds the candidates
    candidate = bets.Candidate(event_id="g1", game="x", market="h2h", side="home", point=None, label="x",
                               probability=0.7, fair=0.62, prices={"Pinnacle": 1.9, "Betfair": 1.95})
    assert all(not tier for tier in (bets.build_tier(t, [candidate]) for t in bets.TIERS))
    view = service.game(NBA, "g1")["books"]
    assert {b["title"]: b["role"] for b in view} == {"Pinnacle": "sharp", "Betfair": "sharp", "DraftKings": "price",
                                                     "FanDuel": "price"}


def test_the_target_builder_never_builds_at_a_sharp_book() -> None:
    def leg(n, book, price):
        return TicketLeg(key=f"{NBA}:g{n}:h2h:home", league=NBA, event_id=f"g{n}", market="h2h", side="home",
                         point=None, price=Decimal(str(price)), label=f"Team {n}", game=f"g{n}",
                         start="2030-01-01T00:00:00Z", ai_prob=0.55, market_prob=0.52, push_prob=0.0, book=book)
    sharp_only = [leg(1, "Pinnacle", 2.0), leg(2, "Betfair", 2.0)]
    out = target.build_for_target(sharp_only, Decimal("10"), Decimal("40"), 2)
    assert out["tickets"] == []
    mixed = [leg(1, "Pinnacle", 2.10), leg(1, "DraftKings", 1.95), leg(2, "Matchbook", 2.1), leg(2, "FanDuel", 1.98)]
    out = target.build_for_target(mixed, Decimal("10"), Decimal("38"), 2)
    used = {l["book"] for ticket in out["tickets"] for l in ticket["legs"]}
    assert used and used <= {"DraftKings", "FanDuel"}


def test_a_pasted_slip_s_better_price_is_never_a_sharp_one() -> None:
    from telegram_betting_bot_simple.dashboard.lab import Lab
    lab = Lab.__new__(Lab)
    entry = {"status": "matched", "best_price": None, "better_price": False}  # as _paste_entry starts one
    card = {"prediction": None}
    sharp = lab._board_chances(dict(entry), card, {"price": 2.4, "book": "Pinnacle", "fair": 0.42}, Decimal("2.10"))
    assert sharp["best_price"] is None and sharp["better_price"] is False and sharp["market_chance"] == 0.42
    us = lab._board_chances(dict(entry), card, {"price": 2.4, "book": "FanDuel", "fair": 0.42}, Decimal("2.10"))
    assert us["better_price"] is True and us["best_price"]["book"] == "FanDuel"


# --- the slate: "Markets say" on every game, in a compact block ------------------------------------------

TEAMS = ["Boston Celtics", "New York Knicks", "Denver Nuggets", "Utah Jazz", "Miami Heat", "Chicago Bulls",
         "Phoenix Suns", "Sacramento Kings", "Golden State Warriors", "Dallas Mavericks", "Atlanta Hawks",
         "Detroit Pistons", "Orlando Magic", "Toronto Raptors", "Houston Rockets", "Memphis Grizzlies"]


def fixture_slate(store, games: int = 60):
    """The fixture slate: 60 NBA games over seven days. Half have a saved odds board (Pinnacle, Betfair and two
    US books), about half a Kalshi market and 60% a Polymarket one (one in ten games neither, nor odds: ratings
    only), every team a BPI-like rating, the odds games the model's own chance and one game in seven ESPN's
    predictor."""
    rng = random.Random(7)
    boards: Dict[Any, Any] = {}
    snapshot, kalshi, poly, predictor = [], [], [], {}
    for n in range(games):
        day = TODAY + datetime.timedelta(days=n % 7)
        home, away = TEAMS[n % 16], TEAMS[(n * 7 + 3) % 16]
        start = at(1 + (n % 7) * 24 + (n % 5) * 0.5)
        espn_id = f"4019{n:05d}"
        boards.setdefault((NBA, day), []).append(espn_game(espn_id, home, away, start, ids=(str(n % 30), str(n % 29))))
        p = round(rng.uniform(0.25, 0.78), 3)
        if n % 2 == 0:
            fair = lambda q: (round(1 / (q * 1.025), 2), round(1 / ((1 - q) * 1.025), 2))  # noqa: E731
            snapshot.append(sharp_and_us(NBA, f"{n:032x}", home, away, start, sharp=fair(p),
                                         us=(fair(p - 0.01), fair(p + 0.012))))
            # Logged two minutes before the test's NOW (the store stamps the wall clock): the model's chance is an
            # opinion from before every game's start
            with patch("telegram_betting_bot_simple.data.store.time.time", return_value=NOW - 120):
                store.log_ai_prediction({"league": NBA, "event_id": f"{n:032x}", "espn_id": espn_id, "home": home,
                                         "away": away, "start": start,
                                         "payload": {"home_win": p,
                                                     "model_home_win": round(p + rng.uniform(-0.06, 0.06), 3)}})
        if n % 3 and n % 5:
            kalshi.append(venue_event("kalshi", NBA, home, away, start, round(p + rng.uniform(-0.03, 0.03), 2)))
        if n % 4 and n % 5:
            poly.append(venue_event("polymarket", NBA, home, away, start, round(p + rng.uniform(-0.03, 0.03), 3),
                                    thin=n % 9 == 0))
        if n % 7 == 0:
            predictor[espn_id] = {"home": round(p + 0.02, 3), "away": round(0.98 - p, 3), "as_of": at(-0.2)}
    store.save_odds_snapshot(NBA, snapshot, NOW - 60)
    sources = {"kalshi": FakeVenue("kalshi", {NBA: kalshi}), "polymarket": FakeVenue("polymarket", {NBA: poly}),
               "ratings": FakeRatings({team: rng.uniform(-8, 8) for team in TEAMS}, predictor)}
    builder = builder_with(store, FakeEspn(boards), **sources)
    return builder, sources


def test_every_game_gets_a_blend_and_a_compact_markets_block(store) -> None:
    builder, sources = fixture_slate(store)
    run_markets_round(builder)
    current = builder.current()
    games = json.loads(current.raw)["games"]
    assert len(games) == 60
    with_mkt = [g for g in games if g.get("mkt")]
    assert len(with_mkt) == 60  # every game has at least the BPI rating
    for game in with_mkt:
        mkt, chance = game["mkt"], game["chance"]
        assert chance["src"] == "blend" and chance["home"] == mkt["p"]["home"] and chance["away"] == mkt["p"]["away"]
        assert mkt["fav"] == max(mkt["p"], key=mkt["p"].get) and mkt["conf"] in ("normal", "low")
        assert set(mkt) <= {"p", "fav", "conf", "dis", "spr", "src", "as_of"}
        assert all(len(entry) == 3 and entry[2] in ("ok", "thin", "stale") for entry in mkt["src"])
    # A game with everything: the books (sharp and US), both venues, the rating, the model and ESPN's predictor
    full = next(g for g in games if {"Sharp", "US", "Kalshi", "Poly", "BPI", "Model", "ESPN"}
                <= {entry[0] for entry in g["mkt"]["src"]})
    assert full["mkt"]["conf"] == "normal"
    # Ratings alone (no odds, no market): low confidence
    low = [g for g in games if {entry[0] for entry in g["mkt"]["src"]} == {"BPI"}]
    assert low and all(g["mkt"]["conf"] == "low" for g in low)
    # A dense slate (every price different, up to seven opinions a game) stays small: about 3.2 KB of it are the
    # varied prices, about 1.4 KB "Markets say"
    assert len(current.gzip) <= 5000, len(current.gzip)
    assert gzip.decompress(current.gzip) == current.raw


def test_the_bookmakers_chance_stays_as_fair_when_the_blend_takes_over_chance(store) -> None:
    # Today shows the books' margin-free chance as "Mkt" and measures the AI's edge against it: the blend
    # replacing ``chance`` must not take that figure away
    from telegram_betting_bot_simple.data.slate import odds_view
    builder, sources = fixture_slate(store)
    run_markets_round(builder)
    games = games_by_id(builder)
    saved = {event["id"]: event for event in store.odds_snapshot(NBA)[1]}
    priced = [g for g in games.values() if g.get("odds")]
    assert len(priced) == 30
    for game in priced:
        assert game["chance"]["src"] == "blend"
        book = odds_view(saved[game["id"].split(":", 1)[1]])[1]
        assert game["fair"] == {k: book[k] for k in ("home", "away", "draw") if book.get(k) is not None}
    # The two are different figures: on most priced games the venues, ratings and model move the blend
    assert sum(abs(g["fair"]["home"] - g["chance"]["home"]) > 0.005 for g in priced) >= 10
    # Games without a board have no books' chance to keep
    assert not [g for g in games.values() if not g.get("odds") and "fair" in g]


def test_the_fixture_slate_with_markets_stays_within_3_4_kb_per_60_games(store) -> None:
    """Stage 1's 60-game fixture slate (tests/test_slate.py: odds on half the games), now with a "Markets say"
    block on every game: the books, Kalshi and Polymarket on the odds games and a rating on all of them."""
    from tests.test_slate import odds_event
    rng = random.Random(3)
    boards: Dict[Any, Any] = {}
    snapshot, kalshi, poly = [], [], []
    for n in range(60):
        day = TODAY + datetime.timedelta(days=n % 7)
        home, away = TEAMS[n % 16], TEAMS[(n * 7 + 3) % 16]
        start = at(1 + (n % 7) * 24 + (n % 5) * 0.5)
        boards.setdefault((NBA, day), []).append(espn_game(f"4019{n:05d}", home, away, start,
                                                           ids=(str(n % 30), str(n % 29))))
        if n % 2 == 0:
            snapshot.append(odds_event(NBA, f"{n:032x}", home, away, start))
            kalshi.append(venue_event("kalshi", NBA, home, away, start, round(0.53 + rng.uniform(-0.02, 0.02), 2)))
            poly.append(venue_event("polymarket", NBA, home, away, start, round(0.53 + rng.uniform(-0.02, 0.02), 3)))
    store.save_odds_snapshot(NBA, snapshot, NOW - 60)
    builder = builder_with(store, FakeEspn(boards), kalshi=FakeVenue("kalshi", {NBA: kalshi}),
                           polymarket=FakeVenue("polymarket", {NBA: poly}),
                           ratings=FakeRatings({team: rng.uniform(-8, 8) for team in TEAMS}))
    run_markets_round(builder)
    current = builder.current()
    games = json.loads(current.raw)["games"]
    assert len(games) == 60 and all(g.get("mkt") for g in games)
    assert sum(len(g["mkt"]["src"]) == 4 for g in games) == 30  # US books, Kalshi, Polymarket, BPI
    assert len(current.gzip) <= 3400, len(current.gzip)


def test_the_markets_block_matches_the_brain_and_flags_disagreement(store) -> None:
    espn = FakeEspn({(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(3)),
                                    espn_game("e2", "Denver Nuggets", "Utah Jazz", at(4))]})
    store.save_odds_snapshot(NBA, [sharp_and_us(NBA, "o1", "Boston Celtics", "New York Knicks", at(3),
                                                sharp=(1.60, 2.55), us=((1.55, 2.50), (1.57, 2.45)))], NOW - 60)
    kalshi = FakeVenue("kalshi", {NBA: [venue_event("kalshi", NBA, "Boston Celtics", "New York Knicks", at(3), 0.70),
                                        venue_event("kalshi", NBA, "Denver Nuggets", "Utah Jazz", at(4), 0.55)]})
    builder = builder_with(store, espn, kalshi=kalshi)
    run_markets_round(builder)
    games = games_by_id(builder)
    game = games["basketball_nba:o1"]
    detail = builder.markets_for(NBA, odds_id="o1")
    assert detail is builder.markets_for(NBA, espn_id="e1")
    names = [s["name"] for s in detail["sources"]]
    assert names == ["Sharp", "US books", "Kalshi"]
    # The blend is engine/consensus.py's, over these very sources
    expected = consensus.blend(detail["sources"], NBA, now=NOW, start=NOW + 3 * 3600)
    assert detail["blend"]["p"]["home"] == pytest.approx(expected["p"]["home"])
    assert game["chance"]["home"] == round(expected["p"]["home"], 3)
    # Kalshi says 70%, the books about 61-62%: 5 points or more apart is a disagreement
    assert game["mkt"]["dis"] is True and game["mkt"]["spr"] >= 5
    assert [entry[0] for entry in game["mkt"]["src"]] == ["Sharp", "US", "Kalshi"]
    assert detail["links"] == [{"name": "Kalshi", "url": "https://kalshi.com/event/celtics-knicks"}]
    # The other game has only Kalshi: a normal blend of one market, with no books on the slate
    other = games["basketball_nba:espn:e2"]
    assert other["mkt"]["src"] == [["Kalshi", 0.55, "ok"]] and other["chance"]["home"] == 0.55
    assert "odds" not in other


def test_a_thin_market_is_shown_but_not_blended(store) -> None:
    espn = FakeEspn({(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(3))]})
    poly = FakeVenue("polymarket", {NBA: [venue_event("polymarket", NBA, "Boston Celtics", "New York Knicks", at(3),
                                                      0.80, thin=True)]})
    ratings = FakeRatings({"Boston Celtics": 4.0, "New York Knicks": 1.0})
    builder = builder_with(store, espn, polymarket=poly, ratings=ratings)
    run_markets_round(builder)
    [game] = games_by_id(builder).values()
    assert [entry[0] for entry in game["mkt"]["src"]] == ["Poly", "BPI"]
    assert game["mkt"]["src"][0][2] == "thin" and game["mkt"]["conf"] == "low"  # no market blended: ratings only
    detail = builder.markets_for(NBA, espn_id="e1")
    assert detail["blend"]["used"] == ["BPI"]


def test_the_client_can_rebuild_the_trimmed_fields() -> None:
    game = {"id": f"{NBA}:abc", "league": NBA, "odds_id": "abc", "espn_id": "401", "_ts": 1.0,
            "home": {"name": "Boston Celtics", "short": "Celtics", "abbr": "BOS",
                     "logo": "https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/bos.png"},
            "away": {"name": "Fernando Verdasco", "short": "F. Verdasco",
                     "logo": "https://a1.espncdn.com/flags/esp.png"}}
    trimmed = trim_game(game)
    assert trimmed["id"] == f"{NBA}:abc" and "odds_id" not in trimmed and trimmed["espn_id"] == "401"
    assert "_ts" not in trimmed
    # The full name stays beside the short name and abbreviation: Today's search finds "Boston" or "Boston Celtics"
    # in it (the city isn't in "Celtics"/"BOS"), and a slip says "Boston Celtics to win" as the game page does
    assert trimmed["home"] == {"name": "Boston Celtics", "short": "Celtics", "abbr": "BOS",
                               "logo": "i/teamlogos/nba/500/scoreboard/bos.png"}
    # Another ESPN host keeps its whole address
    assert trimmed["away"]["name"] == "Fernando Verdasco"
    assert trimmed["away"]["logo"] == "https://a1.espncdn.com/flags/esp.png"
    assert ids_of(trimmed) == ("abc", "401") and ids_of({"id": f"{NBA}:espn:77", "league": NBA}) == (None, "77")
    assert trim_game(trimmed) == trimmed  # trimming twice changes nothing (a saved slate is served again)


# --- the game route: the markets block and typed factors -------------------------------------------------

def app_client(service: DashboardService, slate: Optional[SlateBuilder] = None) -> TestClient:
    from tests.test_slate import signed_in
    return signed_in(TestClient(create_app(service, "s3cret", login_delay=0, slate=slate)))


def test_the_game_route_carries_the_markets_block_and_typed_factors(store) -> None:
    service = DashboardService(BoardOdds(two_games(), store), NoEspn(), NY, store=store)
    game = app_client(service).get("/api/game", params={"league": NBA, "id": "g1"}).json()
    block = game["consensus"]
    assert set(block) == {"blend", "sources", "links"}
    assert [s["name"] for s in block["sources"]] == ["Sharp", "US books"]
    assert all(set(s) == {"name", "kind", "p", "as_of", "status", "url", "detail"} for s in block["sources"])
    expected = consensus.blend(block["sources"], NBA, now=NOW, start="2030-01-01T00:00:00Z")
    assert block["blend"]["p"]["home"] == pytest.approx(expected["p"]["home"])
    assert block["blend"]["confidence"] == "normal" and block["links"] == []
    assert game["markets"]["h2h"]["home"]["book"] == "FanDuel"  # the price grid keeps its old name and shape
    # The model's explanation as typed factors: each number is a baseline plus factors that add up to it
    projections = game["prediction"]["projections"]
    assert {"margin", "total", "home_win"} <= set(projections)
    for projection in projections.values():
        total = projection["baseline"] + sum(f["delta"] for f in projection["factors"])
        assert total == pytest.approx(projection["value"], abs=0.01), projection
        assert all(set(f) >= {"key", "label", "delta", "unit", "direction", "evidence", "source", "confidence"}
                   for f in projection["factors"])
    assert isinstance(game["prediction"]["factors"], list)  # the old text lines stay for older pages


def test_the_game_route_adds_what_the_dashboard_logged_and_freezes_at_the_start(store) -> None:
    service = DashboardService(BoardOdds(two_games(), store), NoEspn(), NY, store=store)
    logged = market_detail(NBA, [venue_kalshi_source(0.70, "2029-12-31T23:30:00Z")], None, NOW)
    store.log_consensus([{"league": NBA, "game_key": "odds:g1", "event_id": "g1", "espn_id": None, "home": "Boston Celtics",
                          "away": "New York Knicks", "start": "2030-01-01T00:00:00Z", "sources": logged["sources"],
                          "blend": logged["blend"]}], now=NOW)
    detail = service.markets_detail(NBA, "g1", now=NOW)
    names = {s["name"]: s for s in detail["sources"]}
    assert set(names) == {"Sharp", "US books", "Kalshi"}  # fresh books, plus the venue the dashboard logged
    assert names["Kalshi"]["url"] == "https://kalshi.com/markets/kxnbagame"
    # A logged price an hour old or more is stale: shown, not blended (the dashboard stopped refreshing it)
    old = market_detail(NBA, [venue_kalshi_source(0.70, "2026-10-26T18:00:00Z")], None, NOW)
    store.log_consensus([{"league": NBA, "game_key": "odds:g2", "event_id": "g2", "espn_id": None, "home": "Denver Nuggets",
                          "away": "Utah Jazz", "start": "2030-01-01T01:00:00Z", "sources": old["sources"],
                          "blend": old["blend"]}], now=NOW)
    stale = {s["name"]: s for s in service.markets_detail(NBA, "g2", now=NOW)["sources"]}["Kalshi"]
    assert stale["status"] == "stale"
    # A game that has started shows its last logged "Markets say", frozen
    before = market_detail(NBA, [venue_kalshi_source(0.70, "2026-10-26T21:55:00Z")], None, NOW)
    store.log_consensus([{"league": NBA, "game_key": "odds:past", "event_id": "past", "espn_id": None, "home": "A",
                          "away": "B", "start": "2026-10-26T23:00:00Z", "sources": before["sources"],
                          "blend": before["blend"]}], now=NOW)
    later = DashboardService(BoardOdds([], store), NoEspn(), NY, store=store)
    assert later.markets_detail(NBA, "past", now=NOW + 2 * 3600) == before
    assert later.markets_detail(NBA, "past", now=NOW) == before  # still to start, but nothing newer to add
    assert later.markets_detail(NBA, "nobody-knows", now=NOW) is None


def venue_kalshi_source(p: float, as_of: str) -> Dict[str, Any]:
    from telegram_betting_bot_simple.data.slate import make_source
    return make_source("Kalshi", "prediction_market", {"home": p, "away": 1 - p}, as_of, "ok",
                       "https://kalshi.com/markets/kxnbagame", "mid price; $25,000 liquidity")


def test_the_game_route_reads_the_slate_s_detail_when_it_has_one(store) -> None:
    builder, sources = fixture_slate(store)
    run_markets_round(builder)
    odds = OddsService("k", store=store)
    service = DashboardService(odds, NoEspn(), NY, store=store)
    service.slate = builder
    odds_id = f"{0:032x}"
    detail = service.markets_detail(NBA, odds_id)
    assert detail is builder.markets_for(NBA, odds_id=odds_id)
    assert {"Sharp", "US books", "BPI", "Model"} <= {s["name"] for s in detail["sources"]}


# --- consensus logging: frozen at the start, graded after the final, the record ----------------------------

class FinalEspn(FakeEspn):
    """ESPN scoreboards, plus game results for grading."""

    def __init__(self, boards, results=None):
        super().__init__(boards)
        self.results = results or {}

    def game_result(self, league, event_id):
        return self.results.get(event_id, {"state": "pending"})

    def find_game(self, *args):
        return None

    def standings(self, league, season=None):
        return {}


def test_consensus_is_logged_until_the_start_then_frozen_and_graded_after_the_final(store) -> None:
    clock = {"now": NOW}
    espn = FinalEspn({(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(1))]})
    store.save_odds_snapshot(NBA, [sharp_and_us(NBA, "o1", "Boston Celtics", "New York Knicks", at(1))], NOW - 60)
    kalshi = FakeVenue("kalshi", {NBA: [venue_event("kalshi", NBA, "Boston Celtics", "New York Knicks", at(1), 0.60)]})
    builder = builder_with(store, espn, now=lambda: clock["now"], kalshi=kalshi)
    run_markets_round(builder)
    [row] = store.consensus_rows()
    assert (row["league"], row["game_key"], row["event_id"], row["espn_id"]) == (NBA, "espn:e1", "o1", "e1")
    assert row["status"] == "pending" and [s["name"] for s in row["sources"]] == ["Sharp", "US books", "Kalshi"]
    first = row["blend"]["p"]["home"]
    # Before the start every change is logged ...
    kalshi.data[NBA][0]["markets"]["h2h"] = {"home": quote(0.66), "away": quote(0.34)}
    clock["now"] = NOW + 600
    run_markets_round(builder)
    [row] = store.consensus_rows()
    assert row["blend"]["p"]["home"] > first and row["logged_at"] == NOW + 600
    before_start = row["blend"]
    # ... after it nothing changes: the row holds the last refresh before the start
    kalshi.data[NBA][0]["markets"]["h2h"] = {"home": quote(0.30), "away": quote(0.70)}
    clock["now"] = NOW + 3700
    run_markets_round(builder)
    [row] = store.consensus_rows()
    assert row["blend"] == before_start and row["logged_at"] == NOW + 600
    # The slate keeps showing the last "Markets say" before the start for a game that has started
    [game] = games_by_id(builder).values()
    assert game["mkt"]["p"]["home"] == round(before_start["p"]["home"], 3)
    # Graded after the final: each source's and the blend's Brier score and log loss
    espn.results["e1"] = {"state": "final", "home": 112, "away": 104}
    service = DashboardService(OddsService("k", store=store), espn, NY, store=store)
    service.grade_due()
    [row] = store.consensus_rows()
    assert row["status"] == "graded" and row["outcome"] == "home"
    result = row["result"]
    assert result["score"] == {"home": 112, "away": 104}
    for source in row["sources"]:
        p = source["p"]["home"]
        assert result["scores"][source["name"]]["brier"] == pytest.approx((1 - p) ** 2, abs=1e-6)
        assert result["scores"][source["name"]]["log_loss"] == pytest.approx(-math.log(p), abs=1e-6)
    assert result["blend"]["brier"] == pytest.approx((1 - before_start["p"]["home"]) ** 2, abs=1e-6)
    assert service.grade_due() == 0  # graded once


def test_grading_follows_espn_s_home_and_away_and_voids_what_it_can_t_settle(store) -> None:
    detail = market_detail(NBA, [venue_kalshi_source(0.7, "2026-10-26T21:00:00Z")], None, NOW)
    base = {"league": NBA, "home": "Sacramento Kings", "away": "Phoenix Suns", "start": "2026-10-26T23:00:00Z",
            "sources": detail["sources"], "blend": detail["blend"]}
    store.log_consensus([{**base, "game_key": "espn:e4", "event_id": "o4", "espn_id": "e4", "swapped": True},
                         {**base, "game_key": "espn:e5", "event_id": "o5", "espn_id": "e5"},
                         {**base, "league": "tennis_atp_paris", "game_key": "odds:t1", "event_id": "t1"}], now=NOW)
    # ESPN lists Phoenix at home (the row is the odds feed's way round): Phoenix 120, Sacramento 101
    espn = FinalEspn({}, {"e4": {"state": "final", "home": 120, "away": 101}, "e5": {"state": "void"}})
    service = DashboardService(OddsService("k", store=store), espn, NY, store=store)
    service.grade_due()
    rows = {row["game_key"]: row for row in store.consensus_rows()}
    assert rows["espn:e4"]["outcome"] == "away" and rows["espn:e4"]["result"]["score"] == {"home": 101, "away": 120}
    assert rows["espn:e5"]["status"] == "void" and rows["odds:t1"]["status"] == "void"
    assert consensus_score({"home": 0.6, "away": 0.4}, "draw") is None  # a tie on a two-way market isn't scored
    three = consensus_score({"home": 0.5, "draw": 0.3, "away": 0.2}, "draw")
    assert three["brier"] == pytest.approx(0.5 * (0.25 + 0.49 + 0.04)) and three["log_loss"] == pytest.approx(1.2039728)


def test_the_markets_record_per_sport_and_source(store) -> None:
    def graded(key, sources, outcome, blend_p):
        detail = market_detail(NBA, sources, None, NOW)
        detail["blend"]["p"] = blend_p
        store.log_consensus([{"league": NBA, "game_key": key, "event_id": key, "espn_id": key, "home": "A",
                              "away": "B", "start": "2026-10-26T23:00:00Z", "sources": detail["sources"],
                              "blend": detail["blend"]}], now=NOW)
        result = {"outcome": outcome, "blend": consensus_score(blend_p, outcome),
                  "scores": {s["name"]: {**consensus_score(s["p"], outcome), "kind": s["kind"]}
                             for s in detail["sources"]}}
        store.grade_consensus(NBA, key, "graded", outcome, result)

    sharp = lambda p: book_sources({"pinnacle": {"home": 1 / p, "away": 1 / (1 - p)}}, "2026-10-26T21:00:00Z", NOW)  # noqa
    graded("g1", sharp(0.6) + [venue_kalshi_source(0.7, "2026-10-26T21:00:00Z")], "home", {"home": 0.62, "away": 0.38})
    graded("g2", sharp(0.4), "away", {"home": 0.40, "away": 0.60})
    service = DashboardService(OddsService("k", store=store), NoEspn(), NY, store=store)
    client = app_client(service)
    record = client.get("/api/record/markets").json()
    [sport] = record["sports"]
    assert sport["league"] == NBA and sport["title"] == "NBA" and sport["n"] == 2 and record["games"] == 2
    assert sport["blend"] == {"n": 2, "brier": pytest.approx(round((0.38 ** 2 + 0.40 ** 2) / 2, 4)),
                              "log_loss": pytest.approx(round(-(math.log(0.62) +
                                                                math.log(0.6)) / 2, 4))}
    by_name = {s["name"]: s for s in sport["sources"]}
    assert by_name["Sharp"]["n"] == 2 and by_name["Sharp"]["brier"] == pytest.approx(0.16, abs=1e-4)
    # Kalshi priced one game: its figures, and the blend's on that same game
    assert by_name["Kalshi"]["n"] == 1 and by_name["Kalshi"]["brier"] == pytest.approx(0.09, abs=1e-4)
    assert by_name["Kalshi"]["blend_brier"] == pytest.approx(0.1444, abs=1e-4)
    assert TestClient(create_app(service, "s3cret", login_delay=0)).get("/api/record/markets").status_code == 401


# --- the AI tool, the bot's line ------------------------------------------------------------------------

def test_the_markets_consensus_tool_and_the_blend_in_odds_get_prices(store) -> None:
    from tests.test_assistant import FakeLab
    tool = next(t for t in TOOLS if t["name"] == "markets_consensus")
    assert "strict" not in tool and tool["input_schema"]["required"] == ["league", "event_id"]
    assert len(TOOLS) == 14
    service = DashboardService(BoardOdds(two_games(), store), NoEspn(), NY, store=store)
    box = Toolbox(FakeLab(service, store), "tg:42", NY)
    run = box.run("markets_consensus", {"league": NBA, "event_id": "g1"})
    assert run.ok, run.content
    content = run.content
    assert content["game"] == "New York Knicks @ Boston Celtics" and content["home"] == "Boston Celtics"
    detail = service.markets_detail(NBA, "g1")
    assert content["blend"] == {o: round(p, 4) for o, p in detail["blend"]["p"].items()}
    assert [s["name"] for s in content["sources"]] == ["Sharp", "US books"]
    assert all({"chance", "status", "as_of"} <= set(s) for s in content["sources"])
    assert content["confidence"] == "normal" and content["disagree"] is False
    assert "market usually wins" in content["note"] and content["source"].startswith("The markets brain")
    assert "markets_consensus" not in PICK_TOOLS  # it prices nothing: on even while picks are paused
    missing = box.run("markets_consensus", {"league": NBA, "event_id": "nope"})
    assert not missing.ok and missing.content["error"] == "not_found"
    # odds_get_prices: the blend beside each game-winner outcome
    prices = box.run("odds_get_prices", {"league": NBA, "event_id": "g1"}).content
    h2h = {o["ref"]["side"]: o for o in prices["outcomes"] if o["ref"]["market"] == "h2h"}
    assert h2h["home"]["blend_chance"] == round(detail["blend"]["p"]["home"], 4)
    assert all("blend_chance" not in o for o in prices["outcomes"] if o["ref"]["market"] != "h2h")
    assert prices["markets_say"]["confidence"] == "normal"
    from telegram_betting_bot_simple.ai.assistant import SYSTEM_PROMPT
    assert "the blend is the app's best estimate; the market usually wins; you explain it, you don't override it" \
        in SYSTEM_PROMPT


def test_a_low_confidence_hint_names_market_prices_it_did_not_count(store, monkeypatch) -> None:
    from tests.test_assistant import FakeLab
    service = DashboardService(BoardOdds(two_games(), store), NoEspn(), NY, store=store)
    box = Toolbox(FakeLab(service, store), "tg:42", NY)
    low = {"blend": {"p": {"home": 0.6, "away": 0.4}, "confidence": "low", "disagree": False, "spread_pp": 0.0},
           "sources": [{"name": "US books", "kind": "us_books", "p": {"home": 0.65, "away": 0.35}, "status": "stale"},
                       {"name": "Torvik", "kind": "model_rating", "p": {"home": 0.6, "away": 0.4}, "status": "ok"}]}
    monkeypatch.setattr(service, "markets_detail", lambda league, event_id, *a, **k: low)
    hint = box.run("markets_consensus", {"league": NBA, "event_id": "g1"}).content["hint"]
    assert "US books" in hint and "thin or out of date" in hint and "no market prices" not in hint
    low["sources"] = low["sources"][1:]  # ratings only: there really is no market price
    hint = box.run("markets_consensus", {"league": NBA, "event_id": "g1"}).content["hint"]
    assert hint.startswith("Low confidence: no market prices this game")


def test_the_bot_s_game_card_says_what_the_markets_say() -> None:
    def source(name, p, status="ok"):
        return {"name": name, "kind": "x", "p": {"home": p, "away": 1 - p}, "status": status}
    card = {"league": NBA, "home": "Boston Celtics", "away": "New York Knicks", "live": False, "when": "Tonight",
            "markets": {}, "prediction": None,
            "consensus": {"blend": {"p": {"home": 0.63, "away": 0.37}, "confidence": "normal", "disagree": False,
                                    "spread_pp": 3.0},
                          "sources": [source("Sharp", 0.64), source("US books", 0.62), source("Kalshi", 0.63),
                                      source("Polymarket", 0.61), source("ESPN", 0.70, "stale")]}}
    # A neutral mark: ⚖️ means "the sources disagree" in the app and Help, so it is only on a disagreement
    assert markets_line(card) == "📊 Markets say: Celtics 63% (sharp 64 · US 62 · Kalshi 63 · Poly 61)"
    assert "📊 Markets say: Celtics 63%" in game_card_text(card, None) and "⚖" not in markets_line(card)
    card["consensus"]["blend"].update(disagree=True, spread_pp=7.5, p={"home": 0.4, "away": 0.6})
    card["consensus"]["sources"] = [source("<b>Elo</b>", 0.35, "thin")]
    line = markets_line(card)
    assert line.startswith("📊 Markets say: Knicks 60% (") and "&lt;b&gt;Elo&lt;/b&gt; 65 thin" in line
    assert line.endswith("· ⚖️ sources differ by 8 pts")  # half up, as the app's Math.round
    card["consensus"]["blend"]["confidence"] = "low"
    assert "low confidence" in markets_line(card)
    assert markets_line({**card, "consensus": None}) is None


def test_the_bot_rounds_every_chance_as_the_app_does() -> None:
    """Half a cent is common on Polymarket and Kalshi (a 1-cent bid/ask gap): 0.585 is 59% on the board and the game
    page (Math.round, whole_percent), so it is 59 in Telegram too, not round()'s 58."""
    from telegram_betting_bot_simple.data.slate import whole_percent
    card = {"league": NBA, "home": "Los Angeles Lakers", "away": "Portland Trail Blazers",
            "consensus": {"blend": {"p": {"home": 0.585, "away": 0.415}, "confidence": "normal", "disagree": False},
                          "sources": [{"name": "Polymarket", "p": {"home": 0.585, "away": 0.415}, "status": "ok"},
                                      {"name": "Kalshi", "p": {"home": 0.605, "away": 0.395}, "status": "ok"},
                                      {"name": "Sharp", "p": {"home": 0.625, "away": 0.375}, "status": "ok"}]}}
    assert markets_line(card) == "📊 Markets say: Lakers 59% (Poly 59 · Kalshi 61 · sharp 63)"
    assert [round(whole_percent(p) * 100) for p in (0.585, 0.605, 0.625)] == [59, 61, 63]  # the slate's figures


# --- no request path fetches a source ------------------------------------------------------------------

def test_only_the_slate_s_markets_worker_fetches_the_sources(store) -> None:
    builder, sources = fixture_slate(store)
    run_markets_round(builder)
    kalshi, poly, ratings = sources["kalshi"], sources["polymarket"], sources["ratings"]
    assert kalshi.fetches == [NBA] and poly.fetches == [NBA] and ("power", NBA) in ratings.fetches
    assert any(f[0] == "predictor" for f in ratings.fetches)
    assert set(kalshi.threads + poly.threads) == {"slate-markets"}
    counts = (len(kalshi.fetches), len(poly.fetches), len(ratings.fetches))
    odds = OddsService("k", store=store)
    service = DashboardService(odds, FinalEspn({}), NY, store=store)
    service.slate = builder
    client = app_client(service, builder)
    assert client.get("/api/slate").status_code == 200
    odds_id = f"{0:032x}"
    game = client.get("/api/game", params={"league": NBA, "id": odds_id})
    assert game.status_code == 200
    assert game.json()["consensus"]["blend"] == builder.markets_for(NBA, odds_id=odds_id)["blend"]
    assert client.get("/api/record/markets").status_code == 200
    from tests.test_assistant import FakeLab
    box = Toolbox(FakeLab(service, store), "tg:42", NY)
    assert box.run("markets_consensus", {"league": NBA, "event_id": odds_id}).ok
    service.markets_detail(NBA, "not-on-the-slate")
    builder.refresh()  # a slate rebuild between worker rounds reads cached copies only
    assert (len(kalshi.fetches), len(poly.fetches), len(ratings.fetches)) == counts


def test_a_restart_shows_a_started_game_s_logged_markets(store) -> None:
    clock = {"now": NOW}
    espn = FakeEspn({(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(1))]})
    store.save_odds_snapshot(NBA, [sharp_and_us(NBA, "o1", "Boston Celtics", "New York Knicks", at(1))], NOW - 60)
    kalshi = FakeVenue("kalshi", {NBA: [venue_event("kalshi", NBA, "Boston Celtics", "New York Knicks", at(1), 0.68)]})
    builder = builder_with(store, espn, now=lambda: clock["now"], kalshi=kalshi)
    run_markets_round(builder)
    logged = builder.markets_for(NBA, odds_id="o1")
    # After the start, a new process (no memory, no venue copies yet) shows the last logged "Markets say"
    clock["now"] = NOW + 2 * 3600
    restarted = builder_with(store, espn, now=lambda: clock["now"], kalshi=FakeVenue("kalshi", {}))
    restarted.refresh()
    assert restarted.markets_for(NBA, odds_id="o1")["blend"] == logged["blend"]
    assert [s["name"] for s in restarted.markets_for(NBA, espn_id="e1")["sources"]] == ["Sharp", "US books", "Kalshi"]


def test_three_way_games_and_games_without_a_usable_source() -> None:
    books = {"pinnacle": {"home": 2.9, "draw": 3.3, "away": 2.6}, "draftkings": {"home": 2.8, "draw": 3.2, "away": 2.5}}
    detail = market_detail("soccer_epl", book_sources(books, "2026-10-26T21:00:00Z", NOW), NOW + 7200, NOW)
    chance, mkt = chance_of(detail), compact(detail)
    assert set(chance) == {"home", "draw", "away", "src"} and chance["src"] == "blend"
    assert sum(chance[o] for o in ("home", "draw", "away")) == pytest.approx(1, abs=0.002)
    assert mkt["fav"] == "away" and set(mkt["p"]) == {"home", "draw", "away"}
    assert [entry[0] for entry in mkt["src"]] == ["Sharp", "US"]
    # A board saved over 12 hours ago is shown as stale, not blended: no blend, no chance, no block
    old = market_detail("soccer_epl", book_sources(books, "2026-10-26T06:00:00Z", NOW), NOW + 7200, NOW)
    assert {s["status"] for s in old["sources"]} == {"stale"} and old["blend"] is None
    assert chance_of(old) is None and compact(old) is None
    assert market_detail("soccer_epl", [], None, NOW) is None


# --- integration: the real sources and brain ------------------------------------------------------------

def test_the_real_sources_and_brain_end_to_end(store, monkeypatch) -> None:
    """The real Kalshi, Polymarket and Ratings (data/markets) over the trimmed live captures in
    tests/fixtures/markets, the real alias matching and engine/consensus.py, through the slate's markets worker,
    the slate, the game route and the bot's line."""
    from telegram_betting_bot_simple.data.markets import Kalshi, Polymarket, Ratings, common
    from telegram_betting_bot_simple.data.markets.common import parse_time
    from tests.test_markets_sources import Clock, Fetch, espn_routes, fixture, kalshi_routes, polymarket_routes
    from tests.test_markets_sources import FakeEspn as SummaryEspn

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(common.requests, "get", refuse)
    clock = Clock(parse_time("2026-10-27T01:15:00Z"))  # when the fixtures were captured
    start = "2026-10-27T23:30Z"  # Brooklyn at Boston, the next evening
    kalshi = Kalshi(store=store, fetch=Fetch(kalshi_routes({"KXNBAGAME": fixture("kalshi_nba_game.json")}), clock),
                    clock=clock, sleep=clock.sleep)
    poly = Polymarket(store=store, fetch=Fetch(polymarket_routes(fixture("polymarket_nba_moneyline.json"),
                                                                 fixture("polymarket_nba_lines.json")), clock),
                      clock=clock)
    ratings = Ratings(SummaryEspn({"401909877": fixture("espn_nba_summary_predictor.json")}), store=store,
                      fetch=Fetch(espn_routes(), clock), clock=clock)
    espn = FakeEspn({(NBA, datetime.date(2026, 10, 27)): [espn_game("401909877", "Boston Celtics", "Brooklyn Nets",
                                                                     start)]})
    store.save_odds_snapshot(NBA, [sharp_and_us(NBA, "bknbos", "Boston Celtics", "Brooklyn Nets", start,
                                                sharp=(1.22, 4.80), us=((1.20, 4.60), (1.21, 4.50)))], clock() - 60)
    builder = SlateBuilder(OddsService("k", store=store), espn, store, now=clock, workers=2, kalshi=kalshi,
                           polymarket=poly, ratings=ratings)
    run_markets_round(builder)
    detail = builder.markets_for(NBA, odds_id="bknbos")
    by_name = {s["name"]: s for s in detail["sources"]}
    assert set(by_name) == {"Sharp", "US books", "Kalshi", "Polymarket", "ESPN", "BPI"}
    assert by_name["Kalshi"]["p"]["home"] == pytest.approx(0.82 / (0.82 + 0.185), abs=0.005)  # bid/ask mids
    assert by_name["Polymarket"]["p"]["home"] == pytest.approx(0.815, abs=0.005)
    assert by_name["ESPN"]["p"]["home"] == pytest.approx(0.84)
    assert all(by_name[name]["url"].startswith("https://") for name in ("Kalshi", "Polymarket"))
    expected = consensus.blend(detail["sources"], NBA, now=clock(), start=parse_time(start))
    assert detail["blend"]["p"]["home"] == pytest.approx(expected["p"]["home"])
    assert detail["blend"]["confidence"] == "normal" and set(detail["blend"]["used"]) == set(by_name)
    [game] = games_by_id(builder).values()
    assert game["chance"]["src"] == "blend" and {e[0] for e in game["mkt"]["src"]} == {
        "Sharp", "US", "Kalshi", "Poly", "ESPN", "BPI"}
    [row] = store.consensus_rows()
    assert row["game_key"] == "espn:401909877" and row["event_id"] == "bknbos"
    # The game route and the bot read the same detail; the AI tool too
    service = DashboardService(OddsService("k", store=store), espn, NY, store=store)
    service.slate = builder
    assert service.markets_detail(NBA, "bknbos") is detail
    line = markets_line({"league": NBA, "home": "Boston Celtics", "away": "Brooklyn Nets", "consensus": detail})
    assert line.startswith("📊 Markets say: Celtics ") and "Kalshi 82" in line and "Poly 82" in line


def test_the_settings_reach_the_odds_service_the_book_roles_and_the_slate(tmp_path, monkeypatch) -> None:
    from telegram_betting_bot_simple.data.markets import Kalshi, Polymarket, Ratings
    from telegram_betting_bot_simple.services import build_service, build_slate
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:x")
    monkeypatch.setenv("THE_ODDS_API_KEY", "k" * 32)
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path))
    for name in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "MARKETS_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ODDS_BOOKMAKERS", "pinnacle,draftkings,fanduel,betmgm")
    monkeypatch.setenv("ODDS_PRICE_BOOKS", "fanduel")
    config = Config(_env_file=None)
    service = build_service(config)
    assert service.odds.bookmakers == ("pinnacle", "draftkings", "fanduel", "betmgm") and service.odds.board_cost == 3
    roles = book_roles()
    assert roles.price == frozenset({"fanduel"}) and roles.role("DraftKings") == "other" and roles.role("Pinnacle") == "sharp"
    slate = build_slate(config, service)
    assert service.slate is slate
    assert isinstance(slate.kalshi, Kalshi) and isinstance(slate.polymarket, Polymarket) and isinstance(slate.ratings, Ratings)
    monkeypatch.setenv("MARKETS_ENABLED", "0")
    off = build_slate(Config(_env_file=None), service)
    assert off.kalshi is None and off.polymarket is None and off.ratings is None
