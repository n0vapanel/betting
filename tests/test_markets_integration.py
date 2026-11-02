"""Stage 2 integration: the real markets sources (data/markets' Polymarket, Kalshi and Ratings over the trimmed
live captures in tests/fixtures/markets), the real alias matching, engine/consensus.py's blend, the slate's markets
worker, the game route, consensus logging and grading, GET /api/record/markets, the markets_consensus tool and the
bot's line, wired together as the app runs them. Only the HTTP getters, ESPN's scoreboards and the clock are fake;
no test touches the network (requests.get fails loudly).

The unit fakes stay in tests/test_markets_wiring.py; these tests catch what they can't: a venue that lists the
teams the other way round, a three-way market, ratings that only exist in the sources' own tables, Kalshi's
refresh window and the request paths' promise never to fetch.
"""

from __future__ import annotations

import copy
import datetime
import json
import math
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.ai.tools import Toolbox
from telegram_betting_bot_simple.bot.handlers import game_card_text, markets_line
from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.dashboard.service import DashboardService
from telegram_betting_bot_simple.data import odds_service
from telegram_betting_bot_simple.data.markets import Kalshi, Polymarket, Ratings, common
from telegram_betting_bot_simple.data.markets.common import parse_time
from telegram_betting_bot_simple.data.markets.kalshi import Pacer
from telegram_betting_bot_simple.data.odds_service import configure_books, normalize_event
from telegram_betting_bot_simple.data.slate import KALSHI_AHEAD, SlateBuilder, whole_percent
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.engine import consensus
from tests.test_markets_sources import Clock, Fetch, espn_routes, fixture, kalshi_routes
from tests.test_markets_sources import FakeEspn as SummaryEspn
from tests.test_slate import FakeEspn, signed_in

NBA, EPL, UFC, NCAAB, NCAAF = ("basketball_nba", "soccer_epl", "mma_mixed_martial_arts", "basketball_ncaab",
                               "americanfootball_ncaaf")
NY = ZoneInfo("America/New_York")
CAPTURED = parse_time("2026-10-27T01:15:00Z")  # when the fixtures were captured


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(common.requests, "get", refuse)
    monkeypatch.setattr(odds_service.requests, "get", refuse)


@pytest.fixture(autouse=True)
def default_roles():
    configure_books()
    yield
    configure_books()


@pytest.fixture()
def store(tmp_path):
    return Store(str(tmp_path / "betting.db"))


# --- the world -----------------------------------------------------------------------------------------------

def team(team_id: str, name: str, abbr: str, sport: str = "nba") -> Dict[str, Any]:
    words = name.split()
    return {"id": team_id, "displayName": name, "shortDisplayName": words[-1], "name": words[-1],
            "location": " ".join(words[:-1]), "abbreviation": abbr,
            "logo": f"https://a.espncdn.com/i/teamlogos/{sport}/500/scoreboard/{abbr.lower()}.png"}


def espn_game(event_id: str, home: Dict[str, Any], away: Dict[str, Any], start: str, neutral: bool = False,
              state: str = "pre", scores=(None, None)) -> Dict[str, Any]:
    def side(where, info, score):
        return {"homeAway": where, "score": "0" if score is None else str(score), "team": info,
                "records": [{"type": "total", "summary": "3-1"}]}
    return {"id": event_id, "date": start, "competitions": [{
        "id": event_id, "date": start, "neutralSite": neutral,
        "venue": {"fullName": "Arena", "address": {"city": "City"}, "indoor": True},
        "status": {"period": 0, "displayClock": "0:00", "type": {"name": "STATUS_FINAL" if state == "post" else "STATUS_SCHEDULED",
                                                               "state": state, "completed": state == "post",
                                                               "shortDetail": "7:30 PM"}},
        "competitors": [side("home", home, scores[0]), side("away", away, scores[1])]}]}


def fight_card(event_id: str, start: str, fights) -> Dict[str, Any]:
    return {"id": event_id, "date": start, "name": "UFC Fight Night", "competitions": [
        {"id": fid, "date": start, "status": {"type": {"name": "STATUS_SCHEDULED", "state": "pre"}},
         "competitors": [{"order": 1, "athlete": {"displayName": a, "shortName": a.split()[-1]}},
                         {"order": 2, "athlete": {"displayName": b, "shortName": b.split()[-1]}}]}
        for fid, a, b in fights]}


def board(league: str, event_id: str, home: str, away: str, start: str, books: Dict[str, tuple],
          spread: Optional[float] = None) -> Dict[str, Any]:
    """A saved odds event: {book key: (home, away[, draw])} moneylines (plus a spread for the NBA model)."""
    bookmakers = []
    for key, prices in books.items():
        h2h = [{"name": home, "price": prices[0]}, {"name": away, "price": prices[1]}]
        if len(prices) > 2:
            h2h.append({"name": "Draw", "price": prices[2]})
        markets = [{"key": "h2h", "outcomes": h2h}]
        if spread is not None:
            markets.append({"key": "spreads", "outcomes": [{"name": home, "price": 1.91, "point": spread},
                                                           {"name": away, "price": 1.91, "point": -spread}]})
        bookmakers.append({"key": key, "title": key.title(), "markets": markets})
    return normalize_event({"id": event_id, "sport_key": league, "sport_title": league, "commence_time": start,
                            "home_team": home, "away_team": away, "bookmakers": bookmakers})


class SavedOdds:
    """The odds service as the game route uses it, over the store's saved boards (never fetches)."""

    quota = {"remaining": 300, "used": 200}
    odds_ttl = 7200
    reserve = 50

    def __init__(self, store: Store) -> None:
        self.store = store

    def sports(self):
        return [{"key": league, "group": "x", "title": league.split("_")[-1].upper(), "active": True,
                 "has_outrights": False} for league in (NBA, EPL, UFC, NCAAB, NCAAF)]

    def league_games(self, league, refresh=False):
        saved = self.store.odds_snapshot(league)
        return copy.deepcopy(saved[1]) if saved else []

    def odds_loaded_at(self, league):
        return self.store.odds_fetched_at().get(league)

    def saved_games(self, league):
        return self.store.odds_snapshot(league)


class NoMatch:
    """The service's ESPN: no match (the NBA model works from the market alone), finals for grading."""

    def __init__(self, results: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.results = results or {}

    def find_game(self, *args):
        return None

    def standings(self, league, season=None):
        return {}

    def game_result(self, league, event_id):
        return self.results.get(str(event_id), {"state": "pending"})


def sources(store: Store, clock: Clock, summaries: Optional[Dict[str, Any]] = None):
    """The real Polymarket, Kalshi and Ratings over the captures: NBA, EPL and UFC venue markets; BPI, FPI,
    Torvik, Tennis Abstract and eloratings.net; ESPN's predictor for Nets @ Celtics."""
    gamma = {"745": ("polymarket_nba_moneyline.json", "polymarket_nba_lines.json"),
             "306": ("polymarket_epl_moneyline.json", "polymarket_epl_lines.json"),
             "279": ("polymarket_ufc_moneyline.json", None)}

    def poly_answer(params):
        if params.get("offset"):
            return 200, []
        names = gamma.get(str(params.get("tag_id")))
        if not names:
            return 200, []
        name = names[0] if params.get("sports_market_types") == "moneyline" else names[1]
        return 200, fixture(name) if name else []

    poly_fetch = Fetch([("gamma-api.polymarket.com/markets", poly_answer)], clock)
    kalshi_fetch = Fetch(kalshi_routes({
        "KXNBAGAME": fixture("kalshi_nba_game.json"), "KXNBASPREAD": fixture("kalshi_nba_spread.json"),
        "KXNBATOTAL": fixture("kalshi_nba_total.json"), "KXEPLGAME": fixture("kalshi_epl_game.json"),
        "KXEPLSPREAD": fixture("kalshi_epl_spread.json"), "KXEPLTOTAL": fixture("kalshi_epl_total.json"),
        "KXUFCFIGHT": fixture("kalshi_ufc_fight.json")}), clock)
    ratings_fetch = Fetch(espn_routes(), clock)
    polymarket = Polymarket(store=store, fetch=poly_fetch, clock=clock)
    kalshi = Kalshi(store=store, fetch=kalshi_fetch, clock=clock, sleep=clock.sleep, pacer=Pacer(1.0, clock, clock.sleep))
    no_predictor = {"header": {"competitions": [{"competitors": []}]}}  # as ESPN's summary for most games
    ratings = Ratings(SummaryEspn(summaries or {"401909877": fixture("espn_nba_summary_predictor.json"),
                                                "401909878": no_predictor}), store=store, fetch=ratings_fetch, clock=clock)
    return polymarket, kalshi, ratings, {"polymarket": poly_fetch, "kalshi": kalshi_fetch, "ratings": ratings_fetch}


BOS = team("2", "Boston Celtics", "BOS")
BKN = team("17", "Brooklyn Nets", "BKN")
NYK = team("18", "New York Knicks", "NY")
DET = team("8", "Detroit Pistons", "DET")
CELTICS_NETS = "2026-10-27T23:30:00Z"  # the captures' Nets @ Celtics, the next evening
KNICKS_PISTONS = "2026-10-28T00:00:00Z"


def nba_world(store: Store, clock: Clock, reversed_feed: bool = False):
    """Nets @ Celtics and Pistons @ Knicks (ESPN), both on a saved odds board with Pinnacle and Betfair (sharp) and
    two US books. With ``reversed_feed`` the odds feed lists the Celtics game the other way round (Nets "home")."""
    espn = FakeEspn({(NBA, datetime.date(2026, 10, 27)): [espn_game("401909877", BOS, BKN, CELTICS_NETS),
                                                          espn_game("401909878", NYK, DET, KNICKS_PISTONS)]})
    books = {"pinnacle": (1.22, 4.70), "betfair_ex_eu": (1.23, 4.80), "draftkings": (1.20, 4.60),
             "fanduel": (1.21, 4.50)}
    if reversed_feed:
        celtics = board(NBA, "bknbos", "Brooklyn Nets", "Boston Celtics", CELTICS_NETS,
                        {k: (v[1], v[0]) for k, v in books.items()}, spread=11.5)
    else:
        celtics = board(NBA, "bknbos", "Boston Celtics", "Brooklyn Nets", CELTICS_NETS, books, spread=-11.5)
    knicks = board(NBA, "detnyk", "New York Knicks", "Detroit Pistons", KNICKS_PISTONS,
                   {"pinnacle": (1.62, 2.42), "draftkings": (1.60, 2.40), "fanduel": (1.63, 2.35)}, spread=-3.5)
    store.save_odds_snapshot(NBA, [celtics, knicks], clock() - 60)
    return espn


def built(store: Store, clock: Clock, espn, **sources_kwargs) -> SlateBuilder:
    polymarket, kalshi, ratings, fetches = sources(store, clock, **sources_kwargs)
    builder = SlateBuilder(SavedOdds(store), espn, store, now=clock, workers=2, polymarket=polymarket, kalshi=kalshi,
                           ratings=ratings)
    builder.fetches = fetches  # type: ignore[attr-defined]
    builder.refresh()
    builder.wait_markets(20)
    builder.refresh()
    return builder


def slate_games(builder: SlateBuilder) -> Dict[str, Dict[str, Any]]:
    return {g["id"]: g for g in json.loads(builder.current().raw)["games"]}


def by_name(detail: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {s["name"]: s for s in detail["sources"]}


def calls(builder: SlateBuilder) -> int:
    return sum(len(fetch.calls) for fetch in builder.fetches.values())  # type: ignore[attr-defined]


# --- tests ---------------------------------------------------------------------------------------------------

def test_every_source_faces_the_game_s_way_once(store) -> None:
    """The odds feed lists Nets @ Celtics the other way round (Nets "home"): the slate's game follows the feed, so
    ESPN's predictor (Celtics at home) is turned round, Kalshi and Polymarket are turned by match_events/orient
    and BPI reads the sides by name. Every source must give the Nets (the feed's home) about 12-20%: a source
    flipped twice, or not at all, would give them 80%+."""
    clock = Clock(CAPTURED)
    builder = built(store, clock, nba_world(store, clock, reversed_feed=True))
    detail = builder.markets_for(NBA, odds_id="bknbos")
    names = by_name(detail)
    assert set(names) == {"Sharp", "US books", "Kalshi", "Polymarket", "ESPN", "BPI"}
    for name, source in names.items():
        assert 0.10 < source["p"]["home"] < 0.22, (name, source["p"])
    assert names["ESPN"]["p"]["home"] == pytest.approx(0.16)
    assert names["Kalshi"]["p"]["home"] == pytest.approx(0.185 / (0.185 + 0.82), abs=0.005)
    assert names["Polymarket"]["p"]["home"] == pytest.approx(0.185, abs=0.001)
    game = slate_games(builder)[f"{NBA}:bknbos"]
    assert game["home"]["abbr"] == "BKN" and game["mkt"]["fav"] == "away" and game["chance"]["home"] < 0.25


def test_the_whole_stack_on_the_captures(store) -> None:
    """Sources -> slate -> game route -> bot line -> AI tool, on the same numbers, and no request fetches."""
    from tests.test_assistant import FakeLab
    clock = Clock(CAPTURED)
    builder = built(store, clock, nba_world(store, clock))
    games = slate_games(builder)
    celtics, knicks = games[f"{NBA}:bknbos"], games[f"{NBA}:detnyk"]
    detail = builder.markets_for(NBA, odds_id="bknbos")
    names = by_name(detail)
    assert list(names) == ["Sharp", "US books", "Kalshi", "Polymarket", "ESPN", "BPI"]
    assert names["Sharp"]["detail"] == "no-vig: Pinnacle · Betfair" and names["ESPN"]["detail"] == "ESPN matchup predictor"
    # The blend is the brain's own over exactly these sources; BPI (88%) is 5+ points from the markets (~80%)
    expected = consensus.blend(detail["sources"], NBA, now=clock(), start=parse_time(CELTICS_NETS))
    assert detail["blend"]["p"] == pytest.approx(expected["p"])
    assert detail["blend"]["disagree"] and celtics["mkt"]["dis"] and celtics["mkt"]["spr"] >= 5
    assert celtics["chance"] == {"home": round(expected["p"]["home"], 3), "away": round(expected["p"]["away"], 3),
                                 "src": "blend"}
    assert [entry[0] for entry in celtics["mkt"]["src"]] == ["Sharp", "US", "Kalshi", "Poly", "ESPN", "BPI"]
    # Pistons @ Knicks: Polymarket only among the venues (Kalshi's capture doesn't list it)
    assert {entry[0] for entry in knicks["mkt"]["src"]} == {"Sharp", "US", "Poly", "BPI"}
    # The game route serves the slate's very detail, and its own prices never come from a sharp book
    service = DashboardService(SavedOdds(store), NoMatch(), NY, store=store)
    service.slate = builder
    client = signed_in(TestClient(create_app(service, "s3cret", login_delay=0, slate=builder)))
    before = calls(builder)
    game = client.get("/api/game", params={"league": NBA, "id": "bknbos"}).json()
    assert game["consensus"] == json.loads(json.dumps(detail))
    assert game["markets"]["h2h"]["away"]["book"] in ("Draftkings", "Fanduel")  # Betfair's 4.80 is never offered
    assert {b["key"]: b["role"] for b in game["books"]}["betfair_ex_eu"] == "sharp"
    projections = game["prediction"]["projections"]
    for projection in projections.values():
        assert projection["baseline"] + sum(f["delta"] for f in projection["factors"]) == pytest.approx(
            projection["value"], abs=0.01)
    # The bot's card reads the same block
    line = markets_line(game)
    assert line.startswith(f"📊 Markets say: Celtics {round(100 * whole_percent(detail['blend']['p']['home']))}%")
    assert "Kalshi 82" in line and "Poly 82" in line and "BPI 88" in line and "sources differ by" in line
    assert line in game_card_text(game, None)
    # The AI tool too
    box = Toolbox(FakeLab(service, store), "tg:42", NY)
    run = box.run("markets_consensus", {"league": NBA, "event_id": "bknbos"})
    assert run.ok and run.content["blend"] == {o: round(p, 4) for o, p in detail["blend"]["p"].items()}
    assert run.content["disagree"] is True and {s["name"] for s in run.content["sources"]} == set(names)
    prices = box.run("odds_get_prices", {"league": NBA, "event_id": "bknbos"}).content
    assert {o["ref"]["side"]: o.get("blend_chance") for o in prices["outcomes"] if o["ref"]["market"] == "h2h"} == {
        side: round(p, 4) for side, p in detail["blend"]["p"].items()}
    # None of these requests fetched a source
    assert client.get("/api/slate").status_code == 200
    assert client.get("/api/record/markets").status_code == 200
    assert calls(builder) == before


def test_soccer_three_way_from_both_venues_and_the_books(store) -> None:
    """Chelsea v Manchester United: Kalshi's three contracts and Polymarket's three Yes/No markets (each
    normalised), the books' 1X2, blended three-way with the draw."""
    clock = Clock(parse_time("2026-10-30T12:00:00Z"))  # inside Kalshi's two-day window for the Saturday games
    che = team("363", "Chelsea", "CHE", "soccer")
    mun = team("360", "Manchester United", "MUN", "soccer")
    espn = FakeEspn({(EPL, datetime.date(2026, 10, 31)): [espn_game("740100", che, mun, "2026-10-31T12:30Z")]})
    store.save_odds_snapshot(EPL, [board(EPL, "munche", "Chelsea", "Manchester United", "2026-10-31T12:30:00Z",
                                         {"pinnacle": (1.85, 4.20, 3.90), "draftkings": (1.83, 4.10, 3.80),
                                          "fanduel": (1.87, 4.00, 3.75)})], clock() - 60)
    builder = built(store, clock, espn)
    detail = builder.markets_for(EPL, odds_id="munche")
    names = by_name(detail)
    assert set(names) == {"Sharp", "US books", "Kalshi", "Polymarket"}
    for source in names.values():
        assert set(source["p"]) == {"home", "away", "draw"} and sum(source["p"].values()) == pytest.approx(1.0)
    assert names["Kalshi"]["p"]["draw"] == pytest.approx(0.245 / (0.535 + 0.235 + 0.245), abs=0.003)
    assert set(detail["blend"]["p"]) == {"home", "away", "draw"} and detail["blend"]["confidence"] == "normal"
    [game] = slate_games(builder).values()
    assert set(game["mkt"]["p"]) == {"home", "away", "draw"} and "draw" in game["chance"]
    assert names["Polymarket"]["url"] == "https://polymarket.com/event/epl-che-mun-2026-10-31"


def test_a_fighter_listed_first_on_one_venue_and_second_on_the_other(store) -> None:
    """Kropschot v Klein: Polymarket lists Kropschot first, Kalshi lists Klein first. Both must give Kropschot
    (the card's first fighter, "home") his own chance: 0.715 and 0.739."""
    clock = Clock(parse_time("2026-10-30T20:00:00Z"))
    espn = FakeEspn({(UFC, datetime.date(2026, 10, 31)): [fight_card("600060", "2026-10-31T21:00Z", [
        ("k1", "Joe Kropschot", "Nick Klein"), ("k2", "Zach Reese", "Mickey Gall")])]})
    builder = built(store, clock, espn)
    games = slate_games(builder)
    fight = next(g for g in games.values() if g.get("espn_id") == "k1" or g["id"].endswith(":k1"))
    detail = builder.markets_for(UFC, espn_id="k1")
    names = by_name(detail)
    assert names["Polymarket"]["p"]["home"] == pytest.approx(0.715, abs=0.001)
    assert names["Kalshi"]["p"]["home"] == pytest.approx(0.74 / (0.74 + 0.26), abs=0.01)
    assert fight["mkt"]["fav"] == "home"
    other = builder.markets_for(UFC, espn_id="k2")
    assert set(by_name(other)) == {"Kalshi"}  # Polymarket's capture doesn't price it


def test_ratings_alone_are_low_confidence_and_shrunk(store) -> None:
    """Kansas @ Duke (Torvik only) and a college football game whose odds board is a day old (stale books, FPI):
    no market is usable, so the ratings are blended alone, shrunk a quarter of the way to the base rate and marked
    low; the stale books are shown, not counted."""
    clock = Clock(CAPTURED)
    duke, kansas = team("150", "Duke Blue Devils", "DUKE", "ncb"), team("2305", "Kansas Jayhawks", "KU", "ncb")
    kenn = team("338", "Kennesaw State Owls", "KENN", "ncf")
    mtsu = team("2393", "Middle Tennessee Blue Raiders", "MTSU", "ncf")
    espn = FakeEspn({(NCAAB, datetime.date(2026, 10, 28)): [espn_game("401950100", duke, kansas, "2026-10-28T23:00Z")],
                     (NCAAF, datetime.date(2026, 10, 30)): [espn_game("401940100", kenn, mtsu, "2026-10-30T23:30Z")]})
    store.save_odds_snapshot(NCAAF, [board(NCAAF, "kenmtsu", "Kennesaw State Owls", "Middle Tennessee Blue Raiders",
                                           "2026-10-30T23:30:00Z", {"draftkings": (1.45, 2.80), "fanduel": (1.47, 2.70)})],
                             clock() - 20 * 3600)
    builder = built(store, clock, espn)
    duke_detail = builder.markets_for(NCAAB, espn_id="401950100")
    [torvik] = duke_detail["sources"]
    assert torvik["name"] == "Torvik" and duke_detail["blend"]["confidence"] == "low"
    # College basketball has no home base rate: shrunk toward an even split
    assert duke_detail["blend"]["p"]["home"] == pytest.approx(0.75 * torvik["p"]["home"] + 0.25 * 0.5)
    college = builder.markets_for(NCAAF, odds_id="kenmtsu")
    names = by_name(college)
    assert names["US books"]["status"] == "stale" and names["FPI"]["status"] == "ok"
    assert college["blend"]["used"] == ["FPI"] and college["blend"]["confidence"] == "low"
    game = slate_games(builder)[f"{NCAAF}:kenmtsu"]
    assert game["mkt"]["conf"] == "low" and ["US", whole_percent(names["US books"]["p"][game["mkt"]["fav"]]), "stale"] \
        in game["mkt"]["src"]


def test_kalshi_is_refreshed_only_for_games_within_two_days(store) -> None:
    """The NBA plays tomorrow, the EPL on Saturday (4+ days away): Kalshi is asked for the NBA's series only (its
    winner series: the spread and total series wait for a consumer), Polymarket for both."""
    clock = Clock(CAPTURED)
    espn = nba_world(store, clock)
    espn.boards[(EPL, datetime.date(2026, 10, 31))] = [espn_game("740100", team("363", "Chelsea", "CHE", "soccer"),
                                                                 team("360", "Manchester United", "MUN", "soccer"),
                                                                 "2026-10-31T12:30Z")]
    builder = built(store, clock, espn)
    series = {call[1].get("series_ticker") for call in builder.fetches["kalshi"].calls}  # type: ignore[attr-defined]
    assert series == {"KXNBAGAME"}
    tags = {str(call[1].get("tag_id")) for call in builder.fetches["polymarket"].calls}  # type: ignore[attr-defined]
    assert tags == {"745", "306"}
    epl = builder.markets_for(EPL, espn_id="740100")
    assert set(by_name(epl)) == {"Polymarket"}
    assert parse_time("2026-10-31T12:30:00Z") - clock() > KALSHI_AHEAD


def test_logged_before_the_start_graded_after_and_in_the_record(store) -> None:
    """The slate logs the real sources' opinions; after the final the real grader scores each against the result
    and GET /api/record/markets reports them next to the blend's."""
    clock = Clock(CAPTURED)
    builder = built(store, clock, nba_world(store, clock))
    rows = {row["event_id"]: row for row in store.consensus_rows()}
    assert set(rows) == {"bknbos", "detnyk"}
    logged = rows["bknbos"]
    assert logged["blend"] == json.loads(json.dumps(builder.markets_for(NBA, odds_id="bknbos")["blend"]))
    # Rows freeze at the start: moved on past it, nothing more is written
    store.log_consensus([{**logged, "blend": {**logged["blend"], "p": {"home": 0.5, "away": 0.5}}}],
                        now=parse_time(CELTICS_NETS) + 60)
    assert {row["event_id"]: row for row in store.consensus_rows()}["bknbos"]["blend"] == logged["blend"]
    # Celtics win; the Knicks game is still to be played
    espn = NoMatch({"401909877": {"state": "final", "home": 118, "away": 99}})
    service = DashboardService(SavedOdds(store), espn, NY, store=store)
    # The grader's cutoff as grade_due sets it once these games are over two hours old (the clock here is fake)
    graded = service._grade_consensus("2026-10-28T02:00:00Z", "2026-10-24T00:00:00Z")
    assert graded == 1
    rows = {row["event_id"]: row for row in store.consensus_rows()}
    result = rows["bknbos"]["result"]
    assert rows["bknbos"]["outcome"] == "home" and rows["detnyk"]["status"] == "pending"
    for source in logged["sources"]:
        p = source["p"]["home"]
        assert result["scores"][source["name"]]["brier"] == pytest.approx((1 - p) ** 2, abs=1e-6)
        assert result["scores"][source["name"]]["log_loss"] == pytest.approx(-math.log(p), abs=1e-6)
    client = signed_in(TestClient(create_app(service, "s3cret", login_delay=0)))
    record = client.get("/api/record/markets").json()
    [sport] = record["sports"]
    assert sport["league"] == NBA and sport["n"] == 1
    assert sport["blend"]["brier"] == pytest.approx(round((1 - logged["blend"]["p"]["home"]) ** 2, 4), abs=1e-4)
    assert {s["name"] for s in sport["sources"]} == {s["name"] for s in logged["sources"]}


def test_a_restart_serves_the_saved_copies_without_fetching(store) -> None:
    """The sources keep their last good copies in the store: a new process's slate shows the same "Markets say"
    from them before its own worker fetches anything."""
    clock = Clock(CAPTURED)
    espn = nba_world(store, clock)
    first = built(store, clock, espn)
    before = first.markets_for(NBA, odds_id="bknbos")
    polymarket, kalshi, ratings, fetches = sources(store, clock)
    later = SlateBuilder(SavedOdds(store), espn, store, now=clock, workers=2, polymarket=polymarket, kalshi=kalshi,
                         ratings=ratings, markets_every=3600)
    later._markets_next = clock() + 3600  # no worker round yet: only what the store holds
    later.refresh()
    after = later.markets_for(NBA, odds_id="bknbos")
    assert sum(len(f.calls) for f in fetches.values()) == 0
    assert {s["name"]: s["p"] for s in after["sources"] if s["name"] != "ESPN"} == {
        s["name"]: s["p"] for s in before["sources"] if s["name"] != "ESPN"}
