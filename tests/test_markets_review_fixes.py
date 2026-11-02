"""Stage 2 review fixes: nothing from after a game's start enters its "Markets say" or the consensus log; tennis
Elo by surface; matching names and start times the venues and ESPN write differently; Torvik's look-alike team
names; ESPN's predictor for the whole week; the markets worker when Kalshi rate-limits; the markets record past
5,000 games; name lookups remembered per rating table. Fakes, captured fixtures and a real Store: no network."""

from __future__ import annotations

import copy
import datetime
import json
import os
from contextlib import closing
from typing import Any, Dict, List

import pytest

from telegram_betting_bot_simple.dashboard.service import DashboardService, consensus_score
from telegram_betting_bot_simple.data import slate as slate_mod
from telegram_betting_bot_simple.data.markets import aliases, ratings as ratings_mod
from telegram_betting_bot_simple.data.markets.aliases import match_events, side_score, team_key
from telegram_betting_bot_simple.data.markets.common import parse_time
from telegram_betting_bot_simple.data.markets.kalshi import Kalshi, Pacer
from telegram_betting_bot_simple.data.markets.ratings import Ratings, surface_of
from telegram_betting_bot_simple.data.slate import SlateBuilder, market_detail, parse_scoreboard
from telegram_betting_bot_simple.data.store import Store
from tests.test_markets_wiring import (NY, BoardOdds, FakeVenue, NoEspn, builder_with, run_markets_round,
                                       sharp_and_us, venue_event, venue_kalshi_source)
from tests.test_slate import NBA, NOW, TODAY, FakeEspn, at, espn_game

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "markets")
ATP = "tennis_atp"
NCAAF = "americanfootball_ncaaf"


@pytest.fixture()
def store(tmp_path):
    return Store(str(tmp_path / "betting.db"))


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- the freeze rule: nothing from the start on --------------------------------------------------------------

def test_a_slow_scoreboard_round_can_t_log_a_price_from_after_the_start(store) -> None:
    """The boards take seconds; meanwhile the game starts and the markets worker reads an in-play Kalshi price. The
    slate reads the clock again after the boards, so the game counts as started: its last view before the start
    stays, and the log isn't touched."""
    start = parse_time(at(1).replace("Z", ":00Z"))
    clock = {"now": start - 20}
    kalshi = FakeVenue("kalshi", {NBA: [venue_event("kalshi", NBA, "Boston Celtics", "New York Knicks", at(1), 0.60)]})

    class SlowEspn(FakeEspn):
        armed = False

        def fetch_scoreboard(self, league, day):
            if self.armed:
                clock["now"] = start + 40
                kalshi.cached[NBA] = [venue_event("kalshi", NBA, "Boston Celtics", "New York Knicks", at(1), 0.20,
                                                  as_of=iso(start + 30))]
            return super().fetch_scoreboard(league, day)

    espn = SlowEspn({(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(1))]})
    store.save_odds_snapshot(NBA, [sharp_and_us(NBA, "o1", "Boston Celtics", "New York Knicks", at(1))], NOW - 60)
    builder = builder_with(store, espn, now=lambda: clock["now"], kalshi=kalshi, live_seconds=1, today_seconds=1)
    run_markets_round(builder)
    [before] = store.consensus_rows()
    espn.armed = True
    builder._boards.clear()  # every board due again
    builder.refresh()
    [row] = store.consensus_rows()
    assert row == before  # the same sources, blend and logged_at
    assert {s["name"]: s["p"]["home"] for s in row["sources"]}["Kalshi"] == 0.60
    shown = builder.markets_for(NBA, odds_id="o1")
    assert all(parse_time(s["as_of"]) < start for s in shown["sources"] if s.get("as_of"))


def test_a_price_read_after_the_start_marks_the_game_started_even_when_the_clock_lags(store) -> None:
    """The clock read at the top of a build says 20 s to go, but the worker has already cached a Kalshi price from
    after the start (it refreshed while the slate was building): that price is the proof the game has started, so
    the build keeps the last view before it and logs nothing new."""
    start = parse_time(at(1).replace("Z", ":00Z"))
    kalshi = FakeVenue("kalshi", {NBA: [venue_event("kalshi", NBA, "Boston Celtics", "New York Knicks", at(1), 0.60)]})
    espn = FakeEspn({(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(1))]})
    store.save_odds_snapshot(NBA, [sharp_and_us(NBA, "o1", "Boston Celtics", "New York Knicks", at(1))], NOW - 60)
    builder = builder_with(store, espn, now=lambda: start - 20, kalshi=kalshi)
    run_markets_round(builder)
    [before] = store.consensus_rows()
    kept = builder.markets_for(NBA, odds_id="o1")
    kalshi.cached[NBA] = [venue_event("kalshi", NBA, "Boston Celtics", "New York Knicks", at(1), 0.20,
                                      as_of=iso(start + 30))]
    builder.refresh()
    assert store.consensus_rows() == [before]
    assert builder.markets_for(NBA, odds_id="o1") is kept


def test_market_detail_leaves_out_every_opinion_from_the_start_on() -> None:
    start = parse_time("2026-10-26T23:00:00Z")
    early = venue_kalshi_source(0.60, "2026-10-26T22:30:00Z")
    late = venue_kalshi_source(0.20, "2026-10-26T23:00:00Z")  # at the start counts as after it
    detail = market_detail(NBA, [early, late], start, start + 600)
    assert [s["p"]["home"] for s in detail["sources"]] == [0.60] and detail["blend"]["as_of"] == "2026-10-26T22:30:00Z"
    assert market_detail(NBA, [late], start, start + 600) is None
    assert len(market_detail(NBA, [early, late], None, start)["sources"]) == 2  # no start known: nothing to compare


def test_a_started_game_with_no_earlier_view_shows_none_made_from_in_play_prices(store) -> None:
    """The league's first board came in play: no "Markets say" on the slate, the game route or the bot (the page
    would call it "as it stood at the start"). A board from before the start still makes one."""
    started = at(-1)
    espn = FakeEspn({(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", started, state="in",
                                              name="STATUS_IN_PROGRESS", period=2)]})
    in_play = [sharp_and_us(NBA, "o1", "Boston Celtics", "New York Knicks", started, sharp=(1.20, 5.0),
                            us=((1.18, 4.8), (1.19, 4.6)))]
    store.save_odds_snapshot(NBA, in_play, NOW - 30)
    builder = builder_with(store, espn)
    builder.refresh()
    assert builder.markets_for(NBA, odds_id="o1") is None
    [game] = json.loads(builder.current().raw)["games"]
    assert "mkt" not in game and store.consensus_rows() == []
    # The service's own fallback (the bot's process, no slate) says the same
    service = DashboardService(BoardOdds(in_play, store, fetched_at=NOW - 30), NoEspn(), NY, store=store)
    assert service.markets_detail(NBA, "o1", now=NOW) is None
    # A board saved before the start is the view as it stood
    before = DashboardService(BoardOdds(in_play, store, fetched_at=NOW - 2 * 3600), NoEspn(), NY, store=store)
    detail = before.markets_detail(NBA, "o1", now=NOW)
    assert detail and parse_time(detail["blend"]["as_of"]) < parse_time(started.replace("Z", ":00Z"))


# --- tennis: surface Elo, and ESPN's placeholder start times ----------------------------------------------------

def tennis_board(name: str, matches: List[tuple]) -> Dict[str, Any]:
    """An ESPN tennis scoreboard event: a tournament whose men's singles are the games."""
    def player(where, order, full):
        first, last = full.split(" ", 1)
        return {"id": f"{where}-{last}", "type": "athlete", "order": order, "homeAway": where,
                "athlete": {"displayName": full, "shortName": f"{first[0]}. {last}", "fullName": full}}
    return {"id": "t-" + name.split()[0].lower(), "name": name, "date": at(-48), "groupings": [
        {"grouping": {"slug": "mens-singles"}, "competitions": [
            {"id": match_id, "date": date, "timeValid": valid,
             "status": {"period": 0, "type": {"name": "STATUS_SCHEDULED", "state": "pre", "completed": False,
                                               "shortDetail": "TBD"}},
             "competitors": [player("home", 1, home), player("away", 2, away)]}
            for match_id, home, away, date, valid in matches]}]}


def tennis_ratings() -> Ratings:
    ratings = Ratings(None)
    page = open(os.path.join(FIXTURES, "tennis_abstract_atp_elo.html")).read()
    ratings.keeper.seed(("power", ATP), NOW, ratings_mod.parse_tennis_abstract(page, ATP))
    return ratings


def test_a_clay_tournament_s_elo_is_half_overall_half_clay(store) -> None:
    """ESPN names the tournament, not its surface: the name says clay (Madrid), so the Elo chip blends overall and
    clay Elo; an indoor hard event (Vienna) uses hard Elo."""
    espn = FakeEspn({(ATP, TODAY): [tennis_board("Mutua Madrid Open", [("m1", "Jannik Sinner", "Carlos Alcaraz",
                                                                        at(2), True)]),
                                    tennis_board("Erste Bank Open", [("v1", "Alexander Zverev", "Novak Djokovic",
                                                                      at(3), True)])]})
    builder = builder_with(store, espn, ratings=tennis_ratings())
    builder.refresh()
    madrid = builder.markets_for(ATP, espn_id="m1")
    [elo] = madrid["sources"]
    assert elo["name"] == "Elo" and elo["detail"].startswith("50/50 overall and clay Elo")
    expected = ratings_mod.elo_p((0.5 * 2296.9 + 0.5 * 2186.8) - (0.5 * 2149.8 + 0.5 * 2088.1))
    assert elo["p"]["home"] == pytest.approx(expected)
    [vienna] = builder.markets_for(ATP, espn_id="v1")["sources"]
    assert vienna["detail"].startswith("50/50 overall and hard Elo")
    # The table, by name or by the odds feed's tournament key
    assert surface_of(ATP, "Roland Garros") == surface_of("tennis_atp_french_open") == "clay"
    assert surface_of("tennis_atp_wimbledon") == surface_of(ATP, "Terra Wortmann Open") == "grass"
    assert surface_of("tennis_wta", "Porsche Tennis Grand Prix Stuttgart") == "clay"
    assert surface_of(ATP, "Boss Open Stuttgart") == "grass" and surface_of(ATP, "Rolex Paris Masters") == "hard"
    assert surface_of(NBA, "Madrid") is None


def test_a_match_without_a_start_time_yet_still_meets_its_markets(store) -> None:
    """ESPN gives tomorrow's matches 04:00Z (midnight Eastern) with timeValid false until the order of play is out;
    Kalshi and Polymarket have the real time. The venues are matched on the date, and the match isn't taken as
    started at 04:00Z."""
    tomorrow = (TODAY + datetime.timedelta(days=1)).isoformat()
    placeholder = f"{tomorrow}T04:00Z"
    board = [tennis_board("Rolex Paris Masters", [("p1", "Novak Djokovic", "Ugo Humbert", placeholder, False)])]
    espn = FakeEspn({(ATP, TODAY): board, (ATP, TODAY + datetime.timedelta(days=1)): board})
    real = f"{tomorrow}T11:00:00Z"
    kalshi = FakeVenue("kalshi", {ATP: [venue_event("kalshi", ATP, "Novak Djokovic", "Ugo Humbert", real, 0.71)]})
    poly = FakeVenue("polymarket", {ATP: [venue_event("polymarket", ATP, "Novak Djokovic", "Ugo Humbert", real, 0.70)]})
    clock = {"now": NOW}
    builder = builder_with(store, espn, now=lambda: clock["now"], kalshi=kalshi, polymarket=poly)
    run_markets_round(builder)
    detail = builder.markets_for(ATP, espn_id="p1")
    assert {s["name"] for s in detail["sources"]} == {"Kalshi", "Polymarket"}
    assert detail["blend"]["confidence"] == "normal"
    # The morning of the match (after 04:00Z, before it is played): still a game to come, still priced
    clock["now"] = parse_time(f"{tomorrow}T09:00:00Z")
    for venue in (kalshi, poly):
        venue.data[ATP][0]["as_of"] = iso(clock["now"] - 60)
    builder._boards.clear()
    run_markets_round(builder)
    later = builder.markets_for(ATP, espn_id="p1")
    assert later is not detail and {s["name"] for s in later["sources"]} == {"Kalshi", "Polymarket"}
    [game] = json.loads(builder.current().raw)["games"]
    assert game["status"] == "pre" and game["mkt"]


def test_one_on_one_sports_match_within_twelve_hours_teams_within_three() -> None:
    """Polymarket gives every fight the card's start (21:00Z); ESPN the main card's (01:00Z, four hours on)."""
    fight = {"id": "f1", "home": {"name": "Alex Pereira"}, "away": {"name": "Magomed Ankalaev"},
             "start": "2026-11-01T01:00:00Z"}
    event = {"id": "ufc-1", "source": "polymarket", "home": "Alex Pereira", "away": "Magomed Ankalaev",
             "start": "2026-10-31T21:00:00Z", "markets": {}}
    assert set(match_events("mma_mixed_martial_arts", [fight], [event])) == {"f1"}
    assert match_events("mma_mixed_martial_arts", [fight], [{**event, "start": "2026-10-31T12:00:00Z"}]) == {}
    game = {"id": "g1", "home": {"name": "Boston Celtics"}, "away": {"name": "New York Knicks"},
            "start": "2026-11-01T01:00:00Z"}
    celtics = {**event, "home": "Boston Celtics", "away": "New York Knicks"}
    assert match_events(NBA, [game], [celtics]) == {}  # four hours apart: another game
    assert set(match_events(NBA, [game], [celtics], tol_hours=5)) == {"g1"}


# --- names ------------------------------------------------------------------------------------------------------

def test_letters_without_an_accent_to_strip_are_spelled_out() -> None:
    """NFKD leaves ø, æ, ł, ı and ß whole, and an ASCII fold dropped them: "Bodø/Glimt" keyed as "bod glimt"."""
    league = "soccer_uefa_champs_league"
    assert team_key(league, "FK Bodø/Glimt") == team_key(league, "Bodo/Glimt") == "bodo glimt"
    assert side_score(league, ["FK Bodø/Glimt"], ["Bodo/Glimt"]) == 1.0
    assert team_key(league, "Brøndby IF") == "brondby if" and team_key("soccer_turkey_super_league", "Kasımpaşa") == "kasimpasa"
    assert team_key(league, "Widzew Łódź") == "widzew lodz" and team_key(league, "FC København") == "copenhagen"
    assert team_key(league, "FC Copenhagen") == "copenhagen"
    assert side_score("soccer_france_ligue_one", ["Stade Brestois 29"], ["Brest", "Stade Brestois"]) == 1.0
    assert team_key(NCAAF, "Louisiana-Monroe") == team_key(NCAAF, "UL Monroe") == team_key(NCAAF, "ULM")
    assert side_score(NCAAF, ["Louisiana-Monroe"], ["UL Monroe Warhawks", "ULM"]) == 1.0
    assert aliases._plain("Æbeltoft Œuvre ß") == "aebeltoft oeuvre ss"


def torvik_row(name: str, adjoe: float, adjde: float) -> List[Any]:
    row: List[Any] = [0] * 45
    row[1], row[4], row[6], row[44] = name, adjoe, adjde, 68.0
    return row


def test_a_school_inside_another_school_s_name_is_told_apart() -> None:
    """"Northern Illinois Huskies" contains both "Illinois" and "Northern Illinois": the one sharing more words is
    the team (it was a tie, so no Torvik rating for about 35 Division I teams)."""
    table = ratings_mod.parse_torvik([torvik_row("Illinois", 120.0, 92.0), torvik_row("Northern Illinois", 101.0, 108.0),
                                      torvik_row("Southern Illinois", 104.0, 104.0), torvik_row("Kentucky", 118.0, 95.0),
                                      torvik_row("Western Kentucky", 105.0, 106.0)], "2026-10-27T00:00:00Z")
    ratings = Ratings(None)
    ratings.keeper.seed(("power", "basketball_ncaab"), NOW, table)
    league = "basketball_ncaab"

    def resolve(name, short, abbr):
        return ratings._resolve(league, table, {"name": name, "short": short, "abbr": abbr})

    assert resolve("Northern Illinois Huskies", "N Illinois", "NIU") == "northern illinois"
    assert resolve("Southern Illinois Salukis", "S Illinois", "SIU") == "southern illinois"
    assert resolve("Illinois Fighting Illini", "Illinois", "ILL") == "illinois"
    assert resolve("Western Kentucky Hilltoppers", "W Kentucky", "WKU") == "western kentucky"
    prob = ratings.game_prob(league, {"name": "Northern Illinois Huskies", "short": "N Illinois", "abbr": "NIU"},
                             {"name": "Western Kentucky Hilltoppers", "short": "W Kentucky", "abbr": "WKU"}, fetch=False)
    assert prob and prob["source"] == "Torvik"


def test_name_lookups_are_remembered_per_table(monkeypatch) -> None:
    """The slate asks for every game's rating on every rebuild: each spelling is looked up once per table (misses
    too), and a new table starts over."""
    ratings = tennis_ratings()
    calls = []
    real = ratings_mod._resolve

    def counting(league, table, side):
        calls.append(side)
        return real(league, table, side)

    monkeypatch.setattr(ratings_mod, "_resolve", counting)
    for _ in range(3):
        assert ratings.game_prob(ATP, "Jannik Sinner", "Carlos Alcaraz", fetch=False)
        assert ratings.game_prob(ATP, "Jannik Sinner", "Nobody Known", fetch=False) is None
    assert len(calls) == 3  # Sinner, Alcaraz, the unknown name: once each
    table = dict(ratings.cached_power(ATP))
    ratings.keeper.put(("power", ATP), table)  # a refreshed table
    assert ratings.game_prob(ATP, "Jannik Sinner", "Carlos Alcaraz", fetch=False)
    assert len(calls) == 5


# --- the markets worker -----------------------------------------------------------------------------------------

class Clock:
    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class CountingVenue:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.calls: List[tuple] = []

    def covers(self, league: str) -> bool:
        return True

    def events(self, league: str) -> List[Dict[str, Any]]:
        self.calls.append((league, self.clock()))
        return []

    def cached_events(self, league: str) -> List[Dict[str, Any]]:
        return []


def test_a_rate_limited_kalshi_holds_up_nothing_else() -> None:
    """Kalshi answers "too many requests" to everything: Polymarket is refreshed for every league before Kalshi is
    asked, and once one request runs out of retries Kalshi rests, so the round makes 5 Kalshi requests, not 5 a
    league, and lasts one back-off (about 30 s), not seven minutes."""
    clock = Clock()
    asked: List[Any] = []

    def limited(url, params=None, text=False):
        asked.append(params.get("series_ticker"))
        return 429, {"error": {"code": "too_many_requests"}}

    kalshi = Kalshi(fetch=limited, clock=clock, sleep=clock.sleep, pacer=Pacer(1.0, clock, clock.sleep))
    poly = CountingVenue(clock)
    builder = SlateBuilder(odds=None, espn=None, store=None, polymarket=poly, kalshi=kalshi, ratings=None, now=clock)
    leagues = sorted(["americanfootball_ncaaf", "americanfootball_nfl", "basketball_nba", "icehockey_nhl", "soccer_epl",
                      "soccer_spain_la_liga", "soccer_usa_mls", "tennis_atp", "tennis_wta", "mma_mixed_martial_arts"])
    start = clock()
    builder._load_markets(leagues, [], leagues)
    assert [league for league, _ in poly.calls] == leagues and all(when == start for _, when in poly.calls)
    assert len(asked) == 5 and kalshi.resting()
    assert clock() - start < 40
    # The next rounds skip Kalshi while it rests, without a request or a wait
    builder._load_markets(leagues, [], leagues)
    assert len(asked) == 5 and clock() - start < 40


class PredictorRatings:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.asked: Dict[str, List[float]] = {}

    def power(self, league: str) -> Dict[str, Any]:
        return {}

    def espn_predictor(self, league: str, espn_id: str) -> Dict[str, Any]:
        self.asked.setdefault(espn_id, []).append(self.clock())
        return {"home": 0.6, "away": 0.4, "as_of": iso(self.clock())}


def test_espn_s_predictor_reaches_every_game_of_the_week() -> None:
    """A college football Saturday: 40 games today (20 at noon, 20 in the afternoon) and 20 more next week. Each
    round's 25 fetches go to games that are due, so every game gets ESPN's predictor within two rounds; games within
    a day are asked again hourly, the rest every 6 hours."""
    clock = Clock()
    ratings = PredictorRatings(clock)
    builder = SlateBuilder(odds=None, espn=None, store=None, ratings=ratings, now=clock, markets_every=60)
    games = [{"league": NCAAF, "status": "pre", "espn_id": str(1000 + i),
              "_ts": NOW + (2 if i < 20 else 6 if i < 40 else 4 * 24) * 3600 + i} for i in range(60)]
    for _round in range(3):
        builder._schedule_markets(games, clock())
        builder.wait_markets(5)
        clock.now += 61
    assert len(ratings.asked) == 60 and all(len(times) == 1 for times in ratings.asked.values())
    assert all(builder._predictors.get((NCAAF, g["espn_id"])) for g in games)
    clock.now = NOW + 3600 + 30  # an hour on: today's games again, next week's not yet
    for _round in range(2):
        builder._schedule_markets(games, clock())
        builder.wait_markets(5)
        clock.now += 61
    assert sum(len(ratings.asked[g["espn_id"]]) == 2 for g in games[:40]) == 40
    assert all(len(ratings.asked[g["espn_id"]]) == 1 for g in games[40:])
    clock.now = NOW + 6 * 3600 + 200  # six hours after next week's games were asked (today's have started)
    builder._schedule_markets(games, clock())
    builder.wait_markets(5)
    assert all(len(ratings.asked[g["espn_id"]]) == 2 for g in games[40:])


def test_a_game_more_than_a_day_away_gets_espn_in_its_blend(store) -> None:
    """ESPN's predictor exists all week (the game page's "Who wins" shows it); the blend has it too, so the board,
    the game page and the log agree."""
    two_days = at(48)
    espn = FakeEspn({(NBA, TODAY + datetime.timedelta(days=2)): [espn_game("e9", "Boston Celtics", "New York Knicks",
                                                                           two_days)]})

    class Predictor(PredictorRatings):
        def game_prob(self, *args, **kwargs):
            return None

    ratings = Predictor(Clock())
    builder = builder_with(store, espn, ratings=ratings)
    run_markets_round(builder)
    detail = builder.markets_for(NBA, espn_id="e9")
    assert [s["name"] for s in detail["sources"]] == ["ESPN"] and "e9" in ratings.asked


# --- the markets record ---------------------------------------------------------------------------------------

def test_the_markets_record_counts_every_graded_game(store) -> None:
    """More than 5,000 graded games: every one counts (consensus_rows' default limit used to cut the record to the
    latest 5,000), and the record is worked out again only when something new is graded."""
    detail = market_detail(NBA, [venue_kalshi_source(0.6, "2026-10-26T20:00:00Z")], None, NOW)
    rows = [{"league": NBA, "game_key": f"espn:{n}", "event_id": f"o{n}", "espn_id": str(n), "home": "A", "away": "B",
             "start": iso(NOW + 3600 + n), "sources": detail["sources"], "blend": detail["blend"]} for n in range(5003)]
    store.log_consensus(rows, now=NOW)
    result = {"score": {"home": 1, "away": 0}, "outcome": "home", "blend": consensus_score(detail["blend"]["p"], "home"),
              "scores": {"Kalshi": {**consensus_score({"home": 0.6, "away": 0.4}, "home"),
                                    "kind": "prediction_market", "status": "ok"}}}
    # Graded the way grade_consensus grades one row, all at once (5,003 commits would take seconds)
    with closing(store._connect()) as db:
        db.execute("UPDATE consensus_log SET status = 'graded', outcome = 'home', result = ?, graded_at = ? "
                   "WHERE status = 'pending'", (json.dumps(result), NOW + 7200))
    service = DashboardService(BoardOdds([], store), NoEspn(), NY, store=store)
    record = service.markets_record()
    [nba] = record["sports"]
    assert record["games"] == nba["n"] == 5003 and nba["sources"][0]["n"] == 5003
    assert nba["sources"][0]["brier"] == pytest.approx(0.16)
    assert service.markets_record() is record  # nothing new graded: the same record
    extra = {**rows[0], "game_key": "espn:new", "event_id": "onew", "espn_id": "new", "start": iso(NOW + 9000)}
    store.log_consensus([extra], now=NOW)
    store.grade_consensus(NBA, "espn:new", "graded", "home", copy.deepcopy(result), now=NOW + 7300)
    assert service.markets_record()["games"] == 5004


# --- the slate keeps team names ----------------------------------------------------------------------------------

def test_the_slate_keeps_each_team_s_full_name(store) -> None:
    """Today's search and slip labels need the full name ("Brooklyn", "Golden State" are in it, not in "Nets")."""
    espn = FakeEspn({(NBA, TODAY): [espn_game("e1", "Golden State Warriors", "Brooklyn Nets", at(2))]})
    builder = builder_with(store, espn)
    builder.refresh()
    [game] = json.loads(builder.current().raw)["games"]
    assert game["home"]["name"] == "Golden State Warriors" and game["away"]["name"] == "Brooklyn Nets"
    assert game["home"]["short"] == "Warriors" and game["home"]["abbr"]


def test_parse_scoreboard_keeps_espn_s_time_flag_and_the_tournament() -> None:
    [match] = parse_scoreboard(ATP, [tennis_board("Mutua Madrid Open", [("m1", "Jannik Sinner", "Carlos Alcaraz",
                                                                         "2026-10-28T04:00Z", False)])])
    assert match["_time_ok"] is False and match["_event"] == "Mutua Madrid Open"
    [game] = parse_scoreboard(NBA, [espn_game("e1", "Boston Celtics", "New York Knicks", at(2))])
    assert game["_time_ok"] is True and game.get("_event") is None
    assert slate_mod.tennis_surface("tennis_atp_madrid_open") == "clay" and slate_mod.tennis_surface(NBA) is None
