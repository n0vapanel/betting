"""Tests for Today's all-sports slate (data/slate.py), GET /api/slate, POST /api/slate/odds and the hashed static
files: fake ESPN scoreboards, a fake odds service and a real Store in a temporary folder (no network)."""

from __future__ import annotations

import copy
import datetime
import gzip
import json
import logging
import re
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import requests
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.dashboard.assets import IMMUTABLE, StaticAssets
from telegram_betting_bot_simple.data.espn_client import ESPN_DAY_TZ, LEAGUES, SCOREBOARD_ONLY, espn_path
from telegram_betting_bot_simple.data import odds_service
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.odds_service import OddsService, normalize_event
from telegram_betting_bot_simple.data.slate import SEASON_EVERY, SEASON_RETRY, SLATE_NAME, SlateBuilder, ids_of
from telegram_betting_bot_simple.data.store import Store

NBA, NHL, EPL, ODI = "basketball_nba", "icehockey_nhl", "soccer_epl", "cricket_odi"
PLAYER = {"X-Player": "browser-1234"}
# 6 pm in New York on Monday 26 October 2026: the ESPN day is the 26th
NOW = datetime.datetime(2026, 10, 26, 18, 0, tzinfo=ESPN_DAY_TZ).timestamp()
TODAY = datetime.date(2026, 10, 26)


def at(hours: float) -> str:
    """An ISO time ``hours`` from NOW (ESPN writes "2026-10-26T23:00Z")."""
    moment = datetime.datetime.fromtimestamp(NOW + hours * 3600, datetime.timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%MZ")


def team(team_id: str, name: str, sport: str = "nba") -> Dict[str, Any]:
    words = name.split()
    return {"id": team_id, "displayName": name, "shortDisplayName": words[-1], "name": words[-1],
            "location": " ".join(words[:-1]), "abbreviation": "".join(w[0] for w in words).upper()[:3],
            "logo": f"https://a.espncdn.com/i/teamlogos/{sport}/500/scoreboard/{team_id}.png"}


def espn_game(event_id: str, home: str, away: str, start: str, state: str = "pre", name: str = "STATUS_SCHEDULED",
              scores=(None, None), period: int = 0, clock: str = "0.0", ids=("1", "2"), sport: str = "nba",
              neutral: bool = False, detail: str = "7:00 PM") -> Dict[str, Any]:
    def side(where, label, score, team_id):
        return {"homeAway": where, "score": "0" if score is None else str(score), "team": team(team_id, label, sport),
                "records": [{"name": "overall", "type": "total", "summary": "3-1"}]}
    return {"id": event_id, "date": start, "competitions": [{
        "id": event_id, "date": start, "neutralSite": neutral,
        "venue": {"fullName": f"{home.split()[-1]} Arena", "address": {"city": "Boston"}, "indoor": True},
        "status": {"period": period, "displayClock": clock,
                   "type": {"name": name, "state": state, "completed": state == "post", "shortDetail": detail}},
        "competitors": [side("home", home, scores[0], ids[0]), side("away", away, scores[1], ids[1])],
    }]}


def odds_event(league: str, event_id: str, home: str, away: str, start: str, home_ml=1.8, away_ml=2.1,
               draw: Optional[float] = None, spread: Optional[float] = -2.5) -> Dict[str, Any]:
    h2h = [{"name": home, "price": home_ml}, {"name": away, "price": away_ml}]
    if draw:
        h2h.append({"name": "Draw", "price": draw})
    markets = [{"key": "h2h", "outcomes": h2h},
               {"key": "totals", "outcomes": [{"name": "Over", "price": 1.91, "point": 220.5},
                                              {"name": "Under", "price": 1.91, "point": 220.5}]}]
    if spread is not None:
        markets.append({"key": "spreads", "outcomes": [{"name": home, "price": 1.91, "point": spread},
                                                       {"name": away, "price": 1.91, "point": -spread}]})
    return normalize_event({"id": event_id, "sport_key": league, "sport_title": league, "commence_time": start,
                            "home_team": home, "away_team": away,
                            "bookmakers": [{"key": "dk", "title": "DraftKings", "markets": markets}]})


class FakeEspn:
    """fetch_scoreboard from fixed boards: {(league, day): events or an exception}; every call is counted."""

    def __init__(self, boards: Optional[Dict[Any, Any]] = None) -> None:
        self.boards = boards or {}
        self.calls: List[Any] = []

    def fetch_scoreboard(self, league, day):
        self.calls.append((league, day))
        value = self.boards.get((league, day), [])
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


class FakeOdds:
    """The odds service as the slate uses it. league_games(refresh=True) "fetches" by saving a new snapshot."""

    odds_ttl = 7200
    reserve = 50

    def __init__(self, store: Store, listing=None, remaining: Optional[int] = 300, now=lambda: NOW) -> None:
        self.store = store
        self.listing = listing if listing is not None else [
            {"key": NBA, "group": "Basketball", "title": "NBA", "active": True, "has_outrights": False},
            {"key": NHL, "group": "Ice Hockey", "title": "NHL", "active": True, "has_outrights": False},
            {"key": EPL, "group": "Soccer", "title": "EPL", "active": True, "has_outrights": False},
            {"key": ODI, "group": "Cricket", "title": "One Day Internationals", "active": True, "has_outrights": False},
            {"key": "golf_masters_tournament_winner", "group": "Golf", "title": "Masters", "active": True,
             "has_outrights": True},
        ]
        self.quota = {"remaining": remaining, "used": 200}
        self.fetches: List[str] = []
        self.now = now

    def sports(self):
        return copy.deepcopy(self.listing)

    def league_games(self, league, refresh=False):
        self.fetches.append(league)
        self.store.save_odds_snapshot(league, [odds_event(league, "fresh1", "Home Side", "Away Side", at(30))],
                                      self.now())
        return []


@pytest.fixture()
def store(tmp_path):
    return Store(str(tmp_path / "betting.db"))


def builder_for(store, espn, odds=None, **kwargs) -> SlateBuilder:
    return SlateBuilder(odds or FakeOdds(store), espn, store, now=kwargs.pop("now", lambda: NOW), workers=2, **kwargs)


def games_of(builder: SlateBuilder) -> List[Dict[str, Any]]:
    return json.loads(builder.current().raw)["games"]


def league_entry(builder: SlateBuilder, key: str) -> Dict[str, Any]:
    return next(entry for entry in json.loads(builder.current().raw)["leagues"] if entry["key"] == key)


# --- assembling the slate ------------------------------------------------------------------------------

def test_an_espn_only_league_lists_its_games_without_odds(store) -> None:
    espn = FakeEspn({(NHL, TODAY): [espn_game("401", "Boston Bruins", "Toronto Maple Leafs", at(1), sport="nhl")]})
    builder = builder_for(store, espn)
    builder.refresh()
    [game] = games_of(builder)
    # The id holds the ESPN id (no "espn_id" too); nulls are left out
    assert game["id"] == "icehockey_nhl:espn:401" and "espn_id" not in game and game["league"] == NHL
    assert ids_of(game) == (None, "401")
    assert "odds_id" not in game and "odds" not in game and "chance" not in game and "mkt" not in game
    # The logo as its path after https://a.espncdn.com/; the full name stays (search by city, slip labels)
    assert game["home"] == {"name": "Boston Bruins", "short": "Bruins", "abbr": "BB", "record": "3-1",
                            "logo": "i/teamlogos/nhl/500/scoreboard/1.png"}
    assert game["venue"] == {"name": "Bruins Arena", "city": "Boston", "indoor": True}
    assert game["status"] == "pre" and game["start"] == at(1).replace("Z", ":00Z")
    entry = league_entry(builder, NHL)
    assert entry == {"key": NHL, "title": "NHL", "group": "Ice Hockey", "label": "🏒 Ice Hockey", "count": 1, "live": 0,
                     "ai": False, "odds": "none", "espn": True}
    # Every in-season league is listed, even with no games yet; futures-only markets are not
    assert league_entry(builder, EPL)["count"] == 0
    assert all(entry["key"] != "golf_masters_tournament_winner" for entry in json.loads(builder.current().raw)["leagues"])
    # Today and the next six days, one request per board (ESPN refuses date ranges), never the odds feed
    assert {(NHL, TODAY + datetime.timedelta(days=n)) for n in range(7)} <= set(espn.calls)


def test_an_odds_only_league_takes_its_games_from_the_saved_snapshot(store) -> None:
    store.save_odds_snapshot(ODI, [
        odds_event(ODI, "odi1", "India", "Australia", at(20), home_ml=1.6, away_ml=2.4, spread=None),
        odds_event(ODI, "odi0", "England", "Ireland", at(-5 * 24), spread=None),  # long over: its result is unknown
    ], NOW - 600)
    odds = FakeOdds(store)
    builder = builder_for(store, FakeEspn(), odds)
    builder.refresh()
    [game] = [g for g in games_of(builder) if g["league"] == ODI]
    assert game["id"] == "cricket_odi:odi1" and "odds_id" not in game and "espn_id" not in game
    assert ids_of(game) == ("odi1", None)
    assert game["home"] == {"name": "India"} and game["status"] == "pre"  # no short name: the full one stays
    assert game["odds"]["h2h"] == {"home": [1.6, "DraftKings"], "away": [2.4, "DraftKings"]}
    assert game["odds"]["totals"]["over"] == [1.91, "DraftKings", 220.5] and "spreads" not in game["odds"]
    # The markets brain's blend, computed in code: one US book's margin-free chances, 1/1.6 and 1/2.4 scaled to
    # add up to 1, are the only opinion
    assert game["chance"] == {"home": 0.6, "away": 0.4, "src": "blend"}
    assert game["mkt"] == {"p": {"home": 0.6, "away": 0.4}, "fav": "home", "conf": "normal", "dis": False,
                           "spr": 0.0, "src": [["US", 0.6, "ok"]], "as_of": "2026-10-26T21:50:00Z"}
    assert game["odds_as_of"] == "2026-10-26T21:50:00Z"
    entry = league_entry(builder, ODI)
    assert entry["label"] == "🏏 Cricket" and entry["odds"] == "fresh" and entry["espn"] is False
    assert entry["odds_as_of"] == "2026-10-26T21:50:00Z"
    assert odds.fetches == []  # building the slate never asks the odds feed for anything but the free listing


def test_odds_join_espn_games_on_team_names_within_three_hours(store) -> None:
    espn = FakeEspn({(NBA, TODAY): [
        espn_game("e1", "Los Angeles Clippers", "Boston Celtics", at(1), ids=("12", "2")),
        espn_game("e2", "Denver Nuggets", "Utah Jazz", at(2), ids=("7", "26")),
        espn_game("e3", "Miami Heat", "Chicago Bulls", at(3), ids=("14", "4")),
        espn_game("e4", "Phoenix Suns", "Sacramento Kings", at(1), ids=("21", "23")),
    ]})
    store.save_odds_snapshot(NBA, [
        odds_event(NBA, "o1", "LA Clippers", "Boston Celtics", at(1)),  # "LA" is Los Angeles
        odds_event(NBA, "o2", "Denver Nuggets", "Utah Jazz", at(2 + 2.98)),  # 2 h 59 min apart: the same game
        odds_event(NBA, "o3", "Miami Heat", "Chicago Bulls", at(3 + 3.05)),  # over 3 hours apart: another game
        odds_event(NBA, "o4", "Sacramento Kings", "Phoenix Suns", at(1), home_ml=2.5, away_ml=1.55),  # sides swapped
    ], NOW - 60)
    builder = builder_for(store, espn)
    builder.refresh()
    games = {g["id"]: g for g in games_of(builder)}
    assert games["basketball_nba:o1"]["espn_id"] == "e1" and games["basketball_nba:o1"]["home"]["abbr"] == "LAC"
    assert games["basketball_nba:o2"]["espn_id"] == "e2"
    assert "espn_id" not in games["basketball_nba:o3"] and "basketball_nba:espn:e3" in games
    # ESPN lists Phoenix at home, the odds feed Sacramento: the game keeps the odds feed's sides, so a price
    # added to a slip names the side the server prices
    swapped = games["basketball_nba:o4"]
    assert swapped["espn_id"] == "e4" and swapped["home"]["short"] == "Kings" and swapped["home"]["abbr"] == "SK"
    assert swapped["odds"]["h2h"]["home"] == [2.5, "DraftKings"] and swapped["chance"]["home"] < 0.5
    assert len(games) == 5


def test_live_final_and_postponed_games(store) -> None:
    espn = FakeEspn({
        (NBA, TODAY): [
            espn_game("l1", "Boston Celtics", "New York Knicks", at(-1), state="in", name="STATUS_IN_PROGRESS",
                      scores=(88, 80), period=3, clock="4:12"),
            espn_game("f1", "Denver Nuggets", "Utah Jazz", at(-4), state="post", name="STATUS_FINAL", scores=(120, 101)),
            espn_game("p1", "Miami Heat", "Chicago Bulls", at(2), state="post", name="STATUS_POSTPONED"),
        ],
        (EPL, TODAY): [espn_game("s1", "Arsenal", "Chelsea", at(-1), state="in", name="STATUS_SECOND_HALF",
                                 scores=(1, 1), period=2, clock="67'", detail="67'", sport="soccer")],
        # Played last night (ESPN's previous day): a back-to-back for Boston tonight
        (NBA, TODAY - datetime.timedelta(days=1)): [espn_game("y1", "Boston Celtics", "Orlando Magic", at(-22),
                                                              state="post", name="STATUS_FINAL", scores=(99, 98),
                                                              ids=("1", "19"))],
    })
    builder = builder_for(store, espn)
    builder.refresh()
    games = {ids_of(g)[1]: g for g in games_of(builder)}
    live = games["l1"]
    assert live["status"] == "live" and live["clock"] == "Q3 4:12"
    assert (live["home"]["score"], live["away"]["score"]) == (88, 80)
    assert live["badges"] == ["live", "b2b-home"]
    assert games["f1"]["status"] == "final" and games["f1"]["home"]["score"] == 120 and "clock" not in games["f1"]
    assert games["p1"]["status"] == "postponed" and "score" not in games["p1"]["home"]
    assert games["s1"]["clock"] == "67'" and games["s1"]["status"] == "live"
    assert "y1" not in games  # last night's finished games aren't on the slate
    assert league_entry(builder, NBA)["live"] == 1 and league_entry(builder, NBA)["count"] == 3


def test_last_night_s_games_show_only_while_still_live(store) -> None:
    # Half past midnight in New York: ESPN's day is the 27th, and last night's board is the 26th's
    late = datetime.datetime(2026, 10, 27, 0, 30, tzinfo=ESPN_DAY_TZ).timestamp()
    espn = FakeEspn({(NBA, TODAY): [
        espn_game("done", "Boston Celtics", "New York Knicks", at(4), state="post", name="STATUS_FINAL", scores=(101, 99)),
        espn_game("late", "Golden State Warriors", "Denver Nuggets", at(5), state="in", name="STATUS_IN_PROGRESS",
                  scores=(90, 91), period=4, clock="2:10", ids=("9", "7")),
    ]})
    store.save_odds_snapshot(NBA, [odds_event(NBA, "o-done", "Boston Celtics", "New York Knicks", at(4)),
                                   odds_event(NBA, "o-late", "Golden State Warriors", "Denver Nuggets", at(5))], NOW)
    builder = builder_for(store, espn, now=lambda: late)
    builder.refresh()
    games = games_of(builder)
    # The finished game isn't brought back as "live" by its saved odds; the one still on is shown with them
    assert [(g["id"], g["status"], g["clock"]) for g in games] == [("basketball_nba:o-late", "live", "Q4 2:10")]


def test_the_nba_model_s_logged_chance_is_shown_never_computed(store) -> None:
    espn = FakeEspn({(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(2))]})
    store.save_odds_snapshot(NBA, [odds_event(NBA, "o1", "Boston Celtics", "New York Knicks", at(2))], NOW - 60)
    store.log_ai_prediction({"league": NBA, "event_id": "o1", "espn_id": "e1", "home": "Boston Celtics",
                             "away": "New York Knicks", "start": at(2), "payload": {"home_win": 0.64}})
    builder = builder_for(store, espn)
    builder.refresh()
    [game] = games_of(builder)
    assert game["ai"] == {"home": 0.64, "away": 0.36} and game["chance"]["src"] == "blend"
    assert league_entry(builder, NBA)["ai"] is True


def test_a_failed_league_keeps_its_last_good_games_even_after_a_restart(store) -> None:
    boards = {(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(2))],
              (NHL, TODAY): [espn_game("h1", "Boston Bruins", "Buffalo Sabres", at(2), sport="nhl")]}
    espn = FakeEspn(boards)
    clock = {"now": NOW}
    builder = builder_for(store, espn, now=lambda: clock["now"])
    builder.refresh()
    first = builder.current()
    boards[(NBA, TODAY)] = RuntimeError("ESPN returned HTTP 503")
    clock["now"] += 3600  # every board is due again
    builder.refresh()
    assert {ids_of(g)[1] for g in games_of(builder)} == {"e1", "h1"}
    assert builder.current().etag == first.etag  # nothing changed, so the same bytes and ETag (a 304 for the app)
    # A restart: the saved slate is served at once, and stands in for boards ESPN won't give
    restarted = builder_for(store, FakeEspn({(NBA, TODAY): RuntimeError("down"),
                                             (NHL, TODAY): boards[(NHL, TODAY)]}), now=lambda: clock["now"])
    assert restarted.load() and restarted.current().etag == first.etag
    restarted.refresh()
    assert {ids_of(g)[1] for g in games_of(restarted)} == {"e1", "h1"}


def test_the_refresh_pace_follows_the_games(store) -> None:
    clock = {"now": NOW}
    boards = {(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(3))]}
    espn = FakeEspn(boards)
    builder = builder_for(store, espn, now=lambda: clock["now"], live_seconds=20, today_seconds=60,
                          idle_seconds=600, future_seconds=3600)
    assert builder.refresh() == 60  # games still to play today: a minute
    calls = len(espn.calls)
    clock["now"] += 61
    builder.refresh()
    # Only today's boards with games were due again (and NBA's last night, kept for back-to-backs, isn't)
    assert set(espn.calls[calls:]) == {(NBA, TODAY)}
    boards[(NBA, TODAY)] = [espn_game("e1", "Boston Celtics", "New York Knicks", at(-0.5), state="in",
                                      name="STATUS_IN_PROGRESS", scores=(10, 8), period=1, clock="5:00")]
    clock["now"] += 61
    builder.refresh()
    assert builder.refresh() == 20  # a game live: every 20 seconds
    boards[(NBA, TODAY)] = []
    clock["now"] += 21
    # Nothing left today anywhere: each board is checked every 10 minutes (the days ahead hourly)
    assert 60 < builder.refresh() <= 600
    calls = len(espn.calls)
    clock["now"] += 61
    builder.refresh()
    assert len(espn.calls) == calls  # no board was due a minute later


def test_the_slate_carries_the_month_s_odds_quota_for_the_header(store) -> None:
    espn = FakeEspn({(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(2))]})
    odds = FakeOdds(store, remaining=310)
    builder = builder_for(store, espn, odds)
    builder.refresh()
    assert json.loads(builder.current().raw)["quota"] == {"remaining": 310, "used": 200}
    etag = builder.current().etag
    builder.refresh()  # nothing changed: the same bytes
    assert builder.current().etag == etag
    odds.quota = {"remaining": 307, "used": 203}  # the bot loaded a board: the header's count follows
    builder.refresh()
    assert json.loads(builder.current().raw)["quota"]["remaining"] == 307 and builder.current().etag != etag
    odds.quota = {"remaining": None, "used": None}  # unknown (a fresh install): left out
    builder.refresh()
    assert "quota" not in json.loads(builder.current().raw)


# --- the refresh schedule: one wake-up per round, backing off while ESPN fails, the listing off the path ----------

class ClockedEspn:
    """ESPN on a simulated clock: an NBA game live today, a game a day ahead in every league; each request takes
    ``latency`` simulated seconds; from ``fail_from`` seconds after NOW every request fails (an outage or a 429).
    Records (time, league, day, ok) of every request."""

    def __init__(self, clock: Dict[str, float], latency: float = 0.0, fail_from: float = float("inf"),
                 live: bool = True, today_games: tuple = ()) -> None:
        self.clock, self.latency, self.fail_from, self.live, self.today_games = clock, latency, fail_from, live, today_games
        self.calls: List[Any] = []
        self.lock = threading.Lock()

    def fetch_scoreboard(self, league, day):
        with self.lock:
            asked = self.clock["now"]
            self.clock["now"] += self.latency
            failing = asked - NOW >= self.fail_from
            self.calls.append((asked, league, day, not failing))
        if failing:
            raise RuntimeError("ESPN returned HTTP 429 for https://site.api.espn.com/apis/site/v2/sports/...")
        if (league, day) == (NBA, TODAY) and self.live:
            return [espn_game("live", "Boston Celtics", "New York Knicks", at(-1), state="in", name="STATUS_IN_PROGRESS",
                              scores=(50, 48), period=2, clock="3:00")]
        if day == TODAY and league in self.today_games:
            return [espn_game(f"{league}-today", "Home Side", "Away Side", at(3))]
        if day > TODAY:
            start = datetime.datetime.combine(day, datetime.time(19), ESPN_DAY_TZ).astimezone(datetime.timezone.utc)
            return [espn_game(f"{league}-{day}", "Home Side", "Away Side", start.strftime("%Y-%m-%dT%H:%MZ"))]
        return []

    def per_hour(self, ok: bool, board=None) -> Counter:
        return Counter(int((t - NOW) // 3600) for t, league, day, good in self.calls
                       if good == ok and (board is None or (league, day) == board))


def run_for(builder: SlateBuilder, clock: Dict[str, float], seconds: float) -> List[float]:
    """Call refresh() whenever the builder asks to be woken, for ``seconds`` of simulated time; the round starts."""
    starts = []
    end = clock["now"] + seconds
    while clock["now"] < end:
        starts.append(clock["now"])
        delay = builder.refresh()  # (the round itself moves the clock on)
        clock["now"] += delay
    return starts


def test_one_round_one_wake_up_however_long_the_round_takes(store) -> None:
    # Each request takes a fifth of a second, so a round of every board takes seconds: the boards of a round are
    # still stamped together, and come due together, so the slate is rebuilt once per live interval, not once per
    # straggler (each extra wake-up rebuilds and compares the whole slate)
    clock = {"now": NOW}
    espn = ClockedEspn(clock, latency=0.2, today_games=(NHL, EPL))
    builder = builder_for(store, espn, now=lambda: clock["now"], live_seconds=20, today_seconds=60,
                          idle_seconds=600, future_seconds=3600)
    starts = run_for(builder, clock, 900)
    gaps = [round(b - a, 1) for a, b in zip(starts, starts[1:])]
    assert len(starts) <= 900 / 20 + 2, gaps
    assert min(gaps[1:]) >= 19.9, gaps  # after the first (long) round, never a straggler a second or two later
    asked = Counter((league, day) for _, league, day, _ in espn.calls)
    assert asked[(NBA, TODAY)] == len(starts) == 45  # the live board every 20 seconds, in every round
    assert asked[(NHL, TODAY)] == 15 and asked[(NHL, TODAY + datetime.timedelta(days=1))] == 1  # every minute; hourly


def test_a_failing_espn_is_asked_no_more_often_than_a_working_one(store, caplog) -> None:
    # Every league, the production pace (live 20 s, today 60 s, idle 10 min, days ahead hourly); ESPN answers for
    # three hours, then fails every board
    clock = {"now": datetime.datetime(2026, 10, 27, 1, 0, tzinfo=ESPN_DAY_TZ).timestamp()}
    start = clock["now"]
    listing = [{"key": k, "group": "Soccer", "title": k, "active": True, "has_outrights": False} for k in LEAGUES]
    espn = ClockedEspn(clock, fail_from=start - NOW + 3 * 3600, live=False)
    builder = builder_for(store, espn, FakeOdds(store, listing=listing), now=lambda: clock["now"],
                          live_seconds=20, today_seconds=60, idle_seconds=600, future_seconds=3600)
    with caplog.at_level(logging.WARNING, logger="telegram_betting_bot_simple.data.slate"):
        starts = run_for(builder, clock, 6 * 3600)
    hour = lambda t: int((t - start) // 3600)  # noqa: E731
    asked = Counter(hour(t) for t, *_ in espn.calls)
    rounds = Counter(hour(t) for t in starts)
    healthy = max(asked[1], asked[2])
    assert healthy > 300
    for failing in (3, 4, 5):
        assert asked[failing] <= healthy, asked  # was about 35 times as many (every board every minute)
        assert rounds[failing] <= max(rounds[1], rounds[2]), rounds
    assert len(builder.current().data["games"]) > 300  # the last good games stay on show
    paused = [r.getMessage() for r in caplog.records if "only boards with a game on" in r.getMessage()]
    assert len(paused) == 1  # said once, not every round


def test_a_live_board_is_still_fetched_while_espn_is_failing_but_further_apart(store) -> None:
    clock = {"now": NOW}
    espn = ClockedEspn(clock, fail_from=3600, today_games=(NHL, EPL))
    builder = builder_for(store, espn, now=lambda: clock["now"], live_seconds=20, today_seconds=60,
                          idle_seconds=600, future_seconds=3600)
    run_for(builder, clock, 3 * 3600)
    live = (NBA, TODAY)
    assert espn.per_hour(True, live)[0] >= 170  # every 20 seconds while ESPN answers
    for failing in (1, 2):
        tried = espn.per_hour(False, live)[failing]
        assert 0 < tried <= 20  # still tried (a game is on), but at most every 5 minutes once it keeps failing
        assert espn.per_hour(False)[failing] <= espn.per_hour(True)[0]  # all boards together, too
    # While paused, only the live board is asked for: nothing else in the 5 minutes after the first failed round
    first = min(t for t, *_, ok in espn.calls if not ok)
    assert {(league, day) for t, league, day, _ in espn.calls if first + 1 <= t < first + 300} <= {live}
    [game] = [g for g in builder.current().data["games"] if g["status"] == "live"]
    assert game["home"]["score"] == 50  # its last good score stays on show


class CountingOdds(FakeOdds):
    """The odds service with a listing that can fail or hang; counts the listing's reads."""

    def __init__(self, store, **kwargs) -> None:
        super().__init__(store, **kwargs)
        self.reads = 0
        self.error: Optional[Exception] = None
        self.gate: Optional[threading.Event] = None

    def sports(self):
        self.reads += 1
        if self.gate is not None:
            self.gate.wait(10)
        if self.error is not None:
            raise self.error
        return super().sports()


def test_the_in_season_listing_is_read_in_the_background_not_on_every_refresh(store) -> None:
    clock = {"now": NOW}
    odds = CountingOdds(store, now=lambda: clock["now"])
    builder = builder_for(store, FakeEspn(), odds, now=lambda: clock["now"])
    builder.refresh()
    assert odds.reads == 1  # the first slate waits for it, to know which leagues are in season
    for _ in range(5):
        clock["now"] += 60
        builder.refresh()
    wait_for(builder)
    assert odds.reads == 1  # not again for SEASON_EVERY seconds
    clock["now"] += SEASON_EVERY
    builder.refresh()
    wait_for(builder)
    assert odds.reads == 2
    # The listing fails (The Odds API unreachable): the last good list is kept, and it's tried again only after
    # SEASON_RETRY seconds
    odds.error = RuntimeError("Couldn't reach The Odds API for /sports (ConnectionError)")
    clock["now"] += SEASON_EVERY + 1
    builder.refresh()
    wait_for(builder)
    assert odds.reads == 3
    for _ in range(3):
        clock["now"] += SEASON_RETRY / 4
        builder.refresh()
    wait_for(builder)
    assert odds.reads == 3
    clock["now"] += SEASON_RETRY / 4 + 1
    builder.refresh()
    wait_for(builder)
    assert odds.reads == 4
    assert league_entry(builder, ODI)["count"] == 0  # still listed: the last good listing stands
    # A listing that hangs (until requests' timeout) doesn't hold up the scores
    odds.error, odds.gate = None, threading.Event()
    clock["now"] += SEASON_EVERY + 1
    began = time.monotonic()
    builder.refresh()
    assert time.monotonic() - began < 2 and odds.reads == 5
    odds.gate.set()
    wait_for(builder)


KEY = "MADEUPKEY0123456789abcdef01234567"  # not a real key


def test_the_odds_api_key_never_reaches_the_log(store, monkeypatch, caplog) -> None:
    def unreachable(url, params=None, timeout=None):
        # What requests says when the host can't be reached: the URL, with the key in its query
        raise requests.exceptions.ConnectionError(
            f"HTTPSConnectionPool(host='api.the-odds-api.com', port=443): Max retries exceeded with url: "
            f"/v4/sports?apiKey={params['apiKey']} (Caused by NewConnectionError('Failed to establish a connection'))")

    monkeypatch.setattr(odds_service.requests, "get", unreachable)
    cache = TTLCache()
    odds = OddsService(KEY, cache, store=store)
    clock = {"now": NOW}
    builder = builder_for(store, FakeEspn(), odds, now=lambda: clock["now"])
    with caplog.at_level(logging.INFO):
        builder.refresh()  # no listing yet: the slate's own warning
        cache._entries[("odds", "sports")] = (time.monotonic() - 7 * 3600, time.time(), [])  # an expired copy
        clock["now"] += SEASON_EVERY + 1
        builder.refresh()  # the shared cache's warning, serving the old copy
        wait_for(builder)
        with pytest.raises(RuntimeError) as error:
            odds.league_games(EPL)
    assert "ConnectionError" in str(error.value) and KEY not in str(error.value)
    # A logged traceback (log.exception) doesn't print the connection error it replaced either
    assert KEY not in "".join(traceback.format_exception(error.value))
    assert "couldn't load sports listing" in caplog.text and "serving the cached copy" in caplog.text
    assert KEY not in caplog.text


def test_a_sixty_game_slate_is_a_few_kilobytes(store) -> None:
    names = ["Boston Celtics", "New York Knicks", "Denver Nuggets", "Utah Jazz", "Miami Heat", "Chicago Bulls",
             "Phoenix Suns", "Sacramento Kings", "Golden State Warriors", "Dallas Mavericks", "Atlanta Hawks",
             "Detroit Pistons", "Orlando Magic", "Toronto Raptors", "Houston Rockets", "Memphis Grizzlies"]
    boards: Dict[Any, Any] = {}
    snapshot = []
    for n in range(60):
        day = TODAY + datetime.timedelta(days=n % 7)
        home, away = names[n % 16], names[(n * 7 + 3) % 16]
        start = at(1 + (n % 7) * 24 + (n % 5) * 0.5)
        boards.setdefault((NBA, day), []).append(espn_game(f"4019{n:05d}", home, away, start, ids=(str(n % 30), str(n % 29))))
        if n % 2 == 0:  # odds on half of them, as when only some leagues' boards were loaded
            snapshot.append(odds_event(NBA, f"{n:032x}", home, away, start))
    store.save_odds_snapshot(NBA, snapshot, NOW - 60)
    builder = builder_for(store, FakeEspn(boards))
    builder.refresh()
    current = builder.current()
    assert len(json.loads(current.raw)["games"]) == 60
    assert gzip.decompress(current.gzip) == current.raw
    # About 3 KB per 60 games without prices; the best prices for 30 of them add about 1.5 KB
    assert len(current.gzip) < 5000, len(current.gzip)


# --- GET /api/slate ------------------------------------------------------------------------------------

@pytest.fixture()
def slate_app(service, store):
    service.store = store
    espn = FakeEspn({(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(2))],
                     (NHL, TODAY): [espn_game("h1", "Boston Bruins", "Buffalo Sabres", at(3), sport="nhl")]})
    builder = builder_for(store, espn)
    builder.refresh()
    return TestClient(create_app(service, "s3cret", login_delay=0, slate=builder)), builder


def signed_in(client: TestClient, confirm_age: bool = True) -> TestClient:
    assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
    client.headers.update(PLAYER)
    if confirm_age:
        assert client.put("/api/me", json={"confirm_age": True}).status_code == 200
    return client


def test_the_slate_is_served_as_ready_bytes_with_an_etag_and_gzip(slate_app) -> None:
    client, builder = slate_app
    assert client.get("/api/slate").status_code == 401  # signed in only
    signed_in(client)
    response = client.get("/api/slate", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip" and response.headers["cache-control"] == "no-cache"
    assert response.headers["etag"] == builder.current().etag and "Accept-Encoding" in response.headers["vary"]
    assert response.content == builder.current().raw  # the client unzipped the ready-made gzip
    assert int(response.headers["content-length"]) == len(builder.current().gzip)
    data = response.json()
    assert {g["id"] for g in data["games"]} == {"basketball_nba:espn:e1", "icehockey_nhl:espn:h1"}
    assert data["generated_at"] and data["as_of"]
    # Unchanged: a 304 with no body
    again = client.get("/api/slate", headers={"If-None-Match": response.headers["etag"]})
    assert again.status_code == 304 and again.content == b"" and again.headers["etag"] == response.headers["etag"]
    assert client.get("/api/slate", headers={"If-None-Match": 'W/"nope"'}).status_code == 200
    # Without gzip, the plain bytes
    plain = client.get("/api/slate", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers and plain.content == builder.current().raw


def test_the_slate_can_be_filtered_on_the_server(slate_app) -> None:
    client, _ = slate_app
    signed_in(client)
    data = client.get("/api/slate?leagues=icehockey_nhl").json()
    assert [g["league"] for g in data["games"]] == [NHL] and [e["key"] for e in data["leagues"]] == [NHL]
    data = client.get("/api/slate?day=2026-10-27").json()
    assert data["games"] == [] and all(entry["count"] == 0 for entry in data["leagues"])
    assert len(client.get("/api/slate?day=2026-10-26").json()["games"]) == 2
    assert client.get("/api/slate?day=26-10-2026").status_code == 422
    assert client.get("/api/slate?day=2026-13-45").status_code == 400
    assert client.get("/api/slate?leagues=NBA!").status_code == 422


def test_the_slate_keeps_the_gates(slate_app, store) -> None:
    client, _ = slate_app
    signed_in(client, confirm_age=False)
    response = client.get("/api/slate")
    assert response.status_code == 400 and "18+" in response.json()["detail"]
    assert client.post("/api/slate/odds", json={"league": NBA}).status_code == 400
    assert client.put("/api/me", json={"confirm_age": True}).status_code == 200
    assert client.get("/api/slate").status_code == 200
    assert client.post("/api/me/cooloff", json={"hours": 24}).status_code == 200
    response = client.get("/api/slate")
    assert response.status_code == 400 and "break" in response.json()["detail"]
    client.headers.pop("X-Player")
    assert client.get("/api/slate").status_code == 400  # a password browser must say which browser it is


def test_without_a_slate_yet_the_app_says_to_try_again(service, store) -> None:
    service.store = store
    builder = builder_for(store, FakeEspn())
    client = signed_in(TestClient(create_app(service, "s3cret", login_delay=0, slate=builder)))
    import telegram_betting_bot_simple.dashboard.app as app_module
    original, app_module.SLATE_WAIT = app_module.SLATE_WAIT, 0.01
    try:
        response = client.get("/api/slate")
    finally:
        app_module.SLATE_WAIT = original
    assert response.status_code == 503 and response.headers["retry-after"] == "3"


def test_a_restarted_dashboard_serves_the_saved_slate_at_once(service, store) -> None:
    first = builder_for(store, FakeEspn({(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(2))]}))
    first.refresh()
    assert store.slate(SLATE_NAME)[1] == first.current().etag
    service.store = store
    # The new process hasn't refreshed (its ESPN is down); it serves what the last one saved
    restarted = builder_for(store, FakeEspn({(NBA, TODAY): RuntimeError("down")}))
    assert restarted.load()
    client = signed_in(TestClient(create_app(service, "s3cret", login_delay=0, slate=restarted)))
    response = client.get("/api/slate")
    assert response.status_code == 200 and response.headers["etag"] == first.current().etag
    assert response.content == first.current().raw


@pytest.mark.parametrize("body", [
    b"[]",  # valid JSON, not a slate
    b'{"leagues": 5, "games": []}',
    json.dumps({"leagues": [], "games": [{"id": "basketball_nba:espn:1", "league": NBA, "espn_id": "1",
                                          "start": "2026-10-26T23:00:00Z", "home": "Boston Celtics",
                                          "away": "New York Knicks"}]}).encode(),  # another version's games
    b'{"leagues": [{"key": ["x"], "title": "X"}], "games": []}',
    b"\xff\xfe not even text",
], ids=["a-list", "leagues-not-a-list", "teams-as-strings", "unhashable-key", "not-utf8"])
def test_a_saved_slate_of_another_shape_is_set_aside_not_a_crash(store, caplog, body) -> None:
    # The dashboard calls load() at startup without a try: a row it can't use (after a rollback, say) must not
    # stop it starting, or systemd would restart it into the same row for ever
    store.save_slate(SLATE_NAME, body, '"old"', NOW)
    builder = builder_for(store, FakeEspn({(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(2))]}))
    with caplog.at_level(logging.WARNING):
        assert builder.load() is False
    assert builder.current() is None and "starting without it" in caplog.text
    builder.refresh()  # the first refresh builds and saves a new slate
    assert [ids_of(g)[1] for g in games_of(builder)] == ["e1"] and store.slate(SLATE_NAME)[1] == builder.current().etag


# --- POST /api/slate/odds: fresh odds on request, within the quota's limits ----------------------------------

def wait_for(builder: SlateBuilder) -> None:
    builder._odds_pool.submit(lambda: None).result(timeout=5)


def test_fresh_odds_are_loaded_once_per_two_hours_per_league(store) -> None:
    clock = {"now": NOW}
    odds = FakeOdds(store, now=lambda: clock["now"])
    espn = FakeEspn({(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(30))]})
    builder = builder_for(store, espn, odds, now=lambda: clock["now"])
    builder.refresh()
    assert builder.request_odds(NBA) == {"queued": True, "code": "queued", "reason": "Loading the latest odds."}
    wait_for(builder)
    assert odds.fetches == [NBA]
    builder.refresh()  # the slate picks up the new board (the background load also wakes the thread)
    assert league_entry(builder, NBA)["odds"] == "fresh"
    assert builder.request_odds(NBA)["code"] == "fresh"
    # Over two hours later the saved odds are stale, and they may be loaded again, once
    clock["now"] += 7300
    builder.refresh()
    assert league_entry(builder, NBA)["odds"] == "stale"
    assert builder.request_odds(NBA)["queued"] is True
    wait_for(builder)
    store.save_odds_snapshot(NBA, [], clock["now"] - 7201)  # say that load failed: the odds are still old
    assert builder.request_odds(NBA) == {"queued": False, "code": "throttled", "reason": "Asked for less than 2 hours ago."}
    assert odds.fetches == [NBA, NBA]
    # A restarted dashboard can't ask again within the two hours either (a lease in the shared store)
    other = builder_for(store, espn, odds, now=lambda: clock["now"])
    other.refresh()
    other_store_holder = store.holder
    store.holder = "another-process"
    try:
        assert other.request_odds(NBA)["queued"] is False
    finally:
        store.holder = other_store_holder


def test_fresh_odds_respect_the_reserve_the_season_the_games_and_a_daily_cap(store) -> None:
    clock = {"now": NOW}
    espn = FakeEspn({(NBA, TODAY): [espn_game("e1", "Boston Celtics", "New York Knicks", at(5))]})
    low = FakeOdds(store, remaining=52)  # 52 - 3 credits would leave less than the reserve of 50
    builder = builder_for(store, espn, low)
    builder.refresh()
    assert builder.request_odds(NBA)["reason"].startswith("The month's odds allowance is down to its reserve")
    assert low.fetches == []
    odds = FakeOdds(store, now=lambda: clock["now"])
    builder = builder_for(store, espn, odds, now=lambda: clock["now"], odds_per_day=1)
    builder.refresh()
    assert builder.request_odds("basketball_euroleague")["queued"] is False  # not in season at The Odds API
    assert builder.request_odds("tennis_atp")["queued"] is False  # ESPN's tour: no odds market of its own
    # ESPN shows no NHL or EPL games this week: no credits spent on an empty board
    assert builder.request_odds(NHL) == {"queued": False, "code": "no_games", "reason": "No games coming up in this league."}
    assert builder.request_odds(EPL)["queued"] is False
    assert builder.request_odds(NBA)["queued"]
    wait_for(builder)
    # One board a day (SLATE_ODDS_PER_DAY), all leagues together; a day later there's room again
    assert builder.request_odds(ODI)["reason"].startswith("Today's odds loads are used up")
    clock["now"] += 24 * 3600 + 1
    builder.refresh()
    assert builder.request_odds(ODI)["queued"]
    wait_for(builder)
    assert odds.fetches == [NBA, ODI]


class PayingOdds(FakeOdds):
    """league_games pays for its board as The Odds API would: 3 credits per region off the reported quota."""

    regions = "us"

    def league_games(self, league, refresh=False):
        result = super().league_games(league, refresh)
        cost = 3 * len(self.regions.split(","))
        self.quota = {"remaining": self.quota["remaining"] - cost, "used": self.quota["used"] + cost}
        return result


def hold_the_odds_pool(builder: SlateBuilder) -> threading.Event:
    """Keep the builder's (one) odds worker busy until the event is set, so requested boards stay queued."""
    gate = threading.Event()
    builder._odds_pool.submit(gate.wait, 10)
    return gate


def test_boards_already_queued_count_against_the_reserve(store) -> None:
    # The app asks for up to four leagues at once, and several people may be on Today: each answer must count
    # the boards already on their way, or they all pass the check and the reserve is spent
    leagues = [EPL, ODI, "soccer_spain_la_liga", "soccer_italy_serie_a"]
    listing = [{"key": key, "group": "Soccer", "title": key, "active": True, "has_outrights": False} for key in leagues]
    odds = PayingOdds(store, listing=listing, remaining=56)  # room for two boards above the reserve of 50
    espn = FakeEspn({(league, TODAY): [espn_game(f"e-{league}", "Home Side", "Away Side", at(30), sport="soccer")]
                     for league in leagues if league != ODI})
    builder = builder_for(store, espn, odds)
    builder.refresh()
    gate = hold_the_odds_pool(builder)
    answers = [builder.request_odds(league)["code"] for league in leagues]
    gate.set()
    wait_for(builder)
    assert answers == ["queued", "queued", "reserve", "reserve"]
    assert odds.fetches == [EPL, ODI] and odds.quota["remaining"] == 50  # never below the reserve


def test_a_queued_board_is_skipped_if_the_quota_fell_meanwhile_and_regions_cost_more(store) -> None:
    listing = [{"key": key, "group": "Soccer", "title": key, "active": True, "has_outrights": False} for key in (EPL, ODI)]
    espn = FakeEspn({(EPL, TODAY): [espn_game("e1", "Arsenal", "Chelsea", at(30), sport="soccer")]})
    odds = PayingOdds(store, listing=listing, remaining=60)
    builder = builder_for(store, espn, odds)
    builder.refresh()
    gate = hold_the_odds_pool(builder)
    assert builder.request_odds(EPL)["code"] == "queued"
    odds.quota = {"remaining": 52, "used": 248}  # the bot loaded boards while this one waited its turn
    gate.set()
    wait_for(builder)
    assert odds.fetches == []  # checked again just before paying for it
    # Two odds regions: a board costs 6 credits, so 60 left leaves room for one board, not two
    odds = PayingOdds(store, listing=listing, remaining=60)
    odds.regions = "us,uk"
    builder = builder_for(store, espn, odds, now=lambda: NOW + 3 * 3600)
    builder.refresh()
    gate = hold_the_odds_pool(builder)
    assert [builder.request_odds(league)["code"] for league in (ODI, EPL)] == ["queued", "reserve"]
    gate.set()
    wait_for(builder)
    assert odds.fetches == [ODI] and odds.quota["remaining"] == 54


def test_the_odds_route_queues_through_the_builder(slate_app) -> None:
    client, builder = slate_app
    signed_in(client)
    assert client.post("/api/slate/odds", json={"league": NBA}).json()["code"] == "queued"
    wait_for(builder)
    assert client.post("/api/slate/odds", json={"league": "NBA!"}).status_code == 422
    assert client.post("/api/slate/odds", json={"league": NBA}).json()["queued"] is False


# --- static files: hashed names, an import map and caching ------------------------------------------------

STATIC = Path(__file__).resolve().parent.parent / "telegram_betting_bot_simple" / "dashboard" / "static"


def test_the_page_names_every_module_by_its_hash(service, store) -> None:
    service.store = store
    client = TestClient(create_app(service, "s3cret", login_delay=0))
    page = client.get("/")
    assert page.status_code == 200 and page.headers["cache-control"] == "no-cache"
    html = page.text
    assets = StaticAssets(STATIC)
    imports = json.loads(re.search(r'<script type="importmap">(.*?)</script>', html, re.S).group(1))["imports"]
    for rel in ("app.js", "js/app.js", "js/api.js", "js/views/today.js", "js/views/ask/cards.js"):
        assert imports[f"/static/{rel}"] == f"/static/{rel}?v={assets.version(rel)}"
    preloads = re.findall(r'<link rel="modulepreload" href="([^"]+)">', html)
    for rel in ("app.js", "js/app.js", "js/api.js", "js/store.js", "js/ui.js", "js/icons.js", "js/telegram.js",
                "js/views/today.js"):
        assert f"/static/{rel}?v={assets.version(rel)}" in preloads
    assert f'href="/static/styles.css?v={assets.version("styles.css")}"' in html
    assert f'src="/static/app.js?v={assets.version("app.js")}"' in html and '?v=6"' not in html
    # The import map comes before every module script and preload, and the charset stays first
    assert html.index('type="importmap"') < min(html.index("modulepreload"), html.index('type="module"'))
    assert html.index("<meta charset") < html.index('type="importmap"')
    # An unchanged page is a 304
    assert client.get("/", headers={"If-None-Match": page.headers["etag"]}).status_code == 304


def test_hashed_files_are_cached_for_good_and_others_rechecked(service, store) -> None:
    client = TestClient(create_app(service, "s3cret", login_delay=0))
    version = StaticAssets(STATIC).version("js/api.js")
    hashed = client.get(f"/static/js/api.js?v={version}")
    assert hashed.status_code == 200 and hashed.headers["cache-control"] == IMMUTABLE
    assert client.get("/static/js/api.js").headers["cache-control"] == "no-cache"
    assert client.get("/static/js/api.js?v=0123456789ab").headers["cache-control"] == "no-cache"  # an old hash
    assert client.get("/static/js/nope.js?v=1").status_code == 404


def test_a_file_changed_without_a_restart_gets_a_new_hash(tmp_path) -> None:
    (tmp_path / "js").mkdir()
    (tmp_path / "index.html").write_text('<!doctype html><html><head><meta charset="utf-8">'
                                         '<script type="module" src="/static/app.js?v=6"></script></head></html>')
    (tmp_path / "app.js").write_text('import "./js/app.js";')
    (tmp_path / "js" / "app.js").write_text("export const a = 1;")
    client = TestClient(create_app(None, "s3cret", login_delay=0, static_dir=tmp_path))
    old = StaticAssets(tmp_path).version("js/app.js")
    assert f"/static/js/app.js?v={old}" in client.get("/").text
    (tmp_path / "js" / "app.js").write_text("export const a = 2; // changed")
    assert client.get(f"/static/js/app.js?v={old}").headers["cache-control"] == "no-cache"
    new = StaticAssets(tmp_path).version("js/app.js")
    assert new != old and f"/static/js/app.js?v={new}" in client.get("/").text


def test_responses_are_compressed_unless_caddy_does_it(service) -> None:
    client = TestClient(create_app(service, "s3cret", login_delay=0))
    direct = client.get("/static/styles.css", headers={"Accept-Encoding": "gzip"})
    assert direct.headers["content-encoding"] == "gzip"
    proxied = client.get("/static/styles.css", headers={"Accept-Encoding": "gzip", "X-Forwarded-For": "203.0.113.9"})
    assert "content-encoding" not in proxied.headers
    small = client.get("/static/app.js", headers={"Accept-Encoding": "gzip"})  # under 1 KB: as it is
    assert "content-encoding" not in small.headers


def test_caddy_compresses_the_site() -> None:
    script = (Path(__file__).resolve().parent.parent / "deploy" / "setup_vps.sh").read_text()
    site = script[script.index('cat > "$CADDY_SITE" <<CADDY'):script.index("\nCADDY\n")]
    assert "\tencode zstd gzip\n" in site


def test_every_espn_league_has_a_path() -> None:
    assert not set(LEAGUES) & set(SCOREBOARD_ONLY)
    for key in list(LEAGUES) + list(SCOREBOARD_ONLY):
        sport, path = espn_path(key)
        assert sport in ("basketball", "football", "hockey", "baseball", "soccer", "tennis", "mma") and path
    assert espn_path("soccer_efl_champ") == ("soccer", "eng.2") and espn_path("cricket_odi") is None
