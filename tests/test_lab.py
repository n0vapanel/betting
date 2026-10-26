"""Tests for the per-person features (dashboard/lab.py) and their API routes.

Fake odds and ESPN clients, a fake player-data source and a fake AI assistant: no network, no
Anthropic calls. A test clock moves time forward for budgets, breaks and the restart wait.
"""

from __future__ import annotations

import copy
import datetime
import sqlite3
import sys
import threading
import time
import types
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.config import Config
from telegram_betting_bot_simple.dashboard import lab as lab_module
from telegram_betting_bot_simple.dashboard.app import INIT_DATA_HEADER, create_app
from telegram_betting_bot_simple.dashboard.lab import (
    AGE_MESSAGE,
    BREAK_SUGGESTION,
    NO_CHAT,
    Lab,
    Run,
    next_stake,
    play_day,
    replay,
)
from telegram_betting_bot_simple.dashboard.service import CallRejected, NotFound
from telegram_betting_bot_simple.data.espn_client import ESPN_DAY_TZ
from telegram_betting_bot_simple.data.odds_service import normalize_event
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.engine import calls
from tests.conftest import GAMES, FakeEspn, FakeOdds, make_event
from tests.test_auth import BOT_TOKEN, make_init_data

NBA = "basketball_nba"
NHL = "icehockey_nhl"
ME = {"X-Player": "browser-1234"}
UID = "web:browser-1234"


# --- fakes ------------------------------------------------------------------------------------------

def more_games() -> List[Dict[str, Any]]:
    """The two shared games plus four more, so tickets can have up to six legs from different games."""
    extra = [
        make_event("g3", "Boston Celtics", "Miami Heat", "2030-01-02T00:00:00Z",
                   {"DraftKings": (1.50, 2.70, -4.5, 1.91), "FanDuel": (1.48, 2.75, -4.5, 1.91)}),
        make_event("g4", "Denver Nuggets", "Utah Jazz", "2030-01-02T01:00:00Z",
                   {"DraftKings": (1.25, 4.20, -8.5, 1.91), "FanDuel": (1.24, 4.30, -8.5, 1.91)}),
        make_event("g5", "Phoenix Suns", "Sacramento Kings", "2030-01-02T02:00:00Z",
                   {"DraftKings": (1.80, 2.05, -2.5, 1.91), "FanDuel": (1.83, 2.02, -2.5, 1.91)}),
        make_event("g6", "Chicago Bulls", "Detroit Pistons", "2030-01-02T03:00:00Z",
                   {"DraftKings": (1.95, 1.87, -1.5, 1.91), "FanDuel": (1.91, 1.91, -1.5, 1.91)}),
    ]
    return copy.deepcopy(GAMES) + extra


NHL_GAMES = [
    make_event("h1", "Boston Bruins", "Toronto Maple Leafs", "2030-01-01T00:30:00Z",
               {"DraftKings": (1.70, 2.20, -1.5, 1.91)}),
]


class Odds(FakeOdds):
    """FakeOdds with more games, a second league in season, one out of season, and a log of fetches."""

    def __init__(self, games: Optional[List[Dict[str, Any]]] = None) -> None:
        self.games = games if games is not None else more_games()
        self.fetched: List[str] = []

    def sports(self):
        return [
            {"key": NBA, "group": "Basketball", "title": "NBA", "active": True, "has_outrights": False},
            {"key": NHL, "group": "Ice Hockey", "title": "NHL", "active": True, "has_outrights": False},
            {"key": "baseball_mlb", "group": "Baseball", "title": "MLB", "active": False, "has_outrights": False},
        ]

    def league_games(self, league, refresh=False):
        self.fetched.append(league)
        return copy.deepcopy(self.games if league == NBA else NHL_GAMES if league == NHL else [])


EPL = "soccer_epl"
MMA = "mma_mixed_martial_arts"


def moneyline_event(event_id: str, league: str, home: str, away: str, prices: Dict[str, float]) -> Dict[str, Any]:
    """A game with only a moneyline, e.g. prices {"home": 2.10, "away": 3.60, "draw": 3.40}."""
    names = {"home": home, "away": away, "draw": "Draw"}
    return normalize_event({"id": event_id, "sport_key": league, "sport_title": league,
                            "commence_time": "2030-01-01T15:00:00Z", "home_team": home, "away_team": away,
                            "bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": [
                                {"key": "h2h", "outcomes": [{"name": names[side], "price": price}
                                                            for side, price in prices.items()]}]}]})


class MoreOdds(Odds):
    """Odds with soccer (a three-way moneyline) and MMA (no ESPN results here, so bets can't be settled)."""

    def sports(self):
        return super().sports() + [
            {"key": EPL, "group": "Soccer", "title": "EPL", "active": True, "has_outrights": False},
            {"key": MMA, "group": "Mixed Martial Arts", "title": "MMA", "active": True, "has_outrights": False},
        ]

    def league_games(self, league, refresh=False):
        if league == EPL:
            return [moneyline_event("s1", EPL, "Arsenal", "Chelsea", {"home": 2.10, "away": 3.60, "draw": 3.40})]
        if league == MMA:
            return [moneyline_event("m1", MMA, "Alex Pereira", "Magomed Ankalaev", {"home": 1.80, "away": 2.05})]
        return super().league_games(league, refresh)


class Results(FakeEspn):
    """ESPN once the games are over: final scores by home team; "void" for a postponed game."""

    def __init__(self, scores: Dict[str, Any]) -> None:
        self.scores = scores

    def find_game(self, league, home, away, start):
        return {"event_id": f"espn-{home}", "home_id": "1", "away_id": "2"} if home in self.scores else None

    def game_result(self, league, event_id):
        score = self.scores[event_id[len("espn-"):]]
        if score == "void":
            return {"state": "void", "status": "STATUS_POSTPONED"}
        if score == "pending":
            return {"state": "pending", "status": "STATUS_IN_PROGRESS"}
        return {"state": "final", "home": score[0], "away": score[1]}


class Clock:
    def __init__(self) -> None:
        self.t = time.time()

    def __call__(self) -> float:
        return self.t

    def advance(self, hours: float) -> None:
        self.t += hours * 3600


ROWS = [{"date": f"2026-10-{20 - i:02d}T23:00:00Z", "opponent": "BOS", "home": i % 2 == 0, "season": "2026-27",
         "minutes": 35.0, "stats": {"pts": pts, "reb": 7.0, "ast": 8.0, "min": 35.0}}
        for i, pts in enumerate([31.0, 22.0, 28.0, 25.0, 30.0, 19.0, 27.0, 33.0, 24.0, 26.0])]


class Players:
    """PlayerData stand-in: one player, his last ten games and two DraftKings lines."""

    def __init__(self, game_date: str = "2030-01-01T00:00:00Z", team: str = "Los Angeles Lakers") -> None:
        self.game_date = game_date
        self.team = team

    def search(self, query, league=None):
        return {"query": query, "reason": None, "players": [
            {"id": "1966", "name": "LeBron James", "league": NBA, "team": self.team, "team_abbr": "LAL",
             "position": "F", "headshot": "https://a.espncdn.com/1966.png", "active": True}]}

    def card(self, league, athlete_id):
        if athlete_id != "1966":
            return {"id": athlete_id, "league": league, "name": None, "reason": "ESPN has no player with that id."}
        return {"id": "1966", "league": league, "name": "LeBron James", "team": self.team, "team_abbr": "LAL",
                "position": "F", "headshot": None, "status": "Active", "injury": None,
                "next_game": {"event_id": "401", "date": self.game_date, "name": "LAL @ OKC", "status": "pre",
                              "home": False, "opponent": "OKC", "opponent_name": "Oklahoma City Thunder"},
                "note": None, "season_label": "2026-27", "season": {"pts": 26.5}, "last5": {}, "last10": {},
                "sample": {"season": 10, "last5": 5, "last10": 10}, "games": ROWS, "splits": {"home": None, "away": None},
                "news": [], "missing": [], "as_of": "2026-10-26T12:00:00Z", "reason": None}

    def gamelog(self, league, athlete_id, last=10):
        return {"rows": ROWS[:last]}

    def props(self, league, event_id):
        return {"event_id": event_id, "league": league, "book": "DraftKings", "count": 2, "as_of": "2026-10-26T12:00:00Z",
                "stale": False, "reason": None, "players": {"1966": [
                    {"market": "player_points", "stat": "pts", "name": "Total Points", "kind": "over_under",
                     "line": 25.5, "open_line": 25.5, "over_price": Decimal("1.87"), "under_price": Decimal("1.95"),
                     "yes_price": None, "sides_known": True, "label": "25.5", "updated": None},
                    {"market": "player_points_rebounds", "stat": "pts+reb", "name": "Total Points and Rebounds",
                     "kind": "over_under", "line": 33.5, "open_line": None, "over_price": None, "under_price": None,
                     "yes_price": None, "sides_known": False, "label": "33.5", "updated": None},
                ]}}


class Assistant:
    """The AI assistant's interface, recording what it was asked."""

    def __init__(self, refuse: Optional[str] = None) -> None:
        self.refuse = refuse
        self.explained: List[Any] = []
        self.replies: List[Any] = []

    def explain(self, uid, fact_sheet):
        if self.refuse:
            raise CallRejected(self.refuse)
        self.explained.append((uid, fact_sheet))
        return {"verdict": "A long shot.", "per_leg": [{"index": 0, "text": "Short price."}],
                "case_against": "Two legs double the cut.", "model": "claude-opus-5-5", "cost": 0.012}

    def reply(self, uid, thread_id=None, message=None, request=None, slip=None):
        self.replies.append({"uid": uid, "thread_id": thread_id, "message": message, "request": request, "slip": slip})
        return {"thread_id": thread_id or 7, "reply": {"text": "Here you go.", "cards": [], "chips": ["More"]},
                "credits": self.credits(uid), "disclosure": "You're chatting with an AI."}

    def credits(self, uid):
        return {"used_cents": 3.0, "daily_cents": 100, "left_cents": 97.0}


# --- fixtures ---------------------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    return Store(str(tmp_path / "betting.db"))


@pytest.fixture()
def clock():
    return Clock()


@pytest.fixture()
def lab(service, store, clock):
    service.odds = Odds()
    service.store = store
    return Lab(service, store, players=Players(), public_url="https://bets.example.com", now=clock)


@pytest.fixture()
def ready(lab):
    """The lab with the browser user's age confirmed."""
    lab.update_me(UID, {"confirm_age": True})
    return lab


@pytest.fixture()
def client(lab):
    client = TestClient(create_app(lab.service, "s3cret", login_delay=0))
    assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
    client.headers.update(ME)
    return client


def ref(event_id: str, market: str = "h2h", side: str = "home", **extra: Any) -> Dict[str, Any]:
    return {"league": NBA, "id": event_id, "market": market, "side": side, **extra}


def set_leg_starts(store: Store, start: str = "2020-01-01T00:00:00Z") -> None:
    """Pretend every saved bet's games started long ago (so the grader looks at them)."""
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE slip_legs SET start = ?", (start,))


def double_tap(lab: Lab, store_method: str, first, second):
    """Run two calls side by side, as a double tap does: the first is held inside the store's write until the
    second has had time to run. Returns (first's result or error, second's result or error, whether the second
    had to wait for the first)."""
    entered, release = threading.Event(), threading.Event()
    original = getattr(lab.store, store_method)

    def held(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            release.wait(5)
        return original(*args, **kwargs)

    out: Dict[str, Any] = {}

    def run(name, call):
        try:
            out[name] = call()
        except Exception as exc:  # the refusal is the result
            out[name] = exc

    setattr(lab.store, store_method, held)
    try:
        a = threading.Thread(target=run, args=("first", first))
        a.start()
        assert entered.wait(5)
        b = threading.Thread(target=run, args=("second", second))
        b.start()
        b.join(0.3)
        waited = b.is_alive()
        release.set()
        a.join(5)
        b.join(5)
    finally:
        release.set()
        delattr(lab.store, store_method)
    return out["first"], out["second"], waited


# --- who is asking ------------------------------------------------------------------------------------

def test_person_is_the_telegram_user_or_the_browsers_own_id(lab) -> None:
    app = create_app(lab.service, "s3cret", login_delay=0, bot_token=BOT_TOKEN, telegram_ids=(42,))
    client = TestClient(app)
    assert client.get("/api/me").status_code == 401
    client.post("/api/login", json={"password": "s3cret"})
    for headers in ({}, {"X-Player": "short"}, {"X-Player": "has spaces in it"}, {"X-Player": "x" * 65}):
        response = client.get("/api/me", headers=headers)
        assert response.status_code == 400 and response.json()["detail"] == "Missing player id"
    assert client.get("/api/me", headers=ME).json()["uid"] == UID
    # Two browsers on the shared password are two people with their own settings
    client.put("/api/me", headers=ME, json={"name": "Sam"})
    other = client.get("/api/me", headers={"X-Player": "other-browser-99"}).json()
    assert other["uid"] == "web:other-browser-99" and other["name"] is None
    # Inside Telegram the person is their Telegram id, whatever X-Player says
    telegram = {INIT_DATA_HEADER: make_init_data({"id": 42, "first_name": "Kim"}, auth_date=int(time.time()))}
    me = TestClient(app).get("/api/me", headers={**telegram, **ME}).json()
    assert me["uid"] == "tg:42" and me["is_telegram"]


def test_me_settings_budget_and_break(client, clock) -> None:
    me = client.get("/api/me").json()
    assert me["age_confirmed"] is False and me["budget"] is None and me["odds_format"] == "decimal"
    assert me["credits"] == {"used_cents": 0.0, "daily_cents": 100, "left_cents": 100.0}
    assert me["public_url"] == "https://bets.example.com" and me["ai_enabled"] is False
    me = client.put("/api/me", json={"confirm_age": True, "odds_format": "american",
                                     "favourite_sports": [NHL, NBA], "risk_tier": "safe", "weekly_budget": 20}).json()
    assert me["age_confirmed"] and me["odds_format"] == "american" and me["favourite_sports"] == [NHL, NBA]
    assert me["budget"]["weekly"] == "20.00" and me["budget"]["left"] == "20.00" and me["budget"]["pending"] is None
    # Raising the budget waits 24 hours; lowering applies at once
    me = client.put("/api/me", json={"weekly_budget": 50}).json()
    assert me["budget"]["weekly"] == "20.00" and me["budget"]["pending"]["amount"] == "50.00"
    clock.advance(25)
    assert client.get("/api/me").json()["budget"]["weekly"] == "50.00"
    assert client.put("/api/me", json={"weekly_budget": 10}).json()["budget"]["weekly"] == "10.00"
    # Only the fields sent change; null clears the optional ones
    me = client.put("/api/me", json={"risk_tier": None}).json()
    assert me["risk_tier"] is None and me["odds_format"] == "american"
    for bad in ({"odds_format": "roman"}, {"favourite_sports": ["NBA!"]}, {"weekly_budget": -1},
                {"reality_check_minutes": 45}, {"favourite_sports": [NBA] * 9}):
        assert client.put("/api/me", json=bad).status_code == 422, bad

    # A break switches picks off until it ends, and can't be cut short
    assert client.post("/api/me/cooloff", json={"hours": 5}).status_code == 422
    me = client.post("/api/me/cooloff", json={"hours": 24}).json()
    assert me["on_break"] and me["cooloff_until"]
    response = client.post("/api/tickets/price", json={"type": "single", "stake": 5, "legs": [ref("g1")]})
    assert response.status_code == 400 and response.json()["detail"].startswith("You're on a break until")
    assert client.get("/api/today").status_code == 400
    assert client.get("/api/slips").status_code == 200  # history stays readable
    clock.advance(25)
    assert client.post("/api/tickets/price", json={"type": "single", "stake": 5, "legs": [ref("g1")]}).status_code == 200


def test_nothing_betting_related_before_the_age_is_confirmed(client) -> None:
    for method, path, body in (("post", "/api/tickets/price", {"stake": 5, "legs": [ref("g1")]}),
                               ("post", "/api/tickets/analyse", {"stake": 5, "legs": [ref("g1")]}),
                               ("post", "/api/tickets/build", {"stake": 5, "target": 20}),
                               ("post", "/api/slips", {"stake": 5, "legs": [ref("g1")]}),
                               ("post", "/api/plans", {"start": 10, "days": 3, "min_odds": 2}),
                               ("get", "/api/today", None), ("get", f"/api/players/{NBA}/1966", None)):
        response = client.request(method, path, json=body)
        assert response.status_code == 400 and response.json()["detail"] == AGE_MESSAGE, path
    assert client.get("/api/sports").json()[0] == {"key": NBA, "title": "NBA", "group": "Basketball",
                                                   "label": "🏀 Basketball", "ai": True}


def test_a_warning_sign_in_the_chat_pauses_picks_in_the_app_too(ready, client, clock) -> None:
    # The chat stores the 24-hour pause in the person's settings; building and plans honour it
    until = clock() + 24 * 3600
    ready.store.update_user(UID, {"settings": {lab_module.PAUSE_SETTING: until}})
    assert client.get("/api/me").json()["picks_paused_until"]
    for path, body in (("/api/tickets/build", {"stake": 5, "target": 20}),
                       ("/api/plans", {"start": 10, "days": 3, "min_odds": 2})):
        response = client.post(path, json=body)
        assert response.status_code == 400 and response.json()["detail"].startswith("Picks are paused until"), path
    # The plain maths on the person's own slip stays open
    assert client.post("/api/tickets/price", json={"stake": 5, "legs": [ref("g1")]}).status_code == 200
    clock.advance(24.01)
    assert client.get("/api/me").json()["picks_paused_until"] is None
    assert client.post("/api/tickets/build", json={"stake": 5, "target": 20}).status_code == 200


def test_the_leaderboard_never_shows_another_persons_id(lab, client) -> None:
    # On the shared password a browser's id is the whole identity, so it must never be shown to anyone else
    for player, name in (("web:victim-browser-1", "Vic"), (UID, "Me")):
        call_id = lab.store.save_prediction({"player": player, "name": name, "league": NBA, "event_id": "g1",
                                             "espn_id": None, "home": "A", "away": "B", "start": "2020-01-01T00:00:00Z",
                                             "call": {"winner": "home"}, "ai": {}})
        lab.store.grade_prediction(call_id, "graded", {"parts": []}, 1.0, 0.5)
    rows = client.get("/api/leaderboard").json()
    assert "victim-browser-1" not in str(rows)
    assert {row["name"]: (row["mine"], row["player"]) for row in rows} == {"Vic": (False, None), "Me": (True, UID)}
    # Still open without a browser id, with nothing marked
    rows = client.get("/api/leaderboard", headers={"X-Player": ""}).json()
    assert "web:" not in str(rows) and not any(row["mine"] for row in rows)


# --- the board -------------------------------------------------------------------------------------

def test_a_pick_is_looked_up_on_the_cached_board(ready) -> None:
    leg = ready.leg_from_ref(ref("g1", "spreads", "home"))
    assert leg.price == Decimal("1.91") and leg.point == -13.5 and leg.group == "g1" and leg.book
    assert leg.label == "Oklahoma City Thunder -13.5" and leg.game == "Los Angeles Clippers @ Oklahoma City Thunder"
    assert leg.key == "basketball_nba:g1:spreads:home:-13.5"
    assert 0 < leg.ai_prob < 1 and 0 < leg.market_prob < 1
    assert ready.leg_from_ref(ref("gone")) is None
    assert ready.leg_from_ref(ref("g1", "totals", "draw")) is None
    assert ready.leg_from_ref({**ref("g1"), "league": "baseball_mlb"}) is None  # out of season: never fetched
    assert "baseball_mlb" not in ready.service.odds.fetched
    with pytest.raises(CallRejected):
        ready.leg_from_ref({"league": NBA, "id": "g1", "market": "props", "side": "home"})


def test_today_loads_only_the_leagues_asked_for(ready, client) -> None:
    odds = ready.service.odds
    data = client.get("/api/today").json()  # no favourites yet: the NBA
    assert data["leagues"] == [NBA] and odds.fetched and set(odds.fetched) == {NBA}
    assert [g["id"] for g in data["games"]] == ["g1", "g2", "g3", "g4", "g5", "g6"]
    assert data["games"][0]["league_title"] == "NBA" and data["quota"]["remaining"] == 480
    data = client.get("/api/today", params={"leagues": f"{NHL},{NBA},baseball_mlb"}).json()
    assert [g["id"] for g in data["games"]][:3] == ["g1", "h1", "g2"]  # soonest first across leagues
    assert data["errors"] == {"baseball_mlb": "Not in season right now, so there are no odds to show."}
    assert "baseball_mlb" not in odds.fetched
    # Favourites are the default; a day keeps only that day's games (local time)
    client.put("/api/me", json={"favourite_sports": [NHL]})
    assert [g["id"] for g in client.get("/api/today").json()["games"]] == ["h1"]
    day = client.get("/api/today", params={"leagues": NBA, "day": "2029-12-31"}).json()
    assert [g["id"] for g in day["games"]] == ["g1", "g2"]
    assert client.get("/api/today", params={"leagues": "NBA"}).status_code == 422
    assert client.get("/api/today", params={"leagues": ",".join([NBA] * 9)}).status_code == 422


def test_a_league_that_fails_says_why_without_leaking_the_request(ready, caplog) -> None:
    def broken(league, refresh=False):
        raise ConnectionError("HTTPSConnectionPool: /v4/sports/x/odds?apiKey=SECRET")

    ready.service.odds.league_games = broken
    data = ready.today(UID, [NBA])
    assert data["games"] == [] and "SECRET" not in data["errors"][NBA]
    # Pricing a pick on that league is a plain 502 with a safe reason, and the log has no request URL
    with pytest.raises(RuntimeError) as failed:
        ready.price(UID, {"type": "single", "stake": 1, "legs": [ref("g1")]})
    assert "Couldn't load this league" in str(failed.value) and "SECRET" not in str(failed.value)
    assert "SECRET" not in caplog.text


# --- pricing ---------------------------------------------------------------------------------------

def test_tickets_are_priced_on_the_server_from_picks_only(ready, client) -> None:
    body = {"type": "multi", "stake": 10, "legs": [ref("g1", price=100.0), ref("g2", side="away")]}
    priced = client.post("/api/tickets/price", json=body).json()
    # Each leg at its best price (a price sent by the app is ignored)
    assert [leg["odds"]["decimal"] for leg in priced["legs"]] == [1.11, 3.6]
    assert priced["odds"]["decimal"] == round(1.11 * 3.6, 2) and priced["max_payout"] == "39.96"
    assert priced["valid"] and priced["removed"] == [] and priced["bets"] == 1 and priced["total_stake"] == "10.00"
    assert priced["refs"] == [ref("g1", point=None), ref("g2", side="away", point=None)]
    assert priced["chance"]["ai_one_in"] and 0 < priced["book_cut"] < 0.2
    assert [leg["book"] for leg in priced["legs"]] == ["FanDuel", "FanDuel"]
    assert not any("different bookmakers" in w for w in priced["warnings"])
    # Best prices at two bookmakers: said plainly, since one bookmaker's ticket would pay less
    split = client.post("/api/tickets/price", json={"stake": 10, "legs": [ref("g1"), ref("g2")]}).json()
    assert [leg["book"] for leg in split["legs"]] == ["FanDuel", "Bovada"]
    assert any("different bookmakers (Bovada, FanDuel)" in w for w in split["warnings"])

    gone = client.post("/api/tickets/price", json={"stake": 10, "legs": [ref("g1"), ref("nope")]}).json()
    assert gone["removed"] == [ref("nope", point=None)] and not gone["valid"] and "no longer offered" in gone["reasons"][-1]
    moved = client.post("/api/tickets/price", json={"type": "single", "stake": 10,
                                                    "legs": [ref("g1", "spreads", point=-10.5)]}).json()
    assert moved["moved"][0]["was"] == -10.5 and moved["moved"][0]["now"] == -13.5
    assert any("line moved" in w for w in moved["warnings"])
    same_game = client.post("/api/tickets/price", json={"stake": 10, "legs": [ref("g1"), ref("g1", "totals", "over")]})
    assert not same_game.json()["valid"] and "same game" in same_game.json()["reasons"][0]

    robin = client.post("/api/tickets/price", json={"type": "round_robin", "k": 2, "stake": 5,
                                                   "legs": [ref("g1"), ref("g2"), ref("g3")]}).json()
    assert robin["bets"] == 3 and robin["total_stake"] == "15.00" and robin["odds"] is None
    trixie = client.post("/api/tickets/price", json={"type": "system", "system": "trixie", "stake": 1,
                                                    "legs": [ref("g1"), ref("g2"), ref("g3")]}).json()
    assert trixie["bets"] == 4 and trixie["total_stake"] == "4.00"
    unknown = client.post("/api/tickets/price", json={"type": "system", "system": "nonsense", "stake": 1,
                                                     "legs": [ref("g1"), ref("g2"), ref("g3")]})
    assert unknown.status_code == 400 and "Unknown system" in unknown.json()["detail"]
    for bad in ({"stake": 10.005, "legs": [ref("g1")]}, {"stake": 0, "legs": [ref("g1")]},
                {"stake": 10, "legs": []}, {"stake": 10, "legs": [ref("g1")] * 13},
                {"stake": 10, "legs": [ref("g1", market="props")]}, {"stake": 10, "k": 9, "legs": [ref("g1")]},
                {"stake": 10, "legs": [{**ref("g1"), "id": "../etc"}]}):
        assert client.post("/api/tickets/price", json=bad).status_code == 422, bad


def test_analysis_without_an_assistant_is_complete_in_code(ready, client) -> None:
    legs = [ref("g1"), ref("g2"), ref("g3"), ref("g4", side="away")]
    data = client.post("/api/tickets/analyse", json={"type": "multi", "stake": 12, "legs": legs}).json()
    assert data["explanation"] is None and data["credits"]["daily_cents"] == 100
    ticket = data["ticket"]
    assert ticket["valid"] and ticket["weakest_leg"] == 3  # the Jazz at 4.30
    notes = data["legs"][0]["notes"]
    assert data["legs"][0]["label"] == "Oklahoma City Thunder to win"
    assert any("Favourite" in n for n in notes) and any("bookmaker keeps" in n for n in notes)
    assert any("market" in n for n in notes)
    titles = [alt["title"] for alt in data["alternatives"]]
    assert titles == ["Drop the weakest leg (Utah Jazz to win)", "Split into 4 singles of $3",
                      "Round robin by 2s: 6 bets of $2", "Round robin by 3s: 4 bets of $3"]
    drop, singles, by_two, by_three = data["alternatives"]
    assert drop["ticket"]["type"] == "multi" and len(drop["refs"]) == 3 and drop["ticket"]["total_stake"] == "12.00"
    assert singles["ticket"]["bets"] == 4 and singles["ticket"]["total_stake"] == "12.00"
    assert by_two["ticket"]["bets"] == 6 and by_two["ticket"]["k"] == 2 and by_two["ticket"]["total_stake"] == "12.00"
    assert by_three["ticket"]["total_stake"] == "12.00"
    # Spreading the same money raises the chance of getting a profit
    assert by_two["ticket"]["chance"]["profit"]["ai"] > ticket["chance"]["profit"]["ai"]
    assert by_two["summary"].startswith("Chance of a profit") and "for your ticket" in by_two["summary"]
    # The alternatives' refs price back to the same legs
    again = client.post("/api/tickets/price", json={"type": "multi", "stake": 12, "legs": drop["refs"]}).json()
    assert again["max_payout"] == drop["ticket"]["max_payout"]


def test_the_assistant_explains_numbers_from_the_fact_sheet_only(ready, lab) -> None:
    lab.assistant = Assistant()
    body = {"type": "multi", "stake": 10, "legs": [ref("g1"), ref("g2", side="away")]}
    data = lab.analyse(UID, body)
    (uid, sheet), = lab.assistant.explained
    assert uid == UID and sheet["currency"] == "$"
    assert sheet["ticket"]["total_stake"] == "10.00" and sheet["ticket"]["odds"] == data["ticket"]["odds"]
    assert [leg["label"] for leg in sheet["legs"]] == ["Oklahoma City Thunder to win", "Orlando Magic to win"]
    assert sheet["legs"][0]["notes"] == data["legs"][0]["notes"]
    assert [alt["title"] for alt in sheet["alternatives"]] == [alt["title"] for alt in data["alternatives"]]
    assert data["explanation"]["verdict"] == "A long shot." and data["explanation"]["model"] == "claude-opus-5-5"
    assert data["explanation"]["alternatives"] == data["alternatives"]
    assert data["credits"] == {"used_cents": 3.0, "daily_cents": 100, "left_cents": 97.0}
    # The chat asks for the analysis as a tool and writes its own words
    lab.analyse(UID, body, explain=False)
    assert len(lab.assistant.explained) == 1
    # An invalid ticket isn't sent to Claude
    lab.analyse(UID, {"stake": 10, "legs": [ref("g1"), ref("g1", "totals", "over")]})
    assert len(lab.assistant.explained) == 1
    # Credits used up: the code's analysis still comes back
    lab.assistant = Assistant(refuse="You've used today's AI credits")
    data = lab.analyse(UID, body)
    assert data["explanation"] is None and data["note"] == "You've used today's AI credits"
    assert data["ticket"]["valid"] and data["alternatives"]


def test_build_returns_addable_tickets_within_the_budget(ready, client) -> None:
    built = client.post("/api/tickets/build", json={"stake": 10, "target": 30, "max_legs": 3}).json()
    assert built["leagues"] == [NBA] and built["multiple"] == 3.0 and built["verdict"]
    assert built["tickets"], built["notes"]
    for ticket in built["tickets"]:
        assert ticket["total_stake"] == "10.00"  # never a bigger stake
        priced = client.post("/api/tickets/price", json={"type": "multi", "stake": 10, "legs": ticket["refs"]}).json()
        assert priced["odds"] == ticket["odds"]
    client.put("/api/me", json={"weekly_budget": 5})
    response = client.post("/api/tickets/build", json={"stake": 10, "target": 30})
    assert response.status_code == 400 and "$5 until Monday" in response.json()["detail"]
    assert client.post("/api/tickets/build", json={"stake": 1, "target": 2, "max_legs": 7}).status_code == 422


# --- saved bets ------------------------------------------------------------------------------------

def test_slips_keep_exact_quotes_share_codes_and_the_budget(ready, client, lab) -> None:
    client.put("/api/me", json={"weekly_budget": 25})
    body = {"type": "multi", "stake": 10, "legs": [ref("g1", "spreads"), ref("g2")]}
    slip = client.post("/api/slips", json=body).json()
    assert slip["status"] == "saved" and slip["stake"] == "10.00" and len(slip["share_code"]) == 8
    assert [leg["price"] for leg in slip["legs"]] == ["1.91", "1.36"]  # exact, as quoted
    assert slip["odds"]["decimal"] == round(1.91 * 1.36, 2) and slip["refs"][0]["point"] == -13.5
    assert client.get("/api/me").json()["budget"]["left"] == "15.00"
    over = client.post("/api/slips", json={**body, "stake": 16})
    assert over.status_code == 400 and over.json()["detail"] == (
        "This ticket of $16 is more than your weekly budget has left: $15 until Monday.")
    assert client.post("/api/slips", json={**body, "legs": [ref("g1"), ref("nope")]}).status_code == 400
    same_game = client.post("/api/slips", json={**body, "legs": [ref("g1"), ref("g1", "totals", "over")]})
    assert same_game.status_code == 400 and "same game" in same_game.json()["detail"]

    assert [s["id"] for s in client.get("/api/slips").json()] == [slip["id"]]
    assert client.get(f"/api/slips/{slip['id']}").json()["share_code"] == slip["share_code"]
    # Anyone signed in can open a share code; it comes back as a slip to price again
    friend = {"X-Player": "friend-browser"}
    shared = client.get(f"/api/slips/share/{slip['share_code'].lower()}", headers=friend).json()
    assert shared == {"type": "multi", "stake": 10.0, "k": None, "system": None, "legs": slip["refs"],
                      "labels": ["Oklahoma City Thunder -13.5", "New York Knicks to win"],
                      "share_code": slip["share_code"], "mine": False}
    assert client.get(f"/api/slips/{slip['id']}", headers=friend).status_code == 404
    assert client.get("/api/slips/share/NOPE2345").status_code == 404
    # A bet can be deleted until its games start
    assert client.delete(f"/api/slips/{slip['id']}", headers=friend).status_code == 404
    second = client.post("/api/slips", json={**body, "stake": 5}).json()
    set_leg_starts(lab.store)
    response = client.delete(f"/api/slips/{second['id']}")
    assert response.status_code == 400 and "started" in response.json()["detail"]
    with sqlite3.connect(lab.store.path) as db:
        db.execute("UPDATE slip_legs SET start = '2030-01-01T00:00:00Z'")
    assert client.delete(f"/api/slips/{second['id']}").json() == {"ok": True}


def test_saved_bets_are_graded_when_their_games_finish(ready, lab) -> None:
    multi = lab.save_slip(UID, {"type": "multi", "stake": 10, "legs": [ref("g1", "spreads"), ref("g2")]})
    lost = lab.save_slip(UID, {"type": "multi", "stake": 4, "legs": [ref("g3"), ref("g4", side="away")]})
    singles = lab.save_slip(UID, {"type": "single", "stake": 2, "legs": [ref("g5"), ref("g6", "totals", "over")]})
    service = lab.service
    service.espn = Results({"Oklahoma City Thunder": (120, 100), "New York Knicks": (101, 99),
                            "Boston Celtics": (110, 100), "Denver Nuggets": (115, 90),
                            "Phoenix Suns": "void", "Chicago Bulls": "pending"})
    assert service.grade_due() == 0  # nothing has started yet
    set_leg_starts(lab.store)
    three_hours_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=3)
    with sqlite3.connect(lab.store.path) as db:  # the Bulls game started recently and is still going
        db.execute("UPDATE slip_legs SET start = ? WHERE event_id = 'g6'", (three_hours_ago.strftime("%Y-%m-%dT%H:%M:%SZ"),))
    assert service.grade_due() == 2  # the multis; the singles wait for the Bulls game
    won = lab.get_slip(UID, multi["id"])
    assert won["status"] == "settled" and won["result"]["status"] == "won"
    assert won["result"]["returns"] == "25.98"  # 10 x 1.91 x 1.36, rounded once
    assert won["result"]["profit"] == "15.98"
    assert [leg["result"]["result"] for leg in won["legs"]] == ["won", "won"]
    assert won["legs"][0]["result"]["home"] == 120
    assert lab.get_slip(UID, lost["id"])["result"]["status"] == "lost"  # the Jazz lost
    waiting = lab.get_slip(UID, singles["id"])
    assert waiting["status"] == "saved" and waiting["legs"][0]["result"]["result"] == "void"
    assert waiting["legs"][1]["result"] is None
    service.espn.scores["Chicago Bulls"] = (100, 130)  # 230 points: the over wins
    assert service.grade_due() == 1
    settled = lab.get_slip(UID, singles["id"])
    # The postponed game's single comes back (2.00) and the over pays 2 x 1.91
    assert settled["result"]["status"] == "won" and settled["result"]["returns"] == "5.82"
    assert [s["status"] for s in lab.list_slips(UID, "settled")] == ["settled"] * 3


def test_a_bet_with_no_result_for_days_is_voided(ready, lab) -> None:
    slip = lab.save_slip(UID, {"type": "single", "stake": 3, "legs": [ref("g1")]})
    lab.service.espn = Results({})  # ESPN never finds the game
    set_leg_starts(lab.store, "2030-01-01T00:00:00Z")
    lab.grade_due(cutoff="2030-01-02T00:00:00Z", give_up="2029-12-01T00:00:00Z")
    assert lab.get_slip(UID, slip["id"])["status"] == "saved"  # still hoping for a result
    lab.grade_due(cutoff="2030-01-02T00:00:00Z", give_up="2030-01-05T00:00:00Z")
    voided = lab.get_slip(UID, slip["id"])
    assert voided["status"] == "void" and voided["result"]["returns"] == "3.00"
    assert voided["legs"][0]["result"] == {"result": "void", "reason": "no result"}


def test_a_draw_loses_a_three_way_moneyline(ready, lab) -> None:
    lab.service.odds = MoreOdds()
    arsenal = lab.save_slip(UID, {"type": "single", "stake": 10, "legs": [{"league": EPL, "id": "s1", "market": "h2h",
                                                                         "side": "home"}]})
    assert arsenal["legs"][0]["market_prob"] < 0.5 and arsenal["chance"]["ai"] is None  # priced over three outcomes
    draw = lab.save_slip(UID, {"type": "single", "stake": 5, "legs": [{"league": EPL, "id": "s1", "market": "h2h",
                                                                     "side": "draw"}]})
    lab.service.espn = Results({"Arsenal": (1, 1)})
    set_leg_starts(lab.store)
    assert lab.service.grade_due() == 2
    lost = lab.get_slip(UID, arsenal["id"])
    assert lost["status"] == "settled" and lost["result"]["status"] == "lost" and lost["result"]["returns"] == "0.00"
    assert lost["legs"][0]["result"]["result"] == "lost"
    assert lab.get_slip(UID, draw["id"])["result"]["returns"] == "17.00"  # 5 x 3.40
    # Bets saved before the market's shape was recorded go by the league; a two-way moneyline still pushes
    home = {"league": EPL, "market": "h2h", "side": "home", "point": None}
    assert lab_module._outcome(home, {"home": 1, "away": 1}, {}) == "lost"
    assert lab_module._outcome({**home, "league": NBA}, {"home": 1, "away": 1}, {"three_way": False}) == "push"
    assert lab_module._outcome(home, {"home": 2, "away": 1}, {"three_way": True}) == "won"


def test_leagues_the_grader_cant_settle_are_priced_but_not_saved(ready, client, lab) -> None:
    lab.service.odds = MoreOdds()
    body = {"type": "single", "stake": 10, "legs": [{"league": MMA, "id": "m1", "market": "h2h", "side": "home"}]}
    priced = client.post("/api/tickets/price", json=body).json()
    assert priced["valid"] and "MMA bets can't be settled automatically yet" in " ".join(priced["warnings"])
    refused = client.post("/api/slips", json={**body, "legs": [*body["legs"], ref("g1")]})
    assert refused.status_code == 400 and refused.json()["detail"].startswith("MMA bets can't be settled")
    assert client.get("/api/slips").json() == []
    created = client.post("/api/plans", json={"start": 10, "days": 3, "min_odds": 1.5}).json()
    day = client.post(f"/api/plans/{created['id']}/day", json={"legs": body["legs"]})
    assert day.status_code == 400 and "MMA bets can't be settled" in day.json()["detail"]


def test_a_moved_line_is_saved_only_once_the_person_has_seen_it(ready, client) -> None:
    tapped = {"type": "single", "stake": 10, "legs": [ref("g3", "spreads", "home", point=-3.5)]}  # now -4.5
    priced = client.post("/api/tickets/price", json=tapped).json()
    assert priced["moved"][0]["now"] == -4.5 and priced["refs"][0]["point"] == -4.5
    refused = client.post("/api/slips", json=tapped)
    assert refused.status_code == 400
    assert "Boston Celtics -4.5 now (it was -3.5)" in refused.json()["detail"]
    assert client.get("/api/slips").json() == []
    saved = client.post("/api/slips", json={**tapped, "legs": priced["refs"]}).json()  # the line they now see
    assert saved["legs"][0]["label"] == "Boston Celtics -4.5"


def test_a_double_tap_cant_pass_the_budget_twice(ready, lab) -> None:
    lab.update_me(UID, {"weekly_budget": "10"})
    first, second, waited = double_tap(
        lab, "save_slip",
        lambda: lab.save_slip(UID, {"type": "single", "stake": 10, "legs": [ref("g3", side="away")]}),
        lambda: lab.save_slip(UID, {"type": "single", "stake": 10, "legs": [ref("g4", side="away")]}))
    assert waited and first["status"] == "saved"
    assert isinstance(second, CallRejected) and "$0 until Monday" in str(second)
    assert lab.me(UID)["budget"]["used"] == "10.00"


# --- rollover plans ----------------------------------------------------------------------------------

def plan(strategy: str, start: int = 1000, days: int = 5) -> Dict[str, Any]:
    return {"start_cents": start, "strategy": strategy, "days": days, "days_played": []}


def test_each_strategy_stakes_from_the_plans_own_money() -> None:
    # Classic: everything, every day; one loss ends it
    run = Run(bankroll=1000)
    assert next_stake(plan("classic"), run) == 1000
    run, banked, rolled = play_day(plan("classic"), run, 1000, 2000)
    assert (run.bankroll, banked, rolled, next_stake(plan("classic"), run)) == (2000, 0, 2000, 2000)
    assert play_day(plan("classic"), run, 2000, 0)[0].ended == "lost"
    # Bank half: the start on day 1, then half of each day's profit is banked
    run = Run(bankroll=1000)
    assert next_stake(plan("bank_half"), run) == 1000
    run, banked, rolled = play_day(plan("bank_half"), run, 1000, 1910)
    assert (banked, rolled, run.bankroll, next_stake(plan("bank_half"), run)) == (455, 1455, 1910, 1455)
    lost = play_day(plan("bank_half"), run, 1455, 0)[0]
    assert lost.ended == "lost" and lost.bankroll == 455 == lost.banked
    # Pot and ladder: a 10% pot rolls for two wins, is banked, and a new pot starts
    run = Run(bankroll=1000)
    assert next_stake(plan("ladder"), run) == 100
    run, _, rolled = play_day(plan("ladder"), run, 100, 200)
    assert rolled == 200 and next_stake(plan("ladder"), run) == 200
    run, banked, rolled = play_day(plan("ladder"), run, 200, 400)
    assert (banked, rolled, run.bankroll, next_stake(plan("ladder"), run)) == (400, 0, 1300, 100)
    push = play_day(plan("ladder"), run, 100, 100)[0]
    assert push.bankroll == 1300 and next_stake(plan("ladder"), push) == 100 and not push.ended
    # It stops once the bankroll is below half the start
    low = Run(bankroll=550)
    assert play_day(plan("ladder"), low, 100, 0)[0].ended == "below_half"
    # Flat: 5% of the start every day, winnings kept
    run = Run(bankroll=1000)
    assert next_stake(plan("flat"), run) == 50
    run, banked, rolled = play_day(plan("flat"), run, 50, 100)
    assert (run.bankroll, banked, rolled, next_stake(plan("flat"), run)) == (1050, 0, 0, 50)
    assert play_day(plan("flat"), Run(bankroll=60), 50, 0)[0].ended == "broke"
    # The state is always worked out again from each day's stake and result
    played = {**plan("ladder"), "days_played": [
        {"day": 1, "stake_cents": 100, "result": {"returns": "2.00"}},
        {"day": 2, "stake_cents": 200, "result": {"returns": "4.00"}},
        {"day": 3, "stake_cents": 100, "result": None}]}
    assert replay(played) == Run(bankroll=1300, banked=400, rolled=0, pot_wins=0, days=2)


def test_rollover_compare_shows_every_strategy_and_the_honest_sentences(client, ready) -> None:
    data = client.post("/api/rollover/compare", json={"start": 10, "days": 10, "min_odds": 2, "legs_per_day": 1}).json()
    assert set(data["strategies"]) == {"classic", "bank_half", "ladder", "flat"}
    assert data["sentences"]["summary"].startswith("To turn $10 into $10,240 in 10 days")
    assert data["sentences"]["cost"]["classic"].startswith("On average a run returns")
    for bad in ({"start": 10, "days": 61, "min_odds": 2}, {"start": 10, "days": 5, "min_odds": 1.1},
                {"start": 10, "days": 5, "min_odds": 2, "legs_per_day": 4}):
        assert client.post("/api/rollover/compare", json=bad).status_code == 422, bad


def test_a_ladder_plan_runs_one_day_at_a_time_and_finishes(ready, client, lab) -> None:
    created = client.post("/api/plans", json={"start": 10, "days": 3, "min_odds": 1.5}).json()
    assert created["strategy"] == "ladder" and created["mode"] == "paper" and created["status"] == "active"
    assert created["next"] == {"day": 1, "stake": "1.00", "min_odds": "1.5", "max_legs": 1}
    assert created["comparison"]["sentences"]["summary"].startswith("To turn $10")
    assert "Nothing is placed until you choose" in created["message"]
    plan_id = created["id"]
    # One plan at a time
    again = client.post("/api/plans", json={"start": 10, "days": 3, "min_odds": 1.5})
    assert again.status_code == 400 and "already have a plan" in again.json()["detail"]
    # A day's pick must reach the plan's minimum odds, with no more legs than the plan takes
    short = client.post(f"/api/plans/{plan_id}/day", json={"legs": [ref("g1")]})
    assert short.status_code == 400 and "needs odds of 1.5 or more" in short.json()["detail"]
    two = client.post(f"/api/plans/{plan_id}/day", json={"legs": [ref("g1", "spreads"), ref("g2", "spreads")]})
    assert two.status_code == 400 and "1 pick" in two.json()["detail"]

    day1 = client.post(f"/api/plans/{plan_id}/day", json={"legs": [ref("g1", "spreads")]}).json()
    assert day1["day"] == 1 and day1["in_play"] == "1.00" and day1["next"] is None
    slip = day1["days_played"][0]["slip"]
    assert slip["stake"] == "1.00" and slip["plan"] == {"id": plan_id, "day": 1, "mode": "paper", "strategy": "ladder"}
    blocked = client.post(f"/api/plans/{plan_id}/day", json={"legs": [ref("g2", "spreads")]})
    assert blocked.status_code == 400 and "still in play" in blocked.json()["detail"]
    # A plan's day stays with the plan
    assert client.delete(f"/api/slips/{slip['id']}").status_code == 400

    lab.service.espn = Results({"Oklahoma City Thunder": (130, 100), "New York Knicks": (120, 100),
                                "Boston Celtics": (110, 100)})
    set_leg_starts(lab.store)
    lab.service.grade_due()
    after1 = client.get(f"/api/plans/{plan_id}").json()
    assert after1["days_played"][0]["result"]["status"] == "won" and after1["bankroll"] == "10.91"
    assert after1["next"]["stake"] == "1.91"  # the pot rolls on
    client.post(f"/api/plans/{plan_id}/day", json={"legs": [ref("g2", "spreads")]})
    set_leg_starts(lab.store)
    lab.service.grade_due()
    after2 = client.get(f"/api/plans/{plan_id}").json()
    assert after2["days_played"][1]["banked"] == "3.65" and after2["banked"] == "3.65"  # two wins: banked
    assert after2["bankroll"] == "12.65" and after2["next"]["stake"] == "1.00"  # and a new pot
    client.post(f"/api/plans/{plan_id}/day", json={"legs": [ref("g3", "spreads")]})
    set_leg_starts(lab.store)
    lab.service.grade_due()
    done = client.get(f"/api/plans/{plan_id}").json()
    assert done["status"] == "done" and done["bankroll"] == "13.56" and done["next"] is None
    assert done["message"].startswith("Finished: all 3 days played")
    assert [p["id"] for p in client.get("/api/plans").json()] == [plan_id]
    # Paper plans are play money: the weekly budget isn't touched
    assert client.put("/api/me", json={"weekly_budget": 1}).json()["budget"]["used"] == "0.00"


def test_a_failed_run_ends_the_plan_and_suggests_a_break(ready, client, lab, clock) -> None:
    created = client.post("/api/plans", json={"start": 10, "days": 5, "min_odds": 1.5, "strategy": "classic"}).json()
    assert created["next"]["stake"] == "10.00"
    client.post(f"/api/plans/{created['id']}/day", json={"legs": [ref("g1", "spreads")]})
    lab.service.espn = Results({"Oklahoma City Thunder": (100, 110)})  # the Thunder lose: -13.5 doesn't cover
    set_leg_starts(lab.store)
    lab.service.grade_due()
    failed = client.get(f"/api/plans/{created['id']}").json()
    assert failed["status"] == "failed" and failed["bankroll"] == "0.00" and failed["next"] is None
    assert failed["message"] == "Day 1 lost, so the run is over and the stake is gone."
    assert failed["suggestion"] == BREAK_SUGGESTION
    # No restart straight away
    response = client.post("/api/plans", json={"start": 10, "days": 5, "min_odds": 1.5, "strategy": "classic"})
    assert response.status_code == 400 and "take a break" in response.json()["detail"]
    assert client.post(f"/api/plans/{created['id']}/day", json={"legs": [ref("g2", "spreads")]}).status_code == 400
    assert lab.store.rg_events(UID) == []  # one failed run isn't a warning sign
    clock.advance(25)
    second = client.post("/api/plans", json={"start": 10, "days": 5, "min_odds": 1.5, "strategy": "classic"}).json()
    client.post(f"/api/plans/{second['id']}/day", json={"legs": [ref("g2", "spreads")]})
    lab.service.espn = Results({"New York Knicks": (100, 110)})
    set_leg_starts(lab.store)
    lab.service.grade_due()
    events = lab.store.rg_events(UID)
    assert [(e["kind"], e["action"]) for e in events] == [("failed_rollovers", "break_suggested")]


def test_tracked_plans_count_against_the_budget_and_can_be_stopped(ready, client) -> None:
    client.put("/api/me", json={"weekly_budget": 5})
    too_big = client.post("/api/plans", json={"start": 10, "days": 3, "min_odds": 1.5, "mode": "tracked"})
    assert too_big.status_code == 400 and "weekly budget has left" in too_big.json()["detail"]
    paper = client.post("/api/plans", json={"start": 10, "days": 3, "min_odds": 1.5})
    assert paper.status_code == 400 and "more than your weekly budget ($5)" in paper.json()["detail"]
    created = client.post("/api/plans", json={"start": 5, "days": 3, "min_odds": 1.5, "mode": "tracked",
                                              "strategy": "flat"}).json()
    played = client.post(f"/api/plans/{created['id']}/day", json={"legs": [ref("g1", "spreads")]}).json()
    assert played["in_play"] == "0.25" and client.get("/api/me").json()["budget"]["used"] == "0.25"
    stopped = client.post(f"/api/plans/{created['id']}/stop").json()
    assert stopped["status"] == "stopped" and stopped["message"].startswith("Stopped after 1 day")
    assert client.post(f"/api/plans/{created['id']}/stop").status_code == 400
    assert client.get("/api/plans/999").status_code == 404
    assert client.post("/api/plans", json={"start": 0.5, "days": 3, "min_odds": 1.5}).status_code == 400


def test_a_double_tap_plays_one_day_and_starts_one_plan(ready, lab) -> None:
    body = {"start": 10, "days": 5, "min_odds": "1.2", "legs_per_day": 1, "strategy": "classic"}
    first, second, waited = double_tap(lab, "create_plan", lambda: lab.create_plan(UID, body),
                                       lambda: lab.create_plan(UID, body))
    assert waited and first["status"] == "active"
    assert isinstance(second, CallRejected) and "already have a plan" in str(second)
    plan_id = first["id"]
    first, second, waited = double_tap(
        lab, "record_plan_day",
        lambda: lab.plan_day(UID, plan_id, {"legs": [ref("g3", side="away")]}),
        lambda: lab.plan_day(UID, plan_id, {"legs": [ref("g4", side="away")]}))
    assert waited and first["in_play"] == "10.00"
    assert isinstance(second, CallRejected) and "Day 1 is still in play" in str(second)
    assert [day["day"] for day in lab.store.plan(UID, plan_id)["days_played"]] == [1]
    assert len(lab.list_slips(UID)) == 1  # the refused day saved no bet


def test_a_winning_plan_can_stake_more_than_a_single_request(ready, lab) -> None:
    created = lab.create_plan(UID, {"start": 60000, "days": 3, "min_odds": "1.5", "strategy": "classic"})
    lab.plan_day(UID, created["id"], {"legs": [ref("g3", side="away")]})  # the Heat at 2.75
    lab.service.espn = Results({"Boston Celtics": (100, 110)})
    set_leg_starts(lab.store)
    lab.service.grade_due()
    assert lab.get_plan(UID, created["id"])["next"]["stake"] == "165000.00"  # past the $100,000 request cap
    day2 = lab.plan_day(UID, created["id"], {"legs": [ref("g5", side="away")]})
    assert day2["day"] == 2 and day2["in_play"] == "165000.00"


def test_stopping_a_plan_mid_day_still_brings_the_break(ready, client, lab, clock) -> None:
    body = {"start": 10, "days": 5, "min_odds": 1.5, "strategy": "classic"}
    created = client.post("/api/plans", json=body).json()
    client.post(f"/api/plans/{created['id']}/day", json={"legs": [ref("g1", "spreads")]})
    assert client.post(f"/api/plans/{created['id']}/stop").json()["status"] == "stopped"
    waiting = client.post("/api/plans", json=body)
    assert waiting.status_code == 400 and "Day 1 of the plan you stopped is still in play" in waiting.json()["detail"]
    lab.service.espn = Results({"Oklahoma City Thunder": (100, 110)})  # and the day loses
    set_leg_starts(lab.store)
    lab.service.grade_due()
    stopped = client.get(f"/api/plans/{created['id']}").json()
    assert stopped["status"] == "stopped" and stopped["bankroll"] == "0.00"
    assert stopped["message"] == "Day 1 lost, so the run is over and the stake is gone."
    assert stopped["suggestion"] == BREAK_SUGGESTION
    response = client.post("/api/plans", json=body)
    assert response.status_code == 400 and "take a break" in response.json()["detail"]
    assert lab.store.rg_events(UID) == []
    clock.advance(25)
    second = client.post("/api/plans", json=body).json()
    client.post(f"/api/plans/{second['id']}/day", json={"legs": [ref("g2", "spreads")]})
    lab.service.espn = Results({"New York Knicks": (100, 110)})
    set_leg_starts(lab.store)
    lab.service.grade_due()
    assert client.get(f"/api/plans/{second['id']}").json()["status"] == "failed"
    events = lab.store.rg_events(UID)  # the stopped run's loss counts as one of two failed runs this week
    assert [(e["kind"], e["action"], e["detail"]["plans"]) for e in events] == [
        ("failed_rollovers", "break_suggested", 2)]


# --- players ---------------------------------------------------------------------------------------

def test_player_search_and_page_with_hit_rates(ready, client) -> None:
    found = client.get("/api/players/search", params={"q": "lebron"}).json()
    assert found == [{"id": "1966", "name": "LeBron James", "team": "Los Angeles Lakers", "team_abbr": "LAL",
                      "position": "F", "league": NBA, "headshot": "https://a.espncdn.com/1966.png"}]
    page = client.get(f"/api/players/{NBA}/1966").json()
    assert page["name"] == "LeBron James" and len(page["games"]) == 10
    points, combo = page["props"]
    assert points["line"] == 25.5 and points["over"]["decimal"] == 1.87 and points["under"]["american"] == "-105"
    assert points["hit_rates"]["last10"] == "6/10" and points["hit_rates"]["last5"] == "3/5"
    assert combo["hit_rates"]["last10"] == "5/10" and combo["ai"] is None  # no measured spread for pts+reb
    assert page["projection"] is None and page["projection_note"] == "The AI projects players on game day."
    assert client.get(f"/api/players/{NBA}/77").status_code == 404
    assert client.get(f"/api/players/{NBA}/abc").status_code == 404


def test_on_game_day_the_model_projects_the_player_and_prices_each_line(ready, lab) -> None:
    today = datetime.datetime.fromtimestamp(lab._now(), ESPN_DAY_TZ).date()
    start = datetime.datetime.combine(today, datetime.time(22, 0), ESPN_DAY_TZ).astimezone(datetime.timezone.utc)
    start_text = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    lab.service.odds.games = [make_event("tonight", "Oklahoma City Thunder", "Los Angeles Lakers", start_text,
                                         {"DraftKings": (1.40, 3.00, -7.5, 1.91)})]
    lab._players = Players(game_date=start_text, team="LA Lakers")  # ESPN's short name still matches
    row = {"id": "1966", "name": "LeBron James", "minutes": 34.0, "pts": 27.0, "reb": 7.5, "ast": 8.0, "fg3m": 2.0,
           "stl": 1.0, "blk": 0.5, "tov": 3.0, "status": "Active", "out": False, "side": "away"}
    lab.service.detail = lambda league, event_id: (None, None, {"1966": row} if event_id == "tonight" else {})
    page = lab.player(NBA, "1966")
    projection = page["projection"]
    assert projection["event_id"] == "tonight" and projection["minutes"] == 34.0
    assert projection["stats"]["pts"]["mean"] == 27.0 and projection["stats"]["pra"]["mean"] == 42.5
    assert projection["stats"]["pts"]["low"] < 27 < projection["stats"]["pts"]["high"]
    over, under, push = calls.stat_probabilities("pts", 27.0, 25.5, 34.0)
    ai = page["props"][0]["ai"]
    assert ai["over"] == round(over, 4) and ai["under"] == round(under, 4) and push == 0
    assert ai["fair_over"] == round(1 / over, 2) and ai["edge_over"] == round(over * 1.87 - 1, 4)
    # Not on the board, or not projected (out): no projection, and a reason
    lab.service.detail = lambda league, event_id: (None, None, {})
    assert lab.player(NBA, "1966")["projection_note"].startswith("The model doesn't project this player today")
    assert lab.player("icehockey_nhl", "1966")["projection_note"] == "The AI projects NBA players for now."


# --- the AI chat -------------------------------------------------------------------------------------

def test_chat_needs_the_assistant_and_passes_requests_through(ready, client, lab) -> None:
    response = client.post("/api/chat", json={"message": "Who wins tonight?"})
    assert response.status_code == 400 and response.json()["detail"] == NO_CHAT
    assert NO_CHAT == "AI chat isn't set up: add ANTHROPIC_API_KEY on the server"
    lab.assistant = Assistant()
    assert client.get("/api/me").json()["ai_enabled"] is True
    reply = client.post("/api/chat", json={"message": "Build me something", "thread_id": 7,
                                           "request": {"leagues": [NBA], "stake": 10, "target": 50, "max_legs": 2}}).json()
    assert reply["reply"]["text"] == "Here you go." and reply["thread_id"] == 7
    sent = lab.assistant.replies[0]
    assert sent["uid"] == UID and sent["thread_id"] == 7 and sent["message"] == "Build me something"
    assert sent["request"]["leagues"] == [NBA] and Decimal(sent["request"]["stake"]) == 10 and sent["slip"] is None
    # The request bar may be partial (the assistant asks for what's missing)
    assert client.post("/api/chat", json={"request": {"leagues": [NBA], "stake": 10}}).status_code == 200
    assert lab.assistant.replies[-1]["request"] == {"leagues": [NBA], "stake": "10", "target": None,
                                                    "max_legs": None, "tier": None}
    assert client.post("/api/chat", json={"request": {"stake": 10.001}}).status_code == 422
    assert client.post("/api/chat", json={}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 2001}).status_code == 422
    # Threads and their messages are each person's own
    thread = lab.store.create_thread(UID, "Tonight")
    lab.store.add_message(UID, thread["id"], "user", "Who wins tonight?")
    assert [t["title"] for t in client.get("/api/chat/threads").json()] == ["Tonight"]
    assert [m["content"] for m in client.get(f"/api/chat/{thread['id']}").json()] == ["Who wins tonight?"]
    assert client.get(f"/api/chat/{thread['id']}", headers={"X-Player": "friend-browser"}).status_code == 404


# --- wiring ------------------------------------------------------------------------------------------

def test_build_service_wires_the_lab_and_an_optional_assistant(tmp_path, monkeypatch) -> None:
    from telegram_betting_bot_simple import services

    for name in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "PUBLIC_URL", "AI_CHAT_DAILY_CENTS"):
        monkeypatch.delenv(name, raising=False)
    settings = dict(TELEGRAM_BOT_TOKEN="123456:test", THE_ODDS_API_KEY="not-a-real-key",
                    STATE_DIRECTORY=str(tmp_path), PUBLIC_URL="https://bets.example.com", _env_file=None)
    service = services.build_service(Config(**settings))
    assert isinstance(service.lab, Lab) and service.lab.assistant is None
    assert service.lab.store is service.store and service.lab.public_url == "https://bets.example.com"
    # A key, but the assistant module can't be imported: the app runs without it
    monkeypatch.setitem(sys.modules, "telegram_betting_bot_simple.ai.assistant", None)
    with_key = Config(**settings, ANTHROPIC_API_KEY="not-a-real-key")
    assert services.build_service(with_key).lab.assistant is None
    built = []
    module = types.ModuleType("telegram_betting_bot_simple.ai.assistant")
    module.build_assistant = lambda config, lab, store: built.append((lab, store)) or "assistant"
    monkeypatch.setitem(sys.modules, "telegram_betting_bot_simple.ai.assistant", module)
    service = services.build_service(with_key)
    assert service.lab.assistant == "assistant" and built == [(service.lab, service.store)]


def test_the_app_uses_the_services_lab_or_makes_one(service) -> None:
    lab = Lab(service)
    create_app(service, "s3cret")
    assert service.lab is lab
    fresh = type(service)(service.odds, service.espn, service.tz)
    create_app(fresh, "s3cret")
    assert isinstance(fresh.lab, Lab) and fresh.lab.service is fresh
    without = TestClient(create_app(None, "s3cret"))
    without.post("/api/login", json={"password": "s3cret"})
    response = without.get("/api/me", headers=ME)
    assert response.status_code == 503 and "THE_ODDS_API_KEY" in response.json()["detail"]
    with pytest.raises(CallRejected):
        Lab(fresh).me(UID)  # no state folder: a plain refusal
    assert lab_module.NotFound is NotFound
