"""Stage 3 wiring: the builder's routes (GET /api/builder/presets, POST /api/builder/run, POST /api/builder/price,
PUT /api/me {builder}), the gates and the pause, the result shared for a minute with the stake applied per person,
the candidates assembled from the slate, the saved boards and "Markets say" (never fetched), leagues without prices
(a near miss and one throttled background request), soccer's goals model on the game route and in the builder, the
AI bets and the target builder through the engine, the ticket_build tool's controls and the bot's AI bets.

Fakes and a real Store in a temporary folder: no network (requests' Session.request fails loudly)."""

from __future__ import annotations

import copy
import datetime
import html
import json
from decimal import Decimal
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest
import requests
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.dashboard import lab as lab_module
from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.dashboard.lab import Lab
from telegram_betting_bot_simple.dashboard.lab import candidate_leg
from telegram_betting_bot_simple.dashboard.service import MODEL_BOOK, DashboardService, soccer_inputs
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.odds_service import configure_books, normalize_event
from telegram_betting_bot_simple.data.slate import SlateBuilder
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.engine import tickets
from tests.test_slate import EPL, NBA, NHL, NOW, TODAY, FakeEspn, FakeOdds, at, espn_game

NY = ZoneInfo("America/New_York")
PLAYER_A = {"X-Player": "browser-aaaa1111"}
PLAYER_B = {"X-Player": "browser-bbbb2222"}


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


# --- the fixture: NBA and EPL boards saved, NHL on ESPN only ---------------------------------------------------

def book(key: str, title: str, home: str, away: str, ml: tuple, spread: Optional[tuple] = None,
         total: Optional[tuple] = None) -> Dict[str, Any]:
    """One bookmaker: ml (home, away[, draw]); spread (home point, home price, away price); total (point, over,
    under)."""
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


def event(league: str, event_id: str, home: str, away: str, start: str, books: List[Dict[str, Any]]) -> Dict[str, Any]:
    return normalize_event({"id": event_id, "sport_key": league, "sport_title": league, "commence_time": start,
                            "home_team": home, "away_team": away, "bookmakers": books})


NBA_GAMES = [  # (id, home, away, hours from NOW, home ml, away ml, spread point, total point)
    ("n1", "Boston Celtics", "New York Knicks", 1, 1.45, 2.85, -6.5, 221.5),
    ("n2", "Denver Nuggets", "Utah Jazz", 2, 1.22, 4.60, -9.5, 230.5),
    ("n3", "Miami Heat", "Orlando Magic", 3, 1.95, 1.90, -1.5, 212.5),
    ("n4", "Phoenix Suns", "Portland Trail Blazers", 4, 1.70, 2.20, -3.5, 225.5),
]
EPL_GAMES = [  # (id, home, away, hours, home, away, draw, over 2.5, under 2.5)
    ("s1", "Arsenal", "Everton", 2, 1.55, 6.00, 4.20, 1.80, 2.05),
    ("s2", "Chelsea", "Fulham", 3, 1.90, 4.10, 3.60, 1.95, 1.90),
]


def nba_board(when=at) -> List[Dict[str, Any]]:
    out = []
    for gid, home, away, hours, hml, aml, spread, total in NBA_GAMES:
        books = [book("pinnacle", "Pinnacle", home, away, (hml + 0.03, aml + 0.05), (spread, 1.95, 1.93),
                      (total, 1.94, 1.94)),
                 book("draftkings", "DraftKings", home, away, (hml, aml), (spread, 1.91, 1.91), (total, 1.91, 1.91)),
                 book("fanduel", "FanDuel", home, away, (hml - 0.01, aml + 0.05), (spread, 1.90, 1.92),
                      (total, 1.89, 1.93))]
        out.append(event(NBA, gid, home, away, when(hours), books))
    return out


def in_2030(hours: float) -> str:
    """A start in 2030: still to come by the real clock (the games service's own)."""
    return f"2030-01-01T{int(hours):02d}:00:00Z"


def epl_board() -> List[Dict[str, Any]]:
    out = []
    for gid, home, away, hours, h, a, d, over, under in EPL_GAMES:
        books = [book("pinnacle", "Pinnacle", home, away, (h + 0.03, a + 0.1, d + 0.05), None, (2.5, over + 0.03,
                                                                                               under + 0.03)),
                 book("draftkings", "DraftKings", home, away, (h, a, d), None, (2.5, over, under)),
                 book("fanduel", "FanDuel", home, away, (h - 0.02, a + 0.2, d), None, (2.5, over - 0.02, under))]
        out.append(event(EPL, gid, home, away, at(hours), books))
    return out


def espn_boards() -> Dict[Any, Any]:
    boards: Dict[Any, Any] = {}
    for n, (gid, home, away, hours, *_) in enumerate(NBA_GAMES):
        boards.setdefault((NBA, TODAY), []).append(espn_game(f"40{n}", home, away, at(hours), ids=(f"{n}1", f"{n}2")))
    for n, (gid, home, away, hours, *_) in enumerate(EPL_GAMES):
        boards.setdefault((EPL, TODAY), []).append(espn_game(f"70{n}", home, away, at(hours), sport="soccer",
                                                             ids=(f"7{n}1", f"7{n}2")))
    boards[(NHL, TODAY)] = [espn_game("901", "Boston Bruins", "Toronto Maple Leafs", at(2), sport="nhl"),
                            espn_game("902", "New York Rangers", "Buffalo Sabres", at(3), sport="nhl", ids=("3", "4"))]
    return boards


class StoreOdds:
    """The odds service as the games service uses it, reading the saved boards (no request is ever made: the request
    path must not fetch, so league_games fails the test)."""

    odds_ttl = 7200
    reserve = 50
    quota = {"remaining": 300, "used": 200}

    def __init__(self, store: Store) -> None:
        self.store = store

    def sports(self):
        return [{"key": NBA, "group": "Basketball", "title": "NBA", "active": True, "has_outrights": False},
                {"key": EPL, "group": "Soccer", "title": "EPL", "active": True, "has_outrights": False},
                {"key": NHL, "group": "Ice Hockey", "title": "NHL", "active": True, "has_outrights": False}]

    def saved_games(self, league):
        return self.store.odds_snapshot(league)

    def odds_loaded_at(self, league):
        saved = self.store.odds_snapshot(league)
        return saved[0] if saved else None

    def league_games(self, league, refresh=False):
        raise AssertionError(f"the request path loaded the {league} board")


class NoEspn:
    def find_game(self, *args):
        return None


class World:
    def __init__(self, store: Store, slate: SlateBuilder, service: DashboardService, lab: Lab,
                 odds: FakeOdds) -> None:
        self.store, self.slate, self.service, self.lab, self.odds = store, slate, service, lab, odds
        self.queued: List[str] = []

    def client(self, player: Dict[str, str] = PLAYER_A, confirm_age: bool = True) -> TestClient:
        client = TestClient(create_app(self.service, "s3cret", login_delay=0, lab=self.lab, slate=self.slate))
        assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
        client.headers.update(player)
        if confirm_age:
            assert client.put("/api/me", json={"confirm_age": True}).status_code == 200
        return client


@pytest.fixture()
def world(tmp_path) -> World:
    return make_world(tmp_path)


def make_world(tmp_path) -> World:
    """The world above in a temporary folder (also tests/test_builder_integration.py's)."""
    store = Store(str(tmp_path / "betting.db"))
    store.save_odds_snapshot(NBA, nba_board(), NOW - 600)
    store.save_odds_snapshot(EPL, epl_board(), NOW - 600)
    odds = FakeOdds(store)
    slate = SlateBuilder(odds, FakeEspn(espn_boards()), store, now=lambda: NOW, workers=2)
    slate.refresh()
    service = DashboardService(StoreOdds(store), NoEspn(), NY, store=store, cache=TTLCache())
    service.slate = slate
    lab = Lab(service, store, now=lambda: NOW)
    w = World(store, slate, service, lab, odds)
    # The slate's background loader runs boards on its own thread: here it is only recorded
    slate._odds_pool.submit = lambda fn, league: w.queued.append(league)  # type: ignore[method-assign]
    return w


def run(client: TestClient, controls: Optional[Dict[str, Any]] = None, stake: Any = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"controls": controls or {"preset": "risky"}}
    if stake is not None:
        body["stake"] = stake
    response = client.post("/api/builder/run", json=body)
    assert response.status_code == 200, response.text
    return response.json()


# --- the candidates ---------------------------------------------------------------------------------------------

def candidates(world: World) -> Dict[str, Any]:
    return world.service.builder_candidates(now=NOW)["by_key"]


def test_candidates_come_from_the_slate_the_saved_boards_and_the_blend(world) -> None:
    by_key = candidates(world)
    ml = by_key[f"{NBA}:n1:h2h:home"]
    blend = world.slate.markets_for(NBA, odds_id="n1")["blend"]["p"]
    # The chance is the markets brain's blend; the price is the best at a book a person can bet at (never Pinnacle)
    assert ml.chance_source == "blend" and ml.chance == blend["home"]
    assert ml.book == "DraftKings" and ml.price == Decimal("1.45")
    assert ml.prices == {"DraftKings": Decimal("1.45"), "FanDuel": Decimal("1.44")}
    assert ml.edge == pytest.approx(ml.chance - 1 / 1.45) and ml.fair is not None
    assert ml.price_age_s == 600 and ml.game_key == f"{NBA}:n1" and ml.teams == ("New York Knicks", "Boston Celtics")
    assert ml.label == "Boston Celtics to win" and ml.why[0].startswith("Markets say ")
    spread = by_key[f"{NBA}:n1:spreads:home:-6.5"]
    assert spread.chance_source == "books" and spread.book == "DraftKings" and spread.price == Decimal("1.91")
    assert spread.why[0].startswith("Books' margin-free chance")
    assert not any(c.book == "Pinnacle" or "Pinnacle" in (c.prices or {}) for c in by_key.values())
    # The NHL has games on ESPN and no saved board: no legs, and the league is listed without prices
    assembled = world.service.builder_candidates(now=NOW)
    assert not any(c.league == NHL for c in assembled["candidates"])
    assert len(assembled["leagues"][NHL]["games"]) == 2 and assembled["leagues"][NHL]["priced"] == 0
    # Soccer adds the goals model's markets at fair odds (engine/soccer.py's legs), flagged
    btts = by_key[f"{EPL}:s1:btts:yes"]
    assert btts.chance_source == "model" and "model_price" in btts.flags and btts.book == MODEL_BOOK
    assert btts.label == "Both teams to score: Yes" and btts.price_age_s == 600
    dnb = by_key[f"{EPL}:s1:draw_no_bet:home"]
    assert dnb.push > 0 and dnb.edge == pytest.approx(0.0, abs=0.01)  # fair odds: the draw gives the stake back
    assert by_key[f"{EPL}:s1:team_totals:home_over:1.5"].label == "Arsenal over 1.5 goals"
    assert {f"{EPL}:s1:double_chance:1X", f"{EPL}:s1:h2h:draw"} <= set(by_key)


def test_the_soccer_legs_and_the_game_page_read_the_same_model(world) -> None:
    event = epl_board()[0]
    blend = world.slate.markets_for(EPL, odds_id="s1")["blend"]["p"]
    inputs, derived = world.service.soccer_derived(EPL, event)
    assert inputs["home"] == blend["home"] and inputs["draw"] == blend["draw"]  # the blend's 1X2, not the books'
    btts = candidates(world)[f"{EPL}:s1:btts:yes"]
    assert btts.chance == derived["markets"]["btts"]["yes"]


# --- the routes, the gates and the pause -----------------------------------------------------------------------

def test_presets_route_gives_the_presets_markets_and_defaults(world) -> None:
    client = world.client()
    data = client.get("/api/builder/presets").json()
    assert set(data["presets"]) == {"safe", "risky", "crazy"}
    from telegram_betting_bot_simple.engine.presets import PRESETS
    assert data["presets"]["safe"] == json.loads(json.dumps(PRESETS["safe"].to_dict()))
    assert data["described"]["crazy"]["subtitle"].endswith("expect to lose most of these.")
    assert data["markets"]["h2h"]["available"] is True and data["markets"]["btts"]["available"] is True
    assert data["markets"]["player_props"]["available"] is False
    assert "Stage 6" in data["markets"]["player_props"]["reason"]
    assert {b["key"] for b in data["defaults"]["books"]} >= {"draftkings", "fanduel"}
    assert not {"pinnacle", "betfair_ex_eu", "matchbook"} & {b["key"] for b in data["defaults"]["books"]}
    assert data["defaults"]["stake"] == "10.00" and data["defaults"]["saved"] is None


def test_every_builder_route_has_the_gates_and_the_pause(world) -> None:
    client = world.client(confirm_age=False)
    ref = {"league": NBA, "id": "n1", "market": "h2h", "side": "home"}
    for method, path, body in (("get", "/api/builder/presets", None), ("post", "/api/builder/run", {}),
                               ("post", "/api/builder/price", {"legs": [ref]})):
        response = getattr(client, method)(path, **({"json": body} if body is not None else {}))
        assert response.status_code == 400 and "18+" in response.json()["detail"], path
    client = world.client()
    world.lab.pause_picks("web:" + PLAYER_A["X-Player"], NOW + 3600)
    for method, path, body in (("get", "/api/builder/presets", None), ("post", "/api/builder/run", {}),
                               ("post", "/api/builder/price", {"legs": [ref]})):
        response = getattr(client, method)(path, **({"json": body} if body is not None else {}))
        assert response.status_code == 400 and "Picks are paused" in response.json()["detail"], path
    assert TestClient(create_app(world.service, "s3cret", login_delay=0, lab=world.lab)).post(
        "/api/builder/run", json={}).status_code == 401


def test_a_break_stops_the_builder(world) -> None:
    client = world.client()
    assert client.post("/api/me/cooloff", json={"hours": 24}).status_code == 200
    response = client.post("/api/builder/run", json={})
    assert response.status_code == 400 and "break" in response.json()["detail"]


def test_a_run_returns_priced_tickets_with_refs_and_the_engine_s_words(world) -> None:
    client = world.client()
    data = run(client, {"preset": "risky", "leagues": [NBA, EPL]})
    assert data["label"] == "Risky" and data["stake"] == "10.00" and data["stake_given"] is False
    assert data["tickets"], data
    for ticket in data["tickets"]:
        assert ticket["unit_stake"] == "10.00" and ticket["valid"] is True
        assert ticket["refs"] == [leg["ref"] for leg in ticket["legs"]]
        assert all(ref["league"] in (NBA, EPL) for ref in ticket["refs"])
        assert all(leg["why"] and leg["risk"] and "swaps" in leg for leg in ticket["legs"])
        assert ticket["reach_target"]["chance"] == ticket["joint"]["chance"]  # no target: the joint chance
    keys = [{leg["key"] for leg in t["legs"]} for t in data["tickets"]]
    assert all(not (a & b) for i, a in enumerate(keys) for b in keys[i + 1:])  # tickets share no leg
    assert {"tickets", "near_misses", "blocked", "stats", "unpriced", "controls", "as_of", "notes"} <= set(data)
    # Choosing the goals model's markets changes the risk (Custom); Safe's rules still keep fair-odds legs out
    safe = run(client, {"preset": "safe", "leagues": [NBA, EPL], "markets": ["h2h", "btts", "double_chance"]})
    assert safe["label"] == "Custom" and safe["controls"]["exclude_model_price"] is True
    assert not any(leg["model_price"] for t in safe["tickets"] for leg in t["legs"])
    risky = run(client, {"preset": "risky", "leagues": [EPL], "markets": ["btts", "double_chance"], "tickets": 2})
    assert risky["tickets"] and all(leg["model_price"] for t in risky["tickets"] for leg in t["legs"])
    assert any("fair odds" in note for t in risky["tickets"] for note in t["notes"])


def test_crazy_says_expect_to_lose_most_of_these(world) -> None:
    data = run(world.client(), {"preset": "crazy"})
    assert any("expect to lose most of these" in note for note in data["notes"])


def test_controls_are_checked(world) -> None:
    client = world.client()
    bad = [({"legs": 13}, 422), ({"tickets": 6}, 422), ({"markets": ["player_props"]}, 400),
           ({"books": ["pinnacle"]}, 400), ({"leg_odds_min": 3.0, "leg_odds_max": 2.0}, 400), ({"nonsense": 1}, 422),
           ({"window": {"from": at(5), "to": at(1)}}, 400), ({"window": {"from": at(1), "to": at(24 * 9)}}, 400),
           ({"window": {"from": at(-30), "to": at(-25)}}, 400), ({"kickoff": {"earliest": "25:00"}}, 422),
           ({"tz": "Mars/Olympus"}, 400), ({"max_per_game": None}, 400)]
    for controls, status in bad:
        response = client.post("/api/builder/run", json={"controls": controls})
        assert response.status_code == status, (controls, response.text)
    assert "Stage 6" in client.post("/api/builder/run", json={"controls": {"markets": ["corners"]}}).json()["detail"]


def test_a_window_is_widened_to_five_minutes_and_may_be_open_ended(world) -> None:
    client = world.client()
    data = run(client, {"preset": "risky", "window": {"from": "2026-10-26T22:03:10Z", "to": "2026-10-27T01:01:00Z"}})
    assert data["controls"]["window"] == {"from": "2026-10-26T22:00:00Z", "to": "2026-10-27T01:05:00Z"}
    assert all(leg["start"] <= "2026-10-27T01:05" for t in data["tickets"] for leg in t["legs"])
    open_ended = run(client, {"preset": "risky", "window": {"from": at(2)}})
    assert open_ended["controls"]["window"] == {"from": at(2).replace("Z", ":00Z")}


def test_editing_a_control_makes_it_custom_and_scope_keeps_the_preset(world) -> None:
    client = world.client()
    assert run(client, {"preset": "safe", "leagues": [NBA], "tickets": 2})["label"] == "Safe"
    assert run(client, {"preset": "safe", "legs": 2})["label"] == "Custom"
    assert run(client, {"preset": "risky", "min_chance": 0.3})["label"] == "Custom"


def test_the_kickoff_window_reads_in_the_person_s_time_zone(world) -> None:
    client = world.client()
    client.put("/api/me", json={"timezone": "America/New_York"})
    data = run(client, {"preset": "risky", "leagues": [NBA], "kickoff": {"earliest": "19:30", "latest": "21:30"}})
    assert data["controls"]["tz"] == "America/New_York"
    starts = {leg["start"][11:16] for t in data["tickets"] for leg in t["legs"]}
    assert starts and starts <= {"00:00", "01:00"}  # 8 pm and 9 pm in New York: n2 and n3


# --- shared for a minute, priced per person ---------------------------------------------------------------------

def test_a_result_is_shared_across_people_and_the_stake_is_applied_per_person(world, monkeypatch) -> None:
    calls = []
    engine = lab_module._engine_builder()
    real = engine.build

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "build", counted)
    a, b = world.client(PLAYER_A), world.client(PLAYER_B)
    controls = {"preset": "risky", "leagues": [NBA, EPL]}
    first = run(a, controls, stake="25.00")
    second = run(b, controls)
    assert len(calls) == 1  # one build for both people
    assert [t["refs"] for t in first["tickets"]] == [t["refs"] for t in second["tickets"]]
    assert first["stake"] == "25.00" and second["stake"] == "10.00"
    by_key = candidates(world)
    for ticket, stake in ((first["tickets"][0], "25.00"), (second["tickets"][0], "10.00")):
        exact = tickets.price_ticket([candidate_leg(by_key[leg["key"]]) for leg in ticket["legs"]],
                                     "single" if len(ticket["legs"]) == 1 else "multi", Decimal(stake),
                                     now=datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc))
        for name in ("max_payout", "expected", "chance", "book_cut", "odds", "unit_stake", "most_likely"):
            assert ticket[name] == exact[name], name  # the brain's numbers to the cent
    # Nothing personal in the shared entry: no person's id and no stake but the default
    entry = next(value for key, value in world.service.cache._entries.items()
                 if isinstance(key, tuple) and key[0] == "builder")
    text = json.dumps(entry[2], default=str)
    assert "browser-" not in text and "25.00" not in text and "web:" not in text


def test_a_new_board_or_other_controls_build_again(world, monkeypatch) -> None:
    calls = []
    engine = lab_module._engine_builder()
    real = engine.build
    monkeypatch.setattr(engine, "build", lambda *a, **k: calls.append(1) or real(*a, **k))
    client = world.client()
    run(client, {"preset": "risky"})
    run(client, {"preset": "risky"})
    run(client, {"preset": "crazy"})
    assert len(calls) == 2
    world.store.save_odds_snapshot(NBA, nba_board(), NOW - 30)
    run(client, {"preset": "risky"})
    assert len(calls) == 3


def test_allowed_books_pick_each_leg_s_best_price_at_those_books(world) -> None:
    client = world.client()
    data = run(client, {"preset": "risky", "leagues": [NBA], "books": ["fanduel"], "legs": 1, "tickets": 5})
    assert data["tickets"] and all(leg["book"] == "FanDuel" for t in data["tickets"] for leg in t["legs"])
    assert data["controls"]["books"] == ["fanduel"]


# --- leagues without prices -------------------------------------------------------------------------------------

def test_a_league_without_prices_is_a_near_miss_and_asks_once_for_odds(world) -> None:
    client = world.client()
    data = run(client, {"preset": "risky", "leagues": [NHL]})
    assert data["tickets"] == []
    [entry] = data["unpriced"]
    assert entry["league"] == NHL and entry["games"] == 2 and entry["request"]["code"] == "queued"
    assert "no prices loaded (2 games)" in entry["reason"] and "loading in the background" in entry["reason"]
    miss = next(m for m in data["near_misses"] if m["rule"] == "no_prices")
    assert miss["league"] == NHL and miss["reason"] == entry["reason"]
    assert data["blocked"]["message"].startswith("No prices are loaded")
    assert world.queued == [NHL]
    # The slate's throttle: asking again (another result) queues nothing more
    again = run(client, {"preset": "crazy", "leagues": [NHL]})
    assert again["unpriced"][0]["request"]["code"] in ("loading", "throttled") and world.queued == [NHL]


def test_background_odds_requests_respect_the_daily_cap(world) -> None:
    world.slate.odds_per_day = 0
    data = run(world.client(), {"preset": "risky", "leagues": [NHL]})
    assert data["unpriced"][0]["request"]["code"] == "daily_cap" and world.queued == []


def test_at_most_two_boards_are_queued_per_run_and_answers_that_load_nothing_don_t_count(world) -> None:
    names = ("icehockey_ahl", "basketball_wnba", "baseball_mlb", "americanfootball_nfl", "cricket_odi", "rugby_x")
    extra = {league: {"title": league, "games": [at(2)] * (6 - n), "priced": 0, "odds": "none"}
             for n, league in enumerate(names)}
    answers = {"icehockey_ahl": "not_in_season", "basketball_wnba": "throttled", "baseball_mlb": "fresh",
               "americanfootball_nfl": "queued", "cricket_odi": "queued", "rugby_x": "queued"}
    asked = []

    def request_odds(league):
        asked.append(league)
        return {"queued": answers[league] == "queued", "code": answers[league], "reason": "Asked."}

    world.slate.request_odds = request_odds
    controls = world.lab.builder_controls({"preset": "risky"})
    out = world.lab._unpriced({"leagues": extra}, controls, None)
    # Most games first. No market, asked lately and a fresh board load nothing: they don't hold the rest back. Two
    # boards queued is this run's share: the sixth league waits for another run
    assert asked == list(names[:5])
    assert [e["league"] for e in out][-1] == "rugby_x" and out[-1]["request"] is None
    assert "asked lately" in out[1]["reason"] or "less than 2 hours ago" in out[1]["reason"]
    # The daily cap stops the asking at once
    answers.update(americanfootball_nfl="daily_cap")
    asked.clear()
    world.lab._unpriced({"leagues": extra}, controls, None)
    assert asked == list(names[:4])


def test_the_request_path_never_fetches(world) -> None:
    # StoreOdds.league_games and requests' Session.request both fail the test if called
    client = world.client()
    for preset in ("safe", "risky", "crazy"):
        run(client, {"preset": preset})
    run(client, {"preset": "risky", "markets": ["btts", "draw_no_bet", "double_chance", "team_totals"],
                 "leagues": [EPL]})
    assert world.odds.fetches == []


# --- soccer's goals model -----------------------------------------------------------------------------------------

def test_the_game_route_carries_the_goals_model_for_soccer(world) -> None:
    class BoardOdds(StoreOdds):
        def league_games(self, league, refresh=False):
            return copy.deepcopy(self.store.odds_snapshot(league)[1])

    service = DashboardService(BoardOdds(world.store), NoEspn(), NY, store=world.store, cache=TTLCache())
    client = TestClient(create_app(service, "s3cret", login_delay=0))
    client.post("/api/login", json={"password": "s3cret"})
    client.headers.update(PLAYER_A)
    client.put("/api/me", json={"confirm_age": True})
    game = client.get("/api/game", params={"league": EPL, "id": "s1"}).json()
    block = game["soccer"]
    assert block["rates"]["home"] > block["rates"]["away"] > 0 and block["rho"] == pytest.approx(-0.13)
    blend = game["consensus"]["blend"]["p"]
    assert block["inputs"]["home"] == round(blend["home"], 4)  # the game's own blend
    over = soccer_inputs(epl_board()[0])["over_2_5"]
    assert block["inputs"]["over_2_5"] == round(over, 4) and isinstance(block["fit_ok"], bool)
    yes, no = block["markets"]["btts"]["yes"], block["markets"]["btts"]["no"]
    assert yes["p"] + no["p"] == pytest.approx(1, abs=1e-3) and yes["book"] is None
    # The fair odds rounded down to the cent: the price the Builder's leg for it has (engine/soccer.py fair_price)
    assert 0 <= 1 / yes["p"] - yes["fair"] < 0.0101
    dnb = block["markets"]["draw_no_bet"]["home"]
    assert dnb["push"] == pytest.approx(block["model_1x2"]["draw"], abs=1e-4)
    assert dnb["fair"] == pytest.approx((1 - dnb["push"]) / dnb["p"], abs=0.0101)
    legs = service.builder_candidates(now=NOW)["by_key"]
    assert yes["fair"] == float(legs[f"{EPL}:s1:btts:yes"].price)
    assert dnb["fair"] == float(legs[f"{EPL}:s1:draw_no_bet:home"].price)
    assert set(block["markets"]["double_chance"]) == {"1X", "12", "X2"}
    assert [entry["line"] for entry in block["markets"]["team_totals"]["home"]] == [0.5, 1.5, 2.5, 3.5]
    assert len(block["correct_score"]) == 6 and "fair odds" in block["note"]
    # An NBA game has no goals model
    assert "soccer" not in client.get("/api/game", params={"league": NBA, "id": "n1"}).json()


def test_a_goals_model_leg_prices_on_the_slip_and_is_never_saved(world, monkeypatch) -> None:
    client = world.client()
    ref = {"league": EPL, "id": "s1", "market": "btts", "side": "yes"}
    priced = client.post("/api/tickets/price", json={"type": "single", "stake": "5.00", "legs": [ref]}).json()
    [leg] = priced["legs"]
    assert leg["book"] == MODEL_BOOK and leg["label"] == "Both teams to score: Yes"
    assert Decimal(leg["price"]) == candidates(world)[f"{EPL}:s1:btts:yes"].price
    # Said up front, before anything is saved: no bookmaker offers this price
    assert priced["model_priced"] is True and leg["model_price"] is True
    assert priced["warnings"][0] == ("Priced at fair odds; check your bookmaker's price (Both teams to score: Yes): "
                                     "no bookmaker here offers this price, so the expected result is at fair odds "
                                     "and the ticket can't be saved.")
    # Goals-model legs only: no bookmaker's cut to add up again; beside a bookmaker's leg, that warning stays
    dnb = {"league": EPL, "id": "s2", "market": "draw_no_bet", "side": "home"}
    fair_only = client.post("/api/tickets/price", json={"type": "multi", "stake": "5.00", "legs": [ref, dnb]}).json()
    assert fair_only["warnings"][0].startswith("Priced at fair odds; check your bookmaker's price (")
    assert "these prices" in fair_only["warnings"][0] and tickets.EXTRA_LEG_WARNING not in fair_only["warnings"]
    # (the NBA pick from the saved board, as the builder prices it: this fixture's live board can't load)
    nba = {"league": NBA, "id": "n1", "market": "h2h", "side": "home"}
    lookup = world.lab._lookup
    saved = candidate_leg(candidates(world)[f"{NBA}:n1:h2h:home"])
    monkeypatch.setattr(world.lab, "_lookup", lambda r: (saved, {}) if r["league"] == NBA else lookup(r))
    mixed = client.post("/api/tickets/price", json={"type": "multi", "stake": "5.00", "legs": [ref, nba]}).json()
    assert mixed["warnings"][0].startswith("Priced at fair odds") and tickets.EXTRA_LEG_WARNING in mixed["warnings"]
    assert [view.get("model_price") for view in mixed["legs"]].count(True) == 1
    plain = client.post("/api/tickets/price", json={"type": "single", "stake": "5.00", "legs": [nba]}).json()
    assert plain["model_priced"] is False and not any("fair odds" in w for w in plain["warnings"])
    response = client.post("/api/slips", json={"type": "single", "stake": "5.00", "legs": [ref]})
    assert response.status_code == 400 and "fair odds" in response.json()["detail"]
    bad = client.post("/api/tickets/price", json={"type": "single", "stake": "5.00",
                                                   "legs": [{**ref, "side": "over"}]})
    assert bad.status_code == 400


def test_builder_price_reprices_a_swapped_ticket_with_the_builder_s_numbers(world) -> None:
    client = world.client()
    refs = [{"league": NBA, "id": "n1", "market": "h2h", "side": "home"},
            {"league": EPL, "id": "s2", "market": "btts", "side": "yes"}]
    priced = client.post("/api/builder/price", json={"legs": refs, "stake": "4.00"}).json()
    by_key = candidates(world)
    exact = tickets.price_ticket([candidate_leg(by_key[f"{NBA}:n1:h2h:home"]), candidate_leg(by_key[f"{EPL}:s2:btts:yes"])],
                                 "multi", Decimal("4.00"),
                                 now=datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc))
    assert priced["max_payout"] == exact["max_payout"] and priced["expected"] == exact["expected"]
    assert priced["model_priced"] is True and priced["removed"] == []
    gone = client.post("/api/builder/price", json={"legs": refs + [{"league": NBA, "id": "zz", "market": "h2h",
                                                                    "side": "home"}]}).json()
    assert len(gone["removed"]) == 1
    fanduel = client.post("/api/builder/price", json={"legs": refs[:1], "books": ["fanduel"]}).json()
    assert fanduel["legs"][0]["book"] == "FanDuel"


# --- the person's saved controls ----------------------------------------------------------------------------------

def test_put_me_keeps_the_builder_controls(world) -> None:
    client = world.client()
    controls = {"preset": "crazy", "leagues": [NBA, EPL], "legs": 5, "markets": ["h2h", "btts"],
                "window": {"from": at(-48), "to": at(-24)}, "ui": {"window": "24h", "stake": "5.00"}}
    me = client.put("/api/me", json={"builder": controls}).json()
    saved = me["builder"]
    assert saved["legs"] == 5 and saved["markets"] == ["h2h", "btts"] and saved["label"] == "Custom"
    assert saved["preset"] == "crazy" and saved["ui"] == {"window": "24h", "stake": "5.00"}
    assert saved["leagues"] == [NBA, EPL] and saved["window"]["from"].endswith("Z")
    assert client.get("/api/me").json()["builder"] == saved
    assert client.get("/api/builder/presets").json()["defaults"]["saved"] == saved
    # A saved copy goes straight back as a run's controls
    assert client.post("/api/builder/run", json={"controls": {k: v for k, v in saved.items() if k not in
                                                              ("label", "window")}}).status_code == 200
    assert client.put("/api/me", json={"builder": {"legs": 40}}).status_code == 422
    assert client.put("/api/me", json={"builder": {"ui": {f"k{n}": "x" for n in range(13)}}}).status_code == 422
    assert client.put("/api/me", json={"builder": {"books": ["pinnacle"]}}).status_code == 400
    assert client.put("/api/me", json={"builder": None}).json()["builder"] is None


# --- the AI bets, the target builder, the ticket_build tool and the bot -------------------------------------------

class BoardOdds(StoreOdds):
    """The board as the games service loads it (from the saved snapshot), for the AI bets' own path."""

    def league_games(self, league, refresh=False):
        saved = self.store.odds_snapshot(league)
        return copy.deepcopy(saved[1]) if saved else []


class QuietEspn(NoEspn):
    def standings(self, league, season=None):
        return {}


def spy_builds(monkeypatch) -> List[Any]:
    from telegram_betting_bot_simple.engine import builder
    seen: List[Any] = []
    real = builder.build

    def spy(candidates, controls, now=None, **kwargs):
        seen.append(controls)
        return real(candidates, controls, now, **kwargs)

    monkeypatch.setattr(builder, "build", spy)
    return seen


def test_the_ai_bets_are_the_builder_s_tickets_under_each_preset_and_are_logged(world, monkeypatch) -> None:
    from telegram_betting_bot_simple.bot.handlers import BUILDER_LINE, bets_text
    from telegram_betting_bot_simple.engine.presets import PRESETS
    seen = spy_builds(monkeypatch)
    world.store.save_odds_snapshot(NBA, nba_board(in_2030), NOW - 600)
    service = DashboardService(BoardOdds(world.store), QuietEspn(), NY, store=world.store, cache=TTLCache())
    data = service.ai_bets(NBA)
    assert {controls.preset for controls in seen} == {"safe", "risky", "crazy"}
    for controls in seen:  # each run is the preset's band and leg rules, one bookmaker's prices at a time
        base = PRESETS[controls.preset]
        assert (controls.joint_min, controls.odds_min, controls.odds_max, controls.multi_leg_min) == \
            (base.joint_min, base.odds_min, base.odds_max, base.multi_leg_min)
    tickets = [t for tier in data["tiers"].values() for t in tier["bets"]]
    assert tickets and {t["book"] for t in tickets} <= {"DraftKings", "FanDuel"}
    for key, tier in data["tiers"].items():
        for ticket in tier["bets"]:
            assert PRESETS[key].joint_min <= ticket["probability"] and len({l["event_id"] for l in ticket["legs"]}) \
                == len(ticket["legs"])
    # The AI record still logs each tier's top ticket for the day
    logged = {row["tier"]: row for row in world.store.ai_tickets(league=NBA)}
    assert set(logged) == {key for key, tier in data["tiers"].items() if tier["bets"]}
    for key, row in logged.items():
        assert [leg["event_id"] for leg in row["payload"]["legs"]] == \
            [leg["event_id"] for leg in data["tiers"][key]["bets"][0]["legs"]]
    # The bot's screen says how they're made, and a preset's own page shows its band
    text = bets_text(data, None, "NBA")
    assert html.escape(BUILDER_LINE) in text and "🛡 <b>Safe</b>" in text
    assert data["tiers"]["crazy"]["detail"] in bets_text(data, "crazy", "NBA")
    assert "expect to lose most of these" in bets_text(data, "crazy", "NBA")


def test_the_target_builder_searches_with_the_builder(monkeypatch) -> None:
    from telegram_betting_bot_simple.engine import target
    seen = spy_builds(monkeypatch)
    start = "2030-01-01T00:00:00Z"
    legs = [tickets.TicketLeg(key=f"{NBA}:e{i}:h2h:home", league=NBA, event_id=f"e{i}", market="h2h", side="home",
                              point=None, price=Decimal(str(price)), label=f"Team {i}", game=f"A{i} @ H{i}",
                              start=start, ai_prob=ai, market_prob=market, book="DraftKings")
            for i, (price, market, ai) in enumerate([(3.6, 0.27, 0.29), (1.9, 0.5, 0.54), (1.9, 0.5, 0.5)])]
    out = target.build_for_target(legs, 10, 35, max_legs=3, now=datetime.datetime(2029, 12, 31, tzinfo=datetime.timezone.utc))
    assert [controls.legs for controls in seen] == [1, 2, 3]
    assert all(controls.odds_min == pytest.approx(3.15) and controls.odds_max == pytest.approx(4.025)
               for controls in seen)
    assert [t["slot"] for t in out["tickets"]] == ["single", "double"]


def test_ticket_build_runs_the_builder_with_the_controls(world) -> None:
    from telegram_betting_bot_simple.ai.tools import TOOLS, Toolbox
    world.client()  # confirms the person's age
    uid = "web:" + PLAYER_A["X-Player"]
    schema = next(tool for tool in TOOLS if tool["name"] == "ticket_build")
    assert "strict" not in schema and schema["input_schema"]["required"] == []
    box = Toolbox(world.lab, uid, NY, {"stake": {Decimal("10")}}, leagues={NBA, EPL})
    done = box.run("ticket_build", {"leagues": [NBA, EPL], "legs": 2, "target_odds": 3.5, "preset": "risky",
                                    "tickets": 3})
    assert done.ok, done.content
    assert done.content["label"] == "Custom" and done.content["stake"] == "10.00"
    assert 1 <= len(done.content["tickets"]) <= 3  # at most three tickets for the chat
    assert all(len(t["legs"]) == 2 for t in done.content["tickets"])
    assert done.content["tickets"][0]["legs"][0]["why"] and "reach_target" in done.content["tickets"][0]
    assert done.summary.startswith("ticket_build(builder: basketball_nba, soccer_epl, Custom, 2 legs, about 3.5x)")
    assert done.cards and all(card["source"] == "ticket_build" and card["refs"] for card in done.cards)
    # The user's stake prices the tickets; one they didn't give is refused, and a target needs its stake
    priced = box.run("ticket_build", {"leagues": [NBA], "preset": "safe", "stake": 10})
    assert priced.ok and priced.content["stake_given"] is True
    assert box.run("ticket_build", {"preset": "risky", "stake": 50}).content["error"] == "amount_not_given"
    assert box.run("ticket_build", {"preset": "risky", "target": 40}).content["error"] == "invalid_input"
    assert box.run("ticket_build", {"tickets": 4}).content["error"] == "invalid_input"
    nothing = box.run("ticket_build", {"leagues": [NHL], "preset": "crazy"})
    assert nothing.ok and nothing.content["tickets"] == [] and "blocked" in nothing.content["hint"]
    paused = Toolbox(world.lab, uid, NY, paused=True).run("ticket_build", {"preset": "risky"})
    assert paused.content["error"] == "picks_paused"


def test_the_router_reads_target_odds_and_builds_without_asking_for_a_stake() -> None:
    from telegram_betting_bot_simple.ai.assistant import ROUTER_SCHEMA, Assistant, parse_route
    assert "target_odds" in ROUTER_SCHEMA["required"]
    fields = {"intent": "build", "leagues": [NBA, EPL], "stake": None, "target": None, "max_legs": 5,
              "tier": "risky", "days": None, "min_odds": None, "legs_per_day": None, "players": [],
              "risk_flags": []}
    route = parse_route({**fields, "target_odds": 10})
    assert route.target_odds == Decimal("10.00") and route.fields()["target_odds"] == "10.00"
    assert Assistant._clarify(None, route) is None  # "at about 10x": the Builder needs no stake
    asked = Assistant._clarify(None, parse_route(fields))
    assert asked is not None and asked.text == "How much do you want to stake?"


def test_an_ai_ticket_is_placed_at_one_bookmaker_that_offers_every_leg() -> None:
    from telegram_betting_bot_simple.engine import bets
    from tests.test_bets import leg
    a, b = leg("g1", 0.75, 1.30, books=("Only A",)), leg("g2", 0.80, 1.25, books=("Only B",))
    # The builder's best Safe ticket (both legs) has no bookmaker offering both: each book is searched alone
    safe = bets.build_tier(bets.TIERS[0], [a, b])
    assert sorted((t.book, [l.event_id for l in t.legs]) for t in safe) == [("Only A", ["g1"]), ("Only B", ["g2"])]
    both = [leg("g1", 0.80, 1.22, books=("A", "B")), leg("g2", 0.80, 1.25, books=("A", "B"))]
    both[0].prices["B"] = 1.24  # B pays more on g1: that single is placed at B
    single = next(t for t in bets.build_tier(bets.TIERS[0], both) if [l.event_id for l in t.legs] == ["g1"])
    assert single.book == "B" and single.decimal == pytest.approx(1.24)


def test_a_league_with_a_board_is_never_called_unpriced(world) -> None:
    controls = world.lab.builder_controls({"preset": "risky"})
    leagues = {NBA: {"title": "NBA", "games": [at(2), at(3)], "priced": 4, "odds": "fresh"},
               NHL: {"title": "NHL", "games": [at(2)], "priced": 0, "odds": "none"}}
    world.slate.request_odds = lambda league: {"queued": True, "code": "queued", "reason": "Loading."}
    assert [entry["league"] for entry in world.lab._unpriced({"leagues": leagues}, controls, None)] == [NHL]


def test_a_builder_ticket_with_a_goals_model_leg_can_be_analysed(world) -> None:
    client = world.client()
    refs = [{"league": EPL, "id": "s1", "market": "double_chance", "side": "1X"},
            {"league": EPL, "id": "s2", "market": "team_totals", "side": "home_over", "point": 0.5}]
    response = client.post("/api/tickets/analyse", json={"type": "multi", "stake": "5.00", "legs": refs})
    assert response.status_code == 200, response.text
    data = response.json()
    assert [leg["label"] for leg in data["ticket"]["legs"]] == ["Arsenal or draw", "Chelsea over 0.5 goals"]
    assert all(leg["book"] == MODEL_BOOK for leg in data["ticket"]["legs"])
    # Priced by the real clock, the fixture's games (on the test's clock) have started: that is the only complaint
    assert all("started" in reason for reason in data["ticket"]["reasons"])
    assert [leg["label"] for leg in data["legs"]] == ["Arsenal or draw", "Chelsea over 0.5 goals"]  # notes per leg
    assert "alternatives" in data and data["explanation"] is None  # no assistant here: the numbers stand alone
    # Whose chance it is, and that no bookmaker offers the price, in the notes, the warnings and the explanation
    notes = data["legs"][0]["notes"]
    assert any("by the goals model's chance" in note for note in notes)
    assert not any("the AI's chance" in note for note in notes)
    assert any(note.startswith("Priced at fair odds; check your bookmaker's price") for note in notes)
    assert data["ticket"]["warnings"][0].startswith("Priced at fair odds; check your bookmaker's price (")
    from telegram_betting_bot_simple.ai.explain import template_explanation
    words = template_explanation(world.lab.fact_sheet(data["ticket"], data["legs"], data["alternatives"]))
    assert "by the goals model" in words["verdict"] and "by the AI" not in words["verdict"]
    assert "at fair odds" in words["verdict"]
    assert "cut again" not in words["case_against"] and "priced at fair odds" in words["case_against"]
    assert all(leg["text"].split(": ", 1)[1].startswith("goals model ") for leg in words["per_leg"])


def test_explain_and_ask_price_a_builder_ticket_at_the_run_s_bookmakers(world) -> None:
    """A Builder ticket explained (/api/tickets/analyse) or taken to Ask (the chat's ticket_analyse) carries the run's
    books: each leg is priced where the Builder's card priced it, not at the best price anywhere."""
    from telegram_betting_bot_simple.ai.tools import TOOLS, Toolbox
    world.service.odds = BoardOdds(world.store)  # the board loads from the saved snapshot, as on the server
    client = world.client()
    refs = [{"league": NBA, "id": "n1", "market": "h2h", "side": "home"}]
    anywhere = client.post("/api/tickets/analyse", json={"type": "single", "stake": "5.00", "legs": refs}).json()
    assert anywhere["ticket"]["legs"][0]["book"] == "DraftKings"  # 1.45, the best price
    response = client.post("/api/tickets/analyse", json={"type": "single", "stake": "5.00", "legs": refs,
                                                         "books": ["fanduel"]})
    assert response.status_code == 200, response.text
    at_fanduel = response.json()["ticket"]
    assert at_fanduel["legs"][0]["book"] == "FanDuel" and Decimal(at_fanduel["legs"][0]["price"]) == Decimal("1.44")
    assert at_fanduel["removed"] == []
    # The price is the Builder's own: the same leg candidate_leg makes for the run's tickets
    exact = candidate_leg(candidates(world)[f"{NBA}:n1:h2h:home"], ["fanduel"])
    assert Decimal(at_fanduel["legs"][0]["price"]) == exact.price
    assert client.post("/api/tickets/analyse", json={"type": "single", "stake": "5.00", "legs": refs,
                                                     "books": ["pinnacle"]}).status_code == 400
    # Ask: the slip's books go to ticket_analyse, which passes them on
    schema = next(tool for tool in TOOLS if tool["name"] == "ticket_analyse")
    assert "books" in schema["input_schema"]["properties"] and "books" not in schema["input_schema"]["required"]
    box = Toolbox(world.lab, "web:" + PLAYER_A["X-Player"], NY, {"stake": {Decimal("5")}}, leagues={NBA})
    done = box.run("ticket_analyse", {"type": "single", "stake": 5, "legs": refs, "books": ["fanduel"]})
    assert done.ok, done.content
    assert done.cards[0]["ticket"]["legs"][0]["book"] == "FanDuel"


def test_builder_ticket_facts_name_the_chance_and_give_the_edge_per_unit() -> None:
    """ticket_build's facts: a leg's chance by its own name and source (never "ai_chance" or "blend_chance"), its
    edge as every other tool gives it (the expected result per unit) and the points against the price apart."""
    from telegram_betting_bot_simple.ai.tools import builder_ticket_facts
    from telegram_betting_bot_simple.engine import builder as engine
    start = "2026-10-27T19:00:00Z"
    legs = [engine.Candidate(ref={"league": NBA, "id": "a", "market": "h2h", "side": "home"}, league=NBA,
                             game_key="a", teams=("A", "B"), start=start, price=Decimal("1.95"), book="DraftKings",
                             chance=0.53, chance_source="blend", fair=0.51),
            engine.Candidate(ref={"league": EPL, "id": "b", "market": "btts", "side": "yes"}, league=EPL,
                             game_key="b", teams=("C", "D"), start=start, price=Decimal("1.76"), book=MODEL_BOOK,
                             chance=0.568, chance_source="model", flags={"model_price"}, label="Both teams to score: Yes")]
    now = datetime.datetime(2026, 10, 27, 12, 0, tzinfo=datetime.timezone.utc)
    ticket = engine.build(legs, engine.Controls(markets={"h2h", "btts"}, legs=2), now).tickets[0]
    facts = builder_ticket_facts(ticket)
    by_label = {leg["label"]: leg for leg in facts["legs"]}
    views = {view["label"]: view for view in ticket["legs"]}
    blend, model = by_label["B"], by_label["Both teams to score: Yes"]
    for fact in (blend, model):
        assert "ai_chance" not in fact and "blend_chance" not in fact and "ai_chance_given_no_push" not in fact
    assert (blend["chance"], blend["chance_source"]) == (0.53, "markets_say")
    assert (model["chance"], model["chance_source"]) == (0.568, "goals_model") and model["fair_odds_only"] is True
    # The edge is the expected result per unit (0.53 x 1.95 - 1), as ticket_price gives it; the points apart
    assert blend["edge"] == views["B"]["ev"] == pytest.approx(0.53 * 1.95 - 1, abs=1e-4)
    assert blend["edge_vs_price"] == pytest.approx(0.53 - 1 / 1.95, abs=1e-4)
    assert model.get("edge_vs_price") is None  # no bookmaker's price to measure it against
