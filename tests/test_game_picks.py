"""Suggested picks on every game the owner opens (SPEC_GAME_PICKS, contract A).

- engine/builder.py's game_picks: only the game's legs; top_picks' price safety (no game under way, no fair-odds leg,
  no stale price, never a sharp book or an exchange); one pick per market (the highest chance, ties by the edge) at
  the main line only, in the Builder's MARKETS order; the pick's fields (market_chance, brain_higher, why).
- Lab.game_picks: the gate first; the Builder's candidates; a game without prices (``unpriced``, one ask of the
  slate's throttled background loader); a game that has started; the player board's props limited to the game
  (likeliest, edges, day) and ``loading``.
- GET /api/game/picks: validated like /api/game, the 400 every pick route gives a paused user, a 404 for a game
  nothing in hand knows.
- The bot's "Suggested picks" block: a paused person gets none, the card stays within Telegram's 4096 characters (the
  block gives way to the card's own content), everything escaped, the pick buttons' callback data unchanged.
- No Odds API request: the counting test runs the real OddsService, whose one entry point to The Odds API (_get)
  records every call and refuses it, as tests/test_review2_data.py's RD4 tests do, under the real games service, the
  real ESPN client (the captured fixtures) and the real player board.

The worlds are tests/test_builder_wiring.py's (a real Store, slate, games service and Lab over saved NBA and EPL
boards; the NHL on ESPN only) and tests/test_player_scan.py's NbaWorld (the counting test). No network: requests'
Session.request fails loudly, except where the counting test answers ESPN from the fixtures."""

from __future__ import annotations

import asyncio
import copy
import datetime
import html
import json
import re
import threading
import time
from decimal import Decimal
from html.parser import HTMLParser
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
import requests
from fastapi.testclient import TestClient
from telegram.ext import Application

from telegram_betting_bot_simple.bot import handlers
from telegram_betting_bot_simple.bot.handlers import (SUGGESTED_HEAD, SUGGESTED_NOTE, game_card_text, parse_pick,
                                                      setup_handlers, suggested_entries)
from telegram_betting_bot_simple.config import Config
from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.dashboard.lab import AGE_MESSAGE, GAME_PROPS, PRICE_REASONS, Lab
from telegram_betting_bot_simple.dashboard.service import CallRejected, DashboardService, NotFound
from telegram_betting_bot_simple.data import odds_service as odds_mod
from telegram_betting_bot_simple.data import player_board as pb_mod
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.espn_client import EspnClient
from telegram_betting_bot_simple.data.espn_players import PlayerData
from telegram_betting_bot_simple.data.odds_service import OddsService, configure_books
from telegram_betting_bot_simple.data.player_board import PlayerBoard
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.engine import builder
from telegram_betting_bot_simple.engine.builder import MARKET_TITLES, Candidate, Controls, game_picks
from tests.test_builder_wiring import (EPL, NBA, NHL, NOW, NY, PLAYER_A, PLAYER_B, BoardOdds, QuietEspn, at, book,
                                       event, in_2030, make_world, nba_board)
from tests.test_player_scan import NOW as SCAN_NOW, NbaWorld, Web

UID = "web:" + PLAYER_A["X-Player"]
TG = "tg:42"
NOTE = f"<i>{html.escape(SUGGESTED_NOTE)}</i>"


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


# --- engine/builder.py's game_picks --------------------------------------------------------------------------------

START = "2030-01-01T00:00:00Z"
WHEN = datetime.datetime(2029, 12, 31, 12, 0, tzinfo=datetime.timezone.utc)  # before START


def leg(gid: str, market: str, side: str, chance: float, price: float, point: Optional[float] = None, *,
        league: str = NBA, book: str = "DraftKings", fair: Optional[float] = None,
        prices: Optional[Dict[str, float]] = None, flags=(), age: int = 600, start: str = START,
        why: Optional[List[str]] = None, **extra: Any) -> Candidate:
    """One Builder leg of a Knicks @ Celtics game ``gid``."""
    return Candidate(ref={"league": league, "id": gid, "market": market, "side": side, "point": point}, league=league,
                     game_key=f"{league}:{gid}", teams=("New York Knicks", "Boston Celtics"), start=start,
                     price=Decimal(str(price)), book=book, chance=chance, fair=fair, flags=set(flags),
                     price_age_s=age, prices={b: Decimal(str(p)) for b, p in prices.items()} if prices else None,
                     why=why if why is not None else [f"{market} {side}: a reason"], **extra)


def test_one_pick_per_market_for_this_game_only_in_the_builders_market_order() -> None:
    legs = [
        leg("g1", "btts", "yes", 0.56, 1.80, fair=0.53, label="Both teams to score: Yes"),  # a book prices it
        leg("g1", "totals", "under", 0.47, 1.93, 221.5, fair=0.48),
        leg("g1", "totals", "over", 0.53, 1.91, 221.5, fair=0.52, why=["one", "two", "three", "four"]),
        leg("g1", "spreads", "away", 0.49, 1.92, 6.5, fair=0.5),
        leg("g1", "spreads", "home", 0.51, 1.91, -6.5, fair=0.5),
        leg("g1", "h2h", "away", 0.34, 2.90, fair=0.33),
        leg("g1", "h2h", "home", 0.66, 1.45, fair=0.6123456,
            why_units={"metric": ["m1", "m2", "m3", "m4"], "imperial": ["i1"]}),
        leg("g1", "draw_no_bet", "home", 0.80, 1.20, flags={"model_price"}, book="Fair odds"),  # fair odds: never
        leg("g2", "h2h", "home", 0.95, 1.05),  # another game
        leg("g1", "h2h", "home", 0.97, 1.03, league=EPL),  # another league's game with the same id
    ]
    picks = game_picks(legs, NBA, "g1", now=WHEN)
    assert [p["market"] for p in picks] == ["h2h", "spreads", "totals", "btts"]
    assert [p["market_title"] for p in picks] == [MARKET_TITLES[p["market"]] for p in picks]
    assert all(p["ref"]["league"] == NBA and p["ref"]["id"] == "g1" for p in picks)
    ml, spread, total, btts = picks
    assert set(ml) == {"ref", "key", "market", "market_title", "label", "chance", "market_chance", "price", "book",
                       "edge", "brain_higher", "chance_source", "why", "why_units"}
    assert set(spread) == set(ml) - {"why_units"}
    assert ml["key"] == f"{NBA}:g1:h2h:home" and ml["label"] == "Boston Celtics"  # the leg's own label
    assert (ml["chance"], ml["market_chance"], ml["price"], ml["book"]) == (0.66, 0.6123, "1.45", "DraftKings")
    assert ml["edge"] == round(0.66 - 1 / 1.45, 4) and ml["why_units"] == {"metric": ["m1", "m2", "m3"],
                                                                           "imperial": ["i1"]}
    assert spread["ref"] == {"league": NBA, "id": "g1", "market": "spreads", "side": "home", "point": -6.5}
    assert total["ref"]["side"] == "over" and total["why"] == ["one", "two", "three"]  # at most 3 lines
    assert btts["label"] == "Both teams to score: Yes" and btts["book"] == "DraftKings"
    json.dumps(picks)
    assert game_picks(legs, NBA, "nope", now=WHEN) == []
    # The id is compared as text: the route's "7" is the board's 7
    assert [p["ref"]["id"] for p in game_picks([leg("7", "h2h", "home", 0.6, 1.7)], NBA, 7, now=WHEN)] == ["7"]


def test_an_adjusted_leg_keeps_its_adjustment_and_its_chance_source() -> None:
    """Review F4: a totals leg whose chance a forecast or roof change moved keeps ``adjusted`` (the slip's "Adj" chip)
    and ``chance_source``; a plain leg has its source and no ``adjusted``."""
    note = {"applied": True, "reason": "Wind 25 km/h blowing in", "before": 0.5}
    moved = leg("g1", "totals", "under", 0.56, 1.91, 221.5, fair=0.5, chance_source="adjusted", adjusted=note)
    [pick] = game_picks([moved], NBA, "g1", now=WHEN)
    assert pick["chance_source"] == "adjusted" and pick["adjusted"] == note and pick["adjusted"] is not note
    [plain] = game_picks([leg("g1", "h2h", "home", 0.6, 1.7, fair=0.58, chance_source="books")], NBA, "g1", now=WHEN)
    assert plain["chance_source"] == "books" and "adjusted" not in plain


def test_the_highest_chance_wins_and_a_tie_goes_to_the_better_price() -> None:
    # The likelier side wins although the other has the bigger edge (0.45 - 1/2.40 against 0.55 - 1/1.70)
    likelier = [leg("g1", "h2h", "home", 0.55, 1.70), leg("g1", "h2h", "away", 0.45, 2.40)]
    assert game_picks(likelier, NBA, "g1", now=WHEN)[0]["ref"]["side"] == "home"
    tie = [leg("g1", "h2h", "home", 0.5, 1.95), leg("g1", "h2h", "away", 0.5, 2.05)]
    assert game_picks(tie, NBA, "g1", now=WHEN)[0]["ref"]["side"] == "away"
    # The edge is at the best price a person can bet: FanDuel's 2.10 beats the away side's 2.05
    best = [leg("g1", "h2h", "home", 0.5, 1.95, prices={"DraftKings": 1.95, "FanDuel": 2.10}),
            leg("g1", "h2h", "away", 0.5, 2.05)]
    [pick] = game_picks(best, NBA, "g1", now=WHEN)
    assert (pick["ref"]["side"], pick["price"], pick["book"]) == ("home", "2.1", "FanDuel")
    assert pick["edge"] == round(0.5 - 1 / 2.10, 4)


def test_price_safety_never_quotes_a_price_the_person_can_t_bet() -> None:
    unsafe = [
        leg("g1", "h2h", "home", 0.90, 1.10, flags={"live"}),  # under way
        leg("g1", "spreads", "home", 0.90, 1.10, -3.5, flags={"stale_price"}),  # an old board
        leg("g1", "totals", "over", 0.90, 1.10, 220.5, age=Controls().max_price_age_s + 1),  # past the Builder's limit
        leg("g1", "btts", "yes", 0.90, 1.10, flags={"model_price"}, book="Fair odds"),  # the goals model's fair odds
        leg("g1", "double_chance", "1X", 0.001, 1.10),  # no chance to speak of
        leg("g1", "draw_no_bet", "home", 0.90, 1.10, book="Pinnacle"),  # a sharp book's price only
        leg("g1", "team_totals", "home_over", 0.90, 1.10, 1.5, book="Smarkets"),  # an exchange's price only
    ]
    assert game_picks(unsafe, NBA, "g1", now=WHEN) == []
    # The best price is a price book's, never Pinnacle's or an exchange's, however much longer theirs is
    [pick] = game_picks([leg("g1", "h2h", "away", 0.4, 2.5, prices={"Pinnacle": 2.9, "Betfair": 3.0, "Smarkets": 3.1,
                                                                      "FanDuel": 2.6})], NBA, "g1", now=WHEN)
    assert (pick["price"], pick["book"]) == ("2.6", "FanDuel")
    # Never once the game has started, nor when its start can't be read
    assert game_picks([leg("g1", "h2h", "home", 0.6, 1.7)], NBA, "g1",
                      now=datetime.datetime(2030, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)) == []
    assert game_picks([leg("g1", "h2h", "home", 0.6, 1.7, start="tonight")], NBA, "g1", now=WHEN) == []
    # Without ``now`` the clock decides; without ``max_price_age_s`` the Builder's own stale limit applies
    assert game_picks([leg("g1", "h2h", "home", 0.6, 1.7, start="2020-01-01T00:00:00Z")], NBA, "g1") == []
    assert len(game_picks([leg("g1", "h2h", "home", 0.6, 1.7)], NBA, "g1")) == 1
    old = [leg("g1", "h2h", "home", 0.6, 1.7, age=Controls().max_price_age_s + 60)]
    assert game_picks(old, NBA, "g1", now=WHEN) == []
    assert len(game_picks(old, NBA, "g1", now=WHEN, max_price_age_s=Controls().max_price_age_s + 120)) == 1


def test_spreads_and_totals_count_only_the_main_line() -> None:
    three = {"DraftKings": 1.91, "FanDuel": 1.90, "BetMGM": 1.92}
    main = [leg("g1", "spreads", "home", 0.52, 1.91, -6.5, prices=three, fair=0.5),
            leg("g1", "spreads", "away", 0.48, 1.91, 6.5, prices=three, fair=0.5),
            leg("g1", "totals", "over", 0.51, 1.91, 221.5, prices=three),
            leg("g1", "totals", "under", 0.49, 1.91, 221.5, prices=three)]
    alternates = [leg("g1", "spreads", "home", 0.78, 1.25, -1.5),  # likelier, but one book's alternate line
                  leg("g1", "spreads", "away", 0.70, 1.40, 12.5, prices={"FanDuel": 1.40}),
                  leg("g1", "totals", "under", 0.85, 1.15, 240.5, prices={"BetMGM": 1.15})]
    by_market = {p["market"]: p for p in game_picks(main + alternates, NBA, "g1", now=WHEN)}
    assert by_market["spreads"]["ref"]["point"] == -6.5 and by_market["totals"]["ref"]["point"] == 221.5
    # The main line is found among all the game's legs: with its own legs ruled out (stale), no alternate takes over
    stale = [leg("g1", c.market, c.side, c.chance, float(c.price), c.point, prices=three, flags={"stale_price"})
             for c in main[:2]] + main[2:] + alternates
    assert [p["market"] for p in game_picks(stale, NBA, "g1", now=WHEN)] == ["totals"]
    # The away side's +6.5 is the -6.5 line
    away = [leg("g1", "spreads", "away", 0.55, 1.91, 6.5, prices=three), leg("g1", "spreads", "home", 0.6, 1.5, -2.5)]
    assert game_picks(away, NBA, "g1", now=WHEN)[0]["ref"]["point"] == 6.5
    # As many books on two lines: the one priced nearest even money
    tie = [leg("g1", "totals", "over", 0.62, 1.60, 219.5, fair=0.62),
           leg("g1", "totals", "over", 0.51, 1.95, 222.5, fair=0.51)]
    assert game_picks(tie, NBA, "g1", now=WHEN)[0]["ref"]["point"] == 222.5


@pytest.mark.parametrize("chance, fair, higher", [
    (0.61, 0.58, True), (0.58, 0.55, True), (0.53, 0.50, True),  # 3 points (0.58 - 0.55 is 0.0299... in floats)
    (0.5899, 0.56, False), (0.64, 0.64, False), (0.50, 0.60, False)])
def test_brain_higher_is_three_points_or_more_above_the_market(chance, fair, higher) -> None:
    [pick] = game_picks([leg("g1", "h2h", "home", chance, 1.9, fair=fair)], NBA, "g1", now=WHEN)
    assert pick["brain_higher"] is higher and pick["market_chance"] == fair and pick["chance"] == chance


def test_without_a_market_chance_there_is_none_and_brain_higher_is_false() -> None:
    [pick] = game_picks([leg("g1", "h2h", "home", 0.9, 1.5)], NBA, "g1", now=WHEN)
    assert pick["market_chance"] is None and pick["brain_higher"] is False


# --- Lab.game_picks ------------------------------------------------------------------------------------------------

def log(values: List[float], key: str = "pts") -> List[Dict[str, Any]]:
    """Game-log rows, newest first, as PlayerData.gamelog gives them."""
    return [{"season": "2027", "stats": {"pts": 20.0, "reb": 6.0, "ast": 5.0, "fg3m": 2.0, "stl": 1.0, "blk": 0.0,
                                         key: value}} for value in values]


def ou(market: str, line: float, over: Optional[float] = None, under: Optional[float] = None) -> Dict[str, Any]:
    """A DraftKings over/under item as the board keeps it (ESPN's often carry no prices)."""
    return {"market": market, "kind": "over_under", "line": line, "label": market, "over_price": over,
            "under_price": under}


def means(pts: float, reb: float = 6.0, ast: float = 5.0, blk: float = 0.4) -> Dict[str, float]:
    return {"pts": pts, "reb": reb, "ast": ast, "fg3m": 2.5, "stl": 1.1, "blk": blk, "tov": 3.0, "pra": pts + reb + ast}


def player(athlete: str, name: str, team: str, opponent: str, event_id: str, projected: Dict[str, float],
           rows: List[Dict[str, Any]], props: List[Dict[str, Any]], status: Optional[str] = None) -> Dict[str, Any]:
    return {"athlete_id": athlete, "name": name, "team": team, "opponent": opponent, "home": team == "BOS",
            "position": "F", "event_id": event_id, "status": status,
            "projection": {"means": projected, "minutes": 34.0}, "rows": rows, "props": props}


def board_snapshot() -> Dict[str, Any]:
    """Tonight's NBA board as data/player_board.py's snapshot gives it: the Knicks @ Celtics game (ESPN 400, the
    slate's n1) and Jazz @ Nuggets (401). The real engine ranks it."""
    players = [
        player("1", "Jayson Tatum", "BOS", "NYK", "400", means(28.0), log([30, 27, 33, 24, 30, 29, 26, 35, 28, 32]),
               [ou("player_points", 24.5, 1.87, 1.95)]),
        player("2", "Jaylen Brown", "BOS", "NYK", "400", means(21.0), log([20, 22, 19, 23, 21, 18, 22, 20, 24, 19]),
               [ou("player_points", 23.5, 1.91, 1.91)]),
        player("3", "Kristaps Porzingis", "BOS", "NYK", "400", means(19.0),
               log([18, 21, 17, 22, 19, 20, 16, 23, 18, 21]), [ou("player_points", 12.5)]),  # no price posted
        player("8", "Al Horford", "BOS", "NYK", "400", means(7.5), log([9, 11, 7, 10, 8, 12, 6, 9, 10, 8]),
               [ou("player_points", 9.5, 1.95, None)]),  # the under is likelier, but only the over is priced
        player("4", "Jalen Brunson", "NYK", "BOS", "400", means(26.0, ast=7.5), log([7, 8, 6, 9, 7, 8, 7, 6, 8, 9], "ast"),
               [ou("player_assists", 6.5, 1.80, 2.00), ou("player_points", 26.5)]),
        player("5", "Josh Hart", "NYK", "BOS", "400", means(12.0, reb=9.5), log([10, 9, 11, 8, 12, 9, 10, 11, 9, 10], "reb"),
               [ou("player_rebounds", 8.5, 3.50, 1.30)]),  # a long price the numbers like far too much: suspect
        player("6", "OG Anunoby", "NYK", "BOS", "400", means(30.0), log([30] * 10), [ou("player_points", 15.5, 1.9, 1.9)],
               status="Out"),
        player("7", "Nikola Jokic", "DEN", "UTA", "401", means(30.0), log([35] * 10),
               [ou("player_points", 15.5, 1.9, 1.9)]),  # another game
    ]

    def game(event_id: str, label: str) -> Dict[str, Any]:
        return {"event_id": event_id, "label": label, "start": "2026-10-26T23:00Z", "outs": {},
                "props": {"book": "DraftKings", "as_of": "2026-10-26T21:40:00Z", "stale": False}}

    return {"league": NBA, "day": "2026-10-26", "as_of": "2026-10-26T21:45:00Z",
            "games": {"400": game("400", "NYK @ BOS"), "401": game("401", "UTA @ DEN")}, "players": players,
            "started_players": [], "games_ready": 2, "games_total": 2, "games_started": 0, "games_failed": 0,
            "building": False, "loaded": True, "error": None}


class FakeBoard:
    """data/player_board.py's PlayerBoard as Lab.game_picks reads it: snapshot(league, day, start, props) answers a
    hand-built snapshot (``view``, which a test may change), each call recorded."""

    def __init__(self) -> None:
        self.view = board_snapshot()
        self.calls: List[tuple] = []

    def snapshot(self, league: str, day: Any = "today", start: bool = True, props: bool = False) -> Dict[str, Any]:
        self.calls.append((league, day, start, props))
        return copy.deepcopy(self.view)


@pytest.fixture()
def world(tmp_path):
    w = make_world(tmp_path)
    w.client()  # PLAYER_A confirms their age
    w.board = FakeBoard()
    w.lab.player_board = w.board
    w.lab.price_wait = 0.2  # a board on its way gets this long (_ensure_prices; 8 s on the server)
    return w


def later_lab(world, seconds: float) -> Lab:
    """The world's Lab on another clock (NOW + seconds), over the same games service, store and board."""
    lab = Lab(world.service, world.store, now=lambda: NOW + seconds)
    lab.player_board = world.board
    lab.price_wait = world.lab.price_wait
    return lab


def test_the_gate_comes_first(world, monkeypatch) -> None:
    asked: List[Any] = []
    real = world.service.builder_candidates
    monkeypatch.setattr(world.service, "builder_candidates", lambda now=None: asked.append(now) or real(now=now))
    world.lab.pause_picks(UID, NOW + 3600)
    with pytest.raises(CallRejected, match="Picks are paused until"):
        world.lab.game_picks(UID, NBA, "n1")
    with pytest.raises(CallRejected, match=re.escape(AGE_MESSAGE)):  # the age check before anything else
        world.lab.game_picks("web:browser-cccc3333", NBA, "n1")
    world.store.get_or_create_user(TG, now=NOW)
    world.store.confirm_age(TG, now=NOW)
    world.store.start_cooloff(TG, 24, now=NOW)
    with pytest.raises(CallRejected, match="on a break"):
        world.lab.game_picks(TG, NBA, "n1")
    assert asked == [] and world.board.calls == [] and world.queued == []  # nothing read, built or asked for


def test_a_game_s_picks_are_the_engine_s_over_the_builder_s_candidates(world) -> None:
    out = world.lab.game_picks(UID, NBA, "n1")
    assembled = world.service.builder_candidates(now=NOW)
    when = datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc)
    assert out["picks"] == builder.game_picks(assembled["candidates"], NBA, "n1", now=when,
                                              max_price_age_s=Controls().max_price_age_s)
    assert [p["market"] for p in out["picks"]] == ["h2h", "spreads", "totals"]
    ml, spread, total = out["picks"]
    assert ml["label"] == "Boston Celtics to win" and ml["market_chance"] is not None
    assert spread["ref"]["point"] == -6.5 and total["ref"]["point"] == 221.5  # the board's main lines
    assert {p["book"] for p in out["picks"]} <= {"DraftKings", "FanDuel"}  # never Pinnacle
    assert out["game"] == {"label": "New York Knicks @ Boston Celtics", "start": at(1), "started": False}
    assert out["as_of"] == "2026-10-26T21:50:00Z"  # the saved board's time, ten minutes before NOW
    assert "unpriced" not in out and "loading" not in out and world.queued == []
    json.dumps(out)


def unpriced_as(code: str, stale: bool = False) -> Dict[str, str]:
    """game_picks' ``unpriced`` for a PRICE_REASONS key: the Builder's words, never "ask again" or "come back"; a game
    whose prices are all past their time says they're out of date."""
    what = "This game's prices are out of date" if stale else "No prices are loaded for this game"
    return {"reason": f"{what}: {PRICE_REASONS[code]}", "code": code}


def test_a_game_without_prices_says_so_and_asks_the_slate_s_loader_once(world) -> None:
    # The NHL's board was saved after the slate's last round (over 2 hours ago): its games are on the slate by their
    # ESPN ids only, so the Builder has no legs for them yet
    nhl = [event(NHL, "h1", "Boston Bruins", "Toronto Maple Leafs", at(2),
                 [book("draftkings", "DraftKings", "Boston Bruins", "Toronto Maple Leafs", (1.80, 2.05))])]
    world.store.save_odds_snapshot(NHL, nhl, NOW - 3 * 3600)
    answers: List[Dict[str, Any]] = []
    real = world.slate.request_odds
    world.slate.request_odds = lambda league: answers.append({"league": league, **real(league)}) or answers[-1]
    out = world.lab.game_picks(UID, NHL, "h1")
    assert out["picks"] == [] and out["game"] == {"label": "Toronto Maple Leafs @ Boston Bruins", "start": at(2),
                                                  "started": False}
    [asked] = answers  # the slate's throttled background loader, asked once for the league's board
    assert (asked["league"], asked["code"]) == (NHL, "queued") and world.queued == [NHL]
    # The board was queued, and the wait for it ended with it still loading: "on_way" (the app looks again soon)
    assert out["unpriced"] == unpriced_as("on_way")
    assert "again" not in out["unpriced"]["reason"] and "later" not in out["unpriced"]["reason"]
    again = world.lab.game_picks(UID, NHL, "h1")
    assert len(answers) == 2 and answers[1]["code"] == "loading" and world.queued == [NHL]
    assert again["unpriced"] == unpriced_as("on_way")
    assert world.odds.fetches == []  # nothing fetched on the request path (StoreOdds.league_games fails the test too)


@pytest.mark.parametrize("answer, code", [
    ({"queued": False, "code": "throttled", "reason": "Asked for less than 2 hours ago."}, "arriving"),
    ({"queued": False, "code": "fresh", "reason": "The odds are up to date."}, "arriving"),
    (RuntimeError("the slate is down"), "arriving"),
    ({"queued": False, "code": "reserve", "reason": "The month's odds allowance is down to its reserve."}, "reserve"),
    ({"queued": False, "code": "daily_cap", "reason": "Today's odds loads are used up."}, "daily_cap"),
    ({"queued": False, "code": "no_games", "reason": "No games coming up in this league."}, "no_games"),
    ({"queued": False, "code": "not_in_season", "reason": "No odds market for this league right now."}, "no_games"),
])
def test_unpriced_carries_the_builder_s_why_as_its_code(world, answer, code) -> None:
    sharp_only = [event(NHL, "h1", "Boston Bruins", "Toronto Maple Leafs", at(2),
                        [book("pinnacle", "Pinnacle", "Boston Bruins", "Toronto Maple Leafs", (1.80, 2.05))])]
    world.store.save_odds_snapshot(NHL, sharp_only, NOW - 600)  # only a sharp book prices it: nothing to bet
    asked: List[str] = []

    def request_odds(league: str) -> Dict[str, Any]:
        asked.append(league)
        if isinstance(answer, Exception):
            raise answer
        return dict(answer)

    world.slate.request_odds = request_odds
    out = world.lab.game_picks(UID, NHL, "h1")
    assert out["picks"] == [] and out["unpriced"] == unpriced_as(code)
    # Asked for the league's board only: by _ensure_prices first, and again for the words once nothing was waited for
    assert set(asked) == {NHL} and 1 <= len(asked) <= 2
    # The bot's process has no slate: nothing is asked, and the prices simply haven't arrived
    world.service.slate = None
    before = len(asked)
    out = world.lab.game_picks(UID, NHL, "h1")
    assert out["unpriced"] == unpriced_as("arriving") and len(asked) == before
    assert out["game"]["label"] == "Toronto Maple Leafs @ Boston Bruins"


@pytest.mark.parametrize("price_wait, answer, code", [
    ({"answers": {NHL: {"code": "queued"}}, "outcome": {NHL: "loading"}}, None, "on_way"),  # the wait ran out
    (None, {"code": "queued"}, "on_way"),
    (None, {"code": "loading"}, "on_way"),
    ({"answers": {NHL: {"code": "queued"}}, "outcome": {NHL: "none"}}, {"code": "throttled"}, "arriving"),
    ({"answers": {NHL: {"code": "queued"}}, "outcome": {NHL: "arrived"}}, {"code": "fresh"}, "arriving"),
    (None, {"code": "fresh"}, "arriving"),
    (None, {"code": "throttled"}, "arriving"),
    (None, {"code": "error", "reason": "Couldn't ask for odds right now."}, "arriving"),
    (None, {"code": "reserve"}, "reserve"),
    (None, {"code": "daily_cap"}, "daily_cap"),
    (None, {"code": "no_games"}, "no_games"),
    (None, {"code": "not_in_season"}, "no_games"),
])
def test_the_code_is_the_same_why_the_builder_s_unpriced_entry_carries(world, price_wait, answer, code) -> None:
    """Each loader answer (and how a wait for it ended) maps to the code the Builder's own _unpriced gives the league."""
    asked: List[str] = []
    world.slate.request_odds = lambda league: asked.append(league) or dict(answer)
    assembled: Dict[str, Any] = {"price_wait": price_wait} if price_wait else {}
    out = world.lab._game_unpriced(NHL, assembled)
    assert out == unpriced_as(code) and asked == ([] if answer is None else [NHL])
    # The Builder's entry for the same league and answers says the same
    builder = {**assembled, "leagues": {NHL: {"title": "NHL", "games": [at(2)], "priced": 0, "odds": "none"}}}
    [entry] = world.lab._unpriced(builder, world.lab.builder_controls({"preset": "risky"}), {NHL})
    assert entry["why"] == out["code"]


def stale_board(world, age: float) -> None:
    """The NBA board saved ``age`` seconds before NOW (past the odds' 2 hours), the slate showing it "stale"."""
    world.store.save_odds_snapshot(NBA, nba_board(), NOW - age)
    world.slate.refresh()


def spend_todays_boards(world) -> None:
    """Today's odds budget used up: the store's count of the boards taken today (both processes keep one)."""
    day = datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc).date().isoformat()
    for _ in range(50):
        world.store.take_odds_board(day, "request", 999)


@pytest.mark.parametrize("hours, setup, code", [
    (2.5, "queued", "on_way"),  # the loader puts the board on its way: the app looks again
    (2.5, "daily_cap", "daily_cap"),
    (3.5, "daily_cap", "daily_cap"),  # past the Builder's own 3-hour limit too
    (2.5, "reserve", "reserve"),
])
def test_a_board_past_its_two_hours_is_unpriced_with_the_loader_s_answer(world, hours, setup, code) -> None:
    """Review F2 (scratchpad/review-gp/checks/c_stale.py): every leg of a board over 2 hours old is stale, so the engine
    picks nothing; ``unpriced`` says why ("on_way" while the board is on its way, the day's budget or the month's
    reserve), where it used to say nothing."""
    if setup == "daily_cap":
        spend_todays_boards(world)
    elif setup == "reserve":
        world.odds.quota["remaining"] = world.odds.reserve - 10  # the month's allowance is at its reserve
    stale_board(world, hours * 3600)
    out = world.lab.game_picks(UID, NBA, "n1")
    assert out["picks"] == [] and out["unpriced"] == unpriced_as(code, stale=True)
    assert (NBA in world.queued) is (setup == "queued")  # through the slate's loader only
    assert world.odds.fetches == []


def test_a_fresh_board_s_picks_are_never_unpriced(world) -> None:
    out = world.lab.game_picks(UID, NBA, "n1")
    assert out["picks"] and "unpriced" not in out and world.queued == []
    stale_board(world, 1.5 * 3600)  # an hour and a half: still within the odds' 2 hours
    out = world.lab.game_picks(UID, NBA, "n1")
    assert out["picks"] and "unpriced" not in out


def test_a_board_on_its_way_gets_a_few_seconds_before_the_picks_are_chosen(world, monkeypatch) -> None:
    """fast-odds' _ensure_prices (the Builder's own wait for boards on their way) runs for the game's league before the
    picks are chosen, with the Lab's price_wait, and the legs it assembles again are the ones picked from. A game under
    way never waits."""
    nhl = [event(NHL, "h1", "Boston Bruins", "Toronto Maple Leafs", at(2),
                 [book("draftkings", "DraftKings", "Boston Bruins", "Toronto Maple Leafs", (1.80, 2.05))])]
    world.store.save_odds_snapshot(NHL, nhl, NOW - 3 * 3600)
    arrived = Candidate(ref={"league": NHL, "id": "h1", "market": "h2h", "side": "home", "point": None}, league=NHL,
                        game_key=f"{NHL}:h1", teams=("Toronto Maple Leafs", "Boston Bruins"), start=at(2),
                        price=Decimal("1.80"), book="DraftKings", chance=0.55, fair=0.54, price_age_s=5)
    calls: List[tuple] = []

    def ensure(assembled, leagues, *, wait):
        calls.append((list(leagues), wait))
        return {**assembled, "candidates": list(assembled["candidates"]) + [arrived]}  # the board came in

    monkeypatch.setattr(world.lab, "_ensure_prices", ensure)
    out = world.lab.game_picks(UID, NHL, "h1")
    assert calls == [([NHL], world.lab.price_wait)]
    assert [p["key"] for p in out["picks"]] == [arrived.key] and "unpriced" not in out
    assert out["as_of"] == "2026-10-26T21:59:55Z"  # the new board's prices: 5 seconds old
    later = later_lab(world, 1.5 * 3600)  # n1 is under way
    monkeypatch.setattr(later, "_ensure_prices", ensure)
    assert later.game_picks(UID, NBA, "n1")["game"]["started"] is True and len(calls) == 1


def test_a_game_that_has_started_has_no_picks(world) -> None:
    lab = later_lab(world, 1.5 * 3600)  # n1 tipped off half an hour ago
    out = lab.game_picks(UID, NBA, "n1")
    assert out["picks"] == [] and out["game"]["started"] is True and "unpriced" not in out and "loading" not in out
    assert out["props"] == {"likeliest": [], "edges": [], "day": "today"}
    assert world.board.calls == [] and world.queued == []
    assert lab.game_picks(UID, NBA, "n2")["picks"]  # still to come


def test_a_game_nothing_in_hand_knows_is_not_found(world) -> None:
    for game_id in ("nope", "bad id!", ""):
        with pytest.raises(NotFound):
            world.lab.game_picks(UID, NBA, game_id)
    with pytest.raises(CallRejected, match="League keys"):
        world.lab.game_picks(UID, "NBA!", "n1")


def test_props_are_this_game_s_best_from_the_player_board(world) -> None:
    out = world.lab.game_picks(UID, NBA, "n1")
    assert world.board.calls == [(NBA, "today", True, True)]  # today's board, started if need be, props read again
    props = out["props"]
    assert set(props) == {"likeliest", "edges", "day"} and props["day"] == "today"
    likeliest, edges = props["likeliest"], props["edges"]
    assert len(likeliest) == GAME_PROPS == 3 and 1 <= len(edges) <= GAME_PROPS
    for row in likeliest + edges:
        assert {"stat", "stat_label", "side", "line", "chance", "name", "team", "rank", "why", "hits"} <= set(row)
        assert row["event_id"] == "400" and row["league"] == NBA and row["addable"] is False
        assert row["line"]["stale"] is False and row["side"] in ("over", "under")
    for rows in (likeliest, edges):
        assert [row["rank"] for row in rows] == list(range(1, len(rows) + 1))
    assert [row["chance"] for row in likeliest] == sorted((row["chance"] for row in likeliest), reverse=True)
    assert [row["edge"] for row in edges] == sorted((row["edge"] for row in edges), reverse=True)
    assert all(row["edge"] > 0 and not row.get("edge_suspect") for row in edges)
    assert [(row["name"], row["stat_label"], row["side"], row["line"]["line"]) for row in likeliest] == [
        ("Jaylen Brown", "points", "under", 23.5), ("Jayson Tatum", "points", "over", 24.5),
        ("Jalen Brunson", "assists", "over", 6.5)]
    names = {row["name"] for row in likeliest + edges}
    assert "Nikola Jokic" not in names and "OG Anunoby" not in names  # another game's; ruled out
    assert "Josh Hart" not in {row["name"] for row in edges}  # a suspect edge never ranks
    assert len({(row["athlete_id"], row["stat"]) for row in likeliest}) == len(likeliest)  # one a player and stat


def test_likeliest_never_suggests_a_line_without_a_price_on_its_side(world) -> None:
    """A likeliest row is a bet a person can place: a line with a price on the side suggested (as the app's lineParts
    reads it). Porzingis' points over 12.5 (no price at all) and Horford's points under 9.5 (only the over is priced)
    are the likeliest two lines on the board; neither is suggested, and their slots go to the next priced lines."""
    assembled = world.lab.game_picks(UID, NBA, "n1")["props"]["likeliest"]
    shown = [(row["name"], row["side"]) for row in assembled]
    assert ("Kristaps Porzingis", "over") not in shown and ("Al Horford", "under") not in shown
    assert shown == [("Jaylen Brown", "under"), ("Jayson Tatum", "over"), ("Jalen Brunson", "over")]
    for row in assembled:
        assert (row["line"]["under"] if row["side"] == "under" else row["line"]["over"]) is not None
    # Unpriced, they would have led: the real engine ranks them likeliest
    from telegram_betting_bot_simple.engine import scan as scan_engine
    view = board_snapshot()
    view["players"] = [p for p in view["players"] if p["event_id"] == "400"]
    porzingis = next(r for r in scan_engine.scan(view, scan_engine.make_ask(NBA, "pts", None, "over"), 25)["players"]
                     if r["name"] == "Kristaps Porzingis")
    horford = next(r for r in scan_engine.scan(view, scan_engine.make_ask(NBA, "pts", None, "under"), 25)["players"]
                   if r["name"] == "Al Horford")
    assert min(porzingis["chance"], horford["chance"]) > max(row["chance"] for row in assembled)
    # Each kind of line, read as the app reads it: a yes bet's price (over only), a milestone's over (no under)
    priced = Lab._side_priced
    over_only = {"kind": "over_under", "over": {"decimal": 1.95}, "under": None, "yes": None}
    assert priced(over_only, "over") and not priced(over_only, "under")
    assert priced({"kind": "yes", "over": None, "under": None, "yes": {"decimal": 2.4}}, "over")
    assert not priced({"kind": "yes", "over": None, "under": None, "yes": {"decimal": 2.4}}, "under")
    assert not priced({"kind": "yes", "over": None, "under": None, "yes": None}, "over")
    assert priced({"kind": "milestone", "over": {"decimal": 1.3}, "under": None, "yes": None}, "over")
    assert not priced({"kind": "milestone", "over": {"decimal": 1.3}, "under": {"decimal": 3.0}, "yes": None}, "under")


def test_stale_lines_are_never_suggested(world) -> None:
    world.board.view["games"]["400"]["props"]["stale"] = True  # DraftKings' copy is past its time to live
    props = world.lab.game_picks(UID, NBA, "n1")["props"]
    assert props["likeliest"] == [] and props["edges"] == []


def test_a_props_failure_never_holds_the_game_s_picks_back(world, monkeypatch) -> None:
    from telegram_betting_bot_simple.engine import scan as scan_engine

    class Broken:
        def snapshot(self, *args, **kwargs):
            raise RuntimeError("ESPN board gone")

    world.lab.player_board = Broken()
    out = world.lab.game_picks(UID, NBA, "n1")
    assert out["picks"] and out["props"] == {"likeliest": [], "edges": [], "day": "today"} and "loading" not in out
    world.lab.player_board = world.board
    monkeypatch.setattr(scan_engine, "scan_edges", lambda *args, **kwargs: 1 / 0)
    out = world.lab.game_picks(UID, NBA, "n1")
    assert out["picks"] and out["props"]["likeliest"] == [] and out["props"]["edges"] == []


@pytest.mark.parametrize("state, loading", [
    ({"loaded": False, "building": True, "games": {}, "players": [], "games_ready": 0, "games_total": 0},
     {"games_ready": 0, "games_total": 0}),  # the day's games are still being read
    ({"games_ready": 3, "games_total": 5}, {"games_ready": 3, "games_total": 5}),
    ({"games_ready": 5, "games_total": 5, "games_failed": 1}, {"games_ready": 5, "games_total": 5, "failed": 1}),
    ({"games_ready": 5, "games_total": 5}, None),
])
def test_loading_says_the_board_is_still_building_as_the_player_scan_does(world, state, loading) -> None:
    world.board.view.update(state)
    out = world.lab.game_picks(UID, NBA, "n1")
    assert out.get("loading") == loading and "loading" not in out["props"]  # at the top level, as the spec has it
    assert out["picks"]  # the game's picks never wait for the board


def test_props_come_from_today_s_or_tomorrow_s_board_only(world) -> None:
    out = later_lab(world, -24 * 3600).game_picks(UID, NBA, "n1")  # a day earlier: n1 is tomorrow's
    assert out["props"]["day"] == "tomorrow" and world.board.calls == [(NBA, "tomorrow", True, True)]
    out = later_lab(world, -72 * 3600).game_picks(UID, NBA, "n1")  # three days ahead: no board is read or built
    assert out["props"] == {"likeliest": [], "edges": [], "day": None} and out["picks"] and len(world.board.calls) == 1
    # Soccer: the Player finder's day, but DraftKings posts no player lines for it here, so no board is read
    assert world.lab.game_picks(UID, EPL, "s1")["props"] == {"likeliest": [], "edges": [], "day": "today"}
    assert len(world.board.calls) == 1
    assert world.lab._props_day("mma_mixed_martial_arts", at(1)) is None  # a league the scan doesn't cover


def test_without_the_slate_espn_s_scoreboard_names_the_game(world) -> None:
    class FindEspn:
        def __init__(self) -> None:
            self.asked: List[tuple] = []

        def find_game(self, league, home, away, commence_time):
            self.asked.append((league, home, away, commence_time))
            return {"event_id": "400", "home_id": "1", "away_id": "2"}

    world.service.slate = None  # the bot's process: no slate, the saved boards' games
    world.service.espn = FindEspn()
    out = world.lab.game_picks(UID, NBA, "n1")
    assert world.service.espn.asked == [(NBA, "Boston Celtics", "New York Knicks", at(1))]
    assert out["picks"] and out["props"]["likeliest"] and {row["event_id"] for row in out["props"]["likeliest"]} == {"400"}
    assert "unpriced" not in out


# --- GET /api/game/picks -------------------------------------------------------------------------------------------

def picks_of(client: TestClient, league: str = NBA, game_id: str = "n1"):
    return client.get("/api/game/picks", params={"league": league, "id": game_id})


def test_the_route_answers_with_the_lab_s_picks(world) -> None:
    response = picks_of(world.client())
    assert response.status_code == 200, response.text
    data = response.json()
    assert set(data) == {"picks", "props", "game", "as_of"}
    assert data == json.loads(json.dumps(world.lab.game_picks(UID, NBA, "n1")))
    assert [p["market"] for p in data["picks"]] == ["h2h", "spreads", "totals"]


def test_the_route_is_checked_like_api_game(world) -> None:
    client = world.client()
    for params in ({"league": NBA}, {"id": "n1"}, {"league": "NBA", "id": "n1"}, {"league": "b", "id": "n1"},
                   {"league": NBA, "id": "x" * 65}):
        assert client.get("/api/game/picks", params=params).status_code == 422, params
        assert client.get("/api/game", params=params).status_code == 422, params  # the same checks
    missing = picks_of(client, game_id="nope")
    assert missing.status_code == 404 and missing.json()["detail"] == "No game nope in basketball_nba"
    anonymous = TestClient(create_app(world.service, "s3cret", login_delay=0, lab=world.lab, slate=world.slate))
    assert picks_of(anonymous).status_code == 401
    unnamed = world.client(player={}, confirm_age=False)
    assert picks_of(unnamed).status_code == 400 and picks_of(unnamed).json()["detail"] == "Missing player id"


class FeedOnly:
    """An odds feed as the kit's launchers have it (scratchpad/kit/launch/s1_e2e_server.py): league_games from its own
    copy (never the network), loaded five minutes ago, and no saved boards (no saved_games): the Builder has no legs
    from it, and the games service has no slate."""

    odds_ttl = 7200
    reserve = 50
    quota = {"remaining": 400, "used": 100}

    def __init__(self, games: List[Dict[str, Any]]) -> None:
        self.games = games
        self.at = time.time() - 300

    def sports(self):
        return [{"key": NBA, "group": "Basketball", "title": "NBA", "active": True, "has_outrights": False}]

    def league_games(self, league, refresh=False):
        assert not refresh
        return copy.deepcopy(self.games) if league == NBA else []

    def odds_loaded_at(self, league):
        return self.at if league == NBA else None


def test_every_game_the_game_route_shows_has_a_picks_answer(tmp_path) -> None:
    """The kit's s1 walk: the game page opened, then GET /api/game/picks answered 404 (the Builder had no legs, no
    slate had a row and the feed has no saved board). The picks route finds the games the game route finds: one it
    can't price is a 200 with no picks and an honest ``unpriced``; only an id the game route 404s is a 404."""
    store = Store(str(tmp_path / "betting.db"))
    service = DashboardService(FeedOnly(nba_board(in_2030)), QuietEspn(), NY, store=store, cache=TTLCache())
    lab = Lab(service, store, now=lambda: NOW)
    client = TestClient(create_app(service, "s3cret", login_delay=0, lab=lab))
    assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
    client.headers.update(PLAYER_A)
    assert client.put("/api/me", json={"confirm_age": True}).status_code == 200
    assert service.builder_candidates(now=NOW)["candidates"] == [] and service.slate is None
    for game_id, home, away in (("n1", "Boston Celtics", "New York Knicks"), ("n4", "Phoenix Suns", "Portland Trail Blazers")):
        assert client.get("/api/game", params={"league": NBA, "id": game_id}).status_code == 200
        response = picks_of(client, game_id=game_id)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["picks"] == [] and data["unpriced"] == unpriced_as("arriving")
        assert data["game"] == {"label": f"{away} @ {home}", "start": in_2030(int(game_id[1:])), "started": False}
    for route in ("/api/game", "/api/game/picks"):  # an id the game route doesn't know: both 404
        assert client.get(route, params={"league": NBA, "id": "zz9"}).status_code == 404, route


def test_a_paused_person_gets_the_400_every_pick_route_gives(world) -> None:
    unconfirmed = world.client(player=PLAYER_B, confirm_age=False)
    refused = picks_of(unconfirmed)
    assert refused.status_code == 400 and "18+" in refused.json()["detail"]
    client = world.client()
    world.lab.pause_picks(UID, NOW + 3600)
    picks, run = picks_of(client), client.post("/api/builder/run", json={})
    assert picks.status_code == run.status_code == 400
    assert picks.json() == run.json() and picks.json()["detail"].startswith("Picks are paused until")
    assert world.board.calls == []


def test_a_break_stops_the_route(world) -> None:
    client = world.client()
    assert client.post("/api/me/cooloff", json={"hours": 24}).status_code == 200
    response = picks_of(client)
    assert response.status_code == 400 and "break" in response.json()["detail"]


# --- the bot's "Suggested picks" block -------------------------------------------------------------------------------

def canned_pick(market: str, side: str, chance: float, market_chance: Optional[float], price: str, book_title: str,
                point: Optional[float] = None, label: str = "") -> Dict[str, Any]:
    return {"ref": {"league": NBA, "id": "g1", "market": market, "side": side, "point": point},
            "key": f"{NBA}:g1:{market}:{side}", "market": market, "market_title": MARKET_TITLES[market],
            "label": label or f"{side} pick", "chance": chance, "market_chance": market_chance, "price": price,
            "book": book_title, "edge": 0.01, "brain_higher": False, "why": ["a reason"]}


def canned_row(name: str, stat_label: str, side: str, line: float, chance: float,
               edge: Optional[float] = None) -> Dict[str, Any]:
    row = {"athlete_id": name, "name": name, "team": "OKC", "stat": stat_label, "stat_label": stat_label, "side": side,
           "chance": chance, "kind": "count", "line": {"kind": "over_under", "line": line, "stale": False}}
    if edge is not None:
        row["edge"] = edge
    return row


# Lab.game_picks' answer for the conftest's Clippers @ Thunder game (g1)
CANNED: Dict[str, Any] = {
    "picks": [
        canned_pick("h2h", "home", 0.64, 0.58, "1.72", "FanDuel"),
        canned_pick("spreads", "home", 0.55, 0.52, "1.91", "DraftKings", -13.5),
        canned_pick("totals", "under", 0.53, None, "1.95", "<Book & Co>", 224.5),
        canned_pick("btts", "yes", 0.51, 0.50, "1.90", "DraftKings", label="Both teams to score: Yes"),  # a 4th
    ],
    "props": {
        "likeliest": [canned_row("Shai Gilgeous-Alexander", "points", "over", 30.5, 0.61),
                      canned_row("Jalen Williams", "rebounds", "under", 5.5, 0.6)],
        "edges": [canned_row("<script>alert(1)</script>", "assists", "under", 6.5, 0.58, edge=0.042)],
        "day": "today"},
    "game": {"label": "Los Angeles Clippers @ Oklahoma City Thunder", "start": "2030-01-01T00:00:00Z",
             "started": False},
    "as_of": "2029-12-31T23:00:00Z",
}
CANNED_LINES = [
    "Thunder to win: 64% (market 58%), 1.72 at FanDuel",
    "Thunder -13.5: 55% (market 52%), 1.91 at DraftKings",
    "Under 224.5: 53%, 1.95 at &lt;Book &amp; Co&gt;",
    "Shai Gilgeous-Alexander points over 30.5: 61%",
    "&lt;script&gt;alert(1)&lt;/script&gt; assists under 6.5: 58% (edge +4.2%)",
]


def test_the_card_gains_a_short_suggested_picks_block(service) -> None:
    card = service.game(NBA, "g1")
    plain = game_card_text(card, "7:00 PM EDT")
    text = game_card_text(card, "7:00 PM EDT", CANNED)
    assert suggested_entries(card, CANNED) == CANNED_LINES  # 3 picks, then 2 props (the likeliest and an edge in turn)
    block = "\n".join(["", SUGGESTED_HEAD] + CANNED_LINES + [NOTE])
    assert block in text and text.replace("\n" + block, "", 1) == plain  # inserted whole; the card itself unchanged
    assert text.index(SUGGESTED_HEAD) < text.index("Odds as of 7:00 PM EDT")
    assert "<script>" not in text and "<Book" not in text  # escaped
    # No block: picks refused (None), a game under way, or nothing to suggest
    assert game_card_text(card, "7:00 PM EDT", None) == plain
    assert game_card_text(card, "7:00 PM EDT", {**CANNED, "game": {**CANNED["game"], "started": True}}) == plain
    assert game_card_text(card, "7:00 PM EDT", {"picks": [], "props": {"likeliest": [], "edges": []}}) == plain


TELEGRAM_TAGS = frozenset({"b", "strong", "i", "em", "u", "ins", "s", "strike", "del", "a", "code", "pre",
                           "tg-spoiler", "span", "blockquote"})


def assert_telegram_html(text: str) -> None:
    """Text Telegram's HTML parse mode takes: only its tags, each closed in order, and every &, < and > outside a tag
    written as an entity (a bare one makes Telegram refuse the whole message)."""
    class Check(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=False)
            self.open: List[str] = []

        def handle_starttag(self, tag, attrs):
            assert tag in TELEGRAM_TAGS, f"<{tag}> isn't one of Telegram's tags"
            self.open.append(tag)

        def handle_endtag(self, tag):
            assert self.open and self.open[-1] == tag, f"</{tag}> closes nothing open ({self.open})"
            self.open.pop()

    check = Check()
    check.feed(text)
    check.close()
    assert check.open == [], f"left open: {check.open}"
    assert not re.search(r"&(?!(?:amp|lt|gt|quot|#\d+|#x[0-9a-fA-F]+);)", text), "a bare & in the card"
    assert ">" not in re.sub(r"<[^<>]*>", "", text), "a bare > in the card"


def test_the_bot_escapes_every_name_in_the_block(service) -> None:
    """Review F3: team names and labels come from the odds feed ("Brighton & Hove Albion", "Texas A&M"). Each is
    escaped in the block, and the card stays HTML Telegram takes."""
    card = {**service.game(NBA, "g1"), "league": "soccer_epl", "home": "Brighton & Hove Albion", "away": "Wolves <U21>"}
    picks = {**CANNED, "props": {"likeliest": [], "edges": []}, "picks": [
        canned_pick("h2h", "home", 0.64, 0.58, "1.72", "FanDuel"),
        canned_pick("spreads", "away", 0.55, 0.52, "1.91", "DraftKings", 0.5),
        canned_pick("draw_no_bet", "away", 0.53, None, "1.95", "DraftKings", label="Wolves <U21> (draw no bet)")]}
    assert suggested_entries(card, picks) == ["Brighton &amp; Hove Albion to win: 64% (market 58%), 1.72 at FanDuel",
                                              "Wolves &lt;U21&gt; +0.5: 55% (market 52%), 1.91 at DraftKings",
                                              "Wolves &lt;U21&gt; (draw no bet): 53%, 1.95 at DraftKings"]
    assert_telegram_html(game_card_text(card, "7:00 PM EDT", picks))
    # A US team keeps its & once the card shortens it to the last word ("Texas A&M" -> "A&M")
    ncaaf = {**card, "league": "americanfootball_ncaaf", "home": "Texas A&M", "away": "LSU Tigers"}
    assert suggested_entries(ncaaf, {**picks, "picks": picks["picks"][:1]}) == [
        "A&amp;M to win: 64% (market 58%), 1.72 at FanDuel"]
    assert_telegram_html(game_card_text(ncaaf, None, picks))
    assert_telegram_html(game_card_text(service.game(NBA, "g1"), "7:00 PM EDT", CANNED))


def test_the_block_gives_way_to_the_card_s_own_content_within_telegram_s_limit(service) -> None:
    card = service.game(NBA, "g1")
    entries = suggested_entries(card, CANNED)

    def padded(extra: int) -> Dict[str, Any]:  # the card with an Out list ``extra`` characters longer
        out = copy.deepcopy(card)
        out["injuries"] = {"away": ["X" * extra + " (Out)"], "home": []}
        return out

    def block(count: int) -> int:  # the characters a block of ``count`` entries adds
        return 1 + len("\n".join(["", SUGGESTED_HEAD] + entries[:count] + [NOTE]))

    base = len(game_card_text(padded(0), None))
    room = padded(4096 - base - block(2))  # exactly room for the block with two entries
    text = game_card_text(room, None, CANNED)
    assert len(text) == 4096 and entries[0] in text and entries[1] in text and entries[2] not in text
    assert text.replace("\n" + "\n".join(["", SUGGESTED_HEAD] + entries[:2] + [NOTE]), "", 1) == \
        game_card_text(room, None)  # the card's own content is all there
    tight = padded(4096 - base - block(1) + 1)  # no room even for one entry: the card alone
    assert game_card_text(tight, None, CANNED) == game_card_text(tight, None) and len(game_card_text(tight, None)) <= 4096
    big = padded(5000)  # a card over the limit on its own stays the card it was
    assert game_card_text(big, None, CANNED) == game_card_text(big, None)


class Accounts:
    """The store as the bot reads a person's account (store.user)."""

    def __init__(self, accounts: Dict[str, Dict[str, Any]]) -> None:
        self.accounts = accounts

    def user(self, user_id: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        return self.accounts.get(user_id)


class FakeLab:
    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.calls: List[tuple] = []

    def game_picks(self, uid: str, league: str, event_id: str) -> Dict[str, Any]:
        self.calls.append((uid, league, event_id))
        if isinstance(self.answer, Exception):
            raise self.answer
        return copy.deepcopy(self.answer)


class BotService:
    """The games service as show_game reads it (games, game), with an account store and a Lab of the test's own."""

    def __init__(self, games: Any, store: Any, lab: Any) -> None:
        self.games_service, self.store, self.lab = games, store, lab

    def games(self, league: str, refresh: bool = False) -> Dict[str, Any]:
        return self.games_service.games(league)

    def game(self, league: str, event_id: str) -> Dict[str, Any]:
        return self.games_service.game(league, event_id)


def open_game(service: Any, game_id: str = "g1", user_id: Optional[int] = 42) -> Dict[str, Any]:
    """The bot's game card for ``game_id`` (its text and buttons), as person ``user_id`` (None: no Telegram user)."""
    application = Application.builder().token("123456:test").build()
    setup_handlers(application, Config(TELEGRAM_BOT_TOKEN="123456:test"))
    application.bot_data["service"] = service
    context = SimpleNamespace(application=application, user_data={})
    screens: List[Dict[str, Any]] = []

    class Query:
        data = f"g:{NBA}:{game_id}"

        async def answer(self, *args, **kwargs):
            pass

        async def edit_message_text(self, text, reply_markup=None, **kwargs):
            screens.append({"text": text, "buttons": [[(b.text, b.callback_data) for b in row]
                                                      for row in reply_markup.inline_keyboard]})

    update = SimpleNamespace(callback_query=Query(), message=None, effective_chat=None,
                             effective_user=SimpleNamespace(id=user_id) if user_id is not None else None)
    asyncio.run(handlers.game_selected(update, context))
    return screens[-1]


CONFIRMED = {TG: {"id": TG, "age_confirmed_at": 1.0}}


def test_the_bot_shows_the_block_to_a_person_the_gates_let_through(service) -> None:
    lab = FakeLab(CANNED)
    screen = open_game(BotService(service, Accounts(CONFIRMED), lab))
    assert lab.calls == [(TG, NBA, "g1")]
    assert SUGGESTED_HEAD in screen["text"] and all(line in screen["text"] for line in CANNED_LINES)
    assert len(screen["text"]) <= 4096


def test_a_paused_person_gets_no_block_and_the_pick_buttons_never_change(service) -> None:
    plain = open_game(BotService(service, Accounts({}), FakeLab(CANNED)), user_id=None)  # the card as it always was
    assert SUGGESTED_HEAD not in plain["text"]
    paused = FakeLab(CallRejected("Picks are paused until Mon Oct 26, 7:00 PM EDT after an earlier message about "
                                  "losses or money worries. The odds and the maths stay open."))
    assert open_game(BotService(service, Accounts(CONFIRMED), paused)) == plain and paused.calls == [(TG, NBA, "g1")]
    shown = open_game(BotService(service, Accounts(CONFIRMED), FakeLab(CANNED)))
    assert shown["buttons"] == plain["buttons"]  # the pick buttons' callback data is the same with the block
    picks = [data for row in shown["buttons"] for _, data in row if data.startswith("p:")]
    assert len(picks) == 6 and all(parse_pick(data) and len(data.encode()) <= 64 for data in picks)
    # No account (so no age confirmed): the Lab isn't asked and no account is made; a failing Lab: the card alone
    nobody = FakeLab(CANNED)
    assert open_game(BotService(service, Accounts({}), nobody)) == plain and nobody.calls == []
    assert open_game(BotService(service, Accounts(CONFIRMED), FakeLab(RuntimeError("boom")))) == plain
    assert open_game(BotService(service, Accounts(CONFIRMED), FakeLab(NotFound("No game")))) == plain


def test_the_bot_s_block_with_the_real_lab_and_its_gates(tmp_path) -> None:
    store = Store(str(tmp_path / "betting.db"))
    store.save_odds_snapshot(NBA, nba_board(in_2030), NOW - 600)
    games = DashboardService(BoardOdds(store), QuietEspn(), NY, store=store, cache=TTLCache())
    lab = Lab(games, store, now=lambda: NOW)  # games.lab is this Lab, as services.build_service sets it up
    lab.player_board = FakeBoard()
    assert open_game(games, "n1")["text"].count(SUGGESTED_HEAD) == 0  # no account yet: no block
    assert store.user(TG) is None  # and none was made
    store.get_or_create_user(TG, now=NOW)
    store.confirm_age(TG, now=NOW)
    text = open_game(games, "n1")["text"]
    assert SUGGESTED_HEAD in text and NOTE in text
    expected = suggested_entries(games.game(NBA, "n1"), lab.game_picks(TG, NBA, "n1"))
    assert expected[0].startswith("Celtics to win: ") and all(line in text for line in expected)
    lab.pause_picks(TG, NOW + 3600)  # a warning sign in the chat: picks paused for the day
    assert SUGGESTED_HEAD not in open_game(games, "n1")["text"]


# --- no Odds API request (the RD4 counter) --------------------------------------------------------------------------

def counting_world(tmp_path, monkeypatch, age: float) -> SimpleNamespace:
    """tests/test_review2_data.py's RD4 world: NbaWorld's ten NBA games over the real OddsService (its saved board ``age``
    seconds old, its _get recording every call and refusing it), the real games service and ESPN client (the captured
    fixtures) and the real player board, with the Lab on top and person TG's age confirmed."""
    nba = NbaWorld()
    t0 = time.time()
    lock = threading.Lock()
    clock = {"now": t0}

    def fake_time() -> float:
        with lock:
            return clock["now"]

    fake = SimpleNamespace(time=fake_time, monotonic=time.monotonic, sleep=time.sleep)
    monkeypatch.setattr(odds_mod, "time", fake)
    monkeypatch.setattr(pb_mod, "time", fake)
    web = Web(monkeypatch, nba.routes)
    store = Store(str(tmp_path / f"odds-{age:.0f}.db"))
    store.save_odds_snapshot(NBA, nba.odds_games(), t0 - age)
    spent: List[str] = []

    def no_api(self, path, params):
        spent.append(path)
        raise RuntimeError("would have spent Odds API credits")

    monkeypatch.setattr(OddsService, "_get", no_api)
    odds = OddsService("key", cache=TTLCache(name="odds", clock=fake_time), store=store)
    cache = TTLCache(name="espn")
    espn = EspnClient(cache, clock=lambda: SCAN_NOW)
    service = DashboardService(odds, espn, NY, store=store, cache=cache)
    players = PlayerData(espn, now=lambda: datetime.datetime.fromtimestamp(SCAN_NOW, datetime.timezone.utc))
    lab = Lab(service, store, players, now=lambda: SCAN_NOW)
    board = PlayerBoard(service, players, cache, now=lambda: SCAN_NOW)
    lab.player_board = board
    store.get_or_create_user(TG, now=SCAN_NOW)
    store.confirm_age(TG, now=SCAN_NOW)
    first = nba.events[0]
    return SimpleNamespace(web=web, spent=spent, service=service, lab=lab, board=board, odds_id=f"odds-{first['id']}",
                           espn_id=str(first["id"]))


def test_game_picks_never_sends_an_odds_api_request(tmp_path, monkeypatch) -> None:
    """With the saved board fresh (600 s) and past the odds service's time to live (7300 s, which any board load
    would fetch again), while the player board builds and once it has: Lab.game_picks reads the Builder's candidates
    from the saved board and the props from the ESPN-only board, and never one Odds API request is made (not even
    the free /sports listing). The counter is checked last: the game page's own board load is a request, and only
    within the day's budget."""
    for age in (600.0, 7300.0):
        w = counting_world(tmp_path, monkeypatch, age)
        try:
            building = w.lab.game_picks(TG, NBA, w.odds_id)  # starts the player board's build in the background
            assert [p["market"] for p in building["picks"]] == ["h2h", "spreads", "totals"], age
            assert building["props"]["day"] == "today" and "unpriced" not in building
            assert w.board.wait(NBA, "today", 120)
            built = w.lab.game_picks(TG, NBA, w.odds_id)
        finally:
            w.board.close()
        assert w.spent == [], (age, w.spent)
        assert "api.the-odds-api.com" not in w.web.hosts()
        rows = built["props"]["likeliest"] + built["props"]["edges"]
        assert built["props"]["likeliest"] and "loading" not in built, (age, built.get("loading"))
        assert {row["event_id"] for row in rows} == {w.espn_id}  # this game's players only
    # The counter counts, and the game page's own board load is within the day's budget (OddsService._take_board):
    # past the time to live, with today's boards used up it serves the saved board and asks for no board; with one
    # left it asks once (refused here, so that board is given back). The free /sports listing costs no credits
    store, day = w.service.store, odds_mod.budget_day(odds_mod.time.time())
    for _ in range(odds_mod.ODDS_PER_DAY):  # the month's count is unknown here: 12 boards a day
        assert store.take_odds_board(day, "request", 99)
    for left in (0, 1):
        if left:
            store.give_back_odds_board(day, "request")
            w.service.odds.cache.invalidate(("odds", NBA))
            w.service.cache.invalidate(("games", NBA))
        try:
            w.service.games(NBA)
        except RuntimeError:
            pass
        assert [path for path in w.spent if path.endswith("/odds")] == [f"/sports/{NBA}/odds"] * left, w.spent
    assert store.odds_boards(day) == {"request": odds_mod.ODDS_PER_DAY - 1}
