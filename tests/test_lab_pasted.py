"""Tests for pasted bet slips in the Lab and the API: matching legs to the boards, pricing at the person's own
odds, the checks that ask the person to confirm a doubtful read, the gates, privacy and the request limits.

Fake odds for several leagues (with which are cached and a log of every fetch that would cost quota), fake ESPN
and player data and a fake assistant: no network, no Anthropic calls, no Odds API quota. The clock is fixed on
a Tuesday morning in New York (2030-01-01), before the evening's games.
"""

from __future__ import annotations

import base64
import copy
import datetime
import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.dashboard.app import BODY_LIMITS, MAX_BODY_BYTES, MAX_SLIP_BODY_BYTES, create_app
from telegram_betting_bot_simple.dashboard.lab import AGE_MESSAGE, NO_CHAT, NO_SUCH_PASTE, Lab
from telegram_betting_bot_simple.dashboard.service import CallRejected, NotFound
from telegram_betting_bot_simple.data.odds_service import normalize_event
from telegram_betting_bot_simple.data.store import MAX_PASTED, Store
from telegram_betting_bot_simple.engine import calls, pasted
from telegram_betting_bot_simple.engine.odds_math import remove_vig
from tests.test_lab import Players

NBA, NHL, EPL, NFL, NCAAB, MLB = ("basketball_nba", "icehockey_nhl", "soccer_epl", "americanfootball_nfl",
                                  "basketball_ncaab", "baseball_mlb")
UID, OTHER = "web:browser-1234", "web:someone-else-99"
ME, THEM = {"X-Player": "browser-1234"}, {"X-Player": "someone-else-99"}
NOW = datetime.datetime(2030, 1, 1, 15, 0, tzinfo=datetime.timezone.utc).timestamp()  # Tue 10:00 AM in New York
ROOT = Path(__file__).resolve().parents[1]


def event(event_id: str, league: str, home: str, away: str, start: str, books: Dict[str, Any], total: float = 224.5,
          draw: Optional[float] = None) -> Dict[str, Any]:
    """An Odds API event: books {title: (home_ml, away_ml, home_spread or None, over_price)}."""
    bookmakers = []
    for title, (home_ml, away_ml, home_spread, over) in books.items():
        h2h = [{"name": home, "price": home_ml}, {"name": away, "price": away_ml}]
        if draw is not None:
            h2h.append({"name": "Draw", "price": draw})
        markets = [{"key": "h2h", "outcomes": h2h}]
        if home_spread is not None:
            markets.append({"key": "spreads", "outcomes": [{"name": home, "price": 1.91, "point": home_spread},
                                                          {"name": away, "price": 1.91, "point": -home_spread}]})
        markets.append({"key": "totals", "outcomes": [{"name": "Over", "price": over, "point": total},
                                                     {"name": "Under", "price": 1.91, "point": total}]})
        bookmakers.append({"key": title.lower(), "title": title, "markets": markets})
    return normalize_event({"id": event_id, "sport_key": league, "sport_title": league, "commence_time": start,
                            "home_team": home, "away_team": away, "bookmakers": bookmakers})


def boards() -> Dict[str, List[Dict[str, Any]]]:
    return {
        NBA: [
            event("n1", NBA, "Boston Celtics", "New York Knicks", "2030-01-02T00:30:00Z",  # Tue 7:30 PM ET
                  {"DraftKings": (1.62, 2.40, -3.5, 1.91), "FanDuel": (1.60, 2.45, -3.5, 1.93)}),
            event("n2", NBA, "Denver Nuggets", "Utah Jazz", "2030-01-02T02:00:00Z",
                  {"DraftKings": (1.25, 4.20, -8.5, 1.91)}, total=231.5),
            event("n3", NBA, "Phoenix Suns", "Sacramento Kings", "2030-01-02T03:00:00Z",
                  {"DraftKings": (1.80, 2.05, -2.5, 1.91), "FanDuel": (1.83, 2.02, -2.5, 1.91)}),
            event("n4", NBA, "Oklahoma City Thunder", "Los Angeles Lakers", "2030-01-02T01:00:00Z",
                  {"DraftKings": (1.40, 3.00, -7.5, 1.91)}),
            event("n5", NBA, "Miami Heat", "Chicago Bulls", "2030-01-01T14:00:00Z",  # an hour in already
                  {"DraftKings": (1.70, 2.20, -2.5, 1.91)}),
            event("n6", NBA, "Boston Celtics", "New York Knicks", "2030-01-05T00:30:00Z",  # Fri 7:30 PM ET, again
                  {"DraftKings": (1.55, 2.55, -4.5, 1.91)}),
        ],
        EPL: [
            event("e1", EPL, "Arsenal", "Chelsea", "2030-01-01T20:00:00Z",
                  {"DraftKings": (2.10, 3.60, None, 1.95)}, total=2.5, draw=3.40),
            event("e2", EPL, "Manchester City", "Brighton and Hove Albion", "2030-01-01T20:00:00Z",
                  {"DraftKings": (1.40, 7.00, None, 1.45)}, total=2.5, draw=5.00),
        ],
        NHL: [event("h1", NHL, "Boston Bruins", "Toronto Maple Leafs", "2030-01-02T00:00:00Z",
                    {"DraftKings": (1.70, 2.20, -1.5, 1.91)}, total=5.5)],
        NFL: [event("f1", NFL, "Kansas City Chiefs", "Buffalo Bills", "2030-01-02T01:15:00Z",
                    {"DraftKings": (1.75, 2.10, -2.5, 1.91)}, total=47.5)],
        NCAAB: [event("c1", NCAAB, "Duke Blue Devils", "North Carolina Tar Heels", "2030-01-02T00:00:00Z",
                      {"DraftKings": (1.50, 2.60, -4.5, 1.91)}, total=150.5)],
    }


class PasteOdds:
    """The odds service: several leagues in season (MLB out), which are cached, and a log of the fetches that
    would cost quota (a cached league's odds come back without one)."""

    odds_ttl = 7200
    reserve = 50

    def __init__(self, cached=()) -> None:
        self.games = boards()
        self.cached = set(cached)
        self.fetched: List[str] = []
        self.quota = {"remaining": 480, "used": 20}

    def sports(self):
        titles = {NBA: "NBA", NHL: "NHL", EPL: "EPL", NFL: "NFL", NCAAB: "NCAAB"}
        return [{"key": key, "group": "x", "title": title, "active": True, "has_outrights": False}
                for key, title in titles.items()] + [
            {"key": MLB, "group": "Baseball", "title": "MLB", "active": False, "has_outrights": False}]

    def league_games(self, league, refresh=False):
        if league not in self.cached:
            self.fetched.append(league)
            self.cached.add(league)  # as the odds service keeps what it fetched
        return copy.deepcopy(self.games.get(league, []))

    def odds_loaded_at(self, league):
        return time.time() - 600 if league in self.cached else None


class SlipAssistant:
    """The assistant's read_slip, recording what it was asked (it would read the slip with Claude)."""

    def __init__(self, refuse: Optional[Exception] = None) -> None:
        self.refuse = refuse
        self.asked: List[Dict[str, Any]] = []

    def read_slip(self, uid, thread_id=None, text=None, image=None, media_type=None, on_step=None):
        if self.refuse:
            raise self.refuse
        self.asked.append({"uid": uid, "thread_id": thread_id, "text": text, "image": image, "media_type": media_type})
        return {"thread_id": thread_id or 5, "message_idx": 1, "reply": {"text": "I read your slip as ...", "cards": [],
                                                                         "chips": []}, "credits": self.credits(uid)}

    def credits(self, uid):
        return {"used_cents": 1.0, "daily_cents": 100, "left_cents": 99.0}


@pytest.fixture()
def lab(service, tmp_path):
    store = Store(str(tmp_path / "betting.db"))
    service.odds = PasteOdds()
    service.store = store
    lab = Lab(service, store, players=Players(game_date="2030-01-02T01:00:00Z"), now=lambda: NOW)
    lab.update_me(UID, {"confirm_age": True})
    return lab


@pytest.fixture()
def client(lab):
    client = TestClient(create_app(lab.service, "s3cret", login_delay=0))
    assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
    client.headers.update(ME)
    return client


def leg(selection: str, market: str, odds_text: Optional[str], **extra: Any) -> Dict[str, Any]:
    return {"raw": f"{selection} {odds_text}", "selection": selection, "market": market, "odds_text": odds_text, **extra}


def read(lab: Lab, legs: List[Dict[str, Any]], source: str = "image", text: Optional[str] = None,
         uid: str = UID, **slip: Any) -> Dict[str, Any]:
    return lab.read_pasted(uid, {"legs": legs, **slip}, source, text=text)


def statuses(view: Dict[str, Any]) -> List[str]:
    return [item["match"]["status"] for item in view["legs"]]


US_PARLAY = """DraftKings
3 Leg Parlay +556
BOS Celtics -3.5 -110
NY Knicks @ BOS Celtics
Tue 7:30 PM
Over 231.5 -110
Utah Jazz @ Denver Nuggets
Phoenix Suns ML -125
Wager $10.00
To Win $55.60"""


# --- matching legs to the boards -----------------------------------------------------------------------

def test_legs_match_by_aliases_nicknames_and_a_draw(lab) -> None:
    view = read(lab, [
        leg("BOS Celtics", "spread", "-110", line=-3.5, event_text="NY Knicks @ BOS Celtics", sport_hint="NBA",
            start_text="Tue 7:30 PM"),
        leg("Knicks", "moneyline", "2.40", sport_hint="NBA"),
        leg("Over", "total", "1.91", line=231.5, over_under="over", event_text="Jazz @ Nuggets", sport_hint="NBA"),
        leg("Man City", "moneyline", "1.40", event_text="Man City v Brighton", sport_hint="Premier League"),
        leg("Draw", "moneyline", "3.40", event_text="Arsenal v Chelsea", sport_hint="EPL"),
    ])
    assert statuses(view) == ["matched"] * 5
    refs = [item["match"]["ref"] for item in view["legs"]]
    assert refs == [
        {"league": NBA, "id": "n1", "market": "spreads", "side": "home", "point": -3.5},
        {"league": NBA, "id": "n1", "market": "h2h", "side": "away", "point": None},  # the next game, not Friday's
        {"league": NBA, "id": "n2", "market": "totals", "side": "over", "point": 231.5},
        {"league": EPL, "id": "e2", "market": "h2h", "side": "home", "point": None},
        {"league": EPL, "id": "e1", "market": "h2h", "side": "draw", "point": None},
    ]
    assert view["matched_refs"] == refs
    first = view["legs"][0]
    assert first["match"]["game"] == "New York Knicks @ Boston Celtics" and first["match"]["start"] == "2030-01-02T00:30:00Z"
    assert first["match"]["live"] is False and first["chance_source"] in ("market", "model")
    assert first["odds"] == {"decimal": "1.91", "american": "-110", "fractional": "10/11"}
    # The draw is priced as one of three outcomes: its margin-free chance from the three-way market
    draw = view["legs"][4]
    assert draw["market_chance"] == pytest.approx(remove_vig([2.10, 3.60, 3.40])[2], abs=1e-4)
    assert draw["ai_chance"] is None and draw["chance_source"] == "market"
    assert draw["best_price"] == {"decimal": "3.40", "book": "DraftKings"} and draw["better_price"] is False
    assert view["legs"][1]["best_price"] == {"decimal": "2.45", "book": "FanDuel"}
    assert view["legs"][1]["better_price"] is True  # 2.45 beats the slip's 2.40 by 0.05


def test_a_start_time_picks_the_game_and_a_live_one_is_flagged(lab) -> None:
    view = read(lab, [leg("Celtics", "moneyline", "1.55", sport_hint="NBA", start_text="Fri 7:30 PM"),
                      leg("Chicago Bulls", "moneyline", "2.20", sport_hint="NBA")])
    friday, bulls = view["legs"]
    assert friday["match"]["ref"]["id"] == "n6" and friday["match"]["status"] == "matched"
    assert bulls["match"]["status"] == "matched" and bulls["match"]["live"] is True
    assert view["matched_refs"] == [friday["match"]["ref"]]  # a game in play can't be loaded into Build
    assert any("game has started" in warning for warning in view["warnings"])


def test_a_different_line_links_to_the_game_with_an_estimated_chance(lab) -> None:
    view = read(lab, [leg("Boston Celtics", "spread", "1.95", line=-2.5, event_text="Knicks @ Celtics", sport_hint="NBA"),
                      leg("Under", "total", "1.80", line=229.5, over_under="under", event_text="Jazz @ Nuggets",
                          sport_hint="NBA")])
    spread, total = view["legs"]
    assert statuses(view) == ["line_differs", "line_differs"]
    assert spread["match"]["ref"]["point"] == -3.5 and spread["match"]["game"] == "New York Knicks @ Boston Celtics"
    assert "The board has -3.5; you have -2.5" in spread["match"]["note"]
    assert "The board has Under 231.5; you have Under 229.5" in total["match"]["note"]
    assert spread["chance_source"] == "estimated" and spread["market_chance"] == pytest.approx((1 / 1.95) / 1.045,
                                                                                              abs=1e-4)
    assert view["matched_refs"] == []


def test_legs_not_on_any_board_are_estimated_and_say_so(lab) -> None:
    view = read(lab, [leg("Atlanta Hawks", "moneyline", "2.50", sport_hint="NBA"),
                      leg("Both teams to score - Yes", "other", "1.70", event_text="Arsenal v Chelsea",
                          sport_hint="EPL"),
                      leg("Over", "total", "1.90", line=220.5, over_under="over")])
    hawks, btts, total = view["legs"]
    assert statuses(view) == ["not_found", "unsupported", "not_found"]
    assert hawks["market_chance"] == pytest.approx((1 / 2.5) / 1.045, abs=1e-4) and hawks["chance_source"] == "estimated"
    assert "Not found on the app's boards" in hawks["match"]["note"] and "4.5%" in hawks["match"]["note"]
    assert btts["match"]["game"] == "Chelsea @ Arsenal" and btts["market_chance"] == pytest.approx((1 / 1.7) / 1.08, abs=1e-4)
    assert "doesn't name this leg's game" in total["match"]["note"]
    assert any("Legs 1, 2 and 3 aren't matched" in warning for warning in view["warnings"])


def test_an_nba_prop_takes_the_models_chance_and_the_markets_same_line(lab) -> None:
    row = {"id": "1966", "name": "LeBron James", "minutes": 34.0, "pts": 27.0, "reb": 7.5, "ast": 8.0, "fg3m": 2.0,
           "stl": 1.0, "blk": 0.5, "tov": 3.0, "status": "Active", "out": False, "side": "away"}
    lab.service.detail = lambda league, event_id: (None, None, {"1966": row} if event_id == "n4" else {})
    view = read(lab, [leg("LeBron James", "player_prop", "-125", line=25.5, prop_stat="points", over_under="over",
                          event_text="Lakers @ Thunder", sport_hint="NBA"),
                      leg("Patrick Mahomes", "player_prop", "-115", line=275.5, prop_stat="passing yards",
                          over_under="over", sport_hint="NFL")])
    lebron, mahomes = view["legs"]
    over, _, _ = calls.stat_probabilities("pts", 27.0, 25.5, 34.0)
    assert lebron["match"]["status"] == "prop" and lebron["chance_source"] == "model"
    assert lebron["ai_chance"] == pytest.approx(over, abs=1e-4)
    assert lebron["market_chance"] == pytest.approx(remove_vig([1.87, 1.95])[0], abs=1e-4)
    assert lebron["best_price"] == {"decimal": "1.87", "book": "DraftKings"} and lebron["better_price"] is True
    assert lebron["match"]["game"] == "Los Angeles Lakers @ Oklahoma City Thunder" and lebron["match"]["ref"] is None
    assert "The NBA model projects 27 points" in lebron["match"]["note"]
    assert mahomes["match"]["status"] == "prop" and mahomes["chance_source"] == "estimated"
    assert "NBA players only" in mahomes["match"]["note"]
    assert mahomes["market_chance"] == pytest.approx((1 / float(pasted.parse_odds("-115"))) / 1.08, abs=1e-4)
    analysis = lab.analyse_pasted(UID, view["id"])
    assert analysis["ticket"]["legs"][0]["ai_prob"] == pytest.approx(over, abs=1e-4)
    assert analysis["ticket"]["legs"][0]["market"] == "player_pts"


def test_cached_boards_come_first_then_the_favourites(lab) -> None:
    odds = lab.service.odds
    odds.cached = {NHL}
    lab.update_me(UID, {"favourite_sports": [NCAAB, NBA]})
    view = read(lab, [leg("Bruins", "moneyline", "1.70")])  # no league named: the cached NHL board has it
    assert view["legs"][0]["match"]["ref"]["league"] == NHL and odds.fetched == []
    view = read(lab, [leg("Knicks", "moneyline", "2.40")])  # not cached: the favourites load, in order
    assert view["legs"][0]["match"]["ref"]["id"] == "n1" and odds.fetched == [NCAAB, NBA]
    view = read(lab, [leg("Duke", "moneyline", "1.50"), leg("Celtics", "moneyline", "1.62")])
    assert statuses(view) == ["matched", "matched"] and odds.fetched == [NCAAB, NBA]  # cached now: no more fetches


def test_one_slip_loads_at_most_three_leagues_and_never_one_out_of_season(lab) -> None:
    odds = lab.service.odds
    view = read(lab, [leg("Chiefs", "moneyline", "1.75", sport_hint="NFL"),
                      leg("Arsenal", "moneyline", "2.10", sport_hint="EPL"),
                      leg("Celtics", "moneyline", "1.62", sport_hint="NBA"),
                      leg("Duke", "moneyline", "1.50", sport_hint="NCAAB"),
                      leg("Yankees", "moneyline", "1.90", sport_hint="MLB")])
    assert odds.fetched == [NFL, EPL, NBA]  # three loads, then no more; MLB is out of season, never fetched
    assert statuses(view) == ["matched", "matched", "matched", "not_found", "not_found"]
    assert "to save the monthly odds allowance" in view["legs"][3]["match"]["note"]
    assert "isn't in season" in view["legs"][4]["match"]["note"]
    again = read(lab, [leg("Duke", "moneyline", "1.50", sport_hint="NCAAB")])  # the next slip may load it
    assert statuses(again) == ["matched"] and odds.fetched == [NFL, EPL, NBA, NCAAB]


def test_no_league_is_loaded_below_the_quota_reserve(lab) -> None:
    odds = lab.service.odds
    odds.quota = {"remaining": 55, "used": 445}  # one fetch (3) still leaves the reserve (50)
    view = read(lab, [leg("Chiefs", "moneyline", "1.75", sport_hint="NFL"), leg("Arsenal", "moneyline", "2.10",
                                                                               sport_hint="EPL")])
    assert odds.fetched == [NFL] and statuses(view) == ["matched", "not_found"]
    odds.quota = {"remaining": 40, "used": 460}
    odds.fetched.clear()
    view = read(lab, [leg("Celtics", "moneyline", "1.62", sport_hint="NBA")])
    assert odds.fetched == [] and statuses(view) == ["not_found"]


def test_a_sport_without_its_league_looks_in_that_sports_leagues(lab) -> None:
    view = read(lab, [leg("Arsenal", "moneyline", "2.10", event_text="Arsenal v Chelsea", sport_hint="Football")])
    assert view["legs"][0]["match"]["ref"]["league"] == EPL and lab.service.odds.fetched == [EPL]


# --- pricing at the person's odds and the checks ---------------------------------------------------------

def test_the_free_reader_prices_the_slip_at_the_persons_odds(lab) -> None:
    result = lab.read_free(UID, US_PARLAY)
    view, analysis = result["pasted"], result["analysis"]
    assert "support" not in result
    assert view["source"] == "basic" and view["bookmaker"] == "DraftKings" and view["bet_type"] == "multi"
    assert statuses(view) == ["matched", "matched", "matched"]
    assert [item["verified"] for item in view["legs"]] == [None, None, None]  # code read it: nothing to verify
    # -110, -110, -125 at $10: 1.9091 x 1.9091 x 1.8 = 6.5603, so $65.60 back; "To Win $55.60" is the profit
    assert (view["stake"], view["total_stake"], view["currency"], view["stated_return"]) == ("10.00", "10.00", "$",
                                                                                             "55.60")
    assert view["return_check"] == "matches" and view["return_basis"] == "profit" and view["computed_return"] == "55.60"
    assert view["needs_confirmation"] is False and view["confirm_reasons"] == []
    ticket = analysis["ticket"]
    assert ticket["type"] == "multi" and ticket["odds"]["decimal"] == 6.56 and ticket["max_payout"] == "65.60"
    assert [leg_view["price"] for leg_view in ticket["legs"]] == ["1.9091", "1.9091", "1.8000"]  # theirs, not the board's
    assert [leg_view["slip_index"] for leg_view in ticket["legs"]] == [1, 2, 3]
    assert [leg_view["market_prob"] for leg_view in ticket["legs"]] == [item["market_chance"] for item in view["legs"]]
    assert ticket["valid"] and ticket["book_cut"] > 0 and Decimal(ticket["expected"]["market"]) < 0
    assert analysis["stake"] == "10.00" and analysis["drop"] == [] and analysis["ticket_type"] == "multi"
    suns = view["legs"][2]
    assert suns["best_price"] == {"decimal": "1.83", "book": "FanDuel"} and suns["better_price"] is True
    notes = {item["index"]: item["notes"] for item in analysis["legs"]}
    assert "Better price: 1.83 at FanDuel (you have 1.80)." in notes[3]
    assert any("bookmaker keeps" in note and "DraftKings" in note for note in notes[1])
    titles = [alt["title"] for alt in analysis["alternatives"]]
    assert titles[0].startswith("Drop the weakest leg") and "Split into 3 singles of $3.33" in titles
    alt = analysis["alternatives"][0]
    assert len(alt["legs"]) == 2 and alt["drop"] == sorted({1, 2, 3} - set(alt["legs"]))
    assert alt["refs"] == [view["legs"][index - 1]["match"]["ref"] for index in alt["legs"]]


def test_dropping_legs_and_changing_the_stake_or_type_reprice_in_code(lab) -> None:
    view = lab.read_free(UID, US_PARLAY)["pasted"]
    two = lab.analyse_pasted(UID, view["id"], drop=[2], stake="5")
    assert two["drop"] == [2] and two["stake"] == "5.00" and two["ticket"]["odds"]["decimal"] == 3.44
    assert two["ticket"]["max_payout"] == "17.18" and [leg_view["slip_index"] for leg_view in two["ticket"]["legs"]] == [1, 3]
    assert next(item for item in two["legs"] if item["index"] == 2)["in_ticket"] is False
    singles = lab.analyse_pasted(UID, view["id"], ticket_type="single", stake=Decimal("2"))
    assert singles["ticket"]["bets"] == 3 and singles["ticket"]["total_stake"] == "6.00"
    robin = lab.analyse_pasted(UID, view["id"], ticket_type="round_robin", k=2)
    assert robin["ticket"]["bets"] == 3 and robin["ticket"]["k"] == 2
    for bad in ({"drop": [1, 2, 3]}, {"drop": [4]}, {"drop": [0]}, {"stake": "0"}, {"stake": "abc"},
                {"stake": "100001"}, {"ticket_type": "lottery"}, {"ticket_type": "system", "system": "nope"}):
        with pytest.raises(CallRejected):
            lab.analyse_pasted(UID, view["id"], **bad)


def test_a_stated_payout_that_matches_and_one_that_doesnt(lab) -> None:
    good = lab.read_free(UID, US_PARLAY.replace("To Win $55.60", "Payout $65.61"))["pasted"]
    assert good["return_check"] == "matches" and good["return_basis"] == "payout" and good["computed_return"] == "65.60"
    bad = lab.read_free(UID, US_PARLAY.replace("To Win $55.60", "Payout $85.00"))["pasted"]
    assert bad["return_check"] == "differs" and bad["needs_confirmation"]
    assert bad["confirm_reasons"] == ["Your bookmaker says $85; at the odds read we get $65.60: check the odds."]


def test_a_doubtful_read_asks_the_person_to_confirm(lab) -> None:
    text = "Celtics -3.5 -110\nKnicks ML +140"
    view = read(lab, [leg("Boston Celtics", "spread", "-110", line=-3.5, sport_hint="NBA"),
                      leg("Knicks", "moneyline", "+150", sport_hint="NBA"),  # the text says +140
                      leg("Suns", "moneyline", "??", sport_hint="NBA")],
                source="text", text=text, stake_text="$10")
    assert [item["verified"] for item in view["legs"]] == [True, False, False]
    assert view["legs"][2]["match"]["status"] == "no_odds" and view["legs"][2]["odds"] is None
    assert view["needs_confirmation"] and view["confirm_reasons"] == [
        "The odds read for leg 2 (+150) aren't in the text you pasted.",
        "Couldn't read the odds for leg 3 (Suns to win)."]
    assert view["computed_return"] is None and view["return_check"] == "unknown"
    analysis = lab.analyse_pasted(UID, view["id"])
    assert [leg_view["slip_index"] for leg_view in analysis["ticket"]["legs"]] == [1, 2]
    assert analysis["ticket"]["warnings"][0] == "Leg 3 (Suns to win): its odds couldn't be read, so it is left out."
    # A screenshot's odds can't be checked against text: never "verified", and no reason to confirm for that
    image = read(lab, [leg("Knicks", "moneyline", "+150", sport_hint="NBA")], source="image")
    assert image["legs"][0]["verified"] is False and image["needs_confirmation"] is False


def test_the_slips_own_stake_and_kind_of_bet(lab) -> None:
    singles = lab.read_free(UID, "Singles\nKnicks ML @ 2.40\nSuns ML @ 1.80\nTotal stake: £10.00\n"
                                 "Potential returns: £21.00")["pasted"]
    assert (singles["bet_type"], singles["stake"], singles["total_stake"], singles["currency"]) == ("single", "5.00",
                                                                                                  "10.00", "£")
    assert singles["return_check"] == "matches"  # 5 x 2.40 + 5 x 1.80
    unknown = lab.read_free(UID, "Knicks ML 2.40\nSuns ML 1.80")
    assert unknown["pasted"]["stake"] is None and unknown["pasted"]["bet_type"] == "unknown"
    assert "priced as a multi" in " ".join(unknown["pasted"]["warnings"])
    assert unknown["analysis"]["stake"] == "10.00" and unknown["analysis"]["ticket"]["type"] == "multi"
    assert "No stake was read from the slip, so it is priced at 10 a bet." in unknown["analysis"]["ticket"]["warnings"]
    euro = read(lab, [leg("Arsenal", "moneyline", "2,10", event_text="Arsenal v Chelsea"),
                      leg("Celtics", "moneyline", "1,62")], stake_text="€5,00", bet_type="multi")
    titles = [alt["title"] for alt in lab.analyse_pasted(UID, euro["id"])["alternatives"]]
    assert "Split into 2 singles of €2.50" in titles


def test_a_same_game_slip_is_priced_but_flagged(lab) -> None:
    view = read(lab, [leg("Boston Celtics", "spread", "-110", line=-3.5, sport_hint="NBA"),
                      leg("Over", "total", "-110", line=224.5, over_under="over", event_text="Knicks @ Celtics",
                          sport_hint="NBA")], bet_type="multi")
    analysis = lab.analyse_pasted(UID, view["id"])
    assert analysis["ticket"]["valid"] is False and "same game" in analysis["ticket"]["reasons"][0]
    assert analysis["ticket"]["max_payout"] == "36.45" and analysis["alternatives"] == []


def test_the_view_reprices_from_cached_boards_and_never_loads(lab) -> None:
    odds = lab.service.odds
    view = lab.read_free(UID, US_PARLAY)["pasted"]
    odds.fetched.clear()
    again = lab.pasted_view(UID, view["id"])  # NBA isn't cached: the match stays as read, nothing is fetched
    assert odds.fetched == [] and again["legs"] == view["legs"]
    odds.cached.add(NBA)
    odds.games[NBA][2] = event("n3", NBA, "Phoenix Suns", "Sacramento Kings", "2030-01-02T03:00:00Z",
                               {"DraftKings": (1.95, 1.90, -1.5, 1.91)})
    odds.games[NBA][0] = event("n1", NBA, "Boston Celtics", "New York Knicks", "2030-01-02T00:30:00Z",
                               {"DraftKings": (1.62, 2.40, -4.5, 1.91)})
    lab.service.forget(NBA)
    fresh = lab.pasted_view(UID, view["id"])
    assert odds.fetched == []
    assert fresh["legs"][2]["best_price"] == {"decimal": "1.95", "book": "DraftKings"}
    assert fresh["legs"][0]["match"]["status"] == "line_differs"
    assert fresh["legs"][0]["match"]["note"].startswith("The board now has -4.5; you have -3.5")
    assert fresh["matched_refs"] == [fresh["legs"][1]["match"]["ref"], fresh["legs"][2]["match"]["ref"]]


# --- gates, privacy and storage -----------------------------------------------------------------------

def test_age_and_breaks_refuse_and_the_pause_keeps_the_numbers(lab) -> None:
    view = lab.read_free(UID, US_PARLAY)["pasted"]
    stranger = "web:not-confirmed-1"
    for call in (lambda: lab.read_free(stranger, US_PARLAY), lambda: read(lab, [leg("Knicks", "moneyline", "2.40")],
                                                                           uid=stranger)):
        with pytest.raises(CallRejected, match="18\\+"):
            call()
    lab.pause_picks(UID, NOW + 3600)
    paused = lab.read_free(UID, US_PARLAY)
    assert paused["analysis"]["ticket"]["max_payout"] == "65.60" and paused["analysis"]["alternatives"] == []
    assert paused["analysis"]["paused_until"] is not None and "Picks are paused" in paused["analysis"]["pause_note"]
    lab.cooloff(UID, 24)
    for call in (lambda: lab.read_free(UID, US_PARLAY), lambda: lab.pasted_view(UID, view["id"]),
                 lambda: lab.analyse_pasted(UID, view["id"])):
        with pytest.raises(CallRejected, match="on a break"):
            call()
    assert AGE_MESSAGE.startswith("Confirm you're 18+")


def test_nobody_else_can_read_a_pasted_slip(lab) -> None:
    view = lab.read_free(UID, US_PARLAY)["pasted"]
    with pytest.raises(NotFound):  # before any gate: someone who hasn't even confirmed their age gets a 404
        lab.pasted_view(OTHER, view["id"])
    lab.update_me(OTHER, {"confirm_age": True})
    for call in (lambda: lab.pasted_view(OTHER, view["id"]), lambda: lab.analyse_pasted(OTHER, view["id"]),
                 lambda: lab.pasted_view(UID, view["id"] + 1)):
        with pytest.raises(NotFound, match=NO_SUCH_PASTE):
            call()
    thread = lab.store.create_thread(OTHER)
    with pytest.raises(NotFound):
        lab.read_free(UID, US_PARLAY, thread_id=thread["id"])  # someone else's chat
    mine = lab.store.create_thread(UID)
    kept = lab.read_free(UID, US_PARLAY, thread_id=mine["id"])["pasted"]
    assert kept["thread_id"] == mine["id"]
    assert [row["id"] for row in lab.store.pasted_for_thread(UID, mine["id"])] == [kept["id"]]
    assert lab.store.pasted_for_thread(OTHER, mine["id"]) == []


def test_a_slip_with_no_legs_or_too_many_is_refused(lab) -> None:
    for bad in ({"legs": []}, {"legs": [{"raw": " ", "selection": "", "market": "other"}]}, "not a slip", None):
        with pytest.raises(CallRejected):
            lab.read_pasted(UID, bad, "text")
    with pytest.raises(CallRejected, match="more than 20 legs"):
        read(lab, [leg(f"Team {i}", "moneyline", "2.00") for i in range(21)])
    with pytest.raises(CallRejected, match="source"):
        lab.read_pasted(UID, {"legs": [leg("Knicks", "moneyline", "2.40")]}, "fax")
    with pytest.raises(CallRejected, match="one bet per line"):
        lab.read_basic(UID, "Who wins tonight?")
    with pytest.raises(CallRejected, match="6,000"):
        lab.read_basic(UID, "Knicks ML 2.40\n" * 500)


def test_the_store_keeps_each_persons_newest_slips(tmp_path) -> None:
    store = Store(str(tmp_path / "betting.db"))
    ids = [store.save_pasted(UID, "basic", {"legs": []}, [], text=f"slip {i}", thread_id=3 if i % 2 else None,
                             now=1000.0 + i) for i in range(MAX_PASTED + 5)]
    other = store.save_pasted(OTHER, "text", {"legs": []}, [], now=5000.0)
    assert store.pasted(UID, ids[0]) is None and store.pasted(UID, ids[4]) is None  # the 5 oldest are gone
    newest = store.pasted(UID, ids[-1])
    assert newest["text"] == f"slip {MAX_PASTED + 4}" and newest["extraction"] == {"legs": []} and newest["matches"] == []
    in_thread = store.pasted_for_thread(UID, 3)
    assert [row["id"] for row in in_thread] == sorted(row["id"] for row in in_thread) and in_thread[-1]["id"] == ids[-2]
    assert store.pasted(UID, other) is None and store.pasted(OTHER, other)["source"] == "text"
    assert store.save_pasted(UID, "basic", {"legs": []}, [], text="x" * 7000) and len(
        store.pasted(UID, ids[-1] + 2)["text"]) == 6000
    for bad in ({"user": "", "source": "basic"}, {"user": "*web", "source": "basic"}, {"user": UID, "source": "fax"}):
        with pytest.raises(ValueError):
            store.save_pasted(bad["user"], bad["source"], {"legs": []}, [])


def test_a_warning_sign_in_pasted_text_pauses_picks_like_the_chat(lab) -> None:
    text = "I lost my rent money this week, need to win it back\n" + US_PARLAY
    result = lab.read_free(UID, text)
    assert result["support"] and lab.picks_paused_until(UID) is not None
    assert result["analysis"]["alternatives"] == [] and result["analysis"]["paused_until"] is not None
    assert result["analysis"]["ticket"]["max_payout"] == "65.60"  # the numbers stay
    events = lab.store.rg_events(UID)
    assert events and events[0]["detail"]["source"] == "pasted_slip"


# --- the API ---------------------------------------------------------------------------------------------

def test_the_free_reader_and_the_cards_toggles_over_the_api(client, lab) -> None:
    response = client.post("/api/pasted", json={"text": US_PARLAY})
    assert response.status_code == 200
    body = response.json()
    pasted_id = body["pasted"]["id"]
    assert body["analysis"]["pasted"]["id"] == pasted_id and body["analysis"]["ticket"]["max_payout"] == "65.60"
    assert client.get(f"/api/pasted/{pasted_id}").json()["legs"] == body["pasted"]["legs"]
    toggled = client.post(f"/api/pasted/{pasted_id}/analyse", json={"drop": [2], "stake": 5, "type": "multi"}).json()
    assert toggled["drop"] == [2] and toggled["ticket"]["max_payout"] == "17.18"
    assert client.post(f"/api/pasted/{pasted_id}/analyse", json={"drop": [9]}).status_code == 400
    assert client.post(f"/api/pasted/{pasted_id}/analyse", json={"stake": 0}).status_code == 422
    lab.update_me(OTHER, {"confirm_age": True})
    assert client.get(f"/api/pasted/{pasted_id}", headers=THEM).status_code == 404
    assert client.post(f"/api/pasted/{pasted_id}/analyse", headers=THEM, json={}).status_code == 404
    assert client.get("/api/pasted/99999").status_code == 404
    nothing = client.post("/api/pasted", json={"text": "Who wins tonight?"})
    assert nothing.status_code == 400 and "one bet per line" in nothing.json()["detail"]
    long = client.post("/api/pasted", json={"text": "Knicks ML 2.40\n" * 500})
    assert long.status_code == 400 and "6,000" in long.json()["detail"]
    assert TestClient(client.app).post("/api/pasted", json={"text": US_PARLAY}).status_code == 401


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200


def test_the_ai_reader_route_checks_the_screenshot_first(client, lab) -> None:
    response = client.post("/api/chat/slip", json={"text": US_PARLAY})
    assert response.status_code == 400 and response.json()["detail"] == NO_CHAT
    lab.assistant = SlipAssistant()
    data = base64.b64encode(PNG).decode()
    body = client.post("/api/chat/slip", json={"thread_id": 4, "text": "my slip",
                                               "image": {"media_type": "image/jpeg", "data": data}}).json()
    assert body["reply"]["text"].startswith("I read your slip as")
    asked = lab.assistant.asked[-1]
    assert asked == {"uid": UID, "thread_id": 4, "text": "my slip", "image": PNG, "media_type": "image/png"}
    assert client.post("/api/chat/slip", json={"image": {"data": f"data:image/png;base64,{data}"}}).status_code == 200
    assert lab.assistant.asked[-1]["text"] is None
    for image, words in (({"data": "not base64!!"}, "base64"),
                         ({"data": base64.b64encode(b"GIF89a" + b"\x00" * 99).decode()}, "JPEG, PNG or WebP")):
        response = client.post("/api/chat/slip", json={"image": image})
        assert response.status_code == 400 and words in response.json()["detail"]
    assert client.post("/api/chat/slip", json={"text": "  "}).status_code == 422
    assert client.post("/api/chat/slip", json={"text": "x" * 6001}).status_code == 400
    lab.assistant = SlipAssistant(refuse=CallRejected("Another answer is still on its way"))
    assert client.post("/api/chat/slip", json={"text": US_PARLAY}).status_code == 400
    lab.assistant = SlipAssistant(refuse=NotFound("That chat doesn't exist"))
    assert client.post("/api/chat/slip", json={"text": US_PARLAY, "thread_id": 77}).status_code == 404


def test_only_the_slip_route_takes_a_large_body(client, lab) -> None:
    assert BODY_LIMITS == {"/api/chat/slip": MAX_SLIP_BODY_BYTES} and MAX_SLIP_BODY_BYTES == 2_500_000
    lab.assistant = SlipAssistant()
    # A 1.8 MB screenshot (2.4 MB as base64) with the longest text fits under 2.5 MB
    biggest = PNG + b"\x00" * (pasted.MAX_IMAGE_BYTES - len(PNG))
    body = json.dumps({"text": "x" * pasted.MAX_PASTE_CHARS,
                       "image": {"media_type": "image/png", "data": base64.b64encode(biggest).decode()}})
    assert len(body) < MAX_SLIP_BODY_BYTES
    response = client.post("/api/chat/slip", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 200 and lab.assistant.asked[-1]["image"] == biggest
    too_big = json.dumps({"image": {"data": "A" * MAX_SLIP_BODY_BYTES}})
    response = client.post("/api/chat/slip", content=too_big, headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    # Everywhere else, 64 KB
    medium = json.dumps({"text": "x" * (MAX_BODY_BYTES + 10)})
    for path in ("/api/pasted", "/api/chat"):
        assert client.post(path, content=medium, headers={"Content-Type": "application/json"}).status_code == 413
    # ... while the slip route takes the body and refuses the text for its length instead
    assert client.post("/api/chat/slip", content=medium, headers={"Content-Type": "application/json"}).status_code == 422


def test_the_https_site_allows_a_large_body_on_the_slip_route_only() -> None:
    script = (ROOT / "deploy" / "setup_vps.sh").read_text()
    site = script[script.index("$DOMAIN {"):script.index("\nCADDY\n")]
    slip = site[site.index("handle @slip {"):site.index("\thandle {")]
    rest = site[site.index("\thandle {"):]
    assert "@slip path /api/chat/slip" in site
    assert "max_size 2500KB" in slip and "reverse_proxy 127.0.0.1:$port" in slip
    assert "max_size 64KB" in rest and "reverse_proxy 127.0.0.1:$port" in rest and "2500KB" not in rest


def test_the_fact_sheet_for_claude_holds_the_slips_numbers_and_each_legs_notes(lab) -> None:
    view = lab.read_free(UID, US_PARLAY)["pasted"]
    analysis = lab.analyse_pasted(UID, view["id"], drop=[1])
    sheet = lab.pasted_fact_sheet(analysis)
    assert sheet["currency"] == "$" and sheet["stake"] == "10.00" and sheet["drop"] == [1] and sheet["paused"] is False
    assert [leg["index"] for leg in sheet["legs"]] == [0, 1]  # the ticket's own legs: slip legs 2 and 3
    assert sheet["legs"][1]["notes"] == next(item["notes"] for item in analysis["legs"] if item["index"] == 3)
    assert sheet["pasted"]["stated_return"] == "55.60" and sheet["pasted"]["return_check"] == "matches"
    assert [leg["odds_as_written"] for leg in sheet["pasted"]["legs"]] == ["-110", "-110", "-125"]
    assert sheet["ticket"]["max_payout"] == analysis["ticket"]["max_payout"]
    json.dumps(sheet)  # plain JSON for the model


# --- review fixes: same-game parlays, prices far off the board, money that doesn't add up, 1X2, leagues -----

SGP = """DraftKings
SGP
Knicks @ Celtics
Jayson Tatum 25+ Points
Jaylen Brown 5+ Rebounds
Celtics Moneyline
+350
Wager $10.00
To Win $35.00"""


def test_a_same_game_parlay_is_read_but_not_priced_leg_by_leg(lab, client) -> None:
    view = lab.read_basic(UID, SGP)
    assert [item["odds"] for item in view["legs"]] == [None, None, None]
    assert pasted.NO_LEG_ODDS in view["warnings"] and view["needs_confirmation"] is False  # nothing to confirm
    with pytest.raises(CallRejected, match="same-game parlay"):
        lab.analyse_pasted(UID, view["id"])
    body = client.post("/api/pasted", json={"text": SGP}).json()  # the free reader: what was read, no numbers
    assert body["analysis"] is None and body["pasted"]["warnings"][0] == pasted.NO_LEG_ODDS
    # The AI's read of the same slip (each leg without odds) is the same
    ai = read(lab, [leg("Jayson Tatum", "player_prop", None, line=24.5, over_under="over", prop_stat="points",
                        sport_hint="NBA"), leg("Boston Celtics", "moneyline", None, sport_hint="NBA")],
              bet_type="multi", stake_text="$10", notes="Same game parlay at +350")
    assert ai["needs_confirmation"] is False and pasted.NO_LEG_ODDS in ai["warnings"]


def test_odds_far_above_the_board_or_a_one_leg_multi_are_checked_first(lab) -> None:
    # A combined price read as one leg's: Celtics at +350 when the board's best is 1.62
    combined = lab.read_basic(UID, "Celtics Moneyline +350\nWager $10.00\nTo Win $35.00")
    assert combined["return_check"] == "matches" and combined["needs_confirmation"]
    assert combined["confirm_reasons"] == ["Leg 1's odds (+350) are far longer than any price on our board (best "
                                           "1.62 at DraftKings): check them. Is it a combined or boosted price?"]
    one = lab.read_basic(UID, "Parlay\nCeltics ML -150\nWager $10.00")
    assert one["confirm_reasons"] == ["Your slip looks like a multi (parlay or accumulator), but only one leg was "
                                      "read: check that none is missing."]
    near = lab.read_basic(UID, "Celtics ML +100\nWager $10.00")  # 2.00 against 1.62: a better bookmaker, maybe
    assert near["needs_confirmation"] is False and near["legs"][0]["better_price"] is False


def test_a_stake_per_bet_that_misses_the_total_is_checked_first(lab) -> None:
    singles = lab.read_basic(UID, "Singles\nKnicks ML @ 2.40\nSuns ML @ 1.80\nStake per bet $10.00\nTotal stake $30.00")
    assert singles["needs_confirmation"] and singles["confirm_reasons"][-1] == (
        "Your slip says $30 in all, but $10 on each of the 2 bets read is $20: a leg may be missing, or it's an "
        "each-way bet.")
    each_way = lab.read_basic(UID, "Single\nKnicks 7/5\nStake £5.00 EW\nTotal stake £10.00")
    assert each_way["confirm_reasons"][-1] == ("Your slip says £10 in all, but the stake read for its one bet is £5: a "
                                               "leg may be missing, or it's an each-way bet.")
    even = lab.read_basic(UID, "Singles\nKnicks ML @ 2.40\nSuns ML @ 1.80\nStake per bet $10.00\nTotal stake $20.00")
    assert even["needs_confirmation"] is False


def test_1x2_picks_and_dash_written_games_match_their_sides(lab) -> None:
    view = read(lab, [leg("1", "moneyline", "2.10", event_text="Arsenal vs Chelsea", sport_hint="Premier League"),
                      leg("2", "moneyline", "7.00", event_text="Man City - Brighton", sport_hint="Premier League"),
                      leg("Draw", "moneyline", "3.40", event_text="Arsenal - Chelsea", sport_hint="Premier League"),
                      leg("Over", "total", "1.95", line=2.5, over_under="over", event_text="Arsenal - Chelsea",
                          sport_hint="Premier League")])
    assert statuses(view) == ["matched"] * 4
    assert [(item["match"]["ref"]["id"], item["match"]["ref"]["side"]) for item in view["legs"]] == [
        ("e1", "home"), ("e2", "away"), ("e1", "draw"), ("e1", "over")]
    free = lab.read_basic(UID, "Bet9ja\nMultiple\nArsenal - Chelsea\n1X2 - 1 @ 2.10\nMan City - Brighton\n1X2 - X @ 5.00")
    assert [(item["selection"], item["match"]["ref"]["side"]) for item in free["legs"]] == [("Arsenal", "home"),
                                                                                          ("Draw", "draw")]


def test_another_league_of_the_same_name_is_neither_searched_nor_loaded(lab) -> None:
    odds = lab.service.odds
    odds.cached = {NBA, EPL}  # the namesakes' boards are right there
    view = read(lab, [leg("Maine Celtics", "moneyline", "1.70", event_text="Westchester Knicks @ Maine Celtics",
                          sport_hint="NBA G League"),
                      leg("Botafogo", "moneyline", "2.10", event_text="Botafogo v Chelsea", sport_hint="Brazil Serie A"),
                      leg("Jalen Smith", "player_prop", "1.90", line=12.5, over_under="over", prop_stat="points",
                          sport_hint="NBA G League")])
    assert statuses(view) == ["not_found", "not_found", "prop"] and odds.fetched == []
    assert [item["chance_source"] for item in view["legs"]] == ["estimated"] * 3
    assert view["legs"][2]["match"]["note"].startswith("The app models NBA players only")
