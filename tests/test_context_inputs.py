"""data/context_inputs.py: home-field rates from ESPN's standings (captured answers), each team's last game from
cached scoreboards (captured NBA and NHL boards, 25-27 October 2026) and days of rest. Pure: nothing fetches."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from telegram_betting_bot_simple.data import context_inputs as ci
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.espn_client import EspnClient
from telegram_betting_bot_simple.data.slate import parse_scoreboard
from telegram_betting_bot_simple.data.venues import load_venues

FIXTURES = Path(__file__).parent / "fixtures" / "context"
UTC = datetime.timezone.utc


def fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def boards(league):
    return {datetime.date.fromisoformat(day): events for day, events in fixture("espn_boards.json")[league].items()}


def home_records(payload, kind="home"):
    """(summary, ...) of every entry's home stat, straight from the payload."""
    found = []
    for child in payload["children"]:
        for entry in child["standings"]["entries"]:
            found.extend(s.get("summary") for s in entry["stats"] if s.get("type") == kind)
    return found


# --- home-field rates -------------------------------------------------------------------------------------------------

def test_home_rates_shrink_toward_the_prior_with_400_games():
    payload = {"children": [{"standings": {"entries": [
        {"team": {"id": "1"}, "stats": [{"name": "Home", "type": "home", "summary": "30-10"}]},
        {"team": {"id": "2"}, "stats": [{"name": "Home", "type": "home", "summary": "20-20"}]}]}}]}
    rates = ci.home_rates("basketball_nba", payload, as_of="2026-10-27T12:00:00Z")
    assert rates["games"] == 80 and rates["raw"] == 0.625 and rates["teams"] == 2
    assert rates["home_win"] == round((50 + 0.554 * 400) / 480, 4)
    assert rates["draw"] is None and rates["prior"] == {"home_win": 0.554, "draw": None}
    assert rates["prior_games"] == 400 and rates["source"] == "ESPN standings, home records"
    assert rates["as_of"] == "2026-10-27T12:00:00Z"
    assert ci.home_rates("basketball_nba", payload, prior=0.5)["home_win"] == round((50 + 200) / 480, 4)


def test_home_rates_from_the_captured_standings():
    nba = fixture("espn_standings_nba.json")
    wins = sum(int(s.split("-")[0]) for s in home_records(nba))
    games = sum(sum(int(x) for x in s.split("-")) for s in home_records(nba))
    rates = ci.home_rates("basketball_nba", nba)
    assert rates["games"] == games > 0 and rates["teams"] == 30
    assert rates["home_win"] == round((wins + 0.554 * 400) / (games + 400), 4)
    # The NHL's home record is W-L-OTL: an overtime loss is a loss
    nhl = fixture("espn_standings_nhl.json")
    records = [[int(x) for x in s.split("-")] for s in home_records(nhl)]
    assert all(len(r) == 3 for r in records)
    rates = ci.home_rates("icehockey_nhl", nhl)
    assert rates["games"] == sum(sum(r) for r in records)
    assert rates["raw"] == round(sum(r[0] for r in records) / sum(sum(r) for r in records), 4)
    # The NFL counts a tie as half a win; MLB's whole 2026 season is in its answer
    assert ci.home_rates("americanfootball_nfl", fixture("espn_standings_nfl.json"))["teams"] == 32
    mlb = ci.home_rates("baseball_mlb", fixture("espn_standings_mlb.json"))
    assert mlb["games"] > 2000 and 0.5 < mlb["raw"] < 0.56 and abs(mlb["home_win"] - mlb["raw"]) < 0.01


def test_nfl_ties_count_half():
    payload = {"children": [{"standings": {"entries": [
        {"team": {"id": "1"}, "stats": [{"type": "home", "summary": "3-0-1"}]}]}}]}
    rates = ci.home_rates("americanfootball_nfl", payload)
    assert rates["games"] == 4 and rates["raw"] == 0.875


def test_mlb_home_counts_without_a_summary():
    payload = {"children": [{"standings": {"entries": [
        {"team": {"id": "1"}, "stats": [{"name": "homeWins", "value": 46.0}, {"name": "homeLosses", "value": 34.0},
                                         {"name": "homeTies", "value": 0.0}]}]}}]}
    assert ci.home_rates("baseball_mlb", payload)["games"] == 80


def test_soccer_standings_carry_no_home_split_so_the_prior_stands():
    rates = ci.home_rates("soccer_epl", fixture("espn_standings_epl.json"))
    assert rates["games"] == 0 and rates["home_win"] == 0.45 and rates["draw"] == 0.27 and rates["raw"] is None
    assert rates["source"].startswith("Prior only")
    # A soccer home record, when there is one, is W-D-L
    payload = {"children": [{"standings": {"entries": [
        {"team": {"id": "1"}, "stats": [{"type": "home", "summary": "10-6-4"}]}]}}]}
    rates = ci.home_rates("soccer_usa_mls", payload)
    assert rates["games"] == 20 and rates["raw"] == 0.5 and rates["raw_draw"] == 0.3
    assert rates["home_win"] == round((10 + 0.45 * 400) / 420, 4) and rates["draw"] == round((6 + 0.27 * 400) / 420, 4)


def test_soccer_rates_from_finished_games():
    def event(event_id, home, away, state="post", neutral=False):
        return {"id": event_id, "competitions": [{"date": "2026-10-25T15:00Z", "neutralSite": neutral,
                                                  "status": {"type": {"state": state, "completed": state == "post"}},
                                                  "competitors": [
                                                      {"homeAway": "home", "score": str(home), "team": {"id": "1"}},
                                                      {"homeAway": "away", "score": str(away), "team": {"id": "2"}}]}]}
    games = [event("a", 2, 0), event("b", 1, 1), event("c", 0, 3), event("d", 1, 0, state="pre"),
             event("e", 5, 0, neutral=True), event("a", 2, 0)]
    rates = ci.results_rates("soccer_epl", games)
    assert rates["games"] == 3 and rates["raw"] == round(1 / 3, 4) and rates["raw_draw"] == round(1 / 3, 4)
    assert rates["source"] == "ESPN scoreboards, finished games"
    assert ci.results_rates("soccer_epl", [])["source"] == "Prior only: no finished games yet"


def test_espn_client_standings_keep_the_home_record():
    payload = fixture("espn_standings_nhl.json")
    espn = EspnClient(TTLCache())
    espn._get = lambda url, params=None: payload  # the captured answer
    teams = espn.standings("icehockey_nhl")
    assert all(team["home"] and team["home"].count("-") == 2 for team in teams.values())
    assert ci.home_rates("icehockey_nhl", teams) == ci.home_rates("icehockey_nhl", payload)
    mlb = fixture("espn_standings_mlb.json")
    espn = EspnClient(TTLCache())
    espn._get = lambda url, params=None: mlb
    assert ci.home_rates("baseball_mlb", espn.standings("baseball_mlb"))["games"] == \
        ci.home_rates("baseball_mlb", mlb)["games"]
    epl = fixture("espn_standings_epl.json")
    espn._get = lambda url, params=None: epl
    assert all(team["home"] is None for team in espn.standings("soccer_epl").values())


def test_priors_and_lookbacks_are_the_specs():
    assert ci.home_prior("americanfootball_nfl") == 0.539 and ci.home_prior("basketball_nba") == 0.554
    assert ci.home_prior("icehockey_nhl") == 0.522 and ci.home_prior("baseball_mlb") == 0.529
    assert ci.home_prior("basketball_wnba") == 0.55 and ci.home_prior("americanfootball_ncaaf") == 0.57
    assert ci.home_prior("basketball_ncaab") == 0.60 and ci.home_prior("soccer_epl") == (0.45, 0.27)
    assert [ci.lookback_for(league) for league in ("basketball_nba", "basketball_wnba", "icehockey_nhl")] == [4, 4, 4]
    assert ci.lookback_for("americanfootball_nfl") == 14 and ci.lookback_for("baseball_mlb") == 2
    assert ci.lookback_for("soccer_epl") == ci.lookback_for("soccer_usa_mls") == 10
    assert ci.lookback_for("americanfootball_ncaaf") == ci.lookback_for("basketball_ncaab") == 7
    with pytest.raises(ValueError):
        ci.home_rates("tennis_atp", {})


# --- each team's last game and rest -----------------------------------------------------------------------------------

def test_last_games_from_the_captured_nba_boards():
    nba = boards("basketball_nba")
    # Celtics (2) v Nets (17), 27 October: Boston played at home the night before; Brooklyn two nights before
    found = ci.last_games("basketball_nba", nba, ["2", "17", "99"], "2026-10-27T23:30Z")
    assert set(found) == {"2", "17"}
    assert found["2"] == {"date": "2026-10-26", "start": "2026-10-26T23:30:00Z",
                          "venue": {"name": "TD Garden", "city": "Boston", "espn_id": "1824"},
                          "opponent": "Chicago Bulls", "home": True, "neutral": False, "status": "final",
                          "event_id": "401909870"}
    assert found["17"]["date"] == "2026-10-25" and found["17"]["opponent"] == "Indiana Pacers"
    assert ci.rest_for("basketball_nba", found["2"], "2026-10-27T23:30Z") == {"rest_days": 0, "back_to_back": True}
    assert ci.rest_for("basketball_nba", found["17"], "2026-10-27T23:30Z") == {"rest_days": 1, "back_to_back": False}
    assert ci.rest_for("basketball_nba", None, "2026-10-27T23:30Z") == {"rest_days": None, "back_to_back": False}
    # The last venue is in the venue table (travel is measured from it)
    assert load_venues().find("basketball_nba", espn_venue_id=found["2"]["venue"]["espn_id"])["id"] == "td-garden"


def test_the_slates_parsed_games_give_the_same_answer():
    nba = boards("basketball_nba")
    parsed = {day: parse_scoreboard("basketball_nba", events) for day, events in nba.items()}
    raw = ci.last_games("basketball_nba", nba, ["2", "17", "13"], "2026-10-28T03:00Z")
    slate = ci.last_games("basketball_nba", parsed, ["2", "17", "13"], "2026-10-28T03:00Z")
    for team_id in raw:
        assert {k: v for k, v in raw[team_id].items() if k != "venue"} == \
               {k: v for k, v in slate[team_id].items() if k != "venue"}
        assert slate[team_id]["venue"]["name"] == raw[team_id]["venue"]["name"]


def test_rest_is_counted_on_the_local_calendar():
    """The Lakers played at Utah on 25 October and host Portland at 8 pm Pacific on 27 October (03:00 GMT on the
    28th): one day off on the US calendar, two if the GMT dates were used."""
    nba = boards("basketball_nba")
    last = ci.last_games("basketball_nba", nba, ["13"], "2026-10-28T03:00Z")["13"]
    assert last["date"] == "2026-10-25" and last["venue"]["espn_id"] == "3093"
    assert ci.rest_days(last["start"], "2026-10-28T03:00Z") == 1
    assert ci.rest_days(last["start"], "2026-10-28T03:00Z", tz="America/Los_Angeles") == 1
    assert ci.rest_days(last["start"], "2026-10-28T03:00Z", tz="UTC") == 2


def test_nhl_back_to_back_and_an_outdoor_game():
    nhl = boards("icehockey_nhl")
    found = ci.last_games("icehockey_nhl", nhl, ["8", "4", "28"], "2026-10-28T00:00Z")
    assert found["8"]["venue"]["name"] == "Madison Square Garden" and found["8"]["home"] is False
    assert ci.rest_days(found["8"]["start"], "2026-10-28T00:00Z") == 0  # the Kings, at the Rangers last night
    assert found["4"]["date"] == "2026-10-25" and ci.rest_days(found["4"]["date"], "2026-10-27") == 1
    jets = found["28"]  # the Jets' last game was outdoors, at Princess Auto Stadium
    assert jets["venue"]["espn_id"] == "4461"
    assert load_venues().find("icehockey_nhl", espn_venue_id="4461")["roof"] == "open"


def test_last_games_skips_postponed_later_and_too_old_games():
    def event(event_id, start, state="post", name="STATUS_FINAL"):
        return {"id": event_id, "competitions": [{"date": start, "status": {"type": {"state": state, "name": name}},
                                                  "venue": {"id": "1", "fullName": "Arena"},
                                                  "competitors": [
                                                      {"homeAway": "home", "team": {"id": "1", "displayName": "A"}},
                                                      {"homeAway": "away", "team": {"id": "2", "displayName": "B"}}]}]}
    days = {"x": [event("old", "2026-10-20T23:00Z"), event("ok", "2026-10-25T23:00Z"),
                  event("off", "2026-10-26T23:00Z", state="pre", name="STATUS_POSTPONED"),
                  event("this", "2026-10-27T23:00Z", state="pre", name="STATUS_SCHEDULED"),
                  event("later", "2026-10-28T23:00Z", state="pre", name="STATUS_SCHEDULED")]}
    found = ci.last_games("basketball_nba", days, ["1"], "2026-10-27T23:00Z")
    assert found["1"]["event_id"] == "ok"
    assert ci.last_games("basketball_nba", days, ["1"], "2026-10-27T23:00Z", lookback_days=1) == {}
    assert ci.last_games("basketball_nba", days, ["1"], None) == {}
    assert ci.last_games("basketball_nba", {"x": [None, {"competitions": []}, {"id": "q"}]}, ["1"],
                         "2026-10-27T23:00Z") == {}


def test_rest_days():
    d = datetime.date
    assert ci.rest_days(d(2026, 10, 26), d(2026, 10, 27)) == 0
    assert ci.rest_days(d(2026, 10, 27), d(2026, 10, 27)) == 0  # a doubleheader
    assert ci.rest_days(d(2026, 10, 25), d(2026, 10, 27)) == 1
    assert ci.rest_days("2026-10-19", "2026-10-26") == 6  # an NFL week
    assert ci.rest_days(d(2026, 10, 25), d(2026, 10, 29)) == 3  # Sunday to Thursday
    late = datetime.datetime(2026, 10, 27, 2, 30, tzinfo=UTC)  # 10:30 pm on the 26th in New York
    assert ci.rest_days(late, datetime.datetime(2026, 10, 27, 23, 0, tzinfo=UTC)) == 0
    assert ci.rest_days(d(2026, 10, 28), d(2026, 10, 27)) is None
    assert ci.rest_days(None, d(2026, 10, 27)) is None and ci.rest_days("soon", d(2026, 10, 27)) is None
    # Epoch seconds, as the slate keeps start times
    assert ci.rest_days(late.timestamp(), datetime.datetime(2026, 10, 27, 23, 0, tzinfo=UTC).timestamp()) == 0
    assert ci.rest_for("basketball_nba", {"start": "2026-10-25T23:00:00Z"},
                       datetime.datetime(2026, 10, 27, 23, 30, tzinfo=UTC).timestamp())["rest_days"] == 1


def test_scoreboard_days():
    assert ci.scoreboard_days("2026-10-27T23:30Z", 4) == [datetime.date(2026, 10, d) for d in (23, 24, 25, 26, 27)]
    assert ci.scoreboard_days(None, 4) == []
