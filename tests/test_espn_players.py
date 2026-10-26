"""Tests for ESPN player data: search, profile, game logs, splits, news, props, cards and hit rates.

Every response is a small fixture trimmed from a live ESPN call (tests/fixtures/espn/),
served by a fake client keyed on URL and query, so no test touches the network.
Anything a test doesn't serve comes back as the HTTP 404 the real client raises.
"""

from __future__ import annotations

import copy
import datetime
import json
from decimal import Decimal
from pathlib import Path

import pytest

from telegram_betting_bot_simple.data import cache as cache_module
from telegram_betting_bot_simple.data import espn_players
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.espn_players import PlayerData, average_stats, hit_rates, stat_value

FIXTURES = Path(__file__).parent / "fixtures" / "espn"
NOW = datetime.datetime(2026, 10, 26, 12, 0, tzinfo=datetime.timezone.utc)

NBA = f"{espn_players.COMMON}/basketball/nba/athletes"
NFL = f"{espn_players.COMMON}/football/nfl/athletes"
MLB = f"{espn_players.COMMON}/baseball/mlb/athletes"
NHL = f"{espn_players.COMMON}/hockey/nhl/athletes"
PHI_SCHEDULE = f"{espn_players.SITE}/basketball/nba/teams/20/schedule"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def url(base, **params):
    """The fake's lookup key: the URL plus its query, sorted."""
    if not params:
        return base
    return base + "?" + "&".join(f"{key}={value}" for key, value in sorted(params.items()))


def props_url(sport, league, event_id):
    return (f"{espn_players.CORE}/{sport}/leagues/{league}/events/{event_id}"
            f"/competitions/{event_id}/odds/100/propBets")


class FakeEspn:
    """Serves saved ESPN responses by URL; anything unknown is a 404, as the real client raises."""

    def __init__(self, routes=None):
        self.cache = TTLCache()
        self.routes = dict(routes or {})
        self.calls = []

    def _get(self, address, params=None):
        key = url(address, **(params or {}))
        self.calls.append(key)
        if key not in self.routes:
            raise RuntimeError(f"ESPN returned HTTP 404 for {address}")
        payload = self.routes[key]
        if isinstance(payload, Exception):
            raise payload
        return copy.deepcopy(payload)


def player_data(routes=None):
    fake = FakeEspn(routes)
    return PlayerData(fake, now=lambda: NOW), fake


class FakeClock:
    """Stands in for the cache module's ``time`` so a test can let ten minutes pass in an instant."""

    def __init__(self, start):
        self.now = start

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock(NOW.timestamp())
    monkeypatch.setattr(cache_module, "time", fake)
    return fake


LEBRON = {
    url(f"{NBA}/1966"): load("nba_athlete_1966.json"),
    url(f"{NBA}/1966/overview"): load("nba_overview_1966.json"),
    url(PHI_SCHEDULE): load("nba_schedule_20.json"),
    url(f"{NBA}/1966/gamelog"): load("nba_gamelog_1966.json"),
    url(f"{NBA}/1966/gamelog", season=2026): load("nba_gamelog_1966_2026.json"),
    url(f"{NBA}/1966/splits"): load("nba_splits_1966.json"),
    url(espn_players.NEWS.format(game="fba"), days=30, playerId="1966"): load("news_fba_1966.json"),
}


# --- search ---

def test_search_keeps_covered_leagues_with_their_odds_api_keys():
    data, _ = player_data({url(espn_players.SEARCH, limit=25, query="lebron", type="player"): load("search_lebron.json")})
    result = data.search("lebron")
    assert result["reason"] is None
    assert [(p["id"], p["league"]) for p in result["players"]] == [
        ("1966", "basketball_nba"), ("5313200", "basketball_ncaab"), ("5198939", "baseball_mlb"),
    ]  # the Copa del Rey player is dropped: the app doesn't cover that league
    lebron = result["players"][0]
    assert lebron["name"] == "LeBron James"
    assert lebron["team"] == "Philadelphia 76ers" and lebron["team_abbr"] == "PHI"
    assert lebron["position"] == "F" and lebron["headshot"].endswith("/1966.png")


def test_search_by_league_passes_espn_filter_and_keeps_only_that_league():
    search = url(espn_players.SEARCH, league="nba", limit=25, query="LeBron", sport="basketball", type="player")
    data, fake = player_data({search: load("search_lebron.json")})
    result = data.search("  LeBron ", league="nba")  # an ESPN slug works as well as "basketball_nba"
    assert fake.calls == [search]
    assert [p["id"] for p in result["players"]] == ["1966"]
    assert result["players"][0]["league"] == "basketball_nba"


def test_search_retries_the_surname_for_names_with_punctuation():
    chase = {"id": "4362628", "displayName": "Ja'Marr Chase", "type": "player", "sport": "football", "league": "nfl"}
    brown = {"id": "4362238", "displayName": "Chase Brown", "type": "player", "sport": "football", "league": "nfl"}
    data, _ = player_data({
        url(espn_players.SEARCH, league="nfl", limit=25, query="jamarr chase", sport="football", type="player"):
            {"count": 0, "items": []},
        url(espn_players.SEARCH, league="nfl", limit=25, query="chase", sport="football", type="player"):
            {"count": 2, "items": [chase, brown]},
    })
    result = data.search("jamarr chase", "americanfootball_nfl")
    assert [p["name"] for p in result["players"]] == ["Ja'Marr Chase"]


def test_search_explains_empty_results_instead_of_raising():
    data, fake = player_data({
        url(espn_players.SEARCH, limit=25, query="zzzq", type="player"): {"count": 0, "items": []},
        url(espn_players.SEARCH, limit=25, query="down", type="player"): RuntimeError("connection reset"),
    })
    assert data.search(" ")["reason"].startswith("Type at least two letters")
    assert "doesn't cover 'cricket'" in data.search("lebron", league="cricket")["reason"]
    assert data.search("zzzq") == {"query": "zzzq", "players": [], "reason": "No player matching 'zzzq'."}
    down = data.search("down")
    assert down["players"] == [] and "couldn't be reached" in down["reason"]
    assert len(fake.calls) == 2  # bad input never reaches ESPN


# --- profile and next game ---

def test_overview_reads_bio_season_line_and_falls_back_to_the_team_schedule():
    data, _ = player_data(LEBRON)
    info = data.overview("basketball_nba", 1966)
    assert info["reason"] is None
    assert (info["name"], info["team_abbr"], info["team_id"], info["position"]) == ("LeBron James", "PHI", "20", "F")
    assert info["status"] == "Active" and info["injury"] is None
    assert info["season_stats"] == {
        "label": "Regular Season", "per_game": True, "games": 2,
        "stats": {"min": 33.5, "reb": 9.5, "ast": 6, "blk": 1, "stl": 2, "tov": 3, "pts": 15.5},
    }
    assert info["note"] == {
        "headline": "James (back) isn't listed on the injury report for Sunday's game against Detroit.",
        "published": "2026-10-25T17:06:18Z",  # RotoWire's "Sun Oct 25 10:06:18 PDT 2026"
        "source": "RotoWire",
    }
    # The overview shows the game just played, so the next one comes from the schedule
    assert info["next_game"] == {
        "event_id": "401909884", "date": "2026-10-28T23:00:00Z", "name": "PHI @ IND", "status": "pre",
        "home": False, "opponent": "IND", "opponent_name": "Indiana Pacers",
    }


def test_overview_uses_its_own_next_game_when_one_is_coming():
    data, fake = player_data({
        url(f"{NHL}/3895074"): load("nhl_athlete_3895074.json"),
        url(f"{NHL}/3895074/overview"): load("nhl_overview_3895074.json"),
    })
    info = data.overview("icehockey_nhl", "3895074")
    assert info["name"] == "Connor McDavid"
    assert info["next_game"]["name"] == "DET @ EDM"
    assert (info["next_game"]["home"], info["next_game"]["opponent"], info["next_game"]["status"]) == (True, "DET", "pre")
    assert info["season_stats"]["per_game"] is False  # hockey's season line is totals
    assert info["season_stats"]["stats"]["SOG"] == 28
    assert not any("schedule" in call for call in fake.calls)


def test_overview_reports_injuries_and_survives_a_missing_overview():
    data, _ = player_data({url(f"{NBA}/5061575"): load("nba_athlete_5061575.json")})
    info = data.overview("basketball_nba", "5061575")
    assert info["injury"] == {
        "status": "Out",
        "detail": "Knueppel (hamstring) has been ruled out for Monday's game against the Wizards.",
        "type": "Hamstring",
        "date": "2026-10-25T21:20:00Z",
        "return_date": "2026-10-28T00:00:00Z",
    }
    assert info["reason"].startswith("Season figures are unavailable")
    assert info["season_stats"] is None and info["next_game"] is None


def test_overview_of_an_unknown_or_malformed_player_gives_a_reason():
    data, fake = player_data()
    missing = data.overview("basketball_nba", "999")
    assert missing["name"] is None
    assert missing["reason"] == "ESPN has no basketball_nba player with id 999."
    calls = len(fake.calls)
    assert data.overview("basketball_nba", "../../x")["reason"].startswith("ESPN ids are numbers")
    assert data.overview("curling", "1")["reason"] == "Player data doesn't cover 'curling'."
    # str.isdigit() passes these, but they aren't ESPN ids
    for odd in ("²", "١٢٣", "1" * 13):
        assert data.overview("basketball_nba", odd)["reason"].startswith("ESPN ids are numbers")
    assert len(fake.calls) == calls


def test_overview_skips_the_schedule_for_a_team_id_that_isnt_ascii_digits():
    bio = load("nba_athlete_1966.json")
    bio["athlete"]["team"]["id"] = "²"
    data, fake = player_data({**LEBRON, url(f"{NBA}/1966"): bio})
    info = data.overview("basketball_nba", "1966")
    assert info["team_id"] == "²" and info["next_game"] is None
    assert not any("schedule" in call for call in fake.calls)


# --- game logs ---

def test_gamelog_maps_basketball_keys_and_reaches_back_a_season():
    data, _ = player_data(LEBRON)
    log = data.gamelog("basketball_nba", "1966", last=10)
    assert log["reason"] is None
    assert (log["season"], log["season_games"], len(log["rows"])) == ("2026-27", 3, 10)
    first = log["rows"][0]
    assert first == {
        "event_id": "401909866", "date": "2026-10-25T23:30:00Z", "opponent": "DET", "home": True,
        "result": "L", "score": "106-93", "season": "2026-27", "season_type": "regular", "minutes": 33,
        "stats": {"min": 33, "fg3m": 0, "reb": 2, "ast": 5, "blk": 0, "stl": 2, "tov": 3, "pts": 18},
    }
    assert [row["season"] for row in log["rows"]] == ["2026-27"] * 3 + ["2025-26"] * 7
    assert log["rows"][3]["season_type"] == "postseason"
    assert "401898407" not in {row["event_id"] for row in log["rows"]}  # preseason is left out
    assert log["labels"]["pts"] == "Points" and log["labels"]["fg3m"] == "3-Point Field Goals Made"


def test_gamelog_says_when_it_was_fetched(clock):
    data, _ = player_data(LEBRON)
    assert data.gamelog("basketball_nba", "1966")["as_of"] == "2026-10-26T12:00:00Z"
    assert data.gamelog("basketball_nba", "1")["as_of"] is None  # nothing fetched, nothing to date


def test_gamelog_and_news_counts_from_a_query_string_never_raise():
    data, _ = player_data(LEBRON)
    for bad in (None, "ten", "", 1.5e400, float("nan")):
        assert len(data.gamelog("basketball_nba", "1966", last=bad)["rows"]) == 10  # the default
    assert len(data.gamelog("basketball_nba", "1966", last="3")["rows"]) == 3
    assert len(data.gamelog("basketball_nba", "1966", last=-5)["rows"]) == 1
    assert len(data.gamelog("basketball_nba", "1966", last=10**9)["rows"]) == 15  # every row there is
    assert len(data.news("basketball_nba", "1966", limit=None)["items"]) == 4  # default 5; the feed has 4
    assert len(data.news("basketball_nba", "1966", limit="2")["items"]) == 2
    assert len(data.news("basketball_nba", "1966", limit=0)["items"]) == 1


def test_gamelog_only_fetches_last_season_when_it_needs_to():
    data, fake = player_data(LEBRON)
    log = data.gamelog("basketball_nba", "1966", last=3)
    assert len(log["rows"]) == 3
    assert not any("season=2026" in call for call in fake.calls)


def test_football_logs_put_the_group_in_front_of_shared_labels():
    data, _ = player_data({
        url(f"{NFL}/3139477/gamelog"): load("nfl_gamelog_3139477.json"),
        url(f"{NFL}/3953687/gamelog"): load("nfl_gamelog_3953687.json"),
    })
    qb = data.gamelog("americanfootball_nfl", "3139477")["rows"][0]["stats"]
    assert (qb["PASS YDS"], qb["RUSH YDS"], qb["PASS TD"], qb["CMP"], qb["INT"]) == (272, 16, 0, 23, 1)
    assert "YDS" not in qb
    kicker = data.gamelog("americanfootball_nfl", "3953687")
    made = kicker["rows"][1]["stats"]  # ESPN's labels end in tabs; "1-1" field goals keep the made count
    assert (made["40-49"], made["FG"], made["FG LNG"], made["PTS"]) == (1, 1, 41, 4)


def test_baseball_innings_and_hockey_time_are_numbers():
    data, _ = player_data({
        url(f"{MLB}/42409/gamelog"): load("mlb_gamelog_42409.json"),
        url(f"{NHL}/3895074/gamelog"): load("nhl_gamelog_3895074.json"),
    })
    pitcher = data.gamelog("baseball_mlb", "42409")["rows"]
    seven_and_one = next(row for row in pitcher if row["date"] == "2026-09-09T02:10:00Z")
    assert seven_and_one["stats"]["IP"] == 7.333  # "7.1" innings is seven innings and one out
    assert "Dec" not in seven_and_one["stats"]  # "W(9-7)" isn't a number
    assert pitcher[0]["season_type"] == "postseason"
    averages = average_stats(pitcher)
    assert "ERA" not in averages and averages["K"] == 7.1  # a season-to-date ERA can't be averaged
    skater = data.gamelog("icehockey_nhl", "3895074")["rows"][0]
    assert skater["minutes"] == 22.45  # "22:27" time on ice
    assert skater["stats"]["SOG"] == 3 and "S" not in skater["stats"]


def test_gamelog_survives_missing_players_and_changed_responses():
    data, _ = player_data({url(f"{NBA}/2/gamelog"): {"seasonTypes": "changed"}})
    missing = data.gamelog("basketball_nba", "1")
    assert missing["rows"] == [] and missing["reason"] == "ESPN has no game log for this player."
    changed = data.gamelog("basketball_nba", "2")
    assert changed["rows"] == [] and "changed shape" in changed["reason"]


# --- splits ---

def test_splits_are_per_game_where_espn_gives_games_played():
    data, _ = player_data({
        url(f"{NBA}/1966/splits"): load("nba_splits_1966.json"),
        url(f"{NHL}/3895074/splits"): load("nhl_splits_3895074.json"),
        url(f"{NFL}/3139477/splits"): load("nfl_splits_3139477.json"),
    })
    nba = data.splits("basketball_nba", "1966")
    assert nba["per_game"] and nba["reason"] is None
    assert (nba["home"]["games"], nba["home"]["stats"]["pts"], nba["away"]["stats"]["pts"]) == (1, 17, 14)
    assert nba["total"]["stats"]["pts"] == 15.5
    assert sorted(nba["by_opponent"]) == ["vs CLE", "vs NY"]
    nhl = data.splits("icehockey_nhl", "3895074")
    assert nhl["per_game"] and nhl["home"]["games"] == 6
    assert nhl["home"]["stats"]["PTS"] == 2.5  # 15 points over 6 home games
    assert nhl["home"]["stats"]["TOI/G"] == 24.667  # already per game, left alone
    nfl = data.splits("americanfootball_nfl", "3139477")
    assert nfl["per_game"] is False  # no games-played column: season totals
    assert (nfl["total"]["stats"]["PASS YDS"], nfl["total"]["stats"]["RUSH YDS"]) == (1309, 57)


def test_mlb_pitcher_splits_divide_counts_by_games_but_keep_rates(clock):
    data, _ = player_data({url(f"{MLB}/42409/splits"): load("mlb_splits_42409.json")})
    splits = data.splits("baseball_mlb", "42409")
    assert splits["reason"] is None and splits["per_game"]  # pitchers' splits carry games played
    assert splits["as_of"] == "2026-10-26T12:00:00Z"
    total = splits["total"]
    assert total["games"] == 26
    assert (total["stats"]["K"], total["stats"]["IP"]) == (7.15, 6.1)  # 186 K and 158.2 innings over 26 games
    assert (total["stats"]["OBA"], total["stats"]["ERA"]) == (0.217, 2.72)  # rates stay as ESPN gives them
    assert "GP" not in total["stats"]
    home = splits["home"]
    assert (home["games"], home["stats"]["K"], home["stats"]["OBA"]) == (14, 7.07, 0.237)
    assert splits["away"]["stats"]["OBA"] == 0.193
    assert sorted(splits["by_opponent"]) == ["vs. ATL", "vs. BOS"]


# --- news ---

def test_news_prefers_written_items_and_drops_duplicates(clock):
    data, _ = player_data(LEBRON)
    three = data.news("basketball_nba", "1966", limit=3)
    assert three["reason"] is None and three["as_of"] == "2026-10-26T12:00:00Z"
    assert [(item["source"], item["published"]) for item in three["items"]] == [
        ("RotoWire", "2026-10-25T17:06:18Z"), ("RotoWire", "2026-10-24T17:06:23Z"), ("ESPN", "2026-10-20T12:33:18Z"),
    ]
    rotowire = three["items"][0]
    assert rotowire["link"] == "http://m.espn.go.com/wireless/story?storyId=53600246"
    assert rotowire["summary"].startswith("James will suit up")  # RotoWire's analysis, not the headline again
    assert three["items"][2]["summary"] == "Our fantasy hoops experts for their boldest predictions for the 2026-27 NBA season."
    every = data.news("basketball_nba", "1966", limit=5)["items"]
    assert len(every) == 4  # the repeated video clip appears once
    # The clip (Oct 23) is newer than the ESPN story (Oct 20) but still comes after every written item
    assert [(item["source"], item["published"][:10]) for item in every] == [
        ("RotoWire", "2026-10-25"), ("RotoWire", "2026-10-24"), ("ESPN", "2026-10-20"), ("ESPN Video", "2026-10-23"),
    ]


def test_news_explains_leagues_without_a_player_feed():
    data, fake = player_data()
    result = data.news("basketball_ncaab", "5313200")
    assert result == {"id": "5313200", "league": "basketball_ncaab", "items": [], "as_of": None,
                      "reason": "ESPN has no per-player news feed for basketball_ncaab."}
    assert fake.calls == []


# --- props ---

def test_props_keep_lines_and_only_prices_whose_side_is_clear():
    data, _ = player_data({
        url(props_url("basketball", "nba", "401909866"), limit=500, page=1): load("props_nba_401909866.json"),
    })
    result = data.props("basketball_nba", "401909866")
    assert result["reason"] is None and result["book"] == "DraftKings"
    assert sorted(result["players"]) == ["1966", "2581018"]  # team totals and game lines are left out
    assert result["count"] == 7
    lebron = {prop["name"]: prop for prop in result["players"]["1966"]}
    points = lebron["Total Points"]
    assert (points["market"], points["stat"], points["kind"]) == ("player_points", "pts", "over_under")
    assert (points["line"], points["open_line"]) == (18.5, 19.5)
    # Two unlabelled DraftKings prices: no way to tell the over from the under, so neither is kept
    assert (points["over_price"], points["under_price"], points["sides_known"]) == (None, None, False)
    assert lebron["Total Points, Rebounds, and Assists"]["stat"] == "pts+reb+ast"
    milestone = lebron["Points Milestones"]
    assert (milestone["kind"], milestone["label"], milestone["line"]) == ("milestone", "20+", 19.5)
    assert milestone["open_line"] == 19.5  # it opened at the same "20+" rung
    assert milestone["over_price"] == Decimal("2.02") and milestone["sides_known"]
    assert lebron["To Record a Double Double"]["yes_price"] == Decimal("2.51")
    assert lebron["First Basket"]["yes_price"] == Decimal("7.30")
    quarter = {prop["name"]: prop for prop in result["players"]["2581018"]}["Points Milestones - 1st Quarter"]
    assert (quarter["market"], quarter["stat"], quarter["line"]) == ("points_milestones_1st_quarter", None, 2.5)


def test_props_read_every_page_and_work_prices_out_from_american_odds():
    base = props_url("baseball", "mlb", "401908044")
    data, fake = player_data({
        url(base, limit=500, page=1): load("props_mlb_401908044_page1.json"),
        url(base, limit=500, page=2): load("props_mlb_401908044_page2.json"),
    })
    result = data.props("baseball_mlb", 401908044)
    assert len(fake.calls) == 2 and result["count"] == 5
    pitcher = result["players"]["39825"]
    assert [(p["market"], p["kind"], p["line"]) for p in pitcher] == [
        ("pitcher_strikeouts", "over_under", 4.5),
        ("pitcher_record_a_win", "yes", None),
        ("pitcher_strikeouts", "milestone", 1.5),
        ("pitcher_strikeouts", "milestone", 2.5),
    ]
    win = pitcher[1]
    assert win["yes_price"] is None and not win["sides_known"]  # a yes/no pair without labels
    assert pitcher[2]["over_price"] == Decimal("1.0286")  # -3500 is 1 + 100/3500
    # ESPN's "open" on the 2+ and 3+ items is the 6+ rung: no opening line rather than a fake move
    assert (pitcher[2]["open_line"], pitcher[3]["open_line"]) == (None, None)
    assert pitcher[3]["over_price"] == Decimal("1.1176")  # -850
    hits = result["players"]["31283"][0]
    assert (hits["stat"], hits["line"], hits["over_price"]) == ("H", 0.5, Decimal("1.5917"))


def test_props_during_a_game_keep_the_line_without_prices():
    data, _ = player_data({
        url(props_url("football", "nfl", "401873022"), limit=500, page=1): load("props_nfl_401873022_live.json"),
    })
    passing = data.props("americanfootball_nfl", "401873022")["players"]["3139477"][0]
    assert (passing["stat"], passing["line"], passing["open_line"]) == ("PASS YDS", 223.5, 220.5)
    assert passing["over_price"] is None and passing["under_price"] is None


def test_props_keep_both_prices_when_items_name_their_side():
    items = copy.deepcopy(load("props_nba_401909866.json")["items"][:2])  # LeBron's Total Points pair
    items[0]["side"], items[1]["side"] = "Over", "Under"
    data, _ = player_data({
        url(props_url("basketball", "nba", "1"), limit=500, page=1): {"pageCount": 1, "items": items},
    })
    points = data.props("basketball_nba", "1")["players"]["1966"][0]
    assert points["sides_known"]
    assert (points["over_price"], points["under_price"]) == (Decimal("1.8475"), Decimal("1.9259"))


def test_props_say_when_they_were_fetched_and_flag_a_copy_espn_did_not_renew(clock):
    route = url(props_url("basketball", "nba", "401909866"), limit=500, page=1)
    data, fake = player_data({route: load("props_nba_401909866.json")})
    first = data.props("basketball_nba", "401909866")
    assert (first["count"], first["as_of"], first["stale"]) == (7, "2026-10-26T12:00:00Z", False)
    # Eleven minutes on, DraftKings has pulled the props: the cache serves its last copy, marked stale
    clock.advance(11 * 60)
    del fake.routes[route]
    old = data.props("basketball_nba", "401909866")
    assert (old["count"], old["as_of"], old["stale"], old["reason"]) == (7, "2026-10-26T12:00:00Z", True, None)
    # Once ESPN answers again the lines are fresh
    fake.routes[route] = load("props_nba_401909866.json")
    again = data.props("basketball_nba", "401909866")
    assert (again["as_of"], again["stale"]) == ("2026-10-26T12:11:00Z", False)


def test_props_not_posted_gives_a_reason():
    data, fake = player_data()
    result = data.props("basketball_nba", "401909876")
    assert result["players"] == {} and result["count"] == 0
    assert result["reason"].startswith("DraftKings hasn't posted player props")
    assert result["as_of"] is None and result["stale"] is False
    assert data.props("basketball_nba", "1; drop")["reason"].startswith("ESPN event ids are numbers")
    assert data.props("basketball_nba", "١٢٣")["reason"].startswith("ESPN event ids are numbers")
    assert len(fake.calls) == 1


# --- the card ---

def test_card_is_a_compact_profile_built_from_every_source():
    data, _ = player_data(LEBRON)
    card = data.card("basketball_nba", "1966")
    assert card["reason"] is None and card["missing"] == []
    assert (card["name"], card["team"], card["position"]) == ("LeBron James", "Philadelphia 76ers", "F")
    assert card["next_game"]["opponent"] == "IND"
    assert card["season_label"] == "2026-27"
    assert card["sample"] == {"season": 3, "last5": 5, "last10": 10}
    assert card["season"]["pts"] == 16.33  # (18 + 17 + 14) / 3
    assert card["last5"]["pts"] == 18.4
    assert card["last10"]["pts"] == 21.4
    assert len(card["games"]) == 10
    assert card["splits"]["home"] == {"games": 2, "stats": {
        "min": 33.0, "fg3m": 0.5, "reb": 6.0, "ast": 5.0, "blk": 0.5, "stl": 2.5, "tov": 4.0, "pts": 17.5,
    }}
    assert card["splits"]["away"]["games"] == 1
    assert [item["source"] for item in card["news"]] == ["RotoWire", "RotoWire", "ESPN"]
    assert all(len(item["summary"]) <= 161 for item in card["news"])
    assert datetime.datetime.strptime(card["as_of"], "%Y-%m-%dT%H:%M:%SZ")
    assert len(json.dumps(card)) < 6000  # plain JSON, small enough to hand to the assistant


def test_card_lookups_are_cached():
    data, fake = player_data(LEBRON)
    data.card("basketball_nba", "1966")
    calls = len(fake.calls)
    data.card("basketball_nba", "1966")
    data.gamelog("basketball_nba", "1966")
    assert len(fake.calls) == calls


def test_card_for_an_unknown_player_is_empty_with_reasons():
    data, _ = player_data()
    card = data.card("basketball_nba", "424242")
    assert card["reason"] == "ESPN has no basketball_nba player with id 424242."
    assert card["games"] == [] and card["season"] == {} and card["news"] == []
    assert card["sample"] == {"season": 0, "last5": 0, "last10": 0}
    assert {line.split(":")[0] for line in card["missing"]} == {"profile", "games", "news", "next game"}
    assert data.card("darts", "1")["reason"] == "Player data doesn't cover 'darts'."


# --- hit rates ---

def test_hit_rates_show_sample_sizes_and_count_pushes():
    data, _ = player_data(LEBRON)
    rows = data.gamelog("basketball_nba", "1966", last=100)["rows"]  # this season's 3 games + last season's 12
    rates = hit_rates(rows, "pts", 18.5)
    # Points, newest first: 18 17 14 | 24 19 23 27 18 28 26 ...
    assert (rates["last5"], rates["last10"], rates["season"]) == ("2/5", "6/10", "0/3")
    assert rates["games"] == {"last5": 5, "last10": 10, "season": 3}
    assert rates["reason"] is None
    whole = hit_rates(rows, "pts", 18)
    assert whole["last5"] == "2/5" and whole["pushes"]["last5"] == 1  # 18 on an 18 line is a push
    combo = hit_rates(rows, "pts+reb+ast", 33.5)
    assert combo["last5"] == "1/5"  # only the 24-12-3 playoff game cleared 33.5


def test_hit_rates_with_few_games_or_no_stat_say_so():
    rows = [{"season": "2026-27", "stats": {"pts": 30}}, {"season": "2026-27", "stats": {}}]
    rates = hit_rates(rows, "pts", 25.5)
    assert (rates["last5"], rates["last10"], rates["season"]) == ("1/1", "1/1", "1/1")
    assert hit_rates(rows, "blk", 0.5)["reason"] == "No games with a blk figure."
    assert hit_rates(rows, "pts", None)["reason"] == "No line to compare against."


def test_stat_value_reads_keys_sums_and_hockey_plus_minus():
    row = {"stats": {"pts": 20, "reb": 8, "ast": 5, "+/-": -2, "PASS YDS": 250, "RUSH YDS": 12}}
    assert stat_value(row, "pts+reb+ast") == 33
    assert stat_value(row, "+/-") == -2
    assert stat_value(row, "pass yds + rush yds") == 262
    assert stat_value(row, "pts+stl") is None
