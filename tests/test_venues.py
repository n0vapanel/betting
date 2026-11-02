"""The venue table (data/venues.json, data/venues.py): its rows checked against the captures they were seeded from
(MLB StatsAPI venues, Open-Meteo geocodes and elevations, ESPN scoreboards), and its lookups and validation."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from telegram_betting_bot_simple.data.venues import (VENUES_PATH, VenueError, VenueTable, approx_venue,
                                                       default_venues, load_venues, normalize_name, surface_for)

FIXTURES = Path(__file__).parent / "fixtures" / "context"
LEAGUE_TEAMS = {"americanfootball_nfl": 32, "baseball_mlb": 30, "basketball_nba": 30, "icehockey_nhl": 32,
                "basketball_wnba": 15, "soccer_usa_mls": 30, "soccer_epl": 20}


def fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


@pytest.fixture(scope="module")
def table():
    return load_venues()


def raw_rows():
    return json.loads(VENUES_PATH.read_text(encoding="utf-8"))


# --- the table's contents ----------------------------------------------------------------------------------------

def test_the_table_loads_and_has_one_row_per_building(table):
    assert 160 <= len(table) <= 185
    assert default_venues() is default_venues() and len(default_venues()) == len(table)
    # Shared buildings are one row with every team
    def teams(venue_id):
        return {(team["league"], team["name"]) for team in table.by_id[venue_id]["teams"]}
    assert teams("metlife-stadium") == {("americanfootball_nfl", "New York Giants"),
                                        ("americanfootball_nfl", "New York Jets")}
    assert teams("sofi-stadium") == {("americanfootball_nfl", "Los Angeles Rams"),
                                     ("americanfootball_nfl", "Los Angeles Chargers")}
    assert {name for _, name in teams("crypto-com-arena")} == {"Los Angeles Lakers", "Los Angeles Kings",
                                                              "Los Angeles Sparks"}
    assert teams("yankee-stadium") == {("baseball_mlb", "New York Yankees"), ("soccer_usa_mls", "New York City FC")}
    assert {league for league, _ in teams("delta-center")} == {"basketball_nba", "icehockey_nhl"}


def test_every_team_of_the_seven_leagues_has_a_home_row(table):
    counts = {league: 0 for league in LEAGUE_TEAMS}
    for row in table:
        for team in row["teams"]:
            if team["league"] in counts:
                counts[team["league"]] += 1
    assert counts == LEAGUE_TEAMS
    captured = fixture("espn_home_venues.json")["leagues"]
    for league in LEAGUE_TEAMS:
        for team_id, team in captured[league]["teams"].items():
            row = table.find(league, home_team_id=team_id)
            assert row is not None, (league, team["name"])
            assert any(t["espn_team_id"] == team_id and t["name"] for t in row["teams"])


def test_espn_venue_ids_come_from_the_captured_scoreboards(table):
    """Each team's usual home venue on ESPN's boards is its row's ESPN id; every venue id seen resolves to a row."""
    captured = fixture("espn_home_venues.json")["leagues"]
    for league, data in captured.items():
        for team_id, team in data["teams"].items():
            usual = max(team["venues"], key=team["venues"].get)
            found = table.find(league, espn_venue_id=usual)
            assert found is not None, (league, team["name"], usual)
            assert found["id"] == table.find(league, home_team_id=team_id)["id"], (league, team["name"])
            # ESPN's own spelling of the name finds the same row
            assert table.find(league, name=data["venue_names"][usual])["id"] == found["id"], (league, usual)
            if league not in LEAGUE_TEAMS:
                continue  # Liga MX and CONMEBOL: only the altitude venues' clubs are in the table
            for venue_id in team["venues"]:  # an occasional other home (the Spurs in Austin) is a row too
                assert table.find(league, espn_venue_id=venue_id) is not None, (league, team["name"], venue_id)


def test_mlb_rows_match_the_statsapi_capture(table):
    captured = {venue["id"]: venue for venue in fixture("mlb_venues.json")["venues"]}
    rows = [row for row in table if row["mlb_id"] is not None]
    assert len(rows) == 30 and {row["mlb_id"] for row in rows} == set(captured)
    roofs = {"Open": "open", "Retractable": "retractable", "Dome": "dome"}
    surfaces = {"Grass": "grass", "Artificial Turf": "turf"}
    for row in rows:
        venue = captured[row["mlb_id"]]
        location, field = venue["location"], venue["fieldInfo"]
        assert row["lat"] == pytest.approx(location["defaultCoordinates"]["latitude"], abs=1e-4), row["id"]
        assert row["lon"] == pytest.approx(location["defaultCoordinates"]["longitude"], abs=1e-4), row["id"]
        assert row["elevation_m"] == round(location["elevation"] * 0.3048), row["id"]  # StatsAPI gives feet
        assert row["azimuth_deg"] == location["azimuthAngle"]
        assert row["roof"] == roofs[field["roofType"]] and row["surface"] == surfaces[field["turfType"]]
        assert row["capacity"] == field["capacity"] and row["tz"] == venue["timeZone"]["id"]
        park = row["park"]
        assert 85 <= park["runs"] <= 120 and 70 <= park["hr"] <= 130
        assert "Savant" in park["source"] and "20" in park["source"]  # its source and years
    coors = table.by_id["coors-field"]
    assert (coors["elevation_m"], coors["azimuth_deg"], coors["park"]["runs"]) == (1582, 4.0, 113)


def test_every_row_is_within_40_km_of_its_city_geocode(table):
    cities = fixture("geocodes.json")["cities"]
    for row in table:
        key = f"{row['city']}|{row['region'] or ''}|{row['country']}"
        geocode = cities[key]
        assert geocode["country_code"] == row["country"], key
        distance = haversine_km(row["lat"], row["lon"], geocode["latitude"], geocode["longitude"])
        assert distance <= 40, (row["id"], key, round(distance, 1))


def test_elevations_come_from_open_meteo(table):
    captured = fixture("elevations.json")["elevation_m"]
    assert set(captured) == {row["id"] for row in table}
    for row in table:
        if row["mlb_id"] is None:
            assert row["elevation_m"] == round(captured[row["id"]]), row["id"]
        else:  # StatsAPI's own figure, which agrees with Open-Meteo's model
            assert abs(row["elevation_m"] - captured[row["id"]]) <= 20, row["id"]


def test_the_altitude_venues(table):
    for venue_id, country in (("estadio-azteca", "MX"), ("estadio-rodrigo-paz-delgado", "EC"),
                              ("estadio-hernando-siles", "BO"), ("estadio-el-campin", "CO")):
        row = table.by_id[venue_id]
        assert row["country"] == country and row["elevation_m"] >= 2000 and row["teams"]
    assert table.by_id["estadio-hernando-siles"]["elevation_m"] > 3500
    for venue_id in ("empower-field", "coors-field", "ball-arena", "dicks-sporting-goods-park"):
        assert 1500 <= table.by_id[venue_id]["elevation_m"] <= 1650, venue_id
    assert 1250 <= table.by_id["delta-center"]["elevation_m"] <= 1350  # Utah, the NBA's other altitude home
    assert table.find("soccer_mexico_ligamx", name="Estadio Banorte")["id"] == "estadio-azteca"


def test_rows_carry_their_sources(table):
    for row in table:
        assert row["source"] and row["as_of"] == "2026-10-27" and row["approx"] is False
        if row["mlb_id"] is None:
            assert row["park"] is None and row["azimuth_deg"] is None


# --- lookups -------------------------------------------------------------------------------------------------------

def test_find_tries_the_espn_id_then_the_name_then_the_home_team(table):
    # The ESPN id wins over a name and a team that point elsewhere
    assert table.find("basketball_nba", espn_venue_id="892", name="TD Garden", home_team_id="2")["id"] == "ball-arena"
    # An unknown id falls through to the name, then to the home team
    assert table.find("basketball_nba", espn_venue_id="999999", name="TD Garden", home_team_id="7")["id"] == "td-garden"
    assert table.find("basketball_nba", espn_venue_id="999999", name="Nowhere Arena", home_team_id="7")["id"] == \
        "ball-arena"
    # At a neutral site the home team's building is no answer
    assert table.find("basketball_nba", name="Nowhere Arena", home_team_id="7", neutral=True) is None
    assert table.find("americanfootball_nfl", espn_venue_id="2455", neutral=True)["id"] == "wembley-stadium"
    assert table.find("basketball_nba") is None
    assert table.find("basketball_nba", home_team_id="7", neutral=False)["name"] == "Ball Arena"
    # ESPN gives the same building different ids in different sports
    assert table.find("soccer_usa_mls", espn_venue_id="7485")["id"] == table.find(
        "americanfootball_nfl", espn_venue_id="5348")["id"] == "mercedes-benz-stadium"


def test_names_match_whatever_their_spelling(table):
    assert table.find("icehockey_nhl", name="crypto.com Arena")["id"] == "crypto-com-arena"
    assert table.find("soccer_epl", name="St. James' Park")["id"] == "st-james-park"
    assert table.find("soccer_conmebol_copa_libertadores", name="Estadio Nemesio Camacho El Campín")["id"] == \
        "estadio-el-campin"
    assert table.find("basketball_nba", name="Toyota Center (Houston)")["id"] == "toyota-center"
    assert table.find("icehockey_nhl", name="SAP Center at San Jose")["id"] == "sap-center"
    assert table.find("americanfootball_nfl", name="Reliant Stadium")["id"] == "nrg-stadium"  # ESPN's old name
    assert normalize_name("  Inter&Co   Stadium ") == "inter and co stadium"


def test_a_found_row_is_a_copy_with_the_leagues_surface(table):
    assert table.find("icehockey_nhl", home_team_id="17")["surface"] == "ice"  # Ball Arena for the Avalanche
    assert table.find("basketball_nba", home_team_id="7")["surface"] == "hardwood"
    assert table.find("basketball_wnba", espn_venue_id="7224")["surface"] == "hardwood"  # the Storm on the Kraken's
    assert table.find("americanfootball_nfl", home_team_id="7")["surface"] == "grass"
    assert surface_for(table.by_id["lumen-field"], "soccer_usa_mls") == "turf"
    found = table.find("baseball_mlb", home_team_id="27")
    found["name"] = "changed"
    assert table.by_id["coors-field"]["name"] == "Coors Field"
    assert table.by_id("coors-field") is table.by_id["coors-field"] and table.by_id("nowhere") is None


def test_an_unknown_venue_is_placed_at_its_citys_geocode():
    geocode = {"name": "Boise", "lat": 43.6135, "lon": -116.20345, "elevation": 824.0, "country": "US",
               "admin1": "Idaho", "timezone": "America/Boise", "population": 235684}
    row = approx_venue("Albertsons Stadium", "Boise", "US", geocode, indoor=False, espn_venue_id="3911")
    assert row["approx"] is True and row["roof"] == "open" and row["elevation_m"] == 824
    assert (row["lat"], row["tz"], row["espn_ids"]) == (43.6135, "America/Boise", ["3911"])
    assert row["lon"] == pytest.approx(-116.2035, abs=1e-4)
    assert row["park"] is None and row["teams"] == [] and "Open-Meteo" in row["source"]
    indoor = approx_venue("Some Arena", "Boise", "US", {**geocode, "elevation": None, "timezone": "Nowhere/Zone"},
                          indoor=True)
    assert indoor["roof"] == "dome" and indoor["elevation_m"] is None and indoor["tz"] == "UTC"
    assert approx_venue("X", "Y", "US", None) is None and approx_venue("X", "Y", "US", {"lat": None}) is None
    abroad = approx_venue("The Den", "London", "England", {"name": "London", "lat": 51.48, "lon": -0.05})
    assert abroad["country"] == "GB" and abroad["tz"] == "UTC" and abroad["roof"] == "open"


# --- validation ----------------------------------------------------------------------------------------------------

def _write(tmp_path, rows):
    path = tmp_path / "venues.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _row(venue_id):
    return copy.deepcopy(next(row for row in raw_rows() if row["id"] == venue_id))


@pytest.mark.parametrize("change, message", [
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "used twice"),
    (lambda rows: rows[1].update(espn_ids=list(rows[0]["espn_ids"])), "ESPN venue id"),
    (lambda rows: rows[1].update(aliases=[rows[0]["name"]]), "the name"),
    (lambda rows: rows[0].update(roof="tent"), "roof"),
    (lambda rows: rows[0].update(surface="clay"), "surface"),
    (lambda rows: rows[0].update(tz="Mars/Olympus"), "time zone"),
    (lambda rows: rows[0].update(lat=123.0), "coordinates"),
    (lambda rows: rows[0].update(park=None), "park factors"),
    (lambda rows: rows[0].update(azimuth_deg=None), "azimuth"),
    (lambda rows: rows[1].update(park={"runs": 100, "hr": 100, "source": "x"}), "MLB parks only"),
    (lambda rows: rows[1]["teams"][0].update(league="basketball_mars"), "league"),
    (lambda rows: rows[0].pop("city"), "missing city"),
    (lambda rows: rows[1].update(teams=rows[1]["teams"] + [dict(rows[1]["teams"][0])]), "already plays"),
    (lambda rows: rows[0].update(espn_ids=[19]), "espn_ids"),
    (lambda rows: rows[0].update(id="Coors Field"), "lowercase"),
    (lambda rows: rows[0].update(capacity=-5), "capacity"),
])
def test_a_bad_row_is_refused(tmp_path, change, message):
    rows = [_row("coors-field"), _row("ball-arena")]
    VenueTable(rows)  # the two rows as they are pass
    change(rows)
    with pytest.raises(VenueError, match=message):
        load_venues(_write(tmp_path, rows))


def test_the_table_file_must_be_a_list(tmp_path):
    with pytest.raises(VenueError):
        load_venues(_write(tmp_path, {"rows": []}))
