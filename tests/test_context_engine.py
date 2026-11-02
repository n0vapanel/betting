"""engine/context.py: the context maths. Attribution explains the market's line and sums to it exactly; only a
forecast or roof change since the odds moves a number, by clamp(0.5 x delta, cap). No network, no clock."""

from __future__ import annotations

import datetime
import math
import random
from statistics import NormalDist

import pytest

from telegram_betting_bot_simple.engine import context as C
from telegram_betting_bot_simple.engine import soccer
from telegram_betting_bot_simple.engine.factors import CONFIDENCES, NOISY, NORMAL, check_sum, fold

N = NormalDist()

# Venue rows in data/venues.json's shape (only the fields the engine reads)
COORS = {"id": "coors-field", "name": "Coors Field", "city": "Denver", "lat": 39.7559, "lon": -104.9942,
         "elevation_m": 1580, "tz": "America/Denver", "roof": "open", "azimuth_deg": 4,
         "park": {"runs": 113, "hr": 105, "source": "Baseball Savant, 3-year, 2023–25"},
         "source": "MLB StatsAPI venue 19; Savant park factors"}
WRIGLEY = {"id": "wrigley-field", "name": "Wrigley Field", "elevation_m": 181, "roof": "open", "azimuth_deg": 37,
           "park": {"runs": 98, "hr": 101, "source": "Baseball Savant, 3-year, 2023–25"}}
DAIKIN = {"id": "daikin-park", "name": "Daikin Park", "elevation_m": 13, "roof": "retractable", "azimuth_deg": 343,
          "park": {"runs": 100, "hr": 104, "source": "Baseball Savant"}}
BALL_ARENA = {"id": "ball-arena", "name": "Ball Arena", "elevation_m": 1580, "roof": "dome"}
DELTA_CENTER = {"id": "delta-center", "name": "Delta Center", "elevation_m": 1288, "roof": "dome"}
FOOTPRINT = {"id": "footprint-center", "name": "Footprint Center", "elevation_m": 331, "roof": "dome"}
EMPOWER = {"id": "empower-field", "name": "Empower Field at Mile High", "elevation_m": 1580, "roof": "open"}
HIGHMARK = {"id": "highmark-stadium", "name": "Highmark Stadium", "elevation_m": 222, "roof": "open"}
SOFI = {"id": "sofi-stadium", "name": "SoFi Stadium", "elevation_m": 30, "roof": "dome"}
LUCAS_OIL = {"id": "lucas-oil-stadium", "name": "Lucas Oil Stadium", "elevation_m": 218, "roof": "retractable"}
AZTECA = {"id": "estadio-azteca", "name": "Estadio Azteca", "elevation_m": 2240, "roof": "open"}
ANFIELD = {"id": "anfield", "name": "Anfield", "elevation_m": 40, "roof": "open"}


def mph(value: float) -> float:
    """km/h for a wind given in mph (Open-Meteo is asked for km/h)."""
    return value * C.KMH_PER_MPH


def keys(factors):
    return {f.key: f.delta for f in factors}


# --- sports and parameters ----------------------------------------------------------------------------------------

def test_sport_of_maps_the_apps_league_keys() -> None:
    assert C.sport_of("americanfootball_nfl") == "nfl"
    assert C.sport_of("americanfootball_ncaaf") == "ncaaf"
    assert C.sport_of("basketball_nba") == "nba"
    assert C.sport_of("basketball_wnba") == "wnba"
    assert C.sport_of("basketball_ncaab") == "ncaab"
    assert C.sport_of("icehockey_nhl") == "nhl"
    assert C.sport_of("baseball_mlb") == "mlb"
    assert C.sport_of("baseball_mlb_preseason") == "mlb"
    for league in ("soccer_epl", "soccer_usa_mls", "soccer_uefa_champs_league", "soccer_mexico_ligamx"):
        assert C.sport_of(league) == "soccer"
    assert C.sport_of("nhl") == "nhl"  # a sport key is its own sport
    for league in ("tennis_atp", "mma_mixed_martial_arts", "basketball_wncaab", "americanfootball_cfl",
                   "soccer_fifa_world_cup_winner", "", None):
        assert C.sport_of(league) is None
    assert C.params("icehockey_nhl") is C.SPORTS["nhl"]
    with pytest.raises(ValueError):
        C.params("tennis_atp")


def test_sport_parameters_are_the_specs() -> None:
    sds = {"nfl": (12.7, 13.2), "ncaaf": (14.5, 15.0), "nba": (12.0, 18.0), "wnba": (11.0, 16.0),
           "ncaab": (10.5, 16.0), "nhl": (2.3, 2.2), "mlb": (4.3, 4.4)}
    for sport, (margin_sd, total_sd) in sds.items():
        assert (C.SPORTS[sport].margin_sd, C.SPORTS[sport].total_sd) == (margin_sd, total_sd)
    totals = {"nfl": 2.0, "ncaaf": 2.0, "mlb": 0.4, "nba": 1.5, "wnba": 1.5, "ncaab": 1.5, "nhl": 0.15,
              "soccer": 0.30}
    for sport, cap in totals.items():
        assert C.SPORTS[sport].caps["total"] == cap
        assert C.SPORTS[sport].caps["win"] == 2.5
    assert C.SPORTS["nfl"].caps["side"] == C.SPORTS["ncaaf"].caps["side"] == 1.0
    assert C.SPORTS["soccer"].caps["side"] == 0.15  # 0.15 goals per team
    # Other sides: the 2.5-point win-chance cap through the margin SD (NBA about 0.75 pts)
    assert C.SPORTS["nba"].caps["side"] == pytest.approx(12.0 * N.inv_cdf(0.525))
    assert C.SPORTS["nba"].caps["side"] == pytest.approx(0.7525, abs=1e-4)
    priors = {"nfl": 0.539, "nba": 0.554, "nhl": 0.522, "mlb": 0.529, "wnba": 0.55, "ncaaf": 0.57, "ncaab": 0.60,
              "soccer": 0.45}
    assert {k: C.SPORTS[k].home_prior for k in priors} == priors
    assert C.SPORTS["soccer"].draw_prior == 0.27 and not C.SPORTS["soccer"].two_way and C.SPORTS["nba"].two_way
    assert {k: C.SPORTS[k].unit for k in C.SPORTS} == {"nfl": "pts", "ncaaf": "pts", "nba": "pts", "wnba": "pts",
                                                       "ncaab": "pts", "nhl": "goals", "mlb": "runs", "soccer": "goals"}
    assert {k for k in C.SPORTS if C.SPORTS[k].outdoor} == {"nfl", "ncaaf", "mlb", "soccer"}
    assert C.ADJUST_WEIGHT == 0.5


# --- distance and time zones -------------------------------------------------------------------------------------

def test_haversine_matches_known_distances() -> None:
    boston_denver = C.haversine_km(42.3601, -71.0589, 39.7392, -104.9903)
    assert boston_denver == pytest.approx(2830, abs=15)  # "about 2,830 km"
    assert boston_denver == pytest.approx(2841.3, abs=1)
    assert C.haversine_km(39.7392, -104.9903, 42.3601, -71.0589) == pytest.approx(boston_denver)
    assert C.haversine_km(51.5074, -0.1278, 48.8566, 2.3522) == pytest.approx(343.6, abs=1)  # London-Paris
    assert C.haversine_km(40.7128, -74.0060, 34.0522, -118.2437) == pytest.approx(3936, abs=3)  # New York-LA
    assert C.haversine_km(10.0, 20.0, 10.0, 20.0) == 0.0
    assert C.haversine_km(0.0, 0.0, 0.0, 180.0) == pytest.approx(math.pi * C.EARTH_RADIUS_KM)


def test_tz_shift_is_signed_eastward() -> None:
    at = datetime.datetime(2026, 10, 27, 23, 0, tzinfo=datetime.timezone.utc)
    assert C.tz_shift_hours("America/Los_Angeles", "America/New_York", at) == 3.0
    assert C.tz_shift_hours("America/New_York", "America/Los_Angeles", at) == -3.0
    assert C.tz_shift_hours("America/Chicago", "America/Chicago", at) == 0.0
    # A naive datetime is UTC; ISO strings and epoch seconds work too
    assert C.tz_shift_hours("America/Los_Angeles", "America/New_York", at.replace(tzinfo=None)) == 3.0
    assert C.tz_shift_hours("America/Los_Angeles", "America/New_York", "2026-10-27T23:00:00Z") == 3.0
    assert C.tz_shift_hours("America/Los_Angeles", "America/New_York", at.timestamp()) == 3.0
    with pytest.raises(ValueError):
        C.tz_shift_hours("Mars/Olympus_Mons", "America/New_York", at)
    with pytest.raises(ValueError):
        C.tz_shift_hours("America/New_York", "America/Denver", "not a time")


def test_tz_shift_follows_daylight_saving_on_the_day() -> None:
    # The US springs forward on 8 March 2026, the UK on 29 March: for three weeks New York is 4 h behind London
    assert C.tz_shift_hours("Europe/London", "America/New_York", "2026-03-15T18:00:00Z") == -4.0
    assert C.tz_shift_hours("Europe/London", "America/New_York", "2026-04-15T18:00:00Z") == -5.0
    # The UK falls back on 25 October 2026, the US on 1 November: Chicago -> London is 5 h that week, then 6
    assert C.tz_shift_hours("America/Chicago", "Europe/London", "2026-10-28T18:00:00Z") == 5.0
    assert C.tz_shift_hours("America/Chicago", "Europe/London", "2026-11-05T18:00:00Z") == 6.0
    # Arizona keeps standard time: Phoenix -> Denver is +1 in summer, 0 in winter
    assert C.tz_shift_hours("America/Phoenix", "America/Denver", "2026-07-01T02:00:00Z") == 1.0
    assert C.tz_shift_hours("America/Phoenix", "America/Denver", "2026-01-10T02:00:00Z") == 0.0
    # Brisbane keeps standard time, Sydney doesn't (southern summer)
    assert C.tz_shift_hours("Australia/Brisbane", "Australia/Sydney", "2026-01-10T08:00:00Z") == 1.0
    assert C.tz_shift_hours("Australia/Brisbane", "Australia/Sydney", "2026-07-10T08:00:00Z") == 0.0
    # Read at one instant: between New York's change (07:00 UTC) and Los Angeles' (10:00 UTC) the gap is 4 h
    assert C.tz_shift_hours("America/Los_Angeles", "America/New_York", "2026-03-08T06:30:00Z") == 3.0
    assert C.tz_shift_hours("America/Los_Angeles", "America/New_York", "2026-03-08T08:00:00Z") == 4.0
    assert C.tz_shift_hours("America/Los_Angeles", "America/New_York", "2026-03-08T11:00:00Z") == 3.0


def test_rest_and_jet_lag_rules() -> None:
    assert C.days_since_last({"rest_days": 0}) == 1
    assert C.days_since_last({"rest_days": 3, "days_since_last": 4}) == 4
    assert C.days_since_last({}) is None and C.days_since_last(None) is None
    assert C.on_zero_rest({"rest_days": 0}) and not C.on_zero_rest({"rest_days": 1})
    assert C.on_zero_rest({"rest_days": 1, "back_to_back": True})
    assert not C.on_zero_rest({"rest_days": 0, "back_to_back": False})
    assert C.jet_lag_hours({"tz_shift_h": 3, "rest_days": 1}) == 3
    assert C.jet_lag_hours({"tz_shift_h": 2, "rest_days": 2}) == 2  # 3 days since the last game
    assert C.jet_lag_hours({"tz_shift_h": 3, "rest_days": 3}) is None  # 4 days: recovered
    assert C.jet_lag_hours({"tz_shift_h": 1.5, "rest_days": 0}) is None
    assert C.jet_lag_hours({"tz_shift_h": -3, "rest_days": 0}) is None  # westward
    assert C.jet_lag_hours({"tz_shift_h": 3}) is None  # no last game known


# --- probabilities from points -----------------------------------------------------------------------------------

def test_implied_margin_and_margin_to_prob_use_the_margin_sd() -> None:
    assert C.implied_margin("nfl", 0.539) == pytest.approx(12.7 * N.inv_cdf(0.539))
    assert C.implied_margin("americanfootball_nfl", 0.5) == pytest.approx(0.0, abs=1e-12)
    assert C.margin_to_prob("nba", 3.0) == pytest.approx(N.cdf(3.0 / 12.0))
    assert C.margin_to_prob("nhl", 0.0) == pytest.approx(0.5)
    for sport in ("nfl", "ncaaf", "nba", "wnba", "ncaab", "nhl", "mlb"):
        for p in (0.1, 0.35, 0.5, 0.62, 0.9):
            assert C.margin_to_prob(sport, C.implied_margin(sport, p)) == pytest.approx(p, abs=1e-12)
        assert C.implied_margin(sport, 0.6) > C.implied_margin(sport, 0.55) > 0
    with pytest.raises(ValueError):
        C.implied_margin("nba", 1.5)
    with pytest.raises(ValueError):
        C.implied_margin("nba", None)  # type: ignore[arg-type]
    assert math.isfinite(C.implied_margin("nba", 1.0))  # kept off the edge


def test_soccer_margins_go_through_the_goals_model() -> None:
    supremacy = C.implied_margin("soccer", 0.6)
    assert C.margin_to_prob("soccer", supremacy) == pytest.approx(0.6, abs=1e-6)
    home, _draw, away = soccer.outcome_probs(soccer.grid(*C._soccer_rates(supremacy, 2.8)))
    assert home == pytest.approx(0.6, abs=1e-6)
    # Level teams: no supremacy, equal chances
    level = soccer.outcome_probs(soccer.grid(1.4, 1.4))
    assert C.implied_margin("soccer", level[0]) == pytest.approx(0.0, abs=1e-6)
    # With a draw chance it matches the home-minus-away gap
    gap = C.implied_margin("soccer", 0.45, p_draw=0.27)
    home, _draw, away = soccer.outcome_probs(soccer.grid(*C._soccer_rates(gap, 2.8)))
    assert home - away == pytest.approx(0.45 - 0.28, abs=1e-6)
    # A higher-scoring match needs more supremacy for the same chance
    assert C.implied_margin("soccer", 0.6, total=3.4) > C.implied_margin("soccer", 0.6, total=2.2)
    with pytest.raises(ValueError):
        C.implied_margin("soccer", 0.7, p_draw=0.3)


def test_home_edge_from_rates_and_priors() -> None:
    assert C.home_edge("nfl") == pytest.approx(12.7 * N.inv_cdf(0.539))  # about +1.2 pts
    assert C.home_edge("nhl") == pytest.approx(0.127, abs=0.001)  # the report's "+0.13 goals a game"
    assert C.home_edge("nba", 0.58) == pytest.approx(12.0 * N.inv_cdf(0.58))
    edge = C.home_edge("soccer")  # 45% home, 27% draw
    assert 0.3 < edge < 0.45  # the notes' "about 0.35-0.4 goals"
    home, _draw, away = soccer.outcome_probs(soccer.grid(*C._soccer_rates(edge, 2.8)))
    assert home - away == pytest.approx(0.17, abs=1e-6)
    assert C.home_edge("soccer", 0.48, 0.26) > edge


# --- weather -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("wind, points", [(0, 0.0), (8, 0.0), (10, -1.1), (12, -2.2), (14, -3.0), (16, -3.8),
                                          (25, -3.8)])
def test_nfl_wind_is_piecewise_linear(wind: float, points: float) -> None:
    assert C.nfl_wind_points(wind) == pytest.approx(points)
    factors = C.weather_factors("nfl", HIGHMARK, {"wind_kmh": mph(wind), "temp_c": 12, "precip_prob": 0,
                                                  "precip_mm": 0})
    assert keys(factors) == ({"wind": pytest.approx(points)} if points else {})
    if points:
        assert factors[0].confidence == NORMAL and factors[0].unit == "pts"
        assert factors[0].label == "Wind at kick-off" and factors[0].source == C.SRC_FORECAST
        assert "mph" in factors[0].evidence and "km/h" in factors[0].evidence


def test_nfl_cold_and_cold_with_wind() -> None:
    assert keys(C.weather_factors("nfl", HIGHMARK, {"temp_f": 32, "wind_mph": 5})) == {"cold": -1.0}
    assert keys(C.weather_factors("nfl", HIGHMARK, {"temp_f": 33, "wind_mph": 5})) == {}
    assert keys(C.weather_factors("nfl", HIGHMARK, {"temp_c": -6, "wind_mph": 14.9})) == {
        "wind": pytest.approx(C.nfl_wind_points(14.9)), "cold": -1.0}
    # Freezing with 15 mph wind: -3.4 wind, -1 cold, -2 more: -6.4, the CBS gap (44.3 calm -> 37.9)
    factors = C.weather_factors("nfl", HIGHMARK, {"temp_f": 20, "wind_mph": 15})
    assert keys(factors) == {"wind": pytest.approx(-3.4), "cold": -1.0, "cold_wind": -2.0}
    assert C.weather_total_effect("nfl", HIGHMARK, {"temp_f": 20, "wind_mph": 15}) == pytest.approx(-6.4)
    assert all(f.confidence == NORMAL for f in factors)


def test_nfl_rain_needs_both_the_chance_and_the_amount() -> None:
    wet = C.weather_factors("nfl", HIGHMARK, {"temp_c": 10, "wind_kmh": 5, "precip_prob": 60, "precip_mm": 0.5})
    assert keys(wet) == {"rain": -1.0} and wet[0].confidence == NOISY
    assert keys(C.weather_factors("nfl", HIGHMARK, {"temp_c": 10, "precip_prob": 59, "precip_mm": 3})) == {}
    assert keys(C.weather_factors("nfl", HIGHMARK, {"temp_c": 10, "precip_prob": 90, "precip_mm": 0.4})) == {}


def test_college_football_uses_the_nfl_figures() -> None:
    factors = C.weather_factors("americanfootball_ncaaf", HIGHMARK, {"temp_f": 20, "wind_mph": 16})
    assert keys(factors) == {"wind": -3.8, "cold": -1.0, "cold_wind": -2.0}
    assert all("college" in f.evidence for f in factors)


def test_indoor_sports_have_no_weather() -> None:
    storm = {"temp_c": -10, "wind_kmh": 60, "precip_prob": 100, "precip_mm": 8}
    for sport in ("nba", "wnba", "ncaab", "nhl"):
        assert C.weather_factors(sport, {"roof": "open"}, storm) == []
        assert C.weather_total_effect(sport, {"roof": "open"}, storm) == 0.0


def test_a_dome_or_closed_roof_gives_zero_weather() -> None:
    storm = {"temp_c": -10, "wind_kmh": 60, "precip_prob": 100, "precip_mm": 8, "wind_dir": 180}
    assert C.weather_factors("nfl", SOFI, storm) == [] and C.weather_total_effect("nfl", SOFI, storm) == 0.0
    assert C.weather_factors("nfl", HIGHMARK, None) == [] and C.weather_total_effect("nfl", HIGHMARK, None) == 0.0
    # Retractable, state unknown (NFL roofs are never announced): no weather counts, no badge
    assert C.roof_state(LUCAS_OIL) is None and C.roof_unknown(LUCAS_OIL, storm)
    assert not C.weather_counts(LUCAS_OIL, storm) and C.weather_factors("nfl", LUCAS_OIL, storm) == []
    assert C.weather_total_effect("nfl", LUCAS_OIL, storm) == 0.0
    assert C.badges("nfl", LUCAS_OIL, storm, None) == []
    assert C.weather_total_effect("nfl", {**LUCAS_OIL, "roof_state": "open"}, storm) == pytest.approx(
        -3.8 - 1.0 - 2.0 - 1.0)  # the venue row's (or an announced) open roof: the weather counts
    # Announced closed (MLB's "Roof Closed", or the flag), or the venue's state: no weather
    closed = {**storm, "announced": {"condition": "Roof Closed", "temp_f": 72, "wind_text": "0 mph, None"}}
    assert C.roof_state(DAIKIN, closed) == "closed" and C.is_covered(DAIKIN, closed)
    assert C.weather_factors("mlb", DAIKIN, closed) == []
    assert C.weather_factors("nfl", LUCAS_OIL, {**storm, "roof_closed": True}) == []
    assert C.weather_factors("nfl", {**LUCAS_OIL, "roof_state": "closed"}, storm) == []
    assert C.weather_total_effect("soccer", {"roof": "dome"}, {"temp_c": 30, "precip_mm": 3}) == 0.0
    # Announced open: the weather counts again, and beats the venue row's state
    sunny = {"temp_f": 90, "announced": {"condition": "Sunny", "temp_f": 90, "wind_text": "3 mph, L To R",
                                         "roof_closed": False}}
    assert C.roof_state({**DAIKIN, "roof_state": "closed"}, sunny) == "open"
    assert keys(C.weather_factors("mlb", DAIKIN, sunny)) == {"temperature": pytest.approx(0.3)}
    assert C.roof_state(SOFI, {"roof_closed": False}) == "dome" and C.roof_state(HIGHMARK) == "open"
    assert C.roof_state(None) == "open"


def test_mlb_temperature() -> None:
    assert keys(C.weather_factors("mlb", WRIGLEY, {"temp_f": 80})) == {"temperature": pytest.approx(0.15)}
    assert keys(C.weather_factors("mlb", WRIGLEY, {"temp_f": 60})) == {"temperature": pytest.approx(-0.15)}
    assert keys(C.weather_factors("mlb", WRIGLEY, {"temp_f": 70})) == {}
    hot = C.weather_factors("mlb", WRIGLEY, {"temp_c": 35})  # 95 F
    assert keys(hot) == {"temperature": pytest.approx(0.375)}
    assert hot[0].confidence == NOISY and hot[0].unit == "runs" and "95°F (35°C)" in hot[0].evidence


def test_mlb_wind_component_from_the_parks_azimuth() -> None:
    # Wrigley faces 37 degrees (home plate to centre field); forecasts give where the wind comes FROM
    assert C.wind_component(37, 217, 15) == pytest.approx(15)  # from behind home plate: straight out
    assert C.wind_component(37, 37, 15) == pytest.approx(-15)  # from centre field: straight in
    assert C.wind_component(37, 127, 15) == pytest.approx(0, abs=1e-9)  # across
    assert C.wind_component(0, 225, 10) == pytest.approx(10 * math.cos(math.radians(45)))
    out = C.weather_factors("mlb", WRIGLEY, {"wind_kmh": mph(15), "wind_dir": 217})
    assert keys(out) == {"wind": pytest.approx(0.8)}  # (15 - 5) x 0.08
    assert out[0].label == "Wind blowing out" and out[0].confidence == NOISY
    blowing_in = C.weather_factors("mlb", WRIGLEY, {"wind_kmh": mph(15), "wind_dir": 37})
    assert keys(blowing_in) == {"wind": pytest.approx(-0.8)} and blowing_in[0].label == "Wind blowing in"
    assert keys(C.weather_factors("mlb", WRIGLEY, {"wind_kmh": mph(15), "wind_dir": 127})) == {}
    assert keys(C.weather_factors("mlb", WRIGLEY, {"wind_kmh": mph(5), "wind_dir": 217})) == {}  # not above 5 mph
    assert keys(C.weather_factors("mlb", WRIGLEY, {"wind_kmh": mph(30), "wind_dir": 217})) == {"wind": 1.0}
    assert keys(C.weather_factors("mlb", WRIGLEY, {"wind_kmh": mph(40), "wind_dir": 37})) == {"wind": -1.0}
    # No azimuth, or no direction: no wind effect
    assert keys(C.weather_factors("mlb", {**WRIGLEY, "azimuth_deg": None}, {"wind_kmh": 40, "wind_dir": 217})) == {}
    assert keys(C.weather_factors("mlb", WRIGLEY, {"wind_kmh": 40})) == {}
    assert C.mlb_wind_runs(-12) == pytest.approx(-0.56) and C.mlb_wind_runs(3) == 0.0


def test_mlb_announced_wind_text_wins_over_the_forecast() -> None:
    # "14 mph, Out To RF": 14 x cos 45 = 9.9 mph out, 4.9 above 5 mph, x 0.08 = +0.39 runs
    forecast = {"temp_c": 20, "wind_kmh": 40, "wind_dir": 37, "fetched_at": "2026-07-18T15:00:00Z"}
    announced = {"condition": "Partly Cloudy", "temp_f": 81, "wind_mph": 14, "wind_text": "14 mph, Out To RF",
                 "wind_kind": "out", "field": "RF", "roof_closed": False}
    factors = C.weather_factors("mlb", WRIGLEY, {**forecast, "announced": announced})
    expected_wind = 0.08 * (14 * math.cos(math.radians(45)) - 5)
    assert keys(factors) == {"temperature": pytest.approx(0.165), "wind": pytest.approx(expected_wind)}
    assert all(f.source == C.SRC_MLB for f in factors)
    assert all(f.as_of == "2026-07-18T15:00:00Z" for f in factors)
    # MLB's own block passed flat works the same; the text alone is enough
    assert keys(C.weather_factors("mlb", WRIGLEY, announced)) == keys(factors)
    text_only = {"temp_f": 70, "announced": {"wind_text": "14 mph, In From LF"}}
    assert keys(C.weather_factors("mlb", WRIGLEY, text_only)) == {"wind": pytest.approx(-expected_wind)}
    assert keys(C.weather_factors("mlb", WRIGLEY, {"temp_f": 70, "wind_text": "10 mph, Out To CF"})) == {
        "wind": pytest.approx(0.4)}
    for calm in ("6 mph, L To R", "8 mph, R To L", "Calm", "0 mph, None", "Varies", "12 mph, Varies"):
        assert keys(C.weather_factors("mlb", WRIGLEY, {"temp_f": 70, "wind_text": calm})) == {}, calm
    # The context block's weather shape (start hour + announcement) reads the same
    block = {"source": "Open-Meteo", "fetched_at": "f", "start": {"t": "x", **forecast}, "announced": announced}
    assert keys(C.weather_factors("mlb", WRIGLEY, block)) == keys(factors)
    # Other sports ignore MLB's announcement
    assert C.read_conditions("nfl", {"wind_kmh": 10, "announced": announced}).wind_mph == pytest.approx(
        10 / C.KMH_PER_MPH)


# Wind and condition strings MLB StatsAPI announced on 10 April, 18 July and 20 September 2026 (schedule with
# hydrate=weather), and how each reads: (out/in component in mph, or None for no wind effect)
MLB_ANNOUNCED = [
    ("Sunny", "10 mph, In From LF", -10 * math.cos(math.radians(45))), ("Clear", "12 mph, Out To CF", 12.0),
    ("Overcast", "12 mph, R To L", 0.0), ("Roof Closed", "0 mph, None", None), ("Dome", "0 mph, None", None),
    ("Partly Cloudy", "7 mph, Out To LF", 7 * math.cos(math.radians(45))), ("Clear", "3 mph, L To R", 0.0),
    ("Cloudy", "8 mph, In From RF", -8 * math.cos(math.radians(45))), ("Sunny", "7 mph, In From CF", -7.0),
    ("Partly Cloudy", "14 mph, Out To RF", 14 * math.cos(math.radians(45))), ("Drizzle", "8 mph, Out To LF",
                                                                              8 * math.cos(math.radians(45))),
    ("Rain", "10 mph, L To R", 0.0), ("Partly Cloudy", "1 mph, Out To LF", math.cos(math.radians(45))),
]


@pytest.mark.parametrize("condition, wind, component", MLB_ANNOUNCED)
def test_every_announced_mlb_wind_seen_live_reads(condition: str, wind: str, component) -> None:
    weather = C.read_conditions("mlb", {"announced": {"condition": condition, "temp_f": 70, "wind_text": wind}})
    assert weather.wind_kind in ("out", "in", "cross", "calm") and weather.wind_source == C.SRC_MLB
    covered = C.is_covered(WRIGLEY, {"announced": {"condition": condition, "wind_text": wind}})
    assert covered == (condition in ("Roof Closed", "Dome"))
    if component is not None:
        got = C.announced_component(weather.wind_mph, weather.wind_kind, weather.wind_field)
        assert got == pytest.approx(component, abs=1e-9)
        assert C.weather_total_effect("mlb", WRIGLEY, {"announced": {"condition": condition, "temp_f": 70,
                                                                     "wind_text": wind}}) == \
            pytest.approx(C.mlb_wind_runs(component))


def test_still_air_at_a_retractable_park_means_the_roof_is_closed() -> None:
    # T-Mobile Park on 10 April 2026: "Overcast", 64 F, "0 mph, None" (the condition doesn't say "Roof Closed")
    seattle = {"name": "T-Mobile Park", "roof": "retractable", "azimuth_deg": 49}
    announced = {"announced": {"condition": "Overcast", "temp_f": 64, "wind_text": "0 mph, None",
                               "roof_closed": False}}
    assert C.roof_state(seattle, announced) == "closed" and C.weather_factors("mlb", seattle, announced) == []
    assert C.badges("mlb", seattle, announced, None) == ["roof"]
    # The same words at an open park are just calm air
    assert C.roof_state(WRIGLEY, announced) == "open"
    assert keys(C.weather_factors("mlb", WRIGLEY, announced)) == {"temperature": pytest.approx(-0.09)}


def test_soccer_weather_multipliers() -> None:
    heat = C.weather_factors("soccer", ANFIELD, {"temp_c": 25, "precip_mm": 0})
    assert keys(heat) == {"heat": pytest.approx(2.8 * 0.068)} and heat[0].confidence == NOISY
    assert keys(C.weather_factors("soccer", ANFIELD, {"temp_c": 3})) == {"cold": pytest.approx(-2.8 * 0.03)}
    assert keys(C.weather_factors("soccer", ANFIELD, {"temp_c": 22})) == {}
    assert keys(C.weather_factors("soccer", ANFIELD, {"temp_c": 6})) == {}
    assert keys(C.weather_factors("soccer", ANFIELD, {"temp_c": 15, "precip_mm": 0.5}, base_total=3.0)) == {
        "rain": pytest.approx(3.0 * 0.014)}
    # Heat and rain together: base x (1.068 x 1.014 - 1), split by each multiplier's share of the log
    both = C.weather_factors("soccer", ANFIELD, {"temp_c": 25, "precip_mm": 1.0}, base_total=3.0)
    total = 3.0 * (1.068 * 1.014 - 1)
    share = math.log(1.068) / (math.log(1.068) + math.log(1.014))
    assert keys(both) == {"heat": pytest.approx(total * share), "rain": pytest.approx(total * (1 - share))}
    assert C.weather_total_effect("soccer", ANFIELD, {"temp_c": 25, "precip_mm": 1.0}, base_total=3.0) == \
        pytest.approx(total)
    assert C.weather_multiplier("soccer", ANFIELD, {"temp_c": 25, "precip_mm": 1.0}) == pytest.approx(1.068 * 1.014)
    assert C.weather_multiplier("nfl", ANFIELD, {"temp_c": 25}) == 1.0


def test_weather_change_in_plain_words() -> None:
    assert C.weather_change("nfl", {"wind_mph": 8}, {"wind_mph": 18}) == "wind 8 → 18 mph"
    assert C.weather_change("soccer", {"wind_kmh": 13, "temp_c": 20}, {"wind_kmh": 29, "temp_c": 12}) == \
        "wind 13 → 29 km/h, 20 → 12°C"
    assert C.weather_change("mlb", {"temp_f": 60, "precip_mm": 0}, {"temp_f": 81, "precip_mm": 2}) == \
        "60 → 81°F, rain now expected"
    assert C.weather_change("nfl", {"wind_mph": 8.2}, {"wind_mph": 8.6}) == ""


# --- attribution -------------------------------------------------------------------------------------------------

def test_denver_altitude_and_utah_scaled() -> None:
    denver = C.attribute_margin("nba", market_margin=6.5, venue=BALL_ARENA, trips={}, neutral=False, as_of="t")
    altitude = next(f for f in denver.factors if f.key == "altitude")
    assert altitude.delta == pytest.approx(2.6) and altitude.confidence == NORMAL
    assert "Ball Arena" in altitude.evidence and "1,580 m" in altitude.evidence
    utah = C.attribute_margin("nba", market_margin=1.0, venue=DELTA_CENTER, trips={}, neutral=False, as_of="t")
    scaled = next(f for f in utah.factors if f.key == "altitude")
    assert scaled.delta == pytest.approx(2.6 * (1288 - 600) / (1580 - 600)) and scaled.confidence == NOISY
    phoenix = C.attribute_margin("nba", market_margin=1.0, venue=FOOTPRINT, trips={}, neutral=False, as_of="t")
    assert "altitude" not in keys(phoenix.factors)
    # A neutral site: no home edge, no altitude
    neutral = C.attribute_margin("nba", market_margin=1.0, venue=BALL_ARENA, trips={}, neutral=True, as_of="t")
    assert keys(neutral.factors) == {"home_field": 0.0, "other": 1.0}
    # NFL at Empower Field: +0.5, small and noisy; no NHL altitude term
    nfl = C.attribute_margin("nfl", market_margin=3.0, venue=EMPOWER, trips={}, neutral=False, as_of="t")
    assert keys(nfl.factors)["altitude"] == 0.5
    assert next(f for f in nfl.factors if f.key == "altitude").confidence == NOISY
    assert "altitude" not in keys(C.attribute_margin("nfl", market_margin=3.0, venue=HIGHMARK, trips={},
                                                     neutral=False, as_of="t").factors)
    assert "altitude" not in keys(C.attribute_margin("nhl", market_margin=0.3, venue=BALL_ARENA, trips={},
                                                     neutral=False, as_of="t").factors)


def test_soccer_altitude() -> None:
    known = C.attribute_margin("soccer", market_margin=0.9, venue=AZTECA,
                               trips={"away": {"home_elevation_m": 20}}, neutral=False, as_of="t")
    assert keys(known.factors)["altitude"] == pytest.approx(0.25 * 2.22)
    unknown = C.attribute_margin("soccer", market_margin=0.9, venue=AZTECA, trips={}, neutral=False, as_of="t")
    assert keys(unknown.factors)["altitude"] == pytest.approx(0.25 * 0.74)  # only the height above 1,500 m
    assert next(f for f in unknown.factors if f.key == "altitude").confidence == NOISY
    higher = C.attribute_margin("soccer", market_margin=0.2, venue=AZTECA,
                                trips={"away": {"home_elevation_m": 2660}}, neutral=False, as_of="t")
    assert "altitude" not in keys(higher.factors)
    assert "altitude" not in keys(C.attribute_margin("soccer", market_margin=0.2, venue=ANFIELD, trips={},
                                                     neutral=False, as_of="t").factors)


def test_back_to_backs_and_zero_rest() -> None:
    home_b2b = C.attribute_margin("nba", market_margin=-2.0, venue=FOOTPRINT,
                                  trips={"home": {"name": "Suns", "rest_days": 0}, "away": {"rest_days": 2}},
                                  neutral=False, as_of="t")
    rest = next(f for f in home_b2b.factors if f.key == "home_rest")
    assert rest.delta == -1.25 and rest.label == "Suns on a back-to-back" and rest.confidence == NORMAL
    away_b2b = C.attribute_margin("nba", market_margin=-2.0, venue=FOOTPRINT,
                                  trips={"away": {"rest_days": 0}}, neutral=False, as_of="t")
    assert keys(away_b2b.factors)["away_rest"] == 1.25
    assert next(f for f in away_b2b.factors if f.key == "away_rest").label == "Away team on a back-to-back"
    nhl = C.attribute_margin("nhl", market_margin=0.4, venue=None, trips={"away": {"back_to_back": True}},
                             neutral=False, as_of="t")
    goals = 2.3 * -N.inv_cdf(0.5 - 0.058)  # 5.8 points of win chance through a 2.3-goal SD
    assert keys(nhl.factors)["away_rest"] == pytest.approx(goals) and goals == pytest.approx(0.336, abs=0.001)
    both = C.attribute_margin("nhl", market_margin=0.2, venue=None,
                              trips={"home": {"rest_days": 0}, "away": {"rest_days": 0}}, neutral=False, as_of="t")
    assert keys(both.factors)["home_rest"] + keys(both.factors)["away_rest"] == pytest.approx(0)
    nfl = C.attribute_margin("nfl", market_margin=3, venue=HIGHMARK, trips={"home": {"rest_days": 0}},
                             neutral=False, as_of="t")
    assert set(keys(nfl.factors)) == {"home_field", "other"}


def test_eastward_jet_lag_is_flagged_untested() -> None:
    trips = {"home": {"name": "Yankees", "tz_shift_h": 3, "rest_days": 0}, "away": {"tz_shift_h": 3, "rest_days": 0}}
    mlb = C.attribute_margin("mlb", market_margin=0.4, venue=WRIGLEY, trips=trips, neutral=False, as_of="t")
    lag = next(f for f in mlb.factors if f.key == "home_tz")
    assert lag.delta == pytest.approx(4.3 * N.inv_cdf(0.5 - 0.035)) and lag.delta == pytest.approx(-0.378, abs=1e-3)
    assert lag.confidence == NOISY and "untested" in lag.evidence and lag.label == "Yankees: 3 h time-zone shift east"
    assert "away_tz" not in keys(mlb.factors)  # the studies found it for home teams
    nba = C.attribute_margin("nba", market_margin=4.0, venue=FOOTPRINT,
                             trips={"home": {"tz_shift_h": 2, "days_since_last": 3}}, neutral=False, as_of="t")
    assert keys(nba.factors)["home_tz"] == -1.3
    assert "untested" in next(f for f in nba.factors if f.key == "home_tz").evidence
    for trip in ({"tz_shift_h": 3, "rest_days": 3}, {"tz_shift_h": -3, "rest_days": 0},
                 {"tz_shift_h": 1, "rest_days": 0}):
        quiet = C.attribute_margin("nba", market_margin=4.0, venue=FOOTPRINT, trips={"home": trip}, neutral=False,
                                   as_of="t")
        assert "home_tz" not in keys(quiet.factors)
    assert "home_tz" not in keys(C.attribute_margin("nhl", market_margin=0.2, venue=None, trips=trips,
                                                    neutral=False, as_of="t").factors)


def test_attribute_margin_sums_to_the_market_and_names_its_parts() -> None:
    rates = {"home_win": 0.548, "draw": None, "games": 312, "raw": 0.561, "source": "ESPN standings, home records",
             "as_of": "2026-10-27T08:00:00Z"}
    trips = {"home": {"name": "Nuggets", "rest_days": 1, "tz_shift_h": 2, "days_since_last": 2},
             "away": {"name": "Lakers", "rest_days": 0, "tz_shift_h": 1}}
    projection = C.attribute_margin("basketball_nba", market_margin=7.5, home_edge=None, venue=BALL_ARENA,
                                    trips=trips, neutral=False, as_of="2026-10-27T14:05:00Z", home_rates=rates)
    assert projection.key == "margin" and projection.unit == "pts"
    assert projection.baseline_label == "Even game" and projection.baseline == 0.0 and projection.value == 7.5
    assert [f.key for f in projection.factors] == ["home_field", "altitude", "away_rest", "home_tz", "other"]
    edge = 12.0 * N.inv_cdf(0.548)
    assert keys(projection.factors) == pytest.approx({"home_field": edge, "altitude": 2.6, "away_rest": 1.25,
                                                      "home_tz": -1.3,
                                                      "other": 7.5 - edge - 2.6 - 1.25 + 1.3})
    home = projection.factors[0]
    assert home.label == "Home court" and home.source == "ESPN standings, home records"
    assert home.as_of == "2026-10-27T08:00:00Z" and "56.1% of 312 games" in home.evidence
    other = projection.factors[-1]
    assert other.label == "Team strength and other" and other.as_of == "2026-10-27T14:05:00Z"
    assert check_sum(projection, tol=1e-9)
    shown = projection.to_dict(1)
    assert round(shown["baseline"] + sum(f["delta"] for f in shown["factors"]), 6) == shown["value"]
    # A home edge passed in is used as given
    given = C.attribute_margin("nba", market_margin=7.5, home_edge=2.0, venue=None, trips=None, neutral=False,
                               as_of=None)
    assert keys(given.factors) == {"home_field": 2.0, "other": 5.5}
    # No rates and no edge: the sport's prior
    prior = C.attribute_margin("nfl", market_margin=-3.0, venue=HIGHMARK, trips=None, neutral=False, as_of=None)
    assert keys(prior.factors)["home_field"] == pytest.approx(12.7 * N.inv_cdf(0.539))
    assert prior.factors[0].source == C.SRC_PRIOR
    with pytest.raises(ValueError):
        C.attribute_margin("nba", market_margin=float("nan"), venue=None, trips=None, neutral=False, as_of=None)


def test_attribute_total_coors_park_and_weather_at_the_odds() -> None:
    at_odds = {"temp_c": 27.2, "wind_kmh": 26, "wind_dir": 184, "precip_prob": 10, "precip_mm": 0,
               "fetched_at": "2026-07-18T14:00:00Z"}
    projection = C.attribute_total("baseball_mlb", market_total=11.5, league_total=8.8, venue=COORS,
                                   conditions_at_odds=at_odds, as_of="2026-07-18T14:05:00Z")
    assert projection.key == "total" and projection.baseline_label == "League average total"
    assert projection.baseline == 8.8 and projection.value == 11.5 and projection.unit == "runs"
    parts = keys(projection.factors)
    assert parts["park"] == pytest.approx(0.13 * 8.8)  # (113 - 100) / 100 x the league average
    park = projection.factors[0]
    assert park.label == "Park: Coors Field" and "home-run index 105, shown, not added" in park.evidence
    assert park.source == "Baseball Savant, 3-year, 2023–25" and park.confidence == NORMAL
    temp_f = 27.2 * 9 / 5 + 32
    assert parts["temperature"] == pytest.approx(0.015 * (temp_f - 70))
    assert parts["wind"] == pytest.approx(0.08 * (26 / C.KMH_PER_MPH - 5))  # 184 -> blowing to 4: straight out
    assert parts["other"] == pytest.approx(11.5 - 8.8 - parts["park"] - parts["temperature"] - parts["wind"])
    assert [f.key for f in projection.factors][-1] == "other"
    assert projection.factors[-1].label == "Teams and other"
    assert check_sum(projection, tol=1e-9)
    # Without the forecast at the odds' time the remainder says the weather is in it
    unknown = C.attribute_total("mlb", market_total=11.5, league_total=8.8, venue=COORS, conditions_at_odds=None,
                                as_of=None)
    assert set(keys(unknown.factors)) == {"park", "other"}
    assert "forecast at the odds' time is unknown" in unknown.factors[-1].evidence
    dome = C.attribute_total("nfl", market_total=47.5, league_total=45.5, venue=SOFI, conditions_at_odds=None,
                             as_of=None)
    assert keys(dome.factors) == {"other": 2.0} and "unknown" not in dome.factors[-1].evidence
    # No league total given: the sport's fallback average
    assert C.attribute_total("nba", market_total=231.5, venue=BALL_ARENA, conditions_at_odds=None,
                             as_of=None).baseline == C.SPORTS["nba"].league_total


def test_attribute_total_nfl_wind_and_soccer_heat() -> None:
    windy = C.attribute_total("nfl", market_total=40.5, league_total=45.5, venue=HIGHMARK,
                              conditions_at_odds={"wind_kmh": mph(12), "temp_c": 5}, as_of="t")
    assert keys(windy.factors) == pytest.approx({"wind": -2.2, "other": 40.5 - 45.5 + 2.2})
    # Soccer: the weather's share of the market's own total (the total before weather is market / multiplier)
    hot = C.attribute_total("soccer_epl", market_total=2.9, league_total=2.8, venue=ANFIELD,
                            conditions_at_odds={"temp_c": 25}, as_of="t")
    assert keys(hot.factors)["heat"] == pytest.approx(2.9 - 2.9 / 1.068)
    assert check_sum(hot, tol=1e-9) and check_sum(windy, tol=1e-9)


def _random_venue(rng: random.Random, sport: str) -> dict:
    roof = rng.choice(["open", "open", "dome", "retractable"])
    venue = {"name": f"Venue {rng.randrange(1000)}", "roof": roof,
             "elevation_m": rng.choice([None, rng.uniform(0, 3600)]),
             "azimuth_deg": rng.choice([None, rng.uniform(0, 360)])}
    if roof == "retractable" and rng.random() < 0.3:
        venue["roof_state"] = rng.choice(["open", "closed"])
    if sport == "mlb" and rng.random() < 0.8:
        venue["park"] = {"runs": rng.uniform(80, 125), "hr": rng.uniform(70, 130), "source": "Savant"}
    return venue


def _random_conditions(rng: random.Random, sport: str):
    if rng.random() < 0.15:
        return None
    conditions = {"temp_c": rng.uniform(-20, 40), "wind_kmh": rng.uniform(0, 70), "wind_dir": rng.uniform(0, 360),
                  "gust_kmh": rng.uniform(0, 90), "precip_prob": rng.uniform(0, 100), "precip_mm": rng.uniform(0, 6),
                  "fetched_at": "2026-10-27T12:00:00Z"}
    if rng.random() < 0.2:
        del conditions[rng.choice(["temp_c", "wind_kmh", "wind_dir", "precip_mm"])]
    if sport == "mlb" and rng.random() < 0.4:
        text = rng.choice(["14 mph, Out To RF", "7 mph, In From CF", "6 mph, L To R", "0 mph, None", "Calm",
                           "22 mph, Out To CF", "11 mph, In From LF", "Varies"])
        conditions["announced"] = {"condition": rng.choice(["Sunny", "Roof Closed", "Cloudy"]),
                                   "temp_f": rng.uniform(40, 100), "wind_text": text}
    if rng.random() < 0.1:
        conditions["roof_closed"] = rng.random() < 0.5
    return conditions


def _random_trip(rng: random.Random) -> dict:
    trip = {"name": rng.choice(["Hawks", "Kings", None]), "rest_days": rng.choice([None, 0, 0, 1, 2, 3, 5, 9]),
            "tz_shift_h": rng.choice([None, -3, -2, -1, 0, 1, 2, 3, 5.5]), "travel_km": rng.uniform(0, 4500)}
    if rng.random() < 0.2:
        trip["back_to_back"] = rng.random() < 0.5
    if rng.random() < 0.3:
        trip["home_elevation_m"] = rng.uniform(0, 3000)
    return trip


@pytest.mark.parametrize("sport", sorted(C.SPORTS))
def test_attribution_sums_to_the_market_for_1000_random_games(sport: str) -> None:
    rng = random.Random(f"context-{sport}")
    sp = C.SPORTS[sport]
    for _ in range(1000):
        venue = rng.choice([None, _random_venue(rng, sport)])
        trips = rng.choice([None, {}, {"home": _random_trip(rng), "away": _random_trip(rng)},
                            {"away": _random_trip(rng)}])
        neutral = rng.random() < 0.1
        rates = None
        if rng.random() < 0.5:
            rates = {"home_win": rng.uniform(0.40, 0.62), "draw": rng.uniform(0.2, 0.3) if sport == "soccer" else None,
                     "games": rng.randrange(0, 900), "raw": rng.uniform(0.3, 0.7), "source": "ESPN standings",
                     "as_of": "2026-10-27"}
        edge = rng.choice([None, None, rng.uniform(-1, 1) * sp.margin_sd / 4])
        market_margin = rng.uniform(-1.5, 1.5) * sp.margin_sd
        margin = C.attribute_margin(sport, market_margin=market_margin, home_edge=edge, venue=venue, trips=trips,
                                    neutral=neutral, as_of="2026-10-27T14:05:00Z", home_rates=rates)
        market_total = sp.league_total * rng.uniform(0.6, 1.5)
        total = C.attribute_total(sport, market_total=market_total,
                                  league_total=rng.choice([None, sp.league_total * rng.uniform(0.9, 1.1)]),
                                  venue=venue, conditions_at_odds=_random_conditions(rng, sport),
                                  as_of="2026-10-27T14:05:00Z")
        for projection, number in ((margin, market_margin), (total, market_total)):
            assert projection.value == number
            assert check_sum(projection)  # to 0.01, as the contract says
            assert abs(projection.residual()) < 1e-9
            assert projection.factors[-1].key == "other"
            assert sum(f.key == "other" for f in projection.factors) == 1
            for f in projection.factors:
                assert f.unit == sp.unit and f.confidence in CONFIDENCES and f.source and f.evidence
            for digits in (1, 2):
                shown = projection.to_dict(digits)
                assert round(shown["baseline"] + sum(f["delta"] for f in shown["factors"]), 6) == shown["value"]
            folded = fold(projection, 3, 1)
            assert round(folded["baseline"] + sum(f["delta"] for f in folded["factors"]), 6) == folded["value"]


# --- adjustment --------------------------------------------------------------------------------------------------

def test_no_adjustment_without_a_forecast_at_the_odds_or_a_change() -> None:
    assert C.adjustment("nfl", "total", -3.8, None) is None
    assert C.adjustment("nfl", "total", -2.2, -2.2) is None
    assert C.adjustment("mlb", "total", 0.301, 0.3) is None  # rounds to no change
    assert C.weather_adjustment("nfl", HIGHMARK, {"wind_mph": 18}, None) is None
    assert C.weather_adjustment("nfl", HIGHMARK, {"wind_mph": 18}, {"wind_mph": 18}) is None
    assert C.weather_adjustment("nfl", SOFI, {"wind_mph": 30}, {"wind_mph": 5}) is None  # a dome: no effect either way


def test_the_specs_worked_adjustment_wind_8_to_18_mph() -> None:
    adj = C.weather_adjustment("americanfootball_nfl", HIGHMARK, {"wind_mph": 18, "temp_f": 50},
                               {"wind_mph": 8, "temp_f": 50}, odds_at="2026-10-27T14:05:00Z")
    assert adj is not None
    assert (adj.kind, adj.raw, adj.applied, adj.cap, adj.weight, adj.capped, adj.full) == \
        ("total", pytest.approx(-3.8), pytest.approx(-1.9), 2.0, 0.5, False, False)
    assert adj.reason == "The forecast changed after the odds were fetched at 14:05 UTC: wind 8 → 18 mph."
    assert adj.unit == "pts" and adj.effect_at_odds == 0.0 and adj.effect_now == pytest.approx(-3.8)
    local = C.weather_adjustment("nfl", HIGHMARK, {"wind_mph": 18}, {"wind_mph": 8}, odds_at="2026-10-27T14:05:00Z",
                                 tz="America/New_York")
    assert local.reason == "The forecast changed after the odds were fetched at 10:05 EDT: wind 8 → 18 mph."
    data = adj.to_dict()
    assert set(data) == {"kind", "raw", "applied", "cap", "weight", "capped", "full", "reason", "unit", "sport",
                         "effect_now", "effect_at_odds"}
    assert data["applied"] == -1.9 and data["sport"] == "nfl"


def test_adjustment_weight_clamp_and_sign() -> None:
    half = C.adjustment("nfl", "total", -1.0, 0.0)
    assert half.applied == pytest.approx(-0.5) and not half.capped and half.weight == 0.5
    capped = C.adjustment("nfl", "total", -6.4, 0.0)  # 0.5 x -6.4 = -3.2, capped at -2.0
    assert capped.applied == -2.0 and capped.capped and capped.raw == pytest.approx(-6.4)
    assert C.adjustment("nfl", "side", 3.0, 0.0).applied == 1.0
    assert C.adjustment("mlb", "total", 1.0, 0.0).applied == 0.4
    assert C.adjustment("nba", "total", 4.0, 0.0).applied == 1.5
    assert C.adjustment("nhl", "total", -0.5, 0.0).applied == -0.15
    assert C.adjustment("soccer", "total", 0.8, 0.0).applied == 0.30
    assert C.adjustment("soccer", "side", 0.8, 0.0).applied == 0.15
    win = C.adjustment("nba", "win", 8.0, 0.0)
    assert win.applied == 2.5 and win.unit == "pp" and win.capped
    # A change of sign: wind out at the odds' time (+0.6 runs), in now (-0.4): delta -1.0, half is -0.5, cap 0.4
    flipped = C.adjustment("mlb", "total", -0.4, 0.6)
    assert flipped.raw == pytest.approx(-1.0) and flipped.applied == -0.4 and flipped.capped
    small_flip = C.adjustment("mlb", "total", -0.2, 0.3)
    assert small_flip.applied == pytest.approx(-0.25) and not small_flip.capped
    assert C.adjustment("mlb", "total", 0.2, -0.1).applied == pytest.approx(0.15)
    assert C.adjustment("nfl", "total", -1.0, 0.0, weight=0.8).applied == pytest.approx(-0.8)
    with pytest.raises(ValueError):
        C.adjustment("nfl", "spread", -1.0, 0.0)
    with pytest.raises(ValueError):
        C.adjustment("nfl", "total", -1.0, 0.0, weight=1.5)


def test_adjustment_applies_the_full_raw_change_without_a_market() -> None:
    full = C.adjustment("nfl", "total", -3.8, 0.0, has_market=False, odds_at="2026-10-27T14:05:00Z",
                        change="wind 8 → 18 mph")
    assert (full.applied, full.raw, full.weight, full.cap, full.capped, full.full) == \
        (pytest.approx(-3.8), pytest.approx(-3.8), 1.0, None, False, True)
    assert full.reason == ("The forecast changed after the odds were fetched at 14:05 UTC: wind 8 → 18 mph. There "
                           "is no market line to anchor to, so the full change applies.")
    assert C.adjustment("nfl", "total", -6.4, 0.0, has_market=False).applied == pytest.approx(-6.4)  # no cap
    assert C.adjustment("nfl", "total", -1.0, -1.0, has_market=False) is None


def test_a_roof_announced_closed_after_the_odds_removes_the_weather() -> None:
    # Hot, blowing straight out at Daikin Park, with the roof announced open when the odds were fetched
    at_odds = {"temp_f": 95, "wind_mph": 20, "wind_dir": 343 + 180, "roof_closed": False}
    effect = C.weather_total_effect("mlb", DAIKIN, at_odds)
    assert effect == pytest.approx(0.375 + 1.0)
    closed = {**at_odds, "roof_closed": None,
              "announced": {"condition": "Roof Closed", "temp_f": 73, "wind_text": "0 mph, None"}}
    adj = C.weather_adjustment("mlb", DAIKIN, closed, at_odds, odds_at="2026-07-18T16:00:00Z")
    assert adj.raw == pytest.approx(0 - effect) and adj.applied == -0.4 and adj.capped
    assert adj.reason == "The roof was announced closed after the odds were fetched at 16:00 UTC."
    # And the other way: closed at the odds' time, announced open since
    reopened = C.weather_adjustment("mlb", DAIKIN, {**at_odds, "roof_closed": False}, {**at_odds, "roof_closed": True})
    assert reopened.raw == pytest.approx(effect) and reopened.reason.startswith("The roof was announced open")


def test_a_retractable_roof_with_no_announced_state_moves_nothing() -> None:
    """Nothing announced at the odds' time: what the market assumed about the roof isn't known, so neither MLB's
    routine "Roof Closed" since nor a forecast change moves a number, and the line holds no weather factor."""
    chase = {"id": "chase-field", "name": "Chase Field", "elevation_m": 331, "roof": "retractable", "azimuth_deg": 0,
             "park": {"runs": 103, "hr": 99, "source": "Savant"}}
    july = {"temp_c": 40.5, "wind_kmh": 12, "wind_dir": 200}  # 105°F in Phoenix, no announcement yet
    assert C.weather_factors("mlb", chase, july) == [] and C.badges("mlb", chase, july, None) == []
    total = C.attribute_total("mlb", market_total=9.5, league_total=8.8, venue=chase, conditions_at_odds=july, as_of="t")
    assert {f.key for f in total.factors} == {"park", "other"} and "wasn't announced" in total.factors[-1].evidence
    closed = {**july, "announced": {"condition": "Roof Closed", "temp_f": 76, "wind_text": "0 mph, None"}}
    assert C.weather_adjustment("mlb", chase, closed, july, odds_at="2026-07-18T16:00:00Z") is None
    assert C.badges("mlb", chase, closed, None) == ["roof"]  # the roof chip, but no "adj"
    # AT&T Stadium (NFL roofs are never announced): a forecast turning windy and freezing moves nothing
    att = {"id": "att-stadium", "name": "AT&T Stadium", "elevation_m": 180, "roof": "retractable"}
    calm, storm = {"wind_kmh": 14, "temp_c": 8}, {"wind_kmh": 32, "temp_c": -2}
    assert C.weather_adjustment("nfl", att, storm, calm) is None and C.badges("nfl", att, storm, None) == []
    assert C.weather_adjustment("nfl", HIGHMARK, storm, calm).applied == -2.0  # open air: it does


def test_weather_effects_share_one_base_for_soccer() -> None:
    now, then = C.weather_effects("soccer", ANFIELD, {"temp_c": 25, "precip_mm": 1}, {"temp_c": 25},
                                  market_total=2.9)
    base = 2.9 / 1.068
    assert then == pytest.approx(2.9 - base) and now == pytest.approx(base * (1.068 * 1.014 - 1))
    assert C.weather_effects("soccer", ANFIELD, {"temp_c": 25}, None, market_total=2.9)[1] is None
    adj = C.weather_adjustment("soccer", ANFIELD, {"temp_c": 25, "precip_mm": 1}, {"temp_c": 25}, market_total=2.9)
    assert adj.applied == pytest.approx(0.5 * (now - then)) and adj.reason.endswith("rain now expected.")


# --- the adjusted chance of the over -----------------------------------------------------------------------------

def test_adjusted_over_matches_the_normal_formula() -> None:
    for sport, line, p, applied in (("nfl", 44.5, 0.52, -1.9), ("mlb", 8.5, 0.47, 0.4), ("nba", 228.5, 0.5, 1.5),
                                    ("nhl", 6.5, 0.44, -0.15), ("ncaaf", 55.5, 0.61, 2.0)):
        sd = C.SPORTS[sport].total_sd
        mu = line + sd * N.inv_cdf(p)
        expected = 1 - N.cdf((line - (mu + applied)) / sd)
        assert C.adjusted_over(sport, line, p, applied) == pytest.approx(expected, abs=1e-12)
    assert C.adjusted_over("nfl", 44.5, 0.5, -1.9) == pytest.approx(N.cdf(-1.9 / 13.2))
    assert C.adjusted_over("nfl", 44.5, 0.53, 0) == 0.53  # nothing applied: unchanged
    with pytest.raises(ValueError):
        C.adjusted_over("nfl", 44.5, 1.3, 1.0)


@pytest.mark.parametrize("sport, line", [("nfl", 44.5), ("mlb", 8.5), ("nhl", 6.5), ("soccer", 2.5),
                                         ("soccer", 3.5)])
def test_adjusted_over_is_monotone(sport: str, line: float) -> None:
    cap = C.SPORTS[sport].caps["total"]
    steps = [cap * k / 10 for k in range(-20, 21)]
    for p in (0.2, 0.5, 0.8):
        chances = [C.adjusted_over(sport, line, p, a) for a in steps]
        assert all(b > a for a, b in zip(chances, chances[1:]))
        assert all(0 < c < 1 for c in chances)
        assert C.adjusted_over(sport, line, p, 0) == p


def test_soccer_adjusted_over_scales_the_goals_models_rates() -> None:
    rates = (1.5, 1.2)
    p = soccer.over_prob(soccer.grid(*rates), 2.5)
    moved = C.adjusted_over("soccer", 2.5, p, 0.3, rates=rates)
    scale = 3.0 / 2.7
    assert moved == pytest.approx(soccer.over_prob(soccer.grid(1.5 * scale, 1.2 * scale), 2.5), abs=1e-9)
    assert moved > p
    # Without rates, the goals model's own home/away split
    p_default = soccer.over_prob(soccer.grid(1.557, 1.245), 2.5)
    assert C.adjusted_over("soccer", 2.5, p_default, -0.3) == pytest.approx(
        soccer.over_prob(soccer.grid(1.557 * 2.502 / 2.802, 1.245 * 2.502 / 2.802), 2.5), abs=1e-9)
    # It is not the normal formula
    assert moved != pytest.approx(1 - N.cdf((2.5 - (2.5 + 1.7 * N.inv_cdf(p) + 0.3)) / 1.7), abs=1e-4)


# --- badges ------------------------------------------------------------------------------------------------------

def test_badges_the_specs_example() -> None:
    conditions = {"temp_c": 27, "wind_kmh": 26.2, "wind_dir": 184, "precip_prob": 10, "precip_mm": 0}
    assert C.badges("baseball_mlb", COORS, conditions, None) == ["wind:26", "alt:1580"]


def test_badges_codes_and_thresholds() -> None:
    assert C.badges("nfl", HIGHMARK, {"wind_kmh": 23.9, "temp_c": 1}, None) == []
    assert C.badges("nfl", HIGHMARK, {"wind_kmh": 24, "temp_c": 0}, None) == ["wind:24", "cold:0"]
    assert C.badges("nfl", HIGHMARK, {"temp_c": -4.6}, None) == ["cold:-5"]
    assert C.badges("soccer", ANFIELD, {"temp_c": 31.2}, None) == ["hot:31"]
    assert C.badges("soccer", ANFIELD, {"temp_c": 15, "precip_prob": 60, "precip_mm": 0.5}, None) == ["rain"]
    assert C.badges("soccer", ANFIELD, {"temp_c": 15, "precip_prob": 59, "precip_mm": 3}, None) == []
    # MLB's announced wind counts for the badge (16 mph = 26 km/h)
    assert C.badges("mlb", WRIGLEY, {"announced": {"wind_text": "16 mph, Out To RF", "temp_f": 75}}, None) == \
        ["wind:26"]
    # Indoors: no weather badges, but altitude stays
    storm = {"temp_c": -12, "wind_kmh": 50, "precip_prob": 90, "precip_mm": 4}
    assert C.badges("nba", BALL_ARENA, storm, None) == ["alt:1580"]
    assert C.badges("nfl", SOFI, storm, None) == []
    # A retractable roof announced closed
    closed = {**storm, "announced": {"condition": "Roof Closed"}}
    assert C.badges("mlb", DAIKIN, closed, None) == ["roof"]
    assert C.badges("nfl", LUCAS_OIL, {**storm, "roof_closed": True}, None) == ["roof"]
    # Travel: 2 h+ east within 3 days; NFL short weeks (4 days or fewer since the last game)
    trips = {"home": {"tz_shift_h": 3, "rest_days": 1}, "away": {"tz_shift_h": 2.5, "rest_days": 2}}
    assert C.badges("nba", FOOTPRINT, None, trips) == ["tz-home:3", "tz-away:2.5"]
    thursday = {"home": {"rest_days": 3}, "away": {"days_since_last": 4}}
    assert C.badges("nfl", SOFI, None, thursday) == ["rest-home:4", "rest-away:4"]
    assert C.badges("nfl", SOFI, None, {"home": {"rest_days": 6}}) == []
    assert C.badges("nba", FOOTPRINT, None, thursday) == []
    assert C.badges("tennis_atp", {"elevation_m": 2240}, None, None) == ["alt:2240"]


def test_badges_put_the_adjustment_first_and_keep_three() -> None:
    adj = C.adjustment("nfl", "total", -3.8, 0.0)
    conditions = {"wind_kmh": 30, "temp_c": -5, "precip_prob": 80, "precip_mm": 2}
    trips = {"home": {"rest_days": 3}, "away": {"tz_shift_h": 3, "rest_days": 2}}
    assert C.badges("nfl", EMPOWER, conditions, trips, adj) == ["adj", "wind:30", "rain"]
    assert C.badges("nfl", EMPOWER, conditions, trips) == ["wind:30", "rain", "cold:-5"]
    assert C.badges("nfl", EMPOWER, conditions, trips, limit=10) == ["wind:30", "rain", "cold:-5", "alt:1580",
                                                                    "tz-away:3", "rest-home:4", "rest-away:3"]
    assert C.badges("nfl", EMPOWER, None, None, adj.to_dict()) == ["adj", "alt:1580"]
    assert C.badges("nfl", EMPOWER, None, None, [adj]) == ["adj", "alt:1580"]
    assert C.badges("nfl", EMPOWER, None, None, []) == ["alt:1580"]
    assert C.badges("nfl", EMPOWER, None, None, None) == ["alt:1580"]


# --- factors.fold ------------------------------------------------------------------------------------------------

def test_fold_keeps_the_top_factors_and_still_sums() -> None:
    trips = {"home": {"rest_days": 0, "tz_shift_h": 3}, "away": {"rest_days": 0}}
    projection = C.attribute_margin("nba", market_margin=5.0, venue=BALL_ARENA, trips=trips, neutral=False,
                                    as_of="t")
    assert len(projection.factors) == 6
    folded = fold(projection, 3, 1)
    labels = [f["label"] for f in folded["factors"]]
    assert len(labels) == 4 and labels[-1] == "Other (3 factors)" and folded["factors"][-1]["folded"] == 3
    sizes = [abs(f["delta"]) for f in folded["factors"][:3]]
    assert sizes == sorted(sizes, reverse=True)
    assert round(folded["baseline"] + sum(f["delta"] for f in folded["factors"]), 6) == folded["value"] == 5.0
    # One factor left over shows as itself
    assert [f["key"] for f in fold(projection, 5)["factors"]] == [factor.key for factor in sorted(
        projection.factors, key=lambda factor: -abs(factor.delta))]
    assert len(fold(projection, 6)["factors"]) == 6
