"""data/weather.py: Open-Meteo's forecast and geocoder through the shared cache, batched, capped and attributed.
Nothing here touches the network: a fake session answers in Open-Meteo's shape (built from a captured answer)."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import pytest

from telegram_betting_bot_simple.data import weather as weather_module
from telegram_betting_bot_simple.data.cache import Policy, TTLCache, ttl_for
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.data.weather import (ATTRIBUTION, ATTRIBUTION_URL, FIELDS, HOURLY, DailyCounter,
                                                        WeatherClient, at, hours_of, parse_forecast)

FIXTURES = Path(__file__).parent / "fixtures" / "context"
UTC = datetime.timezone.utc
NOW = datetime.datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
CAPTURED = json.loads((FIXTURES / "openmeteo_forecast.json").read_text(encoding="utf-8"))


class Clock:
    def __init__(self, start: datetime.datetime = NOW) -> None:
        self.now = start.timestamp()

    def __call__(self) -> float:
        return self.now

    def advance(self, minutes: float) -> None:
        self.now += minutes * 60


class Response:
    def __init__(self, status: int, body) -> None:
        self.status_code, self.ok, self._body = status, 200 <= status < 300, body

    def json(self):
        return self._body


class OpenMeteo:
    """A fake requests session answering Open-Meteo's forecast (one location per latitude asked for, hours from
    today's midnight GMT, the captured answer's values repeated) and geocoder."""

    def __init__(self, clock: Clock, status: int = 200) -> None:
        self.clock, self.status, self.calls = clock, status, []
        self.places = {"Denver": [
            {"name": "Denver", "latitude": 39.73915, "longitude": -104.9847, "elevation": 1609.0, "country_code": "US",
             "admin1": "Colorado", "timezone": "America/Denver", "population": 715522},
            {"name": "Denver", "latitude": 40.2, "longitude": -76.1, "elevation": 120.0, "country_code": "US",
             "admin1": "Pennsylvania", "timezone": "America/New_York", "population": 1800}],
            "Bogota": [{"name": "Bogotá", "latitude": 4.60971, "longitude": -74.08175, "elevation": 2582.0,
                        "country_code": "CO", "admin1": "Bogota D.C.", "timezone": "America/Bogota"}],
            "Sandy": [{"name": "Sandy", "latitude": 45.39734, "longitude": -122.26148, "country_code": "US",
                       "admin1": "Oregon"},
                      {"name": "Sandy", "latitude": 40.59161, "longitude": -111.88411, "country_code": "US",
                       "admin1": "Utah", "elevation": 1369.0},
                      {"name": "Sandy", "latitude": 52.12, "longitude": -0.29, "country_code": "GB",
                       "admin1": "England"}]}

    def get(self, url, params=None, timeout=None, headers=None):
        params = dict(params or {})
        self.calls.append((url, params))
        if self.status != 200:
            return Response(self.status, {"error": True, "reason": "Too many requests"})
        if "geocoding" in url:
            return Response(200, {"results": self.places.get(params["name"], [])} if params["name"] in self.places
                            else {"generationtime_ms": 0.1})
        lats = str(params["latitude"]).split(",")
        lons = str(params["longitude"]).split(",")
        today = datetime.datetime.fromtimestamp(self.clock(), UTC).date()
        past = int(params.get("past_days") or 0)
        first = datetime.datetime.combine(today - datetime.timedelta(days=past), datetime.time(), UTC)
        count = (int(params["forecast_days"]) + past) * 24
        sample = CAPTURED["response"][0]["hourly"]
        times = [(first + datetime.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(count)]
        columns = {variable: [sample[variable][h % len(sample[variable])] for h in range(count)]
                   for variable in HOURLY}
        items = []
        for index, (lat, lon) in enumerate(zip(lats, lons)):
            hourly = {"time": times, **columns}
            hourly["temperature_2m"] = [round(10 + 0.1 * h + index, 1) for h in range(count)]
            items.append({"latitude": float(lat), "longitude": float(lon), "elevation": 100.0 + index,
                          "timezone": "GMT", "hourly": hourly})
        return Response(200, items if len(items) > 1 else items[0])

    def forecast_calls(self):
        return [params for url, params in self.calls if url == weather_module.OPEN_METEO]


def client(clock=None, session=None, store=None, cache=None, **options):
    clock = clock or Clock()
    session = session or OpenMeteo(clock)
    return WeatherClient(cache or TTLCache(clock=clock), store=store, session=session, **options), session, clock


def hours_after(hours: float) -> datetime.datetime:
    return NOW + datetime.timedelta(hours=hours)


# --- the request and the hours ---------------------------------------------------------------------------------------

def test_a_forecast_is_one_call_with_six_variables_and_its_hours():
    weather, session, _ = client()
    found = weather.forecast([("coors-field", 39.756042, -104.994136)], hours_after(2), hours_after(6))
    (params,) = session.forecast_calls()
    assert params == {"latitude": "39.76", "longitude": "-104.99", "hourly": ",".join(HOURLY), "timezone": "GMT",
                      "wind_speed_unit": "kmh", "forecast_days": 3}
    assert len(HOURLY) == 6 and len(FIELDS) == 6
    hours = found["coors-field"]
    assert [hour["t"] for hour in hours] == ["2026-10-27T14:00Z", "2026-10-27T15:00Z", "2026-10-27T16:00Z",
                                             "2026-10-27T17:00Z", "2026-10-27T18:00Z"]
    assert set(hours[0]) == {"t", "temp_c", "precip_prob", "precip_mm", "wind_kmh", "wind_dir", "gust_kmh"}
    assert hours[0]["temp_c"] == 11.4  # hour 14 of the fake's day
    detail = weather.forecasts([("coors-field", 39.756042, -104.994136)], hours_after(2), hours_after(6))
    assert detail["coors-field"]["fetched_at"] == NOW.timestamp() and detail["coors-field"]["stale"] is False
    assert detail["coors-field"]["elevation"] == 100.0 and len(session.forecast_calls()) == 1  # from the cache


def test_forecast_days_cover_the_window_up_to_seven():
    weather, session, _ = client()
    weather.forecast([("a", 10, 10)], hours_after(60), hours_after(70))  # ends on 30 October: day 4 (today is 1)
    weather.forecast([("b", 11, 11)], hours_after(140), hours_after(200))  # past day 7: 7
    assert [params["forecast_days"] for params in session.forecast_calls()] == [4, 7]
    assert weather.forecast([("c", 12, 12)], hours_after(24 * 7 + 1), hours_after(24 * 7 + 4)) == {}  # too far
    assert len(session.forecast_calls()) == 2


def test_a_window_starting_yesterday_asks_for_the_past_day():
    weather, session, _ = client(Clock(datetime.datetime(2026, 10, 27, 0, 30, tzinfo=UTC)))
    start = datetime.datetime(2026, 10, 26, 23, 0, tzinfo=UTC)
    hours = weather.forecast([("late", 40.0, -74.0)], start, start + datetime.timedelta(hours=4))["late"]
    assert session.forecast_calls()[0]["past_days"] == 1
    assert hours[0]["t"] == "2026-10-26T23:00Z" and len(hours) == 5


def test_one_call_per_fifty_points_and_one_location_per_spot():
    weather, session, _ = client()
    points = [(f"v{i}", 30 + i * 0.1, -90 - i * 0.1) for i in range(120)]
    points.append(("v0-again", 30.001, -90.001))  # rounds to v0's spot: asked for once
    found = weather.forecast(points, hours_after(1), hours_after(4))
    calls = session.forecast_calls()
    assert [len(params["latitude"].split(",")) for params in calls] == [50, 50, 20]
    assert len(found) == 121 and found["v0-again"] == found["v0"]
    assert weather.calls_today()["calls"] == 3 and weather.calls_today()["units"] == 120


def test_bad_points_are_skipped():
    weather, session, _ = client()
    assert weather.forecast([("x", None, 10), ("y", 95, 10), ("z", 10, 200)], hours_after(1), hours_after(2)) == {}
    assert session.calls == []


def test_off_means_no_calls():
    weather, session, _ = client(enabled=False)
    assert weather.forecast([("a", 10, 10)], hours_after(1), hours_after(2)) == {}
    assert weather.geocode("Denver") is None and session.calls == []


# --- the cache: an hour, 15 minutes when near ----------------------------------------------------------------------

def test_a_forecast_is_kept_an_hour_or_15_minutes_when_the_game_is_near():
    weather, session, clock = client()
    far = [("far", 41.0, -87.0, hours_after(24))]
    near = [("near", 42.0, -71.0, hours_after(3))]
    weather.forecast(far, hours_after(1), hours_after(30))
    weather.forecast(near, hours_after(1), hours_after(30))
    assert len(session.forecast_calls()) == 2
    clock.advance(16)
    weather.forecast(far, hours_after(1), hours_after(30))
    assert len(session.forecast_calls()) == 2  # the far game's forecast is good for an hour
    weather.forecast(near, hours_after(1), hours_after(30))
    assert len(session.forecast_calls()) == 3  # the near one's for 15 minutes
    clock.advance(45)
    weather.forecast(far, hours_after(1), hours_after(30))
    assert len(session.forecast_calls()) == 4
    assert ttl_for("weather") == Policy(3600, 3600) and ttl_for("weather", "near") == Policy(900, 900)


def test_a_failed_refresh_serves_the_last_forecast_marked_stale_and_pauses():
    weather, session, clock = client()
    points = [("v", 39.0, -105.0, hours_after(20))]
    first = weather.forecasts(points, hours_after(1), hours_after(5))["v"]
    clock.advance(61)
    session.status = 429
    second = weather.forecasts(points, hours_after(1), hours_after(5))["v"]
    assert second["stale"] is True and second["fetched_at"] == first["fetched_at"]
    assert second["hours"] == first["hours"] and len(session.forecast_calls()) == 2
    weather.forecasts(points, hours_after(1), hours_after(5))
    assert len(session.forecast_calls()) == 2  # paused: no call for 5 minutes
    clock.advance(6)
    weather.forecasts(points, hours_after(1), hours_after(5))
    assert len(session.forecast_calls()) == 3  # tried again, failed again: the pause doubles
    clock.advance(6)
    weather.forecasts(points, hours_after(1), hours_after(5))
    assert len(session.forecast_calls()) == 3
    clock.advance(60)  # past the copy's stale window as well: nothing to serve while Open-Meteo fails
    assert weather.forecasts(points, hours_after(1), hours_after(5)) == {}
    session.status = 200
    clock.advance(30)
    assert weather.forecasts(points, hours_after(1), hours_after(5))["v"]["stale"] is False


def test_another_process_reads_the_forecast_from_the_shared_table(tmp_path):
    store = Store(str(tmp_path / "betting.db"))
    clock = Clock()
    mine, session, _ = client(clock, store=store, cache=TTLCache(store=store, clock=clock))
    mine.forecast([("coors-field", 39.756, -104.994)], hours_after(1), hours_after(4))
    theirs = OpenMeteo(clock, status=500)
    other = WeatherClient(TTLCache(store=store, clock=clock), store=store, session=theirs)
    found = other.forecast([("coors-field", 39.756, -104.994)], hours_after(1), hours_after(4))
    assert found["coors-field"] and theirs.calls == []
    with store._connect() as db:
        keys = [row[0] for row in db.execute("SELECT key FROM shared_cache WHERE kind = 'weather'")]
    assert keys == ['["weather",39.76,-104.99,"2026-10-27","2026-10-29"]']  # the spot and the date range only


# --- the daily cap ---------------------------------------------------------------------------------------------------

def test_calls_past_the_daily_cap_are_refused_with_a_warning(caplog):
    weather, session, _ = client(daily_cap=2)
    caplog.set_level(logging.WARNING, logger="telegram_betting_bot_simple.data.weather")
    for index in range(3):
        weather.forecast([(f"v{index}", 30 + index, -90)], hours_after(1), hours_after(3))
    assert len(session.forecast_calls()) == 2
    assert "daily cap" in caplog.text
    assert weather.calls_today()["calls"] == 2


def test_the_weighted_count_has_its_own_cap():
    weather, session, _ = client(daily_units=60)
    weather.forecast([(f"a{i}", 30 + i, -90) for i in range(50)], hours_after(1), hours_after(3))
    weather.forecast([(f"b{i}", 30 + i, -80) for i in range(20)], hours_after(1), hours_after(3))
    assert len(session.forecast_calls()) == 1  # 50 + 20 locations would pass 60


def test_the_count_is_shared_through_the_store_and_resets_each_utc_day(tmp_path):
    store = Store(str(tmp_path / "betting.db"))
    clock = Clock()
    first, one, _ = client(clock, store=store, daily_cap=3)
    second, two, _ = client(clock, store=store, daily_cap=3)
    first.forecast([("a", 30, -90)], hours_after(1), hours_after(3))
    second.forecast([("b", 31, -90)], hours_after(1), hours_after(3))
    first.forecast([("c", 32, -90)], hours_after(1), hours_after(3))
    second.forecast([("d", 33, -90)], hours_after(1), hours_after(3))
    assert len(one.forecast_calls()) + len(two.forecast_calls()) == 3
    assert first.calls_today()["calls"] == second.calls_today()["calls"] == 3
    clock.advance(13 * 60)  # the next UTC day
    second.forecast([("d", 33, -90)], hours_after(1), hours_after(3))
    assert len(two.forecast_calls()) == 2 and second.calls_today()["calls"] == 1


def test_the_counter_counts_in_memory_without_a_store():
    counter = DailyCounter("x", None, lambda: NOW.timestamp())
    assert counter.take(1, 5.0, cap_calls=2) and counter.take(1, 5.0, cap_calls=2)
    assert not counter.take(1, 5.0, cap_calls=2)
    assert counter.count() == (2, 10.0) and counter.day() == "2026-10-27"


# --- at(), the geocoder, attribution --------------------------------------------------------------------------------

def test_at_finds_the_hour_nearest_the_start():
    hours = [{"t": "2026-10-27T18:00Z", "temp_c": 1}, {"t": "2026-10-27T19:00Z", "temp_c": 2},
             {"t": "2026-10-27T20:00Z", "temp_c": 3}]
    assert at(hours, "2026-10-27T19:20:00Z")["temp_c"] == 2
    assert at(hours, datetime.datetime(2026, 10, 27, 19, 40, tzinfo=UTC))["temp_c"] == 3
    assert at(hours, datetime.datetime(2026, 10, 27, 17, 40, tzinfo=UTC).timestamp())["temp_c"] == 1
    assert at(hours, "2026-10-27T23:00Z") is None  # nothing within 90 minutes
    assert at([], NOW) is None and at(hours, None) is None
    assert WeatherClient.at(hours, "2026-10-27T18:00Z")["temp_c"] == 1


def test_geocode_picks_the_country_and_is_kept():
    weather, session, clock = client()
    denver = weather.geocode("Denver", "US")
    assert denver == {"name": "Denver", "lat": 39.73915, "lon": -104.9847, "elevation": 1609.0, "country": "US",
                      "admin1": "Colorado", "timezone": "America/Denver", "population": 715522}
    clock.advance(60 * 24 * 29)
    assert weather.geocode("  Denver ", "us") == denver and len(session.calls) == 1  # kept 30 days
    assert weather.geocode("Bogota", "US") is None  # not in that country
    assert weather.geocode("Bogota")["elevation"] == 2582.0
    assert weather.geocode("Atlantis") is None
    assert session.calls[0][1] == {"name": "Denver", "count": 10, "language": "en", "format": "json"}
    assert ttl_for("geocode") == Policy(30 * 86400, 0)


def test_geocode_reads_espns_city_and_country_spellings():
    weather, session, _ = client()
    assert weather.geocode("Sandy, Utah", "USA")["admin1"] == "Utah"
    assert session.calls[-1][1]["name"] == "Sandy"
    assert weather.geocode("Sandy", "England")["country"] == "GB"
    assert weather.geocode("Sandy, Oregon")["admin1"] == "Oregon"
    assert weather_module.country_code("USA") == "US" and weather_module.country_code("England") == "GB"
    assert weather_module.country_code("mx") == "MX" and weather_module.country_code("Narnia") is None
    assert weather_module.country_code(None) is None


def test_geocode_failure_is_none():
    weather, session, _ = client(session=None)
    session.status = 500
    assert weather.geocode("Denver", "US") is None


def test_attribution():
    assert ATTRIBUTION == "Weather data by Open-Meteo.com" and ATTRIBUTION_URL == "https://open-meteo.com/"
    assert WeatherClient.attribution() == {"text": ATTRIBUTION, "url": ATTRIBUTION_URL}


def test_the_captured_batched_answer_parses():
    """A real three-location answer (api.open-meteo.com, 27 October 2026): one copy per location, in order."""
    copies = [parse_forecast(item) for item in CAPTURED["response"]]
    assert len(copies) == 3 and [copy["elevation"] for copy in copies] == [
        item["elevation"] for item in CAPTURED["response"]]
    hours = hours_of(copies[0])
    first = CAPTURED["response"][0]["hourly"]
    assert len(hours) == 24 and hours[0]["t"] == first["time"][0] + "Z"
    assert [hour["wind_kmh"] for hour in hours] == first["wind_speed_10m"]
    assert [hour["wind_dir"] for hour in hours] == first["wind_direction_10m"]
    assert [hour["precip_prob"] for hour in hours] == first["precipitation_probability"]
    with pytest.raises(ValueError):
        parse_forecast({"elevation": 1.0})


def test_a_simulated_day_stays_well_under_the_cap():
    """60 open-air venues, each with a game a day (starts spread over the day), warmed every 5 minutes for 24 hours:
    at most one call a cycle (288), far under 1,000 calls; the weighted count stays far under Open-Meteo's 10,000."""
    weather, session, clock = client()
    venues = [(f"v{i}", 25 + i * 0.4, -120 + i * 0.7) for i in range(60)]
    first_games = {pid: hours_after(1 + i * 0.4) for i, (pid, _, _) in enumerate(venues)}
    for _ in range(24 * 12):
        now = datetime.datetime.fromtimestamp(clock(), UTC)
        points = []
        for pid, lat, lon in venues:
            game = first_games[pid]
            while game + datetime.timedelta(hours=3) < now:  # the next day's game once this one is over
                game += datetime.timedelta(days=1)
            if game - now <= datetime.timedelta(hours=48):
                points.append((pid, lat, lon, game))
        found = weather.forecast(points, now - datetime.timedelta(hours=1), now + datetime.timedelta(hours=3))
        assert set(found) == {point[0] for point in points}
        clock.advance(5)
    calls = session.forecast_calls()
    assert len(calls) <= 24 * 12  # at most one a cycle
    units = sum(len(params["latitude"].split(",")) for params in calls)  # Open-Meteo counts each location
    assert 60 * 24 <= units < 4000  # every venue hourly, those with a game within 6 hours every 15 minutes


def test_a_finished_game_is_not_near_but_one_on_is():
    weather, session, clock = client()
    over = [("over", 41.0, -87.0, hours_after(-5))]
    playing = [("on", 42.0, -71.0, hours_after(-2))]
    weather.forecast(over, hours_after(-6), hours_after(2))
    weather.forecast(playing, hours_after(-6), hours_after(2))
    clock.advance(16)
    weather.forecast(over, hours_after(-6), hours_after(2))
    weather.forecast(playing, hours_after(-6), hours_after(2))
    assert len(session.forecast_calls()) == 3  # only the game still on is refreshed after 15 minutes
