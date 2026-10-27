"""Open-Meteo forecasts for venues, and its geocoder (v3 Stage 4).

Open-Meteo (https://open-meteo.com/) is free for non-commercial use with attribution (CC BY 4.0): every screen that
shows its data shows ATTRIBUTION linked to ATTRIBUTION_URL.

``WeatherClient(cache, store=None, session=None).forecast(points, start, end)`` takes ``(point id, lat, lon)``
triples (a fourth item, the point's own game start, decides that point's freshness; default ``start``) and gives
``{point id: [hour, ...]}`` for the hours from ``start`` (its hour) to ``end``, each hour
``{"t": "2026-10-27T18:00Z", "temp_c", "precip_prob", "precip_mm", "wind_kmh", "wind_dir", "gust_kmh"}``.
``forecasts`` gives each point's ``{"hours", "fetched_at", "elevation", "lat", "lon", "stale"}``.

- One HTTP call per up to 50 points (comma-separated ``latitude``/``longitude``), with 6 hourly variables (fewer than
  the 10 that count as one call), ``timezone=GMT`` and ``wind_speed_unit=kmh``. ``forecast_days`` covers the window,
  at least 3 days (so every game within 48 hours at a venue shares one forecast) and at most 7; a window starting
  yesterday (a game still on after midnight GMT) adds ``past_days=1``. A window starting more than 6 days ahead gets
  nothing (the slate shows no weather that far out).
- Every forecast is kept in the shared TTLCache under kind "weather", keyed by the point rounded to 2 decimals (about
  1 km; Open-Meteo's grid is coarser) and the date range: fresh for an hour, 15 minutes once the point's game starts
  within 6 hours or is on (started less than 4 hours ago): "near", data/cache.py's policy. A point with a fresh copy
  isn't fetched; one past its time is fetched again, and its last copy (marked stale) is served if the fetch fails or
  the cap is reached. The fetch itself runs inside a cache loader (the coverage audit), so concurrent callers share
  it; each point's copy is then kept on its own, so the other process reads it from the shared table.
- A daily counter (UTC days) in the store's SQLite file, shared by the bot and the dashboard, refuses calls past
  ``daily_cap`` HTTP calls (WEATHER_DAILY_CAP, default 1000, a setting) or past ``daily_units`` of Open-Meteo's
  weighted count (each location in a call counts as one; its free tier allows 10,000 a day), with a logged warning.
  A failed call (Open-Meteo answers 429 when its per-minute limit is hit) pauses fetching for 5 minutes, doubling to
  an hour while failures continue.

``geocode(name, country=None)`` looks a place up through Open-Meteo's geocoder (kind "geocode", kept 30 days):
``{"name", "lat", "lon", "elevation", "country", "admin1", "timezone", "population"}`` or None.
``at(hours, when)`` is the hour nearest ``when`` (None when there's none within 90 minutes).
"""

from __future__ import annotations

import datetime
import logging
import math
import sqlite3
import threading
import time
import unicodedata
from contextlib import closing
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests

from .cache import TTLCache, ttl_for

log = logging.getLogger(__name__)

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
GEOCODER = "https://geocoding-api.open-meteo.com/v1/search"
ATTRIBUTION = "Weather data by Open-Meteo.com"
ATTRIBUTION_URL = "https://open-meteo.com/"
SOURCE = "Open-Meteo"

# The hourly variables asked for, and each one's name in an hour
HOURLY = ("temperature_2m", "precipitation_probability", "precipitation", "wind_speed_10m", "wind_direction_10m",
          "wind_gusts_10m")
FIELDS = ("temp_c", "precip_prob", "precip_mm", "wind_kmh", "wind_dir", "gust_kmh")
MAX_POINTS = 50       # locations per call
MAX_DAYS = 7          # forecast days asked for at most (today and the next six)
MIN_DAYS = 3          # ... and at least: every game within 48 hours shares one cached forecast
NEAR = 6 * 3600       # a game starting within this is "near": its forecast is refreshed every 15 minutes ...
PLAYING = 4 * 3600    # ... as is one that started less than this ago (still on)
NEAREST_GAP = 90 * 60  # at(): an hour further than this from the time asked about is no answer
WEATHER_DAILY_CAP = 1000   # HTTP calls a day (the WEATHER_DAILY_CAP setting)
WEATHER_DAILY_UNITS = 8000  # Open-Meteo's weighted calls a day (each location counts one; its free limit is 10,000)
GEOCODE_DAILY_CAP = 500    # geocoder calls a day
BACKOFF = 5 * 60           # after a failed call, no call for this long ...
BACKOFF_CAP = 60 * 60      # ... doubling with each failure in a row up to this
WARN_EVERY = 3600          # the cap's warning at most this often
TIMEOUT = 15.0


# ESPN's address countries (and other common spellings) as the two-letter codes the geocoder filters on
COUNTRY_CODES = {
    "usa": "US", "united states": "US", "us": "US", "canada": "CA", "england": "GB", "scotland": "GB",
    "wales": "GB", "northern ireland": "GB", "united kingdom": "GB", "uk": "GB", "mexico": "MX", "brazil": "BR",
    "argentina": "AR", "chile": "CL", "colombia": "CO", "ecuador": "EC", "bolivia": "BO", "peru": "PE",
    "uruguay": "UY", "paraguay": "PY", "venezuela": "VE", "spain": "ES", "italy": "IT", "germany": "DE",
    "france": "FR", "netherlands": "NL", "portugal": "PT", "belgium": "BE", "turkey": "TR", "turkiye": "TR",
    "austria": "AT", "switzerland": "CH", "denmark": "DK", "greece": "GR", "norway": "NO", "sweden": "SE",
    "finland": "FI", "ireland": "IE", "republic of ireland": "IE", "russia": "RU", "saudi arabia": "SA",
    "japan": "JP", "australia": "AU", "poland": "PL", "croatia": "HR", "serbia": "RS", "ukraine": "UA",
    "czech republic": "CZ", "czechia": "CZ", "south korea": "KR", "korea republic": "KR", "china": "CN",
}


# ESPN's address states (two-letter postal codes) as the geocoder's admin1 names, US and Canada
REGION_NAMES = {
    "US": {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
           "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida",
           "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
           "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
           "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
           "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
           "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
           "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
           "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
           "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico"},
    "CA": {"AB": "Alberta", "BC": "British Columbia", "MB": "Manitoba", "NB": "New Brunswick",
           "NL": "Newfoundland and Labrador", "NS": "Nova Scotia", "NT": "Northwest Territories", "NU": "Nunavut",
           "ON": "Ontario", "PE": "Prince Edward Island", "QC": "Quebec", "SK": "Saskatchewan", "YT": "Yukon"},
}
UK_NATIONS = ("england", "scotland", "wales", "northern ireland")  # ESPN's "country" there is the geocoder's admin1


def region_name(region: Any, country: Any = None) -> Optional[str]:
    """The geocoder's admin1 name for an address's state or region, folded to plain lower case: ESPN's "AR" in the
    US is "arkansas", "ON" in Canada "ontario"; a name ("Utah") is kept; a UK nation given as the country
    ("England") is its own region. None when there is nothing to go on."""
    code = country_code(country)
    text = " ".join(str(region or "").split())
    if len(text) == 2 and text.isalpha():
        tables = [REGION_NAMES[code]] if code in REGION_NAMES else list(REGION_NAMES.values()) if not code else []
        for table in tables:
            if text.upper() in table:
                return _plain(table[text.upper()])
        return None  # a code the tables don't know: no guess
    if text:
        return _plain(text)
    nation = " ".join(str(country or "").split()).lower()
    return nation if code == "GB" and nation in UK_NATIONS else None


def country_code(country: Any) -> Optional[str]:
    """A two-letter country code from a code or a country's name ("USA", "England", "Mexico"); None if unknown."""
    text = " ".join(str(country or "").replace(".", "").split()).lower()
    if not text:
        return None
    if text in COUNTRY_CODES:
        return COUNTRY_CODES[text]
    return text.upper() if len(text) == 2 and text.isalpha() else None


def _plain(text: Any) -> str:
    return unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower().strip()


class DailyCapReached(RuntimeError):
    """The day's calls to a source are used up (WEATHER_DAILY_CAP)."""


# --- the daily counter ---------------------------------------------------------------------------------------------

class DailyCounter:
    """Calls per UTC day to one source. With a store (anything with a SQLite ``path``, i.e. data.store.Store) the
    count is kept in its file's ``api_daily_calls`` table, so the bot and the dashboard count together and the check
    and the increment are one transaction; otherwise (or if the file fails) in this process."""

    SCHEMA = ("CREATE TABLE IF NOT EXISTS api_daily_calls (source TEXT NOT NULL, day TEXT NOT NULL, "
              "calls INTEGER NOT NULL DEFAULT 0, units REAL NOT NULL DEFAULT 0, PRIMARY KEY (source, day))")
    KEEP_DAYS = 40

    def __init__(self, source: str, store: Any = None, clock: Optional[Callable[[], float]] = None) -> None:
        self.source = source
        self.clock = clock or time.time
        path = getattr(store, "path", None) if store is not None else None
        self._path = path if isinstance(path, str) and path else None
        self._memory: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._ready = False

    def day(self, now: Optional[float] = None) -> str:
        """Today's UTC date (ISO), the counter's day."""
        stamp = self.clock() if now is None else now
        return datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).date().isoformat()

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._path, timeout=10, isolation_level=None)
        if not self._ready:
            db.execute(self.SCHEMA)
            cutoff = (datetime.date.fromisoformat(self.day()) - datetime.timedelta(days=self.KEEP_DAYS)).isoformat()
            db.execute("DELETE FROM api_daily_calls WHERE day < ?", (cutoff,))
            self._ready = True
        return db

    def take(self, calls: int = 1, units: float = 1.0, cap_calls: Optional[int] = None,
             cap_units: Optional[float] = None) -> bool:
        """Count ``calls`` and ``units`` for today unless that would pass a cap; True when counted."""
        day = self.day()
        with self._lock:
            if self._path is not None:
                try:
                    with closing(self._db()) as db:
                        db.execute("BEGIN IMMEDIATE")
                        row = db.execute("SELECT calls, units FROM api_daily_calls WHERE source = ? AND day = ?",
                                         (self.source, day)).fetchone()
                        used_calls, used_units = (int(row[0]), float(row[1])) if row else (0, 0.0)
                        if not _within(used_calls + calls, cap_calls) or not _within(used_units + units, cap_units):
                            db.execute("ROLLBACK")
                            return False
                        db.execute("INSERT INTO api_daily_calls (source, day, calls, units) VALUES (?, ?, ?, ?) "
                                   "ON CONFLICT(source, day) DO UPDATE SET calls = calls + excluded.calls, "
                                   "units = units + excluded.units", (self.source, day, calls, units))
                        db.execute("COMMIT")
                        return True
                except sqlite3.Error as exc:
                    log.warning("%s call counter: the store failed (%s); counting in memory", self.source,
                                type(exc).__name__)
                    self._path = None
            used = self._memory.setdefault(day, [0, 0.0])
            for old in [d for d in self._memory if d < day]:
                del self._memory[old]
            if not _within(used[0] + calls, cap_calls) or not _within(used[1] + units, cap_units):
                return False
            used[0] += calls
            used[1] += units
            return True

    def count(self, day: Optional[str] = None) -> Tuple[int, float]:
        """(calls, units) counted on ``day`` (default today)."""
        day = day or self.day()
        with self._lock:
            if self._path is not None:
                try:
                    with closing(self._db()) as db:
                        row = db.execute("SELECT calls, units FROM api_daily_calls WHERE source = ? AND day = ?",
                                         (self.source, day)).fetchone()
                    return (int(row[0]), float(row[1])) if row else (0, 0.0)
                except sqlite3.Error:
                    pass
            used = self._memory.get(day) or [0, 0.0]
            return int(used[0]), float(used[1])


def _within(value: float, cap: Optional[float]) -> bool:
    return cap is None or value <= cap


# --- times ---------------------------------------------------------------------------------------------------------

def _utc(value: Any) -> Optional[datetime.datetime]:
    """An aware UTC datetime from a datetime (naive is UTC), epoch seconds or ISO text ("Z" allowed); else None."""
    if isinstance(value, datetime.datetime):
        return value.replace(tzinfo=datetime.timezone.utc) if value.tzinfo is None else \
            value.astimezone(datetime.timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return datetime.datetime.fromtimestamp(value, datetime.timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return _utc(parsed)
    return None


def _hour_text(moment: datetime.datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%MZ")


def at(hours: Optional[Sequence[Mapping[str, Any]]], when: Any,
       max_gap: float = NEAREST_GAP) -> Optional[Dict[str, Any]]:
    """The hour in ``hours`` nearest ``when`` (a datetime, epoch seconds or ISO text), or None when ``hours`` is
    empty or the nearest is more than ``max_gap`` seconds away."""
    target = _utc(when)
    if target is None or not hours:
        return None
    best, gap = None, math.inf
    for hour in hours:
        moment = _utc(hour.get("t")) if isinstance(hour, Mapping) else None
        if moment is None:
            continue
        distance = abs((moment - target).total_seconds())
        if distance < gap:
            best, gap = hour, distance
    return dict(best) if best is not None and gap <= max_gap else None


def attribution() -> Dict[str, str]:
    """{"text", "url"}: what every screen showing Open-Meteo's data links to (CC BY 4.0)."""
    return {"text": ATTRIBUTION, "url": ATTRIBUTION_URL}


def _number(value: Any) -> Optional[float]:
    """A finite number as it came (an int stays an int: the cached copy stays small), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def parse_forecast(item: Mapping[str, Any]) -> Dict[str, Any]:
    """One location of Open-Meteo's answer as the cached copy: {"start": first hour (ISO, GMT), "elevation", <the
    FIELDS as lists, one value per hour>}. Raises ValueError for an answer without hourly data."""
    hourly = item.get("hourly") if isinstance(item, Mapping) else None
    times = hourly.get("time") if isinstance(hourly, Mapping) else None
    if not isinstance(times, list) or not times:
        raise ValueError("Open-Meteo's answer has no hourly data")
    first = _utc(str(times[0]) + ("" if str(times[0]).endswith("Z") else "Z"))
    if first is None:
        raise ValueError("Open-Meteo's answer has an unreadable time")
    copy: Dict[str, Any] = {"start": _hour_text(first), "hours": len(times),
                            "elevation": _number(item.get("elevation"))}
    for variable, field in zip(HOURLY, FIELDS):
        values = hourly.get(variable)
        values = values[:len(times)] if isinstance(values, list) else []
        if not all(type(value) in (int, float) and math.isfinite(value) for value in values):
            values = [_number(value) for value in values]
        copy[field] = list(values) + [None] * (len(times) - len(values))
    return copy


def hours_of(copy: Mapping[str, Any], start: Optional[datetime.datetime] = None,
             end: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
    """A cached copy's hours as dicts, those from ``start``'s hour to ``end`` (both optional)."""
    first = _utc(copy.get("start"))
    count = int(copy.get("hours") or 0)
    if first is None or count <= 0:
        return []
    low, high = 0, count - 1
    if start is not None:
        low = max(low, math.ceil((start.replace(minute=0, second=0, microsecond=0) - first).total_seconds() / 3600))
    if end is not None:
        high = min(high, math.floor((end - first).total_seconds() / 3600))
    hours = []
    for index in range(low, high + 1):
        hour: Dict[str, Any] = {"t": _hour_text(first + datetime.timedelta(hours=index))}
        for field in FIELDS:
            values = copy.get(field) or []
            hour[field] = values[index] if index < len(values) else None
        hours.append(hour)
    return hours


# --- the client ------------------------------------------------------------------------------------------------

class WeatherClient:
    """Open-Meteo's forecast and geocoder through the shared TTLCache (see the module's docstring)."""

    at = staticmethod(at)
    attribution = staticmethod(attribution)

    def __init__(self, cache: Optional[TTLCache] = None, store: Any = None, session: Any = None, *,
                 daily_cap: Optional[int] = WEATHER_DAILY_CAP, daily_units: Optional[float] = WEATHER_DAILY_UNITS,
                 geocode_cap: Optional[int] = GEOCODE_DAILY_CAP, enabled: bool = True,
                 clock: Optional[Callable[[], float]] = None, timeout: float = TIMEOUT) -> None:
        self.cache = cache if cache is not None else TTLCache(name="weather")
        self.store = store
        self.session = session
        self.enabled = bool(enabled)
        self.daily_cap = daily_cap
        self.daily_units = daily_units
        self.geocode_cap = geocode_cap
        self.timeout = float(timeout)
        now = getattr(self.cache, "now", None)
        self.clock: Callable[[], float] = clock or (now if callable(now) else time.time)
        self.counter = DailyCounter("open-meteo", store, self.clock)
        self.geocode_counter = DailyCounter("open-meteo-geocoding", store, self.clock)
        self._lock = threading.Lock()
        self._failures = 0
        self._retry_at = -math.inf
        self._warned_at = -math.inf

    # --- HTTP (only ever inside a cache loader) ---

    def _get(self, url: str, params: Mapping[str, Any]) -> Any:
        getter = self.session.get if self.session is not None else requests.get
        response = getter(url, params=dict(params), timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0"})
        if not getattr(response, "ok", False):
            status = getattr(response, "status_code", "?")
            raise RuntimeError(f"Open-Meteo answered HTTP {status} for {url}")
        return response.json()

    # --- the daily cap and the pause after failures ---

    def _warn(self, message: str, *args: Any) -> None:
        now = time.monotonic()
        if now - self._warned_at >= WARN_EVERY:
            self._warned_at = now
            log.warning(message, *args)

    def _paused(self) -> bool:
        with self._lock:
            return self.clock() < self._retry_at

    def _failed(self) -> None:
        with self._lock:
            self._failures += 1
            wait = min(BACKOFF * 2 ** min(self._failures - 1, 10), BACKOFF_CAP)
            self._retry_at = self.clock() + wait

    def _succeeded(self) -> None:
        with self._lock:
            self._failures, self._retry_at = 0, -math.inf

    def calls_today(self) -> Dict[str, Any]:
        """{"day", "calls", "units", "cap", "units_cap", "geocoder_calls"}: today's (UTC) use of Open-Meteo."""
        calls, units = self.counter.count()
        geocoder, _ = self.geocode_counter.count()
        return {"day": self.counter.day(), "calls": calls, "units": units, "cap": self.daily_cap,
                "units_cap": self.daily_units, "geocoder_calls": geocoder}

    # --- forecasts ---

    def forecast(self, points: Iterable[Sequence[Any]], start: Any, end: Any) -> Dict[str, List[Dict[str, Any]]]:
        """{point id: hours from ``start``'s hour to ``end``} for the points with a forecast (see the module's
        docstring)."""
        return {pid: found["hours"] for pid, found in self.forecasts(points, start, end).items()}

    def forecasts(self, points: Iterable[Sequence[Any]], start: Any, end: Any) -> Dict[str, Dict[str, Any]]:
        """{point id: {"hours", "fetched_at" (epoch seconds), "elevation", "lat", "lon", "stale"}}: as forecast,
        with when each forecast was fetched and whether it is past its time (its refresh failed or was refused)."""
        begin, finish = _utc(start), _utc(end)
        if not self.enabled or begin is None or finish is None:
            return {}
        if finish < begin:
            begin, finish = finish, begin
        now = self.clock()
        today = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).date()
        if begin.date() > today + datetime.timedelta(days=MAX_DAYS - 1) or \
                finish.date() < today - datetime.timedelta(days=1):
            return {}
        past = 1 if begin.date() < today else 0
        days = min(MAX_DAYS, max(MIN_DAYS, (finish.date() - today).days + 1))
        first = today - datetime.timedelta(days=past)
        last = today + datetime.timedelta(days=days - 1)
        groups: Dict[Tuple[float, float], List[Tuple[str, datetime.datetime]]] = {}
        for point in points:
            pid, lat, lon = str(point[0]), _number(point[1]), _number(point[2])
            if lat is None or lon is None or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                continue
            game = _utc(point[3]) if len(point) > 3 and point[3] is not None else None
            groups.setdefault((round(lat, 2), round(lon, 2)), []).append((pid, game or begin))
        held: Dict[Tuple[float, float], Tuple[float, Any, bool]] = {}
        due: List[Tuple[float, float]] = []
        rules: Dict[Tuple[float, float], Any] = {}
        for spot, members in groups.items():
            near = any(-PLAYING <= game.timestamp() - now <= NEAR for _, game in members)
            rule = ttl_for("weather", "near" if near else None)
            rules[spot] = rule
            copy = self.cache.peek(self._key(spot, first, last)) if hasattr(self.cache, "peek") else None
            if copy is not None:
                age = now - float(copy[0])
                if age < rule.ttl:
                    held[spot] = (float(copy[0]), copy[1], False)
                    continue
                if age < rule.ttl + rule.stale:
                    held[spot] = (float(copy[0]), copy[1], True)
            due.append(spot)
        if due:
            held.update(self._refresh(due, rules, first, last, days, past))
        found: Dict[str, Dict[str, Any]] = {}
        for spot, members in groups.items():
            if spot not in held:
                continue
            fetched_at, copy, stale = held[spot]
            hours = hours_of(copy, begin, finish)
            for pid, _ in members:
                found[pid] = {"hours": [dict(hour) for hour in hours], "fetched_at": fetched_at,
                              "elevation": copy.get("elevation"), "lat": spot[0], "lon": spot[1], "stale": stale}
        return found

    @staticmethod
    def _key(spot: Tuple[float, float], first: datetime.date, last: datetime.date) -> Tuple[Any, ...]:
        return ("weather", spot[0], spot[1], first.isoformat(), last.isoformat())

    def _refresh(self, spots: List[Tuple[float, float]], rules: Mapping[Tuple[float, float], Any],
                 first: datetime.date, last: datetime.date, days: int, past: int
                 ) -> Dict[Tuple[float, float], Tuple[float, Any, bool]]:
        """Fetch ``spots`` in calls of up to MAX_POINTS (each inside a cache loader), keep each spot's copy in the
        cache, and give {spot: (fetched_at, copy, stale)} for those fetched (or just fetched by another caller)."""
        got: Dict[Tuple[float, float], Tuple[float, Any, bool]] = {}
        units_each = max(1.0, len(HOURLY) / 10.0) * max(1.0, (days + past) / 14.0)
        for index in range(0, len(spots), MAX_POINTS):
            if self._paused():
                log.debug("Open-Meteo: paused after a failure; %d points wait", len(spots) - index)
                break
            chunk = sorted(spots[index:index + MAX_POINTS])
            units = len(chunk) * units_each
            calls_used, units_used = self.counter.count()
            if not _within(calls_used + 1, self.daily_cap) or not _within(units_used + units, self.daily_units):
                self._warn("Open-Meteo's daily cap is reached (%d of %s calls, %.0f of %s weighted): weather is not "
                           "refreshed until tomorrow (UTC); %d points keep their last forecast", calls_used,
                           self.daily_cap, units_used, self.daily_units, len(spots) - index)
                break
            rule = min((rules[spot] for spot in chunk), key=lambda policy: policy.ttl)
            loaded: List[bool] = []
            failed: List[BaseException] = []

            def load(chunk: List[Tuple[float, float]] = chunk, loaded: List[bool] = loaded,
                     failed: List[BaseException] = failed, units: float = units) -> Dict[str, Any]:
                if not self.counter.take(1, units, self.daily_cap, self.daily_units):
                    error = DailyCapReached(f"Open-Meteo's daily cap is reached ({self.daily_cap} calls, "
                                            f"{self.daily_units} weighted)")
                    failed.append(error)
                    raise error
                params: Dict[str, Any] = {
                    "latitude": ",".join(f"{lat:.2f}" for lat, _ in chunk),
                    "longitude": ",".join(f"{lon:.2f}" for _, lon in chunk),
                    "hourly": ",".join(HOURLY), "timezone": "GMT", "wind_speed_unit": "kmh", "forecast_days": days,
                }
                if past:
                    params["past_days"] = past
                try:
                    data = self._get(OPEN_METEO, params)
                    items = data if isinstance(data, list) else [data]
                    if len(items) != len(chunk):
                        raise ValueError(f"Open-Meteo answered {len(items)} locations for {len(chunk)}")
                    copies = {f"{lat:.2f},{lon:.2f}": parse_forecast(item) for (lat, lon), item in zip(chunk, items)}
                except Exception as exc:
                    failed.append(exc)
                    raise
                loaded.append(True)
                return copies

            key = ("weather", "call", first.isoformat(), last.isoformat(), tuple(chunk))
            error: Optional[BaseException] = None
            try:
                copies, fetched_at = self.cache.get_with_time(key, rule.ttl, load, stale=0.0, error_ttl=0.0,
                                                              persist=False, kind="weather")
            except Exception as exc:
                error = exc
            # A failed load may come back as the cache's last copy of the same call: still a failure
            error = error or (failed[-1] if failed else None)
            if isinstance(error, DailyCapReached):
                self._warn("%s: weather is not refreshed until tomorrow (UTC); %d points keep their last forecast",
                           error, len(spots) - index)
                break
            if error is not None:  # Open-Meteo down or limiting: pause; the held copies are served, marked stale
                self._failed()
                log.warning("Open-Meteo forecast for %d points failed (%s); pausing for %.0f minutes", len(chunk),
                            str(error)[:160] or type(error).__name__, max(0.0, self._retry_at - self.clock()) / 60)
                continue
            self._succeeded()
            now = self.clock()
            for spot in chunk:
                copy = copies.get(f"{spot[0]:.2f},{spot[1]:.2f}")
                if copy is None:
                    continue
                spot_rule = rules[spot]
                if loaded:  # fetched now: kept per spot, for every reader here and in the other process
                    at_time = self.cache.put(self._key(spot, first, last), copy, ttl=spot_rule.ttl,
                                             stale=spot_rule.stale, kind="weather")
                    got[spot] = (float(at_time), copy, False)
                else:  # another caller's identical call a moment ago
                    got[spot] = (float(fetched_at), copy, now - float(fetched_at) >= spot_rule.ttl)
            if loaded:  # each spot's copy is kept on its own: the call's copy goes (no stale call to fall back on)
                self.cache.invalidate(key)
        return got

    # --- the geocoder ---

    def geocode(self, name: Any, country: Any = None, region: Any = None) -> Optional[Dict[str, Any]]:
        """The best match for a place name (in ``country``, a two-letter code or a name such as ESPN's "USA" or
        "England", when given) from Open-Meteo's geocoder: {"name", "lat", "lon", "elevation", "country", "admin1",
        "timezone", "population"}; None when nothing matches or geocoding is off. ``region``: the address's state
        (ESPN's "AR", or a name): a place in exactly that admin1 wins over a bigger same-named one elsewhere
        (Fayetteville, AR is not Fayetteville, NC), else the first match as before. A name written "City, Region"
        (ESPN's "Sandy, Utah") is looked up as the city, the region picking among places of that name. Kept 30
        days (kind "geocode")."""
        text = " ".join(str(name or "").split())
        code = country_code(country) or ""
        if not self.enabled or not text:
            return None
        rule = ttl_for("geocode")
        place, _, hint = text.partition(",")  # ESPN writes some cities "Sandy, Utah"
        place, hint = place.strip() or text, _plain(hint.strip(" ."))
        admin1 = region_name(region, country) or ""

        def load() -> Optional[Dict[str, Any]]:
            if not self.geocode_counter.take(1, 1.0, self.geocode_cap, None):
                raise DailyCapReached(f"Open-Meteo geocoder's daily cap is reached ({self.geocode_cap} calls)")
            # A region to match: more same-named places, so a small town's is among them
            data = self._get(GEOCODER, {"name": place, "count": 30 if admin1 else 10, "language": "en",
                                        "format": "json"})
            results = [r for r in (data.get("results") or []) if isinstance(r, Mapping)] if isinstance(data, Mapping) \
                else []
            if code:
                results = [r for r in results if str(r.get("country_code") or "").upper() == code]
            if admin1:  # the address's state: an exact admin1 match first, in the geocoder's order
                exact = [r for r in results if _plain(r.get("admin1")) == admin1]
                results = exact + [r for r in results if _plain(r.get("admin1")) != admin1]
            elif hint:  # the state or region after the comma picks among same-named places
                results = sorted(results, key=lambda r: not any(
                    hint in _plain(r.get(field)) or _plain(r.get(field)) in hint
                    for field in ("admin1", "admin2", "country") if r.get(field)))
            if not results:
                return None
            best = results[0]
            if _number(best.get("latitude")) is None or _number(best.get("longitude")) is None:
                return None
            return {"name": best.get("name"), "lat": float(best["latitude"]), "lon": float(best["longitude"]),
                    "elevation": _number(best.get("elevation")), "country": best.get("country_code"),
                    "admin1": best.get("admin1"), "timezone": best.get("timezone"),
                    "population": best.get("population")}

        try:
            key = ("geocode", text.lower(), code) + ((admin1,) if admin1 else ())
            return self.cache.get(key, rule.ttl, load, stale=rule.stale, error_ttl=300.0, kind="geocode")
        except DailyCapReached as exc:
            self._warn("%s", exc)
            return None
        except Exception as exc:
            log.info("Open-Meteo geocoder failed for %r (%s)", text[:60], type(exc).__name__)
            return None
