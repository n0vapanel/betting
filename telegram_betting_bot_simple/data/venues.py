"""The venue table: where a game is played, for the context layer (v3 Stage 4).

``data/venues.json`` holds every current home venue of the NFL, MLB, NBA, NHL, WNBA, MLS and Premier League, the
altitude venues (Estadio Azteca, Quito's Rodrigo Paz Delgado, La Paz's Hernando Siles, Bogotá's El Campín; Empower
Field is the Broncos' row) and a few other venues the captured boards used, one row per building (a building two
leagues share is one row listing both teams). How it was seeded:

- MLB rows come from MLB StatsAPI's venues (``/api/v1/venues?venueIds=…&hydrate=location,fieldInfo,timezone``,
  captured into tests/fixtures/context/mlb_venues.json): coordinates, elevation, roof, turf, ``azimuthAngle`` (home
  plate to centre field) and capacity. Park factors are Baseball Savant's Statcast runs and home-run indices, typed
  in by hand with their source and years.
- Every other row's coordinates and capacity were entered by hand (to about 1 km); a test checks each row is within
  40 km of its city's Open-Meteo geocode (tests/fixtures/context/geocodes.json). Their elevations come from Open-Meteo
  (the ``elevation`` its forecast answers carry, from a batched capture: tests/fixtures/context/elevations.json).
- ESPN venue ids and team ids come from ESPN scoreboards (tests/fixtures/context/espn_home_venues.json). ESPN gives
  one building different ids in different sports (Mercedes-Benz Stadium is 5348 for the NFL and 7485 for MLS), so a
  row lists every id it is known by; no id belongs to two rows.

``load_venues(path=None) -> VenueTable`` reads and validates the table (VenueError names the bad row).
``VenueTable.find(league, espn_venue_id=None, name=None, home_team_id=None, neutral=False)`` tries an ESPN venue id,
then the venue's name or an alias (normalised: case, accents, punctuation), then the home team's venue unless the game
is at a neutral site, and gives None otherwise. ``VenueTable.by_id`` maps a row id to its row (and can be called:
``by_id("coors-field")``). A row from ``find`` is a copy whose ``surface`` is the league's (an arena two leagues share
is hardwood for basketball, ice for hockey).

An unknown venue (a college ground, a cup tie abroad) can be placed approximately: ``approx_venue`` builds a row from
Open-Meteo's geocode of the game's city (data/weather.py's ``WeatherClient.geocode``), with ``approx: true``, no
elevation unless the geocoder gives one, and the roof from ESPN's ``indoor`` flag (dome when true, open otherwise).
"""

from __future__ import annotations

import datetime
import functools
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .espn_client import LEAGUES as ESPN_LEAGUES
from .weather import country_code

VENUES_PATH = Path(__file__).with_name("venues.json")

ROOFS = ("open", "dome", "retractable")
SURFACES = ("grass", "turf", "hardwood", "ice")
MLB = "baseball_mlb"
LEAGUE_SURFACES = {"basketball": "hardwood", "hockey": "ice"}  # by ESPN sport: an arena's floor for that league
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
REQUIRED = ("id", "name", "aliases", "espn_ids", "mlb_id", "city", "country", "lat", "lon", "elevation_m", "tz",
            "roof", "surface", "azimuth_deg", "capacity", "park", "teams", "source", "as_of")


class VenueError(ValueError):
    """A venue row that doesn't pass validation (the message names the row and the field)."""


def normalize_name(text: Any) -> str:
    """A venue name for matching: lowercase, accent-free, "&" as "and", other punctuation dropped
    ("Crypto.com Arena" and "crypto.com Arena" match, as do "St. James' Park" and "St James' Park")."""
    plain = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()
    plain = plain.replace("&", " and ").replace(".", "").replace("'", "").replace("’", "")
    plain = re.sub(r"[^a-z0-9]+", " ", plain)
    return " ".join(plain.split())


def _sport(league: str) -> Optional[str]:
    path = ESPN_LEAGUES.get(league)
    return path[0] if path else None


def surface_for(row: Mapping[str, Any], league: Optional[str]) -> Optional[str]:
    """The playing surface of ``row`` for ``league``'s games: hardwood for basketball and ice for hockey in an arena,
    else the row's own (a field's grass or turf)."""
    floor = LEAGUE_SURFACES.get(_sport(league or "") or "")
    if floor and row.get("roof") == "dome" and row.get("surface") in ("hardwood", "ice"):
        return floor
    return row.get("surface")


class _ById(dict):
    """Row id -> row; ``by_id("coors-field")`` works as ``by_id.get("coors-field")``."""

    def __call__(self, venue_id: Any) -> Optional[Dict[str, Any]]:
        return self.get(str(venue_id)) if venue_id is not None else None


class VenueTable:
    """The validated venue rows with their lookups (see the module's docstring)."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows: List[Dict[str, Any]] = [dict(row) for row in rows]
        for row in self.rows:
            row.setdefault("approx", False)
            row.setdefault("region", None)
        self.by_id: _ById = _ById()
        self._by_espn: Dict[str, Dict[str, Any]] = {}
        self._by_name: Dict[str, Dict[str, Any]] = {}
        self._by_team: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in self.rows:
            _validate(row)
            if row["id"] in self.by_id:
                raise VenueError(f"venue {row['id']}: the id is used twice")
            self.by_id[row["id"]] = row
            for espn_id in row["espn_ids"]:
                if espn_id in self._by_espn:
                    raise VenueError(f"venue {row['id']}: ESPN venue id {espn_id} is also "
                                     f"{self._by_espn[espn_id]['id']}'s")
                self._by_espn[espn_id] = row
            for name in [row["name"], *row["aliases"]]:
                key = normalize_name(name)
                other = self._by_name.get(key)
                if other is not None and other is not row:
                    raise VenueError(f"venue {row['id']}: the name {name!r} is also {other['id']}'s")
                self._by_name[key] = row
            for team in row["teams"]:
                key = (team["league"], str(team["espn_team_id"]))
                if key in self._by_team:
                    raise VenueError(f"venue {row['id']}: {team['name']} already plays at home at "
                                     f"{self._by_team[key]['id']}")
                self._by_team[key] = row

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.rows)

    def find(self, league: str, espn_venue_id: Any = None, name: Any = None, home_team_id: Any = None,
             neutral: bool = False) -> Optional[Dict[str, Any]]:
        """The game's venue: by ESPN venue id, else by name or alias, else (not ``neutral``) the home team's;
        None when none matches. The row is a copy, its ``surface`` the league's."""
        row = None
        if espn_venue_id not in (None, ""):
            row = self._by_espn.get(str(espn_venue_id).strip())
        if row is None and name:
            row = self._by_name.get(normalize_name(name))
        if row is None and not neutral and home_team_id not in (None, ""):
            row = self._by_team.get((league, str(home_team_id).strip()))
        if row is None:
            return None
        found = dict(row)
        found["surface"] = surface_for(row, league)
        return found

    def home_of(self, league: str, team_id: Any) -> Optional[Dict[str, Any]]:
        """The team's home venue (a copy, as find gives it), or None."""
        return self.find(league, home_team_id=team_id)


# --- validation -----------------------------------------------------------------------------------------------

def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fail(row: Mapping[str, Any], message: str) -> None:
    raise VenueError(f"venue {row.get('id') or row.get('name') or '?'}: {message}")


def _validate(row: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED if key not in row]
    if missing:
        _fail(row, f"missing {', '.join(missing)}")
    if not isinstance(row["id"], str) or not ID_PATTERN.match(row["id"]):
        _fail(row, "the id must be lowercase words joined by hyphens")
    for key in ("name", "city", "source"):
        if not isinstance(row[key], str) or not row[key].strip():
            _fail(row, f"{key} must be text")
    if not isinstance(row["aliases"], list) or not all(isinstance(a, str) and a.strip() for a in row["aliases"]):
        _fail(row, "aliases must be a list of names")
    if not isinstance(row["espn_ids"], list) or not all(isinstance(i, str) and i.isdigit() for i in row["espn_ids"]):
        _fail(row, "espn_ids must be a list of ESPN venue ids (digits, as text)")
    if row["mlb_id"] is not None and (not isinstance(row["mlb_id"], int) or isinstance(row["mlb_id"], bool)):
        _fail(row, "mlb_id must be MLB StatsAPI's venue id or null")
    if not isinstance(row["country"], str) or not re.fullmatch(r"[A-Z]{2}", row["country"]):
        _fail(row, "country must be a two-letter code")
    if not _number(row["lat"]) or not -90 <= row["lat"] <= 90 or not _number(row["lon"]) or \
            not -180 <= row["lon"] <= 180:
        _fail(row, "lat and lon must be coordinates")
    if row["elevation_m"] is not None and (not _number(row["elevation_m"]) or not -100 <= row["elevation_m"] <= 5000):
        _fail(row, "elevation_m must be metres above sea level")
    try:
        ZoneInfo(str(row["tz"]))
    except (ZoneInfoNotFoundError, ValueError):
        _fail(row, f"tz {row['tz']!r} is not an IANA time zone")
    if row["roof"] not in ROOFS:
        _fail(row, f"roof must be one of {', '.join(ROOFS)}")
    if row["surface"] not in SURFACES:
        _fail(row, f"surface must be one of {', '.join(SURFACES)}")
    if row["capacity"] is not None and (not isinstance(row["capacity"], int) or row["capacity"] <= 0):
        _fail(row, "capacity must be a whole number of seats or null")
    teams = row["teams"]
    if not isinstance(teams, list):
        _fail(row, "teams must be a list")
    for team in teams:
        if not isinstance(team, Mapping) or team.get("league") not in ESPN_LEAGUES or \
                not str(team.get("espn_team_id") or "").isdigit() or not isinstance(team.get("name"), str):
            _fail(row, f"a team must be {{league (an app league key), espn_team_id, name}}: {team!r}")
    baseball = any(team["league"] == MLB for team in teams)
    azimuth = row["azimuth_deg"]
    if azimuth is not None and (not _number(azimuth) or not 0 <= azimuth <= 360):
        _fail(row, "azimuth_deg must be a bearing in degrees")
    park = row["park"]
    if baseball:
        if azimuth is None:
            _fail(row, "an MLB park needs azimuth_deg (home plate to centre field)")
        if not isinstance(park, Mapping) or not _number(park.get("runs")) or not _number(park.get("hr")) or \
                not 50 <= park["runs"] <= 150 or not 50 <= park["hr"] <= 150 or \
                not isinstance(park.get("source"), str) or not park["source"].strip():
            _fail(row, "an MLB park needs park factors {runs, hr, source}")
    elif park is not None or azimuth is not None:
        _fail(row, "park factors and azimuth are for MLB parks only")
    try:
        datetime.date.fromisoformat(str(row["as_of"]))
    except ValueError:
        _fail(row, "as_of must be an ISO date")


# --- loading ------------------------------------------------------------------------------------------------------

def load_venues(path: Optional[Any] = None) -> VenueTable:
    """The venue table from ``path`` (default data/venues.json), validated: VenueError on a bad row."""
    source = Path(path) if path is not None else VENUES_PATH
    with open(source, encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise VenueError(f"{source}: the table must be a JSON list of rows")
    return VenueTable(rows)


@functools.lru_cache(maxsize=1)
def default_venues() -> VenueTable:
    """The packaged table, read once per process."""
    return load_venues()


# --- an unknown venue, placed by its city -------------------------------------------------------------------------

def approx_venue(name: Any, city: Any, country: Any = None, geocode: Optional[Mapping[str, Any]] = None,
                 indoor: Optional[bool] = None, espn_venue_id: Any = None) -> Optional[Dict[str, Any]]:
    """A row for a venue the table doesn't have, at its city's geocode (WeatherClient.geocode's answer: name, lat,
    lon, elevation, country, timezone); None without a geocode. ``approx`` is true; the elevation is the geocoder's
    (the city's, when it gives one); the roof is "dome" when ESPN says ``indoor``, else "open"; ``country`` may be a
    code or ESPN's name for it ("USA", "England")."""
    if not geocode or not _number(geocode.get("lat")) or not _number(geocode.get("lon")):
        return None
    label = str(name or city or geocode.get("name") or "venue")
    elevation = geocode.get("elevation")
    tz = geocode.get("timezone") or "UTC"
    try:
        ZoneInfo(str(tz))
    except (ZoneInfoNotFoundError, ValueError):
        tz = "UTC"
    return {
        "id": "approx-" + (normalize_name(label).replace(" ", "-") or "venue"),
        "name": label, "aliases": [], "espn_ids": [str(espn_venue_id)] if espn_venue_id not in (None, "") else [],
        "mlb_id": None, "city": str(city or geocode.get("name") or ""), "region": geocode.get("admin1"),
        "country": country_code(geocode.get("country")) or country_code(country),
        "lat": round(float(geocode["lat"]), 4), "lon": round(float(geocode["lon"]), 4),
        "elevation_m": round(float(elevation)) if _number(elevation) else None, "tz": tz,
        "roof": "dome" if indoor is True else "open", "surface": None, "azimuth_deg": None, "capacity": None,
        "park": None, "teams": [], "approx": True,
        "source": "Approximate: the city's position from Open-Meteo's geocoder",
        "as_of": datetime.date.today().isoformat(),
    }


def rows_for(table: VenueTable, leagues: Iterable[str]) -> List[Dict[str, Any]]:
    """The rows with a home team in any of ``leagues``."""
    wanted = set(leagues)
    return [row for row in table.rows if any(team["league"] in wanted for team in row["teams"])]
