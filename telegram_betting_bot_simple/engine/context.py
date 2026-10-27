"""Context maths: venues, weather, rest and travel explain the market's line; only newer information moves it.

Two layers (report, "Venues, weather and news explain the line; only newer information moves it"), kept apart on
every screen:

1. **Attribution, "Why the line is here".** It changes no number. ``attribute_margin`` and ``attribute_total`` split
   the market's own margin or total into a labelled baseline plus sized, sourced factors (home field, altitude, rest
   and back-to-backs, eastward time-zone shifts, the park, the weather as forecast when the odds were fetched) and
   a remainder, key ``"other"`` ("Team strength and other" / "Teams and other"), that makes them sum exactly to the
   market's number (``engine.factors.check_sum``).
2. **Adjustment, "What moves it".** It moves a number only for what the odds snapshot cannot contain: a forecast
   that changed after the odds were fetched, or a roof announced closed (or open) since. ``adjustment`` applies
   ``clamp(ADJUST_WEIGHT x delta, -cap, +cap)`` with ``ADJUST_WEIGHT = 0.5`` and the sport's cap; with no market to
   anchor to, the raw change applies in full and says so. ``adjusted_over`` turns an applied change to a total into
   a new chance of the over.

Starting effect sizes (each becomes a Factor carrying its source, evidence, as_of and confidence):

    Home field (2-way)   league home win rate (standings, shrunk to the prior with 400 games), as points through
                         the sport's margin SD; soccer as goal supremacy through the goals model
    Altitude             NBA: +2.6 pts at Denver (8.6 vs 6.0 net per 100 possessions), Utah scaled by elevation
                         above 600 m; NFL: +0.5 pts at 1,500 m+; soccer: 0.25 goals per 1,000 m of elevation
                         difference at venues above 1,500 m (half the confounded BMJ slope)
    Zero rest            NBA back-to-back -1.25 pts (the NBA model's setting); NHL -5.8 points of win chance
    Eastward shift       2 h+ east within 3 days, home team: MLB -3.5 points of win chance; NBA -1.3 pts (untested)
    NFL wind             0 at <= 8 mph, -2.2 pts at 12, -3.8 at >= 16, linear between
    NFL cold             -1.0 pt at <= 32 F; with wind >= 15 mph, -2.0 more
    NFL rain             -1.0 pt at a 60%+ chance and 0.5+ mm/h
    MLB temperature      +0.15 runs per 10 F above 70 F (negative below)
    MLB wind             +-0.08 runs per mph of the out/in component above 5 mph, capped at +-1 run
    MLB park             (Savant runs index - 100) / 100 x the league average total
    Soccer weather       total goals x1.068 above 22 C, x0.97 below 6 C, x1.014 in rain (0.5+ mm/h)
    Dome or roof closed  no weather effect
    Retractable roof,    no weather effect either way (counted in "Teams and other"), no weather badge and no
    state unknown        adjustment: the market's assumption about the roof isn't known. A roof adjustment needs
                         the state known at the odds' time (announced) and announced flipped since

College football uses the NFL's weather and altitude figures. Weak effects carry the confidence "small, noisy"; the
jet-lag factors also say "untested" in their evidence (no study checks whether closing lines already hold them).

Probabilities from points: ``implied_margin = sd x inverse-normal(p_home)``, ``margin_to_prob = normal(m / sd)``;
soccer goes through the Stage 3 goals model (engine/soccer.py) instead.

Every number here is a starting value from the literature, listed in Help; the weight and caps are design choices,
not published values. The weight rises only after 200 graded games show the adjusted line beating the unadjusted
one. This module is pure: no I/O and no clock.
"""

from __future__ import annotations

import datetime
import math
import re
from dataclasses import dataclass
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

from . import soccer
from .factors import NOISY, NORMAL, Factor, Projection, allocate
from .nba_model import Params as _NbaParams

_N = NormalDist()

# --- the adjustment layer ----------------------------------------------------------------------------------------

ADJUST_WEIGHT = 0.5  # share of a post-odds change that moves the number (raised only after 200 graded games)
WIN_CAP_PP = 2.5  # the most an adjustment may move a win chance, in percentage points
MIN_DELTA = 0.005  # a change smaller than this shows as 0.00: no adjustment
KINDS = ("total", "side", "win")
CAUSES = ("forecast", "roof_closed", "roof_open")

# --- unit conversions --------------------------------------------------------------------------------------------

KMH_PER_MPH = 1.609344
EARTH_RADIUS_KM = 6371.0088  # mean Earth radius


def kmh_to_mph(kmh: float) -> float:
    return kmh / KMH_PER_MPH


def mph_to_kmh(mph: float) -> float:
    return mph * KMH_PER_MPH


def c_to_f(celsius: float) -> float:
    return celsius * 9 / 5 + 32


def f_to_c(fahrenheit: float) -> float:
    return (fahrenheit - 32) * 5 / 9


# --- effect sizes ------------------------------------------------------------------------------------------------

HOME_PRIOR_GAMES = 400  # games of prior weight the season's home win rate is shrunk with (data/context_inputs.py)

# NFL (and college football): CBS Sports, NFL games since 2000: 44.3 combined points at 0-9 mph, 42.1 at 10-14,
# 40.5 at 15+, 46.9 indoors, 37.9 freezing with 15+ mph wind; Sharp Football Analysis: unders 57.4% in rain (343)
NFL_WIND_POINTS: Tuple[Tuple[float, float], ...] = ((8.0, 0.0), (12.0, -2.2), (16.0, -3.8))  # (mph, pts)
NFL_COLD_F = 32.0
NFL_COLD = -1.0
NFL_COLD_WIND_MPH = 15.0
NFL_COLD_WIND = -2.0
NFL_RAIN = -1.0
RAIN_PROB = 60.0  # % chance of precipitation that counts as rain (NFL and the badge)
RAIN_MM = 0.5  # mm/h at the start that counts as rain

# MLB: about 3 ft of carry per 10 F (Alan Nathan); Wrigley 6.0 runs a game with warm out-blowing wind, 4.1 in
# (FanGraphs 2013)
MLB_TEMP_BASE_F = 70.0
MLB_TEMP_PER_10F = 0.15
MLB_WIND_PER_MPH = 0.08
MLB_WIND_FLOOR_MPH = 5.0
MLB_WIND_CAP = 1.0
FIELD_ANGLE = {"CF": 0.0, "LF": 45.0, "RF": 45.0}  # degrees off the home plate -> centre field line

# Soccer: Poisson coefficients for goals against a 10-14 C baseline (Misak 2025)
SOCCER_HEAT_C = 22.0
SOCCER_HEAT = 1.068
SOCCER_COLD_C = 6.0
SOCCER_COLD = 0.97
SOCCER_RAIN = 1.014
# The goals model's home share of the expected goals (implied means 1.557 home, 1.245 away; engine/soccer.py)
SOCCER_HOME_SHARE = 1.557 / (1.557 + 1.245)

# Altitude
ALTITUDE_M = 1500.0  # venues at or above this count as high (badge, NFL, soccer)
NBA_ALTITUDE_PTS = 2.6  # Denver: home-road net rating gap 8.6 vs 6.0 per 100 possessions (Sportico 2024)
NBA_ALTITUDE_FLOOR_M = 600.0  # elevation above which the NBA term scales (Utah 1,288 m)
NBA_ALTITUDE_REF_M = 1580.0  # Denver, where the full +2.6 applies
NFL_ALTITUDE_PTS = 0.5
SOCCER_ALTITUDE_PER_1000M = 0.25  # half McSharry's BMJ 2007 slope (0.5 goals per 1,000 m, confounded)

# Rest and travel
NBA_BACK_TO_BACK = _NbaParams().back_to_back  # 1.25 pts, the NBA model's own setting
NHL_ZERO_REST_PP = 5.8  # 45.0% on zero rest against 50.8% on one day (nhlscraper vignette, 2005-26)
JETLAG_HOURS = 2.0  # an eastward shift of at least this many hours ...
JETLAG_DAYS = 3  # ... since a game no more than this many days before
MLB_JETLAG_PP = 3.5  # home teams' win rate after eastward travel (Song, Severini & Allada, PNAS 2017)
NBA_JETLAG_PTS = 1.3  # home teams' points after eastward travel (Leota et al. 2022)
NFL_SHORT_REST_DAYS = 4  # days since the last game at or below which NFL rest is unusual (a Thursday game)

# Badges
BADGE_WIND_KMH = 24.0
BADGE_COLD_C = 0.0
BADGE_HOT_C = 30.0
BADGE_LIMIT = 3
BADGE_ORDER = ("adj", "wind", "rain", "cold", "hot", "roof", "alt", "tz-home", "tz-away", "rest-home", "rest-away")

# Sources (Factor.source: where the input came from)
SRC_STANDINGS = "ESPN standings, home records"
SRC_PRIOR = "League prior (recent seasons' home records)"
SRC_FORECAST = "Open-Meteo forecast"
SRC_MLB = "MLB (announced)"
SRC_VENUE = "Venue table"
SRC_SCHEDULE = "ESPN schedule"
SRC_PARK = "Baseball Savant park factors"
SRC_MARKET = "The market's line (what the listed factors leave)"

UNTESTED = "untested"  # the word the waterfall looks for in a factor's evidence


# --- sports ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SportParams:
    """One sport's context parameters. ``caps``: "total" and "side" in the sport's unit, "win" in points of chance.

    ``margin_sd`` and ``total_sd`` turn points into chances (2-way sports); soccer's are rough and unused, since the
    goals model prices soccer. ``league_total`` is the fallback baseline when the caller has no league average.
    ``digits`` is how many decimals the sport's factors are worth showing.
    """

    key: str
    label: str
    unit: str  # "pts" | "runs" | "goals"
    margin_sd: float
    total_sd: float
    caps: Mapping[str, float]
    home_prior: float
    draw_prior: Optional[float] = None
    league_total: float = 0.0
    outdoor: bool = False
    digits: int = 1
    home_label: str = "Home field"

    @property
    def two_way(self) -> bool:
        return self.draw_prior is None


def _side_cap(margin_sd: float) -> float:
    """The win-chance cap (2.5 points either side of even) as a margin in the sport's unit."""
    return margin_sd * _N.inv_cdf(0.5 + WIN_CAP_PP / 100)


def _make(key: str, label: str, unit: str, margin_sd: float, total_sd: float, total_cap: float, home_prior: float,
          league_total: float, *, side_cap: Optional[float] = None, draw_prior: Optional[float] = None,
          outdoor: bool = False, home_label: str = "Home field") -> SportParams:
    caps = {"total": total_cap, "side": side_cap if side_cap is not None else _side_cap(margin_sd),
            "win": WIN_CAP_PP}
    return SportParams(key, label, unit, margin_sd, total_sd, MappingProxyType(caps), home_prior, draw_prior,
                       league_total, outdoor, 1 if unit == "pts" else 2, home_label)


# Margin and total SDs, caps and home-win priors from the Stage 4 spec; league totals are recent seasons' averages,
# only a fallback (the caller passes today's league average when it has one)
SPORTS: Dict[str, SportParams] = {
    "nfl": _make("nfl", "NFL", "pts", 12.7, 13.2, 2.0, 0.539, 45.5, side_cap=1.0, outdoor=True),
    "ncaaf": _make("ncaaf", "College football", "pts", 14.5, 15.0, 2.0, 0.57, 55.0, side_cap=1.0, outdoor=True),
    "nba": _make("nba", "NBA", "pts", 12.0, 18.0, 1.5, 0.554, 228.0, home_label="Home court"),
    "wnba": _make("wnba", "WNBA", "pts", 11.0, 16.0, 1.5, 0.55, 164.0, home_label="Home court"),
    "ncaab": _make("ncaab", "College basketball", "pts", 10.5, 16.0, 1.5, 0.60, 145.0, home_label="Home court"),
    "nhl": _make("nhl", "NHL", "goals", 2.3, 2.2, 0.15, 0.522, 6.1, home_label="Home ice"),
    "mlb": _make("mlb", "MLB", "runs", 4.3, 4.4, 0.4, 0.529, 8.8, outdoor=True),
    # +-0.15 goals per team: 0.30 for the match total, 0.15 for one side
    "soccer": _make("soccer", "Soccer", "goals", 1.7, 1.7, 0.30, 0.45, 2.8, side_cap=0.15, draw_prior=0.27,
                    outdoor=True, home_label="Home advantage"),
}

_LEAGUE_SPORT = {
    "americanfootball_nfl": "nfl",
    "americanfootball_ncaaf": "ncaaf",
    "basketball_nba": "nba",
    "basketball_wnba": "wnba",
    "basketball_ncaab": "ncaab",
    "icehockey_nhl": "nhl",
    "baseball_mlb": "mlb",
}


def sport_of(league: Any) -> Optional[str]:
    """The context sport for a league key ("basketball_nba" -> "nba", any "soccer_..." -> "soccer"); a sport key
    is returned as is; None for leagues with no context parameters (tennis, MMA, the CFL, women's college...)."""
    text = str(league or "").strip().lower()
    if text in SPORTS:
        return text
    if text.endswith("_preseason"):
        text = text[: -len("_preseason")]
    if text in _LEAGUE_SPORT:
        return _LEAGUE_SPORT[text]
    if text.startswith("soccer_") and not text.endswith("_winner"):
        return "soccer"
    return None


def params(sport: Any) -> SportParams:
    """The SportParams for a sport or league key; ValueError when there are none."""
    key = sport_of(sport)
    if key is None:
        raise ValueError(f"No context parameters for {sport!r}")
    return SPORTS[key]


# --- small helpers -----------------------------------------------------------------------------------------------

def _num(value: Any) -> Optional[float]:
    """A finite float, or None (bools and text are not numbers here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _finite(value: Any, name: str) -> float:
    number = _num(value)
    if number is None:
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def _chance(value: Any, name: str = "chance") -> float:
    """A chance kept inside [1e-4, 1 - 1e-4] so the inverse normal stays finite."""
    number = _num(value)
    if number is None or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a chance between 0 and 1, got {value!r}")
    return min(max(number, 1e-4), 1 - 1e-4)


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _when(at: Any) -> datetime.datetime:
    """An aware UTC datetime from a datetime (naive = UTC), an ISO string or epoch seconds."""
    if isinstance(at, datetime.datetime):
        when = at
    elif isinstance(at, str):
        try:
            when = datetime.datetime.fromisoformat(at.strip().replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"Not a time: {at!r}") from None
    elif _num(at) is not None:
        when = datetime.datetime.fromtimestamp(float(at), datetime.timezone.utc)
    else:
        raise ValueError(f"Not a time: {at!r}")
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    return when.astimezone(datetime.timezone.utc)


def _zone(name: Any) -> ZoneInfo:
    try:
        return ZoneInfo(str(name))
    except Exception:  # ZoneInfoNotFoundError, ValueError for odd paths, TypeError
        raise ValueError(f"Unknown time zone {name!r}") from None


def _fmt(value: float) -> str:
    """A badge number: whole numbers without a decimal point, else one decimal."""
    rounded = round(value, 1)
    return str(int(round(rounded))) if abs(rounded - round(rounded)) < 1e-9 else f"{rounded:g}"


_SINGULAR = {"pts": "point", "runs": "run", "goals": "goal"}


def _signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


# --- distance and time zones -------------------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points (degrees), on a sphere of the mean Earth radius."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    half_dlat = (p2 - p1) / 2
    half_dlon = math.radians(lon2 - lon1) / 2
    a = math.sin(half_dlat) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(half_dlon) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def tz_shift_hours(from_tz: str, to_tz: str, at: Any) -> float:
    """Hours the clock moves for a team travelling from ``from_tz`` to ``to_tz`` (IANA names), both read at the
    instant ``at`` (a datetime, naive = UTC; an ISO string; or epoch seconds), so daylight saving is applied where
    it is in force that day. Positive means eastward travel (Los Angeles -> New York is +3)."""
    when = _when(at)
    to_offset = when.astimezone(_zone(to_tz)).utcoffset() or datetime.timedelta(0)
    from_offset = when.astimezone(_zone(from_tz)).utcoffset() or datetime.timedelta(0)
    return (to_offset - from_offset).total_seconds() / 3600


def days_since_last(trip: Optional[Mapping[str, Any]]) -> Optional[int]:
    """Days between a team's last game and this one: ``days_since_last`` when given, else ``rest_days + 1``
    (``rest_days`` 0 is a back-to-back, the day after)."""
    trip = _map(trip)
    days = _num(trip.get("days_since_last"))
    if days is None:
        rest = _num(trip.get("rest_days"))
        days = None if rest is None else rest + 1
    return None if days is None else int(round(days))


def on_zero_rest(trip: Optional[Mapping[str, Any]]) -> bool:
    """True for a team on a back-to-back: ``back_to_back`` when given, else ``rest_days == 0``."""
    trip = _map(trip)
    flag = trip.get("back_to_back")
    if isinstance(flag, bool):
        return flag
    return _num(trip.get("rest_days")) == 0


def jet_lag_hours(trip: Optional[Mapping[str, Any]]) -> Optional[float]:
    """The eastward shift when it counts (at least 2 h, since a game no more than 3 days before), else None."""
    shift = _num(_map(trip).get("tz_shift_h"))
    days = days_since_last(trip)
    if shift is None or shift < JETLAG_HOURS or days is None or days > JETLAG_DAYS:
        return None
    return shift


# --- probabilities from points -----------------------------------------------------------------------------------

_RATE_MIN = 0.05


def _soccer_rates(supremacy: float, total: float) -> Tuple[float, float]:
    room = max(total - 2 * _RATE_MIN, 0.0)
    s = min(max(supremacy, -room), room)
    return (total + s) / 2, (total - s) / 2


def _soccer_outcomes(supremacy: float, total: float) -> Tuple[float, float, float]:
    return soccer.outcome_probs(soccer.grid(*_soccer_rates(supremacy, total)))


def _soccer_total(total: Optional[float]) -> float:
    value = SPORTS["soccer"].league_total if total is None else _finite(total, "total")
    if value <= 2 * _RATE_MIN:
        raise ValueError(f"A soccer total must be above {2 * _RATE_MIN}, got {total!r}")
    return value


def soccer_supremacy(p_home: float, *, p_draw: Optional[float] = None, total: Optional[float] = None) -> float:
    """Goal supremacy (home rate minus away rate) at the match total ``total`` (default the league average) whose
    Dixon-Coles grid gives the home side ``p_home``, or, with ``p_draw``, the gap between home and away chances."""
    p = _chance(p_home, "p_home")
    t = _soccer_total(total)
    if p_draw is None:
        target, index = p, 0
    else:
        d = _chance(p_draw, "p_draw")
        if p + d >= 1:
            raise ValueError(f"p_home + p_draw must be below 1, got {p_home!r} + {p_draw!r}")
        target, index = p - (1 - p - d), 1

    def reading(s: float) -> float:
        home, _draw, away = _soccer_outcomes(s, t)
        return home if index == 0 else home - away

    room = t - 2 * _RATE_MIN
    low, high = -room, room
    for _ in range(60):
        middle = (low + high) / 2
        if reading(middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def implied_margin(sport: Any, p_home: float, *, p_draw: Optional[float] = None,
                   total: Optional[float] = None) -> float:
    """The home margin a home chance implies: ``margin_sd x inverse-normal(p_home)``. Soccer: the goal supremacy the
    goals model needs at ``total`` (default the league average) for that home chance (or, with ``p_draw``, for that
    home-minus-away gap)."""
    sp = params(sport)
    if sp.key == "soccer":
        return soccer_supremacy(p_home, p_draw=p_draw, total=total)
    return sp.margin_sd * _N.inv_cdf(_chance(p_home, "p_home"))


def margin_to_prob(sport: Any, margin: float, *, total: Optional[float] = None) -> float:
    """The home chance for a home margin: ``normal(margin / margin_sd)``. Soccer: the goals model's home-win chance
    at that supremacy and ``total`` (default the league average)."""
    sp = params(sport)
    m = _finite(margin, "margin")
    if sp.key == "soccer":
        return _soccer_outcomes(m, _soccer_total(total))[0]
    return _N.cdf(m / sp.margin_sd)


def _pp_to_margin(sp: SportParams, pp: float) -> float:
    """A change of ``pp`` points of win chance, from an even game, as a margin in the sport's unit."""
    return sp.margin_sd * _N.inv_cdf(0.5 + pp / 100)


def home_edge(sport: Any, home_win: Optional[float] = None, draw: Optional[float] = None, *,
              total: Optional[float] = None) -> float:
    """The home edge in the sport's unit from a home win rate (default the sport's prior): points through the
    margin SD; soccer: goal supremacy from the home and draw rates (default draw the prior) through the goals
    model at ``total``."""
    sp = params(sport)
    p = sp.home_prior if home_win is None else home_win
    if sp.key == "soccer":
        return soccer_supremacy(p, p_draw=sp.draw_prior if draw is None else draw, total=total)
    return implied_margin(sp.key, p)


# --- conditions --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Weather:
    """Conditions at the start, normalised: each number in both units (as given, so thresholds aren't blurred by a
    round trip), wind direction as meteorologists give it (where the wind blows FROM), and where each came from."""

    temp_f: Optional[float] = None
    temp_c: Optional[float] = None
    wind_mph: Optional[float] = None
    wind_kmh: Optional[float] = None
    wind_from: Optional[float] = None
    gust_mph: Optional[float] = None
    precip_prob: Optional[float] = None
    precip_mm: Optional[float] = None
    wind_kind: Optional[str] = None  # MLB's announcement: "out" | "in" | "cross" | "calm" | "varies"
    wind_field: Optional[str] = None  # "CF" | "LF" | "RF"
    wind_text: Optional[str] = None
    source: str = SRC_FORECAST  # the forecast's source (precipitation always comes from it)
    temp_source: str = SRC_FORECAST
    wind_source: str = SRC_FORECAST
    as_of: Optional[str] = None


_WIND_TEXT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*mph\s*,?\s*(.*?)\s*$", re.I)
_ANNOUNCED_KEYS = ("wind_text", "wind_kind", "condition")


def _parse_wind_text(text: Any) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """(mph, kind, field) from MLB's wind text: "14 mph, Out To RF" -> (14, "out", "RF"); "In From LF" -> (None,
    "in", "LF"); "6 mph, L To R" -> (6, "cross", None); "0 mph, None" and "Calm" -> calm; "Varies" -> varies;
    anything else -> (mph or None, None, None). data/mlb_stats.parse_wind does the same for the data layer; this
    keeps the engine pure."""
    if not isinstance(text, str) or not text.strip():
        return None, None, None
    match = _WIND_TEXT.match(text)
    speed, rest = (float(match.group(1)), match.group(2)) if match else (None, text)
    rest = rest.strip().lower()
    words = rest.split()
    field = words[-1].upper() if words and words[-1].upper() in FIELD_ANGLE else None
    if rest in ("none", "calm") or speed == 0:
        return (speed or 0.0), "calm", None
    if rest.startswith("out to"):
        return speed, "out", field
    if rest.startswith("in from"):
        return speed, "in", field
    if rest in ("l to r", "r to l"):
        return speed, "cross", None
    if rest.startswith("varies"):
        return speed, "varies", None
    return speed, None, None


def _announced(conditions: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """MLB's announced weather: ``conditions["announced"]``, or the conditions themselves when they are MLB's
    block (they carry wind_text, wind_kind or condition)."""
    nested = conditions.get("announced")
    if isinstance(nested, Mapping):
        return nested
    if any(key in conditions for key in _ANNOUNCED_KEYS):
        return conditions
    return None


def announced_roof_closed(conditions: Optional[Mapping[str, Any]]) -> Optional[bool]:
    """True when the roof is announced closed, False when announced open, None when nothing is announced."""
    conditions = _map(conditions)
    flag = conditions.get("roof_closed")
    if isinstance(flag, bool):
        return flag
    announced = _announced(conditions)
    if announced is None:
        return None
    flag = announced.get("roof_closed")
    condition = str(announced.get("condition") or "").strip().lower()
    if condition in ("roof closed", "dome"):
        return True
    if isinstance(flag, bool):
        return flag
    return False if condition else None


def _still_air(conditions: Optional[Mapping[str, Any]]) -> bool:
    """MLB announced "0 mph, None": what it shows under a closed roof (and under a dome)."""
    announced = _announced(_map(conditions))
    text = announced.get("wind_text") if announced is not None else None
    return isinstance(text, str) and " ".join(text.lower().replace(",", " ").split()) == "0 mph none"


def roof_state(venue: Optional[Mapping[str, Any]], conditions: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """"dome", "closed" or "open"; None for a retractable roof whose state isn't known (``weather_counts`` is then
    False: no weather effect, badge or adjustment, since nobody can say what the market assumed). An announcement (MLB's "Roof Closed") beats the venue row's ``roof_state``; at a
    retractable park MLB's "0 mph, None" also means closed (it announced Seattle's that way under an "Overcast"
    condition)."""
    venue = _map(venue)
    roof = str(venue.get("roof") or "open").strip().lower()
    if roof == "dome":
        return "dome"
    if roof == "retractable" and _still_air(conditions):
        return "closed"
    announced = announced_roof_closed(conditions)
    if announced is not None:
        return "closed" if announced else "open"
    state = str(venue.get("roof_state") or "").strip().lower()
    if state in ("open", "closed"):
        return state
    return None if roof == "retractable" else "open"


def is_covered(venue: Optional[Mapping[str, Any]], conditions: Optional[Mapping[str, Any]] = None) -> bool:
    """True under a dome or a closed roof: no weather effect."""
    return roof_state(venue, conditions) in ("dome", "closed")


def roof_unknown(venue: Optional[Mapping[str, Any]], conditions: Optional[Mapping[str, Any]] = None) -> bool:
    """True for a retractable roof whose state isn't announced (NFL roofs never are; MLB's only hours before the
    first pitch). ESPN's ``indoor`` flag is no state: it marks every retractable park indoor whatever the day."""
    return roof_state(venue, conditions) is None


def weather_counts(venue: Optional[Mapping[str, Any]], conditions: Optional[Mapping[str, Any]] = None) -> bool:
    """True when the weather counts at the venue: open air, or a retractable roof announced open. False under a dome,
    a closed roof or a retractable roof whose state isn't known."""
    return roof_state(venue, conditions) == "open"


def read_conditions(sport: Any, conditions: Optional[Mapping[str, Any]]) -> Weather:
    """Normalise one hour of conditions. Open-Meteo's hour: ``temp_c``, ``wind_kmh``, ``wind_dir`` (from),
    ``gust_kmh``, ``precip_prob`` (%), ``precip_mm`` (mm/h), ``t``; ``temp_f`` / ``wind_mph`` / ``gust_mph`` work
    too. For MLB, the announced weather (``conditions["announced"]`` or MLB's own block: ``temp_f``, ``wind_mph``,
    ``wind_text``, ``wind_kind``, ``field``) wins over the forecast for temperature and wind."""
    c = _map(conditions)
    key = sport_of(sport)
    start = c.get("start")
    if isinstance(start, Mapping) and not any(name in c for name in ("temp_c", "temp_f", "wind_kmh", "wind_mph")):
        # The context block's weather: the hour at the start, with the announcement and fetch time beside it
        c = {**start, **{name: c[name] for name in ("announced", "fetched_at", "roof_closed", "source") if name in c}}

    temp_f, temp_c = _num(c.get("temp_f")), _num(c.get("temp_c"))
    if temp_f is None and temp_c is not None:
        temp_f = c_to_f(temp_c)
    elif temp_f is not None:
        temp_c = f_to_c(temp_f)
    wind, wind_kmh = _num(c.get("wind_mph")), _num(c.get("wind_kmh"))
    if wind is None and wind_kmh is not None:
        wind = kmh_to_mph(wind_kmh)
    elif wind is not None:
        wind_kmh = mph_to_kmh(wind)
    gust = _num(c.get("gust_mph"))
    if gust is None and _num(c.get("gust_kmh")) is not None:
        gust = kmh_to_mph(float(c["gust_kmh"]))
    source = str(c.get("source") or SRC_FORECAST)
    if source.lower() in ("open-meteo", "openmeteo"):
        source = SRC_FORECAST
    temp_source = wind_source = source
    kind = field = text = None

    announced = _announced(c) if key == "mlb" else None
    if announced is not None:
        a_temp = _num(announced.get("temp_f"))
        if a_temp is not None:
            temp_f, temp_c, temp_source = a_temp, f_to_c(a_temp), SRC_MLB
        a_kind = announced.get("wind_kind")
        a_kind = str(a_kind).lower() if isinstance(a_kind, str) and a_kind else None
        a_field = announced.get("field")
        a_field = str(a_field).upper() if isinstance(a_field, str) and str(a_field).upper() in FIELD_ANGLE else None
        a_speed = _num(announced.get("wind_mph"))
        parsed = _parse_wind_text(announced.get("wind_text"))
        if a_kind is None:
            a_kind, a_field = parsed[1], a_field or parsed[2]
        if a_speed is None:
            a_speed = parsed[0]
        if a_kind is not None:
            kind, field = a_kind, a_field
            wind_source = SRC_MLB
            text = announced.get("wind_text") if isinstance(announced.get("wind_text"), str) else None
            wind = a_speed if a_speed is not None else (0.0 if kind == "calm" else wind)
            wind_kmh = None if wind is None else mph_to_kmh(wind)

    as_of = c.get("fetched_at") or c.get("as_of")
    return Weather(temp_f=temp_f, temp_c=temp_c, wind_mph=wind, wind_kmh=wind_kmh, wind_from=_num(c.get("wind_dir")),
                   gust_mph=gust, precip_prob=_num(c.get("precip_prob")), precip_mm=_num(c.get("precip_mm")),
                   wind_kind=kind, wind_field=field, wind_text=text, source=source, temp_source=temp_source,
                   wind_source=wind_source, as_of=str(as_of) if as_of else None)


def is_rain(weather: Weather) -> bool:
    """A 60%+ chance of precipitation and 0.5+ mm/h at the start (the NFL rule and the badge)."""
    return (weather.precip_prob is not None and weather.precip_prob >= RAIN_PROB
            and weather.precip_mm is not None and weather.precip_mm >= RAIN_MM)


# --- weather effects ---------------------------------------------------------------------------------------------

def nfl_wind_points(mph: float) -> float:
    """Points off an NFL total for sustained wind: 0 at <= 8 mph, -2.2 at 12, -3.8 at >= 16, linear between."""
    points = NFL_WIND_POINTS
    if mph <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if mph <= x1:
            return y0 + (y1 - y0) * (mph - x0) / (x1 - x0)
    return points[-1][1]


def wind_component(azimuth_deg: float, wind_from_deg: float, speed: float) -> float:
    """The part of the wind blowing out towards centre field (negative: blowing in). ``azimuth_deg`` is the
    bearing from home plate to centre field; ``wind_from_deg`` where the wind comes from, as forecasts give it."""
    return speed * math.cos(math.radians(wind_from_deg + 180.0 - azimuth_deg))


def announced_component(speed: float, kind: Optional[str], field: Optional[str]) -> float:
    """The out (+) / in (-) component of MLB's announced wind: "Out To CF" counts in full, "Out To RF" / "In From
    LF" at cos 45 degrees, cross, calm and varying wind as 0."""
    if kind not in ("out", "in"):
        return 0.0
    angle = FIELD_ANGLE.get(field or "", 45.0)
    sign = 1.0 if kind == "out" else -1.0
    return sign * speed * math.cos(math.radians(angle))


def mlb_wind_runs(component_mph: float) -> float:
    """Runs for an out/in wind component: +-0.08 per mph above 5 mph, capped at +-1 run."""
    excess = max(abs(component_mph) - MLB_WIND_FLOOR_MPH, 0.0)
    runs = math.copysign(MLB_WIND_PER_MPH * excess, component_mph) if excess else 0.0
    return max(-MLB_WIND_CAP, min(MLB_WIND_CAP, runs))


def _metric(units: Optional[str], sport: str = "") -> bool:
    """Whether evidence puts metric figures first: ``units`` "metric", or (None) soccer's default."""
    return units == "metric" or (units is None and sport == "soccer")


def _temps(weather: Weather, units: Optional[str] = None, sport: str = "") -> str:
    """"28°F (−2°C)", metric first ("−2°C (28°F)") for ``units`` "metric" (and soccer by default)."""
    if _metric(units, sport):
        return f"{weather.temp_c:.0f}°C ({weather.temp_f:.0f}°F)"
    return f"{weather.temp_f:.0f}°F ({weather.temp_c:.0f}°C)"


def _winds(mph: float, units: Optional[str] = None) -> str:
    """"19 mph (30 km/h)", metric first ("30 km/h (19 mph)") for ``units`` "metric"."""
    if units == "metric":
        return f"{mph_to_kmh(mph):.0f} km/h ({mph:.0f} mph)"
    return f"{mph:.0f} mph ({mph_to_kmh(mph):.0f} km/h)"


def _deg(fahrenheit: float, units: Optional[str]) -> str:
    """A rule's temperature in both scales, the person's first: "32°F (0°C)" / "0°C (32°F)"."""
    celsius = f_to_c(fahrenheit)
    return f"{celsius:.0f}°C ({fahrenheit:.0f}°F)" if units == "metric" else f"{fahrenheit:.0f}°F ({celsius:.0f}°C)"


def _football_weather(sp: SportParams, w: Weather, as_of: Optional[str], units: Optional[str] = None) -> List[Factor]:
    college = " NFL figures, used for college football too." if sp.key == "ncaaf" else ""
    out: List[Factor] = []
    if w.wind_mph is not None:
        points = nfl_wind_points(w.wind_mph)
        if points:
            gust = (f", gusting {_winds(w.gust_mph, units)}" if w.gust_mph is not None and w.gust_mph > w.wind_mph
                    else "")
            if units == "metric":
                rule = "0 at 13 km/h (8 mph) or less, −2.2 pts at 19 km/h (12 mph), −3.8 at 26 km/h (16 mph) or more"
            else:
                rule = "0 at 8 mph or less, −2.2 pts at 12, −3.8 at 16 or more"
            out.append(Factor(
                "wind", "Wind at kick-off", points, sp.unit,
                evidence=(f"{_winds(w.wind_mph, units)} sustained{gust}, open-air: {rule} (NFL games since 2000: "
                          f"44.3 pts at 0–9 mph, 42.1 at 10–14, 40.5 at 15+).{college}"),
                source=w.wind_source, as_of=as_of, confidence=NORMAL))
    if w.temp_f is not None and w.temp_f <= NFL_COLD_F:
        out.append(Factor(
            "cold", "Freezing at kick-off", NFL_COLD, sp.unit,
            evidence=f"{_temps(w, units)}: −1 pt at {_deg(NFL_COLD_F, units)} or below.{college}",
            source=w.temp_source, as_of=as_of, confidence=NORMAL))
        if w.wind_mph is not None and w.wind_mph >= NFL_COLD_WIND_MPH:
            out.append(Factor(
                "cold_wind", "Freezing and windy", NFL_COLD_WIND, sp.unit,
                evidence=(f"{_temps(w, units)} with {_winds(w.wind_mph, units)} wind: 2 pts more when it's freezing "
                          f"with wind of {_winds(NFL_COLD_WIND_MPH, units)} or more (37.9 pts a game in such NFL "
                          f"games since 2000).{college}"),
                source=w.wind_source, as_of=as_of, confidence=NORMAL))
    if is_rain(w):
        out.append(Factor(
            "rain", "Rain at kick-off", NFL_RAIN, sp.unit,
            evidence=(f"{w.precip_prob:.0f}% chance, {w.precip_mm:.1f} mm/h: −1 pt at a 60%+ chance and 0.5+ mm/h "
                      f"(unders won 57.4% of 343 NFL games in measurable rain).{college}"),
            source=w.source, as_of=as_of, confidence=NOISY))
    return out


def _mlb_weather(sp: SportParams, venue: Mapping[str, Any], w: Weather, as_of: Optional[str],
                 units: Optional[str] = None) -> List[Factor]:
    out: List[Factor] = []
    if w.temp_f is not None:
        runs = MLB_TEMP_PER_10F * (w.temp_f - MLB_TEMP_BASE_F) / 10
        if runs:
            rule = ("+0.15 runs per 5.6°C (10°F) above 21°C (70°F), less below (a fly ball carries about 1 m further "
                    "per 5.6°C)" if units == "metric" else
                    "+0.15 runs per 10°F above 70°F, less below (a fly ball carries about 3 ft further per 10°F)")
            out.append(Factor(
                "temperature", "Temperature at first pitch", runs, sp.unit,
                evidence=f"{_temps(w, units)}: {rule}.",
                source=w.temp_source, as_of=as_of, confidence=NOISY))
    component: Optional[float] = None
    described = ""
    if w.wind_kind is not None:
        speed = w.wind_mph or 0.0
        component = announced_component(speed, w.wind_kind, w.wind_field)
        described = w.wind_text or f"{speed:.0f} mph, {w.wind_kind}{' ' + w.wind_field if w.wind_field else ''}"
        described = f"MLB announced \"{described}\""
    else:
        azimuth = _num(venue.get("azimuth_deg"))
        if azimuth is not None and w.wind_mph is not None and w.wind_from is not None:
            component = wind_component(azimuth, w.wind_from, w.wind_mph)
            described = (f"{_winds(w.wind_mph, units)} from {w.wind_from:.0f}°, with the park facing {azimuth:.0f}° "
                         "from home plate to centre field")
    if component is not None:
        runs = mlb_wind_runs(component)
        if runs:
            rule = ("±0.08 runs per mph (1.6 km/h) above 5 mph (8 km/h)" if units == "metric"
                    else "±0.08 runs per mph above 5 mph")
            out.append(Factor(
                "wind", "Wind blowing out" if runs > 0 else "Wind blowing in", runs, sp.unit,
                evidence=(f"{described}: {_winds(abs(component), units)} "
                          f"{'out towards' if component > 0 else 'in from'} centre field; {rule}, at most ±1 run "
                          "(Wrigley: 6.0 runs a game with warm wind blowing out, 4.1 blowing in)."),
                source=w.wind_source, as_of=as_of, confidence=NOISY))
    return out


def _soccer_multipliers(w: Weather, as_of: Optional[str], units: Optional[str] = None) -> Dict[str, Tuple[Any, ...]]:
    mult: Dict[str, Tuple[Any, ...]] = {}
    temp_c = w.temp_c
    imperial = units == "imperial"
    if temp_c is not None and temp_c > SOCCER_HEAT_C:
        above = "72°F (22°C)" if imperial else "22°C"
        mult["heat"] = ("Heat", SOCCER_HEAT, f"{_temps(w, units, 'soccer')}: about 6.8% more goals above {above} "
                        "(Mišák 2025).", w.temp_source, NOISY, as_of)
    elif temp_c is not None and temp_c < SOCCER_COLD_C:
        below = "43°F (6°C)" if imperial else "6°C"
        mult["cold"] = ("Cold", SOCCER_COLD, f"{_temps(w, units, 'soccer')}: about 3% fewer goals below {below} "
                        "(Mišák 2025).", w.temp_source, NOISY, as_of)
    if w.precip_mm is not None and w.precip_mm >= RAIN_MM:
        mult["rain"] = ("Rain", SOCCER_RAIN, f"{w.precip_mm:.1f} mm/h: about 1.4% more goals in rain (Mišák 2025).",
                        w.source, NOISY, as_of)
    return mult


def weather_multiplier(sport: Any, venue: Optional[Mapping[str, Any]],
                       conditions: Optional[Mapping[str, Any]]) -> float:
    """Soccer: the product of the weather multipliers on total goals (1.0 under a roof, with no conditions, or for
    any other sport)."""
    sp = params(sport)
    if sp.key != "soccer" or conditions is None or not weather_counts(venue, conditions):
        return 1.0
    return math.prod(spec[1] for spec in _soccer_multipliers(read_conditions(sp.key, conditions), None).values())


def weather_factors(sport: Any, venue: Optional[Mapping[str, Any]], conditions: Optional[Mapping[str, Any]], *,
                    as_of: Optional[str] = None, base_total: Optional[float] = None,
                    units: Optional[str] = None) -> List[Factor]:
    """The weather's effects on the total, as Factors in the sport's unit (only the non-zero ones).

    [] for indoor sports, with no conditions, under a dome or with the roof closed, and for a retractable roof whose
    state isn't known (``weather_counts``). Soccer's multipliers become goals on ``base_total`` (the total before
    weather; default the league average). ``as_of`` defaults to the conditions' ``fetched_at``. ``units``
    ("metric" / "imperial") puts that system's figures first in the evidence (default: imperial, soccer metric);
    the sizes are the same either way.
    """
    sp = params(sport)
    if not sp.outdoor or conditions is None or not weather_counts(venue, conditions):
        return []
    w = read_conditions(sp.key, conditions)
    when = as_of or w.as_of
    if sp.key in ("nfl", "ncaaf"):
        return _football_weather(sp, w, when, units)
    if sp.key == "mlb":
        return _mlb_weather(sp, _map(venue), w, when, units)
    mult = _soccer_multipliers(w, when, units)
    if not mult:
        return []
    base = sp.league_total if base_total is None else _finite(base_total, "base_total")
    return allocate(base, mult, unit=sp.unit)


def weather_total_effect(sport: Any, venue: Optional[Mapping[str, Any]], conditions: Optional[Mapping[str, Any]], *,
                         base_total: Optional[float] = None) -> float:
    """The sum of ``weather_factors``: the weather's effect on the total in the sport's unit (0 under a roof or with
    no conditions). Soccer: ``base_total x (product of multipliers - 1)``."""
    return math.fsum(f.delta for f in weather_factors(sport, venue, conditions, base_total=base_total))


def weather_change(sport: Any, before: Optional[Mapping[str, Any]], now: Optional[Mapping[str, Any]], *,
                   units: Optional[str] = None) -> str:
    """What changed between two forecasts, in plain words for an adjustment's reason: "wind 8 → 18 mph",
    "31 → 45°F", "rain now expected". ``units``: "imperial" (default, soccer "metric") or "metric"."""
    key = sport_of(sport)
    metric = (units or ("metric" if key == "soccer" else "imperial")) == "metric"
    a, b = read_conditions(key, before), read_conditions(key, now)
    parts: List[str] = []
    if a.wind_mph is not None and b.wind_mph is not None and abs(a.wind_mph - b.wind_mph) >= 1:
        if metric:
            parts.append(f"wind {mph_to_kmh(a.wind_mph):.0f} → {mph_to_kmh(b.wind_mph):.0f} km/h")
        else:
            parts.append(f"wind {a.wind_mph:.0f} → {b.wind_mph:.0f} mph")
    if a.temp_f is not None and b.temp_f is not None and abs(a.temp_f - b.temp_f) >= 2:
        parts.append(f"{a.temp_c:.0f} → {b.temp_c:.0f}°C" if metric else f"{a.temp_f:.0f} → {b.temp_f:.0f}°F")
    wet_a = a.precip_mm is not None and a.precip_mm >= RAIN_MM
    wet_b = b.precip_mm is not None and b.precip_mm >= RAIN_MM
    if wet_a != wet_b:
        parts.append("rain now expected" if wet_b else "rain no longer expected")
    return ", ".join(parts)


# --- attribution -------------------------------------------------------------------------------------------------

def _team(trips: Optional[Mapping[str, Any]], side: str) -> Mapping[str, Any]:
    return _map(_map(trips).get(side))


def _name(trip: Mapping[str, Any], side: str) -> str:
    name = trip.get("name")
    return str(name) if isinstance(name, str) and name.strip() else f"{side.title()} team"


def _home_field_factor(sp: SportParams, edge: Optional[float], rates: Optional[Mapping[str, Any]], neutral: bool,
                       as_of: Optional[str]) -> Factor:
    if neutral:
        return Factor("home_field", sp.home_label, 0.0, sp.unit, evidence="Neutral site: no home edge.",
                      source=SRC_SCHEDULE, as_of=as_of, confidence=NORMAL)
    rates = _map(rates)
    p = _num(rates.get("home_win"))
    draw = _num(rates.get("draw"))
    given = edge is not None
    if edge is None:
        edge = home_edge(sp.key, p, draw) if p is not None else home_edge(sp.key)
    edge = _finite(edge, "home_edge")
    digits = sp.digits
    if sp.key == "soccer":
        how = f"about {_signed(edge, 2)} goals of supremacy through the goals model"
    else:
        spread = f"{sp.margin_sd:g}-{_SINGULAR[sp.unit]}"
        how = f"about {_signed(edge, digits)} {sp.unit} through a {spread} spread of results"
    if p is not None:
        games = _num(rates.get("games"))
        raw = _num(rates.get("raw"))
        seen = (f"Home teams won {raw:.1%} of {games:.0f} games this season" if raw is not None and games
                else "This season's home records")
        drawn = f" (draws {draw:.0%})" if draw is not None else ""
        evidence = (f"{seen}{drawn}, shrunk toward the {sp.home_prior:.1%} prior with {HOME_PRIOR_GAMES} games of "
                    f"weight to {p:.1%}: {how}.")
        source = str(rates.get("source") or SRC_STANDINGS)
        when = rates.get("as_of") or as_of
    elif given:
        evidence = f"The league's home edge from this season's home records: {how}."
        source, when = str(rates.get("source") or SRC_STANDINGS), rates.get("as_of") or as_of
    else:
        drawn = f", {sp.draw_prior:.0%} draws" if sp.draw_prior is not None else ""
        evidence = f"Home teams win about {sp.home_prior:.1%}{drawn} in recent seasons: {how}."
        source, when = SRC_PRIOR, as_of
    return Factor("home_field", sp.home_label, edge, sp.unit, evidence=evidence, source=source,
                  as_of=str(when) if when else None, confidence=NORMAL)


def _altitude_factor(sp: SportParams, venue: Mapping[str, Any], trips: Optional[Mapping[str, Any]],
                     as_of: Optional[str], units: Optional[str] = None) -> Optional[Factor]:
    elevation = _num(venue.get("elevation_m"))
    if elevation is None:
        return None
    where = str(venue.get("name") or "the venue")
    if units == "imperial":
        height = f"{where}, {elevation / 0.3048:,.0f} ft ({elevation:,.0f} m)"
    else:
        height = f"{where}, {elevation:,.0f} m ({elevation / 0.3048:,.0f} ft)"
    source = str(venue.get("source") or SRC_VENUE)
    if sp.key == "nba" and elevation > NBA_ALTITUDE_FLOOR_M:
        share = min(1.0, (elevation - NBA_ALTITUDE_FLOOR_M) / (NBA_ALTITUDE_REF_M - NBA_ALTITUDE_FLOOR_M))
        points = NBA_ALTITUDE_PTS * share
        denver = elevation >= ALTITUDE_M
        scaled = "" if share >= 1 else (f" Scaled by elevation above 600 m: {elevation:,.0f} m gives "
                                        f"{_signed(points, 1)} pts.")
        return Factor("altitude", "Altitude", points, sp.unit,
                      evidence=(f"{height}. Denver's home teams run 8.6 net points per 100 possessions better than on "
                                "the road, against a 6.0 league norm (Sportico 2024): about +2.6 pts at Denver's "
                                "1,580 m." + scaled),
                      source=source, as_of=as_of, confidence=NORMAL if denver else NOISY)
    if sp.key in ("nfl", "ncaaf") and elevation >= ALTITUDE_M:
        college = " The NFL figure, used for college football too." if sp.key == "ncaaf" else ""
        return Factor("altitude", "Altitude", NFL_ALTITUDE_PTS, sp.unit,
                      evidence=(f"{height}: about +0.5 pts for the home side at 1,500 m or more (Denver's extra home "
                                f"edge).{college}"),
                      source=source, as_of=as_of, confidence=NOISY)
    if sp.key == "soccer" and elevation >= ALTITUDE_M:
        visitor = _num(_team(trips, "away").get("home_elevation_m"))
        if visitor is None:
            difference = elevation - ALTITUDE_M
            basis = "the visitors' home elevation isn't known, so only the height above 1,500 m counts"
        else:
            difference = elevation - visitor
            basis = f"the visitors play at home at {visitor:,.0f} m"
        goals = SOCCER_ALTITUDE_PER_1000M * max(difference, 0.0) / 1000
        if not goals:
            return None
        return Factor("altitude", "Altitude", goals, sp.unit,
                      evidence=(f"{height}: 0.25 goals per 1,000 m of elevation difference at venues above 1,500 m; "
                                f"{basis}. "
                                "Half the slope a BMJ study of 1,460 South American internationals found, since team "
                                "quality partly confounds it."),
                      source=source, as_of=as_of, confidence=NOISY)
    return None


def _rest_factor(sp: SportParams, trip: Mapping[str, Any], side: str, as_of: Optional[str]) -> Optional[Factor]:
    if not on_zero_rest(trip):
        return None
    sign = -1.0 if side == "home" else 1.0
    name = _name(trip, side)
    if sp.key == "nba":
        return Factor(f"{side}_rest", f"{name} on a back-to-back", sign * NBA_BACK_TO_BACK, sp.unit,
                      evidence=(f"Second night of a back-to-back: about {NBA_BACK_TO_BACK:g} pts (the NBA model's own "
                                "setting; NBA games 2003–10)."),
                      source=SRC_SCHEDULE, as_of=as_of, confidence=NORMAL)
    if sp.key == "nhl":
        goals = _pp_to_margin(sp, -NHL_ZERO_REST_PP)
        return Factor(f"{side}_rest", f"{name} on a back-to-back", sign * abs(goals), sp.unit,
                      evidence=(f"Teams on zero rest won 45.0% against 50.8% on one day's rest (50,568 team-games, "
                                f"2005–26): −{NHL_ZERO_REST_PP:g} points of win chance, about {abs(goals):.2f} "
                                "goals."),
                      source=SRC_SCHEDULE, as_of=as_of, confidence=NORMAL)
    return None


def _jet_lag_factor(sp: SportParams, trip: Mapping[str, Any], as_of: Optional[str]) -> Optional[Factor]:
    """The home team's eastward shift (the studies found it for home teams)."""
    shift = jet_lag_hours(trip)
    if shift is None or sp.key not in ("mlb", "nba"):
        return None
    name = _name(trip, "home")
    label = f"{name}: {_fmt(shift)} h time-zone shift east"
    if sp.key == "mlb":
        runs = _pp_to_margin(sp, -MLB_JETLAG_PP)
        evidence = (f"Home teams won {MLB_JETLAG_PP:g} points less often after crossing 2+ time zones east (Song, "
                    f"Severini & Allada, PNAS 2017; 1992–2011): about {_signed(runs, 2)} runs. Whether the line "
                    f"already holds it is {UNTESTED}.")
        return Factor("home_tz", label, runs, sp.unit, evidence=evidence, source=SRC_SCHEDULE, as_of=as_of,
                      confidence=NOISY)
    evidence = (f"Eastward jet lag cost NBA home teams about {NBA_JETLAG_PTS:g} pts (Leota et al. 2022). Whether the "
                f"line already holds it is {UNTESTED}.")
    return Factor("home_tz", label, -NBA_JETLAG_PTS, sp.unit, evidence=evidence, source=SRC_SCHEDULE, as_of=as_of,
                  confidence=NOISY)


def attribute_margin(sport: Any, *, market_margin: float, home_edge: Optional[float] = None,
                     venue: Optional[Mapping[str, Any]] = None, trips: Optional[Mapping[str, Mapping[str, Any]]] = None,
                     neutral: bool = False, as_of: Optional[str] = None,
                     home_rates: Optional[Mapping[str, Any]] = None, units: Optional[str] = None) -> Projection:
    """"Why the line is here" for the home margin (home minus away, in the sport's unit): an even game (0), plus
    home field (0 at a neutral site), altitude, each side's back-to-back, the home team's eastward time-zone shift,
    plus "Team strength and other" (key "other"), the remainder, so the factors sum exactly to ``market_margin``.

    ``home_edge``: the league's home edge in the sport's unit (``home_edge()``); None works it out from
    ``home_rates`` (data/context_inputs.home_rates: home_win, draw, games, raw, source, as_of), else the prior.
    ``trips``: {"home"|"away": {"name", "rest_days" (0 = back-to-back), "back_to_back", "tz_shift_h",
    "days_since_last", "travel_km", "home_elevation_m" (soccer altitude: the visitors' home)}}. ``units``
    ("metric" / "imperial") orders the altitude evidence's figures (default metric first).
    """
    sp = params(sport)
    market = _finite(market_margin, "market_margin")
    venue = _map(venue)
    factors: List[Factor] = [_home_field_factor(sp, home_edge, home_rates, bool(neutral), as_of)]
    if not neutral:
        altitude = _altitude_factor(sp, venue, trips, as_of, units)
        if altitude is not None:
            factors.append(altitude)
    for side in ("home", "away"):
        rest = _rest_factor(sp, _team(trips, side), side, as_of)
        if rest is not None:
            factors.append(rest)
    jet_lag = _jet_lag_factor(sp, _team(trips, "home"), as_of)
    if jet_lag is not None:
        factors.append(jet_lag)
    remainder = market - math.fsum(f.delta for f in factors)
    factors.append(Factor(
        "other", "Team strength and other", remainder, sp.unit,
        evidence=("The rest of the market's margin: the teams' strength, injuries and anything not listed here."),
        source=SRC_MARKET, as_of=as_of, confidence=NORMAL))
    return Projection("Even game", 0.0, factors, market, sp.unit, "margin", "Home margin (home minus away)")


def park_factor(sport: Any, venue: Optional[Mapping[str, Any]], league_total: float,
                as_of: Optional[str] = None) -> Optional[Factor]:
    """MLB: (runs index - 100) / 100 x the league average total; the home-run index is shown, not added."""
    sp = params(sport)
    venue = _map(venue)
    park = _map(venue.get("park"))
    runs = _num(park.get("runs"))
    if sp.key != "mlb" or runs is None:
        return None
    hr = _num(park.get("hr"))
    name = str(venue.get("name") or "the park")
    delta = (runs - 100) / 100 * league_total
    evidence = f"{name}: runs index {runs:g} (100 is average) × the {league_total:.1f}-run league average"
    evidence += f"; home-run index {hr:g}, shown, not added." if hr is not None else "."
    return Factor("park", f"Park: {name}", delta, sp.unit, evidence=evidence,
                  source=str(park.get("source") or SRC_PARK), as_of=as_of, confidence=NORMAL)


def attribute_total(sport: Any, *, market_total: float, league_total: Optional[float] = None,
                    venue: Optional[Mapping[str, Any]] = None,
                    conditions_at_odds: Optional[Mapping[str, Any]] = None,
                    as_of: Optional[str] = None, units: Optional[str] = None) -> Projection:
    """"Why the line is here" for the total: the league average total, plus the park (MLB), plus the weather as
    forecast when the odds were fetched, plus "Teams and other" (key "other"), the remainder, so the factors sum
    exactly to ``market_total``. ``league_total`` None uses the sport's fallback average. Soccer's weather
    multipliers are taken off the market's own total (the total before weather is market / multiplier). A
    retractable roof whose state wasn't known at the odds' time counts no weather (the remainder says so).
    ``units`` orders the weather evidence's figures (``weather_factors``)."""
    sp = params(sport)
    total = _finite(market_total, "market_total")
    base = sp.league_total if league_total is None else _finite(league_total, "league_total")
    venue_map = _map(venue)
    factors: List[Factor] = []
    park = park_factor(sp.key, venue_map, base, as_of)
    if park is not None:
        factors.append(park)
    if conditions_at_odds is not None:
        before_weather = None
        if sp.key == "soccer":
            before_weather = total / weather_multiplier(sp.key, venue_map, conditions_at_odds)
        factors.extend(weather_factors(sp.key, venue_map, conditions_at_odds, base_total=before_weather, units=units))
    evidence = "The rest of the market's total: the teams' scoring and pace, lineups, and anything not listed here."
    if sp.outdoor and roof_unknown(venue_map, conditions_at_odds):
        evidence += (" The retractable roof's state wasn't announced when the odds were fetched, so no weather effect "
                     "is counted: any is in here too.")
    elif sp.outdoor and conditions_at_odds is None and not is_covered(venue_map):
        evidence += " The forecast at the odds' time is unknown, so any weather effect is in here too."
    remainder = total - base - math.fsum(f.delta for f in factors)
    factors.append(Factor("other", "Teams and other", remainder, sp.unit, evidence=evidence, source=SRC_MARKET,
                          as_of=as_of, confidence=NORMAL))
    return Projection("League average total", base, factors, total, sp.unit, "total", "Total")


# --- adjustment --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Adjustment:
    """What moves a number: ``raw`` (delta, the effect now minus the effect at the odds' time), ``weight`` times it,
    clamped to +-``cap``, is ``applied``. ``full``: no market to anchor to, so the raw change applies in full
    (weight 1, no cap). ``reason`` is plain words."""

    kind: str  # "total" | "side" | "win"
    raw: float
    applied: float
    cap: Optional[float]
    weight: float
    capped: bool
    full: bool
    reason: str
    unit: str = ""
    sport: str = ""
    effect_now: float = 0.0
    effect_at_odds: float = 0.0

    def to_dict(self, digits: Optional[int] = 3) -> Dict[str, Any]:
        def r(value: Optional[float]) -> Optional[float]:
            return value if value is None or digits is None else round(value, digits)

        return {"kind": self.kind, "raw": r(self.raw), "applied": r(self.applied), "cap": r(self.cap),
                "weight": self.weight, "capped": self.capped, "full": self.full, "reason": self.reason,
                "unit": self.unit, "sport": self.sport, "effect_now": r(self.effect_now),
                "effect_at_odds": r(self.effect_at_odds)}


def _clock(at: Any, tz: Optional[str]) -> str:
    """"14:05 UTC", or "10:05 EDT" on ``tz``'s clock; "" for something that isn't a time."""
    try:
        when = _when(at)
    except ValueError:
        return ""
    try:
        local = when.astimezone(_zone(tz)) if tz else None
    except ValueError:  # an unknown zone: UTC rather than no time at all
        local = None
    if local is not None:
        return f"{local:%H:%M} {local.tzname() or ''}".strip()
    return f"{when:%H:%M} UTC"


def adjustment_reason(cause: str = "forecast", *, odds_at: Any = None, change: Optional[str] = None,
                      full: bool = False, tz: Optional[str] = None) -> str:
    """Plain words: "The forecast changed after the odds were fetched at 14:05 UTC: wind 8 → 18 mph." ``cause``:
    "forecast", "roof_closed" or "roof_open"; ``tz`` names the clock for ``odds_at`` (default UTC)."""
    if cause not in CAUSES:
        raise ValueError(f"Unknown adjustment cause {cause!r}")
    clock = _clock(odds_at, tz) if odds_at is not None else ""
    at = f" at {clock}" if clock else ""
    if cause == "forecast":
        text = f"The forecast changed after the odds were fetched{at}"
        text += f": {change}." if change else "."
    else:
        state = "closed" if cause == "roof_closed" else "open"
        text = f"The roof was announced {state} after the odds were fetched{at}"
        text += f" ({change})." if change else "."
    if full:
        text += " There is no market line to anchor to, so the full change applies."
    return text


def adjustment(sport: Any, kind: str, effect_now: float, effect_at_odds: Optional[float], *, has_market: bool = True,
               weight: float = ADJUST_WEIGHT, reason: Optional[str] = None, cause: str = "forecast",
               odds_at: Any = None, change: Optional[str] = None, tz: Optional[str] = None) -> Optional[Adjustment]:
    """The adjustment for a context effect that changed since the odds were fetched, or None.

    None when ``effect_at_odds`` is None (no forecast from the odds' time to compare with) or the change rounds to
    0 (below ``MIN_DELTA``). Otherwise ``raw = effect_now - effect_at_odds`` and ``applied = clamp(weight x raw,
    -cap, +cap)`` with the sport's cap for ``kind`` ("total", "side": the sport's unit; "win": points of chance).
    Without a market (``has_market=False``) the raw change applies in full: weight 1, no cap, ``full`` True.
    ``reason`` defaults to ``adjustment_reason(cause, odds_at=..., change=..., full=..., tz=...)``.
    """
    sp = params(sport)
    if kind not in KINDS:
        raise ValueError(f"Unknown adjustment kind {kind!r}")
    if effect_at_odds is None:
        return None
    now = _finite(effect_now, "effect_now")
    then = _finite(effect_at_odds, "effect_at_odds")
    raw = now - then
    if abs(raw) < MIN_DELTA:
        return None
    unit = "pp" if kind == "win" else sp.unit
    if not has_market:
        applied, cap, used, capped, full = raw, None, 1.0, False, True
    else:
        used = _finite(weight, "weight")
        if not 0.0 <= used <= 1.0:
            raise ValueError(f"weight must be between 0 and 1, got {weight!r}")
        cap = float(sp.caps[kind])
        wanted = used * raw
        applied = max(-cap, min(cap, wanted))
        capped, full = abs(wanted) > cap, False
    text = reason if reason else adjustment_reason(cause, odds_at=odds_at, change=change, full=full, tz=tz)
    return Adjustment(kind=kind, raw=raw, applied=applied, cap=cap, weight=used, capped=capped, full=full,
                      reason=text, unit=unit, sport=sp.key, effect_now=now, effect_at_odds=then)


def weather_effects(sport: Any, venue: Optional[Mapping[str, Any]], conditions_now: Optional[Mapping[str, Any]],
                    conditions_at_odds: Optional[Mapping[str, Any]], *,
                    market_total: Optional[float] = None) -> Tuple[float, Optional[float]]:
    """(effect now, effect at the odds' time) of the weather on the total, on one base, ready for ``adjustment``.
    The second is None when the forecast at the odds' time is unknown. Soccer's base is the total before weather:
    ``market_total`` (default the league average) over the multiplier at the odds' time."""
    sp = params(sport)
    base: Optional[float] = None
    if sp.key == "soccer":
        anchor = sp.league_total if market_total is None else _finite(market_total, "market_total")
        then = conditions_at_odds if conditions_at_odds is not None else conditions_now
        base = anchor / weather_multiplier(sp.key, venue, then)
    now = weather_total_effect(sp.key, venue, conditions_now, base_total=base)
    if conditions_at_odds is None:
        return now, None
    return now, weather_total_effect(sp.key, venue, conditions_at_odds, base_total=base)


def weather_adjustment(sport: Any, venue: Optional[Mapping[str, Any]], conditions_now: Optional[Mapping[str, Any]],
                       conditions_at_odds: Optional[Mapping[str, Any]], *, market_total: Optional[float] = None,
                       has_market: bool = True, odds_at: Any = None, tz: Optional[str] = None,
                       units: Optional[str] = None, weight: float = ADJUST_WEIGHT) -> Optional[Adjustment]:
    """The total's adjustment for weather or a roof that changed since the odds were fetched (None when nothing
    changed, the forecast at the odds' time is unknown, or a retractable roof's state wasn't announced then or isn't
    now): ``weather_effects`` into ``adjustment``, with the cause ("roof_closed" / "roof_open" when an announced state
    flipped, else "forecast") and the change in plain words."""
    now, then = weather_effects(sport, venue, conditions_now, conditions_at_odds, market_total=market_total)
    if then is None:
        return None
    before_state, now_state = roof_state(venue, conditions_at_odds), roof_state(venue, conditions_now)
    if before_state is None or now_state is None:
        # A retractable roof whose state wasn't known at the odds' time (or isn't now): what the market assumed
        # can't be said, so neither a forecast change nor an announcement moves a number
        return None
    covered_before, covered_now = before_state in ("dome", "closed"), now_state in ("dome", "closed")
    if covered_now and not covered_before:
        cause, change = "roof_closed", None
    elif covered_before and not covered_now:
        cause, change = "roof_open", None
    else:
        cause, change = "forecast", weather_change(sport, conditions_at_odds, conditions_now, units=units) or None
    return adjustment(sport, "total", now, then, has_market=has_market, weight=weight, cause=cause, odds_at=odds_at,
                      change=change, tz=tz)


def _soccer_over(share: float, total: float, line: float) -> float:
    return soccer.over_prob(soccer.grid(share * total, (1 - share) * total), line)


def adjusted_over(sport: Any, line: float, p_over: float, applied: float, *,
                  rates: Optional[Tuple[float, float]] = None) -> float:
    """The chance of the over after moving the expected total by ``applied`` (in the sport's unit).

    Normal sports: ``mu = line + sd x inverse-normal(p_over)``, ``mu' = mu + applied``,
    ``p' = 1 - normal((line - mu') / sd)`` with the sport's total SD. Soccer: the goals model's grid, with the
    home/away split of ``rates`` (default the model's 1.557 : 1.245), is first scaled to the total that gives
    ``p_over`` at ``line``, then to that total plus ``applied``. ``applied`` 0 returns ``p_over`` unchanged.
    """
    sp = params(sport)
    point = _finite(line, "line")
    shift = _finite(applied, "applied")
    p = _chance(p_over, "p_over")
    if shift == 0:
        return float(p_over)
    if sp.key != "soccer":
        mu = point + sp.total_sd * _N.inv_cdf(p)
        moved = mu + shift
        return 1 - _N.cdf((point - moved) / sp.total_sd)
    share = SOCCER_HOME_SHARE
    if rates is not None:
        home, away = (_finite(rates[0], "rates[0]"), _finite(rates[1], "rates[1]"))
        if home <= 0 or away <= 0:
            raise ValueError(f"Goal rates must be positive, got {rates!r}")
        share = home / (home + away)
    low, high = 2 * _RATE_MIN, 15.0
    for _ in range(60):
        middle = (low + high) / 2
        if _soccer_over(share, middle, point) < p:
            low = middle
        else:
            high = middle
    start = (low + high) / 2
    return _soccer_over(share, max(start + shift, 2 * _RATE_MIN), point)


# --- badges ------------------------------------------------------------------------------------------------------

def _adjusts(adjustment: Any) -> bool:
    """True when any adjustment given (an Adjustment, its dict, or a list of them) moves a number."""
    if adjustment is None:
        return False
    if isinstance(adjustment, Adjustment):
        return adjustment.applied != 0
    if isinstance(adjustment, Mapping):
        value = _num(adjustment.get("applied"))
        return bool(value) if value is not None else bool(adjustment)
    if isinstance(adjustment, Iterable) and not isinstance(adjustment, (str, bytes)):
        return any(_adjusts(item) for item in adjustment)
    return bool(adjustment)


def badges(sport: Any, venue: Optional[Mapping[str, Any]], conditions: Optional[Mapping[str, Any]],
           trips: Optional[Mapping[str, Mapping[str, Any]]], adjustment: Any = None, *,
           limit: int = BADGE_LIMIT) -> List[str]:
    """Compact context codes, most important first, at most ``limit`` (3):

    ``adj`` (an adjustment applies), ``wind:<km/h>`` (open-air or a roof announced open, 24+ km/h at the start), ``rain`` (60%+ chance and
    0.5+ mm/h), ``cold:<C>`` (0 C or below, open-air), ``hot:<C>`` (30 C or above, open-air), ``roof`` (a
    retractable roof announced closed), ``alt:<m>`` (1,500 m or higher), ``tz-home:<h>`` / ``tz-away:<h>`` (2 h+
    east within 3 days), ``rest-home:<days>`` / ``rest-away:<days>`` (NFL: 4 days or fewer since the last game).
    The slate's own ``live``, ``neutral`` and ``b2b-home`` / ``b2b-away`` are not made here.
    """
    key = sport_of(sport)
    sp = SPORTS.get(key) if key else None
    venue_map = _map(venue)
    found: Dict[str, str] = {}
    if _adjusts(adjustment):
        found["adj"] = "adj"
    state = roof_state(venue_map, conditions)
    if sp is not None and sp.outdoor and conditions is not None and state == "open":
        w = read_conditions(key, conditions)
        if w.wind_kmh is not None and w.wind_kmh >= BADGE_WIND_KMH:
            found["wind"] = f"wind:{_fmt(round(w.wind_kmh))}"
        if is_rain(w):
            found["rain"] = "rain"
        temp_c = w.temp_c
        if temp_c is not None and temp_c <= BADGE_COLD_C:
            found["cold"] = f"cold:{_fmt(round(temp_c))}"
        elif temp_c is not None and temp_c >= BADGE_HOT_C:
            found["hot"] = f"hot:{_fmt(round(temp_c))}"
    if state == "closed" and str(venue_map.get("roof") or "").lower() == "retractable":
        found["roof"] = "roof"
    elevation = _num(venue_map.get("elevation_m"))
    if elevation is not None and elevation >= ALTITUDE_M:
        found["alt"] = f"alt:{_fmt(round(elevation))}"
    for side in ("home", "away"):
        trip = _team(trips, side)
        shift = jet_lag_hours(trip)
        if shift is not None:
            found[f"tz-{side}"] = f"tz-{side}:{_fmt(shift)}"
        if key == "nfl":
            days = days_since_last(trip)
            if days is not None and 0 < days <= NFL_SHORT_REST_DAYS:
                found[f"rest-{side}"] = f"rest-{side}:{days}"
    ordered = [found[name] for name in BADGE_ORDER if name in found]
    return ordered[:max(int(limit), 0)]
