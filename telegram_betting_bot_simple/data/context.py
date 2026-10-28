"""Venue & conditions for every game (v3 Stage 4): the ContextService.

Context explains the line; only newer information moves it (report, "Venues, weather and news explain the line; only
newer information moves it"). For each game it puts together, in the background, one context block: where the game
is played (data/venues.py), the weather at the start (Open-Meteo, data/weather.py; MLB's own announcement,
data/mlb_stats.py), each team's rest, travel and time-zone shift (from the cached scoreboards, through the slate), the
league's home edge (from the cached standings), "Why the line is here" (engine/context.py's attribution of the
market's margin and total) and "What moves it" (an adjustment, only for a forecast or a roof that changed after the
odds were fetched). The block is a computed entry in the shared cache, ``("game_context", league, game_id)`` (kind
"game_context": 15 minutes, 15 more stale), so the bot and the dashboard read the same one; ``game_id`` is the odds
feed's id when the game has one (as /api/game names it), else ``"espn:<ESPN id>"``.

- ``warm(games)`` runs only in the background: the slate's worker hands it the slate's games each round. It covers
  games starting within 48 hours or live, plus any game a reader asked for (``for_game`` queues it). Fetching is
  bounded to games with an odds board (or asked for): their venues' forecasts (one Open-Meteo call per 50 venues, the
  whole week in one cached copy per venue, refreshed hourly and every 15 minutes within 6 hours of the start), MLB's
  schedule for MLB games, the past scoreboards a lookback needs (immutable once past) and the league's standings.
  Other games get a block from what is already held (the venue table, the slate's own boards, the league's prior).
  Each forecast read is logged with its fetch time (store table ``weather_log``) when it changed, so the forecast as
  it stood when the odds were fetched can be found later; each game with a total line is logged to ``context_log``
  (the chance of the over before and after any adjustment), frozen at the start and graded after the final.
- ``for_game(league, game_id)`` reads the cache only: never a fetch on the request path. Not warmed yet, it answers
  ``{"status": "loading"}`` and queues the game; in the process with the worker that brings the next round forward
  (SlateBuilder.request_context), so the page's one re-check 20 seconds on finds the block. A process without the
  worker (the bot's) reads the blocks the dashboard writes from the shared table, again once its copy is READ_AGAIN
  seconds old.
- ``badges_for(slate_game)`` gives the block's compact badge codes for the slate's row ([] when there is none).
- ``on_adjust(league, game_id)`` (set by DashboardService) hears of every game whose adjustment to its total appeared,
  changed or went, so the cards holding the totals' chances are rebuilt at once; ``adjust_mark()`` signs them all for
  the entries that hold many games' chances (the Builder's).

The forecast at the odds' time is the latest ``weather_log`` row fetched at or before the board's ``odds_as_of``;
without one there is no adjustment, only attribution with "forecast at odds time unknown". A roof announced closed
after the odds removes the at-odds weather effect (the change is 0 minus that effect), but only when the roof's state
was known (announced) at the odds' time: a retractable roof whose state isn't announced (every NFL roof; MLB's before
its announcement) counts no weather, gets no weather badge and is never adjusted, and its line says "Retractable roof:
state not announced". ESPN's ``indoor`` flag is no state (it marks every retractable park indoor whatever the day).
Unknown venues are placed at their city's geocode in the address's state (ESPN's "AR": Fayetteville, Arkansas).

Text the server writes into entries shared by everyone comes in both unit systems: an adjustment's ``reasons``, the
leg's adjusted note's ``reasons``, the Builder's ``why_units`` and each factor's ``evidence_units``; the app shows the
person's Units.

Module helpers used by the games service, the Lab, the AI tool and the bot: ``total_adjustment(block)``,
``adjusted_chance(league, adjustment, side, point, p)``, ``why_line(block, market)``, ``venue_line(block, units)``,
``compact(block)`` and ``units_for(league, units)``. Text the server writes is plain; the app escapes it.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import logging
import math
import statistics
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..engine import context as engine
from ..engine.factors import NOISY, Factor, Projection, fold
from ..learn import registry
from .cache import ttl_for

log = logging.getLogger(__name__)

KIND = "game_context"
WINDOW = 48 * 3600  # games starting within this (or live) are warmed every round
FORECAST_DAYS = 7  # Open-Meteo's week: a game further out has no weather
HOUR = 3600
HOURS_BEFORE, HOURS_AFTER = 1, 3  # the block's hourly strip around the start
QUEUE_MAX = 300  # games asked for and waiting for a round
SEEN_MAX = 3000  # the last slate copy of each game (a queued game is warmed from it)
REWRITE_AFTER = 7.5 * 60  # an unchanged block is written again only this long after it was (its entry lives 15 min)
# A reader of blocks the other process writes (the bot) checks the shared table again once its copy is this old
READ_AGAIN = 60.0
PRUNE_AFTER = 14 * 24 * 3600  # forecast rows of games that started longer ago than this are dropped (daily)
# Leagues shown in imperial units when the person's setting is "auto" (the app's US leagues)
US_LEAGUES = frozenset({"americanfootball_nfl", "americanfootball_ncaaf", "basketball_nba", "basketball_wnba",
                        "basketball_ncaab", "icehockey_nhl", "baseball_mlb", "soccer_usa_mls"})
UNITS = ("auto", "metric", "imperial")
SHOWN_UNITS = ("metric", "imperial")  # what "auto" resolves to: text the server writes comes in both
# Sports whose spread is a point spread priced near even money: its line is the market's margin. MLB's run line and
# the NHL's puck line are fixed at 1.5 and soccer's handicaps carry the draw, so those use the blend's chance instead
POINT_SPREAD_SPORTS = frozenset({"nfl", "ncaaf", "nba", "wnba", "ncaab"})
# The smallest factor worth a Why line, by unit
WHY_MIN = {"pts": 0.5, "runs": 0.1, "goals": 0.05}
FIELD_WORDS = {"LF": "LF", "CF": "CF", "RF": "RF"}
ATTRIBUTION = {"text": "Weather data by Open-Meteo.com", "url": "https://open-meteo.com/"}


# --- small helpers ----------------------------------------------------------------------------------------------

def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, datetime.datetime):
        return (value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)).timestamp()
    if isinstance(value, str) and value.strip():
        try:
            moment = datetime.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return (moment if moment.tzinfo else moment.replace(tzinfo=datetime.timezone.utc)).timestamp()
    return None


def _sig(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()[:20]


def _r(value: Optional[float], digits: int = 1) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def context_key(league: str, game_id: Any) -> Tuple[str, str, str]:
    """The computed entry of a game's context block."""
    return (KIND, str(league), str(game_id))


def game_id_of(game: Mapping[str, Any]) -> Optional[str]:
    """A slate game's context id: the odds feed's id when it has one (as /api/game names the game), else
    "espn:<ESPN id>" (from its own fields or, trimmed, from the slate id)."""
    league, gid = str(game.get("league") or ""), str(game.get("id") or "")
    rest = gid[len(league) + 1:] if gid.startswith(league + ":") else ""
    odds_id = game.get("odds_id") or (rest if rest and not rest.startswith("espn:") else None)
    if odds_id:
        return str(odds_id)
    espn_id = game.get("espn_id") or (rest[5:] if rest.startswith("espn:") else None)
    return f"espn:{espn_id}" if espn_id else None


def units_for(league: Optional[str], units: Optional[str] = "auto") -> str:
    """"metric" or "imperial": the person's setting, or for "auto" imperial in the app's US leagues and metric
    otherwise."""
    if units in ("metric", "imperial"):
        return str(units)
    return "imperial" if league in US_LEAGUES else "metric"


def _wind(kmh: Optional[float], units: str) -> Optional[str]:
    if kmh is None:
        return None
    return f"{engine.kmh_to_mph(kmh):.0f} mph" if units == "imperial" else f"{kmh:.0f} km/h"


def _temp(celsius: Optional[float], units: str) -> Optional[str]:
    if celsius is None:
        return None
    return f"{engine.c_to_f(celsius):.0f}°F" if units == "imperial" else f"{celsius:.0f}°C"


def _height(metres: Optional[float], units: str) -> Optional[str]:
    if metres is None:
        return None
    return f"{metres / 0.3048:,.0f} ft" if units == "imperial" else f"{metres:,.0f} m"


def _signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def _osm(lat: float, lon: float) -> str:
    return f"https://www.openstreetmap.org/?mlat={lat:.4f}&mlon={lon:.4f}#map=16/{lat:.4f}/{lon:.4f}"


# --- what the block says, in words -----------------------------------------------------------------------------

def _wind_words(weather: engine.Weather, units: str) -> Optional[str]:
    """"26 km/h out to RF", "16 mph", "calm": MLB's announced direction when it gave one."""
    if weather.wind_kind == "calm":
        return "calm"
    speed = _wind(weather.wind_kmh, units)
    if speed is None:
        return None
    if weather.wind_kind == "out":
        return f"{speed} out to {FIELD_WORDS.get(weather.wind_field or '', 'CF')}"
    if weather.wind_kind == "in":
        return f"{speed} in from {FIELD_WORDS.get(weather.wind_field or '', 'CF')}"
    if weather.wind_kind == "cross":
        return f"{speed} across the field"
    return f"{speed} wind"


def conditions_line(sport: str, venue: Mapping[str, Any], conditions: Optional[Mapping[str, Any]],
                    effect: Optional[float], units: str) -> Optional[str]:
    """One line of conditions: "26 km/h out to RF · 27°C · +0.3 runs", "Roof closed", "Dome", "Retractable roof:
    state not announced"; None when there's nothing to say."""
    if not engine.SPORTS[sport].outdoor:
        return None
    state = engine.roof_state(venue, conditions)
    if state == "dome":
        return "Dome"
    if state == "closed":
        return "Roof closed"
    if state is None:
        return "Retractable roof: state not announced"
    if conditions is None:
        return None
    weather = engine.read_conditions(sport, conditions)
    parts = [part for part in (_wind_words(weather, units), _temp(weather.temp_c, units)) if part]
    if engine.is_rain(weather):
        parts.append("rain likely")
    sp = engine.SPORTS[sport]
    if effect is not None and abs(effect) >= (0.1 if sp.unit == "pts" else 0.05):
        parts.append(f"{_signed(effect, 1 if sp.unit == 'pts' else 2)} {sp.unit}")
    return " · ".join(parts) or None


def venue_line(block: Optional[Mapping[str, Any]], units: str = "metric") -> Optional[str]:
    """The bot's one context line (plain text): "Coors Field · 5,184 ft · 16 mph out to RF · 81°F" (the venue, its
    elevation from 1,000 m, the conditions); None without a ready block."""
    if not block or block.get("status") != "ready":
        return None
    venue = block.get("venue") or {}
    if not venue.get("name"):
        return None
    parts = [str(venue["name"])]
    elevation = _num(venue.get("elevation_m"))
    if elevation is not None and elevation >= 1000:
        parts.append(_height(elevation, units) or "")
    line = (block.get("line") or {}).get(units)
    if line:
        parts.append(str(line))
    return " · ".join(part for part in parts if part)


def total_adjustment(block: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The block's adjustment to the total that moves a number (kind "total", applied not 0), or None."""
    if not block or block.get("status") != "ready":
        return None
    for adjustment in block.get("adjustments") or []:
        if isinstance(adjustment, Mapping) and adjustment.get("kind") == "total" and _num(adjustment.get("applied")):
            return dict(adjustment)
    return None


def brain_params(league: Any, store: Any = None) -> "registry.Params":
    """The learning loop's live values for a league's sport (learn/registry.py): ADJUST_WEIGHT and the sport's
    priors and SDs as tuned, today's constants (version 0) until a change passes the gate. Read from ``store`` when
    it is a real Store, else from the registry's bound one."""
    return registry.current(league, store if hasattr(store, "_connect") and getattr(store, "path", None) else None)


def engine_sport(league: Any, store: Any = None) -> Any:
    """What the engine's functions take as ``sport`` for a league: its context key ("nfl"), or, once the learning
    loop has tuned the sport's priors or SDs, a SportParams with them (engine.context.sport_params). None for
    leagues with no context parameters."""
    sport = engine.sport_of(league)
    if sport is None:
        return None
    tuned = brain_params(league, store)
    overrides = tuned.context_overrides()
    return engine.sport_params(sport, overrides) if overrides else sport


def adjusted_chance(league: str, adjustment: Optional[Mapping[str, Any]], side: str, point: Any,
                    p: Optional[float]) -> Optional[float]:
    """A totals leg's chance after the adjustment: engine.adjusted_over for the over at its own line (the under is
    1 minus the over moved from 1 - p); ``p`` unchanged without an adjustment or for any other leg. The sport's
    total SD is the learning loop's live one (engine_sport)."""
    applied = _num((adjustment or {}).get("applied"))
    line = _num(point)
    if not applied or line is None or p is None or side not in ("over", "under") or not 0 < p < 1:
        return p
    sport = engine_sport(league)
    if sport is None:
        return p
    if side == "over":
        return engine.adjusted_over(sport, line, p, applied)
    return 1.0 - engine.adjusted_over(sport, line, 1.0 - p, applied)


def adjusted_note(adjustment: Mapping[str, Any]) -> Dict[str, Any]:
    """What a leg carries when an adjustment moved its chance: {applied, reason} (plus the unit, whether the full
    change applied and ``reasons``: the reason in each unit system, {"metric", "imperial"}, so the app shows the
    person's while the entries holding it stay the same for everyone)."""
    note = {"applied": _r(_num(adjustment.get("applied")), 3), "reason": adjustment.get("reason"),
            "unit": adjustment.get("unit"), "full": bool(adjustment.get("full"))}
    reasons = adjustment.get("reasons")
    if isinstance(reasons, Mapping) and all(isinstance(reasons.get(name), str) for name in SHOWN_UNITS):
        note["reasons"] = {name: reasons[name] for name in SHOWN_UNITS}
    return note


def note_reason(note: Optional[Mapping[str, Any]], units: str) -> Optional[str]:
    """An adjusted note's reason in ``units`` ("metric" / "imperial"), else its own."""
    if not isinstance(note, Mapping):
        return None
    reasons = note.get("reasons")
    if isinstance(reasons, Mapping) and isinstance(reasons.get(units), str):
        return str(reasons[units])
    reason = note.get("reason")
    return str(reason) if isinstance(reason, str) else None


def _detail(factor: Mapping[str, Any], block: Mapping[str, Any], units: str) -> str:
    """The condition behind a weather factor of the attribution, for its Why line: "Wind 26 km/h at kick-off". The
    attribution holds the forecast from when the odds were fetched (the block's ``weather.at_odds.start``), so that
    is the condition named; when the forecast has changed since, the line says so ("… as forecast when the odds
    were fetched")."""
    weather = block.get("weather") if isinstance(block.get("weather"), Mapping) else {}
    at_odds = weather.get("at_odds") if isinstance(weather.get("at_odds"), Mapping) else {}
    start = at_odds.get("start") if isinstance(at_odds.get("start"), Mapping) else {}
    key, label = str(factor.get("key") or ""), str(factor.get("label") or "")
    changed = at_odds.get("changed") if isinstance(at_odds.get("changed"), list) else []
    then = " as forecast when the odds were fetched"
    if key in ("wind", "cold_wind") and _num(start.get("wind_kmh")) is not None:
        speed = _wind(_num(start.get("wind_kmh")), units)
        text = f"Wind {speed} at kick-off" if label == "Wind at kick-off" else f"{label} ({speed})"
        return text + (then if "wind" in changed else "")
    if key in ("temperature", "cold", "heat") and _num(start.get("temp_c")) is not None:
        return f"{label} ({_temp(_num(start.get('temp_c')), units)})" + (then if "temp" in changed else "")
    return label


def why_line(block: Optional[Mapping[str, Any]], market: str, units: Optional[str] = None) -> Optional[str]:
    """The Builder's context line for a leg of ``market``: the top factor of the block's attribution (the total's for
    totals, the margin's for the rest; never the remainder or the home edge every game has), e.g. "Wind 26 km/h at
    kick-off: the line already has about −2.2 pts for it (small, noisy)". None when no factor is big enough.
    ``units``: "metric" or "imperial" (None: the league's "auto"); ``why_lines`` gives both."""
    if not block or block.get("status") != "ready":
        return None
    attribution = block.get("attribution") or {}
    projection = attribution.get("total" if market == "totals" else "margin")
    if not isinstance(projection, Mapping):
        return None
    unit = str(projection.get("unit") or "")
    floor = WHY_MIN.get(unit, 0.5)
    factors = [f for f in projection.get("factors") or [] if isinstance(f, Mapping)
               and f.get("key") not in ("other", "home_field", "folded") and abs(_num(f.get("delta")) or 0) >= floor]
    if not factors:
        return None
    top = max(factors, key=lambda f: abs(float(f["delta"])))
    league = block.get("league")
    shown = units or units_for(league if isinstance(league, str) else None)
    digits = 1 if unit == "pts" else 2
    text = (f"{_detail(top, block, shown)}: the line already has about {_signed(float(top['delta']), digits)} {unit} "
            "for it")
    if top.get("confidence") == NOISY:
        text += " (small, noisy)"
    return text


def why_lines(block: Optional[Mapping[str, Any]], market: str) -> Dict[str, Optional[str]]:
    """``why_line`` in both unit systems: {"metric": …, "imperial": …} (entries shared by everyone hold both; the
    app shows the person's)."""
    return {name: why_line(block, market, name) for name in SHOWN_UNITS}


def projection_of(shown: Mapping[str, Any]) -> Projection:
    """A Projection back from its to_dict form (for engine.factors.fold)."""
    factors = [Factor(str(f.get("key") or ""), str(f.get("label") or ""), float(f.get("delta") or 0.0),
                      str(f.get("unit") or ""), "", str(f.get("evidence") or ""), str(f.get("source") or ""),
                      f.get("as_of"), f.get("confidence") if f.get("confidence") in ("normal", NOISY) else "normal")
               for f in shown.get("factors") or [] if isinstance(f, Mapping)]
    return Projection(str(shown.get("baseline_label") or ""), float(shown.get("baseline") or 0.0), factors,
                      float(shown.get("value") or 0.0), str(shown.get("unit") or ""), str(shown.get("key") or ""),
                      str(shown.get("label") or ""))


def compact(block: Optional[Mapping[str, Any]], factors: int = 6) -> Dict[str, Any]:
    """The game_context tool's view of a block: the venue, the conditions at the start (both units), each team's rest
    and travel, the top ``factors`` attribution factors per projection (the rest folded, so they still add up) and
    the adjustments."""
    if not block or block.get("status") != "ready":
        return {"status": (block or {}).get("status") or "loading"}
    venue = block.get("venue") or {}
    out: Dict[str, Any] = {
        "status": "ready", "as_of": block.get("as_of"),
        "venue": {key: venue.get(key) for key in ("name", "city", "country", "elevation_m", "roof", "roof_state",
                                                  "surface", "capacity", "park", "approx") if venue.get(key) is not None},
    }
    weather = block.get("weather")
    if isinstance(weather, Mapping):
        start = weather.get("start") or {}
        conditions = {key: start.get(key) for key in ("temp_c", "wind_kmh", "gust_kmh", "precip_prob", "precip_mm")
                      if start.get(key) is not None}
        if _num(start.get("temp_c")) is not None:
            conditions["temp_f"] = round(engine.c_to_f(float(start["temp_c"])))
        if _num(start.get("wind_kmh")) is not None:
            conditions["wind_mph"] = round(engine.kmh_to_mph(float(start["wind_kmh"])))
        out["conditions"] = {"at_start": conditions, "announced": weather.get("announced"),
                             "effect_on_total": weather.get("effect"), "fetched_at": weather.get("fetched_at"),
                             "source": weather.get("source")}
    out["line"] = block.get("line")
    out["teams"] = {side: {key: team.get(key) for key in ("name", "rest_days", "back_to_back", "travel_km",
                                                           "tz_shift_h", "last") if team.get(key) is not None}
                    for side, team in (block.get("teams") or {}).items() if isinstance(team, Mapping)}
    attribution = {}
    for name, shown in (block.get("attribution") or {}).items():
        if isinstance(shown, Mapping):
            folded = fold(projection_of(shown), factors, digits=2)
            attribution[name] = {
                "label": folded["label"], "baseline_label": folded["baseline_label"], "baseline": folded["baseline"],
                "value": folded["value"], "unit": folded["unit"],
                "factors": [{"label": f["label"], "delta": f["delta"], "confidence": f["confidence"],
                             "evidence": f["evidence"]} for f in folded["factors"]]}
    out["attribution"] = attribution
    out["adjustments"] = [{key: adjustment.get(key) for key in ("kind", "raw", "weight", "cap", "applied", "capped",
                                                                 "full", "reason", "unit", "p_over")}
                          for adjustment in block.get("adjustments") or [] if isinstance(adjustment, Mapping)]
    out["badges"] = list(block.get("badges") or [])
    return out


# --- the service ------------------------------------------------------------------------------------------------

class ContextService:
    """Warms each game's context block in the background and serves it from the cache (see the module's docstring).

    ``venues`` is data/venues.py's VenueTable; ``weather`` the WeatherClient (None when WEATHER_ENABLED is off: no
    forecasts, no geocoding); ``mlb`` the MlbStats client (or None); ``store`` the shared Store (weather_log,
    context_log); ``cache`` the process's TTLCache; ``slate`` the SlateBuilder, for past scoreboards (``scoreboards``);
    ``espn`` the EspnClient, for standings (default the slate's)."""

    def __init__(self, venues: Any, weather: Any, mlb: Any, store: Any, cache: Any, slate: Any = None,
                 now: Callable[[], float] = time.time, *, espn: Any = None, window: float = WINDOW) -> None:
        self.venues = venues
        self.weather = weather
        self.mlb = mlb
        self.store = store
        self.cache = cache
        self.slate = slate
        self.espn = espn
        self._now = now
        self.window = float(window)
        self._lock = threading.Lock()
        self._warming = threading.Lock()
        self._queued: "OrderedDict[Tuple[str, str], None]" = OrderedDict()
        self._asked: set = set()  # queued games a reader is waiting for (they bring the round forward)
        self._seen: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()
        self._blocks: "OrderedDict[Tuple[str, str], Tuple[float, str, Dict[str, Any]]]" = OrderedDict()
        self._logged_context: Dict[Tuple[str, str], str] = {}
        self._approx: Dict[Tuple[str, str, str, str], Tuple[float, Optional[Dict[str, Any]]]] = {}
        self._pruned_at = -math.inf
        self.stats = {"warms": 0, "blocks": 0, "written": 0, "weather_rows": 0, "context_rows": 0, "errors": 0}
        # Called with (league, game id) when a game's adjustment to its total appears, changes or goes, so the
        # computed entries holding the totals' chances (the game's card, the league's cards) are rebuilt at once;
        # DashboardService sets it when it is handed this service
        self.on_adjust: Optional[Callable[[str, str], Any]] = None
        self._adjusted: Dict[Tuple[str, str], str] = {}  # game -> its total adjustment's signature
        self._mark = ""

    # --- reading (the request path: never a fetch) ---

    def _policy(self) -> Tuple[float, float]:
        policy = ttl_for(KIND)
        return policy.ttl, policy.stale

    def _clock(self) -> float:
        now = getattr(self.cache, "now", None)
        return float(now()) if callable(now) else self._now()

    def _held(self, league: str, game_id: str, shared: bool = True) -> Optional[Dict[str, Any]]:
        """The game's block: this process's own last one (the process that warms writes every block, so its copy is
        the newest), else the cache's (within its TTL and stale window). A process that doesn't warm (the bot) reads
        the shared table again once its copy is READ_AGAIN seconds old, so it follows the dashboard's rounds."""
        ttl, stale = self._policy()
        with self._lock:
            mine = self._blocks.get((league, str(game_id)))
        if mine is not None and self._now() - mine[0] <= ttl + stale:
            return mine[2]
        key = context_key(league, game_id)
        found = None
        peek = getattr(self.cache, "peek", None)
        if callable(peek):
            try:
                found = peek(key, shared=shared, recheck=READ_AGAIN if shared else None)
            except TypeError:  # a cache without recheck
                found = peek(key, shared=shared)
            except Exception:
                found = None
        if found is not None and self._clock() - float(found[0]) <= ttl + stale and isinstance(found[1], dict):
            return found[1]
        return None

    def held(self, league: str, game_id: Any) -> Optional[Dict[str, Any]]:
        """The game's block as held (read-only: not a copy, nothing queued), or None. Never fetches."""
        if engine.sport_of(league) is None or game_id in (None, ""):
            return None
        return self._held(league, str(game_id))

    def for_game(self, league: str, game_id: Any) -> Dict[str, Any]:
        """The game's context block from the cache, or {"status": "loading"} (the game is then queued for the next
        round); {"status": "none"} for a sport with no context. Never fetches."""
        if engine.sport_of(league) is None:
            return {"status": "none"}
        game_id = str(game_id)
        block = self._held(league, game_id)
        if block is not None:
            return copy.deepcopy(block)
        self.queue(league, game_id, soon=True)
        return {"status": "loading"}

    def queue(self, league: str, game_id: Any, soon: bool = False) -> None:
        """Ask for a game's block at the next round (whatever its start). ``soon`` (a reader is waiting for it): bring
        that round forward (the slate's request_context; nothing in a process without the worker, such as the bot's)."""
        key = (league, str(game_id))
        with self._lock:
            new = soon and key not in self._asked and len(self._asked) < QUEUE_MAX
            if new:
                self._asked.add(key)
            self._queued[key] = None
            self._queued.move_to_end(key)
            while len(self._queued) > QUEUE_MAX:
                self._queued.popitem(last=False)
        ask = getattr(self.slate, "request_context", None) if new else None
        if callable(ask):
            try:
                ask()
            except Exception:
                log.exception("Context: couldn't ask for a round")

    def pending(self) -> bool:
        """True while a game a reader asked for (``queue(soon=True)``) waits for a round; games queued by a status
        change wait for the worker's next round as usual."""
        with self._lock:
            return bool(self._asked)

    def badges_for(self, slate_game: Mapping[str, Any]) -> List[str]:
        """The context badge codes for a slate row ([] without a block): from this process's memory only, so the
        slate's round never waits on the shared table."""
        league, game_id = str(slate_game.get("league") or ""), game_id_of(slate_game)
        if not game_id or engine.sport_of(league) is None:
            return []
        block = self._held(league, game_id, shared=False)
        return list((block or {}).get("badges") or [])

    def invalidate(self, league: str, game_id: Any) -> None:
        """A game's status or score changed and its cache entry was dropped (DashboardService.invalidate_game): queue
        it for the next round, which writes it again. Until then this process serves its last block, so a scoring
        play doesn't blank the panel; the other process reads the rebuilt entry."""
        self.queue(league, game_id)

    # --- warming (the background only) ---

    def warm(self, games: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
        """Build the context blocks of the games due (see the module's docstring). Returns counts:
        {"games", "rich", "written", "weather_points"}."""
        with self._warming:
            return self._warm(list(games))

    def _warm(self, games: List[Mapping[str, Any]]) -> Dict[str, int]:
        now = self._now()
        with self._lock:
            queued = set(self._queued)
            self._queued.clear()
            self._asked.clear()
        picked: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for game in games:
            league = str(game.get("league") or "")
            game_id = game_id_of(game)
            if not game_id or engine.sport_of(league) is None:
                continue
            key = (league, game_id)
            entry = {k: v for k, v in game.items() if k not in ("_books", "_model", "_names")}
            with self._lock:
                self._seen[key] = entry
                self._seen.move_to_end(key)
                while len(self._seen) > SEEN_MAX:
                    self._seen.popitem(last=False)
            status = game.get("status")
            start = _ts(game.get("_ts") if game.get("_ts") is not None else game.get("start"))
            if status in ("final", "postponed") or start is None:
                continue
            if status == "live" or start - now <= self.window or key in queued:
                picked[key] = entry
        for key in queued - set(picked):
            with self._lock:
                entry = self._seen.get(key)
            if entry is not None and entry.get("status") in ("pre", "live"):
                picked[key] = entry
        rich = {key for key, game in picked.items() if game.get("odds_id") or key in queued}
        venues = {key: self._venue(game, key in rich) for key, game in picked.items()}
        forecasts = self._forecasts(picked, venues, rich, now)
        announced = self._announced(picked, venues, now)
        rates = self._rates({key[0] for key in rich}, now)
        league_totals = self._league_totals(games)
        boards = self._boards(picked, rich, now)
        rates.update(self._soccer_rates({key[0] for key in rich}, boards, now))
        written = 0
        for key, game in picked.items():
            try:
                block, extra = self.build(game, venues.get(key), forecast=forecasts.get(key),
                                          announced=announced.get(key), boards=boards.get(key[0]) or {},
                                          rates=rates.get(key[0]), league_total=league_totals.get(key[0]), now=now)
            except Exception:
                self.stats["errors"] += 1
                log.exception("Context: couldn't build the block of %s %s", *key)
                continue
            written += self._keep(key, block, now)
            self._note_adjustment(key, block)
            self._log(game, key, block, extra, now)
        self.stats["warms"] += 1
        self.stats["blocks"] += len(picked)
        self.stats["written"] += written
        if now - self._pruned_at >= 24 * HOUR and self.store is not None and hasattr(self.store, "prune_weather"):
            self._pruned_at = now
            try:
                self.store.prune_weather(now - PRUNE_AFTER)  # games long over: their forecasts are no longer needed
            except Exception:
                log.exception("Context: couldn't prune old forecasts")
        return {"games": len(picked), "rich": len(rich), "written": written,
                "weather_points": len({(v or {}).get("id") for k, v in venues.items() if k in forecasts})}

    def _keep(self, key: Tuple[str, str], block: Dict[str, Any], now: float) -> int:
        """Put the block in the cache when it changed or its entry is getting old; keep this process's copy."""
        body = {k: v for k, v in block.items() if k != "as_of"}
        signature = _sig(body)
        with self._lock:
            held = self._blocks.get(key)
        cache_key = context_key(*key)
        fresh = held is not None and held[1] == signature and now - held[0] < REWRITE_AFTER
        if fresh:
            peek = getattr(self.cache, "peek", None)
            fresh = callable(peek) and peek(cache_key, shared=False) is not None
        if fresh:
            return 0
        ttl, stale = self._policy()
        self.cache.put(cache_key, block, ttl=ttl, stale=stale, kind=KIND)
        with self._lock:
            self._blocks[key] = (now, signature, block)
            self._blocks.move_to_end(key)
            while len(self._blocks) > SEEN_MAX:
                self._blocks.popitem(last=False)
        return 1

    def adjust_mark(self) -> str:
        """A short signature of every game's adjustment to its total ("" when none): a key part of the entries that
        hold the totals' chances of many games at once (the Builder's legs and results)."""
        with self._lock:
            return self._mark

    def _note_adjustment(self, key: Tuple[str, str], block: Mapping[str, Any]) -> None:
        """Track the game's total adjustment; when it appeared, changed or went, update the mark and tell on_adjust."""
        moved = total_adjustment(block)
        sig = None if moved is None else _sig({k: moved.get(k) for k in ("applied", "reason", "full", "p_over")})
        with self._lock:
            if self._adjusted.get(key) == sig:
                return
            if sig is None:
                self._adjusted.pop(key, None)
            else:
                self._adjusted[key] = sig
                while len(self._adjusted) > SEEN_MAX:
                    self._adjusted.pop(next(iter(self._adjusted)))
            self._mark = _sig(sorted(self._adjusted.items())) if self._adjusted else ""
        callback = self.on_adjust
        if callback is not None and not key[1].startswith("espn:"):  # a game without odds has no card
            try:
                callback(key[0], key[1])
            except Exception:
                log.exception("Context: couldn't pass on the adjustment of %s %s", *key)

    # --- inputs ---

    def _venue(self, game: Mapping[str, Any], rich: bool) -> Optional[Dict[str, Any]]:
        """The game's venue row: by ESPN venue id, name or the home team's; an unknown venue of a game with odds is
        placed at its city's geocode (approx)."""
        league = str(game.get("league") or "")
        shown = game.get("venue") if isinstance(game.get("venue"), Mapping) else {}
        place = game.get("_place") if isinstance(game.get("_place"), Mapping) else {}
        ids = game.get("_ids") or (None, None)
        neutral = bool(game.get("_neutral"))
        found = None
        if self.venues is not None:
            try:
                found = self.venues.find(league, espn_venue_id=place.get("id"), name=shown.get("name"),
                                         home_team_id=ids[0] if ids else None, neutral=neutral)
            except Exception:
                log.exception("Context: the venue table failed for %s", shown.get("name"))
        if found is not None or not rich or self.weather is None or not shown.get("city"):
            return found
        key = (str(shown.get("name") or ""), str(shown.get("city") or ""), str(place.get("country") or ""),
               str(place.get("state") or ""))
        held = self._approx.get(key)
        # A placed venue is kept; a city the geocoder couldn't place (or a failed call) is asked about again hourly
        if held is not None and (held[1] is not None or self._now() - held[0] < HOUR):
            return held[1]
        try:
            from .venues import approx_venue
            country = place.get("country")  # ESPN's name for it ("USA", "England"): both read it as a code
            state = place.get("state")  # ESPN's "AR": picks Fayetteville, Arkansas over the bigger one in NC
            geocode = (self.weather.geocode(shown.get("city"), country, region=state) if state
                       else self.weather.geocode(shown.get("city"), country))
            row = approx_venue(shown.get("name"), shown.get("city"), country, geocode,
                               indoor=shown.get("indoor") if isinstance(shown.get("indoor"), bool) else None,
                               espn_venue_id=place.get("id"))
        except Exception:
            log.exception("Context: couldn't place %s", shown.get("name"))
            row = None
        if len(self._approx) > 2000:
            self._approx.clear()
        self._approx[key] = (self._now(), row)
        return row

    def _forecasts(self, picked: Mapping[Tuple[str, str], Mapping[str, Any]],
                   venues: Mapping[Tuple[str, str], Optional[Mapping[str, Any]]], rich: set, now: float
                   ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """{game key: {"hours", "fetched_at", "stale"}} for open-air or retractable venues of outdoor sports, games
        with odds (or asked for) within the forecast week: one batched forecast for all the venues of the games
        within the window (the client makes one call per 50 venues and keeps one cached copy per venue, refreshed
        hourly, every 15 minutes within 6 hours of its soonest game), and one more for games asked for beyond it."""
        if self.weather is None:
            return {}
        groups: Dict[bool, Dict[str, List[Any]]] = {True: {}, False: {}}  # within the window, or asked for beyond it
        wanted: Dict[Tuple[str, str], Tuple[bool, str]] = {}
        horizon = now + FORECAST_DAYS * 24 * HOUR - 4 * HOUR
        for key, game in picked.items():
            venue = venues.get(key)
            sport = engine.sport_of(key[0])
            start = _ts(game.get("_ts") if game.get("_ts") is not None else game.get("start"))
            if key not in rich or venue is None or sport is None or start is None:
                continue
            if not engine.SPORTS[sport].outdoor or str(venue.get("roof") or "open") == "dome" or start > horizon:
                continue
            lat, lon = _num(venue.get("lat")), _num(venue.get("lon"))
            if lat is None or lon is None:
                continue
            pid = str(venue.get("id") or f"{lat:.2f},{lon:.2f}")
            near = start <= now + self.window
            held = groups[near].get(pid)
            groups[near][pid] = [pid, lat, lon, min(start, held[3]) if held else start,
                                 max(start, held[4]) if held else start]
            wanted[key] = (near, pid)
        today = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).replace(hour=0, minute=0, second=0,
                                                                                    microsecond=0)
        found: Dict[bool, Dict[str, Any]] = {True: {}, False: {}}
        for near, points in groups.items():
            if not points:
                continue
            # One call for the group: from the hour before its first start (not before today, UTC: a forecast copy is
            # one date range) to 3 hours after its last; each venue's freshness follows its soonest game
            begin = max(today, datetime.datetime.fromtimestamp(min(p[3] for p in points.values()) - HOURS_BEFORE * HOUR,
                                                                datetime.timezone.utc))
            end = datetime.datetime.fromtimestamp(max(p[4] for p in points.values()) + HOURS_AFTER * HOUR,
                                                  datetime.timezone.utc)
            triples = [(pid, lat, lon, _iso(first)) for pid, lat, lon, first, _ in points.values()]
            try:
                reader = getattr(self.weather, "forecasts", None)
                if callable(reader):
                    found[near] = reader(triples, begin, end) or {}
                else:  # a client with forecast() only: no fetch times
                    found[near] = {pid: {"hours": hours, "fetched_at": now, "stale": False}
                                   for pid, hours in (self.weather.forecast(triples, begin, end) or {}).items()}
            except Exception:
                log.exception("Context: the forecast failed")
        return {key: dict(found[near][pid]) for key, (near, pid) in wanted.items() if pid in found[near]}

    def _announced(self, picked: Mapping[Tuple[str, str], Mapping[str, Any]],
                   venues: Mapping[Tuple[str, str], Optional[Mapping[str, Any]]], now: float
                   ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """{game key: {"weather", "fetched_at"}} for MLB games MLB has announced weather for (its schedule for each
        day with an MLB game in the window: 10 minutes on the day itself)."""
        if self.mlb is None:
            return {}
        from .espn_client import ESPN_DAY_TZ, team_match_score
        days: Dict[datetime.date, List[Tuple[Tuple[str, str], Mapping[str, Any], float]]] = {}
        for key, game in picked.items():
            start = _ts(game.get("_ts") if game.get("_ts") is not None else game.get("start"))
            if key[0] != "baseball_mlb" or start is None or start - now > self.window:
                continue
            day = datetime.datetime.fromtimestamp(start, ESPN_DAY_TZ).date()
            days.setdefault(day, []).append((key, game, start))
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for day, members in sorted(days.items()):
            try:
                schedule = self.mlb.games(day) or []
            except Exception as exc:
                log.info("Context: MLB's schedule for %s failed (%s)", day, type(exc).__name__)
                continue
            fetched_at = now
            peek = getattr(self.cache, "peek", None)
            held = peek(("mlb", "schedule", day.isoformat()), shared=False) if callable(peek) else None
            if held is not None:
                fetched_at = float(held[0])
            for key, game, start in members:
                venue = venues.get(key) or {}
                best, best_score = None, 0.0
                for row in schedule:
                    weather = row.get("weather") if isinstance(row, Mapping) else None
                    row_start = _ts(row.get("start")) if isinstance(row, Mapping) else None
                    if not weather or row_start is None or abs(row_start - start) > 3 * HOUR:
                        continue
                    score = 0.0
                    if venue.get("mlb_id") is not None and (row.get("venue") or {}).get("mlb_id") == venue["mlb_id"]:
                        score += 1.0
                    home = (game.get("home") or {}).get("name")
                    names = {"displayName": (row.get("home") or {}).get("name")}
                    if home and names["displayName"]:
                        score += team_match_score(str(home), names)
                    if score > best_score:
                        best, best_score = row, score
                if best is not None and best_score >= 0.8:
                    out[key] = {"weather": dict(best["weather"]), "fetched_at": fetched_at}
        return out

    def _rates(self, leagues: Iterable[str], now: float) -> Dict[str, Dict[str, Any]]:
        """Each league's home rates from its cached standings (games with odds only; else the engine's prior)."""
        from . import context_inputs
        espn = self.espn if self.espn is not None else getattr(self.slate, "espn", None)
        out: Dict[str, Dict[str, Any]] = {}
        for league in leagues:
            if engine.sport_of(league) == "soccer" or context_inputs.home_prior(league) is None:
                continue  # ESPN's soccer standings carry no home split: the prior (soccer results are read below)
            reader = getattr(espn, "standings", None)
            if not callable(reader):
                continue
            try:
                standings = reader(league)
                out[league] = context_inputs.home_rates(league, standings, as_of=_iso(now))
            except Exception as exc:
                log.info("Context: no %s standings (%s)", league, type(exc).__name__)
        return out

    @staticmethod
    def _soccer_rates(leagues: Iterable[str], boards: Mapping[str, Mapping[datetime.date, List[Any]]], now: float
                      ) -> Dict[str, Dict[str, Any]]:
        """Soccer's home and draw rates from the finished games in the scoreboards read for the lookbacks (ESPN's
        soccer standings carry no home split), shrunk toward the prior."""
        from . import context_inputs
        out: Dict[str, Dict[str, Any]] = {}
        for league in leagues:
            if engine.sport_of(league) != "soccer" or not boards.get(league):
                continue
            games = [game for rows in boards[league].values() for game in rows or []]
            try:
                out[league] = context_inputs.results_rates(league, games, as_of=_iso(now))
            except Exception:
                log.exception("Context: couldn't count %s's home results", league)
        return out

    @staticmethod
    def _league_totals(games: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
        """Each league's average total line on the slate (3 or more games), the attribution's "League average total"
        (the engine's fallback otherwise)."""
        lines: Dict[str, List[float]] = {}
        for game in games:
            total = _num(((game.get("_lines") or {}) if isinstance(game.get("_lines"), Mapping) else {}).get("total"))
            if total is not None:
                lines.setdefault(str(game.get("league")), []).append(total)
        return {league: round(statistics.median(values), 2) for league, values in lines.items() if len(values) >= 3}

    def _boards(self, picked: Mapping[Tuple[str, str], Mapping[str, Any]], rich: set, now: float
                ) -> Dict[str, Dict[datetime.date, List[Any]]]:
        """Each league's scoreboards over its lookback, through the slate (held boards; past ones fetched once, for
        leagues with a game with odds, and kept: they don't change)."""
        from . import context_inputs
        from .espn_client import ESPN_DAY_TZ
        if self.slate is None or not hasattr(self.slate, "scoreboards"):
            return {}
        wanted: Dict[str, set] = {}
        fetch: Dict[str, bool] = {}
        for key, game in picked.items():
            league = key[0]
            start = _ts(game.get("_ts") if game.get("_ts") is not None else game.get("start"))
            if start is None:
                continue
            for day in context_inputs.scoreboard_days(start, context_inputs.lookback_for(league), ESPN_DAY_TZ):
                wanted.setdefault(league, set()).add(day)
            fetch[league] = fetch.get(league, False) or key in rich
        out: Dict[str, Dict[datetime.date, List[Any]]] = {}
        for league, days in wanted.items():
            try:
                out[league] = self.slate.scoreboards(league, sorted(days), fetch=fetch.get(league, False))
            except Exception:
                log.exception("Context: couldn't read the %s scoreboards", league)
        return out

    def _trips(self, game: Mapping[str, Any], venue: Optional[Mapping[str, Any]],
               boards: Mapping[datetime.date, List[Any]], start: float) -> Dict[str, Dict[str, Any]]:
        """Each side's last game, rest, travel and time-zone shift (and, for soccer altitude, the visitors' home
        elevation)."""
        from . import context_inputs
        league = str(game.get("league") or "")
        ids = list(game.get("_ids") or (None, None))
        games = {day: [self._with_venue_id(g) for g in rows] for day, rows in boards.items()}
        last = context_inputs.last_games(league, games, [i for i in ids if i is not None], start) if games else {}
        trips: Dict[str, Dict[str, Any]] = {}
        for side, team_id in zip(("home", "away"), ids):
            team = game.get(side) if isinstance(game.get(side), Mapping) else {}
            trip: Dict[str, Any] = {"name": team.get("name"), "rest_days": None, "back_to_back": False,
                                    "last": None, "travel_km": None, "tz_shift_h": None, "days_since_last": None}
            row = last.get(str(team_id)) if team_id is not None else None
            if row:
                rest = context_inputs.rest_for(league, row, start)
                trip.update(rest)
                game_day = datetime.datetime.fromtimestamp(start, context_inputs.ESPN_DAY_TZ).date()
                try:
                    trip["days_since_last"] = (game_day - datetime.date.fromisoformat(row["date"])).days
                except (KeyError, TypeError, ValueError):
                    pass
                where = row.get("venue") or {}
                previous = None
                if self.venues is not None:
                    previous = self.venues.find(league, espn_venue_id=where.get("espn_id"), name=where.get("name"),
                                                home_team_id=team_id if row.get("home") and not row.get("neutral")
                                                else None, neutral=bool(row.get("neutral")))
                trip["last"] = {"date": row.get("date"), "opponent": row.get("opponent"),
                                "venue": (previous or {}).get("name") or where.get("name"),
                                "city": (previous or {}).get("city") or where.get("city")}
                if previous is not None and venue is not None:
                    try:
                        trip["travel_km"] = round(engine.haversine_km(float(previous["lat"]), float(previous["lon"]),
                                                                      float(venue["lat"]), float(venue["lon"])))
                        trip["tz_shift_h"] = round(engine.tz_shift_hours(previous["tz"], venue["tz"], start), 1)
                    except (KeyError, TypeError, ValueError):
                        pass
            if side == "away" and team_id is not None and self.venues is not None:
                home = self.venues.home_of(league, team_id) if hasattr(self.venues, "home_of") else None
                if home is not None and home.get("elevation_m") is not None:
                    trip["home_elevation_m"] = home["elevation_m"]
            trips[side] = trip
        return trips

    @staticmethod
    def _with_venue_id(game: Any) -> Any:
        """A slate-parsed game with its ESPN venue id where context_inputs reads it (venue.espn_id)."""
        if isinstance(game, Mapping) and game.get("_place") and isinstance(game.get("venue"), Mapping):
            place = game["_place"]
            if place.get("id") and not game["venue"].get("espn_id"):
                return {**game, "venue": {**game["venue"], "espn_id": place.get("id")}}
        return game

    # --- the block ---

    def build(self, game: Mapping[str, Any], venue: Optional[Mapping[str, Any]], *,
              forecast: Optional[Mapping[str, Any]] = None, announced: Optional[Mapping[str, Any]] = None,
              boards: Optional[Mapping[datetime.date, List[Any]]] = None, rates: Optional[Mapping[str, Any]] = None,
              league_total: Optional[float] = None, now: Optional[float] = None
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """(the context block, what the logs need) for one slate game from its inputs. Pure but for the store's
        weather_log (the current forecast is logged first, then the forecast at the odds' time is read)."""
        now = self._now() if now is None else now
        league = str(game["league"])
        sport = engine.sport_of(league)
        sp = engine.SPORTS[sport]
        # The learning loop's live values (learn/registry.py): the engine gets a SportParams with the tuned priors
        # and SDs, and the tuned ADJUST_WEIGHT, only once a change has passed the gate (until then exactly today's)
        brain = brain_params(league, self.store)
        overrides = brain.context_overrides()
        eng = engine.sport_params(sport, overrides) if overrides else sport
        weight = brain.adjust_weight
        game_id = game_id_of(game)
        start = _ts(game.get("_ts") if game.get("_ts") is not None else game.get("start"))
        status = str(game.get("status") or "pre")
        lines = game.get("_lines") if isinstance(game.get("_lines"), Mapping) else {}
        total_line, p_over = _num(lines.get("total")), _num(lines.get("p_over"))
        odds_at = _ts(game.get("odds_as_of"))
        venue = dict(venue) if venue else None
        trips = self._trips(game, venue, boards or {}, start) if start is not None else {}

        # The weather now: the forecast's hour at the start, with MLB's announcement beside it
        conditions: Optional[Dict[str, Any]] = None
        hours: List[Dict[str, Any]] = []
        fetched_at: Optional[float] = None
        mlb_weather = (announced or {}).get("weather") if announced else None
        if forecast and start is not None and start - now <= FORECAST_DAYS * 24 * HOUR:
            hours = [dict(hour) for hour in forecast.get("hours") or []
                     if (_ts(hour.get("t")) or 0) >= start - HOURS_BEFORE * HOUR - 1
                     and (_ts(hour.get("t")) or 0) <= start + HOURS_AFTER * HOUR + 1]
            first = _nearest(forecast.get("hours") or [], start)
            if first is not None:
                fetched_at = _num(forecast.get("fetched_at")) or now
                conditions = {**{k: first.get(k) for k in ("t", "temp_c", "wind_kmh", "wind_dir", "gust_kmh",
                                                           "precip_prob", "precip_mm")},
                              "fetched_at": _iso(fetched_at), "source": "Open-Meteo"}
        if mlb_weather:
            announced_at = _num((announced or {}).get("fetched_at")) or now
            block_announced = {k: mlb_weather.get(k) for k in ("condition", "temp_f", "wind_mph", "wind_text",
                                                               "wind_kind", "field", "roof_closed")}
            if conditions is None:
                conditions = {"source": "MLB", "fetched_at": _iso(announced_at)}
                fetched_at = announced_at
            else:
                fetched_at = max(fetched_at or announced_at, announced_at)
            conditions["announced"] = block_announced
        if venue is not None and str(venue.get("roof") or "open") == "dome":
            conditions, hours, mlb_weather = None, [], None  # a dome: no weather

        # Log the conditions as they are now, then find them as they stood when the odds were fetched
        effect_now = None
        if conditions is not None and venue is not None:
            effect_now = engine.weather_effects(eng, venue, conditions, None, market_total=total_line)[0]
        if conditions is not None and fetched_at is not None and venue is not None and self.store is not None \
                and hasattr(self.store, "log_weather") and start is not None:
            row = {"league": league, "game_key": game_id, "fetched_at": fetched_at, "start": _iso(start),
                   "venue": venue.get("id"), "source": conditions.get("source") or "Open-Meteo",
                   "effect": effect_now, "conditions": _plain(conditions),
                   # The conditions without their fetch time: a forecast read again unchanged adds no row
                   "sig": _sig({k: v for k, v in _plain(conditions).items() if k != "fetched_at"})}
            try:
                self.stats["weather_rows"] += int(self.store.log_weather([row]) or 0)
            except Exception:
                log.exception("Context: couldn't log the forecast of %s", game_id)
        at_odds: Optional[Dict[str, Any]] = None
        if odds_at is not None and self.store is not None and hasattr(self.store, "weather_at"):
            try:
                found = self.store.weather_at(league, game_id, odds_at)
            except Exception:
                found = None
            if found and isinstance(found.get("conditions"), Mapping):
                at_odds = dict(found["conditions"])
                at_odds.setdefault("fetched_at", _iso(_num(found.get("fetched_at"))))

        # Attribution: the market's margin (its spread, else the blend's chance) and its total line
        margin, margin_note = self._market_margin(sport, game, lines, total_line, eng)
        as_of = game.get("odds_as_of") or _iso(now)
        neutral = bool(game.get("_neutral"))
        attribution: Dict[str, Any] = {"margin": None, "total": None}
        if margin is not None:
            made = {name: engine.attribute_margin(eng, market_margin=margin, venue=venue, trips=trips,
                                                  neutral=neutral, as_of=as_of, home_rates=rates, units=name)
                    for name in (None,) + SHOWN_UNITS}
            shown = _with_evidence_units(made, sp.digits)
            shown["source"] = margin_note
            attribution["margin"] = shown
        if total_line is not None:
            made = {name: engine.attribute_total(eng, market_total=total_line, league_total=league_total,
                                                 venue=venue, conditions_at_odds=at_odds, as_of=as_of, units=name)
                    for name in (None,) + SHOWN_UNITS}
            attribution["total"] = _with_evidence_units(made, sp.digits)

        # Adjustments: only for a forecast or roof that changed since the odds (before the start)
        adjustments: List[Dict[str, Any]] = []
        tz = (venue or {}).get("tz")
        units = units_for(league)
        if status == "pre" and venue is not None and conditions is not None and at_odds is not None:
            moved = engine.weather_adjustment(eng, venue, conditions, at_odds, market_total=total_line,
                                              has_market=total_line is not None, odds_at=odds_at, tz=tz, units=units,
                                              weight=weight)
            if moved is not None:
                entry = moved.to_dict()
                other = "metric" if units == "imperial" else "imperial"
                again = engine.weather_adjustment(eng, venue, conditions, at_odds, market_total=total_line,
                                                  has_market=total_line is not None, odds_at=odds_at, tz=tz,
                                                  units=other, weight=weight)
                entry["reasons"] = {units: moved.reason, other: again.reason if again is not None else moved.reason}
                entry["p_over"] = None
                if total_line is not None and p_over is not None and moved.applied:
                    after = engine.adjusted_over(eng, total_line, p_over, moved.applied)
                    entry["p_over"] = {"line": total_line, "before": round(p_over, 4), "after": round(after, 4)}
                adjustments.append(entry)

        weather_block = None
        if conditions is not None and venue is not None and (hours or conditions.get("announced")):
            start_hour = {k: conditions.get(k) for k in ("t", "temp_c", "wind_kmh", "wind_dir", "gust_kmh",
                                                         "precip_prob", "precip_mm")}
            weather_block = {
                "source": "Open-Meteo" if conditions.get("source") != "MLB" else "MLB",
                "attribution": dict(ATTRIBUTION) if conditions.get("source") != "MLB" else None,
                "fetched_at": _iso(fetched_at),
                "start": start_hour,
                "hours": hours,
                "announced": ({k: mlb_weather.get(k) for k in ("condition", "temp_f", "wind_text")}
                              if mlb_weather else None),
                "effect": {"total": round(effect_now, 3), "unit": sp.unit} if effect_now is not None else None,
                "at_odds": (_at_odds_view(sport, at_odds, conditions) if game.get("odds_as_of") else None),
            }
            if forecast and forecast.get("stale"):
                weather_block["stale"] = True
        badges = engine.badges(sport, venue, conditions, trips, adjustments)
        state = engine.roof_state(venue, conditions) if venue is not None else None
        venue_block = None
        if venue is not None:
            lat, lon = _num(venue.get("lat")), _num(venue.get("lon"))
            park = venue.get("park") if isinstance(venue.get("park"), Mapping) else None
            venue_block = {
                "id": venue.get("id"), "name": venue.get("name"), "city": venue.get("city"),
                "country": venue.get("country"), "lat": lat, "lon": lon,
                "map": _osm(lat, lon) if lat is not None and lon is not None else None,
                "elevation_m": venue.get("elevation_m"), "roof": venue.get("roof"), "roof_state": state,
                "surface": venue.get("surface"), "capacity": venue.get("capacity"),
                "park": {k: park.get(k) for k in ("runs", "hr", "source")} if park else None,
                "approx": bool(venue.get("approx")), "source": venue.get("source"), "tz": venue.get("tz"),
            }
        teams = {side: {k: trip.get(k) for k in ("name", "rest_days", "back_to_back", "last", "travel_km",
                                                  "tz_shift_h")} for side, trip in trips.items()}
        lines_text = None
        if venue is not None:
            lines_text = {name: conditions_line(sport, venue, conditions, effect_now, name)
                          for name in ("metric", "imperial")}
        block = {
            "status": "ready", "league": league, "game_id": game_id, "sport": sport, "unit": sp.unit,
            "as_of": _iso(now), "start": _iso(start), "odds_as_of": game.get("odds_as_of"),
            "venue": venue_block, "teams": teams, "weather": weather_block,
            "attribution": attribution, "adjustments": adjustments, "badges": badges, "line": lines_text,
        }
        extra = {"total_line": total_line, "p_over": p_over, "adjustments": adjustments,
                 "params_version": brain.version}
        return block, extra

    @staticmethod
    def _market_margin(sport: str, game: Mapping[str, Any], lines: Mapping[str, Any], total_line: Optional[float],
                       eng: Any = None) -> Tuple[Optional[float], str]:
        """The market's home margin: the point spread (sports where it is one) or the margin the blend's chance
        implies; with what it came from. ``eng``: the engine's sport (a tuned SportParams; default ``sport``)."""
        eng = sport if eng is None else eng
        spread = _num(lines.get("spread"))
        if spread is not None and sport in POINT_SPREAD_SPORTS:
            return -spread, "The board's point spread"
        chance = game.get("chance") if isinstance(game.get("chance"), Mapping) else {}
        home, away, draw = _num(chance.get("home")), _num(chance.get("away")), _num(chance.get("draw"))
        if home is None or away is None:
            return None, ""
        try:
            if sport == "soccer":
                if draw is None:
                    return None, ""
                total = home + away + draw
                return engine.implied_margin(eng, home / total, p_draw=draw / total,
                                             total=total_line if total_line and total_line > 0.5 else None), \
                    "The margin the markets brain's chance implies"
            share = home / (home + away)
            return engine.implied_margin(eng, share), "The margin the markets brain's chance implies"
        except (ValueError, ZeroDivisionError):
            return None, ""

    # --- the logs ---

    def _log(self, game: Mapping[str, Any], key: Tuple[str, str], block: Mapping[str, Any],
             extra: Mapping[str, Any], now: float) -> None:
        """context_log: the game's total line and chances of the over before and after any adjustment (games still
        to start, with a total line and the books' chance of the over)."""
        if self.store is None or not hasattr(self.store, "log_context") or game.get("status") != "pre":
            return
        line, p_over = extra.get("total_line"), extra.get("p_over")
        if line is None or p_over is None:
            return
        moved = next((a for a in extra.get("adjustments") or [] if a.get("kind") == "total"), None)
        applied = float((moved or {}).get("applied") or 0.0)
        p_after = ((moved or {}).get("p_over") or {}).get("after") if moved else None
        odds_id, espn_id = game.get("odds_id"), game.get("espn_id")
        name = lambda team: (team or {}).get("name") or "?"  # noqa: E731
        row = {"league": key[0], "game_key": f"espn:{espn_id}" if espn_id else f"odds:{odds_id}",
               "event_id": odds_id, "espn_id": espn_id, "home": name(game.get("home")), "away": name(game.get("away")),
               "start": block.get("start"), "line": line, "p_before": round(float(p_over), 4),
               "p_after": round(float(p_after if p_after is not None else p_over), 4), "applied": round(applied, 4),
               "reason": (moved or {}).get("reason"), "adjustment": moved,
               "params_version": int(extra.get("params_version") or 0)}
        signature = _sig(row)
        if self._logged_context.get(key) == signature:
            return
        try:
            self.stats["context_rows"] += int(self.store.log_context([row], now) or 0)
        except Exception:
            log.exception("Context: couldn't log %s", key)
            return
        self._logged_context[key] = signature
        if len(self._logged_context) > 20000:
            self._logged_context = dict(list(self._logged_context.items())[-10000:])


def _with_evidence_units(made: Mapping[Optional[str], Projection], digits: int) -> Dict[str, Any]:
    """The projection's to_dict (``made[None]``, the default wording) with ``evidence_units`` ({"metric",
    "imperial"}) on each factor whose evidence names figures that come first in one system (weather, altitude), so
    the app shows the person's units first; the sizes are the same in every wording."""
    shown = made[None].to_dict(digits)
    worded = {name: {f.key: f.evidence for f in made[name].factors} for name in SHOWN_UNITS}
    for factor in shown.get("factors") or []:
        texts = {name: worded[name].get(factor.get("key")) for name in SHOWN_UNITS}
        if all(isinstance(text, str) for text in texts.values()) and texts["metric"] != texts["imperial"]:
            factor["evidence_units"] = texts
    return shown


def _at_odds_view(sport: str, at_odds: Optional[Mapping[str, Any]], conditions: Optional[Mapping[str, Any]]
                  ) -> Dict[str, Any]:
    """The weather block's ``at_odds``: whether the forecast from when the odds were fetched is known, its fetch time,
    the conditions at the start it gave (``start``: temp_c, wind_kmh, as the attribution reads them, MLB's
    announcement included) and which of them have ``changed`` since (["wind"] by 1 mph or more, ["temp"] by 2°F or
    more: the thresholds of an adjustment's reason)."""
    view: Dict[str, Any] = {"known": at_odds is not None, "fetched_at": (at_odds or {}).get("fetched_at")}
    if at_odds is not None:
        then = engine.read_conditions(sport, at_odds)
        view["start"] = {"temp_c": _r(then.temp_c), "wind_kmh": _r(then.wind_kmh)}
        changed: List[str] = []
        if conditions is not None:
            now = engine.read_conditions(sport, conditions)
            if then.wind_mph is not None and now.wind_mph is not None and abs(then.wind_mph - now.wind_mph) >= 1:
                changed.append("wind")
            if then.temp_f is not None and now.temp_f is not None and abs(then.temp_f - now.temp_f) >= 2:
                changed.append("temp")
        view["changed"] = changed
    return view


def _nearest(hours: Sequence[Mapping[str, Any]], at: float, gap: float = 90 * 60) -> Optional[Dict[str, Any]]:
    best, distance = None, math.inf
    for hour in hours or []:
        moment = _ts(hour.get("t")) if isinstance(hour, Mapping) else None
        if moment is None:
            continue
        if abs(moment - at) < distance:
            best, distance = hour, abs(moment - at)
    return dict(best) if best is not None and distance <= gap else None


def _plain(value: Any) -> Any:
    """A JSON-safe copy (numbers, text, lists, dicts)."""
    return json.loads(json.dumps(value, default=str))


def grade_row(row: Mapping[str, Any], home: Optional[float], away: Optional[float]) -> Tuple[str, Optional[str],
                                                                                             Dict[str, Any]]:
    """(status, outcome, result) for a context_log row from the final score: over or under scores each chance's Brier
    score ((p - won)^2); a push or a missing score is void."""
    if home is None or away is None:
        return "void", None, {"reason": "no final score"}
    total = float(home) + float(away)
    line = float(row["line"])
    if total == line:
        return "void", "push", {"total": total, "reason": "push"}
    won = 1.0 if total > line else 0.0
    before, after = float(row["p_before"]), float(row["p_after"])
    return "graded", "over" if won else "under", {
        "total": total, "home": home, "away": away,
        "brier_before": round((before - won) ** 2, 6), "brier_after": round((after - won) ** 2, 6)}
