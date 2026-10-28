"""MLB StatsAPI's schedule: each game's venue and the weather MLB announces for it (v3 Stage 4).

``MlbStats(cache, session=None).games(date)`` reads
``statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=venue(location,fieldInfo),weather`` through the
shared TTLCache (kind "mlb_schedule": 10 minutes, 10 more served while a refresh runs, on the day itself; 6 hours for
other dates) and gives one dict per game:

    {"game_pk", "start" (ISO, UTC), "status" ("Preview"|"Live"|"Final"), "detailed_state", "double_header",
     "game_number", "day_night", "home": {"name", "id"}, "away": {"name", "id"},
     "venue": {"mlb_id", "name", "roof"},
     "weather": {"condition", "temp_f", "wind_mph", "wind_text", "wind_kind", "field", "roof_closed"} | None}

``weather`` is None until MLB announces it (around first pitch; never for a game days away). ``wind_kind`` is "out",
"in", "cross", "calm", "varies" or None; ``field`` is "CF", "LF" or "RF" for wind blowing out to or in from a field.
A condition of "Roof Closed" or "Dome" sets ``roof_closed``. StatsAPI is public and keyless, but MLB's terms limit
it to personal, non-commercial use.
"""

from __future__ import annotations

import datetime
import logging
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

from .cache import TTLCache, ttl_for

log = logging.getLogger(__name__)

SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
HYDRATE = "venue(location,fieldInfo),weather"
MLB_DAY_TZ = ZoneInfo("America/New_York")  # MLB's schedule dates are US Eastern
TIMEOUT = 15.0
FIELDS = ("LF", "CF", "RF")
ROOF_CLOSED = ("roof closed", "dome")

_SPEED = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*mph\b", re.I)
_OUT = re.compile(r"\bout\s+to\s+(lf|cf|rf)\b", re.I)
_IN = re.compile(r"\bin\s+from\s+(lf|cf|rf)\b", re.I)
_CROSS = re.compile(r"\b(l\s+to\s+r|r\s+to\s+l)\b", re.I)


def parse_wind(text: Any) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """MLB's wind text as (mph, kind, field): "14 mph, Out To RF" -> (14, "out", "RF"); "8 mph, In From LF" ->
    (8, "in", "LF"); "5 mph, L To R" and "R To L" -> (5, "cross", None); "Calm" -> (0, "calm", None); "6 mph,
    Varies" -> (6, "varies", None); "0 mph, None" (a dome or a closed roof) and any 0 mph -> (0, "calm", None).
    Unreadable text gives (None, None, None); a speed with no readable direction (mph, None, None)."""
    if not isinstance(text, str) or not text.strip():
        return None, None, None
    words = text.strip()
    speed_match = _SPEED.match(words)
    mph = int(round(float(speed_match.group(1)))) if speed_match else None
    direction = words.split(",", 1)[1] if "," in words else ("" if speed_match else words)
    if mph == 0 or re.search(r"\bcalm\b", direction, re.I):
        return 0, "calm", None
    found = _OUT.search(direction)
    if found:
        return mph, "out", found.group(1).upper()
    found = _IN.search(direction)
    if found:
        return mph, "in", found.group(1).upper()
    if _CROSS.search(direction):
        return mph, "cross", None
    if re.search(r"\bvar", direction, re.I):
        return mph, "varies", None
    return mph, None, None


def roof_closed(condition: Any) -> bool:
    """True when MLB's condition says the game is played under a roof ("Roof Closed", "Dome")."""
    return isinstance(condition, str) and condition.strip().lower() in ROOF_CLOSED


def _int(value: Any) -> Optional[int]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(round(number))


def parse_weather(weather: Any) -> Optional[Dict[str, Any]]:
    """StatsAPI's ``weather`` ({"condition", "temp", "wind"}) in the client's form; None when MLB hasn't announced
    any (an empty object, or nothing)."""
    if not isinstance(weather, Mapping) or not any(weather.get(k) for k in ("condition", "temp", "wind")):
        return None
    condition = str(weather.get("condition") or "").strip() or None
    wind_text = str(weather.get("wind") or "").strip() or None
    mph, kind, field = parse_wind(wind_text)
    return {"condition": condition, "temp_f": _int(weather.get("temp")), "wind_mph": mph, "wind_text": wind_text,
            "wind_kind": kind, "field": field, "roof_closed": roof_closed(condition)}


def _team(side: Any) -> Dict[str, Any]:
    team = (side or {}).get("team") if isinstance(side, Mapping) else None
    team = team if isinstance(team, Mapping) else {}
    return {"name": team.get("name"), "id": team.get("id")}


def parse_game(game: Mapping[str, Any]) -> Dict[str, Any]:
    """One StatsAPI schedule game in the client's form (see the module's docstring)."""
    venue = game.get("venue") if isinstance(game.get("venue"), Mapping) else {}
    field_info = venue.get("fieldInfo") if isinstance(venue.get("fieldInfo"), Mapping) else {}
    status = game.get("status") if isinstance(game.get("status"), Mapping) else {}
    teams = game.get("teams") if isinstance(game.get("teams"), Mapping) else {}
    return {
        "game_pk": game.get("gamePk"),
        "start": game.get("gameDate"),
        "status": status.get("abstractGameState"),
        "detailed_state": status.get("detailedState"),
        "double_header": game.get("doubleHeader"),
        "game_number": game.get("gameNumber"),
        "day_night": game.get("dayNight"),
        "home": _team(teams.get("home")),
        "away": _team(teams.get("away")),
        "venue": {"mlb_id": venue.get("id"), "name": venue.get("name"),
                  "roof": str(field_info.get("roofType")).lower() if field_info.get("roofType") else None},
        "weather": parse_weather(game.get("weather")),
    }


def parse_schedule(payload: Any) -> List[Dict[str, Any]]:
    """Every game of a StatsAPI schedule answer, in the client's form (unreadable games skipped)."""
    games: List[Dict[str, Any]] = []
    for day in (payload.get("dates") or []) if isinstance(payload, Mapping) else []:
        for game in (day.get("games") or []) if isinstance(day, Mapping) else []:
            if not isinstance(game, Mapping) or game.get("gamePk") is None:
                continue
            try:
                games.append(parse_game(game))
            except (AttributeError, TypeError, ValueError):
                continue
    return games


def _date(value: Any) -> datetime.date:
    if isinstance(value, datetime.datetime):
        return value.astimezone(MLB_DAY_TZ).date() if value.tzinfo else value.date()
    if isinstance(value, datetime.date):
        return value
    return datetime.date.fromisoformat(str(value).strip()[:10])


class MlbStats:
    """MLB StatsAPI's schedule with announced weather, through the shared TTLCache (see the module's docstring)."""

    def __init__(self, cache: Optional[TTLCache] = None, session: Any = None,
                 clock: Optional[Callable[[], float]] = None, timeout: float = TIMEOUT) -> None:
        self.cache = cache if cache is not None else TTLCache(name="mlb")
        self.session = session
        now = getattr(self.cache, "now", None)
        self.clock: Callable[[], float] = clock or (now if callable(now) else time.time)
        self.timeout = float(timeout)

    def _get(self, url: str, params: Mapping[str, Any]) -> Any:
        """One GET to StatsAPI (no key). Only ever called inside a cache loader."""
        getter = self.session.get if self.session is not None else requests.get
        response = getter(url, params=dict(params), timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0"})
        if not getattr(response, "ok", False):
            raise RuntimeError(f"MLB StatsAPI answered HTTP {getattr(response, 'status_code', '?')} for {url}")
        return response.json()

    def day_state(self, day: datetime.date) -> str:
        """The schedule's policy state: "today" (MLB's day, US Eastern), "past" or "future"."""
        today = datetime.datetime.fromtimestamp(self.clock(), MLB_DAY_TZ).date()
        return "today" if day == today else ("past" if day < today else "future")

    def games(self, date: Any) -> List[Dict[str, Any]]:
        """The day's games (``date``: a date, a datetime, or ISO text) with venue and announced weather."""
        day = _date(date)
        rule = ttl_for("mlb_schedule", self.day_state(day))

        def load() -> List[Dict[str, Any]]:
            payload = self._get(SCHEDULE, {"sportId": 1, "date": day.isoformat(), "hydrate": HYDRATE})
            return parse_schedule(payload)

        return self.cache.get(("mlb", "schedule", day.isoformat()), rule.ttl, load, stale=rule.stale,
                              error_ttl=60.0, backoff_cap=15 * 60.0, kind="mlb_schedule")
