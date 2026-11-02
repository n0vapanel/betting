"""nflverse's NFL schedule (``nfldata/master/data/games.csv``): one request covers every season since 1999 with the
result, the closing spread, total and moneylines, roof, surface and the game-time temperature and wind.

Read through the shared TTLCache (kind "hist_file", 12 hours: the file grows each week of the season); the cached
value is the rows of the seasons kept (FIRST_SEASON on) with the columns read here. Licence: CC BY 4.0 (nflverse).

- ``espn`` is ESPN's event id: a game with one is keyed ``espn:<id>`` (the live logs' key), else
  ``nflverse:<game_id>``.
- Lines carry no timestamp and behave like a consensus close (data.md): stored as ``close`` from book ``nflverse``,
  a minute before kick-off. ``spread_line`` > 0 means the home team is favoured (``result = home - away``), so the
  home side's handicap is ``-spread_line``.
- ``gameday``/``gametime`` are US Eastern. ``temp`` (F) and ``wind`` (mph) are observations for outdoor games only,
  filled in after the game: stored as ``actual`` weather (C, km/h).
"""

from __future__ import annotations

import csv
import datetime
import io
from typing import Any, Dict, List, Mapping, Optional
from zoneinfo import ZoneInfo

from ...data.markets.aliases import team_key
from ..history_store import iso, num
from .http import HistHttp, cached

URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
SOURCE = "nflverse"
LEAGUE = "americanfootball_nfl"
EASTERN = ZoneInfo("America/New_York")
FIRST_SEASON = 2019
CLOSE_LEAD = 60.0
PARSE_VERSION = 1
KEPT = ("game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "away_score", "home_team",
        "home_score", "location", "overtime", "espn", "away_rest", "home_rest", "away_moneyline", "home_moneyline",
        "spread_line", "away_spread_odds", "home_spread_odds", "total_line", "under_odds", "over_odds", "div_game",
        "roof", "surface", "temp", "wind", "referee", "stadium")
# nflverse abbreviations the app's NFL table doesn't know (franchises that moved)
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LAR"}


def american_to_decimal(value: Any) -> Optional[float]:
    """+150 -> 2.5, -200 -> 1.5; None for 0, blanks and text."""
    number = num(value)
    if number is None or number == 0 or -100 < number < 100:
        return None
    return round(1 + (number / 100.0 if number > 0 else 100.0 / -number), 4)


def nfl_key(abbr: str) -> str:
    plain = str(abbr or "").strip().upper()
    plain = TEAM_ALIASES.get(plain, plain)
    return team_key(LEAGUE, plain) or plain


def parse_csv(text: str, first_season: int = FIRST_SEASON) -> List[Dict[str, str]]:
    rows = []
    reader = csv.DictReader(io.StringIO((text or "").lstrip("﻿")))
    for row in reader:
        season = num(row.get("season"))
        if season is None or season < first_season:
            continue
        rows.append({name: (row.get(name) or "").strip() for name in KEPT if (row.get(name) or "").strip()})
    return rows


def kickoff(row: Mapping[str, str]) -> Optional[float]:
    try:
        day = datetime.date.fromisoformat(row.get("gameday", ""))
    except ValueError:
        return None
    try:
        hour, minute = (int(part) for part in row.get("gametime", "13:00").split(":")[:2])
    except ValueError:
        hour, minute = 13, 0
    return datetime.datetime(day.year, day.month, day.day, hour, minute, tzinfo=EASTERN).timestamp()


def record(row: Mapping[str, str]) -> Optional[Dict[str, Any]]:
    """One games.csv row as a history record; None for a game not played yet."""
    home_score, away_score = num(row.get("home_score")), num(row.get("away_score"))
    start_ts = kickoff(row)
    if home_score is None or away_score is None or start_ts is None:
        return None
    home, away = nfl_key(row.get("home_team", "")), nfl_key(row.get("away_team", ""))
    espn = (row.get("espn") or "").strip()
    espn = espn.split(".")[0] if espn else ""
    as_of = start_ts - CLOSE_LEAD
    lines: List[Dict[str, Any]] = []

    def add(market: str, side: str, point: Optional[float], price: Optional[float]) -> None:
        if price is not None:
            lines.append({"kind": "close", "book": "nflverse", "market": market, "side": side, "point": point,
                          "price": price, "as_of": as_of})

    add("h2h", "home", None, american_to_decimal(row.get("home_moneyline")))
    add("h2h", "away", None, american_to_decimal(row.get("away_moneyline")))
    spread = num(row.get("spread_line"))
    if spread is not None:
        add("spreads", "home", -spread, american_to_decimal(row.get("home_spread_odds")))
        add("spreads", "away", spread, american_to_decimal(row.get("away_spread_odds")))
    total = num(row.get("total_line"))
    if total is not None:
        add("totals", "over", total, american_to_decimal(row.get("over_odds")))
        add("totals", "under", total, american_to_decimal(row.get("under_odds")))
    weather = []
    temp_f, wind_mph = num(row.get("temp")), num(row.get("wind"))
    if temp_f is not None or wind_mph is not None:
        weather.append({"kind": "actual",
                        "temp_c": None if temp_f is None else round((temp_f - 32) * 5 / 9, 1),
                        "wind_kmh": None if wind_mph is None else round(wind_mph * 1.609344, 1),
                        "precip_mm": None})
    info = {key: row.get(key) for key in ("game_id", "game_type", "week", "roof", "surface", "referee", "stadium",
                                         "location", "overtime", "div_game", "home_rest", "away_rest") if row.get(key)}
    if "spread_line" in row and not any(line["market"] == "h2h" for line in lines):
        info["no_moneyline"] = True
    game = {"league": LEAGUE, "game_key": f"espn:{espn}" if espn else f"nflverse:{row.get('game_id')}",
            "espn_id": espn or None, "start": iso(start_ts), "home": home, "away": away,
            "home_score": int(home_score), "away_score": int(away_score), "status": "final",
            "season": int(num(row.get("season")) or 0), "source": SOURCE, "home_key": home, "away_key": away,
            "info": info}
    return {"game": game, "lines": lines, "stats": [], "weather": weather}


class NflVerse:
    """games.csv through the shared cache: ``records(first_season)`` gives the played games as history records."""

    def __init__(self, http: HistHttp, cache: Any) -> None:
        self.http = http
        self.cache = cache

    def rows(self, first_season: int = FIRST_SEASON) -> List[Dict[str, str]]:
        return cached(self.cache, "hist_file", ("hist", PARSE_VERSION, "nflverse", "games", first_season),
                      lambda: parse_csv(self.http.text(URL), first_season), "current")

    def records(self, first_season: int = FIRST_SEASON) -> List[Dict[str, Any]]:
        found = []
        for row in self.rows(first_season):
            item = record(row)
            if item is not None:
                found.append(item)
        return found
