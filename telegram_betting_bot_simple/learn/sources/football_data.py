"""football-data.co.uk: results, match stats and odds for the soccer leagues the app shows (seasons 2019/20 on).

Files: ``/mmz4281/{YYZZ}/{DIV}.csv`` (one league-season: E0, E1, SP1, I1, D1, F1) and ``/new/{CC}.csv`` (one country,
every season since 2012, closing 1X2 only: USA for MLS). Read through the shared TTLCache (kind "hist_file": a
finished season's file for 7 days, the season being played for 12 hours); the cached value is the parsed rows.

**Which columns are the close.** football-data's notes: plain bookmaker columns (``AvgH``, ``B365>2.5``, ``AHh``,
``PAHH``) are *pre-closing* odds, collected on Friday afternoons for weekend games and Tuesday afternoons for midweek
ones; the same names with a ``C`` after the bookmaker (``AvgCH``, ``B365C>2.5``, ``AHCh``, ``PCAHH``) are the
*closing* odds. COLUMNS below pins that mapping (the old soccer backtest had it the wrong way round: that was its
leak, and tests/test_learn_history.py pins it). Pre-closing prices are stored as ``open`` with ``as_of`` the latest
Tuesday or Friday 18:00 UK time at least an hour before kick-off (an upper bound); closing prices as ``close`` a
minute before kick-off.

**Pinnacle.** Its prices stop after 2026-01-08, and the site says its feed was unreliable ("systematically out of
date") from 2025-07-23: Pinnacle columns are dropped for games from that date (PINNACLE_UNRELIABLE_FROM), so the
sharp baseline is then Betfair Exchange's closing price, or the market average (HistoryReader.sharp_line's order).

Kept books: closing prices of Pinnacle (``PS``/``P``), Betfair Exchange (``BFE``), the market average (``Avg``) and
maximum (``Max``) and Bet365 (``B365``); early-market prices of the first three only (OPEN_BOOKS: the table stays small).
Times are UK local (Europe/London). Team names join the app's through ``aliases.team_key`` after FD_ALIASES (the
site's short names: "Nott'm Forest", "Ath Madrid", "M'gladbach"...).
"""

from __future__ import annotations

import csv
import datetime
import io
from typing import Any, Dict, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

from ...data.markets.aliases import team_key
from ..history_store import iso, num
from .http import HistHttp, cached

BASE = "https://football-data.co.uk"  # the www. host answers 302 to this one
SOURCE = "football-data"
UK = ZoneInfo("Europe/London")
FIRST_SEASON = 2019  # 2019/20: the first season with both pre-closing and closing odds (and AHh)
PARSE_VERSION = 1

# football-data division -> the app's league key
DIVISIONS: Dict[str, str] = {
    "E0": "soccer_epl",
    "E1": "soccer_efl_champ",
    "SP1": "soccer_spain_la_liga",
    "I1": "soccer_italy_serie_a",
    "D1": "soccer_germany_bundesliga",
    "F1": "soccer_france_ligue_one",
}
# "new" leagues: country file -> (the app's league key, the League column's value)
NEW_LEAGUES: Dict[str, Tuple[str, str]] = {"USA": ("soccer_usa_mls", "MLS")}

PINNACLE_UNRELIABLE_FROM = datetime.date(2025, 7, 23)
PINNACLE_LAST = datetime.date(2026, 1, 8)
CLOSE_LEAD = 60.0  # a closing price is known a minute before kick-off (at the latest)
OPEN_HOUR = 18     # pre-closing prices: collected Tuesday or Friday afternoon, by 18:00 UK time
OPEN_DAYS = (1, 4)  # Tuesday, Friday (Monday = 0)

# Column prefixes of the books kept -> book name
BOOKS_1X2 = {"B365": "bet365", "PS": "pinnacle", "Avg": "avg", "Max": "max", "BFE": "betfair_ex"}
BOOKS_OU = {"B365": "bet365", "P": "pinnacle", "Avg": "avg", "Max": "max", "BFE": "betfair_ex"}
BOOKS_AH = {"B365": "bet365", "P": "pinnacle", "Avg": "avg", "Max": "max", "BFE": "betfair_ex"}
OPEN_BOOKS = ("pinnacle", "avg", "betfair_ex")  # early-market prices kept


def _columns() -> Tuple[Tuple[str, str, str, str, str, Optional[str]], ...]:
    """(column, kind, book, market, side, point column) for every price kept. Pre-closing = ``open``, the ``C``
    columns = ``close``."""
    found: List[Tuple[str, str, str, str, str, Optional[str]]] = []
    for kind, c in (("open", ""), ("close", "C")):
        def kept(book: str) -> bool:
            return kind == "close" or book in OPEN_BOOKS

        for prefix, book in BOOKS_1X2.items():
            for letter, side in (("H", "home"), ("D", "draw"), ("A", "away")):
                if kept(book):
                    found.append((f"{prefix}{c}{letter}", kind, book, "h2h", side, None))
        for prefix, book in BOOKS_OU.items():
            if kept(book):
                found.append((f"{prefix}{c}>2.5", kind, book, "totals", "over", "2.5"))
                found.append((f"{prefix}{c}<2.5", kind, book, "totals", "under", "2.5"))
        for prefix, book in BOOKS_AH.items():
            line = "AHCh" if c else "AHh"
            if kept(book):
                found.append((f"{prefix}{c}AHH", kind, book, "spreads", "home", line))
                found.append((f"{prefix}{c}AHA", kind, book, "spreads", "away", line))
    return tuple(found)


COLUMNS = _columns()

# Match stats: (home column, away column) -> stat
STATS = {("HS", "AS"): "shots", ("HST", "AST"): "shots_on_target", ("HF", "AF"): "fouls", ("HC", "AC"): "corners",
         ("HY", "AY"): "yellow", ("HR", "AR"): "red", ("HxG", "AxG"): "xg", ("HO", "AO"): "offsides",
         ("HTHG", "HTAG"): "goals_h1"}

# football-data's short names -> a spelling aliases.team_key (and ESPN) shares
FD_ALIASES = {
    "nott'm forest": "Nottingham Forest", "ath madrid": "Atletico Madrid", "ath bilbao": "Athletic Club",
    "m'gladbach": "Borussia Monchengladbach", "ein frankfurt": "Eintracht Frankfurt", "sociedad": "Real Sociedad",
    "celta": "Celta Vigo", "vallecano": "Rayo Vallecano", "espanol": "Espanyol", "sp gijon": "Sporting Gijon",
    "la coruna": "Deportivo La Coruna", "st etienne": "Saint-Etienne", "hertha": "Hertha Berlin",
    "verona": "Hellas Verona", "preston": "Preston North End", "blackburn": "Blackburn Rovers",
    "plymouth": "Plymouth Argyle", "peterboro": "Peterborough United", "luton": "Luton Town", "derby": "Derby County",
    "huddersfield": "Huddersfield Town", "rotherham": "Rotherham United", "charlton": "Charlton Athletic",
    "wigan": "Wigan Athletic", "birmingham": "Birmingham City", "swansea": "Swansea City",
    "oxford": "Oxford United", "bielefeld": "Arminia Bielefeld",
}


def season_code(season: int) -> str:
    """2025 -> "2526" (the file folder of the 2025/26 season)."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def file_url(div: str, season: int) -> str:
    return f"{BASE}/mmz4281/{season_code(season)}/{div}.csv"


def new_url(country: str) -> str:
    return f"{BASE}/new/{country}.csv"


def fd_name(name: str) -> str:
    """football-data's team name in a spelling team_key folds like ESPN's."""
    plain = " ".join(str(name or "").split())
    return FD_ALIASES.get(plain.lower(), plain)


def fd_team_key(league: str, name: str) -> Optional[str]:
    return team_key(league, fd_name(name))


def _key_slug(key: Optional[str], name: str) -> str:
    return (key or name.lower()).replace(" ", "-")


def _date(text: str) -> Optional[datetime.date]:
    text = (text or "").strip()
    for pattern in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    return None


def kickoff(date_text: str, time_text: str) -> Optional[Tuple[float, bool]]:
    """(epoch seconds, time known) of a UK-local date and time; noon when the time is missing."""
    day = _date(date_text)
    if day is None:
        return None
    known = True
    try:
        hour, minute = (int(part) for part in (time_text or "").strip().split(":")[:2])
    except ValueError:
        hour, minute, known = 12, 0, False
    moment = datetime.datetime(day.year, day.month, day.day, hour, minute, tzinfo=UK)
    return moment.timestamp(), known


def open_as_of(start_ts: float) -> float:
    """When a pre-closing price was on the board at the latest: the last Tuesday or Friday 18:00 UK time at least
    an hour before kick-off."""
    limit = datetime.datetime.fromtimestamp(start_ts - 3600, UK)
    day = limit.date()
    for back in range(0, 8):
        candidate = day - datetime.timedelta(days=back)
        if candidate.weekday() not in OPEN_DAYS:
            continue
        moment = datetime.datetime(candidate.year, candidate.month, candidate.day, OPEN_HOUR, tzinfo=UK)
        if moment <= limit:
            return moment.timestamp()
    return start_ts - 7 * 86400.0  # not reached: one of the last 8 days is a Tuesday or a Friday at 18:00


def _kept() -> frozenset:
    names = {"Div", "Country", "League", "Season", "Date", "Time", "HomeTeam", "AwayTeam", "Home", "Away", "FTHG", "FTAG",
             "HG", "AG", "FTR", "Res", "Referee", "AHh", "AHCh"}
    names.update(column for column, *_ in COLUMNS)
    for pair in STATS:
        names.update(pair)
    return frozenset(names)


KEPT = _kept()


def parse_csv(text: str) -> List[Dict[str, str]]:
    """A football-data CSV as dicts of the columns read here (KEPT: the cached value stays small), header names
    stripped, a byte-order mark dropped, empty cells and rows skipped."""
    text = (text or "").lstrip("﻿")
    reader = csv.reader(io.StringIO(text))
    try:
        header = [name.strip().lstrip("﻿") for name in next(reader)]
    except StopIteration:
        return []
    rows = []
    for values in reader:
        if not any(value.strip() for value in values):
            continue
        row = {name: value.strip() for name, value in zip(header, values) if name in KEPT and value.strip()}
        if row:
            rows.append(row)
    return rows


def _lines(row: Mapping[str, str], start_ts: float, pinnacle_ok: bool) -> List[Dict[str, Any]]:
    lines = []
    opened = open_as_of(start_ts)
    for column, kind, book, market, side, point_column in COLUMNS:
        if book == "pinnacle" and not pinnacle_ok:
            continue
        price = num(row.get(column))
        if price is None or price <= 1.0:
            continue
        point: Optional[float] = None
        if point_column == "2.5":
            point = 2.5
        elif point_column:
            handicap = num(row.get(point_column))
            if handicap is None:
                continue
            point = handicap if side == "home" else -handicap  # AHh is the home team's handicap
        lines.append({"kind": kind, "book": book, "market": market, "side": side, "point": point, "price": price,
                      "as_of": (start_ts - CLOSE_LEAD) if kind == "close" else opened})
    return lines


def _stats(row: Mapping[str, str]) -> List[Dict[str, Any]]:
    stats = []
    for (home_col, away_col), stat in STATS.items():
        for column, side in ((home_col, "home"), (away_col, "away")):
            value = num(row.get(column))
            if value is not None:
                stats.append({"team_side": side, "stat": stat, "value": value})
    return stats


def record(row: Mapping[str, str], league: str, season: int, div: str) -> Optional[Dict[str, Any]]:
    """One CSV row as a history record ({"game", "lines", "stats", "weather"}); None for a row with no result."""
    home_name = row.get("HomeTeam") or row.get("Home") or ""
    away_name = row.get("AwayTeam") or row.get("Away") or ""
    home_goals = num(row.get("FTHG", row.get("HG")))
    away_goals = num(row.get("FTAG", row.get("AG")))
    when = kickoff(row.get("Date", ""), row.get("Time", ""))
    if not home_name or not away_name or home_goals is None or away_goals is None or when is None:
        return None
    start_ts, time_known = when
    day = datetime.datetime.fromtimestamp(start_ts, UK).date()
    home_key, away_key = fd_team_key(league, home_name), fd_team_key(league, away_name)
    info: Dict[str, Any] = {"div": div, "referee": row.get("Referee") or None, "result": row.get("FTR") or None}
    if not time_known:
        info["time_unknown"] = True
    ht_home, ht_away = num(row.get("HTHG")), num(row.get("HTAG"))
    if ht_home is not None and ht_away is not None:
        info["half_time"] = {"home": int(ht_home), "away": int(ht_away)}
    game = {"league": league, "season": season, "source": SOURCE, "status": "final", "start": iso(start_ts),
            "game_key": f"fd:{div}:{day:%Y%m%d}:{_key_slug(home_key, home_name)}:{_key_slug(away_key, away_name)}",
            "espn_id": None, "home": home_name, "away": away_name, "home_score": int(home_goals),
            "away_score": int(away_goals), "home_key": home_key, "away_key": away_key, "info": info}
    pinnacle_ok = day < PINNACLE_UNRELIABLE_FROM
    return {"game": game, "lines": _lines(row, start_ts, pinnacle_ok), "stats": _stats(row), "weather": []}


def season_of(text: str) -> Optional[int]:
    """The "new" files' Season column ("2026", "2012/2013") as the season's first year."""
    head = str(text or "").strip().split("/")[0]
    return int(head) if head.isdigit() else None


class FootballData:
    """football-data.co.uk through the shared cache. ``season_records(div, season, current)`` and
    ``new_records(country, first_season, current)`` give history records; ``current`` marks a season still being
    played (its file is cached 12 hours, a finished one's 7 days)."""

    def __init__(self, http: HistHttp, cache: Any) -> None:
        self.http = http
        self.cache = cache

    def _rows(self, url: str, key: Tuple[Any, ...], current: bool) -> List[Dict[str, str]]:
        return cached(self.cache, "hist_file", key, lambda: parse_csv(self.http.text(url)),
                      "current" if current else "past")

    def season_records(self, div: str, season: int, current: bool = False) -> List[Dict[str, Any]]:
        league = DIVISIONS[div]
        rows = self._rows(file_url(div, season), ("hist", PARSE_VERSION, "football-data", div, season), current)
        found = []
        for row in rows:
            item = record(row, league, season, div)
            if item is not None:
                found.append(item)
        return found

    def new_records(self, country: str, first_season: int = FIRST_SEASON,
                    current: bool = True) -> List[Dict[str, Any]]:
        league, name = NEW_LEAGUES[country]
        rows = self._rows(new_url(country), ("hist", PARSE_VERSION, "football-data", "new", country), current)
        found = []
        for row in rows:
            season = season_of(row.get("Season", ""))
            if season is None or season < first_season or (row.get("League") or name) != name:
                continue
            item = record(row, league, season, country)
            if item is not None:
                found.append(item)
        return found
