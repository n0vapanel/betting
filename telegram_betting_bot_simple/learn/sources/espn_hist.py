"""ESPN's past games for the leagues the app follows (NBA, NHL, MLB, college football and basketball), and MLB
StatsAPI's season schedule.

- **Event ids:** ``sports.core.api.espn.com/v2/sports/{s}/leagues/{l}/events?dates=A-B&limit=1000`` (a date range,
  paged), one window at a time (``events``).
- **Result and box stats:** the game summary (``summary``: ``site.api.espn.com/.../summary?event=``), parsed to the
  score, the per-period scores and a curated set of team box stats per sport (TEAM_STATS).
- **Odds:** ``.../events/{id}/competitions/{id}/odds`` (``odds``). Point in time: only each book's ``open`` and
  ``close`` blocks are read, never ``current`` (often in-play); the live-odds providers (ids 59 and 46, any name with
  "live") and projection sites (numberfire, teamrankings, accuscore) are dropped. Old-era "Opening" (id 23) gives
  the opener through its ``current``. Rows kept: ESPN's own books (ESPN BET, DraftKings) and the median across every
  kept book (``median``: the most common point, then the median price at it), which shrugs off one book's odd line.
  A close is known a minute before the start at the latest (``lead`` 60 s), an open two minutes before (120 s):
  upper bounds, the real time is earlier.
- **MLB** (StatsAPI's terms are "non-bulk"): one season schedule request (``mlb_season``: results, the inning
  linescore, MLB's announced game-time weather), ESPN's scoreboard per date for the ESPN ids (``scoreboard``) and
  ESPN odds. No box scores.

Every read goes through the shared TTLCache: kind "hist_list" (event lists and scoreboards; a past window's for 7
days), "hist_game" (a finished game's summary and odds, 2 days: the hist_* tables keep them) and "hist_file" (MLB's
season schedule). The cached values are the parsed, compact forms.
"""

from __future__ import annotations

import datetime
import re
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ...data.cache import ttl_for
from ...data.espn_client import espn_path
from ...data.mlb_stats import parse_weather
from ..history_store import num
from .http import HistHttp, cached

CORE = "https://sports.core.api.espn.com/v2/sports"
SITE = "https://site.api.espn.com/apis/site/v2/sports"
MLB_SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
PARSE_VERSION = 1
SOURCE = "espn"

# the app's league -> (ESPN sport, league path, season start (month, day), season end (month, day))
LEAGUES: Dict[str, Tuple[str, str, Tuple[int, int], Tuple[int, int]]] = {
    "basketball_nba": ("basketball", "nba", (10, 1), (6, 30)),
    "icehockey_nhl": ("hockey", "nhl", (10, 1), (6, 30)),
    "basketball_ncaab": ("basketball", "mens-college-basketball", (11, 1), (4, 15)),
    "americanfootball_ncaaf": ("football", "college-football", (8, 20), (1, 25)),
    "baseball_mlb": ("baseball", "mlb", (3, 15), (11, 15)),
}
LIVE_PROVIDERS = {"59", "46"}  # "ESPN Bet - Live Odds", "Caesars ... Live Odds": in-play prices
NOT_BOOKS = ("numberfire", "teamrankings", "accuscore")  # projection sites, not prices
OWN_BOOKS = {"espn bet": "espn_bet", "draftkings": "draftkings"}
OPENING_PROVIDER = "23"
CLOSE_LEAD = 60.0
OPEN_LEAD = 120.0
MEDIAN_BOOK = "median"

# ESPN team box stat name -> ours, per sport ("a-b" names split into two)
TEAM_STATS: Dict[str, Dict[str, Any]] = {
    "soccer": {"wonCorners": "corners", "yellowCards": "yellow", "redCards": "red", "totalShots": "shots",
               "shotsOnTarget": "shots_on_target", "foulsCommitted": "fouls", "possessionPct": "possession",
               "offsides": "offsides", "saves": "saves"},
    "basketball": {"fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
                   "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
                   "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"), "totalRebounds": "rebounds",
                   "offensiveRebounds": "offensive_rebounds", "assists": "assists", "steals": "steals",
                   "blocks": "blocks", "totalTurnovers": "turnovers", "fouls": "fouls",
                   "pointsInPaint": "points_in_paint", "fastBreakPoints": "fast_break_points"},
    "hockey": {"shotsTotal": "shots", "hits": "hits", "blockedShots": "blocked_shots",
               "powerPlayGoals": "power_play_goals", "powerPlayOpportunities": "power_play_opportunities",
               "penaltyMinutes": "pim", "faceoffsWon": "faceoffs_won", "giveaways": "giveaways",
               "takeaways": "takeaways"},
    "football": {"firstDowns": "first_downs", "totalYards": "total_yards", "netPassingYards": "passing_yards",
                 "rushingYards": "rushing_yards", "rushingAttempts": "rushing_attempts", "turnovers": "turnovers",
                 "totalPenaltiesYards": ("penalties", "penalty_yards"), "sacksYardsLost": ("sacks", "sack_yards"),
                 "thirdDownEff": ("third_down_conversions", "third_down_attempts"),
                 "possessionTime": "possession_seconds"},
    "baseball": {"batting.hits": "hits", "batting.runs": "runs", "batting.homeRuns": "home_runs",
                 "batting.walks": "walks", "batting.strikeouts": "strikeouts", "fielding.errors": "errors"},
}


def sport_of(league: str) -> Optional[str]:
    """ESPN's sport ("basketball", "hockey", "football", "baseball", "soccer") of an app league; None if unknown."""
    if league in LEAGUES:
        return LEAGUES[league][0]
    found = espn_path(league)
    return found[0] if found else ("soccer" if league.startswith("soccer_") else None)


def espn_league(league: str) -> Tuple[str, str]:
    sport, path, _start, _end = LEAGUES[league]
    return sport, path


def season_dates(league: str, season: int) -> Tuple[datetime.date, datetime.date]:
    """The first and last date of ``season`` (the calendar year it starts in) for the backfill's windows."""
    _sport, _path, (m1, d1), (m2, d2) = LEAGUES[league]
    first = datetime.date(season, m1, d1)
    last = datetime.date(season + (1 if (m2, d2) < (m1, d1) else 0), m2, d2)
    return first, last


def current_season(league: str, today: datetime.date) -> int:
    """The latest season that has started by ``today``."""
    _sport, _path, (m1, d1), _end = LEAGUES[league]
    return today.year if (today.month, today.day) >= (m1, d1) else today.year - 1


# --- prices -------------------------------------------------------------------------------------------------------

def american(value: Any) -> Optional[float]:
    """An American price ("+205", "-110", -250, "EVEN") as decimal odds; None when unreadable."""
    if isinstance(value, str) and value.strip().upper() in ("EVEN", "EV", "PK"):
        return 2.0
    number = num(str(value).replace("+", "")) if isinstance(value, str) else num(value)
    if number is None or -100 < number < 100:
        return None
    return round(1 + (number / 100.0 if number > 0 else 100.0 / -number), 4)


def _price(block: Any) -> Optional[float]:
    """A price block ({"american": "-110"} or {"value"/"decimal": 1.91}) as decimal odds."""
    if not isinstance(block, Mapping):
        return american(block) if isinstance(block, (str, int, float)) else None
    found = american(block.get("american") or block.get("alternateDisplayValue") or block.get("odds"))
    if found is not None:
        return found
    decimal = num(block.get("decimal") or block.get("value"))
    return decimal if decimal is not None and decimal > 1.0 else None


def _point(block: Any) -> Optional[float]:
    if isinstance(block, Mapping):
        block = block.get("american") or block.get("alternateDisplayValue") or block.get("line")
    if isinstance(block, str):
        text = block.strip().lower().lstrip("ou").replace("+", "")
        if text in ("pk", "even"):
            return 0.0
        return num(text)
    return num(block)


def book_name(provider: Mapping[str, Any]) -> Optional[str]:
    """A kept provider's book key ("espn_bet", "draftkings", "caesars sportsbook (new jersey)"...); None for the
    live-odds providers and projection sites."""
    pid = str(provider.get("id") or "")
    name = str(provider.get("name") or "").strip()
    lowered = name.lower()
    if pid in LIVE_PROVIDERS or "live" in lowered or any(word in lowered for word in NOT_BOOKS):
        return None
    if not lowered:
        return None
    return OWN_BOOKS.get(lowered, lowered)


def _item_lines(item: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One provider's open and close prices (never ``current``, except "Opening"'s, which is the opener)."""
    lines: List[Dict[str, Any]] = []
    blocks: List[Tuple[str, str]] = [("open", "open"), ("close", "close")]
    if str((item.get("provider") or {}).get("id") or "") == OPENING_PROVIDER and not any(
            isinstance(item.get(k), Mapping) for k in ("open", "close")):
        blocks = [("open", "current")]
    for kind, block in blocks:
        lead = CLOSE_LEAD if kind == "close" else OPEN_LEAD
        for side, team in (("home", "homeTeamOdds"), ("away", "awayTeamOdds"), ("draw", "drawOdds")):
            odds = (item.get(team) or {}).get(block) if isinstance(item.get(team), Mapping) else None
            if not isinstance(odds, Mapping):
                continue
            price = _price(odds.get("moneyLine"))
            if price is not None:
                lines.append({"kind": kind, "market": "h2h", "side": side, "point": None, "price": price,
                              "lead": lead})
            if side != "draw":
                point, spread_price = _point(odds.get("pointSpread")), _price(odds.get("spread"))
                if point is not None and spread_price is not None:
                    lines.append({"kind": kind, "market": "spreads", "side": side, "point": point,
                                  "price": spread_price, "lead": lead})
        total = item.get(block) if isinstance(item.get(block), Mapping) else None
        if total:
            point = _point(total.get("total"))
            for side in ("over", "under"):
                price = _price(total.get(side))
                if point is not None and price is not None:
                    lines.append({"kind": kind, "market": "totals", "side": side, "point": point, "price": price,
                                  "lead": lead})
    return lines


def _median_lines(by_book: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """Per (kind, market, side): the most common point across books (ties: the one nearest the median), then the
    median price of the books at that point."""
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for lines in by_book.values():
        for line in lines:
            groups.setdefault((line["kind"], line["market"], line["side"]), []).append(line)
    found = []
    for (kind, market, side), lines in sorted(groups.items()):
        if market == "h2h":
            chosen = lines
            point = None
        else:
            points = [line["point"] for line in lines]
            middle = statistics.median(points)
            counts: Dict[float, int] = {}
            for value in points:
                counts[value] = counts.get(value, 0) + 1
            point = sorted(counts, key=lambda value: (-counts[value], abs(value - middle), value))[0]
            chosen = [line for line in lines if line["point"] == point]
        found.append({"kind": kind, "market": market, "side": side, "point": point,
                      "price": round(statistics.median(line["price"] for line in chosen), 4),
                      "lead": max(line["lead"] for line in chosen), "n": len(chosen)})
    return found


def parse_odds(data: Any) -> Dict[str, Any]:
    """The core odds response as {"lines": [{book, kind, market, side, point, price, lead}], "books": [...],
    "dropped": [...]}: ESPN's own books and the median across kept books; ``lead`` is seconds before the start."""
    by_book: Dict[str, List[Dict[str, Any]]] = {}
    dropped = []
    for item in (data or {}).get("items") or [] if isinstance(data, Mapping) else []:
        if not isinstance(item, Mapping):
            continue
        provider = item.get("provider") or {}
        book = book_name(provider)
        if book is None:
            dropped.append(str(provider.get("name") or provider.get("id")))
            continue
        lines = _item_lines(item)
        if lines:
            by_book.setdefault(book, []).extend(lines)
    out: List[Dict[str, Any]] = []
    for book in sorted(by_book):
        if book in OWN_BOOKS.values():
            out.extend({**line, "book": book} for line in by_book[book])
    if by_book:
        out.extend({**line, "book": MEDIAN_BOOK} for line in _median_lines(by_book))
    return {"lines": out, "books": sorted(by_book), "dropped": sorted(set(dropped))}


def parse_pickcenter(pickcenter: Any) -> List[Dict[str, Any]]:
    """A summary's ``pickcenter`` (today's books: DraftKings' open and close per side) as lines with ``lead``."""
    lines: List[Dict[str, Any]] = []
    for pick in pickcenter or []:
        if not isinstance(pick, Mapping):
            continue
        book = book_name(pick.get("provider") or {})
        if book is None:
            continue
        for kind in ("open", "close"):
            lead = CLOSE_LEAD if kind == "close" else OPEN_LEAD
            moneyline = pick.get("moneyline") or {}
            for side in ("home", "away", "draw"):
                price = _price(((moneyline.get(side) or {}).get(kind) or {}).get("odds"))
                if price is not None:
                    lines.append({"book": book, "kind": kind, "market": "h2h", "side": side, "point": None,
                                  "price": price, "lead": lead})
            spread = pick.get("pointSpread") or {}
            for side in ("home", "away"):
                block = (spread.get(side) or {}).get(kind) or {}
                point, price = _point(block.get("line")), _price(block.get("odds"))
                if point is not None and price is not None:
                    lines.append({"book": book, "kind": kind, "market": "spreads", "side": side, "point": point,
                                  "price": price, "lead": lead})
            total = pick.get("total") or {}
            for side in ("over", "under"):
                block = (total.get(side) or {}).get(kind) or {}
                point, price = _point(block.get("line")), _price(block.get("odds"))
                if point is not None and price is not None:
                    lines.append({"book": book, "kind": kind, "market": "totals", "side": side, "point": point,
                                  "price": price, "lead": lead})
    return lines


# --- summaries ----------------------------------------------------------------------------------------------------

def _clock_seconds(text: str) -> Optional[float]:
    parts = str(text or "").split(":")
    if len(parts) != 2:
        return None
    minutes, seconds = num(parts[0]), num(parts[1])
    return None if minutes is None or seconds is None else minutes * 60 + seconds


def team_stats(sport: str, team: Mapping[str, Any]) -> Dict[str, float]:
    """One boxscore team's curated stats (TEAM_STATS) as {stat: value}."""
    wanted = TEAM_STATS.get(sport, {})
    found: Dict[str, float] = {}
    entries: List[Tuple[str, Any]] = []
    for item in team.get("statistics") or []:
        if not isinstance(item, Mapping):
            continue
        if isinstance(item.get("stats"), list):  # baseball: groups of stats
            for stat in item["stats"]:
                if isinstance(stat, Mapping):
                    entries.append((f"{item.get('name')}.{stat.get('name')}", stat.get("displayValue",
                                                                                        stat.get("value"))))
        else:
            entries.append((str(item.get("name")), item.get("displayValue", item.get("value"))))
    for name, value in entries:
        target = wanted.get(name)
        if target is None or name in found:
            continue
        if isinstance(target, tuple):
            parts = re.split(r"[-/]", str(value or ""))
            if len(parts) == 2 and all(num(part) is not None for part in parts):
                found[target[0]], found[target[1]] = num(parts[0]), num(parts[1])  # type: ignore[assignment]
        elif target == "possession_seconds":
            seconds = _clock_seconds(value)
            if seconds is not None:
                found[target] = seconds
        else:
            number = num(value)
            if number is not None:
                found[target] = number
    return found


def _period_scores(competitor: Mapping[str, Any]) -> List[float]:
    found = []
    for period in competitor.get("linescores") or []:
        value = num(period.get("value", period.get("displayValue"))) if isinstance(period, Mapping) else None
        if value is None:
            return []
        found.append(value)
    return found


def parse_summary(data: Any, league: str) -> Dict[str, Any]:
    """A game summary as {"id", "start", "state" (final|void|pending), "status", "home"/"away" ({id, name, abbr,
    score}), "periods" ({"home": [...], "away": [...]}), "stats" ({"home": {...}, "away": {...}}), "neutral",
    "venue", "after_extra_time"}. Soccer's score is the 90-minute one (the two halves) after extra time."""
    data = data if isinstance(data, Mapping) else {}
    competition = ((data.get("header") or {}).get("competitions") or [{}])[0] or {}
    status = (competition.get("status") or {}).get("type") or {}
    name = str(status.get("name") or "").upper()
    if any(word in name for word in ("POSTPONED", "CANCELED", "CANCELLED", "FORFEIT", "ABANDONED", "SUSPENDED")):
        state = "void"
    elif status.get("completed") or status.get("state") == "post":
        state = "final"
    else:
        state = "pending"
    sides: Dict[str, Dict[str, Any]] = {}
    periods: Dict[str, List[float]] = {}
    for competitor in competition.get("competitors") or []:
        side = competitor.get("homeAway") if isinstance(competitor, Mapping) else None
        if side not in ("home", "away"):
            continue
        team = competitor.get("team") or {}
        sides[side] = {"id": team.get("id"), "name": team.get("displayName") or team.get("name"),
                       "abbr": team.get("abbreviation"), "score": num(competitor.get("score"))}
        periods[side] = _period_scores(competitor)
    sport = sport_of(league)
    found: Dict[str, Any] = {"id": str(competition.get("id") or (data.get("header") or {}).get("id") or ""),
                             "start": competition.get("date"), "state": state, "status": name,
                             "home": sides.get("home"), "away": sides.get("away"), "periods": periods,
                             "neutral": bool(competition.get("neutralSite")), "stats": {"home": {}, "away": {}}}
    venue = ((data.get("gameInfo") or {}).get("venue") or {}).get("fullName")
    if venue:
        found["venue"] = venue
    if sport == "soccer" and state == "final" and sides.get("home") and sides.get("away"):
        extra = name in ("STATUS_FINAL_AET", "STATUS_FINAL_PEN") or any(len(periods.get(s) or []) > 2
                                                                         for s in ("home", "away"))
        if extra:
            if all(len(periods.get(s) or []) >= 2 for s in ("home", "away")):
                found["after_extra_time"] = {s: sides[s]["score"] for s in ("home", "away")}
                for s in ("home", "away"):
                    sides[s]["score"] = sum(periods[s][:2])
            else:
                found["state"] = "pending"  # no 90-minute score: never graded on the wrong one
    for team in (data.get("boxscore") or {}).get("teams") or []:
        if isinstance(team, Mapping) and team.get("homeAway") in ("home", "away"):
            found["stats"][team["homeAway"]] = team_stats(sport, team)
    return found


def summary_stats(parsed: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """A parsed summary's team stats and period scores as hist_stats rows."""
    rows = []
    for side in ("home", "away"):
        for stat, value in sorted(((parsed.get("stats") or {}).get(side) or {}).items()):
            rows.append({"team_side": side, "stat": stat, "value": value})
        for index, value in enumerate((parsed.get("periods") or {}).get(side) or [], start=1):
            rows.append({"team_side": side, "stat": f"period_{index}", "value": value})
    return rows


# --- scoreboards and MLB's season ---------------------------------------------------------------------------------

def parse_scoreboard(data: Any) -> List[Dict[str, Any]]:
    """A site scoreboard's events as [{"id", "start", "state", "home": {name, abbr, score}, "away": {...}}]."""
    found = []
    for event in (data or {}).get("events") or [] if isinstance(data, Mapping) else []:
        competition = (event.get("competitions") or [{}])[0] or {}
        parsed = parse_summary({"header": {"competitions": [{**competition, "id": event.get("id"),
                                                            "date": competition.get("date") or event.get("date")}]}},
                               "scoreboard")
        found.append({key: parsed[key] for key in ("id", "start", "state", "status", "home", "away")})
    return found


def parse_mlb_season(data: Any) -> List[Dict[str, Any]]:
    """StatsAPI's season schedule as compact games: {"pk", "start", "date", "state", "home"/"away" (name, score),
    "game_number", "double_header", "day_night", "venue", "weather", "innings" ({"home": [...], "away": [...]})}.
    A game listed twice (postponed, then played) keeps its final listing."""
    games: Dict[str, Dict[str, Any]] = {}
    for day in (data or {}).get("dates") or [] if isinstance(data, Mapping) else []:
        for game in day.get("games") or []:
            status = game.get("status") or {}
            detailed = str(status.get("detailedState") or "")
            if status.get("abstractGameState") == "Final" and detailed in ("Final", "Completed Early", "Game Over"):
                state = "final"
            elif detailed.startswith(("Postponed", "Cancelled", "Canceled")) or status.get("codedGameState") in (
                    "D", "C"):
                state = "void"
            else:
                state = "pending"
            teams = game.get("teams") or {}
            linescore = game.get("linescore") or {}
            innings: Dict[str, List[Optional[float]]] = {"home": [], "away": []}
            for inning in linescore.get("innings") or []:
                for side in ("home", "away"):
                    innings[side].append(num((inning.get(side) or {}).get("runs")))
            entry = {"pk": str(game.get("gamePk")), "start": game.get("gameDate"), "date": game.get("officialDate"),
                     "state": state, "status": detailed,
                     "home": {"name": ((teams.get("home") or {}).get("team") or {}).get("name"),
                              "score": num((teams.get("home") or {}).get("score"))},
                     "away": {"name": ((teams.get("away") or {}).get("team") or {}).get("name"),
                              "score": num((teams.get("away") or {}).get("score"))},
                     "game_number": game.get("gameNumber"), "double_header": game.get("doubleHeader"),
                     "day_night": game.get("dayNight"), "venue": (game.get("venue") or {}).get("name"),
                     "weather": parse_weather(game.get("weather")), "innings": innings}
            old = games.get(entry["pk"])
            if old is None or old["state"] != "final":
                games[entry["pk"]] = entry
    return sorted(games.values(), key=lambda g: (g.get("start") or "", g["pk"]))


# --- the client ---------------------------------------------------------------------------------------------------

_EVENT_ID = re.compile(r"/events/(\d+)")


def _final_life(value: Any) -> Any:
    state = value.get("state") if isinstance(value, Mapping) else None
    return ttl_for("hist_game", "final" if state in ("final", "void") else None)


class EspnHistory:
    """ESPN's past games and MLB's season schedule through the shared cache (see the module's docstring)."""

    def __init__(self, http: HistHttp, cache: Any) -> None:
        self.http = http
        self.cache = cache

    def events(self, league: str, first: datetime.date, last: datetime.date, past: bool = True) -> List[str]:
        """ESPN event ids of games dated ``first``..``last`` (inclusive), every page."""
        sport, path = espn_league(league)
        url = f"{CORE}/{sport}/leagues/{path}/events"

        def load() -> List[str]:
            ids: List[str] = []
            page = 1
            while True:
                data = self.http.json(url, {"dates": f"{first:%Y%m%d}-{last:%Y%m%d}", "limit": 1000, "page": page})
                for item in (data or {}).get("items") or []:
                    found = _EVENT_ID.search(str((item or {}).get("$ref") or ""))
                    if found and found.group(1) not in ids:
                        ids.append(found.group(1))
                if page >= int(num((data or {}).get("pageCount")) or 1) or page >= 20:
                    return ids
                page += 1

        return cached(self.cache, "hist_list", ("hist", PARSE_VERSION, "espn", "events", league, first.isoformat(),
                                                last.isoformat()), load, "past" if past else None)

    def summary(self, league: str, event_id: str) -> Dict[str, Any]:
        sport, path = espn_league(league)
        url = f"{SITE}/{sport}/{path}/summary"
        return cached(self.cache, "hist_game", ("hist", PARSE_VERSION, "espn", "summary", league, str(event_id)),
                      lambda: parse_summary(self.http.json(url, {"event": str(event_id)}), league), "final",
                      life=_final_life)

    def odds(self, league: str, event_id: str) -> Dict[str, Any]:
        sport, path = espn_league(league)
        url = f"{CORE}/{sport}/leagues/{path}/events/{event_id}/competitions/{event_id}/odds"
        return cached(self.cache, "hist_game", ("hist", PARSE_VERSION, "espn", "odds", league, str(event_id)),
                      lambda: parse_odds(self.http.json(url, {"limit": 100})), "final")

    def scoreboard(self, league: str, day: datetime.date, past: bool = True) -> List[Dict[str, Any]]:
        sport, path = espn_league(league)
        url = f"{SITE}/{sport}/{path}/scoreboard"
        return cached(self.cache, "hist_list", ("hist", PARSE_VERSION, "espn", "scoreboard", league,
                                                day.isoformat()),
                      lambda: parse_scoreboard(self.http.json(url, {"dates": f"{day:%Y%m%d}", "limit": 300})),
                      "past" if past else None)

    def mlb_season(self, season: int, current: bool = False) -> List[Dict[str, Any]]:
        params = {"sportId": 1, "season": season, "gameType": "R", "hydrate": "weather,linescore"}
        return cached(self.cache, "hist_file", ("hist", PARSE_VERSION, "mlb", "season", season),
                      lambda: parse_mlb_season(self.http.json(MLB_SCHEDULE, params)),
                      "current" if current else "past")
