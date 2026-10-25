"""Schedules, injuries, ESPN's matchup predictor and team/player stats from ESPN.

These are ESPN's public but undocumented JSON endpoints: they need no key,
but ESPN can change them without notice and doesn't license them for
commercial use, so callers treat every lookup as best-effort. Responses
are cached to keep the request rate low.
"""

from __future__ import annotations

import datetime
import re
import unicodedata
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

from .cache import TTLCache

SITE = "https://site.api.espn.com/apis/site/v2/sports"
STANDINGS = "https://site.api.espn.com/apis/v2/sports"
COMMON = "https://site.web.api.espn.com/apis/common/v3/sports"

# The Odds API league key -> ESPN (sport, league) path
LEAGUES = {
    "basketball_nba": ("basketball", "nba"),
    "basketball_wnba": ("basketball", "wnba"),
    "basketball_ncaab": ("basketball", "mens-college-basketball"),
    "americanfootball_nfl": ("football", "nfl"),
    "americanfootball_ncaaf": ("football", "college-football"),
    "icehockey_nhl": ("hockey", "nhl"),
    "baseball_mlb": ("baseball", "mlb"),
    "soccer_epl": ("soccer", "eng.1"),
    "soccer_spain_la_liga": ("soccer", "esp.1"),
    "soccer_italy_serie_a": ("soccer", "ita.1"),
    "soccer_germany_bundesliga": ("soccer", "ger.1"),
    "soccer_france_ligue_one": ("soccer", "fra.1"),
    "soccer_usa_mls": ("soccer", "usa.1"),
    "soccer_uefa_champs_league": ("soccer", "uefa.champions"),
}

# ESPN's scoreboard days follow US Eastern time
ESPN_DAY_TZ = ZoneInfo("America/New_York")

# Statuses that mean a player won't play
OUT_STATUSES = {"out", "doubtful", "suspension", "suspended", "injured reserve"}


def normalize_team(name: str) -> str:
    """Lowercase, accent-free, punctuation-free team name with common abbreviations expanded."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"^la ", "los angeles ", text.strip())
    return " ".join(word for word in text.split() if word not in {"fc", "cf", "afc", "sc"})


def team_match_score(name: str, team: Dict[str, Any]) -> float:
    """How well an odds-feed team name matches an ESPN team (1.0 = same name)."""
    target = normalize_team(name)
    best = 0.0
    for field in ("displayName", "shortDisplayName", "name", "nickname", "location"):
        candidate = normalize_team(team.get(field) or "")
        if not candidate:
            continue
        if candidate == target:
            return 1.0
        a, b = set(target.split()), set(candidate.split())
        best = max(best, len(a & b) / len(a | b))
    return best


class EspnClient:
    def __init__(self, cache: Optional[TTLCache] = None) -> None:
        self.cache = cache or TTLCache()

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = requests.get(url, params=params, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if not response.ok:
            raise RuntimeError(f"ESPN returned HTTP {response.status_code} for {url}")
        return response.json()

    def scoreboard(self, league: str, day: datetime.date) -> List[Dict[str, Any]]:
        """The league's events on ``day`` (a US Eastern date)."""
        sport, path = LEAGUES[league]
        ttl = 300 if day >= datetime.date.today() - datetime.timedelta(days=1) else 6 * 3600
        return self.cache.get(
            ("espn", "scoreboard", league, day),
            ttl,
            lambda: self._get(f"{SITE}/{sport}/{path}/scoreboard", {"dates": day.strftime("%Y%m%d")}).get("events", []),
        )

    def summary(self, league: str, event_id: str) -> Dict[str, Any]:
        """Game summary: injuries, ESPN predictor, leaders, last-five games and more."""
        sport, path = LEAGUES[league]
        return self.cache.get(
            ("espn", "summary", league, event_id),
            600,
            lambda: self._get(f"{SITE}/{sport}/{path}/summary", {"event": event_id}),
        )

    def standings(self, league: str, season: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """Per-team wins, losses and average points for/against, keyed by ESPN team id."""
        sport, path = LEAGUES[league]

        def load() -> Dict[str, Dict[str, float]]:
            data = self._get(f"{STANDINGS}/{sport}/{path}/standings", {"season": season} if season else None)
            teams = {}
            for group in data.get("children", []):
                for entry in group.get("standings", {}).get("entries", []):
                    stats = {stat.get("name"): stat.get("value") for stat in entry.get("stats", [])}
                    wins, losses = stats.get("wins") or 0, stats.get("losses") or 0
                    teams[entry["team"]["id"]] = {
                        "name": entry["team"].get("displayName"),
                        "games": wins + losses,
                        "wins": wins,
                        "losses": losses,
                        "points_for": stats.get("avgPointsFor"),
                        "points_against": stats.get("avgPointsAgainst"),
                    }
            return teams

        return self.cache.get(("espn", "standings", league, season), 3600 if season is None else 24 * 3600, load)

    def roster(self, league: str, team_id: str) -> List[Dict[str, Any]]:
        """The team's players: id, name, position and injury status ('' when healthy)."""
        sport, path = LEAGUES[league]

        def load() -> List[Dict[str, Any]]:
            data = self._get(f"{SITE}/{sport}/{path}/teams/{team_id}/roster")
            players = []
            for athlete in data.get("athletes", []):
                injuries = athlete.get("injuries") or []
                players.append({
                    "id": athlete["id"],
                    "name": athlete.get("displayName"),
                    "position": (athlete.get("position") or {}).get("abbreviation", ""),
                    "status": injuries[0].get("status", "") if injuries else "",
                })
            return players

        return self.cache.get(("espn", "roster", league, team_id), 3 * 3600, load)

    def player_seasons(self, league: str, athlete_id: str) -> List[Dict[str, Any]]:
        """Per-game averages for each season of an NBA/WNBA player's career, oldest first.

        A season split across teams after a trade is merged into one row,
        weighting each team's averages by games played.
        """
        sport, path = LEAGUES[league]

        def load() -> List[Dict[str, Any]]:
            data = self._get(f"{COMMON}/{sport}/{path}/athletes/{athlete_id}/stats")
            averages = next((c for c in data.get("categories", []) if c.get("name") == "averages"), None)
            if not averages:
                return []
            labels = averages.get("labels", [])
            seasons: Dict[str, Dict[str, float]] = {}
            for row in averages.get("statistics", []):
                values = dict(zip(labels, row.get("stats", [])))
                parsed = _basketball_averages(values)
                if not parsed["gp"]:
                    continue
                season = (row.get("season") or {}).get("displayName", "")
                merged = seasons.get(season)
                if merged:
                    games = merged["gp"] + parsed["gp"]
                    for key in parsed:
                        if key != "gp":
                            merged[key] = (merged[key] * merged["gp"] + parsed[key] * parsed["gp"]) / games
                    merged["gp"] = games
                else:
                    seasons[season] = parsed
            return [{"season": season, **stats} for season, stats in sorted(seasons.items())]

        return self.cache.get(("espn", "player", league, athlete_id), 12 * 3600, load)

    def find_game(self, league: str, home: str, away: str, commence_time: str) -> Optional[Dict[str, Any]]:
        """The ESPN event for an odds-feed game: matched on date and both team names."""
        if league not in LEAGUES:
            return None
        start = datetime.datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        days = {start.astimezone(ESPN_DAY_TZ).date(), start.date()}
        for day in sorted(days):
            try:
                events = self.scoreboard(league, day)
            except Exception:
                continue
            for event in events:
                competitors = event.get("competitions", [{}])[0].get("competitors", [])
                sides = {c.get("homeAway"): c for c in competitors}
                if "home" not in sides or "away" not in sides:
                    continue
                home_team, away_team = sides["home"]["team"], sides["away"]["team"]
                if team_match_score(home, home_team) >= 0.5 and team_match_score(away, away_team) >= 0.5:
                    return {
                        "event_id": event["id"],
                        "home_id": home_team["id"],
                        "away_id": away_team["id"],
                        "day": day,
                    }
        return None

    def played_on(self, league: str, day: datetime.date) -> set:
        """ESPN ids of the teams that played on ``day`` (for back-to-back detection)."""
        try:
            events = self.scoreboard(league, day)
        except Exception:
            return set()
        return {
            competitor["team"]["id"]
            for event in events
            for competitor in event.get("competitions", [{}])[0].get("competitors", [])
        }


def _number(text: Any) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _basketball_averages(values: Dict[str, Any]) -> Dict[str, float]:
    """Parse ESPN basketball per-game averages ("3PT" is "made-attempted")."""
    three = str(values.get("3PT", "0-0")).split("-")[0]
    return {
        "gp": _number(values.get("GP")),
        "min": _number(values.get("MIN")),
        "pts": _number(values.get("PTS")),
        "reb": _number(values.get("REB")),
        "ast": _number(values.get("AST")),
        "fg3m": _number(three),
        "stl": _number(values.get("STL")),
        "blk": _number(values.get("BLK")),
        "tov": _number(values.get("TO")),
    }
