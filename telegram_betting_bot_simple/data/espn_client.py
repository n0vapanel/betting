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
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

from .cache import TTLCache

SITE = "https://site.api.espn.com/apis/site/v2/sports"
STANDINGS = "https://site.api.espn.com/apis/v2/sports"
COMMON = "https://site.web.api.espn.com/apis/common/v3/sports"

# The Odds API league key -> ESPN (sport, league) path, for team sports: each ESPN event is one game between a
# home and an away team, so the grader can settle bets from it (find_game, game_result). Every path here was
# checked against ESPN's scoreboard on 26 October 2026 (a wrong one answers HTTP 400; ESPN has none for
# Germany's 3. Liga, Poland's Ekstraklasa, the AHL, KBO, the Swedish and Finnish hockey leagues, and no games
# for the Euroleague).
LEAGUES = {
    "basketball_nba": ("basketball", "nba"),
    "basketball_wnba": ("basketball", "wnba"),
    "basketball_ncaab": ("basketball", "mens-college-basketball"),
    "basketball_wncaab": ("basketball", "womens-college-basketball"),
    "basketball_nbl": ("basketball", "nbl"),
    "americanfootball_nfl": ("football", "nfl"),
    "americanfootball_ncaaf": ("football", "college-football"),
    "americanfootball_cfl": ("football", "cfl"),
    "icehockey_nhl": ("hockey", "nhl"),
    "baseball_mlb": ("baseball", "mlb"),
    "soccer_epl": ("soccer", "eng.1"),
    "soccer_efl_champ": ("soccer", "eng.2"),
    "soccer_england_league1": ("soccer", "eng.3"),
    "soccer_england_league2": ("soccer", "eng.4"),
    "soccer_england_efl_cup": ("soccer", "eng.league_cup"),
    "soccer_spl": ("soccer", "sco.1"),
    "soccer_spain_la_liga": ("soccer", "esp.1"),
    "soccer_spain_segunda_division": ("soccer", "esp.2"),
    "soccer_italy_serie_a": ("soccer", "ita.1"),
    "soccer_italy_serie_b": ("soccer", "ita.2"),
    "soccer_germany_bundesliga": ("soccer", "ger.1"),
    "soccer_germany_bundesliga2": ("soccer", "ger.2"),
    "soccer_germany_dfb_pokal": ("soccer", "ger.dfb_pokal"),
    "soccer_france_ligue_one": ("soccer", "fra.1"),
    "soccer_france_ligue_two": ("soccer", "fra.2"),
    "soccer_netherlands_eredivisie": ("soccer", "ned.1"),
    "soccer_portugal_primeira_liga": ("soccer", "por.1"),
    "soccer_belgium_first_div": ("soccer", "bel.1"),
    "soccer_turkey_super_league": ("soccer", "tur.1"),
    "soccer_austria_bundesliga": ("soccer", "aut.1"),
    "soccer_switzerland_superleague": ("soccer", "sui.1"),
    "soccer_denmark_superliga": ("soccer", "den.1"),
    "soccer_greece_super_league": ("soccer", "gre.1"),
    "soccer_norway_eliteserien": ("soccer", "nor.1"),
    "soccer_sweden_allsvenskan": ("soccer", "swe.1"),
    "soccer_sweden_superettan": ("soccer", "swe.2"),
    "soccer_finland_veikkausliiga": ("soccer", "fin.1"),
    "soccer_league_of_ireland": ("soccer", "irl.1"),
    "soccer_russia_premier_league": ("soccer", "rus.1"),
    "soccer_saudi_arabia_pro_league": ("soccer", "ksa.1"),
    "soccer_japan_j_league": ("soccer", "jpn.1"),
    "soccer_australia_aleague": ("soccer", "aus.1"),
    "soccer_usa_mls": ("soccer", "usa.1"),
    "soccer_mexico_ligamx": ("soccer", "mex.1"),
    "soccer_brazil_campeonato": ("soccer", "bra.1"),
    "soccer_brazil_serie_b": ("soccer", "bra.2"),
    "soccer_argentina_primera_division": ("soccer", "arg.1"),
    "soccer_chile_campeonato": ("soccer", "chi.1"),
    "soccer_conmebol_copa_libertadores": ("soccer", "conmebol.libertadores"),
    "soccer_conmebol_copa_sudamericana": ("soccer", "conmebol.sudamericana"),
    "soccer_uefa_champs_league": ("soccer", "uefa.champions"),
    "soccer_uefa_europa_league": ("soccer", "uefa.europa"),
    "soccer_uefa_europa_conference_league": ("soccer", "uefa.europa.conf"),
    "soccer_uefa_champs_league_women": ("soccer", "uefa.wchampions"),
    "soccer_uefa_nations_league": ("soccer", "uefa.nations"),
}

# Scoreboards the all-sports slate (data/slate.py) shows but the grader can't settle from: an ESPN tennis event
# is a tournament of matches and an MMA event a card of fights, not one home/away game. The tennis keys are
# ESPN's whole tours (The Odds API keys each tournament on its own, e.g. tennis_atp_vienna_open); MMA is The
# Odds API's key. Checked on ESPN's scoreboard on 26 October 2026.
SCOREBOARD_ONLY = {
    "tennis_atp": ("tennis", "atp"),
    "tennis_wta": ("tennis", "wta"),
    "mma_mixed_martial_arts": ("mma", "ufc"),
}

# Extra scoreboard query parameters: college basketball lists only featured games unless asked for all of
# Division I ("groups": 50)
SCOREBOARD_PARAMS: Dict[str, Dict[str, str]] = {
    "basketball_ncaab": {"groups": "50"},
    "basketball_wncaab": {"groups": "50"},
}
SCOREBOARD_LIMIT = 300  # events per scoreboard request (ESPN's default can cut a busy college Saturday short)


def espn_path(league: str) -> Optional[Tuple[str, str]]:
    """ESPN's (sport, league) path for a league key, from LEAGUES or SCOREBOARD_ONLY; None when ESPN has none."""
    return LEAGUES.get(league) or SCOREBOARD_ONLY.get(league)

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
        ttl = 300 if day >= datetime.date.today() - datetime.timedelta(days=1) else 6 * 3600
        return self.cache.get(("espn", "scoreboard", league, day), ttl, lambda: self.fetch_scoreboard(league, day))

    def fetch_scoreboard(self, league: str, day: datetime.date) -> List[Dict[str, Any]]:
        """The league's events on ``day`` straight from ESPN, not cached: the all-sports slate (data/slate.py)
        keeps its own trimmed copy of every league's boards, which would pile up here as raw JSON. ESPN takes
        one day at a time (a date range is a 400)."""
        target = espn_path(league)
        if target is None:
            raise KeyError(league)
        sport, path = target
        params = {"dates": day.strftime("%Y%m%d"), "limit": SCOREBOARD_LIMIT, **SCOREBOARD_PARAMS.get(league, {})}
        return self._get(f"{SITE}/{sport}/{path}/scoreboard", params).get("events", [])

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

    def game_result(self, league: str, event_id: str) -> Dict[str, Any]:
        """How a game stands: {"state": "final", "home": pts, "away": pts}, {"state": "void"} if it
        was postponed or cancelled, or {"state": "pending"} while it hasn't finished.

        Soccer bets (the three-way moneyline, spreads and totals) settle on 90 minutes plus stoppage
        time, but ESPN's headline score after extra time or penalties includes the extra-time goals.
        Then the score is the two halves' (the first two linescores), with the full one under
        "after_extra_time"; if the halves can't be read the game stays pending, so it is voided in the
        end rather than graded on the wrong score.
        """
        competition = (self.summary(league, event_id).get("header", {}).get("competitions") or [{}])[0]
        status = (competition.get("status") or {}).get("type") or {}
        name = status.get("name", "")
        if "POSTPONED" in name or "CANCELED" in name or "CANCELLED" in name or "FORFEIT" in name:
            return {"state": "void", "status": name}
        if not status.get("completed"):
            return {"state": "pending", "status": name}
        sides = {c.get("homeAway"): c for c in competition.get("competitors", [])}
        try:
            final = {"state": "final", "home": int(sides["home"]["score"]), "away": int(sides["away"]["score"])}
        except (KeyError, TypeError, ValueError):
            return {"state": "pending", "status": name}
        extra_time = name in ("STATUS_FINAL_AET", "STATUS_FINAL_PEN") or any(
            len(sides[side].get("linescores") or []) > 2 for side in ("home", "away"))
        if LEAGUES.get(league, ("",))[0] == "soccer" and extra_time:
            try:
                halves = {side: sum(_goals(period) for period in (sides[side].get("linescores") or [])[:2])
                          for side in ("home", "away")}
                if any(len(sides[side].get("linescores") or []) < 2 for side in ("home", "away")):
                    raise ValueError("no score for each half")
            except (TypeError, ValueError):
                return {"state": "pending", "status": name}
            return {**final, **halves, "after_extra_time": {"home": final["home"], "away": final["away"]},
                    "status": name}
        return final

    def box_score(self, league: str, event_id: str) -> Dict[str, Dict[str, Any]]:
        """Each player's stat line in a basketball game, keyed by ESPN athlete id."""
        lines: Dict[str, Dict[str, Any]] = {}
        for team in self.summary(league, event_id).get("boxscore", {}).get("players", []):
            for group in team.get("statistics", []):
                labels = group.get("labels") or group.get("names") or []
                for entry in group.get("athletes", []):
                    athlete = entry.get("athlete") or {}
                    values = dict(zip(labels, entry.get("stats") or []))
                    played = bool(entry.get("stats")) and not entry.get("didNotPlay")
                    line = _basketball_averages({**values, "GP": 1}) if played else {}
                    lines[str(athlete.get("id"))] = {**line, "name": athlete.get("displayName"), "dnp": not played}
        return lines

    def closing_line(self, league: str, event_id: str) -> Optional[Dict[str, Any]]:
        """The bookmaker line ESPN shows for a game (its last one once the game has started)."""
        for pick in self.summary(league, event_id).get("pickcenter") or []:
            home, away = pick.get("homeTeamOdds") or {}, pick.get("awayTeamOdds") or {}
            return {
                "provider": (pick.get("provider") or {}).get("name"),
                "home_spread": pick.get("spread"),  # from the home team's side, e.g. +4.5
                "total": pick.get("overUnder"),
                "home_ml": home.get("moneyLine"),
                "away_ml": away.get("moneyLine"),
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


def _goals(period: Any) -> int:
    """Goals in one linescore entry ({"displayValue": "1"} or {"value": 1.0}); ValueError if unreadable."""
    value = period.get("value", period.get("displayValue")) if isinstance(period, dict) else None
    number = float(value)  # TypeError/ValueError for None or text
    if not number.is_integer() or number < 0:
        raise ValueError(f"not a goal count: {value!r}")
    return int(number)


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
