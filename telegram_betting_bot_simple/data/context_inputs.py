"""Pure helpers over data the app already caches, for the context layer (v3 Stage 4): home-field rates from ESPN's
standings, each team's last game from the cached scoreboards, and days of rest. Nothing here fetches.

``home_rates(league, standings, prior=None, *, as_of=None)`` reads every team's home record and shrinks the league's
home win rate (and, in soccer, its draw rate) toward a prior with PRIOR_GAMES (400) games of weight:
``{"home_win", "draw" (soccer, else None), "games", "raw", "raw_draw", "prior", "prior_games", "teams", "source",
"as_of"}``. ``standings`` is ESPN's standings answer (``apis/v2/sports/{sport}/{league}/standings``) or
EspnClient.standings()'s per-team mapping (which keeps each team's ``"home"`` record). In ESPN's answer the home
record is the stat of ``type`` "home" (``summary`` "W-L", the NFL's "W-L-T", the NHL's "W-L-OTL") or, in MLB's, the
``homeWins``/``homeLosses``/``homeTies`` stats. ESPN's soccer standings carry no home split (only the overall
"W-D-L"), so for soccer the answer is the prior alone (``games`` 0) unless finished games are given to
``results_rates``, which counts home wins and draws from the cached scoreboards instead.

``last_games(league, scoreboards_by_day, team_ids, before, lookback_days=None)`` finds each team's latest game that
started before ``before`` within the league's lookback (LOOKBACK_DAYS): ``{team_id: {"date", "start", "venue":
{"name", "city", "espn_id"}, "opponent", "home", "neutral", "status", "event_id"}}``. A day's list may hold ESPN
scoreboard events or the slate's parsed games (data/slate.py's parse_scoreboard). ``rest_days(last, game)`` is the
days between, 0 for a back-to-back; ``lookback_for(league)`` the league's lookback; ``scoreboard_days`` the days of
scoreboards a lookback needs.
"""

from __future__ import annotations

import datetime
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from zoneinfo import ZoneInfo

from .espn_client import ESPN_DAY_TZ, LEAGUES as ESPN_LEAGUES

PRIOR_GAMES = 400
SOURCE = "ESPN standings, home records"
RESULTS_SOURCE = "ESPN scoreboards, finished games"
PRIOR_SOURCE = "Prior only: ESPN's standings carry no home records for this league"
NO_RESULTS_SOURCE = "Prior only: no finished games yet"
# The league's long-run home win rate (v3 Stage 4 spec); soccer: (home win, draw)
HOME_PRIORS: Dict[str, Union[float, Tuple[float, float]]] = {
    "americanfootball_nfl": 0.539, "basketball_nba": 0.554, "icehockey_nhl": 0.522, "baseball_mlb": 0.529,
    "basketball_wnba": 0.55, "americanfootball_ncaaf": 0.57, "basketball_ncaab": 0.60, "basketball_wncaab": 0.60,
}
SOCCER_PRIOR = (0.45, 0.27)
# How far back a team's last game is looked for, in days (v3 Stage 4 spec)
LOOKBACK_DAYS = {"basketball_nba": 4, "basketball_wnba": 4, "icehockey_nhl": 4, "americanfootball_nfl": 14,
                 "baseball_mlb": 2, "americanfootball_ncaaf": 7, "basketball_ncaab": 7, "basketball_wncaab": 7}
SOCCER_LOOKBACK = 10
DEFAULT_LOOKBACK = 7
POSTPONED = ("POSTPONED", "CANCELED", "CANCELLED", "SUSPENDED", "FORFEIT", "ABANDONED", "DELAYED_INDEFINITELY")
_RECORD = re.compile(r"^\s*(\d+)\s*-\s*(\d+)(?:\s*-\s*(\d+))?\s*$")


def sport_of(league: str) -> Optional[str]:
    """ESPN's sport for a league key ("basketball", "soccer", ...), None when ESPN has none."""
    path = ESPN_LEAGUES.get(league)
    if path:
        return path[0]
    return "soccer" if league.startswith("soccer_") else None


def home_prior(league: str) -> Union[float, Tuple[float, float], None]:
    """The league's prior: a home win rate, (home win, draw) for soccer, None when there's none."""
    if sport_of(league) == "soccer":
        return SOCCER_PRIOR
    return HOME_PRIORS.get(league)


def lookback_for(league: str) -> int:
    """How many days back a team's last game is looked for in ``league`` (LOOKBACK_DAYS; soccer 10)."""
    if league in LOOKBACK_DAYS:
        return LOOKBACK_DAYS[league]
    return SOCCER_LOOKBACK if sport_of(league) == "soccer" else DEFAULT_LOOKBACK


# --- home-field rates ---------------------------------------------------------------------------------------------

def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _split(summary: Any, sport: Optional[str]) -> Optional[Tuple[float, float, float]]:
    """(wins, draws, games) of a home record summary, read the sport's way: soccer "W-D-L", hockey "W-L-OTL" (an
    overtime loss is a loss), football "W-L-T" (a tie is half a win), else "W-L"."""
    found = _RECORD.match(str(summary or ""))
    if not found:
        return None
    first, second = float(found.group(1)), float(found.group(2))
    third = float(found.group(3)) if found.group(3) is not None else 0.0
    if sport == "soccer":
        return first, second, first + second + third
    if sport == "hockey":
        return first, 0.0, first + second + third
    if sport == "football":
        return first + 0.5 * third, 0.0, first + second + third
    return first + 0.5 * third, 0.0, first + second + third


def _from_counts(wins: Optional[float], losses: Optional[float], ties: Optional[float],
                 sport: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if wins is None or losses is None:
        return None
    ties = ties or 0.0
    if sport == "soccer":
        return wins, ties, wins + losses + ties
    return wins + 0.5 * ties, 0.0, wins + losses + ties


def _entry_record(stats: Any, sport: Optional[str]) -> Optional[Tuple[float, float, float]]:
    """A standings entry's home record from its stats list."""
    if not isinstance(stats, list):
        return None
    by_name: Dict[str, Any] = {}
    for stat in stats:
        if not isinstance(stat, Mapping):
            continue
        if str(stat.get("type") or "").lower() == "home" or str(stat.get("name") or "") == "Home":
            found = _split(stat.get("summary") or stat.get("displayValue"), sport)
            if found is not None:
                return found
        if stat.get("name"):
            by_name[str(stat["name"])] = stat.get("value")
    return _from_counts(_num(by_name.get("homeWins")), _num(by_name.get("homeLosses")),
                        _num(by_name.get("homeTies")), sport)


def _entries(node: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(node, Mapping):
        return
    standings = node.get("standings")
    if isinstance(standings, Mapping):
        for entry in standings.get("entries") or []:
            if isinstance(entry, Mapping):
                yield entry
    for child in node.get("children") or []:
        yield from _entries(child)


def _team_records(standings: Any, sport: Optional[str]) -> Dict[str, Tuple[float, float, float]]:
    """{team id: (wins, draws, games) at home} from ESPN's answer or EspnClient.standings()'s mapping."""
    records: Dict[str, Tuple[float, float, float]] = {}
    if not isinstance(standings, Mapping):
        return records
    if "children" in standings or "standings" in standings:
        for entry in _entries(standings):
            team = entry.get("team") if isinstance(entry.get("team"), Mapping) else {}
            found = _entry_record(entry.get("stats"), sport)
            if found is not None and team.get("id") is not None:
                records[str(team["id"])] = found
        return records
    for team_id, team in standings.items():  # EspnClient.standings(): {team id: {"home": "W-L", ...}}
        if not isinstance(team, Mapping):
            continue
        found = _split(team.get("home"), sport) if team.get("home") else None
        if found is None:
            found = _from_counts(_num(team.get("home_wins")), _num(team.get("home_losses")),
                                 _num(team.get("home_ties")), sport)
        if found is not None:
            records[str(team_id)] = found
    return records


def _prior_pair(league: str, prior: Any) -> Tuple[float, Optional[float]]:
    prior = home_prior(league) if prior is None else prior
    if isinstance(prior, Mapping):
        home, draw = _num(prior.get("home") if "home" in prior else prior.get("home_win")), _num(prior.get("draw"))
    elif isinstance(prior, (tuple, list)):
        home, draw = _num(prior[0]), _num(prior[1]) if len(prior) > 1 else None
    else:
        home, draw = _num(prior), None
    if home is None or not 0 < home < 1:
        raise ValueError(f"{league}: no home prior (pass prior=)")
    return home, draw


def _shrunk(league: str, wins: float, draws: float, games: float, prior: Any, source: str, teams: int,
            as_of: Any) -> Dict[str, Any]:
    home, draw_prior = _prior_pair(league, prior)
    soccer = sport_of(league) == "soccer"
    if soccer and draw_prior is None:
        draw_prior = SOCCER_PRIOR[1]
    weight = float(PRIOR_GAMES)
    return {
        "home_win": round((wins + home * weight) / (games + weight), 4),
        "draw": round((draws + draw_prior * weight) / (games + weight), 4) if soccer else None,
        "games": int(round(games)),
        "raw": round(wins / games, 4) if games else None,
        "raw_draw": (round(draws / games, 4) if games else None) if soccer else None,
        "prior": {"home_win": home, "draw": draw_prior if soccer else None},
        "prior_games": PRIOR_GAMES,
        "teams": teams,
        "source": source if games else (PRIOR_SOURCE if source == SOURCE else NO_RESULTS_SOURCE),
        "as_of": as_of,
    }


def home_rates(league: str, standings_payload: Any, prior: Any = None, *, as_of: Any = None) -> Dict[str, Any]:
    """The league's home win rate (and draw rate in soccer) this season from ESPN's standings home records, shrunk
    toward ``prior`` (default HOME_PRIORS / SOCCER_PRIOR) with PRIOR_GAMES games of weight (see the module's
    docstring). ``as_of`` is passed through (when the standings were fetched)."""
    sport = sport_of(league)
    records = _team_records(standings_payload, sport)
    wins = sum(record[0] for record in records.values())
    draws = sum(record[1] for record in records.values())
    games = sum(record[2] for record in records.values())
    return _shrunk(league, wins, draws, games, prior, SOURCE, len(records), as_of)


def results_rates(league: str, games: Iterable[Any], prior: Any = None, *, as_of: Any = None) -> Dict[str, Any]:
    """As home_rates, counted from finished games instead (ESPN scoreboard events or the slate's parsed games;
    neutral-site and unfinished games are skipped): for soccer, whose standings have no home split."""
    seen = set()
    wins = draws = played = 0.0
    for game in games or []:
        found = _game(game)
        if found is None or found["status"] != "final" or found["neutral"]:
            continue
        if found["event_id"] is not None:
            if found["event_id"] in seen:
                continue
            seen.add(found["event_id"])
        home, away = found["home_score"], found["away_score"]
        if home is None or away is None:
            continue
        played += 1
        if home > away:
            wins += 1
        elif home == away:
            if sport_of(league) == "soccer":
                draws += 1
            else:
                wins += 0.5
    return _shrunk(league, wins, draws, played, prior, RESULTS_SOURCE, 0, as_of)


# --- each team's last game ------------------------------------------------------------------------------------------

def _when(value: Any) -> Optional[datetime.datetime]:
    if isinstance(value, datetime.datetime):
        return value.replace(tzinfo=datetime.timezone.utc) if value.tzinfo is None else value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return datetime.datetime.fromtimestamp(value, datetime.timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return _when(parsed)
    return None


def _score(value: Any) -> Optional[float]:
    if isinstance(value, Mapping):
        value = value.get("value", value.get("displayValue"))
    return _num(value)


def _game(game: Any) -> Optional[Dict[str, Any]]:
    """A scoreboard game (ESPN's event or the slate's parsed game) as {"start", "status", "home_id", "away_id",
    "home_name", "away_name", "home_score", "away_score", "venue", "neutral", "event_id"}; None if unreadable."""
    if not isinstance(game, Mapping):
        return None
    if "_ids" in game or ("home" in game and "competitions" not in game):  # the slate's parsed game
        ids = game.get("_ids") or (None, None)
        home, away = game.get("home") or {}, game.get("away") or {}
        venue = game.get("venue") or {}
        start = _when(game.get("start") if game.get("start") is not None else game.get("_ts"))
        return {"start": start, "status": str(game.get("status") or "pre"),
                "home_id": str(ids[0]) if ids and ids[0] is not None else None,
                "away_id": str(ids[1]) if ids and len(ids) > 1 and ids[1] is not None else None,
                "home_name": home.get("name"), "away_name": away.get("name"),
                "home_score": _num(home.get("score")), "away_score": _num(away.get("score")),
                "venue": {"name": venue.get("name"), "city": venue.get("city"),
                          "espn_id": venue.get("espn_id") or venue.get("id")},
                "neutral": bool(game.get("_neutral") or game.get("neutral")),
                "event_id": game.get("espn_id") or game.get("id")}
    competitions = game.get("competitions") or []
    comp = competitions[0] if competitions and isinstance(competitions[0], Mapping) else None
    if comp is None:
        return None
    sides = {c.get("homeAway"): c for c in comp.get("competitors") or [] if isinstance(c, Mapping)}
    if "home" not in sides or "away" not in sides:
        return None
    kind = (comp.get("status") or game.get("status") or {}).get("type") or {}
    name = str(kind.get("name") or "").upper()
    if any(word in name for word in POSTPONED):
        status = "postponed"
    elif kind.get("state") == "in":
        status = "live"
    elif kind.get("state") == "post" or kind.get("completed"):
        status = "final"
    else:
        status = "pre"
    venue = comp.get("venue") if isinstance(comp.get("venue"), Mapping) else {}
    address = venue.get("address") if isinstance(venue.get("address"), Mapping) else {}
    home_team = sides["home"].get("team") or {}
    away_team = sides["away"].get("team") or {}
    return {"start": _when(comp.get("date") or game.get("date")), "status": status,
            "home_id": str(home_team["id"]) if home_team.get("id") is not None else None,
            "away_id": str(away_team["id"]) if away_team.get("id") is not None else None,
            "home_name": home_team.get("displayName") or home_team.get("name"),
            "away_name": away_team.get("displayName") or away_team.get("name"),
            "home_score": _score(sides["home"].get("score")), "away_score": _score(sides["away"].get("score")),
            "venue": {"name": venue.get("fullName"), "city": address.get("city"),
                      "espn_id": str(venue["id"]) if venue.get("id") is not None else None},
            "neutral": bool(comp.get("neutralSite")), "event_id": game.get("id")}


def _local_date(moment: datetime.datetime, tz: Any) -> datetime.date:
    zone = ZoneInfo(tz) if isinstance(tz, str) else tz
    return moment.astimezone(zone).date()


def last_games(league: str, scoreboards_by_day: Mapping[Any, Sequence[Any]], team_ids: Iterable[Any], before: Any,
               lookback_days: Optional[int] = None, *, tz: Any = ESPN_DAY_TZ) -> Dict[str, Dict[str, Any]]:
    """Each of ``team_ids``' latest game that started before ``before`` (the game's start: a datetime, epoch
    seconds or ISO text) and no more than ``lookback_days`` (default the league's LOOKBACK_DAYS) days earlier,
    skipping postponed and cancelled games: {team id: {"date" (its local date in ``tz``, ESPN's US Eastern day by
    default), "start" (ISO, UTC), "venue": {"name", "city", "espn_id"}, "opponent", "home" (bool), "neutral",
    "status", "event_id"}}. Teams with no such game are left out."""
    cutoff = _when(before)
    if cutoff is None:
        return {}
    days = lookback_days if lookback_days is not None else lookback_for(league)
    earliest = cutoff - datetime.timedelta(days=days, hours=12)
    wanted = {str(team_id) for team_id in team_ids if team_id is not None}
    found: Dict[str, Tuple[datetime.datetime, Dict[str, Any]]] = {}
    for games in scoreboards_by_day.values():
        for raw in games or []:
            game = _game(raw)
            if game is None or game["start"] is None or game["status"] == "postponed":
                continue
            start = game["start"]
            if not earliest <= start < cutoff - datetime.timedelta(minutes=1):
                continue
            for side, other in (("home", "away"), ("away", "home")):
                team_id = game[f"{side}_id"]
                if team_id not in wanted:
                    continue
                held = found.get(team_id)
                if held is not None and held[0] >= start:
                    continue
                found[team_id] = (start, {
                    "date": _local_date(start, tz).isoformat(),
                    "start": start.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "venue": dict(game["venue"]),
                    "opponent": game[f"{other}_name"],
                    "home": side == "home",
                    "neutral": game["neutral"],
                    "status": game["status"],
                    "event_id": game["event_id"],
                })
    return {team_id: row for team_id, (_, row) in found.items()}


def _day(value: Any, tz: Any) -> Optional[datetime.date]:
    if isinstance(value, datetime.datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
        return _local_date(moment, tz)
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):  # epoch seconds
        moment = _when(value)
        return _local_date(moment, tz) if moment is not None else None
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if len(text) == 10:
            try:
                return datetime.date.fromisoformat(text)
            except ValueError:
                return None
        moment = _when(text)
        return _local_date(moment, tz) if moment is not None else None
    return None


def rest_days(last_date: Any, game_date: Any, *, tz: Any = ESPN_DAY_TZ) -> Optional[int]:
    """Full days off between a team's last game and this one: 0 for a back-to-back (or a second game the same day),
    1 with one day between, ... Dates, datetimes or epoch seconds (put on ``tz``'s calendar, ESPN's US Eastern day by
    default) or ISO text; None when either is missing or the last game is after this one."""
    last, game = _day(last_date, tz), _day(game_date, tz)
    if last is None or game is None or last > game:
        return None
    return max(0, (game - last).days - 1)


def rest_for(league: str, last: Optional[Mapping[str, Any]], game_start: Any, *, tz: Any = ESPN_DAY_TZ
             ) -> Dict[str, Any]:
    """{"rest_days", "back_to_back"} for a team whose last game is ``last`` (a last_games row, or None)."""
    if not last:
        return {"rest_days": None, "back_to_back": False}
    days = rest_days(last.get("start") or last.get("date"), game_start, tz=tz)
    return {"rest_days": days, "back_to_back": days == 0}


def scoreboard_days(before: Any, lookback: int, tz: Any = ESPN_DAY_TZ) -> List[datetime.date]:
    """The scoreboard days (``tz``'s calendar, ESPN's US Eastern by default) a lookback of ``lookback`` days before
    ``before`` covers, oldest first, including ``before``'s own day (an earlier game the same day)."""
    moment = _when(before)
    if moment is None:
        return []
    last = _local_date(moment, tz)
    return [last - datetime.timedelta(days=offset) for offset in range(lookback, -1, -1)]
