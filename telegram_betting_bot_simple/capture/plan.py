"""What capture keeps for each sport, and when (v4 P1, plan section 3.3), over the 58 ESPN paths.

**The 58 paths** (``PATHS``): data/espn_client.py's 55 team leagues (``LEAGUES``) plus ATP, WTA and UFC
(``SCOREBOARD_ONLY``), each with its ESPN (sport, league path).

**Kinds** (``KINDS[sport] = {"final": [...], "pre": [...]}``), the endpoints of section 3.3's table:

| Sport | At final | Before the start (inputs, never outcomes) |
|---|---|---|
| basketball | site ``summary`` untrimmed | ``summary`` at t90 (NBA); ``propbets`` at open/t24/t90/close (NBA, WNBA) |
| football | site ``summary`` (nflverse play-by-play and officials: every Tuesday, ``STATIC``) | ``propbets`` (NFL) |
| hockey | site ``summary`` | ``propbets`` (NHL) |
| baseball | site ``summary`` and StatsAPI ``feed_live`` (one game at a time) | ``propbets`` (MLB) |
| soccer | site ``summary``; core ``core_plays`` (all pages) for CAPTURE_CORE_PLAYS_LEAGUES | ``summary`` at t24 and close |
| mma | per card ``fightcenter``; per bout ``core_status`` and ``core_competitor_stats`` only when the fightcenter lacks the bout's result; the day's ``scoreboard`` raw | ``fightcenter`` at t24 |
| tennis | the day's ATP/WTA ``scoreboard`` raw | — |

Which leagues get which kind (``kinds_for``): pre-start summaries only for wave 1-2 (plan section 9.3: the NBA and the
wave-1 soccer leagues; the UFC's pre kind is its fightcenter), core plays only for CAPTURE_CORE_PLAYS_LEAGUES (default:
the ten wave-1 soccer leagues), propBets only for ``PROPBETS_LEAGUES`` while CAPTURE_PROPBETS is on, Kalshi's series
for the leagues CAPTURE_KALSHI_SERIES names (``kalshi`` at each of their games' horizons, one snapshot a series per 10
minutes at most). Not yet: the NHL's api-web play-by-play (section 3.3's wave-2 note: P1 has no fetcher for it).

**When.** A pre item (kind, horizon) is due at its horizon (``HORIZON_LEAD``: t24 is the start less 24 hours, t90
less 90 minutes, close less 10 minutes; "open" at first sight) and is missed once the start has passed. A fetch serves
every horizon of that kind already due, so a game first seen 5 hours before it starts takes one snapshot for "open"
and "t24". The final is fetched 30 minutes after the final whistle (seen on the slate, the sweep, or estimated from
the start: ``GAME_MINUTES``), then retried at +2 h, +12 h and +36 h (``FINAL_DELAYS``) while anything is missing; a
job gives up 4 days after its start (``GIVE_UP``).

**Static history** (``STATIC``): Greco1899's six UFC CSVs every week, the Match Charting Project's CSVs every month,
football-data's current-season CSVs twice a week and nflverse's play-by-play of the current season with its officials
file every Tuesday.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..data.espn_client import LEAGUES, SCOREBOARD_ONLY

# The 58 paths: league key -> (ESPN sport, league path)
PATHS: Dict[str, Tuple[str, str]] = {**LEAGUES, **SCOREBOARD_ONLY}
TENNIS = ("tennis_atp", "tennis_wta")
UFC = "mma_mixed_martial_arts"
TENNIS_PREFIXES = {"tennis_atp_": "tennis_atp", "tennis_wta_": "tennis_wta"}  # odds-feed tournaments -> ESPN tours

# Waves (plan section 1.3): where models, prices and pre-start inputs go first; every path is captured at final
WAVE1_SOCCER: Tuple[str, ...] = (
    "soccer_epl", "soccer_efl_champ", "soccer_spain_la_liga", "soccer_germany_bundesliga", "soccer_italy_serie_a",
    "soccer_france_ligue_one", "soccer_spl", "soccer_netherlands_eredivisie", "soccer_portugal_primeira_liga",
    "soccer_uefa_champs_league")
WAVE1: Tuple[str, ...] = ("basketball_nba", "americanfootball_nfl") + WAVE1_SOCCER
WAVE2: Tuple[str, ...] = ("icehockey_nhl", UFC, "baseball_mlb")
CORE_PLAYS_LEAGUES: Tuple[str, ...] = WAVE1_SOCCER  # CAPTURE_CORE_PLAYS_LEAGUES' default
PROPBETS_LEAGUES: Tuple[str, ...] = ("basketball_nba", "basketball_wnba", "americanfootball_nfl", "icehockey_nhl",
                                     "baseball_mlb")  # capture/sources/espn_propbets.py LEAGUES
PRE_SUMMARY_LEAGUES: Tuple[str, ...] = ("basketball_nba",) + WAVE1_SOCCER  # wave 1-2 sports with a pre summary

KINDS: Dict[str, Dict[str, List[str]]] = {
    "basketball": {"final": ["summary"], "pre": ["summary", "propbets"]},
    "football": {"final": ["summary"], "pre": ["propbets"]},
    "hockey": {"final": ["summary"], "pre": ["propbets"]},
    "baseball": {"final": ["summary", "feed_live"], "pre": ["propbets"]},
    "soccer": {"final": ["summary", "core_plays"], "pre": ["summary"]},
    "mma": {"final": ["fightcenter", "core_status", "core_competitor_stats", "scoreboard"], "pre": ["fightcenter"]},
    "tennis": {"final": ["scoreboard"], "pre": []},
}
# MMA's per-bout kinds are fetched only for a bout the card's fightcenter has no result for
CONDITIONAL: FrozenSet[str] = frozenset({"core_status", "core_competitor_stats"})
# The horizons of each pre kind (per sport; "*" for every sport)
PRE_HORIZONS: Dict[Tuple[str, str], Tuple[str, ...]] = {
    ("basketball", "summary"): ("t90",),
    ("soccer", "summary"): ("t24", "close"),
    ("mma", "fightcenter"): ("t24",),
    ("*", "propbets"): ("open", "t24", "t90", "close"),
    ("*", "kalshi"): ("open", "t24", "t90", "close"),
}
HORIZON_ORDER = ("open", "t24", "t90", "close")
HORIZON_LEAD: Dict[str, float] = {"t24": 24 * 3600.0, "t90": 90 * 60.0, "close": 10 * 60.0}
FINAL_DELAYS: Tuple[float, ...] = (30 * 60.0, 2 * 3600.0, 12 * 3600.0, 36 * 3600.0)  # after the final whistle
GIVE_UP = 4 * 86400.0  # after the start
# How long a game usually runs, to guess its final whistle when nothing reports it (a card: prelims to main event)
GAME_MINUTES: Dict[str, float] = {"basketball": 150, "football": 210, "hockey": 165, "baseball": 195, "soccer": 115,
                                  "mma": 420, "tennis": 180}
PRICE_KINDS: FrozenSet[str] = frozenset({"propbets", "kalshi"})  # billed to the "prices" budget
KALSHI_EVERY = 600.0  # one snapshot of a series per 10 minutes at most

# (source, endpoint) each kind is stored as
SOURCE_OF: Dict[str, str] = {"summary": "espn_site", "scoreboard": "espn_site", "core_plays": "espn_core",
                             "core_status": "espn_core", "core_competitor_stats": "espn_core",
                             "fightcenter": "espn_fightcenter", "feed_live": "mlb_statsapi",
                             "propbets": "espn_propbets", "kalshi": "kalshi"}
ENDPOINT_OF: Dict[str, str] = {"kalshi": "events"}  # every other kind is its own endpoint
# The host each kind is fetched from (whether a job can run while a host backs off)
HOST_OF: Dict[str, str] = {"summary": "site.api.espn.com", "scoreboard": "site.api.espn.com",
                           "core_plays": "sports.core.api.espn.com", "core_status": "sports.core.api.espn.com",
                           "core_competitor_stats": "sports.core.api.espn.com",
                           "propbets": "sports.core.api.espn.com", "fightcenter": "site.web.api.espn.com",
                           "feed_live": "statsapi.mlb.com", "kalshi": "api.elections.kalshi.com"}


def endpoint_of(kind: str) -> str:
    return ENDPOINT_OF.get(kind, kind)


def espn_league(league: str) -> Optional[str]:
    """The ESPN league key a slate or odds league is captured under: itself on the 58 paths, a tennis tournament's
    tour (tennis_atp_paris_masters -> tennis_atp), else None."""
    if league in PATHS:
        return league
    return next((tour for prefix, tour in TENNIS_PREFIXES.items() if str(league).startswith(prefix)), None)


def sport_of(league: str) -> Optional[str]:
    """ESPN's sport of a league on the 58 paths ("basketball", "soccer", ..., "mma", "tennis"); None otherwise."""
    found = PATHS.get(espn_league(league) or "")
    return found[0] if found else None


@dataclass(frozen=True)
class Options:
    """The settings that decide which kinds a league gets (config's CAPTURE_*)."""

    core_plays_leagues: Tuple[str, ...] = CORE_PLAYS_LEAGUES
    propbets: bool = True
    kalshi_leagues: Tuple[str, ...] = ()  # leagues with at least one CAPTURE_KALSHI_SERIES ticker


def kinds_for(league: str, phase: str, options: Optional[Options] = None) -> List[str]:
    """The kinds ``league`` gets in ``phase`` ("final" or "pre"), in KINDS' order (MMA's conditional per-bout kinds
    included: see CONDITIONAL)."""
    options = options or Options()
    sport = sport_of(league)
    if sport is None or phase not in ("final", "pre"):
        return []
    out = []
    for kind in KINDS[sport][phase]:
        if phase == "final" and kind == "core_plays" and league not in options.core_plays_leagues:
            continue
        if phase == "pre" and kind == "summary" and league not in PRE_SUMMARY_LEAGUES:
            continue
        if kind == "propbets" and not (options.propbets and league in PROPBETS_LEAGUES):
            continue
        out.append(kind)
    if phase == "pre" and league in options.kalshi_leagues:
        out.append("kalshi")
    return out


def required_final(league: str, options: Optional[Options] = None) -> List[str]:
    """The final kinds a game needs to be done (MMA's conditional per-bout kinds are not among them)."""
    return [kind for kind in kinds_for(league, "final", options) if kind not in CONDITIONAL]


def horizons_for(league: str, kind: str) -> Tuple[str, ...]:
    sport = sport_of(league) or ""
    return PRE_HORIZONS.get((sport, kind)) or PRE_HORIZONS.get(("*", kind)) or ()


def fire_time(horizon: str, start: float, first_seen: float) -> float:
    """When a pre horizon is due: "open" at first sight, the others their lead before the start."""
    if horizon == "open":
        return float(first_seen)
    return float(start) - HORIZON_LEAD[horizon]


def pre_items(league: str, options: Optional[Options] = None) -> List[Tuple[str, str]]:
    """(kind, horizon) for every pre-start fetch of a game of ``league``."""
    return [(kind, horizon) for kind in kinds_for(league, "pre", options) for horizon in horizons_for(league, kind)]


def pre_name(kind: str, horizon: str) -> str:
    """The kinds_done entry of a pre item: "pre:t24:summary"."""
    return f"pre:{horizon}:{kind}"


def final_name(kind: str) -> str:
    """The kinds_done entry of a final kind: "final:summary"."""
    return f"final:{kind}"


def estimated_end(league: str, start: float) -> float:
    return float(start) + GAME_MINUTES.get(sport_of(league) or "", 180.0) * 60.0


def next_final_delay(attempts: int) -> Optional[float]:
    """Seconds from attempt ``attempts`` (1-based, the one just made) to the next, on the schedule +30 min, +2 h,
    +12 h, +36 h after the final; None after the last."""
    if attempts < 1 or attempts >= len(FINAL_DELAYS):
        return None
    return FINAL_DELAYS[attempts] - FINAL_DELAYS[attempts - 1]


# --- static history files -----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Static:
    """A static source's schedule: every ``every`` seconds, or once a week from ``weekday`` (Monday 0) at ``hour``
    server time."""

    name: str
    every: Optional[float] = None
    weekday: Optional[int] = None
    hour: int = 0


STATIC: Tuple[Static, ...] = (
    Static("greco1899", every=7 * 86400.0),
    Static("mcp", every=30 * 86400.0),
    Static("football_data", every=3.5 * 86400.0),
    Static("nflverse", weekday=1, hour=10),  # Tuesday: Monday night's game is in
)


def static_period(entry: Static, now: float, tz: Optional[datetime.tzinfo] = None) -> Optional[str]:
    """The period a weekly-on-a-day source's run belongs to (the ISO week of its latest due day at or before
    ``now``, "2026-W44"), or None for an interval source."""
    if entry.weekday is None:
        return None
    local = datetime.datetime.fromtimestamp(now, tz) if tz is not None else datetime.datetime.fromtimestamp(now)
    day = local.date() - datetime.timedelta(days=(local.weekday() - entry.weekday) % 7)
    due = datetime.datetime.combine(day, datetime.time(entry.hour), tzinfo=tz) if tz is not None else \
        datetime.datetime.combine(day, datetime.time(entry.hour))
    if local < due:
        day -= datetime.timedelta(days=7)
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def static_due(entry: Static, last: Optional[Mapping[str, object]], now: float,
               tz: Optional[datetime.tzinfo] = None) -> bool:
    """Whether a static source is due, from its last run ({"at": epoch, "period": ...}; None: never ran)."""
    if entry.weekday is not None:
        period = static_period(entry, now, tz)
        return not last or last.get("period") != period
    if not last or not isinstance(last.get("at"), (int, float)):
        return True
    return now - float(last["at"]) >= float(entry.every or 0)  # type: ignore[arg-type]


def league_list(text: Optional[str], default: Sequence[str]) -> Tuple[str, ...]:
    """A comma-separated list of league keys (CAPTURE_CORE_PLAYS_LEAGUES): None or blank is the default, "none" is
    empty; keys not on the 58 paths are refused (ValueError), so a typo fails at startup."""
    if text is None or not str(text).strip():
        return tuple(default)
    if str(text).strip().lower() == "none":
        return ()
    out: List[str] = []
    for part in str(text).split(","):
        key = part.strip()
        if not key:
            continue
        if key not in PATHS:
            raise ValueError(f"{key[:60]!r} isn't one of the 58 ESPN league keys")
        if key not in out:
            out.append(key)
    return tuple(out)


def iter_paths(leagues: Optional[Iterable[str]] = None) -> List[Tuple[str, str, str]]:
    """(league, ESPN sport, path) of the 58 paths (or of ``leagues``), in data/espn_client.py's order."""
    wanted = set(leagues) if leagues is not None else None
    return [(league, sport, path) for league, (sport, path) in PATHS.items() if wanted is None or league in wanted]


__all__ = ["PATHS", "TENNIS", "UFC", "WAVE1", "WAVE1_SOCCER", "WAVE2", "CORE_PLAYS_LEAGUES", "PROPBETS_LEAGUES",
           "PRE_SUMMARY_LEAGUES", "KINDS", "CONDITIONAL", "PRE_HORIZONS", "HORIZON_ORDER", "HORIZON_LEAD",
           "FINAL_DELAYS", "GIVE_UP", "GAME_MINUTES", "PRICE_KINDS", "KALSHI_EVERY", "SOURCE_OF", "HOST_OF", "Options",
           "STATIC", "Static", "kinds_for", "required_final", "horizons_for", "fire_time", "pre_items", "pre_name",
           "final_name", "estimated_end", "next_final_delay", "espn_league", "sport_of", "endpoint_of",
           "static_due", "static_period", "league_list", "iter_paths"]
