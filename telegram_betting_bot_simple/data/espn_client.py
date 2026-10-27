"""Schedules, injuries, ESPN's matchup predictor and team/player stats from ESPN.

These are ESPN's public but undocumented JSON endpoints: they need no key,
but ESPN can change them without notice and doesn't license them for
commercial use, so callers treat every lookup as best-effort. Responses
are cached to keep the request rate low.
"""

from __future__ import annotations

import datetime
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

from .cache import TTLCache, freshest, ttl_for
from .utils import policy

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


# Scoreboard keys ESPN sends that nothing here reads, dropped (at any depth) before a board is cached: leaders,
# headlines, highlights, team and athlete links, box-score statistics, the books' odds, play details and tennis
# rosters are 80-90% of a board (a 330 KB NHL night keeps 38 KB). find_game, played_on and the slate's
# parse_scoreboard read ids, names, logos, records, scores, linescores, status, dates, venue and groupings
LEAN_DROP = frozenset({
    "links", "leaders", "statistics", "headlines", "highlights", "odds", "details", "featuredAthletes", "roster",
    "broadcasts", "broadcast", "geoBroadcasts", "tickets", "situation", "probables", "notes",
})
# The slate's own fetch (fetch_scoreboard) reuses a board at most this old (one the other process or a request
# loaded a moment ago) and otherwise loads it, so the slate keeps its own intervals (20 s live at the fastest)
SLATE_REUSE = 5.0
LIVE_SOON = 10 * 60  # a board with a game starting this soon is paced as live (as the slate's LIVE_SOON)
STATES_KEPT = 5000  # games whose state (pre, live, final) and score the client remembers, newest first


def lean_events(value: Any) -> Any:
    """A scoreboard's events without the keys in LEAN_DROP."""
    if isinstance(value, dict):
        return {key: lean_events(item) for key, item in value.items() if key not in LEAN_DROP}
    if isinstance(value, list):
        return [lean_events(item) for item in value]
    return value


# The parts of a game summary anything here reads (the game's status and score, ESPN's predictor, the injury lists,
# the box score and the books' lines: game_result, box_score, closing_line, ratings.parse_predictor, the dashboard's
# cards). The rest (play-by-play, win probability, news, videos, leaders, standings) is about 90% of a summary
# (a final NBA game's 516 KB keeps about 57 KB), so it is dropped before the summary is cached
SUMMARY_KEEP = ("header", "predictor", "injuries", "boxscore", "pickcenter")


def lean_summary(data: Any) -> Any:
    """A game summary with only the SUMMARY_KEEP parts."""
    if not isinstance(data, dict):
        return data
    return {key: data[key] for key in SUMMARY_KEEP if key in data}


def _score(value: Any) -> Optional[int]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def competition_state(competition: Any) -> Optional[Tuple[str, Optional[int], Optional[int], Optional[float]]]:
    """(state, home score, away score, start in epoch seconds) of an ESPN competition: state is "pre", "live" or
    "final" (a postponed or cancelled game counts as final: nothing more will happen to it); None if unreadable."""
    if not isinstance(competition, dict):
        return None
    kind = (competition.get("status") or {}).get("type") or {}
    if not isinstance(kind, dict):
        return None
    name = str(kind.get("name") or "").upper()
    if kind.get("completed") or kind.get("state") == "post" or any(
            word in name for word in ("POSTPONED", "CANCELED", "CANCELLED", "FORFEIT", "ABANDONED")):
        state = "final"
    elif kind.get("state") == "in":
        state = "live"
    elif kind.get("state") == "pre":
        state = "pre"
    else:
        return None
    sides = {c.get("homeAway"): c for c in competition.get("competitors") or [] if isinstance(c, dict)}
    start = None
    if isinstance(competition.get("date"), str):
        try:
            start = datetime.datetime.fromisoformat(competition["date"].replace("Z", "+00:00")).timestamp()
        except ValueError:
            start = None
    return (state, _score((sides.get("home") or {}).get("score")), _score((sides.get("away") or {}).get("score")),
            start)


class EspnClient:
    """ESPN's public endpoints, every request through the shared TTLCache (data/cache.py) with a kind and the
    policy table's time to live (data/utils.py's policy). Game summaries are state-aware: a live game's for
    seconds, a pre-game one for minutes, a final one for a day. The client remembers each game's state and score
    from the scoreboards and summaries it reads (``game_state``, ``event_state``)."""

    def __init__(self, cache: Optional[TTLCache] = None, clock: Optional[Callable[[], float]] = None) -> None:
        self.cache = cache if cache is not None else TTLCache()  # an empty TTLCache is falsy (len 0)
        self.clock = clock or time.time
        self._states: "OrderedDict[Tuple[str, str], Tuple[str, Optional[int], Optional[int], Optional[float]]]" = \
            OrderedDict()
        self._states_lock = threading.Lock()

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """One GET to ESPN (no key needed). Only ever called inside a cache loader."""
        response = requests.get(url, params=params, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if not response.ok:
            raise RuntimeError(f"ESPN returned HTTP {response.status_code} for {url}")
        return response.json()

    def _cached(self, key: Tuple[Any, ...], kind: str, loader: Callable[[], Any], state: Optional[str] = None,
                stale: Optional[float] = None, life: Any = None) -> Any:
        return self._read(key, kind, loader, policy(kind, state), stale, life)[0]

    def _read(self, key: Tuple[Any, ...], kind: str, loader: Callable[[], Any], rule: Tuple[float, float],
              stale: Optional[float] = None, life: Any = None) -> Tuple[Any, float]:
        """(value, when it was fetched on the cache's clock) under ``rule`` (ttl, stale window)."""
        ttl, window = rule
        options: Dict[str, Any] = {"stale": window if stale is None else stale, "kind": kind}
        if life is not None:
            options["life"] = life
        read = getattr(self.cache, "get_with_time", None)
        if callable(read):
            return read(key, ttl, loader, **options)
        return self.cache.get(key, ttl, loader, **options), self._cache_now()

    def _cache_now(self) -> float:
        """Now on the cache's clock (the one its copies' fetch times are on), else the client's."""
        now = getattr(self.cache, "now", None)
        return float(now()) if callable(now) else self.clock()

    # --- what the client knows of each game's state ---

    def _remember(self, league: str, event_id: Any, competition: Any, at: float) -> None:
        """Note a game's state and score as read from data fetched at ``at`` (the cache's clock), unless what is
        noted already comes from data fetched later: an old copy (a pre-game summary, a board from before the
        final whistle) never moves a game back."""
        found = competition_state(competition)
        if found is None or event_id in (None, ""):
            return
        with self._states_lock:
            key = (league, str(event_id))
            known = self._states.get(key)
            if known is not None and known[4] > at:
                return
            self._states[key] = (*found, at)
            self._states.move_to_end(key)
            while len(self._states) > STATES_KEPT:
                self._states.popitem(last=False)

    def _remember_events(self, league: str, events: Any, at: float) -> None:
        if espn_path(league) is None or league in SCOREBOARD_ONLY:
            return  # tennis and MMA events are tournaments and cards, not one game
        for event in events or []:
            if isinstance(event, dict):
                self._remember(league, event.get("id"), (event.get("competitions") or [None])[0], at)

    def event_state(self, league: str, event_id: Any) -> Optional[Dict[str, Any]]:
        """{"state": "pre"|"live"|"final", "home": score, "away": score} as last read from a scoreboard or a summary
        (the most recently fetched; a "pre" game whose start has passed reads as "live"); None when the client hasn't
        seen the game."""
        with self._states_lock:
            found = self._states.get((league, str(event_id)))
        if found is None:
            return None
        state, home, away, start = found[:4]
        if state == "pre" and start is not None and start <= self.clock():
            state = "live"
        return {"state": state, "home": home, "away": away}

    def game_state(self, league: str, event_id: Any) -> str:
        """The game's state for the cache's policy: "pre", "live" or "final" ("pre" when not seen yet)."""
        found = self.event_state(league, event_id)
        return found["state"] if found else "pre"

    # --- scoreboards ---

    @staticmethod
    def _scoreboard_key(league: str, day: datetime.date) -> Tuple[Any, ...]:
        return ("espn", "scoreboard", league, day.isoformat())

    def _load_scoreboard(self, league: str, day: datetime.date) -> List[Dict[str, Any]]:
        target = espn_path(league)
        if target is None:
            raise KeyError(league)
        sport, path = target
        params = {"dates": day.strftime("%Y%m%d"), "limit": SCOREBOARD_LIMIT, **SCOREBOARD_PARAMS.get(league, {})}
        events = lean_events(self._get(f"{SITE}/{sport}/{path}/scoreboard", params).get("events", []))
        self._remember_events(league, events, self._cache_now())  # here too: a background refresh's count at once
        return events

    def _board_life(self, day: datetime.date) -> Callable[[Any], Tuple[float, float]]:
        """A loaded board's life: the policy of the board it is (board_state of its own events)."""
        return lambda events: ttl_for("scoreboard", self.board_state(day, events))

    def board_state(self, day: datetime.date, events: Any = None) -> str:
        """A board's state for the policy table's "scoreboard" rows, as the slate paces its boards: "live" (a game
        on, or starting within 10 minutes), "today" (games still to finish), "idle" (all done), "future" (a day
        ahead) or "past" (two days or more ago). A board not held yet is "today" (or "future"/"past")."""
        today = datetime.datetime.fromtimestamp(self.clock(), ESPN_DAY_TZ).date()
        if day < today - datetime.timedelta(days=1):
            return "past"
        if day > today:
            return "future"
        if events is None:
            return "today"
        now, waiting = self.clock(), False
        for event in events or []:
            found = competition_state((event.get("competitions") or [None])[0]) if isinstance(event, dict) else None
            if found is None:
                continue
            state, _, _, start = found
            if state == "live" or (state == "pre" and start is not None and start - now < LIVE_SOON):
                return "live"
            waiting = waiting or state == "pre"
        return "today" if waiting else "idle"

    def scoreboard(self, league: str, day: datetime.date) -> List[Dict[str, Any]]:
        """The league's events on ``day`` (a US Eastern date), from the shared cache, kept as long as the slate
        keeps that board (board_state): the slate refreshes today's boards, so a request rarely waits for ESPN."""
        if espn_path(league) is None:
            raise KeyError(league)
        key = self._scoreboard_key(league, day)
        held = self.cache.peek(key) if hasattr(self.cache, "peek") else None
        # Past its time to live a board is served at once while one background refresh runs (the policy's stale
        # window): a game page needs only the event's and teams' ids from it, which never change
        events, at = self._read(key, "scoreboard", lambda: self._load_scoreboard(league, day),
                                policy("scoreboard", self.board_state(day, held[1] if held else None)),
                                life=self._board_life(day))
        self._remember_events(league, events, at)
        return events

    def fetch_scoreboard(self, league: str, day: datetime.date) -> List[Dict[str, Any]]:
        """The league's events on ``day`` for the all-sports slate (data/slate.py), which fetches each board on its
        own schedule: a copy younger than SLATE_REUSE seconds is reused (the other process's, say), anything older
        is loaded. ESPN takes one day at a time (a date range is a 400). Raises when ESPN fails, even if the cache
        still holds the last good board, so the slate's own back-off and fallback see the failure."""
        if espn_path(league) is None:
            raise KeyError(league)
        key = self._scoreboard_key(league, day)
        # error_ttl=0: the slate backs off a failing board itself (its pace, the round breaker), so the cache
        # neither remembers nor delays the slate's failures
        # life: the board's own policy, so the request path (scoreboard) keeps the slate's copy as long as the slate
        # paces that board, not SLATE_REUSE seconds
        events, at = self.cache.get_with_time(key, SLATE_REUSE, lambda: self._load_scoreboard(league, day),
                                              stale=0.0, error_ttl=0.0, kind="scoreboard", life=self._board_life(day))
        age = self.cache.age(key)
        if age is not None and age >= SLATE_REUSE + 1.0:
            raise RuntimeError(f"ESPN didn't refresh the {league} scoreboard for {day.isoformat()}")
        self._remember_events(league, events, at)
        return events

    # --- one game ---

    def summary(self, league: str, event_id: str) -> Dict[str, Any]:
        """Game summary (lean_summary: the header with status and score, ESPN's predictor, injuries, box score and
        the books' lines). Kept for the game's state: seconds while it is live, minutes before it, a day once final.
        The state is both what the client knows of the game (its scoreboards) and what the copy itself says, the
        less final of the two: a pre-game or live copy is never kept as a final one, however final the board says
        the game is (nor is a final copy kept as long while a board says the game is still on)."""
        sport, path = LEAGUES[league]
        key = ("espn", "summary", league, str(event_id))

        def load() -> Dict[str, Any]:
            data = lean_summary(self._get(f"{SITE}/{sport}/{path}/summary", {"event": event_id}))
            self._remember_summary(league, event_id, data, self._cache_now())  # a background refresh's at once
            return data

        # This process's copy (a copy only in the shared table is judged by the life it was saved with)
        held = self.cache.peek(key, shared=False) if hasattr(self.cache, "peek") else None
        known = self.event_state(league, event_id)
        states = [state for state in ((known or {}).get("state"), self._copy_state(held[1]) if held else None)
                  if state] or ["pre"]
        rule = freshest(*(ttl_for("summary", state) for state in states))
        data, at = self._read(key, "summary", load, rule,
                              life=lambda copy: ttl_for("summary", self._copy_state(copy) or "pre"))
        self._remember_summary(league, event_id, data, at)  # the other process's copy, from the shared table
        return data

    def _copy_state(self, data: Any) -> Optional[str]:
        """A summary copy's own state ("pre", "live" or "final"; a pre-game copy whose start has passed is "live"),
        None if unreadable."""
        competitions = (data.get("header") or {}).get("competitions") if isinstance(data, dict) else None
        found = competition_state(competitions[0]) if competitions else None
        if found is None:
            return None
        state, _, _, start = found
        return "live" if state == "pre" and start is not None and start <= self.clock() else state

    def _remember_summary(self, league: str, event_id: Any, data: Any, at: float) -> None:
        competitions = (data.get("header") or {}).get("competitions") if isinstance(data, dict) else None
        if competitions:
            self._remember(league, event_id, competitions[0], at)

    def standings(self, league: str, season: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """Per-team wins, losses and average points for/against, keyed by ESPN team id, with the team's home record
        as ESPN writes it ("home": "W-L", the NFL's "W-L-T", the NHL's "W-L-OTL"; None when ESPN gives none, as
        in soccer) for data/context_inputs.py's home_rates."""
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
                        "home": _home_record(entry.get("stats", [])),
                    }
            return teams

        return self._cached(("espn", "standings", league, season), "standings", load,
                            state=None if season is None else "past")

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

        return self._cached(("espn", "roster", league, str(team_id)), "roster", load)

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

        return self._cached(("espn", "player", league, str(athlete_id)), "career", load)

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


def _home_record(stats: Any) -> Optional[str]:
    """A standings entry's home record: the stat of type "home" ("W-L", "W-L-T", "W-L-OTL"), else MLB's homeWins,
    homeLosses and homeTies as "W-L" (with "-T" when there were ties); None when there's neither."""
    by_name = {}
    for stat in stats or []:
        if not isinstance(stat, dict):
            continue
        if str(stat.get("type") or "").lower() == "home" and (stat.get("summary") or stat.get("displayValue")):
            return str(stat.get("summary") or stat.get("displayValue"))
        by_name[stat.get("name")] = stat.get("value")
    if by_name.get("homeWins") is None or by_name.get("homeLosses") is None:
        return None
    try:
        wins, losses, ties = (int(float(by_name.get(name) or 0)) for name in ("homeWins", "homeLosses", "homeTies"))
    except (TypeError, ValueError):
        return None
    return f"{wins}-{losses}-{ties}" if ties else f"{wins}-{losses}"


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
