"""Player pages from ESPN: search, game logs, splits, news, DraftKings props and a compact card.

Prop bets and the player page need a player's recent games, home/away form,
injury news and the lines a bookmaker has posted, and the AI assistant should
read a few hundred tokens about a player rather than ESPN's megabytes. ESPN's
public JSON has all of it for the NBA, WNBA, NFL, MLB and NHL, so this module
fetches it, keeps only what matters and caches it (10 to 60 minutes; a
finished season for a day).

Personal mode only: ESPN licenses its data for personal, noncommercial use
and bars automated and AI use, so a public launch must swap this adapter for
a licensed feed (see the product blueprint). The endpoints are undocumented
and change without notice, so nothing here raises for a data gap: every
method returns its usual structure, empty where data is missing, with a
plain-English ``reason`` the page or the assistant can show instead.

Stat keys: basketball columns get short keys (pts, reb, ast, fg3m, stl, blk,
tov, min) so props and hit rates line up across the app. Other sports keep
ESPN's labels (YDS, TD, REC, H, HR, RBI, G, A, ...); football's shared labels
get their group in front ("PASS YDS", "RUSH YDS", "REC YDS") and hockey's
shots column is called SOG. A stat can also be a sum such as "pts+reb+ast".

Prices are Decimal decimal odds worked out from DraftKings' American price.
ESPN's DraftKings over/under items carry no side label, so for those only
the line is kept: guessing which price is the over would be worse than none.
"""

from __future__ import annotations

import copy
import datetime
import html
import logging
import re
import threading
import unicodedata
from collections import Counter, OrderedDict
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Callable, Dict, Hashable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from .espn_client import COMMON, ESPN_DAY_TZ, LEAGUES, SITE, EspnClient
from .cache import freshest, ttl_for
from .utils import policy

SEARCH = "https://site.web.api.espn.com/apis/common/v3/search"
NEWS = "https://site.api.espn.com/apis/fantasy/v2/games/{game}/news/players"
CORE = "https://sports.core.api.espn.com/v2/sports"
DRAFTKINGS = "100"  # ESPN's provider id for DraftKings in the core odds API
MAX_PROP_PAGES = 5  # 500 props a page; a busy MLB game runs to three pages
MAX_GAMES = 200  # the most game-log rows a caller can ask for: an MLB season plus playoffs
MAX_NEWS = 20

log = logging.getLogger(__name__)

# Each lookup's kind of data in the shared cache's policy table (data/cache.py's ttl_for, read through
# data/utils.policy): a player's overview and game log keep for an hour on his game days and six hours otherwise,
# a finished season's log ("past") for a day; splits six hours; news ten minutes; DraftKings' props ten minutes,
# never served past that without being flagged stale (props)
KINDS = {
    "search": "search",
    "bio": "bio",
    "overview": "player",
    "gamelog": "gamelog",
    "splits": "splits",
    "news": "news",
    "props": "props",
    "schedule": "schedule",
}
GAME_DAY = "gameday"  # the policy state of a player whose team plays today (ESPN's day), or who isn't known yet
NEXT_GAMES_KEPT = 5000  # players whose next game date is remembered (for the game-day state)

# The fantasy game whose news feed carries one player's items (ESPN's ?athletes= filter doesn't)
FANTASY_GAMES = {
    "basketball_nba": "fba",
    "basketball_wnba": "wfba",
    "americanfootball_nfl": "ffl",
    "baseball_mlb": "flb",
    "icehockey_nhl": "fhl",
}
NEWS_SOURCES = {"Rotowire": "RotoWire", "Media": "ESPN Video", "Story": "ESPN", "HeadlineNews": "ESPN"}

BASKETBALL_KEYS = {"MIN": "min", "PTS": "pts", "REB": "reb", "AST": "ast", "3PT": "fg3m", "STL": "stl", "BLK": "blk", "TO": "tov"}
HOCKEY_KEYS = {"S": "SOG"}
# Football labels that mean different things in each group, so they always carry the group
FOOTBALL_SHARED = {"YDS", "AVG", "TD", "LNG", "ATT"}
GROUP_PREFIX = {
    "passing": "PASS", "rushing": "RUSH", "receiving": "REC", "fumbles": "FUM", "tackles": "TKL",
    "interceptions": "INT", "defensiveInterceptions": "INT", "defensive": "DEF", "fieldgoals": "FG",
    "pats": "XP", "kicking": "KICK", "punting": "PUNT", "returning": "RET", "kickReturns": "KR",
    "puntReturns": "PR", "scoring": "SCR",
}
# Words in ESPN's stat names that give a column's group when a response has no category list
NAME_GROUPS = (
    ("puntreturn", "PR"), ("kickreturn", "KR"), ("pass", "PASS"), ("rush", "RUSH"), ("receiv", "REC"),
    ("reception", "REC"), ("intercept", "INT"), ("fieldgoal", "FG"), ("punt", "PUNT"), ("fumble", "FUM"),
    ("tackle", "TKL"),
)
# Rates (and MLB's season-to-date columns) that can't be averaged across games or divided by them;
# OBA, BAA and OAVG are names for the batting average pitchers allow
RATE_KEYS = {
    "AVG", "OBP", "SLG", "OPS", "ERA", "WHIP", "GAA", "RTG", "QBR", "PROD", "GSC", "SPCT", "OBA", "BAA", "OAVG",
}

# ESPN prop name (lower case, "(incl. overtime)" dropped) -> (market key, game-log stat or None)
PROP_MARKETS: Dict[str, Dict[str, Tuple[str, Optional[str]]]] = {
    "basketball": {
        "total points": ("player_points", "pts"),
        "points milestones": ("player_points", "pts"),
        "total rebounds": ("player_rebounds", "reb"),
        "rebounds milestones": ("player_rebounds", "reb"),
        "total assists": ("player_assists", "ast"),
        "assists milestones": ("player_assists", "ast"),
        "total 3-point field goals": ("player_threes", "fg3m"),
        "3-point field goals milestones": ("player_threes", "fg3m"),
        "total steals": ("player_steals", "stl"),
        "steals milestones": ("player_steals", "stl"),
        "total blocks": ("player_blocks", "blk"),
        "blocks milestones": ("player_blocks", "blk"),
        "total turnovers": ("player_turnovers", "tov"),
        "total points and rebounds": ("player_points_rebounds", "pts+reb"),
        "points + rebounds milestones": ("player_points_rebounds", "pts+reb"),
        "total points and assists": ("player_points_assists", "pts+ast"),
        "points + assists milestones": ("player_points_assists", "pts+ast"),
        "total assists and rebounds": ("player_rebounds_assists", "reb+ast"),
        "rebounds + assists milestones": ("player_rebounds_assists", "reb+ast"),
        "total points, rebounds, and assists": ("player_points_rebounds_assists", "pts+reb+ast"),
        "points + assists + rebounds milestones": ("player_points_rebounds_assists", "pts+reb+ast"),
        "steals + blocks milestones": ("player_blocks_steals", "stl+blk"),
        "to record a double double": ("player_double_double", None),
        "to record a triple double": ("player_triple_double", None),
        "first basket": ("player_first_basket", None),
    },
    "football": {
        "total passing yards": ("player_pass_yds", "PASS YDS"),
        "total pass completions": ("player_pass_completions", "CMP"),
        "total passing attempts": ("player_pass_attempts", "PASS ATT"),
        "total passing touchdowns": ("player_pass_tds", "PASS TD"),
        "total passing interceptions": ("player_pass_interceptions", "INT"),
        "longest passing completion": ("player_pass_longest_completion", "PASS LNG"),
        "total passing plus rushing yards": ("player_pass_rush_yds", "PASS YDS+RUSH YDS"),
        "total rushing yards": ("player_rush_yds", "RUSH YDS"),
        "total carries": ("player_rush_attempts", "CAR"),
        "longest rush": ("player_rush_longest", "RUSH LNG"),
        "total receiving yards": ("player_reception_yds", "REC YDS"),
        "total receptions": ("player_receptions", "REC"),
        "longest reception": ("player_reception_longest", "REC LNG"),
        "total rushing plus receiving yards": ("player_rush_reception_yds", "RUSH YDS+REC YDS"),
        "passing yards milestones": ("player_pass_yds", "PASS YDS"),
        "passing completions milestones": ("player_pass_completions", "CMP"),
        "passing attempts milestones": ("player_pass_attempts", "PASS ATT"),
        "passing touchdown milestones": ("player_pass_tds", "PASS TD"),
        "interceptions thrown milestones": ("player_pass_interceptions", "INT"),
        "passing + rushing yards milestones": ("player_pass_rush_yds", "PASS YDS+RUSH YDS"),
        "rushing yards milestones": ("player_rush_yds", "RUSH YDS"),
        "rushing attempts milestones": ("player_rush_attempts", "CAR"),
        "receiving yards milestones": ("player_reception_yds", "REC YDS"),
        "receptions milestones": ("player_receptions", "REC"),
        "rushing + receiving yards milestones": ("player_rush_reception_yds", "RUSH YDS+REC YDS"),
        "total tackles": ("player_solo_tackles", "SOLO"),
        "tackles milestones": ("player_solo_tackles", "SOLO"),
        "total assists": ("player_tackle_assists", "AST"),
        "assists milestones (football)": ("player_tackle_assists", "AST"),
        "total tackles plus assists": ("player_tackles_assists", "TOT"),
        "tackles + assists milestones": ("player_tackles_assists", "TOT"),
        "total sacks": ("player_sacks", "SACK"),
        "sacks milestones": ("player_sacks", "SACK"),
        "total kicking points": ("player_kicking_points", "PTS"),
        "kicking points milestones": ("player_kicking_points", "PTS"),
        "total field goals made": ("player_field_goals", "FG"),
        "field goals made milestones": ("player_field_goals", "FG"),
        "total extra points made": ("player_pats", "XP"),
        "anytime touchdown scorer": ("player_anytime_td", None),
        "first touchdown scorer": ("player_1st_td", None),
        "last touchdown scorer": ("player_last_td", None),
    },
    "baseball": {
        "total hits": ("batter_hits", "H"),
        "hits milestones": ("batter_hits", "H"),
        "total runs scored": ("batter_runs_scored", "R"),
        "runs milestones": ("batter_runs_scored", "R"),
        "total rbis": ("batter_rbis", "RBI"),
        "rbis milestones": ("batter_rbis", "RBI"),
        "total hits + runs + rbis": ("batter_hits_runs_rbis", "H+R+RBI"),
        "hits + runs + rbis milestones": ("batter_hits_runs_rbis", "H+R+RBI"),
        "home runs milestones": ("batter_home_runs", "HR"),
        "total doubles hit": ("batter_doubles", "2B"),
        "doubles milestones": ("batter_doubles", "2B"),
        "triples milestones": ("batter_triples", "3B"),
        "extra base hits milestones": ("batter_extra_base_hits", "2B+3B+HR"),
        "total singles hit": ("batter_singles", None),
        "singles milestones": ("batter_singles", None),
        # Total bases = H + 2B + 2x3B + 3xHR; a stat sum may name a column more than once
        "total bases": ("batter_total_bases", "H+2B+3B+3B+HR+HR+HR"),
        "total bases milestones": ("batter_total_bases", "H+2B+3B+3B+HR+HR+HR"),
        "total stolen bases": ("batter_stolen_bases", "SB"),
        "stolen bases milestones": ("batter_stolen_bases", "SB"),
        "hits + stolen bases milestones": ("batter_hits_stolen_bases", "H+SB"),
        "hits + runs + stolen bases milestones": ("batter_hits_runs_stolen_bases", "H+R+SB"),
        "hits + walks + stolen bases milestones": ("batter_hits_walks_stolen_bases", "H+BB+SB"),
        "runs + rbis milestones": ("batter_runs_rbis", "R+RBI"),
        "total walks (batter)": ("batter_walks", "BB"),
        "walks (batter) milestones": ("batter_walks", "BB"),
        "strikeouts (batter) milestones": ("batter_strikeouts", "SO"),
        "total strikeouts": ("pitcher_strikeouts", "K"),
        "strikeouts thrown milestones": ("pitcher_strikeouts", "K"),
        "total hits allowed": ("pitcher_hits_allowed", "H"),
        "hits allowed milestones": ("pitcher_hits_allowed", "H"),
        "earned runs allowed": ("pitcher_earned_runs", "ER"),
        "earned runs allowed milestones": ("pitcher_earned_runs", "ER"),
        "total walks allowed": ("pitcher_walks", "BB"),
        "walks allowed milestones": ("pitcher_walks", "BB"),
        "hits + walks + earned runs allowed": ("pitcher_hits_walks_earned_runs", "H+BB+ER"),
        "hits + walks + earned runs milestones": ("pitcher_hits_walks_earned_runs", "H+BB+ER"),
        "total outs recorded": ("pitcher_outs", None),
        "to record win": ("pitcher_record_a_win", None),
        "first home run hitter": ("batter_first_home_run", None),
    },
    "hockey": {
        "total points": ("player_points", "PTS"),
        "points milestones": ("player_points", "PTS"),
        "total assists": ("player_assists", "A"),
        "assists milestones": ("player_assists", "A"),
        "goals milestones": ("player_goals", "G"),
        "total shots on goal": ("player_shots_on_goal", "SOG"),
        "shots on goal milestones": ("player_shots_on_goal", "SOG"),
        "powerplay points milestones": ("player_power_play_points", "PPG+PPA"),
        "total blocked shots": ("player_blocked_shots", None),
        "blocked shots milestones": ("player_blocked_shots", None),
        "total saves": ("player_total_saves", "SV"),
        "goalkeeper saves milestones": ("player_total_saves", "SV"),
        "anytime goalscorer": ("player_goal_scorer_anytime", None),
        "first goalscorer": ("player_goal_scorer_first", None),
        "last goalscorer": ("player_goal_scorer_last", None),
    },
}

# RotoWire stamps its notes like "Sun Oct 25 10:06:18 PDT 2026"
US_ZONES = {"EST": -5, "EDT": -4, "CST": -6, "CDT": -5, "MST": -7, "MDT": -6, "PST": -8, "PDT": -7, "UTC": 0, "GMT": 0}

_MADE_ATTEMPTED = re.compile(r"^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$")
_ATHLETE_REF = re.compile(r"/athletes/(\d+)")
# ESPN ids go into URL paths; str.isdigit() would also pass "²" and Arabic-Indic digits
_ESPN_ID = re.compile(r"\d{1,12}", re.ASCII)
_LEAGUE_KEYS = {espn: key for key, espn in LEAGUES.items()}


class _League(NamedTuple):
    key: str  # The Odds API key used across the app, e.g. "basketball_nba"
    sport: str  # ESPN path parts, e.g. "basketball" and "nba"
    path: str


def _resolve(league: Optional[str]) -> Optional[_League]:
    """The league for an Odds API key ("basketball_nba") or an ESPN slug ("nba", "basketball/nba")."""
    text = str(league or "").strip()
    if text in LEAGUES:
        return _League(text, *LEAGUES[text])
    for key, (sport, path) in LEAGUES.items():
        if text.lower() in (path, f"{sport}/{path}"):
            return _League(key, sport, path)
    return None


def _check(league: Optional[str], athlete_id: Any) -> Tuple[Optional[_League], str, Optional[str]]:
    """Resolve the league and validate the id (it goes into a URL path) before any request."""
    target = _resolve(league)
    athlete_id = str(athlete_id if athlete_id is not None else "").strip()
    if not target:
        return None, athlete_id, f"Player data doesn't cover {league!r}."
    if not _is_id(athlete_id):
        return target, athlete_id, "ESPN ids are numbers, e.g. 1966."
    return target, athlete_id, None


def _is_id(value: Any) -> bool:
    """True for an ESPN id: 1 to 12 ASCII digits and nothing else."""
    return bool(_ESPN_ID.fullmatch(str(value)))


def _count(value: Any, default: int, most: int) -> int:
    """A caller's count (it may come straight from a query string) as a whole number in 1..``most``.

    Anything that isn't a number gives ``default`` rather than an error.
    """
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(1, min(number, most))


CHANGED_SHAPE = "ESPN's response has changed shape"
UNREACHABLE = "ESPN couldn't be reached"


def _why(error: Optional[BaseException], missing: str) -> str:
    """A plain-English reason: ``missing`` when ESPN simply has nothing, else what went wrong."""
    if error is None or "HTTP 404" in str(error):
        return missing
    if isinstance(error, (KeyError, TypeError, ValueError, AttributeError, IndexError)):
        return f"{CHANGED_SHAPE}, so this part is unavailable."
    return f"{UNREACHABLE} ({str(error)[:120]})."


def _failed_part(card: Mapping[str, Any]) -> bool:
    """Whether a card was built while part of ESPN failed (not merely had nothing to show)."""
    texts = [str(card.get("reason") or "")] + [str(line) for line in card.get("missing") or []]
    return any(UNREACHABLE in text or CHANGED_SHAPE in text for text in texts)


class PlayerData:
    """Player lookups on top of an EspnClient, sharing its HTTP helper and its TTL cache."""

    def __init__(self, espn: EspnClient, now: Optional[Callable[[], datetime.datetime]] = None) -> None:
        self.espn = espn
        self.cache = espn.cache
        self._now = now or (lambda: datetime.datetime.now(datetime.timezone.utc))
        # (league, athlete id) -> his next game's date ("" when none is listed), from the overviews read so far
        self._next_games: "OrderedDict[Tuple[str, str], str]" = OrderedDict()
        self._next_lock = threading.Lock()

    # --- plumbing ---

    def _cached(self, key: Hashable, lookup: str, loader: Callable[[], Any],
                state: Optional[str] = None) -> Tuple[Any, Optional[BaseException]]:
        """The cached value (loaded if needed) or None plus the error; never raises.

        Every lookup goes through the shared cache with its kind (KINDS) and the policy table's time to live
        for ``state``. Loaders parse inside, so the cache keeps a few kilobytes per player instead of ESPN's
        full responses, and a changed response shape fails here like a network error does.
        """
        kind = KINDS[lookup]
        ttl, stale = policy(kind, state)
        try:
            return self.cache.get(key, ttl, loader, stale=stale, kind=kind), None
        except Exception as exc:  # network, HTTP and parsing errors alike
            log.info("ESPN player lookup %r failed: %s", key, exc)
            return None, exc

    def _note_next_game(self, target: _League, athlete_id: str, game: Optional[Mapping[str, Any]]) -> None:
        with self._next_lock:
            key = (target.key, athlete_id)
            self._next_games[key] = str((game or {}).get("date") or "")
            self._next_games.move_to_end(key)
            while len(self._next_games) > NEXT_GAMES_KEPT:
                self._next_games.popitem(last=False)

    def _player_state(self, target: _League, athlete_id: str) -> Optional[str]:
        """GAME_DAY when the player's next game is today (ESPN's day) or already under way, or isn't known yet
        (the shorter time to live); None on other days."""
        with self._next_lock:
            when = self._next_games.get((target.key, athlete_id))
        if when is None:
            return GAME_DAY
        start = _parse_utc(when)
        if start is None:
            return None
        today = self._now().astimezone(ESPN_DAY_TZ).date()
        return GAME_DAY if start.astimezone(ESPN_DAY_TZ).date() <= today else None

    @staticmethod
    def _key(kind: str, target: _League, *parts: Any) -> Tuple[Any, ...]:
        return ("espn-player", kind, target.key, *parts)

    def _fetched_at(self, key: Hashable) -> Optional[str]:
        """When the cached copy behind ``key`` came from ESPN, or None if nothing is cached."""
        stamp = self.cache.loaded_at(key)
        return _utc_text(datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc)) if stamp else None

    def _as_of(self, keys: Sequence[Hashable]) -> str:
        """When the oldest piece of data in a response was fetched (now if nothing is cached)."""
        stamps = [stamp for stamp in (self.cache.loaded_at(key) for key in keys) if stamp]
        if stamps:
            moment = datetime.datetime.fromtimestamp(min(stamps), datetime.timezone.utc)
        else:
            moment = self._now()
        return _utc_text(moment)

    # --- search ---

    def search(self, query: str, league: Optional[str] = None) -> Dict[str, Any]:
        """Players whose name matches ``query``: {query, players: [{id, name, league, team, ...}], reason}.

        Only leagues the app covers are returned, each with its Odds API key,
        so a result can go straight into the other lookups.
        """
        text = " ".join(str(query or "").split())[:60]
        result: Dict[str, Any] = {"query": text, "players": [], "reason": None}
        target = _resolve(league) if league else None
        if league and not target:
            result["reason"] = f"Player search doesn't cover {league!r}."
            return result
        if len(text) < 2:
            result["reason"] = "Type at least two letters of a player's name."
            return result
        players, error = self._cached(
            ("espn-player", "search", text.lower(), target.key if target else None),
            "search",
            lambda: self._search(text, target),
        )
        if players is None:
            result["reason"] = _why(error, "ESPN's player search returned nothing.")
            return result
        result["players"] = players
        if not players:
            where = f" in {target.key}" if target else ""
            result["reason"] = f"No player matching {text!r}{where}."
        return result

    def _search(self, text: str, target: Optional[_League]) -> List[Dict[str, Any]]:
        players = self._search_once(text, target)
        words = text.split()
        if not players and len(words) > 1:
            # ESPN matches whole words, so "jamarr chase" misses "Ja'Marr Chase": search the
            # surname and keep the names that read the same without punctuation
            wanted = _plain(text)
            players = [p for p in self._search_once(words[-1], target) if wanted in _plain(p["name"] or "")]
        return players[:10]

    def _search_once(self, text: str, target: Optional[_League]) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"query": text, "type": "player", "limit": 25}
        if target:
            params.update(sport=target.sport, league=target.path)
        items = self.espn._get(SEARCH, params).get("items") or []
        return [hit for hit in (_search_hit(item, target) for item in items) if hit]

    # --- profile ---

    def overview(self, league: str, athlete_id: Any) -> Dict[str, Any]:
        """Who the player is and what's next: team, position, injury, next game, ESPN's season line.

        ``season_stats`` is ESPN's own regular-season row: per game for
        basketball, season totals elsewhere (see ``per_game``).
        """
        target, athlete_id, problem = _check(league, athlete_id)
        result: Dict[str, Any] = {
            "id": athlete_id, "league": target.key if target else league, "name": None, "team": None,
            "team_abbr": None, "team_id": None, "position": None, "jersey": None, "headshot": None,
            "status": None, "injury": None, "next_game": None, "season_stats": None, "note": None, "reason": problem,
        }
        if problem:
            return result
        base = f"{COMMON}/{target.sport}/{target.path}/athletes/{athlete_id}"
        state = self._player_state(target, athlete_id)
        bio, bio_error = self._cached(
            self._key("bio", target, athlete_id), "bio", lambda: _parse_bio(self.espn._get(base)), state
        )
        summary, summary_error = self._cached(
            self._key("overview", target, athlete_id),
            "overview",
            lambda: _parse_overview(target.sport, self.espn._get(f"{base}/overview")),
            state,
        )
        if bio:
            result.update(bio)
        if summary:
            result["season_stats"] = summary["season_stats"]
            result["note"] = summary["note"]
            result["next_game"] = _next_game(summary["next_event"], result["team_id"])
        if result["next_game"] is None and result["team_id"]:
            # Between games the overview shows the previous one, so ask the team's schedule
            result["next_game"] = self._scheduled_game(target, result["team_id"])
        if bio:
            self._note_next_game(target, athlete_id, result["next_game"])
        if not bio:
            result["reason"] = _why(bio_error, f"ESPN has no {target.key} player with id {athlete_id}.")
        elif not summary:
            result["reason"] = "Season figures are unavailable: " + _why(summary_error, "ESPN has no overview.")
        return result

    def _scheduled_game(self, target: _League, team_id: str) -> Optional[Dict[str, Any]]:
        if not _is_id(team_id):
            return None
        url = f"{SITE}/{target.sport}/{target.path}/teams/{team_id}/schedule"
        events, _ = self._cached(
            self._key("schedule", target, team_id),
            "schedule",
            lambda: [_schedule_brief(event) for event in self.espn._get(url).get("events") or []],
        )
        # A postponed game can sit at "pre" with an old date, so skip anything long past
        cutoff = _utc_text(self._now() - datetime.timedelta(hours=12))
        upcoming = sorted(
            (e for e in events or [] if e["status"] in ("pre", "in") and (e["date"] or "") >= cutoff),
            key=lambda e: e["date"],
        )
        return _next_game(upcoming[0], team_id) if upcoming else None

    # --- games ---

    def gamelog(self, league: str, athlete_id: Any, last: int = 10) -> Dict[str, Any]:
        """The player's last ``last`` games (1 to 200), newest first, preseason left out.

        Early in a season the rows reach back into the previous one (each row
        says which). Rows: {event_id, date, opponent, home, result, score,
        season, season_type, minutes, stats: {key: value}}; ``labels`` names
        each stat key; ``as_of`` is when this season's log was fetched.
        """
        target, athlete_id, problem = _check(league, athlete_id)
        result: Dict[str, Any] = {
            "id": athlete_id, "league": target.key if target else league, "season": None,
            "season_games": 0, "rows": [], "labels": {}, "as_of": None, "reason": problem,
        }
        if problem:
            return result
        current, rows, labels, reason = self._recent(target, athlete_id, _count(last, 10, MAX_GAMES))
        result.update(rows=rows, labels=labels, reason=reason)
        if current:
            result.update(
                season=current["season"], season_games=len(current["rows"]),
                as_of=self._fetched_at(self._key("gamelog", target, athlete_id, None)),
            )
        return result

    def _season_log(
        self, target: _League, athlete_id: str, year: Optional[int] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[BaseException]]:
        url = f"{COMMON}/{target.sport}/{target.path}/athletes/{athlete_id}/gamelog"
        return self._cached(
            self._key("gamelog", target, athlete_id, year),
            "gamelog",
            lambda: _parse_gamelog(target.sport, self.espn._get(url, {"season": year} if year else None)),
            "past" if year else self._player_state(target, athlete_id),  # a finished season doesn't change
        )

    def _recent(
        self, target: _League, athlete_id: str, last: int
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str], Optional[str]]:
        """(this season's log, the last ``last`` rows, stat labels, reason or None)."""
        current, error = self._season_log(target, athlete_id)
        if current is None:
            return None, [], {}, _why(error, "ESPN has no game log for this player.")
        rows, labels = list(current["rows"]), dict(current["labels"])
        if len(rows) < last and current.get("year"):
            # Early in a season the last 10 games reach back into the previous one
            older, _ = self._season_log(target, athlete_id, current["year"] - 1)
            if older:
                rows += older["rows"]
                labels = {**older["labels"], **labels}
        rows = rows[:last]
        return current, rows, labels, None if rows else "No games logged this season or last."

    def splits(self, league: str, athlete_id: Any) -> Dict[str, Any]:
        """ESPN's season splits: home, away, total and by opponent, each {games, stats}.

        ``per_game`` follows ESPN's games-played column: basketball, hockey
        and MLB pitchers come back per game (rates such as ERA and OBA left
        as they are); football and MLB batters have no such column, so theirs
        are season totals (``per_game`` False). ``as_of`` is when ESPN was asked.
        """
        target, athlete_id, problem = _check(league, athlete_id)
        result: Dict[str, Any] = {
            "id": athlete_id, "league": target.key if target else league, "season": None, "per_game": False,
            "home": None, "away": None, "total": None, "by_opponent": {}, "as_of": None, "reason": problem,
        }
        if problem:
            return result
        url = f"{COMMON}/{target.sport}/{target.path}/athletes/{athlete_id}/splits"
        key = self._key("splits", target, athlete_id)
        data, error = self._cached(key, "splits", lambda: _parse_splits(target.sport, self.espn._get(url)))
        if data is None:
            result["reason"] = _why(error, "ESPN has no splits for this player.")
            return result
        result.update(data, as_of=self._fetched_at(key))
        if not (data["home"] or data["away"] or data["total"]):
            result["reason"] = "No splits yet this season."
        return result

    # --- news ---

    def news(self, league: str, athlete_id: Any, limit: int = 5) -> Dict[str, Any]:
        """The player's latest news (30 days): {items: [{headline, published, source, link, summary}], as_of, reason}.

        Up to ``limit`` items (1 to 20): written items first, then video
        clips, each newest first, duplicates dropped.
        """
        target, athlete_id, problem = _check(league, athlete_id)
        result: Dict[str, Any] = {
            "id": athlete_id, "league": target.key if target else league, "items": [], "as_of": None, "reason": problem,
        }
        if problem:
            return result
        game = FANTASY_GAMES.get(target.key)
        if not game:
            result["reason"] = f"ESPN has no per-player news feed for {target.key}."
            return result
        key = self._key("news", target, athlete_id)
        items, error = self._cached(
            key,
            "news",
            lambda: _parse_news(self.espn._get(NEWS.format(game=game), {"days": 30, "playerId": athlete_id})),
        )
        if items is None:
            result["reason"] = _why(error, "ESPN has no news for this player.")
            return result
        result["as_of"] = self._fetched_at(key)
        result["items"] = _pick_news(items, _count(limit, 5, MAX_NEWS))
        if not result["items"]:
            result["reason"] = "No news about this player in the last 30 days."
        return result

    # --- props ---

    def props(self, league: str, espn_event_id: Any) -> Dict[str, Any]:
        """DraftKings player props for one ESPN event: {players: {athlete_id: [prop, ...]}, ...}.

        Each prop: {market, stat, name, kind, line, open_line, over_price,
        under_price, yes_price, sides_known, label, updated}. ``kind`` is
        "over_under" (only the line: ESPN doesn't say which price is the
        over), "milestone" ("20+" = over 19.5, a clear over price) or "yes"
        (a one-sided bet such as first basket). ``stat`` matches the game-log
        keys so ``hit_rates`` can be run against the line. Team totals and
        game lines are left out.

        ``as_of`` is when these lines were fetched. If a refresh fails (ESPN
        down, or DraftKings has pulled the props) the last copy is served
        with ``stale`` True, so anything that prices a ticket can check its age.
        """
        target = _resolve(league)
        event_id = str(espn_event_id if espn_event_id is not None else "").strip()
        result: Dict[str, Any] = {
            "event_id": event_id, "league": target.key if target else league, "book": "DraftKings",
            "players": {}, "count": 0, "as_of": None, "stale": False, "reason": None,
        }
        if not target:
            result["reason"] = f"Props don't cover {league!r}."
            return result
        if not _is_id(event_id):
            result["reason"] = "ESPN event ids are numbers, e.g. 401909866."
            return result
        url = (
            f"{CORE}/{target.sport}/leagues/{target.path}/events/{event_id}"
            f"/competitions/{event_id}/odds/{DRAFTKINGS}/propBets"
        )
        key = ("espn-player", "props", target.key, event_id)
        # Kept as plain JSON (prices as text) so the bot and the dashboard share one copy; Decimal again here
        saved, error = self._cached(
            key, "props", lambda: _props_to_json(_parse_props(target.sport, self._prop_items(url))))
        if saved is None:
            result["reason"] = _why(error, "DraftKings hasn't posted player props for this game (or has taken them down).")
            return result
        players = _props_from_json(saved)
        # The cache serves its last copy when a refresh fails, so a copy older than the TTL is one ESPN didn't renew
        age = self.cache.age(key)
        result.update(as_of=self._fetched_at(key), stale=age is not None and age >= policy(KINDS["props"])[0])
        result["players"] = players
        result["count"] = sum(len(props) for props in players.values())
        if not players:
            result["reason"] = "No player props posted for this game yet."
        return result

    def _prop_items(self, url: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for page in range(1, MAX_PROP_PAGES + 1):
            data = self.espn._get(url, {"limit": 500, "page": page})
            items += data.get("items") or []
            if page >= int(data.get("pageCount") or 1):
                break
        return items

    # --- the compact card ---

    def card(self, league: str, athlete_id: Any) -> Dict[str, Any]:
        """The compact card (see _card), built once per time to live for everyone: a computed entry (kind
        "player_card") in the shared cache keyed by the league and the player, kept for its freshest input's time to
        live (the news', ten minutes)."""
        target, athlete_id, problem = _check(league, athlete_id)
        if problem:
            return self._card(league, athlete_id)
        state = self._player_state(target, athlete_id)
        fresh = freshest(ttl_for("player_card", state), *(ttl_for(KINDS[part], state)
                                                           for part in ("bio", "overview", "gamelog", "news")))
        key = ("computed", "player_card", target.key, athlete_id)
        # Built while part of ESPN failed: kept a few seconds for everyone, then rebuilt (TTLCache.get_computed)
        card = self.cache.get_computed(key, fresh.ttl, lambda: self._card(target.key, athlete_id), _failed_part,
                                       stale=fresh.stale, kind="player_card")
        return copy.deepcopy(card)

    def _card(self, league: str, athlete_id: Any) -> Dict[str, Any]:
        """Everything a player page or the AI needs in a few hundred tokens.

        {id, league, name, team, team_abbr, position, headshot, status, injury,
        next_game, note, season_label, season, last5, last10 (per-game
        averages), sample (games behind each average), games (last 10 rows),
        splits {home, away} (this season, per game), news (3 items), missing
        (parts that came back empty and why), as_of (when the oldest piece was
        fetched), reason (set when the player couldn't be found)}.
        """
        target, athlete_id, problem = _check(league, athlete_id)
        card: Dict[str, Any] = {
            "id": athlete_id, "league": target.key if target else league, "name": None, "team": None,
            "team_abbr": None, "position": None, "headshot": None, "status": None, "injury": None,
            "next_game": None, "note": None, "season_label": None, "season": {}, "last5": {}, "last10": {},
            "sample": {"season": 0, "last5": 0, "last10": 0}, "games": [], "splits": {"home": None, "away": None},
            "news": [], "missing": [], "as_of": _utc_text(self._now()), "reason": problem,
        }
        if problem:
            return card
        info = self.overview(target.key, athlete_id)
        for field in ("name", "team", "team_abbr", "position", "headshot", "status", "injury", "next_game", "note"):
            card[field] = info[field]
        current, rows, _, games_reason = self._recent(target, athlete_id, 10)
        season_rows = current["rows"] if current else []
        news = self.news(target.key, athlete_id, limit=3)
        card.update(
            season_label=current["season"] if current else None,
            season=average_stats(season_rows),
            last5=average_stats(rows[:5]),
            last10=average_stats(rows),
            sample={"season": len(season_rows), "last5": len(rows[:5]), "last10": len(rows)},
            games=rows,
            splits={"home": _home_away(season_rows, True), "away": _home_away(season_rows, False)},
            news=[{**item, "summary": _shorten(item["summary"], 160)} for item in news["items"]],
        )
        missing = []
        if info["reason"]:
            missing.append(f"profile: {info['reason']}")
        if games_reason:
            missing.append(f"games: {games_reason}")
        if news["reason"]:
            missing.append(f"news: {news['reason']}")
        if not card["next_game"]:
            missing.append("next game: none listed on ESPN")
        card["missing"] = missing
        card["reason"] = info["reason"] if card["name"] is None else None
        card["as_of"] = self._as_of([
            self._key("bio", target, athlete_id),
            self._key("overview", target, athlete_id),
            self._key("gamelog", target, athlete_id, None),
            self._key("news", target, athlete_id),
        ])
        return card


def _props_to_json(players: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Props with their Decimal prices as text (JSON for the shared cache's table)."""
    return {athlete: [{field: str(value) if isinstance(value, Decimal) else value for field, value in prop.items()}
                      for prop in props] for athlete, props in players.items()}


def _props_from_json(players: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """_props_to_json undone: the three prices as Decimal again (new dicts: the cached copy stays as it is)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for athlete, props in players.items():
        rows = []
        for prop in props:
            row = dict(prop)
            for field in ("over_price", "under_price", "yes_price"):
                if row.get(field) is not None:
                    row[field] = Decimal(str(row[field]))
            rows.append(row)
        out[athlete] = rows
    return out


def _parse_utc(text: Any) -> Optional[datetime.datetime]:
    try:
        moment = datetime.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=datetime.timezone.utc)


# --- hit rates and averages (pure functions over game-log rows) ---

def stat_value(row: Dict[str, Any], stat: str) -> Optional[float]:
    """A row's value for ``stat``: a key ("pts", "PASS YDS") or a sum ("pts+reb+ast"); None if missing."""
    stats = row.get("stats") or {}
    if stat in stats:  # also covers keys that contain "+", like hockey's "+/-"
        return stats[stat]
    total = 0.0
    for part in str(stat).split("+"):
        part = part.strip()
        value = stats.get(part)
        if value is None:  # ESPN labels are upper case; accept "yds" for "YDS"
            value = next((v for k, v in stats.items() if k.lower() == part.lower()), None)
        if value is None:
            return None
        total += value
    return total


def hit_rates(rows: Sequence[Dict[str, Any]], stat: str, line: Any) -> Dict[str, Any]:
    """How often ``stat`` went over ``line`` in the last 5 and 10 games and this season.

    Each rate carries its sample size ("4/5") because a bare percentage
    misleads when a player's minutes change. ``rows`` are game-log rows,
    newest first (pass the whole season for the season figure; it counts the
    rows from the newest row's season). Over means strictly above the line,
    so a whole-number line can push; pushes are counted separately.
    """
    result: Dict[str, Any] = {
        "stat": stat, "line": line, "last5": "0/0", "last10": "0/0", "season": "0/0",
        "games": {}, "hits": {}, "pushes": {}, "reason": None,
    }
    try:
        target = float(line)
    except (TypeError, ValueError):
        result["reason"] = "No line to compare against."
        return result
    played = []
    for row in rows:
        value = stat_value(row, stat)
        if value is not None:
            played.append((row.get("season"), value))
    season = played[0][0] if played else None
    windows = {"last5": played[:5], "last10": played[:10], "season": [p for p in played if p[0] == season]}
    for name, sample in windows.items():
        hits = sum(1 for _, value in sample if value > target)
        result[name] = f"{hits}/{len(sample)}"
        result["games"][name] = len(sample)
        result["hits"][name] = hits
        result["pushes"][name] = sum(1 for _, value in sample if value == target)
    if not played:
        result["reason"] = f"No games with a {stat} figure."
    return result


def average_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Per-game averages of the counting stats in ``rows`` (rates like FG% or ERA are skipped)."""
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in rows:
        for key, value in (row.get("stats") or {}).items():
            if _is_rate(key) or not isinstance(value, (int, float)):
                continue
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: round(totals[key] / counts[key], 2) for key in totals}


def _home_away(rows: Sequence[Dict[str, Any]], home: bool) -> Dict[str, Any]:
    sample = [row for row in rows if row.get("home") is home]
    return {"games": len(sample), "stats": average_stats(sample)}


def _is_rate(key: str) -> bool:
    return "%" in key or key in RATE_KEYS or key.endswith(" AVG")


# --- parsing ESPN responses ---

def _value(text: Any, label: str = "") -> Optional[float]:
    """A stat cell as a number: "1,309", ".241", "22:27" (minutes), "8-15" (made), "6.1" innings."""
    if text is None or isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return float(text)
    text = str(text).strip().replace(",", "")
    if not text or text in ("-", "--"):
        return None
    if ":" in text:  # time on ice or minutes, "22:27"
        minutes, _, seconds = text.partition(":")
        try:
            return int(minutes) + int(seconds) / 60
        except ValueError:
            return None
    made = _MADE_ATTEMPTED.match(text)
    if made:  # "8-15" field goals: the made count is the stat props settle on
        return float(made.group(1))
    try:
        number = float(text)
    except ValueError:
        return None  # e.g. a pitcher's decision "W(12-7)"
    if label == "IP":  # 6.1 innings is six innings and one out
        whole = int(number)
        return whole + round((number - whole) * 10) / 3
    return number


def _tidy(value: float) -> float:
    return int(value) if float(value).is_integer() else round(value, 3)


def _stat_keys(sport: str, labels: List[str], names: Sequence[Any], categories: Sequence[Any]) -> List[Optional[str]]:
    """One key per ESPN stat column (None = column dropped); see the module docstring."""
    if sport == "basketball":
        return [BASKETBALL_KEYS.get(label) for label in labels]
    groups = _column_groups(len(labels), names, categories)
    repeated = {label for label, count in Counter(labels).items() if count > 1}
    keys: List[Optional[str]] = []
    for label, group in zip(labels, groups):
        key = label
        if group and (label in repeated or (sport == "football" and label in FOOTBALL_SHARED)):
            key = f"{group} {label}"
        if sport == "hockey":
            key = HOCKEY_KEYS.get(key, key)
        keys.append(key)
    # A label still repeated (its group unknown) gets a column number so no value overwrites another
    seen: Counter = Counter()
    for index, key in enumerate(keys):
        seen[key] += 1
        if seen[key] > 1:
            keys[index] = f"{key} {seen[key]}"
    return keys


def _column_groups(count: int, names: Sequence[Any], categories: Sequence[Any]) -> List[str]:
    """Each column's group (PASS, RUSH, REC, ...) from ESPN's category counts, else from its stat names."""
    groups: List[str] = []
    for category in categories:
        name = str(category.get("name") or "")
        groups += [GROUP_PREFIX.get(name, name.upper()[:4])] * int(category.get("count") or 0)
    if len(groups) == count:
        return groups
    hinted = []
    for index in range(count):
        name = str(names[index]).lower() if index < len(names) else ""
        hinted.append(next((group for word, group in NAME_GROUPS if word in name), ""))
    return hinted


def _labels(data: Dict[str, Any]) -> List[str]:
    return [str(label).strip() for label in data.get("labels") or []]  # kickers' labels end in tabs


def _stats(keys: Sequence[Optional[str]], labels: Sequence[str], values: Sequence[Any]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for key, label, raw in zip(keys, labels, values):
        if key is None:
            continue
        value = _value(raw, label)
        if value is not None:
            stats[key] = _tidy(value)
    return stats


def _games_played(labels: Sequence[str], values: Sequence[Any]) -> Optional[int]:
    if "GP" not in labels or labels.index("GP") >= len(values):
        return None
    games = _value(values[labels.index("GP")])
    return int(games) if games else None


def _season_type(name: str) -> Tuple[str, str]:
    """"2026-27 Regular Season" -> ("2026-27", "regular"); also "postseason" and "preseason"."""
    season, _, rest = name.strip().partition(" ")
    lowered = rest.lower()
    if "preseason" in lowered:
        return season, "preseason"
    if "post" in lowered:
        return season, "postseason"
    if "regular" in lowered or not lowered:
        return season, "regular"
    return season, lowered


def _season_filter(filters: Sequence[Dict[str, Any]]) -> Tuple[Optional[int], Optional[str]]:
    """The season a game log covers: (2027, "2026-27")."""
    for item in filters:
        if item.get("name") != "season":
            continue
        value = str(item.get("value") or "")
        shown = next((o.get("displayValue") for o in item.get("options") or [] if str(o.get("value")) == value), None)
        return (int(value) if value.isdigit() else None), shown or value or None
    return None, None


def _parse_gamelog(sport: str, data: Dict[str, Any]) -> Dict[str, Any]:
    labels = _labels(data)
    keys = _stat_keys(sport, labels, data.get("names") or [], data.get("categories") or [])
    events = data.get("events") or {}
    rows: List[Dict[str, Any]] = []
    seen = set()
    for season_type in data.get("seasonTypes") or []:
        season, kind = _season_type(season_type.get("displayName") or "")
        if kind == "preseason":
            continue
        for category in season_type.get("categories") or []:
            for entry in category.get("events") or []:
                event_id = str(entry.get("eventId") or "")
                if not event_id or event_id in seen:
                    continue
                seen.add(event_id)
                stats = _stats(keys, labels, entry.get("stats") or [])
                rows.append(_game_row(event_id, events.get(event_id) or {}, stats, season, kind))
    rows.sort(key=lambda row: row["date"] or "", reverse=True)
    year, season = _season_filter(data.get("filters") or [])
    display = data.get("displayNames") or []
    # "8-15" cells keep the made count, so name them that way
    labels_by_key = {
        key: str(display[i]).replace("Made-Attempted", "Made") for i, key in enumerate(keys) if key and i < len(display)
    }
    return {"year": year, "season": season or (rows[0]["season"] if rows else None), "rows": rows, "labels": labels_by_key}


def _game_row(event_id: str, meta: Dict[str, Any], stats: Dict[str, float], season: str, kind: str) -> Dict[str, Any]:
    at_vs = meta.get("atVs")
    return {
        "event_id": event_id,
        "date": _iso(meta.get("gameDate")),
        "opponent": (meta.get("opponent") or {}).get("abbreviation"),
        "home": (at_vs == "vs") if at_vs in ("vs", "@") else None,
        "result": meta.get("gameResult"),
        "score": meta.get("score"),
        "season": season,
        "season_type": kind,
        "minutes": stats.get("min", stats.get("TOI/G")),
        "stats": stats,
    }


def _parse_bio(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    athlete = data.get("athlete") or {}
    if not athlete.get("id"):
        return None
    team = athlete.get("team") or {}
    injury = None
    if athlete.get("injuries"):
        first = athlete["injuries"][0] or {}
        details = first.get("details") or {}
        injury = {
            "status": first.get("status"),
            "detail": _clean(first.get("shortComment")),
            "type": details.get("type"),
            "date": _iso(first.get("date")),
            "return_date": _iso(details.get("returnDate")),
        }
    return {
        "id": str(athlete["id"]),
        "name": athlete.get("displayName"),
        "team": team.get("displayName"),
        "team_abbr": team.get("abbreviation"),
        "team_id": str(team["id"]) if team.get("id") else None,
        "position": (athlete.get("position") or {}).get("abbreviation"),
        "jersey": athlete.get("jersey"),
        "headshot": (athlete.get("headshot") or {}).get("href"),
        "status": (athlete.get("status") or {}).get("name"),
        "injury": injury,
    }


def _parse_overview(sport: str, data: Dict[str, Any]) -> Dict[str, Any]:
    stats = data.get("statistics") or {}
    labels = _labels(stats)
    keys = _stat_keys(sport, labels, stats.get("names") or [], stats.get("categories") or [])
    season_stats = None
    for split in stats.get("splits") or []:
        if split.get("displayName") == "Regular Season":
            values = split.get("stats") or []
            season_stats = {
                "label": "Regular Season",
                "per_game": sport == "basketball",
                "games": _games_played(labels, values),
                "stats": _stats(keys, labels, values),
            }
            break
    events = ((data.get("nextGame") or {}).get("league") or {}).get("events") or []
    rotowire = data.get("rotowire") or {}
    note = None
    if rotowire.get("headline"):
        note = {"headline": _clean(rotowire["headline"]), "published": _loose_time(rotowire.get("published")), "source": "RotoWire"}
    return {"season_stats": season_stats, "next_event": _event_brief(events[0]) if events else None, "note": note}


def _event_brief(event: Dict[str, Any]) -> Dict[str, Any]:
    """An overview's game: id, date, name, status ("pre", "in", "post") and both teams."""
    status = event.get("status")
    if isinstance(status, dict):
        status = ((status.get("type") or {}).get("state"))
    return {
        "event_id": str(event.get("id") or ""),
        "date": _iso(event.get("date")),
        "name": event.get("shortName"),
        "status": status,
        "competitors": [
            {"id": str(c.get("id") or ""), "abbr": c.get("abbreviation"), "name": c.get("displayName"),
             "home": c.get("homeAway") == "home"}
            for c in event.get("competitors") or []
        ],
    }


def _schedule_brief(event: Dict[str, Any]) -> Dict[str, Any]:
    """A team-schedule game in the same shape as ``_event_brief``."""
    competition = (event.get("competitions") or [{}])[0]
    return {
        "event_id": str(event.get("id") or ""),
        "date": _iso(event.get("date")),
        "name": event.get("shortName"),
        "status": ((competition.get("status") or {}).get("type") or {}).get("state"),
        "competitors": [
            {"id": str((c.get("team") or {}).get("id") or ""), "abbr": (c.get("team") or {}).get("abbreviation"),
             "name": (c.get("team") or {}).get("displayName"), "home": c.get("homeAway") == "home"}
            for c in competition.get("competitors") or []
        ],
    }


def _next_game(event: Optional[Dict[str, Any]], team_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """The game as the player sees it (opponent, home or away), if it is still to come or under way."""
    if not event or event.get("status") not in ("pre", "in"):
        return None
    ours = next((c for c in event["competitors"] if team_id and c["id"] == str(team_id)), None)
    theirs = next((c for c in event["competitors"] if ours and c is not ours), None)
    return {
        "event_id": event["event_id"],
        "date": event["date"],
        "name": event["name"],
        "status": event["status"],
        "home": ours["home"] if ours else None,
        "opponent": theirs["abbr"] if theirs else None,
        "opponent_name": theirs["name"] if theirs else None,
    }


def _parse_splits(sport: str, data: Dict[str, Any]) -> Dict[str, Any]:
    labels = _labels(data)
    keys = _stat_keys(sport, labels, data.get("names") or [], [])
    has_games = "GP" in labels
    result: Dict[str, Any] = {
        "season": data.get("displayName"), "per_game": sport == "basketball" or has_games,
        "home": None, "away": None, "total": None, "by_opponent": {},
    }
    for category in data.get("splitCategories") or []:
        for split in category.get("splits") or []:
            values = split.get("stats") or []
            games = _games_played(labels, values)
            stats = _stats(keys, labels, values)
            stats.pop("GP", None)
            if sport != "basketball" and games:
                # Hockey splits are season totals with games played: make them per game
                stats = {k: v if _is_rate(k) or k.endswith("/G") else round(v / games, 2) for k, v in stats.items()}
            entry = {"games": games, "stats": stats}
            name = str(split.get("displayName") or "").lower()
            if category.get("name") == "byOpponent":
                # Out of season ESPN fills this group with copies of the season total
                if name != "all splits":
                    result["by_opponent"][split.get("abbreviation") or split.get("displayName")] = entry
            elif name == "home" and result["home"] is None:
                result["home"] = entry
            elif name in ("road", "away") and result["away"] is None:
                result["away"] = entry
            elif name == "all splits" and result["total"] is None:
                result["total"] = entry
    return result


def _parse_news(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen = set()
    for entry in data.get("feed") or []:
        headline = _clean(entry.get("headline"))
        if not headline or headline.lower() in seen:
            continue
        seen.add(headline.lower())
        kind = entry.get("type") or ""
        links = entry.get("links") or {}
        link = next(
            (links[k]["href"] for k in ("web", "mobile") if isinstance(links.get(k), dict) and links[k].get("href")),
            None,
        )
        # RotoWire's "story" is the useful analysis; elsewhere it is the whole article
        summary = entry.get("story") if kind == "Rotowire" else entry.get("description")
        items.append({
            "headline": headline,
            "published": _iso(entry.get("published")),
            "source": NEWS_SOURCES.get(kind, "ESPN"),
            "link": link,
            "summary": _shorten(_clean(summary or entry.get("description")), 240),
        })
    return items


def _pick_news(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Written items first, newest first; video clips only fill the gaps after them."""
    newest = sorted(items, key=lambda item: item["published"] or "", reverse=True)
    written = [item for item in newest if item["source"] != "ESPN Video"]
    clips = [item for item in newest if item["source"] == "ESPN Video"]
    return (written + clips)[:limit]


def _parse_props(sport: str, items: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    # DraftKings posts an over/under as two items with the same line, so group before reading prices
    groups: Dict[Tuple[str, str, Optional[float], Optional[str]], List[Dict[str, Any]]] = {}
    for item in items:
        match = _ATHLETE_REF.search(str((item.get("athlete") or {}).get("$ref") or ""))
        if not match:
            continue  # team totals and game lines
        name = str((item.get("type") or {}).get("name") or "")
        target = (item.get("current") or {}).get("target") or {}
        line = _float(target.get("value"))
        if line is None:
            line = _float(((item.get("odds") or {}).get("total") or {}).get("value"))
        groups.setdefault((match.group(1), name, line, target.get("displayValue")), []).append(item)
    players: Dict[str, List[Dict[str, Any]]] = {}
    for (athlete_id, name, line, shown), group in groups.items():
        players.setdefault(athlete_id, []).append(_prop_entry(sport, name, line, shown, group))
    return players


def _prop_entry(sport: str, name: str, line: Optional[float], shown: Optional[str], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    market, stat = _market(sport, name)
    first = items[0]
    opened = _float(((first.get("open") or {}).get("target") or {}).get("value"))
    prices = [_price(item.get("odds")) for item in items]
    entry: Dict[str, Any] = {
        "market": market, "stat": stat, "name": name, "kind": "over_under", "line": line, "open_line": opened,
        "over_price": None, "under_price": None, "yes_price": None, "sides_known": False,
        "label": shown, "updated": _iso(first.get("lastUpdated")),
    }
    if line is not None and ("milestone" in name.lower() or str(shown or "").endswith("+")):
        # "20+" pays if the player reaches 20: the over side of 19.5. ESPN's "open" on a milestone
        # item is often another rung ("6+" on the "2+" item), so it only counts when it is this one.
        same_rung = opened is not None and opened == line
        entry.update(
            kind="milestone", line=line - 0.5, open_line=line - 0.5 if same_rung else None,
            label=shown or f"{line:g}+",
        )
        if len(items) == 1:
            entry.update(over_price=prices[0], sides_known=True)
    elif line is None:
        # One item is a yes-only bet (first basket); a pair is yes/no with no way to tell which is which
        entry["kind"] = "yes"
        if len(items) == 1:
            entry.update(yes_price=prices[0], sides_known=True)
    else:
        sides = {_side_label(item): price for item, price in zip(items, prices)}
        if "over" in sides or "under" in sides:
            entry.update(over_price=sides.get("over"), under_price=sides.get("under"), sides_known=True)
    return entry


def _market(sport: str, name: str) -> Tuple[str, Optional[str]]:
    plain = re.sub(r"\s*\((?:incl|excl)[^)]*\)", "", name, flags=re.IGNORECASE).strip().lower()
    known = PROP_MARKETS.get(sport, {}).get(plain)
    if known:
        return known
    return re.sub(r"[^a-z0-9]+", "_", plain).strip("_"), None


def _side_label(item: Dict[str, Any]) -> Optional[str]:
    """"over"/"under" when an item names its side (ESPN's DraftKings items usually don't)."""
    for field in ("side", "outcome", "description"):
        text = item.get(field)
        if isinstance(text, dict):
            text = text.get("name") or text.get("type")
        text = str(text or "").strip().lower()
        if text.startswith("over"):
            return "over"
        if text.startswith("under"):
            return "under"
    return None


def _price(odds: Optional[Dict[str, Any]]) -> Optional[Decimal]:
    """Decimal odds worked out from the American price (ESPN's own decimal is cut to two places)."""
    odds = odds or {}
    american = str((odds.get("american") or {}).get("value") or "").replace("+", "").strip()
    try:
        value = int(american)
    except ValueError:
        value = 0
    try:
        if value >= 100:
            price = 1 + Decimal(value) / 100
        elif value <= -100:
            price = 1 + Decimal(100) / Decimal(-value)
        else:
            price = Decimal(str((odds.get("decimal") or {}).get("value")))
            if not price > 1:
                return None
    except (InvalidOperation, ValueError):
        return None
    return price.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _search_hit(item: Dict[str, Any], target: Optional[_League]) -> Optional[Dict[str, Any]]:
    if item.get("type") != "player" or not item.get("id"):
        return None
    espn_league = (item.get("sport"), item.get("league"))
    if target and espn_league != (target.sport, target.path):
        return None
    key = target.key if target else _LEAGUE_KEYS.get(espn_league)
    if not key:
        return None  # a league the app doesn't cover (cups, friendlies, ...)
    team = (item.get("teamRelationships") or [{}])[0] or {}
    return {
        "id": str(item["id"]),
        "name": item.get("displayName"),
        "league": key,
        "team": team.get("displayName"),
        "team_abbr": (team.get("core") or {}).get("abbreviation"),
        "position": (item.get("position") or {}).get("abbreviation"),
        "headshot": (item.get("headshot") or {}).get("href"),
        "active": item.get("isActive"),
    }


# --- text and time helpers ---

def _plain(text: str) -> str:
    """Lowercase, accent-free name with apostrophes and dots dropped: "Ja'Marr" -> "jamarr"."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"['.’]", "", text)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _clean(text: Any) -> str:
    """Plain text from ESPN's sometimes-HTML strings."""
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    return " ".join(html.unescape(text).split())


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _utc_text(moment: datetime.datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    return moment.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso(text: Any) -> Optional[str]:
    """ESPN's assorted ISO timestamps as "2026-10-25T23:30:00Z" (unparseable text is returned as is)."""
    if not text:
        return None
    try:
        moment = datetime.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return str(text)
    return _utc_text(moment)


def _loose_time(text: Any) -> Optional[str]:
    """RotoWire's "Sun Oct 25 10:06:18 PDT 2026" (or an ISO time) as UTC ISO text."""
    parts = str(text or "").split()
    if len(parts) == 6 and parts[4] in US_ZONES:
        try:
            moment = datetime.datetime.strptime(" ".join(parts[:4] + parts[5:]), "%a %b %d %H:%M:%S %Y")
        except ValueError:
            return str(text)
        zone = datetime.timezone(datetime.timedelta(hours=US_ZONES[parts[4]]))
        return _utc_text(moment.replace(tzinfo=zone))
    return _iso(text)
