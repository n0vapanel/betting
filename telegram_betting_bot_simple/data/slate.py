"""One all-sports slate for Today: built in the background, served as ready-made bytes.

Today used to load each league's odds board when someone opened it, at most eight leagues at a time. The
slate instead holds every game the app can see for today and the next six days, in one small JSON
document that the API serves as it is (GET /api/slate, with an ETag, so an unchanged slate is a 304):

- Leagues are the union of The Odds API's in-season leagues (its free /v4/sports listing, which the odds
  service caches for hours) and the leagues ESPN has scoreboards for (espn_client.LEAGUES and
  SCOREBOARD_ONLY). Every in-season league is listed with its game count, even one with no games yet.
- ESPN's scoreboards are fetched one day at a time (ESPN refuses date ranges), eight at once, each on its
  own schedule: boards with a game live (or about to start) every LIVE seconds, today's other boards
  every TODAY seconds while they have games still to finish and every IDLE seconds when they don't, and
  the days ahead every FUTURE seconds. A board that fails keeps its last good copy (the last slate's games for
  that league and day, which is also what a restart starts from) and is retried ever further apart, from its
  own pace (from a minute while there is nothing to show for it); when most boards of a round fail (an ESPN
  outage or rate limit), only boards with a game on are fetched for the next 5 to 15 minutes. The boards of a
  round are stamped with its start, so they come due together: one wake-up and one rebuild per round. The Odds API's listing is re-read in the background at
  most every 15 minutes, so a slow listing never holds up the scores.
- Odds come only from each league's saved snapshot (Store.odds_snapshot): building the slate never calls
  The Odds API. An odds event is the ESPN game in the same league starting within 3 hours whose teams'
  names match (espn_client.team_match_score); it then carries the odds feed's orientation and id, so a
  price added to a slip names the same side the server prices. Leagues ESPN doesn't cover take their games
  from the snapshot alone. The numbers are code's: the best price per side with its bookmaker, the
  market's margin-free chances, and the NBA model's own chance where it has already made one (its logged
  pre-game prediction; nothing new is computed here).
- A league whose odds are older than the odds service's 2 hours can ask for a new board (request_odds,
  POST /api/slate/odds): in the background, at most once per league per 2 hours (a lease in the shared
  store keeps that across restarts), never below the odds service's reserve (counting the boards already on
  their way), at most ODDS_PER_DAY boards a day, and never for a league ESPN shows no games for. Each board
  costs 3 of the month's credits per odds region.

The slate also carries the month's Odds API quota as last reported ({"remaining", "used"}, read from the
shared store, never fetched), for the header's "N odds left".

The slate is serialised once per change, kept in memory as raw bytes, gzip bytes and an ETag (a hash of
the bytes), and saved to the store, so a restarted dashboard serves the last slate before its first
refresh. Nothing in it changes unless the games do, so an unchanged slate keeps its ETag.
"""

from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from .espn_client import ESPN_DAY_TZ, LEAGUES as ESPN_LEAGUES, SCOREBOARD_ONLY, espn_path, team_match_score
from .leagues import POPULAR_LEAGUES, SPORTS
from .odds_service import MARKETS, summarize_markets

log = logging.getLogger(__name__)

SLATE_NAME = "today"  # the store's name for it
DAYS_AHEAD = 6  # today and the next six days
WORKERS = 8  # ESPN scoreboards fetched at once
MATCH_WINDOW = 3 * 3600  # an odds event and an ESPN game further apart than this are different games
MATCH = 0.5  # both teams' names at least this close (team_match_score) ...
STRONG_MATCH = 0.8  # ... or one team's this close: a team plays one game in a 3-hour window
LIVE_SOON = 10 * 60  # a game starting this soon refreshes at the live pace
STUCK = 6 * 3600  # ... but not a "scheduled" game this long past its start (ESPN never updated it)
FETCH_COST = len(MARKETS)  # Odds API credits one league's board costs per region (moneyline, spreads, totals)
ODDS_TTL = 2 * 3600  # odds older than this are stale (the odds service's own odds_ttl when it has one)
ODDS_PER_DAY = 12  # boards request_odds may load in 24 hours, all leagues together
BACKOFF_CAP = 30 * 60  # a failing board is retried at most this far apart (or at its own pace, if slower) ...
LIVE_BACKOFF_CAP = 5 * 60  # ... and one with a game on, this far
BREAKER_MIN = 8  # when at least this many boards of a round fail, and more than half of them, ...
BREAKER_PAUSE = (5 * 60, 10 * 60, 15 * 60)  # ... only live boards are fetched for this long (longer each time in a row)
SEASON_EVERY = 15 * 60  # The Odds API's in-season listing is re-read this often (its service caches it for hours) ...
SEASON_RETRY = 10 * 60  # ... and this long after a failed read
AI_LEAGUES = frozenset({"basketball_nba"})  # as dashboard/service.py: leagues with the NBA model
B2B_SPORTS = ("basketball", "hockey")  # where playing the day before is a badge
TENNIS_PREFIXES = {"tennis_atp_": "tennis_atp", "tennis_wta_": "tennis_wta"}  # odds tournaments -> ESPN tours
SINGLES = ("mens-singles", "womens-singles")
# A game without ESPN data is shown as live this long after its start, then dropped (its result is unknown)
ODDS_ONLY_LIVE = (("cricket_test_match", 5 * 24 * 3600), ("cricket_", 10 * 3600), ("", 4 * 3600))
POSTPONED = ("POSTPONED", "CANCELED", "CANCELLED", "SUSPENDED", "FORFEIT", "ABANDONED", "DELAYED_INDEFINITELY")
GROUP_OF_SPORT = {"basketball": "Basketball", "hockey": "Ice Hockey", "soccer": "Soccer", "tennis": "Tennis",
                  "baseball": "Baseball", "football": "American Football", "mma": "Mixed Martial Arts"}
LABELS = {group: label for label, group in SPORTS}
LABELS.update({"Cricket": "🏏 Cricket", "Rugby League": "🏉 Rugby League", "Rugby Union": "🏉 Rugby Union",
               "Aussie Rules": "🏉 Aussie Rules", "Boxing": "🥊 Boxing", "Handball": "🤾 Handball",
               "Lacrosse": "🥍 Lacrosse", "Golf": "⛳ Golf"})
GROUP_ORDER = [group for _, group in SPORTS]
# Titles for leagues ESPN shows when The Odds API's listing doesn't name them (not in season there, or ESPN's own)
TITLES = {
    "tennis_atp": "ATP", "tennis_wta": "WTA", "mma_mixed_martial_arts": "MMA", "basketball_nba": "NBA",
    "basketball_wnba": "WNBA", "basketball_ncaab": "NCAAB", "basketball_wncaab": "NCAAW", "basketball_nbl": "NBL",
    "americanfootball_nfl": "NFL", "americanfootball_ncaaf": "NCAAF", "americanfootball_cfl": "CFL",
    "icehockey_nhl": "NHL", "baseball_mlb": "MLB", "soccer_epl": "EPL", "soccer_usa_mls": "MLS",
    "soccer_uefa_champs_league": "UEFA Champions League", "soccer_uefa_europa_league": "UEFA Europa League",
}
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏ -‮⁦-⁩]")


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(text: Any) -> Optional[float]:
    """Epoch seconds from an ISO time ("2026-10-26T23:00Z", "...:00Z", "+00:00"), or None."""
    if not isinstance(text, str) or not text:
        return None
    value = text.strip().replace("Z", "+00:00")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}\+00:00", value):
        value = value.replace("+00:00", ":00+00:00")
    try:
        moment = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    return moment.timestamp()


def _text(value: Any, limit: int = 80) -> Optional[str]:
    """Outside text, tidied: no control or direction characters, one space between words, at most ``limit``
    characters. The app still escapes it wherever it is shown."""
    if value is None or isinstance(value, (dict, list)):
        return None
    text = " ".join(CONTROL.sub(" ", str(value)).split())
    return text[:limit] or None


def _logo(url: Any) -> Optional[str]:
    """An ESPN image address, only over HTTPS from ESPN's image hosts; anything else is dropped."""
    if not isinstance(url, str) or len(url) > 300:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not (host == "espncdn.com" or host.endswith(".espncdn.com")):
        return None
    if any(ch in url for ch in "\"'<> \\"):
        return None
    return url


def _lean(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """The mapping without its None values (the slate omits nulls)."""
    return {key: value for key, value in mapping.items() if value is not None}


def _int(value: Any) -> Optional[int]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() and abs(number) < 10_000 else None


def _odds_only_live(league: str) -> float:
    return next(seconds for prefix, seconds in ODDS_ONLY_LIVE if league.startswith(prefix))


def _answer(queued: bool, code: str, reason: str) -> Dict[str, Any]:
    return {"queued": queued, "reason": reason, "code": code}


def _tour(league: str) -> Optional[str]:
    """The ESPN tour (tennis_atp, tennis_wta) whose board shows an odds feed tennis tournament's matches."""
    return next((tour for prefix, tour in TENNIS_PREFIXES.items() if league.startswith(prefix)), None)


def _sport(league: str) -> str:
    """ESPN's sport for a league key ("basketball", "soccer", ...), "" when ESPN doesn't show it."""
    return (espn_path(league) or espn_path(_tour(league) or "") or ("",))[0]


# --- ESPN's scoreboards ----------------------------------------------------------------------------

def _status(status: Mapping[str, Any]) -> str:
    kind = status.get("type") or {}
    name = str(kind.get("name") or "").upper()
    if any(word in name for word in POSTPONED):
        return "postponed"
    state = kind.get("state")
    if state == "in":
        return "live"
    if state == "post" or kind.get("completed"):
        return "final"
    return "pre"


def _clock(sport: str, league: str, status: Mapping[str, Any]) -> Optional[str]:
    """A live game's clock, short: "Q3 4:12", "H2 11:05", "P2 8:40", "67'", "Top 5th", "R2 3:12", "Set 2"."""
    kind = status.get("type") or {}
    period = _int(status.get("period")) or 0
    clock = _text(status.get("displayClock"), 12)
    if str(kind.get("name") or "") != "STATUS_IN_PROGRESS" or not period:
        return _text(kind.get("shortDetail") or kind.get("detail"), 24)
    if sport == "soccer":
        return clock or _text(kind.get("shortDetail"), 24)
    if sport in ("basketball", "football"):
        halves = league in ("basketball_ncaab",)
        regular = 2 if halves else 4
        label = (f"H{period}" if halves else f"Q{period}") if period <= regular else (
            "OT" if period == regular + 1 else f"{period - regular}OT")
        return f"{label} {clock}" if clock else label
    if sport == "hockey":
        label = f"P{period}" if period <= 3 else ("OT" if period == 4 else "SO")
        return f"{label} {clock}" if clock else label
    if sport == "mma":
        return f"R{period} {clock}" if clock else f"R{period}"
    if sport == "tennis":
        return f"Set {period}"
    return _text(kind.get("shortDetail"), 24)


def _record(competitor: Mapping[str, Any]) -> Optional[str]:
    records = competitor.get("records") or competitor.get("record") or []
    if not isinstance(records, list):
        return None
    best = next((r for r in records if isinstance(r, dict) and r.get("type") == "total"), None) or next(
        (r for r in records if isinstance(r, dict)), None)
    return _text((best or {}).get("summary"), 16)


def _side(competitor: Mapping[str, Any], sport: str, status: str) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
    """(the team as the slate shows it, its names for matching, ESPN's team id)."""
    team = competitor.get("team")
    if isinstance(team, dict):
        names = {field: team.get(field) for field in ("displayName", "shortDisplayName", "name", "nickname", "location")}
        shown = {"name": _text(team.get("displayName") or team.get("name")), "short": _text(team.get("shortDisplayName"), 40),
                 "abbr": _text(team.get("abbreviation"), 8), "logo": _logo(team.get("logo"))}
        team_id = team.get("id")
    else:
        athlete = competitor.get("athlete") or {}
        names = {"displayName": athlete.get("displayName"), "shortDisplayName": athlete.get("shortName"),
                 "name": athlete.get("fullName")}
        shown = {"name": _text(athlete.get("displayName") or athlete.get("fullName")),
                 "short": _text(athlete.get("shortName"), 40), "logo": _logo((athlete.get("flag") or {}).get("href"))}
        team_id = competitor.get("id")
    score = None
    if status in ("live", "final"):
        if sport == "tennis":
            score = sum(1 for period in competitor.get("linescores") or [] if isinstance(period, dict) and period.get("winner"))
        elif sport != "mma":
            score = _int(competitor.get("score"))
    shown["score"] = score
    shown["record"] = _record(competitor)
    return _lean(shown), names, team_id


def parse_competition(league: str, sport: str, competition: Mapping[str, Any], event_id: Any,
                      event: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """One ESPN game in the slate's form, with its matching names and team ids under "_" keys; None when it
    hasn't two sides or a start time."""
    start = _parse_time(competition.get("date") or competition.get("startDate") or event.get("date"))
    competitors = [c for c in competition.get("competitors") or [] if isinstance(c, dict)]
    if start is None or len(competitors) != 2:
        return None
    by_side = {c.get("homeAway"): c for c in competitors}
    if "home" in by_side and "away" in by_side:
        home, away = by_side["home"], by_side["away"]
    else:  # fights: no home or away, so the first-listed fighter is shown first, as "home"
        home, away = sorted(competitors, key=lambda c: _int(c.get("order")) or 0)
    status_info = competition.get("status") or event.get("status") or {}
    status = _status(status_info)
    home_view, home_names, home_id = _side(home, sport, status)
    away_view, away_names, away_id = _side(away, sport, status)
    if not home_view.get("name") or not away_view.get("name"):
        return None
    venue = competition.get("venue") or {}
    address = venue.get("address") or {}
    venue_view = _lean({"name": _text(venue.get("fullName"), 80), "city": _text(address.get("city"), 40),
                        "indoor": venue.get("indoor") if isinstance(venue.get("indoor"), bool) else None})
    return {
        "espn_id": _text(event_id, 40),
        "start": _iso(start),
        "status": status,
        "clock": _clock(sport, league, status_info) if status == "live" else None,
        "home": home_view,
        "away": away_view,
        "venue": venue_view or None,
        "_ts": start,
        "_neutral": bool(competition.get("neutralSite")),
        "_names": (home_names, away_names),
        "_ids": (home_id, away_id),
    }


def parse_scoreboard(league: str, events: Iterable[Any]) -> List[Dict[str, Any]]:
    """ESPN's scoreboard events as slate games. A team sport's event is one game; a tennis event is a
    tournament (its singles matches are the games) and an MMA event a card (its fights)."""
    sport = _sport(league)
    games = []
    for event in events or []:
        if not isinstance(event, dict):
            continue
        if sport == "tennis":
            pairs = [(competition, competition.get("id")) for grouping in event.get("groupings") or []
                     if ((grouping.get("grouping") or {}).get("slug") in SINGLES)
                     for competition in grouping.get("competitions") or []]
        elif sport == "mma":
            pairs = [(competition, competition.get("id")) for competition in event.get("competitions") or []]
        else:
            pairs = [(competition, event.get("id")) for competition in (event.get("competitions") or [])[:1]]
        for competition, game_id in pairs:
            if not isinstance(competition, dict):
                continue
            try:
                game = parse_competition(league, sport, competition, game_id, event)
            except (AttributeError, TypeError, ValueError, KeyError):
                game = None  # one odd entry costs only itself, not the league's board
            if game is not None:
                games.append(game)
    return games


# --- odds snapshots ----------------------------------------------------------------------------------

def odds_view(event: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """(best prices, market chances) of one saved odds event: {"h2h": {"home": [price, book], ...},
    "spreads": {"home": [price, book, point], ...}, "totals": {...}} and {"home", "away", "draw", "src"}."""
    try:
        summary = summarize_markets(dict(event))
    except Exception:  # a malformed saved event: no prices rather than no slate
        return None, None
    prices: Dict[str, Dict[str, List[Any]]] = {}
    for market in ("h2h", "spreads", "totals"):
        sides = summary.get(market) or {}
        view = {}
        for side, offer in sides.items():
            price = offer.get("price")
            if not isinstance(price, (int, float)) or price <= 1:
                continue
            entry: List[Any] = [round(float(price), 3), _text(offer.get("book"), 40) or "?"]
            if market != "h2h":
                entry.append(offer.get("point"))
            view[side] = entry
        if view:
            prices[market] = view
    fair = {side: offer.get("fair") for side, offer in (summary.get("h2h") or {}).items()}
    chance = None
    if isinstance(fair.get("home"), float) and isinstance(fair.get("away"), float):
        chance = _lean({"home": round(fair["home"], 3), "away": round(fair["away"], 3),
                        "draw": round(fair["draw"], 3) if isinstance(fair.get("draw"), float) else None,
                        "src": "market"})
    return prices or None, chance


@dataclass
class _Board:
    """One league's scoreboard for one day."""

    fetched: float = 0.0  # when it was last fetched (wall clock); 0 when it never has been
    failed: float = 0.0  # when its last fetch failed, if after ``fetched``
    failures: int = 0  # fetches failed in a row
    games: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SlateBytes:
    """A slate as served: the JSON, its gzip, its ETag (quoted) and the parsed document (for filters)."""

    raw: bytes
    gzip: bytes
    etag: str
    generated_at: float
    data: Dict[str, Any]


def pack(data: Mapping[str, Any], generated_at: float) -> SlateBytes:
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8", "replace")
    return SlateBytes(raw, gzip.compress(raw, compresslevel=6, mtime=0),
                      '"' + hashlib.sha256(raw).hexdigest()[:24] + '"', generated_at, dict(data))


class SlateBuilder:
    """Builds the all-sports slate (see the module docstring). ``odds`` is the OddsService (its sports()
    listing, quota, reserve, odds_ttl and league_games for request_odds); ``espn`` the EspnClient;
    ``store`` the shared Store (odds snapshots, the NBA model's logged predictions, the saved slate);
    ``service`` the DashboardService, which loads an AI league's board with its model (optional)."""

    def __init__(self, odds: Any, espn: Any, store: Any = None, *, service: Any = None,
                 tz: Optional[datetime.tzinfo] = None, live_seconds: float = 20, today_seconds: float = 60,
                 idle_seconds: float = 600, future_seconds: float = 3600, odds_per_day: int = ODDS_PER_DAY,
                 ai_leagues: Iterable[str] = AI_LEAGUES, workers: int = WORKERS,
                 now: Callable[[], float] = time.time) -> None:
        self.odds = odds
        self.espn = espn
        self.store = store
        self.service = service
        self.tz = tz or ZoneInfo("America/New_York")
        self.live_seconds = max(1.0, float(live_seconds))
        self.today_seconds = max(self.live_seconds, float(today_seconds))
        self.idle_seconds = max(self.today_seconds, float(idle_seconds))
        self.future_seconds = max(self.today_seconds, float(future_seconds))
        self.odds_per_day = max(0, int(odds_per_day))
        self.ai_leagues = frozenset(ai_leagues)
        self._now = now
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="slate-espn")
        self._odds_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="slate-odds")
        self._boards: Dict[Tuple[str, datetime.date], _Board] = {}
        self._fallback: Dict[Tuple[str, datetime.date], List[Dict[str, Any]]] = {}
        self._odds: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}  # league -> (fetched_at, events)
        self._season: Optional[List[Dict[str, Any]]] = None
        self._season_next = 0.0  # when to read the in-season listing again (0: never read yet)
        self._season_busy = False  # a read is on its way, in the background
        self._paused_until = 0.0  # while most boards fail: only live boards are fetched before this time
        self._trips = 0  # rounds in a row that paused the boards
        self._titles: Dict[str, str] = {}
        self._league_info: Dict[str, Dict[str, Any]] = {}
        self._current: Optional[SlateBytes] = None
        self._core: Optional[bytes] = None  # the last slate's leagues and games, to tell a change
        self._filtered: "OrderedDict[Tuple[str, Tuple[str, ...], str], SlateBytes]" = OrderedDict()
        self._filter_lock = threading.Lock()
        self._ready = threading.Event()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._refreshing = threading.Lock()
        self._lock = threading.Lock()  # request_odds' bookkeeping
        self._requested: Dict[str, float] = {}
        self._loads: List[float] = []
        self._loading: Set[str] = set()
        self._logged: Dict[str, float] = {}
        self._thread: Optional[threading.Thread] = None

    # --- serving -----------------------------------------------------------------------------------

    def current(self, wait: float = 0.0) -> Optional[SlateBytes]:
        """The latest slate, waiting up to ``wait`` seconds for the first one; None if there is none yet."""
        if self._current is None and wait > 0:
            self._ready.wait(wait)
        return self._current

    def filtered(self, leagues: Optional[Sequence[str]] = None, day: Optional[str] = None) -> Optional[SlateBytes]:
        """The slate with only these leagues and/or one day (YYYY-MM-DD in the builder's time zone), its
        counts recounted. The app filters in the browser; this is for other callers."""
        current = self._current
        if current is None:
            return None
        wanted = tuple(sorted(set(leagues or ())))
        key = (current.etag, wanted, day or "")
        with self._filter_lock:
            cached = self._filtered.get(key)
        if cached is not None:
            return cached
        wanted_day = datetime.date.fromisoformat(day) if day else None
        games = [game for game in current.data.get("games") or []
                 if (not wanted or game.get("league") in wanted)
                 and (wanted_day is None or self._local_day(_parse_time(game.get("start"))) == wanted_day)]
        counts: Dict[str, List[int]] = {}
        for game in games:
            count = counts.setdefault(game["league"], [0, 0])
            count[0] += 1
            count[1] += game.get("status") == "live"
        leagues_view = [{**entry, "count": counts.get(entry["key"], [0, 0])[0], "live": counts.get(entry["key"], [0, 0])[1]}
                        for entry in current.data.get("leagues") or [] if not wanted or entry.get("key") in wanted]
        data = {**current.data, "leagues": leagues_view, "games": games}
        result = pack(data, current.generated_at)
        with self._filter_lock:
            self._filtered[key] = result
            while len(self._filtered) > 32:
                self._filtered.popitem(last=False)
        return result

    def _local_day(self, ts: Optional[float]) -> Optional[datetime.date]:
        return datetime.datetime.fromtimestamp(ts, self.tz).date() if ts is not None else None

    # --- the saved slate ---------------------------------------------------------------------------

    def load(self) -> bool:
        """Serve the slate saved by the last run (if any) until the first refresh finishes."""
        if self.store is None or not hasattr(self.store, "slate"):
            return False
        try:
            saved = self.store.slate(SLATE_NAME)
        except Exception:
            log.exception("Couldn't read the saved slate")
            return False
        if not saved:
            return False
        try:
            generated_at, _, body = saved
            data = json.loads(body.decode("utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("leagues") or [], list) \
                    or not isinstance(data.get("games") or [], list):
                raise ValueError("not a slate")
            self._seed(data)
            current = pack(data, float(generated_at))
            core = self._core_bytes(data.get("leagues") or [], data.get("games") or [], data.get("quota"))
        except Exception as exc:
            # Another version's slate (after a rollback, say) or a damaged row: the first refresh builds a new one
            log.warning("The saved slate can't be used (%s); starting without it", type(exc).__name__)
            self._fallback.clear()
            self._titles.clear()
            return False
        self._core = core
        self._current = current
        self._ready.set()
        return True

    def _seed(self, data: Mapping[str, Any]) -> None:
        """Keep the saved slate's ESPN games, by league and ESPN day, as the last good copy of boards not
        fetched yet; and its league titles."""
        for entry in data.get("leagues") or []:
            if isinstance(entry, dict) and entry.get("key") and entry.get("title"):
                self._titles[entry["key"]] = entry["title"]
        for game in data.get("games") or []:
            if not isinstance(game, dict) or not all(isinstance(game.get(side), dict) for side in ("home", "away")):
                raise ValueError("not a game")  # not a slate this version wrote: load() starts without it
            if not game.get("espn_id"):
                continue
            ts = _parse_time(game.get("start"))
            league = str(game.get("league") or "")
            source = league if espn_path(league) else _tour(league)
            if ts is None or source is None:
                continue
            day = datetime.datetime.fromtimestamp(ts, ESPN_DAY_TZ).date()
            names = tuple({"displayName": _text(game[side].get("name")),
                           "shortDisplayName": _text(game[side].get("short"))} for side in ("home", "away"))
            self._fallback.setdefault((source, day), []).append({
                "espn_id": game["espn_id"], "start": game.get("start"), "status": game.get("status") or "pre",
                "clock": game.get("clock"), "home": dict(game["home"]), "away": dict(game["away"]),
                "venue": game.get("venue"), "_ts": ts, "_neutral": "neutral" in (game.get("badges") or []),
                "_names": names, "_ids": (None, None),
            })

    # --- the background thread ---------------------------------------------------------------------

    def start(self) -> threading.Thread:
        """Refresh on the schedule in a daemon thread (once; later calls return the same thread)."""
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self.run, name="slate", daemon=True)
            self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        """Refresh now (an odds board arrived)."""
        self._wake.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                delay = self.refresh()
            except Exception:
                log.exception("Building the slate failed")
                delay = self.today_seconds
            self._wake.wait(delay)
            self._wake.clear()

    # --- building ----------------------------------------------------------------------------------

    def refresh(self) -> float:
        """Fetch the ESPN boards that are due, rebuild the slate and publish it if anything changed.
        Returns the seconds until the next board is due."""
        with self._refreshing:
            now = self._now()
            season = self._read_season(now)
            today = datetime.datetime.fromtimestamp(now, ESPN_DAY_TZ).date()
            due = self._due(season, today, now)
            if due:
                failed = 0
                for league, day, games, error in self._pool.map(self._fetch, due):
                    board = self._boards.setdefault((league, day), _Board())
                    # Each board of a round is stamped with the round's start, so they come due again together
                    # (one wake-up and one rebuild per round, not one per straggler)
                    if error is None:
                        board.games = games
                        board.fetched = now
                        board.failed = 0.0
                        board.failures = 0
                        self._fallback.pop((league, day), None)
                    else:
                        failed += 1
                        board.failed = now
                        board.failures += 1
                        self._log_failure(league, error)
                self._check_round(len(due), failed, now)
            for key in [key for key in self._boards if key[1] < today - datetime.timedelta(days=1)]:
                del self._boards[key]
            for key in [key for key in self._fallback if key[1] < today - datetime.timedelta(days=1)]:
                del self._fallback[key]
            leagues, games, as_of = self._assemble(season, today, now)
            self._publish(leagues, games, as_of, now, self._quota())
            return self._next_delay(season, today, self._now())

    def _read_season(self, now: float) -> Optional[List[Dict[str, Any]]]:
        """The Odds API's in-season leagues (its free listing), else the last good. Read at once the first time,
        so the first slate knows which leagues are in season; after that in the background, at most every
        SEASON_EVERY seconds (SEASON_RETRY after a failure), so a slow or failing listing never holds up the scores."""
        if not self._season_next:
            self._load_season()
        elif now >= self._season_next and not self._season_busy:
            self._season_busy = True
            try:
                self._odds_pool.submit(self._load_season)
            except RuntimeError:  # shutting down
                self._season_busy = False
        return self._season

    def _load_season(self) -> None:
        ok = False
        try:
            listing = self.odds.sports()
            if isinstance(listing, list):
                self._season = [sport for sport in listing if isinstance(sport, dict) and sport.get("key")]
                ok = True
        except Exception as exc:
            # Only the type: a connection error's message can quote the request URL, which holds the API key
            self._log_failure("sports listing", exc, detail=False)
        finally:
            self._season_next = self._now() + (SEASON_EVERY if ok else SEASON_RETRY)
            self._season_busy = False

    @staticmethod
    def _active(season: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Dict[str, Any]]]:
        if season is None:
            return None
        return {sport["key"]: sport for sport in season if sport.get("active") and not sport.get("has_outrights")}

    def _espn_keys(self, season: Optional[List[Dict[str, Any]]], today: datetime.date) -> Dict[str, bool]:
        """ESPN leagues to fetch -> whether they are in season (all days fetched) or only checked today."""
        active = self._active(season)
        keys = {}
        for league in list(ESPN_LEAGUES) + list(SCOREBOARD_ONLY):
            in_season = active is None or league in active or league in TENNIS_PREFIXES.values()
            board = self._boards.get((league, today))
            keys[league] = in_season or bool(board and board.games)
        return keys

    def _days(self, league: str, in_season: bool, today: datetime.date) -> List[datetime.date]:
        sport = _sport(league)
        if sport == "tennis" or not in_season:
            days = [today]  # a tennis board lists whole tournaments; a league out of season is only checked
        else:
            days = [today + datetime.timedelta(days=n) for n in range(DAYS_AHEAD + 1)]
        yesterday = today - datetime.timedelta(days=1)
        previous = self._boards.get((league, yesterday))
        if (in_season and sport in B2B_SPORTS) or (previous and any(g["status"] == "live" for g in previous.games)):
            days.insert(0, yesterday)
        return days

    def _pace(self, league: str, day: datetime.date, board: Optional[_Board], today: datetime.date, now: float
              ) -> Tuple[float, bool]:
        """(seconds between the board's fetches, whether it has a game on or about to start)."""
        sport = _sport(league)
        live, busy = self.live_seconds, self.today_seconds
        if sport == "tennis":  # its boards are large (whole tournaments): a slower pace
            live, busy = max(live, 60.0), max(busy, 300.0)
        games = board.games if board else []
        if any(g["status"] == "live" or (g["status"] == "pre" and -STUCK < g["_ts"] - now < LIVE_SOON) for g in games):
            return live, True
        if day == today:
            return (busy if any(g["status"] in ("pre", "live") for g in games) else self.idle_seconds), False
        return self.future_seconds, False

    def _due_at(self, key: Tuple[str, datetime.date], board: Optional[_Board], ttl: float, live: bool) -> float:
        """When the board is next due: at once if it was never tried, its pace after its last fetch, and after a
        failure ever further apart (doubling from its own pace, or from a minute while there is nothing to show
        for it yet), so a failing ESPN is never asked more often than a working one."""
        if board is None or not (board.fetched or board.failed):
            return 0.0
        if board.fetched >= board.failed:
            return board.fetched + ttl
        step = ttl if board.fetched or key in self._fallback else min(ttl, max(self.today_seconds, 60.0))
        cap = max(ttl, LIVE_BACKOFF_CAP if live else BACKOFF_CAP)
        return board.failed + min(cap, step * 2 ** min(max(board.failures - 1, 0), 16))

    def _schedule(self, season: Optional[List[Dict[str, Any]]], today: datetime.date, now: float
                  ) -> List[Tuple[str, datetime.date, float, float, bool]]:
        """Every board the slate keeps: (league, day, its pace, when it is next due, whether it is held back).
        While ESPN is failing most boards (see _check_round) only those with a game on are fetched."""
        paused = now < self._paused_until
        items = []
        for league, in_season in self._espn_keys(season, today).items():
            for day in self._days(league, in_season, today):
                board = self._boards.get((league, day))
                ttl, live = self._pace(league, day, board, today, now)
                at = self._due_at((league, day), board, ttl, live)
                held = paused and not live
                items.append((league, day, ttl, max(at, self._paused_until) if held else at, held))
        return items

    def _due(self, season: Optional[List[Dict[str, Any]]], today: datetime.date, now: float
             ) -> List[Tuple[str, datetime.date, float]]:
        # A round also takes the boards due within half the live pace: boards that drifted apart (a long first
        # round, a board fetched on a wake-up) come due together again, so there is one wake-up and one rebuild
        # per round. Boards on the same footing (stamped by one round, paces a multiple of the live pace) are
        # never fetched early.
        horizon = now + self.live_seconds / 2
        due = [(league, day, ttl) for league, day, ttl, at, held in self._schedule(season, today, now)
               if not held and at < horizon]
        # Today's boards first, so a slow future day never holds them up
        due.sort(key=lambda item: (item[1] != today, item[1], item[0]))
        return due

    def _next_delay(self, season: Optional[List[Dict[str, Any]]], today: datetime.date, now: float) -> float:
        soonest = min((item[3] for item in self._schedule(season, today, now)), default=now + self.idle_seconds)
        return max(1.0, min(self.idle_seconds, soonest - now))

    def _check_round(self, tried: int, failed: int, now: float) -> None:
        """When most of a round's boards failed (ESPN down, or pushing back with 429s), fetch only boards with
        a game on for the next 5 minutes, 10 the time after, then 15, until a round mostly succeeds again."""
        if failed >= BREAKER_MIN and 2 * failed > tried:
            self._trips += 1
            pause = BREAKER_PAUSE[min(self._trips, len(BREAKER_PAUSE)) - 1]
            self._paused_until = now + pause
            if self._trips == 1:
                log.warning("Slate: %d of %d ESPN scoreboards failed; only boards with a game on are fetched for "
                            "the next %d minutes (longer while it lasts)", failed, tried, pause // 60)
        elif self._trips and now >= self._paused_until and 2 * failed <= tried:
            log.info("Slate: ESPN's scoreboards answer again")
            self._trips = 0

    def _fetch(self, item: Tuple[str, datetime.date, float]
               ) -> Tuple[str, datetime.date, List[Dict[str, Any]], Optional[BaseException]]:
        """One board, fetched and trimmed in a worker (the raw JSON is dropped here)."""
        league, day, _ = item
        try:
            return league, day, parse_scoreboard(league, self.espn.fetch_scoreboard(league, day)), None
        except Exception as exc:
            return league, day, [], exc

    def _log_failure(self, what: str, exc: BaseException, detail: bool = True) -> None:
        """A warning at most once an hour per league. ESPN's errors carry only its own public URLs; with
        ``detail`` False only the error's type is logged (an Odds API error could quote its URL, and the key)."""
        now = self._now()
        if now - self._logged.get(what, 0.0) >= 3600:
            self._logged[what] = now
            log.warning("Slate: couldn't load %s (%s%s); keeping its last good copy", what, type(exc).__name__,
                        f": {str(exc)[:160]}" if detail else "")

    # --- odds ---------------------------------------------------------------------------------------

    def _odds_ttl(self) -> float:
        ttl = getattr(self.odds, "odds_ttl", None)
        return float(ttl) if isinstance(ttl, (int, float)) and ttl > 0 else ODDS_TTL

    def _odds_times(self) -> Dict[str, float]:
        if self.store is None or not hasattr(self.store, "odds_fetched_at"):
            return {}
        try:
            return dict(self.store.odds_fetched_at())
        except Exception as exc:
            self._log_failure("saved odds", exc)
            return {}

    def _odds_events(self, league: str, fetched_at: Optional[float]) -> List[Dict[str, Any]]:
        """The league's saved odds events with their best prices and chances (re-read only when the saved
        snapshot changes). Never fetches."""
        if fetched_at is None:
            self._odds.pop(league, None)
            return []
        cached = self._odds.get(league)
        if cached and cached[0] == fetched_at:
            return cached[1]
        try:
            snapshot = self.store.odds_snapshot(league)
        except Exception as exc:
            self._log_failure(f"{league} odds", exc)
            return cached[1] if cached else []
        events = []
        for event in (snapshot or (fetched_at, []))[1]:
            if not isinstance(event, dict):
                continue
            start = _parse_time(event.get("commence_time"))
            home, away = _text(event.get("home")), _text(event.get("away"))
            event_id = _text(event.get("id"), 64)
            if start is None or not home or not away or not event_id or not re.fullmatch(r"[A-Za-z0-9_-]+", event_id):
                continue
            prices, chance = odds_view(event)
            events.append({"odds_id": event_id, "_ts": start, "home": home, "away": away, "odds": prices,
                           "chance": chance})
        self._odds[league] = (fetched_at, events)
        return events

    def _ai_chances(self) -> Dict[Tuple[str, str], Tuple[float, str]]:
        """The NBA model's logged pre-game home-win chances: (league, odds id) and ("espn", ESPN id) ->
        (chance, odds id). Only what it already worked out; nothing is computed here."""
        chances: Dict[Tuple[str, str], Tuple[float, str]] = {}
        if self.store is None or not hasattr(self.store, "ai_predictions"):
            return chances
        for league in sorted(self.ai_leagues):
            try:
                rows = self.store.ai_predictions(league=league, status="pending")
            except Exception as exc:
                self._log_failure(f"{league} model predictions", exc)
                continue
            for row in rows:
                home_win = (row.get("payload") or {}).get("home_win")
                if not isinstance(home_win, (int, float)) or not 0 <= home_win <= 1:
                    continue
                chances[(league, str(row.get("event_id")))] = (float(home_win), str(row.get("event_id")))
                if row.get("espn_id"):
                    chances[("espn", str(row["espn_id"]))] = (float(home_win), str(row.get("event_id")))
        return chances

    # --- assembling -----------------------------------------------------------------------------------

    def _pool_games(self, league: str, today: datetime.date) -> List[Dict[str, Any]]:
        """An ESPN league's games from yesterday to six days ahead (yesterday's only to recognise an odds event
        for a game already over; _shown keeps those out unless still live)."""
        last = today + datetime.timedelta(days=DAYS_AHEAD)
        games = []
        for day in [today - datetime.timedelta(days=1)] + [today + datetime.timedelta(days=n) for n in range(DAYS_AHEAD + 1)]:
            board = self._boards.get((league, day))
            source = board.games if board is not None and board.fetched else self._fallback.get((league, day), [])
            for game in source:
                game_day = datetime.datetime.fromtimestamp(game["_ts"], ESPN_DAY_TZ).date()
                if game_day <= last and game_day >= today - datetime.timedelta(days=1):
                    games.append(game)
        unique: Dict[Any, Dict[str, Any]] = {}
        for game in games:  # a tennis tournament's matches come with every day's board
            unique.setdefault(game["espn_id"] or id(game), game)
        return list(unique.values())

    @staticmethod
    def _shown(game: Mapping[str, Any], today: datetime.date) -> bool:
        """Today's and later games, and earlier ones still being played (a late game past midnight)."""
        return game["status"] == "live" or datetime.datetime.fromtimestamp(game["_ts"], ESPN_DAY_TZ).date() >= today

    @staticmethod
    def _match(event: Mapping[str, Any], pool: List[Dict[str, Any]], used: Set[int]
               ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """The ESPN game this odds event is, and whether ESPN lists its teams the other way round."""
        best: Tuple[float, float, Optional[Dict[str, Any]], bool] = (0.0, 0.0, None, False)
        for game in pool:
            if id(game) in used or abs(game["_ts"] - event["_ts"]) > MATCH_WINDOW:
                continue
            home_names, away_names = game["_names"]
            for swapped, (first, second) in ((False, (home_names, away_names)), (True, (away_names, home_names))):
                scores = (team_match_score(event["home"], first), team_match_score(event["away"], second))
                if min(scores) >= MATCH or max(scores) >= STRONG_MATCH:
                    rank = (sum(scores), -abs(game["_ts"] - event["_ts"]))
                    if rank > best[:2]:
                        best = (rank[0], rank[1], game, swapped)
        return best[2], best[3]

    def _game(self, league: str, espn: Optional[Mapping[str, Any]], event: Optional[Mapping[str, Any]], swapped: bool,
              odds_at: Optional[float], ai: Mapping[Tuple[str, str], Tuple[float, str]], now: float
              ) -> Optional[Dict[str, Any]]:
        sport = _sport(league)
        if espn is not None:
            home, away = dict(espn["home"]), dict(espn["away"])
            ids = espn["_ids"]
            if swapped:
                home, away = away, home
                ids = (ids[1], ids[0])
            status, start_ts = espn["status"], espn["_ts"]
        else:
            home, away = {"name": event["home"]}, {"name": event["away"]}
            ids = (None, None)
            start_ts = event["_ts"]
            if start_ts > now:
                status = "pre"
            elif now - start_ts < _odds_only_live(league):
                status = "live"
            else:
                return None  # long started and no scoreboard: its result is unknown here
        badges = []
        if status == "live":
            badges.append("live")
        if espn is not None and espn.get("_neutral"):
            badges.append("neutral")
        if sport in B2B_SPORTS and status in ("pre", "live"):
            day = datetime.datetime.fromtimestamp(start_ts, ESPN_DAY_TZ).date()
            played = self._played(league, day - datetime.timedelta(days=1))
            for side, team_id in zip(("home", "away"), ids):
                if team_id is not None and team_id in played:
                    badges.append(f"b2b-{side}")
        odds_id = event["odds_id"] if event else None
        espn_id = espn.get("espn_id") if espn else None
        model = ai.get((league, odds_id)) if odds_id else None
        if model is None and espn_id:
            model = ai.get(("espn", espn_id))
            if model is not None and odds_id and model[1] != odds_id:
                model = None
        return _lean({
            "id": f"{league}:{odds_id}" if odds_id else f"{league}:espn:{espn_id}",
            "league": league,
            "espn_id": espn_id,
            "odds_id": odds_id,
            "start": _iso(start_ts),
            "status": status,
            "clock": espn.get("clock") if espn else None,
            "home": home,
            "away": away,
            "venue": espn.get("venue") if espn else None,
            "odds": event.get("odds") if event else None,
            "chance": event.get("chance") if event else None,
            "ai": {"home": round(model[0], 3), "away": round(1 - model[0], 3)} if model else None,
            "badges": badges or None,
            "odds_as_of": _iso(odds_at) if event else None,
        })

    def _played(self, league: str, day: datetime.date) -> Set[Any]:
        """ESPN ids of the league's teams that played (or play) on ``day``."""
        board = self._boards.get((league, day))
        return {team_id for game in (board.games if board else []) for team_id in game["_ids"] if team_id is not None}

    def _assemble(self, season: Optional[List[Dict[str, Any]]], today: datetime.date, now: float
                  ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        active = self._active(season) or {}
        espn_keys = self._espn_keys(season, today)
        pools = {league: self._pool_games(league, today) for league in espn_keys}
        used: Set[int] = set()
        odds_times = self._odds_times()
        ai = self._ai_chances()
        window_end = datetime.datetime.combine(today + datetime.timedelta(days=DAYS_AHEAD + 1), datetime.time(),
                                               ESPN_DAY_TZ).timestamp()
        games: List[Dict[str, Any]] = []
        for league in sorted(set(active) | set(pools)):
            fetched_at = odds_times.get(league)
            events = self._odds_events(league, fetched_at)
            pool = pools.get(league if league in pools else _tour(league) or "", [])
            for event in sorted(events, key=lambda e: (e["_ts"], e["odds_id"])):
                if event["_ts"] >= window_end:
                    continue
                match, swapped = self._match(event, pool, used) if pool else (None, False)
                if match is not None:
                    used.add(id(match))
                    if not self._shown(match, today):
                        continue  # last night's game, over: not on today's slate
                game = self._game(league, match, event, swapped, fetched_at, ai, now)
                if game is not None:
                    games.append(game)
        for league, pool in pools.items():
            for espn_game in pool:
                if id(espn_game) not in used and self._shown(espn_game, today):
                    game = self._game(league, espn_game, None, False, None, ai, now)
                    if game is not None:
                        games.append(game)
        games.sort(key=lambda game: (game["start"], game["league"], game["id"]))
        leagues = self._league_entries(season, active, today, games, odds_times, now)
        fetched = [board.fetched for board in self._boards.values() if board.fetched]
        return leagues, games, max(fetched) if fetched else now

    def _league_entries(self, season: Optional[List[Dict[str, Any]]], active: Mapping[str, Mapping[str, Any]],
                        today: datetime.date, games: List[Dict[str, Any]], odds_times: Mapping[str, float],
                        now: float) -> List[Dict[str, Any]]:
        counts: Dict[str, List[int]] = {}
        upcoming: Dict[str, int] = {}
        for game in games:
            count = counts.setdefault(game["league"], [0, 0])
            count[0] += 1
            count[1] += game["status"] == "live"
            if game["status"] in ("pre", "live"):
                upcoming[game["league"]] = upcoming.get(game["league"], 0) + 1
        listed = {sport["key"]: sport for sport in season or []}
        keys = set(active) | set(counts)
        odds_ttl = self._odds_ttl()
        entries = []
        info: Dict[str, Dict[str, Any]] = {}
        for key in keys:
            sport = listed.get(key) or {}
            path = espn_path(key)
            group = _text(sport.get("group"), 40) or GROUP_OF_SPORT.get((path or ("",))[0]) or "Other"
            title = _text(sport.get("title"), 60) or self._titles.get(key) or TITLES.get(key) or key
            fetched = odds_times.get(key)
            odds = "none" if fetched is None else ("fresh" if now - fetched < odds_ttl else "stale")
            ai = key in self.ai_leagues
            entries.append(_lean({
                "key": key, "title": title, "group": group, "label": LABELS.get(group, group),
                "count": counts.get(key, [0, 0])[0], "live": counts.get(key, [0, 0])[1], "ai": ai, "odds": odds,
                "odds_as_of": _iso(fetched), "espn": path is not None or any(key.startswith(p) for p in TENNIS_PREFIXES),
            }))
            board = self._boards.get((key, today))
            # ESPN's word that a league has no games counts only once its board for today has loaded
            info[key] = {"in_season": key in active, "espn": path is not None and bool(board and board.fetched),
                         "upcoming": upcoming.get(key, 0), "odds_fetched": fetched}
        self._league_info = info

        def rank(entry: Mapping[str, Any]) -> Tuple[Any, ...]:
            group = entry["group"]
            key = entry["key"]
            return (GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER), group,
                    POPULAR_LEAGUES.index(key) if key in POPULAR_LEAGUES else len(POPULAR_LEAGUES), entry["title"], key)

        return sorted(entries, key=rank)

    @staticmethod
    def _core_bytes(leagues: List[Dict[str, Any]], games: List[Dict[str, Any]],
                    quota: Optional[Mapping[str, int]] = None) -> bytes:
        return json.dumps({"leagues": leagues, "games": games, "quota": quota}, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8", "replace")

    def _quota(self) -> Optional[Dict[str, int]]:
        """The month's Odds API requests left and used, as last reported (the odds service reads the shared
        store; no request is made), for the header's "N odds left"; None when unknown."""
        try:
            quota = getattr(self.odds, "quota", None)
        except Exception as exc:
            self._log_failure("odds quota", exc)
            return None
        if not isinstance(quota, Mapping):
            return None
        view = {key: int(quota[key]) for key in ("remaining", "used")
                if isinstance(quota.get(key), int) and not isinstance(quota.get(key), bool)}
        return view if "remaining" in view else None

    def _publish(self, leagues: List[Dict[str, Any]], games: List[Dict[str, Any]], as_of: float, now: float,
                 quota: Optional[Dict[str, int]] = None) -> None:
        """Serialise the slate once, if it changed; keep it in memory and in the store."""
        core = self._core_bytes(leagues, games, quota)
        if self._current is not None and core == self._core:
            return
        data = _lean({"generated_at": _iso(now), "as_of": _iso(min(as_of, now)), "quota": quota,
                      "leagues": leagues, "games": games})
        current = pack(data, now)
        self._core = core
        self._current = current
        with self._filter_lock:
            self._filtered.clear()
        self._ready.set()
        if self.store is not None and hasattr(self.store, "save_slate"):
            try:
                self.store.save_slate(SLATE_NAME, current.raw, current.etag, now)
            except Exception:
                log.exception("Couldn't save the slate")

    # --- fresh odds on request ------------------------------------------------------------------------

    def request_odds(self, league: str) -> Dict[str, Any]:
        """Ask for a fresh odds board for one league, loaded in the background: {"queued", "reason", "code"}
        (reason in words; code one of queued, loading, fresh, throttled, no_games, reserve, daily_cap,
        not_in_season, no_listing, error).

        Runs only when the league is in season at The Odds API, its saved odds are older than the odds
        service's 2 hours (or missing), it hasn't been asked for in that time (by anyone, across restarts),
        ESPN doesn't show it without games, the quota stays at or above the odds service's reserve after this
        board and every board already on its way, and fewer than ``odds_per_day`` boards were loaded this way
        in the last 24 hours. The slate updates when it arrives.
        """
        now = self._now()
        ttl = self._odds_ttl()
        active = self._active(self._season)
        info = self._league_info.get(league)
        if active is None:
            return _answer(False, "no_listing", "The list of leagues in season isn't available right now.")
        if league not in active:
            return _answer(False, "not_in_season", "No odds market for this league right now.")
        fetched = self._odds_times().get(league)
        if fetched is not None and now - fetched < ttl:
            return _answer(False, "fresh", "The odds are up to date.")
        with self._lock:
            if league in self._loading:
                return _answer(True, "loading", "The latest odds are on their way.")
            last = self._requested.get(league)
            if last is not None and now - last < ttl:
                return _answer(False, "throttled", "Asked for less than 2 hours ago.")
            if info and info.get("espn") and not info.get("upcoming"):
                return _answer(False, "no_games", "No games coming up in this league.")
            # Boards already queued haven't been paid for yet: count them too (the app asks for several at once)
            if not self._affordable(boards=len(self._loading) + 1):
                return _answer(False, "reserve", "The month's odds allowance is down to its reserve.")
            self._loads = [t for t in self._loads if now - t < 24 * 3600]
            if len(self._loads) >= self.odds_per_day:
                return _answer(False, "daily_cap", "Today's odds loads are used up; saved odds stay on show.")
            if self.store is not None and hasattr(self.store, "try_lease"):
                try:
                    if not self.store.try_lease(f"slate-odds:{league}", ttl):
                        return _answer(False, "throttled", "Asked for less than 2 hours ago.")
                except Exception:
                    log.exception("Couldn't take the odds lease for %s", league)
                    return _answer(False, "error", "Couldn't ask for odds right now.")
            self._requested[league] = now
            self._loads.append(now)
            self._loading.add(league)
        self._odds_pool.submit(self._load_odds, league)
        return _answer(True, "queued", "Loading the latest odds.")

    def _fetch_cost(self) -> int:
        """Credits one league's board costs: a credit per market per odds region ("us" or "us,uk", say)."""
        regions = getattr(self.odds, "regions", None)
        count = len([r for r in regions.split(",") if r.strip()]) if isinstance(regions, str) else 1
        return FETCH_COST * max(1, count)

    def _affordable(self, boards: int = 1) -> bool:
        """Whether ``boards`` more boards keep the month's quota (as last reported) at or above the reserve."""
        try:
            quota = getattr(self.odds, "quota", None) or {}
        except Exception:
            log.exception("Couldn't read the odds quota")
            return False
        remaining = quota.get("remaining") if isinstance(quota, Mapping) else None
        reserve = getattr(self.odds, "reserve", 0) or 0
        return not isinstance(remaining, int) or remaining - self._fetch_cost() * boards >= reserve

    def _load_odds(self, league: str) -> None:
        try:
            # Loads run one at a time, so the quota read here already counts every board loaded before this one
            if not self._affordable():
                log.info("Slate: not loading %s odds: the month's allowance is down to its reserve", league)
            elif self.service is not None and league in self.ai_leagues:
                self.service.games(league, refresh=True)  # the model's chances are logged with the board
            else:
                self.odds.league_games(league, refresh=True)
        except Exception as exc:
            # Only the type: a connection error's message can quote the request URL, which holds the API key
            log.warning("Slate: loading %s odds failed (%s)", league, type(exc).__name__)
        finally:
            with self._lock:
                self._loading.discard(league)
            self.wake()
