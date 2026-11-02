"""Player boards: per league and day, everything the player scan (engine/scan.py) reads.

A board holds one day's games in a league (ESPN's scoreboard, games not yet started) and, for each game:

- **projections:** NBA games use the NBA model's own projection for the game (``DashboardService.detail``: the
  same rows the player card reads), but only while the league's odds board can be read without a fetch (loaded
  or saved within the odds service's time to live less ODDS_MARGIN, or the quota under its reserve), checked again
  inside each projection's pool task as it runs: this feature never spends The Odds API's credits. Without it, and
  in every other league, the scan's form model works from the game logs;
- **game logs:** each player's (``PlayerData.gamelog``: this season, and last season's tail early on), kept by
  the shared cache's player-log time to live; while this season is short of the trend's TREND_BASE games, also last
  season's whole log (``last_season_rows``, the trend's baseline: engine.trend), from the copy the tail came from;
- **injury status:** ESPN's injury report for the game (its summary), else the roster's;
- **props:** DraftKings' player props for the event through ESPN (``PlayerData.props``), never The Odds API.

Which players: an NBA game's projected rotation; elsewhere the players DraftKings has props for (when it has them
for at least PROPS_POOL players of the game), else the roster at the positions that matter (the NFL's QB, RB,
WR, TE; soccer's outfield players), ruled-out players left out.

Fetching: everything goes through the existing ESPN client, PlayerData and the shared cache (no new hosts), at
most FETCHES requests at once for the board: a copy past its time to live that a fetch reads is loaded on the
board's own pool (data/cache.py's inline_refreshes), never on the cache's background threads. An NBA game's
players' season averages are loaded at that width before the model runs, so its own pool finds them cached. A
board builds in the background, game by game in start order (LANES games side by side), starting on the first scan
of the day for the league or from the slate worker's game-day prewarm (leagues with games in the next 12 hours: the
NBA first, then the NHL, NFL and MLB).

Each game's props and injuries are read again every REFRESH seconds until tip-off, and the player list is built
again from them: the injury report read then is the authority, so a player cleared since joins (on an NBA game the
model left him out of, on the form model until the model has him) and a status off the report goes. A part that
failed while loading (a roster, the injury report, the props, the NBA projection, a game log) is tried again then;
until it comes in the game counts in the snapshot's ``games_failed``. Props older than their time to live are flagged
stale (no edge); a scan that finds a game's props stale reads them again on their own (PROPS_RETRY apart at most:
never more often than their time to live lets a read renew them) and waits for that, SCAN_WAIT seconds at most. A read
that renews nothing (ESPN failing: the last copy served) holds every read of that game's props off for the props' time
to live, the 30-minute refresh's too, so an outage costs one request per game per time to live.

Games under way are left out of the scan, but their players' names stay matchable (the snapshot's
``started_players``): those of games that started since the build, and of games already under way when it ran
(their rosters, read after the games still to come).

A scan never waits long: it answers with what is ready, plus {"loading": {"games_ready", "games_total", "failed"?}}
while the board is still building (SCAN_WAIT seconds the first time) or a game is missing a part that failed.
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..engine import scan as engine
from .cache import inline_refreshes, ttl_for
from .espn_client import ESPN_DAY_TZ, competition_state
from .espn_players import UNREACHABLE, PlayerData

log = logging.getLogger(__name__)

FETCHES = 4  # ESPN requests at once, every board together
LANES = 2  # games built side by side (their requests share the FETCHES)
REFRESH = 30 * 60  # seconds between reads of a game's props and injuries, until tip-off
# A scan reads a game's props again on their own once their copy is past the props' time to live (stale: no edge),
# at most this often for one game (a read that brought no newer copy, ESPN failing, isn't tried on every scan)
PROPS_RETRY = 60.0
SCAN_WAIT = 2.0  # seconds a scan waits for a board that has just started building, or for stale props read again
PREWARM_HOURS = 12  # the slate's prewarm builds boards for leagues with a game starting within this many hours
PREWARM_LEAGUES = (engine.NBA, engine.NHL, engine.NFL, engine.MLB, engine.WNBA)  # ... in this order
MATCH_WINDOW = 3 * 3600  # an odds game and an ESPN game further apart than this are different games
ODDS_MARGIN = 900.0  # odds this close to the odds service's time to live count as stale (no fetch can start: the
# check runs again inside each NBA projection's pool task, and one projection never takes 15 minutes)
PROPS_POOL = 6  # with props for this many of a game's players, they are the game's players
TEAM_CAP = 22  # without props, at most this many of a team's roster
NBA_MINUTES = 8.0  # the NBA model's rows below this many minutes are left out (as service._players does)
GAMELOG_MAX = 200
KEEP = 4  # boards kept (today's and tomorrow's for a couple of leagues)
RETRY_FAILED = 60.0  # seconds before a board whose scoreboard couldn't be read is built again
POSITIONS = {
    engine.NFL: frozenset({"QB", "RB", "WR", "TE", "FB"}),
}
SOCCER_POSITIONS = frozenset({"D", "M", "F", "DF", "MF", "FW"})
SKIP_GROUPS = frozenset({"practicesquad", "practice squad"})


class _Game:
    """One game on a board (mutated only under the board's lock)."""

    def __init__(self, league: str, event: Mapping[str, Any]) -> None:
        self.league = league
        self.event_id = str(event["event_id"])
        self.start = event["start"]  # epoch seconds
        self.start_text = event["start_text"]
        self.home = event["home"]
        self.away = event["away"]
        self.label = _label(event)
        self.odds_id: Optional[str] = None
        self.ready = False
        self.players: Dict[str, Dict[str, Any]] = {}
        self.props_meta: Dict[str, Any] = {}
        self.props_at: Optional[float] = None  # when its props were fetched, on the board's clock (their staleness)
        self.props_players: Dict[str, List[Dict[str, Any]]] = {}  # the last props read that worked, by player
        self.outs: Dict[str, List[str]] = {}
        self.refreshed_at = 0.0
        self.refreshing = False
        # Parts that failed and are still missing ("summary", "props", "roster", "projection", "logs"), retried on
        # each refresh; the players whose game log failed; and, for an NBA game, whether the model's projection is
        # still to come (it failed, or the odds board couldn't be read for free)
        self.failed: Set[str] = set()
        self.log_failed: Set[str] = set()
        self.model_pending = False
        # Set while no refresh runs (a scan waits on it for stale props read again); when props were last read alone;
        # and the earliest any read of its props is tried again after one that brought no newer copy (ESPN failing):
        # the props' time to live on, so an outage costs one request per game per time to live, scans or not
        self.idle = threading.Event()
        self.idle.set()
        self.props_tried = 0.0
        self.props_next = 0.0
        # An NBA game's players the board left out for an out status (the report's, else the roster's: the model's
        # own reading): {athlete id: {name, side, position}}. One the injury report read on a refresh no longer rules
        # out joins (RD6 on the model's games)
        self.benched: Dict[str, Dict[str, Any]] = {}


def _label(event: Mapping[str, Any]) -> str:
    """A game's label, "AWAY @ HOME" (abbreviations, else names)."""
    home, away = event.get("home") or {}, event.get("away") or {}
    return f"{away.get('abbr') or away.get('name')} @ {home.get('abbr') or home.get('name')}"


class _Board:
    def __init__(self, league: str, day: datetime.date) -> None:
        self.league = league
        self.day = day
        self.games: Dict[str, _Game] = {}
        self.started = 0  # the day's games already under way or over when the board was built
        # Their players, for name matching only ({athlete_id, name, team, event_id, game}: the rosters)
        self.started_players: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.building = False
        self.loaded = False  # the scoreboard has been read
        self.done = threading.Event()
        self.built_at: Optional[float] = None
        self.error: Optional[str] = None
        self.thread: Optional[threading.Thread] = None


class _OddsNotFree(Exception):
    """The league's odds board can no longer be read without a fetch: the NBA projection is skipped (form model)."""


def _pooled(fn: Callable[..., Any], *args: Any) -> Any:
    """One of the board's fetches, on its pool: any copy past its time to live that it reads is loaded here, inside
    the FETCHES limit, never refreshed on the shared cache's own background threads."""
    with inline_refreshes():
        return fn(*args)


def _unreachable(found: Any) -> bool:
    """Whether PlayerData's answer is a failed read (ESPN down), not an empty one (nothing posted)."""
    return isinstance(found, Mapping) and UNREACHABLE in str(found.get("reason") or "")


def _team(competitor: Mapping[str, Any]) -> Dict[str, Any]:
    team = competitor.get("team") or {}
    return {"id": str(team.get("id") or competitor.get("id") or ""), "abbr": team.get("abbreviation"),
            "name": team.get("displayName") or team.get("name")}


def parse_events(events: Iterable[Any]) -> List[Dict[str, Any]]:
    """A scoreboard's games: {event_id, state ("pre", "live", "final"), start (epoch), start_text, home, away}, each
    team {id, abbr, name}; unreadable events are skipped."""
    games = []
    for event in events or []:
        if not isinstance(event, Mapping) or not event.get("id"):
            continue
        competition = (event.get("competitions") or [None])[0]
        found = competition_state(competition)
        if found is None:
            continue
        state, _, _, start = found
        if start is None:
            start = _epoch(event.get("date"))
        sides = {c.get("homeAway"): c for c in (competition or {}).get("competitors") or [] if isinstance(c, Mapping)}
        if "home" not in sides or "away" not in sides or start is None:
            continue
        games.append({"event_id": str(event["id"]), "state": state, "start": start, "start_text": _iso(start),
                      "home": _team(sides["home"]), "away": _team(sides["away"])})
    games.sort(key=lambda game: (game["start"], game["event_id"]))
    return games


def _epoch(text: Any) -> Optional[float]:
    try:
        moment = datetime.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    return moment.timestamp()


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PlayerBoard:
    """The player scan's boards, one per (league, day), built in the background.

    ``service`` is the DashboardService (its ESPN client, odds service and ``detail``); ``players`` the PlayerData
    (one is made over the service's ESPN client when None); ``cache`` the shared TTLCache (its clock stamps
    ``as_of``); ``now`` the clock (epoch seconds). ``events`` (tests, launchers) replaces the ESPN scoreboard read:
    a function (league, date) -> ESPN events."""

    def __init__(self, service: Any, players: Optional[Any] = None, cache: Any = None,
                 now: Optional[Callable[[], float]] = None, *, fetches: int = FETCHES, lanes: int = LANES,
                 refresh_every: float = REFRESH, wait: float = SCAN_WAIT,
                 events: Optional[Callable[[str, datetime.date], Sequence[Any]]] = None) -> None:
        self.service = service
        self.espn = getattr(service, "espn", None)
        self.players = players if players is not None else PlayerData(self.espn)
        self.cache = cache if cache is not None else getattr(service, "cache", None)
        self._now = now or time.time
        self.lanes = max(1, int(lanes))
        self.refresh_every = float(refresh_every)
        self.wait_seconds = max(0.0, float(wait))
        self._events = events
        self._fetch = ThreadPoolExecutor(max_workers=max(1, int(fetches)), thread_name_prefix="player-board")
        self._boards: Dict[Tuple[str, datetime.date], _Board] = {}
        self._lock = threading.Lock()
        self._closed = False

    # --- days and boards -------------------------------------------------------------------------------

    def day_of(self, day: Any = "today") -> datetime.date:
        """ESPN's day (US Eastern) for "today", "tomorrow", a date or "YYYY-MM-DD"."""
        today = datetime.datetime.fromtimestamp(self._now(), ESPN_DAY_TZ).date()
        if isinstance(day, datetime.date):
            return day
        text = str(day or "today").strip().lower()
        if text == "today":
            return today
        if text == "tomorrow":
            return today + datetime.timedelta(days=1)
        return datetime.date.fromisoformat(text)

    def ensure(self, league: str, day: Any = "today") -> _Board:
        """The board for the league and day, its build started in the background if it hasn't been."""
        date = self.day_of(day)
        key = (league, date)
        with self._lock:
            board = self._boards.get(key)
            if board is None:
                board = _Board(league, date)
                self._boards[key] = board
                self._prune_locked()
            failed = board.built_at is not None and board.error is not None \
                and self._now() - board.built_at >= RETRY_FAILED
            start = not board.building and (board.built_at is None or failed) and not self._closed
            if start:
                board.building = True
                board.error = None
                board.done.clear()
        if start:
            thread = threading.Thread(target=self._build, args=(board,), name=f"player-board-{league}", daemon=True)
            board.thread = thread
            try:
                thread.start()
            except RuntimeError:  # shutting down
                board.building = False
        return board

    def _prune_locked(self) -> None:
        today = datetime.datetime.fromtimestamp(self._now(), ESPN_DAY_TZ).date()
        for key in [key for key in self._boards if key[1] < today]:
            del self._boards[key]
        while len(self._boards) > KEEP * 4:
            oldest = min(self._boards, key=lambda key: (key[1], self._boards[key].built_at or 0))
            del self._boards[oldest]

    def wait(self, league: str, day: Any = "today", timeout: Optional[float] = None) -> bool:
        """Wait for the board's build (for tests and tools); True when it finished."""
        board = self.ensure(league, day)
        return board.done.wait(timeout)

    def close(self) -> None:
        self._closed = True
        self._fetch.shutdown(wait=False, cancel_futures=True)

    # --- reading -----------------------------------------------------------------------------------------

    def snapshot(self, league: str, day: Any = "today", start: bool = True, props: bool = False) -> Dict[str, Any]:
        """What engine.scan reads (its docstring has the shape): the games not started yet and the players loaded so
        far, with ``games_failed`` (games with a part that failed and is still missing: a roster, the injury report,
        the props, the NBA projection, a game log), ``error`` (the last scoreboard error, or None) and
        ``started_players`` ({athlete_id, name, team, event_id, game}: the players of the day's games already under
        way, whose names a scan still knows). A game's props are ``stale`` once their copy is older than the props'
        time to live, and are then shown but never priced for an edge. Starts the board's build (``start``), any
        refresh of props and injuries that is due, and with ``props`` (a scan's) a read of the stale props alone."""
        if start:
            board = self.ensure(league, day)
        else:
            with self._lock:
                board = self._boards.get((league, self.day_of(day)))
            if board is None:
                return self._empty(league, self.day_of(day))
        return self._view_of(board, props)[0]

    def _view_of(self, board: _Board, props: bool = False) -> Tuple[Dict[str, Any], List[_Game]]:
        """(snapshot's view of the board, the games whose props were stale and are being read again: a refresh of
        theirs is running)."""
        now = self._now()
        props_ttl = ttl_for("props").ttl
        stale: List[_Game] = []
        with board.lock:
            games = list(board.games.values())
            started = board.started + sum(1 for game in games if game.start <= now)
            live = [game for game in games if game.start > now]
            view_games = {}
            for game in live:
                meta = dict(game.props_meta)
                if game.props_at is not None and now - game.props_at >= props_ttl:
                    meta["stale"] = True
                    stale.append(game)
                view_games[game.event_id] = {"event_id": game.event_id, "label": game.label,
                                             "start": game.start_text, "outs": dict(game.outs), "props": meta}
            players = [player for game in live for player in game.players.values()]
            begun = list(board.started_players) + [
                {"athlete_id": player.get("athlete_id"), "name": player.get("name"), "team": player.get("team"),
                 "event_id": game.event_id, "game": game.label}
                for game in games if game.start <= now for player in game.players.values()]
            ready = sum(1 for game in live if game.ready)
            failed = sum(1 for game in live if game.ready and game.failed)
            running = [game for game in stale if game.refreshing]  # a refresh already reading them again
        kicked = self._refresh_due(board, live, now, stale if props else ())
        # Every stale game whose read is running or was just started (a fast one may be over already: its idle is set)
        renewing = running + [game for game in kicked if game in stale and game not in running]
        stamps = [game.refreshed_at for game in live if game.refreshed_at]
        view = {"league": board.league, "day": board.day.isoformat(),
                "as_of": _iso(min(stamps)) if stamps else _iso(now), "games": view_games, "players": players,
                "started_players": begun, "games_ready": ready, "games_total": len(live), "games_started": started,
                "games_failed": failed, "building": board.building, "loaded": board.loaded, "error": board.error}
        return view, renewing

    def _empty(self, league: str, day: datetime.date) -> Dict[str, Any]:
        return {"league": league, "day": day.isoformat(), "as_of": _iso(self._now()), "games": {}, "players": [],
                "started_players": [], "games_ready": 0, "games_total": 0, "games_started": 0, "games_failed": 0,
                "building": False, "loaded": False, "error": None}

    def scan(self, league: str, stat: Any, threshold: Any = None, side: str = "over", day: str = "today",
             n: int = 10, wait: Optional[float] = None, *, names: Optional[Sequence[str]] = None,
             sort: Optional[str] = None) -> Dict[str, Any]:
        """The scan for one league: engine.scan's result for make_ask(league, stat, threshold, side, day) (with its
        ``names`` and ``sort``), plus {league, day, title, no_games, error?}. A board that has just started building is
        waited for up to ``wait`` seconds (SCAN_WAIT by default); after that whatever is ready is answered, with
        ``loading``. ``no_games`` is true only for a day whose scoreboard was read and holds no game at all: none to
        come, none started, no error (after the last tip-off, or with ESPN down, the day had games). Raises ValueError
        for an ask that doesn't make sense (an unknown league or stat, a bad number)."""
        ask = engine.make_ask(league, stat, threshold, side, day)
        board, view = self._view(league, day, wait)
        result = engine.scan(view, ask, n, names=names, sort=sort)
        return self._answer(result, view, league)

    def scan_edges(self, league: str, side: Optional[str] = None, day: str = "today", n: int = 10,
                   wait: Optional[float] = None, *, names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """engine.scan_edges over the league's board (every stat with a posted line that isn't stale, by edge; only
        the ``names`` players' with names), plus what scan adds. Raises ValueError for a league the scan doesn't cover
        or a bad side."""
        if league not in engine.STATS:
            raise ValueError(f"The player scan doesn't cover {league!r}")
        board, view = self._view(league, day, wait)
        result = engine.scan_edges(view, league, n, side=side, names=names)
        return self._answer(result, view, league)

    def _view(self, league: str, day: str, wait: Optional[float]) -> Tuple[_Board, Dict[str, Any]]:
        """The board a scan reads: waited for while it has just started building, and while the props of a game
        found stale are read again (``wait`` seconds each at most, SCAN_WAIT by default), so a copy a read can renew
        is priced, and one it couldn't renew (ESPN failing) stays stale: no edge (RD8)."""
        board = self.ensure(league, day)
        limit = self.wait_seconds if wait is None else max(0.0, float(wait))
        if limit and not board.done.is_set():
            board.done.wait(limit)
        view, renewing = self._view_of(board, props=True)
        if renewing and limit:
            deadline = time.monotonic() + limit
            for game in renewing:
                game.idle.wait(max(0.0, deadline - time.monotonic()))
            view, _ = self._view_of(board, props=True)
        return board, view

    @staticmethod
    def _answer(result: Dict[str, Any], view: Mapping[str, Any], league: str) -> Dict[str, Any]:
        error = view.get("error")
        result.update(league=league, day=view["day"], title=engine.TITLES.get(league, league),
                      no_games=bool(view["loaded"]) and not view["building"] and view["games_total"] == 0
                      and not view.get("games_started") and not error)
        if not view["loaded"]:
            result["loading"] = {"games_ready": 0, "games_total": 0}  # the day's games are still being read
        if error and not view["players"]:
            result["error"] = error
        return result

    def games_on(self, league: str, day: Any = "today") -> Optional[int]:
        """How many games the league has that day (to come, under way or over) once its board has read the day's
        scoreboard; None before that, or when the read failed. Starts nothing and fetches nothing (the Lab orders the
        leagues it tries by it)."""
        with self._lock:
            board = self._boards.get((league, self.day_of(day)))
        if board is None:
            return None
        with board.lock:
            if not board.loaded or board.building and not board.games or board.error:
                return None
            return board.started + len(board.games)

    def status(self) -> List[Dict[str, Any]]:
        """Each board's progress (for logs and tools)."""
        with self._lock:
            boards = list(self._boards.values())
        out = []
        for board in boards:
            with board.lock:
                games = list(board.games.values())
                out.append({"league": board.league, "day": board.day.isoformat(), "building": board.building,
                            "games": len(games), "ready": sum(1 for game in games if game.ready),
                            "players": sum(len(game.players) for game in games), "built_at": _iso(board.built_at),
                            "error": board.error})
        return out

    # --- the slate's game-day prewarm ------------------------------------------------------------------

    def prewarm(self, games: Optional[Iterable[Mapping[str, Any]]] = None,
                leagues: Optional[Iterable[str]] = None) -> List[str]:
        """Start (or refresh) the boards for leagues with a game starting within PREWARM_HOURS: the slate's rows
        ({league, start, status}) or ``leagues`` directly. The NBA first, then the NHL, NFL, MLB and WNBA, one cold
        build at a time (the slate calls this every few minutes, so the next starts on a later call); other leagues
        build on their first scan. Built boards get their due props and injuries refresh. Returns the (league, day)
        boards touched as "league:YYYY-MM-DD"."""
        now = self._now()
        wanted: Dict[Tuple[str, datetime.date], None] = {}
        if leagues is not None:
            for league in leagues:
                if league in PREWARM_LEAGUES:
                    wanted[(league, self.day_of("today"))] = None
        for game in games or []:
            league = str(game.get("league") or "")
            if league not in PREWARM_LEAGUES or game.get("status") not in (None, "pre", "scheduled"):
                continue
            start = _epoch(game.get("start"))
            if start is None or not now < start <= now + PREWARM_HOURS * 3600:
                continue
            wanted[(league, datetime.datetime.fromtimestamp(start, ESPN_DAY_TZ).date())] = None
        touched = []
        with self._lock:
            busy = any(board.building for board in self._boards.values())
        for league, day in sorted(wanted, key=lambda key: (PREWARM_LEAGUES.index(key[0]), key[1])):
            with self._lock:
                board = self._boards.get((league, day))
            built = board is not None and board.built_at is not None
            if not built and busy:
                continue  # one cold build at a time: the next prewarm round starts it
            # Starts the build (the first not built yet), or the props and injuries refresh that is due
            self.snapshot(league, day)
            busy = busy or not built
            touched.append(f"{league}:{day.isoformat()}")
        return touched

    # --- building ----------------------------------------------------------------------------------------

    def _submit(self, fn: Callable[..., Any], *args: Any) -> "Future[Any]":
        return self._fetch.submit(_pooled, fn, *args)

    @staticmethod
    def _result(future: "Future[Any]", what: str) -> Any:
        return PlayerBoard._attempt(future, what)[1]

    @staticmethod
    def _attempt(future: "Future[Any]", what: str) -> Tuple[bool, Any]:
        """(whether it worked, its value or None): a failed read is logged and told apart from an empty one."""
        try:
            return True, future.result()
        except _OddsNotFree:
            return True, None  # the odds board would need a fetch: the form model, by design
        except Exception as exc:
            log.info("Player board: %s failed (%s)", what, type(exc).__name__)
            return False, None

    def _build(self, board: _Board) -> None:
        try:
            events = self._read_events(board.league, board.day)
            now = self._now()
            parsed = parse_events(events)
            upcoming = [event for event in parsed if event["state"] == "pre" and event["start"] > now]
            under_way = [event for event in parsed if event not in upcoming]
            with board.lock:
                board.started = len(parsed) - len(upcoming)
                for event in upcoming:
                    board.games[event["event_id"]] = _Game(board.league, event)
                board.loaded = True
            games = list(board.games.values())
            if board.league in engine.MODEL_LEAGUES and games:
                self._result(self._submit(self._match_odds, board.league, games), f"{board.league} odds match")
            if self.lanes > 1 and len(games) > 1:
                with ThreadPoolExecutor(max_workers=self.lanes, thread_name_prefix="player-board-game") as lanes:
                    list(lanes.map(lambda game: self._build_game(board, game), games))
            else:
                for game in games:
                    self._build_game(board, game)
            if under_way:
                self._read_started(board, under_way)  # after the games to come: only their names are wanted
        except Exception as exc:
            log.warning("Player board for %s %s failed: %s", board.league, board.day, type(exc).__name__)
            board.error = "ESPN couldn't be reached for this league's games just now."
        finally:
            with board.lock:
                board.loaded = True
                board.building = False
                board.built_at = self._now()
            board.done.set()

    def _read_started(self, board: _Board, events: Sequence[Mapping[str, Any]]) -> None:
        """The players of the games already under way (or over) when the board was built, for name matching only (a
        scan can't rank them, but says their game has started instead of not finding them): each team's roster at
        the positions the league's stats are about, ruled-out players left out, as the games to come take theirs. A
        roster that fails is left out (logged), not retried: it feeds no number, and never fails the board."""
        found: List[Dict[str, Any]] = []
        try:
            reads = [(event, side, self._submit(self.espn.roster, board.league, event[side]["id"]))
                     for event in events for side in ("home", "away") if (event.get(side) or {}).get("id")]
            for event, side, future in reads:
                team = event[side]
                for player in self._result(future, f"{board.league} roster (a game under way)") or []:
                    athlete_id = str(player.get("id") or "")
                    if not athlete_id or str(player.get("group") or "").lower() in SKIP_GROUPS \
                            or engine.ruled_out(player.get("status")) \
                            or not _plays(board.league, player.get("position")):
                        continue
                    found.append({"athlete_id": athlete_id, "name": player.get("name"),
                                  "team": team.get("abbr") or team.get("name"), "event_id": str(event["event_id"]),
                                  "game": _label(event)})
        except Exception as exc:  # a pool shut down: the names of what's under way are all that's lost
            log.info("Player board: %s rosters of the games under way failed (%s)", board.league, type(exc).__name__)
        with board.lock:
            board.started_players = found

    def _read_events(self, league: str, day: datetime.date) -> Sequence[Any]:
        """The day's ESPN events (raises when ESPN fails: the board is then marked failed and built again later)."""
        if self._events is not None:
            return self._events(league, day) or []
        return self._submit(self.espn.scoreboard, league, day).result() or []

    # odds: only ever read when that can't fetch

    def _odds_ready(self, league: str) -> bool:
        """Whether the league's odds board can be read without spending credits: loaded or saved within the odds
        service's time to live (less a margin), or saved at all while the quota is under its reserve."""
        odds = getattr(self.service, "odds", None)
        if odds is None:
            return False
        ttl = float(getattr(odds, "odds_ttl", 7200) or 7200) - ODDS_MARGIN
        now = time.time()  # the odds service's own clock
        loaded = getattr(odds, "odds_loaded_at", None)
        at = loaded(league) if callable(loaded) else None
        if at is not None and now - at < ttl:
            return True
        saved = self._saved(league)
        if saved is None:
            return False
        if now - saved[0] < ttl:
            return True
        try:
            remaining = (getattr(odds, "quota", None) or {}).get("remaining")
        except Exception:
            remaining = None
        return remaining is not None and remaining < int(getattr(odds, "reserve", 0) or 0)

    def _saved(self, league: str) -> Optional[Tuple[float, List[Dict[str, Any]]]]:
        saved_games = getattr(getattr(self.service, "odds", None), "saved_games", None)
        if not callable(saved_games):
            return None
        try:
            return saved_games(league)
        except Exception:
            return None

    def _odds_games(self, league: str) -> List[Dict[str, Any]]:
        """The league's odds board, read only when that can't fetch (else nothing)."""
        if not self._odds_ready(league):
            return []
        saved = self._saved(league)
        if saved is not None:
            return list(saved[1] or [])
        try:
            return list(self.service.odds.league_games(league))  # fresh in memory: no fetch
        except Exception:
            return []

    def _match_odds(self, league: str, games: Sequence[_Game]) -> None:
        """Each game's odds-board id, for the NBA model (service.detail goes by it)."""
        by_teams = {(game.home["id"], game.away["id"]): game for game in games}
        find = getattr(self.espn, "find_game", None)
        if not callable(find):
            return
        starts = [game.start for game in games]
        for odds_game in self._odds_games(league):
            start = _epoch(odds_game.get("commence_time"))
            if start is None or not any(abs(start - other) <= MATCH_WINDOW for other in starts):
                continue  # another day's game: no scoreboard is read for it
            try:
                match = find(league, odds_game["home"], odds_game["away"], odds_game["commence_time"])
            except Exception:
                match = None
            if not match:
                continue
            game = next((g for g in games if g.event_id == str(match.get("event_id"))), None) \
                or by_teams.get((str(match.get("home_id")), str(match.get("away_id"))))
            if game is not None and game.odds_id is None:
                game.odds_id = str(odds_game["id"])

    def _build_game(self, board: _Board, game: _Game) -> None:
        """Read the game's injury report, props and rosters, the NBA model's projections (a model league whose odds
        board can be read for free) and each player's game log, noting each part that failed and is missing
        (``game.failed``; the refresh tries them again). A game built again (the NBA model came in) swaps its players
        at the end."""
        league = board.league
        failed: Set[str] = set()
        log_failed: Set[str] = set()
        benched: Dict[str, Dict[str, Any]] = {}
        status = "none"
        try:
            summary_f = self._submit(self.espn.summary, league, game.event_id)
            props_f = self._submit(self.players.props, league, game.event_id)
            roster_f = {side: self._submit(self.espn.roster, league, team["id"])
                        for side, team in (("home", game.home), ("away", game.away)) if team.get("id")}
            ok, summary = self._attempt(summary_f, f"{league} summary {game.event_id}")
            if not ok:
                failed.add("summary")
            statuses, _ = _injuries(summary)
            ok, props = self._attempt(props_f, f"{league} props {game.event_id}")
            props = props or {}
            props_ok = ok and not _unreachable(props)
            self._props_tried(board, game, props_ok and not props.get("stale"))
            rosters: Dict[str, List[Dict[str, Any]]] = {}
            rosters_ok = True
            for side, future in roster_f.items():
                ok, roster = self._attempt(future, f"{league} roster")
                rosters_ok = rosters_ok and ok
                rosters[side] = list(roster or [])
            projected, outs, status = self._projections(league, game, rosters)
            if status == "failed":
                failed.add("projection")
            if not rosters_ok and not projected:
                failed.add("roster")  # a team's players are missing (the model's rotation doesn't need its roster)
            with board.lock:
                if not props_ok and game.props_players:
                    by_player = dict(game.props_players)  # a rebuild keeps the last props that came through
                else:
                    by_player = dict(props.get("players") or {})
                    game.props_meta = {"book": props.get("book") or "DraftKings", "as_of": props.get("as_of"),
                                       "stale": bool(props.get("stale"))}
                    game.props_at = self._props_time(props)
                    game.props_players = by_player
                if not props_ok and not by_player:
                    failed.add("props")
                game.outs = outs
                game.refreshed_at = self._now()
                again = bool(game.players)  # a rebuild (the NBA model came in): swap the players at the end
            pool = self._pool(league, game, rosters, statuses, {"players": by_player}, projected)
            benched = self._benched(game, rosters, statuses) if projected else {}
            built: Dict[str, Dict[str, Any]] = {}
            futures = [(player, self._submit(self._logs, league, player["athlete_id"])) for player in pool]
            for player, future in futures:
                ok, logs = self._attempt(future, f"{league} game log {player['athlete_id']}")
                if not ok:
                    log_failed.add(player["athlete_id"])
                if logs is None and player.get("projection") is None:
                    continue
                entry = {**player, **_log_fields(logs), "props": list(by_player.get(player["athlete_id"]) or [])}
                built[player["athlete_id"]] = entry
                if not again:
                    with board.lock:
                        game.players[player["athlete_id"]] = entry
            if again:
                with board.lock:
                    game.players = built
        except Exception as exc:
            log.warning("Player board: %s game %s failed (%s)", league, game.event_id, type(exc).__name__)
            failed.add("build")
        finally:
            if log_failed:
                failed.add("logs")
            with board.lock:
                game.failed = failed
                game.log_failed = log_failed
                game.model_pending = league in engine.MODEL_LEAGUES and (status in ("failed", "skipped")
                                                                         or not game.odds_id)
                game.benched = benched
                game.ready = True

    def _props_time(self, props: Mapping[str, Any]) -> Optional[float]:
        """When a props read's copy was fetched, on the board's clock: now less its age on the cache's clock (its
        ``as_of`` is on the cache's). None without one."""
        fetched = _epoch(props.get("as_of")) if props.get("as_of") else None
        if fetched is None:
            return None
        clock = getattr(getattr(self.players, "cache", None) or self.cache, "now", None)
        try:
            cache_now = float(clock()) if callable(clock) else time.time()
        except Exception:
            cache_now = time.time()
        return self._now() - max(0.0, cache_now - fetched)

    def _logs(self, league: str, athlete_id: str) -> Dict[str, Any]:
        """{rows, last_season_rows?}: the player's game-log rows, newest first (this whole season, or the last 10
        games reaching into last season early on), and while this season is short of TREND_BASE games last season's
        whole log ({season, stats} per game: the trend's baseline). That comes from the copy the last 10 games
        reached into, already read: no request of its own."""
        found = self.players.gamelog(league, athlete_id, last=engine.RECENT)
        rows = list(found.get("rows") or [])
        season_games = int(found.get("season_games") or 0)
        if season_games > engine.RECENT:
            more = self.players.gamelog(league, athlete_id, last=min(season_games, GAMELOG_MAX))
            rows = list(more.get("rows") or rows)
        if not rows and found.get("reason") and "couldn't be reached" in str(found.get("reason")):
            raise RuntimeError(found["reason"])
        out: Dict[str, Any] = {"rows": rows}
        if season_games < engine.TREND_BASE and len(rows) > season_games:
            # The last 10 reached into last season, so its log is in the cache: all of it, for the trend's baseline
            whole = self.players.gamelog(league, athlete_id, last=GAMELOG_MAX)
            out["last_season_rows"] = [{"season": row.get("season"), "stats": row.get("stats") or {}}
                                       for row in list(whole.get("rows") or [])[season_games:]]
        return out

    def _detail(self, league: str, odds_id: str) -> Any:
        """service.detail (the NBA model's rows for the game), only while the league's odds board can still be read
        without a fetch, checked as the pool task runs: it may have waited behind the rotation's season averages, and
        service.detail reads the odds board, which past its time to live would be fetched (spending credits)."""
        if not self._odds_ready(league):
            raise _OddsNotFree(league)
        return self.service.detail(league, odds_id)

    def _projections(self, league: str, game: _Game, rosters: Mapping[str, Sequence[Mapping[str, Any]]]
                     ) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Dict[str, List[str]], str]:
        """(the NBA model's rows for the game by player id, or None; the significant players out per team, from the
        model's team inputs; how it went: "ok", "failed" (service.detail raised), "skipped" (the odds board could no
        longer be read for free when the task ran) or "none" (not a model league, or no odds game for it)). The
        rotation's season averages are loaded first, FETCHES at once, so the model's own loading finds them cached."""
        if league not in engine.MODEL_LEAGUES or not game.odds_id:
            return None, {}, "none"
        seasons = getattr(self.espn, "player_seasons", None)
        if callable(seasons):
            futures = [self._submit(seasons, league, str(player["id"]))
                       for roster in rosters.values() for player in roster if player.get("id")]
            for future in futures:
                self._result(future, f"{league} player seasons")
        try:
            found = self._submit(self._detail, league, game.odds_id).result()
        except _OddsNotFree:
            log.info("Player board: the %s odds board can't be read for free now; %s stays on recent form", league,
                     game.event_id)
            return None, {}, "skipped"
        except Exception as exc:
            log.info("Player board: NBA projection %s failed (%s)", game.odds_id, type(exc).__name__)
            return None, {}, "failed"
        if not found:
            return None, {}, "ok"
        _card, info, players = found
        return dict(players or {}), self._outs(game, info), "ok"

    @staticmethod
    def _outs(game: _Game, info: Any) -> Dict[str, List[str]]:
        """The significant players out per team (the abbreviation), from the model's team inputs."""
        outs: Dict[str, List[str]] = {}
        teams = (info or {}).get("teams") or {}
        for side, team in (("home", game.home), ("away", game.away)):
            missing = getattr(teams.get(side), "missing", None) or []
            if missing:
                outs[str(team.get("abbr"))] = [str(name) for name in missing]
        return outs

    def _pool(self, league: str, game: _Game, rosters: Mapping[str, Sequence[Mapping[str, Any]]],
              statuses: Mapping[str, str], props: Mapping[str, Any],
              projected: Optional[Mapping[str, Mapping[str, Any]]], report_only: bool = False) -> List[Dict[str, Any]]:
        """The game's players for the scan: the NBA model's rotation, else DraftKings' props players, else the
        roster at the positions that matter. Ruled-out players are left out. A status is the game's injury report's,
        else (unless ``report_only``: a refresh, whose report is newer than the cached roster) the roster's (the
        model's row's: the roster's too, when the report doesn't list him)."""
        sides = {"home": (game.home, game.away, True), "away": (game.away, game.home, False)}
        pool: List[Dict[str, Any]] = []
        if projected:
            for athlete_id, row in projected.items():
                if row.get("out") or (row.get("minutes") or 0) < NBA_MINUTES:
                    continue
                team, other, home = sides.get(row.get("side"), sides["home"])
                status = statuses.get(str(athlete_id)) or (None if report_only else row.get("status")) or None
                means = {stat: row.get(stat) for stat in ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov")}
                pool.append(self._player(game, str(athlete_id), row.get("name"), team, other, home,
                                         row.get("position"), status, {"means": means, "minutes": row.get("minutes")}))
            return pool
        posted = set((props.get("players") or {}).keys())
        roster_players = []
        for side, roster in rosters.items():
            team, other, home = sides[side]
            for player in roster:
                athlete_id = str(player.get("id") or "")
                if not athlete_id or str(player.get("group") or "").lower() in SKIP_GROUPS:
                    continue
                status = statuses.get(athlete_id) or (None if report_only else player.get("status")) or None
                if engine.ruled_out(status) or not _plays(league, player.get("position")):
                    continue
                roster_players.append((side, self._player(game, athlete_id, player.get("name"), team, other, home,
                                                          player.get("position"), status, None)))
        with_props = [player for _, player in roster_players if player["athlete_id"] in posted]
        if len(with_props) >= PROPS_POOL:
            return with_props
        counts: Dict[str, int] = {}
        for side, player in roster_players:
            counts[side] = counts.get(side, 0) + 1
            if counts[side] <= TEAM_CAP or player["athlete_id"] in posted:
                pool.append(player)
        return pool

    @staticmethod
    def _benched(game: _Game, rosters: Mapping[str, Sequence[Mapping[str, Any]]],
                 statuses: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
        """An NBA game's players the model leaves out for an out status as it reads one (the injury report's, else
        the roster's, whose copy may be hours old): {athlete id: {name, side, position}}, from the build's rosters."""
        out: Dict[str, Dict[str, Any]] = {}
        for side, roster in rosters.items():
            for player in roster:
                athlete_id = str(player.get("id") or "")
                if athlete_id and engine.ruled_out(statuses.get(athlete_id) or player.get("status")):
                    out[athlete_id] = {"name": player.get("name"), "side": side, "position": player.get("position")}
        return out

    def _cleared(self, game: _Game, benched: Mapping[str, Mapping[str, Any]], statuses: Mapping[str, str],
                 taken: Iterable[str]) -> List[Dict[str, Any]]:
        """The benched players the injury report read now doesn't rule out, and no one else has taken (the model's
        rotation, the board): each on the form model (no projection) until the model has him, his status the
        report's."""
        sides = {"home": (game.home, game.away, True), "away": (game.away, game.home, False)}
        skip = set(taken)
        cleared = []
        for athlete_id, info in benched.items():
            status = statuses.get(athlete_id) or None
            if athlete_id in skip or engine.ruled_out(status):
                continue
            team, other, home = sides.get(str(info.get("side")), sides["home"])
            cleared.append(self._player(game, athlete_id, info.get("name"), team, other, home, info.get("position"),
                                        status, None))
        return cleared

    @staticmethod
    def _player(game: _Game, athlete_id: str, name: Any, team: Mapping[str, Any], other: Mapping[str, Any],
                home: bool, position: Any, status: Any, projection: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {"athlete_id": athlete_id, "name": name, "team": team.get("abbr") or team.get("name"),
                "team_id": team.get("id"), "opponent": other.get("abbr") or other.get("name"), "home": home,
                "position": position or None, "event_id": game.event_id, "status": status or None,
                "projection": projection}

    # --- props and injuries, again until tip-off -----------------------------------------------------------

    def _refresh_due(self, board: _Board, games: Sequence[_Game], now: float,
                     stale: Sequence[_Game] = ()) -> List[_Game]:
        """Start, in the background, each game's refresh that is due (props and injuries every REFRESH seconds) and,
        for the ``stale`` games (a scan's: their props past the props' time to live), a read of the props alone, at
        most PROPS_RETRY apart; returns the games whose refresh it started. One refresh at a time per game;
        ``game.idle`` is clear while it runs."""
        due = []
        with board.lock:
            for game in games:
                if not game.ready or game.refreshing or game.start <= now:
                    continue
                if game.refreshed_at and now - game.refreshed_at >= self.refresh_every:
                    due.append((game, self._refresh_game))
                elif game in stale and now - game.props_tried >= PROPS_RETRY and now >= game.props_next:
                    game.props_tried = now
                    due.append((game, self._refresh_props))
                else:
                    continue
                game.refreshing = True
                game.idle.clear()
        for game, work in due:
            try:
                threading.Thread(target=work, args=(board, game), name="player-board-refresh", daemon=True).start()
            except RuntimeError:
                with board.lock:
                    game.refreshing = False
                game.idle.set()
        return [game for game, _ in due]

    def _props_tried(self, board: _Board, game: _Game, renewed: bool) -> None:
        """After a read of a game's props: one that renewed the copy leaves the next to the copy's time to live; one
        that didn't (ESPN failing, the last copy served) holds every read of them off for the props' time to live."""
        with board.lock:
            game.props_next = 0.0 if renewed else self._now() + ttl_for("props").ttl

    def _refresh_props(self, board: _Board, game: _Game) -> None:
        """Read the game's props again on their own (their copy is past its time to live, and a scan wants edges from
        them): every player's lines, and the game's book, ``as_of`` and stale flag. The injuries and the player list
        wait for the REFRESH. A read that fails, or brings no newer copy, leaves the lines stale: no edge (RD8), and the
        game's props aren't tried again for their time to live (``props_next``)."""
        league = board.league
        renewed = False
        try:
            ok, props = self._attempt(self._submit(self.players.props, league, game.event_id),
                                      f"{league} props {game.event_id}")
            if ok and isinstance(props, Mapping) and not _unreachable(props):
                renewed = not props.get("stale")  # a stale copy is the last one, served while ESPN fails
                by_player = dict(props.get("players") or {})
                with board.lock:
                    game.props_meta = {"book": props.get("book") or "DraftKings", "as_of": props.get("as_of"),
                                       "stale": bool(props.get("stale"))}
                    game.props_at = self._props_time(props)
                    game.props_players = by_player
                    game.failed.discard("props")
                    for athlete_id, player in list(game.players.items()):
                        game.players[athlete_id] = {**player, "props": list(by_player.get(athlete_id) or [])}
        except Exception as exc:
            log.info("Player board: reading %s %s props again failed (%s)", league, game.event_id, type(exc).__name__)
        finally:
            self._props_tried(board, game, renewed)
            with board.lock:
                game.refreshing = False
            game.idle.set()

    def _refresh_game(self, board: _Board, game: _Game) -> None:
        """Read the game's props and injuries again (and the NBA model's rows, which follow the injuries), try the
        parts that failed again, and rebuild the player list: each status is the injury report's now (a player off it
        has none, and one cleared to play joins, his game log read then), a team whose roster failed comes in, a game
        log that failed is read again. On an NBA game the model projects, the report is the authority too: the
        model's row reads the roster's status when the report doesn't list a player, and the roster's copy may be
        hours old, so a chip the report no longer has goes, and a player the model left out for an out status the
        report no longer has joins (on the form model until the model has him). An NBA game whose odds board came in
        since, or whose model projection failed or waited for the odds, is built again in full once the odds board
        can be read for free."""
        league = board.league
        try:
            if league in engine.MODEL_LEAGUES:
                if not game.odds_id:
                    self._result(self._submit(self._match_odds, league, [game]), f"{league} odds match")
                    if game.odds_id:  # the odds board came in since: the NBA model's rotation and projections now
                        self._build_game(board, game)
                        return
                elif game.model_pending and self._odds_ready(league):
                    self._build_game(board, game)  # the projection failed, or waited for the odds: try it again
                    return
            with board.lock:
                if "build" in game.failed:
                    rebuild = True
                else:
                    rebuild = False
                    known = dict(game.players)
                    failed = set(game.failed)
                    log_failed = set(game.log_failed)
                    last_props = dict(game.props_players)
                    benched = dict(game.benched)
                    props_held = self._now() < game.props_next  # a props read failed under a time to live ago
            if rebuild:
                self._build_game(board, game)
                return
            modelled = any(player.get("projection") is not None for player in known.values())
            summary_f = self._submit(self.espn.summary, league, game.event_id)
            props_f = None if props_held else self._submit(self.players.props, league, game.event_id)
            roster_f = {} if modelled else {side: self._submit(self.espn.roster, league, team["id"])
                                             for side, team in (("home", game.home), ("away", game.away))
                                             if team.get("id")}
            ok, summary = self._attempt(summary_f, f"{league} summary {game.event_id}")
            statuses: Optional[Dict[str, str]] = _injuries(summary)[0] if ok else None
            # Props held off: the last ones stay (as when the read fails); else read, and hold off if not renewed
            ok, props = self._attempt(props_f, f"{league} props {game.event_id}") if props_f else (False, None)
            props_ok = ok and not _unreachable(props)
            if props_f is not None:
                self._props_tried(board, game, props_ok and not (props or {}).get("stale"))
            by_player: Optional[Dict[str, Any]] = dict((props or {}).get("players") or {}) if props_ok else None
            rosters: Dict[str, List[Dict[str, Any]]] = {}
            rosters_ok = True
            for side, future in roster_f.items():
                ok, roster = self._attempt(future, f"{league} roster")
                rosters_ok = rosters_ok and ok
                rosters[side] = list(roster or [])
            projected: Optional[Dict[str, Dict[str, Any]]] = None
            outs = None
            if modelled and game.odds_id and self._odds_ready(league):
                found = self._result(self._submit(self._detail, league, game.odds_id), f"NBA projection {game.odds_id}")
                if found:
                    _card, info, rows = found
                    projected = dict(rows or {})
                    outs = self._outs(game, info)
            # Who plays now: the model's rotation, or the props or roster pool by the report as it stands
            posted = by_player if by_player is not None else last_props
            if modelled:
                pool = self._pool(league, game, {}, statuses or {}, {}, projected,
                                  report_only=statuses is not None) if projected else []
                if statuses is not None:  # left out for an out status the report no longer has: in, on the form model
                    pool += self._cleared(game, benched, statuses, [p["athlete_id"] for p in pool] + list(known))
            else:
                pool = self._pool(league, game, rosters, statuses or {}, {"players": posted}, None,
                                  report_only=statuses is not None)
            fresh = [player for player in pool if player["athlete_id"] not in known]
            again = [known[athlete_id] for athlete_id in sorted(log_failed) if athlete_id in known]
            futures = [(player, self._submit(self._logs, league, player["athlete_id"])) for player in fresh + again]
            loaded: Dict[str, Optional[Dict[str, Any]]] = {}
            for player, future in futures:
                ok, logs = self._attempt(future, f"{league} game log {player['athlete_id']}")
                loaded[player["athlete_id"]] = logs if ok else None
            with board.lock:
                for athlete_id, player in list(game.players.items()):
                    update = dict(player)
                    row = (projected or {}).get(athlete_id)
                    on_model = player.get("projection") is not None
                    if projected is not None and row is not None:
                        # The model's row (a cleared player on the form model moves onto it once the model has him)
                        update["projection"] = {"means": {stat: row.get(stat) for stat in
                                                          ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov")},
                                                "minutes": row.get("minutes")}
                        if statuses is not None:  # the report as it stands is the authority, not the roster's copy
                            update["status"] = statuses.get(athlete_id) or None
                        else:
                            update["status"] = row.get("status") or update.get("status")
                    elif projected is not None and on_model:
                        update["status"] = (statuses or {}).get(athlete_id) or "Out"  # the model has him out now
                    elif statuses is not None:
                        # The report as it stands: off it, no status (a "Q" a player left the report with goes)
                        update["status"] = statuses.get(athlete_id) or None
                    if by_player is not None:
                        update["props"] = list(by_player.get(athlete_id) or [])
                    if loaded.get(athlete_id) is not None:
                        update.update(_log_fields(loaded[athlete_id]))
                    game.players[athlete_id] = update
                for player in fresh:
                    logs = loaded.get(player["athlete_id"])
                    if logs is None and player.get("projection") is None:
                        continue
                    game.players[player["athlete_id"]] = {**player, **_log_fields(logs),
                                                          "props": list(posted.get(player["athlete_id"]) or [])}
                still = {athlete_id for athlete_id, logs in loaded.items() if logs is None}  # failed again
                game.log_failed = still
                if statuses is not None:
                    failed.discard("summary")
                if props_ok:
                    failed.discard("props")
                    game.props_meta = {"book": props.get("book") or "DraftKings", "as_of": props.get("as_of"),
                                       "stale": bool(props.get("stale"))}
                    game.props_at = self._props_time(props)
                    game.props_players = dict(by_player or {})
                if rosters_ok and not modelled:
                    failed.discard("roster")  # (a roster that fails now, after its team came in, leaves no hole)
                if still:
                    failed.add("logs")
                else:
                    failed.discard("logs")
                game.failed = failed
                if outs is not None:
                    game.outs = outs
        except Exception as exc:
            log.info("Player board: refreshing %s %s failed (%s)", league, game.event_id, type(exc).__name__)
        finally:
            with board.lock:
                game.refreshed_at = self._now()
                game.refreshing = False
            game.idle.set()


def _log_fields(logs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """A board entry's game-log fields from PlayerBoard._logs' answer (None: the read failed): ``rows``, and
    ``last_season_rows`` when they were read."""
    out: Dict[str, Any] = {"rows": list((logs or {}).get("rows") or [])}
    if logs and logs.get("last_season_rows") is not None:
        out["last_season_rows"] = list(logs["last_season_rows"])
    return out


def _injuries(summary: Any) -> Tuple[Dict[str, str], Dict[str, str]]:
    """({athlete id: status}, {athlete id: name}) from an ESPN game summary's injury report."""
    statuses: Dict[str, str] = {}
    names: Dict[str, str] = {}
    if not isinstance(summary, Mapping):
        return statuses, names
    for team in summary.get("injuries") or []:
        for item in (team or {}).get("injuries") or []:
            athlete = (item or {}).get("athlete") or {}
            athlete_id = str(athlete.get("id") or "")
            if athlete_id:
                statuses[athlete_id] = str(item.get("status") or "")
                names[athlete_id] = str(athlete.get("displayName") or "")
    return statuses, names


def _plays(league: str, position: Any) -> bool:
    """Whether a roster position is one the league's stats are about (all, except the NFL's linemen, defense
    and kickers, and soccer's goalkeepers)."""
    wanted = POSITIONS.get(league)
    text = str(position or "").upper()
    if wanted is not None:
        return text in wanted
    if league.startswith("soccer_"):
        return text in SOCCER_POSITIONS or not text
    return True


def board_for(service: Any, players: Optional[Any] = None, **options: Any) -> PlayerBoard:
    """The process's PlayerBoard, made once and kept on the service (``service.player_board``), so the Lab's scans and
    the slate worker's prewarm share its boards."""
    board = getattr(service, "player_board", None)
    if isinstance(board, PlayerBoard):
        return board
    lock = getattr(board_for, "_lock")
    with lock:
        board = getattr(service, "player_board", None)
        if not isinstance(board, PlayerBoard):
            board = PlayerBoard(service, players, getattr(service, "cache", None), **options)
            service.player_board = board
    return board


board_for._lock = threading.Lock()  # type: ignore[attr-defined]
