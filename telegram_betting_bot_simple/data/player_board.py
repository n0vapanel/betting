"""Player boards: per league and day, everything the player scan (engine/scan.py) reads.

A board holds one day's games in a league (ESPN's scoreboard, games not yet started) and, for each game:

- **projections:** NBA games use the NBA model's own projection for the game (``DashboardService.detail``: the
  same rows the player card reads), but only while the league's odds board can be read without a fetch (loaded
  or saved within the odds service's time to live, or the quota under its reserve): this feature never spends
  The Odds API's credits. Without it, and in every other league, the scan's form model works from the game logs;
- **game logs:** each player's (``PlayerData.gamelog``: this season, and last season's tail early on), kept by
  the shared cache's player-log time to live;
- **injury status:** ESPN's injury report for the game (its summary), else the roster's;
- **props:** DraftKings' player props for the event through ESPN (``PlayerData.props``), never The Odds API.

Which players: an NBA game's projected rotation; elsewhere the players DraftKings has props for (when it has them
for at least PROPS_POOL players of the game), else the roster at the positions that matter (the NFL's QB, RB,
WR, TE; soccer's outfield players), ruled-out players left out.

Fetching: everything goes through the existing ESPN client, PlayerData and the shared cache (no new hosts), at
most FETCHES requests at once for every board together; an NBA game's players' season averages are loaded at that
width before the model runs, so its own pool finds them cached. A board builds in the background, game by game in
start order (LANES games side by side), starting on the first scan of the day for the league or from the slate
worker's game-day prewarm (leagues with games in the next 12 hours: the NBA first, then the NHL, NFL and MLB).
Each game's props and injuries are read again every REFRESH seconds until tip-off.

A scan never waits long: it answers with what is ready, plus {"loading": {"games_ready", "games_total"}} while the
board is still building (SCAN_WAIT seconds the first time).
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..engine import scan as engine
from .espn_client import ESPN_DAY_TZ, competition_state
from .espn_players import PlayerData

log = logging.getLogger(__name__)

FETCHES = 4  # ESPN requests at once, every board together
LANES = 2  # games built side by side (their requests share the FETCHES)
REFRESH = 30 * 60  # seconds between reads of a game's props and injuries, until tip-off
SCAN_WAIT = 2.0  # seconds a scan waits for a board that has just started building
PREWARM_HOURS = 12  # the slate's prewarm builds boards for leagues with a game starting within this many hours
PREWARM_LEAGUES = (engine.NBA, engine.NHL, engine.NFL, engine.MLB, engine.WNBA)  # ... in this order
MATCH_WINDOW = 3 * 3600  # an odds game and an ESPN game further apart than this are different games
ODDS_MARGIN = 120.0  # odds this close to the odds service's time to live count as stale (no fetch can start)
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
        self.label = f"{self.away.get('abbr') or self.away.get('name')} @ {self.home.get('abbr') or self.home.get('name')}"
        self.odds_id: Optional[str] = None
        self.ready = False
        self.players: Dict[str, Dict[str, Any]] = {}
        self.props_meta: Dict[str, Any] = {}
        self.outs: Dict[str, List[str]] = {}
        self.refreshed_at = 0.0
        self.refreshing = False


class _Board:
    def __init__(self, league: str, day: datetime.date) -> None:
        self.league = league
        self.day = day
        self.games: Dict[str, _Game] = {}
        self.started = 0  # the day's games already under way or over when the board was built
        self.lock = threading.Lock()
        self.building = False
        self.loaded = False  # the scoreboard has been read
        self.done = threading.Event()
        self.built_at: Optional[float] = None
        self.error: Optional[str] = None
        self.thread: Optional[threading.Thread] = None


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

    def snapshot(self, league: str, day: Any = "today", start: bool = True) -> Dict[str, Any]:
        """What engine.scan reads (its docstring has the shape): the games not started yet and the players loaded so
        far. Starts the board's build (``start``) and any refresh of props and injuries that is due."""
        if start:
            board = self.ensure(league, day)
        else:
            with self._lock:
                board = self._boards.get((league, self.day_of(day)))
            if board is None:
                return self._empty(league, self.day_of(day))
        now = self._now()
        with board.lock:
            games = list(board.games.values())
            started = board.started + sum(1 for game in games if game.start <= now)
            live = [game for game in games if game.start > now]
            view_games = {game.event_id: {"event_id": game.event_id, "label": game.label, "start": game.start_text,
                                          "outs": dict(game.outs), "props": dict(game.props_meta)}
                          for game in live}
            players = [player for game in live for player in game.players.values()]
            ready = sum(1 for game in live if game.ready)
        self._refresh_due(board, live, now)
        stamps = [game.refreshed_at for game in live if game.refreshed_at]
        return {"league": league, "day": board.day.isoformat(), "as_of": _iso(min(stamps)) if stamps else _iso(now),
                "games": view_games, "players": players, "games_ready": ready, "games_total": len(live),
                "games_started": started, "building": board.building, "loaded": board.loaded,
                "error": board.error}

    def _empty(self, league: str, day: datetime.date) -> Dict[str, Any]:
        return {"league": league, "day": day.isoformat(), "as_of": _iso(self._now()), "games": {}, "players": [],
                "games_ready": 0, "games_total": 0, "games_started": 0, "building": False, "loaded": False,
                "error": None}

    def scan(self, league: str, stat: Any, threshold: Any = None, side: str = "over", day: str = "today",
             n: int = 10, wait: Optional[float] = None) -> Dict[str, Any]:
        """The scan for one league: engine.scan's result for make_ask(league, stat, threshold, side, day), plus
        {league, day, title, no_games}. A board that has just started building is waited for up to ``wait`` seconds
        (SCAN_WAIT by default); after that whatever is ready is answered, with ``loading``. Raises ValueError for an
        ask that doesn't make sense (an unknown league or stat, a bad number)."""
        ask = engine.make_ask(league, stat, threshold, side, day)
        board = self.ensure(league, day)
        limit = self.wait_seconds if wait is None else max(0.0, float(wait))
        if limit and not board.done.is_set():
            board.done.wait(limit)
        view = self.snapshot(league, day)
        result = engine.scan(view, ask, n)
        result.update(league=league, day=view["day"], title=engine.TITLES.get(league, league),
                      no_games=bool(view["loaded"]) and not view["building"] and view["games_total"] == 0)
        if not view["loaded"]:
            result["loading"] = {"games_ready": 0, "games_total": 0}  # the day's games are still being read
        if board.error and not view["players"]:
            result["error"] = board.error
        return result

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
        return self._fetch.submit(fn, *args)

    @staticmethod
    def _result(future: "Future[Any]", what: str) -> Any:
        try:
            return future.result()
        except Exception as exc:
            log.info("Player board: %s failed (%s)", what, type(exc).__name__)
            return None

    def _build(self, board: _Board) -> None:
        try:
            events = self._read_events(board.league, board.day)
            now = self._now()
            parsed = parse_events(events)
            upcoming = [event for event in parsed if event["state"] == "pre" and event["start"] > now]
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
        except Exception as exc:
            log.warning("Player board for %s %s failed: %s", board.league, board.day, type(exc).__name__)
            board.error = "ESPN couldn't be reached for this league's games just now."
        finally:
            with board.lock:
                board.loaded = True
                board.building = False
                board.built_at = self._now()
            board.done.set()

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
        league = board.league
        try:
            summary_f = self._submit(self.espn.summary, league, game.event_id)
            props_f = self._submit(self.players.props, league, game.event_id)
            roster_f = {side: self._submit(self.espn.roster, league, team["id"])
                        for side, team in (("home", game.home), ("away", game.away)) if team.get("id")}
            statuses, _ = _injuries(self._result(summary_f, f"{league} summary {game.event_id}"))
            props = self._result(props_f, f"{league} props {game.event_id}") or {}
            rosters = {side: self._result(future, f"{league} roster") or [] for side, future in roster_f.items()}
            projected, outs = self._projections(league, game, rosters)
            pool = self._pool(league, game, rosters, statuses, props, projected)
            with board.lock:
                game.props_meta = {"book": props.get("book") or "DraftKings", "as_of": props.get("as_of"),
                                   "stale": bool(props.get("stale"))}
                game.outs = outs
                game.refreshed_at = self._now()
            by_player = props.get("players") or {}
            with board.lock:
                again = bool(game.players)  # a rebuild (the NBA model came in): swap the players at the end
            built: Dict[str, Dict[str, Any]] = {}
            futures = [(player, self._submit(self._logs, league, player["athlete_id"])) for player in pool]
            for player, future in futures:
                rows = self._result(future, f"{league} game log {player['athlete_id']}")
                if rows is None and player.get("projection") is None:
                    continue
                entry = {**player, "rows": rows or [], "props": list(by_player.get(player["athlete_id"]) or [])}
                built[player["athlete_id"]] = entry
                if not again:
                    with board.lock:
                        game.players[player["athlete_id"]] = entry
            if again:
                with board.lock:
                    game.players = built
        except Exception as exc:
            log.warning("Player board: %s game %s failed (%s)", league, game.event_id, type(exc).__name__)
        finally:
            with board.lock:
                game.ready = True

    def _logs(self, league: str, athlete_id: str) -> List[Dict[str, Any]]:
        """The player's game-log rows, newest first: this whole season, or the last 10 games reaching into last
        season early on."""
        found = self.players.gamelog(league, athlete_id, last=engine.RECENT)
        rows = list(found.get("rows") or [])
        season_games = int(found.get("season_games") or 0)
        if season_games > engine.RECENT:
            more = self.players.gamelog(league, athlete_id, last=min(season_games, GAMELOG_MAX))
            rows = list(more.get("rows") or rows)
        if not rows and found.get("reason") and "couldn't be reached" in str(found.get("reason")):
            raise RuntimeError(found["reason"])
        return rows

    def _projections(self, league: str, game: _Game, rosters: Mapping[str, Sequence[Mapping[str, Any]]]
                     ) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Dict[str, List[str]]]:
        """(the NBA model's rows for the game by player id, or None; the significant players out per team, from the
        model's team inputs). The rotation's season averages are loaded first, FETCHES at once, so the model's own
        loading finds them cached."""
        if league not in engine.MODEL_LEAGUES or not game.odds_id:
            return None, {}
        seasons = getattr(self.espn, "player_seasons", None)
        if callable(seasons):
            futures = [self._submit(seasons, league, str(player["id"]))
                       for roster in rosters.values() for player in roster if player.get("id")]
            for future in futures:
                self._result(future, f"{league} player seasons")
        found = self._result(self._submit(self.service.detail, league, game.odds_id), f"NBA projection {game.odds_id}")
        if not found:
            return None, {}
        _card, info, players = found
        outs: Dict[str, List[str]] = {}
        teams = (info or {}).get("teams") or {}
        for side, team in (("home", game.home), ("away", game.away)):
            missing = getattr(teams.get(side), "missing", None) or []
            if missing:
                outs[str(team.get("abbr"))] = [str(name) for name in missing]
        return dict(players or {}), outs

    def _pool(self, league: str, game: _Game, rosters: Mapping[str, Sequence[Mapping[str, Any]]],
              statuses: Mapping[str, str], props: Mapping[str, Any],
              projected: Optional[Mapping[str, Mapping[str, Any]]]) -> List[Dict[str, Any]]:
        """The game's players for the scan: the NBA model's rotation, else DraftKings' props players, else the
        roster at the positions that matter. Ruled-out players are left out."""
        sides = {"home": (game.home, game.away, True), "away": (game.away, game.home, False)}
        pool: List[Dict[str, Any]] = []
        if projected:
            for athlete_id, row in projected.items():
                if row.get("out") or (row.get("minutes") or 0) < NBA_MINUTES:
                    continue
                team, other, home = sides.get(row.get("side"), sides["home"])
                status = statuses.get(str(athlete_id)) or row.get("status") or None
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
                status = statuses.get(athlete_id) or player.get("status") or None
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
    def _player(game: _Game, athlete_id: str, name: Any, team: Mapping[str, Any], other: Mapping[str, Any],
                home: bool, position: Any, status: Any, projection: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {"athlete_id": athlete_id, "name": name, "team": team.get("abbr") or team.get("name"),
                "team_id": team.get("id"), "opponent": other.get("abbr") or other.get("name"), "home": home,
                "position": position or None, "event_id": game.event_id, "status": status or None,
                "projection": projection}

    # --- props and injuries, again until tip-off -----------------------------------------------------------

    def _refresh_due(self, board: _Board, games: Sequence[_Game], now: float) -> None:
        due = []
        with board.lock:
            for game in games:
                if game.ready and not game.refreshing and game.refreshed_at \
                        and now - game.refreshed_at >= self.refresh_every and game.start > now:
                    game.refreshing = True
                    due.append(game)
        for game in due:
            try:
                threading.Thread(target=self._refresh_game, args=(board, game), name="player-board-refresh",
                                 daemon=True).start()
            except RuntimeError:
                with board.lock:
                    game.refreshing = False

    def _refresh_game(self, board: _Board, game: _Game) -> None:
        """Read the game's props and injuries again (and the NBA model's rows, which follow the injuries)."""
        league = board.league
        try:
            if league in engine.MODEL_LEAGUES and not game.odds_id:
                self._match_odds(league, [game])
                if game.odds_id:  # the odds board came in since: the NBA model's rotation and projections now
                    self._build_game(board, game)
                    return
            summary_f = self._submit(self.espn.summary, league, game.event_id)
            props_f = self._submit(self.players.props, league, game.event_id)
            statuses, _ = _injuries(self._result(summary_f, f"{league} summary {game.event_id}"))
            props = self._result(props_f, f"{league} props {game.event_id}")
            projected = None
            outs = None
            if league in engine.MODEL_LEAGUES and game.odds_id and self._odds_ready(league):
                found = self._result(self._submit(self.service.detail, league, game.odds_id),
                                     f"NBA projection {game.odds_id}")
                if found:
                    _card, info, rows = found
                    projected = rows or {}
                    teams = (info or {}).get("teams") or {}
                    outs = {str(team.get("abbr")): [str(n) for n in getattr(teams.get(side), "missing", None) or []]
                            for side, team in (("home", game.home), ("away", game.away))}
            with board.lock:
                by_player = (props or {}).get("players") if props else None
                for athlete_id, player in list(game.players.items()):
                    update = dict(player)
                    if statuses.get(athlete_id):
                        update["status"] = statuses[athlete_id]
                    if by_player is not None:
                        update["props"] = list(by_player.get(athlete_id) or [])
                    row = (projected or {}).get(athlete_id)
                    if projected is not None and row is not None and player.get("projection") is not None:
                        update["projection"] = {"means": {stat: row.get(stat) for stat in
                                                          ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov")},
                                                "minutes": row.get("minutes")}
                        update["status"] = statuses.get(athlete_id) or row.get("status") or update.get("status")
                    elif projected is not None and row is None and player.get("projection") is not None:
                        update["status"] = statuses.get(athlete_id) or "Out"  # the model no longer has him playing
                    game.players[athlete_id] = update
                if props:
                    game.props_meta = {"book": props.get("book") or "DraftKings", "as_of": props.get("as_of"),
                                       "stale": bool(props.get("stale"))}
                if outs is not None:
                    game.outs = {team: names for team, names in outs.items() if names}
        except Exception as exc:
            log.info("Player board: refreshing %s %s failed (%s)", league, game.event_id, type(exc).__name__)
        finally:
            with board.lock:
                game.refreshed_at = self._now()
                game.refreshing = False


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
