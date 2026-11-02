"""The brain's every-game record (v4 P3-lite): a logged prediction for every game ESPN's boards hold, graded after the
final, and the last RETRO_DAYS days back-tested walk-forward. These are learning records, never picks: nothing here is
shown as a suggestion, and the picks' rules (responsible gambling, the 24-hour pause) are untouched.

**Live** (``log_round``, from data/slate.py's round, after the markets brain): every game of the 55 team leagues
(espn_client.LEAGUES: the ones whose single-game result the grader can read) starting within LEAD (60 minutes) is
logged into ``ai_predictions`` (Store.log_ai_prediction) and updated each round until its start, then frozen. Its market
input, in order:

1. ``odds_api``: the Odds API board the slate already holds for the game (its "Sharp" and "US books" sources, when one
   is "ok"); a board is never fetched for this;
2. ``espn_line``: ESPN's own free line, from the board the slate already fetched (EspnClient.event_line: the
   competition's ``odds``, read before the lean copy drops them) or a summary's ``pickcenter`` already in the lean
   cache or raw.db (never fetched for this), parsed as learn/sources/espn_hist.parse_pickcenter does (``parse_lines``):
   the close, else the open (``pick_line``);
3. ``model``: the brain's own ratings (ESPN's predictor, the power ratings, the NBA team model), where the sport has them.

With none of the three nothing is logged (``no_input`` in the round's counts): never a made-up 50%. The brain's view
(``view``) is the markets brain's blend (engine/consensus.py) of the market input with the game's other opinions at the
sport's live parameters, logged as ``{"source", "retro", "p", "market_p", "home_win", "market_home_win", "margin",
"total", "market_margin", "market_total", "line", "as_of", "params_version", "confidence", "swapped", "odds_id"}``:
``p`` and ``market_p`` are {"home", "away", "draw"?} in the row's orientation (the slate's: the odds feed's when
``swapped``), the spread and total are the line's (``margin`` is the home side's expected margin: minus its spread).
A game is keyed by the odds feed's id when it has one, else ESPN's id, one row per game: a priced NBA prediction (a
payload with no ``source``) is never overwritten, and its record stays apart (``is_every_game``).

**Grading** (``grade``, from DashboardService.grade_due, matched by ESPN's id): the outcome in the row's orientation,
then for the brain and for the market line it used whether the most likely outcome happened (``hit``), the Brier score
and the log loss (engine/consensus.score; None on a two-way market's tie). No result by the grader's give-up time: void.

**Back-test** (``backtest``): every final of the last RETRO_DAYS days that raw.db's capture holds (its jobs and final
summaries) or the consensus log names, its summary read from raw.db, else the lean cache (never fetched here: a game
without one waits for the capture), predicted as at LEAD before its start from only what was known before it: the
summary's pickcenter by espn_hist's lead rule (the close dated CLOSE_LEAD before the start, the open OPEN_LEAD), the
consensus log's pre-start opinions and the parameters live at that time (``params_at``). Logged with ``retro: true``
and graded at once; a re-run adds nothing (a game already logged is left as it is).

``summary(store)``: the Brain page's and /brain's numbers per sport (live, back-tested and the priced picks apart).
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import threading
import time
from contextlib import closing
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..data.espn_client import LEAGUES as ESPN_LEAGUES
from ..engine import consensus
from . import registry
from .sources import espn_hist

log = logging.getLogger(__name__)

LEAD = 3600.0                 # a game is logged from an hour before its start until the start
RETRO_DAYS = 14
RETRO_EVERY = 6 * 3600.0      # the grader runs the back-test at most this often (it is idempotent)
SETTLED = 3 * 3600.0          # the back-test's newest games started at least this long ago
STARTED_LIVE = 6 * 3600.0     # the grader trusts ESPN's board saying "still on" for this long after the start
SOURCES = ("odds_api", "espn_line", "model")
BOOK_KINDS = ("sharp", "us_books")
OPINION_KINDS = ("prediction_market", "model_rating", "team_model")
RATING_KINDS = ("model_rating", "team_model")
LINE_NAME = "ESPN line"
BOOK_ORDER = ("espn_bet", "draftkings")
OUTCOMES = ("home", "away", "draw")
COUNTS = ("logged", "unchanged", "held", "no_input", "failed") + SOURCES
BLOCKS = ("live", "retro", "priced")
_summary_lock = threading.Lock()
_summary_cache: Dict[Any, Tuple[Any, Dict[str, Any]]] = {}


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.datetime.fromtimestamp(float(ts), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def is_every_game(payload: Any) -> bool:
    """Whether a logged ai_predictions payload is this record's (it names its market input), not a priced pick."""
    return isinstance(payload, Mapping) and payload.get("source") in SOURCES


# --- ESPN's own line ------------------------------------------------------------------------------------------------

def _flat(odds: Any) -> List[Dict[str, Any]]:
    """The older flat odds shape (each side's ``moneyLine``, the home ``spread``, the ``overUnder``) as close lines."""
    lines: List[Dict[str, Any]] = []
    for pick in odds if isinstance(odds, list) else []:
        if not isinstance(pick, Mapping):
            continue
        book = espn_hist.book_name(pick.get("provider") or {})
        if book is None:
            continue
        base = {"book": book, "kind": "close", "lead": espn_hist.CLOSE_LEAD}
        for side, key in (("home", "homeTeamOdds"), ("away", "awayTeamOdds"), ("draw", "drawOdds")):
            block = pick.get(key)
            price = espn_hist.american(block.get("moneyLine")) if isinstance(block, Mapping) else None
            if price is not None:
                lines.append({**base, "market": "h2h", "side": side, "point": None, "price": price})
        spread, total = _num(pick.get("spread")), _num(pick.get("overUnder"))
        if spread is not None:
            lines.append({**base, "market": "spreads", "side": "home", "point": spread, "price": None})
        if total is not None:
            lines.append({**base, "market": "totals", "side": "over", "point": total, "price": None})
    return lines


def parse_lines(odds: Any) -> List[Dict[str, Any]]:
    """A board competition's ``odds`` or a summary's ``pickcenter`` as espn_hist.parse_pickcenter's lines (each kept
    book's open and close, with its ``lead``), plus the older flat shape's prices as close lines when it gave no
    moneyline. Never raises."""
    try:
        lines = espn_hist.parse_pickcenter(odds if isinstance(odds, list) else [])
    except Exception:  # noqa: BLE001 - one odd entry costs only its line
        lines = []
    if not any(line.get("market") == "h2h" for line in lines):
        try:
            lines += _flat(odds)
        except Exception:  # noqa: BLE001
            pass
    return lines


def devig(prices: Mapping[str, Any], three_way: bool = False) -> Optional[Dict[str, float]]:
    """Decimal prices per outcome as margin-free chances (each 1/price over their sum); None without both sides (and
    the draw, ``three_way``)."""
    inverse = {side: 1.0 / float(price) for side, price in (prices or {}).items()
               if side in OUTCOMES and isinstance(price, (int, float)) and math.isfinite(price) and price > 1.0}
    if "home" not in inverse or "away" not in inverse or (three_way and "draw" not in inverse):
        return None
    if not three_way:
        inverse.pop("draw", None)
    total = math.fsum(inverse.values())
    return {side: round(value / total, 6) for side, value in inverse.items()}


def _book_rank(book: str) -> Tuple[int, str]:
    return (BOOK_ORDER.index(book) if book in BOOK_ORDER else len(BOOK_ORDER), book)


def pick_line(lines: Sequence[Mapping[str, Any]], *, as_of: Optional[float] = None, start: Optional[float] = None,
              three_way: bool = False) -> Optional[Dict[str, Any]]:
    """The ESPN line the brain uses: the close, else the open; within a kind ESPN BET, DraftKings, then any other book,
    the first with a full moneyline. Dated ``as_of`` (a board's fetch time), else, with ``start``, by espn_hist's lead
    rule (the close CLOSE_LEAD, the open OPEN_LEAD before the start). {"p", "spread" (the home side's point), "total",
    "kind", "book", "as_of"} or None."""
    for kind in ("close", "open"):
        books: Dict[str, List[Mapping[str, Any]]] = {}
        for line in lines or []:
            if isinstance(line, Mapping) and line.get("kind") == kind and line.get("book"):
                books.setdefault(str(line["book"]), []).append(line)
        for book in sorted(books, key=_book_rank):
            rows = books[book]
            p = devig({line["side"]: line.get("price") for line in rows if line.get("market") == "h2h"}, three_way)
            if p is None:
                continue
            spread = next((_num(line.get("point")) for line in rows if line.get("market") == "spreads"
                           and line.get("side") == "home" and _num(line.get("point")) is not None), None)
            if spread is None:
                away = next((_num(line.get("point")) for line in rows if line.get("market") == "spreads"
                             and line.get("side") == "away" and _num(line.get("point")) is not None), None)
                spread = -away if away is not None else None
            total = next((_num(line.get("point")) for line in rows if line.get("market") == "totals"
                          and _num(line.get("point")) is not None), None)
            lead = float(rows[0].get("lead") or (espn_hist.CLOSE_LEAD if kind == "close" else espn_hist.OPEN_LEAD))
            when = as_of if as_of is not None else (start - lead if start is not None else None)
            return {"p": p, "spread": spread, "total": total, "kind": kind, "book": book, "as_of": _iso(when)}
    return None


def held_line(espn: Any, league: str, espn_id: str, start: Optional[float],
              raw: Optional[Callable[[str, str], Optional[Tuple[float, Any]]]] = None) -> Optional[Dict[str, Any]]:
    """ESPN's own line for a game still to start (in ESPN's orientation) from what is already held: the newest board's
    odds (EspnClient.event_line), else a summary copy's pickcenter in the lean cache (EspnClient.held_summary), else
    raw.db's newest summary (``raw(league, espn_id)`` -> (fetched_at, data)); each only when read before the start.
    Nothing is fetched."""
    three_way = league.startswith("soccer_")

    def read(name: str) -> Optional[Any]:
        reader = getattr(espn, name, None)
        if not callable(reader):
            return None
        try:
            return reader(league, espn_id)
        except Exception:  # noqa: BLE001
            return None

    board = read("event_line")
    if board and (start is None or board[1] < start):
        line = pick_line(board[0], as_of=board[1], three_way=three_way)
        if line:
            return line
    held = [read("held_summary")]
    if raw is not None:
        try:
            held.append(raw(league, espn_id))
        except Exception:  # noqa: BLE001
            pass
    for copy in held:
        if copy and isinstance(copy[1], Mapping) and (start is None or float(copy[0]) < start):
            line = pick_line(parse_lines(copy[1].get("pickcenter")), as_of=float(copy[0]), three_way=three_way)
            if line:
                return line
    return None


def flip(p: Mapping[str, Any]) -> Dict[str, Any]:
    return {{"home": "away", "away": "home"}.get(k, k): v for k, v in p.items()}


def oriented(line: Optional[Mapping[str, Any]], swapped: bool) -> Optional[Dict[str, Any]]:
    """An ESPN line (ESPN's home and away) turned to a row whose home is ESPN's away (``swapped``)."""
    if not line or not swapped:
        return dict(line) if line else None
    spread = line.get("spread")
    return {**line, "p": flip(line["p"]), "spread": -spread if spread is not None else None}


def line_source(line: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """An ESPN line as the brain's "US books"-kind source (a bookmaker's margin-free moneyline)."""
    try:
        source = consensus.make_source(LINE_NAME, "us_books", line.get("p"), line.get("as_of"),
                                       detail=f"ESPN's own line ({line.get('book')}, {line.get('kind')})")
    except ValueError:
        return None
    return dict(source) if source.get("status") == "ok" and source.get("p") else None


# --- the brain's view of one game -----------------------------------------------------------------------------------

def _before(source: Mapping[str, Any], start: Optional[float]) -> bool:
    ts = _ts(source.get("as_of")) if source.get("as_of") else None
    return start is None or ts is None or ts < start


def view(league: str, market: str, sources: Sequence[Optional[Mapping[str, Any]]], *, start: Optional[float],
         now: float, params: Any = None, line: Optional[Mapping[str, Any]] = None, neutral: bool = False,
         retro: bool = False, swapped: bool = False, odds_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The payload logged for one game (see the module docstring): the blend of ``sources`` (only "ok" ones count,
    none dated from the start on) at ``params``; None when nothing blends, or the market input has no chance."""
    usable = [dict(source) for source in sources if source and _before(source, start)]
    if not usable:
        return None
    tuned = params.blend_kwargs() if params is not None else {}
    try:
        blend = consensus.blend(usable, league, now=now, start=start, neutral=neutral, **tuned)
    except Exception:  # noqa: BLE001 - one game's odd opinion costs only that game
        log.warning("Every-game record: blending %s failed", league, exc_info=True)
        return None
    if not blend or not blend.get("p"):
        return None
    market_p: Optional[Dict[str, float]] = None
    if market == "odds_api":
        ok = [s for s in usable if s.get("status") == "ok"]
        book = next((s for kind in BOOK_KINDS for s in ok if s.get("kind") == kind), None)
        market_p = dict(book["p"]) if book else None
        if market_p is None:
            return None
    elif market == "espn_line":
        market_p = dict(line["p"]) if line and line.get("p") else None
        if market_p is None:
            return None
    p = {k: round(float(v), 6) for k, v in blend["p"].items() if k in OUTCOMES}
    spread = _num((line or {}).get("spread"))
    total = _num((line or {}).get("total"))
    margin = -spread if spread is not None else None
    return {"source": market, "retro": bool(retro), "p": p,
            "market_p": {k: round(float(v), 6) for k, v in market_p.items()} if market_p else None,
            "home_win": p.get("home"), "market_home_win": round(float(market_p["home"]), 6) if market_p else None,
            "margin": margin, "total": total, "market_margin": margin, "market_total": total,
            "line": (line or {}).get("kind") if market == "espn_line" else None,
            "as_of": (line or {}).get("as_of") if market == "espn_line" else None,
            "params_version": int(getattr(params, "version", 0) or 0), "confidence": blend.get("confidence"),
            "swapped": bool(swapped), "odds_id": odds_id}


def choose(league: str, sources: Sequence[Mapping[str, Any]], line: Optional[Mapping[str, Any]], *,
           start: Optional[float], now: float, params: Any = None, board: Optional[Mapping[str, Any]] = None,
           neutral: bool = False, retro: bool = False, swapped: bool = False,
           odds_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The brain's view from the first market input there is (see the module docstring): the Odds API board's book
    sources (spread and total from ``board``, else the ESPN line), else the ESPN ``line`` (already in the row's
    orientation) with the other opinions, else the ratings alone; None with none of the three."""
    ok = [s for s in sources if s and s.get("status") == "ok" and _before(s, start)]
    common = {"start": start, "now": now, "params": params, "neutral": neutral, "retro": retro, "swapped": swapped,
              "odds_id": odds_id}
    if any(s.get("kind") in BOOK_KINDS for s in ok):
        lines = board if board and (board.get("spread") is not None or board.get("total") is not None) else line
        found = view(league, "odds_api", ok, line=lines, **common)
        if found:
            return found
    if line:
        source = line_source(line)
        if source:
            found = view(league, "espn_line", [source] + [s for s in ok if s.get("kind") in OPINION_KINDS], line=line,
                         **common)
            if found:
                return found
    ratings = [s for s in ok if s.get("kind") in RATING_KINDS]
    if ratings:
        return view(league, "model", ratings, **common)
    return None


def _name(team: Any) -> str:
    if isinstance(team, Mapping):
        return str(team.get("name") or team.get("short") or team.get("abbr") or "?")
    return str(team or "?")


def game_row(game: Mapping[str, Any], detail: Optional[Mapping[str, Any]], line: Optional[Mapping[str, Any]],
             params: Any, now: float) -> Optional[Dict[str, Any]]:
    """The ai_predictions row for one slate game (``line``: ESPN's, in ESPN's orientation), or None without a market
    input."""
    swapped = bool(game.get("_swapped"))
    payload = choose(game["league"], list((detail or {}).get("sources") or []), oriented(line, swapped),
                     start=game.get("_ts"), now=now, params=params, board=game.get("_lines"),
                     neutral=bool(game.get("_neutral")), swapped=swapped, odds_id=game.get("odds_id"))
    if payload is None:
        return None
    return {"league": game["league"], "event_id": str(game.get("odds_id") or game["espn_id"]),
            "espn_id": str(game["espn_id"]), "home": _name(game.get("home")), "away": _name(game.get("away")),
            "start": game["start"], "payload": payload}


def due(game: Mapping[str, Any], now: float) -> bool:
    """Whether a slate game is logged this round: a team-league game ESPN has, still to start, within LEAD."""
    start = game.get("_ts")
    return (game.get("league") in ESPN_LEAGUES and game.get("status") == "pre" and bool(game.get("espn_id"))
            and bool(game.get("_time_ok", True)) and isinstance(start, (int, float)) and now < start <= now + LEAD)


def log_round(store: Any, games: Iterable[Mapping[str, Any]], details: Mapping[str, Mapping[str, Any]],
              line_of: Callable[[Mapping[str, Any]], Optional[Mapping[str, Any]]], params_of: Callable[[str], Any],
              now: float, memo: Optional[Dict[Tuple[str, str], str]] = None) -> Dict[str, int]:
    """Log (or update) each due game's view (``due``); ``memo`` remembers what each game last got, so an unchanged
    view is not written again. Returns the round's counts (COUNTS). One game's failure costs only that game."""
    counts = dict.fromkeys(COUNTS, 0)
    if store is None or not hasattr(store, "log_ai_prediction"):
        return counts
    for game in games:
        if not due(game, now):
            continue
        try:
            line = line_of(game)
            row = game_row(game, details.get(str(game.get("id"))), line, params_of(game["league"]), now)
            if row is None:
                counts["no_input"] += 1
                continue
            counts[row["payload"]["source"]] += 1
            key = (row["league"], row["espn_id"])
            signature = json.dumps({k: v for k, v in row.items() if k != "payload"} |
                                   {"payload": {k: v for k, v in row["payload"].items() if k != "as_of"}},
                                   sort_keys=True, default=str)
            if memo is not None and memo.get(key) == signature:
                counts["unchanged"] += 1
                continue
            written = store.log_ai_prediction(row, now=now)
            counts["logged" if written else "held"] += 1
            if memo is not None:
                memo[key] = signature
                if len(memo) > 20000:
                    for old in list(memo)[:10000]:
                        memo.pop(old, None)
        except Exception:  # noqa: BLE001 - the record must never cost the slate
            counts["failed"] += 1
            log.warning("Every-game record: logging %s failed", game.get("id"), exc_info=True)
    return counts


# --- grading --------------------------------------------------------------------------------------------------------

def outcome_of(final: Mapping[str, Any], swapped: bool = False) -> Optional[str]:
    """A final score's outcome in the row's orientation (ESPN's home and away turned round when ``swapped``)."""
    home, away = _num(final.get("home")), _num(final.get("away"))
    if home is None or away is None:
        return None
    if swapped:
        home, away = away, home
    return "home" if home > away else "away" if away > home else "draw"


def _scored(p: Any, outcome: Optional[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(p, Mapping) or not p or outcome is None:
        return None
    found = consensus.score({k: v for k, v in p.items() if k in OUTCOMES}, outcome)
    if not found:
        return {"hit": None, "brier": None, "log_loss": None}  # a tie on a two-way market: a push
    best = max((k for k in OUTCOMES if isinstance(p.get(k), (int, float))), key=lambda k: float(p[k]))
    return {"hit": best == outcome, "brier": round(found["brier"], 6), "log_loss": round(found["log_loss"], 6)}


def grade(payload: Mapping[str, Any], final: Mapping[str, Any]) -> Dict[str, Any]:
    """One finished game's score: {"outcome", "brain": {"hit", "brier", "log_loss"}, "market": ... | None}."""
    outcome = outcome_of(final, bool(payload.get("swapped")))
    return {"outcome": outcome, "brain": _scored(payload.get("p"), outcome),
            "market": _scored(payload.get("market_p"), outcome) if payload.get("market_p") else None}


def still_on(espn: Any, row: Mapping[str, Any], now: float) -> bool:
    """Whether ESPN's boards (as the client last read them; nothing is fetched) say the game hasn't finished yet,
    within STARTED_LIVE of its start: the grader then asks for no summary this round."""
    read = getattr(espn, "event_state", None)
    start = _ts(row.get("start"))
    if not callable(read) or not row.get("espn_id") or start is None or now - start > STARTED_LIVE:
        return False
    try:
        state = (read(row["league"], row["espn_id"]) or {}).get("state")
    except Exception:  # noqa: BLE001
        return False
    return state in ("pre", "live")


# --- the walk-forward back-test -------------------------------------------------------------------------------------

def params_at(league: str, store: Any, ts: float) -> "registry.Params":
    """The parameters live for the league's sport at ``ts`` (a version promoted later never counts: walk-forward);
    the defaults when none was live then or the store can't be read."""
    sport = registry.sport_key(league)
    row = None
    try:
        with closing(store._connect()) as db:
            row = db.execute(
                "SELECT version, params_json, source, effective_from FROM brain_params WHERE sport = ? AND status IN "
                "('champion', 'retired', 'rolled_back') AND effective_from IS NOT NULL AND effective_from <= ? AND "
                "(ended_at IS NULL OR ended_at > ?) ORDER BY effective_from DESC, version DESC LIMIT 1",
                (sport, float(ts), float(ts))).fetchone()
    except Exception:  # noqa: BLE001 - no learning tables: the defaults
        row = None
    if row is not None:
        try:
            stored = {k: v for k, v in (json.loads(row[1]) or {}).items()
                      if k in registry.KNOBS and registry.KNOBS[k].applies(sport)}
            return registry.Params(sport, stored, version=int(row[0]), source=row[2] or "refit",
                                   effective_from=row[3])
        except Exception:  # noqa: BLE001
            log.warning("Every-game back-test: version %s for %s unusable", row[0], sport)
    return registry.Params(sport)


def _summary_of(raw: Any, cache: Any, league: str, espn_id: str) -> Optional[Mapping[str, Any]]:
    """The game's summary as already held: raw.db's newest one (the capture's final), else the lean cache's copy."""
    if raw is not None:
        try:
            found = raw.latest(f"espn:{espn_id}", "espn_site", "summary")
            data = found.json() if found is not None else None
            if isinstance(data, Mapping):
                return data
        except Exception:  # noqa: BLE001 - an unreadable payload: try the cache
            log.warning("Every-game back-test: raw summary of %s unreadable", espn_id, exc_info=True)
    peek = getattr(cache, "peek", None)
    if callable(peek):
        try:
            held = peek(("espn", "summary", league, str(espn_id)))
        except Exception:  # noqa: BLE001
            held = None
        if held and isinstance(held[1], Mapping):
            return held[1]
    return None


def _games(store: Any, raw: Any, since: float, until: float) -> List[Tuple[str, str, float]]:
    """(league, ESPN id, start) of the team-league games that started in [since, until): raw.db's capture jobs and
    the consensus log's games, each once."""
    found: Dict[Tuple[str, str], float] = {}
    if raw is not None:
        try:
            for job in raw.jobs(since=since, until=until, prefix="espn:"):
                if job.league in ESPN_LEAGUES and job.espn_id:
                    found.setdefault((job.league, str(job.espn_id)), float(job.start))
        except Exception:  # noqa: BLE001
            log.warning("Every-game back-test: raw.db's jobs unreadable", exc_info=True)
    rows = getattr(store, "consensus_rows", None)
    if callable(rows):
        try:
            for row in rows(started_before=_iso(until), limit=10000):
                start = _ts(row.get("start"))
                if row.get("league") in ESPN_LEAGUES and row.get("espn_id") and start is not None and \
                        since <= start < until:
                    found.setdefault((row["league"], str(row["espn_id"])), start)
        except Exception:  # noqa: BLE001
            log.warning("Every-game back-test: the consensus log unreadable", exc_info=True)
    return sorted(((league, espn_id, start) for (league, espn_id), start in found.items()), key=lambda g: (g[2], g))


def _logged(store: Any, since: float) -> set:
    """(league, ESPN id) of every row the record already has for games from ``since`` on (one read)."""
    try:
        with closing(store._connect()) as db:
            return {(league, str(espn_id)) for league, espn_id in db.execute(
                "SELECT league, espn_id FROM ai_predictions WHERE espn_id IS NOT NULL AND start >= ?",
                (_iso(since - 86400.0),))}
    except Exception:  # noqa: BLE001 - the store's own checks still keep one row per game
        return set()


def retro_row(league: str, espn_id: str, start: float, summary: Mapping[str, Any],
              logged: Optional[Mapping[str, Any]], params: Any) -> Optional[Dict[str, Any]]:
    """The back-test's row for one game, from only what was known before its start: the summary's pickcenter (by
    the lead rule) and the consensus log's pre-start opinions (``logged``), at ``params``. Never the result."""
    parsed = espn_hist.parse_summary(summary, league)
    swapped = bool(logged and logged.get("swapped"))
    line = pick_line(parse_lines(summary.get("pickcenter")), start=start, three_way=league.startswith("soccer_"))
    sources = [s for s in ((logged or {}).get("sources") or []) if isinstance(s, Mapping)]
    payload = choose(league, sources, oriented(line, swapped), start=start, now=start - LEAD, params=params,
                     neutral=bool(parsed.get("neutral")), retro=True, swapped=swapped,
                     odds_id=(logged or {}).get("event_id"))
    if payload is None:
        return None
    home, away = parsed.get("home") or {}, parsed.get("away") or {}
    names = (_name(home), _name(away)) if not swapped else (_name(away), _name(home))
    if logged:
        names = (str(logged.get("home") or names[0]), str(logged.get("away") or names[1]))
    return {"league": league, "event_id": str((logged or {}).get("event_id") or espn_id), "espn_id": str(espn_id),
            "home": names[0], "away": names[1], "start": _iso(start), "payload": payload}


def backtest(store: Any, raw: Any = None, cache: Any = None, now: Optional[float] = None,
             days: int = RETRO_DAYS) -> Dict[str, Any]:
    """Back-test the last ``days`` days walk-forward (see the module docstring) and grade each new row at once.
    Returns {"games", "logged", "graded", "void", "held", "no_summary", "not_final", "no_input", "failed",
    "by_league": {league: {"logged", "graded"}}}. Fetches nothing."""
    now = time.time() if now is None else float(now)
    report: Dict[str, Any] = {k: 0 for k in ("games", "logged", "graded", "void", "held", "no_summary", "not_final",
                                             "no_input", "failed")}
    report["by_league"] = {}
    if store is None or not hasattr(store, "log_ai_prediction"):
        return report
    since = now - days * 86400.0
    logged = _logged(store, since)
    for league, espn_id, start in _games(store, raw, since, now - SETTLED):
        report["games"] += 1
        if (league, espn_id) in logged:
            report["held"] += 1  # already in the record (live, or an earlier run): its summary isn't read again
            continue
        try:
            summary = _summary_of(raw, cache, league, espn_id)
            if summary is None:
                report["no_summary"] += 1
                continue
            parsed = espn_hist.parse_summary(summary, league)
            if parsed.get("state") not in ("final", "void"):
                report["not_final"] += 1
                continue
            finder = getattr(store, "consensus_row", None)
            opinions = finder(league, espn_id=espn_id) if callable(finder) else None
            row = retro_row(league, espn_id, start, summary, opinions, params_at(league, store, start - LEAD))
            if row is None:
                report["no_input"] += 1
                continue
            if not store.log_ai_prediction(row, now=now):
                report["held"] += 1
                continue
            report["logged"] += 1
            per = report["by_league"].setdefault(league, {"logged": 0, "graded": 0})
            per["logged"] += 1
            if parsed["state"] == "void":
                store.grade_ai_prediction(league, row["event_id"], "void", {"state": "void", "espn_id": espn_id,
                                                                             "status": parsed.get("status")})
                report["void"] += 1
                continue
            final = {"state": "final", "home": (parsed.get("home") or {}).get("score"),
                     "away": (parsed.get("away") or {}).get("score"), "espn_id": espn_id}
            store.grade_ai_prediction(league, row["event_id"], "graded", {**final, "brain": grade(row["payload"], final)})
            report["graded"] += 1
            per["graded"] += 1
        except Exception:  # noqa: BLE001 - one game's odd summary costs only that game
            report["failed"] += 1
            log.warning("Every-game back-test: %s %s failed", league, espn_id, exc_info=True)
    return report


# --- the record -----------------------------------------------------------------------------------------------------

def _empty() -> Dict[str, Any]:
    return {"predicted": 0, "graded": 0, "pending": 0, "void": 0, "scored": 0, "hits": 0, "hit_rate": None,
            "brain_brier": None, "brain_log_loss": None, "market_n": 0, "market_brier": None,
            "market_log_loss": None, "market_brain_brier": None}


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(math.fsum(values) / len(values), 4) if values else None


def _priced_score(payload: Mapping[str, Any], result: Mapping[str, Any]) -> Dict[str, Any]:
    """A priced NBA prediction scored the same way (its home-win chance and the market's)."""
    outcome = outcome_of(result)
    out: Dict[str, Any] = {"brain": None, "market": None}
    for key, name in (("home_win", "brain"), ("market_home_win", "market")):
        chance = _num(payload.get(key))
        if chance is not None and 0 <= chance <= 1:
            out[name] = _scored({"home": chance, "away": 1 - chance}, outcome)
    return out


def summary(store: Any, now: Optional[float] = None) -> Dict[str, Any]:
    """Per sport: {"sport", "label", "live", "retro", "priced"}, each block {"predicted", "graded", "pending", "void",
    "scored", "hits", "hit_rate", "brain_brier", "brain_log_loss", "market_n", "market_brier", "market_log_loss",
    "market_brain_brier" (the brain's Brier on the games the market line also priced)}; plus "games" (every row)
    and a "note". Read from ai_predictions only (kept until a row is logged or graded)."""
    from .loop import SPORT_LABELS
    out: Dict[str, Any] = {"sports": [], "games": 0}
    if store is None or not hasattr(store, "_connect"):
        return out
    with closing(store._connect()) as db:
        mark = db.execute("SELECT COUNT(*), MAX(logged_at), MAX(graded_at) FROM ai_predictions").fetchone()
        key = getattr(store, "path", None) or id(store)
        with _summary_lock:
            cached = _summary_cache.get(key)
            if cached is not None and cached[0] == tuple(mark):
                return cached[1]
        rows = db.execute("SELECT league, status, payload, result FROM ai_predictions").fetchall()
    sports: Dict[str, Dict[str, Any]] = {}
    scores: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for league, status, payload_text, result_text in rows:
        try:
            payload = json.loads(payload_text) if payload_text else {}
            result = json.loads(result_text) if result_text else {}
        except ValueError:
            continue
        sport = registry.sport_key(league)
        block_name = ("retro" if payload.get("retro") else "live") if is_every_game(payload) else "priced"
        entry = sports.setdefault(sport, {"sport": sport, "label": SPORT_LABELS.get(sport, sport),
                                          **{name: _empty() for name in BLOCKS}})
        block = entry[block_name]
        block["predicted"] += 1
        if status == "pending":
            block["pending"] += 1
            continue
        if status != "graded":
            block["void"] += 1
            continue
        block["graded"] += 1
        scored = (result.get("brain") or {}) if block_name != "priced" else {}
        brain, market = (scored.get("brain"), scored.get("market")) if block_name != "priced" else \
            tuple(_priced_score(payload, result).values())
        bucket = scores.setdefault((sport, block_name), {"brier": [], "ll": [], "m_brier": [], "m_ll": [],
                                                         "mb_brier": []})
        if brain and brain.get("brier") is not None:
            block["scored"] += 1
            block["hits"] += 1 if brain.get("hit") else 0
            bucket["brier"].append(float(brain["brier"]))
            bucket["ll"].append(float(brain["log_loss"]))
            if market and market.get("brier") is not None:
                block["market_n"] += 1
                bucket["m_brier"].append(float(market["brier"]))
                bucket["m_ll"].append(float(market["log_loss"]))
                bucket["mb_brier"].append(float(brain["brier"]))
    for (sport, name), bucket in scores.items():
        block = sports[sport][name]
        block["hit_rate"] = round(block["hits"] / block["scored"], 4) if block["scored"] else None
        block.update({"brain_brier": _mean(bucket["brier"]), "brain_log_loss": _mean(bucket["ll"]),
                      "market_brier": _mean(bucket["m_brier"]), "market_log_loss": _mean(bucket["m_ll"]),
                      "market_brain_brier": _mean(bucket["mb_brier"])})
    order = list(registry.SPORTS)
    out["sports"] = sorted(sports.values(), key=lambda s: order.index(s["sport"]) if s["sport"] in order else 99)
    out["games"] = len(rows)
    out["note"] = ("Every game on ESPN's boards gets the brain's chance an hour before the start, scored after the "
                   "final: learning records, not picks. Priced picks, the app's own priced games, are scored apart. "
                   "Brier score: lower is better (0.25 is a coin flip on a two-way game).")
    with _summary_lock:
        if len(_summary_cache) > 20:
            _summary_cache.clear()
        _summary_cache[key] = (tuple(mark), out)
    return out


def _pct(value: Optional[float]) -> str:
    return "–" if value is None else f"{value * 100:.0f}%"


def block_text(name: str, block: Mapping[str, Any]) -> Optional[str]:
    """One line of /brain for a block ("live", "retro" or "priced"); None when it has no game."""
    if not block.get("predicted"):
        return None
    label = {"live": "Every game, live", "retro": "Every game, last 14 days (back-tested)",
             "priced": "Priced picks"}[name]
    parts = [f"{label}: {block['predicted']} predicted, {block['graded']} graded, {block['pending']} pending"]
    if block.get("scored"):
        parts.append(f"hit rate {_pct(block.get('hit_rate'))}")
        if block.get("market_n"):
            n = block["market_n"]
            parts.append(f"Brier {block['market_brain_brier']:.3f} vs the line's {block['market_brier']:.3f} "
                         f"({n} game{'' if n == 1 else 's'})")
        else:
            parts.append(f"Brier {block['brain_brier']:.3f}")
    return "; ".join(parts) + "."


def brain_lines(store: Any, limit: int = 1800) -> List[str]:
    """The bot's /brain lines for the record: per sport its live, back-tested and priced lines, within ``limit``
    characters (the message stays under Telegram's 4096)."""
    lines: List[str] = []
    used = 0
    for sport in summary(store).get("sports") or []:
        parts = [text for name in BLOCKS if (text := block_text(name, sport[name]))]
        if not parts:
            continue
        text = f"{sport['label']}:\n" + "\n".join(f"• {part}" for part in parts)
        if used + len(text) > limit:
            lines.append("More sports on the Brain progress page.")
            break
        lines.append(text)
        used += len(text) + 2
    return lines


__all__ = ["LEAD", "RETRO_DAYS", "SOURCES", "is_every_game", "parse_lines", "pick_line", "devig", "view", "choose",
           "game_row", "log_round", "grade", "still_on", "params_at", "backtest", "summary", "brain_lines"]
