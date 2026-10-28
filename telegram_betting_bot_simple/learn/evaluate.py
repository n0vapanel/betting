"""Walk-forward, point-in-time evaluation of the brain (method.md sections 2-3).

Inputs, by trust:

- **live logs** (trusted, frozen before the start): ``consensus_log`` (every source's chance and the blend as served,
  with the ``params_version`` it was blended with), ``context_log`` (the total line and the over before and after the
  adjustment) and ``ai_predictions`` (the NBA model). ``load_corpus`` reads them into plain records;
- **history replays**, only for what closing lines can rebuild: the goals model from the 1X2 and over 2.5, home
  edge and the margin/total spreads, and the corner and card count models (``MatchRecord``s built by the caller or
  by ``match_records`` from the history tables).

Targets and scores: moneyline/1X2 by log loss with Brier as a guard; totals by the log loss of the over; goals,
corners and cards by the count log score, discrete CRPS and a PIT histogram; calibration by the CORP reliability
curve with its consistency band, and the calibration slope and intercept with 95% ranges. Baselines: the sharp
market on the same games (the books' consensus where no sharp price was logged), the base rate, and for counts the
league's climatology (a negative binomial with no game inputs, fitted on the training window).

Replays go through the real engine functions (``consensus.blend``, ``context.adjusted_over``,
``soccer.rates_from_odds``/``grid``) with the candidate's values, so a candidate is scored by exactly the code
that would serve it. The leakage checklist (``leakage_problems``) is enforced by the walk-forward folds and pinned
by tests: nothing dated at or after the start, folds grouped by day, no shuffling, and no AI forecast or proposal
judged on games before it was made.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..engine import consensus, context, soccer
from . import registry, stats

MIN_TRAIN = 300  # graded games per sport before any refit
MIN_EVAL = 300  # out-of-sample games per sport for a gate decision
TRAILING = 300  # games in the trailing calibration window
HORIZONS: Tuple[Tuple[str, float], ...] = (("under 15 min", 900), ("15-60 min", 3600), ("1-6 h", 6 * 3600),
                                           ("6-24 h", 86400), ("over 24 h", math.inf))
EVEN = 0.5


# --- records -------------------------------------------------------------------------------------------------------

def _ts(value: Any) -> Optional[float]:
    return consensus._ts(value)


def _json(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _columns(db: sqlite3.Connection, table: str) -> List[str]:
    return [row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()]


def _neutral(blend: Mapping[str, Any], league: str) -> bool:
    """A low-confidence blend shrunk toward an even split where the sport has a home base rate was a neutral site."""
    base = blend.get("base")
    if not isinstance(base, Mapping) or not base:
        return False
    outcomes = tuple(base)
    even = all(abs(float(v) - 1 / len(outcomes)) < 1e-12 for v in base.values())
    if not even:
        return False
    rate = consensus.base_rate(league, outcomes)
    return any(abs(rate[o] - 1 / len(outcomes)) > 1e-12 for o in outcomes)


def _snapshot(row: Mapping[str, Any]) -> bool:
    """A fixed-horizon snapshot row (kind "t24"), not the last-before-start row the scores use."""
    kind = row.get("kind")
    return bool(kind) and kind not in ("last", "close") or str(row.get("game_key") or "").endswith("#t24")


def consensus_records(db: sqlite3.Connection, *, since: Optional[float] = None, until: Optional[float] = None,
                      status: Optional[str] = "graded", snapshots: bool = False) -> List[Dict[str, Any]]:
    """Graded (by default) consensus_log rows as records, oldest start first: league, sport (registry key),
    game_key, start and logged_at (epoch), sources, blend, outcome, result, params_version (0 when the column is
    missing), neutral, day and week. Snapshot rows (kind "t24") only with ``snapshots``."""
    columns = _columns(db, "consensus_log")
    if not columns:
        return []
    wanted = [c for c in ("league", "game_key", "start", "logged_at", "sources", "blend", "status", "outcome", "result",
                          "params_version", "kind") if c in columns]
    sql = f"SELECT {', '.join(wanted)} FROM consensus_log"
    args: List[Any] = []
    if status is not None:
        sql += " WHERE status = ?"
        args.append(status)
    out = []
    for values in db.execute(sql, args).fetchall():
        row = dict(zip(wanted, values))
        if _snapshot(row) != snapshots:
            continue
        start = _ts(row["start"])
        if start is None or (since is not None and start < since) or (until is not None and start >= until):
            continue
        blend = _json(row["blend"]) or {}
        out.append({
            "league": row["league"], "sport": registry.sport_key(row["league"]), "game_key": row["game_key"],
            "start": start, "logged_at": float(row["logged_at"]), "sources": _json(row["sources"]) or [],
            "blend": blend, "outcome": row.get("outcome"), "result": _json(row.get("result")) or {},
            "params_version": int(row.get("params_version") or 0), "neutral": _neutral(blend, row["league"]),
            "day": stats.day_of(start), "week": stats.week_of(start)})
    out.sort(key=lambda r: (r["start"], r["league"], r["game_key"]))
    return out


def context_records(db: sqlite3.Connection, *, since: Optional[float] = None, until: Optional[float] = None,
                    status: Optional[str] = "graded") -> List[Dict[str, Any]]:
    """Graded context_log rows: league, sport, game_key, start, logged_at, line, p_before, p_after, applied,
    adjustment, y (1 over, 0 under), total (the final total), params_version, day, week."""
    columns = _columns(db, "context_log")
    if not columns:
        return []
    wanted = [c for c in ("league", "game_key", "start", "logged_at", "line", "p_before", "p_after", "applied",
                          "adjustment", "status", "outcome", "result", "params_version") if c in columns]
    sql = f"SELECT {', '.join(wanted)} FROM context_log"
    args: List[Any] = []
    if status is not None:
        sql += " WHERE status = ?"
        args.append(status)
    out = []
    for values in db.execute(sql, args).fetchall():
        row = dict(zip(wanted, values))
        start = _ts(row["start"])
        if start is None or (since is not None and start < since) or (until is not None and start >= until):
            continue
        if status == "graded" and row.get("outcome") not in ("over", "under"):
            continue
        result = _json(row.get("result")) or {}
        out.append({
            "league": row["league"], "sport": registry.sport_key(row["league"]), "game_key": row["game_key"],
            "start": start, "logged_at": float(row["logged_at"]), "line": float(row["line"]),
            "p_before": float(row["p_before"]), "p_after": float(row["p_after"]), "applied": float(row["applied"] or 0),
            "adjustment": _json(row.get("adjustment")), "y": 1 if row.get("outcome") == "over" else 0,
            "total": result.get("total"), "params_version": int(row.get("params_version") or 0),
            "day": stats.day_of(start), "week": stats.week_of(start)})
    out.sort(key=lambda r: (r["start"], r["league"], r["game_key"]))
    return out


def ai_records(db: sqlite3.Connection, *, since: Optional[float] = None,
               until: Optional[float] = None) -> List[Dict[str, Any]]:
    """Graded ai_predictions (the NBA model) with a final score: league, start, logged_at, payload, y (home won),
    margin_final, total_final, day, week."""
    if not _columns(db, "ai_predictions"):
        return []
    out = []
    for league, start_text, logged_at, payload, result in db.execute(
            "SELECT league, start, logged_at, payload, result FROM ai_predictions WHERE status = 'graded'"):
        start = _ts(start_text)
        payload, result = _json(payload) or {}, _json(result) or {}
        home, away = result.get("home"), result.get("away")
        if start is None or home is None or away is None or home == away:
            continue
        if (since is not None and start < since) or (until is not None and start >= until):
            continue
        out.append({"league": league, "sport": registry.sport_key(league), "start": start,
                    "logged_at": float(logged_at), "payload": payload, "y": 1 if float(home) > float(away) else 0,
                    "margin_final": float(home) - float(away), "total_final": float(home) + float(away),
                    "day": stats.day_of(start), "week": stats.week_of(start)})
    out.sort(key=lambda r: r["start"])
    return out


@dataclass
class MatchRecord:
    """One finished soccer match as of kickoff: the closing margin-free 1X2 and over 2.5, the 90-minute score and the
    match stats (``stats`` {"corners": (home, away), "cards": (home, away)} when known)."""

    league: str
    game_key: str
    start: float
    p_home: float
    p_draw: float
    p_away: float
    p_over: float
    home_goals: int
    away_goals: int
    stats: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    lines_as_of: Optional[float] = None  # the newest price time used (must be before the start)

    @property
    def day(self) -> str:
        return stats.day_of(self.start)

    @property
    def week(self) -> str:
        return stats.week_of(self.start)


def match_records(db: sqlite3.Connection, *, since: Optional[float] = None,
                  until: Optional[float] = None) -> List[MatchRecord]:
    """Soccer matches from the history tables (``hist_games``, ``hist_lines``, ``hist_stats``) with a final score and a
    pre-start closing 1X2 and over 2.5 from one book (the sharpest present), lines dated before the start only."""
    if not _columns(db, "hist_games") or not _columns(db, "hist_lines"):
        return []
    games = {}
    for league, key, start_text, hs, as_, status in db.execute(
            "SELECT league, game_key, start, home_score, away_score, status FROM hist_games WHERE league LIKE "
            "'soccer%' AND home_score IS NOT NULL AND away_score IS NOT NULL"):
        start = _ts(start_text)
        if start is None or (since is not None and start < since) or (until is not None and start >= until):
            continue
        if status not in (None, "", "final", "ft", "FT", "post"):
            continue
        games[key] = {"league": league, "start": start, "home": int(hs), "away": int(as_)}
    if not games:
        return []
    lines: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    for key, kind, book, market, side, point, price, as_of in db.execute(
            "SELECT game_key, kind, book, market, side, point, price, as_of FROM hist_lines WHERE kind = 'close'"):
        game = games.get(key)
        when = _ts(as_of)
        if game is None or price is None or float(price) <= 1 or when is None or when >= game["start"]:
            continue
        name = str(market or "").lower()
        if name in ("h2h", "1x2", "moneyline") and side in ("home", "draw", "away"):
            lines.setdefault(key, {}).setdefault(str(book), {})[f"1x2:{side}"] = (float(price), when)
        elif name in ("totals", "total", "ou") and point is not None and abs(float(point) - 2.5) < 1e-9 \
                and side in ("over", "under"):
            lines.setdefault(key, {}).setdefault(str(book), {})[f"ou:{side}"] = (float(price), when)
    counts: Dict[str, Dict[str, List[Optional[float]]]] = {}
    if _columns(db, "hist_stats"):
        # One source per game (the one with the most of these stats), so two sources never add up
        by_source: Dict[str, Dict[str, Dict[Tuple[str, str], float]]] = {}
        has_source = "source" in _columns(db, "hist_stats")
        sql = "SELECT game_key, team_side, stat, value" + (", source" if has_source else ", ''") + " FROM hist_stats"
        for key, side, stat, value, source in db.execute(sql):
            if key in games and value is not None and side in ("home", "away") and stat in ("corners", "yellow",
                                                                                                "red"):
                by_source.setdefault(key, {}).setdefault(str(source), {})[(side, str(stat))] = float(value)
        for key, sources in by_source.items():
            chosen = max(sources.values(), key=len)
            for target, names in (("corners", ("corners",)), ("cards", ("yellow", "red"))):
                if not any((side, name) in chosen for side in ("home", "away") for name in names):
                    continue
                if target == "corners" and not all(("home" if i == 0 else "away", "corners") in chosen
                                                   for i in (0, 1)):
                    continue
                if target == "cards" and not all((side, "yellow") in chosen for side in ("home", "away")):
                    continue
                counts.setdefault(key, {})[target] = [
                    sum(chosen.get((side, name), 0.0) for name in names) for side in ("home", "away")]
    preference = ("pinnacle", "betfair_ex", "betfair", "avg", "average", "max", "bet365")
    out = []
    for key, game in games.items():
        books = lines.get(key) or {}
        full = [b for b, v in books.items() if all(k in v for k in ("1x2:home", "1x2:draw", "1x2:away", "ou:over",
                                                                     "ou:under"))]
        if not full:
            continue
        book = sorted(full, key=lambda b: next((i for i, p in enumerate(preference) if p in b.lower()), 99))[0]
        v = books[book]
        inv = {k: 1 / v[k][0] for k in v}
        x = inv["1x2:home"] + inv["1x2:draw"] + inv["1x2:away"]
        o = inv["ou:over"] + inv["ou:under"]
        made = {t: (int(h), int(a)) for t, (h, a) in (counts.get(key) or {}).items() if h is not None and a is not None}
        out.append(MatchRecord(game["league"], key, game["start"], inv["1x2:home"] / x, inv["1x2:draw"] / x,
                               inv["1x2:away"] / x, inv["ou:over"] / o, game["home"], game["away"], made,
                               max(w for _, w in v.values())))
    out.sort(key=lambda m: (m.start, m.game_key))
    return out


def line_records(db: sqlite3.Connection, *, since: Optional[float] = None,
                 until: Optional[float] = None) -> List[Dict[str, Any]]:
    """Finished non-soccer games from the history tables with a closing spread and/or total dated before the start:
    league, sport, game_key, start, logged_at (the price time), outcome (home/away), margin_final, market_margin
    (minus the home handicap), total_final, line (the closing total). For the margin and total SDs and home rates."""
    if not _columns(db, "hist_games") or not _columns(db, "hist_lines"):
        return []
    games = {}
    for league, key, start_text, hs, as_ in db.execute(
            "SELECT league, game_key, start, home_score, away_score FROM hist_games WHERE league NOT LIKE 'soccer%' "
            "AND status = 'final' AND home_score IS NOT NULL AND away_score IS NOT NULL"):
        start = _ts(start_text)
        if start is None or (since is not None and start < since) or (until is not None and start >= until):
            continue
        games[key] = {"league": league, "start": start, "home": float(hs), "away": float(as_)}
    preference = ("pinnacle", "betfair_ex", "avg", "average", "median", "consensus", "draftkings", "espn_bet",
                  "nflverse")
    best: Dict[Tuple[str, str], Tuple[int, float, float]] = {}
    for key, market, side, point, as_of, book in db.execute(
            "SELECT game_key, market, side, point, as_of, book FROM hist_lines WHERE kind = 'close' AND point IS NOT "
            "NULL AND ((market = 'spreads' AND side = 'home') OR (market = 'totals' AND side = 'over'))"):
        game = games.get(key)
        when = _ts(as_of)
        if game is None or when is None or when >= game["start"]:
            continue
        rank = next((i for i, p in enumerate(preference) if p in str(book).lower()), len(preference))
        slot = (key, market)
        if slot not in best or rank < best[slot][0]:
            best[slot] = (rank, float(point), when)
    out = []
    for key, game in games.items():
        spread, total = best.get((key, "spreads")), best.get((key, "totals"))
        if spread is None and total is None:
            continue
        margin = game["home"] - game["away"]
        out.append({"league": game["league"], "sport": registry.sport_key(game["league"]), "game_key": key,
                    "start": game["start"], "logged_at": max(x[2] for x in (spread, total) if x is not None),
                    "outcome": "home" if margin > 0 else "away" if margin < 0 else "tie", "margin_final": margin,
                    "market_margin": -spread[1] if spread else None, "total_final": game["home"] + game["away"],
                    "line": total[1] if total else None, "day": stats.day_of(game["start"]),
                    "week": stats.week_of(game["start"])})
    out.sort(key=lambda r: (r["start"], r["game_key"]))
    return out


@dataclass
class Corpus:
    """Everything the evaluator and the tuner read, loaded once: records by sport, oldest first."""

    consensus: List[Dict[str, Any]] = field(default_factory=list)
    context: List[Dict[str, Any]] = field(default_factory=list)
    ai: List[Dict[str, Any]] = field(default_factory=list)
    matches: List[MatchRecord] = field(default_factory=list)
    lines: List[Dict[str, Any]] = field(default_factory=list)  # history: closing spreads/totals with results

    def sport(self, name: str, kind: str = "consensus") -> List[Any]:
        rows = getattr(self, kind)
        if kind == "matches":
            return list(rows) if name == "soccer" else []
        return [r for r in rows if r["sport"] == name]

    def sports(self) -> List[str]:
        found = {r["sport"] for rows in (self.consensus, self.context, self.ai, self.lines) for r in rows}
        if self.matches:
            found.add("soccer")
        return [s for s in registry.SPORTS if s in found]


def load_corpus(db: sqlite3.Connection, *, until: Optional[float] = None,
                matches: Optional[Sequence[MatchRecord]] = None) -> Corpus:
    """The graded live logs (and the history's soccer matches unless ``matches`` is given) of games starting before
    ``until``."""
    return Corpus(consensus_records(db, until=until), context_records(db, until=until), ai_records(db, until=until),
                  list(matches) if matches is not None else match_records(db, until=until),
                  line_records(db, until=until))


# --- the leakage checklist -----------------------------------------------------------------------------------------

def leakage_problems(train: Sequence[Any] = (), test: Sequence[Any] = (), *, ai: Sequence[Mapping[str, Any]] = (),
                     proposals: Sequence[Mapping[str, Any]] = ()) -> List[str]:
    """Plain-words problems with a train/test split (empty when clean):

    1. nothing dated at or after the start (a record's ``logged_at`` or ``lines_as_of``);
    2. folds grouped by day: no day on both sides;
    3. no shuffle: every training game starts before every test game, and each side is in time order;
    4. no AI forecast graded on a game that started before the forecast was asked (``ai``: {"start", "asked_at"}),
       and no AI proposal judged on a game before it was made (``proposals``: {"created_at", "judged_on": starts}).
    """
    problems = []

    def get(r: Any, name: str) -> Any:
        return r.get(name) if isinstance(r, Mapping) else getattr(r, name, None)

    for side, rows in (("train", train), ("test", test)):
        for r in rows:
            start = get(r, "start")
            for stamp in ("logged_at", "lines_as_of", "as_of"):
                value = get(r, stamp)
                value = _ts(value) if value is not None else None
                if value is not None and start is not None and value >= start:
                    problems.append(f"{side}: {get(r, 'game_key') or '?'} has {stamp} at or after its start")
        starts = [get(r, "start") for r in rows]
        if any(b < a for a, b in zip(starts, starts[1:])):
            problems.append(f"{side}: games out of time order (shuffled)")
    if train and test:
        train_days = {stats.day_of(get(r, "start")) for r in train}
        test_days = {stats.day_of(get(r, "start")) for r in test}
        shared = sorted(train_days & test_days)
        if shared:
            problems.append(f"day split across train and test: {', '.join(shared[:3])}")
        if max(get(r, "start") for r in train) >= min(get(r, "start") for r in test):
            problems.append("a training game starts after a test game")
    for row in ai:
        if row.get("asked_at") is not None and row.get("start") is not None and row["start"] <= row["asked_at"]:
            problems.append(f"AI forecast on a game that started before it was asked ({row.get('game_key', '?')})")
    for proposal in proposals:
        made = proposal.get("created_at")
        early = [s for s in proposal.get("judged_on", ()) if made is not None and s <= made]
        if early:
            problems.append(f"AI proposal {proposal.get('id', '?')} judged on {len(early)} game(s) before it was made")
    return problems


def weekly_folds(rows: Sequence[Any], *, min_train: int = MIN_TRAIN, weeks: Optional[int] = None,
                 start: Callable[[Any], float] = lambda r: r["start"] if isinstance(r, Mapping) else r.start
                 ) -> Iterator[Tuple[str, List[Any], List[Any]]]:
    """Rolling origin: for each calendar week (Monday to Sunday, UTC) with at least ``min_train`` games before it,
    (week, every game before the week, the week's games). Only the last ``weeks`` folds when given. Rows must be in
    time order; days never straddle a fold (a week boundary is a day boundary)."""
    ordered = sorted(rows, key=start)
    by_week: Dict[str, List[Any]] = {}
    for r in ordered:
        by_week.setdefault(stats.week_of(start(r)), []).append(r)
    keys = sorted(by_week, key=lambda w: stats.week_bounds(w)[0])
    folds = []
    seen = 0
    for week in keys:
        if seen >= min_train:
            folds.append(week)
        seen += len(by_week[week])
    if weeks is not None:
        folds = folds[-weeks:]
    for week in folds:
        begin = stats.week_bounds(week)[0]
        yield week, [r for r in ordered if start(r) < begin], by_week[week]


# --- scores of one game --------------------------------------------------------------------------------------------

def market_source(sources: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """The benchmark on a game: the ok sharp source, else the ok US-books consensus."""
    for kind in ("sharp", "us_books"):
        for source in sources:
            if isinstance(source, Mapping) and source.get("kind") == kind and source.get("status") == "ok":
                return source
    return None


def _scored(p: Any, outcome: Optional[str]) -> Optional[Dict[str, float]]:
    return consensus.score(p, outcome) if isinstance(p, Mapping) and outcome else None


def game_scores(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The brain's (as served), the market's and the base rate's log loss and Brier on one graded game, plus the
    brain's chance of the home side and whether it happened."""
    p = (row.get("blend") or {}).get("p")
    brain = _scored(p, row.get("outcome"))
    if brain is None:
        return None
    market_src = market_source(row.get("sources") or [])
    market = _scored(market_src.get("p"), row["outcome"]) if market_src else None
    base = consensus.base_rate(row["league"], tuple(p), row.get("neutral", False))
    baseline = _scored(base, row["outcome"])
    return {"brain_ll": brain["log_loss"], "brain_brier": brain["brier"],
            "market_ll": market["log_loss"] if market else None, "market_brier": market["brier"] if market else None,
            "baseline_ll": baseline["log_loss"] if baseline else None,
            "baseline_brier": baseline["brier"] if baseline else None,
            "p_home": float(p["home"]) / math.fsum(float(v) for v in p.values()), "home_won": row["outcome"] == "home",
            "market_kind": market_src.get("kind") if market_src else None}


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    found = [v for v in values if v is not None]
    return math.fsum(found) / len(found) if found else None


def _horizon(seconds: float) -> str:
    for label, limit in HORIZONS:
        if seconds < limit:
            return label
    return HORIZONS[-1][0]


def summarise_results(rows: Sequence[Mapping[str, Any]], *, trailing: Optional[Sequence[Mapping[str, Any]]] = None,
                      reps: int = 1000, seed: int = 0, curve: bool = True) -> Dict[str, Any]:
    """A brain_scores row's numbers for the moneyline/1X2 target over ``rows``: means of the brain's, market's and
    base rate's log loss and Brier; the gap (brain minus market on the games the market priced) with its 95% range
    (``stats.t_interval``: a week has only 2-7 game days, too few for a bootstrap over days); calibration slope and intercept (home side, one-vs-rest) and the CORP curve over ``trailing`` (default
    ``rows``); per-version and per-horizon counts in ``details``."""
    scored = [(r, s) for r in rows if (s := game_scores(r)) is not None]
    n = len(scored)
    out: Dict[str, Any] = {"n": n, "brain_ll": None, "market_ll": None, "baseline_ll": None, "brier": None,
                           "market_brier": None, "baseline_brier": None, "gap": None, "gap_lo": None, "gap_hi": None,
                           "n_market": 0, "details": {}}
    if not n:
        return out
    out["brain_ll"] = _mean(s["brain_ll"] for _, s in scored)
    out["brier"] = _mean(s["brain_brier"] for _, s in scored)
    out["baseline_ll"] = _mean(s["baseline_ll"] for _, s in scored)
    out["baseline_brier"] = _mean(s["baseline_brier"] for _, s in scored)
    paired = [(r, s) for r, s in scored if s["market_ll"] is not None]
    out["n_market"] = len(paired)
    if paired:
        out["market_ll"] = _mean(s["market_ll"] for _, s in paired)
        out["market_brier"] = _mean(s["market_brier"] for _, s in paired)
        band = stats.t_interval([s["brain_ll"] - s["market_ll"] for _, s in paired], [r["day"] for r, _ in paired])
        out.update({"gap": band["mean"], "gap_lo": band["lo"], "gap_hi": band["hi"]})
    window = [s for r in (trailing if trailing is not None else rows) if (s := game_scores(r)) is not None]
    window = window[-TRAILING:]
    p = np.array([s["p_home"] for s in window])
    y = np.array([1.0 if s["home_won"] else 0.0 for s in window])
    out.update({k: v for k, v in stats.calibration(p, y).items() if k != "n"})
    versions: Dict[str, Dict[str, Any]] = {}
    horizons: Dict[str, Dict[str, Any]] = {}
    for r, s in scored:
        for bucket, key in ((versions, str(r.get("params_version", 0))), (horizons, _horizon(r["start"] - r["logged_at"]))):
            slot = bucket.setdefault(key, {"n": 0, "brain_ll": 0.0, "gap_n": 0, "gap": 0.0})
            slot["n"] += 1
            slot["brain_ll"] += s["brain_ll"]
            if s["market_ll"] is not None:
                slot["gap_n"] += 1
                slot["gap"] += s["brain_ll"] - s["market_ll"]
    for bucket in (versions, horizons):
        for slot in bucket.values():
            slot["brain_ll"] /= slot["n"]
            slot["gap"] = slot["gap"] / slot["gap_n"] if slot["gap_n"] else None
    details: Dict[str, Any] = {"versions": versions, "horizons": horizons, "calibration_n": int(p.size),
                               "market_kinds": {k: sum(1 for _, s in paired if s["market_kind"] == k)
                                                for k in ("sharp", "us_books")}}
    if curve and p.size >= 30:
        details["reliability"] = stats.reliability(p, y, reps=200, seed=seed)
    out["details"] = details
    return out


def summarise_totals(rows: Sequence[Mapping[str, Any]], *, trailing: Optional[Sequence[Mapping[str, Any]]] = None,
                     reps: int = 1000, seed: int = 0) -> Dict[str, Any]:
    """The totals target: the over's log loss after the adjustment (the brain) against before it (the books' own
    margin-free chance: the market) and an even split (the baseline), the paired gap with its 95% range
    (``stats.t_interval``) on the adjusted games only, calibration of p_after over ``trailing``, and the continuous residual slope (final total minus
    line on the applied shift) in ``details``."""
    n = len(rows)
    out: Dict[str, Any] = {"n": n, "brain_ll": None, "market_ll": None, "baseline_ll": None, "brier": None,
                           "market_brier": None, "baseline_brier": None, "gap": None, "gap_lo": None, "gap_hi": None,
                           "n_market": n, "details": {}}
    if not n:
        return out
    after = np.array([r["p_after"] for r in rows])
    before = np.array([r["p_before"] for r in rows])
    y = np.array([r["y"] for r in rows], dtype=float)
    out.update({"brain_ll": float(stats.log_loss(after, y).mean()), "market_ll": float(stats.log_loss(before, y).mean()),
                "baseline_ll": math.log(2), "brier": float(stats.brier(after, y).mean()),
                "market_brier": float(stats.brier(before, y).mean()), "baseline_brier": 0.25})
    adjusted = [r for r in rows if r["applied"] != 0]
    details: Dict[str, Any] = {"adjusted": len(adjusted)}
    if adjusted:
        d = stats.log_loss([r["p_after"] for r in adjusted], [r["y"] for r in adjusted]) - stats.log_loss(
            [r["p_before"] for r in adjusted], [r["y"] for r in adjusted])
        band = stats.t_interval(d, [r["day"] for r in adjusted])
        out.update({"gap": band["mean"], "gap_lo": band["lo"], "gap_hi": band["hi"]})
        db = stats.brier([r["p_after"] for r in adjusted], [r["y"] for r in adjusted]) - stats.brier(
            [r["p_before"] for r in adjusted], [r["y"] for r in adjusted])
        details["brier_gap"] = stats.t_interval(db, [r["day"] for r in adjusted])
        details["slope"] = residual_slope(adjusted)
        causes: Dict[str, int] = {}
        for r in adjusted:
            cause = cause_of(r.get("adjustment"))
            causes[cause] = causes.get(cause, 0) + 1
        details["causes"] = causes
    window = list(trailing if trailing is not None else rows)[-TRAILING:]
    if window:
        out.update({k: v for k, v in stats.calibration([r["p_after"] for r in window],
                                                        [r["y"] for r in window]).items() if k != "n"})
    out["details"] = details
    return out


def cause_of(adjustment: Any) -> str:
    """"forecast", "roof_closed" or "roof_open" of a logged adjustment (its cause, else read from its reason)."""
    if not isinstance(adjustment, Mapping):
        return "forecast"
    if adjustment.get("cause") in context.CAUSES:
        return str(adjustment["cause"])
    reason = str(adjustment.get("reason") or "")
    if "roof was announced closed" in reason:
        return "roof_closed"
    if "roof was announced open" in reason:
        return "roof_open"
    return "forecast"


def residual_slope(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    """beta from (final total - line) = beta x applied (through the origin) on adjusted games: the share of the applied
    shift that showed up in the results (1: right size, 0: noise), with its standard error and 95% range."""
    pairs = [(float(r["total"]) - r["line"], r["applied"]) for r in rows if r.get("total") is not None and r["applied"]]
    if len(pairs) < 3:
        return {"n": len(pairs), "beta": None, "se": None, "lo": None, "hi": None}
    resid = np.array([p for p, _ in pairs])
    x = np.array([a for _, a in pairs])
    sxx = float(np.dot(x, x))
    beta = float(np.dot(x, resid) / sxx)
    sigma2 = float(np.sum((resid - beta * x) ** 2) / (len(pairs) - 1))
    se = math.sqrt(sigma2 / sxx)
    return {"n": len(pairs), "beta": beta, "se": se, "lo": beta - stats.Z95 * se, "hi": beta + stats.Z95 * se}


# --- replays through the engine ------------------------------------------------------------------------------------

def blend_kwargs(sport: str, values: Mapping[str, float]) -> Dict[str, Any]:
    """``consensus.blend`` keywords for a value set (nothing for knobs at their default)."""
    return registry.Params(sport, values).blend_kwargs()


def reblend(row: Mapping[str, Any], kwargs: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """The blend of a logged game's sources with other parameters, as the slate would have computed it then."""
    out = consensus.blend(row["sources"], row["league"], now=row["logged_at"], start=row["start"],
                          neutral=row.get("neutral", False), **kwargs)
    return out["p"] if out else None


def replay_results(rows: Sequence[Mapping[str, Any]], sport: str, values: Mapping[str, float]
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """(log loss, Brier) per game of the blend recomputed with ``values`` (nan where the outcome isn't priced)."""
    kwargs = blend_kwargs(sport, values)
    ll, br = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        p = reblend(row, kwargs)
        scored = _scored(p, row.get("outcome"))
        if scored:
            ll[i], br[i] = scored["log_loss"], scored["brier"]
    return ll, br


def readjusted(row: Mapping[str, Any], sport: str, values: Mapping[str, float]) -> Optional[float]:
    """p(over) of a context row with another adjustment weight / total SD, through ``context.adjusted_over``; None
    for a row whose logged adjustment can't be replayed."""
    adj = row.get("adjustment")
    if not row["applied"] and not adj:
        return row["p_before"]
    if not isinstance(adj, Mapping) or adj.get("raw") is None:
        return None
    sp = context.sport_params(sport, {k.split(".", 1)[1]: v for k, v in values.items()
                                      if k in ("context.total_sd", "context.margin_sd", "context.home_prior",
                                               "context.draw_prior") and v != registry.default_value(k, sport)})
    raw = float(adj["raw"])
    if adj.get("full"):
        applied = raw
    else:
        weight = float(values.get("context.adjust_weight", context.ADJUST_WEIGHT))
        cap = adj.get("cap")
        cap = float(cap) if cap is not None else float(sp.caps.get(adj.get("kind") or "total", math.inf))
        applied = max(-cap, min(cap, weight * raw))
    try:
        return context.adjusted_over(sp, row["line"], min(max(row["p_before"], 1e-6), 1 - 1e-6), applied)
    except ValueError:
        return None


def replay_totals(rows: Sequence[Mapping[str, Any]], sport: str, values: Mapping[str, float]) -> np.ndarray:
    """p(over) per context row replayed with ``values`` (nan where it can't be)."""
    out = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        p = readjusted(row, sport, values)
        if p is not None:
            out[i] = p
    return out


@dataclass
class Paired:
    """Per-game scores of a challenger against a champion on the same games."""

    d_ll: np.ndarray
    d_brier: np.ndarray
    days: List[str]
    starts: np.ndarray
    p_champion: np.ndarray  # the champion's chance of the event scored (home side / over), for calibration
    p_challenger: np.ndarray
    y: np.ndarray

    @property
    def n(self) -> int:
        return int(self.d_ll.size)

    def after(self, moment: float) -> "Paired":
        keep = self.starts > moment
        return Paired(self.d_ll[keep], self.d_brier[keep], [d for d, k in zip(self.days, keep) if k],
                      self.starts[keep], self.p_champion[keep], self.p_challenger[keep], self.y[keep])


def _paired_from(ll_a: np.ndarray, br_a: np.ndarray, ll_b: np.ndarray, br_b: np.ndarray, rows: Sequence[Any],
                 p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray) -> Paired:
    keep = np.isfinite(ll_a) & np.isfinite(ll_b)
    starts = np.array([r["start"] if isinstance(r, Mapping) else r.start for r in rows], dtype=float)
    days = [r["day"] if isinstance(r, Mapping) else r.day for r in rows]
    return Paired((ll_b - ll_a)[keep], (br_b - br_a)[keep], [d for d, k in zip(days, keep) if k], starts[keep],
                  p_a[keep], p_b[keep], y[keep])


def pair_results(rows: Sequence[Mapping[str, Any]], sport: str, champion: Mapping[str, float],
                 challenger: Mapping[str, float]) -> Paired:
    """Champion and challenger blends replayed on the same graded games (log loss and Brier differences,
    challenger minus champion: negative is better)."""
    kw_a, kw_b = blend_kwargs(sport, champion), blend_kwargs(sport, challenger)
    n = len(rows)
    y = np.array([1.0 if r.get("outcome") == "home" else 0.0 for r in rows])
    sides = []
    for kw in (kw_a, kw_b):
        if sides and kw == kw_a:
            sides.append(sides[0])
            continue
        ll, br, ph = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        for i, row in enumerate(rows):
            p = reblend(row, kw)
            scored = _scored(p, row.get("outcome"))
            if scored:
                ll[i], br[i], ph[i] = scored["log_loss"], scored["brier"], p["home"] / math.fsum(p.values())
        sides.append((ll, br, ph))
    (ll_a, br_a, p_a), (ll_b, br_b, p_b) = sides
    return _paired_from(ll_a, br_a, ll_b, br_b, rows, p_a, p_b, y)


def pair_totals(rows: Sequence[Mapping[str, Any]], sport: str, champion: Mapping[str, float],
                challenger: Mapping[str, float]) -> Paired:
    """The same for the totals layer: p(over) replayed with each value set on the same context rows."""
    pa, pb = replay_totals(rows, sport, champion), replay_totals(rows, sport, challenger)
    y = np.array([r["y"] for r in rows], dtype=float)
    ll_a, ll_b = stats.log_loss(np.nan_to_num(pa, nan=0.5), y), stats.log_loss(np.nan_to_num(pb, nan=0.5), y)
    ll_a[~np.isfinite(pa)] = np.nan
    ll_b[~np.isfinite(pb)] = np.nan
    return _paired_from(ll_a, stats.brier(pa, y), ll_b, stats.brier(pb, y), rows, pa, pb, y)


def compare(paired: Paired, *, reps: int = 2000, seed: int = 0, lags: int = 2) -> Dict[str, Any]:
    """The gate's evidence on a pairing: n, days, mean change in log loss with its day-block 95% range, mean change in
    Brier, the one-sided Diebold-Mariano (HLN) p-value, and the calibration slope of each side."""
    boot = stats.block_bootstrap(paired.d_ll, paired.days, reps=reps, seed=seed)
    dm = stats.diebold_mariano(paired.d_ll, paired.days, lags=lags)
    return {"n": paired.n, "days": boot["days"], "dll": boot["mean"], "lo": boot["lo"], "hi": boot["hi"],
            "dbrier": float(paired.d_brier.mean()) if paired.n else None, "p": dm["p"], "p_worse": dm["p_worse"],
            "dm": dm["stat"], "calibration_champion": stats.calibration(paired.p_champion, paired.y),
            "calibration_challenger": stats.calibration(paired.p_challenger, paired.y)}


# --- the goals model and count models (soccer) ---------------------------------------------------------------------

def rates_many(matches: Sequence[MatchRecord], rho: float, iters: int = 100,
               start: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Home and away goal rates for many matches at once: the least-squares fit of ``soccer.rates_from_odds`` (the
    same four chances, the same Dixon-Coles grid, Levenberg-Marquardt on log rates within 0.05-6 goals), vectorised;
    tests pin it to the real function. ``start`` (rates from a nearby rho) skips the coarse search."""
    n = len(matches)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    target = np.array([[m.p_home, m.p_draw, m.p_away, m.p_over] for m in matches], dtype=float)
    total = target[:, :3].sum(axis=1, keepdims=True)
    target[:, :3] /= total
    lo, hi = math.log(soccer._RATE_LOW), math.log(soccer._RATE_HIGH)
    if start is not None:
        x = np.clip(np.column_stack([np.log(start[0]), np.log(start[1])]), lo, hi)
    else:
        goals = _poisson_total_for_over(target[:, 3])
        x, best_cost = None, None
        for s in [i / 20 for i in range(1, 20)]:
            x0 = np.column_stack([np.log(np.maximum(goals * s, soccer._RATE_LOW)),
                                  np.log(np.maximum(goals * (1 - s), soccer._RATE_LOW))])
            cost = np.sum((_fit_values_many(x0, rho) - target) ** 2, axis=1)
            if x is None:
                x, best_cost = x0, cost
            else:
                better = cost < best_cost
                x = np.where(better[:, None], x0, x)
                best_cost = np.where(better, cost, best_cost)
    r = _fit_values_many(x, rho) - target
    damping = np.full(n, 1e-3)
    step = 1e-6
    active = np.arange(n)
    for _ in range(iters):
        if active.size == 0:
            break
        xa, ra, ta = x[active], r[active], target[active]
        cols = []
        for i in range(2):
            e = np.zeros(2)
            e[i] = step
            cols.append((_fit_values_many(xa + e, rho) - _fit_values_many(xa - e, rho)) / (2 * step))
        j0, j1 = cols
        a11, a12, a22 = (j0 * j0).sum(1), (j0 * j1).sum(1), (j1 * j1).sum(1)
        g0, g1 = (j0 * ra).sum(1), (j1 * ra).sum(1)
        cost = (ra * ra).sum(1)
        damp = damping[active]
        pending = np.ones(active.size, dtype=bool)
        moved = np.zeros(active.size)
        for _ in range(20):
            idx = np.where(pending)[0]
            if idx.size == 0:
                break
            a = a11[idx] * (1 + damp[idx])
            d = a22[idx] * (1 + damp[idx])
            det = a * d - a12[idx] * a12[idx]
            det = np.where(np.abs(det) < 1e-300, np.inf, det)
            dx = (-g0[idx] * d + a12[idx] * g1[idx]) / det
            dy = (-a * g1[idx] + a12[idx] * g0[idx]) / det
            cand = np.clip(xa[idx] + np.column_stack([dx, dy]), lo, hi)
            rc = _fit_values_many(cand, rho) - ta[idx]
            ok = (rc * rc).sum(1) < cost[idx]
            good = idx[ok]
            moved[good] = np.abs(cand[ok] - xa[good]).sum(1)
            xa[good], ra[good] = cand[ok], rc[ok]
            damp[good] = np.maximum(damp[good] / 3, 1e-9)
            damp[idx[~ok]] *= 4
            pending[good] = False
        x[active], r[active], damping[active] = xa, ra, damp
        # A match is done when no damping gave a better fit, or the step was negligible (as in rates_from_odds)
        active = active[~pending & (moved >= 1e-12)]
    return np.exp(x[:, 0]), np.exp(x[:, 1])


def _poisson_total_for_over(over: np.ndarray) -> np.ndarray:
    low, high = np.full(over.shape, 0.1), np.full(over.shape, 12.0)
    for _ in range(60):
        mid = (low + high) / 2
        under = np.exp(-mid) * (1 + mid + mid * mid / 2)
        up = 1 - under < over
        low, high = np.where(up, mid, low), np.where(up, high, mid)
    return (low + high) / 2


def grids_many(lam: np.ndarray, mu: np.ndarray, rho: float, max_goals: int = soccer.MAX_GOALS) -> np.ndarray:
    """Dixon-Coles score grids (n, G, G), as ``soccer.grid``."""
    k = np.arange(max_goals + 1, dtype=float)
    lf = stats.lgamma(k + 1)
    ph = np.exp(-lam[:, None] + k[None, :] * np.log(lam[:, None]) - lf[None, :])
    pa = np.exp(-mu[:, None] + k[None, :] * np.log(mu[:, None]) - lf[None, :])
    g = ph[:, :, None] * pa[:, None, :]
    g[:, 0, 0] *= np.maximum(1 - lam * mu * rho, 0)
    g[:, 0, 1] *= np.maximum(1 + lam * rho, 0)
    g[:, 1, 0] *= np.maximum(1 + mu * rho, 0)
    g[:, 1, 1] *= max(1 - rho, 0)
    return g / g.sum(axis=(1, 2), keepdims=True)


_G = soccer.MAX_GOALS + 1
_IDX = np.arange(_G)
_HOME = (_IDX[:, None] > _IDX[None, :]).astype(float)
_DRAW = (_IDX[:, None] == _IDX[None, :]).astype(float)
_AWAY = (_IDX[:, None] < _IDX[None, :]).astype(float)
_OVER = ((_IDX[:, None] + _IDX[None, :]) > 2.5).astype(float)
_BOTH = ((_IDX[:, None] >= 1) & (_IDX[None, :] >= 1)).astype(float)


def _fit_values_many(x: np.ndarray, rho: float) -> np.ndarray:
    g = grids_many(np.exp(x[:, 0]), np.exp(x[:, 1]), rho)
    return np.column_stack([(g * _HOME).sum((1, 2)), (g * _DRAW).sum((1, 2)), (g * _AWAY).sum((1, 2)),
                            (g * _OVER).sum((1, 2))])


def goals_scores(matches: Sequence[MatchRecord], rho: float, nudge: float,
                 rates: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 start: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
    """Per-match scores of the goals model at ``rho``/``nudge``: exact-score log score, total-goals CRPS and PIT, the
    BTTS-yes chance (with the nudge, clamped like ``soccer.markets``) and its log loss."""
    lam, mu = rates if rates is not None else rates_many(matches, rho, start=start)
    g = grids_many(lam, mu, rho)
    hg = np.array([min(m.home_goals, _G - 1) for m in matches])
    ag = np.array([min(m.away_goals, _G - 1) for m in matches])
    exact = g[np.arange(len(matches)), hg, ag]
    totals = np.zeros((len(matches), 2 * _G - 1))
    for x in range(_G):
        for y in range(_G):
            totals[:, x + y] += g[:, x, y]
    observed = np.array([m.home_goals + m.away_goals for m in matches])
    observed = np.minimum(observed, totals.shape[1] - 1)
    counted = stats.count_scores(totals, observed)
    btts_raw = (g * _BOTH).sum((1, 2))
    btts = np.clip(btts_raw + nudge, 0.01, 0.99)
    both = np.array([1.0 if m.home_goals >= 1 and m.away_goals >= 1 else 0.0 for m in matches])
    return {"exact_ls": -np.log(np.maximum(exact, 1e-300)), "goals_crps": counted["per_game_crps"],
            "goals_pit": counted["pit"], "btts_raw": btts_raw, "btts": btts, "btts_y": both,
            "btts_ll": stats.log_loss(btts, both), "rates": (lam, mu)}


CLIMATOLOGY_MIN = 100  # training matches a league needs for its own climatology (else the pooled one is used)


def _by_league(train: Sequence[MatchRecord], test: Sequence[MatchRecord], fit: Callable[[Sequence[MatchRecord]], Any]
               ) -> List[Any]:
    """Per test match the baseline fitted on its league's training matches (``fit``), or on all of them for a league
    with fewer than CLIMATOLOGY_MIN: a model shouldn't beat "climatology" just by knowing which league it is in."""
    pooled = fit(train)
    leagues: Dict[str, Any] = {}
    for league in {m.league for m in test}:
        mine = [m for m in train if m.league == league]
        leagues[league] = fit(mine) if len(mine) >= CLIMATOLOGY_MIN else pooled
    return [leagues[m.league] for m in test]


def climatology_goals(train: Sequence[MatchRecord], test: Sequence[MatchRecord]) -> np.ndarray:
    """Exact-score log score of the no-input baseline: independent Poissons at the training window's mean home and
    away goals of the match's league (``_by_league``)."""
    def fit(rows: Sequence[MatchRecord]) -> Tuple[float, float]:
        return (max(float(np.mean([m.home_goals for m in rows])) if rows else 1.5, 0.05),
                max(float(np.mean([m.away_goals for m in rows])) if rows else 1.2, 0.05))

    rates = _by_league(train, test, fit)
    lam = np.array([r[0] for r in rates], dtype=float)
    mu = np.array([r[1] for r in rates], dtype=float)
    return -(stats.poisson_logpmf([m.home_goals for m in test], lam) + stats.poisson_logpmf(
        [m.away_goals for m in test], mu))


def count_inputs(matches: Sequence[MatchRecord], rho: float = soccer.DC_RHO,
                 rates: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(supremacy, total) of each match from the goals model's rates (home minus away, and their sum)."""
    lam, mu = rates if rates is not None else rates_many(matches, rho)
    return lam - mu, lam + mu


def count_means(model: Mapping[str, float], supremacy: np.ndarray, total: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-team means of a count model (registry's counts.<target>.* terms)."""
    centre = total - registry.COUNT_TOTAL_CENTRE
    base = model["intercept"] + model["total"] * centre
    home = np.exp(base + model["home"] / 2 + model["supremacy"] * supremacy)
    away = np.exp(base - model["home"] / 2 - model["supremacy"] * supremacy)
    return home, away


def count_loglik(model: Mapping[str, float], y_home: np.ndarray, y_away: np.ndarray, supremacy: np.ndarray,
                 total: np.ndarray) -> np.ndarray:
    """Per-match log likelihood (both teams) of a count model."""
    home, away = count_means(model, supremacy, total)
    return stats.nb_logpmf(y_home, home, model["size"]) + stats.nb_logpmf(y_away, away, model["size"])


def climatology_counts(train_home: np.ndarray, train_away: np.ndarray) -> Dict[str, float]:
    """The no-input baseline for a count target: home and away means and one dispersion (method of moments, at
    least 2), fitted on the training window."""
    mh, ma = float(np.mean(train_home)), float(np.mean(train_away))
    var = (float(np.var(train_home, ddof=1)) + float(np.var(train_away, ddof=1))) / 2 if len(train_home) > 1 else 0.0
    mean = (mh + ma) / 2
    excess = var - mean
    size = min(max(mean * mean / excess, 2.0), 500.0) if excess > 1e-9 else 500.0
    return {"home": max(mh, 0.05), "away": max(ma, 0.05), "size": size}


def climatology_loglik(target: str, train: Sequence[MatchRecord], test: Sequence[MatchRecord]
                       ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Per test match the log likelihood (both teams) of a count target's climatology: the match's league's mean and
    dispersion over the training window (``_by_league``, at least CLIMATOLOGY_MIN matches, else all leagues'), and
    the fits used ({"pooled", "leagues"})."""
    have = [m for m in train if target in m.stats]

    def fit(rows: Sequence[MatchRecord]) -> Dict[str, float]:
        return climatology_counts(np.array([m.stats[target][0] for m in rows]),
                                  np.array([m.stats[target][1] for m in rows]))

    clims = _by_league(have, test, fit)
    yh = np.array([m.stats[target][0] for m in test])
    ya = np.array([m.stats[target][1] for m in test])
    home = np.array([c["home"] for c in clims], dtype=float)
    away = np.array([c["away"] for c in clims], dtype=float)
    size = np.array([c["size"] for c in clims], dtype=float)
    ll = stats.nb_logpmf(yh, home, size) + stats.nb_logpmf(ya, away, size)
    leagues = {m.league: c for m, c in zip(test, clims)}
    return ll, {"pooled": fit(have), "leagues": leagues}


def count_target_scores(target: str, model: Mapping[str, float], train: Sequence[MatchRecord],
                        test: Sequence[MatchRecord], rho: float = soccer.DC_RHO) -> Dict[str, Any]:
    """Model against climatology (per league: ``climatology_loglik``) on ``test`` for one count target (both teams'
    counts, per match): mean log score of each, the per-match difference, CRPS, PIT and variance ratio of the model,
    and n."""
    have_test = [m for m in test if target in m.stats]
    have_train = [m for m in train if target in m.stats]
    if not have_test or len(have_train) < 2:
        return {"n": len(have_test), "model_ls": None, "baseline_ls": None}
    yh = np.array([m.stats[target][0] for m in have_test])
    ya = np.array([m.stats[target][1] for m in have_test])
    sup, tot = count_inputs(have_test, rho)
    base_ll, clim = climatology_loglik(target, have_train, have_test)
    model_ll = count_loglik(model, yh, ya, sup, tot)
    home, away = count_means(model, sup, tot)
    kmax = int(max(yh.max(), ya.max(), 1) + 40)
    table = stats.nb_pmf_table(np.concatenate([home, away]), model["size"], kmax)
    scored = stats.count_scores(table, np.concatenate([yh, ya]))
    return {"n": len(have_test), "model_ls": float(-model_ll.mean()), "baseline_ls": float(-base_ll.mean()),
            "d": -(model_ll - base_ll), "days": [m.day for m in have_test], "crps": scored["crps"],
            "pit": scored["pit"], "variance_ratio": scored["variance_ratio"], "climatology": clim}


# --- weekly scores -------------------------------------------------------------------------------------------------

def week_scores(corpus: Corpus, week: str, *, reps: int = 1000, seed: int = 0,
                params: Optional[Mapping[str, registry.Params]] = None) -> List[Dict[str, Any]]:
    """brain_scores rows for one ISO week: per sport the "result" target (consensus_log) and the "total" target
    (context_log); for soccer also "goals" and "btts" (the goals model on the history's matches) and "corners" and
    "cards" (count models against climatology), when the week has such games."""
    begin, end = stats.week_bounds(week)
    out = []
    params = params or {}
    for sport in registry.SPORTS:
        live = params.get(sport) or registry.Params(sport)
        rows = corpus.sport(sport)
        week_rows = [r for r in rows if begin <= r["start"] < end]
        if week_rows:
            before = [r for r in rows if r["start"] < end]
            summary = summarise_results(week_rows, trailing=before, reps=reps, seed=seed)
            out.append({"week": week, "sport": sport, "target": "result", "params_version": _latest_version(week_rows),
                        **summary})
        ctx = corpus.sport(sport, "context")
        week_ctx = [r for r in ctx if begin <= r["start"] < end]
        if week_ctx:
            summary = summarise_totals(week_ctx, trailing=[r for r in ctx if r["start"] < end], reps=reps, seed=seed)
            out.append({"week": week, "sport": sport, "target": "total", "params_version": _latest_version(week_ctx),
                        **summary})
        if sport == "soccer" and corpus.matches:
            out += _soccer_week(corpus.matches, week, begin, end, live, reps, seed)
    return out


def _latest_version(rows: Sequence[Mapping[str, Any]]) -> int:
    return int(max(rows, key=lambda r: r["start"]).get("params_version") or 0)


def _soccer_week(matches: Sequence[MatchRecord], week: str, begin: float, end: float, live: registry.Params,
                 reps: int, seed: int) -> List[Dict[str, Any]]:
    test = [m for m in matches if begin <= m.start < end]
    train = [m for m in matches if m.start < begin]
    if not test:
        return []
    base = {"week": week, "sport": "soccer", "params_version": live.version}
    goals = goals_scores(test, live.rho, live.btts_nudge)
    clim = climatology_goals(train, test)
    days = [m.day for m in test]
    d = goals["exact_ls"] - clim
    boot = stats.t_interval(d, days)
    rows = [{**base, "target": "goals", "n": len(test), "brain_ll": float(goals["exact_ls"].mean()),
             "baseline_ll": float(clim.mean()), "market_ll": None, "gap": boot["mean"], "gap_lo": boot["lo"],
             "gap_hi": boot["hi"], "n_market": 0,
             "details": {"gap_vs": "climatology", "crps": float(goals["goals_crps"].mean()), "pit": goals["goals_pit"]}}]
    y = goals["btts_y"]
    cal = stats.calibration(goals["btts"], y)
    rows.append({**base, "target": "btts", "n": len(test), "brain_ll": float(goals["btts_ll"].mean()),
                 "brier": float(stats.brier(goals["btts"], y).mean()),
                 "baseline_ll": float(stats.log_loss(np.full(len(test), _rate(train, "btts")), y).mean()),
                 **{k: v for k, v in cal.items() if k != "n"}, "details": {}})
    for target in registry.COUNT_TARGETS:
        scored = count_target_scores(target, live.counts(target), train, test, live.rho)
        if scored.get("model_ls") is None:
            continue
        boot = stats.t_interval(scored["d"], scored["days"])
        rows.append({**base, "target": target, "n": scored["n"], "brain_ll": scored["model_ls"],
                     "baseline_ll": scored["baseline_ls"], "gap": boot["mean"], "gap_lo": boot["lo"],
                     "gap_hi": boot["hi"], "details": {"gap_vs": "climatology", "crps": scored["crps"],
                                                       "pit": scored["pit"],
                                                       "variance_ratio": scored["variance_ratio"]}})
    return rows


def _rate(matches: Sequence[MatchRecord], what: str) -> float:
    if not matches:
        return 0.5
    if what == "btts":
        hits = sum(1 for m in matches if m.home_goals >= 1 and m.away_goals >= 1)
    else:
        hits = sum(1 for m in matches if m.home_goals + m.away_goals > 2.5)
    return min(max(hits / len(matches), 0.01), 0.99)


def params_at(db: sqlite3.Connection, sport: str, moment: float) -> registry.Params:
    """The values that were live for ``sport`` at ``moment``: the current champion, or the version it replaced (its
    parent, and so on back) when it went live after ``moment``; the defaults before any change, or when safe mode has
    been on since before ``moment``."""
    from .brain_store import champion, get_version, safe_mode

    safe = safe_mode(db)
    if safe and float(safe.get("since") or 0) <= moment:
        return registry.Params(sport)
    row = champion(db, sport)
    seen = set()
    while row is not None and int(row["version"]) not in seen and float(row.get("effective_from") or 0) > moment:
        seen.add(int(row["version"]))
        parent = int(row.get("parent") or 0)
        row = get_version(db, parent) if parent else None
    if row is None:
        return registry.Params(sport)
    stored = {k: v for k, v in (row.get("params") or {}).items() if k in registry.KNOBS
              and registry.KNOBS[k].applies(sport)}
    try:
        return registry.Params(sport, stored, version=int(row["version"]), source=str(row.get("source") or "refit"),
                               effective_from=row.get("effective_from"))
    except ValueError:
        return registry.Params(sport)


def week_params(db: sqlite3.Connection, corpus: Corpus, week: str) -> Dict[str, registry.Params]:
    """The soccer values each scored week is replayed with (goals, BTTS, corners, cards): those live at the start of
    the week's first match (point in time: a change made later never rescores the weeks before it)."""
    begin, end = stats.week_bounds(week)
    starts = [m.start for m in corpus.matches if begin <= m.start < end]
    return {"soccer": params_at(db, "soccer", min(starts) if starts else begin)}


def write_week_scores(db: sqlite3.Connection, corpus: Corpus, week: str, *, now: Optional[float] = None,
                      reps: int = 1000, seed: int = 0,
                      params: Optional[Mapping[str, registry.Params]] = None) -> List[Dict[str, Any]]:
    """Compute ``week_scores`` and upsert them into brain_scores (nightly runs rewrite the week to date). Without
    ``params`` the values live during the week are used (``week_params``)."""
    from .brain_store import put_score, transaction

    if params is None:
        params = week_params(db, corpus, week)
    rows = week_scores(corpus, week, reps=reps, seed=seed, params=params)
    with transaction(db):
        for row in rows:
            put_score(db, row, now)
    return rows


def headline(rows: Sequence[Mapping[str, Any]], last: int = 200, *, reps: int = 1000, seed: int = 0
             ) -> Dict[str, Any]:
    """The page's headline numbers for one sport: over the last ``last`` graded games with a market price, the brain's
    log loss minus the market's with its day-block 95% range, and a word for it from code: "level" when the range
    holds 0, "behind" when it sits above 0, "ahead" when below."""
    scored = [(r, s) for r in rows if (s := game_scores(r)) is not None and s["market_ll"] is not None][-last:]
    if not scored:
        return {"n": 0, "gap": None, "lo": None, "hi": None, "word": "no games yet"}
    boot = stats.block_bootstrap([s["brain_ll"] - s["market_ll"] for _, s in scored], [r["day"] for r, _ in scored],
                                 reps=reps, seed=seed)
    lo, hi = boot["lo"], boot["hi"]
    if lo is None or hi is None:
        word = "too few days to say"
    elif lo <= 0 <= hi:
        word = "about level with the sharp market"
    elif lo > 0:
        word = "behind the sharp market"
    else:
        word = "ahead of the sharp market"
    return {"n": len(scored), "gap": boot["mean"], "lo": lo, "hi": hi, "word": word,
            "since": scored[0][0]["start"], "until": scored[-1][0]["start"]}
