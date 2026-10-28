"""The weekly tuner and the nightly guard (method.md sections 4-5): refits, gates, promotions, rollbacks, safe mode.

Candidates (at most 6 a sport a week, at most 3 of them the AI's):

- deterministic refits, each a penalised fit centred on the champion that keeps only its 95%-sure part (an LCB
  step), clipped to one cycle's step and the bounds (learn/registry.py): the blend's kind weights (fitted through a
  vectorised copy of ``consensus.blend`` that is checked against the real function, with a ridge to the champion and
  partial pooling toward the all-sports fit), the pool's sharpness (structural), the near-start factor and the
  low-confidence shrink, Dixon-Coles rho (profile likelihood of the exact score) and the BTTS nudge, ADJUST_WEIGHT
  (the continuous residual slope), the home and draw priors, the total and margin SDs, the NBA model's weights and
  the corner and card count models (structural);
- AI proposals: a direction and a bound for an allow-listed knob; the size is refitted by the LCB rule on games
  before the proposal, frozen as a challenger, and judged only on games that start after it was made.

Gates:

- a **nudge** (within one step, no structural knob) is promoted when, on out-of-sample games of a rolling-origin
  replay (refit on everything before each 4-week block, score the block), n >= MIN_EVAL, the upper end of the
  day-block 95% range of the mean change in log loss is at most +0.0005, the mean change in Brier is at most 0 and
  the calibration slope stays sound; else "held" with the reason (usually "needs N more games");
- a **structural** change needs superiority: the one-sided Diebold-Mariano (HLN) p-value under the Holm-adjusted
  0.05 across the week's structural candidates and the 95% range entirely below 0; it then runs as a challenger in
  shadow for 2 weeks and is promoted at a later weekly run if its forward range has not moved above 0.

Every candidate is a row of the trials ledger. A claimed gain bigger than 0.01 log loss (or 0.005 Brier) is treated
as a data bug and rejected. Rollback is automatic (nightly and weekly) when, at one of the planned looks (the first
200, 400 and 800 games logged since a promotion), the range of the change against the previous champion (always
scored in shadow by replay; 98.3% each, a Bonferroni split) sits above 0, or a one-sided CUSUM (k 0.25, h 5) of the
weekly standardised change (scale frozen from the first 200 games) fires. The tripwire puts the whole brain in
safe mode (the defaults) when the blend sits more than 4 points from the sharp price on more than 5% of a night's
games. The owner's Undo rolls a version back to its parent.
"""

from __future__ import annotations

import datetime
import math
import sqlite3
import time
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..engine import consensus, nba_model, soccer
from . import brain_store as bs
from . import evaluate as ev
from . import registry, stats

MIN_TRAIN = ev.MIN_TRAIN
MIN_EVAL = ev.MIN_EVAL
MIN_ID = 150  # games where a kind sits beside another kind before its weight may move
NONINF_MARGIN = 0.0005  # a nudge's upper 95% bound of mean change in log loss
ALPHA = 0.05
SHADOW_WEEKS = 2
AI_LOOKS_FACTOR = (1, 2)  # an AI proposal is judged at planned looks only: MIN_EVAL and 2 x MIN_EVAL later games
AI_LOOK_ALPHAS = (0.005, 0.045)  # the 0.05 spent across the looks (O'Brien-Fleming-like: little at the first)
ROLLBACK_MIN = 200
COOL_OFF_WEEKS = 4  # a setting the owner undid, or that was rolled back, is not moved again for this long
CUSUM_K, CUSUM_H = 0.25, 5.0
TRIP_PP, TRIP_SHARE, TRIP_MIN_GAMES = 4.0, 0.05, 20
MAX_CANDIDATES, MAX_AI = 6, 3
LEAK_LL, LEAK_BRIER = 0.01, 0.005  # a claimed gain past these is a data bug until proven otherwise
EVAL_WEEKS = 26  # the rolling-origin replay covers the last 26 weeks
BLOCK_WEEKS = 4  # refit once per 4-week block inside it
PRIOR_GAMES = 400  # the champion's weight in the shrunk rates (home and draw priors), as games
RIDGE = {"weights": 20.0, "pool": 5.0, "sharpness": 50.0, "counts": 2.0, "nba": 50.0}
BOOT_REPS = 2000
_N = NormalDist()


# --- data kinds: what each knob is scored on ------------------------------------------------------------------------

def data_kind(knob: str) -> str:
    """The scorer a knob is judged by: results (the blend), totals (the adjustment layer), home (home and draw
    priors), residual_total / residual_margin (the SDs), goals (rho), btts (the nudge), counts:<target>, nba."""
    if knob.startswith("consensus."):
        return "results"
    if knob == "context.adjust_weight":
        return "totals"
    if knob in ("context.home_prior", "context.draw_prior"):
        return "home"
    if knob == "context.total_sd":
        return "residual_total"
    if knob == "context.margin_sd":
        return "residual_margin"
    if knob == "soccer.rho":
        return "goals"
    if knob == "soccer.btts_nudge":
        return "btts"
    if knob.startswith("counts."):
        return "counts:" + knob.split(".")[1]
    if knob.startswith("nba."):
        return "nba"
    raise ValueError(f"No scorer for {knob!r}")


def rows_for(kind: str, corpus: ev.Corpus, sport: str) -> List[Any]:
    """The records a data kind is scored on, oldest first."""
    if kind == "results":
        return corpus.sport(sport)
    if kind == "home":
        # The live record, plus the history's finished games of the sport (one row per game key)
        live = corpus.sport(sport)
        seen = {r["game_key"] for r in live}
        extra = [r for r in corpus.lines if r["sport"] == sport and r["game_key"] not in seen]
        return sorted(live + extra, key=lambda r: r["start"])
    if kind == "totals":
        return corpus.sport(sport, "context")
    if kind == "residual_total":
        rows = [{**r, "residual": float(r["total"]) - r["line"]} for r in corpus.sport(sport, "context")
                if r.get("total") is not None]
        seen = {r["game_key"] for r in rows}
        rows += [{**r, "residual": r["total_final"] - r["line"]} for r in corpus.lines
                 if r["sport"] == sport and r.get("line") is not None and r["game_key"] not in seen]
        return sorted(rows, key=lambda r: r["start"])
    if kind == "residual_margin":
        rows = [{**r, "residual": r["margin_final"] - float(r["payload"]["market_margin"])} for r in corpus.ai
                if r["sport"] == sport and r["payload"].get("market_margin") is not None]
        rows += [{**r, "residual": r["margin_final"] - r["market_margin"]} for r in corpus.lines
                 if r["sport"] == sport and r.get("market_margin") is not None]
        return sorted(rows, key=lambda r: r["start"])
    if kind in ("goals", "btts") or kind.startswith("counts:"):
        if sport != "soccer":
            return []
        matches = soccer_matches(corpus)
        if kind.startswith("counts:"):
            target = kind.split(":", 1)[1]
            return [m for m in matches if target in m.stats]
        return matches
    if kind == "nba":
        return [r for r in corpus.ai if r["sport"] == sport]
    raise ValueError(f"Unknown data kind {kind!r}")


def soccer_matches(corpus: ev.Corpus) -> List[ev.MatchRecord]:
    """History matches plus the live join (the blend's 1X2 from consensus_log with the books' over 2.5 from
    context_log, and the stored final score), one per game key, oldest first."""
    out = {m.game_key: m for m in corpus.matches}
    overs = {(r["league"], r["game_key"]): r for r in corpus.sport("soccer", "context") if abs(r["line"] - 2.5) < 1e-9}
    for r in corpus.sport("soccer"):
        p = (r.get("blend") or {}).get("p") or {}
        score = (r.get("result") or {}).get("score") or {}
        ctx = overs.get((r["league"], r["game_key"]))
        if ctx is None or "draw" not in p or score.get("home") is None or score.get("away") is None:
            continue
        if r["game_key"] in out:
            continue
        total = math.fsum(float(v) for v in p.values())
        out[r["game_key"]] = ev.MatchRecord(r["league"], r["game_key"], r["start"], float(p["home"]) / total,
                                            float(p["draw"]) / total, float(p["away"]) / total, ctx["p_before"],
                                            int(score["home"]), int(score["away"]), {},
                                            max(r["logged_at"], ctx["logged_at"]))
    return sorted(out.values(), key=lambda m: (m.start, m.game_key))


def losses(kind: str, rows: Sequence[Any], sport: str, values: Mapping[str, float]
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row (loss, Brier-like guard, chance of the scored event, event happened) of ``values`` on ``rows``: the
    loss is a log loss or log score (lower is better); nan where a row can't be scored."""
    n = len(rows)
    if n == 0:
        empty = np.zeros(0)
        return empty, empty, empty, empty
    if kind == "results":
        kw = ev.blend_kwargs(sport, values)
        ll, br, p = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        for i, row in enumerate(rows):
            q = ev.reblend(row, kw)
            scored = ev._scored(q, row.get("outcome"))
            if scored:
                ll[i], br[i] = scored["log_loss"], scored["brier"]
                p[i] = q["home"] / math.fsum(q.values())
        y = np.array([1.0 if r.get("outcome") == "home" else 0.0 for r in rows])
        return ll, br, p, y
    if kind == "totals":
        p = ev.replay_totals(rows, sport, values)
        y = np.array([r["y"] for r in rows], dtype=float)
        ll = stats.log_loss(np.nan_to_num(p, nan=0.5), y)
        ll[~np.isfinite(p)] = np.nan
        return ll, stats.brier(p, y), p, y
    if kind == "home":
        outcome = np.array([r.get("outcome") for r in rows])
        if sport == "soccer":
            pd_ = values["context.draw_prior"]
            ph = values["context.home_prior"]
            pa = max(1 - ph - pd_, 1e-4)
            chance = np.where(outcome == "home", ph, np.where(outcome == "draw", pd_, pa))
            y = (outcome == "home").astype(float)
            br = 0.5 * ((ph - y) ** 2 + (pd_ - (outcome == "draw")) ** 2 + (pa - (outcome == "away")) ** 2)
            return -np.log(np.maximum(chance, 1e-4)), br, np.full(n, ph), y
        keep = np.isin(outcome, ["home", "away"])
        y = (outcome == "home").astype(float)
        p = np.full(n, values["context.home_prior"])
        ll, br = stats.log_loss(p, y), stats.brier(p, y)
        ll[~keep], br[~keep] = np.nan, np.nan
        return ll, br, p, y
    if kind in ("residual_total", "residual_margin"):
        r = np.array([x["residual"] for x in rows], dtype=float)
        sd = values["context.total_sd" if kind == "residual_total" else "context.margin_sd"]
        ls = 0.5 * np.log(2 * np.pi * sd * sd) + r * r / (2 * sd * sd)
        return ls, (r / sd) ** 2, np.full(n, sd), r
    if kind in ("goals", "btts"):
        scored = ev.goals_scores(rows, values["soccer.rho"], values["soccer.btts_nudge"])
        if kind == "goals":
            return scored["exact_ls"], scored["goals_crps"], scored["btts"], scored["btts_y"]
        return scored["btts_ll"], stats.brier(scored["btts"], scored["btts_y"]), scored["btts"], scored["btts_y"]
    if kind.startswith("counts:"):
        target = kind.split(":", 1)[1]
        model = {t: values[f"counts.{target}.{t}"] for t in registry.COUNT_TERMS}
        yh = np.array([m.stats[target][0] for m in rows])
        ya = np.array([m.stats[target][1] for m in rows])
        sup, tot = ev.count_inputs(rows, values.get("soccer.rho", soccer.DC_RHO))
        ll = -ev.count_loglik(model, yh, ya, sup, tot)
        home, away = ev.count_means(model, sup, tot)
        return ll, (yh - home) ** 2 + (ya - away) ** 2, home, yh
    if kind == "nba":
        p, ok = nba_replay(rows, values)
        y = np.array([r["y"] for r in rows], dtype=float)
        ll = stats.log_loss(np.nan_to_num(p, nan=0.5), y)
        br = stats.brier(np.nan_to_num(p, nan=0.5), y)
        ll[~ok], br[~ok] = np.nan, np.nan
        return ll, br, p, y
    raise ValueError(f"Unknown data kind {kind!r}")


def pair(kind: str, rows: Sequence[Any], sport: str, champion: Mapping[str, float],
         challenger: Mapping[str, float]) -> ev.Paired:
    """Champion and challenger scored on the same rows (challenger minus champion)."""
    if kind == "results":
        return ev.pair_results(rows, sport, champion, challenger)
    la, ba, pa, y = losses(kind, rows, sport, champion)
    lb, bb, pb, _ = losses(kind, rows, sport, challenger)
    return ev._paired_from(la, ba, lb, bb, rows, pa, pb, y)


# --- the NBA model's weights (replayed from ai_predictions) ---------------------------------------------------------

def nba_replay(rows: Sequence[Mapping[str, Any]], values: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """The NBA model's home-win chance recomputed with other market/ESPN weights from each logged prediction (the
    margin blend and the moneyline anchor of ``nba_model.predict_game``); (chances, usable): a row is usable only when
    the replay at the weights it was logged with reproduces its logged chance (to 1e-6)."""
    base = nba_model.Params()
    wm = values.get("nba.market_weight", base.market_weight)
    we = values.get("nba.espn_weight", base.espn_weight)
    out = np.full(len(rows), np.nan)
    ok = np.zeros(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        q = row["payload"]
        here = _nba_chance(q, wm, we, base)
        then = _nba_chance(q, base.market_weight, base.espn_weight, base)
        if here is None or then is None or q.get("home_win") is None:
            continue
        if abs(then - float(q["home_win"])) < 1e-6:
            out[i], ok[i] = here, True
    return out, ok


def _nba_chance(q: Mapping[str, Any], wm: float, we: float, base: nba_model.Params) -> Optional[float]:
    total = q.get("total")
    if total is None:
        return None
    win_sd = base.win_sd_factor * math.sqrt(max(float(total), 150.0))
    margins = []
    if q.get("model_home_win") is not None:
        p = float(q["model_home_win"])
        if not 0 < p < 1:
            return None
        margins.append((win_sd * _N.inv_cdf(p), 1 - wm - we))
    market_margin = q.get("market_margin")
    if market_margin is not None:
        margins.append((float(market_margin), wm))
    if q.get("espn_home_win") is not None:
        margins.append((win_sd * _N.inv_cdf(min(max(float(q["espn_home_win"]), 0.01), 0.99)), we))
    weight = math.fsum(w for _, w in margins)
    if not margins or weight <= 0:
        return None
    margin = math.fsum(m * w for m, w in margins) / weight
    fair = q.get("market_home_win")
    if fair is not None and market_margin is not None:
        start = _N.inv_cdf(min(max(float(fair), 0.005), 0.995))
        return _N.cdf(start + (margin - float(market_margin)) / win_sd)
    return _N.cdf(margin / win_sd)


# --- the blend, vectorised (for fitting only; every score goes through the real blend) ------------------------------

@dataclass
class BlendArrays:
    """Games with a market source, as arrays: per pooled source its kind, its log chance per outcome and whether it is
    a prediction market inside the near-start window; the outcome index; low-confidence games apart."""

    logs: np.ndarray  # (G, S, 3) log chances (home, away, draw), 0 for padding / absent outcomes
    kind: np.ndarray  # (G, S) kind index into registry.KINDS, -1 for padding
    near_pm: np.ndarray  # (G, S) 1.0 for a prediction market near the start
    per_kind: np.ndarray  # (G, S) sources of the same kind in that game (1 for padding)
    outcomes: np.ndarray  # (G, 3) 1.0 where the outcome is in the game's pool
    y: np.ndarray  # (G,) index of the outcome that happened
    rows: List[Mapping[str, Any]]
    identifying: Dict[str, int]  # kind -> games where it sits beside another kind


OUT_INDEX = {"home": 0, "away": 1, "draw": 2}


def blend_arrays(rows: Sequence[Mapping[str, Any]], near_seconds: float = consensus.NEAR_START_SECONDS
                 ) -> BlendArrays:
    """The market-confidence games of ``rows`` (graded, outcome priced) as BlendArrays, built with the same rules as
    ``consensus.blend``: ok sources of a known kind, three-way when a market prices the draw, sources that price every
    outcome, chances clamped and renormalised, the near-start window on the prediction markets."""
    games = []
    for row in rows:
        ok = [s for s in row["sources"] if isinstance(s, Mapping) and s.get("status") == "ok"
              and s.get("kind") in consensus.KINDS]
        leading = [s for s in ok if s["kind"] in consensus.MARKET_KINDS] or ok
        outcomes = ("home", "away", "draw") if any(
            isinstance(s.get("p"), Mapping) and consensus._number(s["p"].get("draw")) is not None for s in leading
        ) else ("home", "away")
        pool = [(s, c) for s in ok if (c := consensus._normalised(s.get("p"), outcomes)) is not None]
        if not pool or not any(s["kind"] in consensus.MARKET_KINDS for s, _ in pool):
            continue
        if row.get("outcome") not in outcomes:
            continue
        near = consensus._near_start(row["logged_at"], row["start"], near_seconds) and any(
            s["kind"] == "prediction_market" for s, _ in pool)
        games.append((row, pool, outcomes, near))
    G = len(games)
    S = max((len(p) for _, p, _, _ in games), default=1)
    logs = np.zeros((G, S, 3))
    kind = np.full((G, S), -1, dtype=int)
    near_pm = np.zeros((G, S))
    per_kind = np.ones((G, S))
    outs = np.zeros((G, 3))
    y = np.zeros(G, dtype=int)
    identifying = {k: 0 for k in consensus.KINDS}
    for g, (row, pool, outcomes, near) in enumerate(games):
        counts: Dict[str, int] = {}
        for s, _ in pool:
            counts[s["kind"]] = counts.get(s["kind"], 0) + 1
        for j, (s, chances) in enumerate(pool):
            kind[g, j] = consensus.KINDS.index(s["kind"])
            per_kind[g, j] = counts[s["kind"]]
            near_pm[g, j] = 1.0 if near and s["kind"] == "prediction_market" else 0.0
            for o in outcomes:
                logs[g, j, OUT_INDEX[o]] = math.log(min(max(chances[o], consensus.P_FLOOR), 1 - consensus.P_FLOOR))
        for o in outcomes:
            outs[g, OUT_INDEX[o]] = 1.0
        y[g] = OUT_INDEX[row["outcome"]]
        if len(counts) > 1:
            for k in counts:
                identifying[k] += 1
    return BlendArrays(logs, kind, near_pm, per_kind, outs, y, [g[0] for g in games], identifying)


def blend_probs(arr: BlendArrays, weights: np.ndarray, factor: float = consensus.NEAR_START_FACTOR,
                sharpness: float = 1.0) -> np.ndarray:
    """(G, 3) pooled chances for kind weights (5,), as ``consensus.blend`` computes them."""
    if arr.kind.size == 0:
        return np.zeros((0, 3))
    w = np.where(arr.kind >= 0, np.maximum(weights, 0.0)[np.maximum(arr.kind, 0)], 0.0)
    w = w * np.where(arr.near_pm > 0, factor, 1.0) / arr.per_kind
    total = w.sum(axis=1, keepdims=True)
    count = (arr.kind >= 0).sum(axis=1, keepdims=True)
    shares = np.where(total > 0, w / np.where(total > 0, total, 1.0), (arr.kind >= 0) / np.maximum(count, 1))
    z = sharpness * np.einsum("gs,gsk->gk", shares, arr.logs)
    z = np.where(arr.outcomes > 0, z, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def blend_nll(arr: BlendArrays, weights: np.ndarray, factor: float = consensus.NEAR_START_FACTOR,
              sharpness: float = 1.0) -> np.ndarray:
    p = blend_probs(arr, weights, factor, sharpness)
    return -np.log(np.maximum(p[np.arange(len(arr.y)), arr.y], consensus.P_FLOOR))


def verify_blend(arr: BlendArrays, values: Mapping[str, float], sport: str, sample: int = 400) -> float:
    """The largest gap between the vectorised blend and the real ``consensus.blend`` on up to ``sample`` games at
    ``values`` (the fit is refused when it isn't ~0)."""
    if not arr.rows:
        return 0.0
    w = np.array([values[f"consensus.weight.{k}"] for k in consensus.KINDS])
    p = blend_probs(arr, w, values["consensus.near_start_factor"], values["consensus.sharpness"])
    kw = ev.blend_kwargs(sport, values)
    step = max(1, len(arr.rows) // sample)
    worst = 0.0
    for g in range(0, len(arr.rows), step):
        real = ev.reblend(arr.rows[g], kw)
        if real is None:
            return math.inf
        for o, value in real.items():
            worst = max(worst, abs(value - p[g, OUT_INDEX[o]]))
    return worst


def _softmax(eta: np.ndarray) -> np.ndarray:
    e = np.exp(eta - eta.max())
    return e / e.sum()


def fit_weights(arr: BlendArrays, champion: Mapping[str, float], *, centre: Optional[np.ndarray] = None,
                lam: float = RIDGE["weights"], lam_pool: float = RIDGE["pool"]) -> Dict[str, Any]:
    """Penalised maximum likelihood of the kind weights through the vectorised blend: weights = softmax(eta), with
    (lam/2)||eta - eta_champion||^2 (+ (lam_pool/2)||eta - centre||^2 toward the all-sports fit), kinds with fewer than
    MIN_ID identifying games held at the champion, and the first free kind as the reference (only ratios between
    weights are seen by the pool, so one log weight is pinned). Standard errors of the weights come from the inverse
    Hessian through the softmax (delta method), and each weight takes only its 95%-sure move (the LCB step).
    Returns {"weights" (new, before the step clip), "fit" (unshrunk), "se", "n", "free"}."""
    kinds = consensus.KINDS
    w0 = np.array([max(champion[f"consensus.weight.{k}"], 1e-3) for k in kinds])
    w0 = w0 / w0.sum()
    eta0 = np.log(w0)
    free = [i for i, k in enumerate(kinds) if arr.identifying.get(k, 0) >= MIN_ID]
    factor, a = champion["consensus.near_start_factor"], champion["consensus.sharpness"]
    n = len(arr.y)
    if n == 0 or len(free) < 2:
        return {"weights": w0, "fit": w0, "se": np.full(len(kinds), np.inf), "n": n, "free": [kinds[i] for i in free]}

    est = free[1:]  # free[0] is the reference
    if centre is not None:
        # Only ratios count (softmax ignores a common shift), and the reference's log weight is pinned at the
        # champion's: put the pooled centre on that same level, so the pull is toward the pooled fit's ratios
        centre = np.asarray(centre, dtype=float)
        centre = centre - centre[free[0]] + eta0[free[0]]

    def full(x: np.ndarray) -> np.ndarray:
        eta = eta0.copy()
        eta[est] = x
        return eta

    def objective(x: np.ndarray) -> float:
        eta = full(x)
        value = float(blend_nll(arr, _softmax(eta), factor, a).sum()) + lam / 2 * float(np.sum((eta - eta0) ** 2))
        if centre is not None and lam_pool > 0:
            value += lam_pool / 2 * float(np.sum((eta - centre) ** 2))
        return value

    x_hat, hess, _ = stats.newton(objective, eta0[est])
    w_hat = _softmax(full(x_hat))
    try:
        cov_est = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov_est = np.full((len(est), len(est)), np.inf)
    cov = np.zeros((len(kinds), len(kinds)))
    cov[np.ix_(est, est)] = cov_est
    jac = np.diag(w_hat) - np.outer(w_hat, w_hat)
    var_w = np.diag(jac @ cov @ jac.T)
    se = np.where(np.isfinite(var_w) & (var_w >= 0), np.sqrt(np.abs(var_w)), np.inf)
    se[[i for i in range(len(kinds)) if i not in free]] = np.inf  # held kinds don't move
    stepped = np.maximum(stats.lcb_step(w0, w_hat, se), 0.0)
    held = [i for i in range(len(kinds)) if i not in free]
    stepped[held] = w0[held]  # kinds the data can't see keep their share; the free ones share the rest
    room = 1.0 - float(w0[held].sum())
    if stepped[free].sum() > 0:
        stepped[free] *= room / stepped[free].sum()
    return {"weights": stepped, "fit": w_hat, "se": se, "n": n, "free": [kinds[i] for i in free]}


def fit_scalar(nll: Callable[[float], float], start: float, low: float, high: float) -> Tuple[float, float]:
    """(estimate, standard error) of a one-parameter negative log likelihood on [low, high]: golden section, then the
    curvature at the estimate."""
    x = stats.golden(nll, low, high, tol=1e-6)
    h = max(1e-4 * max(abs(x), 1.0), 1e-6)
    lo_x, hi_x = max(low, x - h), min(high, x + h)
    mid = (lo_x + hi_x) / 2
    curv = (nll(hi_x) - 2 * nll(mid) + nll(lo_x)) / ((hi_x - lo_x) / 2) ** 2
    se = 1 / math.sqrt(curv) if curv > 0 else math.inf
    return x, se


def fit_rho(matches: Sequence[ev.MatchRecord], r0: float, low: float, high: float, nudge: float = soccer.BTTS_NUDGE,
            span: float = 0.06, points: int = 7) -> Tuple[float, float]:
    """Dixon-Coles rho by profile likelihood of the exact 90-minute scores: the rates are re-derived from the market
    at each rho on a grid of ``points`` around ``r0`` (+-``span``, inside the bounds), and a parabola through the
    profile gives the estimate and its standard error (inf when the profile isn't concave)."""
    grid = np.linspace(max(low, r0 - span), min(high, r0 + span), points)
    start = ev.rates_many(matches, r0)
    nll = np.array([float(ev.goals_scores(matches, float(r), nudge, start=start)["exact_ls"].sum()) for r in grid])
    a, b, _ = np.polyfit(grid - r0, nll, 2)
    if not a > 0:
        return r0, math.inf
    r_hat = float(min(max(r0 - b / (2 * a), low), high))
    return r_hat, 1 / math.sqrt(2 * a)


def lcb_scalar(old: float, fit: float, se: float) -> float:
    return float(stats.lcb_step([old], [fit], [se])[0])


# --- one family's fit on a training window -----------------------------------------------------------------------

@dataclass
class Fit:
    """A refit's proposed values (complete set, already LCB-stepped, step-clipped and bounded), what it moved and
    why."""

    family: str
    values: Dict[str, float]
    knobs: List[str]
    train_n: int
    note: str
    structural: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


FAMILIES: Dict[str, Tuple[str, ...]] = {
    "weights": registry.WEIGHT_KNOBS,
    "sharpness": ("consensus.sharpness",),
    "near_start": ("consensus.near_start_factor",),
    "shrink": ("consensus.low_confidence_shrink",),
    "rho": ("soccer.rho",),
    "btts": ("soccer.btts_nudge",),
    "adjust_weight": ("context.adjust_weight",),
    "home_prior": ("context.home_prior", "context.draw_prior"),
    "total_sd": ("context.total_sd",),
    "margin_sd": ("context.margin_sd",),
    "nba": ("nba.market_weight", "nba.espn_weight"),
    "counts.corners": tuple(f"counts.corners.{t}" for t in registry.COUNT_TERMS),
    "counts.cards": tuple(f"counts.cards.{t}" for t in registry.COUNT_TERMS),
}


def family_kind(family: str) -> str:
    return data_kind(FAMILIES[family][0])


def families_for(sport: str) -> List[str]:
    return [f for f, knobs in FAMILIES.items() if registry.allowed(knobs[0], sport)]


def _finish(family: str, sport: str, champion: Mapping[str, float], wanted: Mapping[str, float], train_n: int,
            note: str, details: Optional[Dict[str, Any]] = None) -> Fit:
    values, cut = registry.clip_step(sport, champion, wanted)
    knobs = registry.changed(sport, champion, values, tol=1e-9)
    structural = any(registry.KNOBS[k].structural for k in knobs)
    return Fit(family, values, knobs, train_n, note, structural, {**(details or {}), "cut": cut})


def refit(family: str, sport: str, rows: Sequence[Any], champion: Mapping[str, float], *,
          pooled: Optional[np.ndarray] = None) -> Optional[Fit]:
    """One family's penalised, LCB-stepped, step-clipped refit on training ``rows`` (the family's data kind); None
    when there is too little to fit."""
    n = len(rows)
    if family == "weights":
        arr = blend_arrays(rows, champion["consensus.near_start_seconds"])
        if len(arr.y) < MIN_TRAIN:
            return None
        got = fit_weights(arr, champion, centre=pooled)
        wanted = {f"consensus.weight.{k}": float(v) for k, v in zip(consensus.KINDS, got["weights"])}
        fit = _finish(family, sport, champion, wanted, len(arr.y),
                      "kind weights refitted through the blend, ridge to the current values",
                      {"fit": dict(zip(consensus.KINDS, got["fit"].tolist())), "free": got["free"],
                       "identifying": arr.identifying})
        if fit.knobs:
            gap = verify_blend(arr, fit.values, sport)
            if gap > 1e-9:
                fit.details["refused"] = f"the vectorised blend differs from the real one by {gap:.2g}"
                fit.knobs = []
        return fit
    if family == "sharpness":
        arr = blend_arrays(rows, champion["consensus.near_start_seconds"])
        if len(arr.y) < MIN_TRAIN:
            return None
        w = np.array([champion[f"consensus.weight.{k}"] for k in consensus.KINDS])
        a0 = champion["consensus.sharpness"]
        lam = RIDGE["sharpness"]
        nll = lambda a: float(blend_nll(arr, w, champion["consensus.near_start_factor"], a).sum()) + lam / 2 * (  # noqa: E731
            a - a0) ** 2
        low, high = registry.KNOBS["consensus.sharpness"].bounds(sport)
        a_hat, se = fit_scalar(nll, a0, low, high)
        return _finish(family, sport, champion, {"consensus.sharpness": lcb_scalar(a0, a_hat, se)}, len(arr.y),
                       f"pool sharpness fitted at {a_hat:.3f} (standard error {se:.3f})", {"fit": a_hat, "se": se})
    if family == "near_start":
        arr = blend_arrays(rows, champion["consensus.near_start_seconds"])
        near = np.where(arr.near_pm.max(axis=1) > 0)[0] if arr.kind.size else np.zeros(0, dtype=int)
        if near.size < 100:
            return None
        sub = BlendArrays(arr.logs[near], arr.kind[near], arr.near_pm[near], arr.per_kind[near], arr.outcomes[near],
                          arr.y[near], [arr.rows[i] for i in near], arr.identifying)
        w = np.array([champion[f"consensus.weight.{k}"] for k in consensus.KINDS])
        f0 = champion["consensus.near_start_factor"]
        nll = lambda f: float(blend_nll(sub, w, f, champion["consensus.sharpness"]).sum())  # noqa: E731
        low, high = registry.KNOBS["consensus.near_start_factor"].bounds(sport)
        f_hat, se = fit_scalar(nll, f0, low, high)
        return _finish(family, sport, champion, {"consensus.near_start_factor": lcb_scalar(f0, f_hat, se)},
                       int(near.size), f"near-start factor fitted at {f_hat:.2f} on {near.size} near-start games",
                       {"fit": f_hat, "se": se})
    if family == "shrink":
        low_rows = [r for r in rows if (r.get("blend") or {}).get("confidence") == "low" and r.get("outcome")]
        if len(low_rows) < 100:
            return None
        pools, bases, ys = [], [], []
        for r in low_rows:
            raw = consensus.blend(r["sources"], r["league"], now=r["logged_at"], start=r["start"],
                                  neutral=r.get("neutral", False), low_confidence_shrink=0.0)
            if raw is None or r["outcome"] not in raw["p"] or raw["base"] is None:
                continue
            pools.append(raw["p"][r["outcome"]])
            bases.append(raw["base"][r["outcome"]])
            ys.append(1)
        if len(pools) < 100:
            return None
        pool_p, base_p = np.array(pools), np.array(bases)
        s0 = champion["consensus.low_confidence_shrink"]
        nll = lambda s: float(-np.log(np.maximum((1 - s) * pool_p + s * base_p, consensus.P_FLOOR)).sum())  # noqa: E731
        low, high = registry.KNOBS["consensus.low_confidence_shrink"].bounds(sport)
        s_hat, se = fit_scalar(nll, s0, low, high)
        return _finish(family, sport, champion, {"consensus.low_confidence_shrink": lcb_scalar(s0, s_hat, se)},
                       len(pools), f"low-confidence shrink fitted at {s_hat:.2f}", {"fit": s_hat, "se": se})
    if family == "rho":
        if n < MIN_TRAIN:
            return None
        r0 = champion["soccer.rho"]
        low, high = registry.KNOBS["soccer.rho"].bounds(sport)
        r_hat, se = fit_rho(rows, r0, low, high, champion["soccer.btts_nudge"])
        return _finish(family, sport, champion, {"soccer.rho": lcb_scalar(r0, r_hat, se)}, n,
                       f"Dixon-Coles rho fitted at {r_hat:.3f} (standard error {se:.3f}) on the exact scores",
                       {"fit": r_hat, "se": se})
    if family == "btts":
        if n < MIN_TRAIN:
            return None
        scored = ev.goals_scores(rows, champion["soccer.rho"], 0.0)
        raw, yy = scored["btts_raw"], scored["btts_y"]
        n0 = champion["soccer.btts_nudge"]
        nll = lambda b: float(stats.log_loss(np.clip(raw + b, 0.01, 0.99), yy).sum())  # noqa: E731
        low, high = registry.KNOBS["soccer.btts_nudge"].bounds(sport)
        b_hat, se = fit_scalar(nll, n0, low, high)
        return _finish(family, sport, champion, {"soccer.btts_nudge": lcb_scalar(n0, b_hat, se)}, n,
                       f"BTTS nudge fitted at {b_hat * 100:+.1f} points", {"fit": b_hat, "se": se})
    if family == "adjust_weight":
        adjusted = [r for r in rows if r["applied"] and r.get("total") is not None]
        if len(adjusted) < 30:
            return None
        slope = ev.residual_slope(adjusted)
        if slope["beta"] is None:
            return None
        w0 = champion["context.adjust_weight"]
        return _finish(family, sport, champion,
                       {"context.adjust_weight": lcb_scalar(w0, w0 * slope["beta"], w0 * slope["se"])}, len(adjusted),
                       f"the results showed {slope['beta']:.2f} of the applied shift (standard error "
                       f"{slope['se']:.2f})", {"slope": slope})
    if family == "home_prior":
        if n < MIN_TRAIN:
            return None
        outcomes = [r.get("outcome") for r in rows]
        wanted, details = {}, {}
        for name, hit in (("context.home_prior", "home"), ("context.draw_prior", "draw")):
            if name not in champion:
                continue
            pool = [o for o in outcomes if o in ("home", "away", "draw")] if sport == "soccer" else [
                o for o in outcomes if o in ("home", "away")]
            if len(pool) < MIN_TRAIN:
                continue
            p0 = champion[name]
            hits = sum(1 for o in pool if o == hit)
            p_hat = (hits + PRIOR_GAMES * p0) / (len(pool) + PRIOR_GAMES)
            se = math.sqrt(max(p_hat * (1 - p_hat), 1e-6) / (len(pool) + PRIOR_GAMES))
            wanted[name] = lcb_scalar(p0, p_hat, se)
            details[name] = {"rate": hits / len(pool), "shrunk": p_hat, "se": se, "n": len(pool)}
        if not wanted:
            return None
        return _finish(family, sport, champion, wanted, n, "home and draw rates, shrunk toward the current priors",
                       details)
    if family in ("total_sd", "margin_sd"):
        name = f"context.{family}"
        r = np.array([x["residual"] for x in rows], dtype=float)
        if r.size < MIN_TRAIN:
            return None
        sd0 = champion[name]
        sd_hat = float(math.sqrt(np.mean(r * r)))
        se = sd_hat / math.sqrt(2 * r.size)
        return _finish(family, sport, champion, {name: lcb_scalar(sd0, sd_hat, se)}, int(r.size),
                       f"spread of results {sd_hat:.2f} (standard error {se:.2f})", {"fit": sd_hat, "se": se})
    if family == "nba":
        p, ok = nba_replay(rows, champion)
        usable = [r for r, good in zip(rows, ok) if good]
        if len(usable) < MIN_TRAIN:
            return None
        y = np.array([r["y"] for r in usable], dtype=float)
        x0 = np.array([champion["nba.market_weight"], champion["nba.espn_weight"]])
        lam = RIDGE["nba"]

        def objective(x: np.ndarray) -> float:
            values = {"nba.market_weight": float(x[0]), "nba.espn_weight": float(x[1])}
            q, _ = nba_replay(usable, values)
            q = np.nan_to_num(q, nan=0.5)
            return float(stats.log_loss(q, y).sum()) + lam / 2 * float(np.sum((x - x0) ** 2))

        x_hat, hess, _ = stats.newton(objective, x0, iters=10)
        se = stats.standard_errors(hess)
        stepped = stats.lcb_step(x0, x_hat, se)
        return _finish(family, sport, champion, {"nba.market_weight": float(stepped[0]),
                                                 "nba.espn_weight": float(stepped[1])}, len(usable),
                       "NBA model weights refitted on its graded predictions", {"fit": x_hat.tolist()})
    if family.startswith("counts."):
        target = family.split(".")[1]
        have = [m for m in rows if target in m.stats]
        if len(have) < MIN_TRAIN:
            return None
        yh = np.array([m.stats[target][0] for m in have], dtype=float)
        ya = np.array([m.stats[target][1] for m in have], dtype=float)
        sup, tot = ev.count_inputs(have, champion.get("soccer.rho", soccer.DC_RHO))
        terms = registry.COUNT_TERMS
        x0 = np.array([champion[f"counts.{target}.{t}"] if t != "size" else math.log(champion[f"counts.{target}.size"])
                       for t in terms])
        lam = RIDGE["counts"]

        def objective(x: np.ndarray) -> float:
            model = dict(zip(terms, x.tolist()))
            model["size"] = math.exp(min(max(x[-1], -0.7), 7.0))  # dispersion kept within 0.5-1100
            return float(-ev.count_loglik(model, yh, ya, sup, tot).sum()) + lam / 2 * float(np.sum((x - x0) ** 2))

        x_hat, hess, _ = stats.newton(objective, x0, iters=40)
        se = stats.standard_errors(hess)
        stepped = stats.lcb_step(x0, x_hat, se)
        wanted = {f"counts.{target}.{t}": float(v) for t, v in zip(terms[:-1], stepped[:-1])}
        wanted[f"counts.{target}.size"] = float(math.exp(stepped[-1]))
        return _finish(family, sport, champion, wanted, len(have),
                       f"{target} count model refitted (negative binomial) on {len(have)} matches",
                       {"fit": dict(zip(terms, x_hat.tolist())), "se": se.tolist()})
    raise ValueError(f"Unknown family {family!r}")


def pooled_centre(corpus: ev.Corpus, champions: Mapping[str, Mapping[str, float]],
                  until: Optional[float] = None) -> Optional[np.ndarray]:
    """The all-sports weight fit (eta, the log of the weights): the centre each sport's fit is pulled toward."""
    rows = [r for r in corpus.consensus if until is None or r["start"] < until]
    if len(rows) < MIN_TRAIN:
        return None
    arr = blend_arrays(rows)
    if len(arr.y) < MIN_TRAIN:
        return None
    base = registry.defaults("other")
    got = fit_weights(arr, base, lam=RIDGE["weights"], lam_pool=0.0)
    return np.log(np.maximum(got["fit"], 1e-6))


def _pooled(corpus: ev.Corpus, until: float, cache: Optional[Dict[float, Optional[np.ndarray]]]
            ) -> Optional[np.ndarray]:
    if cache is not None and until in cache:
        return cache[until]
    centre = pooled_centre(corpus, {}, until=until)
    if cache is not None:
        cache[until] = centre
    return centre


def data_health(corpus: ev.Corpus, sport: str, now: float) -> List[str]:
    """Reasons to pause refits for a sport (fitting on a broken feed teaches the brain the outage): a source kind
    whose share of ok statuses halved last week against the week before (each with at least 20 games showing it)."""
    week_end = stats.week_bounds(stats.week_of(now))[0]
    last, before = (week_end - 7 * 86400, week_end), (week_end - 14 * 86400, week_end - 7 * 86400)
    problems = []
    rows = corpus.sport(sport)
    for kind in consensus.KINDS:
        shares = []
        for lo, hi in (before, last):
            seen = ok = 0
            for r in rows:
                if not lo <= r["start"] < hi:
                    continue
                mine = [s for s in r["sources"] if isinstance(s, Mapping) and s.get("kind") == kind]
                if mine:
                    seen += 1
                    ok += any(s.get("status") == "ok" for s in mine)
            shares.append((seen, ok / seen if seen else None))
        (n0, s0), (n1, s1) = shares
        if n0 >= 20 and n1 >= 20 and s0 and s1 is not None and s1 < s0 / 2:
            problems.append(f"{kind.replace('_', ' ')} prices were usable on {s1:.0%} of games last week against "
                            f"{s0:.0%} the week before")
    return problems


# --- rolling-origin replay of a refit procedure ---------------------------------------------------------------------

def rolling(family: str, sport: str, rows: Sequence[Any], champion: Mapping[str, float], *,
            weeks: int = EVAL_WEEKS, block: int = BLOCK_WEEKS, pooled_rows: Optional[Sequence[Any]] = None,
            cache: Optional[Dict[float, Optional[np.ndarray]]] = None
            ) -> Tuple[Optional[ev.Paired], List[Dict[str, Any]]]:
    """Out-of-sample evidence for a refit procedure: over the last ``weeks`` weeks, in blocks of ``block`` weeks,
    refit on every game before the block and score the block against the champion. Returns (the pooled pairing,
    the per-block fits)."""
    if not rows:
        return None, []
    start_of = (lambda r: r["start"]) if isinstance(rows[0], Mapping) else (lambda r: r.start)
    folds = list(ev.weekly_folds(rows, min_train=MIN_TRAIN, weeks=weeks, start=start_of))
    if not folds:
        return None, []
    kind = family_kind(family)
    pieces: List[ev.Paired] = []
    fits = []
    for i in range(0, len(folds), block):
        chunk = folds[i:i + block]
        train = chunk[0][1]
        test = [r for _, _, week_rows in chunk for r in week_rows]
        problems = ev.leakage_problems(train, test)
        if problems:
            raise AssertionError(f"leakage in the rolling replay of {family}: {problems[0]}")
        centre = None
        if family == "weights" and pooled_rows is not None:
            begin = stats.week_bounds(chunk[0][0])[0]
            centre = _pooled(ev.Corpus(consensus=list(pooled_rows)), begin, cache)
        fit = refit(family, sport, train, champion, pooled=centre)
        if fit is None:
            continue
        fits.append({"from_week": chunk[0][0], "to_week": chunk[-1][0], "train_n": fit.train_n, "knobs": fit.knobs,
                     "values": {k: fit.values[k] for k in FAMILIES[family] if k in fit.values}})
        pieces.append(pair(kind, test, sport, champion, fit.values))
    if not pieces:
        return None, fits
    return _concat(pieces), fits


def _concat(pieces: Sequence[ev.Paired]) -> ev.Paired:
    return ev.Paired(np.concatenate([p.d_ll for p in pieces]), np.concatenate([p.d_brier for p in pieces]),
                     [d for p in pieces for d in p.days], np.concatenate([p.starts for p in pieces]),
                     np.concatenate([p.p_champion for p in pieces]), np.concatenate([p.p_challenger for p in pieces]),
                     np.concatenate([p.y for p in pieces]))


# --- gates ---------------------------------------------------------------------------------------------------------

def _calibration_ok(evidence: Mapping[str, Any]) -> bool:
    """The challenger's calibration slope range still holds 1, or its slope is no further from 1 than the
    champion's."""
    new, old = evidence.get("calibration_challenger") or {}, evidence.get("calibration_champion") or {}
    if new.get("slope") is None:
        return True
    if new["slope_lo"] <= 1 <= new["slope_hi"]:
        return True
    return old.get("slope") is not None and abs(new["slope"] - 1) <= abs(old["slope"] - 1)


CHANCE_KINDS = ("results", "totals", "home", "btts", "nba")  # scored as chances against a market or a base rate


def leak_suspect(evidence: Mapping[str, Any], kind: str = "results") -> bool:
    """A claimed gain past what a market-anchored chance can plausibly gain (0.01 log loss, 0.005 Brier): a data bug
    until proven otherwise. Count and spread models have no market to anchor them, so the check is for chances."""
    if kind not in CHANCE_KINDS:
        return False
    dll, dbrier = evidence.get("dll"), evidence.get("dbrier")
    return (dll is not None and dll < -LEAK_LL) or (dbrier is not None and dbrier < -LEAK_BRIER)


def gate_nudge(evidence: Mapping[str, Any], *, min_eval: int = MIN_EVAL, kind: str = "results") -> Tuple[str, str]:
    """("promoted" | "held" | "rejected", the reason in plain words) for a nudge's out-of-sample evidence."""
    n = evidence.get("n") or 0
    if n < min_eval:
        return "held", f"Not enough evidence yet: about {min_eval - n} more graded games needed ({n} of {min_eval})."
    if leak_suspect(evidence, kind):
        return "rejected", "The gain is too big to be real: treated as a data problem until checked."
    if evidence.get("hi") is None:
        return "held", "Not enough separate days to judge yet."
    if evidence["hi"] > NONINF_MARGIN:
        return "held", (f"Could be worse: the upper end of the 95% range ({stats.signed(evidence['hi'])}) is above "
                        f"the {NONINF_MARGIN:+.4f} allowed.")
    if (evidence.get("dbrier") or 0.0) > 0:
        return "held", f"Brier score got worse ({stats.signed(evidence['dbrier'])})."
    if not _calibration_ok(evidence):
        return "held", "Calibration would get worse."
    return "promoted", (f"Out of sample on {n} games the change in log loss was {stats.signed(evidence['dll'])} "
                        f"(95% range {stats.signed(evidence['lo'])} to {stats.signed(evidence['hi'])}), within the "
                        "allowed margin.")


def gate_structural(evidence: Mapping[str, Any], holm_p: Optional[float], *, alpha: float = ALPHA,
                    min_eval: int = MIN_EVAL, kind: str = "results") -> Tuple[str, str]:
    """("pass" | "held" | "rejected", the reason) for a structural candidate's superiority test (before shadow)."""
    n = evidence.get("n") or 0
    if n < min_eval:
        return "held", f"Not enough evidence yet: about {min_eval - n} more graded games needed ({n} of {min_eval})."
    if leak_suspect(evidence, kind):
        return "rejected", "The gain is too big to be real: treated as a data problem until checked."
    if evidence.get("p") is None or evidence.get("hi") is None:
        return "held", "Not enough separate days to judge yet."
    if holm_p is None or holm_p > alpha:
        return "held", (f"Not clearly better: one-sided test p = {evidence['p']:.3f} (Holm-adjusted "
                        f"{(holm_p if holm_p is not None else 1):.3f}, needs {alpha}).")
    if evidence["hi"] >= 0:
        return "held", f"Not clearly better: the 95% range reaches {stats.signed(evidence['hi'])}."
    if (evidence.get("dbrier") or 0.0) > 0:
        return "held", f"Brier score got worse ({stats.signed(evidence['dbrier'])})."
    if not _calibration_ok(evidence):
        return "held", "Calibration would get worse."
    return "pass", (f"Better on {n} games: change in log loss {stats.signed(evidence['dll'])} (95% range "
                    f"{stats.signed(evidence['lo'])} to {stats.signed(evidence['hi'])}), p = {evidence['p']:.3f} "
                    f"(Holm {holm_p:.3f}).")


def evidence_of(paired: Optional[ev.Paired], seed: int = 0) -> Dict[str, Any]:
    if paired is None or paired.n == 0:
        return {"n": 0, "days": 0, "dll": None, "lo": None, "hi": None, "dbrier": None, "p": None}
    return ev.compare(paired, reps=BOOT_REPS, seed=seed)


# --- AI proposals --------------------------------------------------------------------------------------------------

def check_proposal(proposal: Mapping[str, Any], *, known_evidence: Iterable[str] = (),
                   ledger: Sequence[Mapping[str, Any]] = ()) -> Optional[str]:
    """The code-side checks of method.md 6.2: None when the proposal may be frozen, else why not (plain words)."""
    knob = proposal.get("knob")
    sport = registry.sport_key(proposal.get("sport"))
    # The reasons name nothing the AI wrote (they are shown on the page as the gate's words)
    if not isinstance(knob, str) or knob not in registry.KNOBS:
        return "the setting isn't an allowed one"
    if not registry.allowed(knob, sport) or (proposal.get("sport") and sport == "other"
                                             and str(proposal.get("sport")).lower() not in ("other", "")):
        return f"{registry.KNOBS[knob].label} doesn't apply to that sport"
    if proposal.get("direction") not in ("increase", "decrease"):
        return "direction must be increase or decrease"
    try:
        size = float(proposal.get("size"))
    except (TypeError, ValueError):
        return "size must be a number"
    k = registry.KNOBS[knob]
    current = registry.default_value(knob, sport)
    if not 0 < size <= k.max_step(sport, current) + 1e-12:
        return f"the size is outside one cycle's step (at most {k.max_step(sport, current):g})"
    known = set(known_evidence)
    if known:
        missing = [e for e in proposal.get("evidence") or [] if e not in known]
        if missing:
            return (f"{len(missing)} evidence id{'s' if len(missing) != 1 else ''} "
                    f"{'aren' if len(missing) != 1 else 'isn'}'t in the fact sheet")
    for row in ledger:
        if row.get("knob") == knob and row.get("sport") == sport and str(row.get("source", "")).startswith("ai:") \
                and (row.get("details") or {}).get("direction") == proposal.get("direction"):
            return "the same change was tried in the last 8 weeks"
    return None


def freeze_proposal(db: sqlite3.Connection, proposal: Mapping[str, Any], corpus: ev.Corpus, *, provider: str,
                    now: float, known_evidence: Iterable[str] = ()) -> Dict[str, Any]:
    """Check an AI proposal and freeze it as a challenger: the size within the proposed direction and bound is set by
    the LCB rule on games before ``now`` (the AI never supplies the final value); it will be judged only on games
    that start after ``now``. Returns {"version"} or {"rejected": reason}; both are ledger rows."""
    sport = registry.sport_key(proposal.get("sport"))
    week = stats.week_of(now)
    since = now - 8 * 7 * 86400
    reason = check_proposal(proposal, known_evidence=known_evidence,
                            ledger=bs.trials(db, sport=sport, since=since, limit=None))
    knob = str(proposal.get("knob"))
    source = f"ai:{provider}"
    base = {"week": week, "sport": sport, "knob": knob, "source": source, "gate": "structural",
            "details": {"proposal": dict(proposal), "direction": proposal.get("direction")}}
    if reason:
        bs.add_trial(db, {**base, "decision": "rejected", "reason": f"Proposal refused: {reason}."}, now)
        return {"rejected": reason}
    champion = _champion_values(db, sport)
    old = champion[knob]
    size = float(proposal["size"])
    sign = 1.0 if proposal["direction"] == "increase" else -1.0
    k = registry.KNOBS[knob]
    low, high = k.bounds(sport)
    bound = min(max(old + sign * size, low), high)
    kind = data_kind(knob)
    rows = [r for r in rows_for(kind, corpus, sport) if (r["start"] if isinstance(r, Mapping) else r.start) < now]
    if len(rows) < 30:
        bs.add_trial(db, {**base, "old": old, "decision": "rejected", "train_n": len(rows),
                          "reason": "Proposal refused: too few graded games before it to size the change."}, now)
        return {"rejected": "too few games"}

    def nll(theta: float) -> float:
        values = dict(champion)
        values[knob] = theta
        ll = losses(kind, rows, sport, values)[0]
        return float(np.nansum(ll))

    # Fit over the knob's range on the proposed side (only values the registry accepts: raising one blend weight
    # can't take the market kinds under their floor), keep the 95%-sure part, then cap it at the proposed size
    far = feasible_end(sport, champion, knob, high if sign > 0 else low)
    lo_t, hi_t = (old, far) if sign > 0 else (far, old)
    try:
        if abs(hi_t - lo_t) <= 1e-6:
            raise ValueError("no room to move it that way")
        theta, se = fit_scalar(nll, old, lo_t, hi_t)
        new = lcb_scalar(old, theta, se)
        new = min(new, bound, far) if sign > 0 else max(new, bound, far)
        if sign * (new - old) <= 1e-12:
            bs.add_trial(db, {**base, "old": old, "new": old, "decision": "rejected", "train_n": len(rows),
                              "reason": "Proposal refused: the games before it give no 95%-sure move in that "
                                        "direction."}, now)
            return {"rejected": "no sure move"}
        values, _ = registry.clip_step(sport, champion, {knob: new})
    except ValueError as exc:  # a move the registry's rules don't allow: refused, never a crash
        bs.add_trial(db, {**base, "old": old, "decision": "rejected", "train_n": len(rows),
                          "reason": f"Proposal refused: the setting can't move that way within the allowed values "
                                    f"({exc})."}, now)
        return {"rejected": "not allowed"}
    version = bs.insert_version(db, sport, values, status="challenger", parent=_champion_version(db, sport),
                                source=source, evidence={"proposal": dict(proposal), "fit": theta, "se": se,
                                                         "plan": {"looks": ai_looks(), "shadow_weeks": SHADOW_WEEKS,
                                                                  "alphas": list(AI_LOOK_ALPHAS)}},
                                effective_from=now, now=now, knob=",".join(registry.changed(sport, champion, values)),
                                gate="structural")
    bs.add_trial(db, {**base, "old": old, "new": values[knob], "train_n": len(rows), "decision": "shadow",
                      "version": version, "reason": "Frozen as a challenger: judged only on games after today."}, now)
    return {"version": version, "value": values[knob]}


def feasible_end(sport: str, champion: Mapping[str, float], knob: str, end: float) -> float:
    """The furthest value toward ``end`` (from the champion's) that ``registry.check`` accepts with only ``knob``
    moved: the joint rules (market kinds at least MARKET_FLOOR of the blend weight, NBA market plus ESPN at most 1)
    are monotone along the move, so a bisection finds the edge."""
    old = champion[knob]

    def ok(value: float) -> bool:
        try:
            registry.check(sport, {**champion, knob: value})
            return True
        except ValueError:
            return False

    if ok(end):
        return end
    good, bad = old, end
    for _ in range(60):
        mid = (good + bad) / 2
        if ok(mid):
            good = mid
        else:
            bad = mid
    return good


def freeze_proposals(db: sqlite3.Connection, proposals: Sequence[Mapping[str, Any]], corpus: ev.Corpus, *,
                     provider: str, now: float, known_evidence: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """``freeze_proposal`` for one AI call's proposals: at most MAX_AI are considered, the rest are refused."""
    out = []
    known = list(known_evidence)
    for i, proposal in enumerate(proposals):
        if i >= MAX_AI:
            sport = registry.sport_key(proposal.get("sport"))
            bs.add_trial(db, {"week": stats.week_of(now), "sport": sport, "knob": str(proposal.get("knob")),
                              "source": f"ai:{provider}", "gate": "structural", "decision": "rejected",
                              "reason": f"Proposal refused: at most {MAX_AI} proposals are taken from one call.",
                              "details": {"proposal": dict(proposal)}}, now)
            out.append({"rejected": "too many proposals"})
            continue
        out.append(freeze_proposal(db, proposal, corpus, provider=provider, now=now, known_evidence=known))
    return out


# --- store helpers -------------------------------------------------------------------------------------------------

def _champion_values(db: sqlite3.Connection, sport: str) -> Dict[str, float]:
    row = bs.champion(db, sport)
    if row is None or bs.safe_mode(db):
        return registry.defaults(sport)
    stored = {k: v for k, v in (row.get("params") or {}).items() if k in registry.KNOBS
              and registry.KNOBS[k].applies(sport)}
    try:
        return registry.check(sport, stored)
    except ValueError:
        return registry.defaults(sport)


def _champion_version(db: sqlite3.Connection, sport: str) -> int:
    row = bs.champion(db, sport)
    return int(row["version"]) if row else 0


def _values_of(db: sqlite3.Connection, version: int, sport: str) -> Dict[str, float]:
    if not version:
        return registry.defaults(sport)
    row = bs.get_version(db, version)
    stored = {k: v for k, v in ((row or {}).get("params") or {}).items() if k in registry.KNOBS
              and registry.KNOBS[k].applies(sport)}
    return registry.check(sport, stored)


def challenger_values(db: sqlite3.Connection, row: Mapping[str, Any], champion: Mapping[str, float]
                      ) -> Tuple[Dict[str, float], List[str]]:
    """A challenger as it would go live today: the current champion with only the challenger's own knobs (its
    ``knob`` list) set to its values, so a promotion since it was frozen is never undone. (values, its knobs that
    still differ from the champion)."""
    sport = row["sport"]
    stored = _values_of(db, row["version"], sport)
    own = [k for k in str(row.get("knob") or "").split(",") if k in stored]
    if not own:
        own = registry.changed(sport, _values_of(db, int(row.get("parent") or 0), sport), stored)
    try:
        values = registry.check(sport, {**champion, **{k: stored[k] for k in own}})
    except ValueError:
        return dict(champion), []
    return values, registry.changed(sport, champion, values)


def _db(target: Any) -> Tuple[sqlite3.Connection, bool]:
    """A connection for a Store, BrainStore, path or connection (and whether to close it)."""
    if isinstance(target, sqlite3.Connection):
        bs.init_brain_schema(target)
        return target, False
    if isinstance(target, str):
        db = sqlite3.connect(target, timeout=10, isolation_level=None)
    else:
        store = target.store if isinstance(target, bs.BrainStore) else target
        db = store._connect() if not isinstance(store, str) else sqlite3.connect(store, timeout=10,
                                                                                  isolation_level=None)
    bs.init_brain_schema(db)
    return db, True


# --- rollback watch, CUSUM -----------------------------------------------------------------------------------------

ROLLBACK_LOOKS = (ROLLBACK_MIN, 2 * ROLLBACK_MIN, 4 * ROLLBACK_MIN)  # the planned looks of the "range above 0" check
LOOK_LEVEL = 1 - ALPHA / len(ROLLBACK_LOOKS)  # each look's two-sided range: Bonferroni across the looks
WEEK_GRACE = 86400.0  # a week enters the CUSUM a day after it ends (its last games graded)


def frozen_sd(d: np.ndarray, days: Sequence[Any]) -> Optional[float]:
    """The per-game scale the CUSUM standardises by, from the first games after a promotion: the larger of the plain
    SD and the day-clustered one (same-day games share news, weather and slates, so a week's mean varies more than
    independent games would)."""
    d = np.asarray(d, dtype=float)
    if d.size < 2:
        return None
    plain = float(np.std(d, ddof=1))
    _, sums, counts = stats.daily(d, days)
    clustered = math.sqrt(float(np.sum((sums - counts * float(d.mean())) ** 2)) / d.size)
    sd = max(plain, clustered)
    return sd if sd > 0 else None


def detector(paired: ev.Paired, *, now: float, scale: Optional[Mapping[str, Any]] = None, seed: int = 0
             ) -> Dict[str, Any]:
    """The rollback checks on a live champion against the version before it (``paired``: champion minus parent, per
    game since the promotion), built so that a change that is not worse is rarely taken back (method.md 4.2 on
    peeking):

    - the "range above 0" check runs only at the planned looks (the first 200, 400 and 800 games, in start order),
      each with a Bonferroni share of the error rate (a 98.3% range, day-clustered: ``stats.t_interval``), so
      re-running it every night never adds a look;
    - the one-sided CUSUM (k 0.25, h 5) runs on complete ISO weeks (a week enters a day after it ends), each week's
      mean standardised by a per-game scale frozen from the first ROLLBACK_MIN games (``frozen_sd``), starting with
      the first week after those games. Its path only grows (a week once scored never changes), and on weekly
      scores its in-control run length is about 140 weeks, several seasons (on daily scores it was about 140 days).

    ``scale`` is the stored {"sd", "after"} (the scale and the week it was frozen in), else it is worked out here.
    Returns {"n", "scale", "looks": [{"n", "days", "dll", "lo", "hi", "level"}], "weeks", "cusum", "cusum_max",
    "cusum_fired", "fired": None | "range" | "cusum", "look": the look that fired}."""
    order = np.argsort(paired.starts, kind="stable")
    d = paired.d_ll[order]
    days = [paired.days[i] for i in order]
    starts = paired.starts[order]
    n = int(d.size)
    out: Dict[str, Any] = {"n": n, "scale": dict(scale) if scale else None, "looks": [], "weeks": 0, "cusum": 0.0,
                           "cusum_max": 0.0, "cusum_fired": False, "fired": None, "look": None}
    for look in ROLLBACK_LOOKS:
        if n < look:
            break
        found = stats.t_interval(d[:look], days[:look], LOOK_LEVEL)
        entry = {"n": look, "days": found["days"], "dll": found["mean"], "lo": found["lo"], "hi": found["hi"],
                 "level": LOOK_LEVEL}
        out["looks"].append(entry)
        if out["fired"] is None and found["lo"] is not None and found["lo"] > 0:
            out["fired"], out["look"] = "range", entry
    if out["scale"] is None and n >= ROLLBACK_MIN:
        sd = frozen_sd(d[:ROLLBACK_MIN], days[:ROLLBACK_MIN])
        if sd is not None:
            out["scale"] = {"sd": sd, "after": stats.week_of(float(starts[ROLLBACK_MIN - 1]))}
    if out["scale"]:
        sd, after = float(out["scale"]["sd"]), str(out["scale"]["after"])
        weeks = [stats.week_of(float(s)) for s in starts]
        keep = np.array([w > after and stats.week_bounds(w)[1] + WEEK_GRACE <= now for w in weeks], dtype=bool)
        z = stats.daily_z(d[keep], [w for w, k in zip(weeks, keep) if k], sd=sd) if keep.any() else []
        cs = stats.cusum(z, CUSUM_K, CUSUM_H)
        out.update({"weeks": len(z), "cusum": cs["last"], "cusum_max": cs["max"], "cusum_fired": cs["fired"]})
        if out["fired"] is None and cs["fired"]:
            out["fired"] = "cusum"
    return out


def watch(db: sqlite3.Connection, corpus: ev.Corpus, sport: str, *, now: float, seed: int = 0) -> Dict[str, Any]:
    """The live champion against its parent (always replayed in shadow) on games logged since its promotion: per data
    kind the change in log loss with its 95% range (shown), and the rollback checks (``detector``: planned looks and
    a weekly CUSUM with its scale frozen in brain_state ``watch_scale:<version>``). ``rollback`` names the trigger when
    one fired."""
    row = bs.champion(db, sport)
    if row is None or row.get("effective_from") is None:
        return {"version": 0, "rollback": None}
    champion = _champion_values(db, sport)
    base = watched_base(db, row)
    parent = _values_of(db, base, sport)
    knobs = registry.changed(sport, parent, champion, tol=1e-12)
    kinds = sorted({data_kind(k) for k in knobs})
    out: Dict[str, Any] = {"version": row["version"], "parent": base, "kinds": {}, "rollback": None}
    since = float(row["effective_from"])
    scale_key = f"watch_scale:{row['version']}"
    frozen = dict(bs.get_state(db, scale_key) or {})
    for kind in kinds:
        rows = [r for r in rows_for(kind, corpus, sport) if _logged(r) > since]
        paired = pair(kind, rows, sport, parent, champion)  # challenger = the live champion, champion = its parent
        evidence = evidence_of(paired, seed)
        found = detector(paired, now=now, scale=frozen.get(kind), seed=seed)
        if kind not in frozen and found["scale"] is not None:
            frozen[kind] = found["scale"]
            bs.set_state(db, scale_key, frozen, now)
        out["kinds"][kind] = {**{k: evidence.get(k) for k in ("n", "days", "dll", "lo", "hi", "p_worse")},
                              "cusum": found["cusum"], "cusum_max": found["cusum_max"],
                              "cusum_fired": found["cusum_fired"], "weeks": found["weeks"], "looks": found["looks"]}
        if out["rollback"] is None:
            look = found["look"]
            if found["fired"] == "range":
                out["rollback"] = (f"Worse than the version before it: on the first {look['n']} games since the change "
                                   f"the log loss rose {stats.signed(look['dll'])} ({look['level']:.1%} range "
                                   f"{stats.signed(look['lo'])} to {stats.signed(look['hi'])}, one of "
                                   f"{len(ROLLBACK_LOOKS)} planned checks).")
            elif found["fired"] == "cusum":
                out["rollback"] = (f"A steady drift against the version before it set off the change detector "
                                   f"(CUSUM {found['cusum_max']:.1f} over {CUSUM_H:g} on {found['weeks']} weekly "
                                   "scores).")
    return out


def watched_base(db: sqlite3.Connection, row: Mapping[str, Any]) -> int:
    """The version live before ``row``'s promotion: its parent, or, when one weekly run promoted several changes for
    the sport one after another (each on top of the last, at the same moment), the version live before that run. The
    run's changes are watched, and rolled back, together against it."""
    base = int(row.get("parent") or 0)
    seen = {int(row["version"])}
    while base and base not in seen:
        seen.add(base)
        earlier = bs.get_version(db, base)
        if earlier is None or earlier.get("effective_from") != row.get("effective_from"):
            break
        base = int(earlier.get("parent") or 0)
    return base


def _logged(r: Any) -> float:
    if isinstance(r, Mapping):
        return float(r.get("logged_at") or r["start"])
    return float(r.lines_as_of or r.start)


def roll_back(db: sqlite3.Connection, sport: str, reason: str, *, now: float, source: str = "refit",
              decision: str = "rolled_back", evidence: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Retire the live champion of ``sport`` as rolled back, make its parent live again, and write the ledger row."""
    row = bs.champion(db, sport)
    if row is None:
        return None
    old = _champion_values(db, sport)
    # An automatic rollback goes back to the version the watch compared against (before the whole run's changes);
    # the owner's Undo takes one change back
    target = watched_base(db, row) if decision == "rolled_back" else int(row.get("parent") or 0)
    retired: List[int] = []
    with bs.transaction(db):
        parent = None
        live: Optional[Dict[str, Any]] = row
        while live is not None and int(live["version"]) != target:
            retired.append(int(live["version"]))
            parent = bs.roll_back(db, live["version"], now)
            live = parent
        new = _values_of(db, parent["version"], sport) if parent else registry.defaults(sport)
        knobs = registry.changed(sport, new, old)
        bs.add_trial(db, {"week": stats.week_of(now), "sport": sport, "knob": ",".join(knobs) or "-",
                          "old": {k: old[k] for k in knobs}, "new": {k: new[k] for k in knobs}, "source": source,
                          "decision": decision, "reason": reason, "gate": "rollback" if decision == "rolled_back"
                          else "owner", "version": row["version"],
                          "details": {**dict(evidence or {}), "versions": retired}}, now)
    registry.invalidate()
    return {"version": row["version"], "now_live": parent["version"] if parent else 0, "knobs": knobs,
            "reason": reason, "versions": retired}


def undo(target: Any, version: int, *, now: Optional[float] = None, by: str = "owner") -> Dict[str, Any]:
    """The owner's Undo: a live champion goes back to its parent (``rolled_back``, ledger decision "undone"); a
    challenger is withdrawn (nothing live changes). Returns {"undone", "now_live"} (plus "withdrawn": True for a
    challenger) or {"error"}."""
    now = time.time() if now is None else now
    db, close = _db(target)
    try:
        row = bs.get_version(db, version)
        if row is None:
            return {"error": "no such version"}
        if row["status"] == "challenger":
            bs.end_challenger(db, version, "rolled_back", now)
            bs.add_trial(db, {"week": stats.week_of(now), "sport": row["sport"], "knob": row.get("knob") or "-",
                              "source": by, "decision": "undone", "gate": "owner", "version": version,
                              "reason": "Withdrawn by the owner before it went live."}, now)
            return {"undone": version, "withdrawn": True, "now_live": _champion_version(db, row["sport"])}
        if row["status"] != "champion":
            return {"error": f"version {version} is not live"}
        done = roll_back(db, row["sport"], "Undone by the owner.", now=now, source=by, decision="undone")
        return {"undone": version, "now_live": done["now_live"] if done else 0}
    finally:
        if close:
            db.close()


# --- the tripwire and safe mode ------------------------------------------------------------------------------------

def _gap(p: Mapping[str, Any], sharp: Mapping[str, Any]) -> float:
    return max((abs(float(p[o]) - float(sharp[o])) for o in p if sharp.get(o) is not None
                and isinstance(p[o], (int, float))), default=0.0)


def tripwire(rows: Sequence[Mapping[str, Any]], *, pp: float = TRIP_PP, share: float = TRIP_SHARE,
             min_games: int = TRIP_MIN_GAMES, baseline: Optional[Sequence[Optional[Mapping[str, Any]]]] = None
             ) -> Dict[str, Any]:
    """The cheap bug guard over one night's logged blends: ``fired`` when any blend chance is missing, not a number
    or outside 0-1, or when the blend sits more than ``pp`` points from the ok sharp price on more than ``share`` of
    the games that have one (at least ``min_games`` of them). With ``baseline`` (one chance dict per row: the same
    sources blended with today's defaults) the live blends must also be that far on ``share`` more of the games than
    the defaults' blends: where the sources simply disagree, the defaults sit far from the sharp price too, and that
    is the data, not a fault in the live settings."""
    bad, far, far_default, sharp_games = [], 0, 0, 0
    for i, r in enumerate(rows):
        p = (r.get("blend") or {}).get("p")
        values = list(p.values()) if isinstance(p, Mapping) else []
        if not values or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
                             for v in values) or abs(math.fsum(values) - 1) > 1e-6:
            bad.append(r.get("game_key"))
            continue
        sharp = next((s for s in r.get("sources") or [] if isinstance(s, Mapping) and s.get("kind") == "sharp"
                      and s.get("status") == "ok" and isinstance(s.get("p"), Mapping)), None)
        if sharp is None:
            continue
        sharp_games += 1
        if _gap(p, sharp["p"]) * 100 > pp:
            far += 1
        base = baseline[i] if baseline is not None and i < len(baseline) else None
        if isinstance(base, Mapping) and _gap(base, sharp["p"]) * 100 > pp:
            far_default += 1
    rate = far / sharp_games if sharp_games else 0.0
    rate_default = far_default / sharp_games if sharp_games and baseline is not None else 0.0
    fired = bool(bad) or (sharp_games >= min_games and rate > share and rate - rate_default > share)
    if bad:
        reason = f"{len(bad)} blend(s) with a broken chance"
    elif fired:
        reason = f"the blend sat more than {pp:g} points from the sharp price on {rate:.0%} of {sharp_games} games"
        if baseline is not None:
            reason += f" (today's defaults: {rate_default:.0%})"
    else:
        reason = ""
    return {"fired": fired, "games": len(rows), "sharp_games": sharp_games, "far": far, "share": rate,
            "far_default": far_default if baseline is not None else None,
            "share_default": rate_default if baseline is not None else None, "broken": bad[:10], "reason": reason}


def _default_blend(row: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """A logged game's sources blended with today's defaults (None when they can't be)."""
    try:
        return ev.reblend(row, {})
    except Exception:  # noqa: BLE001 - a row the blend can't read: judged on its own blend
        return None


def check_tripwire(target: Any, *, now: Optional[float] = None, hours: float = 24.0) -> Dict[str, Any]:
    """Run the tripwire over the blends logged in the last ``hours`` and switch safe mode on when it fires and
    something other than the defaults is live (the defaults then serve every sport until the owner clears it)."""
    now = time.time() if now is None else now
    db, close = _db(target)
    try:
        rows = [r for r in ev.consensus_records(db, status=None) if r["logged_at"] >= now - hours * 3600]
        result = tripwire(rows, baseline=[_default_blend(r) for r in rows])
        result["safe_mode"] = False
        if result["fired"]:
            tuned = bool(bs.champions(db))
            if tuned and bs.set_safe_mode(db, True, f"Tripwire: {result['reason']}.", now,
                                          {"share": result["share"], "games": result["sharp_games"]}):
                bs.add_trial(db, {"week": stats.week_of(now), "sport": "all", "knob": "-", "source": "tripwire",
                                  "decision": "safe_mode", "gate": "tripwire",
                                  "reason": f"Safe mode: {result['reason']}. The default settings serve every sport "
                                            "until the owner clears it.", "details": result}, now)
                registry.invalidate()
                result["safe_mode"] = True
            elif not tuned:
                result["note"] = "The default settings are live, so this points at the data, not the tuning."
        return result
    finally:
        if close:
            db.close()


def clear_safe_mode(target: Any, *, now: Optional[float] = None, by: str = "owner") -> bool:
    now = time.time() if now is None else now
    db, close = _db(target)
    try:
        changed = bs.set_safe_mode(db, False, now=now)
        if changed:
            bs.add_trial(db, {"week": stats.week_of(now), "sport": "all", "knob": "-", "source": by,
                              "decision": "cleared", "gate": "owner", "reason": "Safe mode cleared by the owner."}, now)
            registry.invalidate()
        return changed
    finally:
        if close:
            db.close()


# --- the weekly run ------------------------------------------------------------------------------------------------

@dataclass
class Candidate:
    sport: str
    family: str
    knobs: List[str]
    values: Dict[str, float]
    old: Dict[str, float]
    source: str
    gate: str  # nudge or structural
    train_n: int
    note: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    version: Optional[int] = None  # a stored challenger
    created_at: Optional[float] = None
    decision: str = "held"
    reason: str = ""
    holm_p: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


def count_vs_climatology(matches: Sequence[ev.MatchRecord], target: str, model: Mapping[str, float], *,
                         rho: float = soccer.DC_RHO, weeks: int = EVAL_WEEKS, block: int = BLOCK_WEEKS
                         ) -> Optional[ev.Paired]:
    """A count model (fixed) against climatology (each league's mean and dispersion: ``evaluate.climatology_loglik``)
    refitted before each 4-week block of the last ``weeks`` weeks: per-match change in log score (model minus
    climatology; negative: the model is better)."""
    have = [m for m in matches if target in m.stats]
    folds = list(ev.weekly_folds(have, min_train=MIN_TRAIN, weeks=weeks, start=lambda m: m.start))
    pieces = []
    for i in range(0, len(folds), block):
        chunk = folds[i:i + block]
        train, test = chunk[0][1], [m for _, _, rows in chunk for m in rows]
        clim_ll, _ = ev.climatology_loglik(target, train, test)  # each league's own, fitted before the block
        yh = np.array([m.stats[target][0] for m in test])
        ya = np.array([m.stats[target][1] for m in test])
        sup, tot = ev.count_inputs(test, rho)
        model_ll = ev.count_loglik(model, yh, ya, sup, tot)
        n = len(test)
        pieces.append(ev.Paired(-(model_ll - clim_ll), np.zeros(n), [m.day for m in test],
                                np.array([m.start for m in test]), np.zeros(n), np.zeros(n), np.zeros(n)))
    return _concat(pieces) if pieces else None


def _show_candidates(db: sqlite3.Connection, corpus: ev.Corpus, now: float, seed: int) -> List[Candidate]:
    """Whether each count model now beats climatology out of sample (to be shown as "model only, low confidence")."""
    if not corpus.matches and not corpus.sport("soccer"):
        return []
    matches = [m for m in soccer_matches(corpus) if m.start < now]
    champion = _champion_values(db, "soccer")
    out = []
    for target in registry.COUNT_TARGETS:
        model = {t: champion[f"counts.{target}.{t}"] for t in registry.COUNT_TERMS}
        paired = count_vs_climatology(matches, target, model, rho=champion["soccer.rho"])
        if paired is None:
            continue
        out.append(Candidate("soccer", f"show.{target}", [], dict(champion), {}, "refit", "structural",
                             sum(1 for m in matches if target in m.stats), f"the {target} model against climatology",
                             evidence_of(paired, seed), details={"target": target}))
    return out


def still_watched(db: sqlite3.Connection, corpus: ev.Corpus, sport: str) -> Optional[Tuple[int, int]]:
    """(the live version, graded games since it went live) while the sport's last change has had fewer than
    ROLLBACK_MIN games to be judged on (its rollback watch needs them), else None. A new change for the sport waits:
    stacking weekly nudges would restart the watch every week, and a regression would never be seen."""
    row = bs.champion(db, sport)
    if row is None or row.get("effective_from") is None:
        return None
    since = float(row["effective_from"])
    n = sum(1 for r in corpus.sport(sport) if _logged(r) > since)
    return (int(row["version"]), n) if n < ROLLBACK_MIN else None


def cooling_off(db: sqlite3.Connection, sport: str, knobs: Sequence[str], now: float) -> Optional[Dict[str, Any]]:
    """The owner's Undo or an automatic rollback of any of ``knobs`` for ``sport`` in the last COOL_OFF_WEEKS
    (the ledger row), else None: such a setting is held rather than moved straight back next Monday."""
    wanted = set(knobs)
    for t in bs.trials(db, sport=sport, since=now - COOL_OFF_WEEKS * 7 * 86400, limit=None):
        if t["decision"] in ("undone", "rolled_back") and wanted & set(str(t.get("knob") or "").split(",")):
            return t
    return None


SCORE_WEEKS = EVAL_WEEKS  # the weekly run also scores the earlier weeks within this many that have no scores yet


def backfill_scores(db: sqlite3.Connection, corpus: ev.Corpus, week: str, *, now: float, seed: int = 0) -> int:
    """Score the weeks before ``week`` (within SCORE_WEEKS of it) that have graded games but no scores yet: the logged
    blends as they were, against the market on the same games (a record, nothing refitted), so the page's score over
    time starts with the games already graded when the loop first runs. Returns the rows written."""
    begin = stats.week_bounds(week)[0]
    first = begin - SCORE_WEEKS * 7 * 86400
    weeks = {stats.week_of(r["start"]) for rows in (corpus.consensus, corpus.context) for r in rows
             if first <= r["start"] < begin}
    weeks |= {m.week for m in corpus.matches if first <= m.start < begin}
    have = {row["week"] for row in bs.scores(db)}
    written = 0
    for earlier in sorted(weeks - have):
        written += len(ev.write_week_scores(db, corpus, earlier, now=now, seed=seed))
    return written


def decision_week(now: float) -> str:
    """The week a run on ``now`` decides: the last full ISO week before it."""
    return stats.week_of(now - 7 * 86400)


def run_week(target: Any, *, now: Optional[float] = None, corpus: Optional[ev.Corpus] = None,
             matches: Optional[Sequence[ev.MatchRecord]] = None, paused: bool = False, force: bool = False,
             seed: int = 0, families: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """The weekly decision: refit candidates per sport, gate them, promote nudges that pass, shadow structural ones
    that pass, promote challengers whose shadow is done, roll back what got worse, write the ledger. Idempotent per
    week (``force`` reruns) and resumable: the decisions and the rollbacks are each written in one transaction with
    the report (brain_state ``week_decided:<week>``), so a run that stops before its scores are written carries on
    from there and returns the same report. ``paused`` (LEARN_ENABLED=0 or the owner's switch) and safe mode score and record but
    change nothing. Returns a report: {"week", "skipped", "sports": {sport: {...}}, "trials", "promoted",
    "rolled_back", "shadow", "held"}."""
    now = time.time() if now is None else now
    db, close = _db(target)
    try:
        week = decision_week(now)
        report: Dict[str, Any] = {"week": week, "ran_at": now, "skipped": None, "sports": {}, "trials": [],
                                  "promoted": [], "rolled_back": [], "shadow": [], "held": [], "rejected": []}
        mark = bs.get_state(db, f"week_done:{week}")
        if mark and not force:
            report["skipped"] = "already ran"
            return report
        corpus = corpus or ev.load_corpus(db, until=now, matches=matches)
        frozen = paused or bool(bs.safe_mode(db))
        # Resumable: the decisions are written (with the report) in one transaction, then the rollbacks (with the
        # report again); a run that stopped after that (scoring) carries on from the stored report and never decides
        # the week twice
        decided = None if force else bs.get_state(db, f"week_decided:{week}")
        if isinstance(decided, Mapping) and isinstance(decided.get("report"), Mapping):
            report = {**report, **decided["report"], "resumed": True}
            phase = decided.get("phase")
        else:
            _decide_week(db, corpus, now=now, week=week, seed=seed, families=families, frozen=frozen, paused=paused,
                         report=report)
            phase = "recorded"
        if phase == "recorded":
            fired = {}
            for sport in corpus.sports():
                if frozen:
                    continue
                w = watch(db, corpus, sport, now=now, seed=seed)
                if w.get("rollback"):
                    fired[sport] = w
            with bs.transaction(db):
                for sport, w in fired.items():
                    done = roll_back(db, sport, w["rollback"], now=now, evidence=w)
                    if done:
                        report["rolled_back"].append({"sport": sport, **done})
                bs.set_state(db, f"week_decided:{week}", {"phase": "decided", "report": report}, now)
        report["scores"] = len(ev.write_week_scores(db, corpus, week, now=now, seed=seed))  # the decided week, final
        report["scores_backfilled"] = backfill_scores(db, corpus, week, now=now, seed=seed)
        bs.set_state(db, f"week_done:{week}", {"at": now, "trials": len(report["trials"])}, now)
        bs.set_state(db, "last_week_run", {"week": week, "at": now}, now)
        return report
    finally:
        if close:
            db.close()


def _decide_week(db: sqlite3.Connection, corpus: ev.Corpus, *, now: float, week: str, seed: int,
                 families: Optional[Sequence[str]], frozen: bool, paused: bool, report: Dict[str, Any]) -> None:
    """The week's candidates, gates and holds, then every decision written in one transaction with the report (brain
    state ``week_decided:<week>``, phase "recorded")."""
    candidates: List[Candidate] = []
    pooled_cache: Dict[float, Optional[np.ndarray]] = {}
    for sport in corpus.sports():
        candidates += _sport_candidates(db, corpus, sport, now=now, seed=seed, families=families,
                                        pooled_cache=pooled_cache)
    if families is None or any(f.startswith("show.") for f in families):
        candidates += _show_candidates(db, corpus, now, seed)
    _decide(candidates, frozen)
    for c in candidates:
        flag = bs.get_state(db, f"freeze:{c.sport}")
        if isinstance(flag, Mapping) and flag.get("frozen") and c.decision in ("promoted", "shadow"):
            c.details["would_have"] = c.decision
            c.decision, c.reason = "held", ("Calibration drifted lately, so promotions are frozen for this sport "
                                            "until the weekly review. " + c.reason)
    watched: Dict[str, Optional[Tuple[int, int]]] = {}
    for c in candidates:
        if c.decision != "promoted":
            continue
        if c.sport not in watched:
            watched[c.sport] = still_watched(db, corpus, c.sport)
        if watched[c.sport] is not None:
            version, n = watched[c.sport]
            c.details["would_have"] = c.decision
            c.decision, c.reason = "held", (f"Waits its turn: version {version}, the last change, has {n} of the "
                                            f"{ROLLBACK_MIN} graded games it is watched on before anything is "
                                            "built on it (changes go live one at a time). " + c.reason)
    for c in candidates:
        if c.decision not in ("promoted", "shadow") or not c.knobs:
            continue
        taken_back = cooling_off(db, c.sport, c.knobs, now)
        if taken_back is not None:
            c.details["would_have"] = c.decision
            when = datetime.datetime.fromtimestamp(float(taken_back["created_at"]),
                                                   datetime.timezone.utc).strftime("%Y-%m-%d")
            until = datetime.datetime.fromtimestamp(float(taken_back["created_at"]) + COOL_OFF_WEEKS * 7 * 86400,
                                                    datetime.timezone.utc).strftime("%Y-%m-%d")
            why = "the owner took it back" if taken_back["decision"] == "undone" else "it was rolled back"
            c.decision, c.reason = "held", (f"Not moved again yet: {why} on {when}, so this setting waits until "
                                            f"{until}. " + c.reason)
    with bs.transaction(db):
        for c in candidates:
            _record(db, c, week, now, frozen, report)
        report["skipped"] = "paused" if paused else ("safe mode" if frozen else None)
        bs.set_state(db, f"week_decided:{week}", {"phase": "recorded", "report": report}, now)


def _sport_candidates(db: sqlite3.Connection, corpus: ev.Corpus, sport: str, *, now: float, seed: int,
                      families: Optional[Sequence[str]],
                      pooled_cache: Optional[Dict[float, Optional[np.ndarray]]] = None) -> List[Candidate]:
    champion = _champion_values(db, sport)
    out: List[Candidate] = []
    # Challengers already in shadow (AI proposals and structural refits): judged on games after they were frozen
    for row in bs.versions(db, sport=sport, status="challenger"):
        values, knobs = challenger_values(db, row, champion)
        if not knobs:
            bs.end_challenger(db, row["version"], "retired", now)
            continue
        kind = data_kind(knobs[0])
        created = float(row.get("effective_from") or row["created_at"])
        rows = [r for r in rows_for(kind, corpus, sport) if (r["start"] if isinstance(r, Mapping) else r.start)
                > created]
        paired = pair(kind, rows, sport, champion, values)
        ai = str(row["source"]).startswith("ai:")
        if ai and sum(1 for c in out if c.details.get("ai")) >= MAX_AI:
            continue  # at most 3 AI proposals judged a week; the rest wait their turn
        e = evidence_of(paired, seed)
        week = stats.week_of(now)
        done = [look for look in ((row.get("evidence") or {}).get("looks") or []) if look.get("week") != week]
        # An AI proposal that passed a planned look but couldn't go live then (paused, another change still watched)
        # goes live once it can, like a refit that passed before its shadow, unless its range moves above 0
        details = {"shadow_since": created, "shadow_days": (now - created) / 86400, "ai": ai,
                   "passed_before": (row.get("evidence") or {}).get("passed")
                   or any(look.get("verdict") == "pass" for look in done)}
        if ai and not details["passed_before"]:
            details.update({"looks_done": done, "look": planned_look(e.get("n") or 0, details["shadow_days"], done)})
        c = Candidate(sport, "challenger", knobs, values, {k: champion[k] for k in knobs}, row["source"], "structural",
                      0, "in shadow since it was frozen", e, row["version"], created, details=details)
        out.append(c)
    health = data_health(corpus, sport, now)
    if health:
        out.append(Candidate(sport, "health", [], dict(champion), {}, "refit", "nudge", 0, "data health",
                             decision="held", reason="Refits paused for this sport: " + "; ".join(health) + "."))
        return out[:MAX_CANDIDATES]
    menu = [f for f in families_for(sport) if families is None or f in families]
    quiet: List[Candidate] = []  # families whose refit found nothing sure to move: recorded, not tested
    pooled = None
    if "weights" in menu:
        pooled = _pooled(corpus, now, pooled_cache)
    for family in menu:
        if len(out) >= MAX_CANDIDATES:
            break
        kind = family_kind(family)
        rows = [r for r in rows_for(kind, corpus, sport) if (r["start"] if isinstance(r, Mapping) else r.start) < now]
        fit = refit(family, sport, rows, champion, pooled=pooled if family == "weights" else None)
        if fit is None:
            continue
        if not fit.knobs:
            if fit.details.get("refused"):
                quiet.append(Candidate(sport, family, [], dict(champion), {}, "refit", "nudge", fit.train_n, fit.note,
                                       decision="rejected", reason=f"Refit refused: {fit.details['refused']}.",
                                       details={**fit.details, "no_move": True}))
            else:
                quiet.append(Candidate(sport, family, [], dict(champion), {}, "refit", "nudge", fit.train_n, fit.note,
                                       decision="held", reason=f"No change worth testing: {fit.note}, and the part of "
                                       "the move that is 95% sure is zero (or within rounding).",
                                       details={**fit.details, "no_move": True}))
            continue
        paired, fits = rolling(family, sport, rows, champion,
                               pooled_rows=corpus.consensus if family == "weights" else None, cache=pooled_cache)
        gate = "nudge" if registry.is_nudge(sport, champion, fit.values) else "structural"
        c = Candidate(sport, family, fit.knobs, fit.values, {k: champion[k] for k in fit.knobs}, "refit", gate,
                      fit.train_n, fit.note, evidence_of(paired, seed), details={**fit.details, "blocks": fits})
        out.append(c)
    return out[:MAX_CANDIDATES] + quiet


def _decide(candidates: List[Candidate], frozen: bool) -> None:
    """Holm across the week's structural candidates, then each gate."""
    # The family is this week's superiority tests: a challenger in shadow is tested only at an AI proposal's planned
    # look (a structural refit's shadow is a confirmation, not a test)
    structural = [c for c in candidates if c.gate == "structural" and c.decision != "rejected"
                  and not c.details.get("no_move")
                  and (c.family != "challenger" or c.details.get("look") is not None)]
    adjusted, _ = stats.holm([c.evidence.get("p") for c in structural], ALPHA)
    for c, holm_p in zip(structural, adjusted):
        c.holm_p = holm_p
    for c in candidates:
        if c.decision == "rejected" or c.details.get("no_move") or c.family == "health":
            continue
        if c.family == "challenger":
            _decide_challenger(c)
        elif c.family.startswith("show."):
            _decide_show(c)
        elif c.gate == "nudge":
            c.decision, c.reason = gate_nudge(c.evidence, kind=_kind_of(c))
        else:
            verdict, c.reason = gate_structural(c.evidence, c.holm_p, kind=_kind_of(c))
            c.decision = "shadow" if verdict == "pass" else verdict
        if frozen and c.decision in ("promoted", "shadow"):
            c.details["would_have"] = c.decision
            c.decision, c.reason = "held", "Learning is paused (or in safe mode): nothing changes. " + c.reason


def _kind_of(c: Candidate) -> str:
    if c.family.startswith("show."):
        return "counts:" + c.family.split(".", 1)[1]
    return data_kind(c.knobs[0]) if c.knobs else "results"


def _decide_show(c: Candidate) -> None:
    verdict, reason = gate_structural(c.evidence, c.holm_p, kind=_kind_of(c))
    c.details["verdict"] = verdict
    c.decision, c.reason = ("shadow", "Beats climatology so far: " + reason) if verdict == "pass" else ("held", reason)


def ai_looks() -> List[int]:
    """The graded games after an AI proposal at which it is judged (MIN_EVAL, then 2 x MIN_EVAL)."""
    return [MIN_EVAL * k for k in AI_LOOKS_FACTOR]


def planned_look(n: int, shadow_days: float, done: Sequence[Mapping[str, Any]]) -> Optional[int]:
    """The index of the planned look an AI challenger is at this week (its next one, once it has the games and
    SHADOW_WEEKS in shadow), else None."""
    looks = ai_looks()
    k = len(done)
    if k >= len(looks) or shadow_days < SHADOW_WEEKS * 7 or n < looks[k]:
        return None
    return k


def _decide_challenger(c: Candidate) -> None:
    e = c.evidence
    n = e.get("n") or 0
    if c.details.get("ai") and not c.details.get("passed_before"):
        # An AI proposal: the full structural gate on games after it was made, at its planned looks only (the first
        # weekly run with MIN_EVAL later games and SHADOW_WEEKS in shadow, then once more at 2 x MIN_EVAL), the 0.05
        # split between them (0.005, then 0.045), so judging it week after week never inflates false promotions
        look = c.details.get("look")
        looks = ai_looks()
        if look is None:
            if e.get("lo") is not None and e["lo"] > 0 and n >= MIN_EVAL:
                c.decision, c.reason = "rejected", (
                    f"Worse on games after the proposal ({stats.signed(e['dll'])}, 95% range {stats.signed(e['lo'])} "
                    f"to {stats.signed(e['hi'])}).")
                return
            done = len(c.details.get("looks_done") or [])
            target = looks[min(done, len(looks) - 1)]
            c.decision, c.reason = "held", (
                f"Waiting for its planned look: judged once {target} graded games after the proposal are in and it "
                f"has had {SHADOW_WEEKS} weeks in shadow ({n} so far).")
            return
        verdict, reason = gate_structural(e, c.holm_p, alpha=AI_LOOK_ALPHAS[look], kind=_kind_of(c))
        c.details["look_verdict"] = verdict
        where = f"planned look {look + 1} of {len(looks)}"
        if verdict == "pass":
            c.decision, c.reason = "promoted", f"Judged only on games after the proposal, at its {where}: " + reason
        elif verdict == "rejected":
            c.decision, c.reason = "rejected", reason
        elif look + 1 >= len(looks):
            c.decision, c.reason = "rejected", (f"Not shown to be better by its last planned look ({n} games after "
                                                "the proposal), so it is retired. " + reason)
        else:
            c.decision, c.reason = "held", (f"Not shown to be better at its {where}; judged once more at "
                                            f"{looks[look + 1]} games. " + reason)
        return
    # A structural refit that passed: promoted after SHADOW_WEEKS of forward games unless its range moved above 0
    if e.get("lo") is not None and e["lo"] > 0:
        c.decision, c.reason = "rejected", (f"Worse in shadow ({stats.signed(e['dll'])}, 95% range "
                                            f"{stats.signed(e['lo'])} to {stats.signed(e['hi'])}).")
    elif c.details.get("shadow_days", 0) < SHADOW_WEEKS * 7:
        c.decision, c.reason = "held", f"In shadow: {n} games so far, {SHADOW_WEEKS} weeks needed."
    else:
        c.decision, c.reason = "promoted", (f"Held up in {SHADOW_WEEKS} weeks of shadow on {n} games"
                                            + (f" ({stats.signed(e['dll'])})." if e.get("dll") is not None else "."))


def _record(db: sqlite3.Connection, c: Candidate, week: str, now: float, frozen: bool,
            report: Dict[str, Any]) -> None:
    e = c.evidence
    row = {"week": week, "sport": c.sport, "knob": ",".join(c.knobs) or c.family, "old": c.old,
           "new": {k: c.values[k] for k in c.knobs}, "source": c.source, "train_n": c.train_n,
           "eval_n": e.get("n"), "dll": e.get("dll"), "lo": e.get("lo"), "hi": e.get("hi"), "p": e.get("p"),
           "holm_p": c.holm_p, "decision": c.decision, "reason": c.reason, "gate": c.gate, "version": c.version,
           "details": {"family": c.family, "note": c.note, "dbrier": e.get("dbrier"), "days": e.get("days"),
                       "calibration": e.get("calibration_challenger"), **{k: v for k, v in c.details.items()
                                                                           if k in ("would_have", "cut", "free",
                                                                                    "shadow_since", "ai")}}}
    if c.family.startswith("show."):
        _record_show(db, c, row, now, frozen)
        report["trials"].append(row)
        bucket = c.decision if c.decision in ("held", "shadow", "promoted", "rolled_back") else "held"
        report[bucket].append({"sport": c.sport, "family": c.family, "reason": c.reason})
        return
    with bs.transaction(db):
        if c.family == "challenger" and c.version is not None and c.details.get("look") is not None:
            _record_look(db, c, week)
        if c.decision == "promoted" and not frozen:
            # On top of whatever is live now (an earlier promotion this run touched other knobs)
            merged = registry.check(c.sport, {**_champion_values(db, c.sport), **{k: c.values[k] for k in c.knobs}})
            evidence = _evidence_row(c)
            if c.version is None:
                c.version = bs.insert_version(db, c.sport, merged, status="challenger",
                                              parent=_champion_version(db, c.sport), source=c.source,
                                              evidence=evidence, now=now, knob=",".join(c.knobs), gate=c.gate)
            else:
                bs.set_params(db, c.version, merged)
                kept = (bs.get_version(db, c.version) or {}).get("evidence")
                evidence = {**(kept if isinstance(kept, Mapping) else {}), **evidence}  # the proposal and its looks stay
            bs.promote(db, c.version, now, evidence=evidence)
            row["version"] = c.version
            report["promoted"].append({"sport": c.sport, "version": c.version, "knobs": c.knobs,
                                       "source": c.source})
        elif c.decision == "shadow" and not frozen and c.version is None:
            c.version = bs.insert_version(db, c.sport, c.values, status="challenger",
                                          parent=_champion_version(db, c.sport), source=c.source,
                                          evidence={**_evidence_row(c), "passed": True}, effective_from=now, now=now,
                                          knob=",".join(c.knobs), gate="structural")
            row["version"] = c.version
            report["shadow"].append({"sport": c.sport, "version": c.version, "knobs": c.knobs})
        elif c.decision == "rejected" and c.version is not None:
            bs.end_challenger(db, c.version, "retired", now)
        trial_id = bs.add_trial(db, row, now)
    if c.decision == "promoted":
        registry.invalidate()
    elif c.decision == "held":
        report["held"].append({"sport": c.sport, "family": c.family, "reason": c.reason})
    elif c.decision == "rejected":
        report["rejected"].append({"sport": c.sport, "family": c.family, "reason": c.reason})
    report["trials"].append({**row, "id": trial_id})
    report["sports"].setdefault(c.sport, {"candidates": 0, "promoted": 0, "held": 0, "rejected": 0, "shadow": 0})
    report["sports"][c.sport]["candidates"] += 1
    if c.decision in report["sports"][c.sport]:
        report["sports"][c.sport][c.decision] += 1


def _record_look(db: sqlite3.Connection, c: Candidate, week: str) -> None:
    """An AI challenger's planned look, kept in its version's evidence (a rerun of the same week replaces it)."""
    row = bs.get_version(db, int(c.version)) or {}
    evidence = dict(row.get("evidence") or {})
    e = c.evidence
    looks = [look for look in evidence.get("looks") or [] if look.get("week") != week]
    looks.append({"look": int(c.details["look"]) + 1, "week": week, "n": e.get("n"), "dll": e.get("dll"),
                  "lo": e.get("lo"), "hi": e.get("hi"), "p": e.get("p"), "holm_p": c.holm_p,
                  "alpha": AI_LOOK_ALPHAS[int(c.details["look"])], "verdict": c.details.get("look_verdict")})
    evidence["looks"] = looks
    bs.set_evidence(db, int(c.version), evidence)


def _record_show(db: sqlite3.Connection, c: Candidate, row: Dict[str, Any], now: float, frozen: bool) -> None:
    """A count model is shown once it has beaten climatology at two weekly runs at least SHADOW_WEEKS apart; it is
    hidden again when its range against climatology moves above 0."""
    target = c.details["target"]
    key = f"counts_shown:{target}"
    state = bs.get_state(db, key) or {}
    e = c.evidence
    if c.decision == "shadow" and not frozen:
        since = state.get("pending_since") if not state.get("shown") else None
        if state.get("shown"):
            c.decision, c.reason = "held", "Still beats climatology: " + c.reason.split(": ", 1)[-1]
        elif since is not None and now - since >= SHADOW_WEEKS * 7 * 86400:
            state = {"shown": True, "since": now, "evidence": _evidence_row(c)}
            c.decision = "promoted"
            c.reason = (f"The {target} model beat climatology again after {SHADOW_WEEKS} weeks: shown from now as "
                        "\"model only, low confidence\". " + c.reason.split(": ", 1)[-1])
        else:
            state = {"shown": False, "pending_since": since or now, "evidence": _evidence_row(c)}
    elif state.get("shown") and e.get("lo") is not None and e["lo"] > 0 and not frozen:
        state = {"shown": False, "hidden_at": now, "evidence": _evidence_row(c)}
        c.decision, c.reason = "rolled_back", f"The {target} model fell behind climatology: hidden again."
    elif not state.get("shown"):
        state = {"shown": False, "evidence": _evidence_row(c)}
    bs.set_state(db, key, state, now)
    row.update({"decision": c.decision, "reason": c.reason, "knob": f"counts.{target}"})
    row["id"] = bs.add_trial(db, row, now)


def _evidence_row(c: Candidate) -> Dict[str, Any]:
    e = c.evidence
    return {"family": c.family, "n": e.get("n"), "days": e.get("days"), "dll": e.get("dll"), "lo": e.get("lo"),
            "hi": e.get("hi"), "dbrier": e.get("dbrier"), "p": e.get("p"), "holm_p": c.holm_p, "reason": c.reason,
            "train_n": c.train_n, "note": c.note}


# --- the nightly run -----------------------------------------------------------------------------------------------

def run_night(target: Any, *, now: Optional[float] = None, corpus: Optional[ev.Corpus] = None,
              matches: Optional[Sequence[ev.MatchRecord]] = None, paused: bool = False, seed: int = 0
              ) -> Dict[str, Any]:
    """The nightly guard: the tripwire, the rollback watch (never a promotion at night), shadow progress for the
    challengers, the calibration freeze flags and the week-to-date scores. Returns a report."""
    now = time.time() if now is None else now
    db, close = _db(target)
    try:
        report: Dict[str, Any] = {"ran_at": now, "tripwire": None, "rolled_back": [], "shadow": [], "freeze": {},
                                  "scores": 0}
        report["tripwire"] = check_tripwire(db, now=now)
        corpus = corpus or ev.load_corpus(db, until=now, matches=matches)
        for sport in corpus.sports():
            if not paused and not bs.safe_mode(db):
                w = watch(db, corpus, sport, now=now, seed=seed)
                bs.set_state(db, f"watch:{sport}", {k: v for k, v in w.items() if k != "rollback"}, now)
                if w.get("rollback"):
                    done = roll_back(db, sport, w["rollback"], now=now, evidence=w)
                    if done:
                        report["rolled_back"].append({"sport": sport, **done})
            champion = _champion_values(db, sport)
            for row in bs.versions(db, sport=sport, status="challenger"):
                values, knobs = challenger_values(db, row, champion)
                if not knobs:
                    continue
                kind = data_kind(knobs[0])
                created = float(row.get("effective_from") or row["created_at"])
                rows = [r for r in rows_for(kind, corpus, sport) if (r["start"] if isinstance(r, Mapping)
                                                                     else r.start) > created]
                e = evidence_of(pair(kind, rows, sport, champion, values), seed)
                progress = {"version": row["version"], "sport": sport, "knobs": knobs, "n": e["n"], "needed": MIN_EVAL,
                            "dll": e.get("dll"), "lo": e.get("lo"), "hi": e.get("hi"), "since": created,
                            "source": row["source"], "updated_at": now}
                bs.set_state(db, f"shadow:{row['version']}", progress, now)
                report["shadow"].append(progress)
            recent = corpus.sport(sport)[-ev.TRAILING:]
            if len(recent) >= 100:
                scored = [s for r in recent if (s := ev.game_scores(r)) is not None]
                cal = stats.calibration([s["p_home"] for s in scored], [1.0 if s["home_won"] else 0.0 for s in scored])
                frozen = cal.get("slope") is not None and (cal["slope_hi"] < 0.8 or cal["slope_lo"] > 1.25 or (
                    cal["intercept_lo"] > 0.1 or cal["intercept_hi"] < -0.1))
                bs.set_state(db, f"freeze:{sport}", {"frozen": frozen, **cal, "at": now}, now)
                report["freeze"][sport] = frozen
        week = stats.week_of(now)
        report["scores"] = len(ev.write_week_scores(db, corpus, week, now=now, seed=seed))
        bs.set_state(db, "last_night_run", {"at": now}, now)
        return report
    finally:
        if close:
            db.close()


def counts_shown(target: Any, target_name: str) -> bool:
    """Whether a count target (corners, cards) has beaten climatology through the gate (shown as "model only, low
    confidence"); until then only its record is shown."""
    db, close = _db(target)
    try:
        value = bs.get_state(db, f"counts_shown:{target_name}")
        return bool(isinstance(value, Mapping) and value.get("shown"))
    finally:
        if close:
            db.close()
