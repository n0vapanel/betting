"""Synthetic worlds for the learning-loop tests (no network): graded consensus_log and context_log rows, soccer
matches with closing odds, scores and corners/cards, and NBA model predictions, each drawn from a known truth so a
test can plant an improvement, a regression or pure noise.

Every chance shown to the brain is a noisy view of the true chance; outcomes are drawn from the truth (or, with
``truth="pool"``, from the log pool of the sources at planted weights, so a refit has a right answer to recover).
"""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from telegram_betting_bot_simple.engine import consensus, soccer
from telegram_betting_bot_simple.learn import evaluate as ev
from telegram_betting_bot_simple.learn import stats

DAY = 86400.0
T0 = 1_700_006_400.0  # Wednesday 2023-11-15 00:00 UTC


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _src(name: str, kind: str, p_home: float, as_of: float, status: str = "ok") -> Dict[str, Any]:
    p = min(max(p_home, 0.02), 0.98)
    return {"name": name, "kind": kind, "p": {"home": p, "away": 1 - p}, "as_of": consensus._iso(as_of),
            "status": status}


NOISE = {"sharp": 0.20, "us_books": 0.25, "prediction_market": 0.30, "model_rating": 0.60}


def consensus_world(n: int, *, league: str = "basketball_nba", seed: int = 0, per_day: int = 8,
                    start: float = T0, noise: Optional[Mapping[str, float]] = None, pm_share: float = 0.6,
                    truth: str = "latent", pool_weights: Optional[Mapping[str, float]] = None,
                    near_share: float = 0.1, params_version: int = 0, values: Optional[Mapping[str, float]] = None,
                    game_prefix: str = "g") -> List[Dict[str, Any]]:
    """``n`` graded games (records shaped like ``evaluate.consensus_records``), ``per_day`` a day from ``start``.
    Sources: sharp, US books, a prediction market (``pm_share`` of games), a rating; each a logit-normal view of the
    truth with ``noise`` SDs. The stored blend is the real blend with ``values`` (registry values; default the
    defaults). ``truth="pool"`` draws the outcome from the pool of the sources at ``pool_weights`` instead."""
    rng = np.random.default_rng(seed)
    noise = {**NOISE, **(noise or {})}
    kw = ev.blend_kwargs(ev.registry.sport_key(league), values or {})
    out = []
    for i in range(n):
        day = i // per_day
        begin = start + day * DAY + 15 * 3600 + (i % per_day) * 1800
        near = rng.random() < near_share
        logged = begin - (rng.uniform(60, 800) if near else rng.uniform(1800, 20 * 3600))
        q = _sig(rng.normal(0.25, 0.9))
        sources = [_src("Sharp", "sharp", _sig(_logit(q) + rng.normal(0, noise["sharp"])), logged),
                   _src("US books", "us_books", _sig(_logit(q) + rng.normal(0, noise["us_books"])), logged)]
        if rng.random() < pm_share:
            sources.append(_src("Kalshi", "prediction_market",
                                _sig(_logit(q) + rng.normal(0, noise["prediction_market"])), logged))
        sources.append(_src("ESPN", "model_rating", _sig(0.8 * _logit(q) + rng.normal(0, noise["model_rating"])),
                            logged))
        blend = consensus.blend(sources, league, now=logged, start=begin, **kw)
        if truth == "pool":
            pooled = consensus.blend(sources, league, now=logged, start=begin, weights=dict(pool_weights or {}))
            q = pooled["p"]["home"]
        outcome = "home" if rng.random() < q else "away"
        out.append({"league": league, "sport": ev.registry.sport_key(league), "game_key": f"espn:{game_prefix}{i}",
                    "start": begin, "logged_at": logged, "sources": sources, "blend": blend, "outcome": outcome,
                    "result": {"score": {"home": 100 + (outcome == "home") * 5, "away": 100}},
                    "params_version": params_version, "neutral": False, "day": stats.day_of(begin),
                    "week": stats.week_of(begin), "truth": q})
    return out


def write_consensus(db: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> None:
    """Insert graded rows into consensus_log (creating the table's ``params_version`` column if missing)."""
    columns = [r[1] for r in db.execute("PRAGMA table_info(consensus_log)").fetchall()]
    has_version = "params_version" in columns
    for r in rows:
        values = [r["league"], r["game_key"], None, None, 0, "Home", "Away", consensus._iso(r["start"]),
                  r["logged_at"], json.dumps(r["sources"]), json.dumps(r["blend"]), "graded", r["outcome"],
                  json.dumps(r.get("result") or {}), r["start"] + 4 * 3600]
        names = ("league, game_key, event_id, espn_id, swapped, home, away, start, logged_at, sources, blend, status, "
                 "outcome, result, graded_at")
        if has_version:
            names += ", params_version"
            values.append(r.get("params_version", 0))
        db.execute(f"INSERT OR REPLACE INTO consensus_log ({names}) VALUES ({', '.join('?' * len(values))})", values)


def context_world(n: int, *, league: str = "americanfootball_nfl", seed: int = 0, per_day: int = 6,
                  start: float = T0, adjusted_share: float = 0.3, beta: float = 1.0, sd: float = 13.2,
                  line_mean: float = 45.0, weight: float = 0.5) -> List[Dict[str, Any]]:
    """Graded context_log records: a line, the books' p(over), and on ``adjusted_share`` of games a weather change
    (raw, in points) of which ``beta`` x raw really shows up in the final total. ``p_after`` is the real
    ``context.adjusted_over`` with ``weight``."""
    from telegram_betting_bot_simple.engine import context

    rng = np.random.default_rng(seed)
    sp = context.params(league)
    out = []
    for i in range(n):
        day = i // per_day
        begin = start + day * DAY + 17 * 3600 + (i % per_day) * 3600
        logged = begin - rng.uniform(600, 6 * 3600)
        line = round(line_mean + rng.normal(0, 3)) + 0.5
        mu_market = line + rng.normal(0, 1.0)
        p_before = 1 - context._N.cdf((line - mu_market) / sd)
        raw = float(rng.choice([-4.0, -3.0, -2.0, 2.0, 3.0])) if rng.random() < adjusted_share else 0.0
        adjustment = None
        applied = 0.0
        p_after = p_before
        if raw:
            adj = context.adjustment(league, "total", raw, 0.0, weight=weight)
            applied = adj.applied
            p_after = context.adjusted_over(sp, line, p_before, applied)
            adjustment = {"kind": "total", "raw": adj.raw, "applied": adj.applied, "cap": adj.cap,
                          "weight": adj.weight, "capped": adj.capped, "full": adj.full, "reason": adj.reason}
        total = mu_market + beta * raw + rng.normal(0, sd)
        total = float(round(total))
        if total == line:
            total += 1
        out.append({"league": league, "sport": ev.registry.sport_key(league), "game_key": f"espn:c{i}",
                    "start": begin, "logged_at": logged, "line": line, "p_before": round(p_before, 4),
                    "p_after": round(p_after, 4), "applied": round(applied, 4), "adjustment": adjustment,
                    "y": 1 if total > line else 0, "total": total, "params_version": 0, "day": stats.day_of(begin),
                    "week": stats.week_of(begin)})
    return out


def matches_world(n: int, *, seed: int = 0, rho: float = -0.13, per_day: int = 10, start: float = T0,
                  corners: Optional[Mapping[str, float]] = None, cards: Optional[Mapping[str, float]] = None
                  ) -> List[ev.MatchRecord]:
    """Soccer matches whose closing 1X2 and over 2.5 are the true Dixon-Coles grid's (at ``rho``), with the score
    drawn from that grid and corners and cards from negative binomials with the given count-model terms."""
    rng = np.random.default_rng(seed)
    corners = corners or {"intercept": 1.63, "home": 0.18, "supremacy": 0.12, "total": 0.05, "size": 15.0}
    cards = cards or {"intercept": 0.64, "home": -0.12, "supremacy": -0.08, "total": -0.03, "size": 12.0}
    out = []
    for i in range(n):
        day = i // per_day
        begin = start + day * DAY + 14 * 3600 + (i % per_day) * 1200
        lam, mu = float(rng.uniform(0.7, 2.4)), float(rng.uniform(0.5, 1.9))
        g = np.array(soccer.grid(lam, mu, rho))
        h, d, a = soccer.outcome_probs(g.tolist())
        over = soccer.over_prob(g.tolist(), 2.5)
        flat = rng.choice(g.size, p=g.ravel() / g.sum())
        hg, ag = divmod(int(flat), g.shape[1])
        made = {}
        for name, model in (("corners", corners), ("cards", cards)):
            mh, ma = ev.count_means(model, np.array([lam - mu]), np.array([lam + mu]))
            r = model["size"]
            made[name] = (int(rng.negative_binomial(r, r / (r + mh[0]))), int(rng.negative_binomial(r, r / (r + ma[0]))))
        out.append(ev.MatchRecord("soccer_epl", f"espn:m{i}", begin, h, d, a, over, hg, ag, made, begin - 120))
    return out


def nba_world(n: int, *, seed: int = 0, per_day: int = 8, start: float = T0) -> List[Dict[str, Any]]:
    """Graded ai_predictions records whose logged chance is the NBA model's own at its default weights."""
    from telegram_betting_bot_simple.learn import tune

    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        day = i // per_day
        begin = start + day * DAY + 23 * 3600 + (i % per_day) * 1800
        truth = rng.normal(2.0, 7.0)
        total = 228 + rng.normal(0, 8)
        win_sd = 0.80 * math.sqrt(total)
        market_margin = truth + rng.normal(0, 1.0)
        payload = {"model": "v1", "total": total, "market_margin": market_margin,
                   "market_home_win": tune._N.cdf(market_margin / win_sd),
                   "espn_home_win": tune._N.cdf((truth + rng.normal(0, 3)) / win_sd),
                   "model_home_win": tune._N.cdf((truth + rng.normal(0, 4)) / win_sd)}
        payload["home_win"] = tune._nba_chance(payload, 0.9, 0.05, tune.nba_model.Params())
        final = truth + rng.normal(0, 12)
        y = 1 if final > 0 else 0
        out.append({"league": "basketball_nba", "sport": "nba", "start": begin, "logged_at": begin - 3600,
                    "payload": payload, "y": y, "margin_final": float(final), "total_final": float(total),
                    "day": stats.day_of(begin), "week": stats.week_of(begin)})
    return out
