"""How the AI learns from its analysts' graded forecasts.

Each analyst's forecast is read as a move away from the AI's own win chance
at the time it was asked. After the games, a logistic regression with the
AI's chance as a fixed offset fits how much of each analyst's move the
results support. The AI only keeps the part it is 95% sure of (the fit
minus 1.645 standard errors), and an analyst needs a minimum number of
graded games before it gets any say. That way a lucky week can't buy an
analyst influence, and one that keeps adding real information slowly earns
it (its "trust", 0-100%).
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..engine.calls import winner_points

MIN_GAMES = 40  # graded forecasts an analyst needs before the AI uses it at all
RIDGE = 0.5     # a light penalty that keeps the fit stable when moves are tiny
SURE = 1.645    # standard errors taken off the fit: keep only what's 95% sure to help

Row = Tuple[float, int, Dict[str, float]]  # (the AI's log-odds, home won, each analyst's move)


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def training_rows(analyses: Sequence[Dict[str, Any]]) -> List[Row]:
    """One row per graded game: the AI's chance (as log-odds), the result, and each analyst's move."""
    games: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in analyses:
        payload, result = row.get("payload") or {}, row.get("result") or {}
        if row.get("status") != "graded" or payload.get("base_home_win") is None or "home_won" not in result:
            continue
        game = games.setdefault((row["league"], row["event_id"]), {
            "base": payload["base_home_win"], "won": int(bool(result["home_won"])), "moves": {},
        })
        game["moves"][row["analyst"]] = logit(payload["home_win_prob"]) - logit(payload["base_home_win"])
    return [(logit(g["base"]), g["won"], g["moves"]) for g in games.values()]


def _solve(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Solve a small linear system by Gaussian elimination with partial pivoting."""
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (a[r][n] - sum(a[r][c] * x[c] for c in range(r + 1, n))) / a[r][r]
    return x


def fit_trust(rows: Sequence[Row], min_games: int = MIN_GAMES, ridge: float = RIDGE,
              sure: float = SURE) -> Dict[str, float]:
    """Each analyst's trust (0-1): the share of its move away from the AI that the AI keeps."""
    counts = Counter(key for _, _, moves in rows for key in moves)
    keys = sorted(key for key, n in counts.items() if n >= min_games)
    trust = {key: 0.0 for key in counts}
    if not keys:
        return trust
    weights = [0.0] * len(keys)
    for _ in range(50):  # Newton's method; converges in a handful of steps
        gradient = [ridge * w for w in weights]
        hessian = [[ridge if i == j else 0.0 for j in range(len(keys))] for i in range(len(keys))]
        for offset, won, moves in rows:
            x = [moves.get(key, 0.0) for key in keys]
            if not any(x):
                continue
            p = sigmoid(offset + sum(w * xi for w, xi in zip(weights, x)))
            for i, xi in enumerate(x):
                gradient[i] += (p - won) * xi
                for j, xj in enumerate(x):
                    hessian[i][j] += p * (1 - p) * xi * xj
        step = _solve(hessian, gradient)
        weights = [w - s for w, s in zip(weights, step)]
        if max(abs(s) for s in step) < 1e-7:
            break
    # Standard errors from the inverse Hessian at the fit
    for i, (key, w) in enumerate(zip(keys, weights)):
        unit = [1.0 if j == i else 0.0 for j in range(len(keys))]
        error = math.sqrt(max(_solve(hessian, unit)[i], 0.0))
        trust[key] = min(max(w - sure * error, 0.0), 1.0)
    return trust


def record(analyses: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One analyst's graded forecasts next to the AI's and the bookmakers' on the same games."""
    graded = [row for row in analyses if row.get("status") == "graded" and "home_won" in (row.get("result") or {})]
    if not graded:
        return {"games": 0}
    out: Dict[str, Any] = {"games": len(graded)}
    won = [int(bool(row["result"]["home_won"])) for row in graded]
    yours = [row["payload"]["home_win_prob"] for row in graded]
    ai = [row["payload"].get("base_home_win") for row in graded]
    market = [row["payload"].get("market_home_win") for row in graded]

    def brier(probs: List[Optional[float]]) -> Optional[float]:
        pairs = [(p, y) for p, y in zip(probs, won) if p is not None]
        return round(mean((p - y) ** 2 for p, y in pairs), 4) if len(pairs) == len(won) else None

    out["brier"], out["brier_ai"], out["brier_market"] = brier(yours), brier(ai), brier(market)
    calls = [(p, y) for p, y in zip(yours, won) if p != 0.5]
    out["accuracy"] = round(mean((p > 0.5) == bool(y) for p, y in calls), 3) if calls else None
    out["points"] = round(sum(winner_points(p if y else 1 - p) for p, y in zip(yours, won)), 1)
    if all(p is not None for p in ai):
        out["ai_points"] = round(sum(winner_points(p if y else 1 - p) for p, y in zip(ai, won)), 1)
    # How far it strays from the bookmakers, and whether it is bolder than them
    paired = [(p, m) for p, m in zip(yours, market) if m is not None]
    if paired:
        out["average_move"] = round(mean(abs(p - m) for p, m in paired), 3)
        out["bolder_by"] = round(mean(abs(p - 0.5) - abs(m - 0.5) for p, m in paired), 3)
    misses = sorted(zip(graded, yours, won), key=lambda item: item[1] if not item[2] else 1 - item[1], reverse=True)
    out["misses"] = [
        {"game": f"{row['away']} @ {row['home']}", "home_win_prob": p, "home_won": bool(y),
         "market_home_win": row["payload"].get("market_home_win")}
        for row, p, y in misses[:3] if (p if not y else 1 - p) > 0.5
    ]
    return out


def feedback_note(stats: Dict[str, Any]) -> str:
    """The analyst's record in a few sentences, for its next dossier."""
    games = stats.get("games", 0)
    if not games:
        return "No graded forecasts yet."
    parts = [f"{games} graded forecast{'s' if games != 1 else ''}"
             + (" (a small sample: read it loosely)" if games < 30 else "") + "."]
    if stats.get("accuracy") is not None:
        parts.append(f"Winners called: {stats['accuracy']:.0%}.")
    if stats.get("brier") is not None and stats.get("brier_market") is not None:
        verdict = "better than" if stats["brier"] < stats["brier_market"] - 0.002 else (
            "worse than" if stats["brier"] > stats["brier_market"] + 0.002 else "about the same as")
        parts.append(f"Brier score {stats['brier']:.3f}, {verdict} the bookmakers' {stats['brier_market']:.3f} "
                     "on the same games (lower is better).")
        if verdict != "better than":
            parts.append("Your moves away from the market have not helped so far: stay closer to it unless you "
                         "have specific news.")
    if stats.get("bolder_by") is not None and abs(stats["bolder_by"]) >= 0.02:
        parts.append(f"On average you are {'bolder' if stats['bolder_by'] > 0 else 'more cautious'} than the "
                     f"bookmakers by {abs(stats['bolder_by']) * 100:.0f} percentage points.")
    for miss in stats.get("misses", [])[:2]:
        parts.append(f"Missed: {miss['game']}, you gave the home team {miss['home_win_prob']:.0%} and "
                     f"{'it won' if miss['home_won'] else 'it lost'}.")
    return " ".join(parts)
