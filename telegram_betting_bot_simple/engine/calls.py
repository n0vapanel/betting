"""People's own predictions ("calls"): what the AI thinks of them, and how they score.

A call on a game has a winner with a confidence (the chance the person gives
that team, 50-100%), optionally a final score, and optionally player stat
lines ("LeBron over 27.5 points"). The AI judges each part with its own
probabilities, and once the game is over both are scored on the same scale,
so people can see whether they beat the AI:

- winner: FiveThirtyEight's forecasting-game points, 25 - 100 x (1 - p)^2,
  where p is the chance given to the team that actually won (+25 for a
  100% call that hits, 0 for a coin flip, -75 for a 100% call that misses);
- score: up to 10 points each for how close the margin and the total were;
- player lines: +10 when right, -10 when wrong (the AI takes the side it
  rates above 50%).
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional

from .nba_model import GamePrediction

_NORMAL = NormalDist()
STAT_NAMES = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes", "stl": "steals",
              "blk": "blocks", "tov": "turnovers", "pra": "pts+reb+ast"}
PROP_POINTS = 10.0
CLOSE_POINTS = 10.0  # for a margin or total within 1 point; less the further off


def winner_points(probability_of_actual: float) -> float:
    """FiveThirtyEight forecasting-game points for a win probability once the result is known."""
    return 25.0 - 100.0 * (1.0 - probability_of_actual) ** 2


def closeness_points(error: float, full_within: float, zero_at: float) -> float:
    """CLOSE_POINTS when ``error`` <= ``full_within``, falling linearly to 0 at ``zero_at``."""
    error = abs(error)
    if error <= full_within:
        return CLOSE_POINTS
    if error >= zero_at:
        return 0.0
    return CLOSE_POINTS * (zero_at - error) / (zero_at - full_within)


# --- player stat probabilities ------------------------------------------------------------

def stat_over_prob(stat: str, mean: float, line: float) -> float:
    """Chance a player's stat goes over ``line`` given the projected ``mean``.

    Counting stats use a negative binomial (a Poisson with extra game-to-game
    spread); points and points+rebounds+assists use a normal with a spread
    that grows with the mean. Half-point lines have no push; for a whole
    number line, "over" means at least line + 1.
    """
    if mean <= 0:
        return 0.0
    if stat in ("pts", "pra"):
        sd = max(1.0, STAT_SD_SCALE[stat] * math.sqrt(mean) + STAT_SD_FLOOR[stat])
        threshold = math.floor(line) + 1  # whole points: over 27.5 means 28 or more
        return 1 - _NORMAL.cdf((threshold - 0.5 - mean) / sd)
    dispersion = STAT_DISPERSION.get(stat, 1.2)
    threshold = math.floor(line) + 1
    return 1 - _negative_binomial_cdf(threshold - 1, mean, dispersion)


# Spread of points around the projection: sd = scale * sqrt(mean) + floor
STAT_SD_SCALE = {"pts": 1.2, "pra": 1.25}
STAT_SD_FLOOR = {"pts": 1.0, "pra": 1.5}
# Variance / mean for counting stats (1.0 would be a plain Poisson)
STAT_DISPERSION = {"reb": 1.3, "ast": 1.4, "fg3m": 1.25, "stl": 1.1, "blk": 1.4, "tov": 1.15}


def _negative_binomial_cdf(k: int, mean: float, variance_ratio: float) -> float:
    """P(X <= k) for a count with this mean and variance = variance_ratio x mean."""
    if k < 0:
        return 0.0
    if variance_ratio <= 1.0001:
        # Poisson
        term = math.exp(-mean)
        total = term
        for i in range(1, k + 1):
            term *= mean / i
            total += term
        return min(total, 1.0)
    p = 1.0 / variance_ratio  # success probability
    r = mean * p / (1 - p)  # number of successes
    term = p ** r  # P(X = 0)
    total = term
    for i in range(1, k + 1):
        term *= (i - 1 + r) / i * (1 - p)
        total += term
    return min(total, 1.0)


# --- the AI's view of a call ----------------------------------------------------------------

def judge_call(prediction: GamePrediction, call: Dict[str, Any], home: str, away: str,
               players: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The AI's probabilities for each part of a call, and a one-line verdict.

    ``players`` maps player id -> projection (with stat means) for player lines.
    """
    view: Dict[str, Any] = {
        "home_win": prediction.home_win_prob,
        "home_score": prediction.home_score,
        "away_score": prediction.away_score,
        "margin": prediction.margin,
        "total": prediction.total,
        "margin_sd": prediction.margin_sd,
        "total_sd": prediction.total_sd,
    }
    notes: List[str] = []
    side = call.get("winner")
    if side in ("home", "away"):
        team = home if side == "home" else away
        ai_prob = prediction.home_win_prob if side == "home" else 1 - prediction.home_win_prob
        view["winner"] = {"side": side, "team": team, "ai_prob": ai_prob, "your_prob": call.get("confidence")}
        if ai_prob >= 0.5:
            notes.append(f"The AI also picks {team} ({ai_prob:.0%}).")
        else:
            notes.append(f"The AI disagrees: it gives {team} {ai_prob:.0%}.")

    if call.get("home_score") is not None and call.get("away_score") is not None:
        margin = call["home_score"] - call["away_score"]
        total = call["home_score"] + call["away_score"]
        winner_side = "home" if margin > 0 else "away"
        by = abs(margin)
        # Chance the predicted winner wins by at least the predicted margin
        if winner_side == "home":
            by_at_least = 1 - _NORMAL.cdf((by - 0.5 - prediction.margin) / prediction.margin_sd)
        else:
            by_at_least = _NORMAL.cdf((-by + 0.5 - prediction.margin) / prediction.margin_sd)
        within = lambda centre, mu, sd, k: _NORMAL.cdf((centre + k - mu) / sd) - _NORMAL.cdf((centre - k - mu) / sd)
        view["score"] = {
            "margin": margin,
            "total": total,
            "winner_side": winner_side,
            "wins_by_at_least": by_at_least,
            "margin_within_5": within(margin, prediction.margin, prediction.margin_sd, 5.5),
            "total_over_yours": 1 - _NORMAL.cdf((total - prediction.total) / prediction.total_sd),
            "total_within_10": within(total, prediction.total, prediction.total_sd, 10.5),
        }
        team = home if winner_side == "home" else away
        notes.append(f"{team} by {by}+ happens {by_at_least:.0%} of the time in the AI's model; "
                     f"it expects a total near {prediction.total:.0f} (yours {total}).")

    props = []
    seen = set()
    for prop in call.get("props") or []:
        projection = (players or {}).get(str(prop.get("player_id")))
        stat = prop.get("stat")
        key = (str(prop.get("player_id")), stat, float(prop.get("line", 0)), prop.get("side"))
        if not projection or stat not in STAT_NAMES or key in seen:
            continue
        seen.add(key)
        mean = projection.get(stat) if stat != "pra" else sum(projection.get(s, 0.0) for s in ("pts", "reb", "ast"))
        over = stat_over_prob(stat, mean or 0.0, float(prop["line"]))
        ai_prob = over if prop.get("side") == "over" else 1 - over
        props.append({**prop, "player": projection.get("name", prop.get("player")), "ai_mean": mean, "ai_prob": ai_prob})
    if props:
        view["props"] = props
        agree = sum(1 for p in props if p["ai_prob"] >= 0.5)
        notes.append(f"The AI backs {agree} of your {len(props)} player call(s).")
    view["verdict"] = " ".join(notes)
    return view


# --- scoring once the game is over -----------------------------------------------------------

def grade_call(call: Dict[str, Any], ai: Dict[str, Any], final: Dict[str, Any],
               box: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Points for the person and the AI, part by part.

    ``final`` has the home and away scores; ``box`` maps player id -> stat line
    (with "dnp" for players who didn't play, whose lines are void).
    """
    home, away = final["home"], final["away"]
    margin, total = home - away, home + away
    parts: List[Dict[str, Any]] = []
    user = ai_total = 0.0

    if call.get("winner") in ("home", "away") and margin != 0:
        won = "home" if margin > 0 else "away"
        confidence = float(call.get("confidence") or 0.5)
        user_p = confidence if call["winner"] == won else 1 - confidence
        ai_home = ai["home_win"]
        ai_p = ai_home if won == "home" else 1 - ai_home
        part = {"part": "winner", "hit": call["winner"] == won,
                "points": winner_points(user_p), "ai_points": winner_points(ai_p)}
        parts.append(part)
        user += part["points"]
        ai_total += part["ai_points"]

    if call.get("home_score") is not None and call.get("away_score") is not None:
        your_margin = call["home_score"] - call["away_score"]
        your_total = call["home_score"] + call["away_score"]
        part = {
            "part": "score",
            "points": closeness_points(your_margin - margin, 1, 12) + closeness_points(your_total - total, 2, 25),
            "ai_points": closeness_points(ai["margin"] - margin, 1, 12) + closeness_points(ai["total"] - total, 2, 25),
            "margin_off": your_margin - margin,
            "total_off": your_total - total,
        }
        parts.append(part)
        user += part["points"]
        ai_total += part["ai_points"]

    for prop in ai.get("props") or []:
        line = (box or {}).get(str(prop.get("player_id")))
        if not line or line.get("dnp"):
            parts.append({"part": "prop", "player": prop.get("player"), "void": True, "points": 0.0, "ai_points": 0.0})
            continue
        stat = prop["stat"]
        actual = line.get(stat) if stat != "pra" else sum(line.get(s, 0) for s in ("pts", "reb", "ast"))
        if actual == float(prop["line"]):
            parts.append({"part": "prop", "player": prop.get("player"), "void": True, "actual": actual,
                          "points": 0.0, "ai_points": 0.0})
            continue
        went_over = actual > float(prop["line"])
        hit = went_over == (prop["side"] == "over")
        ai_hit = hit if prop["ai_prob"] >= 0.5 else not hit
        part = {"part": "prop", "player": prop.get("player"), "stat": stat, "line": prop["line"], "side": prop["side"],
                "actual": actual, "hit": hit, "points": PROP_POINTS if hit else -PROP_POINTS,
                "ai_points": PROP_POINTS if ai_hit else -PROP_POINTS}
        parts.append(part)
        user += part["points"]
        ai_total += part["ai_points"]

    return {"home": home, "away": away, "parts": parts, "points": round(user, 2), "ai_points": round(ai_total, 2)}


# --- the AI's track record --------------------------------------------------------------------

CALIBRATION_BANDS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0))


def _record() -> Dict[str, int]:
    return {"won": 0, "lost": 0, "push": 0}


def track_record(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise graded AI predictions (store rows with payload and result).

    Brier score is the mean squared error of the home-win probability (lower
    is better; always saying 50% scores 0.25). The spread and total records
    are against ESPN's closing line, the honest test of a betting model.
    """
    games = hits = market_games = 0
    brier = market_brier = margin_error = total_error = 0.0
    ats, totals = _record(), _record()
    bands = [{"low": low, "high": high, "games": 0, "predicted": 0.0, "won": 0} for low, high in CALIBRATION_BANDS]
    recent = []
    for row in rows:
        payload, result = row.get("payload") or {}, row.get("result") or {}
        if result.get("home") is None or payload.get("home_win") is None:
            continue
        margin, total = result["home"] - result["away"], result["home"] + result["away"]
        if margin == 0:
            continue
        home_won = margin > 0
        home_win = payload["home_win"]
        games += 1
        hits += (home_win >= 0.5) == home_won
        brier += (home_win - home_won) ** 2
        if payload.get("market_home_win") is not None:
            market_games += 1
            market_brier += (payload["market_home_win"] - home_won) ** 2
        margin_error += abs(payload["margin"] - margin)
        total_error += abs(payload["total"] - total)
        favourite = max(home_win, 1 - home_win)
        band = next(b for b in bands if favourite < b["high"] or b["high"] == 1.0)
        band["games"] += 1
        band["predicted"] += favourite
        band["won"] += home_won if home_win >= 0.5 else not home_won

        closing = result.get("closing") or {}
        spread, line = closing.get("home_spread"), closing.get("total")
        if spread is not None and payload["margin"] + spread != 0:
            cover = margin + spread
            ats["push" if cover == 0 else "won" if (cover > 0) == (payload["margin"] + spread > 0) else "lost"] += 1
        if line is not None and payload["total"] != line:
            totals["push" if total == line else "won" if (total > line) == (payload["total"] > line) else "lost"] += 1
        if len(recent) < 20:
            recent.append({"home": row["home"], "away": row["away"], "start": row["start"],
                           "home_win": home_win, "margin": payload["margin"], "total": payload["total"],
                           "final_home": result["home"], "final_away": result["away"],
                           "hit": (home_win >= 0.5) == home_won, "closing": closing or None})
    return {
        "games": games,
        "winner_accuracy": hits / games if games else None,
        "brier": brier / games if games else None,
        "market_brier": market_brier / market_games if market_games else None,
        "margin_error": margin_error / games if games else None,
        "total_error": total_error / games if games else None,
        "ats": ats,
        "totals": totals,
        "calibration": [
            {"range": f"{b['low']:.0%}-{b['high']:.0%}", "games": b["games"],
             "predicted": b["predicted"] / b["games"] if b["games"] else None,
             "actual": b["won"] / b["games"] if b["games"] else None}
            for b in bands
        ],
        "recent": recent,
    }
