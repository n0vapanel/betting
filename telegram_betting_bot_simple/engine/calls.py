"""People's own predictions ("calls"): what the AI thinks of them, and how they score.

A call on a game has a winner with a confidence (the chance the person gives
that team, 50-100%), optionally a final score, and optionally player stat
lines ("LeBron over 27.5 points", each with its own confidence). The AI
judges each part with its own probabilities. Once the game is over the
person, the AI and the bookmakers are scored on the same rules, so "vs AI"
(your points minus the AI's on the games you called) shows who reads games
better; copying the AI scores exactly 0 on it:

- winner: FiveThirtyEight's forecasting game, 25 - 100 x (1 - p)^2 with p the
  chance given to the team that won, rounded to a whole percent, and points
  rounded half up to 0.1: +25 for a 100% call that hits, 0 for 50%, -75 for
  a 100% call that misses. A calibrated 70% call averages only about +4.
- score: max(0, 15 - margin miss) + max(0, 20 - total miss), plus a 10-point
  bullseye when both teams' scores are within 2.
- player lines: the winner rule at 40% weight (+10 to -30); the AI is scored
  on its own chance for the same side (from a negative binomial with each
  stat's measured spread); a player who doesn't play, or an exact whole-number
  line, voids it.

The AI and the market are scored on their last numbers before tip-off.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

from .nba_model import GamePrediction

_NORMAL = NormalDist()
STAT_NAMES = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes", "stl": "steals",
              "blk": "blocks", "tov": "turnovers", "pra": "pts+reb+ast"}
PROP_WEIGHT = 0.4  # a player line counts 40% of a winner call
BULLSEYE = 10.0
CAUTION_GAP = 0.08  # the AI this far from the market gets a "check the news" warning


def _half_up(value: float, places: int = 1) -> float:
    return float(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def forecast_points(probability_of_actual: float, weight: float = 1.0) -> float:
    """FiveThirtyEight forecasting-game points once the result is known.

    ``probability_of_actual`` is the chance the forecaster gave to what
    actually happened; it is rounded to a whole percent first, as in 538's game.
    """
    p = int(Decimal(str(probability_of_actual * 100)).quantize(Decimal(1), rounding=ROUND_HALF_UP)) / 100
    return _half_up(weight * (25.0 - 100.0 * (1.0 - p) ** 2))


def winner_points(probability_of_actual: float) -> float:
    return forecast_points(probability_of_actual)


def margin_points(error: float) -> float:
    return max(0.0, 15.0 - abs(error))


def total_points(error: float) -> float:
    return max(0.0, 20.0 - abs(error))


def score_points(margin: float, total: float, final_home: int, final_away: int) -> float:
    """Points for a predicted margin (home minus away) and total against the final score."""
    home, away = (total + margin) / 2, (total - margin) / 2
    points = margin_points(margin - (final_home - final_away)) + total_points(total - (final_home + final_away))
    if abs(home - final_home) <= 2 and abs(away - final_away) <= 2:
        points += BULLSEYE
    return _half_up(points)


def agreement(ai_probability: float) -> str:
    """How the AI rates a call: agrees (60%+), coin flip (40-60%) or disagrees (under 40%)."""
    if ai_probability >= 0.6:
        return "AI agrees"
    if ai_probability >= 0.4:
        return "Coin flip for the AI"
    return "AI disagrees"


def compare_confidence(yours: float, ai: float) -> str:
    """Your confidence in a side next to the AI's chance for the same side."""
    if ai < 0.5:
        return "Against the AI"
    if abs(yours - ai) <= 0.05:
        return "Same as the AI"
    return "Bolder than the AI" if yours > ai else "More cautious than the AI"


# --- player stat probabilities ------------------------------------------------------------

# Game-to-game variance / mean of each stat for rotation players (2025-26 residuals, matching fitted
# repos and DraftKings' alt ladders): points vary far more than a Poisson count, mostly because minutes do
STAT_DISPERSION = {"pts": 2.9, "reb": 1.4, "ast": 1.4, "fg3m": 1.3, "stl": 1.1, "blk": 1.15, "tov": 1.07, "pra": 3.5}
# Points also have a "short night" (early foul trouble, injury, blowout): about 4% of games at ~30% of normal
SHORT_NIGHT = (0.04, 0.3)
BENCH_DISPERSION = 1.3  # extra spread for players projected under 20 minutes


def stat_probabilities(stat: str, mean: float, line: float, minutes: Optional[float] = None) -> Tuple[float, float, float]:
    """(over, under, push) chances for a player stat line, from a negative binomial around ``mean``.

    Half-point lines can't push; on a whole-number line the exact number is a
    push (the bet is void), so over means line + 1 or more.
    """
    if mean <= 0:
        return 0.0, 1.0, 0.0
    dispersion = STAT_DISPERSION.get(stat, 1.3) * (BENCH_DISPERSION if minutes is not None and minutes < 20 else 1.0)

    def cdf(k: int) -> float:
        if stat == "pts":
            weight, scale = SHORT_NIGHT
            return ((1 - weight) * _negative_binomial_cdf(k, mean / (1 - weight + weight * scale), dispersion)
                    + weight * _negative_binomial_cdf(k, mean * scale / (1 - weight + weight * scale), dispersion))
        return _negative_binomial_cdf(k, mean, dispersion)

    if float(line).is_integer():
        k = int(line)
        under, at_most = cdf(k - 1), cdf(k)
        return 1 - at_most, under, at_most - under
    k = math.floor(line)
    return 1 - cdf(k), cdf(k), 0.0


def stat_over_prob(stat: str, mean: float, line: float, minutes: Optional[float] = None) -> float:
    return stat_probabilities(stat, mean, line, minutes)[0]


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
    """The AI's probabilities for each part of a call, and a short verdict.

    ``players`` maps player id -> projection (with stat means) for player lines.
    The market's numbers are kept too, so the market can be scored as a player.
    """
    market = prediction.sources.get("market", {})
    view: Dict[str, Any] = {
        "home_win": prediction.home_win_prob,
        "home_score": prediction.home_score,
        "away_score": prediction.away_score,
        "margin": prediction.margin,
        "total": prediction.total,
        "margin_sd": prediction.margin_sd,
        "total_sd": prediction.total_sd,
        "market_home_win": market.get("home_win_prob"),
        "market_margin": market.get("margin"),
        "market_total": market.get("total"),
    }
    view["caution"] = (view["market_home_win"] is not None
                       and abs(prediction.home_win_prob - view["market_home_win"]) >= CAUTION_GAP)
    notes: List[str] = []
    side = call.get("winner")
    if side in ("home", "away"):
        team = home if side == "home" else away
        ai_prob = prediction.home_win_prob if side == "home" else 1 - prediction.home_win_prob
        market_prob = None
        if view["market_home_win"] is not None:
            market_prob = view["market_home_win"] if side == "home" else 1 - view["market_home_win"]
        yours = float(call.get("confidence") or 0.5)
        compare = compare_confidence(yours, ai_prob)
        view["winner"] = {"side": side, "team": team, "ai_prob": ai_prob, "your_prob": yours,
                          "market_prob": market_prob, "label": agreement(ai_prob),
                          "compare": compare}
        notes.append(f"{agreement(ai_prob)}: it gives {team} {ai_prob:.0%}"
                     + (f" (bookmakers {market_prob:.0%})" if market_prob is not None else "")
                     + f". You: {yours:.0%}, {compare[0].lower() + compare[1:]}.")
        if view["caution"]:
            notes.append("Caution: the AI is far from the bookmakers on this game; check the injury news.")

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
        over, under, push = stat_probabilities(stat, mean or 0.0, float(prop["line"]), projection.get("minutes"))
        # A push voids the line, so the chance that matters is the chance of winning given it isn't a push
        ai_prob = (over if prop.get("side") == "over" else under) / max(1 - push, 1e-9)
        props.append({**prop, "confidence": float(prop.get("confidence") or 0.6),
                      "player": projection.get("name", prop.get("player")), "ai_mean": mean, "ai_prob": ai_prob,
                      "label": agreement(ai_prob)})
    if props:
        view["props"] = props
        agree = sum(1 for p in props if p["ai_prob"] >= 0.6)
        against = sum(1 for p in props if p["ai_prob"] < 0.4)
        notes.append(f"Player lines: the AI agrees with {agree}, disagrees with {against}"
                     f"{f', and calls {len(props) - agree - against} a coin flip' if len(props) - agree - against else ''}.")
    view["verdict"] = " ".join(notes)
    return view


# --- scoring once the game is over -----------------------------------------------------------

def grade_call(call: Dict[str, Any], ai: Dict[str, Any], final: Dict[str, Any],
               box: Optional[Dict[str, Dict[str, Any]]] = None,
               at_tip_off: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Points for the person, the AI and the market, part by part.

    ``ai`` is the AI's view saved with the call; ``at_tip_off`` (when known) has
    the AI's and the market's last pre-game numbers, which are what they're
    scored on. ``final`` has the home and away scores; ``box`` maps player id ->
    stat line (with "dnp" for players who didn't play).
    """
    home, away = final["home"], final["away"]
    margin, total = home - away, home + away
    last = {**ai, **{k: v for k, v in (at_tip_off or {}).items() if v is not None}}
    parts: List[Dict[str, Any]] = []
    user = ai_total = 0.0
    market_total = None

    if call.get("winner") in ("home", "away") and margin != 0:
        won = "home" if margin > 0 else "away"
        confidence = float(call.get("confidence") or 0.5)
        user_p = confidence if call["winner"] == won else 1 - confidence
        ai_p = last["home_win"] if won == "home" else 1 - last["home_win"]
        part = {"part": "winner", "hit": call["winner"] == won,
                "points": forecast_points(user_p), "ai_points": forecast_points(ai_p)}
        if last.get("market_home_win") is not None:
            market_p = last["market_home_win"] if won == "home" else 1 - last["market_home_win"]
            part["market_points"] = forecast_points(market_p)
            market_total = part["market_points"]
        parts.append(part)
        user += part["points"]
        ai_total += part["ai_points"]

    if call.get("home_score") is not None and call.get("away_score") is not None:
        your_margin = call["home_score"] - call["away_score"]
        your_total = call["home_score"] + call["away_score"]
        part = {
            "part": "score",
            "points": score_points(your_margin, your_total, home, away),
            "ai_points": score_points(last["margin"], last["total"], home, away),
            "margin_off": your_margin - margin,
            "total_off": your_total - total,
        }
        parts.append(part)
        user += part["points"]
        ai_total += part["ai_points"]

    for prop in ai.get("props") or []:
        line = (box or {}).get(str(prop.get("player_id")))
        name = prop.get("player")
        if not line or line.get("dnp"):
            parts.append({"part": "prop", "player": name, "void": True, "points": 0.0, "ai_points": 0.0})
            continue
        stat = prop["stat"]
        actual = line.get(stat) if stat != "pra" else sum(line.get(s, 0) for s in ("pts", "reb", "ast"))
        if actual == float(prop["line"]):
            parts.append({"part": "prop", "player": name, "void": True, "actual": actual, "points": 0.0, "ai_points": 0.0})
            continue
        hit = (actual > float(prop["line"])) == (prop["side"] == "over")
        confidence = float(prop.get("confidence") or 0.6)
        part = {"part": "prop", "player": name, "stat": stat, "line": prop["line"], "side": prop["side"],
                "actual": actual, "hit": hit,
                "points": forecast_points(confidence if hit else 1 - confidence, PROP_WEIGHT),
                "ai_points": forecast_points(prop["ai_prob"] if hit else 1 - prop["ai_prob"], PROP_WEIGHT)}
        parts.append(part)
        user += part["points"]
        ai_total += part["ai_points"]

    return {"home": home, "away": away, "parts": parts, "points": _half_up(user), "ai_points": _half_up(ai_total),
            "market_points": market_total, "vs_ai": _half_up(user - ai_total)}


# --- the AI's track record --------------------------------------------------------------------

CALIBRATION_BANDS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0))


def _record() -> Dict[str, int]:
    return {"won": 0, "lost": 0, "push": 0}


def _log_loss(p: float, happened: bool) -> float:
    p = min(max(p, 0.01), 0.99)
    return -math.log(p if happened else 1 - p)


def track_record(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise graded AI predictions (store rows with payload and result).

    Brier score is the mean squared error of the home-win probability (lower
    is better; always saying 50% scores 0.25), shown next to the bookmakers'
    and an "unskilled" baseline that always gives the home team this sample's
    home-win rate. The spread and total records are against ESPN's closing
    line, the honest test of a betting model.
    """
    games = hits = market_games = home_wins = 0
    brier = market_brier = log_loss = margin_error = total_error = 0.0
    ats, totals = _record(), _record()
    bands = [{"low": low, "high": high, "games": 0, "predicted": 0.0, "won": 0} for low, high in CALIBRATION_BANDS]
    outcomes = []
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
        home_wins += home_won
        outcomes.append(home_won)
        hits += (home_win >= 0.5) == home_won
        brier += (home_win - home_won) ** 2
        log_loss += _log_loss(home_win, home_won)
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
    base_rate = home_wins / games if games else None
    return {
        "games": games,
        "winner_accuracy": hits / games if games else None,
        "brier": brier / games if games else None,
        "market_brier": market_brier / market_games if market_games else None,
        "baseline_brier": sum((base_rate - o) ** 2 for o in outcomes) / games if games else None,
        "log_loss": log_loss / games if games else None,
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
