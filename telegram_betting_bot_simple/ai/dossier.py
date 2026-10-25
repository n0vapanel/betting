"""The facts an analyst gets about one game, as compact JSON.

Everything comes from what the dashboard already has: the bookmakers'
prices, the AI's prediction (before any analyst moved it), ESPN's
predictor, injuries, rest, team scoring and the players' projections.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _chance(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 3)


def news_request(card: Dict[str, Any], today: str) -> str:
    """What Claude's news check is asked for one game."""
    injuries = card.get("injuries") or {}
    known = "; ".join(f"{card[side]}: {', '.join(injuries.get(side) or []) or 'none listed'}" for side in ("away", "home"))
    return (f"Game: {card['away']} at {card['home']}, tip-off {card['when']} ({card['start']}). Today is {today}.\n"
            f"Injury report we already have (ESPN): {known}.")


def build_dossier(card: Dict[str, Any], info: Dict[str, Any], players: Dict[str, Dict[str, Any]],
                  record_note: str, news: str = "") -> Dict[str, Any]:
    prediction = card["prediction"]
    markets = card["markets"]
    before = prediction["sources"].get("before_analysts", {})
    h2h, spreads, totals = markets.get("h2h", {}), markets.get("spreads", {}), markets.get("totals", {})
    dossier: Dict[str, Any] = {
        "game": {"away": card["away"], "home": card["home"], "tip_off": card["when"], "start_utc": card["start"]},
        "bookmakers": {
            "home_win": _chance((h2h.get("home") or {}).get("fair")),
            "moneyline": {side: (h2h.get(side) or {}).get("american") for side in ("home", "away")},
            "spread": {"home_point": (spreads.get("home") or {}).get("point"),
                       "home_cover": _chance((spreads.get("home") or {}).get("fair"))},
            "total": {"line": (totals.get("over") or {}).get("point"),
                      "over": _chance((totals.get("over") or {}).get("fair"))},
        },
        "ai_model": {
            "home_win": _chance(before.get("home_win_prob", prediction["home_win"])),
            "home_score": prediction["home_score"],
            "away_score": prediction["away_score"],
            "home_margin": round(before.get("margin", prediction["margin"]), 1),
            "total": prediction["total"],
            "home_win_by_source": {name: values.get("home_win_prob") for name, values in prediction["sources"].items()
                                   if name in ("market", "espn", "model")},
            "factors": prediction.get("factors", []),
        },
        "espn_home_win": _chance(card.get("espn_home_win")),
        "teams": {},
        "players": {},
        "your_record": record_note,
    }
    for side in ("away", "home"):
        team = (info.get("teams") or {}).get(side)
        rating = getattr(team, "rating", None)
        dossier["teams"][side] = {
            "name": card[side],
            "injury_report": (card.get("injuries") or {}).get(side, []),
            "out_in_model": list(getattr(team, "injured", []) or []),
            "back_to_back": bool(getattr(team, "back_to_back", False)),
            "points_for": round(rating.points_for, 1) if rating else None,
            "points_against": round(rating.points_against, 1) if rating else None,
            "games_this_season": int(rating.games) if rating else None,
        }
        rotation: List[Dict[str, Any]] = sorted(
            (row for row in players.values() if row.get("side") == side and not row.get("out")),
            key=lambda row: row.get("minutes") or 0, reverse=True,
        )[:8]
        dossier["players"][side] = [
            {"name": row["name"], "status": row.get("status") or "active", "minutes": row.get("minutes"),
             "pts": row.get("pts"), "reb": row.get("reb"), "ast": row.get("ast"), "threes": row.get("fg3m")}
            for row in rotation
        ]
    if news:
        dossier["latest_news"] = news
    return dossier


def dossier_text(dossier: Dict[str, Any]) -> str:
    return json.dumps(dossier, ensure_ascii=False, separators=(",", ":"))
