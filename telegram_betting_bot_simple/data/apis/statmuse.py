"""StatMuse counting stats for the chat. A lookup, not a price.

The public API answers one plain-English question for a league. The result
is shown beside the ticket. It is never a consensus source and never a win
probability. https://developer.statmuse.com/docs/asking-questions
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

from ..utils import fetch_json

BASE = "https://api.statmuse.com"
# Odds API league key -> StatMuse sport. Anything absent is unsupported
# (MMA, MLS, and the rest of soccer). The top European leagues share `fc`.
DOMAINS = {
    "basketball_nba": "nba",
    "americanfootball_nfl": "nfl",
    "icehockey_nhl": "nhl",
    "baseball_mlb": "mlb",
    "basketball_wnba": "wnba",
    "americanfootball_ncaaf": "cfb",
    "soccer_epl": "fc",
    "soccer_spain_la_liga": "fc",
    "soccer_germany_bundesliga": "fc",
    "soccer_italy_serie_a": "fc",
    "soccer_france_ligue_one": "fc",
    "soccer_uefa_champs_league": "fc",
}
MAX_QUESTION = 200
MAX_ROWS = 8
NOTE = "A stat lookup. Not a win probability and it does not change the blend."


def _label(item: Mapping[str, Any]) -> str:
    for key in ("longName", "shortName", "display", "abbreviation"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(item.get("type") or "")


def _rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
    raw = data.get("results") if isinstance(data, Mapping) else None
    rows: List[Dict[str, Any]] = []
    for item in list(raw or [])[:MAX_ROWS]:
        if not isinstance(item, Mapping):
            continue
        context = item.get("context") if isinstance(item.get("context"), list) else []
        stats = item.get("stats") if isinstance(item.get("stats"), Mapping) else {}
        numbers = {}
        for name, cell in stats.items():
            if isinstance(cell, Mapping) and isinstance(cell.get("value"), (int, float)):
                numbers[str(name)] = cell["value"]
        who = [_label(part) for part in context if isinstance(part, Mapping)]
        rows.append({"who": [label for label in who if label], "stats": numbers})
    return rows


class StatMuseClient:
    """One question, one league. The key stays in the request header and never in the result."""

    def __init__(self, api_key: str, fetch: Optional[Callable[..., Dict[str, Any]]] = None,
                 base: str = BASE) -> None:
        self.api_key = (api_key or "").strip()
        self.fetch = fetch or fetch_json
        self.base = base.rstrip("/")

    def ask(self, league: str, question: str) -> Dict[str, Any]:
        domain = DOMAINS.get(league)
        if domain is None:
            return {"supported": False, "league": league, "error": "statmuse_unsupported",
                    "hint": "StatMuse has no feed for this league. Use the markets blend."}
        if not self.api_key:
            return {"supported": True, "league": league, "domain": domain, "error": "statmuse_off",
                    "hint": "StatMuse is not configured."}
        text = " ".join((question or "").split())[:MAX_QUESTION]
        if not text:
            return {"supported": True, "league": league, "domain": domain, "error": "empty_question"}
        url = f"{self.base}/sports/{domain}/v1/ask"
        try:
            payload = self.fetch(url, params={"input": text}, headers={"x-statmuse-api-key": self.api_key},
                                  timeout=10.0)
        except Exception:
            return {"supported": True, "league": league, "domain": domain, "error": "statmuse_unavailable",
                    "hint": "StatMuse did not answer. Do not invent the stat."}
        if not isinstance(payload, dict):
            return {"supported": True, "league": league, "domain": domain, "error": "statmuse_unavailable"}
        answer = payload.get("answer") if isinstance(payload.get("answer"), Mapping) else {}
        body = answer.get("text") if isinstance(answer, Mapping) else ""
        return {
            "supported": True,
            "league": league,
            "domain": domain,
            "type": payload.get("type") if isinstance(payload.get("type"), str) else "unknown",
            "question": text,
            "answer": str(body or "")[:600],
            "rows": _rows(payload),
        }
