"""ESPN's MMA fightcenter (``site.web.api.espn.com``): one UFC card in one call, raw (plan section 3.3).

``card(http, event_id)``: ``{COMMON}/mma/ufc/fightcenter/{eventId}``, the URL in the v4 research notes
(combat_tennis_cricket_other.md, probed 31 October 2026 on UFC 333 = 600061267). Params ``{}``. A card is stored
under ``espn:<cardEventId>`` with league ``mma_mixed_martial_arts`` (``card_key``); its bouts are
``espn-bout:<competitionId>`` (core ``status`` and ``competitors/{id}/statistics``, only for a bout fightcenter has
no result for). The payload: ``cards.{main, prelims1, prelims2}.competitions[]``, each bout with
``status.result.{name, displayName, description, target.name}`` (the method), ``officials[]``, ``bets`` and
``competitors[].stats[]``. ``bouts(payload)`` lists them for the worker. The fightcenter at t24 holds the bets and
the athletes (an input, never an outcome).

ESPN's MMA site summary answers 404, so fightcenter is the card's record.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ...data.espn_client import COMMON
from . import RawFetch, espn_key, fetch
from .espn_site import espn_target

SOURCE = "espn_fightcenter"
LEAGUE = "mma_mixed_martial_arts"
SEGMENTS = ("main", "prelims1", "prelims2")  # the card's parts, in the order they are shown


def card_url(event_id: Any, league: Any = LEAGUE) -> str:
    event = str(event_id if event_id is not None else "").strip()
    if not event or "/" in event or any(ch.isspace() for ch in event):
        raise ValueError(f"bad card event id: {event_id!r}")
    sport, path = espn_target(league)
    return f"{COMMON}/{sport}/{path}/fightcenter/{event}"


def card(http: Any, event_id: Any, *, league: Any = LEAGUE) -> RawFetch:
    """The card's fightcenter, raw. Params ``{}``."""
    return fetch(http, card_url(event_id, league), source=SOURCE, endpoint="fightcenter", params={})


def card_key(event_id: Any) -> str:
    """The game key a card is stored under: ``espn:<cardEventId>``."""
    return espn_key(event_id)


def _state(competition: Dict[str, Any]) -> str:
    kind = (competition.get("status") or {}).get("type") or {}
    if not isinstance(kind, dict):
        return ""
    if kind.get("completed") or kind.get("state") == "post":
        return "final"
    return str(kind.get("state") or "")


def bouts(payload: Any) -> List[Dict[str, Any]]:
    """The card's bouts as ``{"id", "segment", "competitors": [ids], "state", "result"}`` ("final" once over;
    ``result`` the method's ``name``, e.g. "kotko", or None when fightcenter has none), in card order. Unreadable
    parts are skipped."""
    found: List[Dict[str, Any]] = []
    cards = payload.get("cards") if isinstance(payload, dict) else None
    if not isinstance(cards, dict):
        return found
    names = list(SEGMENTS) + [name for name in cards if name not in SEGMENTS]
    for segment in names:
        part = cards.get(segment)
        competitions = part.get("competitions") if isinstance(part, dict) else None
        for competition in competitions if isinstance(competitions, list) else []:
            if not isinstance(competition, dict) or competition.get("id") in (None, ""):
                continue
            ids = []
            for competitor in competition.get("competitors") or []:
                if isinstance(competitor, dict):
                    who = competitor.get("id") or (competitor.get("athlete") or {}).get("id")
                    if who not in (None, ""):
                        ids.append(str(who))
            result = ((competition.get("status") or {}).get("result") or {})
            name = result.get("name") if isinstance(result, dict) else None
            found.append({"id": str(competition["id"]), "segment": segment, "competitors": ids,
                          "state": _state(competition), "result": str(name) if name else None})
    return found
