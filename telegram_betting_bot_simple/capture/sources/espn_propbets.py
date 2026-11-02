"""Raw ESPN core ``propBets`` (DraftKings' props) for the capture store: archive only (v4 P1, plan sections 3.3 and
3.13). Normalising them into odds rows comes in P6; nothing here reads a price.

    GET https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{league}/events/{id}/competitions/{id}
        /odds/100/propBets?limit=500&page={n}

100 is ESPN's provider id for DraftKings in the core odds API (data/espn_players.py ``DRAFTKINGS``). The answer is one
of ESPN's core paged lists, ``{"count", "pageIndex", "pageSize", "pageCount", "items"}``: one NBA game had 443 props,
and a busy MLB game runs to three pages of 500, so ``propbets_all`` reads page 1 and then each page up to its
``pageCount`` (at most ``MAX_PAGES``). Each item carries ESPN's ``open`` line beside its ``current`` one. ESPN's
stored "close" is an in-game number, so P6 never uses it as a close (plan section 3.13).

Leagues (``LEAGUES``): NBA, WNBA, NFL, NHL and MLB, the leagues whose DraftKings categories the audit confirmed for
basketball, football, baseball and hockey.

When (``HORIZONS``): at "open" (the first sighting: due when the worker first schedules the game), then at the
worker's pre horizons "t24", "t90" and "close" (T-10 min), while CAPTURE_PROPBETS is on. A snapshot whose bytes
haven't changed adds no row (RawStore deduplicates on sha256). The worker stores each page with state "pre" under
the game's key ``espn:<eventId>``, spending its "prices" budget.

Shape (capture/sources/__init__.py): ``propbets(http, league, event_id, *, page=1, limit=500) -> RawFetch`` (one
page) and ``propbets_all(http, league, event_id, *, limit=500, max_pages=MAX_PAGES) -> Iterator[RawFetch]`` (the
pager). Neither raises on an HTTP status (it goes into the RawFetch); connection errors and timeouts raise
``capture.http.Unreachable``, and a guard's ``Refused`` passes through. ``params`` is ``{"limit": 500, "page": n}``
(ints): the event id lives in the game key, never in params.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from ...data.espn_client import espn_path
from . import RawFetch, espn_key, fetch

SOURCE = "espn_propbets"
ENDPOINT = "propbets"
CORE = "https://sports.core.api.espn.com/v2/sports"
DRAFTKINGS = "100"  # ESPN's provider id for DraftKings (data/espn_players.py)
LIMIT = 500  # props a page: ESPN's own maximum for this list
MAX_PAGES = 5  # as data/espn_players.py MAX_PROP_PAGES: a busy MLB game runs to three pages
LEAGUES = ("basketball_nba", "basketball_wnba", "americanfootball_nfl", "icehockey_nhl", "baseball_mlb")
HORIZONS = ("open", "t24", "t90", "close")


def covers(league: Any) -> bool:
    """Whether propBets are archived for this league key (NBA, WNBA, NFL, NHL, MLB)."""
    return league in LEAGUES


def _event(event_id: Any) -> str:
    text = str(event_id if event_id is not None else "").strip()
    if not text.isdigit() or not text.isascii():
        raise ValueError(f"an ESPN event id is a number, not {event_id!r}")
    return text


def url_for(league: str, event_id: Any) -> str:
    """The propBets list's URL for one game (no query). ValueError for a league not in LEAGUES or an id that isn't
    a number (the id is put into the path)."""
    path = espn_path(league) if covers(league) else None
    if path is None:
        raise ValueError(f"propBets aren't archived for {league!r}")
    sport, slug = path
    event = _event(event_id)
    return f"{CORE}/{sport}/leagues/{slug}/events/{event}/competitions/{event}/odds/{DRAFTKINGS}/propBets"


def game_key(event_id: Any) -> str:
    """The raw store's key for the game the props belong to: ``espn:<eventId>``."""
    return espn_key(_event(event_id))


def propbets(http: Any, league: str, event_id: Any, *, page: int = 1, limit: int = LIMIT) -> RawFetch:
    """One page of the game's raw propBets (see the module's docstring)."""
    if int(page) < 1 or int(limit) < 1:
        raise ValueError("page and limit start at 1")
    return fetch(http, url_for(league, event_id), source=SOURCE, endpoint=ENDPOINT,
                 params={"limit": int(limit), "page": int(page)})


def page_count(raw: RawFetch) -> Optional[int]:
    """The answer's ``pageCount`` (at least 1) when it is a 2xx core paged list; None otherwise (an error, a body that
    isn't JSON, or another shape), which ends the pager."""
    if not raw.ok:
        return None
    try:
        data = raw.json()
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return None
    try:
        return max(1, int(data.get("pageCount") or 1))
    except (TypeError, ValueError):
        return None


def propbets_all(http: Any, league: str, event_id: Any, *, limit: int = LIMIT,
                 max_pages: int = MAX_PAGES) -> Iterator[RawFetch]:
    """Every page of the game's raw propBets: page 1, then pages 2 to its ``pageCount`` (at most ``max_pages``). A
    page that isn't a 2xx list is yielded and ends the walk (the caller sees its status)."""
    url_for(league, event_id)  # a bad league or id fails before any request
    first = propbets(http, league, event_id, page=1, limit=limit)
    yield first
    pages = page_count(first)
    if pages is None:
        return
    for page in range(2, min(pages, max(1, int(max_pages))) + 1):
        raw = propbets(http, league, event_id, page=page, limit=limit)
        yield raw
        if page_count(raw) is None:
            return


__all__ = ["SOURCE", "ENDPOINT", "LEAGUES", "HORIZONS", "LIMIT", "MAX_PAGES", "covers", "url_for", "game_key",
           "propbets", "page_count", "propbets_all"]
