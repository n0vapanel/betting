"""ESPN's core API (``sports.core.api.espn.com``): play-by-play pages, a competition's status and a competitor's
statistics, raw (plan section 3.3).

URLs (from the v4 audit's request log, 31 October 2026)::

    {CORE}/{sport}/leagues/{path}/events/{eventId}/competitions/{competitionId}/plays?limit=300&page=N
    {CORE}/{sport}/leagues/{path}/events/{eventId}/competitions/{competitionId}/status
    {CORE}/{sport}/leagues/{path}/events/{eventId}/competitions/{competitionId}/competitors/{id}/statistics
    {CORE}/{sport}/leagues/{path}/events?dates=A-B&limit=1000&page=N      (event ids in a date range)

A team game's competition id is its event id (the default). An MMA card is the event and each bout a competition
(game key ``espn-bout:<competitionId>``); a tennis tournament is the event and each match a competition.

- ``plays(http, league, event_id, *, page, limit=300)``: one page; params ``{"page": n, "limit": 300}``. A page
  answers ``{count, pageIndex, pageSize, pageCount, items}`` (a soccer match: 1,481 items is 5 pages).
- ``plays_all(...)``: the pager. It stops after the last page (``pageCount``), after ``max_pages``, at a page that
  is not a 2xx answer (yielded, so the caller sees it), or at one it cannot read.
- ``status(...)``: params ``{}``. ``competitor_stats(...)``: params ``{"competitor": "<id>"}``.
- ``events(http, league, *, dates, page=1, limit=1000)`` / ``events_all(...)``: endpoint ``core_events``, params
  ``{"dates", "page", "limit"}``. The items are ``$ref`` links only (no date, no status): ``event_ids`` reads them.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator, List, Optional

from . import RawFetch, fetch
from .espn_site import dates_text, espn_target

SOURCE = "espn_core"
CORE = "https://sports.core.api.espn.com/v2/sports"
PLAYS_LIMIT = 300
MAX_PAGES = 10      # a long soccer match is 5-7 pages of 300
EVENTS_LIMIT = 1000
EVENTS_MAX_PAGES = 20
_EVENT_REF = re.compile(r"/events/([^/?#]+)")


def _id(value: Any, what: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text or "/" in text or any(ch.isspace() for ch in text):
        raise ValueError(f"bad {what}: {value!r}")
    return text


def league_url(league: Any) -> str:
    sport, path = espn_target(league)
    return f"{CORE}/{sport}/leagues/{path}"


def competition_url(league: Any, event_id: Any, competition_id: Any = None) -> str:
    event = _id(event_id, "event id")
    competition = _id(competition_id, "competition id") if competition_id is not None else event
    return f"{league_url(league)}/events/{event}/competitions/{competition}"


def plays(http: Any, league: Any, event_id: Any, *, page: int, limit: int = PLAYS_LIMIT,
          competition_id: Any = None) -> RawFetch:
    """One page of the competition's plays."""
    page, limit = int(page), int(limit)
    if page < 1 or limit < 1:
        raise ValueError("page and limit start at 1")
    return fetch(http, f"{competition_url(league, event_id, competition_id)}/plays", source=SOURCE,
                 endpoint="core_plays", params={"page": page, "limit": limit}, query={"limit": limit, "page": page})


def page_count(raw: RawFetch) -> Optional[int]:
    """A page's ``pageCount`` (None when the body is not a readable page)."""
    try:
        data = raw.json()
        return int(data["pageCount"]) if isinstance(data, dict) else None
    except (ValueError, KeyError, TypeError):
        return None


def _items(raw: RawFetch) -> List[Any]:
    try:
        data = raw.json()
    except ValueError:
        return []
    items = data.get("items") if isinstance(data, dict) else None
    return items if isinstance(items, list) else []


def _pages(get_page: Any, first_page: int, max_pages: int) -> Iterator[RawFetch]:
    if int(max_pages) < 1:
        raise ValueError("max_pages starts at 1")
    page = int(first_page)
    for _ in range(int(max_pages)):
        raw = get_page(page)
        yield raw
        if not raw.ok:
            return
        count = page_count(raw)
        if count is None or page >= count or not _items(raw):
            return
        page += 1


def plays_all(http: Any, league: Any, event_id: Any, *, limit: int = PLAYS_LIMIT, max_pages: int = MAX_PAGES,
              competition_id: Any = None, first_page: int = 1) -> Iterator[RawFetch]:
    """Every page of plays from ``first_page`` (a resumed capture can start later), one RawFetch each."""
    return _pages(lambda page: plays(http, league, event_id, page=page, limit=limit,
                                     competition_id=competition_id), first_page, max_pages)


def status(http: Any, league: Any, event_id: Any, competition_id: Any = None) -> RawFetch:
    """The competition's status (an MMA bout's ``result``: method, round, clock). Params ``{}``."""
    return fetch(http, f"{competition_url(league, event_id, competition_id)}/status", source=SOURCE,
                 endpoint="core_status", params={})


def competitor_stats(http: Any, league: Any, event_id: Any, competition_id: Any, competitor_id: Any) -> RawFetch:
    """One competitor's statistics in the competition (an MMA fighter's strikes and takedowns)."""
    competitor = _id(competitor_id, "competitor id")
    url = f"{competition_url(league, event_id, competition_id)}/competitors/{competitor}/statistics"
    return fetch(http, url, source=SOURCE, endpoint="core_competitor_stats", params={"competitor": competitor},
                 query={})  # the competitor is in the path, not the query


def events(http: Any, league: Any, *, dates: Any, page: int = 1, limit: int = EVENTS_LIMIT) -> RawFetch:
    """One page of the league's event list for ``dates`` (a day or a ``YYYYMMDD-YYYYMMDD`` range: the core API takes
    ranges)."""
    when, page, limit = dates_text(dates), int(page), int(limit)
    if page < 1 or limit < 1:
        raise ValueError("page and limit start at 1")
    return fetch(http, f"{league_url(league)}/events", source=SOURCE, endpoint="core_events",
                 params={"dates": when, "page": page, "limit": limit},
                 query={"dates": when, "limit": limit, "page": page})


def events_all(http: Any, league: Any, *, dates: Any, limit: int = EVENTS_LIMIT,
               max_pages: int = EVENTS_MAX_PAGES) -> Iterator[RawFetch]:
    """Every page of the event list (the pager's rules)."""
    return _pages(lambda page: events(http, league, dates=dates, page=page, limit=limit), 1, max_pages)


def event_ids(pages: Iterable[RawFetch]) -> List[str]:
    """The event ids in event-list pages (from each item's ``$ref``), in order, once each."""
    found: List[str] = []
    for raw in pages:
        for item in _items(raw):
            match = _EVENT_REF.search(str((item or {}).get("$ref") or "")) if isinstance(item, dict) else None
            if match and match.group(1) not in found:
                found.append(match.group(1))
    return found
