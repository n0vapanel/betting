"""ESPN's site API (``site.api.espn.com``): a game's summary and a league's scoreboard, untrimmed (plan section 3.3).

- ``summary(http, league, event_id)``: ``{SITE}/{sport}/{path}/summary?event={id}``, kept whole: plays, boxscore,
  ``gameInfo.officials``, winprobability, drives, scoringPlays, keyEvents, commentary, rosters, onIce. Params ``{}``
  (the event is the game key, ``espn:<eventId>``). ESPN has no site summary for MMA (404) or tennis (400): those come
  from fightcenter and the scoreboard.
- ``scoreboard(http, league, day=None, *, dates=None, params=None)``: ``{SITE}/{sport}/{path}/scoreboard`` with the
  query the lean client sends (``dates``, ``limit`` SCOREBOARD_LIMIT and the league's SCOREBOARD_PARAMS, e.g. college
  basketball's ``groups``), kept raw: nothing is dropped (LEAN_DROP is the lean cache's), so MMA keeps each bout's
  ``details`` and tennis its ``notes``. Params: ``{"dates": ...}`` plus any ``params`` given; ``{}`` for the
  default board (today's; tennis shows its tournaments' current days).
- **Date ranges.** ``dates`` takes a day (``YYYYMMDD``) or a range (``YYYYMMDD-YYYYMMDD``), but this API answers a
  range with HTTP 400 ``{"code":400,"message":"Failed to get events endpoint."}`` (probed on 1 November 2026 for the
  NBA, the EPL and the NFL; the repo's slate already says ESPN refuses ranges). The fetch returns that status like
  any other. ``scoreboard_days`` reads a window as one board per day, the newest first; a range of event ids comes
  from the core API (``espn_core.events``).

League arguments are the app's league keys (``data/espn_client.py`` LEAGUES and SCOREBOARD_ONLY) or an ESPN
``(sport, path)`` pair.
"""

from __future__ import annotations

import datetime
import re
from typing import Any, Iterator, Mapping, Optional, Tuple

from ...data.espn_client import SCOREBOARD_LIMIT, SCOREBOARD_PARAMS, SITE, espn_path
from . import RawFetch, fetch

SOURCE = "espn_site"
_DAY = re.compile(r"\d{8}")
_DATES = re.compile(r"\d{8}(?:-\d{8})?")


def espn_target(league: Any) -> Tuple[str, str]:
    """ESPN's (sport, league path) for a league key, or a (sport, path) pair as it is; KeyError when ESPN has none."""
    if isinstance(league, (tuple, list)) and len(league) == 2:
        return str(league[0]), str(league[1])
    found = espn_path(str(league))
    if found is None:
        raise KeyError(league)
    return found


def day_text(day: Any) -> str:
    """``YYYYMMDD`` for a date (a datetime counts as its date) or for ``YYYYMMDD`` / ``YYYY-MM-DD`` text."""
    if isinstance(day, datetime.datetime):
        day = day.date()
    if isinstance(day, datetime.date):
        return day.strftime("%Y%m%d")
    text = str(day or "").strip().replace("-", "")
    if not _DAY.fullmatch(text):
        raise ValueError(f"not a day: {day!r}")
    datetime.datetime.strptime(text, "%Y%m%d")  # a real date (ValueError otherwise)
    return text


def dates_text(dates: Any) -> str:
    """A ``dates`` value checked: ``YYYYMMDD`` or ``YYYYMMDD-YYYYMMDD`` (the first day not after the last)."""
    text = str(dates or "").strip()
    if not _DATES.fullmatch(text):
        raise ValueError(f"not a dates value (YYYYMMDD or YYYYMMDD-YYYYMMDD): {dates!r}")
    days = [datetime.datetime.strptime(part, "%Y%m%d").date() for part in text.split("-")]
    if len(days) == 2 and days[0] > days[1]:
        raise ValueError(f"a dates range runs forward: {dates!r}")
    return text


def summary_url(league: Any) -> str:
    sport, path = espn_target(league)
    return f"{SITE}/{sport}/{path}/summary"


def scoreboard_url(league: Any) -> str:
    sport, path = espn_target(league)
    return f"{SITE}/{sport}/{path}/scoreboard"


def summary(http: Any, league: Any, event_id: Any) -> RawFetch:
    """The game's whole summary (``?event=<id>``, nothing else sent). Params ``{}``."""
    event = str(event_id if event_id is not None else "").strip()
    if not event:
        raise ValueError("an event id is needed")
    return fetch(http, summary_url(league), source=SOURCE, endpoint="summary", params={}, query={"event": event})


def scoreboard(http: Any, league: Any, day: Any = None, *, dates: Any = None,
               params: Optional[Mapping[str, Any]] = None) -> RawFetch:
    """The league's scoreboard, raw: for ``day`` (a date or ``YYYYMMDD``), for ``dates`` (a day or a range; a range
    is refused by ESPN, see the module's docstring), or the default board when neither is given. ``params`` adds to
    (or overrides) the query and is kept in the canonical params; the lean client's own ``limit`` and the league's
    SCOREBOARD_PARAMS are sent but not kept (they never change a board's identity)."""
    if day is not None and dates is not None:
        raise ValueError("give day or dates, not both")
    when = day_text(day) if day is not None else (dates_text(dates) if dates is not None else None)
    extra = {str(key): value for key, value in (params or {}).items()}
    canonical: dict = {"dates": when} if when else {}
    canonical.update(extra)
    query: dict = {"dates": when} if when else {}
    query["limit"] = SCOREBOARD_LIMIT
    if isinstance(league, str):
        query.update(SCOREBOARD_PARAMS.get(league, {}))
    query.update(extra)
    return fetch(http, scoreboard_url(league), source=SOURCE, endpoint="scoreboard", params=canonical, query=query)


def scoreboard_days(http: Any, league: Any, first: Any, last: Any, *, newest_first: bool = True,
                    params: Optional[Mapping[str, Any]] = None) -> Iterator[RawFetch]:
    """One board per day from ``first`` to ``last`` (both included), the newest first unless ``newest_first`` is
    False: a 7-day window is 7 requests (ESPN refuses a range). Stops where the caller stops; whatever the http
    raises (Unreachable, Backoff, a guard's Refused) comes out unchanged."""
    start = datetime.datetime.strptime(day_text(first), "%Y%m%d").date()
    end = datetime.datetime.strptime(day_text(last), "%Y%m%d").date()
    days = [start + datetime.timedelta(days=n) for n in range((end - start).days + 1)]
    for day in (reversed(days) if newest_first else days):
        yield scoreboard(http, league, day, params=params)
