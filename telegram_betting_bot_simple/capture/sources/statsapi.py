"""MLB StatsAPI's live feed (``statsapi.mlb.com``): one game's pitch-by-pitch record, raw (plan section 3.3).

``feed_live(http, game_pk)``: ``https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live`` (the URL in the v4
research notes, us_sports_events.md: ``gameData.weather``, ``liveData.plays.allPlays``,
``liveData.boxscore.officials``). Params exactly ``{"game_pk": <int>}``; the worker stores it under the game's
``espn:<eventId>`` key (it maps the ESPN game to its gamePk itself, through data/mlb_stats.py's schedule).

**Forward games only, one game at a time.** StatsAPI's terms are non-bulk: this is never used for a backfill of
older games (tests/test_capture_sources.py checks that the backfill modules don't import it), and it takes exactly
one gamePk per call.
"""

from __future__ import annotations

from typing import Any

from . import RawFetch, fetch

SOURCE = "mlb_statsapi"
LEAGUE = "baseball_mlb"
FEED = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"


def game_pk(value: Any) -> int:
    """One gamePk as an int (a positive whole number, or its digits as text); ValueError for anything else, such as a
    list or a comma-joined batch."""
    if isinstance(value, bool):
        raise ValueError(f"not a gamePk: {value!r}")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    else:
        raise ValueError(f"not a gamePk (one game at a time): {value!r}")
    if number <= 0:
        raise ValueError(f"not a gamePk: {value!r}")
    return number


def feed_url(pk: Any) -> str:
    return FEED.format(game_pk=game_pk(pk))


def feed_live(http: Any, pk: Any) -> RawFetch:
    """One game's live feed (a live or final game the worker is capturing now). Params ``{"game_pk": <int>}``."""
    number = game_pk(pk)
    return fetch(http, FEED.format(game_pk=number), source=SOURCE, endpoint="feed_live",
                 params={"game_pk": number}, query={})
