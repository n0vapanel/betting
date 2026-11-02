"""nflverse's NFL files on GitHub releases, raw (plan section 3.3): a season's play-by-play and the officials file.

URLs (the v4 research notes, us_sports_events.md, and the event-level report)::

    https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz
    https://github.com/nflverse/nflverse-data/releases/download/officials/officials.csv

A release download answers a redirect to GitHub's asset host (urllib follows it; PoliteHttp keeps every GitHub host
in one 0.2 request/s bucket). The play-by-play body is itself a gzip file: it is stored as it came (the store's
``identity`` codec), never unzipped here. Licence: CC BY 4.0 (nflverse).

- ``pbp_season(http, season)``: params ``{"season": <int>}``, game key ``static:nflverse:pbp:<season>``
  (``pbp_key``).
- ``officials(http)``: params ``{}``, game key ``static:nflverse:officials`` (OFFICIALS_KEY).
Both are league ``americanfootball_nfl`` (``game_key(raw)`` / ``league_of(raw)`` for a fetch). The worker fetches
the current season's and the officials every Tuesday; the backfill takes each backfill season's once.
"""

from __future__ import annotations

from typing import Any

from . import RawFetch, fetch, static_key

SOURCE = "nflverse"
LEAGUE = "americanfootball_nfl"
RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
OFFICIALS_URL = f"{RELEASES}/officials/officials.csv"
FIRST_SEASON = 1999  # nflverse's play-by-play starts in 1999
OFFICIALS_KEY = static_key(SOURCE, "officials")


def season_number(season: Any) -> int:
    if isinstance(season, bool):
        raise ValueError(f"not a season: {season!r}")
    try:
        number = int(str(season).strip()) if not isinstance(season, int) else season
    except ValueError:
        raise ValueError(f"not a season: {season!r}") from None
    if not FIRST_SEASON <= number <= 2100:
        raise ValueError(f"nflverse play-by-play covers {FIRST_SEASON} on, not {season!r}")
    return number


def pbp_url(season: Any) -> str:
    return f"{RELEASES}/pbp/play_by_play_{season_number(season)}.csv.gz"


def pbp_season(http: Any, season: Any) -> RawFetch:
    """One season's play-by-play (``.csv.gz``, 372 columns). Params ``{"season": <int>}``."""
    number = season_number(season)
    return fetch(http, pbp_url(number), source=SOURCE, endpoint="pbp_season", params={"season": number}, query={})


def officials(http: Any) -> RawFetch:
    """The officials file (every game's crew). Params ``{}``."""
    return fetch(http, OFFICIALS_URL, source=SOURCE, endpoint="officials", params={})


def pbp_key(season: Any) -> str:
    """``static:nflverse:pbp:<season>``: one season's play-by-play."""
    return static_key(SOURCE, f"pbp:{season_number(season)}")


def game_key(raw: RawFetch) -> str:
    """The game key a fetch of this module is stored under."""
    if raw.endpoint == "pbp_season":
        return pbp_key(raw.params["season"])
    if raw.endpoint == "officials":
        return OFFICIALS_KEY
    raise ValueError(f"not an nflverse fetch: {raw.source}/{raw.endpoint}")


def league_of(raw: Any = None) -> str:
    """Every nflverse file is the NFL's."""
    return LEAGUE
