"""Raw fetchers for the capture store (v4 P1, plan section 3.3): the shared ``RawFetch`` contract and game keys.

Every source function has the shape ``fn(http, ...) -> RawFetch`` (or an iterator of them, for pagers), where
``http`` is a ``capture.http.PoliteHttp`` (or anything with its ``get(url, params=None, *, headers=None) ->
HttpResponse(status, body, headers)``). A fetcher never raises on an HTTP status: the status goes into the
``RawFetch`` and the caller decides. It raises ``capture.http.Unreachable`` on connection errors and timeouts.

``RawFetch.params`` is the canonical query stored with the payload (``raw_payloads.params``): only the parts that
tell two payloads of one game apart (page, limit, competitor, season, file, dates), never the event id (that is in
the game key). Values are JSON-native and typed the same way by every fetcher, so a lookup with the same dict finds
the row: ``page``, ``limit``, ``season`` (nflverse) and ``game_pk`` (StatsAPI) are ints; ids (``competitor``),
``dates`` (``YYYYMMDD`` or ``YYYYMMDD-YYYYMMDD``), ``file`` and football-data's ``season`` (``"2425"``) are strings.
``{}`` when none.
``RawFetch.url`` is the full URL that was asked (with its query); none of these sources takes a key. A body sent
with ``Content-Encoding: gzip`` is stored decoded (ESPN gzips some error bodies unasked); a body that is itself a
``.gz`` file (nflverse's play-by-play) has no such header and is kept as it came.

Game keys (section 3.3): ``espn:<eventId>`` for team games (as ``hist_games``) and for a UFC card (its fightcenter),
``espn-bout:<competitionId>`` for an MMA bout, ``espn-match:<competitionId>`` for a tennis match, ``mlb:<gamePk>``
for a StatsAPI game with no ESPN event (an ESPN game's live feed is stored under its ``espn:<eventId>``, params
``{"game_pk": ...}``) and ``static:<source>:<unit>`` for everything that is not one game: a scoreboard
(``scoreboard_key``) and each static file (``static_unit``).
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.parse
import zlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Tuple

from ..http import Unreachable

SOURCES = ("espn_site", "espn_core", "espn_fightcenter", "mlb_statsapi", "nflverse", "greco1899", "mcp",
           "football_data", "espn_propbets", "kalshi")
ENDPOINTS = ("summary", "scoreboard", "core_plays", "core_status", "core_competitor_stats", "core_events",
             "fightcenter", "feed_live", "pbp_season", "officials", "csv", "propbets", "events")


class HttpLike(Protocol):
    """What a fetcher needs from PoliteHttp."""

    def get(self, url: str, params: Optional[Mapping[str, Any]] = None, *,
            headers: Optional[Mapping[str, str]] = None) -> Any: ...


@dataclass(frozen=True)
class RawFetch:
    source: str              # espn_site | espn_core | espn_fightcenter | mlb_statsapi | nflverse | greco1899 | mcp |
                             # football_data | espn_propbets | kalshi
    endpoint: str            # summary | scoreboard | core_plays | core_status | core_competitor_stats | fightcenter |
                             # feed_live | pbp_season | officials | csv | propbets | events | core_events
    params: Mapping[str, Any]   # the canonical query (page, limit, competitor, season, file, dates); {} when none
    url: str                 # no keys: none of these sources take one
    status: int
    body: bytes
    fetched_at: float

    @property
    def ok(self) -> bool:
        """A 2xx answer."""
        return 200 <= int(self.status) < 300

    def json(self) -> Any:
        """The body parsed as JSON (raises ValueError when it is not JSON)."""
        return json.loads(self.body.decode("utf-8-sig"))


# --- game keys -----------------------------------------------------------------------------------------------------


def _part(value: Any, what: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text or any(ch.isspace() for ch in text):
        raise ValueError(f"bad {what} for a game key: {value!r}")
    return text


def espn_key(event_id: Any) -> str:
    """``espn:<eventId>``: a team game (the key hist_games and the live logs use)."""
    return f"espn:{_part(event_id, 'event id')}"


def bout_key(competition_id: Any) -> str:
    """``espn-bout:<competitionId>``: one MMA bout (a card's competition)."""
    return f"espn-bout:{_part(competition_id, 'competition id')}"


def match_key(competition_id: Any) -> str:
    """``espn-match:<competitionId>``: one tennis match."""
    return f"espn-match:{_part(competition_id, 'competition id')}"


def mlb_key(game_pk: Any) -> str:
    """``mlb:<gamePk>``: one MLB StatsAPI game without an ESPN event (an ESPN game's feed goes under its espn key)."""
    return f"mlb:{_part(game_pk, 'gamePk')}"


def static_key(source: str, unit: Any) -> str:
    """``static:<source>:<unit>``: anything that is not one game (the unit may itself contain colons)."""
    return f"static:{_part(source, 'source')}:{_part(unit, 'unit')}"


def scoreboard_key(league: str, dates: Any) -> str:
    """The unit an ESPN scoreboard is stored under: ``static:espn_site:scoreboard:<league>:<dates>`` (one board for
    one day ``YYYYMMDD`` or one ``dates`` range ``YYYYMMDD-YYYYMMDD`` of one league key)."""
    return static_key("espn_site", f"scoreboard:{_part(league, 'league')}:{_part(dates, 'dates')}")


def static_unit(raw: "RawFetch") -> Tuple[str, str]:
    """(game key, league) a static source's fetch is stored under, the same for the worker and the backfill:

    ======================  ===========================================  ==========================================
    nflverse play-by-play   ``static:nflverse:pbp:<season>``             americanfootball_nfl
    nflverse officials      ``static:nflverse:officials``                americanfootball_nfl
    greco1899               ``static:greco1899:<file>``                  mma_mixed_martial_arts
    mcp                     ``static:mcp:<file>``                        tennis_atp (tennis_wta: ``charting-w-*``)
    football_data           ``static:football_data:<code>:<YYZZ>``       the code's league (E0: soccer_epl)
    ======================  ===========================================  ==========================================

    (a football-data country file, every season in one: ``static:football_data:<code>``). Each module's
    ``game_key`` / ``league_of`` give the same. ValueError for a fetch of any other source."""
    from . import football_data_raw, greco, mcp, nflverse  # the modules import this one
    modules = {nflverse.SOURCE: nflverse, greco.SOURCE: greco, mcp.SOURCE: mcp,
               football_data_raw.SOURCE: football_data_raw}
    module = modules.get(raw.source)
    if module is None:
        raise ValueError(f"not a static source: {raw.source}")
    return module.game_key(raw), module.league_of(raw)


KEY_KINDS = ("espn", "espn-bout", "espn-match", "mlb", "static")


def parse_game_key(game_key: str) -> Tuple[str, str]:
    """(kind, rest): ``("espn", "401")``, ``("static", "nflverse:pbp:2024")``. ValueError for an unknown kind."""
    kind, sep, rest = str(game_key or "").partition(":")
    if not sep or kind not in KEY_KINDS or not rest:
        raise ValueError(f"not a capture game key: {game_key!r}")
    return kind, rest


# --- the one GET every fetcher makes ----------------------------------------------------------------------------------


def full_url(url: str, query: Optional[Mapping[str, Any]]) -> str:
    """``url`` with ``query`` appended (in the given order), as recorded in RawFetch.url."""
    if not query:
        return url
    return f"{url}{'&' if '?' in url else '?'}{urllib.parse.urlencode([(k, str(v)) for k, v in query.items()])}"


def _now(http: Any) -> float:
    clock = getattr(http, "clock", None)
    if not callable(clock):
        clock = getattr(http, "_clock", None)
    return float(clock()) if callable(clock) else time.time()


def _header(headers: Any, name: str) -> str:
    """One response header, whatever form the headers come in (a dict, an email Message, (name, value) pairs)."""
    if not headers:
        return ""
    wanted = name.lower()
    items = headers.items() if hasattr(headers, "items") else headers
    try:
        for key, value in items:
            if str(key).lower() == wanted:
                return str(value)
    except (TypeError, ValueError):
        return ""
    return ""


def _decoded(body: bytes, headers: Any) -> bytes:
    """``body`` gunzipped when its headers say ``Content-Encoding: gzip`` and it is gzip; else as it came."""
    if body[:2] == b"\x1f\x8b" and _header(headers, "Content-Encoding").strip().lower() in ("gzip", "x-gzip"):
        try:
            return gzip.decompress(body)
        except (OSError, EOFError, zlib.error):
            return body
    return body


def fetch(http: HttpLike, url: str, *, source: str, endpoint: str, params: Optional[Mapping[str, Any]] = None,
          query: Optional[Mapping[str, Any]] = None, headers: Optional[Mapping[str, str]] = None) -> RawFetch:
    """One GET through ``http`` as a RawFetch. ``params`` is the canonical query kept with the payload; ``query`` is
    what is sent (default: ``params``). Any HTTP status is returned, never raised (an ``http`` that raises
    urllib's HTTPError has its status kept); connection errors and timeouts raise ``Unreachable``. What PoliteHttp
    raises itself comes out unchanged: ``Unreachable``, its ``Backoff`` and a budget guard's ``Refused`` (and any
    other non-OSError exception of whatever ``http`` is used)."""
    canonical = dict(params or {})
    sent = {key: str(value) for key, value in (canonical if query is None else query).items()}
    try:
        response = http.get(url, sent or None, headers=dict(headers) if headers else None)
    except Unreachable:
        raise
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read() or b""
        except Exception:  # noqa: BLE001 - a closed error body: the status is what matters
            body = b""
        return RawFetch(source, endpoint, canonical, full_url(url, sent), int(exc.code),
                        _decoded(bytes(body), getattr(exc, "headers", None)), _now(http))
    except OSError as exc:  # URLError, timeouts, resets (PoliteHttp raises Unreachable itself)
        raise Unreachable(f"{url}: {exc}") from exc
    status = int(getattr(response, "status", 0) or 0)
    body = getattr(response, "body", b"") or b""
    body = _decoded(body if isinstance(body, bytes) else bytes(body), getattr(response, "headers", None))
    return RawFetch(source, endpoint, canonical, full_url(url, sent), status, body, _now(http))
