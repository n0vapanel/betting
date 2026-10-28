"""One all-sports slate for Today: built in the background, served as ready-made bytes.

Today used to load each league's odds board when someone opened it, at most eight leagues at a time. The
slate instead holds every game the app can see for today and the next six days, in one small JSON
document that the API serves as it is (GET /api/slate, with an ETag, so an unchanged slate is a 304):

- Leagues are the union of The Odds API's in-season leagues (its free /v4/sports listing, which the odds
  service caches for hours) and the leagues ESPN has scoreboards for (espn_client.LEAGUES and
  SCOREBOARD_ONLY). Every in-season league is listed with its game count, even one with no games yet.
- ESPN's scoreboards are fetched one day at a time (ESPN refuses date ranges), eight at once, each on its
  own schedule: boards with a game live (or about to start) every LIVE seconds, today's other boards
  every TODAY seconds while they have games still to finish and every IDLE seconds when they don't, and
  the days ahead every FUTURE seconds. A board that fails keeps its last good copy (the last slate's games for
  that league and day, which is also what a restart starts from) and is retried ever further apart, from its
  own pace (from a minute while there is nothing to show for it); when most boards of a round fail (an ESPN
  outage or rate limit), only boards with a game on are fetched for the next 5 to 15 minutes. The boards of a
  round are stamped with its start, so they come due together: one wake-up and one rebuild per round. The Odds API's listing is re-read in the background at
  most every 15 minutes, so a slow listing never holds up the scores.
- Odds come only from each league's saved snapshot (Store.odds_snapshot): building the slate never calls
  The Odds API. An odds event is the ESPN game in the same league starting within 3 hours whose teams'
  names match (espn_client.team_match_score); it then carries the odds feed's orientation and id, so a
  price added to a slip names the same side the server prices. Leagues ESPN doesn't cover take their games
  from the snapshot alone. The numbers are code's: the best price per side with its bookmaker, the
  market's margin-free chances, and the NBA model's own chance where it has already made one (its logged
  pre-game prediction; nothing new is computed here).
- A league whose odds are older than the odds service's 2 hours can ask for a new board (request_odds,
  POST /api/slate/odds): in the background, at most once per league per 2 hours (a lease in the shared
  store keeps that across restarts), never below the odds service's reserve (counting the boards already on
  their way), at most ODDS_PER_DAY boards a day, and never for a league ESPN shows no games for. Each board
  costs 3 of the month's credits per odds region.

The slate also carries the month's Odds API quota as last reported ({"remaining", "used"}, read from the
shared store, never fetched), for the header's "N odds left".

The markets brain ("Markets say", engine/consensus.py) runs here too. Each game's opinions are gathered as
consensus Sources: the sharp books' and the US books' margin-free moneylines from the saved odds board
(OddsService.consensus_inputs' shape: no credits), Kalshi's and Polymarket's mid prices, ESPN's predictor,
power ratings (BPI, FPI, Torvik, Elo) and the NBA team model's own chance. They are blended into one chance per
outcome, which becomes the game's ``chance`` (src "blend"), with a compact ``mkt`` block:

    "mkt": {"p": {"home", "away", "draw"?}, "fav": "home"|"away"|"draw", "conf": "normal"|"low", "dis": bool,
            "spr": the sources' widest gap in whole points, "src": [[short name, chance of "fav" (2 decimals),
            status], ...], "as_of": to the minute}

The full detail (the Blend, every Source, the venues' links) is kept in memory for the game route
(``markets_for``). Prediction markets and ratings are fetched only by the slate's own background markets worker
(one thread, every MARKETS_EVERY seconds, each source on its own TTL: Polymarket and the ratings first, then ESPN's
predictor for the week's games, then Kalshi, which paces itself and rests when rate-limited); building the slate
and every request read only cached copies. A game's sources and blend are logged to the store's consensus_log on
every change until it starts (then frozen; grading scores them after the final), and once a game has started its
"Markets say" stays the last one before the start. Nothing from the start on enters a game's "Markets say": the
clock is read again after the scoreboards (which can take seconds) and before logging, a source whose copy is from
the start or later is left out (market_detail), and such a price seen before the clock says so marks the game as
started. A start ESPN flags as not valid (a tennis match whose order of play isn't out: midnight Eastern) is only
a day: the venues are matched on the date, and the game starts when ESPN says it has.

Venue & conditions (v3 Stage 4, data/context.py): a context worker (one daemon thread, at most every CONTEXT_EVERY
seconds) hands the games still to play to the ContextService's ``warm``, which builds each game's context block in the
background (venue, weather, rest and travel, the attribution and any adjustment) as a computed entry in the shared
cache; it reads past scoreboards through ``scoreboards`` (the boards held here, else past ones fetched once through
ESPN's cache and kept: they don't change). Each row's ``badges`` then gain the block's codes (wind:26, alt:1580,
tz-away:3, adj, ...) after the slate's own (live, neutral, b2b-home/b2b-away); building the slate never waits for the
worker and never fetches for it.

The player scan (v3, data/player_board.py): at most every PLAYERS_EVERY seconds a round hands the games still to start
to the PlayerBoard's ``prewarm`` (the ``player_board`` given here, else the service's or its Lab's), which starts the
boards of leagues with a game in the next 12 hours in the background (the NBA first, one cold build at a time) and
their due props and injuries refreshes. It returns at once; nothing is fetched on the slate's thread.

Size: each game leaves out ``espn_id`` or ``odds_id`` when its ``id`` already holds it and sends ESPN logos as their
path after https://a.espncdn.com/ (the app puts the host back). Each team keeps its full ``name``: Today's search
finds a game by city or full name ("Brooklyn", "Golden State") and a slip names the team as the game page does,
which the short name and abbreviation can't give back. About 3.2 KB gzipped per 60 games, names, prices and
"Markets say" included.

The slate is serialised once per change, kept in memory as raw bytes, gzip bytes and an ETag (a hash of
the bytes), and saved to the store, so a restarted dashboard serves the last slate before its first
refresh. Nothing in it changes unless the games do, so an unchanged slate keeps its ETag.

The learning loop (v3, learn/): every blend is made with the live parameter version of its sport
(learn/registry.py ``current``; version 0, today's constants, until a change passes the gate), and the logged row
keeps that ``params_version``. Each game also gets one snapshot row a day before its start (kind "t24", logged on
the first round 12 to 24 hours before it, whether or not anything changed), so the brain can be scored by horizon.
"""

from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from ..engine import consensus
from ..learn import registry
from .espn_client import ESPN_DAY_TZ, LEAGUES as ESPN_LEAGUES, SCOREBOARD_ONLY, espn_path, team_match_score
from .leagues import POPULAR_LEAGUES, SPORTS
from .odds_service import MARKETS, book_roles, moneylines, summarize_markets
from .store import T24_WINDOW

log = logging.getLogger(__name__)

SLATE_NAME = "today"  # the store's name for it
DAYS_AHEAD = 6  # today and the next six days
WORKERS = 8  # ESPN scoreboards fetched at once
MATCH_WINDOW = 3 * 3600  # an odds event and an ESPN game further apart than this are different games
MATCH = 0.5  # both teams' names at least this close (team_match_score) ...
STRONG_MATCH = 0.8  # ... or one team's this close: a team plays one game in a 3-hour window
LIVE_SOON = 10 * 60  # a game starting this soon refreshes at the live pace
STUCK = 6 * 3600  # ... but not a "scheduled" game this long past its start (ESPN never updated it)
FETCH_COST = len(MARKETS)  # Odds API credits one league's board costs per region (moneyline, spreads, totals)
ODDS_TTL = 2 * 3600  # odds older than this are stale (the odds service's own odds_ttl when it has one)
ODDS_PER_DAY = 12  # boards request_odds may load in 24 hours, all leagues together
BACKOFF_CAP = 30 * 60  # a failing board is retried at most this far apart (or at its own pace, if slower) ...
LIVE_BACKOFF_CAP = 5 * 60  # ... and one with a game on, this far
BREAKER_MIN = 8  # when at least this many boards of a round fail, and more than half of them, ...
BREAKER_PAUSE = (5 * 60, 10 * 60, 15 * 60)  # ... only live boards are fetched for this long (longer each time in a row)
SEASON_EVERY = 15 * 60  # The Odds API's in-season listing is re-read this often (its service caches it for hours) ...
SEASON_RETRY = 10 * 60  # ... and this long after a failed read
AI_LEAGUES = frozenset({"basketball_nba"})  # as dashboard/service.py: leagues with the NBA model
B2B_SPORTS = ("basketball", "hockey")  # where playing the day before is a badge
TENNIS_PREFIXES = {"tennis_atp_": "tennis_atp", "tennis_wta_": "tennis_wta"}  # odds tournaments -> ESPN tours
SINGLES = ("mens-singles", "womens-singles")
# A game without ESPN data is shown as live this long after its start, then dropped (its result is unknown)
ODDS_ONLY_LIVE = (("cricket_test_match", 5 * 24 * 3600), ("cricket_", 10 * 3600), ("", 4 * 3600))
POSTPONED = ("POSTPONED", "CANCELED", "CANCELLED", "SUSPENDED", "FORFEIT", "ABANDONED", "DELAYED_INDEFINITELY")
GROUP_OF_SPORT = {"basketball": "Basketball", "hockey": "Ice Hockey", "soccer": "Soccer", "tennis": "Tennis",
                  "baseball": "Baseball", "football": "American Football", "mma": "Mixed Martial Arts"}
LABELS = {group: label for label, group in SPORTS}
LABELS.update({"Cricket": "🏏 Cricket", "Rugby League": "🏉 Rugby League", "Rugby Union": "🏉 Rugby Union",
               "Aussie Rules": "🏉 Aussie Rules", "Boxing": "🥊 Boxing", "Handball": "🤾 Handball",
               "Lacrosse": "🥍 Lacrosse", "Golf": "⛳ Golf"})
GROUP_ORDER = [group for _, group in SPORTS]
# Titles for leagues ESPN shows when The Odds API's listing doesn't name them (not in season there, or ESPN's own)
TITLES = {
    "tennis_atp": "ATP", "tennis_wta": "WTA", "mma_mixed_martial_arts": "MMA", "basketball_nba": "NBA",
    "basketball_wnba": "WNBA", "basketball_ncaab": "NCAAB", "basketball_wncaab": "NCAAW", "basketball_nbl": "NBL",
    "americanfootball_nfl": "NFL", "americanfootball_ncaaf": "NCAAF", "americanfootball_cfl": "CFL",
    "icehockey_nhl": "NHL", "baseball_mlb": "MLB", "soccer_epl": "EPL", "soccer_usa_mls": "MLS",
    "soccer_uefa_champs_league": "UEFA Champions League", "soccer_uefa_europa_league": "UEFA Europa League",
}
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏ -‮⁦-⁩]")

# --- the markets brain ("Markets say") ---
MARKETS_EVERY = 60.0  # the markets worker's round: each source refreshes on its own TTL within it
BOOKS_STALE = 12 * 3600  # a saved odds board older than this no longer counts in the blend ("stale")
MARKET_STALE = 3600  # a prediction market's copy older than this is stale (it refreshes every 1-5 minutes)
RATING_STALE = 3 * 24 * 3600  # ratings refresh every 12 hours
PREDICTOR_STALE = 12 * 3600  # ESPN's predictor refreshes every hour
PREDICTOR_TTL = 3600  # as data/markets/ratings.py: ESPN's predictor is refetched when asked for after an hour
PREDICTOR_AHEAD = 24 * 3600  # ESPN's predictor is asked for hourly for games starting within this, ...
PREDICTOR_FAR_EVERY = 6 * 3600  # ... every 6 hours for the rest of the slate's week ...
PREDICTOR_PER_ROUND = 25  # ... at most this many a round (one ESPN game summary each), the soonest due first
ESPN_LOGO = "https://a.espncdn.com/"  # logos are sent as their path after this
# A source's short name in the slate's "mkt" block and the bot's line
SHORT_NAMES = {"Sharp": "Sharp", "US books": "US", "Kalshi": "Kalshi", "Polymarket": "Poly", "ESPN": "ESPN",
               "Model": "Model"}  # ratings keep their own short names (BPI, FPI, Torvik, Elo)
OUTCOMES = ("home", "away", "draw")
PRICE_KINDS = ("sharp", "us_books", "prediction_market")  # sources whose as_of is when this app read the price
KALSHI_AHEAD = 48 * 3600  # Kalshi is refreshed only for leagues with a game on or starting within two days: its
# anonymous limit answers "too many requests" to bursts of large calls (data/markets/kalshi.py)
# The team model's own chance as the brain's "Model" source (the same words on the game route)
MODEL_DETAIL = "NBA team model alone: points for and against, home court, rest, injuries"
# --- context (v3 Stage 4: data/context.py) ---
CONTEXT_EVERY = 60.0  # the context worker's round: each game's venue & conditions block, in the background
CONTEXT_SOON = 5.0  # a reader asked for a block not built yet: a round now, at most this often (request_context)
PLAYERS_EVERY = 5 * 60.0  # the player boards' game-day prewarm (data/player_board.py): at most this often
PAST_KEPT = 600  # past scoreboards (league, day) the context worker keeps: they don't change


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(text: Any) -> Optional[float]:
    """Epoch seconds from an ISO time ("2026-10-26T23:00Z", "...:00Z", "+00:00"), or None."""
    if not isinstance(text, str) or not text:
        return None
    value = text.strip().replace("Z", "+00:00")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}\+00:00", value):
        value = value.replace("+00:00", ":00+00:00")
    try:
        moment = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    return moment.timestamp()


def _text(value: Any, limit: int = 80) -> Optional[str]:
    """Outside text, tidied: no control or direction characters, one space between words, at most ``limit``
    characters. The app still escapes it wherever it is shown."""
    if value is None or isinstance(value, (dict, list)):
        return None
    text = " ".join(CONTROL.sub(" ", str(value)).split())
    return text[:limit] or None


def _logo(url: Any) -> Optional[str]:
    """An ESPN image address, only over HTTPS from ESPN's image hosts; anything else is dropped."""
    if not isinstance(url, str) or len(url) > 300:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not (host == "espncdn.com" or host.endswith(".espncdn.com")):
        return None
    if any(ch in url for ch in "\"'<> \\"):
        return None
    return url


def _lean(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """The mapping without its None values (the slate omits nulls)."""
    return {key: value for key, value in mapping.items() if value is not None}


def _int(value: Any) -> Optional[int]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() and abs(number) < 10_000 else None


def _odds_only_live(league: str) -> float:
    return next(seconds for prefix, seconds in ODDS_ONLY_LIVE if league.startswith(prefix))


def _covers(source: Any, league: str) -> bool:
    covers = getattr(source, "covers", None)
    try:
        return bool(covers(league)) if callable(covers) else True
    except Exception:
        return False


def _answer(queued: bool, code: str, reason: str) -> Dict[str, Any]:
    return {"queued": queued, "reason": reason, "code": code}


def _tour(league: str) -> Optional[str]:
    """The ESPN tour (tennis_atp, tennis_wta) whose board shows an odds feed tennis tournament's matches."""
    return next((tour for prefix, tour in TENNIS_PREFIXES.items() if league.startswith(prefix)), None)


def _sport(league: str) -> str:
    """ESPN's sport for a league key ("basketball", "soccer", ...), "" when ESPN doesn't show it."""
    return (espn_path(league) or espn_path(_tour(league) or "") or ("",))[0]


# --- the markets brain's sources ---------------------------------------------------------------------

def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _age(as_of: Any, now: float) -> Optional[float]:
    ts = as_of if isinstance(as_of, (int, float)) and not isinstance(as_of, bool) else _parse_time(as_of)
    return None if ts is None else now - float(ts)


def _https(url: Any) -> Optional[str]:
    """A venue's link, only when it is a plain https:// address (the app still escapes it)."""
    if not isinstance(url, str) or len(url) > 300 or not url.startswith("https://"):
        return None
    if any(ch in url for ch in "\"'<> \\") or CONTROL.search(url):
        return None
    return url


def _checked(source: Optional[Mapping[str, Any]], now: float, stale_after: Optional[float] = None
             ) -> Optional[Dict[str, Any]]:
    """A Source from the brain (engine/consensus.py) made ready for the slate: None when it has no usable chance,
    its link kept only as a plain https:// address, its detail tidied, and "stale" when its copy is older than
    ``stale_after`` (an "ok" one only: a thin market stays "thin")."""
    if not source or source.get("status") == "missing" or not source.get("p"):
        return None
    out = dict(source)
    out["url"] = _https(out.get("url"))
    out["detail"] = _text(out.get("detail"), 160)
    if not isinstance(out.get("as_of"), str):
        out["as_of"] = None
    if stale_after is not None and out.get("status") == "ok":
        age = _age(out.get("as_of"), now)
        if age is not None and age > stale_after:
            out["status"] = "stale"
    return out


def make_source(name: str, kind: str, p: Optional[Mapping[str, Any]], as_of: Any = None, status: str = "ok",
                url: Optional[str] = None, detail: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """engine/consensus.py's make_source (its chances checked and scaled to sum to 1), ready for the slate; None
    when ``p`` doesn't give at least home and away."""
    if not isinstance(p, Mapping) or _num(p.get("home")) is None or _num(p.get("away")) is None:
        return None
    try:
        return _checked(consensus.make_source(name, kind, p, as_of, status, url, detail), 0.0)
    except ValueError:  # an unknown kind or status: a programming error, logged rather than raised into a build
        log.warning("Slate: unusable %s source (%s, %s)", name, kind, status)
        return None


def book_sources(books: Mapping[str, Mapping[str, Any]], as_of: Any, now: float,
                 sharp: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """The brain's "Sharp" (Pinnacle and the exchanges, no-vig) and "US books" (median no-vig) sources of one game's
    moneylines at every book ({book key: {outcome: decimal}}, OddsService.consensus_inputs). A board older than
    BOOKS_STALE is "stale": shown but not blended."""
    sharp_keys = tuple(sharp if sharp is not None else book_roles().sharp)
    try:
        sources = consensus.book_sources(dict(books or {}), as_of, sharp=sharp_keys)
    except Exception as exc:  # a malformed saved board: no book sources rather than no slate
        log.warning("Slate: book consensus failed (%s)", type(exc).__name__)
        return []
    return [checked for source in sources if (checked := _checked(source, now, BOOKS_STALE))]


def market_source(event: Mapping[str, Any], now: float) -> Optional[Dict[str, Any]]:
    """A Kalshi or Polymarket MarketEvent's game-winner prices (bid/ask mids) as the brain's source
    (consensus.prediction_market_source): "thin" when any outcome's market is thin, "stale" when its copy is over
    an hour old. The event must already face the game's way: match_events and orient turn it round, so it is
    never flipped here."""
    try:
        source = consensus.prediction_market_source(event, flip=False)
    except Exception as exc:
        log.warning("Slate: unusable prediction market (%s)", type(exc).__name__)
        return None
    return _checked(source, now, MARKET_STALE)


def rating_source(rating: Optional[Mapping[str, Any]], now: float, name: Optional[str] = None,
                  stale_after: float = RATING_STALE) -> Optional[Dict[str, Any]]:
    """Ratings.game_prob's (or espn_predictor's) chances as a "model_rating" source."""
    if not rating:
        return None
    source = make_source(name or str(rating.get("source") or "Ratings"), "model_rating", rating, rating.get("as_of"),
                         detail=rating.get("detail"))
    return _checked(source, now, stale_after)


def model_source(chance: Optional[float], as_of: Any, now: float) -> Optional[Dict[str, Any]]:
    """The NBA team model's own home-win chance (not the prediction, which already leans on the market) as the
    "Model" source."""
    if chance is None:
        return None
    return make_source("Model", "team_model", {"home": chance, "away": 1 - chance}, as_of, detail=MODEL_DETAIL)


def tennis_surface(league: str, event_name: Any = None) -> Optional[str]:
    """A tennis match's surface for its Elo ("clay", "grass", "hard"; data/markets/ratings.py), None elsewhere."""
    if not str(league).startswith("tennis_"):
        return None
    from .markets.ratings import surface_of  # the markets package loads on first use
    return surface_of(league, event_name)


def _flip(p: Mapping[str, Any]) -> Dict[str, Any]:
    return {{"home": "away", "away": "home"}.get(k, k): v for k, v in p.items()}


def after_start(source: Mapping[str, Any], start: Optional[float]) -> bool:
    """Whether a source's copy (its ``as_of``) is from the game's start or later."""
    if start is None:
        return False
    ts = _parse_time(source.get("as_of"))
    return ts is not None and ts >= start


def brain_params(league: str, store: Any = None) -> "registry.Params":
    """The learning loop's live values for a league's sport (learn/registry.py), read from ``store`` when it is a
    real Store (else the registry's bound one): today's constants (version 0) until a change passes the gate, in
    safe mode, or when the store can't be read."""
    return registry.current(league, store if hasattr(store, "_connect") and getattr(store, "path", None) else None)


def market_detail(league: str, sources: Sequence[Optional[Mapping[str, Any]]], start: Optional[float], now: float,
                  neutral: bool = False, params: Optional["registry.Params"] = None) -> Optional[Dict[str, Any]]:
    """{"blend": Blend | None, "sources": [Source...], "links": [{"name", "url"}]} for one game, or None without
    any source. The blend (engine/consensus.py) uses only the "ok" sources. Only opinions from before ``start``
    count: a copy from the start or later (an odds board loaded in play, a venue refreshed after the start) is left
    out, so a started game's "Markets say" is never made from in-play prices. ``params``: the sport's live values
    (brain_params); at the defaults the call is exactly the untuned one."""
    usable = [dict(source) for source in sources if source and not after_start(source, start)]
    if not usable:
        return None
    try:
        tuned = params.blend_kwargs() if params is not None else {}
        blend = consensus.blend(usable, league, now=now, start=start, neutral=neutral, **tuned)
    except Exception as exc:
        log.warning("Slate: blending %s failed (%s)", league, type(exc).__name__)
        blend = None
    links = [{"name": source["name"], "url": source["url"]} for source in usable if source.get("url")]
    return {"blend": dict(blend) if blend else None, "sources": usable, "links": links}


def whole_percent(p: Any) -> float:
    """A chance to the whole percent, as a fraction (0.815 -> 0.82), the way the app's pct() rounds it."""
    return math.floor(float(p) * 100 + 0.5) / 100


def compact(detail: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The slate's "mkt" block of a market detail (None without a blend)."""
    blend = (detail or {}).get("blend")
    if not blend or not isinstance(blend.get("p"), Mapping):
        return None
    p = {o: round(float(v), 3) for o, v in blend["p"].items() if o in OUTCOMES}
    fav = max(p, key=lambda o: (p[o], o == "home"))
    # Each source's chance to the whole percent the app shows (the game route has the full detail), rounded half
    # up as the app's pct() rounds the full figure (Python's round(0.815, 2) gives 0.81; the game page says 82%)
    src = [[SHORT_NAMES.get(s["name"], s["name"]), whole_percent(s["p"][fav]), s.get("status") or "ok"]
           for s in detail.get("sources") or [] if isinstance(s.get("p"), Mapping) and s["p"].get(fav) is not None]
    as_of = _parse_time(blend.get("as_of"))
    return _lean({"p": p, "fav": fav, "conf": blend.get("confidence") or "normal", "dis": bool(blend.get("disagree")),
                  "spr": math.floor(float(blend.get("spread_pp") or 0.0) + 0.5), "src": src,  # half up, as the app
                  "as_of": _iso(as_of - as_of % 60) if as_of is not None else None})


def chance_of(detail: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The slate's "chance" from a market detail: the blend, src "blend"."""
    blend = (detail or {}).get("blend")
    if not blend or not isinstance(blend.get("p"), Mapping):
        return None
    p = blend["p"]
    if p.get("home") is None or p.get("away") is None:
        return None
    return _lean({"home": round(float(p["home"]), 3), "away": round(float(p["away"]), 3),
                  "draw": round(float(p["draw"]), 3) if p.get("draw") is not None else None, "src": "blend"})


def _short_logo(url: Any) -> Any:
    return url[len(ESPN_LOGO):] if isinstance(url, str) and url.startswith(ESPN_LOGO) else url


def trim_game(game: Mapping[str, Any]) -> Dict[str, Any]:
    """A game as the slate sends it: no "_" keys, no ``espn_id``/``odds_id`` its ``id`` already holds and ESPN logos
    as their path after https://a.espncdn.com/. Teams keep their full ``name`` (search and slip labels need it)."""
    out = {key: value for key, value in game.items() if not key.startswith("_")}
    league, gid = str(out.get("league") or ""), str(out.get("id") or "")
    if out.get("espn_id") and gid == f"{league}:espn:{out['espn_id']}":
        del out["espn_id"]
    if out.get("odds_id") and gid == f"{league}:{out['odds_id']}":
        del out["odds_id"]
    for side in ("home", "away"):
        team = out.get(side)
        if isinstance(team, Mapping):
            team = dict(team)
            if team.get("logo"):
                team["logo"] = _short_logo(team["logo"])
            out[side] = team
    return out


def ids_of(game: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """(odds id, ESPN id) of a slate game, from its own fields or, trimmed, from its ``id``."""
    league, gid = str(game.get("league") or ""), str(game.get("id") or "")
    rest = gid[len(league) + 1:] if gid.startswith(league + ":") else ""
    odds_id = game.get("odds_id") or (rest if rest and not rest.startswith("espn:") else None)
    espn_id = game.get("espn_id") or (rest[5:] if rest.startswith("espn:") else None)
    return odds_id, espn_id


# --- ESPN's scoreboards ----------------------------------------------------------------------------

def _status(status: Mapping[str, Any]) -> str:
    kind = status.get("type") or {}
    name = str(kind.get("name") or "").upper()
    if any(word in name for word in POSTPONED):
        return "postponed"
    state = kind.get("state")
    if state == "in":
        return "live"
    if state == "post" or kind.get("completed"):
        return "final"
    return "pre"


def _clock(sport: str, league: str, status: Mapping[str, Any]) -> Optional[str]:
    """A live game's clock, short: "Q3 4:12", "H2 11:05", "P2 8:40", "67'", "Top 5th", "R2 3:12", "Set 2"."""
    kind = status.get("type") or {}
    period = _int(status.get("period")) or 0
    clock = _text(status.get("displayClock"), 12)
    if str(kind.get("name") or "") != "STATUS_IN_PROGRESS" or not period:
        return _text(kind.get("shortDetail") or kind.get("detail"), 24)
    if sport == "soccer":
        return clock or _text(kind.get("shortDetail"), 24)
    if sport in ("basketball", "football"):
        halves = league in ("basketball_ncaab",)
        regular = 2 if halves else 4
        label = (f"H{period}" if halves else f"Q{period}") if period <= regular else (
            "OT" if period == regular + 1 else f"{period - regular}OT")
        return f"{label} {clock}" if clock else label
    if sport == "hockey":
        label = f"P{period}" if period <= 3 else ("OT" if period == 4 else "SO")
        return f"{label} {clock}" if clock else label
    if sport == "mma":
        return f"R{period} {clock}" if clock else f"R{period}"
    if sport == "tennis":
        return f"Set {period}"
    return _text(kind.get("shortDetail"), 24)


def _record(competitor: Mapping[str, Any]) -> Optional[str]:
    records = competitor.get("records") or competitor.get("record") or []
    if not isinstance(records, list):
        return None
    best = next((r for r in records if isinstance(r, dict) and r.get("type") == "total"), None) or next(
        (r for r in records if isinstance(r, dict)), None)
    return _text((best or {}).get("summary"), 16)


def _side(competitor: Mapping[str, Any], sport: str, status: str) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
    """(the team as the slate shows it, its names for matching, ESPN's team id)."""
    team = competitor.get("team")
    if isinstance(team, dict):
        names = {field: team.get(field) for field in ("displayName", "shortDisplayName", "name", "nickname", "location")}
        shown = {"name": _text(team.get("displayName") or team.get("name")), "short": _text(team.get("shortDisplayName"), 40),
                 "abbr": _text(team.get("abbreviation"), 8), "logo": _logo(team.get("logo"))}
        team_id = team.get("id")
    else:
        athlete = competitor.get("athlete") or {}
        names = {"displayName": athlete.get("displayName"), "shortDisplayName": athlete.get("shortName"),
                 "name": athlete.get("fullName")}
        shown = {"name": _text(athlete.get("displayName") or athlete.get("fullName")),
                 "short": _text(athlete.get("shortName"), 40), "logo": _logo((athlete.get("flag") or {}).get("href"))}
        team_id = competitor.get("id")
    score = None
    if status in ("live", "final"):
        if sport == "tennis":
            score = sum(1 for period in competitor.get("linescores") or [] if isinstance(period, dict) and period.get("winner"))
        elif sport != "mma":
            score = _int(competitor.get("score"))
    shown["score"] = score
    shown["record"] = _record(competitor)
    return _lean(shown), names, team_id


def parse_competition(league: str, sport: str, competition: Mapping[str, Any], event_id: Any,
                      event: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """One ESPN game in the slate's form, with its matching names and team ids under "_" keys; None when it
    hasn't two sides or a start time."""
    start = _parse_time(competition.get("date") or competition.get("startDate") or event.get("date"))
    competitors = [c for c in competition.get("competitors") or [] if isinstance(c, dict)]
    if start is None or len(competitors) != 2:
        return None
    by_side = {c.get("homeAway"): c for c in competitors}
    if "home" in by_side and "away" in by_side:
        home, away = by_side["home"], by_side["away"]
    else:  # fights: no home or away, so the first-listed fighter is shown first, as "home"
        home, away = sorted(competitors, key=lambda c: _int(c.get("order")) or 0)
    status_info = competition.get("status") or event.get("status") or {}
    status = _status(status_info)
    home_view, home_names, home_id = _side(home, sport, status)
    away_view, away_names, away_id = _side(away, sport, status)
    if not home_view.get("name") or not away_view.get("name"):
        return None
    venue = competition.get("venue") or {}
    address = venue.get("address") or {}
    venue_view = _lean({"name": _text(venue.get("fullName"), 80), "city": _text(address.get("city"), 40),
                        "indoor": venue.get("indoor") if isinstance(venue.get("indoor"), bool) else None})
    # For the context worker only (data/context.py): ESPN's venue id and the address's state and country
    place = _lean({"id": _text(venue.get("id"), 20), "state": _text(address.get("state"), 40),
                   "country": _text(address.get("country"), 40)})
    return {
        "espn_id": _text(event_id, 40),
        "start": _iso(start),
        "status": status,
        "clock": _clock(sport, league, status_info) if status == "live" else None,
        "home": home_view,
        "away": away_view,
        "venue": venue_view or None,
        "_ts": start,
        # ESPN's placeholder start (midnight Eastern) for a match whose order of play isn't out: only the day counts
        "_time_ok": competition.get("timeValid") is not False,
        "_event": _text(event.get("name") or event.get("shortName"), 80) if sport == "tennis" else None,
        "_neutral": bool(competition.get("neutralSite")),
        "_names": (home_names, away_names),
        "_ids": (home_id, away_id),
        "_place": place or None,
    }


def parse_scoreboard(league: str, events: Iterable[Any]) -> List[Dict[str, Any]]:
    """ESPN's scoreboard events as slate games. A team sport's event is one game; a tennis event is a
    tournament (its singles matches are the games) and an MMA event a card (its fights)."""
    sport = _sport(league)
    games = []
    for event in events or []:
        if not isinstance(event, dict):
            continue
        if sport == "tennis":
            pairs = [(competition, competition.get("id")) for grouping in event.get("groupings") or []
                     if ((grouping.get("grouping") or {}).get("slug") in SINGLES)
                     for competition in grouping.get("competitions") or []]
        elif sport == "mma":
            pairs = [(competition, competition.get("id")) for competition in event.get("competitions") or []]
        else:
            pairs = [(competition, event.get("id")) for competition in (event.get("competitions") or [])[:1]]
        for competition, game_id in pairs:
            if not isinstance(competition, dict):
                continue
            try:
                game = parse_competition(league, sport, competition, game_id, event)
            except (AttributeError, TypeError, ValueError, KeyError):
                game = None  # one odd entry costs only itself, not the league's board
            if game is not None:
                games.append(game)
    return games


# --- odds snapshots ----------------------------------------------------------------------------------

def odds_view(event: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """(best prices, market chances) of one saved odds event: {"h2h": {"home": [price, book], ...},
    "spreads": {"home": [price, book, point], ...}, "totals": {...}} and {"home", "away", "draw", "src"}."""
    try:
        summary = summarize_markets(dict(event))
    except Exception:  # a malformed saved event: no prices rather than no slate
        return None, None
    prices: Dict[str, Dict[str, List[Any]]] = {}
    for market in ("h2h", "spreads", "totals"):
        sides = summary.get(market) or {}
        view = {}
        for side, offer in sides.items():
            price = offer.get("price")
            if not isinstance(price, (int, float)) or price <= 1:
                continue
            entry: List[Any] = [round(float(price), 3), _text(offer.get("book"), 40) or "?"]
            if market != "h2h":
                entry.append(offer.get("point"))
            view[side] = entry
        if view:
            prices[market] = view
    fair = {side: offer.get("fair") for side, offer in (summary.get("h2h") or {}).items()}
    chance = None
    if isinstance(fair.get("home"), float) and isinstance(fair.get("away"), float):
        chance = _lean({"home": round(fair["home"], 3), "away": round(fair["away"], 3),
                        "draw": round(fair["draw"], 3) if isinstance(fair.get("draw"), float) else None,
                        "src": "market"})
    return prices or None, chance


def board_lines(event: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """The context worker's lines from one saved odds event (data/context.py): the consensus spread (the home side's
    point), the consensus total and the books' margin-free chance of the over at it; None when there are none."""
    try:
        summary = summarize_markets(dict(event))
    except Exception:
        return None
    lines: Dict[str, float] = {}
    home = (summary.get("spreads") or {}).get("home") or {}
    if isinstance(home.get("point"), (int, float)):
        lines["spread"] = float(home["point"])
    over = (summary.get("totals") or {}).get("over") or {}
    if isinstance(over.get("point"), (int, float)):
        lines["total"] = float(over["point"])
        if isinstance(over.get("fair"), float) and 0 < over["fair"] < 1:
            lines["p_over"] = round(over["fair"], 6)
    return lines or None


@dataclass
class _Board:
    """One league's scoreboard for one day."""

    fetched: float = 0.0  # when it was last fetched (wall clock); 0 when it never has been
    failed: float = 0.0  # when its last fetch failed, if after ``fetched``
    failures: int = 0  # fetches failed in a row
    games: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SlateBytes:
    """A slate as served: the JSON, its gzip, its ETag (quoted) and the parsed document (for filters)."""

    raw: bytes
    gzip: bytes
    etag: str
    generated_at: float
    data: Dict[str, Any]


def pack(data: Mapping[str, Any], generated_at: float) -> SlateBytes:
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8", "replace")
    return SlateBytes(raw, gzip.compress(raw, compresslevel=6, mtime=0),
                      '"' + hashlib.sha256(raw).hexdigest()[:24] + '"', generated_at, dict(data))


class SlateBuilder:
    """Builds the all-sports slate (see the module docstring). ``odds`` is the OddsService (its sports()
    listing, quota, reserve, odds_ttl and league_games for request_odds); ``espn`` the EspnClient;
    ``store`` the shared Store (odds snapshots, the NBA model's logged predictions, the saved slate);
    ``service`` the DashboardService, which loads an AI league's board with its model (optional)."""

    def __init__(self, odds: Any, espn: Any, store: Any = None, *, service: Any = None,
                 tz: Optional[datetime.tzinfo] = None, live_seconds: float = 20, today_seconds: float = 60,
                 idle_seconds: float = 600, future_seconds: float = 3600, odds_per_day: int = ODDS_PER_DAY,
                 ai_leagues: Iterable[str] = AI_LEAGUES, workers: int = WORKERS,
                 now: Callable[[], float] = time.time, polymarket: Any = None, kalshi: Any = None,
                 ratings: Any = None, markets_every: float = MARKETS_EVERY, log_consensus: bool = True,
                 on_change: Optional[Callable[..., Any]] = None, context: Any = None,
                 context_every: float = CONTEXT_EVERY, player_board: Any = None,
                 players_every: float = PLAYERS_EVERY) -> None:
        self.odds = odds
        self.espn = espn
        self.store = store
        self.service = service
        self.tz = tz or ZoneInfo("America/New_York")
        self.live_seconds = max(1.0, float(live_seconds))
        self.today_seconds = max(self.live_seconds, float(today_seconds))
        self.idle_seconds = max(self.today_seconds, float(idle_seconds))
        self.future_seconds = max(self.today_seconds, float(future_seconds))
        self.odds_per_day = max(0, int(odds_per_day))
        self.ai_leagues = frozenset(ai_leagues)
        self._now = now
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="slate-espn")
        self._odds_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="slate-odds")
        self._boards: Dict[Tuple[str, datetime.date], _Board] = {}
        self._fallback: Dict[Tuple[str, datetime.date], List[Dict[str, Any]]] = {}
        self._odds: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}  # league -> (fetched_at, events)
        self._season: Optional[List[Dict[str, Any]]] = None
        self._season_next = 0.0  # when to read the in-season listing again (0: never read yet)
        self._season_busy = False  # a read is on its way, in the background
        self._paused_until = 0.0  # while most boards fail: only live boards are fetched before this time
        self._trips = 0  # rounds in a row that paused the boards
        self._titles: Dict[str, str] = {}
        self._league_info: Dict[str, Dict[str, Any]] = {}
        self._current: Optional[SlateBytes] = None
        self._core: Optional[bytes] = None  # the last slate's leagues and games, to tell a change
        self._filtered: "OrderedDict[Tuple[str, Tuple[str, ...], str], SlateBytes]" = OrderedDict()
        self._filter_lock = threading.Lock()
        self._ready = threading.Event()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._refreshing = threading.Lock()
        self._lock = threading.Lock()  # request_odds' bookkeeping
        self._requested: Dict[str, float] = {}
        self._loads: List[float] = []
        self._loading: Set[str] = set()
        self._logged: Dict[str, float] = {}
        self._thread: Optional[threading.Thread] = None
        # The markets brain: the free outside sources (data/markets; None when MARKETS_ENABLED is off), fetched only
        # by the markets worker; each game's market detail by slate id, and an index by odds and ESPN ids
        self.polymarket, self.kalshi, self.ratings = polymarket, kalshi, ratings
        self.markets_every = max(5.0, float(markets_every))
        self.log_consensus = log_consensus
        # A daemon thread per round (Kalshi paces itself to a request a second, so a round can take a minute or two:
        # it must never hold up the process's exit)
        self._markets_thread: Optional[threading.Thread] = None
        self._markets_busy = False
        self._markets_next = 0.0
        self._markets: Dict[str, Dict[str, Any]] = {}
        self._markets_index: Dict[Tuple[str, str, str], str] = {}
        self._matches: Dict[str, Tuple[Any, Dict[str, List[Tuple[str, Any, bool]]]]] = {}  # league -> pairings
        self._predictors: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (league, ESPN id) -> ESPN's predictor
        self._predictor_asked: Dict[Tuple[str, str], float] = {}  # (league, ESPN id) -> when the worker last asked
        self._logged_sig: Dict[Tuple[str, str], str] = {}  # what consensus_log last got for each game
        self._t24_logged: Set[Tuple[str, str]] = set()  # games whose one-day-before snapshot was sent (kind "t24")
        self._saved_seen: Dict[str, Optional[Dict[str, Any]]] = {}  # started games' logged detail, read once
        # Told when a game's status or score changes between rounds: on_change(league, odds_id=, espn_id=, board=)
        # (services.build_slate passes DashboardService.invalidate_game, so the game's computed entries in the
        # shared cache are dropped and rebuilt once for everyone; ``board`` is True when the status moved)
        self.on_change = on_change
        self._marks: Dict[str, Tuple[Any, Any, Any]] = {}  # game id -> (status, home score, away score) last round
        # Venue & conditions (data/context.py's ContextService): warmed by a background round at most every
        # ``context_every`` seconds; each row's badges gain the context codes from the computed entries
        self.context = context
        self.context_every = max(1.0, float(context_every))
        self._context_thread: Optional[threading.Thread] = None
        self._context_busy = False
        self._context_next = 0.0
        self._context_asked = -math.inf
        self._past: "OrderedDict[Tuple[str, datetime.date], List[Dict[str, Any]]]" = OrderedDict()
        self._past_lock = threading.Lock()
        # The player scan's boards (data/player_board.py's PlayerBoard; else the service's ``player_board`` when one
        # is set): prewarmed for leagues with a game in the next 12 hours, at most every ``players_every`` seconds
        self.player_board = player_board
        self.players_every = max(1.0, float(players_every))
        self._players_next = 0.0

    # --- serving -----------------------------------------------------------------------------------

    def current(self, wait: float = 0.0) -> Optional[SlateBytes]:
        """The latest slate, waiting up to ``wait`` seconds for the first one; None if there is none yet."""
        if self._current is None and wait > 0:
            self._ready.wait(wait)
        return self._current

    def filtered(self, leagues: Optional[Sequence[str]] = None, day: Optional[str] = None) -> Optional[SlateBytes]:
        """The slate with only these leagues and/or one day (YYYY-MM-DD in the builder's time zone), its
        counts recounted. The app filters in the browser; this is for other callers."""
        current = self._current
        if current is None:
            return None
        wanted = tuple(sorted(set(leagues or ())))
        key = (current.etag, wanted, day or "")
        with self._filter_lock:
            cached = self._filtered.get(key)
        if cached is not None:
            return cached
        wanted_day = datetime.date.fromisoformat(day) if day else None
        games = [game for game in current.data.get("games") or []
                 if (not wanted or game.get("league") in wanted)
                 and (wanted_day is None or self._local_day(_parse_time(game.get("start"))) == wanted_day)]
        counts: Dict[str, List[int]] = {}
        for game in games:
            count = counts.setdefault(game["league"], [0, 0])
            count[0] += 1
            count[1] += game.get("status") == "live"
        leagues_view = [{**entry, "count": counts.get(entry["key"], [0, 0])[0], "live": counts.get(entry["key"], [0, 0])[1]}
                        for entry in current.data.get("leagues") or [] if not wanted or entry.get("key") in wanted]
        data = {**current.data, "leagues": leagues_view, "games": games}
        result = pack(data, current.generated_at)
        with self._filter_lock:
            self._filtered[key] = result
            while len(self._filtered) > 32:
                self._filtered.popitem(last=False)
        return result

    def _local_day(self, ts: Optional[float]) -> Optional[datetime.date]:
        return datetime.datetime.fromtimestamp(ts, self.tz).date() if ts is not None else None

    # --- the saved slate ---------------------------------------------------------------------------

    def load(self) -> bool:
        """Serve the slate saved by the last run (if any) until the first refresh finishes."""
        if self.store is None or not hasattr(self.store, "slate"):
            return False
        try:
            saved = self.store.slate(SLATE_NAME)
        except Exception:
            log.exception("Couldn't read the saved slate")
            return False
        if not saved:
            return False
        try:
            generated_at, _, body = saved
            data = json.loads(body.decode("utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("leagues") or [], list) \
                    or not isinstance(data.get("games") or [], list):
                raise ValueError("not a slate")
            self._seed(data)
            current = pack(data, float(generated_at))
            core = self._core_bytes(data.get("leagues") or [], data.get("games") or [], data.get("quota"))
        except Exception as exc:
            # Another version's slate (after a rollback, say) or a damaged row: the first refresh builds a new one
            log.warning("The saved slate can't be used (%s); starting without it", type(exc).__name__)
            self._fallback.clear()
            self._titles.clear()
            return False
        self._core = core
        self._current = current
        self._ready.set()
        return True

    def _seed(self, data: Mapping[str, Any]) -> None:
        """Keep the saved slate's ESPN games, by league and ESPN day, as the last good copy of boards not
        fetched yet; and its league titles."""
        for entry in data.get("leagues") or []:
            if isinstance(entry, dict) and entry.get("key") and entry.get("title"):
                self._titles[entry["key"]] = entry["title"]
        for game in data.get("games") or []:
            if not isinstance(game, dict) or not all(isinstance(game.get(side), dict) for side in ("home", "away")):
                raise ValueError("not a game")  # not a slate this version wrote: load() starts without it
            espn_id = ids_of(game)[1]  # trimmed: the id holds an ESPN-only game's ESPN id
            if not espn_id:
                continue
            ts = _parse_time(game.get("start"))
            league = str(game.get("league") or "")
            source = league if espn_path(league) else _tour(league)
            if ts is None or source is None:
                continue
            day = datetime.datetime.fromtimestamp(ts, ESPN_DAY_TZ).date()
            names = tuple({"displayName": _text(game[side].get("name")),
                           "shortDisplayName": _text(game[side].get("short")),
                           "abbreviation": _text(game[side].get("abbr"), 8)} for side in ("home", "away"))
            self._fallback.setdefault((source, day), []).append({
                "espn_id": espn_id, "start": game.get("start"), "status": game.get("status") or "pre",
                "clock": game.get("clock"), "home": dict(game["home"]), "away": dict(game["away"]),
                "venue": game.get("venue"), "_ts": ts, "_neutral": "neutral" in (game.get("badges") or []),
                "_names": names, "_ids": (None, None),
            })

    # --- the background thread ---------------------------------------------------------------------

    def start(self) -> threading.Thread:
        """Refresh on the schedule in a daemon thread (once; later calls return the same thread)."""
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self.run, name="slate", daemon=True)
            self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        """Refresh now (an odds board arrived)."""
        self._wake.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                delay = self.refresh()
            except Exception:
                log.exception("Building the slate failed")
                delay = self.today_seconds
            self._wake.wait(delay)
            self._wake.clear()

    # --- building ----------------------------------------------------------------------------------

    def refresh(self) -> float:
        """Fetch the ESPN boards that are due, rebuild the slate and publish it if anything changed.
        Returns the seconds until the next board is due."""
        with self._refreshing:
            now = self._now()
            season = self._read_season(now)
            today = datetime.datetime.fromtimestamp(now, ESPN_DAY_TZ).date()
            due = self._due(season, today, now)
            if due:
                failed = 0
                for league, day, games, error in self._pool.map(self._fetch, due):
                    board = self._boards.setdefault((league, day), _Board())
                    # Each board of a round is stamped with the round's start, so they come due again together
                    # (one wake-up and one rebuild per round, not one per straggler)
                    if error is None:
                        board.games = games
                        board.fetched = now
                        board.failed = 0.0
                        board.failures = 0
                        self._fallback.pop((league, day), None)
                    else:
                        failed += 1
                        board.failed = now
                        board.failures += 1
                        self._log_failure(league, error)
                self._check_round(len(due), failed, now)
            for key in [key for key in self._boards if key[1] < today - datetime.timedelta(days=1)]:
                del self._boards[key]
            for key in [key for key in self._fallback if key[1] < today - datetime.timedelta(days=1)]:
                del self._fallback[key]
            # The clock again: the boards can take seconds, and meanwhile a game may have started (the markets worker
            # and request_odds keep updating the sources); the slate is built as of now
            now = self._now()
            leagues, games, as_of = self._assemble(season, today, now)
            self._note_changes(games)
            try:
                self._apply_markets(games, now)
            except Exception:  # the markets brain must never cost the scores
                log.exception("Slate: the markets brain failed this round")
            self._schedule_markets(games, now)
            self._apply_context(games)
            self._schedule_context(games, now)
            self._prewarm_players(games, now)
            self._publish(leagues, games, as_of, now, self._quota())
            return self._next_delay(season, today, self._now())

    def _note_changes(self, games: Sequence[Mapping[str, Any]]) -> None:
        """Tell on_change about each game whose status or score moved since the last round (the first round only
        notes them). A failing hook costs only its own call."""
        marks: Dict[str, Tuple[Any, Any, Any]] = {}
        changed: List[Tuple[Mapping[str, Any], bool]] = []
        for game in games:
            mark = (game.get("status"), (game.get("home") or {}).get("score"), (game.get("away") or {}).get("score"))
            marks[game["id"]] = mark
            before = self._marks.get(game["id"])
            if before is not None and before != mark:
                changed.append((game, before[0] != mark[0]))
        self._marks = marks
        if self.on_change is None:
            return
        for game, status_moved in changed:
            try:
                self.on_change(game["league"], odds_id=game.get("odds_id"), espn_id=game.get("espn_id"),
                               board=status_moved)
            except Exception:
                log.exception("Slate: dropping the computed entries of %s failed", game.get("id"))

    def _read_season(self, now: float) -> Optional[List[Dict[str, Any]]]:
        """The Odds API's in-season leagues (its free listing), else the last good. Read at once the first time,
        so the first slate knows which leagues are in season; after that in the background, at most every
        SEASON_EVERY seconds (SEASON_RETRY after a failure), so a slow or failing listing never holds up the scores."""
        if not self._season_next:
            self._load_season()
        elif now >= self._season_next and not self._season_busy:
            self._season_busy = True
            try:
                self._odds_pool.submit(self._load_season)
            except RuntimeError:  # shutting down
                self._season_busy = False
        return self._season

    def _load_season(self) -> None:
        ok = False
        try:
            listing = self.odds.sports()
            if isinstance(listing, list):
                self._season = [sport for sport in listing if isinstance(sport, dict) and sport.get("key")]
                ok = True
        except Exception as exc:
            # Only the type: a connection error's message can quote the request URL, which holds the API key
            self._log_failure("sports listing", exc, detail=False)
        finally:
            self._season_next = self._now() + (SEASON_EVERY if ok else SEASON_RETRY)
            self._season_busy = False

    @staticmethod
    def _active(season: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Dict[str, Any]]]:
        if season is None:
            return None
        return {sport["key"]: sport for sport in season if sport.get("active") and not sport.get("has_outrights")}

    def _espn_keys(self, season: Optional[List[Dict[str, Any]]], today: datetime.date) -> Dict[str, bool]:
        """ESPN leagues to fetch -> whether they are in season (all days fetched) or only checked today."""
        active = self._active(season)
        keys = {}
        for league in list(ESPN_LEAGUES) + list(SCOREBOARD_ONLY):
            in_season = active is None or league in active or league in TENNIS_PREFIXES.values()
            board = self._boards.get((league, today))
            keys[league] = in_season or bool(board and board.games)
        return keys

    def _days(self, league: str, in_season: bool, today: datetime.date) -> List[datetime.date]:
        sport = _sport(league)
        if sport == "tennis" or not in_season:
            days = [today]  # a tennis board lists whole tournaments; a league out of season is only checked
        else:
            days = [today + datetime.timedelta(days=n) for n in range(DAYS_AHEAD + 1)]
        yesterday = today - datetime.timedelta(days=1)
        previous = self._boards.get((league, yesterday))
        if (in_season and sport in B2B_SPORTS) or (previous and any(g["status"] == "live" for g in previous.games)):
            days.insert(0, yesterday)
        return days

    def _pace(self, league: str, day: datetime.date, board: Optional[_Board], today: datetime.date, now: float
              ) -> Tuple[float, bool]:
        """(seconds between the board's fetches, whether it has a game on or about to start)."""
        sport = _sport(league)
        live, busy = self.live_seconds, self.today_seconds
        if sport == "tennis":  # its boards are large (whole tournaments): a slower pace
            live, busy = max(live, 60.0), max(busy, 300.0)
        games = board.games if board else []
        if any(g["status"] == "live" or (g["status"] == "pre" and -STUCK < g["_ts"] - now < LIVE_SOON) for g in games):
            return live, True
        if day == today:
            return (busy if any(g["status"] in ("pre", "live") for g in games) else self.idle_seconds), False
        return self.future_seconds, False

    def _due_at(self, key: Tuple[str, datetime.date], board: Optional[_Board], ttl: float, live: bool) -> float:
        """When the board is next due: at once if it was never tried, its pace after its last fetch, and after a
        failure ever further apart (doubling from its own pace, or from a minute while there is nothing to show
        for it yet), so a failing ESPN is never asked more often than a working one."""
        if board is None or not (board.fetched or board.failed):
            return 0.0
        if board.fetched >= board.failed:
            return board.fetched + ttl
        step = ttl if board.fetched or key in self._fallback else min(ttl, max(self.today_seconds, 60.0))
        cap = max(ttl, LIVE_BACKOFF_CAP if live else BACKOFF_CAP)
        return board.failed + min(cap, step * 2 ** min(max(board.failures - 1, 0), 16))

    def _schedule(self, season: Optional[List[Dict[str, Any]]], today: datetime.date, now: float
                  ) -> List[Tuple[str, datetime.date, float, float, bool]]:
        """Every board the slate keeps: (league, day, its pace, when it is next due, whether it is held back).
        While ESPN is failing most boards (see _check_round) only those with a game on are fetched."""
        paused = now < self._paused_until
        items = []
        for league, in_season in self._espn_keys(season, today).items():
            for day in self._days(league, in_season, today):
                board = self._boards.get((league, day))
                ttl, live = self._pace(league, day, board, today, now)
                at = self._due_at((league, day), board, ttl, live)
                held = paused and not live
                items.append((league, day, ttl, max(at, self._paused_until) if held else at, held))
        return items

    def _due(self, season: Optional[List[Dict[str, Any]]], today: datetime.date, now: float
             ) -> List[Tuple[str, datetime.date, float]]:
        # A round also takes the boards due within half the live pace: boards that drifted apart (a long first
        # round, a board fetched on a wake-up) come due together again, so there is one wake-up and one rebuild
        # per round. Boards on the same footing (stamped by one round, paces a multiple of the live pace) are
        # never fetched early.
        horizon = now + self.live_seconds / 2
        due = [(league, day, ttl) for league, day, ttl, at, held in self._schedule(season, today, now)
               if not held and at < horizon]
        # Today's boards first, so a slow future day never holds them up
        due.sort(key=lambda item: (item[1] != today, item[1], item[0]))
        return due

    def _next_delay(self, season: Optional[List[Dict[str, Any]]], today: datetime.date, now: float) -> float:
        soonest = min((item[3] for item in self._schedule(season, today, now)), default=now + self.idle_seconds)
        return max(1.0, min(self.idle_seconds, soonest - now))

    def _check_round(self, tried: int, failed: int, now: float) -> None:
        """When most of a round's boards failed (ESPN down, or pushing back with 429s), fetch only boards with
        a game on for the next 5 minutes, 10 the time after, then 15, until a round mostly succeeds again."""
        if failed >= BREAKER_MIN and 2 * failed > tried:
            self._trips += 1
            pause = BREAKER_PAUSE[min(self._trips, len(BREAKER_PAUSE)) - 1]
            self._paused_until = now + pause
            if self._trips == 1:
                log.warning("Slate: %d of %d ESPN scoreboards failed; only boards with a game on are fetched for "
                            "the next %d minutes (longer while it lasts)", failed, tried, pause // 60)
        elif self._trips and now >= self._paused_until and 2 * failed <= tried:
            log.info("Slate: ESPN's scoreboards answer again")
            self._trips = 0

    def _fetch(self, item: Tuple[str, datetime.date, float]
               ) -> Tuple[str, datetime.date, List[Dict[str, Any]], Optional[BaseException]]:
        """One board, fetched and trimmed in a worker (the raw JSON is dropped here)."""
        league, day, _ = item
        try:
            return league, day, parse_scoreboard(league, self.espn.fetch_scoreboard(league, day)), None
        except Exception as exc:
            return league, day, [], exc

    def _log_failure(self, what: str, exc: BaseException, detail: bool = True) -> None:
        """A warning at most once an hour per league. ESPN's errors carry only its own public URLs; with
        ``detail`` False only the error's type is logged (an Odds API error could quote its URL, and the key)."""
        now = self._now()
        if now - self._logged.get(what, 0.0) >= 3600:
            self._logged[what] = now
            log.warning("Slate: couldn't load %s (%s%s); keeping its last good copy", what, type(exc).__name__,
                        f": {str(exc)[:160]}" if detail else "")

    # --- odds ---------------------------------------------------------------------------------------

    def _odds_ttl(self) -> float:
        ttl = getattr(self.odds, "odds_ttl", None)
        return float(ttl) if isinstance(ttl, (int, float)) and ttl > 0 else ODDS_TTL

    def _odds_times(self) -> Dict[str, float]:
        if self.store is None or not hasattr(self.store, "odds_fetched_at"):
            return {}
        try:
            return dict(self.store.odds_fetched_at())
        except Exception as exc:
            self._log_failure("saved odds", exc)
            return {}

    def _odds_events(self, league: str, fetched_at: Optional[float]) -> List[Dict[str, Any]]:
        """The league's saved odds events with their best prices and chances (re-read only when the saved
        snapshot changes). Never fetches."""
        if fetched_at is None:
            self._odds.pop(league, None)
            return []
        cached = self._odds.get(league)
        if cached and cached[0] == fetched_at:
            return cached[1]
        try:
            snapshot = self.store.odds_snapshot(league)
        except Exception as exc:
            self._log_failure(f"{league} odds", exc)
            return cached[1] if cached else []
        events = []
        for event in (snapshot or (fetched_at, []))[1]:
            if not isinstance(event, dict):
                continue
            start = _parse_time(event.get("commence_time"))
            home, away = _text(event.get("home")), _text(event.get("away"))
            event_id = _text(event.get("id"), 64)
            if start is None or not home or not away or not event_id or not re.fullmatch(r"[A-Za-z0-9_-]+", event_id):
                continue
            prices, chance = odds_view(event)
            try:
                books = moneylines(event)
            except Exception:
                books = {}
            events.append({"odds_id": event_id, "_ts": start, "home": home, "away": away, "odds": prices,
                           "chance": chance, "_books": books, "_fetched": fetched_at, "_lines": board_lines(event)})
        self._odds[league] = (fetched_at, events)
        return events

    def _ai_chances(self) -> Dict[Tuple[str, str], Tuple[float, str, Optional[float], Optional[float]]]:
        """The NBA model's logged pre-game home-win chances: (league, odds id) and ("espn", ESPN id) ->
        (chance, odds id, the team model's own chance or None, when it was logged). Only what it already worked
        out; nothing is computed here."""
        chances: Dict[Tuple[str, str], Tuple[float, str, Optional[float], Optional[float]]] = {}
        if self.store is None or not hasattr(self.store, "ai_predictions"):
            return chances
        for league in sorted(self.ai_leagues):
            try:
                rows = self.store.ai_predictions(league=league, status="pending")
            except Exception as exc:
                self._log_failure(f"{league} model predictions", exc)
                continue
            for row in rows:
                payload = row.get("payload") or {}
                home_win = payload.get("home_win")
                if not isinstance(home_win, (int, float)) or not 0 <= home_win <= 1:
                    continue
                model = payload.get("model_home_win")
                model = float(model) if isinstance(model, (int, float)) and 0 < model < 1 else None
                logged = row.get("logged_at") if isinstance(row.get("logged_at"), (int, float)) else None
                entry = (float(home_win), str(row.get("event_id")), model, logged)
                chances[(league, str(row.get("event_id")))] = entry
                if row.get("espn_id"):
                    chances[("espn", str(row["espn_id"]))] = entry
        return chances

    # --- assembling -----------------------------------------------------------------------------------

    def _pool_games(self, league: str, today: datetime.date) -> List[Dict[str, Any]]:
        """An ESPN league's games from yesterday to six days ahead (yesterday's only to recognise an odds event
        for a game already over; _shown keeps those out unless still live)."""
        last = today + datetime.timedelta(days=DAYS_AHEAD)
        games = []
        for day in [today - datetime.timedelta(days=1)] + [today + datetime.timedelta(days=n) for n in range(DAYS_AHEAD + 1)]:
            board = self._boards.get((league, day))
            source = board.games if board is not None and board.fetched else self._fallback.get((league, day), [])
            for game in source:
                game_day = datetime.datetime.fromtimestamp(game["_ts"], ESPN_DAY_TZ).date()
                if game_day <= last and game_day >= today - datetime.timedelta(days=1):
                    games.append(game)
        unique: Dict[Any, Dict[str, Any]] = {}
        for game in games:  # a tennis tournament's matches come with every day's board
            unique.setdefault(game["espn_id"] or id(game), game)
        return list(unique.values())

    @staticmethod
    def _shown(game: Mapping[str, Any], today: datetime.date) -> bool:
        """Today's and later games, and earlier ones still being played (a late game past midnight)."""
        return game["status"] == "live" or datetime.datetime.fromtimestamp(game["_ts"], ESPN_DAY_TZ).date() >= today

    @staticmethod
    def _match(event: Mapping[str, Any], pool: List[Dict[str, Any]], used: Set[int]
               ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """The ESPN game this odds event is, and whether ESPN lists its teams the other way round."""
        best: Tuple[float, float, Optional[Dict[str, Any]], bool] = (0.0, 0.0, None, False)
        for game in pool:
            if id(game) in used or abs(game["_ts"] - event["_ts"]) > MATCH_WINDOW:
                continue
            home_names, away_names = game["_names"]
            for swapped, (first, second) in ((False, (home_names, away_names)), (True, (away_names, home_names))):
                scores = (team_match_score(event["home"], first), team_match_score(event["away"], second))
                if min(scores) >= MATCH or max(scores) >= STRONG_MATCH:
                    rank = (sum(scores), -abs(game["_ts"] - event["_ts"]))
                    if rank > best[:2]:
                        best = (rank[0], rank[1], game, swapped)
        return best[2], best[3]

    def _game(self, league: str, espn: Optional[Mapping[str, Any]], event: Optional[Mapping[str, Any]], swapped: bool,
              odds_at: Optional[float], ai: Mapping[Tuple[str, str], Tuple[float, str, Optional[float], Optional[float]]],
              now: float) -> Optional[Dict[str, Any]]:
        sport = _sport(league)
        if espn is not None:
            home, away = dict(espn["home"]), dict(espn["away"])
            ids = espn["_ids"]
            if swapped:
                home, away = away, home
                ids = (ids[1], ids[0])
            status, start_ts = espn["status"], espn["_ts"]
        else:
            home, away = {"name": event["home"]}, {"name": event["away"]}
            ids = (None, None)
            start_ts = event["_ts"]
            if start_ts > now:
                status = "pre"
            elif now - start_ts < _odds_only_live(league):
                status = "live"
            else:
                return None  # long started and no scoreboard: its result is unknown here
        badges = []
        if status == "live":
            badges.append("live")
        if espn is not None and espn.get("_neutral"):
            badges.append("neutral")
        if sport in B2B_SPORTS and status in ("pre", "live"):
            day = datetime.datetime.fromtimestamp(start_ts, ESPN_DAY_TZ).date()
            played = self._played(league, day - datetime.timedelta(days=1))
            for side, team_id in zip(("home", "away"), ids):
                if team_id is not None and team_id in played:
                    badges.append(f"b2b-{side}")
        odds_id = event["odds_id"] if event else None
        espn_id = espn.get("espn_id") if espn else None
        model = ai.get((league, odds_id)) if odds_id else None
        if model is None and espn_id:
            model = ai.get(("espn", espn_id))
            if model is not None and odds_id and model[1] != odds_id:
                model = None
        return _lean({
            "id": f"{league}:{odds_id}" if odds_id else f"{league}:espn:{espn_id}",
            "league": league,
            "espn_id": espn_id,
            "odds_id": odds_id,
            "start": _iso(start_ts),
            "status": status,
            "clock": espn.get("clock") if espn else None,
            "home": home,
            "away": away,
            "venue": espn.get("venue") if espn else None,
            "odds": event.get("odds") if event else None,
            "chance": event.get("chance") if event else None,
            "ai": {"home": round(model[0], 3), "away": round(1 - model[0], 3)} if model else None,
            "badges": badges or None,
            "odds_as_of": _iso(odds_at) if event else None,
            # For the markets brain only (trim_game drops "_" keys before the slate is sent)
            "_ts": start_ts, "_swapped": swapped, "_neutral": bool(espn and espn.get("_neutral")),
            "_time_ok": bool(espn.get("_time_ok", True)) if espn else True,
            "_surface": tennis_surface(league, espn.get("_event") if espn else None),
            "_books": (event.get("_books") or None) if event else None, "_model": model,
            # For the context worker (data/context.py): the teams' ESPN ids (as oriented here), ESPN's venue id and
            # address, and the board's lines
            "_ids": tuple(ids) if espn is not None else None, "_place": espn.get("_place") if espn else None,
            "_lines": event.get("_lines") if event else None,
        })

    def _played(self, league: str, day: datetime.date) -> Set[Any]:
        """ESPN ids of the league's teams that played (or play) on ``day``."""
        board = self._boards.get((league, day))
        return {team_id for game in (board.games if board else []) for team_id in game["_ids"] if team_id is not None}

    def _assemble(self, season: Optional[List[Dict[str, Any]]], today: datetime.date, now: float
                  ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        active = self._active(season) or {}
        espn_keys = self._espn_keys(season, today)
        pools = {league: self._pool_games(league, today) for league in espn_keys}
        used: Set[int] = set()
        odds_times = self._odds_times()
        ai = self._ai_chances()
        window_end = datetime.datetime.combine(today + datetime.timedelta(days=DAYS_AHEAD + 1), datetime.time(),
                                               ESPN_DAY_TZ).timestamp()
        games: List[Dict[str, Any]] = []
        for league in sorted(set(active) | set(pools)):
            fetched_at = odds_times.get(league)
            events = self._odds_events(league, fetched_at)
            pool = pools.get(league if league in pools else _tour(league) or "", [])
            for event in sorted(events, key=lambda e: (e["_ts"], e["odds_id"])):
                if event["_ts"] >= window_end:
                    continue
                match, swapped = self._match(event, pool, used) if pool else (None, False)
                if match is not None:
                    used.add(id(match))
                    if not self._shown(match, today):
                        continue  # last night's game, over: not on today's slate
                game = self._game(league, match, event, swapped, fetched_at, ai, now)
                if game is not None:
                    games.append(game)
        for league, pool in pools.items():
            for espn_game in pool:
                if id(espn_game) not in used and self._shown(espn_game, today):
                    game = self._game(league, espn_game, None, False, None, ai, now)
                    if game is not None:
                        games.append(game)
        games.sort(key=lambda game: (game["start"], game["league"], game["id"]))
        leagues = self._league_entries(season, active, today, games, odds_times, now)
        fetched = [board.fetched for board in self._boards.values() if board.fetched]
        return leagues, games, max(fetched) if fetched else now

    def _league_entries(self, season: Optional[List[Dict[str, Any]]], active: Mapping[str, Mapping[str, Any]],
                        today: datetime.date, games: List[Dict[str, Any]], odds_times: Mapping[str, float],
                        now: float) -> List[Dict[str, Any]]:
        counts: Dict[str, List[int]] = {}
        upcoming: Dict[str, int] = {}
        for game in games:
            count = counts.setdefault(game["league"], [0, 0])
            count[0] += 1
            count[1] += game["status"] == "live"
            if game["status"] in ("pre", "live"):
                upcoming[game["league"]] = upcoming.get(game["league"], 0) + 1
        listed = {sport["key"]: sport for sport in season or []}
        keys = set(active) | set(counts)
        odds_ttl = self._odds_ttl()
        entries = []
        info: Dict[str, Dict[str, Any]] = {}
        for key in keys:
            sport = listed.get(key) or {}
            path = espn_path(key)
            group = _text(sport.get("group"), 40) or GROUP_OF_SPORT.get((path or ("",))[0]) or "Other"
            title = _text(sport.get("title"), 60) or self._titles.get(key) or TITLES.get(key) or key
            fetched = odds_times.get(key)
            odds = "none" if fetched is None else ("fresh" if now - fetched < odds_ttl else "stale")
            ai = key in self.ai_leagues
            entries.append(_lean({
                "key": key, "title": title, "group": group, "label": LABELS.get(group, group),
                "count": counts.get(key, [0, 0])[0], "live": counts.get(key, [0, 0])[1], "ai": ai, "odds": odds,
                "odds_as_of": _iso(fetched), "espn": path is not None or any(key.startswith(p) for p in TENNIS_PREFIXES),
            }))
            board = self._boards.get((key, today))
            # ESPN's word that a league has no games counts only once its board for today has loaded
            info[key] = {"in_season": key in active, "espn": path is not None and bool(board and board.fetched),
                         "upcoming": upcoming.get(key, 0), "odds_fetched": fetched}
        self._league_info = info

        def rank(entry: Mapping[str, Any]) -> Tuple[Any, ...]:
            group = entry["group"]
            key = entry["key"]
            return (GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER), group,
                    POPULAR_LEAGUES.index(key) if key in POPULAR_LEAGUES else len(POPULAR_LEAGUES), entry["title"], key)

        return sorted(entries, key=rank)

    @staticmethod
    def _core_bytes(leagues: List[Dict[str, Any]], games: List[Dict[str, Any]],
                    quota: Optional[Mapping[str, int]] = None) -> bytes:
        return json.dumps({"leagues": leagues, "games": games, "quota": quota}, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8", "replace")

    def _quota(self) -> Optional[Dict[str, int]]:
        """The month's Odds API requests left and used, as last reported (the odds service reads the shared
        store; no request is made), for the header's "N odds left"; None when unknown."""
        try:
            quota = getattr(self.odds, "quota", None)
        except Exception as exc:
            self._log_failure("odds quota", exc)
            return None
        if not isinstance(quota, Mapping):
            return None
        view = {key: int(quota[key]) for key in ("remaining", "used")
                if isinstance(quota.get(key), int) and not isinstance(quota.get(key), bool)}
        return view if "remaining" in view else None

    def _publish(self, leagues: List[Dict[str, Any]], games: List[Dict[str, Any]], as_of: float, now: float,
                 quota: Optional[Dict[str, int]] = None) -> None:
        """Serialise the slate once, if it changed; keep it in memory and in the store."""
        games = [trim_game(game) for game in games]
        core = self._core_bytes(leagues, games, quota)
        if self._current is not None and core == self._core:
            return
        data = _lean({"generated_at": _iso(now), "as_of": _iso(min(as_of, now)), "quota": quota,
                      "leagues": leagues, "games": games})
        current = pack(data, now)
        self._core = core
        self._current = current
        with self._filter_lock:
            self._filtered.clear()
        self._ready.set()
        if self.store is not None and hasattr(self.store, "save_slate"):
            try:
                self.store.save_slate(SLATE_NAME, current.raw, current.etag, now)
            except Exception:
                log.exception("Couldn't save the slate")

    # --- the markets brain ----------------------------------------------------------------------------------

    def markets_for(self, league: str, odds_id: Optional[str] = None, espn_id: Optional[str] = None
                    ) -> Optional[Dict[str, Any]]:
        """A game's market detail ({"blend", "sources", "links"}) as last worked out, by the odds feed's or ESPN's
        id; None when the slate has none. Never fetches anything."""
        index = self._markets_index
        for kind, value in (("odds", odds_id), ("espn", espn_id)):
            if value:
                gid = index.get((kind, league, str(value)))
                if gid is not None and gid in self._markets:
                    return self._markets[gid]
        return None

    def _sources(self, game: Mapping[str, Any], matched: Sequence[Mapping[str, Any]], now: float
                 ) -> List[Dict[str, Any]]:
        """Every opinion on one game, oriented to its home and away (see the module docstring)."""
        league = game["league"]
        sources: List[Optional[Dict[str, Any]]] = []
        books = game.get("_books")
        if books:
            sources += book_sources(books, game.get("odds_as_of"), now)
        for event in matched:
            sources.append(market_source(event, now))
        espn_id = game.get("espn_id")
        predictor = self._predictors.get((league, str(espn_id))) if espn_id else None
        if predictor:
            oriented = _flip(predictor) if game.get("_swapped") else predictor
            sources.append(rating_source({**oriented, "detail": "ESPN matchup predictor"}, now, name="ESPN",
                                         stale_after=PREDICTOR_STALE))
        if self.ratings is not None:
            try:
                surface = {"surface": game["_surface"]} if game.get("_surface") else {}
                rating = self.ratings.game_prob(league, game["home"], game["away"], neutral=bool(game.get("_neutral")),
                                                fetch=False, **surface)
            except Exception as exc:
                self._log_failure(f"{league} ratings", exc, detail=False)
                rating = None
            sources.append(rating_source(rating, now))
        model = game.get("_model")
        if model and model[2] is not None:
            sources.append(model_source(model[2], model[3], now))
        return [source for source in sources if source]

    def _venue_events(self, league: str) -> List[Dict[str, Any]]:
        """The league's cached Kalshi and Polymarket events (never fetched here)."""
        events: List[Dict[str, Any]] = []
        for source in (self.kalshi, self.polymarket):
            if source is None or not _covers(source, league):
                continue
            try:
                events += [event for event in source.cached_events(league) or [] if isinstance(event, dict)]
            except Exception as exc:
                self._log_failure(f"{league} prediction markets", exc, detail=False)
        return events

    def _pm_matches(self, league: str, games: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """{slate id: the venues' events for that game, turned to the game's home and away}, each venue matched on
        its own (data/markets/aliases.py). The pairing is worked out again only when the venues' events or the
        games change; the prices always come from the events as cached now."""
        from .markets.aliases import match_events, orient  # the alias tables load once, on first use
        events = self._venue_events(league)
        if not events:
            self._matches.pop(league, None)
            return {}
        signature = (tuple((e.get("source"), e.get("id"), e.get("start"), e.get("home"), e.get("away")) for e in events),
                     tuple((g["id"], g.get("start"), g.get("_time_ok", True)) for g in games))
        cached = self._matches.get(league)
        if cached and cached[0] == signature:
            pairs = cached[1]
        else:
            # A start ESPN marks as not valid is a placeholder (midnight Eastern): only its day is compared
            plain = [{"id": g["id"], "home": g["home"], "away": g["away"], "start": g.get("start")}
                     if g.get("_time_ok", True) or g.get("_ts") is None else
                     {"id": g["id"], "home": g["home"], "away": g["away"], "start": None,
                      "date": datetime.datetime.fromtimestamp(g["_ts"], ESPN_DAY_TZ).date().isoformat()}
                     for g in games]
            pairs = {}
            for venue in sorted({str(e.get("source")) for e in events}):
                try:
                    found = match_events(league, plain, [e for e in events if str(e.get("source")) == venue])
                except Exception as exc:
                    self._log_failure(f"{league} {venue} matching", exc, detail=False)
                    continue
                for gid, event in found.items():
                    pairs.setdefault(gid, []).append((venue, event.get("id"), bool(event.get("swapped"))))
            self._matches[league] = (signature, pairs)
        current = {(str(e.get("source")), e.get("id")): e for e in events}
        matches: Dict[str, List[Dict[str, Any]]] = {}
        for gid, keys in pairs.items():
            for venue, event_id, swapped in keys:
                event = current.get((venue, event_id))
                if event is not None:
                    matches.setdefault(gid, []).append(orient(event, swapped) if swapped else event)
        return matches

    def _saved_detail(self, game: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """A started game's last market detail before its start, from consensus_log (after a restart); looked up
        once per game."""
        if self.store is None or not hasattr(self.store, "consensus_row"):
            return None
        if game["id"] in self._saved_seen:
            return self._saved_seen[game["id"]]
        odds_id, espn_id = game.get("odds_id"), game.get("espn_id")
        try:
            row = self.store.consensus_row(game["league"], event_id=odds_id, espn_id=espn_id)
        except Exception as exc:
            self._log_failure("saved consensus", exc)
            return None
        detail = None
        if row and row.get("blend"):
            sources = row.get("sources") or []
            detail = {"blend": row["blend"], "sources": sources,
                      "links": [{"name": s["name"], "url": s["url"]} for s in sources if s.get("url")]}
        self._saved_seen[game["id"]] = detail
        return detail

    def _apply_markets(self, games: List[Dict[str, Any]], now: float) -> None:
        """Work out every game's market detail, set its ``chance`` (the blend) and ``mkt``, and log the consensus
        of games still to start. A started game keeps its last detail from before the start."""
        details: Dict[str, Dict[str, Any]] = {}
        index: Dict[Tuple[str, str, str], str] = {}
        by_league: Dict[str, List[Dict[str, Any]]] = {}
        for game in games:
            by_league.setdefault(game["league"], []).append(game)
        rows = []
        for league, league_games in by_league.items():
            params = brain_params(league, self.store)
            try:
                upcoming = [g for g in league_games if g["status"] == "pre"]
                matches = self._pm_matches(league, upcoming) if (self.kalshi or self.polymarket) and upcoming else {}
            except Exception as exc:
                self._log_failure(f"{league} prediction markets", exc, detail=False)
                matches = {}
            for game in league_games:
                gid = game["id"]
                detail = None
                # A start ESPN marks as not valid (a match whose order of play isn't out) is only its day: such a game
                # has started when ESPN says so
                start = game.get("_ts") if game.get("_time_ok", True) or game["status"] != "pre" else None
                started = game["status"] != "pre" or (start is not None and start <= now)
                if started:  # its "Markets say" stays the last one before the start, as logged
                    detail = self._markets.get(gid) or self._saved_detail(game)
                if detail is None and game["status"] != "postponed":
                    try:
                        sources = self._sources(game, matches.get(gid, []), now)
                        if not started and any(s.get("kind") in PRICE_KINDS and after_start(s, start) for s in sources):
                            # A price from the start or later: the game has started since this round read the clock
                            started = True
                            detail = self._markets.get(gid) or self._saved_detail(game)
                        if detail is None:  # only opinions from before the start (market_detail)
                            detail = market_detail(league, sources, start, now, neutral=bool(game.get("_neutral")),
                                                   params=params)
                    except Exception as exc:
                        self._log_failure(f"{league} markets", exc, detail=False)
                        detail = None
                if not detail:
                    continue
                details[gid] = detail
                for kind, value in (("odds", game.get("odds_id")), ("espn", game.get("espn_id"))):
                    if value:
                        index[(kind, league, str(value))] = gid
                chance, mkt = chance_of(detail), compact(detail)
                if chance:
                    # Keep the bookmakers' margin-free chance: Today's "Mkt" figure, edge and AI-vs-market sort
                    book = game.get("chance")  # odds_view's chance, src "market"
                    if isinstance(book, dict) and book.get("src") == "market":
                        game["fair"] = {k: book[k] for k in ("home", "away", "draw") if book.get(k) is not None}
                    game["chance"] = chance
                if mkt:
                    game["mkt"] = mkt
                if not started and detail.get("blend"):
                    rows.append(self._consensus_row(game, detail, params.version))
        self._markets = details
        self._markets_index = index
        current = {game["id"] for game in games}
        self._saved_seen = {gid: found for gid, found in self._saved_seen.items() if gid in current}
        if rows and self.log_consensus:
            # Read the clock once more: the store writes nothing for a game whose start has passed by now
            self._log_consensus(rows, max(now, self._now()))

    @staticmethod
    def _consensus_row(game: Mapping[str, Any], detail: Mapping[str, Any], version: int = 0) -> Dict[str, Any]:
        odds_id, espn_id = game.get("odds_id"), game.get("espn_id")
        name = lambda team: team.get("name") or team.get("short") or team.get("abbr") or "?"  # noqa: E731
        return {"league": game["league"], "game_key": f"espn:{espn_id}" if espn_id else f"odds:{odds_id}",
                "event_id": odds_id, "espn_id": espn_id, "swapped": bool(game.get("_swapped")),
                "home": name(game["home"]), "away": name(game["away"]), "start": game["start"],
                "sources": detail["sources"], "blend": detail["blend"], "params_version": int(version or 0)}

    def _log_consensus(self, rows: List[Dict[str, Any]], now: float) -> None:
        """Write the rows whose sources or blend changed since they were last written (one transaction), plus each
        game's one-day-before snapshot (kind "t24") on its first round 12 to 24 hours before the start."""
        if self.store is None or not hasattr(self.store, "log_consensus"):
            return
        changed, signatures, snapshots = [], {}, []
        for row in rows:
            key = (row["league"], row["game_key"])
            signature = hashlib.sha1(json.dumps([row["sources"], row["blend"], row["start"], row["event_id"],
                                                 row.get("params_version") or 0],
                                                sort_keys=True, default=str).encode()).hexdigest()
            if self._logged_sig.get(key) != signature:
                changed.append(row)
                signatures[key] = signature
            start = _parse_time(row.get("start"))
            if start is not None and key not in self._t24_logged and \
                    T24_WINDOW[0] <= start - now <= T24_WINDOW[1]:
                snapshots.append({**row, "kind": "t24"})
        if not changed and not snapshots:
            return
        try:
            self.store.log_consensus(changed + snapshots, now)
        except Exception as exc:
            self._log_failure("consensus log", exc)
            return
        self._logged_sig.update(signatures)
        self._t24_logged.update((row["league"], row["game_key"]) for row in snapshots)
        if len(self._t24_logged) > 20000:
            self._t24_logged = set(list(self._t24_logged)[-10000:])
        if len(self._logged_sig) > 20000:  # games long gone
            self._logged_sig = dict(list(self._logged_sig.items())[-10000:])

    def _schedule_markets(self, games: Sequence[Mapping[str, Any]], now: float) -> None:
        """Start a markets worker round when one is due: the leagues with games to come or on, and ESPN's
        predictor for the week's games that are due (hourly within a day of the start, every 6 hours before),
        soonest first, PREDICTOR_PER_ROUND a round."""
        if self.polymarket is None and self.kalshi is None and self.ratings is None:
            return
        if self._markets_busy or now < self._markets_next:
            return
        leagues = sorted({g["league"] for g in games if g["status"] in ("pre", "live")})
        # Kalshi only where a game is on or starts within two days (its anonymous rate limit is tight)
        kalshi_leagues = sorted({g["league"] for g in games if g["status"] == "live"
                                 or (g["status"] == "pre" and (g.get("_ts") or 0) - now < KALSHI_AHEAD)})
        predictor: List[Tuple[str, str]] = []
        if self.ratings is not None:
            from .markets.ratings import PREDICTOR_LEAGUES
            # Every game of the week, soonest first, but only those due (asked for over an hour ago when the game is
            # within a day, over 6 hours ago when further off, or never): each round's few fetches go where needed
            asked = self._predictor_asked
            due = sorted((g for g in games if g["status"] == "pre" and g.get("espn_id") and g["league"] in PREDICTOR_LEAGUES
                          and (g.get("_ts") or 0) > now
                          and now - asked.get((g["league"], str(g["espn_id"])), -math.inf)
                          >= (PREDICTOR_TTL if (g.get("_ts") or 0) - now < PREDICTOR_AHEAD else PREDICTOR_FAR_EVERY)),
                         key=lambda g: g.get("_ts") or 0)
            predictor = [(g["league"], str(g["espn_id"])) for g in due[:PREDICTOR_PER_ROUND]]
            for key in predictor:
                asked[key] = now
            if len(asked) > 4000:  # games long gone
                self._predictor_asked = dict(sorted(asked.items(), key=lambda item: item[1])[-2000:])
        self._markets_busy = True
        thread = threading.Thread(target=self._load_markets, args=(leagues, predictor, kalshi_leagues),
                                  name="slate-markets", daemon=True)
        self._markets_thread = thread
        try:
            thread.start()
        except RuntimeError:  # shutting down
            self._markets_busy = False

    def wait_markets(self, timeout: float = 30.0) -> None:
        """Wait for the markets worker's round in progress, if any (for tests and tools)."""
        thread = self._markets_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _load_markets(self, leagues: Sequence[str], predictor: Sequence[Tuple[str, str]],
                      kalshi_leagues: Optional[Sequence[str]] = None) -> None:
        """One markets worker round, in the background: each source refreshes whatever its TTL says is due (the
        sources never raise; failures keep their last good copy). Polymarket and the ratings first, then ESPN's
        predictor, then Kalshi last and only for ``kalshi_leagues`` (games within two days): a rate-limited Kalshi
        backs off for half a minute inline, so it must not hold up the others, and once it is resting
        (data/markets/kalshi.py) the round skips it. The slate is rebuilt when the others are done and again at
        the end."""
        kalshi_list = [league for league in (leagues if kalshi_leagues is None else kalshi_leagues)
                       if self.kalshi is not None and _covers(self.kalshi, league)]
        try:
            for league in leagues:
                if self._stop.is_set():
                    return
                if self.polymarket is not None and _covers(self.polymarket, league):
                    try:
                        self.polymarket.events(league)
                    except Exception as exc:
                        self._log_failure(f"Polymarket {league}", exc, detail=False)
                if self.ratings is not None:
                    try:
                        self.ratings.power(league)
                    except Exception as exc:
                        self._log_failure(f"ratings {league}", exc, detail=False)
            for league, espn_id in predictor:
                if self._stop.is_set():
                    return
                try:
                    value = self.ratings.espn_predictor(league, espn_id)
                except Exception as exc:
                    self._log_failure(f"ESPN predictor {league}", exc, detail=False)
                    continue
                if value:
                    self._predictors[(league, espn_id)] = dict(value)
            if len(self._predictors) > 2000:
                self._predictors = dict(list(self._predictors.items())[-1000:])
            if kalshi_list:
                self.wake()  # show the others' prices now, not after Kalshi's turn
            for league in kalshi_list:
                resting = getattr(self.kalshi, "resting", None)
                if self._stop.is_set() or (callable(resting) and resting()):
                    break
                try:
                    self.kalshi.events(league)
                except Exception as exc:
                    self._log_failure(f"Kalshi {league}", exc, detail=False)
        finally:
            self._markets_next = self._now() + self.markets_every
            self._markets_busy = False
            self.wake()

    # --- venue & conditions (v3 Stage 4) ------------------------------------------------------------------

    def _apply_context(self, games: List[Dict[str, Any]]) -> None:
        """Add each game's context badge codes (wind, rain, cold, hot, roof, alt, tz-*, rest-*, adj) from its computed
        entry to the row's badges, after the slate's own (live, neutral, b2b-*). Never fetches."""
        if self.context is None:
            return
        for game in games:
            try:
                codes = self.context.badges_for(game)
            except Exception as exc:
                self._log_failure("context badges", exc, detail=False)
                return
            if codes:
                badges = list(game.get("badges") or [])
                game["badges"] = badges + [code for code in codes if code not in badges]

    def _context_input(self, game: Mapping[str, Any]) -> Dict[str, Any]:
        """What the context worker reads of a game (a copy: the worker runs beside the next round)."""
        return {key: (dict(value) if isinstance(value, dict) else value) for key, value in game.items()
                if key not in ("_books", "_model", "_names", "badges")}

    def _schedule_context(self, games: Sequence[Mapping[str, Any]], now: float) -> None:
        """Start a context round in the background when one is due: ContextService.warm over the games still to
        play; the slate is rebuilt when a row's badges changed."""
        if self.context is None or self._context_busy or now < self._context_next:
            return
        snapshot = [self._context_input(game) for game in games if game.get("status") in ("pre", "live")]
        self._context_busy = True
        thread = threading.Thread(target=self._warm_context, args=(snapshot,), name="slate-context", daemon=True)
        self._context_thread = thread
        try:
            thread.start()
        except RuntimeError:  # shutting down
            self._context_busy = False

    def _warm_context(self, games: List[Dict[str, Any]]) -> None:
        again = False
        try:
            before = {game["id"]: self.context.badges_for(game) for game in games}
            self.context.warm(games)
            if any(self.context.badges_for(game) != before.get(game["id"]) for game in games):
                self.wake()
        except Exception:
            log.exception("Slate: the context round failed")
        finally:
            # A game asked for while this round ran: another round at once rather than a minute on
            pending = getattr(self.context, "pending", None)
            again = bool(pending()) if callable(pending) else False
            self._context_next = self._now() + (0.0 if again else self.context_every)
            self._context_busy = False
        if again:
            self.wake()

    def _prewarm_players(self, games: Sequence[Mapping[str, Any]], now: float) -> None:
        """The player boards' game-day prewarm: hand the games to be played to PlayerBoard.prewarm, which starts the
        boards of leagues with a game in the next 12 hours (in the background: it returns at once) and the props and
        injuries refreshes that are due. At most every ``players_every`` seconds; never fetches here."""
        if now < self._players_next:
            return
        board = self.player_board if self.player_board is not None else getattr(self.service, "player_board", None)
        if board is None:
            # The dashboard's Lab makes the process's board on first use (data/player_board.board_for): asked here, the
            # prewarm runs from the first round rather than from the first scan
            try:
                board = getattr(getattr(self.service, "lab", None), "player_board", None)
            except Exception:
                log.exception("Slate: the player boards couldn't start")
                board = None
        prewarm = getattr(board, "prewarm", None)
        if not callable(prewarm):
            return
        self._players_next = now + self.players_every
        rows = [{"league": game.get("league"), "start": game.get("start"), "status": game.get("status")}
                for game in games if game.get("status") == "pre"]
        try:
            prewarm(rows)
        except Exception:
            log.exception("Slate: the player boards' prewarm failed")

    def request_context(self) -> None:
        """A reader asked for a game whose context block isn't built yet (ContextService.for_game queued it): run the
        context round at a slate round now, instead of up to a minute (or an idle slate's ten) later, so the page's one
        re-check finds it. At most once every CONTEXT_SOON seconds; a round in progress runs again when it ends."""
        if self.context is None:
            return
        now = self._now()
        with self._lock:
            if now - self._context_asked < CONTEXT_SOON:
                return
            self._context_asked = now
            self._context_next = min(self._context_next, now)
        self.wake()

    def wait_context(self, timeout: float = 30.0) -> None:
        """Wait for the context round in progress, if any (for tests and tools)."""
        thread = self._context_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def scoreboards(self, league: str, days: Iterable[datetime.date], fetch: bool = False
                    ) -> Dict[datetime.date, List[Dict[str, Any]]]:
        """The league's games on each of ``days`` (ESPN days), parsed as the slate parses them: the boards the slate
        holds, else past boards kept from before (they don't change), else, with ``fetch`` (the context worker's
        background only), the board read through the ESPN client's cache and kept once it is past."""
        if espn_path(league) is None:
            return {}
        today = datetime.datetime.fromtimestamp(self._now(), ESPN_DAY_TZ).date()
        out: Dict[datetime.date, List[Dict[str, Any]]] = {}
        for day in days:
            board = self._boards.get((league, day))
            if board is not None and board.fetched:
                out[day] = list(board.games)
                continue
            with self._past_lock:
                kept = self._past.get((league, day))
            if kept is not None:
                out[day] = kept
                continue
            if (league, day) in self._fallback:
                out[day] = list(self._fallback.get((league, day)) or [])
                continue
            if not fetch or day > today:
                continue
            reader = getattr(self.espn, "scoreboard", None) or getattr(self.espn, "fetch_scoreboard", None)
            try:
                games = parse_scoreboard(league, reader(league, day))
            except Exception as exc:
                self._log_failure(f"{league} past scoreboard", exc)
                continue
            out[day] = games
            if day < today - datetime.timedelta(days=1):
                with self._past_lock:
                    self._past[(league, day)] = games
                    while len(self._past) > PAST_KEPT:
                        self._past.popitem(last=False)
        return out

    # --- fresh odds on request ------------------------------------------------------------------------

    def request_odds(self, league: str) -> Dict[str, Any]:
        """Ask for a fresh odds board for one league, loaded in the background: {"queued", "reason", "code"}
        (reason in words; code one of queued, loading, fresh, throttled, no_games, reserve, daily_cap,
        not_in_season, no_listing, error).

        Runs only when the league is in season at The Odds API, its saved odds are older than the odds
        service's 2 hours (or missing), it hasn't been asked for in that time (by anyone, across restarts),
        ESPN doesn't show it without games, the quota stays at or above the odds service's reserve after this
        board and every board already on its way, and fewer than ``odds_per_day`` boards were loaded this way
        in the last 24 hours. The slate updates when it arrives.
        """
        now = self._now()
        ttl = self._odds_ttl()
        active = self._active(self._season)
        info = self._league_info.get(league)
        if active is None:
            return _answer(False, "no_listing", "The list of leagues in season isn't available right now.")
        if league not in active:
            return _answer(False, "not_in_season", "No odds market for this league right now.")
        fetched = self._odds_times().get(league)
        if fetched is not None and now - fetched < ttl:
            return _answer(False, "fresh", "The odds are up to date.")
        with self._lock:
            if league in self._loading:
                return _answer(True, "loading", "The latest odds are on their way.")
            last = self._requested.get(league)
            if last is not None and now - last < ttl:
                return _answer(False, "throttled", "Asked for less than 2 hours ago.")
            if info and info.get("espn") and not info.get("upcoming"):
                return _answer(False, "no_games", "No games coming up in this league.")
            # Boards already queued haven't been paid for yet: count them too (the app asks for several at once)
            if not self._affordable(boards=len(self._loading) + 1):
                return _answer(False, "reserve", "The month's odds allowance is down to its reserve.")
            self._loads = [t for t in self._loads if now - t < 24 * 3600]
            if len(self._loads) >= self.odds_per_day:
                return _answer(False, "daily_cap", "Today's odds loads are used up; saved odds stay on show.")
            if self.store is not None and hasattr(self.store, "try_lease"):
                try:
                    if not self.store.try_lease(f"slate-odds:{league}", ttl):
                        return _answer(False, "throttled", "Asked for less than 2 hours ago.")
                except Exception:
                    log.exception("Couldn't take the odds lease for %s", league)
                    return _answer(False, "error", "Couldn't ask for odds right now.")
            self._requested[league] = now
            self._loads.append(now)
            self._loading.add(league)
        self._odds_pool.submit(self._load_odds, league)
        return _answer(True, "queued", "Loading the latest odds.")

    def _fetch_cost(self) -> int:
        """Credits one league's board costs: a credit per market per region (ten listed bookmakers count as one
        region: OddsService.board_cost), else per odds region asked for ("us" or "us,uk", say)."""
        cost = getattr(self.odds, "board_cost", None)
        if isinstance(cost, int) and not isinstance(cost, bool) and cost > 0:
            return cost
        regions = getattr(self.odds, "regions", None)
        count = len([r for r in regions.split(",") if r.strip()]) if isinstance(regions, str) else 1
        return FETCH_COST * max(1, count)

    def _affordable(self, boards: int = 1) -> bool:
        """Whether ``boards`` more boards keep the month's quota (as last reported) at or above the reserve."""
        try:
            quota = getattr(self.odds, "quota", None) or {}
        except Exception:
            log.exception("Couldn't read the odds quota")
            return False
        remaining = quota.get("remaining") if isinstance(quota, Mapping) else None
        reserve = getattr(self.odds, "reserve", 0) or 0
        return not isinstance(remaining, int) or remaining - self._fetch_cost() * boards >= reserve

    def _load_odds(self, league: str) -> None:
        try:
            # Loads run one at a time, so the quota read here already counts every board loaded before this one
            if not self._affordable():
                log.info("Slate: not loading %s odds: the month's allowance is down to its reserve", league)
            elif self.service is not None and league in self.ai_leagues:
                self.service.games(league, refresh=True)  # the model's chances are logged with the board
            else:
                self.odds.league_games(league, refresh=True)
        except Exception as exc:
            # Only the type: a connection error's message can quote the request URL, which holds the API key
            log.warning("Slate: loading %s odds failed (%s)", league, type(exc).__name__)
        finally:
            with self._lock:
                self._loading.discard(league)
            self.wake()
