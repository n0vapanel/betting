"""Polymarket's game prices (moneyline, plus the main spread and total), from its public Gamma API: no key.

Endpoints (live-checked 2026-10-27; no authentication; Gamma allows 300 /markets requests per 10 s):

  GET https://gamma-api.polymarket.com/markets
        ?sports_market_types=moneyline&tag_id={league tag}&closed=false
        &end_date_min={now - 1 day}&end_date_max={now + 15 days}&limit=100&offset={0, 100, ...}
      Every open moneyline market of the league. The end-date window drops the season-long futures (season
      series, tournament winners) that share the moneyline type. About 6 KB a market, ~150 KB for an NBA week.
  GET .../markets?sports_market_types=spreads&sports_market_types=totals&tag_id=...&closed=false
        &liquidity_num_min=1000&end_date_min={now - 1 day}&end_date_max={now + 2 days (5 for football and
        soccer)}&limit=100&offset=...
      The full-game spread and total lines of the coming days with at least $1,000 of liquidity (thinner
      lines would only be shown as thin; a game has 25-35 of each, so the filter keeps the download small).
      Fetched only with ``lines=True``: nothing uses a venue's lines before Stage 6, and they were more than half
      of the bytes (12 of 23 MB for one pass over 16 leagues).
  (GET /events?series_id=... returns every market of a game, ~480 with the player props, 16 MB for an NBA
  week, so it is not used; GET /sports lists each league's code, series, tags and home/away ordering, from
  which the table below was taken.)

Fields used (per market): slug, question, sportsMarketType, line, outcomes and outcomePrices (JSON strings),
bestBid, bestAsk, lastTradePrice (all for the first outcome's token), liquidityNum and volumeNum (US dollars),
gameStartTime ("2026-10-27 23:30:00+00"), groupItemTitle, active, closed, and the nested events[0]: slug,
title, startTime, eventDate, live, ended.

Shapes:
- A game's event slug is "{code}-{a}-{b}-{YYYY-MM-DD}". Two-way games (US sports, tennis, UFC) have one
  moneyline market whose two outcomes are the teams in slug order; three-way soccer has three Yes/No markets
  "...-{a}", "...-draw", "...-{b}". Leagues whose /sports ordering is "away" (NBA, NFL, NHL, MLB, college,
  WNBA, CFL) list the away team first; the others list the home team first (tennis and UFC have no home: the
  first-listed player is "home", as the app does for fights).
- Spread slugs are "{game slug}-spread-{home|away}-{n}pt{m}": the named side gives ``line`` points and is
  outcomes[0]. Totals are "{game slug}-total-{n}pt{m}" with outcomes Over/Under. (Soccer lines hang off a
  separate "{game slug}-more-markets" event, so the game is read from the market's slug.) The main line is
  the one closest to 50/50 that isn't thin.

A two-way market's chances are one token's price and its complement, so they sum to 1. Soccer's three
separate markets are normalised to sum to 1 (common.normalise). bid/ask/last stay as quoted.

Leagues (Odds API/app key -> Polymarket code, tag id). Tags were checked live on 2026-10-27 (the market slugs
came back with the league's code) except where marked "unconfirmed": those leagues had no open games that day,
and their tag is /sports' primaryTagId. Note Serie A and the Champions League are not on their primaryTagId.
Polymarket covers every league asked for: NBA, NFL, NHL, MLB, college football and basketball, EPL, La Liga,
Serie A, Bundesliga, Ligue 1, MLS, the Champions/Europa/Conference Leagues, ATP, WTA and UFC. Tennis matches
carry only the shared "tennis" tag (864; the ATP and WTA tags hold just tournament-winner futures), so one call
serves both tours. It asks only for markets with at least $1,000 of liquidity: most of the tag is ITF and doubles
(which the slug pattern leaves out), and a thinner match market would be shown as thin, never blended.
A fetch that fills all MAX_PAGES pages may have been cut short: that is logged (at most hourly per tag). Tennis and UFC
have no spreads/totals of the plain types (only set/game handicaps), so they get moneylines only.
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..cache import ttl_for
from .aliases import team_key
from .common import (HttpError, Keeper, as_list, complement, http_get, iso, make_quote, normalise, num,
                     parse_time, pick_line, rounded, text)

log = logging.getLogger(__name__)

BASE = "https://gamma-api.polymarket.com"
PAGE = 100  # Gamma's page size
MAX_PAGES = 8  # at most 800 markets per league and kind
TRUNCATED_LOG_EVERY = 3600  # a fetch cut short at MAX_PAGES is logged at most this often per tag
# From the shared cache's policy table (data/cache.py's ttl_for), the one place these times live:
TTL = ttl_for("markets").ttl  # prediction markets: 5 minutes ...
LIVE_TTL = ttl_for("markets", "live").ttl  # ... and 1 minute while a game of the league is on
LINES_TTL = ttl_for("market_lines").ttl  # spreads and totals move less and cost more: every 15 minutes
LIVE_FOR = 4 * 3600  # a game counts as possibly live from its start for this long when the venue doesn't say
DROP_AFTER = 12 * 3600  # an open market for a game that started longer ago than this is left behind

# app league key -> (Polymarket code, tag id, first-listed side)
LEAGUES: Dict[str, Tuple[str, int, str]] = {
    "basketball_nba": ("nba", 745, "away"),
    "basketball_wnba": ("wnba", 100254, "away"),
    "basketball_ncaab": ("cbb", 101178, "away"),  # unconfirmed (season starts in November)
    "basketball_wncaab": ("cwbb", 102003, "home"),  # unconfirmed
    "basketball_euroleague": ("euroleague", 102669, "home"),
    "basketball_nbl": ("bknbl", 103096, "home"),
    "americanfootball_nfl": ("nfl", 450, "away"),
    "americanfootball_ncaaf": ("cfb", 100351, "away"),
    "americanfootball_cfl": ("cfl", 105200, "away"),
    "icehockey_nhl": ("nhl", 899, "away"),
    "baseball_mlb": ("mlb", 100381, "away"),
    "soccer_epl": ("epl", 306, "home"),
    "soccer_efl_champ": ("elc", 102643, "home"),
    "soccer_england_league1": ("el1", 104319, "home"),
    "soccer_england_league2": ("el2", 104320, "home"),
    "soccer_england_efl_cup": ("efl", 102525, "home"),
    "soccer_fa_cup": ("efa", 101807, "home"),
    "soccer_spl": ("scop", 102872, "home"),
    "soccer_spain_la_liga": ("lal", 780, "home"),
    "soccer_spain_segunda_division": ("es2", 102866, "home"),
    "soccer_spain_copa_del_rey": ("cdr", 101783, "home"),
    "soccer_italy_serie_a": ("sea", 101962, "home"),
    "soccer_italy_serie_b": ("itsb", 102870, "home"),
    "soccer_italy_coppa_italia": ("itc", 102008, "home"),  # unconfirmed
    "soccer_germany_bundesliga": ("bun", 1494, "home"),
    "soccer_germany_bundesliga2": ("bl2", 102864, "home"),
    "soccer_germany_dfb_pokal": ("dfb", 102154, "home"),
    "soccer_france_ligue_one": ("fl1", 102070, "home"),
    "soccer_france_ligue_two": ("fr2", 102871, "home"),
    "soccer_netherlands_eredivisie": ("ere", 101735, "home"),
    "soccer_portugal_primeira_liga": ("por", 101772, "home"),
    "soccer_turkey_super_league": ("tur", 102564, "home"),
    "soccer_saudi_arabia_pro_league": ("spl", 102650, "home"),
    "soccer_japan_j_league": ("jap", 102649, "home"),
    "soccer_australia_aleague": ("aus", 102765, "home"),
    "soccer_mexico_ligamx": ("mex", 102448, "home"),
    "soccer_brazil_campeonato": ("bra", 102648, "home"),
    "soccer_brazil_serie_b": ("bra2", 104387, "home"),
    "soccer_argentina_primera_division": ("arg", 102561, "home"),
    "soccer_conmebol_copa_libertadores": ("lib", 102562, "home"),  # unconfirmed
    "soccer_conmebol_copa_sudamericana": ("sud", 102563, "home"),  # unconfirmed
    "soccer_uefa_champs_league": ("ucl", 100977, "home"),
    "soccer_uefa_europa_league": ("uel", 101787, "home"),
    "soccer_uefa_europa_conference_league": ("col", 102763, "home"),
    "soccer_uefa_nations_league": ("unl", 100816, "home"),  # unconfirmed
    "soccer_usa_mls": ("mls", 100100, "home"),
    "tennis_atp": ("atp", 864, "home"),  # matches carry only the "tennis" tag, not ATP's own 101232
    "tennis_wta": ("wta", 864, "home"),
    "mma_mixed_martial_arts": ("ufc", 279, "home"),
}
LINE_DAYS = 2  # spreads/totals of the next two days (a busy NBA week has ~40 lines a game) ...
LINE_DAYS_WEEKLY = 5  # ... or five for the weekly sports, whose games are days apart
WEEKLY = ("americanfootball_", "soccer_")
TENNIS_TAG = 864
TENNIS_DAYS = 9  # a tennis match's market ends about a week after it is played
TENNIS_LIQUIDITY = 1000  # tennis moneylines with less liquidity (mostly ITF) aren't downloaded: they'd be thin
NO_LINES = {"tennis_atp", "tennis_wta", "mma_mixed_martial_arts"}  # no plain spreads/totals (only set/game handicaps)
# The Odds API keys each tennis tournament on its own (tennis_atp_vienna_open): they share the tour's markets
PREFIXES = (("tennis_atp_", "tennis_atp"), ("tennis_wta_", "tennis_wta"))


def league_entry(league: str) -> Optional[Tuple[str, int, str]]:
    if league in LEAGUES:
        return LEAGUES[league]
    for prefix, tour in PREFIXES:
        if league.startswith(prefix):
            return LEAGUES[tour]
    return None


def _event_of(market: Mapping[str, Any]) -> Dict[str, Any]:
    events = market.get("events")
    return events[0] if isinstance(events, list) and events and isinstance(events[0], dict) else {}


def _quote(market: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The first outcome's quote (Gamma's bestBid/bestAsk/lastTradePrice are that token's)."""
    prices = as_list(market.get("outcomePrices"))
    return make_quote(market.get("bestBid"), market.get("bestAsk"), market.get("lastTradePrice"),
                      market.get("liquidityNum", market.get("liquidity")),
                      market.get("volumeNum", market.get("volume")), prices[0] if prices else None)


def _start(market: Mapping[str, Any], event: Mapping[str, Any]) -> Optional[float]:
    return parse_time(market.get("gameStartTime")) or parse_time(event.get("startTime"))


def parse_moneylines(league: str, markets: List[Mapping[str, Any]], fetched_at: float) -> List[Dict[str, Any]]:
    """MarketEvents (h2h only) from a page of Gamma moneyline markets. Futures, closed and ended markets, other
    leagues' markets and games that started more than 12 hours ago are left out."""
    entry = league_entry(league)
    if entry is None:
        return []
    code, _tag, first = entry
    pattern = re.compile(rf"^{re.escape(code)}-([a-z0-9]+)-([a-z0-9]+)-(\d{{4}}-\d{{2}}-\d{{2}})$")
    games: Dict[str, Dict[str, Any]] = {}
    for market in markets:
        if not isinstance(market, dict) or market.get("closed") or market.get("active") is False:
            continue
        event = _event_of(market)
        if event.get("ended") or event.get("closed"):
            continue
        slug = str(event.get("slug") or "")
        match = pattern.match(slug)
        if not match:
            continue
        start = _start(market, event)
        if start is not None and start < fetched_at - DROP_AFTER:
            continue
        game = games.setdefault(slug, {"a": match.group(1), "b": match.group(2), "date": match.group(3),
                                       "start": start, "event": event, "parts": {}})
        market_slug = str(market.get("slug") or "")
        outcomes = [str(o) for o in as_list(market.get("outcomes"))]
        quote = _quote(market)
        if quote is None:
            continue
        if market_slug == slug and len(outcomes) == 2:  # two-way: one market, both teams
            game["parts"]["a"] = (outcomes[0], quote)
            game["parts"]["b"] = (outcomes[1], complement(quote))
        elif market_slug.startswith(slug + "-") and outcomes[:1] == ["Yes"]:  # three-way soccer
            suffix = market_slug[len(slug) + 1:]
            name = text(market.get("groupItemTitle"))
            if suffix == "draw":
                game["parts"]["draw"] = ("Draw", quote)
            elif suffix in (game["a"], game["b"]):
                game["parts"]["a" if suffix == game["a"] else "b"] = (name or suffix, quote)
    events = []
    for slug, game in games.items():
        parts = game["parts"]
        if "a" not in parts or "b" not in parts:
            continue
        three_way = "draw" in parts
        quotes = {"a": parts["a"][1], "b": parts["b"][1]}
        if three_way:
            quotes["draw"] = parts["draw"][1]
        quotes = normalise(quotes) if three_way else quotes
        a_side, b_side = ("away", "home") if first == "away" else ("home", "away")
        h2h = {a_side: rounded(quotes["a"]), b_side: rounded(quotes["b"])}
        if three_way:
            h2h["draw"] = rounded(quotes["draw"])
        names = {a_side: (text(parts["a"][0], 60), game["a"]), b_side: (text(parts["b"][0], 60), game["b"])}
        event = game["event"]
        events.append({
            "source": "polymarket",
            "league": league,
            "id": slug,
            "home": names["home"][0],
            "away": names["away"][0],
            "home_abbr": names["home"][1],
            "away_abbr": names["away"][1],
            "home_key": team_key(league, names["home"][0]) or team_key(league, names["home"][1]),
            "away_key": team_key(league, names["away"][0]) or team_key(league, names["away"][1]),
            "start": iso(game["start"]),
            "date": game["date"],
            "live": bool(event.get("live")),
            "url": f"https://polymarket.com/event/{slug}",
            "as_of": iso(fetched_at),
            "markets": {"h2h": h2h},
        })
    events.sort(key=lambda e: (e["start"] or "", e["id"]))
    return events


def parse_lines(league: str, markets: List[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{game slug: {"spreads": {...}, "totals": {...}}} from Gamma spread/total markets: the main line of each
    (closest to 50/50, not thin first). The spread's point is the home side's (-3.5 = home gives 3.5). The game
    is read from the market's own slug: soccer lines sit under a separate "...-more-markets" event."""
    entry = league_entry(league)
    if entry is None:
        return {}
    code = entry[0]
    pattern = re.compile(rf"^({re.escape(code)}-[a-z0-9]+-[a-z0-9]+-\d{{4}}-\d{{2}}-\d{{2}})-"
                         r"(?:spread-(home|away)-\d+pt\d+|total-\d+pt\d+)$")
    spreads: Dict[str, List[Tuple[float, Dict[str, Any], Dict[str, Any], Optional[float]]]] = {}
    totals: Dict[str, List[Tuple[float, Dict[str, Any], Dict[str, Any], Optional[float]]]] = {}
    for market in markets:
        if not isinstance(market, dict) or market.get("closed") or market.get("active") is False:
            continue
        if _event_of(market).get("ended"):
            continue
        match = pattern.match(str(market.get("slug") or ""))
        line = num(market.get("line"))
        quote = _quote(market)
        if not match or line is None or quote is None:
            continue
        slug, side = match.group(1), match.group(2)
        liquidity = quote.get("liquidity")
        kind = market.get("sportsMarketType")
        if kind == "spreads" and side:
            if side == "home":  # outcomes[0] is the home side giving |line| points
                point, home, away = line, quote, complement(quote)
            else:
                point, home, away = -line, complement(quote), quote
            spreads.setdefault(slug, []).append((point, home, away, liquidity))
        elif kind == "totals" and not side:
            outcomes = [str(o).lower() for o in as_list(market.get("outcomes"))]
            over, under = (quote, complement(quote)) if outcomes[:1] == ["over"] else (complement(quote), quote)
            totals.setdefault(slug, []).append((line, over, under, liquidity))
    lines: Dict[str, Dict[str, Any]] = {}
    for slug, candidates in spreads.items():
        best = pick_line(candidates)
        if best:
            lines.setdefault(slug, {})["spreads"] = {"home": rounded(best[1]), "away": rounded(best[2]),
                                                     "point": best[0]}
    for slug, candidates in totals.items():
        best = pick_line(candidates)
        if best:
            lines.setdefault(slug, {})["totals"] = {"over": rounded(best[1]), "under": rounded(best[2]),
                                                    "point": best[0]}
    return lines


class Polymarket:
    """Polymarket game prices per league, as MarketEvents (see data/markets/__init__.py).

    ``events(league)`` refreshes at most every 5 minutes (every minute while one of the league's games is on;
    lines every 15 minutes) and is meant for the background worker; ``cached_events(league)`` never fetches
    and is what request handlers should read. Failures keep the last good copy; nothing raises.
    """

    source = "polymarket"

    def __init__(self, cache: Any = None, store: Any = None, *, fetch: Optional[Callable[..., Tuple[int, Any]]] = None,
                 clock: Optional[Callable[[], float]] = None, lines: bool = False) -> None:
        self.cache = cache  # accepted for the data clients' shared-cache convention; copies live in the keeper
        self.store = store
        self.clock = clock or time.time
        self._fetch = fetch or http_get
        self.lines = lines  # fetch spreads and totals too (no consumer before Stage 6)
        self.keeper = Keeper(self.source, store, self.clock)
        self._truncated: Dict[Any, float] = {}  # tag -> when a cut-short fetch was last logged

    @staticmethod
    def leagues() -> List[str]:
        return sorted(LEAGUES)

    def covers(self, league: str) -> bool:
        return league_entry(league) is not None

    def _canonical(self, league: str) -> Optional[str]:
        if league in LEAGUES:
            return league
        for prefix, tour in PREFIXES:
            if league.startswith(prefix):
                return tour
        return None

    def _get(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        markets: List[Dict[str, Any]] = []
        for page in range(MAX_PAGES):
            status, body = self._fetch(f"{BASE}/markets", {**params, "limit": PAGE, "offset": page * PAGE})
            if status != 200 or not isinstance(body, list):
                raise HttpError(status, "Polymarket")
            markets.extend(item for item in body if isinstance(item, dict))
            if len(body) < PAGE:
                break
        else:  # every page full: there may be more than MAX_PAGES pages
            key = (params.get("tag_id"), str(params.get("sports_market_types")))
            now = self.clock()
            if now - self._truncated.get(key, -math.inf) >= TRUNCATED_LOG_EVERY:
                self._truncated[key] = now
                log.warning("Polymarket: tag %s %s filled all %d pages; later markets are left out", key[0], key[1],
                            MAX_PAGES)
        return markets

    def _window(self, days_ahead: float) -> Dict[str, str]:
        now = self.clock()
        return {"end_date_min": iso(now - 86400), "end_date_max": iso(now + days_ahead * 86400)}

    @staticmethod
    def _sharing(tag: int) -> List[str]:
        """The leagues whose markets come with one tag (ATP and WTA share "tennis")."""
        return sorted(league for league, entry in LEAGUES.items() if entry[1] == tag)

    def fetch_moneylines(self, tag: int) -> Dict[str, List[Dict[str, Any]]]:
        """{league: MarketEvents with moneylines} for the leagues of one tag, straight from Gamma (raises on
        failure)."""
        tennis = tag == TENNIS_TAG
        markets = self._get({"sports_market_types": "moneyline", "tag_id": tag, "closed": "false",
                             **({"liquidity_num_min": TENNIS_LIQUIDITY} if tennis else {}),
                             **self._window(TENNIS_DAYS if tennis else 15)})
        now = self.clock()
        return {league: parse_moneylines(league, markets, now) for league in self._sharing(tag)}

    def fetch_lines(self, tag: int) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """{league: {event slug: spreads/totals}} for the next two days, straight from Gamma (raises on
        failure)."""
        weekly = any(league.startswith(WEEKLY) for league in self._sharing(tag))
        markets = self._get({"sports_market_types": ["spreads", "totals"], "tag_id": tag, "closed": "false",
                             "liquidity_num_min": 1000, **self._window(LINE_DAYS_WEEKLY if weekly else LINE_DAYS)})
        return {league: parse_lines(league, markets) for league in self._sharing(tag)}

    def _ttl(self, tag: int) -> float:
        if tag == TENNIS_TAG:
            return TTL  # some match is nearly always on, and the tour list is ~2.5 MB: every 5 minutes
        copy = self.keeper.peek(("h2h", tag))
        now = self.clock()
        for events in ((copy[1] if copy else None) or {}).values():
            for event in events or []:
                start = parse_time(event.get("start"))
                if event.get("live") or (start is not None and start <= now < start + LIVE_FOR):
                    return LIVE_TTL
        return TTL

    def events(self, league: str) -> List[Dict[str, Any]]:
        """The league's open games on Polymarket (refreshing when due). [] for a league it doesn't cover."""
        league = self._canonical(league) or ""
        if not league:
            return []
        tag = LEAGUES[league][1]
        h2h = self.keeper.get(("h2h", tag), self._ttl(tag), lambda: self.fetch_moneylines(tag), empty={}) or {}
        lines: Mapping[str, Any] = {}
        if self.lines and league not in NO_LINES and h2h.get(league):
            lines = self.keeper.get(("lines", tag), LINES_TTL, lambda: self.fetch_lines(tag), empty={}) or {}
        return _merge(h2h.get(league) or [], lines.get(league) or {})

    def cached_events(self, league: str) -> List[Dict[str, Any]]:
        """The last good copy, without fetching (for request handlers)."""
        league = self._canonical(league) or ""
        if not league:
            return []
        tag = LEAGUES[league][1]
        h2h = self.keeper.peek(("h2h", tag))
        lines = self.keeper.peek(("lines", tag))
        return _merge(((h2h[1] if h2h else None) or {}).get(league) or [],
                      ((lines[1] if lines else None) or {}).get(league) or {})


def _merge(events: List[Dict[str, Any]], lines: Mapping[str, Any]) -> List[Dict[str, Any]]:
    merged = []
    for event in events or []:
        extra = (lines or {}).get(event.get("id")) or {}
        merged.append({**event, "markets": {**event.get("markets", {}), **extra}})
    return merged
