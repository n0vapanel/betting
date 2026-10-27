"""Kalshi's game prices (winner, plus the main spread and total), from its public market data: no key.

Endpoint (live-checked 2026-10-27; "No authentication headers are required" for market data, per Kalshi's
quick start; api.elections.kalshi.com serves every Kalshi market, not only elections):

  GET https://api.elections.kalshi.com/trade-api/v2/events
        ?series_ticker={series}&status=open&with_nested_markets=true&with_milestones=true&limit=200[&cursor=]
      One call per series: every open event with its markets, plus "milestones" (Kalshi's game records) that
      carry the real start time ("start_date"), a title ("MIN Timberwolves at CLE Cavaliers") and the game's
      status ("details.status": not_started, scheduled, inprogress, ...). Pages continue while "cursor" is
      non-empty (college football needs two). Responses are 20 KB to 1.1 MB, cache-control max-age=15.

Rate: the documented limits are for authenticated keys; anonymous reads answered {"error": {"code":
"too_many_requests"}} (HTTP 429) after a few large calls even 1.5-3 s apart. So requests are paced at least
1 s apart (one pacer per process) and a 429 is retried after 2, 4, 8 and 16 s before giving up; the keeper
then keeps the last good copy and waits before trying again. A request that runs out of retries rests Kalshi for
REST seconds: every request in that time fails at once (resting() says so, and the slate's markets worker skips
Kalshi for the rest of its round), so one throttled minute costs one back-off, not one per league.

Spreads and totals (the KX...SPREAD and KX...TOTAL series) are fetched only with ``lines=True``: nothing uses a
venue's lines before Stage 6, and they are two more large calls per league every 15 minutes.

Fields used: event_ticker ("{SERIES}-{YY}{MON}{DD}[{HHMM}]{FIRST}{SECOND}", e.g. KXNBAGAME-26OCT26MINCLE,
KXMLBGAME-26OCT262000MILTB: the time is there when a day can hold two games), and per market: ticker (its last
part is the side: the team code, "TIE" for a soccer draw, or for lines the team code plus the strike, "CFC2"),
yes_sub_title (the team or player), yes_bid_dollars / yes_ask_dollars / last_price_dollars ("0.4200";
0 means none), volume_fp (contracts, each paying notional_value_dollars = $1), floor_strike (the line), status.
Kalshi reports no liquidity in dollars, so its quotes have liquidity None and are judged thin by spread alone.

``occurrence_datetime`` and ``expected_expiration_time`` are NOT the start (02:00Z for a 23:00Z tip): the start
comes only from the milestone; without one it stays None and matching uses the ticker's date.

Order: US leagues, CFL and EuroLeague tickers put the away team first (milestone titles say "X at Y"); soccer
puts the home team first ("Chelsea vs Manchester United" = CFCMUN, Chelsea at home); tennis and UFC list the
first player as "home", as the app does. A milestone title with " at " overrides the league default.

Winner markets of one game are mutually exclusive, so their midpoints are normalised to sum to 1 (two or three
contracts quote about 1.01 together). Spreads are "{team} wins by over/more than X points/goals" (one market
per team and strike; the main line is the one closest to 50/50 that isn't thin), totals "Over X".

Coverage (series found in GET /series?category=Sports, 4,053 series, on 2026-10-27; open events confirmed
for the leagues in season): NBA, WNBA, NFL, NHL, MLB, college football (FBS; FCS is KXNCAAFCSGAME, not used),
men's and women's college basketball (no open games before their November start), EPL, Championship, League
One, EFL Cup, FA Cup, Scottish Premiership, La Liga and LaLiga 2, Copa del Rey, Serie A and B, Coppa Italia,
Bundesliga and 2. Bundesliga, DFB-Pokal, Ligue 1 and 2, Eredivisie, Liga Portugal, Super Lig, Saudi Pro League,
J League, A-League, Liga MX, Brasileiro A and B, Argentina, Libertadores, Sudamericana, Champions/Europa/
Conference Leagues, Nations League, MLS, ATP, WTA, UFC, CFL, EuroLeague, Australia's NBL. Kalshi has no
League Two series (Polymarket has one). J League (home first, with TIE) and women's college basketball
("Villanova at Notre Dame", away first) were checked live; the A-League read was rate-limited that day, so its
order follows the soccer default.
"""

from __future__ import annotations

import datetime
import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..cache import ttl_for
from .aliases import team_key
from .common import (HttpError, Keeper, complement, http_get, iso, make_quote, normalise, num, parse_time,
                     pick_line, rounded, text)

log = logging.getLogger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
# From the shared cache's policy table (data/cache.py's ttl_for): 5 minutes, 1 while live; lines 15 minutes
TTL = ttl_for("markets").ttl
LIVE_TTL = ttl_for("markets", "live").ttl
LINES_TTL = ttl_for("market_lines").ttl
LIVE_FOR = 4 * 3600
MAX_PAGES = 5
MIN_INTERVAL = 1.0  # at most one request a second
BACKOFF = (2.0, 4.0, 8.0, 16.0)  # waits before each retry after "too_many_requests"
REST = 5 * 60  # after a request runs out of retries, Kalshi isn't asked again for this long
LIVE_STATUSES = {"inprogress", "in_progress", "live", "started"}

# app league key -> (winner series, spread series, total series, the side listed first)
LEAGUES: Dict[str, Tuple[str, Optional[str], Optional[str], str]] = {
    "basketball_nba": ("KXNBAGAME", "KXNBASPREAD", "KXNBATOTAL", "away"),
    "basketball_wnba": ("KXWNBAGAME", "KXWNBASPREAD", "KXWNBATOTAL", "away"),
    "basketball_ncaab": ("KXNCAAMBGAME", "KXNCAAMBSPREAD", "KXNCAAMBTOTAL", "away"),
    "basketball_wncaab": ("KXNCAAWBGAME", None, None, "away"),
    "basketball_euroleague": ("KXEUROLEAGUEGAME", None, None, "away"),
    "basketball_nbl": ("KXNBLGAME", None, None, "away"),
    "americanfootball_nfl": ("KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL", "away"),
    "americanfootball_ncaaf": ("KXNCAAFGAME", "KXNCAAFSPREAD", "KXNCAAFTOTAL", "away"),
    "americanfootball_cfl": ("KXCFLGAME", None, None, "away"),
    "icehockey_nhl": ("KXNHLGAME", "KXNHLSPREAD", "KXNHLTOTAL", "away"),
    "baseball_mlb": ("KXMLBGAME", "KXMLBSPREAD", "KXMLBTOTAL", "away"),
    "soccer_epl": ("KXEPLGAME", "KXEPLSPREAD", "KXEPLTOTAL", "home"),
    "soccer_efl_champ": ("KXEFLCHAMPIONSHIPGAME", "KXEFLCHAMPIONSHIPSPREAD", "KXEFLCHAMPIONSHIPTOTAL", "home"),
    "soccer_england_league1": ("KXEFLL1GAME", "KXEFLL1SPREAD", "KXEFLL1TOTAL", "home"),
    "soccer_england_efl_cup": ("KXEFLCUPGAME", "KXEFLCUPSPREAD", "KXEFLCUPTOTAL", "home"),
    "soccer_fa_cup": ("KXFACUPGAME", None, None, "home"),
    "soccer_spl": ("KXSCOTTISHPREMGAME", None, None, "home"),
    "soccer_spain_la_liga": ("KXLALIGAGAME", "KXLALIGASPREAD", "KXLALIGATOTAL", "home"),
    "soccer_spain_segunda_division": ("KXLALIGA2GAME", "KXLALIGA2SPREAD", "KXLALIGA2TOTAL", "home"),
    "soccer_spain_copa_del_rey": ("KXCOPADELREYGAME", None, None, "home"),
    "soccer_italy_serie_a": ("KXSERIEAGAME", "KXSERIEASPREAD", "KXSERIEATOTAL", "home"),
    "soccer_italy_serie_b": ("KXSERIEBGAME", None, None, "home"),
    "soccer_italy_coppa_italia": ("KXCOPPAITALIAGAME", None, None, "home"),
    "soccer_germany_bundesliga": ("KXBUNDESLIGAGAME", "KXBUNDESLIGASPREAD", "KXBUNDESLIGATOTAL", "home"),
    "soccer_germany_bundesliga2": ("KXBUNDESLIGA2GAME", "KXBUNDESLIGA2SPREAD", "KXBUNDESLIGA2TOTAL", "home"),
    "soccer_germany_dfb_pokal": ("KXDFBPOKALGAME", None, None, "home"),
    "soccer_france_ligue_one": ("KXLIGUE1GAME", "KXLIGUE1SPREAD", "KXLIGUE1TOTAL", "home"),
    "soccer_france_ligue_two": ("KXLIGUE2GAME", None, None, "home"),
    "soccer_netherlands_eredivisie": ("KXEREDIVISIEGAME", None, None, "home"),
    "soccer_portugal_primeira_liga": ("KXLIGAPORTUGALGAME", None, None, "home"),
    "soccer_turkey_super_league": ("KXSUPERLIGGAME", None, None, "home"),
    "soccer_saudi_arabia_pro_league": ("KXSAUDIPLGAME", None, None, "home"),
    "soccer_japan_j_league": ("KXJLEAGUEGAME", None, None, "home"),
    "soccer_australia_aleague": ("KXALEAGUEGAME", None, None, "home"),
    "soccer_mexico_ligamx": ("KXLIGAMXGAME", None, None, "home"),
    "soccer_brazil_campeonato": ("KXBRASILEIROGAME", None, None, "home"),
    "soccer_brazil_serie_b": ("KXBRASILEIROBGAME", None, None, "home"),
    "soccer_argentina_primera_division": ("KXARGPREMDIVGAME", None, None, "home"),
    "soccer_conmebol_copa_libertadores": ("KXCONMEBOLLIBGAME", None, None, "home"),
    "soccer_conmebol_copa_sudamericana": ("KXCONMEBOLSUDGAME", None, None, "home"),
    "soccer_uefa_champs_league": ("KXUCLGAME", "KXUCLSPREAD", "KXUCLTOTAL", "home"),
    "soccer_uefa_europa_league": ("KXUELGAME", "KXUELSPREAD", "KXUELTOTAL", "home"),
    "soccer_uefa_europa_conference_league": ("KXUECLGAME", "KXUECLSPREAD", "KXUECLTOTAL", "home"),
    "soccer_uefa_nations_league": ("KXUEFANLGAME", None, None, "home"),
    "soccer_usa_mls": ("KXMLSGAME", "KXMLSSPREAD", "KXMLSTOTAL", "home"),
    "tennis_atp": ("KXATPMATCH", None, None, "home"),
    "tennis_wta": ("KXWTAMATCH", None, None, "home"),
    "mma_mixed_martial_arts": ("KXUFCFIGHT", None, None, "home"),
}
PREFIXES = (("tennis_atp_", "tennis_atp"), ("tennis_wta_", "tennis_wta"))
_BODY = re.compile(r"^(\d{2}[A-Z]{3}\d{2})([A-Z0-9]+)$")


def canonical(league: str) -> Optional[str]:
    if league in LEAGUES:
        return league
    for prefix, tour in PREFIXES:
        if league.startswith(prefix):
            return tour
    return None


class Pacer:
    """At most one request per ``interval`` seconds, across threads."""

    def __init__(self, interval: float = MIN_INTERVAL, clock: Optional[Callable[[], float]] = None,
                 sleep: Optional[Callable[[float], None]] = None) -> None:
        self.interval = interval
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self.clock()
            if now < self._next:
                self.sleep(self._next - now)
                now = self._next
            self._next = now + self.interval


_PROCESS_PACER = Pacer()  # shared by every Kalshi instance in the process


def _ticker_date(body: str) -> Optional[str]:
    try:
        return datetime.datetime.strptime(body[:7].title(), "%y%b%d").date().isoformat()
    except ValueError:
        return None


def _order(rest: str, codes: List[str]) -> Optional[Tuple[str, str]]:
    """Which of the two team codes the ticker lists first ("MINCLE" -> MIN, CLE), allowing an HHMM start
    ("2000MILTB")."""
    if len(codes) != 2:
        return None
    a, b = codes
    for body in (rest, rest[4:] if len(rest) > 4 and rest[:4].isdigit() else None):
        if body is None:
            continue
        if body == a + b:
            return a, b
        if body == b + a:
            return b, a
    return None


def _side_code(ticker: str) -> str:
    return ticker.rsplit("-", 1)[-1] if isinstance(ticker, str) else ""


def _quote(market: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    notional = num(market.get("notional_value_dollars")) or 1.0
    volume = num(market.get("volume_fp"))
    last = num(market.get("last_price_dollars"))
    if last is not None and last <= 0 and not volume:
        last = None  # "0.0000" with no volume: never traded
    return make_quote(market.get("yes_bid_dollars"), market.get("yes_ask_dollars"), last,
                      None, volume * notional if volume is not None else None)


def _milestones(body: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    by_ticker: Dict[str, Dict[str, Any]] = {}
    for milestone in body.get("milestones") or []:
        if not isinstance(milestone, dict):
            continue
        tickers = list(milestone.get("primary_event_tickers") or [])
        main = (milestone.get("details") or {}).get("main_game_event_ticker")
        if main:
            tickers.append(main)
        for ticker in tickers:
            by_ticker.setdefault(str(ticker), milestone)
    return by_ticker


def parse_games(league: str, body: Mapping[str, Any], fetched_at: float) -> List[Dict[str, Any]]:
    """MarketEvents (h2h only) from one /events answer of a winner series."""
    league = canonical(league) or league
    entry = LEAGUES.get(league)
    if entry is None or not isinstance(body, Mapping):
        return []
    series, _spread, _total, first = entry
    milestones = _milestones(body)
    events = []
    for event in body.get("events") or []:
        if not isinstance(event, dict):
            continue
        ticker = str(event.get("event_ticker") or "")
        if not ticker.startswith(series + "-"):
            continue
        match = _BODY.match(ticker[len(series) + 1:])
        if not match:
            continue
        markets = [m for m in event.get("markets") or [] if isinstance(m, dict) and m.get("status") in
                   (None, "active", "open")]
        sides: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for market in markets:
            code = _side_code(market.get("ticker"))
            quote = _quote(market)
            if code and quote is not None:
                sides[code] = (text(market.get("yes_sub_title"), 60) or code, quote)
        draw = sides.pop("TIE", None)
        order = _order(match.group(2), list(sides))
        if order is None:
            continue
        milestone = milestones.get(ticker) or {}
        title = str(milestone.get("title") or "")
        lead = "away" if " at " in title else first
        first_side, second_side = ("away", "home") if lead == "away" else ("home", "away")
        quotes = {first_side: sides[order[0]][1], second_side: sides[order[1]][1]}
        if draw is not None:
            quotes["draw"] = draw[1]
        quotes = normalise(quotes)
        names = {first_side: (sides[order[0]][0], order[0]), second_side: (sides[order[1]][0], order[1])}
        start = parse_time(milestone.get("start_date"))
        status = str((milestone.get("details") or {}).get("status") or "").lower()
        events.append({
            "source": "kalshi",
            "league": league,
            "id": ticker,
            "home": names["home"][0],
            "away": names["away"][0],
            "home_abbr": names["home"][1],
            "away_abbr": names["away"][1],
            "home_key": team_key(league, names["home"][0]) or team_key(league, names["home"][1]),
            "away_key": team_key(league, names["away"][0]) or team_key(league, names["away"][1]),
            "start": iso(start),
            "date": _ticker_date(match.group(1)),
            "live": status in LIVE_STATUSES,
            "url": f"https://kalshi.com/markets/{series.lower()}",
            "as_of": iso(fetched_at),
            "markets": {"h2h": {side: rounded(quote) for side, quote in quotes.items()}},
        })
    events.sort(key=lambda e: (e["start"] or e["date"] or "", e["id"]))
    return events


def parse_lines(league: str, spread_body: Optional[Mapping[str, Any]], total_body: Optional[Mapping[str, Any]],
                games: List[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{winner event ticker: {"spreads": ..., "totals": ...}} from the spread and total series' /events answers,
    joined to ``games`` (parse_games' events) by the ticker's date-and-teams part."""
    league = canonical(league) or league
    entry = LEAGUES.get(league)
    if entry is None:
        return {}
    series, spread_series, total_series, _first = entry
    by_body = {}
    for game in games:
        body = str(game.get("id", ""))[len(series) + 1:]
        by_body[body] = game
    lines: Dict[str, Dict[str, Any]] = {}
    for kind, prefix, answer in (("spreads", spread_series, spread_body), ("totals", total_series, total_body)):
        if not prefix or not isinstance(answer, Mapping):
            continue
        for event in answer.get("events") or []:
            if not isinstance(event, dict):
                continue
            ticker = str(event.get("event_ticker") or "")
            game = by_body.get(ticker[len(prefix) + 1:]) if ticker.startswith(prefix + "-") else None
            if game is None:
                continue
            candidates = []
            for market in event.get("markets") or []:
                if not isinstance(market, dict) or market.get("status") not in (None, "active", "open"):
                    continue
                strike, quote = num(market.get("floor_strike")), _quote(market)
                if strike is None or quote is None:
                    continue
                if kind == "totals":
                    candidates.append((strike, quote, complement(quote), None))
                    continue
                code = _side_code(market.get("ticker"))
                team = next((side for side in ("home", "away") if code.startswith(str(game[f"{side}_abbr"]))
                             and code[len(str(game[f"{side}_abbr"])):].isdigit()), None)
                if team == "home":  # "home wins by over X": home -X
                    candidates.append((-strike, quote, complement(quote), None))
                elif team == "away":  # "away wins by over X": home +X, and the home side is the No
                    candidates.append((strike, complement(quote), quote, None))
            best = pick_line(candidates)
            if best is None:
                continue
            sides = ("home", "away") if kind == "spreads" else ("over", "under")
            lines.setdefault(game["id"], {})[kind] = {sides[0]: rounded(best[1]), sides[1]: rounded(best[2]),
                                                      "point": best[0]}
    return lines


class Kalshi:
    """Kalshi game prices per league, as MarketEvents (see data/markets/__init__.py).

    ``events(league)`` refreshes at most every 5 minutes (every minute while a game of the league is on; lines
    every 15) and is for the background worker; ``cached_events(league)`` never fetches. Requests are paced
    one a second with back-off on "too_many_requests". Failures keep the last good copy; nothing raises.
    """

    source = "kalshi"

    def __init__(self, cache: Any = None, store: Any = None, *, fetch: Optional[Callable[..., Tuple[int, Any]]] = None,
                 clock: Optional[Callable[[], float]] = None, sleep: Optional[Callable[[float], None]] = None,
                 pacer: Optional[Pacer] = None, lines: bool = False) -> None:
        self.cache = cache  # accepted for the data clients' shared-cache convention; copies live in the keeper
        self.store = store
        self.clock = clock or time.time
        self.sleep = sleep or time.sleep
        self._fetch = fetch or http_get
        if pacer is None:
            pacer = _PROCESS_PACER if (clock is None and sleep is None) else Pacer(MIN_INTERVAL, clock, sleep)
        self.pacer = pacer
        self.lines = lines  # fetch the spread and total series too (no consumer before Stage 6)
        self.keeper = Keeper(self.source, store, self.clock)
        self._rest_until = 0.0  # Kalshi is resting (rate-limited) until then

    @staticmethod
    def leagues() -> List[str]:
        return sorted(LEAGUES)

    def covers(self, league: str) -> bool:
        return canonical(league) is not None

    def resting(self) -> bool:
        """Whether Kalshi is resting after a request ran out of retries (requests fail at once until then)."""
        return self.clock() < self._rest_until

    def request(self, path: str, params: Dict[str, Any]) -> Any:
        """One paced GET, retried after 2, 4, 8 and 16 s on "too_many_requests"; raises HttpError otherwise. After
        the last retry Kalshi rests for REST seconds, and every request in that time raises at once."""
        if self.resting():
            raise HttpError(429, "Kalshi")
        for attempt in range(len(BACKOFF) + 1):
            self.pacer.wait()
            status, body = self._fetch(f"{BASE}{path}", params)
            limited = status == 429 or (isinstance(body, dict) and
                                        (body.get("error") or {}).get("code") == "too_many_requests")
            if not limited:
                if status != 200 or not isinstance(body, dict) or body.get("error"):
                    raise HttpError(status, "Kalshi")
                return body
            if attempt < len(BACKOFF):
                self.sleep(BACKOFF[attempt])
        self._rest_until = self.clock() + REST
        log.warning("Kalshi: still rate-limited after %d retries; resting for %d minutes", len(BACKOFF), REST // 60)
        raise HttpError(429, "Kalshi")

    def fetch_series(self, series: str) -> Dict[str, Any]:
        """Every open event of a series with markets and milestones, all pages merged."""
        merged: Dict[str, Any] = {"events": [], "milestones": []}
        cursor = None
        for _ in range(MAX_PAGES):
            params = {"series_ticker": series, "status": "open", "with_nested_markets": "true",
                      "with_milestones": "true", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            body = self.request("/events", params)
            merged["events"].extend(body.get("events") or [])
            merged["milestones"].extend(body.get("milestones") or [])
            cursor = body.get("cursor")
            if not cursor:
                break
        return merged

    def fetch_games(self, league: str) -> List[Dict[str, Any]]:
        league = canonical(league) or league
        return parse_games(league, self.fetch_series(LEAGUES[league][0]), self.clock())

    def fetch_lines(self, league: str, games: List[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        league = canonical(league) or league
        _series, spread, total, _first = LEAGUES[league]
        spread_body = self.fetch_series(spread) if spread else None
        total_body = self.fetch_series(total) if total else None
        return parse_lines(league, spread_body, total_body, games)

    def _ttl(self, league: str) -> float:
        copy = self.keeper.peek(("h2h", league))
        now = self.clock()
        for event in (copy[1] if copy else []) or []:
            start = parse_time(event.get("start"))
            if event.get("live") or (start is not None and start <= now < start + LIVE_FOR):
                return LIVE_TTL
        return TTL

    def events(self, league: str) -> List[Dict[str, Any]]:
        """The league's open games on Kalshi (refreshing when due). [] for a league it doesn't cover."""
        league = canonical(league) or ""
        if not league:
            return []
        games = self.keeper.get(("h2h", league), self._ttl(league), lambda: self.fetch_games(league), empty=[])
        entry = LEAGUES[league]
        lines: Mapping[str, Any] = {}
        if self.lines and games and (entry[1] or entry[2]):
            lines = self.keeper.get(("lines", league), LINES_TTL, lambda: self.fetch_lines(league, games), empty={})
        return _merge(games, lines)

    def cached_events(self, league: str) -> List[Dict[str, Any]]:
        """The last good copy, without fetching (for request handlers)."""
        league = canonical(league) or ""
        games = self.keeper.peek(("h2h", league)) if league else None
        lines = self.keeper.peek(("lines", league)) if league else None
        return _merge(games[1] if games else [], lines[1] if lines else {})


def _merge(events: List[Dict[str, Any]], lines: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [{**event, "markets": {**event.get("markets", {}), **((lines or {}).get(event.get("id")) or {})}}
            for event in events or []]
