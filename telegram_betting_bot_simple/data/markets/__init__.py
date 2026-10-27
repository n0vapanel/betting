"""Free outside opinions for the markets brain: Polymarket and Kalshi prices, ESPN's predictor and power
ratings, Torvik, Tennis Abstract and eloratings.net. Stage 2's Contract A (SPEC_V3_STAGE2.md).

None of this costs Odds API credits. Every source is fetched only by the background worker (``events``,
``power``, ``espn_predictor``), cached with a TTL (prediction markets 5 minutes, 1 minute while a league has a
game on, their spreads/totals 15 minutes; ratings 12 hours; the predictor 1 hour), saved to the store so a
restart has data at once, kept on failure (only the error's type is logged) and never raises into callers.
Request handlers read ``cached_events`` / ``cached_power`` / ``game_prob(..., fetch=False)``, which never fetch.

Shapes:

    Quote = {"p": float,            # implied chance: bid/ask midpoint when both exist, else last price
             "bid": float|None, "ask": float|None, "last": float|None,
             "liquidity": float|None, "volume": float|None,   # US dollars where the venue reports them
             "thin": bool}         # spread > 0.06, no bid or no ask, or liquidity < 1000 (when known)
    MarketEvent = {"source": "polymarket"|"kalshi", "league": str, "home": str, "away": str,
                   "start": iso|None, "url": str|None, "as_of": iso,
                   "markets": {"h2h": {"home": Quote, "away": Quote, "draw": Quote?},
                               "spreads": {"home": Quote, "away": Quote, "point": float}?,   # home's point
                               "totals": {"over": Quote, "under": Quote, "point": float}?},
                   # additions to the contract:
                   "id": str,                      # the venue's event slug / ticker
                   "home_abbr", "away_abbr": str,  # the venue's codes ("was", "WAS")
                   "home_key", "away_key": str|None,  # aliases.team_key of the names
                   "date": "YYYY-MM-DD"|None,      # the game day the venue gives (US Eastern for Kalshi)
                   "live": bool,                   # the venue says the game is on
                   "swapped": bool}                # only on match_events' results: turned to the game's home

Mutually exclusive outcomes are normalised so the h2h chances sum to 1; spreads and totals are one binary
contract and its complement. Never multiply prediction-market leg prices into a parlay price: the venues' own
parlays are priced above the product of their legs (research notes, Moshrefi 2026).

Classes and functions (exact names):

    Polymarket(cache=None, store=None).events(league) -> List[MarketEvent]; .cached_events(league)
    Kalshi(cache=None, store=None).events(league) -> List[MarketEvent]; .cached_events(league)
    Ratings(espn, cache=None, store=None)
        .espn_predictor(league, espn_event_id) -> {"home", "away", "as_of"} | None
        .power(league) -> {team_key: {"rating", "kind": "BPI"|"FPI"|"Elo"|"Torvik", "as_of", ...}}
        .cached_power(league)
        .game_prob(league, home, away, neutral=False, surface=None, *, fetch=True)
            -> {"home", "away", "draw"?, "source", "as_of", "margin"?, "detail"} | None
    team_key(league, name_or_abbr) -> str|None
    match_events(league, games, market_events, tol_hours=None) -> {game_id: MarketEvent} (3 h, 12 h for players)
    is_thin(bid, ask, liquidity), make_quote(...), normalise(quotes), COVERAGE

League keys are the app's (The Odds API's, plus "tennis_atp"/"tennis_wta" for ESPN's tours; the Odds API's
per-tournament tennis keys such as "tennis_atp_vienna_open" map to their tour).

Coverage on 2026-10-27 (see COVERAGE): both venues cover the NBA, NFL, NHL, MLB, college football and men's
college basketball (no games open until its November start), the EPL, La Liga, Serie A, Bundesliga, Ligue 1,
MLS, the Champions/Europa/Conference Leagues, ATP, WTA and UFC, plus many second divisions and cups
(Kalshi has no League Two series; otherwise the two lists match). Ratings exist for
the NBA (BPI), NFL and college football (FPI), college basketball (Torvik), tennis (Tennis Abstract Elo) and
national teams (eloratings.net); the NHL, MLB, club soccer, MMA and the rest have none (ClubElo is down), so
their "Ratings" chip is missing, and ESPN's predictor exists only for the NFL, college football and the NBA.
"""

from __future__ import annotations

from typing import Dict, List

from . import kalshi as _kalshi
from . import polymarket as _polymarket
from .aliases import match_events, orient, team_key
from .common import THIN_LIQUIDITY, THIN_SPREAD, Keeper, complement, is_thin, make_quote, normalise
from .kalshi import Kalshi
from .polymarket import Polymarket
from .ratings import PREDICTOR_LEAGUES, Ratings, source_kind

TARGET_LEAGUES = [
    "basketball_nba", "americanfootball_nfl", "icehockey_nhl", "baseball_mlb", "americanfootball_ncaaf",
    "basketball_ncaab", "soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a", "soccer_germany_bundesliga",
    "soccer_france_ligue_one", "soccer_usa_mls", "soccer_uefa_champs_league", "soccer_uefa_europa_league",
    "tennis_atp", "tennis_wta", "mma_mixed_martial_arts",
]


def coverage() -> Dict[str, Dict[str, object]]:
    """Per league known to any source: which of Polymarket, Kalshi, a rating and ESPN's predictor cover it."""
    leagues = sorted(set(_polymarket.LEAGUES) | set(_kalshi.LEAGUES) | set(TARGET_LEAGUES))
    return {league: {"polymarket": league in _polymarket.LEAGUES, "kalshi": league in _kalshi.LEAGUES,
                     "rating": source_kind(league), "predictor": league in PREDICTOR_LEAGUES}
            for league in leagues}


COVERAGE = coverage()


def missing(source: str) -> List[str]:
    """The target leagues a source lacks ("polymarket", "kalshi", "rating" or "predictor")."""
    return [league for league in TARGET_LEAGUES if not COVERAGE[league][source]]


__all__ = [
    "Polymarket", "Kalshi", "Ratings", "team_key", "match_events", "orient", "is_thin", "make_quote",
    "normalise", "complement", "Keeper", "THIN_SPREAD", "THIN_LIQUIDITY", "COVERAGE", "TARGET_LEAGUES",
    "coverage", "missing", "PREDICTOR_LEAGUES", "source_kind",
]
