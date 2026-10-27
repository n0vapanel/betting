"""Everything the dashboard shows for a league, a game or a bet slip.

Joins the odds feed with ESPN data and the NBA model: each game gets its
best prices, the market's margin-free probabilities and, for the NBA, the
model's predicted score, win/cover/over chances, injuries and player
projections. Bet slips are always priced here from the cached odds, never
from prices the browser sends.

For the Builder (v3 Stage 3) it assembles every usable leg from what is
already in hand, never fetching (builder_candidates): the slate's games, each
league's saved odds board (every price book's price at the consensus line),
"Markets say" (the blend as the chance; disagreement and low-confidence
flags) and the NBA model's cached lines for "why". Soccer games whose board
has the 1X2 and over/under 2.5 add the goals model's markets (engine/soccer.py)
at fair odds, and the game route carries that model as ``soccer``.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from ..ai.desk import AnalystDesk
from ..ai.forecast import AnalystError
from ..data.cache import TTLCache, freshest, ttl_for
from ..data import slate as slate_mod
from ..data.espn_client import ESPN_DAY_TZ, LEAGUES as ESPN_LEAGUES, OUT_STATUSES, EspnClient
from ..data.leagues import SPORTS, pick_leagues
from ..data.odds_service import OddsService, book_roles, is_price_book, moneylines, summarize_markets
from ..data.store import Store
from ..engine import bets, calls, consensus, nba_model
from ..engine.odds_math import decimal_to_american, format_american, remove_vig
from ..engine.slip import Leg, evaluate_slip

MODEL_VERSION = "v1"
# Leagues with the full model (predicted score, players); others show the market's view
AI_LEAGUES = {"basketball_nba"}

# Computed results in the shared cache (data/cache.py), keyed by their inputs and kept for their freshest input's
# time to live, so a game's card is built once for everyone (the bot and the dashboard share the SQLite tier):
GAMES_TTL = ttl_for("games").ttl  # a league's game cards
GAME_KINDS = ("game_card", "game_parts", "consensus")  # one game's entries: dropped together (invalidate_game)


def _partial(value: Any) -> bool:
    """Whether a game's computed entry was built while part of ESPN failed (see TTLCache.get_computed)."""
    return bool(isinstance(value, dict) and value.get("partial"))


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _started_now(card: Dict[str, Any]) -> Dict[str, Any]:
    """A card from a computed entry with its "live" (the game has started) worked out now from its start: the entry
    can be older than tip-off, and calls close at tip-off."""
    card["live"] = _parse_time(card["start"]) <= _now_utc()
    return card


class NotFound(Exception):
    pass


def consensus_score(p: Dict[str, Any], outcome: str) -> Optional[Dict[str, float]]:
    """engine/consensus.py's score, rounded for the record: Brier score (half the squared distance summed over the
    outcomes, which for two is the usual (p - won)^2 the AI record uses) and log loss (-ln p(outcome)); None when
    the forecast didn't price that outcome (a tie on a two-way market)."""
    scored = consensus.score(p or {}, outcome)
    return {key: round(value, 6) for key, value in scored.items()} if scored else None


def _mean(values: Any) -> Optional[float]:
    values = [float(v) for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def _frozen(row: Dict[str, Any]) -> Dict[str, Any]:
    sources = row.get("sources") or []
    return {"blend": row.get("blend"), "sources": sources,
            "links": [{"name": s["name"], "url": s["url"]} for s in sources if s.get("url")]}


def _restale(source: Dict[str, Any], now: float) -> Dict[str, Any]:
    """A logged source with its status checked again for its age (the dashboard may have stopped refreshing it)."""
    limit = (slate_mod.MARKET_STALE if source.get("kind") == "prediction_market"
             else slate_mod.PREDICTOR_STALE if source.get("name") == "ESPN" else slate_mod.RATING_STALE)
    age = slate_mod._age(source.get("as_of"), now)
    if source.get("status") == "ok" and age is not None and age > limit:
        return {**source, "status": "stale"}
    return dict(source)


class CallRejected(Exception):
    """A call that can't be saved (game started, no prediction yet...); the message says why."""


def _parse_time(value: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def outcome_label(game: Dict[str, Any], market: str, side: str, point: Optional[float]) -> str:
    """How a pick reads on a bet slip, e.g. "Oklahoma City Thunder -13.5" or "Over 224.5"; a soccer goals-model pick
    reads "Both teams to score: Yes", "Arsenal (draw no bet)", "Arsenal or draw" or "Arsenal over 1.5 goals"."""
    if market in DERIVED_MARKETS:
        return derived_label(game["home"], game["away"], market, side, point)
    team = {"home": game["home"], "away": game["away"], "draw": "Draw"}.get(side, side.title())
    if market == "h2h":
        return team if side == "draw" else f"{team} to win"
    if market == "spreads" and point is not None:
        return f"{team} {point:+g}"
    if market == "totals" and point is not None:
        return f"{side.title()} {point:g}"
    return team


# --- soccer's goals model (engine/soccer.py) and the builder's markets ------------------------------------
#
# A soccer game whose saved board has the 1X2 and over/under 2.5 (every book quoting both sides at 2.5) is turned into
# goal rates and a Dixon-Coles grid (soccer.derive: rho -0.13, a +1.5-point nudge to both teams to score), and from
# that the markets the board doesn't carry: both teams to score, draw no bet, double chance, team totals and the
# likeliest scores. The 1X2 fed in is the markets brain's blend where the slate has one, else the books' margin-free
# 1X2, so the game page and the builder read the same model. No bookmaker prices those markets here (the board asks
# only for moneylines, spreads and totals: more markets cost more credits), so they are shown, and built with, at
# fair odds ("model_price"): check your bookmaker's price.

BOARD_MARKETS = ("h2h", "spreads", "totals")
DERIVED_MARKETS = ("btts", "draw_no_bet", "double_chance", "team_totals")  # engine/builder.py's SOCCER_MARKETS
DERIVED_SIDES: Dict[str, Tuple[str, ...]] = {
    "btts": ("yes", "no"), "draw_no_bet": ("home", "away"), "double_chance": ("1X", "12", "X2"),
    "team_totals": ("home_over", "home_under", "away_over", "away_under"),
}
MODEL_BOOK = "Fair odds"  # engine/soccer.py's MODEL_BOOK: the "book" of a leg priced at the goals model's fair odds
FAIR_PRICE_NOTE = "Priced at fair odds; check your bookmaker's price."
SOCCER_LINE = 2.5  # the over/under line the goal rates are fitted to
SOCCER_TTL = 2 * 3600.0  # a game's goals model, keyed by its inputs: a new board is a new entry
STALE_PRICE = 2 * 3600  # a board older than the odds service's 2 hours marks its legs "stale_price"


def derived_label(home: str, away: str, market: str, side: str, point: Optional[float]) -> str:
    """engine/soccer.py's words for a goals-model pick ("Both teams to score: Yes", "Arsenal (draw no bet)", "Arsenal
    or draw", "Arsenal over 1.5 goals")."""
    if market == "btts":
        return "Both teams to score: Yes" if side == "yes" else "Both teams to score: No"
    if market == "draw_no_bet":
        return f"{home if side == 'home' else away} (draw no bet)"
    if market == "double_chance":
        return {"1X": f"{home} or draw", "X2": f"Draw or {away}", "12": f"{home} or {away}"}.get(side, side)
    if market == "team_totals":
        team, _, way = side.partition("_")
        name = home if team == "home" else away
        return f"{name} {way} {point:g} goals" if point is not None else f"{name} {way}"
    return side


def _fair_chance(chances: List[float]) -> Optional[float]:
    return sum(chances) / len(chances) if chances else None


def soccer_inputs(game: Mapping[str, Any], one_x_two: Optional[Mapping[str, Any]] = None
                  ) -> Optional[Dict[str, float]]:
    """The goals model's inputs from a saved board: the 1X2 (``one_x_two``, the blend, when given with all three
    outcomes; else summarize_markets' margin-free 1X2, every book pricing all three sides, sharp books included) and
    the margin-free chance of over 2.5 goals (each book quoting both sides at 2.5, de-vigged, averaged). None when
    either is missing."""
    fair: Dict[str, Any] = {}
    if one_x_two and all(isinstance(one_x_two.get(k), (int, float)) for k in ("home", "draw", "away")):
        fair = {k: float(one_x_two[k]) for k in ("home", "draw", "away")}
    else:
        try:
            h2h = summarize_markets(dict(game)).get("h2h") or {}
        except Exception:
            return None
        fair = {side: (h2h.get(side) or {}).get("fair") for side in ("home", "draw", "away")}
    if not all(isinstance(value, float) and 0 < value < 1 for value in fair.values()):
        return None
    overs = []
    for book in game.get("books") or []:
        outcomes = ((book.get("markets") or {}).get("totals") or {}) if isinstance(book, Mapping) else {}
        over, under = outcomes.get("over"), outcomes.get("under")
        if not (isinstance(over, Mapping) and isinstance(under, Mapping)):
            continue
        if over.get("point") != SOCCER_LINE or under.get("point") != SOCCER_LINE:
            continue
        try:
            overs.append(remove_vig([float(over["price"]), float(under["price"])])[0])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
    over_2_5 = _fair_chance(overs)
    if over_2_5 is None or not 0 < over_2_5 < 1:
        return None
    return {"home": fair["home"], "draw": fair["draw"], "away": fair["away"], "over_2_5": over_2_5}


def _offer(p: float, push: float = 0.0) -> Dict[str, Any]:
    """One goals-model outcome for the game page: its chance (outright), the push (draw no bet's draw), its fair odds
    (soccer.fair_odds: (1 - push) / chance) and the book price, which the board never has for these markets."""
    from ..engine import soccer
    fair = soccer.fair_price(p, push)  # the price the Builder's leg has (rounded down to the cent)
    entry: Dict[str, Any] = {"p": round(p, 4), "fair": float(fair) if fair is not None else None, "book": None}
    if push:
        entry["push"] = round(push, 4)
    return entry


def soccer_view(home: str, away: str, inputs: Mapping[str, float], derived: Mapping[str, Any]) -> Dict[str, Any]:
    """The game route's "Goals model" block from soccer.derive's result."""
    from ..engine import soccer
    m = derived["markets"]
    one = m["1x2"]
    team_totals = {team: [{"line": float(line), "over": _offer(sides["over"]), "under": _offer(sides["under"])}
                          for line, sides in sorted(m["team_totals"][team].items(), key=lambda item: float(item[0]))]
                   for team in ("home", "away")}
    return {
        "rates": {"home": round(derived["rates"]["home"], 3), "away": round(derived["rates"]["away"], 3)},
        "rho": derived["rho"],
        "btts_nudge": derived.get("btts_nudge"),
        "inputs": {key: round(float(value), 4) for key, value in inputs.items()},
        "fit": derived.get("fit"), "fit_ok": derived.get("fit_ok"),
        "model_1x2": {key: round(one[key], 4) for key in ("home", "draw", "away")},
        "expected_goals": {key: round(value, 3) for key, value in (m.get("expected_goals") or {}).items()},
        "markets": {
            "btts": {"yes": _offer(m["btts"]["yes"]), "no": _offer(m["btts"]["no"])},
            # Draw no bet: the draw gives the stake back, so the chance shown is the win outright, with the push
            "draw_no_bet": {side: _offer(one[side], one["draw"]) for side in ("home", "away")},
            "double_chance": {side: _offer(m["double_chance"][side]) for side in ("1X", "12", "X2")},
            "team_totals": team_totals,
        },
        "correct_score": [{"score": row["score"], "p": round(row["p"], 4),
                           "fair": float(soccer.fair_price(row["p"])) if row["p"] > 0 else None}
                          for row in m.get("correct_score") or []],
        "teams": {"home": home, "away": away},
        "note": ("Goal rates fitted to the market's 1X2 and over/under 2.5, then a Dixon-Coles grid (rho "
                 f"{derived['rho']:g}) with a +1.5-point nudge to both teams to score. No bookmaker prices these "
                 "markets here: priced at fair odds; check your bookmaker's price."),
    }


def leg_key(league: str, event_id: str, market: str, side: str, point: Optional[float]) -> str:
    """The pricing engine's stable id for a pick, e.g. "basketball_nba:<event>:spreads:home:-3.5"."""
    key = f"{league}:{event_id}:{market}:{side}"
    return key if point is None else f"{key}:{point:g}"


def _line_prices(event: Mapping[str, Any], market: str, side: str, point: Optional[float]) -> Dict[str, Decimal]:
    """Every price book's price for one outcome at one line (titles as keys): the builder picks the best among the
    books a person allows. Never a sharp book or an exchange."""
    prices: Dict[str, Decimal] = {}
    for book in event.get("books") or []:
        if not isinstance(book, Mapping) or not is_price_book(book):
            continue
        offer = ((book.get("markets") or {}).get(market) or {}).get(side)
        if not isinstance(offer, Mapping) or offer.get("point") != point:
            continue
        price = offer.get("price")
        if isinstance(price, (int, float)) and price > 1:
            title = str(book.get("title") or book.get("key"))
            value = Decimal(str(price))
            if title not in prices or value > prices[title]:
                prices[title] = value
    return prices


def _pct(p: float) -> str:
    return f"{p:.0%}"


class DashboardService:
    def __init__(
        self,
        odds: OddsService,
        espn: EspnClient,
        tz: ZoneInfo,
        params: nba_model.Params = nba_model.Params(),
        workers: int = 8,
        store: Optional[Store] = None,
        analysts: Optional[AnalystDesk] = None,
        cache: Optional[TTLCache] = None,
    ) -> None:
        self.odds = odds
        self.espn = espn
        self.tz = tz
        self.params = params
        # The process's one shared cache (services.build_service passes it); an empty TTLCache is falsy, hence "is"
        self.cache = cache if cache is not None else TTLCache()
        self.pool = ThreadPoolExecutor(max_workers=workers)
        # People's predictions and the AI's track record; None turns those features off
        self.store = store
        # The AI analysts (Claude, DeepSeek), when an API key for one is set
        self.analysts = analysts
        # The per-person features (dashboard/lab.py) set this; its saved bets and rollover plans are
        # graded with everything else in grade_due
        self.lab: Any = None
        # The all-sports slate (data/slate.py), in the dashboard process: its markets brain's detail for each game
        # is what the game route shows. The bot's process has none and reads the dashboard's consensus_log instead
        self.slate: Any = None
        self._markets_record: Optional[Tuple[Any, Dict[str, Any]]] = None  # (the store's graded mark, the record)

    # --- menus -----------------------------------------------------------

    def leagues(self) -> List[Dict[str, Any]]:
        sports = self.odds.sports()
        groups = []
        for label, group in SPORTS:
            leagues = [
                {"key": league["key"], "title": league["title"], "ai": league["key"] in AI_LEAGUES}
                for league in pick_leagues(sports, group)
            ]
            if leagues:
                groups.append({"group": group, "label": label, "leagues": leagues})
        return groups

    # --- league and game views ---------------------------------------------

    def games(self, league: str, refresh: bool = False) -> Dict[str, Any]:
        """Every game in the league with odds and predictions (cached for a minute).

        ``refresh`` asks the odds service for new odds (it refuses if the
        current ones are only minutes old, to save quota).
        """
        if refresh:
            before = self.odds.odds_loaded_at(league)
            self.odds.league_games(league, refresh=True)
            if self.odds.odds_loaded_at(league) != before:
                self.cache.invalidate(("games", league))

        def load() -> Dict[str, Any]:
            context = self._league_context(league)
            games = self.odds.league_games(league)
            results = list(self.pool.map(lambda game: self._card(league, game, context), games))
            cards = [card for card, _ in results]
            self._log_ai_predictions(league, results)
            if self.analysts and league in AI_LEAGUES:
                briefs = self.analysts.briefs(league)
                for card in cards:
                    card["analysts"] = briefs.get(card["id"], [])
            return {
                "league": league,
                "title": next((s["title"] for s in self.odds.sports() if s.get("key") == league), league),
                "ai": league in AI_LEAGUES,
                "games": cards,
                "updated": datetime.datetime.now(self.tz).strftime("%-I:%M %p %Z"),
            }

        ttl, stale = ttl_for("games")
        result = self.cache.get(("games", league), ttl, load, stale=stale, kind="games")
        loaded = self.odds.odds_loaded_at(league)
        odds_as_of = (
            datetime.datetime.fromtimestamp(loaded, self.tz).strftime("%-I:%M %p %Z") if loaded else None
        )
        return {**result, "quota": dict(self.odds.quota), "model": MODEL_VERSION, "odds_as_of": odds_as_of}

    def game(self, league: str, event_id: str) -> Dict[str, Any]:
        """One game in full: all bookmakers' odds, prediction breakdown and player projections.

        Built once for everyone: a computed entry (kind "game_card") keyed by the game's inputs (the odds board it
        came from, ESPN's state and score of the game) and kept for its freshest input's time to live (seconds
        while live, a minute before); the slate drops it when the game's status or score changes. "Markets say"
        is its own entry (``consensus``)."""
        game, sig, state = self._game_inputs(league, event_id)
        key = ("game_card", league, event_id, sig)
        ttl, stale = self._game_policy(state)
        # Built while ESPN failed ("partial"): kept a few seconds for everyone, then rebuilt (data/cache.py's
        # get_computed), so a down ESPN costs one rebuild per PARTIAL_KEEP, not one per person
        payload = self.cache.get_computed(key, ttl, lambda: self._game_payload(league, event_id, game, sig, state),
                                          _partial, stale=stale, kind="game_card")
        card = _started_now(copy.deepcopy(payload["card"]))
        card["consensus"] = self._consensus(league, event_id, game, payload.get("model"), state)
        # Soccer's goals model, from the same 1X2 the builder uses (the blend, else the books'): its own computed entry
        blend = ((card["consensus"] or {}).get("blend") or {}).get("p") if isinstance(card["consensus"], dict) else None
        soccer = self.soccer_model(league, game, blend if isinstance(blend, dict) and "draw" in blend else None)
        if soccer is not None:
            card["soccer"] = soccer
        return card

    # --- computed entries: one game's inputs, its parts, and dropping them ----------------------------------

    def _game_inputs(self, league: str, event_id: str) -> Tuple[Dict[str, Any], str, str]:
        """(the game on the league's odds board, its inputs' signature, its state for the policy table). The
        signature covers when the board was fetched, the game's own prices and ESPN's state and score of it, so a
        new board or a scoring play is a new entry. Raises NotFound when the board has no such game."""
        game = next((g for g in self.odds.league_games(league) if g["id"] == event_id), None)
        if game is None:
            raise NotFound(f"No game {event_id} in {league}")
        espn_state = None
        if league in ESPN_LEAGUES:
            try:
                match = self.espn.find_game(league, game["home"], game["away"], game["commence_time"])
            except Exception:
                match = None
            read = getattr(self.espn, "event_state", None)
            if match and callable(read):
                espn_state = read(league, match["event_id"])
        if espn_state and espn_state.get("state"):
            state = espn_state["state"]
        else:
            state = "live" if _parse_time(game["commence_time"]) <= datetime.datetime.now(datetime.timezone.utc) \
                else "pre"
        loaded = getattr(self.odds, "odds_loaded_at", None)
        inputs = {"odds_at": loaded(league) if callable(loaded) else None, "game": game, "espn": espn_state,
                  "state": state}
        text = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
        return game, hashlib.sha1(text.encode("utf-8")).hexdigest()[:20], state

    @staticmethod
    def _game_policy(state: str) -> Tuple[float, float]:
        """A game's computed entries: the freshest of the game's own policy and ESPN's summary for its state."""
        policy = freshest(ttl_for("game_card", state), ttl_for("summary", state))
        return policy.ttl, policy.stale

    def _parts(self, league: str, event_id: str, game: Dict[str, Any], sig: str, state: str) -> Dict[str, Any]:
        """The game's card with its prediction, the prediction itself (info) and its players' projections: built
        once per signature for game() and detail(). In this process's memory only (the prediction is an object)."""
        def load() -> Dict[str, Any]:
            context = self._league_context(league)
            card, info = self._card(league, game, context)
            self._explain(card, info)
            players = None
            if league in AI_LEAGUES and info.get("prediction") and info.get("match") and context:
                players = self._players(league, info, context)
            prediction = info.get("prediction")
            model = (prediction.sources.get("model") or {}).get("home_win_prob") if prediction else None
            return {"card": card, "info": info, "players": players, "model": model, "partial": bool(info.get("partial"))}

        key = ("game_parts", league, event_id, sig)
        ttl, stale = self._game_policy(state)
        return self.cache.get_computed(key, ttl, load, _partial, stale=stale, persist=False, kind="game_parts")

    def _game_payload(self, league: str, event_id: str, game: Dict[str, Any], sig: str, state: str
                      ) -> Dict[str, Any]:
        """game()'s cached value: the card with every book, the players and the analysts (plain JSON, so the bot
        and the dashboard share it), the team model's chance for "Markets say", and whether ESPN failed meanwhile."""
        parts = self._parts(league, event_id, game, sig, state)
        card = copy.deepcopy(parts["card"])
        card["books"] = [self._book_view(book) for book in game["books"]]
        if parts["players"] is not None:
            card["players"] = copy.deepcopy(parts["players"])
        if self.analysts and league in AI_LEAGUES:
            card["analysts"] = self.analysts.game_view(league, event_id)
        return {"card": card, "model": parts["model"], "partial": parts["partial"]}

    def invalidate_game(self, league: str, odds_id: Optional[str] = None, espn_id: Optional[str] = None,
                        board: bool = False) -> int:
        """Drop a game's computed entries (its card, parts and "Markets say") here and in the shared table, and with
        ``board`` the league's game cards too: the slate calls this when it sees the game's status or score change,
        so the next view rebuilds once for everyone. Returns how many entries this process dropped."""
        if not odds_id:
            return 0  # a game the odds feed doesn't list has no card here (game() needs its board)
        dropped = 0
        prefixes = [(kind, league, str(odds_id)) for kind in GAME_KINDS]
        drop_prefix = getattr(self.cache, "invalidate_prefix", None)
        if callable(drop_prefix):
            dropped += int(drop_prefix(*prefixes) or 0)
        if board:
            self.cache.invalidate(("games", league))
        return dropped

    def _explain(self, card: Dict[str, Any], info: Dict[str, Any]) -> None:
        """The game route's typed factors (engine/factors.py): each of the prediction's numbers as a labelled
        baseline plus sized factors that add up to it ("margin", "total", "home_win" in percent, and the team
        model's own "model_margin" and "model_total"), rounded so the deltas still sum to the value shown. Only
        on one game's detail: the boards stay lean."""
        if card.get("prediction") and info.get("prediction") is not None:
            card["prediction"]["projections"] = self._projections(info["prediction"])

    def forget(self, league: str) -> None:
        """Rebuild the league's game cards, and each game's computed entries, on the next request (new analyst
        forecasts)."""
        self.cache.invalidate(("games", league))
        drop_prefix = getattr(self.cache, "invalidate_prefix", None)
        if callable(drop_prefix):
            drop_prefix(*[(kind, league) for kind in GAME_KINDS])

    def _league_context(self, league: str) -> Optional[Dict[str, Any]]:
        """Standings this season and last, for the team model (NBA only): the team context, a computed entry
        (kind "team_context") keyed by the league and when its odds board was fetched (the scoring level reads the
        board's totals), shared by the bot and the dashboard."""
        if league not in AI_LEAGUES:
            return None

        def load() -> Dict[str, Any]:
            today = datetime.datetime.now(ESPN_DAY_TZ).date()
            previous_end_year = today.year if today.month >= 8 else today.year - 1
            current = self.espn.standings(league)
            previous = self.espn.standings(league, previous_end_year)
            games = [team["games"] for team in current.values() if team.get("games")]
            league_avg = nba_model.blend(
                nba_model.league_average(current), sum(games) / len(games) if games else 0,
                nba_model.league_average(previous), self.params.team_prior_games,
            )
            context = {"current": current, "previous": previous, "avg": league_avg,
                       "label": nba_model.season_label(today),
                       # Each season's own average, so last season's strengths can be regressed to the mean (a list:
                       # plain JSON for the shared table)
                       "averages": [nba_model.league_average(current), nba_model.league_average(previous), league_avg]}
            context["level"] = self._scoring_level(league, context)
            return context

        loaded = getattr(self.odds, "odds_loaded_at", None)
        board_at = loaded(league) if callable(loaded) else None
        ttl, stale = freshest(ttl_for("team_context"), ttl_for("standings"))
        try:
            return self.cache.get(("team_context", league, board_at), ttl, load, stale=stale, kind="team_context")
        except Exception as exc:
            logging.warning("No ESPN standings for %s: %s", league, exc)
            return None

    def _scoring_level(self, league: str, context: Dict[str, Any]) -> Optional[float]:
        """The average team score today's market totals imply, net of who is playing whom.

        Our standings blend this season with last, so their scoring level lags
        the league's (early in 2026-27 it ran ~4 points a game low against the
        market, making every total lean under). The market's totals set the
        level; our ratings still say which matchups score more or less.
        """
        avg = context.get("avg")
        if not avg:
            return None
        offsets = []
        try:
            games = self.odds.league_games(league)
        except Exception:
            return None
        for game in games:
            line = (summarize_markets(game).get("totals", {}).get("over") or {}).get("point")
            match = self.espn.find_game(league, game["home"], game["away"], game["commence_time"]) if line else None
            if not match:
                continue
            ratings = [
                nba_model.team_rating(context["current"].get(match[f"{side}_id"]),
                                      context["previous"].get(match[f"{side}_id"]), self.params, context["averages"])
                for side in ("home", "away")
            ]
            if not all(ratings):
                continue
            matchup = sum(r.points_for - avg + r.points_against - avg for r in ratings)
            offsets.append(line - (2 * avg + matchup))
        if len(offsets) < 3:
            return None
        # The median keeps one odd line (a star ruled out late) from moving every game
        return avg + statistics.median(offsets) / 2

    def _card(self, league: str, game: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        summary = summarize_markets(game)
        start = _parse_time(game["commence_time"])
        local = start.astimezone(self.tz)
        info: Dict[str, Any] = {}
        try:
            info = self._predict(league, game, summary, context)
        except Exception:
            logging.exception("Prediction failed for %s @ %s", game["away"], game["home"])
        prediction = info.get("prediction")
        if prediction is not None and self.analysts:
            try:
                self.analysts.adjust(league, game["id"], prediction, self.params)
            except Exception:
                logging.exception("Couldn't apply the analysts' forecasts to %s", game["id"])
        card = {
            "id": game["id"],
            "league": league,
            "start": game["commence_time"],
            "when": f"{local:%a %b} {local.day}, {local:%-I:%M %p %Z}",
            "live": start <= datetime.datetime.now(datetime.timezone.utc),
            "home": game["home"],
            "away": game["away"],
            "markets": self._market_view(game, summary, prediction),
            "prediction": self._prediction_view(game, prediction, info) if prediction else None,
            "espn_home_win": info.get("espn_home_win"),
            "injuries": info.get("injury_names", {}),
        }
        return card, info

    def _predict(self, league: str, game: Dict[str, Any], summary: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        match = self.espn.find_game(league, game["home"], game["away"], game["commence_time"]) if league in ESPN_LEAGUES else None
        info["match"] = match
        injuries: Dict[str, List[Dict[str, str]]] = {}
        if match:
            try:
                espn_summary = self.espn.summary(league, match["event_id"])
            except Exception as exc:
                logging.warning("No ESPN summary for %s: %s", match["event_id"], exc)
                espn_summary = {}
                info["partial"] = True  # a computed entry built now is kept only seconds (get_computed)
            predictor = espn_summary.get("predictor") or {}
            home_projection = (predictor.get("homeTeam") or {})
            if home_projection.get("id") == match["home_id"] and home_projection.get("gameProjection"):
                info["espn_home_win"] = float(home_projection["gameProjection"]) / 100
            for team in espn_summary.get("injuries", []):
                injuries[team.get("team", {}).get("id")] = [
                    {"id": item["athlete"]["id"], "name": item["athlete"].get("displayName"), "status": item.get("status", "")}
                    for item in team.get("injuries", []) if item.get("athlete")
                ]
            info["injury_names"] = {
                side: [f"{i['name']} ({i['status']})" for i in injuries.get(match[f"{side}_id"], [])]
                for side in ("home", "away")
            }
        info["injuries"] = injuries
        if league not in AI_LEAGUES:
            return info

        teams = {}
        for side in ("home", "away"):
            name = game[side]
            rating, impact, missing, back_to_back = None, 0.0, [], False
            if match and context:
                team_id = match[f"{side}_id"]
                rating = nba_model.team_rating(context["current"].get(team_id), context["previous"].get(team_id),
                                               self.params, context["averages"])
                team_games = (context["current"].get(team_id) or {}).get("games") or 0
                for player in injuries.get(team_id, []):
                    if player["status"].lower() not in OUT_STATUSES:
                        continue
                    try:
                        seasons = self.espn.player_seasons(league, player["id"])
                    except Exception:
                        continue
                    profile = nba_model.player_profile(seasons, context["label"], self.params)
                    if profile and profile["min"] >= 10:
                        impact += nba_model.absence_impact(profile, team_games, self.params)
                        missing.append(player["name"])
                yesterday = match["day"] - datetime.timedelta(days=1)
                back_to_back = team_id in self.espn.played_on(league, yesterday)
            teams[side] = nba_model.TeamInputs(name, rating, impact, missing, back_to_back)
        info["teams"] = teams
        info["prediction"] = nba_model.predict_game(
            teams["home"], teams["away"], (context or {}).get("avg"), summary, info.get("espn_home_win"), self.params,
            level=(context or {}).get("level"),
        )
        return info

    def _market_view(self, game: Dict[str, Any], summary: Dict[str, Any], prediction: Optional[nba_model.GamePrediction]) -> Dict[str, Any]:
        view: Dict[str, Any] = {}
        for market, sides in summary.items():
            view[market] = {}
            for side, offer in sides.items():
                chance = prediction.probability(market, side, offer["point"]) if prediction else offer["fair"]
                # A push hands the stake back, so it counts in expected value but not as a win
                push = prediction.push_probability(market, side, offer["point"]) if prediction else 0.0
                view[market][side] = {
                    "price": offer["price"],
                    "american": format_american(decimal_to_american(offer["price"])) if offer["price"] > 1 else None,
                    "point": offer["point"],
                    "book": offer["book"],
                    "books": offer["books"],
                    "fair": offer["fair"],
                    "ai": chance,
                    "push": push,
                    "ev": chance * offer["price"] + push - 1 if chance is not None else None,
                    "label": outcome_label(game, market, side, offer["point"]),
                }
        return view

    def _prediction_view(self, game: Dict[str, Any], prediction: nba_model.GamePrediction, info: Dict[str, Any]) -> Dict[str, Any]:
        home_win = prediction.home_win_prob
        favourite = "home" if home_win >= 0.5 else "away"
        return {
            "home_score": round(prediction.home_score, 1),
            "away_score": round(prediction.away_score, 1),
            "margin": round(prediction.margin, 1),
            "total": round(prediction.total, 1),
            "home_win": home_win,
            "pick": game[favourite],
            "pick_prob": max(home_win, 1 - home_win),
            "sources": {name: {k: round(v, 3) for k, v in values.items()} for name, values in prediction.sources.items()},
            "factors": prediction.factors,  # text lines, for older pages (the game route adds "projections")
            "method": self._method_note(),
        }

    @staticmethod
    def _projections(prediction: nba_model.GamePrediction) -> Optional[Dict[str, Any]]:
        explain = getattr(prediction, "projections", None)
        if not callable(explain):
            return None
        try:
            return {key: projection.to_dict(digits=1) for key, projection in explain().items()}
        except Exception:
            logging.exception("Couldn't explain the prediction's numbers")
            return None

    def _method_note(self) -> str:
        p = self.params
        model = 1 - p.market_weight - p.espn_weight
        return (
            f"Model {MODEL_VERSION} leans on the bookmakers' consensus ({p.market_weight:.0%}), with ESPN's matchup "
            f"predictor ({p.espn_weight:.0%}) and our team model ({model:.0%}: points scored and allowed this season "
            f"and last, home court, back-to-backs, injured players' scoring). On ~13,900 NBA games (2011-22) a "
            f"results-only team model added nothing on top of the closing odds, so the market dominates whenever "
            f"there is a line; the team model takes over when there isn't one. Final margins spread around the "
            f"prediction by about {p.margin_sd_factor:g}·√total points (13 at a 230 total), totals by "
            f"{p.total_sd_factor:g}·√total (19). For scale: the closing market picks about 68% of NBA winners, "
            f"the realistic ceiling for anyone."
        )

    def _book_view(self, book: Dict[str, Any]) -> Dict[str, Any]:
        markets = {}
        for market, outcomes in book["markets"].items():
            markets[market] = {
                side: {
                    "american": format_american(decimal_to_american(offer["price"])) if offer["price"] > 1 else None,
                    "price": offer["price"],
                    "point": offer["point"],
                }
                for side, offer in outcomes.items()
            }
        # "sharp": Pinnacle or an exchange, shown for the consensus only (no price of theirs is ever offered);
        # "price": a book a person can bet at; "other": a listed book left out of ODDS_PRICE_BOOKS
        return {"title": book["title"], "key": book.get("key"), "role": book_roles().role(book), "markets": markets}

    def _players(self, league: str, info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        match, prediction, teams = info["match"], info["prediction"], info["teams"]
        out: Dict[str, List[Dict[str, Any]]] = {}
        for side in ("home", "away"):
            team_id = match[f"{side}_id"]
            try:
                roster = self.espn.roster(league, team_id)
            except Exception as exc:
                logging.warning("No roster for team %s: %s", team_id, exc)
                info["partial"] = True
                out[side] = []
                continue
            statuses = {p["id"]: p["status"] for p in info["injuries"].get(team_id, [])}
            seasons = list(self.pool.map(lambda p: self._safe_seasons(league, p["id"]), roster))
            players = []
            for player, history in zip(roster, seasons):
                status = statuses.get(player["id"], player["status"])
                players.append({
                    **player,
                    "status": status,
                    "out": status.lower() in OUT_STATUSES,
                    "profile": nba_model.player_profile(history, context["label"], self.params),
                })
            rating = teams[side].rating
            team_points = prediction.home_score if side == "home" else prediction.away_score
            # The team's usual game total, at today's scoring level
            shift = 2 * ((context.get("level") or context["avg"] or 0) - (context["avg"] or 0))
            normal_total = rating.points_for + rating.points_against + shift if rating else None
            pace = prediction.total / normal_total if normal_total else 1.0
            team_games = (context["current"].get(team_id) or {}).get("games") or 0
            projections = nba_model.project_players(
                players, team_points, pace, self.params,
                margin=prediction.margin if side == "home" else -prediction.margin,
                home=side == "home", team_games=team_games,
            )
            out[side] = [
                {key: (round(value, 1) if isinstance(value, float) else value) for key, value in vars(p).items()}
                for p in projections
                if p.out or p.minutes >= 8
            ]
        return out

    def _safe_seasons(self, league: str, athlete_id: str) -> List[Dict[str, Any]]:
        try:
            return self.espn.player_seasons(league, athlete_id)
        except Exception:
            return []

    # --- people's predictions ("calls") and the AI's track record ------------------

    def detail(self, league: str, event_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """A game's card, the prediction behind it, and its players' projections by player id (from the game's
        computed parts, shared with game(); copies, so a caller can't change the cached ones)."""
        game, sig, state = self._game_inputs(league, event_id)
        parts = self._parts(league, event_id, game, sig, state)
        card = _started_now(copy.deepcopy(parts["card"]))  # "calls close at tip-off" reads it: never a cached flag
        card["consensus"] = self._consensus(league, event_id, game, parts["model"], state)
        players: Dict[str, Dict[str, Any]] = {}
        for side, rows in (parts["players"] or {}).items():
            for row in rows:
                if not row.get("out"):
                    players[str(row["id"])] = {**row, "side": side}
        return card, copy.deepcopy(parts["info"]), players

    def judge(self, league: str, event_id: str, call: Dict[str, Any]) -> Dict[str, Any]:
        """What the AI thinks of a call, part by part (shown live while someone makes it)."""
        card, info, players = self.detail(league, event_id)
        prediction = info.get("prediction")
        if prediction is None:
            raise CallRejected("Calls are open for NBA games for now: the AI has no prediction for this game.")
        return calls.judge_call(prediction, call, card["home"], card["away"], players)

    def save_call(self, player: str, name: str, league: str, event_id: str, call: Dict[str, Any]) -> Dict[str, Any]:
        """Save someone's call on a game (replacing their earlier one) with the AI's view of it."""
        if self.store is None:
            raise CallRejected("Saving calls needs the server's state folder.")
        card, info, players = self.detail(league, event_id)
        if card["live"]:
            raise CallRejected("This game has started: calls close at tip-off.")
        prediction = info.get("prediction")
        if prediction is None:
            raise CallRejected("Calls are open for NBA games for now: the AI has no prediction for this game.")
        verdict = calls.judge_call(prediction, call, card["home"], card["away"], players)
        call_id = self.store.save_prediction({
            "player": player, "name": name, "league": league, "event_id": event_id,
            "espn_id": (info.get("match") or {}).get("event_id"),
            "home": card["home"], "away": card["away"], "start": card["start"], "call": call, "ai": verdict,
        })
        return {"id": call_id, "ai": verdict}

    def my_calls(self, player: str) -> List[Dict[str, Any]]:
        if self.store is None:
            return []
        self.grade_due()
        return self.store.predictions(player=player, limit=200)

    def delete_call(self, player: str, call_id: int) -> bool:
        return bool(self.store and self.store.delete_prediction(player, call_id))

    def leaderboard(self) -> List[Dict[str, Any]]:
        if self.store is None:
            return []
        self.grade_due()
        return self.store.leaderboard()

    def _log_ai_predictions(self, league: str, results: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> None:
        """Keep the AI's latest pre-game prediction for every game, for its track record."""
        if self.store is None or league not in AI_LEAGUES:
            return
        for card, info in results:
            prediction = info.get("prediction")
            if card["live"] or prediction is None:
                continue
            market = prediction.sources.get("market", {})
            model = prediction.sources.get("model", {})
            try:
                self.store.log_ai_prediction({
                    "league": league, "event_id": card["id"], "espn_id": (info.get("match") or {}).get("event_id"),
                    "home": card["home"], "away": card["away"], "start": card["start"],
                    "payload": {
                        "model": MODEL_VERSION,
                        "home_win": prediction.home_win_prob,
                        "margin": prediction.margin,
                        "total": prediction.total,
                        "market_home_win": market.get("home_win_prob"),
                        "market_margin": market.get("margin"),
                        "market_total": market.get("total"),
                        "espn_home_win": info.get("espn_home_win"),
                        # The team model alone: the markets brain's "Model" source (the prediction itself already
                        # leans on the market, which the brain counts on its own)
                        "model_home_win": model.get("home_win_prob"),
                    },
                })
            except Exception:
                logging.exception("Couldn't log the AI prediction for %s", card["id"])

    def final(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """A finished game's score from ESPN (state final, void or pending)."""
        espn_id = row.get("espn_id")
        if not espn_id and row["league"] in ESPN_LEAGUES:
            match = self.espn.find_game(row["league"], row["home"], row["away"], row["start"])
            espn_id = match["event_id"] if match else None
        if not espn_id:
            return {"state": "pending"}
        return {**self.espn.game_result(row["league"], espn_id), "espn_id": espn_id}

    def grade_due(self) -> int:
        """Score calls, AI predictions and saved bets on games that have finished; returns how many were graded.

        Runs in whichever process gets to it first (a short lease stops the
        bot and the dashboard grading the same games at once).
        """
        if self.store is None or not self.store.try_lease("grading", 120):
            return 0
        now = datetime.datetime.now(datetime.timezone.utc)
        # NBA games last about 2h15m: nothing that started more recently can be over
        cutoff = (now - datetime.timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        give_up = (now - datetime.timedelta(days=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
        graded = 0
        for row in self.store.predictions(status="pending", started_before=cutoff):
            try:
                final = self.final(row)
                if final["state"] == "pending":
                    if row["start"] < give_up:
                        self.store.grade_prediction(row["id"], "void", {"reason": "no result"}, None, None)
                    continue
                if final["state"] == "void":
                    self.store.grade_prediction(row["id"], "void", final, None, None)
                    continue
                box = self.espn.box_score(row["league"], final["espn_id"]) if row["call"].get("props") else None
                # The AI and the market are scored on their last numbers before tip-off
                logged = self.store.ai_prediction(row["league"], row["event_id"])
                result = calls.grade_call(row["call"], row["ai"], final, box, (logged or {}).get("payload"))
                self.store.grade_prediction(row["id"], "graded", result, result["points"], result["ai_points"])
                graded += 1
            except Exception:
                logging.exception("Couldn't grade call %s", row["id"])
        for row in self.store.ai_predictions(status="pending", started_before=cutoff):
            try:
                final = self.final(row)
                if final["state"] == "pending":
                    if row["start"] < give_up:
                        self.store.grade_ai_prediction(row["league"], row["event_id"], "void", {"reason": "no result"})
                    continue
                if final["state"] == "final":
                    try:
                        final["closing"] = self.espn.closing_line(row["league"], final["espn_id"])
                    except Exception:
                        final["closing"] = None
                self.store.grade_ai_prediction(
                    row["league"], row["event_id"], "graded" if final["state"] == "final" else "void", final
                )
                graded += 1
            except Exception:
                logging.exception("Couldn't grade the AI prediction for %s", row["event_id"])
        for row in self.store.ai_tickets(status="pending"):
            if row["last_start"] >= cutoff:
                continue
            try:
                finals = []
                for leg in row["payload"]["legs"]:
                    final = self.final({"league": row["league"], **leg})
                    finals.append(final)
                if any(f["state"] == "pending" for f in finals):
                    if row["first_start"] < give_up:
                        self.store.grade_ai_ticket(row["day"], row["league"], row["tier"], "void", {"reason": "no result"})
                    continue
                status, result = bets.grade_ticket(
                    row["payload"]["legs"], [f if f["state"] == "final" else None for f in finals]
                )
                self.store.grade_ai_ticket(row["day"], row["league"], row["tier"], status, result)
                graded += 1
            except Exception:
                logging.exception("Couldn't grade the AI's %s ticket for %s", row["tier"], row["day"])
        graded += self._grade_consensus(cutoff, give_up)
        if self.lab is not None:
            graded += self.lab.grade_due(cutoff, give_up)  # saved bets, then the plan days they belong to
        if self.analysts:
            graded += self.analysts.grade_due(self, cutoff, give_up)
        return graded

    # --- the markets brain ("Markets say") -------------------------------------------------------------

    def _consensus(self, league: str, event_id: str, game: Optional[Dict[str, Any]], model: Optional[float],
                   state: str) -> Optional[Dict[str, Any]]:
        """The game's "Markets say": the slate's own detail when this process runs the slate (worked out once per
        round for everyone), else a computed entry (kind "consensus") keyed by the game, the board's time, the team
        model's chance and whether the game has started, kept for its state's time to live."""
        if self.slate is not None:
            found = self._slate_detail(league, event_id)
            if found:
                return found
        loaded = getattr(self.odds, "odds_loaded_at", None)
        board_at = loaded(league) if callable(loaded) and game is not None else None
        start = slate_mod._parse_time((game or {}).get("commence_time"))
        started = start is not None and start <= time.time()
        key = ("consensus", league, event_id, board_at, None if model is None else round(float(model), 6), started)
        policy = ttl_for("consensus", state)
        return self.cache.get(key, policy.ttl, lambda: self._markets_detail(league, event_id, game, model, None),
                              stale=policy.stale, kind="consensus")

    def _slate_detail(self, league: str, event_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.slate.markets_for(league, odds_id=event_id)
        except Exception:
            logging.exception("Couldn't read the slate's markets for %s", event_id)
            return None

    def markets_detail(self, league: str, event_id: str, game: Optional[Dict[str, Any]] = None,
                       prediction: Optional[nba_model.GamePrediction] = None, now: Optional[float] = None
                       ) -> Optional[Dict[str, Any]]:
        """A game's "Markets say": {"blend": Blend|None, "sources": [Source...], "links": [{"name", "url"}]}, or
        None when nothing has an opinion yet. Never fetches a source or spends an Odds API credit.

        The slate's own detail when the dashboard's slate has one; else (the bot's process, or before the slate's
        first round) the dashboard's last logged opinions from consensus_log (prediction markets, ESPN, ratings),
        with the books' consensus from the board in hand (or the saved one) and the team model's own chance. Once
        a game has started its logged detail is frozen and shown as it was.

        Without a slate it is the shared cache's "consensus" entry for the game (see _consensus); with ``now`` (a
        test's clock) it is worked out afresh."""
        if self.slate is not None:
            found = self._slate_detail(league, event_id)
            if found:
                return found
        model = (prediction.sources.get("model") or {}).get("home_win_prob") if prediction else None
        if now is not None:
            return self._markets_detail(league, event_id, game, model, now)
        start = slate_mod._parse_time((game or {}).get("commence_time"))
        state = "pre" if start is None or start > time.time() else "live"
        return self._consensus(league, event_id, game, model, state)

    def _markets_detail(self, league: str, event_id: str, game: Optional[Dict[str, Any]], model: Optional[float],
                        now: Optional[float]) -> Optional[Dict[str, Any]]:
        """markets_detail worked out from the store's logged opinions and the board (no slate, no cache)."""
        now = time.time() if now is None else now
        row = None
        if self.store is not None:
            try:
                row = self.store.consensus_row(league, event_id=event_id)
            except Exception:
                logging.exception("Couldn't read the logged consensus for %s", event_id)
        started = slate_mod._parse_time((row or {}).get("start"))
        if row and row.get("blend") and (row["status"] != "pending" or (started is not None and started <= now)):
            return _frozen(row)
        if game is None:
            saved = getattr(self.odds, "saved_games", None)
            found_saved = saved(league) if callable(saved) else None
            game = next((g for g in (found_saved or (None, []))[1] if g.get("id") == event_id), None)
            as_of = found_saved[0] if found_saved else None
        else:
            loaded = getattr(self.odds, "odds_loaded_at", None)
            as_of = loaded(league) if callable(loaded) else None
        sources: List[Dict[str, Any]] = []
        if game is not None:
            sources += slate_mod.book_sources(moneylines(game), as_of, now)
        if row:
            sources += [_restale(source, now) for source in row.get("sources") or []
                        if source.get("kind") in ("prediction_market", "model_rating")]
        if model is not None:
            sources.append(slate_mod.model_source(model, now, now))
        elif row:
            sources += [source for source in row.get("sources") or [] if source.get("kind") == "team_model"]
        start = slate_mod._parse_time((game or {}).get("commence_time") or (row or {}).get("start"))
        return slate_mod.market_detail(league, sources, start, now)

    def _grade_consensus(self, cutoff: str, give_up: str) -> int:
        """Score the logged consensus of finished games: each source's and the blend's Brier score and log loss."""
        if self.store is None or not hasattr(self.store, "consensus_rows"):
            return 0
        graded = 0
        for row in self.store.consensus_rows(status="pending", started_before=cutoff):
            try:
                if row["league"] not in ESPN_LEAGUES:
                    # Tennis tournaments, MMA cards and leagues ESPN doesn't cover: no single-game result to read
                    self.store.grade_consensus(row["league"], row["game_key"], "void", None,
                                               {"reason": "no game result here"})
                    continue
                final = self.final(row)
                if final["state"] == "pending":
                    if row["start"] < give_up:
                        self.store.grade_consensus(row["league"], row["game_key"], "void", None, {"reason": "no result"})
                    continue
                if final["state"] != "final":
                    self.store.grade_consensus(row["league"], row["game_key"], "void", None, final)
                    continue
                home, away = (final["away"], final["home"]) if row.get("swapped") else (final["home"], final["away"])
                outcome = "home" if home > away else "away" if away > home else "draw"
                result = {"score": {"home": home, "away": away}, "outcome": outcome,
                          "blend": consensus_score((row.get("blend") or {}).get("p") or {}, outcome),
                          "scores": {}}
                for source in row.get("sources") or []:
                    if source.get("status") not in ("ok", "thin"):
                        continue  # stale and missing opinions weren't live prices when the game started
                    scored = consensus_score(source.get("p") or {}, outcome)
                    if scored:
                        result["scores"][str(source.get("name"))] = {**scored, "kind": source.get("kind"),
                                                                     "status": source.get("status")}
                self.store.grade_consensus(row["league"], row["game_key"], "graded", outcome, result)
                graded += 1
            except Exception:
                logging.exception("Couldn't grade the consensus for %s", row.get("game_key"))
        return graded

    def markets_record(self) -> Dict[str, Any]:
        """Per league and per source, on graded games: n, mean Brier score and log loss, and the blend's on the
        same games, so the record shows which source has earned its weight."""
        if self.store is None or not hasattr(self.store, "consensus_graded"):
            return {"sports": [], "games": 0}
        # Read only: the grader thread (grade_due, every 10 minutes) scores finished games. Every graded game counts
        # (no cap), and the record is worked out again only when a game has been graded since
        cached = getattr(self, "_markets_record", None)
        if cached is not None:
            try:
                if self.store.consensus_graded_mark() == cached[0]:
                    return cached[1]
            except Exception:
                logging.exception("Couldn't read the markets record's mark")
        mark, graded = self.store.consensus_graded()
        record = self._markets_record_of(graded)
        self._markets_record = (mark, record)
        return record

    @staticmethod
    def _markets_record_of(graded: List[Dict[str, Any]]) -> Dict[str, Any]:
        leagues: Dict[str, Dict[str, Any]] = {}
        for row in graded:
            result = row.get("result") or {}
            blend = result.get("blend")
            if not blend:
                continue  # a draw on a two-way market: nobody's forecast priced it
            entry = leagues.setdefault(row["league"], {"blend": [], "sources": {}, "low": 0})
            entry["blend"].append(blend)
            if (row.get("blend") or {}).get("confidence") == "low":
                entry["low"] += 1
            for name, scored in (result.get("scores") or {}).items():
                source = entry["sources"].setdefault(name, {"kind": scored.get("kind"), "rows": []})
                source["rows"].append((scored, blend))
        sports = []
        for league, entry in leagues.items():
            sources = []
            for name, source in entry["sources"].items():
                rows = source["rows"]
                sources.append({
                    "name": name, "kind": source["kind"], "n": len(rows),
                    "brier": _mean(r[0]["brier"] for r in rows), "log_loss": _mean(r[0]["log_loss"] for r in rows),
                    "blend_brier": _mean(r[1]["brier"] for r in rows),
                    "blend_log_loss": _mean(r[1]["log_loss"] for r in rows),
                })
            sources.sort(key=lambda item: (-item["n"], item["name"]))
            sports.append({"league": league, "title": slate_mod.TITLES.get(league), "n": len(entry["blend"]),
                           "low_confidence": entry["low"],
                           "blend": {"n": len(entry["blend"]), "brier": _mean(b["brier"] for b in entry["blend"]),
                                     "log_loss": _mean(b["log_loss"] for b in entry["blend"])},
                           "sources": sources})
        sports.sort(key=lambda item: (-item["n"], item["league"]))
        return {"sports": sports, "games": sum(item["n"] for item in sports),
                "note": ("Brier score (lower is better; 0.25 is a coin flip on a two-way game) and log loss on games "
                         "graded after the final, from each source's last opinion before the start. The blend is "
                         "the app's best estimate; the market usually wins.")}

    def track_record(self, league: str = "basketball_nba") -> Dict[str, Any]:
        """How the AI's pre-game predictions did: accuracy, Brier score next to the market's,
        record against the closing spread and total, and calibration."""
        if self.store is None:
            return {"games": 0}
        self.grade_due()
        rows = self.store.ai_predictions(league=league, status="graded")
        record = calls.track_record(rows)
        record["tiers"] = bets.tier_record(self.store.ai_tickets(league=league))
        record["analysts"] = self.analysts.record() if self.analysts else []
        return record

    # --- the AI analysts ----------------------------------------------------------------

    def ask_analyst(self, league: str, event_id: str, analyst: str) -> Dict[str, Any]:
        """Ask one analyst about a game now (within its daily limit); returns the game's analyst view."""
        if self.analysts is None:
            raise CallRejected("No AI analysts are set up: add ANTHROPIC_API_KEY or DEEPSEEK_API_KEY on the server.")
        try:
            result = self.analysts.analyze(self, league, event_id, only=analyst, asked=True)[analyst]
        except AnalystError as exc:
            raise CallRejected(str(exc)) from None
        if result["status"] != "done":
            raise CallRejected(result["message"])
        return self.analysts.game_view(league, event_id)

    # --- AI-built bets: safe, risky, crazy --------------------------------------

    def ai_bets(self, league: str, source: str = "all", player: Optional[str] = None) -> Dict[str, Any]:
        """Bets at three risk levels from the league's upcoming games, or only from someone's calls: the Builder's
        tickets (engine/builder.py) under the Safe, Risky and Crazy presets, one bookmaker at a time (engine/bets.py)."""
        board = self.games(league)
        allowed = self._legs_from_calls(league, player, board) if source == "mine" and player else None
        candidates = []
        for card in board["games"]:
            if card["live"]:
                continue
            for market, sides in card["markets"].items():
                for side, offer in sides.items():
                    if offer["ai"] is None or (allowed is not None and (card["id"], market, side) not in allowed):
                        continue
                    prices = self._book_prices(league, card["id"], market, side, offer["point"], max_hold=bets.MAX_HOLD)
                    if prices:
                        candidates.append(bets.Candidate(
                            event_id=card["id"], game=f"{card['away']} @ {card['home']}", market=market, side=side,
                            point=offer["point"], label=offer["label"], probability=offer["ai"], fair=offer["fair"],
                            prices=prices, push=offer.get("push") or 0.0, start=card["start"],
                        ))
        if league in AI_LEAGUES:
            note = ("Built from the AI's chances and the best price at one bookmaker per bet. The AI leans on the "
                    "bookmakers, so expect small negative value: these are for fun, not tips. " + bets.VIG_NOTE)
        else:
            note = ("No AI model for this league yet, so these use the bookmakers' own chances: "
                    "expect slightly negative value on all of them. " + bets.VIG_NOTE)
        if allowed is not None and not allowed:
            note = "You have no open calls in this league yet: make some in Games, then come back."
        tiers = bets.build_all(candidates)
        if source == "all" and league in AI_LEAGUES:
            self._log_ai_tickets(league, board, tiers)
        return {"league": league, "source": source, "note": note, "tiers": tiers}

    def _log_ai_tickets(self, league: str, board: Dict[str, Any], tiers: Dict[str, Dict[str, Any]]) -> None:
        """Keep each tier's top ticket for the day (until its first game starts), to grade it later."""
        if self.store is None:
            return
        cards = {card["id"]: card for card in board["games"]}
        now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for key, tier in tiers.items():
            if not tier["bets"]:
                continue
            ticket = tier["bets"][0]
            legs = [{**leg, "home": cards[leg["event_id"]]["home"], "away": cards[leg["event_id"]]["away"],
                     "start": cards[leg["event_id"]]["start"]} for leg in ticket["legs"] if leg["event_id"] in cards]
            if len(legs) != len(ticket["legs"]):
                continue
            starts = sorted(leg["start"] for leg in legs)
            day = _parse_time(starts[0]).astimezone(ESPN_DAY_TZ).date().isoformat()
            try:
                self.store.log_ai_ticket(day, league, key, starts[0], starts[-1], {**ticket, "legs": legs}, now_iso)
            except Exception:
                logging.exception("Couldn't log the AI's %s ticket", key)

    def _legs_from_calls(self, league: str, player: str, board: Dict[str, Any]) -> set:
        """The bets someone's open calls imply: their winner, and from their score the side of the
        bookmakers' spread and total they're on."""
        if self.store is None:
            return set()
        cards = {card["id"]: card for card in board["games"]}
        legs = set()
        for row in self.store.predictions(player=player, status="pending"):
            card = cards.get(row["event_id"])
            if row["league"] != league or card is None:
                continue
            call = row["call"]
            if call.get("winner"):
                legs.add((card["id"], "h2h", call["winner"]))
            if call.get("home_score") is not None:
                margin = call["home_score"] - call["away_score"]
                total = call["home_score"] + call["away_score"]
                spread = card["markets"].get("spreads", {}).get("home")
                if spread and spread["point"] is not None and margin + spread["point"] != 0:
                    legs.add((card["id"], "spreads", "home" if margin + spread["point"] > 0 else "away"))
                line = (card["markets"].get("totals", {}).get("over") or {}).get("point")
                if line is not None and total != line:
                    legs.add((card["id"], "totals", "over" if total > line else "under"))
        return legs

    # --- soccer's goals model and the builder's candidates (engine/builder.py) ---------------------------

    def _blend_of(self, league: str, event_id: str) -> Optional[Dict[str, float]]:
        """The markets brain's blended chances for a game, from the slate's detail (this process's slate only)."""
        if self.slate is None:
            return None
        detail = self._slate_detail(league, str(event_id))
        blend = (detail or {}).get("blend") or {}
        p = blend.get("p") if isinstance(blend, Mapping) else None
        return dict(p) if isinstance(p, Mapping) else None

    def soccer_derived(self, league: str, game: Optional[Mapping[str, Any]],
                       one_x_two: Optional[Mapping[str, Any]] = None) -> Optional[Tuple[Dict[str, float], Dict[str, Any]]]:
        """(inputs, soccer.derive's result) for a soccer game whose saved board has the 1X2 and over/under 2.5, else
        None. A computed entry (kind "soccer_model") keyed by the game and its inputs, so the game route, the builder
        and both processes fit it once per board. ``one_x_two``: the blend (default: the slate's, when it has one)."""
        if not league.startswith("soccer_") or not game or not game.get("id"):
            return None
        if one_x_two is None:
            one_x_two = self._blend_of(league, str(game["id"]))
        inputs = soccer_inputs(game, one_x_two)
        if inputs is None:
            return None
        sig = tuple(round(inputs[key], 6) for key in ("home", "draw", "away", "over_2_5"))
        key = ("soccer_model", league, str(game["id"]), sig)

        def fit() -> Dict[str, Any]:
            from ..engine import soccer  # the engine's goals model (Stage 3)
            return soccer.derive(inputs["home"], inputs["draw"], inputs["away"], inputs["over_2_5"])

        try:
            return inputs, self.cache.get(key, SOCCER_TTL, fit, kind="soccer_model", error_ttl=60)
        except Exception:
            logging.exception("Couldn't fit the goals model for %s", game.get("id"))
            return None

    def soccer_model(self, league: str, game: Optional[Mapping[str, Any]],
                     one_x_two: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """The game route's "Goals model" block (soccer_view), or None when the game has no 1X2 and over/under 2.5."""
        found = self.soccer_derived(league, game, one_x_two)
        if found is None:
            return None
        inputs, derived = found
        return soccer_view(str(game.get("home") or "Home"), str(game.get("away") or "Away"), inputs, derived)

    def _builder_games(self, now: float) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], str]:
        """The games the builder may use, the leagues' entries ({key: {title, odds, ...}}) and an ETag: the
        dashboard's slate (Stage 1: every in-season league, with "Markets say"), else (the bot's process, or no
        slate yet) the saved odds boards' games. Never fetches."""
        current = None
        if self.slate is not None:
            try:
                current = self.slate.current()
            except Exception:
                logging.exception("Couldn't read the slate")
        if current is not None:
            data = current.data or {}
            entries = {entry["key"]: entry for entry in data.get("leagues") or [] if entry.get("key")}
            return list(data.get("games") or []), entries, current.etag
        times = self._odds_times()
        games: List[Dict[str, Any]] = []
        entries = {}
        for league in sorted(times):
            saved = self.odds.saved_games(league) if hasattr(self.odds, "saved_games") else None
            if not saved:
                continue
            fetched_at, events = saved
            entries[league] = {"key": league, "title": league,
                               "odds": "fresh" if now - fetched_at < STALE_PRICE else "stale"}
            for event in events or []:
                start = slate_mod._parse_time(event.get("commence_time"))
                if start is None or not event.get("id"):
                    continue
                games.append({"id": f"{league}:{event['id']}", "league": league, "odds_id": str(event["id"]),
                              "start": event["commence_time"], "status": "pre" if start > now else "live",
                              "home": {"name": event.get("home")}, "away": {"name": event.get("away")}})
        return games, entries, f'"boards-{self.odds_mark()}"'

    def _odds_times(self) -> Dict[str, float]:
        if self.store is None or not hasattr(self.store, "odds_fetched_at"):
            return {}
        try:
            return dict(self.store.odds_fetched_at())
        except Exception:
            logging.exception("Couldn't read the saved odds' times")
            return {}

    def odds_mark(self) -> str:
        """A short signature of when every league's saved odds were fetched (a new board changes it)."""
        return hashlib.sha1(json.dumps(sorted(self._odds_times().items())).encode()).hexdigest()[:16]

    def builder_candidates(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Every leg the builder may use, assembled from what is already in hand (the slate's games, each league's
        saved odds board, "Markets say" and the NBA model's cached factor lines), never fetched:

        {"candidates": [engine.builder.Candidate], "by_key": {leg key: Candidate}, "leagues": {league: {"title",
        "odds", "odds_as_of", "games": [start of each game without prices], "priced"}}, "etag", "consensus_as_of"}

        Each leg carries every price book's price at the line most books offer (``prices``: the engine picks the best
        among the books a person allows) and the best of them as its price; its chance is the markets brain's blend
        (moneylines), else the price books' margin-free chance (``fair``). A line that can push (a whole number) is
        left out: its chance of a push isn't known here. Soccer games add the goals model's markets at fair odds
        (engine/soccer.py's candidates, flag "model_price"). Kept a minute in this process, per slate and set of
        boards (``now``: the caller's clock, for price ages and games under way)."""
        now = time.time() if now is None else now
        games, entries, etag = self._builder_games(now)
        key = ("builder_candidates", etag, self.odds_mark())
        return self.cache.get(key, 60.0, lambda: self._assemble_candidates(games, entries, etag, now),
                              persist=False, kind="builder_candidates")

    def _assemble_candidates(self, games: Sequence[Mapping[str, Any]], entries: Mapping[str, Mapping[str, Any]],
                             etag: str, now: float) -> Dict[str, Any]:
        from ..engine import builder as engine_builder  # the engine's module (Stage 3)

        boards: Dict[str, Optional[Tuple[float, Dict[str, Dict[str, Any]]]]] = {}
        model_lines = self._model_lines()
        candidates: List[Any] = []
        leagues: Dict[str, Dict[str, Any]] = {}
        consensus_as_of: Optional[str] = None
        for game in games:
            league = str(game.get("league") or "")
            if not league or game.get("status") not in ("pre", "live"):
                continue
            entry = entries.get(league) or {}
            info = leagues.setdefault(league, {"title": entry.get("title") or league, "odds": entry.get("odds"),
                                               "odds_as_of": entry.get("odds_as_of"), "games": [], "priced": 0})
            odds_id, _ = slate_mod.ids_of(game)
            if league not in boards:
                saved = None
                try:
                    saved = self.odds.saved_games(league) if hasattr(self.odds, "saved_games") else None
                except Exception:
                    logging.exception("Couldn't read the saved %s odds", league)
                boards[league] = (saved[0], {str(g.get("id")): g for g in saved[1] or [] if isinstance(g, Mapping)}) \
                    if saved else None
            board = boards[league]
            event = board[1].get(str(odds_id)) if board and odds_id else None
            if event is None:
                info["games"].append(game.get("start"))
                continue
            mkt = game.get("mkt") if isinstance(game.get("mkt"), Mapping) else {}
            if mkt.get("as_of") and (consensus_as_of is None or mkt["as_of"] > consensus_as_of):
                consensus_as_of = mkt["as_of"]
            try:
                made = self._game_candidates(engine_builder, league, game, event, board[0], mkt, model_lines, now)
            except Exception:
                logging.exception("Couldn't make the builder's legs for %s", odds_id)
                made = []
            if made:
                info["priced"] += 1
            candidates += made
        by_key = {candidate.key: candidate for candidate in candidates}
        return {"candidates": candidates, "by_key": by_key, "leagues": leagues, "etag": etag,
                "consensus_as_of": consensus_as_of}

    def _model_lines(self) -> Dict[str, List[str]]:
        """The NBA team model's lines per game ("Team model: Boston Celtics by 4.2, total 226", then its top factors),
        from the league's cached game cards (peeked, never built here). Only where the team model itself ran (its
        "model" source): a card without it carries the market's own spread and total, which are no model's view."""
        out: Dict[str, List[str]] = {}
        peek = getattr(self.cache, "peek", None)
        for league in AI_LEAGUES:
            try:
                found = peek(("games", league)) if callable(peek) else None
            except Exception:
                found = None
            board = found[1] if found else None
            for card in (board or {}).get("games") or []:
                prediction = card.get("prediction") or {}
                model = ((prediction.get("sources") or {}).get("model") or {}) if isinstance(prediction, Mapping) else {}
                if not isinstance(model, Mapping) or not isinstance(model.get("margin"), (int, float)):
                    continue
                margin = float(model["margin"])
                leader = card.get("home") if margin >= 0 else card.get("away")
                head = f"Team model: {leader} by {abs(margin):.1f}"
                if isinstance(model.get("total"), (int, float)):
                    head += f", total {float(model['total']):.0f}"
                out[str(card.get("id"))] = [head] + [str(line) for line in prediction.get("factors") or [] if line][:1]
        return out

    def _game_candidates(self, engine_builder: Any, league: str, game: Mapping[str, Any], event: Mapping[str, Any],
                         fetched_at: float, mkt: Mapping[str, Any], model_lines: Mapping[str, List[str]],
                         now: float) -> List[Any]:
        odds_id = str(event.get("id"))
        home, away = str(event.get("home") or "Home"), str(event.get("away") or "Away")
        start = str(event.get("commence_time") or game.get("start"))
        started = game.get("status") == "live" or (slate_mod._parse_time(start) or now) <= now
        age = max(0, int(now - fetched_at))
        summary = summarize_markets(dict(event))
        detail_p = self._blend_of(league, odds_id)
        chance = game.get("chance") if isinstance(game.get("chance"), Mapping) else {}
        blend = detail_p or (chance if chance.get("src") == "blend" else {})
        game_flags = set()
        if mkt.get("dis"):
            game_flags.add("disagree")
        if mkt.get("conf") == "low":
            game_flags.add("low_conf")
        if started:
            game_flags.add("live")
        if age > STALE_PRICE:
            game_flags.add("stale_price")
        game_key = f"{league}:{odds_id}"
        model = model_lines.get(odds_id) or []
        model_line = ["; ".join(model[:2])] if model else []
        flag_line = []
        if "disagree" in game_flags:
            flag_line = [f"Sources disagree by {int(mkt.get('spr') or 5)} points"]
        elif "low_conf" in game_flags:
            flag_line = ["Low confidence: no market price in the blend, ratings only"]
        strip = self._strip_words(mkt)
        out: List[Any] = []
        for market in BOARD_MARKETS:
            for side, offer in (summary.get(market) or {}).items():
                price, point, fair = offer.get("price"), offer.get("point"), offer.get("fair")
                if not isinstance(price, (int, float)) or price <= 1:
                    continue
                if market != "h2h" and (point is None or float(point).is_integer()):
                    continue  # a line that can push: its chance of a push isn't known here
                p, source = None, "books"
                if market == "h2h" and isinstance(blend.get(side), (int, float)):
                    p, source = float(blend[side]), "blend"
                elif isinstance(fair, float):
                    p = fair
                if p is None or not 0 < p < 1:
                    continue
                flags = set(game_flags)
                count = int(offer.get("books") or 0)
                if count <= 1:
                    flags.add("thin")
                if source == "blend":
                    first = strip(side, p)
                else:
                    first = f"Books' margin-free chance {_pct(p)} ({count} book{'s' if count != 1 else ''} at this line)"
                why = [first] + model_line + flag_line
                if "thin" in flags and len(why) < 3:
                    why.append("Thin market: one bookmaker at this line")
                out.append(engine_builder.Candidate(
                    ref={"league": league, "id": odds_id, "market": market, "side": side, "point": point},
                    league=league, game_key=game_key, teams=(away, home), start=start, price=Decimal(str(price)),
                    book=str(offer.get("book") or "?"), chance=p, chance_source=source, price_age_s=age,
                    flags=flags, why=why[:3], label=outcome_label({"home": home, "away": away}, market, side, point),
                    fair=fair if isinstance(fair, float) else None,
                    prices=_line_prices(event, market, side, point) or None))
        found = self.soccer_derived(league, event, detail_p if detail_p and "draw" in detail_p else None)
        if found is not None:
            from ..engine import soccer
            out += soccer.candidates(league, game_key, (away, home), start, found[1], ref_id=odds_id,
                                     price_age_s=age, flags=game_flags)
        return out

    @staticmethod
    def _strip_words(mkt: Mapping[str, Any]) -> Any:
        """The "Markets say" summary for one side: "Markets say 63% (Sharp 64% · US 62% · Kalshi 63%)"."""
        def words(side: str, p: float) -> str:
            fav = mkt.get("fav")
            parts = []
            two_way = "draw" not in (mkt.get("p") or {})
            for row in mkt.get("src") or []:
                if not isinstance(row, (list, tuple)) or len(row) < 2 or not isinstance(row[1], (int, float)):
                    continue
                if (row[2] if len(row) > 2 else "ok") != "ok":
                    continue
                if side == fav:
                    parts.append(f"{row[0]} {_pct(row[1])}")
                elif two_way and fav in ("home", "away") and side in ("home", "away"):
                    parts.append(f"{row[0]} {_pct(1 - row[1])}")
            text = f"Markets say {_pct(p)}"
            return f"{text} ({' · '.join(parts[:4])})" if parts else text
        return words

    # --- bet slip ------------------------------------------------------------

    def price_slip(self, picks: List[Dict[str, Any]], mode: str, stake: float, stakes: Dict[int, float]) -> Dict[str, Any]:
        """Price picks ({league, id, market, side}) from current odds and evaluate the slip.

        Singles use each pick's best price at any bookmaker. A multi has to
        be placed at one bookmaker, so it is priced at every bookmaker that
        offers all its picks at the same lines, and the best of those is used.
        """
        legs: List[Leg] = []
        offers_by_leg: List[Dict[str, float]] = []
        fair_probs: List[Optional[float]] = []
        missing = []
        for pick in picks:
            board = self.games(pick["league"])
            card = next((g for g in board["games"] if g["id"] == pick["id"]), None)
            offer = (card or {}).get("markets", {}).get(pick["market"], {}).get(pick["side"])
            if not card or not offer:
                missing.append(pick)
                continue
            legs.append(Leg(
                event_id=card["id"], market=pick["market"], side=pick["side"], price=offer["price"],
                label=offer["label"], game=f"{card['away']} @ {card['home']}", start=card["start"],
                point=offer["point"], probability=offer["ai"],
            ))
            fair_probs.append(offer["fair"])
            offers_by_leg.append(self._book_prices(pick["league"], pick["id"], pick["market"], pick["side"], offer["point"]))

        books = []
        if mode == "multi" and legs:
            common = set.intersection(*(set(offers) for offers in offers_by_leg))
            for title in common:
                combined = 1.0
                for offers in offers_by_leg:
                    combined *= offers[title]
                books.append({"book": title, "decimal": combined})
            books.sort(key=lambda entry: entry["decimal"], reverse=True)
            if books:
                best = books[0]["book"]
                for leg, offers in zip(legs, offers_by_leg):
                    leg.price = offers[best]

        result = evaluate_slip(legs, mode, stake=stake, stakes=stakes)
        if mode == "multi" and legs:
            for entry in books:
                entry["american"] = format_american(decimal_to_american(entry["decimal"]))
                entry["returns"] = round(stake * entry["decimal"], 2)
            result["books"] = books
            result["book"] = books[0]["book"] if books else None
            # The bookmaker's edge compounds with every leg: show how much of the stake it keeps on average
            if result.get("odds") and all(p is not None for p in fair_probs):
                fair = 1.0
                for p in fair_probs:
                    fair *= p
                result["book_edge"] = 1 - fair * result["odds"]["decimal"]
            if not books and len(legs) > 1:
                result["reasons"].append("No single bookmaker offers all these picks at these lines")
                result["valid"] = False
        if missing:
            result["reasons"].append(f"{len(missing)} pick(s) are no longer offered and were removed")
            result["valid"] = False
        result["removed"] = missing
        return result

    def _book_prices(self, league: str, event_id: str, market: str, side: str, point: Optional[float],
                     max_hold: Optional[float] = None) -> Dict[str, float]:
        """Each bookmaker's decimal price for one pick, at the same line as the pick: price books only
        (ODDS_PRICE_BOOKS), never a sharp book or an exchange.

        With ``max_hold``, bookmakers whose margin on this market (at this
        line) is higher are left out.
        """
        game = next((g for g in self.odds.league_games(league) if g["id"] == event_id), None)
        prices = {}
        for book in (game or {}).get("books", []):
            if not is_price_book(book):
                continue  # Pinnacle and the exchanges anchor the consensus; nobody is offered their prices
            outcomes = book["markets"].get(market, {})
            offer = outcomes.get(side)
            if not offer or offer["point"] != point:
                continue
            if max_hold is not None:
                if any(o["point"] != (-point if market == "spreads" and s != side and point is not None else point)
                       for s, o in outcomes.items() if s != "draw"):
                    continue  # the other side is at a different line: the margin can't be measured
                hold = sum(1 / o["price"] for o in outcomes.values()) - 1
                if hold > max_hold:
                    continue
            prices[book["title"]] = offer["price"]
        return prices
