"""Everything the dashboard shows for a league, a game or a bet slip.

Joins the odds feed with ESPN data and the NBA model: each game gets its
best prices, the market's margin-free probabilities and, for the NBA, the
model's predicted score, win/cover/over chances, injuries and player
projections. Bet slips are always priced here from the cached odds, never
from prices the browser sends.
"""

from __future__ import annotations

import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from ..data.cache import TTLCache
from ..data.espn_client import ESPN_DAY_TZ, LEAGUES as ESPN_LEAGUES, OUT_STATUSES, EspnClient
from ..data.leagues import SPORTS, pick_leagues
from ..data.odds_service import OddsService, summarize_markets
from ..data.store import Store
from ..engine import bets, calls, nba_model
from ..engine.odds_math import decimal_to_american, format_american
from ..engine.slip import Leg, evaluate_slip

MODEL_VERSION = "v1"
# Leagues with the full model (predicted score, players); others show the market's view
AI_LEAGUES = {"basketball_nba"}


class NotFound(Exception):
    pass


class CallRejected(Exception):
    """A call that can't be saved (game started, no prediction yet...); the message says why."""


def _parse_time(value: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def outcome_label(game: Dict[str, Any], market: str, side: str, point: Optional[float]) -> str:
    """How a pick reads on a bet slip, e.g. "Oklahoma City Thunder -13.5" or "Over 224.5"."""
    team = {"home": game["home"], "away": game["away"], "draw": "Draw"}.get(side, side.title())
    if market == "h2h":
        return team if side == "draw" else f"{team} to win"
    if market == "spreads" and point is not None:
        return f"{team} {point:+g}"
    if market == "totals" and point is not None:
        return f"{side.title()} {point:g}"
    return team


class DashboardService:
    def __init__(
        self,
        odds: OddsService,
        espn: EspnClient,
        tz: ZoneInfo,
        params: nba_model.Params = nba_model.Params(),
        workers: int = 8,
        store: Optional[Store] = None,
    ) -> None:
        self.odds = odds
        self.espn = espn
        self.tz = tz
        self.params = params
        self.cache = TTLCache()
        self.pool = ThreadPoolExecutor(max_workers=workers)
        # People's predictions and the AI's track record; None turns those features off
        self.store = store

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
            return {
                "league": league,
                "title": next((s["title"] for s in self.odds.sports() if s.get("key") == league), league),
                "ai": league in AI_LEAGUES,
                "games": cards,
                "updated": datetime.datetime.now(self.tz).strftime("%-I:%M %p %Z"),
            }

        result = self.cache.get(("games", league), 60, load)
        loaded = self.odds.odds_loaded_at(league)
        odds_as_of = (
            datetime.datetime.fromtimestamp(loaded, self.tz).strftime("%-I:%M %p %Z") if loaded else None
        )
        return {**result, "quota": dict(self.odds.quota), "model": MODEL_VERSION, "odds_as_of": odds_as_of}

    def game(self, league: str, event_id: str) -> Dict[str, Any]:
        """One game in full: all bookmakers' odds, prediction breakdown and player projections."""
        game = next((g for g in self.odds.league_games(league) if g["id"] == event_id), None)
        if game is None:
            raise NotFound(f"No game {event_id} in {league}")
        context = self._league_context(league)
        card, info = self._card(league, game, context)
        card["books"] = [self._book_view(book) for book in game["books"]]
        if league in AI_LEAGUES and info.get("prediction") and info.get("match"):
            card["players"] = self._players(league, info, context)
        return card

    def _league_context(self, league: str) -> Optional[Dict[str, Any]]:
        """Standings this season and last, for the team model (NBA only)."""
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
            return {"current": current, "previous": previous, "avg": league_avg,
                    "label": nba_model.season_label(today)}

        try:
            return self.cache.get(("context", league), 1800, load)
        except Exception as exc:
            logging.warning("No ESPN standings for %s: %s", league, exc)
            return None

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
                rating = nba_model.team_rating(context["current"].get(team_id), context["previous"].get(team_id), self.params)
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
            teams["home"], teams["away"], (context or {}).get("avg"), summary, info.get("espn_home_win"), self.params
        )
        return info

    def _market_view(self, game: Dict[str, Any], summary: Dict[str, Any], prediction: Optional[nba_model.GamePrediction]) -> Dict[str, Any]:
        view: Dict[str, Any] = {}
        for market, sides in summary.items():
            view[market] = {}
            for side, offer in sides.items():
                chance = prediction.probability(market, side, offer["point"]) if prediction else offer["fair"]
                view[market][side] = {
                    "price": offer["price"],
                    "american": format_american(decimal_to_american(offer["price"])) if offer["price"] > 1 else None,
                    "point": offer["point"],
                    "book": offer["book"],
                    "books": offer["books"],
                    "fair": offer["fair"],
                    "ai": chance,
                    "ev": chance * offer["price"] - 1 if chance is not None else None,
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
            "factors": prediction.factors,
        }

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
        return {"title": book["title"], "markets": markets}

    def _players(self, league: str, info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        match, prediction, teams = info["match"], info["prediction"], info["teams"]
        out: Dict[str, List[Dict[str, Any]]] = {}
        for side in ("home", "away"):
            team_id = match[f"{side}_id"]
            try:
                roster = self.espn.roster(league, team_id)
            except Exception as exc:
                logging.warning("No roster for team %s: %s", team_id, exc)
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
            normal_total = rating.points_for + rating.points_against if rating else None
            pace = prediction.total / normal_total if normal_total else 1.0
            projections = nba_model.project_players(players, team_points, pace, self.params, margin=prediction.margin)
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

    def _detail(self, league: str, event_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """A game's card, the prediction behind it, and its players' projections by player id."""
        game = next((g for g in self.odds.league_games(league) if g["id"] == event_id), None)
        if game is None:
            raise NotFound(f"No game {event_id} in {league}")
        context = self._league_context(league)
        card, info = self._card(league, game, context)
        players: Dict[str, Dict[str, Any]] = {}
        if league in AI_LEAGUES and info.get("prediction") and info.get("match") and context:
            for side, rows in self._players(league, info, context).items():
                for row in rows:
                    if not row.get("out"):
                        players[str(row["id"])] = {**row, "side": side}
        return card, info, players

    def judge(self, league: str, event_id: str, call: Dict[str, Any]) -> Dict[str, Any]:
        """What the AI thinks of a call, part by part (shown live while someone makes it)."""
        card, info, players = self._detail(league, event_id)
        prediction = info.get("prediction")
        if prediction is None:
            raise CallRejected("Calls are open for NBA games for now: the AI has no prediction for this game.")
        return calls.judge_call(prediction, call, card["home"], card["away"], players)

    def save_call(self, player: str, name: str, league: str, event_id: str, call: Dict[str, Any]) -> Dict[str, Any]:
        """Save someone's call on a game (replacing their earlier one) with the AI's view of it."""
        if self.store is None:
            raise CallRejected("Saving calls needs the server's state folder.")
        card, info, players = self._detail(league, event_id)
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
                    },
                })
            except Exception:
                logging.exception("Couldn't log the AI prediction for %s", card["id"])

    def _final(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """A finished game's score from ESPN (state final, void or pending)."""
        espn_id = row.get("espn_id")
        if not espn_id and row["league"] in ESPN_LEAGUES:
            match = self.espn.find_game(row["league"], row["home"], row["away"], row["start"])
            espn_id = match["event_id"] if match else None
        if not espn_id:
            return {"state": "pending"}
        return {**self.espn.game_result(row["league"], espn_id), "espn_id": espn_id}

    def grade_due(self) -> int:
        """Score calls and AI predictions on games that have finished; returns how many were graded.

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
                final = self._final(row)
                if final["state"] == "pending":
                    if row["start"] < give_up:
                        self.store.grade_prediction(row["id"], "void", {"reason": "no result"}, None, None)
                    continue
                if final["state"] == "void":
                    self.store.grade_prediction(row["id"], "void", final, None, None)
                    continue
                box = self.espn.box_score(row["league"], final["espn_id"]) if row["call"].get("props") else None
                result = calls.grade_call(row["call"], row["ai"], final, box)
                self.store.grade_prediction(row["id"], "graded", result, result["points"], result["ai_points"])
                graded += 1
            except Exception:
                logging.exception("Couldn't grade call %s", row["id"])
        for row in self.store.ai_predictions(status="pending", started_before=cutoff):
            try:
                final = self._final(row)
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
        return graded

    def track_record(self, league: str = "basketball_nba") -> Dict[str, Any]:
        """How the AI's pre-game predictions did: accuracy, Brier score next to the market's,
        record against the closing spread and total, and calibration."""
        if self.store is None:
            return {"games": 0}
        self.grade_due()
        rows = self.store.ai_predictions(league=league, status="graded")
        return calls.track_record(rows)

    # --- AI-built bets: safe, risky, crazy --------------------------------------

    def ai_bets(self, league: str, source: str = "all", player: Optional[str] = None) -> Dict[str, Any]:
        """Bets at three risk levels from the league's upcoming games, or only from someone's calls."""
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
                    prices = self._book_prices(league, card["id"], market, side, offer["point"])
                    if prices:
                        candidates.append(bets.Candidate(
                            event_id=card["id"], game=f"{card['away']} @ {card['home']}", market=market, side=side,
                            point=offer["point"], label=offer["label"], probability=offer["ai"], fair=offer["fair"],
                            prices=prices,
                        ))
        if league in AI_LEAGUES:
            note = ("Built from the AI's probabilities and the best price at one bookmaker per bet. "
                    "Most multis lose money over time: the bookmaker's edge grows with every leg.")
        else:
            note = ("No AI model for this league yet, so these use the bookmakers' own chances: "
                    "expect slightly negative value on all of them.")
        if allowed is not None and not allowed:
            note = "You have no open calls in this league yet: make some in Games, then come back."
        return {"league": league, "source": source, "note": note, "tiers": bets.build_all(candidates)}

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

    def _book_prices(self, league: str, event_id: str, market: str, side: str, point: Optional[float]) -> Dict[str, float]:
        """Each bookmaker's decimal price for one pick, at the same line as the pick."""
        game = next((g for g in self.odds.league_games(league) if g["id"] == event_id), None)
        prices = {}
        for book in (game or {}).get("books", []):
            offer = book["markets"].get(market, {}).get(side)
            if offer and offer["point"] == point:
                prices[book["title"]] = offer["price"]
        return prices
