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
from ..engine import nba_model
from ..engine.odds_math import decimal_to_american, format_american
from ..engine.slip import Leg, evaluate_slip

MODEL_VERSION = "v1"
# Leagues with the full model (predicted score, players); others show the market's view
AI_LEAGUES = {"basketball_nba"}


class NotFound(Exception):
    pass


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
    ) -> None:
        self.odds = odds
        self.espn = espn
        self.tz = tz
        self.params = params
        self.cache = TTLCache()
        self.pool = ThreadPoolExecutor(max_workers=workers)

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
            cards = list(self.pool.map(lambda game: self._card(league, game, context)[0], games))
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
