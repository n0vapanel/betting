"""NBA game and player predictions (model v1).

Game model: each team's points scored and allowed per game (this season,
shrunk toward last season while the sample is small) give an expected
score, adjusted for home court, back-to-backs and injured players. That
estimate is blended with the betting market's consensus (spread, total,
margin-free moneyline) and ESPN's matchup predictor, because the market
is hard to beat on its own; the blend weights live in ``Params``. Margins
and totals are treated as normally distributed around the prediction to
get win, cover and over/under probabilities.

Player model: projected minutes (season averages blended with last
season, injured teammates' minutes shared across the rotation) times
per-minute rates, with points scaled so the players add up to the team's
predicted score.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Dict, List, Optional

_NORMAL = NormalDist()
STATS = ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov")


@dataclass(frozen=True)
class Params:
    home_court: float = 2.0  # points; ~2.05 implied by 2022-25 closing lines (was ~3.5 historically)
    back_to_back: float = 1.25  # points lost on the second night of a back-to-back
    team_prior_games: float = 10.0  # this season's games that weigh as much as last season
    player_prior_games: float = 8.0
    # Margin lost per point an absent player scores (scaled by minutes / 36): a 31-point star is
    # worth ~4 points, in line with market moves for stars (1.5-5.5 points by depth)
    injury_impact: float = 0.15
    margin_sd: float = 12.5  # spread of final margins around the prediction (12-13 in recent eras)
    total_sd: float = 18.0  # spread of final totals around the prediction
    # Shares of the predicted margin: the closing market beats public NBA models, and blends
    # that improved on it still weighted the market 0.60-0.74; the rest is our team model
    market_weight: float = 0.65
    espn_weight: float = 0.10
    market_total_weight: float = 0.6  # share of the market in the predicted total
    team_minutes: float = 240.0
    max_minutes: float = 38.0
    # Starters sit late in lopsided games: they lose this share of their minutes per point
    # of expected margin beyond blowout_margin (RotoGrinders' rule of thumb), the bench gains it
    blowout_margin: float = 7.0
    blowout_cut: float = 0.015


def season_label(day: datetime.date) -> str:
    """ESPN's label for the NBA season containing ``day``, e.g. "2026-27"."""
    start = day.year if day.month >= 8 else day.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def blend(current: Optional[float], current_n: float, prior: Optional[float], prior_n: float) -> Optional[float]:
    """Weighted mean of this season's value and last season's (the prior)."""
    if current is None and prior is None:
        return None
    if prior is None:
        return current
    if current is None or current_n <= 0:
        return prior
    return (current * current_n + prior * prior_n) / (current_n + prior_n)


@dataclass
class TeamRating:
    points_for: float
    points_against: float
    games: float  # this season


def team_rating(current: Optional[Dict[str, Any]], previous: Optional[Dict[str, Any]], params: Params) -> Optional[TeamRating]:
    games = (current or {}).get("games") or 0
    values = {}
    for key in ("points_for", "points_against"):
        now = (current or {}).get(key) if games else None
        before = (previous or {}).get(key)
        values[key] = blend(now, games, before, params.team_prior_games)
    if values["points_for"] is None or values["points_against"] is None:
        return None
    return TeamRating(values["points_for"], values["points_against"], games)


def league_average(standings: Dict[str, Dict[str, Any]]) -> Optional[float]:
    values = [team["points_for"] for team in standings.values() if team.get("games") and team.get("points_for")]
    return sum(values) / len(values) if values else None


def player_profile(seasons: List[Dict[str, Any]], current_label: str, params: Params) -> Optional[Dict[str, float]]:
    """A player's expected per-game line: this season blended with last season."""
    by_label = {row["season"]: row for row in seasons}
    current = by_label.get(current_label)
    start = int(current_label[:4])
    previous = by_label.get(f"{start - 1}-{start % 100:02d}")
    if not current and not previous:
        return None
    gp = current["gp"] if current else 0.0
    profile = {"gp": gp}
    for key in ("min",) + STATS:
        profile[key] = blend(
            current[key] if current else None, gp,
            previous[key] if previous else None, params.player_prior_games if previous else 0,
        ) or 0.0
    return profile


def absence_impact(profile: Dict[str, float], team_games: float, params: Params) -> float:
    """Margin points a team loses without this player.

    The part of his absence already reflected in this season's team
    numbers (games he has missed) is not counted twice.
    """
    impact = params.injury_impact * profile["pts"] * min(profile["min"], 36.0) / 36.0
    if team_games > 0:
        missed_share = 1 - min(profile["gp"] / team_games, 1.0)
        already_counted = missed_share * team_games / (team_games + params.team_prior_games)
        impact *= 1 - already_counted
    return impact


@dataclass
class GamePrediction:
    home_score: float
    away_score: float
    margin: float  # home minus away
    total: float
    home_win_prob: float
    sources: Dict[str, Dict[str, float]] = field(default_factory=dict)
    factors: List[str] = field(default_factory=list)
    margin_sd: float = 12.5
    total_sd: float = 18.0

    def cover_prob(self, home_point: float) -> float:
        """Chance the home team covers a spread of ``home_point`` (e.g. -3.5)."""
        return _NORMAL.cdf((self.margin + home_point) / self.margin_sd)

    def over_prob(self, line: float) -> float:
        return 1 - _NORMAL.cdf((line - self.total) / self.total_sd)

    def probability(self, market: str, side: str, point: Optional[float] = None) -> Optional[float]:
        """Model chance of one bet winning: market h2h/spreads/totals, side home/away/over/under."""
        if market == "h2h":
            return {"home": self.home_win_prob, "away": 1 - self.home_win_prob}.get(side)
        if market == "spreads" and point is not None:
            if side == "home":
                return self.cover_prob(point)
            if side == "away":
                return 1 - self.cover_prob(-point)
        if market == "totals" and point is not None:
            if side == "over":
                return self.over_prob(point)
            if side == "under":
                return 1 - self.over_prob(point)
        return None


@dataclass
class TeamInputs:
    name: str
    rating: Optional[TeamRating]
    injury_impact: float = 0.0
    injured: List[str] = field(default_factory=list)
    back_to_back: bool = False


def _prob_to_margin(probability: float, params: Params) -> float:
    probability = min(max(probability, 0.01), 0.99)
    return params.margin_sd * _NORMAL.inv_cdf(probability)


def predict_game(
    home: TeamInputs,
    away: TeamInputs,
    league_avg: Optional[float],
    market: Dict[str, Dict[str, Dict[str, Any]]],
    espn_home_win: Optional[float],
    params: Params = Params(),
    level: Optional[float] = None,
) -> Optional[GamePrediction]:
    """Blend our team model, the market and ESPN's predictor into one prediction.

    ``league_avg`` is the average team score in our standings data, the
    baseline each team's strengths are measured from; ``level`` (when known)
    is the average team score today's market totals imply, so the model's
    scoring level matches the current game rather than a blend of seasons.
    """
    sources: Dict[str, Dict[str, float]] = {}
    factors: List[str] = []
    margins: List[tuple] = []

    model_total = None
    if home.rating and away.rating and league_avg:
        base = level or league_avg
        home_pts = base + (home.rating.points_for - league_avg) + (away.rating.points_against - league_avg)
        away_pts = base + (away.rating.points_for - league_avg) + (home.rating.points_against - league_avg)
        home_pts += params.home_court / 2
        away_pts -= params.home_court / 2
        factors.append(f"Home court: {home.name} +{params.home_court:.1f}")
        for team, is_home in ((home, True), (away, False)):
            loss = team.injury_impact + (params.back_to_back if team.back_to_back else 0.0)
            if not loss:
                continue
            # Most of the loss comes off the team's own scoring, some is defensive
            own, other = 0.7 * loss, 0.3 * loss
            if is_home:
                home_pts, away_pts = home_pts - own, away_pts + other
            else:
                away_pts, home_pts = away_pts - own, home_pts + other
            if team.back_to_back:
                factors.append(f"{team.name} on a back-to-back: -{params.back_to_back:.1f}")
            if team.injury_impact:
                factors.append(f"{team.name} missing {', '.join(team.injured)}: -{team.injury_impact:.1f}")
        model_margin = home_pts - away_pts
        model_total = home_pts + away_pts
        sources["model"] = {
            "home_points": home_pts,
            "away_points": away_pts,
            "margin": model_margin,
            "total": model_total,
            "home_win_prob": _NORMAL.cdf(model_margin / params.margin_sd),
        }
        margins.append((model_margin, 1 - params.market_weight - params.espn_weight))

    market_margin = None
    spread = market.get("spreads", {}).get("home")
    moneyline = market.get("h2h", {}).get("home")
    if spread and spread.get("point") is not None:
        market_margin = -spread["point"]
    elif moneyline and moneyline.get("fair"):
        market_margin = _prob_to_margin(moneyline["fair"], params)
    market_total = (market.get("totals", {}).get("over") or {}).get("point")
    if market_margin is not None:
        sources["market"] = {"margin": market_margin, "home_win_prob": _NORMAL.cdf(market_margin / params.margin_sd)}
        if moneyline and moneyline.get("fair"):
            sources["market"]["home_win_prob"] = moneyline["fair"]
        if market_total:
            sources["market"]["total"] = market_total
        margins.append((market_margin, params.market_weight))
    if espn_home_win is not None:
        sources["espn"] = {"home_win_prob": espn_home_win, "margin": _prob_to_margin(espn_home_win, params)}
        margins.append((sources["espn"]["margin"], params.espn_weight))

    if not margins:
        return None
    weight = sum(w for _, w in margins)
    margin = sum(m * w for m, w in margins) / weight
    if market_total and model_total:
        total = params.market_total_weight * market_total + (1 - params.market_total_weight) * model_total
    else:
        total = market_total or model_total
    if total is None:
        # No total from anyone: a typical NBA game total keeps the score display sensible
        total = 2 * (league_avg or 115.0)
    return GamePrediction(
        home_score=(total + margin) / 2,
        away_score=(total - margin) / 2,
        margin=margin,
        total=total,
        home_win_prob=_NORMAL.cdf(margin / params.margin_sd),
        sources=sources,
        factors=factors,
        margin_sd=params.margin_sd,
        total_sd=params.total_sd,
    )


@dataclass
class PlayerProjection:
    id: str
    name: str
    position: str
    status: str
    minutes: float
    pts: float = 0.0
    reb: float = 0.0
    ast: float = 0.0
    fg3m: float = 0.0
    stl: float = 0.0
    blk: float = 0.0
    tov: float = 0.0
    out: bool = False


def project_players(
    players: List[Dict[str, Any]],
    team_points: Optional[float],
    pace_factor: float,
    params: Params = Params(),
    margin: float = 0.0,
) -> List[PlayerProjection]:
    """Project each player's stat line for this game.

    ``players`` items have id, name, position, status, out (bool) and
    profile (from ``player_profile``, may be None for players with no data).
    ``margin`` is the predicted final margin, used for blowout minutes.
    """
    available = [p for p in players if not p["out"] and p.get("profile") and p["profile"]["min"] >= 4]
    available.sort(key=lambda p: p["profile"]["min"], reverse=True)
    # Fill the team's 240 minutes from the top of the rotation down: regulars keep their usual
    # minutes and a crowded roster squeezes the end of the bench, not the stars
    minutes: Dict[str, float] = {}
    remaining = params.team_minutes
    for p in available:
        minutes[p["id"]] = min(p["profile"]["min"], params.max_minutes, remaining)
        remaining -= minutes[p["id"]]
    # Minutes left over (injuries thinned the rotation) go to the players who are on the floor,
    # in proportion to their usual playing time, up to the per-player cap
    for _ in range(5):
        uncapped = [pid for pid, m in minutes.items() if 0 < m < params.max_minutes]
        if remaining < 0.5 or not uncapped:
            break
        uncapped_total = sum(minutes[pid] for pid in uncapped)
        for pid in uncapped:
            minutes[pid] = min(minutes[pid] + remaining * minutes[pid] / uncapped_total, params.max_minutes)
        remaining = params.team_minutes - sum(minutes.values())

    cut = params.blowout_cut * max(0.0, abs(margin) - params.blowout_margin)
    starters = [p["id"] for p in available[:5]]
    bench = [p for p in available[5:]]
    if cut and bench:
        freed = 0.0
        for pid in starters:
            freed += minutes[pid] * cut
            minutes[pid] *= 1 - cut
        bench_usual = sum(p["profile"]["min"] for p in bench)
        for p in bench:
            minutes[p["id"]] = min(minutes[p["id"]] + freed * p["profile"]["min"] / bench_usual, params.max_minutes)

    projections = []
    for p in available:
        if minutes[p["id"]] <= 0:
            continue
        profile = p["profile"]
        share = minutes[p["id"]] / profile["min"]
        projections.append(PlayerProjection(
            id=p["id"], name=p["name"], position=p["position"], status=p["status"],
            minutes=minutes[p["id"]], **{stat: profile[stat] * share for stat in STATS},
        ))
    pace = min(max(pace_factor, 0.92), 1.08)
    projected_points = sum(proj.pts for proj in projections)
    scoring = 1.0
    if team_points and projected_points:
        scoring = min(max(team_points / projected_points, 0.85), 1.15)
    for proj in projections:
        proj.pts *= scoring
        proj.fg3m *= scoring
        for stat in ("reb", "ast", "stl", "blk", "tov"):
            setattr(proj, stat, getattr(proj, stat) * pace)
    projections.sort(key=lambda proj: proj.minutes, reverse=True)
    for p in players:
        if p["out"]:
            projections.append(PlayerProjection(id=p["id"], name=p["name"], position=p["position"],
                                                status=p["status"], minutes=0.0, out=True))
    return projections
