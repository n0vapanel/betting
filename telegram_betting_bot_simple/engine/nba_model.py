"""NBA game and player predictions (model v1).

Game model: each team's points scored and allowed per game (this season,
shrunk toward last season's, itself regressed 38% to the mean, while the
sample is small) give an expected score, adjusted for home court,
back-to-backs and injured players. That estimate is blended with the
betting market's consensus (spread, total, margin-free moneyline) and
ESPN's matchup predictor. On 2011-22 NBA closing lines a results-only team
model added nothing on top of the market, so the market carries 90% of the
blend whenever there is a line; the team model leads only without one.
Margins and totals are treated as normally distributed around the
prediction, with spreads that grow with the expected total (fitted on the
same data), to get win, cover and over/under probabilities; whole-number
lines can push.

Player model (fitted on 2025-26 player games and open-source projection
repos; see research_notes): minutes blended with last season's (this
season counts n/(n+4)), per-minute rates shrunk toward last season with
stat-specific padding, the team's 241 minutes filled from the top of the
rotation down, injured teammates' minutes only 40% to the rotation plus a
usage bump for players in the same position group, blowout minutes cut
from the favourite's starters, and pace, the team's predicted score and
home court applied at half strength.

Every number comes with its explanation (engine/factors.py): the game's margin, total and win chance, the
team model's own margin and total, and each player's stat line are Projections, a labelled baseline plus
typed factors that sum to the value shown. ``GamePrediction.factors`` keeps the old text lines
(``factor_text``) for older callers.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field, replace
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .consensus import Source, make_source
from .factors import NOISY, Factor, Projection, factor_lines, project, split_by_share

_NORMAL = NormalDist()
STATS = ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov")


@dataclass(frozen=True)
class Params:
    home_court: float = 2.0  # points; ~2.05 implied by 2022-25 closing lines (was ~3.5 historically)
    back_to_back: float = 1.25  # points lost on the second night of a back-to-back
    team_prior_games: float = 10.0  # this season's games that weigh as much as last season
    # Share of last season's strength (above or below average) that carries over: betting
    # markets imply 62% for the NBA (Lopez, Matthews & Baumer 2018)
    season_carryover: float = 0.62
    # Minutes: this season counts n/(n + 4) against last season's minutes per game
    minutes_prior_games: float = 4.0
    # Margin lost per point an absent player scores (scaled by minutes / 36): a 31-point star is
    # worth ~4 points, in line with market moves for stars (1.5-5.5 points by depth)
    injury_impact: float = 0.15
    # Spread of results around the closing line, fitted on ~13,900 NBA games (2011-22): it grows with
    # the expected total. Final margins: 0.86 x sqrt(total) (13.0 at 230); totals: 1.24 x sqrt(total).
    # Win chance from a margin uses 0.80 x sqrt(total) (12.1 at 230): results bunch near the spread.
    margin_sd_factor: float = 0.86
    win_sd_factor: float = 0.80
    total_sd_factor: float = 1.24
    # Shares of the predicted margin and total. A results-only team model (Elo, least squares or one
    # like ours) added nothing over the closing line in 2017-22 tests (our old 65/10/25 blend raised
    # log loss from 0.5978 to 0.6007), so the market dominates when there is a line
    market_weight: float = 0.90
    espn_weight: float = 0.05
    market_total_weight: float = 0.90
    team_minutes: float = 241.0  # 240 plus the average overtime
    max_minutes: float = 38.0
    # Of the minutes an injured rotation player leaves, the tracked rotation absorbs only 40% (100%
    # overshot by 2.5 minutes a player); the rest goes to the end of the bench
    injury_minutes_fill: float = 0.4
    rotation_size: int = 9
    # Starters sit late in lopsided games (2025-26 minutes by pre-game spread): the favourite's lose
    # 0.77% of their minutes per point beyond 6 (up to 15%), the underdog's 0.25% per point beyond 9;
    # the deep bench gets those minutes
    blowout_margin: float = 6.0
    blowout_cut: float = 0.0077
    blowout_cap: float = 0.15
    underdog_margin: float = 9.0
    underdog_cut: float = 0.0025


TUNABLE = ("market_weight", "espn_weight", "market_total_weight")  # the fields learn/registry.py may override


def params_with(overrides: Optional[Mapping[str, Any]] = None, base: Optional[Params] = None) -> Params:
    """``base`` (default ``Params()``) with the learning loop's tuned values (learn/registry.py) for the market and
    ESPN weights; ``base`` itself when nothing differs. ValueError for another field, a weight outside 0-1, or market
    and ESPN weights summing past 1."""
    params = base if base is not None else Params()
    changes: Dict[str, float] = {}
    for name, value in (overrides or {}).items():
        if name not in TUNABLE:
            raise ValueError(f"{name!r} is not a tunable NBA model parameter")
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value!r}")
        if number != getattr(params, name):
            changes[name] = number
    if not changes:
        return params
    out = replace(params, **changes)
    if out.market_weight + out.espn_weight > 1.0 + 1e-12:
        raise ValueError("market_weight + espn_weight must not exceed 1")
    return out


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


def team_rating(current: Optional[Dict[str, Any]], previous: Optional[Dict[str, Any]], params: Params,
                averages: Optional[Tuple[Optional[float], Optional[float], float]] = None) -> Optional[TeamRating]:
    """A team's points scored and allowed per game: this season, shrunk toward last season.

    With ``averages`` (this season's league average, last season's, and the
    blended average the model works from) each season counts as distance from
    its own league average, and last season's distance is regressed by
    ``season_carryover``, since teams change between seasons.
    """
    games = (current or {}).get("games") or 0
    values = {}
    for key in ("points_for", "points_against"):
        now = (current or {}).get(key) if games else None
        before = (previous or {}).get(key)
        if averages and averages[2] is not None:
            current_avg, previous_avg, league_avg = averages
            now = now - current_avg + league_avg if now is not None and current_avg else None
            if before is not None and previous_avg:
                before = league_avg + params.season_carryover * (before - previous_avg)
            else:
                before = None
        values[key] = blend(now, games, before, params.team_prior_games)
    if values["points_for"] is None or values["points_against"] is None:
        return None
    return TeamRating(values["points_for"], values["points_against"], games)


def league_average(standings: Dict[str, Dict[str, Any]]) -> Optional[float]:
    values = [team["points_for"] for team in standings.values() if team.get("games") and team.get("points_for")]
    return sum(values) / len(values) if values else None


# Per-minute rates are shrunk toward last season's with this much "padding", in minutes (stabilisation
# points from Medvedovsky's table converted to minutes): steals take ~300 minutes to say much
RATE_PADDING = {"pts": 45.0, "reb": 30.0, "ast": 30.0, "fg3m": 15.0, "stl": 300.0, "blk": 75.0, "tov": 200.0}


def player_profile(seasons: List[Dict[str, Any]], current_label: str, params: Params) -> Optional[Dict[str, float]]:
    """A player's expected per-game line: minutes and per-minute rates, this season shrunk toward last."""
    by_label = {row["season"]: row for row in seasons}
    current = by_label.get(current_label)
    start = int(current_label[:4])
    previous = by_label.get(f"{start - 1}-{start % 100:02d}")
    if not current and not previous:
        return None
    gp = current["gp"] if current and current.get("min") else 0.0
    minutes = blend(current["min"] if gp else None, gp,
                    previous["min"] if previous else None, params.minutes_prior_games if previous else 0) or 0.0
    profile = {"gp": gp, "min": minutes}
    season_minutes = current["min"] * gp if gp else 0.0
    for stat in STATS:
        prior = previous[stat] / previous["min"] if previous and previous.get("min") else None
        if gp and prior is not None:
            rate = (current[stat] * gp + RATE_PADDING[stat] * prior) / (season_minutes + RATE_PADDING[stat])
        elif gp:
            rate = current[stat] / current["min"]
        else:
            rate = prior or 0.0
        profile[stat] = rate * minutes
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
    factors: List[str] = field(default_factory=list)  # text lines for older callers (see factor_text)
    margin_sd: float = 13.0
    total_sd: float = 18.8
    # The typed explanations as predict_game made them: "margin", "total", "home_win" (percent) and, when the
    # team model ran, "model_margin" and "model_total". Read them through projections(), which also
    # accounts for anything that moved the numbers afterwards (the AI analysts' desk)
    base_projections: Dict[str, Projection] = field(default_factory=dict, repr=False, compare=False)

    def projections(self) -> Dict[str, Projection]:
        """The prediction's numbers as Projections (baseline + typed factors = value), by key.

        A later change to the margin, total or win chance (ai/desk.py moves them by the analysts' trusted
        forecasts) shows as one more factor, so every projection still sums to the number shown.
        """
        current = {"margin": self.margin, "total": self.total, "home_win": 100 * self.home_win_prob}
        out: Dict[str, Projection] = {}
        for key, projection in self.base_projections.items():
            target = current.get(key)
            if target is not None and abs(target - projection.value) > 1e-9:
                analysts = "before_analysts" in self.sources
                extra = Factor(
                    "analysts" if analysts else "adjustment",
                    "AI analysts, weighted by their graded record" if analysts else "Later adjustment",
                    target - projection.value, "pp" if key == "home_win" else projection.unit,
                    evidence="moved after the model ran" + ("; trust is earned from graded forecasts" if analysts else ""),
                    source="AI analyst desk" if analysts else "",
                )
                projection = replace(projection, factors=[*projection.factors, extra], value=target)
            out[key] = projection
        return out

    def _above(self, mean: float, sd: float, line: float) -> float:
        """P(result > line) for a whole-number result: a whole-number line pushes when hit exactly."""
        return 1 - _NORMAL.cdf((math.floor(line) + 0.5 - mean) / sd)

    def _below(self, mean: float, sd: float, line: float) -> float:
        return _NORMAL.cdf((math.ceil(line) - 0.5 - mean) / sd)

    def cover_prob(self, home_point: float) -> float:
        """Chance the home team covers a spread of ``home_point`` (e.g. -3.5); pushes don't count."""
        return self._above(self.margin, self.margin_sd, -home_point)

    def over_prob(self, line: float) -> float:
        return self._above(self.total, self.total_sd, line)

    def probability(self, market: str, side: str, point: Optional[float] = None) -> Optional[float]:
        """Model chance of one bet winning (a push is not a win): market h2h/spreads/totals,
        side home/away/over/under."""
        if market == "h2h":
            return {"home": self.home_win_prob, "away": 1 - self.home_win_prob}.get(side)
        if market == "spreads" and point is not None:
            if side == "home":
                return self.cover_prob(point)
            if side == "away":  # the away team at +point covers when home wins by less than point
                return self._below(self.margin, self.margin_sd, point)
        if market == "totals" and point is not None:
            if side == "over":
                return self.over_prob(point)
            if side == "under":
                return self._below(self.total, self.total_sd, point)
        return None

    def push_probability(self, market: str, side: str, point: Optional[float]) -> float:
        """Chance a spread or total bet is a push (only whole-number lines can be)."""
        if point is None or market not in ("spreads", "totals") or point != math.floor(point):
            return 0.0
        if market == "spreads":
            at = -point if side == "home" else point  # the final margin (home minus away) that pushes
            if at == 0:
                return 0.0  # NBA games can't end tied
            mean, sd = self.margin, self.margin_sd
        else:
            at, mean, sd = point, self.total, self.total_sd
        return _NORMAL.cdf((at + 0.5 - mean) / sd) - _NORMAL.cdf((at - 0.5 - mean) / sd)


@dataclass
class TeamInputs:
    name: str
    rating: Optional[TeamRating]
    injury_impact: float = 0.0
    injured: List[str] = field(default_factory=list)
    back_to_back: bool = False


def spreads_for(total: float, params: Params) -> Tuple[float, float, float]:
    """(margin SD, win-chance SD, total SD) for a game with this expected total."""
    root = math.sqrt(max(total, 150.0))
    return params.margin_sd_factor * root, params.win_sd_factor * root, params.total_sd_factor * root


def _prob_to_margin(probability: float, win_sd: float) -> float:
    probability = min(max(probability, 0.01), 0.99)
    return win_sd * _NORMAL.inv_cdf(probability)


# Where each factor's input comes from (Factor.source)
SRC_STANDINGS = "ESPN standings (this season shrunk toward last)"
SRC_PARAMS = "Model setting (fitted on closing lines)"
SRC_SCHEDULE = "ESPN schedule"
SRC_INJURIES = "ESPN injury report"
SRC_MODEL = "Team model"
SRC_ESPN = "ESPN matchup predictor"
EVEN_GAME = "Even teams on a neutral court"
# Factor keys factor_text renders in the old text format (the team model's home court, rest and injuries)
TEXT_KEYS = ("home_court", "home_back_to_back", "home_injuries", "away_back_to_back", "away_injuries")


def factor_text(factors: Iterable[Factor]) -> List[str]:
    """Text lines for older callers and the bot: the team model's home court, back-to-back and injury factors
    exactly as before ("Home court: Knicks +2.0", "Magic on a back-to-back: -1.2", "Magic missing X: -3.1"),
    any other factor as "Label: +0.4 pts"."""
    lines = []
    for f in factors:
        if f.key == "home_court":
            lines.append(f"{f.label} +{abs(f.delta):.1f}")
        elif f.key in TEXT_KEYS:
            lines.append(f"{f.label}: -{abs(f.delta):.1f}")
        else:
            lines.extend(factor_lines([f]))
    return lines


def _team_model(home: TeamInputs, away: TeamInputs, league_avg: float, base: float, params: Params,
                level: Optional[float]) -> Tuple[float, float, Projection, Projection]:
    """(home points, away points, the team model's margin and total as Projections)."""
    hr, ar = home.rating, away.rating
    if hr is None or ar is None:
        raise ValueError("The team model needs both teams' ratings")
    home_pts = base + (hr.points_for - league_avg) + (ar.points_against - league_avg)
    away_pts = base + (ar.points_for - league_avg) + (hr.points_against - league_avg)
    home_pts += params.home_court / 2
    away_pts -= params.home_court / 2

    def record(team: TeamInputs) -> str:
        rating = team.rating
        return (f"{team.name} score {rating.points_for:.1f} and allow {rating.points_against:.1f} a game, "
                f"net {rating.points_for - rating.points_against:+.1f} ({rating.games:g} games this season, "
                "shrunk toward last season)")

    margin_factors = [
        Factor("home_rating", f"{home.name} strength", hr.points_for - hr.points_against, "pts",
               evidence=record(home), source=SRC_STANDINGS),
        # Named for its effect on the home margin: a weak opponent (net -6.9) adds +6.9, a strong one takes away
        Factor("away_rating", f"Opponent: {away.name} (net {ar.points_for - ar.points_against:+.1f})",
               -(ar.points_for - ar.points_against), "pts", evidence=record(away), source=SRC_STANDINGS),
        Factor("home_court", f"Home court: {home.name}", params.home_court, "pts",
               evidence=f"{params.home_court:g} points, what 2022-25 closing lines imply (about 3.5 before)",
               source=SRC_PARAMS),
    ]
    total_factors = [
        Factor("home_scoring", f"{home.name} games", (hr.points_for - league_avg) + (hr.points_against - league_avg),
               "pts", evidence=f"{record(home)}, against a {league_avg:.1f} average", source=SRC_STANDINGS),
        Factor("away_scoring", f"{away.name} games", (ar.points_for - league_avg) + (ar.points_against - league_avg),
               "pts", evidence=f"{record(away)}, against a {league_avg:.1f} average", source=SRC_STANDINGS),
    ]
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
        side, sign = ("home", -1.0) if is_home else ("away", 1.0)
        # The total loses 0.4 of each point lost: 0.7 off the team's own scoring, 0.3 back to the opponent
        if team.back_to_back:
            label = f"{team.name} on a back-to-back"
            evidence = f"second night of a back-to-back: about {params.back_to_back:g} points"
            margin_factors.append(Factor(f"{side}_back_to_back", label, sign * params.back_to_back, "pts",
                                         evidence=evidence, source=SRC_SCHEDULE))
            total_factors.append(Factor(f"{side}_back_to_back", label, -0.4 * params.back_to_back, "pts",
                                        evidence=evidence + ", 70% of it off their own scoring", source=SRC_SCHEDULE))
        if team.injury_impact:
            label = f"{team.name} missing {', '.join(team.injured)}"
            evidence = (f"{team.injury_impact:.1f} points: their scoring weighted by minutes, less what this "
                        "season's numbers already show")
            margin_factors.append(Factor(f"{side}_injuries", label, sign * team.injury_impact, "pts",
                                         evidence=evidence, source=SRC_INJURIES))
            total_factors.append(Factor(f"{side}_injuries", label, -0.4 * team.injury_impact, "pts",
                                        evidence=evidence + ", 70% of it off their own scoring", source=SRC_INJURIES))
    level_label = "today's scoring level (market totals)" if level else "league scoring level"
    model_margin = Projection(EVEN_GAME, 0.0, margin_factors, home_pts - away_pts, "pts", "model_margin",
                              f"Team model: {home.name} margin")
    model_total = Projection(f"2 × {level_label}, {base:.1f}", 2 * base, total_factors, home_pts + away_pts, "pts",
                             "model_total", "Team model: total points")
    return home_pts, away_pts, model_margin, model_total


def _scaled(factors: Iterable[Factor], share: float, note: str) -> List[Factor]:
    """Factors times a blend share (the share is shown in the evidence when it is less than all)."""
    if share >= 1 - 1e-12:
        return list(factors)
    return [replace(f, delta=f.delta * share, evidence=f"{f.evidence}; ×{share:.0%} {note}".lstrip("; "))
            for f in factors]


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

    The margin, total and win chance are explained from the market's number
    when there is one (the team model and ESPN move it by their weights), and
    from an even game otherwise; see ``GamePrediction.projections``.
    """
    sources: Dict[str, Dict[str, float]] = {}
    projections: Dict[str, Projection] = {}

    model_margin = model_total = None
    if home.rating and away.rating and league_avg:
        base = level or league_avg
        home_pts, away_pts, projections["model_margin"], projections["model_total"] = _team_model(
            home, away, league_avg, base, params, level)
        model_margin = home_pts - away_pts
        model_total = home_pts + away_pts
        sources["model"] = {"home_points": home_pts, "away_points": away_pts, "margin": model_margin,
                            "total": model_total}

    spread = market.get("spreads", {}).get("home")
    moneyline = market.get("h2h", {}).get("home")
    market_total = (market.get("totals", {}).get("over") or {}).get("point")
    # The total first: the spread of results (and so every probability) grows with it
    if market_total and model_total:
        total = params.market_total_weight * market_total + (1 - params.market_total_weight) * model_total
        share = 1 - params.market_total_weight
        projections["total"] = Projection(f"Market total {market_total:g}", market_total, [Factor(
            "team_model", "Team model", share * (model_total - market_total), "pts",
            evidence=f"team model {model_total:.1f} vs market {market_total:g}; {share:.0%} weight",
            source=SRC_MODEL, confidence=NOISY)], total, "pts")
    else:
        total = market_total or model_total
        if market_total:
            projections["total"] = Projection(f"Market total {market_total:g}", market_total, [], total, "pts")
        elif model_total:
            projections["total"] = replace(projections["model_total"])
    if total is None:
        # No total from anyone: a typical NBA game total keeps the score display sensible
        total = 2 * (level or league_avg or 115.0)
        projections["total"] = Projection("Typical NBA game (no total from the market or the team model)", total,
                                          [], total, "pts")
    margin_sd, win_sd, total_sd = spreads_for(total, params)

    margins: List[Tuple[str, float, float]] = []
    if model_margin is not None:
        sources["model"]["home_win_prob"] = _NORMAL.cdf(model_margin / win_sd)
        margins.append(("model", model_margin, 1 - params.market_weight - params.espn_weight))
    market_margin = None
    if spread and spread.get("point") is not None:
        market_margin = -spread["point"]
    elif moneyline and moneyline.get("fair"):
        market_margin = _prob_to_margin(moneyline["fair"], win_sd)
    if market_margin is not None:
        sources["market"] = {"margin": market_margin, "home_win_prob": _NORMAL.cdf(market_margin / win_sd)}
        if moneyline and moneyline.get("fair"):
            sources["market"]["home_win_prob"] = moneyline["fair"]
        if market_total:
            sources["market"]["total"] = market_total
        margins.append(("market", market_margin, params.market_weight))
    if espn_home_win is not None:
        sources["espn"] = {"home_win_prob": espn_home_win, "margin": _prob_to_margin(espn_home_win, win_sd)}
        margins.append(("espn", sources["espn"]["margin"], params.espn_weight))

    if not margins:
        return None
    weight = sum(w for _, _, w in margins)
    margin = sum(m * w for _, m, w in margins) / weight
    home_win = _NORMAL.cdf(margin / win_sd)
    shares = {name: w / weight for name, _, w in margins}

    # The margin: from the market's when there is one (the others move it by their share of the blend),
    # else from an even game, built up from the team model's parts and ESPN's view
    anchor = market_margin if market_margin is not None else 0.0
    if market_margin is None:
        baseline_label = EVEN_GAME
    elif spread and spread.get("point") is not None:
        baseline_label = f"Market spread: {home.name} {spread['point']:+g}"
    else:
        baseline_label = f"Market moneyline ({moneyline['fair']:.0%}) as a margin"
    margin_factors: List[Factor] = []
    if model_margin is not None:
        if market_margin is not None:
            margin_factors.append(Factor(
                "team_model", "Team model", shares["model"] * (model_margin - market_margin), "pts",
                evidence=(f"{home.name} {model_margin:+.1f} in our team model vs {market_margin:+.1f} in the market; "
                          f"{shares['model']:.0%} weight (it adds little over a fresh line)"),
                source=SRC_MODEL, confidence=NOISY))
        else:
            margin_factors += _scaled(projections["model_margin"].factors, shares["model"], "team model weight")
    if espn_home_win is not None:
        espn_margin = sources["espn"]["margin"]
        margin_factors.append(Factor(
            "espn_predictor", "ESPN predictor", shares["espn"] * (espn_margin - anchor), "pts",
            evidence=(f"ESPN gives {home.name} {espn_home_win:.0%} (a {espn_margin:+.1f} margin); "
                      f"{shares['espn']:.0%} weight"), source=SRC_ESPN))
    projections["margin"] = Projection(baseline_label, anchor, margin_factors, margin, "pts", "margin",
                                       f"{home.name} margin (home minus away)")

    if moneyline and moneyline.get("fair") and market_margin is not None:
        # The moneyline's own margin-free chance is the market's best win estimate; the other sources
        # only move it by however far they move the margin
        fair = min(max(moneyline["fair"], 0.005), 0.995)
        start_z = _NORMAL.inv_cdf(fair)
        home_win = _NORMAL.cdf(start_z + (margin - market_margin) / win_sd)
        win_label = f"Market moneyline, no-vig {fair:.1%}"
    else:
        start_z = anchor / win_sd
        win_label = baseline_label if market_margin is None else f"{baseline_label}, as a chance"
    # Each factor's share of the move in normal-score units, converted to percentage points
    start = 100 * _NORMAL.cdf(start_z)
    parts = split_by_share(100 * home_win - start, {f.key: f.delta / win_sd for f in margin_factors},
                           slope=100 * math.exp(-start_z * start_z / 2) / math.sqrt(2 * math.pi))
    win_factors = [replace(f, delta=parts[f.key], unit="pp", evidence=f"{f.evidence} ({f.delta:+.1f} pts of margin)")
                   for f in margin_factors]
    projections["home_win"] = Projection(win_label, start, win_factors, 100 * home_win, "%", "home_win",
                                         f"{home.name} win chance")
    projections["total"] = replace(projections["total"], key="total", label="Game total")
    legacy = projections["model_margin"].factors if "model_margin" in projections else []
    return GamePrediction(
        home_score=(total + margin) / 2,
        away_score=(total - margin) / 2,
        margin=margin,
        total=total,
        home_win_prob=home_win,
        sources=sources,
        factors=factor_text(f for f in legacy if f.key in TEXT_KEYS),
        margin_sd=margin_sd,
        total_sd=total_sd,
        base_projections=projections,
    )


def model_source(prediction: Optional[GamePrediction], as_of: Any = None) -> Optional[Source]:
    """Our team model's own home-win chance as a consensus source ("Model", kind team_model), or None when
    the team model didn't run. It is the model alone, not the prediction (which already leans 90% on the
    market), so the market isn't counted twice in the blend."""
    model = (prediction.sources.get("model") or {}) if prediction else {}
    chance = model.get("home_win_prob")
    if chance is None:
        return None
    return make_source("Model", "team_model", {"home": chance, "away": 1 - chance}, as_of,
                       detail="team model: points for and against, home court, rest, injuries")


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


POSITION_GROUPS = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}
# Usage bump per unit of the missing teammate's share of the team's stat (same position group: full;
# others: half)
USAGE_BUMP = {"pts": 0.30, "ast": 0.30, "fg3m": 0.30, "reb": 0.175}


STAT_LABELS = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "Threes made", "stl": "Steals",
               "blk": "Blocks", "tov": "Turnovers"}
STAT_UNITS = {"pts": "pts", "reb": "reb", "ast": "ast", "fg3m": "3PM", "stl": "stl", "blk": "blk", "tov": "tov"}


def project_players(
    players: List[Dict[str, Any]],
    team_points: Optional[float],
    pace_factor: float,
    params: Params = Params(),
    margin: float = 0.0,
    home: Optional[bool] = None,
    team_games: float = 0.0,
) -> List[PlayerProjection]:
    """Project each player's stat line for this game.

    ``players`` items have id, name, position, status, out (bool) and
    profile (from ``player_profile``, may be None for players with no data).
    ``margin`` is the team's own expected margin (positive when favoured),
    used for blowout minutes; ``home`` adds a small home/away effect; and
    ``team_games`` (games played this season) says how much of an injured
    teammate's absence is already in everyone's numbers.
    """
    return _project_players(players, team_points, pace_factor, params, margin, home, team_games)[0]


def explain_players(
    players: List[Dict[str, Any]],
    team_points: Optional[float],
    pace_factor: float,
    params: Params = Params(),
    margin: float = 0.0,
    home: Optional[bool] = None,
    team_games: float = 0.0,
) -> Dict[str, Dict[str, Projection]]:
    """The same projections as ``project_players`` (same arguments), explained: {player id: {stat:
    Projection}} for every player who plays. Each starts from his usual line (usual minutes at this
    season's per-minute rate, shrunk toward last season's); minute changes come in as additive factors
    and usage, team scoring, pace and home/away as multipliers split by log share."""
    return _project_players(players, team_points, pace_factor, params, margin, home, team_games)[1]


def _project_players(
    players: List[Dict[str, Any]],
    team_points: Optional[float],
    pace_factor: float,
    params: Params,
    margin: float,
    home: Optional[bool],
    team_games: float,
) -> Tuple[List[PlayerProjection], Dict[str, Dict[str, Projection]]]:
    available = [p for p in players if not p["out"] and p.get("profile") and p["profile"]["min"] >= 4]
    available.sort(key=lambda p: p["profile"]["min"], reverse=True)
    rotation = {p["id"] for p in available[: params.rotation_size]}
    out = [p for p in players if p["out"] and p.get("profile") and p["profile"]["min"] >= 15]

    # Fill the team's minutes from the top of the rotation down: regulars keep their usual minutes
    # and a crowded roster squeezes the end of the bench, not the stars
    minutes: Dict[str, float] = {}
    remaining = params.team_minutes
    for p in available:
        minutes[p["id"]] = min(p["profile"]["min"], params.max_minutes, remaining)
        remaining -= minutes[p["id"]]
    filled = dict(minutes)  # each player's minutes before the open ones are handed out

    def share_out(amount: float, receivers: List[Dict[str, Any]]) -> float:
        """Give ``amount`` minutes to ``receivers`` in proportion to their usual minutes, up to the cap."""
        for _ in range(5):
            open_ = [p for p in receivers if minutes[p["id"]] < params.max_minutes]
            usual = sum(p["profile"]["min"] for p in open_)
            if amount < 0.25 or not usual:
                break
            given = 0.0
            for p in open_:
                extra = min(amount * p["profile"]["min"] / usual, params.max_minutes - minutes[p["id"]])
                minutes[p["id"]] += extra
                given += extra
            amount -= given
        return amount

    # Minutes left over (injuries thinned the rotation): only 40% go to the regular rotation, the rest
    # to the end of the bench (or to the rotation when there is no bench)
    open_minutes = remaining
    if remaining > 0.25:
        deep = [p for p in available if p["id"] not in rotation]
        left = share_out(remaining * params.injury_minutes_fill, [p for p in available if p["id"] in rotation])
        left += remaining * (1 - params.injury_minutes_fill)
        left = share_out(left, deep) if deep else left
        share_out(left, available)
    before_blowout = dict(minutes)

    # Blowouts: the favourite's starters sit late, the underdog's a little; the deep bench plays
    starters = [p for p in available[:5]]
    if margin >= 0:
        cut = min(params.blowout_cut * max(0.0, margin - params.blowout_margin), params.blowout_cap)
    else:
        cut = params.underdog_cut * max(0.0, -margin - params.underdog_margin)
    if cut:
        freed = 0.0
        for p in starters:
            freed += minutes[p["id"]] * cut
            minutes[p["id"]] *= 1 - cut
        bench = [p for p in available if p["id"] not in rotation] or available[5:]
        share_out(freed, bench)

    # Usage: a missing teammate's shots and passes go mostly to players at his position. If he has
    # already missed games this season, part of that is in everyone's numbers already
    bumps = {p["id"]: {stat: 1.0 for stat in USAGE_BUMP} for p in available}
    usage: Dict[str, Dict[str, List[Tuple[Dict[str, Any], float, str]]]] = {
        p["id"]: {stat: [] for stat in USAGE_BUMP} for p in available}
    totals = {stat: sum(p["profile"][stat] for p in available + out) for stat in USAGE_BUMP}
    for missing in out:
        played_share = min(missing["profile"]["gp"] / team_games, 1.0) if team_games else 1.0
        group = POSITION_GROUPS.get(missing.get("position", ""), "")
        for p in available:
            overlap = 1.0 if group and POSITION_GROUPS.get(p.get("position", "")) == group else 0.5
            for stat, alpha in USAGE_BUMP.items():
                if totals[stat]:
                    bump = 1 + alpha * missing["profile"][stat] / totals[stat] * overlap * played_share
                    bumps[p["id"]][stat] *= bump
                    note = "same position group" if overlap == 1.0 else "other position, half the bump"
                    if played_share < 1:
                        note += f"; he played {played_share:.0%} of the team's games, so the rest is already in"
                    usage[p["id"]][stat].append((missing, bump, note))

    projections = []
    for p in available:
        if minutes[p["id"]] <= 0:
            continue
        profile = p["profile"]
        share = minutes[p["id"]] / profile["min"]
        projections.append(PlayerProjection(
            id=p["id"], name=p["name"], position=p["position"], status=p["status"], minutes=minutes[p["id"]],
            **{stat: profile[stat] * share * bumps[p["id"]].get(stat, 1.0) for stat in STATS},
        ))
    # Game context at half strength (full-strength pace and score scaling overshot in testing)
    pace = min(max(pace_factor ** 0.5, 0.94), 1.06)
    projected_points = sum(proj.pts for proj in projections)
    scoring = 1.0
    if team_points and projected_points:
        scoring = min(max((team_points / projected_points) ** 0.5, 0.90), 1.10)
    home_points = {True: 1.007, False: 0.993}.get(home, 1.0)
    home_blocks = {True: 1.035, False: 0.965}.get(home, 1.0)
    home_turnovers = {True: 0.99, False: 1.01}.get(home, 1.0)
    for proj in projections:
        proj.pts *= scoring * home_points
        proj.fg3m *= scoring
        for stat in ("reb", "ast", "stl", "blk", "tov"):
            setattr(proj, stat, getattr(proj, stat) * pace)
        proj.blk *= home_blocks
        proj.tov *= home_turnovers

    # The same numbers as typed factors
    by_id = {p["id"]: p for p in available}
    venue = "Home floor" if home else "Road game"
    context: Dict[str, List[Tuple[str, str, float, str, str]]] = {stat: [] for stat in STATS}
    if scoring != 1.0:
        evidence = (f"team projected for {team_points:.1f} against {projected_points:.1f} from the player lines; "
                    "half strength, capped at ±10%")
        for stat in ("pts", "fg3m"):
            context[stat].append(("scoring", "Team's projected score", scoring, evidence, SRC_MODEL))
    if pace != 1.0:
        evidence = f"game pace ×{pace_factor:.3f} against the team's usual; half strength, capped at ±6%"
        for stat in ("reb", "ast", "stl", "blk", "tov"):
            context[stat].append(("pace", "Pace", pace, evidence, SRC_MODEL))
    for stat, factor in (("pts", home_points), ("blk", home_blocks), ("tov", home_turnovers)):
        if factor != 1.0:
            context[stat].append(("venue", venue, factor, f"×{factor:g} {'at home' if home else 'on the road'}",
                                  "Model setting (2025-26 home/away splits)"))
    explained: Dict[str, Dict[str, Projection]] = {}
    for proj in projections:
        p = by_id[proj.id]
        profile = p["profile"]
        usual = profile["min"]
        steps = (
            ("minutes_rotation", "Minutes: crowded rotation" if usual <= params.max_minutes else "Minutes: 38-minute cap",
             filled[proj.id] - usual, f"usual {usual:.1f} min, {filled[proj.id]:.1f} left for him"),
            ("minutes_open", "Minutes: teammates out" if out else "Minutes: thin roster",
             before_blowout[proj.id] - filled[proj.id], f"share of {open_minutes:.1f} open minutes"),
            ("minutes_blowout", "Minutes: blowout risk" if minutes[proj.id] < before_blowout[proj.id]
             else "Minutes: garbage time", minutes[proj.id] - before_blowout[proj.id],
             f"team expected to win by {margin:.1f}" if margin >= 0 else f"team expected to lose by {-margin:.1f}"),
        )
        explained[proj.id] = {}
        for stat in STATS:
            rate = profile[stat] / usual
            unit = STAT_UNITS[stat]
            additive = {key: (label, rate * change, f"{evidence}: {change:+.1f} min × {rate:.3g} {unit}/min",
                              "Team model (minutes)")
                        for key, label, change, evidence in steps if abs(change) > 1e-9}
            multipliers = {}
            for missing, bump, note in usage[proj.id].get(stat, []):
                if bump != 1.0:
                    multipliers[f"usage_{missing['id']}"] = (
                        f"{missing['name']} out: usage", bump,
                        f"he averages {missing['profile'][stat]:.1f} {unit}; {note}", SRC_INJURIES)
            for key, label, factor, evidence, source in context[stat]:
                multipliers[key] = (label, factor, evidence, source, NOISY if key == "venue" else "normal")
            explained[proj.id][stat] = project(
                f"Usual: {usual:.1f} min at this season's rate (shrunk toward last)", profile[stat],
                multipliers, additive, unit, stat, f"{proj.name}: {STAT_LABELS[stat]}", getattr(proj, stat))

    projections.sort(key=lambda proj: proj.minutes, reverse=True)
    for p in players:
        if p["out"]:
            projections.append(PlayerProjection(id=p["id"], name=p["name"], position=p["position"],
                                                status=p["status"], minutes=0.0, out=True))
    return projections, explained
