"""Analytical functions for computing win probabilities and team metrics.

This module contains helper functions that combine raw statistics,
bookmaker odds and contextual information (such as injury news) into
win probabilities.  They are deliberately simple and transparent; you
can replace or extend them with more advanced models as needed.
"""

from __future__ import annotations

import logging
from typing import Dict

def normalize_probabilities(probs: Dict[str, float]) -> Dict[str, float]:
    """Scale probability values so they sum to 1."""
    total = sum(probs.values())
    if total <= 0:
        return {k: 0.0 for k in probs}
    return {k: v / total for k, v in probs.items()}

def implied_probabilities_from_odds(odds: Dict[str, float]) -> Dict[str, float]:
    """Convert decimal odds into implied probabilities."""
    probs: Dict[str, float] = {}
    for outcome, price in odds.items():
        try:
            prob = 1.0 / float(price) if price > 0 else 0.0
        except Exception:
            prob = 0.0
        probs[outcome] = prob
    return normalize_probabilities(probs) if probs else {}

def calculate_team_strength(stats: Dict[str, float]) -> float:
    """Derive a simple strength score from a team's statistics."""
    offense = stats.get("points_for") or stats.get("goals_for") or stats.get("runs_for")
    defense = stats.get("points_against") or stats.get("goals_against") or stats.get("runs_against")
    games_played = stats.get("games_played") or stats.get("matches") or 1
    try:
        offense_rate = float(offense) / games_played if offense is not None else 1.0
        defense_rate = float(defense) / games_played if defense is not None else 1.0
        strength = offense_rate / (defense_rate if defense_rate > 0 else 1.0)
    except Exception:
        strength = 1.0
    return max(strength, 0.1)

def adjust_for_injuries(strength: float, injury_factor: float) -> float:
    """Reduce a team's strength based on injuries or suspensions."""
    return strength * max(min(injury_factor, 1.0), 0.0)

def calculate_win_probabilities(
    home_team: str,
    away_team: str,
    home_stats: Dict[str, float],
    away_stats: Dict[str, float],
    odds: Dict[str, float],
    home_injury_factor: float = 1.0,
    away_injury_factor: float = 1.0,
    weight_stats: float = 0.6,
    weight_odds: float = 0.4,
) -> Dict[str, float]:
    """Compute win probabilities by combining team stats and bookmaker odds."""
    try:
        home_strength = calculate_team_strength(home_stats)
        away_strength = calculate_team_strength(away_stats)
        home_strength = adjust_for_injuries(home_strength, home_injury_factor)
        away_strength = adjust_for_injuries(away_strength, away_injury_factor)
        stats_total = home_strength + away_strength
        home_stat_prob = home_strength / stats_total if stats_total else 0.5
        away_stat_prob = away_strength / stats_total if stats_total else 0.5
    except Exception:
        home_stat_prob = away_stat_prob = 0.5

    implied = implied_probabilities_from_odds(odds)
    home_odds_prob = implied.get("home", 0.5)
    away_odds_prob = implied.get("away", 0.5)

    home_final = (weight_stats * home_stat_prob) + (weight_odds * home_odds_prob)
    away_final = (weight_stats * away_stat_prob) + (weight_odds * away_odds_prob)

    return normalize_probabilities({home_team: home_final, away_team: away_final})
