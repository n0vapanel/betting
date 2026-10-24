"""Domain models for sports data used by the prediction engine.

This module defines lightweight data classes representing common entities
retrieved from external APIs.  Using structured models helps to
normalize data coming from disparate sources (e.g. API‑SPORTS, The
Odds API, news feeds) into a consistent shape that the analytics
layer can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class Team:
    """Representation of a sports team."""
    id: str
    name: str
    league: Optional[str] = None
    country: Optional[str] = None
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class Player:
    """Representation of a player."""
    id: str
    name: str
    team_id: Optional[str] = None
    position: Optional[str] = None
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class Game:
    """Representation of a scheduled or completed game."""
    id: str
    sport: str
    date: datetime
    home_team: Team
    away_team: Team
    venue: Optional[str] = None
    odds: Dict[str, float] = field(default_factory=dict)


@dataclass
class Odds:
    """Odds information for a single game across different markets."""
    game_id: str
    markets: Dict[str, Dict[str, float]]  # e.g. {"h2h": {"home": 1.5, "away": 2.8}}
    last_update: datetime


@dataclass
class NewsArticle:
    """Represents a news article or report relevant to a team or player."""
    title: str
    url: str
    published_at: datetime
    summary: Optional[str] = None
    source: Optional[str] = None
