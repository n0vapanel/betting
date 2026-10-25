"""Which sports and leagues the bot and dashboard offer, and in what order."""

from __future__ import annotations

from typing import Any, Dict, List

# Sport menu: (label, The Odds API sport group)
SPORTS = [
    ("🏀 Basketball", "Basketball"),
    ("🏒 Ice Hockey", "Ice Hockey"),
    ("⚽ Soccer", "Soccer"),
    ("🎾 Tennis", "Tennis"),
    ("⚾ Baseball", "Baseball"),
    ("🏈 Football", "American Football"),
    ("🥊 MMA", "Mixed Martial Arts"),
]

# Leagues listed first when a sport has more in season than fit in a menu
POPULAR_LEAGUES = [
    "basketball_nba",
    "basketball_ncaab",
    "icehockey_nhl",
    "americanfootball_nfl",
    "americanfootball_ncaaf",
    "baseball_mlb",
    "soccer_epl",
    "soccer_uefa_champs_league",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_usa_mls",
    "mma_mixed_martial_arts",
]
MAX_LEAGUES = 12


def pick_leagues(sports: List[Dict[str, Any]], group: str, limit: int = MAX_LEAGUES) -> List[Dict[str, Any]]:
    """Return ``group``'s leagues that are in season and have games (not futures), popular first."""
    leagues = [
        sport for sport in sports
        if sport.get("group") == group and sport.get("active") and not sport.get("has_outrights")
    ]

    def rank(league: Dict[str, Any]) -> tuple:
        key = league.get("key")
        popularity = POPULAR_LEAGUES.index(key) if key in POPULAR_LEAGUES else len(POPULAR_LEAGUES)
        return popularity, league.get("title", "")

    return sorted(leagues, key=rank)[:limit]
