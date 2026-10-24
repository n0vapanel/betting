"""Fetch NBA game logs via nba_api and build no-lookahead team stats.

Every statistic handed to the prediction engine is computed strictly from
games played BEFORE the game being predicted.  The game itself and any
later game never contribute to a prediction.
"""

from __future__ import annotations

import bisect
import time
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

try:
    from nba_api.stats.endpoints.leaguegamefinder import LeagueGameFinder
except Exception as exc:  # pragma: no cover - depends on env
    LeagueGameFinder = None
    NBA_API_IMPORT_ERROR: Optional[Exception] = exc
else:
    NBA_API_IMPORT_ERROR = None

SEASON = "2025-26"
WINDOW_START = date(2026, 3, 1)
WINDOW_END = date(2026, 3, 31)

# All 30 NBA franchises (static IDs so no extra stats.nba.com lookup is needed).
TEAM_IDS = [
    1610612737,  # ATL
    1610612738,  # BOS
    1610612751,  # BKN
    1610612766,  # CHA
    1610612741,  # CHI
    1610612739,  # CLE
    1610612742,  # DAL
    1610612743,  # DEN
    1610612765,  # DET
    1610612744,  # GSW
    1610612745,  # HOU
    1610612754,  # IND
    1610612746,  # LAC
    1610612747,  # LAL
    1610612763,  # MEM
    1610612748,  # MIA
    1610612749,  # MIL
    1610612750,  # MIN
    1610612740,  # NOP
    1610612752,  # NYK
    1610612760,  # OKC
    1610612753,  # ORL
    1610612755,  # PHI
    1610612756,  # PHX
    1610612757,  # POR
    1610612758,  # SAC
    1610612759,  # SAS
    1610612761,  # TOR
    1610612762,  # UTA
    1610612764,  # WAS
]

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_PATH = CACHE_DIR / f"nba_logs_{SEASON.replace('-', '_')}.csv"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns the harness needs; date as plain YYYY-MM-DD."""
    keep = ["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "PTS"]
    df = df[keep].copy()
    df["GAME_DATE"] = df["GAME_DATE"].astype(str).str[:10]
    return df


def _call_finder(team_id: int, attempts: int) -> Optional[pd.DataFrame]:
    """One LeagueGameFinder call with backoff retries (stats.nba.com 429s)."""
    if LeagueGameFinder is None:
        raise RuntimeError(f"nba_api is not importable: {NBA_API_IMPORT_ERROR}")
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            finder = LeagueGameFinder(
                team_id_nullable=team_id,
                league_id_nullable="00",
                season_nullable=SEASON,
                season_type_nullable="Regular Season",
                timeout=120,
            )
            return finder.get_data_frames()[0]
        except Exception as exc:  # noqa: BLE001 - any failure -> retry
            last_exc = exc
            time.sleep(min(20.0, 2 ** attempt))
    print(f"WARN: fetch failed for team_id={team_id}: {last_exc}")
    return None


def fetch_season_logs() -> pd.DataFrame:
    """Pull every 2025-26 regular-season game-log row for all 30 teams.

    Tries a single all-teams request first, then falls back to one request
    per team (with a small sleep between calls and exponential-backoff
    retries) so transient 429s from stats.nba.com do not kill the run.
    """
    if NBA_API_IMPORT_ERROR is not None:
        raise RuntimeError(f"nba_api is not importable: {NBA_API_IMPORT_ERROR}")

    try:
        df = _call_finder(team_id=0, attempts=4)
        if df is not None and len(df):
            return _normalize(df)
    except Exception:
        pass  # fall through to the per-team loop below

    frames: List[pd.DataFrame] = []
    for team_id in TEAM_IDS:
        df = _call_finder(team_id=team_id, attempts=5)
        if df is not None and len(df):
            frames.append(df)
        time.sleep(0.75)
    if not frames:
        raise RuntimeError(
            "Could not fetch any game logs from stats.nba.com "
            "(network blocked or API unreachable)"
        )
    return _normalize(pd.concat(frames, ignore_index=True))


def load_or_fetch_logs() -> pd.DataFrame:
    """Return season logs, using the local cache when present."""
    if CACHE_PATH.exists():
        cached = pd.read_csv(CACHE_PATH, dtype={"GAME_ID": str})
        if len(cached):
            return cached
    df = fetch_season_logs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def build_march_games(
    df: pd.DataFrame,
) -> Tuple[List[Dict], List[Tuple[str, str, str, str]], int]:
    """Extract the March window from raw game-log rows.

    Each NBA game appears twice in the logs (once per team).  Rows are
    merged on GAME_ID and we require both sides' points so every game has a
    final score.  Returns:
        games:       [{"date": date, "home", "away", "home_won": bool}]
        skipped:     [(date, home, away, reason), ...]
        parse_fail:  number of rows whose MATCHUP could not be parsed
    """
    start_iso = WINDOW_START.isoformat()
    end_iso = WINDOW_END.isoformat()
    window = df[(df["GAME_DATE"] >= start_iso) & (df["GAME_DATE"] <= end_iso)]

    acc: Dict[str, Dict] = {}
    parse_fail = 0
    for _, row in window.iterrows():
        abbr = str(row["TEAM_ABBREVIATION"])
        matchup = str(row["MATCHUP"])
        if " vs. " in matchup:
            home, away = matchup.split(" vs. ")
        elif " @ " in matchup:
            away, home = matchup.split(" @ ")
        else:
            parse_fail += 1
            continue
        home, away = home.strip(), away.strip()
        if abbr not in (home, away):
            parse_fail += 1
            continue
        gid = str(row["GAME_ID"])
        game = acc.setdefault(
            gid,
            {"date": row["GAME_DATE"], "home": home, "away": away, "pts": {}, "home_wl": None},
        )
        game["pts"][abbr] = row["PTS"]
        if abbr == home:
            game["home_wl"] = row["WL"]

    games: List[Dict] = []
    skipped: List[Tuple[str, str, str, str]] = []
    for gid, game in sorted(acc.items(), key=lambda kv: (kv[1]["date"], kv[1]["home"])):
        home, away = game["home"], game["away"]
        if game["pts"].get(home) is None or game["pts"].get(away) is None:
            skipped.append(
                (game["date"], home, away, "incomplete game log (missing one side's points)")
            )
            continue
        home_won = str(game["home_wl"]) == "W"
        games.append(
            {"date": date.fromisoformat(game["date"]), "home": home, "away": away, "home_won": home_won}
        )
    return games, skipped, parse_fail


def build_prior_stats_provider(
    df: pd.DataFrame,
) -> Callable[[date, str], Dict[str, float]]:
    """Build a stats lookup that only ever sees games BEFORE a given date.

    Returns provider(game_date, team_abbr) -> {"points_for", "points_against",
    "games_played"} where all three values come exclusively from that team's
    games with GAME_DATE strictly earlier than `game_date`.
    """
    # Opponent points per game, so we can compute points_against.
    pts_by_game: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        gid = str(row["GAME_ID"])
        pts_by_game.setdefault(gid, {})[str(row["TEAM_ABBREVIATION"])] = float(row["PTS"])

    # Per team: sorted dates with cumulative points_for / points_against /
    # games_played *before* each date (prefix arrays, lookahead-free).
    team_prefix: Dict[str, Tuple[List[str], List[float], List[float], List[int]]] = {}
    for team, group in df.groupby("TEAM_ABBREVIATION"):
        group = group.sort_values("GAME_DATE")
        dates: List[str] = []
        pf: List[float] = []
        pa: List[float] = []
        n: List[int] = []
        cum_pf = cum_pa = cum_n = 0.0
        cum_n_i = 0
        for _, row in group.iterrows():
            gid = str(row["GAME_ID"])
            dates.append(row["GAME_DATE"])
            pf.append(cum_pf)
            pa.append(cum_pa)
            n.append(cum_n_i)
            opp_pts = 0.0
            for other, pts in pts_by_game.get(gid, {}).items():
                if other != team:
                    opp_pts = pts
                    break
            cum_pf += float(row["PTS"])
            cum_pa += opp_pts
            cum_n_i += 1
        team_prefix[str(team)] = (dates, pf, pa, n)

    def provider(game_date: date, team: str) -> Dict[str, float]:
        dates, pf, pa, n = team_prefix.get(team, ([], [], [], []))
        # First index with date >= game_date; everything before it is prior.
        # pf[idx] is the cumulative total *before* the game at index idx,
        # i.e. exactly the games played strictly before game_date.
        idx = bisect.bisect_left(dates, game_date.isoformat())
        if idx == 0:
            return {"points_for": 0.0, "points_against": 0.0, "games_played": 0}
        return {"points_for": pf[idx], "points_against": pa[idx], "games_played": n[idx]}

    return provider
