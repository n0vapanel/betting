"""v2 data layer: fetch + cache NBA team game logs, player game logs, Kaggle
closing moneylines, and OFFICIAL NBA injury reports (nbainjuries package).

Lookahead rules honoured here:
- Injury reports: for a game on date g we use the 5PM ET snapshot of g-1 (the
  NBA-mandated day-before filing), filtered to rows with Game Date == g.
  Fallback: the 1PM ET game-day snapshot of g itself.  Both snapshots are
  strictly pre-game information (the day-before filing is issued ~20h before
  tip; the 1PM game-day update is issued before evening tips).
- Team/player logs and Kaggle rows are raw historical data; the *rolling*
  (no-lookahead) part lives in features.py.

JVM note: nbainjuries requires a JVM.  Run with
    export JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home
before importing this module's injury functions (imports are lazy so the
rest of the module works without a JVM).
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "backtest" / "cache"
KAGGLE_CSV = (
    Path.home()
    / ".cache/kagglehub/datasets/cviaxmiwnptr/"
    "nba-betting-data-october-2007-to-june-2024/versions/4/nba_2008-2026.csv"
)
SEASON = "2021-22"
TRAIN_START = "2021-10-20"
TRAIN_END = "2022-02-28"
TEST_START = "2022-03-01"
TEST_END = "2022-03-31"

CODE_MAP = {"gs": "GSW", "ny": "NYK", "no": "NOP", "sa": "SAS", "utah": "UTA", "wsh": "WAS"}
STATUS_WEIGHTS = {"Out": 1.0, "Doubtful": 0.5, "Questionable": 0.25, "Probable": 0.0}

TEAMLOG_CACHE = CACHE_DIR / "nba_teamlog_2021_22.csv"
PLAYERLOG_CACHE = CACHE_DIR / "nba_playerlog_2021_22.csv"
INJURY_CACHE = CACHE_DIR / "injury_reports_2021_22.csv"
INJURY_STATUS = CACHE_DIR / "injury_fetch_status.json"


def kaggle_to_abbr(code: str) -> str:
    return CODE_MAP.get(str(code).lower().strip(), str(code).upper().strip())


def american_to_decimal(ml: float) -> float:
    if ml > 0:
        return 1.0 + ml / 100.0
    return 1.0 + 100.0 / abs(ml)


# ---------------------------------------------------------------- team/player logs

def fetch_team_log() -> pd.DataFrame:
    """Full 2021-22 regular-season team game log (2 rows per game, one per team)."""
    if TEAMLOG_CACHE.exists():
        df = pd.read_csv(TEAMLOG_CACHE, dtype={"GAME_ID": str})
        if len(df):
            return df
    from nba_api.stats.endpoints.leaguegamelog import LeagueGameLog

    raw = LeagueGameLog(
        season=SEASON,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T",
    ).get_data_frames()[0]
    raw["GAME_DATE"] = raw["GAME_DATE"].astype(str).str[:10]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(TEAMLOG_CACHE, index=False)
    return raw


def fetch_player_log() -> pd.DataFrame:
    """Full 2021-22 regular-season player game log (MIN per player per game)."""
    if PLAYERLOG_CACHE.exists():
        df = pd.read_csv(PLAYERLOG_CACHE, dtype={"GAME_ID": str})
        if len(df):
            return df
    from nba_api.stats.endpoints.leaguegamelog import LeagueGameLog

    raw = LeagueGameLog(
        season=SEASON,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="P",
    ).get_data_frames()[0]
    raw["GAME_DATE"] = raw["GAME_DATE"].astype(str).str[:10]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(PLAYERLOG_CACHE, index=False)
    return raw


# ---------------------------------------------------------------- kaggle odds/scores

def load_kaggle_games(start: str = "2021-10-01", end: str = "2022-03-31") -> pd.DataFrame:
    """Regular-season games with scores + closing moneylines, abbr-normalized."""
    df = pd.read_csv(KAGGLE_CSV)
    sel = df[
        (df["date"] >= start)
        & (df["date"] <= end)
        & (df["regular"] == True)  # noqa: E712
    ].copy()
    sel = sel[
        sel["moneyline_home"].notna()
        & sel["moneyline_away"].notna()
        & sel["score_home"].notna()
        & sel["score_away"].notna()
    ]
    out = pd.DataFrame(
        {
            "date": sel["date"].astype(str).str[:10],
            "home": sel["home"].apply(kaggle_to_abbr),
            "away": sel["away"].apply(kaggle_to_abbr),
            "score_home": sel["score_home"].astype(float),
            "score_away": sel["score_away"].astype(float),
            "ml_home": sel["moneyline_home"].astype(float),
            "ml_away": sel["moneyline_away"].astype(float),
            "dec_home": sel["moneyline_home"].astype(float).apply(american_to_decimal),
            "dec_away": sel["moneyline_away"].astype(float).apply(american_to_decimal),
        }
    )
    out["home_won"] = out["score_home"] > out["score_away"]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- injury reports

def _normalise_injury_df(raw: pd.DataFrame, snapshot_dt: datetime) -> pd.DataFrame:
    """Normalise one nbainjuries report to cache schema."""
    out = pd.DataFrame(
        {
            "snapshot_date": snapshot_dt.strftime("%Y-%m-%d"),
            "snapshot_time": f"{snapshot_dt.hour:02d}:00",
            "game_date": pd.to_datetime(raw["Game Date"], format="%m/%d/%Y")
            .dt.strftime("%Y-%m-%d"),
            "game_time": raw["Game Time"].astype(str),
            "matchup": raw["Matchup"].astype(str),
            "team": raw["Team"].astype(str),
            "player": raw["Player Name"].astype(str),
            "status": raw["Current Status"].astype(str),
            "reason": raw["Reason"].astype(str).str.slice(0, 120),
        }
    )
    return out


def _fetch_snapshot(dt: datetime) -> Optional[pd.DataFrame]:
    """Fetch one nbainjuries snapshot (lazy JVM import). Returns None on failure."""
    from nbainjuries import injury  # noqa: PLC0415 (needs JVM)

    try:
        valid = True
        try:
            valid = injury.check_reportvalid(dt)
        except Exception:  # noqa: BLE001 - signature/behaviour varies; trust fetch below
            valid = True
        if valid is False:
            return None
        raw = injury.get_reportdata(dt, return_df=True)
        if raw is None or not len(raw):
            return pd.DataFrame()
        return _normalise_injury_df(raw, dt)
    except Exception:  # noqa: BLE001 - any failure = report unavailable
        return None


def _load_status() -> Dict[str, Dict[str, Dict]]:
    if INJURY_STATUS.exists():
        return json.loads(INJURY_STATUS.read_text())
    return {}


def _save_status(status: Dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INJURY_STATUS.write_text(json.dumps(status, indent=1, sort_keys=True))


def _append_cache(rows: pd.DataFrame) -> None:
    if not len(rows):
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if INJURY_CACHE.exists():
        rows.to_csv(INJURY_CACHE, mode="a", header=False, index=False)
    else:
        rows.to_csv(INJURY_CACHE, index=False)


def _fetch_and_record(dt: datetime, status: Dict) -> pd.DataFrame:
    """Fetch snapshot for dt, record outcome, return parsed rows."""
    key = dt.strftime("%Y-%m-%d")
    tkey = f"{dt.hour:02d}:00"
    status.setdefault(key, {})
    if tkey in status[key]:
        return pd.DataFrame()  # already attempted
    try:
        rows = _fetch_snapshot(dt)
    except Exception as exc:  # noqa: BLE001
        status[key][tkey] = {"ok": False, "rows": 0, "error": str(exc)[:200]}
        _save_status(status)
        return pd.DataFrame()
    if rows is None:
        status[key][tkey] = {"ok": False, "rows": 0, "error": "no report / invalid"}
        _save_status(status)
        return pd.DataFrame()
    status[key][tkey] = {"ok": True, "rows": int(len(rows)), "error": None}
    _save_status(status)
    _append_cache(rows)
    return rows


def _game_dates() -> List[str]:
    """Every regular-season game date in the training + test windows (Kaggle)."""
    games = load_kaggle_games(start=TRAIN_START, end=TEST_END)
    return sorted(games["date"].unique().tolist())


def _game_hour(t: str) -> int:
    m = re.search(r"(\d{1,2}):(\d{2})", str(t))
    return int(m.group(1)) if m else 99


def fetch_all_injuries(verbose: bool = True) -> pd.DataFrame:
    """Fetch every snapshot the feature builder needs, incrementally:

    Pass 1 — 5PM game-day snapshots (g @ 17:00) for every game date g, plus
              the day before the first game date.  The 5PM report of day g
              carries the complete pre-game filings for g's evening games
              (the NBA-mandated pre-game filing window) and the day-before
              filings for g+1.
    Pass 2 — 1PM game-day snapshots (g @ 13:00) for game dates that need
              them: early tips (noon–5PM ET, where the 5PM report would be
              post-game) or 17:00 fetch failures / empty same-day rows.
    Reruns skip already-attempted (snapshot_date, time) pairs.
    """
    import re  # noqa: F401

    status = _load_status()
    game_dates = _game_dates()
    g_dates = {date.fromisoformat(s) for s in game_dates}

    # Pass 1: 5PM snapshots for every date covering the window.
    first = date.fromisoformat(min(game_dates)) - timedelta(days=1)
    last = date.fromisoformat(max(game_dates))
    total = (last - first).days + 1
    for i, d in enumerate((first + timedelta(days=n) for n in range(total)), 1):
        if verbose and (i % 20 == 0 or i == total):
            print(f"[injuries] 5PM pass {i}/{total} ({d})", flush=True)
        _fetch_and_record(datetime(d.year, d.month, d.day, 17, 0), status)

    # Pass 2: 1PM game-day snapshots where routing needs them.
    cache = (
        pd.read_csv(INJURY_CACHE, dtype=str)
        if INJURY_CACHE.exists()
        else pd.DataFrame(columns=["snapshot_date", "snapshot_time", "game_date"])
    )
    same_day = cache[
        (cache["snapshot_time"] == "17:00")
        & (cache["snapshot_date"] == cache["game_date"])
    ].copy()
    same_day["hr"] = same_day["game_time"].apply(_game_hour)
    early_dates = {
        date.fromisoformat(d) for d in same_day[same_day["hr"] < 6]["game_date"].unique()
    }
    failed_dates = {
        date.fromisoformat(k) for k in status
        if not status[k].get("17:00", {}).get("ok", False)
    }
    present = set(zip(same_day["game_date"], same_day["matchup"]))
    kg = load_kaggle_games(start=TRAIN_START, end=TEST_END)
    kg["m"] = kg["away"] + "@" + kg["home"]
    missing_dates = {
        date.fromisoformat(d) for d, _ in set(zip(kg["date"], kg["m"])) - present
        if d in game_dates
    }
    need13 = sorted(early_dates | failed_dates | missing_dates)
    if verbose:
        print(f"[injuries] 5PM pass done; 1PM fallback needed for "
              f"{len(need13)} dates (early tips={len(early_dates)}, "
              f"failures={len(failed_dates)}, empty={len(missing_dates)})", flush=True)
    for i, g in enumerate(need13, 1):
        if verbose and (i % 10 == 0 or i == len(need13)):
            print(f"[injuries] 1PM fallback pass {i}/{len(need13)} ({g})", flush=True)
        _fetch_and_record(datetime(g.year, g.month, g.day, 13, 0), status)

    cache = pd.read_csv(INJURY_CACHE, dtype=str) if INJURY_CACHE.exists() else pd.DataFrame()
    if verbose:
        print(f"[injuries] done: {len(cache)} cached rows "
              f"({cache['snapshot_date'].nunique()} snapshot dates)", flush=True)
    return cache


if __name__ == "__main__":
    fetch_all_injuries(verbose=True)
