"""No-lookahead feature builder for the v2 model.

Every feature for game g uses ONLY information available before tip-off:
- Rolling Four Factors: each team's own + opponent metrics over its last <=10
  games (min 5) STRICTLY before g (shift(1).rolling(10)).
- Elo: margin-based (K=20, home-court +100), updated chronologically AFTER each
  game; the feature is the pre-game rating gap.
- Rest: days since each team's previous game + back-to-back flags.
- Injury impact: official NBA reports — the 5PM game-day filing (complete
  NBA-mandated pre-game report; verified pre-tip for evening games), with the
  1PM game-day snapshot for noon–5PM tips, weighted by listed players' prior
  rolling minutes share.
- Market: no-vig closing-moneyline implied home probability.
- Season-to-date win% / avg margin (prior games only).
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import data as D

WINDOW = 10
MIN_GAMES = 5

FEATURE_NAMES = [
    "off_eFG_home", "off_eFG_away",
    "tov_home", "tov_away",
    "orb_home", "orb_away",
    "ftr_home", "ftr_away",
    "pace_diff",
    "xprod_home", "xprod_away",
    "elo_diff",
    "rest_diff", "b2b_home", "b2b_away",
    "inj_home", "inj_away", "inj_diff",
    "market_home",
    "winpct_diff", "margin_diff",
]

STATS_ONLY_COLS = [
    "off_eFG_home", "off_eFG_away", "tov_home", "tov_away", "orb_home", "orb_away",
    "ftr_home", "ftr_away", "pace_diff", "xprod_home", "xprod_away", "elo_diff",
    "rest_diff", "b2b_home", "b2b_away",
]
NO_INJURY_COLS = [c for c in FEATURE_NAMES if not c.startswith("inj_")]
NO_MARKET_COLS = [c for c in FEATURE_NAMES if c != "market_home"]


# ------------------------------------------------------------------ name utils

def norm_player_name(name: str) -> str:
    """'Bagley III, Marvin' -> 'marvin bagley'; 'Stephen Curry' -> 'stephen curry'."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^a-z ,]+", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) > 1:  # 'Last, First' -> 'First Last'
        s = parts[-1] + " " + parts[0]
    else:
        s = " ".join(parts)
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------ team stats

def build_team_game_stats(teamlog: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game with own + opponent metrics and rolling lookbacks."""
    df = teamlog.copy()
    df["date"] = df["GAME_DATE"]
    df["is_home"] = df["MATCHUP"].str.contains(" vs. ", regex=False)
    df["opp"] = df["MATCHUP"].str.split().str[-1].str.strip()
    df["poss"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]
    df["off_eFG"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"].replace(0, np.nan)
    df["off_TOVpct"] = df["TOV"] / df["poss"].replace(0, np.nan)
    df["off_FTr"] = df["FTM"] / df["FGA"].replace(0, np.nan)
    df["win"] = (df["WL"] == "W").astype(int)

    # Attach opponent metrics (same GAME_ID, other team).
    opp = df.rename(
        columns={
            "TEAM_ABBREVIATION": "opp_abbr", "OREB": "opp_OREB", "DREB": "opp_DREB",
            "PTS": "opp_PTS", "poss": "opp_poss", "off_eFG": "opp_off_eFG",
            "off_TOVpct": "opp_off_TOVpct", "off_FTr": "opp_off_FTr",
        }
    )[["GAME_ID", "opp_abbr", "opp_OREB", "opp_DREB", "opp_PTS", "opp_poss",
       "opp_off_eFG", "opp_off_TOVpct", "opp_off_FTr"]]
    df = df.merge(opp, on="GAME_ID", how="left")
    df = df[df["opp_abbr"] != df["TEAM_ABBREVIATION"]].copy()  # drop self-match
    df = df[df["opp_abbr"].notna()].copy()

    df["ORBpct"] = df["OREB"] / (df["OREB"] + df["opp_OREB"]).replace(0, np.nan)
    df["def_ORBpct"] = df["opp_OREB"] / (df["opp_OREB"] + df["DREB"]).replace(0, np.nan)
    df["def_eFG"] = df["opp_off_eFG"]
    df["def_TOVpct"] = df["opp_off_TOVpct"]
    df["def_FTr"] = df["opp_off_FTr"]
    df["margin"] = df["PTS"] - df["opp_PTS"]
    df["pace"] = 0.5 * (df["poss"] + df["opp_poss"]) * 48.0 / (df["MIN"] / 5.0)

    roll_cols = [
        "off_eFG", "off_TOVpct", "off_FTr", "ORBpct",
        "def_eFG", "def_TOVpct", "def_FTr", "def_ORBpct", "pace",
    ]
    out_rows = []
    for team, grp in df.groupby("TEAM_ABBREVIATION", sort=False):
        g = grp.sort_values(["date", "GAME_ID"])
        for col in roll_cols:
            g[col + "_r"] = g[col].shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean()
        n_prior = g["win"].cumsum().shift(1)
        g["s2d_winpct"] = n_prior / g["win"].shift(1).notna().cumsum().shift(1)
        g["s2d_margin"] = g["margin"].cumsum().shift(1) / g["win"].shift(1).notna().cumsum().shift(1)
        out_rows.append(g)
    out = pd.concat(out_rows, ignore_index=True)
    out["rest_days"] = np.nan
    out["b2b"] = 0
    for team, grp in out.groupby("TEAM_ABBREVIATION", sort=False):
        g = grp.sort_values(["date", "GAME_ID"])
        prev = g["date"].shift(1)
        delta = (pd.to_datetime(g["date"]) - pd.to_datetime(prev)).dt.days
        out.loc[g.index, "rest_days"] = (delta - 1).where(prev.notna(), np.nan)
        out.loc[g.index, "b2b"] = (delta == 1).astype(int).where(prev.notna(), 0)
    return out


def build_elo_gaps(games: pd.DataFrame) -> pd.DataFrame:
    """Margin-based Elo (K=20, HCA +100), chronological; returns pre-game gap."""
    games = games.sort_values(["date", "GAME_ID"]).reset_index(drop=True)
    elo: Dict[str, float] = {}
    gaps = np.empty(len(games))
    for i, r in games.iterrows():
        h, a = r["home"], r["away"]
        eh = elo.setdefault(h, 1500.0)
        ea = elo.setdefault(a, 1500.0)
        gaps[i] = eh - ea
        exp_h = 1.0 / (1.0 + 10 ** ((ea - (eh + 100.0)) / 400.0))
        margin = abs(float(r["score_home"]) - float(r["score_away"]))
        mult = ((margin + 3.0) ** 0.8) / (7.5 + 0.006 * abs(eh - ea))
        win_h = 1.0 if float(r["score_home"]) > float(r["score_away"]) else 0.0
        delta = 20.0 * mult * (win_h - exp_h)
        elo[h] += delta
        elo[a] -= delta
    games["elo_diff"] = gaps
    return games


# ------------------------------------------------------------------ injuries

def build_player_avg_min(playerlog: pd.DataFrame) -> Dict[Tuple[str, str], List[Tuple[str, float]]]:
    """{(norm_name, team): sorted [(date, avg_min_over_strictly_prior_games)]}."""
    df = playerlog[["PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_DATE", "MIN"]].copy()
    df["player"] = df["PLAYER_NAME"].apply(norm_player_name)
    df = df.sort_values("GAME_DATE")
    df["avg_min"] = (
        df.groupby(["player", "TEAM_ABBREVIATION"])["MIN"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )
    lookup: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
    for (p, t), grp in df.groupby(["player", "TEAM_ABBREVIATION"], sort=False):
        vals = grp.sort_values("GAME_DATE")[["GAME_DATE", "avg_min"]].values
        lookup[(p, t)] = [(str(d), float(v)) for d, v in vals if pd.notna(v)]
    return lookup


def _avg_min_before(lookup, key: Tuple[str, str], game_date: str) -> Optional[float]:
    entries = lookup.get(key)
    if not entries:
        return None
    dates = [e[0] for e in entries]
    idx = bisect.bisect_left(dates, game_date)
    if idx == 0:
        return None
    return entries[idx - 1][1]


def prepare_injury_cache(inj_df: pd.DataFrame, teamlog: pd.DataFrame) -> pd.DataFrame:
    """Injury rows with team abbr and normalized player names."""
    name_map = dict(
        zip(teamlog["TEAM_NAME"].astype(str), teamlog["TEAM_ABBREVIATION"].astype(str))
    )
    out = inj_df.copy()
    out["team_abbr"] = out["team"].map(name_map)
    out["player_norm"] = out["player"].apply(norm_player_name)
    out = out[out["team_abbr"].notna()].copy()
    return out


def _hour(t: str) -> int:
    m = re.search(r"(\d{1,2}):(\d{2})", str(t))
    return int(m.group(1)) if m else 99


def _team_filtered(rows: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    rows = rows[rows["team_abbr"].isin({home, away})].copy()
    rows = rows.drop_duplicates(subset=["team_abbr", "player_norm"], keep="last")
    return rows


def injury_rows_for_game(
    inj: pd.DataFrame, g_date: str, home: str, away: str
) -> Optional[Tuple[pd.DataFrame, str]]:
    """Pre-game injury rows for (g_date, home, away), strictly no lookahead.

    Routing:
    - Primary: the 5PM game-day report (snapshot (g, 17:00), Game Date == g) —
      the complete NBA-mandated pre-game filing for evening games.
    - Early tips (noon–5PM ET, 12-hour clock hour < 6): use the 1PM game-day
      snapshot (g, 13:00) so the info is strictly pre-tip; fall back to the
      5PM day-before filing if the 1PM snapshot has nothing for this game.
    - Games absent from the 5PM report: 1PM game-day snapshot; then drop.
    """
    matchup = f"{away}@{home}"
    cand17 = inj[
        (inj["game_date"] == g_date)
        & (inj["snapshot_date"] == g_date)
        & (inj["snapshot_time"] == "17:00")
    ]
    rows17 = cand17[cand17["matchup"] == matchup]
    if len(rows17):
        if _hour(rows17["game_time"].iloc[0]) < 6:
            rows13 = _snapshot_rows(inj, g_date, g_date, "13:00", matchup)
            if len(rows13):
                return _team_filtered(rows13, home, away), "game-day 1PM (early tip)"
            day_before = (date.fromisoformat(g_date) - timedelta(days=1)).isoformat()
            rows_db = _snapshot_rows(inj, day_before, g_date, "17:00", matchup)
            if len(rows_db):
                return _team_filtered(rows_db, home, away), "day-before 5PM (early tip, no 1PM)"
            return None
        return _team_filtered(rows17, home, away), "game-day 5PM"
    rows13 = _snapshot_rows(inj, g_date, g_date, "13:00", matchup)
    if len(rows13):
        return _team_filtered(rows13, home, away), "game-day 1PM (fallback)"
    return None


def _snapshot_rows(
    inj: pd.DataFrame, snapshot_date: str, game_date: str, time: str, matchup: str
) -> pd.DataFrame:
    return inj[
        (inj["snapshot_date"] == snapshot_date)
        & (inj["game_date"] == game_date)
        & (inj["snapshot_time"] == time)
        & (inj["matchup"] == matchup)
    ]


def team_injury_impact(
    rows: pd.DataFrame, team_abbr: str, g_date: str, lookup
) -> float:
    """Sum of status weight x prior-minutes share for one team."""
    rows = rows[rows["team_abbr"] == team_abbr]
    total = 0.0
    for _, r in rows.iterrows():
        w = D.STATUS_WEIGHTS.get(str(r["status"]).strip(), 0.0)
        if w <= 0:
            continue
        avg = _avg_min_before(lookup, (r["player_norm"], team_abbr), g_date)
        share = min(avg / 240.0, 1.0) if avg is not None else 0.0
        total += w * share
    return float(total)


# ------------------------------------------------------------------ assembly

def build_features(verbose: bool = True) -> Dict:
    """Build the full feature matrix + meta for training and test windows."""
    teamlog = D.fetch_team_log()
    playerlog = D.fetch_player_log()
    kaggle = D.load_kaggle_games(start=D.TRAIN_START, end=D.TEST_END)

    stats = build_team_game_stats(teamlog)
    # Game table from team logs (covers every played game, incl. those w/o market).
    game_rows = []
    for gid, grp in stats.groupby("GAME_ID", sort=False):
        home_rows = grp[grp["is_home"]]
        away_rows = grp[~grp["is_home"]]
        if len(home_rows) != 1 or len(away_rows) != 1:
            continue
        h, a = home_rows.iloc[0], away_rows.iloc[0]
        game_rows.append(
            {
                "GAME_ID": gid,
                "date": h["date"],
                "home": h["TEAM_ABBREVIATION"],
                "away": a["TEAM_ABBREVIATION"],
                "score_home": float(h["PTS"]),
                "score_away": float(a["PTS"]),
                "MIN_home": float(h["MIN"]),
            }
        )
    games = pd.DataFrame(game_rows)
    games = build_elo_gaps(games)

    # Rolling + rest + s2d lookup frames per team-game.
    roll = stats[
        ["TEAM_ABBREVIATION", "GAME_ID", "date"] + [c + "_r" for c in [
            "off_eFG", "off_TOVpct", "off_FTr", "ORBpct",
            "def_eFG", "def_TOVpct", "def_FTr", "def_ORBpct", "pace",
        ]] + ["s2d_winpct", "s2d_margin", "rest_days", "b2b"]
    ].copy()
    roll_key = {(r.TEAM_ABBREVIATION, r.GAME_ID): r for r in roll.itertuples()}

    # Injury support (must be fetched first — see data.py fetch_all_injuries).
    if not D.INJURY_CACHE.exists():
        raise RuntimeError(
            f"Injury cache {D.INJURY_CACHE} missing — run "
            "`python backtest/v2/data.py` first (with JAVA_HOME exported)."
        )
    inj = prepare_injury_cache(pd.read_csv(D.INJURY_CACHE, dtype=str), teamlog)
    lookup = build_player_avg_min(playerlog)

    # Merge Kaggle (market + target) onto the game table.
    kg = kaggle.rename(columns={"home_won": "home_won"})
    joined = games.merge(
        kg[["date", "home", "away", "dec_home", "dec_away", "home_won"]],
        on=["date", "home", "away"], how="inner", suffixes=("", "_kg"),
    )
    joined["p_market_home"] = (1.0 / joined["dec_home"]) / (
        1.0 / joined["dec_home"] + 1.0 / joined["dec_away"]
    )

    rows: List[Dict] = []
    dropped: Dict[str, int] = {"no_rolling_stats": 0, "no_injury_report": 0, "other": 0}
    inj_source_counts: Dict[str, int] = {}
    for _, g in joined.iterrows():
        h_key = (g["home"], g["GAME_ID"])
        a_key = (g["away"], g["GAME_ID"])
        if h_key not in roll_key or a_key not in roll_key:
            dropped["no_rolling_stats"] += 1
            continue
        hr, ar = roll_key[h_key], roll_key[a_key]
        if pd.isna(hr.off_eFG_r) or pd.isna(ar.off_eFG_r):
            dropped["no_rolling_stats"] += 1  # <5 prior games for a team
            continue
        inj_rows = injury_rows_for_game(inj, g["date"], g["home"], g["away"])
        if inj_rows is None:
            dropped["no_injury_report"] += 1
            continue
        inj_rows, inj_source = inj_rows
        inj_source_counts[inj_source] = inj_source_counts.get(inj_source, 0) + 1
        inj_home = team_injury_impact(inj_rows, g["home"], g["date"], lookup)
        inj_away = team_injury_impact(inj_rows, g["away"], g["date"], lookup)

        rows.append(
            {
                "date": g["date"], "home": g["home"], "away": g["away"],
                "home_won": bool(g["home_won"]),
                "dec_home": float(g["dec_home"]), "dec_away": float(g["dec_away"]),
                "p_market_home": float(g["p_market_home"]),
                "off_eFG_home": hr.off_eFG_r - ar.def_eFG_r,
                "off_eFG_away": ar.off_eFG_r - hr.def_eFG_r,
                "tov_home": ar.def_TOVpct_r - hr.off_TOVpct_r,
                "tov_away": hr.def_TOVpct_r - ar.off_TOVpct_r,
                "orb_home": hr.ORBpct_r - ar.def_ORBpct_r,
                "orb_away": ar.ORBpct_r - hr.def_ORBpct_r,
                "ftr_home": hr.off_FTr_r - ar.def_FTr_r,
                "ftr_away": ar.off_FTr_r - hr.def_FTr_r,
                "pace_diff": hr.pace_r - ar.pace_r,
                "xprod_home": (hr.off_eFG_r - ar.def_eFG_r) * (hr.ORBpct_r - ar.def_ORBpct_r),
                "xprod_away": (ar.off_eFG_r - hr.def_eFG_r) * (ar.ORBpct_r - hr.def_ORBpct_r),
                "elo_diff": float(g["elo_diff"]),
                "rest_diff": (hr.rest_days - ar.rest_days) if pd.notna(hr.rest_days) and pd.notna(ar.rest_days) else 0.0,
                "b2b_home": int(hr.b2b), "b2b_away": int(ar.b2b),
                "inj_home": inj_home, "inj_away": inj_away,
                "inj_diff": inj_home - inj_away,
                "market_home": float(g["p_market_home"]),
                "winpct_diff": float(hr.s2d_winpct - ar.s2d_winpct),
                "margin_diff": float(hr.s2d_margin - ar.s2d_margin),
            }
        )
    frame = pd.DataFrame(rows)

    train = frame[(frame["date"] >= D.TRAIN_START) & (frame["date"] <= D.TRAIN_END)]
    test = frame[(frame["date"] >= D.TEST_START) & (frame["date"] <= D.TEST_END)]
    if verbose:
        print(f"[features] games with complete features: {len(frame)} "
              f"(train {len(train)}, test {len(test)})")
        print(f"[features] dropped: {dropped}")
        print(f"[features] injury source: {inj_source_counts}")
    return {
        "train": train.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "dropped": dropped,
        "injury_sources": inj_source_counts,
        "feature_names": FEATURE_NAMES,
    }
