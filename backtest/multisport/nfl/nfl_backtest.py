from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import unicodedata
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtr, ndtri
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
YEARS = [2020, 2021, 2022]
SEED = 20261024
CALIBRATION_WEEKS = 6
BOOTSTRAP_REPETITIONS = 2000
INJURY_WEIGHTS = {"Out": 1.0, "Doubtful": 0.5, "Questionable": 0.25}
GATES = [(kind, eps) for kind in ("lean", "winner_conviction") for eps in (0.0, 0.02, 0.05)]
MARGIN_SCALE = 13.5
PBP_COLUMNS = [
    "game_id", "season", "season_type", "week", "game_date", "home_team", "away_team",
    "posteam", "defteam", "epa", "yards_gained", "pass_attempt", "rush_attempt",
    "qb_dropback", "qb_scramble", "play_type", "no_play", "vegas_line", "vegas_wp",
    "vegas_home_wp", "spread_line", "play_id", "qtr", "game_seconds_remaining",
]
SOURCE_URLS = {
    "schedules": "http://www.habitatring.com/games.csv",
    "schedule_fallback": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    "pbp": "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet",
    "injuries": "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.parquet",
    "snaps": "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{year}.parquet",
    "depth": "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{year}.parquet",
    "ngs": "https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_{kind}.parquet",
    "ids": "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv",
}


def american_decimal(odds):
    odds = np.asarray(odds, dtype=float)
    if np.any(~np.isfinite(odds) | (np.abs(odds) < 100)):
        raise ValueError("American odds must be finite with absolute value at least 100.")
    return 1.0 + np.where(odds > 0, odds / 100.0, 100.0 / np.abs(odds))


def no_vig_home(home_odds, away_odds):
    home = 1.0 / american_decimal(home_odds)
    away = 1.0 / american_decimal(away_odds)
    return home / (home + away)


def moneyline_mask(games):
    home = pd.to_numeric(games.get("home_moneyline", pd.Series(np.nan, index=games.index)),
                         errors="coerce").to_numpy(float)
    away = pd.to_numeric(games.get("away_moneyline", pd.Series(np.nan, index=games.index)),
                         errors="coerce").to_numpy(float)
    return np.isfinite(home) & np.isfinite(away) & (np.abs(home) >= 100) & (np.abs(away) >= 100)


def market_probabilities(games):
    valid = moneyline_mask(games)
    probabilities = np.empty(len(games))
    if valid.any():
        probabilities[valid] = no_vig_home(games.loc[valid, "home_moneyline"],
                                           games.loc[valid, "away_moneyline"])
    if (~valid).any():
        spread = games.loc[~valid, "spread_line"].to_numpy(float)
        if not np.isfinite(spread).all():
            raise ValueError("A game has neither valid moneylines nor a closing spread.")
        probabilities[~valid] = ndtr(spread / MARGIN_SCALE)
    return probabilities


def bet_decimal(games, home_side):
    home_side = np.asarray(home_side, dtype=bool)
    valid = moneyline_mask(games)
    odds = np.full(len(games), -110.0)
    if valid.any():
        odds[valid] = np.where(home_side[valid], games.loc[valid, "home_moneyline"],
                               games.loc[valid, "away_moneyline"])
    return american_decimal(odds)


def unit_profit(games, home_side):
    home_side = np.asarray(home_side, dtype=bool)
    margin = games["result"].to_numpy(float).copy()
    valid = moneyline_mask(games)
    if (~valid).any():
        margin[~valid] -= games.loc[~valid, "spread_line"].to_numpy(float)
    signed = np.where(home_side, margin, -margin)
    return np.where(signed > 0, bet_decimal(games, home_side) - 1,
                    np.where(signed < 0, -1.0, 0.0))


def gate_bets(probabilities, market, kind, epsilon):
    probabilities = np.asarray(probabilities, dtype=float)
    difference = probabilities - np.asarray(market, dtype=float)
    side = (difference > 0).astype(int)
    selected = np.abs(difference) > epsilon + 1e-12
    if kind == "winner_conviction":
        selected &= side == (probabilities >= 0.5)
    elif kind != "lean":
        raise ValueError(f"Unknown gate: {kind}")
    return side, selected


def gate_name(kind, epsilon):
    return f"{kind}_eps_{int(round(epsilon * 100)):02d}"


def cluster_ratio_ci(numerator, denominator, weeks, repetitions=BOOTSTRAP_REPETITIONS):
    table = pd.DataFrame({"week": weeks, "num": numerator, "den": denominator})
    groups = table.groupby("week", sort=True)[["num", "den"]].sum().to_numpy(float)
    if not len(groups) or groups[:, 1].sum() <= 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(groups), size=(repetitions, len(groups)))
    totals = groups[samples].sum(axis=1)
    values = np.divide(totals[:, 0], totals[:, 1],
                       out=np.full(repetitions, np.nan), where=totals[:, 1] > 0)
    return np.nanquantile(values, [0.025, 0.975]).tolist()


def score_predictions(games, probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    result = games["result"].to_numpy(float)
    decisive = result != 0
    target = result > 0
    correct = ((probabilities >= 0.5) == target) & decisive
    squared = np.where(decisive, (probabilities - target) ** 2, 0)
    profit = unit_profit(games, probabilities >= 0.5)
    weeks = games["week_key"].to_numpy()
    return {
        "games": len(games), "decisive_games": int(decisive.sum()),
        "ties": int((~decisive).sum()),
        "accuracy": float(correct.sum() / decisive.sum()),
        "accuracy_ci": cluster_ratio_ci(correct, decisive, weeks),
        "brier": float(squared.sum() / decisive.sum()),
        "brier_ci": cluster_ratio_ci(squared, decisive, weeks),
        "log_loss": float(log_loss(target[decisive], probabilities[decisive], labels=[False, True])),
        "flat_roi": float(profit.mean()), "flat_profit": float(profit.sum()),
        "flat_roi_ci": cluster_ratio_ci(profit, np.ones(len(games)), weeks),
        "flat_bets": len(games),
    }


def gate_metrics(games, probabilities, market):
    output = []
    for kind, epsilon in GATES:
        side, mask = gate_bets(probabilities, market, kind, epsilon)
        profit = unit_profit(games, side) * mask
        output.append({
            "name": gate_name(kind, epsilon), "kind": kind, "epsilon": epsilon,
            "bets": int(mask.sum()), "weeks": int(games.loc[mask, "week_key"].nunique()),
            "roi": float(profit.sum() / mask.sum()) if mask.any() else None,
            "profit": float(profit.sum()),
            "roi_ci": cluster_ratio_ci(profit, mask, games.week_key),
        })
    return output


def kelly_path(games, probabilities, sides, selected):
    probabilities = np.asarray(probabilities, dtype=float)
    sides = np.asarray(sides, dtype=bool)
    selected = np.asarray(selected, dtype=bool)
    valid = moneyline_mask(games)
    home_probability = probabilities.copy()
    if (~valid).any():
        home_probability[~valid] = ndtr(
            ndtri(np.clip(probabilities[~valid], 1e-6, 1 - 1e-6))
            - games.loc[~valid, "spread_line"].to_numpy(float) / MARGIN_SCALE)
    p_side = np.where(sides, home_probability, 1 - home_probability)
    net_odds = bet_decimal(games, sides) - 1.0
    fractions = np.where(selected, np.clip(0.25 * (net_odds * p_side - (1 - p_side))
                                          / net_odds, 0, 0.02), 0.0)
    profits = unit_profit(games, sides)
    bankroll = 1000.0
    path = pd.DataFrame(index=np.arange(len(games)), data={
        "kelly_fraction": fractions, "kelly_stake": 0.0, "kelly_profit": 0.0,
        "kelly_bankroll_after_week": 0.0,
    })
    weekly_returns = []
    bankroll_history = [bankroll]
    for week in sorted(games.week_key.unique()):
        positions = np.flatnonzero(games.week_key.to_numpy() == week)
        allocation = fractions[positions].copy()
        if allocation.sum() > 1:
            allocation /= allocation.sum()
        stakes = bankroll * allocation
        week_profit = stakes * profits[positions]
        weekly_returns.append(float(week_profit.sum() / bankroll))
        bankroll += float(week_profit.sum())
        bankroll_history.append(bankroll)
        path.loc[positions, "kelly_fraction"] = allocation
        path.loc[positions, "kelly_stake"] = stakes
        path.loc[positions, "kelly_profit"] = week_profit
        path.loc[positions, "kelly_bankroll_after_week"] = bankroll
    history = np.asarray(bankroll_history)
    drawdown = history / np.maximum.accumulate(history) - 1
    rng = np.random.default_rng(SEED)
    samples = rng.choice(weekly_returns, size=(BOOTSTRAP_REPETITIONS, len(weekly_returns)))
    bootstrap_return = np.prod(1 + samples, axis=1) - 1
    total_stake = path.kelly_stake.sum()
    summary = {
        "initial_bankroll": 1000.0, "final_bankroll": bankroll,
        "total_return": bankroll / 1000 - 1,
        "total_return_ci": np.quantile(bootstrap_return, [0.025, 0.975]).tolist(),
        "max_week_end_drawdown": float(drawdown.min()),
        "bets": int((path.kelly_stake > 0).sum()), "total_staked": float(total_stake),
        "stake_roi": float(path.kelly_profit.sum() / total_stake) if total_stake else None,
        "weekly_returns": weekly_returns,
    }
    return path, summary


def clean_name(value):
    value = "" if pd.isna(value) else str(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    words = "".join(c if c.isalnum() else " " for c in value).split()
    while words and words[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        words.pop()
    return "".join(words)


def clean_team(series):
    return series.replace({"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "OAK": "LV", "SD": "LAC"})


def weighted_mean(frame, value, weight):
    if frame.empty or value not in frame or weight not in frame:
        return float("nan")
    values = pd.to_numeric(frame[value], errors="coerce").to_numpy(float)
    weights = pd.to_numeric(frame[weight], errors="coerce").to_numpy(float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else float("nan")


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else float("nan")


def prepare_table(frame, team_column, type_column):
    if frame.empty:
        return frame.copy()
    frame = frame.copy()
    frame = frame.loc[frame[type_column].eq("REG") & frame.week.gt(0)].copy()
    frame["team"] = clean_team(frame[team_column])
    frame["week_key"] = frame.season.astype(int) * 100 + frame.week.astype(int)
    return frame


def role_lookups(prior_snaps, id_map):
    if prior_snaps.empty:
        return {}, {}
    recent = prior_snaps.sort_values(["week_key", "player"]).groupby("pfr_player_id").tail(4)
    roles = recent.groupby("pfr_player_id", sort=False).agg(
        offense=("offense_pct", "mean"), defense=("defense_pct", "mean"),
        special=("st_pct", "mean"), name=("player", "last"), week_key=("week_key", "max"))
    roles["usage"] = roles[["offense", "defense", "special"]].max(axis=1)
    roles["normal_name"] = roles.name.map(clean_name)
    unique_names = roles.normal_name.value_counts()
    by_name, by_gsis = {}, {}
    for pfr_id, row in roles.iterrows():
        record = row.to_dict()
        if unique_names[row.normal_name] == 1:
            by_name[row.normal_name] = record
        gsis = id_map.get(str(pfr_id), "")
        if gsis:
            by_gsis[gsis] = record
    return by_gsis, by_name


def lookup_role(gsis, name, by_gsis, by_name):
    if pd.notna(gsis) and str(gsis) in by_gsis:
        return by_gsis[str(gsis)]
    return by_name.get(clean_name(name))


def player_profile(team, key, kickoff, injuries, snaps, depth, passing, id_map):
    report_all = injuries.loc[injuries.team.eq(team) & injuries.week_key.eq(key)] if not injuries.empty else injuries
    report = report_all.loc[report_all.date_modified.lt(kickoff)].copy() if not report_all.empty else report_all
    if not report.empty:
        report["_identity"] = report.gsis_id.fillna(report.full_name.map(clean_name))
        report = report.sort_values("date_modified").drop_duplicates("_identity", keep="last")
    prior_snaps = snaps.loc[snaps.team.eq(team) & snaps.week_key.lt(key)] if not snaps.empty else snaps
    by_gsis, by_name = role_lookups(prior_snaps, id_map)
    prior_depth = depth.loc[depth.team.eq(team) & depth.week_key.lt(key)
                            & depth.season.eq(key // 100)] if not depth.empty else depth
    if not prior_depth.empty:
        prior_depth = prior_depth.loc[prior_depth.week_key.eq(prior_depth.week_key.max())]
        starters = prior_depth.loc[pd.to_numeric(prior_depth.depth_team, errors="coerce").eq(1)
                                   & prior_depth.formation.isin(["Offense", "Defense"])].copy()
        starters["_identity"] = starters.gsis_id.fillna(starters.full_name.map(clean_name))
        starters = starters.drop_duplicates("_identity")
    else:
        starters = prior_depth
    prior_passing = passing.loc[passing.team.eq(team) & passing.week_key.lt(key)] if not passing.empty else passing
    qb_rows = starters.loc[starters.position.eq("QB")] if not starters.empty else starters
    qb_id, qb_name = "", ""
    if not qb_rows.empty:
        candidates = []
        for row in qb_rows.itertuples():
            role = lookup_role(row.gsis_id, row.full_name, by_gsis, by_name)
            candidates.append((role["offense"] if role else -1, str(row.gsis_id), row.full_name))
        _, qb_id, qb_name = sorted(candidates, reverse=True)[0]
    elif not prior_passing.empty:
        last = prior_passing.loc[prior_passing.week_key.eq(prior_passing.week_key.max())]
        qb = last.sort_values(["attempts", "player_gsis_id"], ascending=[False, True]).iloc[0]
        qb_id, qb_name = str(qb.player_gsis_id), qb.player_display_name
    impact = dict(injury_impact=0.0, injury_offense=0.0, injury_defense=0.0, injury_special=0.0)
    positive_count, matched_count = 0, 0
    injury_by_id, injury_by_name = {}, {}
    qb_on_report, qb_injury_weight = 0.0, 0.0
    if not report.empty:
        for row in report.itertuples():
            weight = INJURY_WEIGHTS.get(row.report_status, 0.0)
            injury_by_id[str(row.gsis_id)] = weight
            injury_by_name[clean_name(row.full_name)] = weight
            if qb_id and (str(row.gsis_id) == qb_id or clean_name(row.full_name) == clean_name(qb_name)):
                qb_on_report = 1.0
                qb_injury_weight = max(qb_injury_weight, weight)
            if weight <= 0:
                continue
            positive_count += 1
            role = lookup_role(row.gsis_id, row.full_name, by_gsis, by_name)
            if role is not None:
                matched_count += 1
                impact["injury_impact"] += weight * role["usage"]
                for phase in ("offense", "defense", "special"):
                    impact[f"injury_{phase}"] += weight * role[phase]
    starter_usage, available_usage, starter_loss = [], [], []
    if not starters.empty:
        for row in starters.itertuples():
            role = lookup_role(row.gsis_id, row.full_name, by_gsis, by_name)
            if role is not None:
                weight = injury_by_id.get(str(row.gsis_id),
                                          injury_by_name.get(clean_name(row.full_name), 0.0))
                starter_usage.append(role["usage"])
                available_usage.append(role["usage"] * (1 - weight))
                starter_loss.append(role["usage"] * weight)
    qb_history = prior_passing
    if not prior_passing.empty:
        qb_history = prior_passing.loc[
            prior_passing.player_gsis_id.astype(str).eq(qb_id)
            | prior_passing.player_display_name.map(clean_name).eq(clean_name(qb_name))
        ].sort_values("week_key").tail(4)
    features = {
        **impact, "injured_players": float(positive_count),
        "injury_unmatched_players": float(positive_count - matched_count),
        "injury_matched_share": matched_count / positive_count if positive_count else
                                (1.0 if not report.empty else np.nan),
        "injury_report_available": float(not report.empty),
        "snap_history_available": float(not prior_snaps.empty),
        "qb_on_report": qb_on_report, "qb_injury_weight": qb_injury_weight,
        "qb_cpoe": weighted_mean(qb_history, "completion_percentage_above_expectation", "attempts"),
        "qb_passer_rating": weighted_mean(qb_history, "passer_rating", "attempts"),
        "qb_yards_per_attempt": safe_ratio(qb_history.pass_yards.sum(), qb_history.attempts.sum())
                                if not qb_history.empty else np.nan,
        "depth_available": float(not starters.empty),
        "starter_usage_quality": float(np.mean(starter_usage)) if starter_usage else np.nan,
        "available_starter_usage": float(np.mean(available_usage)) if available_usage else np.nan,
        "starter_injury_impact": float(np.sum(starter_loss)) if starter_usage else np.nan,
        "starter_snap_coverage": safe_ratio(len(starter_usage), len(starters)),
    }
    metadata = {
        "expected_qb_id": qb_id, "expected_qb_name": qb_name,
        "depth_week_key": int(prior_depth.week_key.max()) if not prior_depth.empty else 0,
        "snap_week_key": int(prior_snaps.week_key.max()) if not prior_snaps.empty else 0,
        "injury_latest_timestamp": report.date_modified.max().isoformat() if not report.empty else "",
        "injury_positive_players": positive_count, "injury_matched_players": matched_count,
        "injury_rows_after_kickoff_excluded": len(report_all) - len(report),
    }
    return features, metadata


def ngs_profile(team, key, ngs):
    output = {}
    for kind in ("passing", "rushing", "receiving"):
        source = ngs.get(kind, pd.DataFrame())
        prior = source.loc[source.team.eq(team) & source.week_key.lt(key)] if not source.empty else source
        if not prior.empty:
            last_weeks = sorted(prior.week_key.unique())[-4:]
            prior = prior.loc[prior.week_key.isin(last_weeks)]
        output[f"ngs_{kind}_available"] = float(not prior.empty)
        if kind == "passing":
            output["ngs_team_cpoe"] = weighted_mean(prior, "completion_percentage_above_expectation", "attempts")
        elif kind == "rushing":
            output["ngs_rush_yards_per_attempt"] = safe_ratio(prior.rush_yards.sum(), prior.rush_attempts.sum()) if not prior.empty else np.nan
            output["ngs_rush_ryoe"] = weighted_mean(prior, "rush_yards_over_expected_per_att", "rush_attempts")
        else:
            output["ngs_receive_yac_over_expected"] = weighted_mean(prior, "avg_yac_above_expectation", "receptions")
            output["ngs_receive_separation"] = weighted_mean(prior, "avg_separation", "targets")
    return output


def aggregate_plays(plays):
    eligible = plays.loc[plays.season_type.eq("REG") & plays.play_type.isin(["run", "pass"])
                         & plays.epa.notna() & plays.posteam.notna()].copy()
    if "no_play" in eligible:
        eligible = eligible.loc[eligible.no_play.fillna(0).eq(0)]
    eligible["team"] = clean_team(eligible.posteam)
    dropback = eligible.qb_dropback.fillna(0).eq(1)
    eligible["plays"] = 1
    eligible["pass_plays"] = dropback.astype(int)
    eligible["rush_plays"] = (~dropback).astype(int)
    eligible["pass_epa"] = eligible.epa.where(dropback, 0.0)
    eligible["rush_epa"] = eligible.epa.where(~dropback, 0.0)
    columns = ["epa", "yards_gained", "plays", "pass_plays", "rush_plays", "pass_epa", "rush_epa"]
    return eligible.groupby(["game_id", "team"], sort=False)[columns].sum()


def rolling_profile(history):
    output = {"prior_games": float(len(history))}
    for window in (4, 8):
        rows = history[-window:]
        for name in ("points_for", "points_against", "yards_for", "yards_against"):
            output[f"{name}_{window}"] = float(np.mean([r[name] for r in rows])) if rows else np.nan
        for name, numerator, denominator in (
            ("off_epa_per_play", "off_epa", "off_plays"),
            ("def_epa_allowed", "def_epa", "def_plays"),
            ("yards_per_play", "yards_for", "off_plays"),
            ("pass_epa_per_play", "pass_epa", "pass_plays"),
            ("rush_epa_per_play", "rush_epa", "rush_plays"),
            ("pass_share", "pass_plays", "off_plays"),
        ):
            output[f"{name}_{window}"] = safe_ratio(
                sum(r[numerator] for r in rows), sum(r[denominator] for r in rows))
        output[f"rush_share_{window}"] = 1 - output[f"pass_share_{window}"]
    return output


def build_features(schedules, plays, injuries, snaps, depth, ngs, ids):
    games = schedules.loc[schedules.game_type.eq("REG")
                          & schedules.season.isin(YEARS)
                          & schedules.home_score.notna() & schedules.away_score.notna()].copy()
    games["home_team"] = clean_team(games.home_team)
    games["away_team"] = clean_team(games.away_team)
    games["week_key"] = games.season.astype(int) * 100 + games.week.astype(int)
    local = pd.to_datetime(games.gameday + " " + games.gametime, format="%Y-%m-%d %H:%M")
    games["kickoff_utc"] = local.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    games["result"] = games.home_score - games.away_score
    games = games.sort_values(["week_key", "kickoff_utc", "game_id"]).reset_index(drop=True)
    if games.game_id.duplicated().any():
        raise ValueError("Schedules contain duplicate game IDs.")
    injuries = prepare_table(injuries, "team", "game_type")
    if not injuries.empty:
        injuries["date_modified"] = pd.to_datetime(injuries.date_modified, utc=True, errors="coerce")
    snaps = prepare_table(snaps, "team", "game_type")
    snap_columns = ["offense_pct", "defense_pct", "st_pct"]
    clipped_rows = 0
    if not snaps.empty:
        clipped_rows = int(((snaps[snap_columns] < 0) | (snaps[snap_columns] > 1)).any(axis=1).sum())
        snaps[snap_columns] = snaps[snap_columns].clip(0, 1)
    depth = prepare_table(depth, "club_code", "game_type")
    ngs = {kind: prepare_table(table, "team_abbr", "season_type") for kind, table in ngs.items()}
    id_map = {}
    if not ids.empty:
        crosswalk = ids[["pfr_id", "gsis_id"]].dropna().drop_duplicates()
        valid_ids = crosswalk.groupby("pfr_id").gsis_id.nunique()
        crosswalk = crosswalk.loc[crosswalk.pfr_id.isin(valid_ids[valid_ids.eq(1)].index)]
        id_map = dict(zip(crosswalk.pfr_id.astype(str), crosswalk.gsis_id.astype(str)))
    aggregates = aggregate_plays(plays)
    histories, ratings, output = {}, {}, []
    previous_season = None
    audit = {"injury_rows_after_kickoff_excluded": 0, "snap_fraction_rows_clipped": clipped_rows}
    feature_columns = None
    for key, group in games.groupby("week_key", sort=True):
        season = int(key // 100)
        if previous_season is not None and season != previous_season:
            ratings = {team: 1500 + (rating - 1500) * 2 / 3 for team, rating in ratings.items()}
        previous_season = season
        pending = []
        for game in group.to_dict("records"):
            home, away = game["home_team"], game["away_team"]
            advantage = float(str(game.get("location", "Home")).lower() != "neutral")
            elo_difference = ratings.get(home, 1500.0) - ratings.get(away, 1500.0)
            elo_probability = float(expit(np.log(10) * (elo_difference + 55 * advantage) / 400))
            row = {
                **game, "home_field": advantage, "elo_difference": elo_difference / 400,
                "elo_home_probability": elo_probability,
            }
            team_profiles = {}
            for prefix, team in (("home", home), ("away", away)):
                history = histories.get(team, [])
                profile = rolling_profile(history)
                rest = (game["kickoff_utc"].date() - history[-1]["date"]).days if history else 7
                profile["rest_days"] = float(min(rest, 30))
                profile["season_opener"] = float(not any(h["week_key"] // 100 == season for h in history))
                player, metadata = player_profile(
                    team, key, game["kickoff_utc"], injuries, snaps, depth,
                    ngs.get("passing", pd.DataFrame()), id_map)
                profile.update(player)
                profile.update(ngs_profile(team, key, ngs))
                row.update({f"{prefix}_{name}": value for name, value in profile.items()})
                row.update({f"{prefix}_{name}": value for name, value in metadata.items()})
                row[f"{prefix}_history_week_key"] = history[-1]["week_key"] if history else 0
                audit["injury_rows_after_kickoff_excluded"] += metadata["injury_rows_after_kickoff_excluded"]
                team_profiles[prefix] = profile
            differences = [f"diff_{name}" for name in team_profiles["home"]]
            for name in team_profiles["home"]:
                row[f"diff_{name}"] = team_profiles["home"][name] - team_profiles["away"][name]
            availability = [f"{prefix}_{name}" for prefix in ("home", "away")
                            for name in ("injury_report_available", "snap_history_available",
                                         "depth_available", "ngs_passing_available",
                                         "ngs_rushing_available", "ngs_receiving_available")]
            if feature_columns is None:
                feature_columns = differences + availability + [
                    "home_field", "elo_difference", "elo_home_probability"]
            output.append(row)
            pending.append((game, elo_probability))
        for game, elo_probability in pending:
            home, away = game["home_team"], game["away_team"]
            actual = 1.0 if game["result"] > 0 else 0.0 if game["result"] < 0 else 0.5
            shift = 20 * (actual - elo_probability)
            ratings[home] = ratings.get(home, 1500.0) + shift
            ratings[away] = ratings.get(away, 1500.0) - shift
            for team, opponent, points_for, points_against in (
                (home, away, game["home_score"], game["away_score"]),
                (away, home, game["away_score"], game["home_score"]),
            ):
                if (game["game_id"], team) not in aggregates.index:
                    raise ValueError(f"Missing offensive play-by-play: {game['game_id']} {team}")
                offense = aggregates.loc[(game["game_id"], team)]
                defense = aggregates.loc[(game["game_id"], opponent)]
                histories.setdefault(team, []).append({
                    "week_key": game["week_key"], "date": game["kickoff_utc"].date(),
                    "points_for": points_for, "points_against": points_against,
                    "yards_for": offense.yards_gained, "yards_against": defense.yards_gained,
                    "off_epa": offense.epa, "def_epa": defense.epa,
                    "off_plays": offense.plays, "def_plays": defense.plays,
                    "pass_epa": offense.pass_epa, "rush_epa": offense.rush_epa,
                    "pass_plays": offense.pass_plays, "rush_plays": offense.rush_plays,
                })
    frame = pd.DataFrame(output)
    if frame.empty:
        raise ValueError("No completed regular-season games.")
    for prefix in ("home", "away"):
        for field in ("history_week_key", "snap_week_key", "depth_week_key"):
            if not (frame[f"{prefix}_{field}"] < frame.week_key).all():
                raise AssertionError(f"Lookahead in {prefix}_{field}")
    frame["market_home_probability"] = market_probabilities(frame)
    frame["market_mode"] = np.where(moneyline_mask(frame), "moneyline", "ATS_-110")
    return frame, feature_columns, audit


def week_splits(frame, n_splits=3):
    weeks = np.array(sorted(frame.week_key.unique()))
    size = max(2, len(weeks) // 5)
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=size)
    return [(np.flatnonzero(frame.week_key.isin(weeks[train])),
             np.flatnonzero(frame.week_key.isin(weeks[validation])))
            for train, validation in splitter.split(weeks)]


def training_blocks(history, calibration_weeks=CALIBRATION_WEEKS):
    weeks = sorted(history.week_key.unique())
    if len(weeks) <= calibration_weeks:
        raise ValueError("Insufficient history for an independent calibration block.")
    cutoff = weeks[-calibration_weeks]
    return history.loc[history.week_key.lt(cutoff)], history.loc[history.week_key.ge(cutoff)]


def new_model(family, parameters):
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if family == "logreg":
        return Pipeline([
            ("imputer", imputer), ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=parameters["C"], max_iter=3000,
                                          solver="lbfgs", random_state=SEED)),
        ])
    return Pipeline([
        ("imputer", imputer),
        ("model", LGBMClassifier(
            n_estimators=100, learning_rate=0.03, max_depth=3,
            num_leaves=parameters["num_leaves"], min_child_samples=30,
            reg_lambda=5.0, n_jobs=2, random_state=SEED,
            deterministic=True, force_col_wise=True, verbosity=-1,
        )),
    ])


def raw_probability(model, frame, columns):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        return np.clip(model.predict_proba(frame[columns].to_numpy(float))[:, 1], 1e-6, 1 - 1e-6)


def fit_base(family, parameters, frame, columns):
    decisive = frame.loc[frame.result.ne(0)]
    model = new_model(family, parameters)
    model.fit(decisive[columns].to_numpy(float), decisive.result.gt(0).astype(int).to_numpy())
    return model


def tune_models(core, columns):
    candidates = {
        "logreg": [{"C": value} for value in (0.1, 1.0, 10.0)],
        "lightgbm": [{"num_leaves": value} for value in (3, 7)],
    }
    choices, records = {}, []
    splits = week_splits(core)
    for family, settings in candidates.items():
        family_results = []
        for parameters in settings:
            losses = []
            for fold, (train_indices, validation_indices) in enumerate(splits, 1):
                train = core.iloc[train_indices]
                validation = core.iloc[validation_indices]
                validation = validation.loc[validation.result.ne(0)]
                model = fit_base(family, parameters, train, columns)
                probabilities = raw_probability(model, validation, columns)
                loss = log_loss(validation.result.gt(0), probabilities, labels=[False, True])
                losses.append(float(loss))
                records.append({
                    "family": family, "parameters": parameters, "fold": fold,
                    "train_last_week": int(train.week_key.max()),
                    "validation_first_week": int(validation.week_key.min()),
                    "validation_last_week": int(validation.week_key.max()),
                    "train_games": len(train), "validation_decisive_games": len(validation),
                    "log_loss": float(loss),
                })
            family_results.append((float(np.mean(losses)), parameters))
        choices[family] = min(family_results, key=lambda item: item[0])[1]
    return choices, records


def calibrated_predict(family, parameters, base, calibration, target, columns):
    if base.week_key.max() >= calibration.week_key.min() or calibration.week_key.max() >= target.week_key.min():
        raise AssertionError("Base, calibration, and prediction weeks overlap.")
    model = fit_base(family, parameters, base, columns)
    cal = calibration.loc[calibration.result.ne(0)]
    raw_cal = raw_probability(model, cal, columns)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED)
    calibrator.fit(logit(raw_cal).reshape(-1, 1), cal.result.gt(0).astype(int).to_numpy())
    raw = raw_probability(model, target, columns)
    calibrated = calibrator.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
    metadata = {
        "base_last_week": int(base.week_key.max()), "base_games": len(base),
        "calibration_first_week": int(calibration.week_key.min()),
        "calibration_last_week": int(calibration.week_key.max()),
        "calibration_games": len(calibration), "calibration_decisive_games": len(cal),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
    }
    return calibrated, raw, metadata


def walk_forward(features, columns):
    initial = features.loc[features.week_key.le(202114)]
    core = initial.loc[initial.week_key.le(202104)]
    calibration = initial.loc[initial.week_key.between(202105, 202110)]
    validation = initial.loc[initial.week_key.between(202111, 202114)].copy()
    parameters, cv_records = tune_models(core, columns)
    validation_scores, validation_gates = {}, {}
    for family in ("logreg", "lightgbm"):
        probabilities, raw, _ = calibrated_predict(
            family, parameters[family], core, calibration, validation, columns)
        validation[f"{family}_probability"] = probabilities
        validation[f"{family}_raw_probability"] = raw
        validation_scores[family] = score_predictions(validation, probabilities)
        validation_gates[family] = gate_metrics(
            validation, probabilities, validation.market_home_probability.to_numpy())
    chosen_family = min(validation_scores, key=lambda family: validation_scores[family]["log_loss"])
    eligible = [gate for gate in validation_gates[chosen_family] if gate["bets"] >= 20]
    fallback = not eligible
    if not eligible:
        eligible = [validation_gates[chosen_family][0]]
    chosen_gate = max(eligible, key=lambda gate: (gate["roi"] if gate["roi"] is not None else -np.inf,
                                                gate["bets"], gate["name"]))
    selection = {
        "chosen_model": chosen_family, "chosen_gate": chosen_gate, "minimum_gate_bets": 20,
        "gate_selection_fallback": fallback, "parameters": parameters,
        "validation_scores": validation_scores, "validation_gates": validation_gates,
        "cv_records": cv_records,
        "initial_training_games": len(initial), "tuning_core_games": len(core),
        "selection_calibration_games": len(calibration), "gate_validation_games": len(validation),
        "initial_training_decisive_games": int(initial.result.ne(0).sum()),
    }
    validation.to_csv(CACHE / "validation_predictions.csv", index=False)
    predictions, fit_audit = [], []
    test = features.loc[features.week_key.ge(202115)]
    for key, target in test.groupby("week_key", sort=True):
        history = features.loc[features.week_key.lt(key)]
        base, calibration = training_blocks(history)
        target = target.copy()
        for family in ("logreg", "lightgbm"):
            probabilities, raw, metadata = calibrated_predict(
                family, parameters[family], base, calibration, target, columns)
            target[f"{family}_probability"] = probabilities
            target[f"{family}_raw_probability"] = raw
            for name, value in metadata.items():
                target[f"{family}_{name}"] = value
            fit_audit.append({"family": family, "prediction_week": int(key), **metadata})
        target["primary_model"] = chosen_family
        target["primary_probability"] = target[f"{chosen_family}_probability"]
        predictions.append(target)
        print(f"week {int(key)}: base={len(base)}, calibration={len(calibration)}, test={len(target)}", flush=True)
    return pd.concat(predictions, ignore_index=True), selection, fit_audit


def closing_line_validation(schedules, plays):
    games = schedules.loc[schedules.game_type.eq("REG")].copy()
    regular = plays.loc[plays.season_type.eq("REG")].sort_values(["game_id", "play_id"])
    lines = regular.groupby("game_id").spread_line.first().rename("pbp_spread")
    merged = games.merge(lines, on="game_id", how="left", validate="one_to_one")
    valid = merged.spread_line.notna() & merged.pbp_spread.notna()
    differences = (merged.loc[valid, "spread_line"] - merged.loc[valid, "pbp_spread"]).abs()
    pregame = regular.loc[regular.game_seconds_remaining.eq(3600) & regular.qtr.eq(1)]
    probabilities = pregame.groupby("game_id").vegas_home_wp.first().rename("pbp_pregame_home_wp")
    joined = merged.merge(probabilities, on="game_id", how="left", validate="one_to_one")
    eligible = moneyline_mask(joined) & joined.pbp_pregame_home_wp.notna()
    market = no_vig_home(joined.loc[eligible, "home_moneyline"], joined.loc[eligible, "away_moneyline"])
    initial_wp = joined.loc[eligible, "pbp_pregame_home_wp"].to_numpy(float)
    details = joined[["game_id", "spread_line", "pbp_spread", "pbp_pregame_home_wp"]].copy()
    details.to_csv(CACHE / "closing_line_validation.csv", index=False)
    return {
        "regular_games": len(games), "pbp_spread_covered": int(valid.sum()),
        "spread_exact_matches": int((differences <= 1e-8).sum()),
        "spread_mismatches": int((differences > 1e-8).sum()),
        "spread_max_difference": float(differences.max()),
        "vegas_line_column_present": "vegas_line" in plays,
        "pregame_wp_games": int(eligible.sum()),
        "pregame_wp_vs_moneyline_mae": float(np.mean(np.abs(market - initial_wp))),
        "pregame_wp_vs_moneyline_correlation": float(np.corrcoef(market, initial_wp)[0, 1]),
    }


def load_data(offline=False):
    import nfl_data_py as nfl

    socket.setdefaulttimeout(180)
    errors = {}

    def load(name, loader, required=True):
        path = CACHE / f"{name}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        if offline:
            if required:
                raise FileNotFoundError(f"Offline cache missing: {path}")
            errors[name] = "Unavailable in offline cache."
            return pd.DataFrame()
        try:
            frame = loader()
            if frame.empty:
                raise ValueError(f"Empty source: {name}")
            frame.to_parquet(path, index=False)
            return frame
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            if required:
                raise
            return pd.DataFrame()

    def schedules_loader():
        try:
            data = nfl.import_schedules(YEARS)
            source = "nfl_data_py.import_schedules"
        except Exception as exc:
            errors["schedule_primary"] = f"{type(exc).__name__}: {exc}"
            data = pd.read_csv(SOURCE_URLS["schedule_fallback"])
            data = data.loc[data.season.isin(YEARS)].copy()
            source = SOURCE_URLS["schedule_fallback"]
        (CACHE / "schedule_source.txt").write_text(source + "\n")
        return data

    schedules = load("schedules", schedules_loader)
    tables = {"schedules": schedules}
    plays = []
    for year in YEARS:
        def loader(year=year):
            data = nfl.import_pbp_data([year], include_participation=False, downcast=False)
            if data.empty:
                raise ValueError(f"Play-by-play import returned empty for {year}")
            return data[[column for column in PBP_COLUMNS if column in data]].copy()
        frame = load(f"pbp_{year}", loader)
        tables[f"pbp_{year}"] = frame
        plays.append(frame)
    ngs = {}
    for kind in ("passing", "rushing", "receiving"):
        ngs[kind] = load(f"ngs_{kind}", lambda kind=kind: nfl.import_ngs_data(kind, YEARS))
        tables[f"ngs_{kind}"] = ngs[kind]
    injuries = load("injuries", lambda: nfl.import_injuries([2021, 2022]))
    snaps = load("snaps", lambda: nfl.import_snap_counts([2021, 2022]))
    depth = load("depth", lambda: nfl.import_depth_charts([2021, 2022]), required=False)
    ids = load("ids", nfl.import_ids, required=False)
    tables.update(injuries=injuries, snaps=snaps, depth=depth, ids=ids)
    manifest = {
        name: {
            "rows": len(frame), "columns": frame.columns.tolist(),
            "sha256": hashlib.sha256((CACHE / f"{name}.parquet").read_bytes()).hexdigest()
                      if (CACHE / f"{name}.parquet").exists() else None,
        } for name, frame in tables.items()
    }
    manifest["errors"] = errors
    manifest["source_urls"] = SOURCE_URLS
    manifest["schedule_import"] = (CACHE / "schedule_source.txt").read_text().strip()
    return (schedules, pd.concat(plays, ignore_index=True), injuries, snaps, depth, ngs, ids), manifest


def data_coverage(features, raw, feature_audit):
    schedules, plays, injuries, snaps, depth, ngs, ids = raw
    test = features.loc[features.week_key.ge(202115)]
    positive = sum(test[f"{prefix}_injury_positive_players"].sum() for prefix in ("home", "away"))
    matched = sum(test[f"{prefix}_injury_matched_players"].sum() for prefix in ("home", "away"))
    team_game_keys = set(
        (int(row.season), int(row.week), team)
        for row in features.itertuples() for team in (row.home_team, row.away_team))
    regular_injuries = prepare_table(injuries, "team", "game_type")
    orphan = sum((int(row.season), int(row.week), row.team) not in team_game_keys
                 for row in regular_injuries.itertuples())
    coverage = {
        **feature_audit, "regular_games_by_season": features.groupby("season").size().to_dict(),
        "test_games_by_season": test.groupby("season").size().to_dict(),
        "test_weeks": int(test.week_key.nunique()), "test_moneyline_games": int(moneyline_mask(test).sum()),
        "test_ats_fallback_games": int((~moneyline_mask(test)).sum()),
        "test_injured_player_reports": int(positive), "test_injured_reports_with_prior_snap_match": int(matched),
        "test_injury_snap_match_rate": safe_ratio(matched, positive),
        "injury_rows_without_completed_game": int(orphan),
        "source_rows": {"injuries": len(injuries), "snaps": len(snaps), "depth": len(depth),
                        **{f"ngs_{kind}": len(table) for kind, table in ngs.items()}},
        "team_game_test_coverage": {},
    }
    for field in ("injury_report_available", "snap_history_available", "depth_available",
                  "ngs_passing_available", "ngs_rushing_available", "ngs_receiving_available"):
        count = sum(test[f"{prefix}_{field}"].sum() for prefix in ("home", "away"))
        coverage["team_game_test_coverage"][field] = {
            "covered": int(count), "total": 2 * len(test), "rate": count / (2 * len(test))}
    return coverage


def evaluate_predictions(predictions, selection, fit_audit, features, columns, raw, manifest):
    games = predictions.copy()
    market = games.market_home_probability.to_numpy(float)
    models, gates, by_season = {}, {}, {}
    for family in ("logreg", "lightgbm"):
        probabilities = games[f"{family}_probability"].to_numpy(float)
        models[family] = score_predictions(games, probabilities)
        gates[family] = gate_metrics(games, probabilities, market)
        decisive = games.result.ne(0).to_numpy()
        target = games.result.gt(0).to_numpy()
        difference = np.where(decisive, (probabilities - target) ** 2 - (market - target) ** 2, 0)
        models[family]["brier_minus_market"] = float(difference.sum() / decisive.sum())
        models[family]["brier_minus_market_ci"] = cluster_ratio_ci(difference, decisive, games.week_key)
        games[f"{family}_pick"] = np.where(probabilities >= 0.5, games.home_team, games.away_team)
        games[f"{family}_flat_profit"] = unit_profit(games, probabilities >= 0.5)
        by_season[family] = {
            str(season): score_predictions(group, group[f"{family}_probability"])
            for season, group in games.groupby("season")}
        for kind, epsilon in GATES:
            side, selected = gate_bets(probabilities, market, kind, epsilon)
            name = f"{family}_{gate_name(kind, epsilon)}"
            games[f"{name}_bet"] = selected.astype(int)
            games[f"{name}_side"] = np.where(selected, np.where(side, games.home_team, games.away_team), "")
            games[f"{name}_profit"] = unit_profit(games, side) * selected
    chosen = selection["chosen_model"]
    chosen_gate = selection["chosen_gate"]
    side, mask = gate_bets(games[f"{chosen}_probability"], market,
                           chosen_gate["kind"], chosen_gate["epsilon"])
    games["selected_gate"] = chosen_gate["name"]
    games["selected_gate_bet"] = mask.astype(int)
    games["selected_gate_side"] = np.where(mask, np.where(side, games.home_team, games.away_team), "")
    games["selected_gate_profit"] = unit_profit(games, side) * mask
    games["home_win_outcome"] = np.where(games.result.gt(0), 1.0,
                                         np.where(games.result.lt(0), 0.0, 0.5))
    path, kelly = kelly_path(games, games[f"{chosen}_probability"].to_numpy(), side, mask)
    games = pd.concat([games.reset_index(drop=True), path], axis=1)
    selected_result = next(gate for gate in gates[chosen] if gate["name"] == chosen_gate["name"])
    best_observed = max(
        [{"model": family, **gate} for family in gates for gate in gates[family] if gate["bets"] > 0],
        key=lambda gate: gate["roi"])
    output = ROOT / "nfl_predictions.csv"
    games.to_csv(output, index=False, float_format="%.10g")
    coverage = data_coverage(features, raw, {})
    summary = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "test_games": len(games), "test_first_game": games.gameday.min(),
        "test_last_game": games.gameday.max(), "features": columns, "feature_count": len(columns),
        "models": models, "market": score_predictions(games, market), "gates": gates,
        "by_season": by_season, "selection": selection, "selected_gate_test": selected_result,
        "best_observed_test_gate_posthoc": best_observed, "kelly": kelly, "coverage": coverage,
        "fit_audit": fit_audit, "source_manifest": manifest,
        "closing_line_validation": closing_line_validation(raw[0], raw[1]),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("nfl_data_py", "pandas", "numpy", "scipy", "scikit-learn", "lightgbm", "pyarrow")},
        "python_version": sys.version.split()[0],
        "prediction_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    return games, summary


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def percent(value):
    return "n/a" if value is None else f"{value:.2%}"


def interval(values):
    return f"[{percent(values[0])}, {percent(values[1])}]"


def render_report(summary):
    selection = summary["selection"]
    chosen = selection["chosen_model"]
    primary = summary["models"][chosen]
    selected = summary["selected_gate_test"]
    coverage = summary["coverage"]
    line_check = summary["closing_line_validation"]
    mode = "closing moneyline" if coverage["test_ats_fallback_games"] == 0 else "moneyline where present; otherwise ATS at -110"
    rows = []
    for name in ("logreg", "lightgbm"):
        model = summary["models"][name]
        rows.append(
            f"| {name}{' (preselected)' if name == chosen else ''} | {percent(model['accuracy'])} "
            f"{interval(model['accuracy_ci'])} | {model['brier']:.5f} | {model['log_loss']:.5f} "
            f"| {percent(model['flat_roi'])} {interval(model['flat_roi_ci'])} |")
    market = summary["market"]
    rows.append(f"| No-vig market | {percent(market['accuracy'])} {interval(market['accuracy_ci'])} "
                f"| {market['brier']:.5f} | {market['log_loss']:.5f} "
                f"| {percent(market['flat_roi'])} {interval(market['flat_roi_ci'])} |")
    gate_rows = []
    for family in ("logreg", "lightgbm"):
        for gate in summary["gates"][family]:
            marker = " **preselected**" if family == chosen and gate["name"] == selected["name"] else ""
            gate_rows.append(
                f"| {family} | {gate['kind']}, ε={gate['epsilon']:.2f}{marker} "
                f"| {gate['bets']} | {gate['weeks']} | {percent(gate['roi'])} "
                f"| {interval(gate['roi_ci'])} |")
    year_rows = []
    for family, seasons in summary["by_season"].items():
        for year, metrics in seasons.items():
            year_rows.append(f"| {family} | {year} | {metrics['games']} | {percent(metrics['accuracy'])} "
                             f"| {percent(metrics['flat_roi'])} | {metrics['brier']:.5f} |")
    coverage_rows = [
        f"| {name} | {item['covered']}/{item['total']} | {percent(item['rate'])} |"
        for name, item in coverage["team_game_test_coverage"].items()]
    cv_rows = [
        f"| {record['family']} | `{json.dumps(record['parameters'], sort_keys=True)}` | {record['fold']} "
        f"| {record['train_last_week']} | {record['validation_first_week']}–{record['validation_last_week']} "
        f"| {record['log_loss']:.5f} |"
        for record in selection["cv_records"]]
    brier_rows = [
        f"- **{family}:** Brier minus market = {model['brier_minus_market']:+.5f}; "
        f"week-clustered 95% CI [{model['brier_minus_market_ci'][0]:+.5f}, "
        f"{model['brier_minus_market_ci'][1]:+.5f}] (negative favours the model)."
        for family, model in summary["models"].items()]
    observed = summary["best_observed_test_gate_posthoc"]
    kelly = summary["kelly"]
    verdict = (
        f"The preselected {chosen} model produced {percent(primary['accuracy'])} decisive-game accuracy "
        f"and {percent(primary['flat_roi'])} flat-stake ROI. Its preselected "
        f"{selected['name']} gate returned {percent(selected['roi'])} on {selected['bets']} bets, "
        f"with a week-clustered 95% ROI interval of {interval(selected['roi_ci'])}. "
    )
    if selected["roi_ci"][0] is not None and selected["roi_ci"][0] > 0:
        verdict += ("That interval excludes zero in this sample, but a single retrospective period and "
                    "historically revised source data do not establish a durable or deployable edge. ")
    else:
        verdict += ("This does not establish a positive betting edge: the uncertainty interval does not "
                    "exclude zero on the profitable side. ")
    verdict += (
        f"The selected model's Brier score is {primary['brier']:.5f} versus "
        f"{summary['market']['brier']:.5f} for the no-vig market; lower is better. "
    )
    if selection["chosen_gate"]["roi"] is not None and selection["chosen_gate"]["roi"] < 0:
        verdict += (
            "Even the least-losing pre-test gate had negative validation ROI in this run, so its "
            "selection is a comparison among the requested gates, not a recommendation to bet. "
        )
    verdict += ("Injury, starter and time-played features are explicitly present, but this experiment "
                "does not isolate their causal or incremental contribution; that requires a separately "
                "specified ablation and a new untouched evaluation period.")
    report = f"""# NFL walk-forward backtest

Generated UTC: {summary['created_utc']}

## Executive result

- **Test:** {summary['test_games']} completed regular-season games, {summary['test_first_game']} through {summary['test_last_game']}, in {coverage['test_weeks']} season-weeks.
- **Preselected model:** {chosen}; **accuracy:** {percent(primary['accuracy'])} on {primary['decisive_games']} decisive games. The {primary['ties']} ties remain in the prediction file and betting denominator but are two-way moneyline pushes.
- **Bet accounting:** {mode}; $1 risked per flat bet, net profit / dollars risked. Moneyline-covered test games: {coverage['test_moneyline_games']}; ATS fallback: {coverage['test_ats_fallback_games']}.
- **Flat ROI:** {percent(primary['flat_roi'])}, week-clustered 95% CI {interval(primary['flat_roi_ci'])}.
- **Best gate selected before testing:** {selected['name']}, {selected['bets']} test bets, ROI {percent(selected['roi'])}, 95% CI {interval(selected['roi_ci'])}.
- **Quarter-Kelly:** $1,000 → ${kelly['final_bankroll']:.2f}; total return {percent(kelly['total_return'])}, bootstrapped 95% CI {interval(kelly['total_return_ci'])}.

## Splits and no-lookahead protocol

Only REG games are modelled. Initial available training data are all 2020 regular-season games and 2021 weeks 1–14 ({selection['initial_training_games']} games; {selection['initial_training_decisive_games']} decisive). Test games are 2021 weeks 15–18 and every completed 2022 regular-season game. There are {coverage['test_games_by_season'].get('2021', coverage['test_games_by_season'].get(2021))} late-2021 and {coverage['test_games_by_season'].get('2022', coverage['test_games_by_season'].get(2022))} 2022 games. The cancelled 2022 BUF–CIN game has no completed result and is not a bet.

The initial training period is partitioned in advance:

1. **Tuning core:** 2020 plus 2021 weeks 1–4 ({selection['tuning_core_games']} games). Three expanding time-ordered CV folds group whole season-weeks. Median imputation, missingness indicators and scaling are fitted inside each fold. Hyperparameters minimise mean validation log loss, not test ROI.
2. **Selection-model calibration:** 2021 weeks 5–10 ({selection['selection_calibration_games']} games), entirely excluded from that base model's fit. A regularised sigmoid maps logit(raw probability) to calibrated probability.
3. **Model/gate validation:** 2021 weeks 11–14 ({selection['gate_validation_games']} games), not used to fit that base model or calibrator. Select model family by calibrated log loss; within that family choose the highest flat-ROI gate with at least {selection['minimum_gate_bets']} validation bets (ties resolved deterministically). Gate fallback used: {selection['gate_selection_fallback']}.
4. **Weekly test refits:** freeze hyperparameters, model choice and gate. Before each test season-week, use all earlier regular-season weeks; reserve its latest {CALIBRATION_WEEKS} observed weeks exclusively for a fresh sigmoid calibrator and fit each base model on the earlier remainder. First test calibration is 2021 weeks 9–14. Previously tested games may enter later training only after their week is over. Both families are predicted regardless of which was selected.

All games in a week receive features before that week's outcome, Elo, snap or NGS updates. Injury rows are the sole same-week player input and must have `date_modified < kickoff_utc`. Kickoffs are parsed in America/New_York and converted to UTC. Current-game QB names/IDs from schedules are **never** model features. Closing prices, spreads, totals, PBP win probabilities and current outcomes are **not** predictors. Prediction rows record base/calibration week cutoffs plus lagged feature-source weeks and injury timestamps.

Frozen hyperparameters: `{json.dumps(selection['parameters'], sort_keys=True)}`. Feature count: {summary['feature_count']}. All selected numeric feature names are in `.cache/evaluation.json`.

## Feature formulas

### Team form, time and Elo

For each team, use its previous **4 and 8 games**, carrying observed form across the offseason, never same-week games:

- Offensive EPA/play = Σ offensive EPA / number of eligible offensive plays. Defensive EPA allowed/play uses the opponent's eligible offensive plays; lower is better.
- Eligible PBP is non-null EPA on `play_type ∈ {{run, pass}}`; penalty/no-play records, kicks, kneels and spikes are excluded. `qb_dropback=1` defines passing (including sacks and QB scrambles); other eligible plays are rushing. Report passing/rushing EPA/play and play shares.
- Points for/against and scrimmage yards for/against are rolling per-game averages; yards/play is Σ eligible yards / eligible plays. These are scrimmage-play yards, not an official all-purpose yardage total.
- Rest = calendar days since that team's previous completed regular-season game, capped at 30 for the model; 7 when no history exists. Include a season-opener indicator and historical game count.
- Elo starts at 1500; pregame P(home) = 1 / (1 + 10^(-(E_home − E_away + 55 H)/400)), where H=1 at a non-neutral home venue, otherwise 0. After a completed week, E_home += 20(actual − expected), E_away -= the same, with actual=0.5 on ties. Between seasons regress ratings one-third toward 1500. Include Elo difference/400, Elo home probability and H.

Models use home-minus-away differences of team/player form, explicit home/away source-availability flags and home-field/Elo terms. Availability is not silently treated as health or zero talent.

### Snap-count-weighted injury impact

For player p, obtain their **latest four prior appearances for the same team**, using only weeks before the target week. Source snap fractions are clipped to [0,1] to correct rounding such as 1.01 ({coverage.get('snap_fraction_rows_clipped', 0)} regular-season source rows clipped). Let ō, d̄ and s̄ be mean offensive, defensive and special-team snap fractions; define ū_p = max(ō_p, d̄_p, s̄_p).

**I(team, week) = Σ_p a(status_p,week) × ū_p**, where a(Out)=1.0, a(Doubtful)=0.5, a(Questionable)=0.25 and otherwise 0. The sum is over that week's timestamp-eligible official injury-report players. Each player is counted once; if multiple report revisions exist, use the latest pre-kickoff one. Also include phase-specific Σ a×ō, Σ a×d̄, Σ a×s̄, number of positive-status reports, unmatched-player count and prior-snap match share.

These are **snap-equivalent lost-usage units**, not literal player-minutes or a medical absence forecast. Averaging prior appearances preserves a player's pre-injury role while absent; it is not zero-filled with subsequent missed games. No previous snaps means unknown usage: contribution is zero with explicit unmatched/missing flags, never an inferred league-average starter. Injured reserve and unlisted injuries absent from the weekly official report are not fabricated.

Player joins first use static PFR↔GSIS ID links, otherwise exact accent/punctuation/suffix-normalised full name within the same team. Ambiguous normalised names are not joined; there is no fuzzy matching. Only ID columns, not current team/age/draft attributes, are read from the contemporary crosswalk.

### Expected starter and quality/participation proxies

Use only the **latest prior-week depth snapshot in the same season**. A rank-1 QB is the expected starter; if unavailable, use the highest-attempt QB in that team's most recent prior NGS week. This is an auditable **lagged starter proxy**, not knowledge of the actual game-day announcement. `qb_on_report` flags that proxy appearing anywhere on the current pregame report (including a null game-status practice listing); `qb_injury_weight` uses the status weights above.

Prior depth rank-1 offense/defense players yield mean prior snap usage, mean usage × (1−injury weight), summed starter injury impact and starter snap-match coverage. This is **role/continuity quality**, not an independently estimated talent rating. For the expected QB use the last four prior observed NGS games: attempt-weighted CPOE and passer rating, and total passing yards / attempts.

Team NGS uses the last four observed prior team-weeks: attempt-weighted passing CPOE, rushing yards/attempt and rushing yards over expected/attempt; reception-weighted receiving YAC over expected and target-weighted separation. `week=0` full-season aggregates and postseason records are always discarded before feature calculation. NGS often covers qualifying players only, so these are covered-player aggregates, not complete official team totals.

## Predictive and flat-bet results

Intervals below are 95% percentile bootstraps resampling **whole season-weeks**, with {BOOTSTRAP_REPETITIONS:,} replicates and seed {SEED}; all games/bets in a sampled week stay together. No iid-game bootstrap is used.

| Predictor | Accuracy [95% CI] | Brier | Log loss | Flat $1 ROI [95% CI] |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

{chr(10).join(brier_rows)}

Brier/log loss/accuracy use decisive games only, matching two-way tie-refund moneyline semantics. Brier = mean((P(home win)−1_home_win)²). Market probabilities remove two-sided vig: q_home = (1/decimal_home) / ((1/decimal_home)+(1/decimal_away)). The market baseline bets its more probable winner at the actual offered price. No commission, limits, slippage, line shopping or transaction costs beyond quoted vig are modelled.

| Model | Test season | Games | Accuracy | Flat ROI | Brier |
|---|---|---:|---:|---:|---:|
{chr(10).join(year_rows)}

## Edge gates

Let δ = P_model(home) − q_market(home). The lean backs home when δ>0, otherwise away, requiring |δ|>ε (strictly). Winner-conviction additionally requires that side to be the model's ≥50% winner. At equality there is no lean. ε ∈ {{0, 0.02, 0.05}} is in absolute probability points. A no-vig edge at ε=0 need not be positive EV at the vigged offered price.

| Model | Gate | Bets | Weeks with bets | Flat ROI | Week-clustered 95% CI |
|---|---|---:|---:|---:|---:|
{chr(10).join(gate_rows)}

Pre-test-selected gate validation ROI was {percent(selection['chosen_gate']['roi'])} on {selection['chosen_gate']['bets']} bets. **Post-hoc descriptive maximum only:** {observed['model']} / {observed['name']} returned {percent(observed['roi'])} on {observed['bets']} test bets. This test-selected maximum was not used for quarter-Kelly or the primary verdict; its nominal CI does not correct for searching gates/models.

For an unavailable moneyline pair only, q_market(home)=Φ(spread_line/13.5); ATS settles `side × (home_margin−spread_line)` at −110. Positive spread means home is favoured. Kelly ATS cover probability would be Φ(Φ⁻¹(P_home_win)−spread_line/13.5), an explicit latent-normal fallback rather than a directly fitted cover model. In this run the fallback was used for **{coverage['test_ats_fallback_games']}** test games.

## Quarter-Kelly on the preselected gate

For offered net decimal payout b and model probability p for the selected side, f = min(0.02, max(0, 0.25×(b p−(1−p))/b)). Even if the no-vig gate passes, negative offered-price EV produces **zero stake**. Cap each bet at 2% of bankroll; if weekly fractions ever exceed 100%, rescale proportionally. Every wager in the same week uses the **start-of-week bankroll**; do not finance a Sunday bet using a Thursday result.

- Actual positive-stake bets: {kelly['bets']}; total staked: ${kelly['total_staked']:.2f}; net profit / total staked: {percent(kelly['stake_roi'])}.
- Initial/final bankroll: $1,000 / ${kelly['final_bankroll']:.2f}; total return: {percent(kelly['total_return'])}.
- Week-end maximum drawdown: {percent(kelly['max_week_end_drawdown'])}; this does not measure intraday mark-to-market risk.
- 95% total-return interval: {interval(kelly['total_return_ci'])}, from resampling realised weekly portfolio returns and compounding them. Neither this interval nor flat-bet intervals refit/reselect models, account for training uncertainty or guarantee future bankroll behaviour.

## Data coverage and closing-line audit

Raw source counts: `{json.dumps(coverage['source_rows'], sort_keys=True)}`.

| Lagged/pregame source | Test team-games covered | Coverage |
|---|---:|---:|
{chr(10).join(coverage_rows)}

- Positive-status test player-reports: {coverage['test_injured_player_reports']}; matched to prior same-team snaps: {coverage['test_injured_reports_with_prior_snap_match']} ({percent(coverage['test_injury_snap_match_rate'])}). Unknown players are flagged, not assigned hindsight usage.
- Injury rows excluded for non-pre-kickoff/missing timestamps: {coverage.get('injury_rows_after_kickoff_excluded', 0)}; report rows with no completed regular-season game: {coverage['injury_rows_without_completed_game']}. The latter are not turned into observations.
- Injury/snap/depth imports cover **2021–2022 only**, as requested. The 2020 training rows have source-missing indicators; early 2021 has no 2020 snap/depth warm-up. A team without report rows may have no listed injuries or missing source data; the availability flag does not distinguish these cases.
- Depth charts and NGS are historical archives, not guaranteed contemporaneous immutable snapshots. Same-week depth charts are deliberately excluded. Weekly injury `date_modified` enforces a documented timestamp cutoff but cannot recover earlier report revisions that the archive no longer contains. PBP EPA and other nflverse fields may have later vendor/model corrections. This is therefore a **strictly lagged historical-data backtest, not a certified point-in-time production simulation**.
- Schedule betting columns inspected include `home_moneyline`, `away_moneyline`, `spread_line`, `home_spread_odds`, `away_spread_odds` and `total_line`; original columns are preserved in the source manifest. These are treated as the historical closing fields supplied by nflverse's schedule source, not independently timestamped sportsbook quotes.
- Schedule/PBP spread comparison: {line_check['pbp_spread_covered']}/{line_check['regular_games']} games covered, {line_check['spread_exact_matches']} exact matches, {line_check['spread_mismatches']} mismatches, maximum absolute difference {line_check['spread_max_difference']:.3f} points.
- `vegas_line` column present: {line_check['vegas_line_column_present']}. Available validation fields are `spread_line`, `vegas_wp`, `vegas_home_wp`; use only initial q1/3600-second `vegas_home_wp`, **not** its in-game values. Pregame comparisons: {line_check['pregame_wp_games']} games; MAE versus no-vig moneyline {line_check['pregame_wp_vs_moneyline_mae']:.5f}, correlation {line_check['pregame_wp_vs_moneyline_correlation']:.5f}. That PBP value is a spread-informed win-probability model, not an independent moneyline quote or proof of closing-price timing.
- Actual schedule loader: `{summary['source_manifest']['schedule_import']}`. Optional-source errors: `{json.dumps(summary['source_manifest']['errors'], sort_keys=True)}`.

Source parquet SHA-256 hashes, schemas, model-fold cutoffs, runtime versions and full feature names are frozen in `.cache/evaluation.json`; cached parquet remains under `.cache/`. Prediction CSV SHA-256: `{summary['prediction_sha256']}`.

### Source endpoints

- nfl_data_py / nflverse schedules: {SOURCE_URLS['schedules']} (official NFLverse fallback: {SOURCE_URLS['schedule_fallback']}).
- PBP: `{SOURCE_URLS['pbp']}` for 2020–2022.
- NGS: `{SOURCE_URLS['ngs']}` for passing, rushing and receiving; filter 2020–2022.
- Official injury-report archive: `{SOURCE_URLS['injuries']}` for 2021–2022.
- Snap participation: `{SOURCE_URLS['snaps']}` for 2021–2022.
- Depth snapshots: `{SOURCE_URLS['depth']}` for 2021–2022.
- Static identity crosswalk: {SOURCE_URLS['ids']}.

These endpoints were read through installed nfl_data_py 0.3.3 on the run date. They are related data pipelines, not independent corroborating sources; the PBP spread agreement validates internal alignment only.

## Time-ordered tuning evidence

| Model | Candidate | Fold | Last training week | Validation weeks | Log loss |
|---|---|---:|---:|---|---:|
{chr(10).join(cv_rows)}

## Verdict

{verdict}
"""
    (ROOT / "NFL_BACKTEST_REPORT.md").write_text(report)
    readme = f"""# Reproduce the NFL walk-forward backtest

All source, caches and outputs live in this directory. No git command, bookmaker account, API key or paid data is required.

## Run

Use the specified Python 3.11 interpreter:

```sh
cd "/Users/x/Desktop/x/Coding Projects/betting bot/backtest/multisport/nfl"
~/.hermes/cache/scratch/msports-venv/bin/python -B nfl_backtest.py
```

On the first run, missing local parquet is imported using nfl_data_py. Subsequent runs reuse `.cache/`. To prohibit network downloads:

```sh
~/.hermes/cache/scratch/msports-venv/bin/python -B nfl_backtest.py --offline
```

The CLI exits nonzero on missing required data, failed leakage assertions or invalid predictions. It prints game count, preselected-model accuracy/flat ROI, preselected best gate/ROI and output paths. It does not place bets.

## Verify

```sh
~/.hermes/cache/scratch/msports-venv/bin/python -B -m unittest -v test_nfl_backtest
~/.hermes/cache/scratch/msports-venv/bin/python -B nfl_backtest.py --verify-only
```

The tests cover odds/vig, push and ATS-sign settlement, snap-weighted injuries, lagged QB/depth, same/future-week mutation invariance, injury timestamp filtering, chronological CV/calibration, gate semantics, week-clustered bootstrap, price-aware simultaneous-week Kelly and tie scoring.

To regenerate only the two Markdown documents from frozen measured results, without downloading data, recalculating features, refitting or rescoring:

```sh
~/.hermes/cache/scratch/msports-venv/bin/python -B nfl_backtest.py --render-only
```

## Outputs

- `nfl_predictions.csv`: {summary['test_games']} completed test games, one row per game; both raw/calibrated model probabilities, no-vig market probability, source/cutoff audit fields, home/away feature values, all gate selections/unit profits and the preselected gate's quarter-Kelly path.
- `NFL_BACKTEST_REPORT.md`: formulas, source coverage, chronological model/gate selection, prediction/betting metrics, uncertainty and verdict.
- `README_nfl.md`: this reproduction guide.
- `.cache/evaluation.json`: frozen measured results, complete feature list, CV/fit audit, versions and source/prediction hashes.
- `.cache/features.parquet`: all lagged training/test feature rows.
- `.cache/validation_predictions.csv`: pre-test 2021 weeks 11–14 predictions used for model/gate selection.
- `.cache/closing_line_validation.csv`: schedule/PBP closing-spread and pregame-WP audit.

In the CSV, `result` is home score minus away score; positive `spread_line` means home is favoured. `primary_probability` is the preselected model's calibrated home-win probability. A gate's `_bet` is 0/1, `_side` is a team abbreviation or blank and `_profit` is net dollars from a $1 risk. `selected_gate_*` is frozen before test outcomes; `kelly_*` uses actual offered-price EV and start-of-week bankroll. `home_win_outcome=0.5` marks a tie, which is excluded from binary scoring but pushes a two-way moneyline bet. Current-game outcomes/odds/QB metadata are exported for auditing/settlement, never included in the explicit model feature list.

## Dependencies and scope

Runtime used Python {summary['python_version']} and:

```text
{chr(10).join(f"{name}=={version}" for name, version in summary['versions'].items())}
```

All were already installed in the supplied venv; no packages were installed. Use `-B` to avoid Python bytecode outside this directory. CLI-managed temporary/cache locations are scoped inside this directory. This implementation changes no shared venv, repository configuration, other sport directory or git state.

The final evaluation uses {mode}; consult the report for complete assumptions. Historical injury/depth/NGS archives are not certified point-in-time snapshots. No claim of profitable deployment, incremental injury-feature benefit or future performance follows from these results.
"""
    (ROOT / "README_nfl.md").write_text(readme)


def verify_artifacts(summary):
    path = ROOT / "nfl_predictions.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != summary["prediction_sha256"]:
        raise AssertionError("Prediction CSV differs from its frozen evaluation hash.")
    predictions = pd.read_csv(path)
    if len(predictions) != summary["test_games"] or predictions.game_id.duplicated().any():
        raise AssertionError("Invalid prediction count or duplicate games.")
    if len(predictions) < 300 or not predictions.week_key.ge(202115).all():
        raise AssertionError("The required test window/coverage is missing.")
    for family in ("logreg", "lightgbm"):
        p = predictions[f"{family}_probability"]
        if not (p.notna() & p.between(0, 1)).all():
            raise AssertionError(f"Invalid {family} probabilities.")
        if not (predictions[f"{family}_calibration_last_week"] < predictions.week_key).all():
            raise AssertionError("Training/calibration lookahead.")
    for prefix in ("home", "away"):
        for source in ("history_week_key", "snap_week_key", "depth_week_key"):
            if not (predictions[f"{prefix}_{source}"] < predictions.week_key).all():
                raise AssertionError("Feature lookahead.")
        timestamps = pd.to_datetime(predictions[f"{prefix}_injury_latest_timestamp"], utc=True)
        kickoff = pd.to_datetime(predictions.kickoff_utc, utc=True)
        if not (timestamps.isna() | timestamps.lt(kickoff)).all():
            raise AssertionError("Post-kickoff injury report entered features.")
    for filename in ("NFL_BACKTEST_REPORT.md", "README_nfl.md"):
        if not (ROOT / filename).is_file() or (ROOT / filename).stat().st_size == 0:
            raise AssertionError(f"Missing deliverable: {filename}")
    return predictions


def print_final(summary):
    family = summary["selection"]["chosen_model"]
    model = summary["models"][family]
    gate = summary["selected_gate_test"]
    print(f"Test games: {summary['test_games']}")
    print(f"Model: {family}; accuracy: {percent(model['accuracy'])} ({model['decisive_games']} decisive)")
    print(f"Flat $1 ROI: {percent(model['flat_roi'])}")
    print(f"Best gate (pre-test selected): {gate['name']}; ROI: {percent(gate['roi'])}; bets: {gate['bets']}")
    for name in ("nfl_predictions.csv", "NFL_BACKTEST_REPORT.md", "README_nfl.md"):
        print(ROOT / name)


def main():
    parser = argparse.ArgumentParser(description="Leakage-audited NFL walk-forward backtest")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use the requested Python 3.11 interpreter.")
    CACHE.mkdir(exist_ok=True)
    (ROOT / ".tmp").mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(ROOT / ".tmp")
    os.environ["XDG_CACHE_HOME"] = str(CACHE)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    summary_path = CACHE / "evaluation.json"
    if args.render_only or args.verify_only:
        summary = json.loads(summary_path.read_text())
        if args.render_only:
            render_report(summary)
        verify_artifacts(summary)
        print_final(summary)
        return
    raw, manifest = load_data(offline=args.offline)
    features, columns, feature_audit = build_features(*raw)
    features.to_parquet(CACHE / "features.parquet", index=False)
    predictions, selection, fit_audit = walk_forward(features, columns)
    _, summary = evaluate_predictions(predictions, selection, fit_audit, features, columns, raw, manifest)
    summary["coverage"].update(feature_audit)
    summary = json_safe(summary)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    render_report(summary)
    verify_artifacts(summary)
    print_final(summary)


if __name__ == "__main__":
    main()
