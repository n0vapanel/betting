"""Build player-aggregate backtest features from strictly prior calendar dates.

Each target uses player and team histories filtered by GAME_DATE < target_date,
including rotation selection. Target-game participants, outcomes, and all other
same-date games are excluded. assert_no_lookahead checks the latest source dates
for every output row; tests also replace/remove same-date and future inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HALFLIFE = 10
ROTATION_WINDOW = 10
START_DATE = "2021-11-01"
CACHE_DIR = Path(__file__).resolve().parents[2].parent / "betting bot/backtest/cache"
OUTPUT_PATH = Path(__file__).resolve().with_name("player_features_2021_22.csv")
KEYS = ["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"]
METRICS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT"]
PRODUCTIVITY_WEIGHTS = np.array([1, 1.2, 1.5, 3, 3, -1], dtype=float)
STAT_COLUMNS = [
    f"{metric}_{unit}_ewma"
    for unit in ("per_game", "per_min")
    for metric in METRICS
]
OUTPUT_COLUMNS = KEYS + [
    "team_rating", "top8_share", "depth_dropoff", "pts_margin_ewma",
    "margin_std_ewma", "plus_minus_ewma",
]
DETAIL_COLUMNS = ["team_min_ewma"] + [f"lineup_{column}" for column in STAT_COLUMNS]
AUDIT_COLUMNS = ["player_history_max_date", "team_history_max_date"]
PLAYER_COLUMNS = ["TEAM_ABBREVIATION", "PLAYER_ID"] + STAT_COLUMNS + [
    "player_min_ewma", "plus_minus_ewma", "rating_per36",
]


def _calendar_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype(str).str[:10], format="%Y-%m-%d", errors="raise")


def _prepare(
    logs: pd.DataFrame, columns: list[str], unique: list[str], label: str
) -> pd.DataFrame:
    missing = set(columns) - set(logs.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    frame = logs.loc[:, columns].copy()
    identifiers = [column for column in columns if column.endswith("_ID")]
    required = ["TEAM_ABBREVIATION", "GAME_DATE"] + identifiers
    if frame[required].isna().any().any():
        raise ValueError(f"{label}: missing game/team/player identifiers or dates")
    for column in ["TEAM_ABBREVIATION"] + identifiers:
        frame[column] = frame[column].astype(str)
    frame["GAME_DATE"] = _calendar_dates(frame["GAME_DATE"])
    if frame["GAME_DATE"].isna().any():
        raise ValueError(f"{label}: invalid GAME_DATE")
    if frame.duplicated(unique).any():
        raise ValueError(f"{label}: duplicate rows for {unique}")
    return frame.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def _prepare_players(logs: pd.DataFrame) -> pd.DataFrame:
    numeric = ["MIN", "FGA", "FG3A"] + METRICS + ["PLUS_MINUS"]
    frame = _prepare(
        logs, KEYS + ["PLAYER_ID"] + numeric,
        ["TEAM_ABBREVIATION", "GAME_ID", "PLAYER_ID"], "player logs",
    )
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    for percentage, attempts in (("FG_PCT", "FGA"), ("FG3_PCT", "FG3A")):
        missing = frame[percentage].isna()
        if (missing & frame[attempts].ne(0)).any():
            raise ValueError(f"{percentage}: missing percentage with nonzero attempts")
        frame.loc[missing, percentage] = 0.0
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("player logs: nonfinite numeric values")
    if frame[numeric[:-1]].lt(0).any().any():
        raise ValueError("player logs: negative minutes, attempts, or box-score values")
    if frame[["FG_PCT", "FG3_PCT"]].gt(1).any().any():
        raise ValueError("player logs: shooting percentages must be fractions in [0, 1]")
    return frame


def _weights(length: int) -> np.ndarray:
    return np.exp2(-np.arange(length - 1, -1, -1, dtype=float) / HALFLIFE)


def _assert_prior(history: pd.DataFrame, target_date: pd.Timestamp) -> None:
    assert history.empty or history["GAME_DATE"].max() < target_date, (
        "source games must be strictly before the target calendar date"
    )


def _ratings_from_history(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (team, player_id), games in history.groupby(["TEAM_ABBREVIATION", "PLAYER_ID"]):
        games = games.sort_values(["GAME_DATE", "GAME_ID"])
        weights = _weights(len(games))
        minutes = games["MIN"].to_numpy(dtype=float)
        values = games[METRICS].to_numpy(dtype=float)
        per_game = weights @ values / weights.sum()
        positive = minutes > 0
        per_min = np.full(len(METRICS), np.nan)
        if positive.any():
            per_min = (
                weights[positive] @ (values[positive] / minutes[positive, None])
                / weights[positive].sum()
            )
        rows.append({
            "TEAM_ABBREVIATION": team,
            "PLAYER_ID": player_id,
            **dict(zip(STAT_COLUMNS, np.concatenate([per_game, per_min]))),
            "player_min_ewma": weights @ minutes / weights.sum(),
            "plus_minus_ewma": weights @ games["PLUS_MINUS"].to_numpy() / weights.sum(),
            "rating_per36": 36 * (per_min[:6] @ PRODUCTIVITY_WEIGHTS),
        })
    return pd.DataFrame(rows, columns=PLAYER_COLUMNS)


def player_rolling_ratings(
    history: pd.DataFrame, target_date: str | pd.Timestamp
) -> pd.DataFrame:
    """Return team-specific player EWMAs; assert every input date is strictly prior.

    Per-game means include all recorded appearances. Per-minute means omit zero-
    minute rates but retain those appearances in the exponential age clock.
    Missing percentages are zero only when the corresponding attempt count is zero.
    """
    history = _prepare_players(history)
    target = pd.Timestamp(target_date).normalize()
    _assert_prior(history, target)
    return _ratings_from_history(history)


def _features_before(
    players: pd.DataFrame, teams: pd.DataFrame, target_date: pd.Timestamp
) -> dict:
    players = players.loc[players["GAME_DATE"].lt(target_date)]
    teams = teams.loc[teams["GAME_DATE"].lt(target_date)].sort_values(["GAME_DATE", "GAME_ID"])
    _assert_prior(players, target_date)
    _assert_prior(teams, target_date)
    values = dict.fromkeys(OUTPUT_COLUMNS[3:] + DETAIL_COLUMNS, np.nan)
    values.update({
        "player_history_max_date": players["GAME_DATE"].max(),
        "team_history_max_date": teams["GAME_DATE"].max(),
    })
    historical_keys = ["GAME_DATE", "GAME_ID"]
    player_games = set(map(tuple, players[historical_keys].drop_duplicates().to_numpy()))
    team_games = set(map(tuple, teams[historical_keys].to_numpy()))
    if player_games != team_games:
        raise ValueError("player and team logs have mismatched historical game coverage")
    if teams.empty:
        return values
    margins = teams["PLUS_MINUS"].ewm(halflife=HALFLIFE, adjust=True)
    values["pts_margin_ewma"] = margins.mean().iloc[-1]
    values["margin_std_ewma"] = margins.std(bias=False).iloc[-1]
    total_minutes = players.groupby("GAME_ID")["MIN"].sum().reindex(teams["GAME_ID"])
    history_weights = _weights(len(teams))
    values["team_min_ewma"] = history_weights @ total_minutes.to_numpy() / history_weights.sum()
    recent_games = teams["GAME_ID"].iloc[-ROTATION_WINDOW:]
    minutes = (
        players.loc[players["GAME_ID"].isin(recent_games)]
        .pivot(index="GAME_ID", columns="PLAYER_ID", values="MIN")
        .reindex(recent_games)
        .fillna(0)
    )
    recent_weights = _weights(len(recent_games))
    rotation_minutes = pd.Series(
        recent_weights @ minutes.to_numpy() / recent_weights.sum(),
        index=minutes.columns, name="rotation_minutes_ewma",
    )
    rotation = _ratings_from_history(players).join(rotation_minutes, on="PLAYER_ID")
    rotation = rotation.loc[rotation["rotation_minutes_ewma"].gt(0)].sort_values(
        ["rotation_minutes_ewma", "PLAYER_ID"], ascending=[False, True]
    )
    if rotation.empty:
        return values
    top8 = rotation.iloc[:8]
    expected_minutes = top8["rotation_minutes_ewma"].to_numpy()
    top8_minutes = expected_minutes.sum()
    values["team_rating"] = expected_minutes @ top8["rating_per36"].to_numpy() / top8_minutes
    values["top8_share"] = top8_minutes / rotation["rotation_minutes_ewma"].sum()
    values["plus_minus_ewma"] = (
        expected_minutes @ top8["plus_minus_ewma"].to_numpy() / top8_minutes
    )
    aggregates = expected_minutes @ top8[STAT_COLUMNS].to_numpy() / top8_minutes
    values.update(zip((f"lineup_{column}" for column in STAT_COLUMNS), aggregates))
    if len(rotation) >= 9:
        values["depth_dropoff"] = rotation.iloc[4]["rating_per36"] - rotation.iloc[8]["rating_per36"]
    return values


def build_player_features(
    player_logs: pd.DataFrame,
    team_logs: pd.DataFrame,
    schedule: pd.DataFrame,
    start_date: str = START_DATE,
) -> pd.DataFrame:
    """Return one row per scheduled team-game, detailed aggregates, and source dates.

    Only the schedule's identifiers/dates are used, never its scores or outcomes.
    The CSV schema is OUTPUT_COLUMNS; DETAIL_COLUMNS remain available in memory.
    """
    players = _prepare_players(player_logs)
    teams = _prepare(
        team_logs, KEYS + ["PLUS_MINUS"],
        ["TEAM_ABBREVIATION", "GAME_ID"], "team logs",
    )
    teams["PLUS_MINUS"] = pd.to_numeric(teams["PLUS_MINUS"], errors="raise")
    if not np.isfinite(teams["PLUS_MINUS"].to_numpy(dtype=float)).all():
        raise ValueError("team logs: nonfinite points margins")
    games = _prepare(schedule, KEYS, ["TEAM_ABBREVIATION", "GAME_ID"], "schedule")
    targets = games.loc[games["GAME_DATE"].ge(pd.Timestamp(start_date).normalize())]
    targets = targets.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBREVIATION"])
    players_by_team = dict(tuple(players.groupby("TEAM_ABBREVIATION")))
    logs_by_team = dict(tuple(teams.groupby("TEAM_ABBREVIATION")))
    rows = []
    for team, game_date, game_id in targets[KEYS].itertuples(index=False, name=None):
        rows.append({
            "TEAM_ABBREVIATION": team,
            "GAME_DATE": game_date,
            "GAME_ID": game_id,
            **_features_before(
                players_by_team.get(team, players.iloc[:0]),
                logs_by_team.get(team, teams.iloc[:0]), game_date,
            ),
        })
    features = pd.DataFrame(rows, columns=OUTPUT_COLUMNS + DETAIL_COLUMNS + AUDIT_COLUMNS)
    features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    for column in AUDIT_COLUMNS:
        features[column] = pd.to_datetime(features[column])
    assert_no_lookahead(features)
    return features


def assert_no_lookahead(features: pd.DataFrame) -> None:
    """Assert zero output rows use a player/team source on or after their game date."""
    target_dates = _calendar_dates(features["GAME_DATE"])
    for column in AUDIT_COLUMNS:
        source_dates = pd.to_datetime(features[column])
        leaked = source_dates.notna() & source_dates.ge(target_dates)
        assert not leaked.any(), f"lookahead in {column}: {int(leaked.sum())} rows"


def main() -> None:
    players = pd.read_csv(
        CACHE_DIR / "nba_playerlog_2021_22.csv", dtype={"GAME_ID": str, "PLAYER_ID": str}
    )
    teams = pd.read_csv(CACHE_DIR / "nba_teamlog_2021_22.csv", dtype={"GAME_ID": str})
    schedule = pd.read_csv(CACHE_DIR / "nba_logs_2021_22.csv", dtype={"GAME_ID": str})
    features = build_player_features(players, teams, schedule)
    output = features.loc[:, OUTPUT_COLUMNS]
    output.to_csv(OUTPUT_PATH, index=False)
    print("10-row preview:")
    print(output.head(10).to_string(index=False))
    print(f"\nRow count: {len(output):,}")
    print(f"Date range: {output['GAME_DATE'].min()} to {output['GAME_DATE'].max()}")
    print("\nNull counts:")
    print(output.isna().sum().to_string())
    print("\nNumeric sanity stats:")
    print(output[OUTPUT_COLUMNS[3:]].agg(["min", "max", "mean"]).to_string())
    print(f"\nNo-lookahead assertion passed for all {len(output):,} rows: source dates < GAME_DATE.")
    print(f"Feature file: {OUTPUT_PATH}")
    print(f"Row count: {len(output)}")


if __name__ == "__main__":
    main()
