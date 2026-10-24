from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.v3 import player_features as pf


KEYS = ["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"]
METRICS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT"]
OUTPUT_COLUMNS = KEYS + [
    "team_rating", "top8_share", "depth_dropoff", "pts_margin_ewma",
    "margin_std_ewma", "plus_minus_ewma",
]
MINUTES = np.array([40, 35, 30, 28, 25, 22, 20, 18, 12, 10], dtype=float)
DECAY = 2 ** (-1 / 10)


def player_row(
    game_id: str, game_date: str, player_id: int, minutes: float
) -> dict:
    return {
        "TEAM_ABBREVIATION": "AAA",
        "GAME_ID": game_id,
        "GAME_DATE": game_date,
        "PLAYER_ID": str(player_id),
        "MIN": minutes,
        "PTS": minutes * (0.25 + 0.01 * player_id),
        "REB": minutes * 0.2,
        "AST": minutes * 0.1,
        "STL": minutes * 0.02,
        "BLK": minutes * 0.03,
        "TOV": minutes * 0.05,
        "FGA": minutes * 0.4,
        "FG3A": minutes * 0.2,
        "FG_PCT": 0.5 if minutes else np.nan,
        "FG3_PCT": 0.4 if minutes else np.nan,
        "PLUS_MINUS": float(player_id - 5),
    }


def make_data(days: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players, teams, games = [], [], []
    for number, day in enumerate(pd.date_range("2021-10-20", periods=days)):
        key = {
            "TEAM_ABBREVIATION": "AAA",
            "GAME_ID": f"002210{number:04d}",
            "GAME_DATE": day.strftime("%Y-%m-%d"),
        }
        teams.append({**key, "PLUS_MINUS": float(8 + 4 * number)})
        games.append({**key, "PTS": 100 + number, "WL": "W"})
        players.extend(
            player_row(key["GAME_ID"], key["GAME_DATE"], pid, minutes)
            for pid, minutes in enumerate(MINUTES, start=1)
        )
    games.append({
        "TEAM_ABBREVIATION": "AAA",
        "GAME_ID": "0022199999",
        "GAME_DATE": "2021-11-01",
        "PTS": -999999,
        "WL": "L",
    })
    return pd.DataFrame(players), pd.DataFrame(teams), pd.DataFrame(games)


def test_features_match_hand_computed_single_game_history() -> None:
    players, teams, games = make_data()
    players = pd.concat([
        players, pd.DataFrame([player_row("0022199999", "2021-11-01", 99, 9999)])
    ], ignore_index=True)
    teams = pd.concat([
        teams,
        pd.DataFrame([{
            **games.iloc[-1][KEYS].to_dict(), "PLUS_MINUS": -999999,
        }]),
    ], ignore_index=True)
    result = pf.build_player_features(players, teams, games)
    assert len(result) == 1
    row = result.iloc[0]
    ratings = 36 * (0.74 + 0.01 * np.arange(1, 11))
    assert row["team_rating"] == pytest.approx(MINUTES[:8] @ ratings[:8] / 218)
    assert row["top8_share"] == pytest.approx(218 / 240)
    assert row["depth_dropoff"] == pytest.approx(ratings[4] - ratings[8])
    assert row["pts_margin_ewma"] == pytest.approx(8)
    assert np.isnan(row["margin_std_ewma"])
    assert row["plus_minus_ewma"] == pytest.approx(
        MINUTES[:8] @ (np.arange(1, 9) - 5) / 218
    )
    assert row["team_min_ewma"] == pytest.approx(240)
    for metric in METRICS:
        rates = players.iloc[:8][metric].to_numpy() / MINUTES[:8]
        assert row[f"lineup_{metric}_per_min_ewma"] == pytest.approx(
            MINUTES[:8] @ rates / 218
        )
        assert row[f"lineup_{metric}_per_game_ewma"] == pytest.approx(
            MINUTES[:8] @ players.iloc[:8][metric].to_numpy() / 218
        )
    assert row["GAME_ID"] == "0022199999"
    pf.assert_no_lookahead(result)


def test_player_ewmas_match_pandas_and_separate_team_histories() -> None:
    players, _, _ = make_data(3)
    players = players.loc[players["PLAYER_ID"].eq("1")].copy()
    players["MIN"] = [10, 20, 0]
    players["PTS"] = [10, 40, 0]
    players["FG_PCT"] = [0.4, 0.8, np.nan]
    players["FG3_PCT"] = [0.0, 0.5, np.nan]
    players.loc[players["MIN"].eq(0), ["FGA", "FG3A"]] = 0
    other_team = players.iloc[[0]].copy()
    other_team["TEAM_ABBREVIATION"] = "BBB"
    other_team["PTS"] = 1_000_000
    combined = pd.concat([players, other_team], ignore_index=True)
    before = combined.copy(deep=True)
    result = pf.player_rolling_ratings(combined, "2021-11-01")
    assert len(result) == 2
    row = result.set_index("TEAM_ABBREVIATION").loc["AAA"]
    for metric in METRICS:
        values = players[metric].fillna(0)
        expected_game = values.ewm(halflife=10, adjust=True).mean().iloc[-1]
        expected_min = (
            values / players["MIN"].replace(0, np.nan)
        ).ewm(halflife=10, adjust=True, ignore_na=False).mean().iloc[-1]
        assert row[f"{metric}_per_game_ewma"] == pytest.approx(expected_game)
        assert row[f"{metric}_per_min_ewma"] == pytest.approx(expected_min)
    assert row["plus_minus_ewma"] == pytest.approx(-4)
    pd.testing.assert_frame_equal(combined, before)


def test_halflife_is_ten_player_games() -> None:
    players, _, _ = make_data(11)
    players = players.loc[players["PLAYER_ID"].eq("1")].copy()
    players["PTS"] = [10] + [0] * 9 + [30]
    result = pf.player_rolling_ratings(players, "2021-11-01")
    assert len(result) == 1
    expected = (0.5 * 10 + 30) / np.power(DECAY, np.arange(10, -1, -1)).sum()
    assert result.iloc[0]["PTS_per_game_ewma"] == pytest.approx(expected)


def test_rotation_uses_ten_team_games_with_absences_as_zero() -> None:
    players, teams, games = make_data(12)
    players = players.loc[~(
        players["PLAYER_ID"].eq("1") & players["GAME_DATE"].eq("2021-10-31")
    )].copy()
    old_star = pd.DataFrame([
        player_row(f"002210{number:04d}", f"2021-10-{20 + number}", 99, 100)
        for number in (0, 1)
    ])
    players = pd.concat([players, old_star], ignore_index=True)
    result = pf.build_player_features(players, teams, games)
    assert len(result) == 1
    row = result.iloc[0]
    rotation_minutes = MINUTES.copy()
    weights = np.power(DECAY, np.arange(9, -1, -1))
    rotation_minutes[0] *= (weights.sum() - 1) / weights.sum()
    order = np.argsort(-rotation_minutes, kind="stable")
    top = order[:8]
    ratings = 36 * (0.74 + 0.01 * np.arange(1, 11))
    assert row["team_rating"] == pytest.approx(
        rotation_minutes[top] @ ratings[top] / rotation_minutes[top].sum()
    )
    assert row["top8_share"] == pytest.approx(
        rotation_minutes[top].sum() / rotation_minutes.sum()
    )
    assert row["depth_dropoff"] == pytest.approx(ratings[order[4]] - ratings[order[8]])
    totals = np.array([340, 340] + [240] * 9 + [200])
    all_weights = np.power(DECAY, np.arange(11, -1, -1))
    assert row["team_min_ewma"] == pytest.approx(all_weights @ totals / all_weights.sum())
    margins = teams["PLUS_MINUS"].ewm(halflife=10, adjust=True)
    assert row["pts_margin_ewma"] == pytest.approx(margins.mean().iloc[-1])
    assert row["margin_std_ewma"] == pytest.approx(margins.std(bias=False).iloc[-1])


@pytest.mark.parametrize("cutoff", ["2021-11-01", "2021-11-05", "2021-11-12"])
@pytest.mark.parametrize("perturbation", ["replace_and_add", "remove"])
def test_current_and_future_data_cannot_change_features(
    cutoff: str, perturbation: str
) -> None:
    players, teams, games = make_data(25)
    targets = games.loc[games["GAME_DATE"].eq(cutoff)].copy()
    originals = [frame.copy(deep=True) for frame in (players, teams, targets)]
    reference = pf.build_player_features(players, teams, targets)
    assert len(reference) == len(targets)
    altered_players, altered_teams = players.copy(deep=True), teams.copy(deep=True)
    future_players = altered_players["GAME_DATE"].ge(cutoff)
    future_teams = altered_teams["GAME_DATE"].ge(cutoff)
    if perturbation == "remove":
        altered_players = altered_players.loc[~future_players]
        altered_teams = altered_teams.loc[~future_teams]
    else:
        altered_players.loc[future_players, ["MIN", "PTS", "REB", "AST", "STL",
                                             "BLK", "TOV", "PLUS_MINUS"]] = 999999
        altered_players.loc[future_players, ["FG_PCT", "FG3_PCT"]] = 0.99
        newcomer = altered_players.loc[future_players].iloc[[0]].copy()
        newcomer["PLAYER_ID"] = "99999999"
        altered_players = pd.concat([altered_players, newcomer], ignore_index=True)
        altered_teams.loc[future_teams, "PLUS_MINUS"] = -999999
    shuffled_targets = targets.sample(frac=1, random_state=23)
    shuffled_targets.loc[:, "PTS"] = 1_000_000
    shuffled_targets.loc[:, "WL"] = "changed"
    actual = pf.build_player_features(
        altered_players.sample(frac=1, random_state=7),
        altered_teams.sample(frac=1, random_state=11),
        shuffled_targets,
    )
    pd.testing.assert_frame_equal(reference, actual, check_exact=True)
    for frame, original in zip((players, teams, targets), originals):
        pd.testing.assert_frame_equal(frame, original)
    pf.assert_no_lookahead(actual)


@pytest.mark.parametrize("bad_date", ["2021-11-01", "2021-11-02"])
def test_rating_entry_point_asserts_strictly_prior_dates(bad_date: str) -> None:
    players, _, _ = make_data()
    players.loc[0, "GAME_DATE"] = bad_date
    with pytest.raises(AssertionError, match="strictly before"):
        pf.player_rolling_ratings(players, "2021-11-01")


def test_same_calendar_day_is_excluded_even_with_earlier_time_or_game_id() -> None:
    players, teams, games = make_data()
    baseline = pf.build_player_features(players, teams, games)
    assert len(baseline) == 1
    players = pd.concat([
        players,
        pd.DataFrame([player_row("0000000000", "2021-11-01 01:00:00", 99, 1000)]),
    ], ignore_index=True)
    teams = pd.concat([
        teams, pd.DataFrame([{
            "TEAM_ABBREVIATION": "AAA", "GAME_ID": "0000000000",
            "GAME_DATE": "2021-11-01 01:00:00", "PLUS_MINUS": 1000,
        }]),
    ], ignore_index=True)
    games.loc[games["GAME_DATE"].eq("2021-11-01"), "GAME_DATE"] = "2021-11-01 23:00:00"
    pd.testing.assert_frame_equal(
        baseline, pf.build_player_features(players, teams, games), check_exact=True
    )


def test_cold_starts_and_short_rotations_are_not_backfilled() -> None:
    players, teams, games = make_data()
    targets = games.iloc[[-1]].copy()
    cold = pf.build_player_features(players.iloc[:0], teams.iloc[:0], targets)
    assert len(cold) == 1
    assert cold[OUTPUT_COLUMNS[3:] + ["team_min_ewma"]].isna().all().all()
    short = pf.build_player_features(
        players.loc[players["PLAYER_ID"].isin(["1", "2", "3", "4"])], teams, targets
    )
    assert len(short) == 1
    assert short.iloc[0]["top8_share"] == pytest.approx(1)
    assert np.isnan(short.iloc[0]["depth_dropoff"])
    assert np.isnan(short.iloc[0]["margin_std_ewma"])


def test_zero_minutes_and_no_attempt_percentages_do_not_create_infinity() -> None:
    players, teams, games = make_data()
    zero = pd.DataFrame([player_row("0022100000", "2021-10-20", 99, 0)])
    zero["PLUS_MINUS"] = 0
    players = pd.concat([players, zero], ignore_index=True)
    ratings = pf.player_rolling_ratings(players, "2021-11-01")
    assert len(ratings) == 11
    zero_rating = ratings.loc[ratings["PLAYER_ID"].eq("99")].iloc[0]
    assert zero_rating["FG_PCT_per_game_ewma"] == 0
    assert np.isnan(zero_rating["PTS_per_min_ewma"])
    result = pf.build_player_features(players, teams, games)
    assert len(result) == 1
    assert np.isfinite(result[OUTPUT_COLUMNS[3:]].drop(columns="margin_std_ewma")).all().all()
    assert result.iloc[0]["top8_share"] == pytest.approx(218 / 240)


@pytest.mark.parametrize("percentage,attempts", [("FG_PCT", "FGA"), ("FG3_PCT", "FG3A")])
def test_missing_percentages_with_attempts_are_not_silently_imputed(
    percentage: str, attempts: str
) -> None:
    players, teams, games = make_data()
    players.loc[0, percentage] = np.nan
    players.loc[0, attempts] = 1
    with pytest.raises(ValueError, match="attempt"):
        pf.build_player_features(players, teams, games)


@pytest.mark.parametrize("duplicate_input", [0, 1, 2])
def test_duplicate_source_rows_fail_instead_of_double_counting(duplicate_input: int) -> None:
    inputs = list(make_data())
    frame = inputs[duplicate_input]
    inputs[duplicate_input] = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        pf.build_player_features(*inputs)


def test_missing_historical_player_game_is_rejected() -> None:
    players, teams, games = make_data(2)
    players = players.loc[players["GAME_DATE"].eq("2021-10-20")]
    with pytest.raises(ValueError, match="histor"):
        pf.build_player_features(players, teams, games)


@pytest.mark.parametrize("column", ["player_history_max_date", "team_history_max_date"])
def test_output_provenance_check_rejects_a_same_date_source(column: str) -> None:
    players, teams, games = make_data()
    result = pf.build_player_features(players, teams, games)
    assert len(result) == 1
    result[column] = pd.Timestamp("2021-11-01")
    with pytest.raises(AssertionError, match="lookahead"):
        pf.assert_no_lookahead(result)
