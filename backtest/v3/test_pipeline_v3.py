import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest


def module(name):
    qualified = f"backtest.v3.{name}"
    assert importlib.util.find_spec(qualified) is not None, f"{name} is not implemented"
    return importlib.import_module(qualified)


def test_rolling_features_ignore_all_same_day_and_future_results():
    f = module("features_v3")
    rng = np.random.default_rng(202210)
    for length in range(8, 24):
        dates = pd.date_range("2022-01-01", periods=length).strftime("%Y-%m-%d")
        source = pd.DataFrame({
            "team": "A", "date": dates, "game_id": [str(i) for i in range(length)],
            "points": rng.normal(110, 12, length),
        })
        target = dates[-3]
        original = source.copy(deep=True)
        before = f.lagged_aggregates(source, ["points"], min_periods=2)
        changed = source.copy()
        changed.loc[changed.date.ge(target), "points"] = 99999
        changed = pd.concat([changed, changed.loc[changed.date.eq(target)].assign(game_id="same-day")])
        after = f.lagged_aggregates(changed, ["points"], min_periods=2)
        pd.testing.assert_frame_equal(source, original)
        cols = ["points_mean", "points_ewma", "points_std", "history_max_date"]
        pd.testing.assert_series_equal(
            before.loc[before.date.eq(target), cols].iloc[0],
            after.loc[after.game_id.eq(str(length - 3)), cols].iloc[0],
            check_names=False,
        )
        prior = source.loc[source.date.lt(target), "points"]
        row = before.loc[before.date.eq(target)].iloc[0]
        assert row.points_mean == pytest.approx(prior.iloc[-10:].mean())
        assert row.points_ewma == pytest.approx(prior.ewm(halflife=10, adjust=True).mean().iloc[-1])
        assert row.history_max_date < target


def test_current_odds_features_use_only_open_marker():
    f = module("features_v3")
    source = pd.DataFrame({
        "game_id": ["0022100001"] * 4,
        "game_date": ["2021-11-01T00:00:00"] * 4,
        "odds_date": ["open", "close", "2021-11-01T18:00:00Z", "2021-11-02T04:00:00Z"],
        "moneyline_home": [-150, -200, -170, -250],
        "moneyline_away": [130, 180, 150, 200],
    })
    expected = f.opening_features(source)
    original = source.copy(deep=True)
    for price in [-1000, -120, 150, 600]:
        changed = source.copy()
        changed.loc[changed.odds_date.ne("open"), "moneyline_home"] = price
        changed.loc[changed.odds_date.ne("open"), "moneyline_away"] = -price
        pd.testing.assert_frame_equal(expected, f.opening_features(changed))
    pd.testing.assert_frame_equal(source, original)
    row = expected.iloc[0]
    assert row.open_implied_home == pytest.approx(150 / 250)
    assert row.open_implied_away == pytest.approx(100 / 230)
    assert row.open_home == pytest.approx((150 / 250) / (150 / 250 + 100 / 230))
    assert not any("move" in col or "steam" in col or "updates" in col for col in expected)


def test_time_folds_never_split_a_date_or_cross_february():
    m = module("model_v3")
    dates = pd.Series(np.repeat(pd.date_range("2021-11-01", periods=70).strftime("%Y-%m-%d"), 3))
    folds = m.time_folds(dates)
    assert len(folds) == 5
    seen = set()
    for train, validation in folds:
        assert dates.iloc[train].max() < dates.iloc[validation].min() < "2022-02-01"
        assert not set(train) & set(validation)
        assert not set(validation) & seen
        seen.update(validation)
        for date in dates.iloc[validation].unique():
            assert set(np.flatnonzero(dates.eq(date))) <= set(validation)
    with pytest.raises(ValueError, match="February"):
        m.time_folds(pd.Series(["2022-02-01"] * 30))


def test_injury_snapshot_is_strictly_before_tip_including_one_pm_games():
    f = module("features_v3")
    source = pd.DataFrame({
        "game_date": ["2022-03-01"] * 3, "matchup": ["A@H"] * 3,
        "game_time": ["01:00 (ET)"] * 3,
        "snapshot_date": ["2022-02-28", "2022-03-01", "2022-03-01"],
        "snapshot_time": ["17:00", "13:00", "17:00"],
        "team": ["H"] * 3, "player": ["Player"] * 3, "status": ["Out"] * 3,
    })
    rows, snapshot, tip = f.injury_snapshot(source, "2022-03-01", "H", "A")
    assert len(rows) == 1
    assert snapshot == "2022-02-28 17:00"
    assert tip == "2022-03-01 13:00"
    assert snapshot < tip
    late = source.assign(game_time="07:30 (ET)")
    _, snapshot, tip = f.injury_snapshot(late, "2022-03-01", "H", "A")
    assert snapshot == "2022-03-01 17:00" < tip


def test_lean_and_winner_gates_are_distinct_and_zero_edge_does_not_bet():
    e = module("evaluation_v3")
    frame = pd.DataFrame({
        "date": ["2022-03-01"] * 3,
        "home_won": [True, False, True],
        "dec_home": [1.5, 2.2, 1.8], "dec_away": [2.8, 1.7, 2.1],
        "p_market_home": [0.7, 0.4, 0.6],
    })
    p = np.array([0.6, 0.6, 0.6])
    original = frame.copy(deep=True)
    lean = e.bet_frame(frame, p, "lean", 0)
    winner = e.bet_frame(frame, p, "winner", 0)
    assert lean.bet.tolist() == [True, True, False]
    assert lean.side.tolist()[:2] == ["away", "home"]
    assert winner.bet.tolist() == [False, True, False]
    assert e.bet_metrics(lean)["pnl"] == pytest.approx(-2)
    flat = e.bet_frame(frame, p, "flat", 0)
    assert e.bet_metrics(flat)["roi"] == pytest.approx((0.5 - 1 + 0.8) / 3)
    pd.testing.assert_frame_equal(frame, original)


def test_quarter_kelly_uses_same_day_starting_bankroll_and_zero_for_negative_ev():
    e = module("evaluation_v3")
    bets = pd.DataFrame({
        "date": ["2022-03-01", "2022-03-01", "2022-03-02"],
        "bet": [True, True, True], "p_side": [0.7, 0.7, 0.51],
        "decimal": [2.0, 2.0, 1.9], "won": [True, False, True],
        "pnl": [1.0, -1.0, 0.9],
    })
    original = bets.copy(deep=True)
    out = e.quarter_kelly(bets)
    assert out["bankroll_multiple"] == pytest.approx(1.0)
    assert out["total_staked"] == pytest.approx(0.1)
    assert out["n_staked"] == 2
    pd.testing.assert_frame_equal(bets, original)


def test_bootstrap_is_seeded_and_preserves_constant_returns():
    e = module("evaluation_v3")
    frame = pd.DataFrame({
        "date": np.repeat(["2022-03-01", "2022-03-02", "2022-03-03"], 2),
        "home_won": [True] * 6, "dec_home": [2.0] * 6, "dec_away": [2.0] * 6,
        "p_market_home": [0.5] * 6,
    })
    first = e.bootstrap_intervals(frame, np.full(6, 0.7), "winner", 0.01, n_boot=1000)
    second = e.bootstrap_intervals(frame, np.full(6, 0.7), "winner", 0.01, n_boot=1000)
    assert first == second
    assert first["flat_roi"] == [1.0, 1.0]
    assert first["gate_roi"] == [1.0, 1.0]
    assert first["accuracy"] == [1.0, 1.0]
    assert first["brier"][0] == pytest.approx(0.09)


def test_schema_discovery_covers_every_column_without_writing_inputs(tmp_path):
    s = module("schema_discovery")
    p = tmp_path / "sample.csv"
    p.write_text("game_id,game_date,value,label\n1,2022-03-01,2,open\n2,2022-03-02,4,close\n")
    before = p.read_bytes()
    text = s.describe_csv(p, chunksize=1)
    assert all(col in text for col in ["game_id", "game_date", "value", "label"])
    assert "2022-03-01" in text
    assert "2" in text and "4" in text
    assert p.read_bytes() == before


def test_prediction_is_outcome_blind_and_incomplete_rows_use_market():
    m = module("model_v3")
    fitted = m.estimator({"kind": "logreg", "parameters": {"C": 0.05}})
    fitted.fit(np.arange(10).reshape(-1, 1), np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1]))
    bundle = {
        "models": [fitted, fitted], "calibrator": {"kind": "none", "model": None},
        "selection": {"selected_features": ["signal"], "vote_weight": 0.75, "model_weight": 0.5},
    }
    frame = pd.DataFrame({"signal": [4.0, np.nan, 8.0], "p_market_home": [0.6, 0.7, 0.8],
                          "home_won": [False, True, False]})
    before = frame.copy(deep=True)
    p = m.predict(bundle, frame, [True, False, True])
    changed = frame.assign(home_won=~frame.home_won)
    np.testing.assert_array_equal(p, m.predict(bundle, changed, [True, False, True]))
    assert p[1] == 0.7
    assert np.isfinite(p).all() and (p > 0).all() and (p < 1).all()
    pd.testing.assert_frame_equal(frame, before)


def test_american_conversion_preserves_payout_and_removes_vig():
    f = module("features_v3")
    moneylines = np.array([-1500, -333, -100, 100, 137, 2500], dtype=float)
    expected = np.array([1 + 100 / 1500, 1 + 100 / 333, 2, 2, 2.37, 26])
    np.testing.assert_allclose(f.american_decimal(moneylines), expected)
    home = f.no_vig(moneylines, moneylines[::-1])
    away = f.no_vig(moneylines[::-1], moneylines)
    np.testing.assert_allclose(home + away, 1)
    with pytest.raises(ValueError, match="nonzero"):
        f.american_decimal([0])
    with pytest.raises(ValueError, match="finite"):
        f.american_decimal([np.nan])
