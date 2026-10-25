import unittest

import numpy as np
import pandas as pd

import nfl_backtest as nb


def fixture():
    schedules = pd.DataFrame(
        [
            dict(game_id=f"2021_0{w}_BAL_ATL", season=2021, week=w, game_type="REG",
                 gameday=f"2021-09-{5 + 7 * w:02d}", gametime="13:00",
                 home_team="ATL", away_team="BAL", home_score=20 + w,
                 away_score=17, location="Home", spread_line=3.0, total_line=45.0,
                 home_moneyline=-150.0, away_moneyline=130.0, result=3 + w)
            for w in range(1, 4)
        ]
    )
    plays = pd.DataFrame(
        [
            dict(game_id=g.game_id, season=2021, week=g.week, season_type="REG",
                 posteam=t, defteam="BAL" if t == "ATL" else "ATL",
                 epa=0.2 if t == "ATL" else -0.1, yards_gained=6.0,
                 qb_dropback=float(p == "pass"), play_type=p, play_id=j,
                 spread_line=3.0, vegas_home_wp=0.6,
                 game_seconds_remaining=3600.0, qtr=1)
            for g in schedules.itertuples()
            for t in ("ATL", "BAL")
            for j, p in enumerate(("pass", "run"), 1)
        ]
    )
    snaps = pd.DataFrame(
        [
            dict(game_id="2021_01_BAL_ATL", season=2021, week=1, game_type="REG",
                 team="ATL", player=name, pfr_player_id=pfr, position=position,
                 offense_pct=off, defense_pct=defense, st_pct=0.0)
            for name, pfr, position, off, defense in [
                ("Test Receiver", "ReceTe00", "WR", 0.8, 0.0),
                ("Matt Ryan", "RyanMa00", "QB", 0.95, 0.0),
                ("Test Defender", "DefeTe00", "LB", 0.0, 0.6),
            ]
        ]
    )
    injuries = pd.DataFrame(
        [
            dict(season=2021, week=2, game_type="REG", team="ATL", full_name=name,
                 gsis_id=gsis, position=position, report_status=status,
                 date_modified=stamp)
            for name, gsis, position, status, stamp in [
                ("Test Receiver", "WR1", "WR", "Out", "2021-09-17T12:00:00Z"),
                ("Matt Ryan", "QB1", "QB", "Questionable", "2021-09-17T12:00:00Z"),
                ("Test Defender", "LB1", "LB", "Doubtful", "2021-09-17T12:00:00Z"),
                ("Unknown Rookie", "NEW1", "TE", "Out", "2021-09-17T12:00:00Z"),
                ("Late Addition", "LATE1", "TE", "Out", "2021-09-20T12:00:00Z"),
            ]
        ]
    )
    depth = pd.DataFrame(
        [
            dict(season=2021, week=w, game_type="REG", club_code="ATL",
                 gsis_id="QB1" if w == 1 else "FAKE1",
                 full_name="Matt Ryan" if w == 1 else "Future Quarterback",
                 position="QB", depth_position="QB", depth_team="1", formation="Offense")
            for w in (1, 2, 3)
        ]
    )
    passing = pd.DataFrame(
        [
            dict(season=2021, week=w, season_type="REG", team_abbr="ATL",
                 player_gsis_id="QB1", player_display_name="Matt Ryan",
                 attempts=30, pass_yards=240, passer_rating=95.0,
                 completion_percentage_above_expectation=4.0 if w else 999.0)
            for w in (0, 1, 2, 3)
        ]
    )
    ids = pd.DataFrame([dict(pfr_id="RyanMa00", gsis_id="QB1"),
                        dict(pfr_id="ReceTe00", gsis_id="WR1")])
    return schedules, plays, injuries, snaps, depth, {"passing": passing}, ids


class NFLBacktestTests(unittest.TestCase):
    def test_american_odds_and_no_vig_probability_properties(self):
        np.testing.assert_allclose(nb.american_decimal([-200, 100, 150]), [1.5, 2, 2.5])
        rng = np.random.default_rng(42)
        for _ in range(30):
            odds = rng.choice([-400, -150, -110, 100, 125, 350], size=(10, 2))
            p = nb.no_vig_home(odds[:, 0], odds[:, 1])
            reverse = nb.no_vig_home(odds[:, 1], odds[:, 0])
            np.testing.assert_allclose(p + reverse, 1.0)
            self.assertTrue(np.all((p > 0) & (p < 1)))
        with self.assertRaises(ValueError):
            nb.american_decimal([0])

    def test_moneyline_push_and_ats_spread_sign(self):
        games = pd.DataFrame([
            dict(result=7.0, spread_line=3.0, home_moneyline=-200.0, away_moneyline=180.0),
            dict(result=0.0, spread_line=3.0, home_moneyline=-200.0, away_moneyline=180.0),
            dict(result=-1.0, spread_line=3.0, home_moneyline=-200.0, away_moneyline=180.0),
        ])
        np.testing.assert_allclose(nb.unit_profit(games, np.ones(3)), [0.5, 0.0, -1.0])
        np.testing.assert_allclose(nb.unit_profit(games, np.zeros(3)), [-1.0, 0.0, 1.8])
        games[["home_moneyline", "away_moneyline"]] = np.nan
        games["result"] = [7.0, 3.0, 0.0]
        np.testing.assert_allclose(nb.unit_profit(games, np.ones(3)),
                                   [100 / 110, 0.0, -1.0])

    def test_injury_formula_prior_snaps_and_lagged_starting_quarterback(self):
        data = fixture()
        frame, columns, audit = nb.build_features(*data)
        game = frame.loc[frame.week.eq(2)].iloc[0]
        self.assertAlmostEqual(game.home_injury_impact, 0.8 + 0.25 * 0.95 + 0.5 * 0.6)
        self.assertAlmostEqual(game.home_injury_matched_share, 0.75)
        self.assertEqual(game.home_qb_on_report, 1.0)
        self.assertEqual(game.home_qb_injury_weight, 0.25)
        self.assertEqual(game.home_expected_qb_id, "QB1")
        self.assertEqual(game.home_depth_week_key, 202101)
        self.assertAlmostEqual(game.home_qb_cpoe, 4.0)
        self.assertEqual(audit["injury_rows_after_kickoff_excluded"], 1)
        self.assertNotIn("spread_line", columns)
        self.assertNotIn("result", columns)
        self.assertTrue(all(frame.home_history_week_key < frame.week_key))

    def test_future_and_same_week_outcomes_cannot_change_current_features(self):
        data = fixture()
        before, columns, _ = nb.build_features(*data)
        changed = [x.copy(deep=True) if isinstance(x, pd.DataFrame) else
                   {k: v.copy(deep=True) for k, v in x.items()} for x in data]
        changed[0].loc[changed[0].week.ge(2), ["home_score", "away_score", "result"]] = 999
        changed[1].loc[changed[1].week.ge(2), ["epa", "yards_gained"]] = 999
        extra_snap = changed[3].copy()
        extra_snap["week"] = 2
        extra_snap["offense_pct"] = 0.01
        changed[3] = pd.concat([changed[3], extra_snap], ignore_index=True)
        changed[4].loc[changed[4].week.ge(2), "gsis_id"] = "DIFFERENT"
        changed[5]["passing"].loc[changed[5]["passing"].week.ne(1),
                                  "completion_percentage_above_expectation"] = -999
        after, after_columns, _ = nb.build_features(*changed)
        self.assertEqual(columns, after_columns)
        pd.testing.assert_frame_equal(
            before.loc[before.week.eq(2), columns].reset_index(drop=True),
            after.loc[after.week.eq(2), columns].reset_index(drop=True),
        )

    def test_only_this_weeks_pre_kickoff_injury_report_may_change_features(self):
        data = fixture()
        before, _, _ = nb.build_features(*data)
        data[2].loc[data[2].gsis_id.eq("WR1"), "report_status"] = "Questionable"
        after, _, _ = nb.build_features(*data)
        old = before.loc[before.week.eq(2), "home_injury_impact"].item()
        new = after.loc[after.week.eq(2), "home_injury_impact"].item()
        self.assertAlmostEqual(old - new, 0.75 * 0.8)
        self.assertEqual(after.loc[after.week.eq(1), "home_injury_impact"].item(), 0.0)

    def test_vendor_snap_rounding_does_not_exceed_full_participation(self):
        data = fixture()
        data[3].loc[data[3].pfr_player_id.eq("ReceTe00"), "st_pct"] = 1.01
        frame, _, audit = nb.build_features(*data)
        impact = frame.loc[frame.week.eq(2), "home_injury_impact"].item()
        self.assertAlmostEqual(impact, 1.0 + 0.25 * 0.95 + 0.5 * 0.6)
        self.assertEqual(audit["snap_fraction_rows_clipped"], 1)

    def test_grouped_cv_and_calibration_blocks_are_chronological_and_disjoint(self):
        frame = pd.DataFrame({"week_key": np.repeat(np.arange(202001, 202022), 8)})
        for train, validation in nb.week_splits(frame):
            self.assertLess(frame.iloc[train].week_key.max(),
                            frame.iloc[validation].week_key.min())
            self.assertFalse(set(frame.iloc[train].week_key) &
                             set(frame.iloc[validation].week_key))
        base, calibration = nb.training_blocks(frame, calibration_weeks=6)
        self.assertEqual(calibration.week_key.nunique(), 6)
        self.assertLess(base.week_key.max(), calibration.week_key.min())

    def test_lean_can_back_an_underdog_but_conviction_requires_model_winner(self):
        p = np.array([0.45, 0.60, 0.70, 0.50])
        market = np.array([0.40, 0.65, 0.60, 0.50])
        side, lean = nb.gate_bets(p, market, "lean", 0.02)
        _, conviction = nb.gate_bets(p, market, "winner_conviction", 0.02)
        np.testing.assert_array_equal(side, [1, 0, 1, 0])
        np.testing.assert_array_equal(lean, [True, True, True, False])
        np.testing.assert_array_equal(conviction, [False, False, True, False])
        _, threshold = nb.gate_bets(np.array([0.62]), np.array([0.60]), "lean", 0.02)
        self.assertFalse(threshold.item())

    def test_week_cluster_bootstrap_is_invariant_to_duplicate_rows(self):
        weeks = np.repeat(np.arange(20), 4)
        denominator = np.ones(80)
        numerator = np.repeat(np.linspace(-0.5, 0.5, 20), 4)
        ci = nb.cluster_ratio_ci(numerator, denominator, weeks, repetitions=500)
        doubled = nb.cluster_ratio_ci(np.repeat(numerator, 2),
                                      np.repeat(denominator, 2),
                                      np.repeat(weeks, 2), repetitions=500)
        np.testing.assert_allclose(ci, doubled)
        np.testing.assert_allclose(
            nb.cluster_ratio_ci(denominator, denominator, weeks, repetitions=500),
            [1, 1],
        )
        self.assertLess(ci[0], 0)
        self.assertGreater(ci[1], 0)

    def test_kelly_uses_start_of_week_bankroll_and_actual_price(self):
        games = pd.DataFrame([
            dict(week_key=202115, result=3.0, spread_line=0.0,
                 home_moneyline=-110.0, away_moneyline=-110.0),
            dict(week_key=202115, result=-3.0, spread_line=0.0,
                 home_moneyline=-110.0, away_moneyline=-110.0),
        ])
        path, summary = nb.kelly_path(games, np.array([0.6, 0.6]),
                                     np.ones(2), np.ones(2, dtype=bool))
        np.testing.assert_allclose(path["kelly_stake"], [20.0, 20.0])
        self.assertAlmostEqual(summary["final_bankroll"], 1000 + 20 * 100 / 110 - 20)
        reverse, reverse_summary = nb.kelly_path(
            games.iloc[::-1].reset_index(drop=True), np.array([0.6, 0.6]),
            np.ones(2), np.ones(2, dtype=bool))
        self.assertAlmostEqual(summary["final_bankroll"], reverse_summary["final_bankroll"])
        no_edge, _ = nb.kelly_path(games, np.array([0.51, 0.51]),
                                 np.ones(2), np.ones(2, dtype=bool))
        np.testing.assert_array_equal(no_edge["kelly_stake"], [0.0, 0.0])

    def test_tied_games_are_pushes_and_excluded_from_binary_scores(self):
        games = pd.DataFrame([
            dict(week_key=202115, result=3.0, home_moneyline=-110, away_moneyline=-110),
            dict(week_key=202116, result=0.0, home_moneyline=-110, away_moneyline=-110),
        ])
        metrics = nb.score_predictions(games, np.array([0.8, 0.99]))
        self.assertEqual(metrics["decisive_games"], 1)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["brier"], 0.04)
        self.assertAlmostEqual(metrics["flat_roi"], (100 / 110) / 2)


if __name__ == "__main__":
    unittest.main()
