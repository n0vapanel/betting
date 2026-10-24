"""Main entry point: runs the two walk-forward phases, evaluation, and deliverables."""
import json

import numpy as np
import pandas as pd

from run_backtest import (
    f, VAL_LO, VAL_HI, TEST_LO, TEST_HI, run_phase, tune, evaluate, gate_table,
    choose_gate, bet_frame, quarter_kelly, bootstrap_intervals,
)


def dec_to_ml(d):
    if pd.isna(d):
        return np.nan
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def main():
    # Tune hyperparameters on the phase-1 training window only (through Apr 15).
    cfg = tune(f[f.date < VAL_LO].reset_index(drop=True))
    # Phase 1: train through Apr 15 -> validate Apr 16-30 (gate selection)
    _, val, _, cv_info = run_phase(
        (f.date < VAL_LO),
        (f.date >= VAL_LO) & (f.date < TEST_LO),
        (f.date >= TEST_LO) & (f.date <= TEST_HI),
        "phase1", cfg)
    # Phase 2: train through Apr 30 -> test May 1 - Jun 30
    _, _, test, cv2 = run_phase(
        (f.date < TEST_LO),
        (f.date < TEST_LO) & (f.date < VAL_LO),  # unused placeholder
        (f.date >= TEST_LO) & (f.date <= TEST_HI),
        "phase2", cfg)

    val_odds = val[val.has_odds == 1].reset_index(drop=True)
    test_odds = test[test.has_odds == 1].reset_index(drop=True)

    gate_choice, val_gates = choose_gate(val_odds, val_odds.p_model_home.to_numpy())
    print("chosen gate:", json.dumps(gate_choice))

    val_eval = evaluate(val_odds, val_odds.p_model_home.to_numpy(), "validation (odds subset)")
    test_eval = evaluate(test_odds, test_odds.p_model_home.to_numpy(), "test (odds subset)")
    test_eval_all = evaluate(test, test.p_model_home.to_numpy(), "test (all games)")

    market_p = test_odds.p_market_home.to_numpy()
    market_eval = evaluate(test_odds, market_p, "market (odds subset)")

    gates_test = gate_table(test_odds, test_odds.p_model_home.to_numpy())
    chosen_v, chosen_eps = gate_choice["variant"], gate_choice["eps"]
    chosen_bets = bet_frame(test_odds, test_odds.p_model_home.to_numpy(), chosen_v, chosen_eps)
    kelly = quarter_kelly(chosen_bets)
    boot = bootstrap_intervals(test_odds, test_odds.p_model_home.to_numpy(), chosen_v, chosen_eps)

    pred = test.copy()
    pred["predicted_side"] = np.where(pred.p_model_home > 0.5, "home", "away")
    pred["decimal_of_side"] = np.where(pred.predicted_side == "home", pred.dec_home, pred.dec_away)
    pred["correct"] = (pred.predicted_side == "home") == (pred.home_won == 1)
    pred["pnl_1usd"] = np.where(pred.has_odds == 1,
                                np.where(pred.correct, pred.decimal_of_side - 1, -1.0), np.nan)
    pred["ml_of_side"] = pred.decimal_of_side.apply(dec_to_ml)
    pred["prediction_source"] = "logreg_lgbm_ensemble_isotonic_cal_rolling_nolookahead"

    out_cols = ["date", "home_std", "away_std", "p_model_home", "p_market_home", "predicted_side",
                "ml_of_side", "correct", "pnl_1usd", "decimal_of_side", "prediction_source"]
    pred_out = pred[out_cols].rename(columns={"home_std": "home", "away_std": "away"})
    pred_out["date"] = pred_out.date.dt.strftime("%Y-%m-%d")
    pred_out.to_csv("mlb_predictions.csv", index=False)

    results = {
        "config": cfg,
        "cv_info": cv2,
        "val_eval": val_eval, "test_eval": test_eval, "test_eval_all": test_eval_all,
        "market_eval": market_eval,
        "gate_choice": gate_choice, "val_gates": val_gates, "test_gates": gates_test,
        "kelly": kelly, "bootstrap": boot,
    }
    with open("backtest_results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=float)

    print(json.dumps({
        "test_eval_all": test_eval_all, "test_eval": test_eval, "market_eval": market_eval,
        "gate_choice": gate_choice, "kelly": kelly,
        "bootstrap_flat_roi": boot["flat_roi"], "bootstrap_accuracy": boot["accuracy"],
        "bootstrap_gate_roi": boot["gate_roi"], "bootstrap_brier": boot["brier"],
        "bootstrap_market_brier": boot["market_brier"],
        "bootstrap_brier_minus_market": boot["brier_minus_market"],
    }, indent=1, default=float))
    print("n_test_all:", len(test), "n_test_odds:", len(test_odds))


if __name__ == "__main__":
    main()
