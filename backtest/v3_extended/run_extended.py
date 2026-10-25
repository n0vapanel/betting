"""NBA 3-month extension runner: Jan 1 - Mar 31 2022 walk-forward backtest with
v3 features + official-filing injury impact + minutes-load features, four
registered candidates (v3_only / plus_injuries / plus_minutes / all), edge
gates frozen on December validation, and the full evaluation protocol.

Run:
    JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home \
    /Users/x/.hermes/cache/scratch/msports-venv/bin/python -B \
    backtest/v3_extended/run_extended.py            [full pipeline]
    ... run_extended.py --report-only               [render frozen capture]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def configure_runtime():
    sys.dont_write_bytecode = True
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[variable] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if sys.platform != "darwin" or os.environ.get("V3X_RUNTIME_READY") == "1":
        return
    spec = importlib.util.find_spec("sklearn")
    runtime = Path(spec.origin).parent / ".dylibs" if spec and spec.origin else None
    if runtime is not None and (runtime / "libomp.dylib").is_file():
        environment = os.environ.copy()
        previous = environment.get("DYLD_LIBRARY_PATH", "")
        environment["DYLD_LIBRARY_PATH"] = str(runtime) + (os.pathsep + previous if previous else "")
        environment["V3X_RUNTIME_READY"] = "1"
        os.execve(sys.executable, [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]], environment)


def percentage(value, signed=False):
    if value is None:
        return "n/a"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def number(value):
    return "n/a" if value is None else f"{value:.6f}"


def markdown_table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ])


def render_report(results):
    primary_name = results["primary_name"]
    primary = results["candidates"][primary_name]
    performance = primary["metrics"]
    selection = primary["selection"]
    gate = primary["frozen_gate"]
    test_gate = primary["frozen_gate_test"]
    best = primary["hindsight_best_gate"]
    features = results["features"]
    ci = primary["confidence_intervals"]
    calibrated = selection["calibration"]
    v3_reference = results["v3_reference"]
    lines = [
        "# NBA 3-month extended backtest (Jan 1 - Mar 31 2022)\n",
        "## Headline results\n",
        f"**Primary configuration: `{primary_name}`**, chosen by the lowest pre-December ensemble "
        "CV log loss among the four registered candidates, not by Jan-Mar results. "
        f"{len(selection['selected_features'])}/{len(selection['candidate_features'])} candidate "
        f"features survived backward selection; calibration: **{calibrated}**. "
        "Flat $1 at Kaggle closing moneylines.\n",
        markdown_table(
            ["configuration", "games", "accuracy", "flat $1 ROI", "P&L", "model Brier", "market Brier"],
            [
                ["v3 reference (March 2022, 1-month)", v3_reference.get("march_n", 229),
                 percentage(v3_reference.get("march_accuracy")),
                 percentage(v3_reference.get("march_flat_roi"), True),
                 f"{v3_reference['march_pnl']:+.4f}" if v3_reference.get("march_pnl") is not None else "n/a",
                 number(v3_reference.get("march_brier")), number(v3_reference.get("march_market_brier"))],
                [f"extended: {primary_name}", performance["n_games"], percentage(performance["accuracy"]),
                 percentage(performance["flat_roi"], True), f"{performance['flat_pnl']:+.4f}",
                 number(performance["brier"]), number(performance["market_brier"])],
            ],
        ),
        "\nThe extended benchmark contains the locked 623 Jan-Mar 2022 games with Kaggle closing "
        "moneylines (Jan 231, Feb 163, Mar 229 — the January/February schedule was reduced by COVID "
        "postponements). The v3 reference row is v3's published March-2022 result on the same "
        "settlement source; it is not recomputed here.\n",
        "## Protocol\n",
        "- **Strict walk-forward:** the model trains once on everything before 2022-01-01 "
        f"(complete-feature rows {features['train_date_min']}..{features['train_date_max']}); weights, "
        "feature selection, calibration, mixture and the deployable gate are frozen on 2021-12-31. "
        "Jan-Mar test rows only update rolling statistics from strictly earlier completed games; "
        "model parameters never update.",
        "- **Model:** v3 methodology — greedy backward selection (LogReg C=0.05, 5 date-grouped "
        "expanding folds over pre-Dec-16 development), LogReg/LightGBM/XGBoost grids, top-two-family "
        "soft vote, optional no-vig market shrinkage, chronological calibration (fit Dec 16-23, "
        "choose Dec 24-31), final refit through Dec 31.",
        "- **Edge gates:** frozen on the held-out Dec 24-31 predictions (highest ROI with ≥20 bets; "
        "fallback winner ε=0). Applied to Jan-Mar without seeing Jan-Mar.",
        "- **Eval:** flat $1 at closing ML; accuracy; ROI; Brier vs no-vig market; 1000-resample "
        "bootstrap CIs clustered by Eastern calendar date; quarter-Kelly diagnostics.",
        "- **Fallback policy:** any Jan-Mar test row missing a candidate's features gets that "
        "candidate's Kaggle no-vig market probability (disclosed below); the game stays in the 623.",
        "\n## Data\n",
        f"- Closing moneylines: Kaggle `cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024` "
        "(coverage 2007-10-30..2023-01-16), team codes normalized (`gs/ny/no/sa/utah/wsh` handled). "
        "All 623 test games have closing moneylines; NBA and Kaggle scores cross-asserted equal.",
        f"- Official injury reports: `nbainjuries` 5PM ET filings (verified against a live fetch "
        "of 2022-01-05 17:00 — 150/156 exact row matches). "
        + f"Filing usage in this window: {json.dumps(features['official_filing_counts'])}.",
        f"- Minutes load: cached player boxscores, {features['player_read']['rows_scanned']:,} rows "
        f"scanned in {features['player_read']['chunksize']:,}-row chunks "
        f"({features['player_read']['rows_retained']:,} retained); the 766 MB file is never loaded whole.",
        f"- Engineered predictors: **{features['n_features']}** = v3's 56 + 5 minutes-load + 6 "
        "official-filing injury features. Full formulas in `features_extended.py` docstring and the manifest.",
        f"- Complete training games: **{features['n_train']}** "
        f"({features['training_dropped_incomplete']} of {features['training_candidates_with_closing_lines']} "
        "closing-priced candidates excluded for warm-up or unavailable precomputed player features — "
        "the supplied player feature file starts Nov 1; earlier values are not invented).",
        f"- No cached input is written; all sources are SHA-256 fingerprinted in the manifests.\n",
    ]
    for missing in features["test_incomplete_games"]:
        lines.append(
            f"- **Disclosed fallback:** {missing['date']} {missing['away']} at {missing['home']} "
            f"(`{missing['game_id']}`) lacks `{missing['missing_features']}` for the all-features "
            "candidate; those candidates fall back to market probability."
        )
    if features.get("test_complete_per_candidate"):
        lines.append("- Complete test rows per candidate: "
                     + ", ".join(f"{k}={v}" for k, v in features["test_complete_per_candidate"].items()) + ".\n")
    lines += [
        "## Ablation — same locked 623-game Jan-Mar window\n",
        markdown_table(
            ["config", "features kept / candidates", "pre-Dec CV LL", "calibration", "accuracy",
             "flat ROI", "P&L", "Brier", "market Brier", "frozen gate", "gate bets", "gate ROI"],
            [[name, f"{len(row['selection']['selected_features'])}/{len(row['selection']['candidate_features'])}",
              number(row["selection"]["cv_log_loss"]), row["selection"]["calibration"],
              percentage(row["metrics"]["accuracy"]), percentage(row["metrics"]["flat_roi"], True),
              f"{row['metrics']['flat_pnl']:+.4f}", number(row["metrics"]["brier"]),
              number(row["metrics"]["market_brier"]),
              f"{row['frozen_gate']['variant']} eps={row['frozen_gate']['eps']}",
              row["frozen_gate_test"]["n_bets"], percentage(row["frozen_gate_test"]["roi"], True)]
             for name, row in results["candidates"].items()],
        ),
        "\nEach candidate reruns backward selection, all three family grids, top-two voting, market "
        "shrinkage and chronological December calibration using only its permitted features. Training "
        "and test rows are held fixed across candidates. The primary configuration was selected by "
        "pre-December CV, not by this Jan-Mar table. `v3_only` keeps exactly v3's 56 features "
        "(which already include v3's own injury- and boxscore-derived columns); `plus_injuries` adds "
        "the new official-5PM-filing injury-impact features; `plus_minutes` adds the minutes-load "
        "features; `all` uses every feature.\n",
        "## Monthly detail (primary)\n",
    ]
    monthly = primary["monthly"]
    lines.append(markdown_table(
        ["month", "games", "accuracy", "flat ROI", "P&L", "Brier", "market Brier", "gate bets", "gate ROI"],
        [[row["month"], row["n_games"], percentage(row["accuracy"]), percentage(row["flat_roi"], True),
          f"{row['flat_pnl']:+.4f}", number(row["brier"]), number(row["market_brier"]),
          row["gate_bets"], percentage(row["gate_roi"], True)]
         for row in monthly],
    ))
    lines += [
        "\n## Edge gates (primary, all registered thresholds)\n",
        "Let d = p_model_home − p_market_home. **Lean-side** bets home if d>0, away if d<0, only when "
        "|d|≥ε. **Winner-side conviction** bets the model's >0.5 winner only when the model exceeds "
        "the no-vig market probability on that same side by ≥ε and strictly >0. Exact zero edge never bets.\n",
        markdown_table(
            ["variant", "eps", "bets", "win rate", "ROI", "P&L"],
            [[row["variant"], row["eps"], row["n_bets"], percentage(row["win_rate"]),
              percentage(row["roi"], True), f"{row['pnl']:+.4f}"] for row in primary["gates"]],
        ),
        f"\n**Deployable gate, frozen from December:** `{gate['variant']}`, eps={gate['eps']:.2f}; "
        f"Dec 24-31 validation {gate['n_bets']} bets, ROI {percentage(gate['roi'], True)}; "
        f"Jan-Mar **{test_gate['n_bets']} bets, ROI {percentage(test_gate['roi'], True)}**. "
        f"Minimum validation bets: {gate['minimum_bets']}; insufficient-sample fallback used: {gate['fallback']}.\n",
        f"**Descriptive Jan-Mar-best gate (hindsight, not selected for deployment):** `{best['variant']}`, "
        f"eps={best['eps']:.2f}, {best['n_bets']} bets, ROI **{percentage(best['roi'], True)}**. "
        "This is the best of ten reported Jan-Mar cells and is explicitly selection-biased.\n",
        "## Quarter-Kelly (primary, frozen gate)\n",
        f"Stake fraction = clip[0.25·(p·decimal−1)/(decimal−1), 0, 0.05]; simultaneous games use "
        "start-of-Eastern-date bankroll. Bankroll multiple "
        f"**{primary['quarter_kelly']['bankroll_multiple']:.6f}x** "
        f"({percentage(primary['quarter_kelly']['net_return'], True)}), staked "
        f"{primary['quarter_kelly']['total_staked']:.6f}, ROI on staked "
        f"{percentage(primary['quarter_kelly']['roi_on_staked'], True)}, max daily drawdown "
        f"{percentage(primary['quarter_kelly']['max_daily_drawdown'])}. Frictionless closing-price "
        "simulation, not executable intraday replay.\n",
        "## Bootstrap 95% confidence intervals (primary, date-clustered)\n",
        f"{ci['n_resamples']} resamples, seeded at {results['bootstrap_seed']}; {ci['n_dates']} whole "
        "Eastern calendar dates sampled with replacement. Games from the same day stay together. "
        "Percentile intervals condition on the fitted model and fixed December gate; they exclude "
        "model-selection and future-season uncertainty.\n",
        markdown_table(
            ["metric", "point estimate", "95% CI"],
            [
                ["accuracy", percentage(performance["accuracy"]), f"[{percentage(ci['accuracy'][0])}, {percentage(ci['accuracy'][1])}]"],
                ["flat ROI", percentage(performance["flat_roi"], True), f"[{percentage(ci['flat_roi'][0], True)}, {percentage(ci['flat_roi'][1], True)}]"],
                ["model Brier", number(performance["brier"]), f"[{number(ci['brier'][0])}, {number(ci['brier'][1])}]"],
                ["market Brier", number(performance["market_brier"]), f"[{number(ci['market_brier'][0])}, {number(ci['market_brier'][1])}]"],
                ["paired Brier − market", number(performance["brier"] - performance["market_brier"]),
                 f"[{number(ci['brier_minus_market'][0])}, {number(ci['brier_minus_market'][1])}]"],
                ["December-frozen gate ROI", percentage(test_gate["roi"], True),
                 f"[{percentage(ci['gate_roi'][0], True)}, {percentage(ci['gate_roi'][1], True)}]"],
            ],
        ),
        "\n## Verdict\n",
    ]
    roi_ci = ci["flat_roi"]
    roi_positive = roi_ci[0] is not None and roi_ci[0] > 0
    brier_better = performance["brier"] < performance["market_brier"]
    verdict = (
        f"The December-selected `{primary_name}` configuration posts **{percentage(performance['accuracy'])}** "
        f"accuracy and **{percentage(performance['flat_roi'], True)}** flat ROI over the locked 623-game "
        f"Jan-Mar 2022 window ({performance['n_games']} games), versus v3's March-2022 reference of "
        "65.5% accuracy / -4.52% flat ROI on its own 229-game window. "
        f"The flat-ROI bootstrap {'excludes zero on the positive side' if roi_positive else 'does not establish positive profitability'} "
        f"(95% CI [{percentage(ci['flat_roi'][0], True)}, {percentage(ci['flat_roi'][1], True)}]). "
        f"Model Brier is {'better' if brier_better else 'worse'} than the closing market "
        f"({number(performance['brier'])} vs {number(performance['market_brier'])}). "
        f"The December-frozen gate returns {percentage(test_gate['roi'], True)} on {test_gate['n_bets']} Jan-Mar bets; "
        "the hindsight best gate is disclosed but must not be treated as an out-of-sample policy. "
        "One locked historical benchmark; no live betting recommendation."
    )
    lines.append(verdict + "\n")
    lines += [
        "## Reproduction and artifacts\n",
        "```sh\nJAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home \\\n"
        "  /Users/x/.hermes/cache/scratch/msports-venv/bin/python -B backtest/v3_extended/run_extended.py\n"
        "/Users/x/.hermes/cache/scratch/msports-venv/bin/python -B backtest/v3_extended/run_extended.py --report-only\n"
        "```\n",
        f"Package versions: `{json.dumps(results['versions'], sort_keys=True)}`.\n",
        "- `backtest/v3_extended/nba_3month_predictions.csv`: primary per-game Jan-Mar predictions plus "
        "all four candidates' probabilities, settlement P&L and gates.",
        "- `backtest/v3_extended/features_train.csv`, `features_test.csv`, `feature_audit.csv`, "
        "`feature_manifest.json`: captured features and temporal provenance.",
        "- `backtest/v3_extended/metrics.json`: captured numbers — all candidates, gates, bootstrap "
        "intervals, source manifests.",
        "- `backtest/v3_extended/experiment_plan.json`, `frozen_protocol.json`: registered grids, "
        "thresholds, code fingerprints and the pre-test frozen selection.",
        "- `backtest/v3_extended/selection_*.json`, `cv_results_*.csv`, `validation_*.csv`, "
        "`model_*.joblib`, `predictions_*.csv`: per-candidate artifacts.",
    ]
    (HERE / "NBA_3MONTH_REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    configure_runtime()
    import warnings

    import numpy as np
    import pandas as pd
    from backtest.v3_extended import evaluation_extended as E
    from backtest.v3_extended import features_extended as F
    from backtest.v3_extended import model_extended as M
    from backtest.v3.baseline_v2 import fingerprints

    parser = argparse.ArgumentParser(description="Locked Jan-Mar 2022 NBA 3-month extension")
    parser.add_argument("--report-only", action="store_true",
                        help="Render the frozen metrics capture without recomputation")
    args = parser.parse_args()
    if args.report_only:
        results = json.loads((HERE / "metrics.json").read_text())
        render_report(results)
    else:
        warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")
        built = F.build_features()
        train, test = built["train"], built["test"]
        plan = {
            "version": 1,
            "training_window": [F.TRAIN_START, F.TRAIN_END],
            "test_window": [F.TEST_START, F.TEST_END],
            "expected_test_games": F.EXPECTED_TEST_GAMES,
            "primary_selection": "Lowest pre-December ensemble CV log loss across the four registered candidates; ties prefer registration order.",
            "candidate_features": F.CANDIDATE_SPECS,
            "n_date_grouped_time_folds": 5,
            "greedy_logreg_C": 0.05, "minimum_backward_improvement": 1e-5, "minimum_features": 3,
            "logreg_C_grid": M.LR_C, "lightgbm_grid": M.LGB_GRID, "xgboost_grid": M.XGB_GRID,
            "soft_vote_weights_first_of_top_two": M.VOTE_WEIGHTS,
            "model_vs_market_weights": M.MODEL_WEIGHTS,
            "calibration_fit": ["2021-12-16", "2021-12-23"],
            "calibration_selection_and_gate_validation": ["2021-12-24", "2021-12-31"],
            "calibration_methods": ["none", "platt", "isotonic"],
            "gate_selection": "Highest Dec 24-31 validation ROI with >=20 bets; tie more bets, lower epsilon, variant name; fallback winner eps=0.",
            "gate_variants": ["lean", "winner"], "eps_grid": E.EPS_GRID,
            "quarter_kelly": {"fraction": 0.25, "per_bet_cap": 0.05, "simultaneous_date_bankroll": True},
            "bootstrap": {"n_resamples": E.N_BOOT, "seed": E.SEED, "unit": "Eastern calendar date"},
            "injury_policy": built["manifest"]["official_filing_policy"],
            "minutes_policy": built["manifest"]["minutes_load_policy"],
            "missing_feature_policy": built["manifest"]["test_fallback_policy"],
            "code_fingerprints": fingerprints([Path(__file__), Path(F.__file__), Path(M.__file__), Path(E.__file__)]),
        }
        (HERE / "experiment_plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
        trained, selections, validations, frozen_gates = {}, {}, {}, {}
        for name, columns in F.CANDIDATE_SPECS.items():
            print(f"\n== Train-only candidate: {name} ==", flush=True)
            trained[name], selections[name], validations[name] = M.select_and_fit(train, columns, name)
            frozen_gates[name], table = E.choose_gate(validations[name])
            pd.DataFrame(table).to_csv(HERE / f"validation_gates_{name}.csv", index=False)
        primary_name = min(F.CANDIDATE_SPECS, key=lambda name: selections[name]["cv_log_loss"])
        frozen = {
            "primary_name": primary_name, "primary_selection": plan["primary_selection"],
            "selected_gates": frozen_gates, "candidate_cv_log_losses": {
                name: row["cv_log_loss"] for name, row in selections.items()
            }, "training_input_fingerprints": fingerprints([HERE / "features_train.csv"]),
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (HERE / "frozen_protocol.json").write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
        print(f"\n== SELECTION FROZEN; first Jan-Mar predictions now permitted. Primary: {primary_name} ==", flush=True)
        candidates, probabilities = {}, {}
        for name, columns in F.CANDIDATE_SPECS.items():
            complete = np.isfinite(test[columns].to_numpy(dtype=float)).all(axis=1)
            p = M.predict(trained[name], test, complete)
            probabilities[name] = p
            gate = frozen_gates[name]
            gates = E.gate_table(test, p)
            frozen_metrics = E.bet_metrics(E.bet_frame(test, p, gate["variant"], gate["eps"]))
            flat = E.bet_frame(test, p, "flat")
            prediction = test[["date", "home", "away", "p_market_home"]].copy()
            prediction["p_model_home"] = p
            prediction["predicted_side"] = flat.side.to_numpy()
            decimal = flat.decimal.to_numpy()
            prediction["ml_of_side"] = np.rint(np.where(decimal >= 2, (decimal - 1) * 100, -100 / (decimal - 1))).astype(int)
            prediction["decimal_of_side"] = decimal
            prediction["correct"] = flat.won.to_numpy()
            prediction["pnl_1usd"] = flat.pnl.to_numpy()
            prediction["edge"] = flat.edge.to_numpy()
            prediction["prediction_source"] = np.where(complete, name, "market_fallback_missing_feature")
            prediction = prediction[["date", "home", "away", "p_model_home", "p_market_home",
                                     "predicted_side", "ml_of_side", "decimal_of_side",
                                     "correct", "pnl_1usd", "edge", "prediction_source"]]
            prediction.to_csv(HERE / f"predictions_{name}.csv", index=False)
            candidates[name] = {
                "selection": selections[name], "metrics": E.evaluate(test, p), "gates": gates,
                "frozen_gate": gate, "frozen_gate_test": frozen_metrics,
                "hindsight_best_gate": E.hindsight_best(gates),
            }
        primary = candidates[primary_name]
        p = probabilities[primary_name]
        gate, best = primary["frozen_gate"], primary["hindsight_best_gate"]
        primary["quarter_kelly"] = E.quarter_kelly(E.bet_frame(test, p, gate["variant"], gate["eps"]))
        primary["confidence_intervals"] = E.bootstrap_intervals(test, p, gate["variant"], gate["eps"])
        monthly = []
        for month in ["2022-01", "2022-02", "2022-03"]:
            mask = test.date.str.startswith(month).to_numpy()
            subset = test.loc[mask]
            metrics = E.evaluate(subset, p[mask])
            gated = E.bet_metrics(E.bet_frame(subset, p[mask], gate["variant"], gate["eps"]))
            monthly.append({"month": month, "n_games": len(subset), "accuracy": metrics["accuracy"],
                            "flat_roi": metrics["flat_roi"], "flat_pnl": metrics["flat_pnl"],
                            "brier": metrics["brier"], "market_brier": metrics["market_brier"],
                            "gate_bets": gated["n_bets"], "gate_roi": gated["roi"]})
        primary["monthly"] = monthly
        combined = test[["game_id", "date", "home", "away", "home_won", "dec_home", "dec_away",
                         "p_market_home"]].copy()
        combined["p_model_home"] = p
        for name in F.CANDIDATE_SPECS:
            combined[f"p_{name}"] = probabilities[name]
        flat_primary = E.bet_frame(test, p, "flat")
        decimal = flat_primary.decimal.to_numpy()
        combined["predicted_side"] = flat_primary.side.to_numpy()
        combined["ml_of_side"] = np.rint(np.where(decimal >= 2, (decimal - 1) * 100, -100 / (decimal - 1))).astype(int)
        combined["decimal_of_side"] = decimal
        combined["correct"] = flat_primary.won.to_numpy()
        combined["pnl_1usd"] = flat_primary.pnl.to_numpy()
        combined["edge"] = flat_primary.edge.to_numpy()
        gated_primary = E.bet_frame(test, p, gate["variant"], gate["eps"])
        combined["gate_bet"] = gated_primary.bet.to_numpy()
        combined["gate_pnl_1usd"] = gated_primary.pnl.to_numpy()
        combined.to_csv(HERE / "nba_3month_predictions.csv", index=False)
        v3_metrics_path = ROOT / "backtest/v3/metrics.json"
        if v3_metrics_path.exists():
            v3_saved = json.loads(v3_metrics_path.read_text())
            v3_primary = v3_saved["candidates"][v3_saved["primary_name"]]["metrics"]
            v3_reference = {
                "march_n": v3_primary["n_games"], "march_accuracy": v3_primary["accuracy"],
                "march_flat_roi": v3_primary["flat_roi"], "march_pnl": v3_primary["flat_pnl"],
                "march_brier": v3_primary["brier"], "march_market_brier": v3_primary["market_brier"],
                "note": "v3 published March-2022 result (read-only reference from backtest/v3/metrics.json).",
            }
        else:
            v3_reference = {"march_n": 229, "march_accuracy": 0.655, "march_flat_roi": -0.0452,
                            "march_pnl": -10.35, "march_brier": None, "march_market_brier": None,
                            "note": "Fallback: v3 published summary 65.5% / -4.52%."}
        results = {
            "primary_name": primary_name, "candidates": candidates, "features": built["manifest"],
            "protocol": frozen, "bootstrap_seed": E.SEED, "v3_reference": v3_reference,
            "versions": {
                package: importlib.metadata.version(package)
                for package in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "nbainjuries"]
            },
        }
        (HERE / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        render_report(results)
    results = json.loads((HERE / "metrics.json").read_text())
    primary = results["candidates"][results["primary_name"]]
    metric = primary["metrics"]
    chosen, chosen_result = primary["frozen_gate"], primary["frozen_gate_test"]
    best = primary["hindsight_best_gate"]
    print(f"\nEXTENDED n={metric['n_games']}  accuracy={metric['accuracy']:.6%}  flat_roi={metric['flat_roi']:+.6%}")
    for name, row in results["candidates"].items():
        m = row["metrics"]
        print(f"  {name:14s} acc={m['accuracy']:.4%}  flat_roi={m['flat_roi']:+.4%}  "
              f"brier={m['brier']:.5f} (market {m['market_brier']:.5f})  "
              f"gate={row['frozen_gate']['variant']}/eps={row['frozen_gate']['eps']:.2f} "
              f"gate_roi={percentage(row['frozen_gate_test']['roi'], True)} "
              f"({row['frozen_gate_test']['n_bets']} bets)")
    print(f"Frozen gate on Jan-Mar: {chosen['variant']}, eps={chosen['eps']:.2f}; "
          f"ROI={percentage(chosen_result['roi'], True)}, {chosen_result['n_bets']} bets")
    print(f"Hindsight best gate (selection-biased): {best['variant']}, eps={best['eps']:.2f}; "
          f"ROI={percentage(best['roi'], True)}, {best['n_bets']} bets")
    print("Exact output paths:")
    for path in sorted(HERE.glob("*")):
        if path.is_file():
            print(path)
    print("END_TO_END_EXIT: 0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
