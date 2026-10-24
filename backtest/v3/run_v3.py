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
    if sys.platform != "darwin" or os.environ.get("V3_RUNTIME_READY") == "1":
        return
    spec = importlib.util.find_spec("sklearn")
    runtime = Path(spec.origin).parent / ".dylibs" if spec and spec.origin else None
    if runtime is not None and (runtime / "libomp.dylib").is_file():
        environment = os.environ.copy()
        previous = environment.get("DYLD_LIBRARY_PATH", "")
        environment["DYLD_LIBRARY_PATH"] = str(runtime) + (os.pathsep + previous if previous else "")
        environment["V3_RUNTIME_READY"] = "1"
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


def output_inventory():
    existing_inputs = {"README.md", "player_features.py", "player_features_2021_22.csv",
                       "test_player_features.py", "verification.log"}
    paths = [
        path for path in HERE.iterdir()
        if path.is_file() and path.name not in existing_inputs and path.name != "output_manifest.json"
    ]
    paths += [ROOT / name for name in ["README_v3.md", "v3_SCHEMA.md", "V3_BACKTEST_REPORT.md"]]
    output = HERE / "output_manifest.json"
    inventory = {"outputs": sorted(str(path) for path in paths) + [str(output)]}
    output.write_text(json.dumps(inventory, indent=2) + "\n")
    return inventory


def render_report(results):
    primary_name = results["primary_name"]
    primary = results["candidates"][primary_name]
    performance = primary["metrics"]
    selection = primary["selection"]
    gate = primary["frozen_gate"]
    march_gate = primary["frozen_gate_march"]
    best = primary["hindsight_best_gate"]
    baseline = results["v2"]
    features = results["features"]
    ci = primary["confidence_intervals"]
    calibrated = selection["calibration"]
    lines = [
        "# V3 NBA walk-forward backtest report\n",
        "## Headline results\n",
        f"**Primary configuration: `{primary_name}`**, chosen by the lowest pre-February "
        "five-fold out-of-fold log loss among the five registered feature-group configurations, not by March results. "
        f"{len(selection['selected_features'])}/{len(selection['candidate_features'])} candidate features survived backward selection; "
        f"calibration: **{calibrated}**.\n",
        markdown_table(
            ["configuration", "games", "accuracy", "flat $1 ROI", "P&L", "model Brier", "market Brier"],
            [
                ["v2 reproduced", baseline["n_test"], percentage(baseline["accuracy"]),
                 percentage(baseline["flat_roi"], True), f"{baseline['pnl']:+.4f}", number(baseline["model_brier"]), number(baseline["market_brier"])],
                [f"v3: {primary_name}", performance["n_games"], percentage(performance["accuracy"]),
                 percentage(performance["flat_roi"], True), f"{performance['flat_pnl']:+.4f}",
                 number(performance["brier"]), number(performance["market_brier"])],
            ],
        ),
        "\nThe primary result includes the same 229 March games and the same Kaggle closing-moneyline "
        "settlement prices as v2. Predictions are stored at full precision. The all-features candidate and every ablation "
        "are reported below regardless of whether they perform well.\n",
        "## V2 sanity gate\n",
        f"The frozen published specification was actually refit: LogReg C=0.05, its nine published features, "
        f"and the original February isotonic procedure. Training rows: {baseline['n_train']}; test rows: {baseline['n_test']}. "
        f"Gate passed: **{baseline['sanity_gate_passed']}**. Rounded results reproduce 68.1%, +0.10%, 0.2131 and 0.2010. "
        "The original v2 feature-selection search was not rerun; this gate reproduces the published chosen model. "
        "Baseline probabilities, exact metrics and source fingerprints are preserved under `backtest/v3/v2_*`.\n",
        "v2 limitations preserved for reproduction, but corrected in v3:\n",
        *[f"- {item}" for item in baseline["known_limitations"]],
        "\nAdditional timing audit: v2's hour<6 routing also selects the game-day 17:00 report "
        "for six 12:00/12:30 ET training games (2021-12-12, 2021-12-18, 2021-12-25, "
        "2022-01-09, 2022-01-17, 2022-02-26), after their scheduled tips. "
        "This is confirmed in `backtest/v3/baseline_temporal_audit.log`. "
        "The published baseline is reproduced, not retroactively repaired; its training-feature timing "
        "is therefore not a fully leakage-free control. V3 uses strictly previous-day snapshots for all six.\n",
        "\n## Data coverage and source usability\n",
        f"- Complete training games: **{features['n_train']}**, {features['train_date_min']} through {features['train_date_max']}; "
        f"authorized training window is 2021-10-20 through 2022-02-28. "
        f"{features['training_dropped_incomplete']} of {features['training_candidates_with_closing_lines']} closing-priced training candidates "
        "are excluded for warm-up, unavailable player features, missing openings or incomplete pre-tip injuries. No future backfill is used.",
        f"- Engineered predictors: **{features['n_features']}**. Full formulas and group assignments are in `README_v3.md`.",
        f"- Advanced player CSV: **{features['player_read']['rows_scanned']:,} rows scanned in "
        f"{features['player_read']['chunksize']:,}-row chunks**, {features['player_read']['rows_retained']:,} retained. "
        "The 766 MB file is never loaded whole. Only selected columns and the current regular season are retained.",
        "- No cached input is written. Every source and relevant implementation has a SHA-256 fingerprint in the manifests.",
        "- The supplied precomputed player file starts on November 1. Its source implementation and 23 existing temporal tests "
        "are retained unchanged; earlier feature values are not invented.",
    ]
    for missing in features["test_incomplete_games"]:
        lines.append(
            f"- **Disclosed benchmark fallback:** {missing['date']} {missing['away']} at {missing['home']} "
            f"(`{missing['game_id']}`) lacks `{missing['missing_features']}`. All five configurations use the "
            "Kaggle no-vig market probability for this row. The fallback was specified before March predictions; "
            "the game stays in the 229-game denominator and is not presented as a learned-model prediction."
        )
    for missing in features.get("unusable_injury_snapshots", []):
        lines.append(f"- Unusable injury source: {missing['date']} {missing['away']} at {missing['home']}: "
                     f"{missing['injury_source_issue']}. Excluded from complete-case training, not treated as healthy.")
    lines += [
        "\n## Temporal discipline and exact availability rules\n",
        *[f"- **{key}:** {value}" for key, value in features["timestamp_policy"].items()],
        "- `feature_audit.csv` records, for each train/test game, all rolling-history maximum source dates, "
        "the actual selected injury snapshot and tip in Eastern time, completeness and missing-source reasons. "
        "Every dated historical source is asserted strictly earlier than its target; injury timestamps must be strictly less than tip.",
        "- Schedule-clock provenance: injury routing uses the first cached clock label for each game, as recorded in the audit. "
        "The sole changed label is 2022-02-03 MIA@SAS (20:30 versus 19:00 ET); the selected 17:00 report is strictly pre-tip under both. "
        "This ambiguity changes no predictor or selected snapshot in this benchmark.",
        "- **Important limitation:** neither semantic opening/closing markers nor the supplied Kaggle close have original publication timestamps. "
        "This is an opening/closing-convention historical backtest, not a point-in-time feed replay. The semantic open is presumed pregame; "
        "that assumption cannot be proven to the second from these caches. `last_fetched` is never used as a historical availability timestamp.",
        "- Current-game close, movement, steam, update counts and quarter results do not enter the predictor matrix. "
        "The sole current closing signal is Kaggle no-vig market probability (and its logit). Prior-game CLV/steam/update features are lagged.",
        "- March model weights do not update. Statistics can incorporate completed earlier March games; model parameters, features, "
        "calibration choice, mixture weights, primary configuration and deployable gate are frozen before evaluating March.",
        "\n## Cross-check: TeamRankings close versus fixed Kaggle close\n",
        markdown_table(
            ["period", "matched / games", "exact both ML", "both within 10 ML", "mean abs Δp", "median abs Δp", "max abs Δp", "same favorite"],
            [
                [name, f"{row['n_matched']} / {row['n_games']}", percentage(row["fraction_exact_both"]),
                 percentage(row["fraction_within_10_both"]), number(row["mean_abs_home_probability_difference"]),
                 number(row["median_abs_home_probability_difference"]), number(row["max_abs_home_probability_difference"]),
                 percentage(row["favorite_side_agreement"])]
                for name, row in features["closing_line_agreement"].items()
            ],
        ),
        "\nThese are different source/book snapshots, not interchangeable settlement prices. No best-line shopping or "
        "replacement of Kaggle closes occurs. `odds_diagnostics.csv` includes opening probability, both-side movement, "
        "largest adjacent intraday probability move, update count, source close and timed-update bounds. "
        "Current-game diagnostic fields are excluded from every candidate's feature list.\n",
        "## Registered model selection and calibration\n",
        f"Development games (before Feb 1): {selection['n_development']}; February holdout: {selection['n_february']}. "
        "Five expanding folds are grouped by calendar date, so simultaneous games never cross a fold boundary. "
        "Fold standardization is fit only on that fold's training rows. Greedy backward selection uses LogReg C=0.05 "
        "and removes the single feature giving the best CV log-loss improvement above 0.00001 until no improvement remains "
        "or three features remain. Family grids are then retuned on that selected subset.\n",
        markdown_table(
            ["fold", "train dates", "train n", "validation dates", "validation n"],
            [[i + 1, f"{r['train_start']}..{r['train_end']}", r["n_train"],
              f"{r['validation_start']}..{r['validation_end']}", r["n_validation"]]
             for i, r in enumerate(selection["folds"])],
        ),
        "\n" + markdown_table(
            ["family", "parameters", "CV log loss"],
            [[row["config"]["kind"], json.dumps(row["config"]["parameters"]), number(row["cv_log_loss"])]
             for row in sorted(selection["cv_results"], key=lambda row: row["cv_log_loss"])],
        ),
        f"\nTop two distinct families: `{json.dumps(selection['top_two'])}`. Soft vote puts "
        f"{selection['vote_weight']:.0%} on the first and {1 - selection['vote_weight']:.0%} on the second. "
        f"The pre-registered CV-selected final probability mixes {selection['model_weight']:.0%} soft vote "
        f"with {1 - selection['model_weight']:.0%} no-vig market probability. Ensemble CV LL "
        f"**{selection['cv_log_loss']:.6f}** versus market **{selection['cv_market_log_loss']:.6f}**. "
        "This transparent market shrinkage is part of the model, not hidden postprocessing.\n",
        selection["calibration_protocol"] + "\n",
        markdown_table(
            ["calibration", "Feb 15–28 held-out log loss"],
            [[row["kind"], number(row["validation_log_loss"])] for row in selection["calibration_validation"]],
        ),
        f"\nCalibration-fit rows (Feb 1–14): {selection['calibration_fit_rows']}; "
        f"calibration-selection/gate-validation rows (Feb 15–28): {selection['calibration_validation_rows']}. "
        "Using the second block for both calibration choice and gate selection is disclosed multiple use of validation data; "
        "it does not make February an unbiased performance estimate. March remains untouched by selection. "
        "CV scores are selection estimates, not nested-CV generalization guarantees.\n",
        "Selected predictors: `" + ", ".join(selection["selected_features"]) + "`.\n",
        "## Edge gates: both interpretations, all registered thresholds\n",
        "Let d = p_model_home − p_market_home. **Lean-side** bets home if d>0, away if d<0, "
        "only when |d|≥ε. **Winner-side conviction** bets the model's >0.5 winner only when the model exceeds "
        "the no-vig market probability on that same side by ≥ε and strictly >0. Exact zero edge never bets. "
        "No-vig edge is not the same as positive EV at the vig-inclusive payout; gates deliberately match the requested definitions.\n",
        markdown_table(
            ["variant", "ε", "bets", "win rate", "ROI", "P&L"],
            [[row["variant"], row["eps"], row["n_bets"], percentage(row["win_rate"]),
              percentage(row["roi"], True), f"{row['pnl']:+.4f}"] for row in primary["gates"]],
        ),
        f"\n**Deployable gate, frozen from February:** `{gate['variant']}`, ε={gate['eps']:.2f}; "
        f"February validation {gate['n_bets']} bets, ROI {percentage(gate['roi'], True)}; "
        f"March **{march_gate['n_bets']} bets, ROI {percentage(march_gate['roi'], True)}**. "
        f"Minimum validation bets: {gate['minimum_bets']}; insufficient-sample fallback used: {gate['fallback']}.\n",
        f"**Descriptive March-best gate (hindsight, not selected for deployment):** `{best['variant']}`, "
        f"ε={best['eps']:.2f}, {best['n_bets']} bets, ROI **{percentage(best['roi'], True)}**. "
        "This is the best of ten reported March cells and is explicitly selection-biased.\n",
        "## Quarter-Kelly\n",
        markdown_table(
            ["gate", "gate bets", "positive-EV stakes", "bankroll multiple", "return", "total staked", "ROI on staked", "daily max drawdown"],
            [[label, row["n_gate_bets"], row["n_staked"], f"{row['bankroll_multiple']:.6f}",
              percentage(row["net_return"], True), f"{row['total_staked']:.6f}",
              percentage(row["roi_on_staked"], True), percentage(row["max_daily_drawdown"])]
             for label, row in [("February-frozen", primary["quarter_kelly"]),
                                ("March-best, hindsight", primary["hindsight_quarter_kelly"])]],
        ),
        "\nStake fraction = clip[0.25·(p·decimal−1)/(decimal−1), 0, 0.05]. Negative-EV gate bets get zero Kelly stake. "
        "All bets on an Eastern date use that day's starting bankroll; winnings are not recycled into games that may overlap. "
        "Daily total stake is capped at bankroll. These are frictionless closing-price simulations, not executable intraday bankroll replays.\n",
        "## Ablations — fixed common complete-case training and benchmark rows\n",
        markdown_table(
            ["config", "features kept / candidates", "pre-Feb CV LL", "calibration", "accuracy", "flat ROI", "Brier", "Feb gate", "March bets", "frozen-gate March ROI"],
            [[name, f"{len(row['selection']['selected_features'])}/{len(row['selection']['candidate_features'])}",
              number(row["selection"]["cv_log_loss"]), row["selection"]["calibration"],
              percentage(row["metrics"]["accuracy"]), percentage(row["metrics"]["flat_roi"], True),
              number(row["metrics"]["brier"]), f"{row['frozen_gate']['variant']} ε={row['frozen_gate']['eps']}",
              row["frozen_gate_march"]["n_bets"], percentage(row["frozen_gate_march"]["roi"], True)]
             for name, row in results["candidates"].items()],
        ),
        "\nEach ablation reruns backward selection, all three family grids, top-two voting, market shrinkage and chronological "
        "February calibration using only its permitted features. Training and test rows are held fixed across ablations. "
        "`no_clv` removes current opening probabilities plus historical line movement/steam/update features, but retains the "
        "required current Kaggle closing-market signal. `no_players` removes both the supplied six player-feature differences "
        "and the six chunked advanced-player differences. The primary configuration was selected by pre-Feb CV, not this March table.\n",
        "## Bootstrap 95% confidence intervals\n",
        f"{ci['n_resamples']} resamples, seeded at 202203; {ci['n_dates']} whole Eastern dates sampled with replacement. "
        "Games from the same day stay together. Percentile intervals condition on the fitted model and the fixed February gate; "
        "they do not include model-selection or future-season uncertainty.\n",
        markdown_table(
            ["metric", "point estimate", "95% CI"],
            [
                ["accuracy", percentage(performance["accuracy"]), f"[{percentage(ci['accuracy'][0])}, {percentage(ci['accuracy'][1])}]"],
                ["flat ROI", percentage(performance["flat_roi"], True), f"[{percentage(ci['flat_roi'][0], True)}, {percentage(ci['flat_roi'][1], True)}]"],
                ["model Brier", number(performance["brier"]), f"[{number(ci['brier'][0])}, {number(ci['brier'][1])}]"],
                ["market Brier", number(performance["market_brier"]), f"[{number(ci['market_brier'][0])}, {number(ci['market_brier'][1])}]"],
                ["paired Brier − market", number(performance["brier"] - performance["market_brier"]),
                 f"[{number(ci['brier_minus_market'][0])}, {number(ci['brier_minus_market'][1])}]"],
                ["February-frozen gate ROI", percentage(march_gate["roi"], True),
                 f"[{percentage(ci['gate_roi'][0], True)}, {percentage(ci['gate_roi'][1], True)}]"],
                ["March-best gate ROI (conditional, selection-biased)", percentage(best["roi"], True),
                 f"[{percentage(primary['hindsight_confidence_intervals']['gate_roi'][0], True)}, "
                 f"{percentage(primary['hindsight_confidence_intervals']['gate_roi'][1], True)}]"],
            ],
        ),
        "\nNo interval for the hindsight-best gate corrects for choosing that gate on March. "
        "A single month and correlated teams limit the inference; no profitability claim is made merely from a positive point estimate.\n",
        "## Verdict\n",
    ]
    accuracy_delta = performance["accuracy"] - baseline["accuracy"]
    roi_delta = performance["flat_roi"] - baseline["flat_roi"]
    evidence = "excludes zero on the positive side" if ci["flat_roi"][0] is not None and ci["flat_roi"][0] > 0 else "does not establish positive profitability"
    lines.append(
        f"The pre-March-selected v3 configuration changes accuracy by **{accuracy_delta * 100:+.2f} percentage points** "
        f"and flat ROI by **{roi_delta * 100:+.2f} points** versus the reproduced v2 benchmark. "
        f"Its model Brier is {'better' if performance['brier'] < performance['market_brier'] else 'worse'} than the closing market "
        f"({performance['brier']:.6f} versus {performance['market_brier']:.6f}). "
        f"The flat-ROI bootstrap {evidence}. The deployable February-frozen gate returns "
        f"{percentage(march_gate['roi'], True)} in March; the hindsight March-best gate returns "
        f"{percentage(best['roi'], True)} and must not be mistaken for an out-of-sample-selected policy. "
        f"Quarter-Kelly on the frozen gate finishes at {primary['quarter_kelly']['bankroll_multiple']:.6f}x "
        f"starting bankroll ({percentage(primary['quarter_kelly']['net_return'], True)}), so the positive flat-gated "
        "point estimate did not translate into profitable fractional staking. "
        f"The registered no_players ablation reached {percentage(results['candidates']['no_players']['metrics']['accuracy'])} accuracy "
        f"and {percentage(results['candidates']['no_players']['metrics']['flat_roi'], True)} flat ROI; it was not selected "
        "by pre-February CV and is not promoted after seeing March. "
        "Richer inputs do not guarantee an edge over an efficient closing line. "
        "This is one locked historical benchmark with one disclosed missing-opening market fallback, not a live betting recommendation."
    )
    lines += [
        "\n## Reproduction and artifacts\n",
        "```sh\nPYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B backtest/v3/run_v3.py\n"
        "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B backtest/v3/run_v3.py --report-only\n"
        "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider backtest/v3 -q "
        "--basetemp=backtest/v3/test_tmp_final\n```\n",
        "The runner re-execs itself on macOS with the already installed sklearn OpenMP library on "
        "`DYLD_LIBRARY_PATH`; no Homebrew installation or system/venv library patch is needed. "
        "Model/feature caches are accepted only when source SHA-256 fingerprints match. "
        "`--report-only` renders captured numbers without retraining or rereading source datasets.\n",
        f"Package versions: `{json.dumps(results['versions'], sort_keys=True)}`.\n",
        "- `backtest/v3/v3_predictions.csv`: required primary per-game predictions and settlement P&L.",
        "- `backtest/v3/metrics.json`: captured numerical results, all candidate selections, all gates, bootstrap intervals and source manifests.",
        "- `backtest/v3/frozen_protocol.json`: primary configuration and every February gate, written before March prediction.",
        "- `backtest/v3/experiment_plan.json`: registered grids, thresholds and code fingerprints.",
        "- `backtest/v3/features_train.csv`, `features_test.csv`, `feature_audit.csv`, `feature_manifest.json`: captured features and temporal provenance.",
        "- `backtest/v3/odds_diagnostics.csv`: diagnostic current-game movement/steam/updates/close, never passed to model selection.",
        "- `backtest/v3/selection_*.json`, `cv_results_*.csv`, `validation_*.csv`, `model_*.joblib`, `predictions_*.csv`: all five candidate artifacts.",
        "- `backtest/v3/output_manifest.json`: complete absolute output paths, including implementation, logs and all generated artifacts.",
        "- `README_v3.md`, `v3_SCHEMA.md`: formulas, protocol, schema and March 1 odds examples.",
        "- Verification evidence: `backtest/v3/final_verification.log`, `backtest/v3/numerical_verification.log`, "
        "`backtest/v3/feature_arithmetic_verification.log`, and `backtest/v3/baseline_temporal_audit.log`. "
        "Independent numerical verification checks every candidate's settlement and gate table, quarter-Kelly and all seven primary bootstrap intervals.",
    ]
    (ROOT / "V3_BACKTEST_REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    configure_runtime()
    import numpy as np
    import pandas as pd
    import warnings
    from backtest.v3 import evaluation_v3 as E
    from backtest.v3 import features_v3 as F
    from backtest.v3 import model_v3 as M
    from backtest.v3.baseline_v2 import fingerprints, reproduce_v2
    from backtest.v3.schema_discovery import discover

    parser = argparse.ArgumentParser(description="Locked March-2022 NBA v3 backtest")
    parser.add_argument("--report-only", action="store_true", help="Render the frozen metrics capture without recomputation")
    args = parser.parse_args()
    if args.report_only:
        results = json.loads((HERE / "metrics.json").read_text())
        render_report(results)
        inventory = output_inventory()
    else:
        warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")
        print("== V2 published-model reproduction sanity gate ==", flush=True)
        baseline = reproduce_v2()
        if not (ROOT / "v3_SCHEMA.md").exists():
            discover()
        built = F.build_features()
        train, test = built["train"], built["test"]
        specifications = {
            "all_features": F.FEATURE_NAMES,
            "no_players": [c for c in F.FEATURE_NAMES if F.FEATURE_GROUPS[c] != "players"],
            "no_clv": [c for c in F.FEATURE_NAMES if F.FEATURE_GROUPS[c] != "clv"],
            "no_boxscores": [c for c in F.FEATURE_NAMES if F.FEATURE_GROUPS[c] != "boxscores"],
            "no_injuries": [c for c in F.FEATURE_NAMES if F.FEATURE_GROUPS[c] != "injuries"],
        }
        plan = {
            "version": 1, "training_window": [F.TRAIN_START, F.TRAIN_END],
            "test_window": [F.TEST_START, F.TEST_END], "expected_test_games": 229,
            "primary_selection": "Lowest pre-February ensemble CV log loss across all five registered configurations; ties prefer registration order.",
            "candidate_features": specifications, "n_date_grouped_time_folds": 5,
            "greedy_logreg_C": 0.05, "minimum_backward_improvement": 1e-5, "minimum_features": 3,
            "logreg_C_grid": M.LR_C, "lightgbm_grid": M.LGB_GRID, "xgboost_grid": M.XGB_GRID,
            "soft_vote_weights_first_of_top_two": M.VOTE_WEIGHTS, "model_vs_market_weights": M.MODEL_WEIGHTS,
            "calibration_fit": ["2022-02-01", "2022-02-14"],
            "calibration_selection_and_gate_validation": ["2022-02-15", "2022-02-28"],
            "calibration_methods": ["none", "platt", "isotonic"],
            "gate_selection": "Highest Feb-validation ROI with >=30 bets; tie more bets, lower epsilon, variant name; fallback winner eps=0 if no eligible gate.",
            "gate_variants": ["lean", "winner"], "eps_grid": E.EPS_GRID,
            "quarter_kelly": {"fraction": 0.25, "per_bet_cap": 0.05, "simultaneous_date_bankroll": True},
            "bootstrap": {"n_resamples": E.N_BOOT, "seed": E.SEED, "unit": "Eastern calendar date"},
            "missing_feature_policy": built["manifest"]["test_fallback_policy"],
            "code_fingerprints": fingerprints([Path(__file__), Path(F.__file__), Path(M.__file__), Path(E.__file__)]),
        }
        (HERE / "experiment_plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
        trained, selections, validations, frozen_gates = {}, {}, {}, {}
        for name, columns in specifications.items():
            print(f"\n== Train-only candidate: {name} ==", flush=True)
            trained[name], selections[name], validations[name] = M.select_and_fit(train, columns, name)
            frozen_gates[name], table = E.choose_gate(validations[name])
            pd.DataFrame(table).to_csv(HERE / f"validation_gates_{name}.csv", index=False)
        primary_name = min(specifications, key=lambda name: selections[name]["cv_log_loss"])
        frozen = {
            "primary_name": primary_name, "primary_selection": plan["primary_selection"],
            "selected_gates": frozen_gates, "candidate_cv_log_losses": {
                name: row["cv_log_loss"] for name, row in selections.items()
            }, "training_input_fingerprints": fingerprints([HERE / "features_train.csv"]),
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (HERE / "frozen_protocol.json").write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
        print(f"\n== SELECTION FROZEN; first March predictions now permitted. Primary: {primary_name} ==", flush=True)
        complete = np.isfinite(test[F.FEATURE_NAMES].to_numpy(dtype=float)).all(axis=1)
        candidates = {}
        probabilities = {}
        for name in specifications:
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
            prediction["prediction_source"] = np.where(complete, name, "market_fallback_missing_feature")
            prediction = prediction[["date", "home", "away", "p_model_home", "p_market_home",
                                     "predicted_side", "ml_of_side", "correct", "pnl_1usd",
                                     "decimal_of_side", "prediction_source"]]
            prediction.to_csv(HERE / f"predictions_{name}.csv", index=False)
            if name == primary_name:
                prediction.to_csv(HERE / "v3_predictions.csv", index=False)
            candidates[name] = {
                "selection": selections[name], "metrics": E.evaluate(test, p), "gates": gates,
                "frozen_gate": gate, "frozen_gate_march": frozen_metrics,
                "hindsight_best_gate": E.hindsight_best(gates),
            }
        primary = candidates[primary_name]
        p = probabilities[primary_name]
        gate, best = primary["frozen_gate"], primary["hindsight_best_gate"]
        primary["quarter_kelly"] = E.quarter_kelly(E.bet_frame(test, p, gate["variant"], gate["eps"]))
        primary["hindsight_quarter_kelly"] = E.quarter_kelly(E.bet_frame(test, p, best["variant"], best["eps"]))
        primary["confidence_intervals"] = E.bootstrap_intervals(test, p, gate["variant"], gate["eps"])
        primary["hindsight_confidence_intervals"] = E.bootstrap_intervals(test, p, best["variant"], best["eps"])
        results = {
            "primary_name": primary_name, "candidates": candidates, "v2": baseline, "features": built["manifest"],
            "protocol": frozen, "versions": {
                package: importlib.metadata.version(package)
                for package in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost"]
            },
        }
        (HERE / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        render_report(results)
        inventory = output_inventory()
    primary = results["candidates"][results["primary_name"]]
    metric = primary["metrics"]
    chosen, chosen_result = primary["frozen_gate"], primary["frozen_gate_march"]
    best = primary["hindsight_best_gate"]
    print(f"\nv3 accuracy: {metric['accuracy']:.6%}")
    print(f"v3 flat ROI: {metric['flat_roi']:+.6%}")
    print(f"February-frozen best gate: {chosen['variant']}, eps={chosen['eps']:.2f}; "
          f"March ROI={percentage(chosen_result['roi'], True)}, {chosen_result['n_bets']} bets")
    print(f"Descriptive March-best gate (hindsight): {best['variant']}, eps={best['eps']:.2f}; "
          f"ROI={percentage(best['roi'], True)}, {best['n_bets']} bets")
    print("Exact output paths:")
    for path in inventory["outputs"]:
        print(path)
    print("END_TO_END_EXIT: 0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
