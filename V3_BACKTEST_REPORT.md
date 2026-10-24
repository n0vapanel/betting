# V3 NBA walk-forward backtest report

## Headline results

**Primary configuration: `all_features`**, chosen by the lowest pre-February five-fold out-of-fold log loss among the five registered feature-group configurations, not by March results. 15/56 candidate features survived backward selection; calibration: **platt**.

| configuration | games | accuracy | flat $1 ROI | P&L | model Brier | market Brier |
| --- | --- | --- | --- | --- | --- | --- |
| v2 reproduced | 229 | 68.12% | +0.10% | +0.2277 | 0.213078 | 0.201028 |
| v3: all_features | 229 | 65.50% | -4.52% | -10.3519 | 0.209219 | 0.201028 |

The primary result includes the same 229 March games and the same Kaggle closing-moneyline settlement prices as v2. Predictions are stored at full precision. The all-features candidate and every ablation are reported below regardless of whether they perform well.

## V2 sanity gate

The frozen published specification was actually refit: LogReg C=0.05, its nine published features, and the original February isotonic procedure. Training rows: 843; test rows: 229. Gate passed: **True**. Rounded results reproduce 68.1%, +0.10%, 0.2131 and 0.2010. The original v2 feature-selection search was not rerun; this gate reproduces the published chosen model. Baseline probabilities, exact metrics and source fingerprints are preserved under `backtest/v3/v2_*`.

v2 limitations preserved for reproduction, but corrected in v3:

- v2 chooses calibration by in-sample February calibration loss; preserved for exact reproduction only.
- v2's 1PM early-tip routing can use a report at tip for a 1PM game; v3 requires strict snapshot < tip.
- v2 ORB denominator uses opposing OREB instead of opposing DREB; v3 corrects this.
- v2 injury minutes are double-lagged and season-to-date denominators are off by one; v3 corrects both.

Additional timing audit: v2's hour<6 routing also selects the game-day 17:00 report for six 12:00/12:30 ET training games (2021-12-12, 2021-12-18, 2021-12-25, 2022-01-09, 2022-01-17, 2022-02-26), after their scheduled tips. This is confirmed in `backtest/v3/baseline_temporal_audit.log`. The published baseline is reproduced, not retroactively repaired; its training-feature timing is therefore not a fully leakage-free control. V3 uses strictly previous-day snapshots for all six.


## Data coverage and source usability

- Complete training games: **820**, 2021-11-01 through 2022-02-28; authorized training window is 2021-10-20 through 2022-02-28. 99 of 919 closing-priced training candidates are excluded for warm-up, unavailable player features, missing openings or incomplete pre-tip injuries. No future backfill is used.
- Engineered predictors: **56**. Full formulas and group assignments are in `README_v3.md`.
- Advanced player CSV: **787,531 rows scanned in 50,000-row chunks**, 24,381 retained. The 766 MB file is never loaded whole. Only selected columns and the current regular season are retained.
- No cached input is written. Every source and relevant implementation has a SHA-256 fingerprint in the manifests.
- The supplied precomputed player file starts on November 1. Its source implementation and 23 existing temporal tests are retained unchanged; earlier feature values are not invented.
- **Disclosed benchmark fallback:** 2022-03-03 CHI at ATL (`0022100710`) lacks `open_implied_home,open_implied_away,open_home`. All five configurations use the Kaggle no-vig market probability for this row. The fallback was specified before March predictions; the game stays in the 229-game denominator and is not presented as a learned-model prediction.
- Unusable injury source: 2021-12-31 PHX at BOS: Only incomplete pre-tip injury reports for 2021-12-31 PHX@BOS. Excluded from complete-case training, not treated as healthy.

## Temporal discipline and exact availability rules

- **statistics:** Completed games with source date strictly less than target date; rolling last 10, minimum 5, EWMA half-life 10.
- **opening:** Only odds_date == open; no marker publication timestamp exists. Semantic historical opening, not exact point-in-time provable.
- **closing:** Kaggle supplied closing moneylines are the sole current closing market input and settlement price; actual publication timestamps unavailable.
- **movement:** Current game close/move/steam/update-count fields are never predictors. Historical versions use strictly previous game dates; timed updates restricted to that game's Eastern calendar day.
- **injuries:** Latest complete official snapshot with Eastern timestamp strictly less than scheduled Eastern tip; 1–11 in the 12-hour cache are afternoon/evening.
- **march:** Weights and selections freeze before March. Earlier completed March games update statistics, never model weights.
- `feature_audit.csv` records, for each train/test game, all rolling-history maximum source dates, the actual selected injury snapshot and tip in Eastern time, completeness and missing-source reasons. Every dated historical source is asserted strictly earlier than its target; injury timestamps must be strictly less than tip.
- Schedule-clock provenance: injury routing uses the first cached clock label for each game, as recorded in the audit. The sole changed label is 2022-02-03 MIA@SAS (20:30 versus 19:00 ET); the selected 17:00 report is strictly pre-tip under both. This ambiguity changes no predictor or selected snapshot in this benchmark.
- **Important limitation:** neither semantic opening/closing markers nor the supplied Kaggle close have original publication timestamps. This is an opening/closing-convention historical backtest, not a point-in-time feed replay. The semantic open is presumed pregame; that assumption cannot be proven to the second from these caches. `last_fetched` is never used as a historical availability timestamp.
- Current-game close, movement, steam, update counts and quarter results do not enter the predictor matrix. The sole current closing signal is Kaggle no-vig market probability (and its logit). Prior-game CLV/steam/update features are lagged.
- March model weights do not update. Statistics can incorporate completed earlier March games; model parameters, features, calibration choice, mixture weights, primary configuration and deployable gate are frozen before evaluating March.

## Cross-check: TeamRankings close versus fixed Kaggle close

| period | matched / games | exact both ML | both within 10 ML | mean abs Δp | median abs Δp | max abs Δp | same favorite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| training | 912 / 919 | 10.42% | 63.38% | 0.007419 | 0.005805 | 0.106833 | 98.57% |
| march | 228 / 229 | 10.09% | 56.14% | 0.006887 | 0.005906 | 0.029395 | 100.00% |

These are different source/book snapshots, not interchangeable settlement prices. No best-line shopping or replacement of Kaggle closes occurs. `odds_diagnostics.csv` includes opening probability, both-side movement, largest adjacent intraday probability move, update count, source close and timed-update bounds. Current-game diagnostic fields are excluded from every candidate's feature list.

## Registered model selection and calibration

Development games (before Feb 1): 659; February holdout: 161. Five expanding folds are grouped by calendar date, so simultaneous games never cross a fold boundary. Fold standardization is fit only on that fold's training rows. Greedy backward selection uses LogReg C=0.05 and removes the single feature giving the best CV log-loss improvement above 0.00001 until no improvement remains or three features remain. Family grids are then retuned on that selected subset.

| fold | train dates | train n | validation dates | validation n |
| --- | --- | --- | --- | --- |
| 1 | 2021-11-01..2021-11-15 | 116 | 2021-11-16..2021-12-01 | 118 |
| 2 | 2021-11-01..2021-12-01 | 234 | 2021-12-02..2021-12-16 | 103 |
| 3 | 2021-11-01..2021-12-16 | 337 | 2021-12-17..2022-01-01 | 102 |
| 4 | 2021-11-01..2022-01-01 | 439 | 2022-01-02..2022-01-16 | 110 |
| 5 | 2021-11-01..2022-01-16 | 549 | 2022-01-17..2022-01-31 | 110 |

| family | parameters | CV log loss |
| --- | --- | --- |
| logreg | {"C": 0.3} | 0.600208 |
| logreg | {"C": 0.1} | 0.600497 |
| logreg | {"C": 1.0} | 0.601106 |
| logreg | {"C": 3.0} | 0.602112 |
| logreg | {"C": 0.05} | 0.602142 |
| logreg | {"C": 0.03} | 0.604563 |
| logreg | {"C": 0.01} | 0.614557 |
| lightgbm | {"n_estimators": 120, "learning_rate": 0.03, "num_leaves": 3, "min_child_samples": 40, "reg_lambda": 10.0} | 0.625325 |
| lightgbm | {"n_estimators": 80, "learning_rate": 0.05, "num_leaves": 5, "min_child_samples": 30, "reg_lambda": 10.0} | 0.626984 |
| xgboost | {"n_estimators": 80, "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 10, "reg_lambda": 30.0} | 0.628804 |
| lightgbm | {"n_estimators": 120, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 40, "reg_lambda": 30.0} | 0.629115 |
| lightgbm | {"n_estimators": 160, "learning_rate": 0.03, "num_leaves": 5, "min_child_samples": 50, "reg_lambda": 50.0} | 0.630577 |
| lightgbm | {"n_estimators": 60, "learning_rate": 0.03, "num_leaves": 3, "min_child_samples": 40, "reg_lambda": 10.0} | 0.632842 |
| xgboost | {"n_estimators": 120, "learning_rate": 0.03, "max_depth": 1, "min_child_weight": 5, "reg_lambda": 10.0} | 0.633136 |
| logreg | {"C": 0.003} | 0.633399 |
| xgboost | {"n_estimators": 60, "learning_rate": 0.03, "max_depth": 1, "min_child_weight": 5, "reg_lambda": 10.0} | 0.639469 |
| xgboost | {"n_estimators": 120, "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 20, "reg_lambda": 30.0} | 0.644426 |
| xgboost | {"n_estimators": 160, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 20, "reg_lambda": 100.0} | 0.648457 |

Top two distinct families: `[{"kind": "logreg", "parameters": {"C": 0.3}}, {"kind": "lightgbm", "parameters": {"n_estimators": 120, "learning_rate": 0.03, "num_leaves": 3, "min_child_samples": 40, "reg_lambda": 10.0}}]`. Soft vote puts 90% on the first and 10% on the second. The pre-registered CV-selected final probability mixes 100% soft vote with 0% no-vig market probability. Ensemble CV LL **0.600424** versus market **0.615967**. This transparent market shrinkage is part of the model, not hidden postprocessing.

Base/feature/family/ensemble selection only before February. Fit calibrators Feb 1–14; choose using Feb 15–28 log loss. Gate validation uses those genuinely held-out predictions. Refit the chosen calibrator on all February out-of-sample raw predictions, then fit base models through Feb 28. No March labels enter any choice.

| calibration | Feb 15–28 held-out log loss |
| --- | --- |
| none | 0.635266 |
| platt | 0.631874 |
| isotonic | 0.665453 |

Calibration-fit rows (Feb 1–14): 101; calibration-selection/gate-validation rows (Feb 15–28): 60. Using the second block for both calibration choice and gate selection is disclosed multiple use of validation data; it does not make February an unbiased performance estimate. March remains untouched by selection. CV scores are selection estimates, not nested-CV generalization guarantees.

Selected predictors: `off_eFG_home, tov_home, ftr_away, b2b_home, market_home, winpct_diff, market_logit, player_margin_std_ewma_diff, player_adv_creation_diff, open_implied_home, open_implied_away, open_home, prior_updates_diff, box_ts_diff, box_q4_margin_diff`.

## Edge gates: both interpretations, all registered thresholds

Let d = p_model_home − p_market_home. **Lean-side** bets home if d>0, away if d<0, only when |d|≥ε. **Winner-side conviction** bets the model's >0.5 winner only when the model exceeds the no-vig market probability on that same side by ≥ε and strictly >0. Exact zero edge never bets. No-vig edge is not the same as positive EV at the vig-inclusive payout; gates deliberately match the requested definitions.

| variant | ε | bets | win rate | ROI | P&L |
| --- | --- | --- | --- | --- | --- |
| lean | 0.0 | 228 | 40.79% | -4.15% | -9.4561 |
| lean | 0.01 | 207 | 40.10% | -1.97% | -4.0828 |
| lean | 0.02 | 193 | 40.93% | +2.22% | +4.2812 |
| lean | 0.03 | 179 | 39.66% | +1.37% | +2.4561 |
| lean | 0.05 | 144 | 41.67% | +9.68% | +13.9378 |
| winner | 0.0 | 110 | 56.36% | -5.96% | -6.5561 |
| winner | 0.01 | 99 | 54.55% | -6.30% | -6.2328 |
| winner | 0.02 | 94 | 53.19% | -7.31% | -6.8688 |
| winner | 0.03 | 85 | 50.59% | -9.64% | -8.1939 |
| winner | 0.05 | 72 | 50.00% | -8.49% | -6.1122 |

**Deployable gate, frozen from February:** `lean`, ε=0.05; February validation 38 bets, ROI +29.84%; March **144 bets, ROI +9.68%**. Minimum validation bets: 30; insufficient-sample fallback used: False.

**Descriptive March-best gate (hindsight, not selected for deployment):** `lean`, ε=0.05, 144 bets, ROI **+9.68%**. This is the best of ten reported March cells and is explicitly selection-biased.

## Quarter-Kelly

| gate | gate bets | positive-EV stakes | bankroll multiple | return | total staked | ROI on staked | daily max drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| February-frozen | 144 | 144 | 0.896167 | -10.38% | 5.475166 | -1.90% | 45.81% |
| March-best, hindsight | 144 | 144 | 0.896167 | -10.38% | 5.475166 | -1.90% | 45.81% |

Stake fraction = clip[0.25·(p·decimal−1)/(decimal−1), 0, 0.05]. Negative-EV gate bets get zero Kelly stake. All bets on an Eastern date use that day's starting bankroll; winnings are not recycled into games that may overlap. Daily total stake is capped at bankroll. These are frictionless closing-price simulations, not executable intraday bankroll replays.

## Ablations — fixed common complete-case training and benchmark rows

| config | features kept / candidates | pre-Feb CV LL | calibration | accuracy | flat ROI | Brier | Feb gate | March bets | frozen-gate March ROI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_features | 15/56 | 0.600424 | platt | 65.50% | -4.52% | 0.209219 | lean ε=0.05 | 144 | +9.68% |
| no_players | 11/44 | 0.607534 | platt | 69.00% | +2.12% | 0.207190 | lean ε=0.05 | 144 | +6.51% |
| no_clv | 12/50 | 0.600876 | platt | 67.25% | -2.79% | 0.207599 | lean ε=0.05 | 143 | +4.56% |
| no_boxscores | 11/42 | 0.600889 | platt | 68.56% | +1.36% | 0.210625 | lean ε=0.05 | 140 | -10.00% |
| no_injuries | 15/51 | 0.600424 | platt | 65.50% | -4.52% | 0.209219 | lean ε=0.05 | 144 | +9.68% |

Each ablation reruns backward selection, all three family grids, top-two voting, market shrinkage and chronological February calibration using only its permitted features. Training and test rows are held fixed across ablations. `no_clv` removes current opening probabilities plus historical line movement/steam/update features, but retains the required current Kaggle closing-market signal. `no_players` removes both the supplied six player-feature differences and the six chunked advanced-player differences. The primary configuration was selected by pre-Feb CV, not this March table.

## Bootstrap 95% confidence intervals

1000 resamples, seeded at 202203; 31 whole Eastern dates sampled with replacement. Games from the same day stay together. Percentile intervals condition on the fitted model and the fixed February gate; they do not include model-selection or future-season uncertainty.

| metric | point estimate | 95% CI |
| --- | --- | --- |
| accuracy | 65.50% | [58.47%, 71.82%] |
| flat ROI | -4.52% | [-14.91%, +5.09%] |
| model Brier | 0.209219 | [0.187522, 0.232266] |
| market Brier | 0.201028 | [0.178392, 0.225014] |
| paired Brier − market | 0.008191 | [-0.002681, 0.020423] |
| February-frozen gate ROI | +9.68% | [-16.57%, +34.69%] |
| March-best gate ROI (conditional, selection-biased) | +9.68% | [-16.57%, +34.69%] |

No interval for the hindsight-best gate corrects for choosing that gate on March. A single month and correlated teams limit the inference; no profitability claim is made merely from a positive point estimate.

## Verdict

The pre-March-selected v3 configuration changes accuracy by **-2.62 percentage points** and flat ROI by **-4.62 points** versus the reproduced v2 benchmark. Its model Brier is worse than the closing market (0.209219 versus 0.201028). The flat-ROI bootstrap does not establish positive profitability. The deployable February-frozen gate returns +9.68% in March; the hindsight March-best gate returns +9.68% and must not be mistaken for an out-of-sample-selected policy. Quarter-Kelly on the frozen gate finishes at 0.896167x starting bankroll (-10.38%), so the positive flat-gated point estimate did not translate into profitable fractional staking. The registered no_players ablation reached 69.00% accuracy and +2.12% flat ROI; it was not selected by pre-February CV and is not promoted after seeing March. Richer inputs do not guarantee an edge over an efficient closing line. This is one locked historical benchmark with one disclosed missing-opening market fallback, not a live betting recommendation.

## Reproduction and artifacts

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B backtest/v3/run_v3.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B backtest/v3/run_v3.py --report-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider backtest/v3 -q --basetemp=backtest/v3/test_tmp_final
```

The runner re-execs itself on macOS with the already installed sklearn OpenMP library on `DYLD_LIBRARY_PATH`; no Homebrew installation or system/venv library patch is needed. Model/feature caches are accepted only when source SHA-256 fingerprints match. `--report-only` renders captured numbers without retraining or rereading source datasets.

Package versions: `{"lightgbm": "4.6.0", "numpy": "2.5.4", "pandas": "2.3.3", "scikit-learn": "1.9.1", "scipy": "1.18.1", "xgboost": "3.1.1"}`.

- `backtest/v3/v3_predictions.csv`: required primary per-game predictions and settlement P&L.
- `backtest/v3/metrics.json`: captured numerical results, all candidate selections, all gates, bootstrap intervals and source manifests.
- `backtest/v3/frozen_protocol.json`: primary configuration and every February gate, written before March prediction.
- `backtest/v3/experiment_plan.json`: registered grids, thresholds and code fingerprints.
- `backtest/v3/features_train.csv`, `features_test.csv`, `feature_audit.csv`, `feature_manifest.json`: captured features and temporal provenance.
- `backtest/v3/odds_diagnostics.csv`: diagnostic current-game movement/steam/updates/close, never passed to model selection.
- `backtest/v3/selection_*.json`, `cv_results_*.csv`, `validation_*.csv`, `model_*.joblib`, `predictions_*.csv`: all five candidate artifacts.
- `backtest/v3/output_manifest.json`: complete absolute output paths, including implementation, logs and all generated artifacts.
- `README_v3.md`, `v3_SCHEMA.md`: formulas, protocol, schema and March 1 odds examples.
- Verification evidence: `backtest/v3/final_verification.log`, `backtest/v3/numerical_verification.log`, `backtest/v3/feature_arithmetic_verification.log`, and `backtest/v3/baseline_temporal_audit.log`. Independent numerical verification checks every candidate's settlement and gate table, quarter-Kelly and all seven primary bootstrap intervals.
