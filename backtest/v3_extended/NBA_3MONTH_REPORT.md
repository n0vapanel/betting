# NBA 3-month extended backtest (Jan 1 - Mar 31 2022)

## Headline results

**Primary configuration: `plus_minutes`**, chosen by the lowest pre-December ensemble CV log loss among the four registered candidates, not by Jan-Mar results. 19/61 candidate features survived backward selection; calibration: **none**. Flat $1 at Kaggle closing moneylines.

| configuration | games | accuracy | flat $1 ROI | P&L | model Brier | market Brier |
| --- | --- | --- | --- | --- | --- | --- |
| v3 reference (March 2022, 1-month) | 229 | 65.50% | -4.52% | -10.3519 | 0.209219 | 0.201028 |
| extended: plus_minutes | 623 | 65.97% | -4.30% | -26.7675 | 0.212247 | 0.207814 |

The extended benchmark contains the locked 623 Jan-Mar 2022 games with Kaggle closing moneylines (Jan 231, Feb 163, Mar 229 — the January/February schedule was reduced by COVID postponements). The v3 reference row is v3's published March-2022 result on the same settlement source; it is not recomputed here.

## Protocol

- **Strict walk-forward:** the model trains once on everything before 2022-01-01 (complete-feature rows 2021-11-01..2021-12-31); weights, feature selection, calibration, mixture and the deployable gate are frozen on 2021-12-31. Jan-Mar test rows only update rolling statistics from strictly earlier completed games; model parameters never update.
- **Model:** v3 methodology — greedy backward selection (LogReg C=0.05, 5 date-grouped expanding folds over pre-Dec-16 development), LogReg/LightGBM/XGBoost grids, top-two-family soft vote, optional no-vig market shrinkage, chronological calibration (fit Dec 16-23, choose Dec 24-31), final refit through Dec 31.
- **Edge gates:** frozen on the held-out Dec 24-31 predictions (highest ROI with ≥20 bets; fallback winner ε=0). Applied to Jan-Mar without seeing Jan-Mar.
- **Eval:** flat $1 at closing ML; accuracy; ROI; Brier vs no-vig market; 1000-resample bootstrap CIs clustered by Eastern calendar date; quarter-Kelly diagnostics.
- **Fallback policy:** any Jan-Mar test row missing a candidate's features gets that candidate's Kaggle no-vig market probability (disclosed below); the game stays in the 623.

## Data

- Closing moneylines: Kaggle `cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024` (coverage 2007-10-30..2023-01-16), team codes normalized (`gs/ny/no/sa/utah/wsh` handled). All 623 test games have closing moneylines; NBA and Kaggle scores cross-asserted equal.
- Official injury reports: `nbainjuries` 5PM ET filings (verified against a live fetch of 2022-01-05 17:00 — 150/156 exact row matches). Filing usage in this window: {"game-day 5PM official filing": 1089, "day-before 5PM official filing (early tip)": 49, "day-before 5PM filing has only NOT YET SUBMITTED rows": 12}.
- Minutes load: cached player boxscores, 787,531 rows scanned in 50,000-row chunks (24,381 retained); the 766 MB file is never loaded whole.
- Engineered predictors: **67** = v3's 56 + 5 minutes-load + 6 official-filing injury features. Full formulas in `features_extended.py` docstring and the manifest.
- Complete training games: **428** (99 of 527 closing-priced candidates excluded for warm-up or unavailable precomputed player features — the supplied player feature file starts Nov 1; earlier values are not invented).
- No cached input is written; all sources are SHA-256 fingerprinted in the manifests.

- **Disclosed fallback:** 2022-01-08 MEM at LAC (`0022100591`) lacks `injx_home,injx_away,injx_diff,injout_home,injout_away,injout_diff` for the all-features candidate; those candidates fall back to market probability.
- **Disclosed fallback:** 2022-01-09 ATL at LAC (`0022100597`) lacks `injx_home,injx_away,injx_diff,injout_home,injout_away,injout_diff` for the all-features candidate; those candidates fall back to market probability.
- **Disclosed fallback:** 2022-01-10 PHI at HOU (`0022100711`) lacks `open_implied_home,open_implied_away,open_home` for the all-features candidate; those candidates fall back to market probability.
- **Disclosed fallback:** 2022-01-16 PHX at DET (`0022100617`) lacks `open_implied_home,open_implied_away,open_home` for the all-features candidate; those candidates fall back to market probability.
- **Disclosed fallback:** 2022-01-17 IND at LAC (`0022100659`) lacks `injx_home,injx_away,injx_diff,injout_home,injout_away,injout_diff` for the all-features candidate; those candidates fall back to market probability.
- **Disclosed fallback:** 2022-01-24 CHI at OKC (`0022100730`) lacks `open_implied_home,open_implied_away,open_home` for the all-features candidate; those candidates fall back to market probability.
- **Disclosed fallback:** 2022-01-28 DEN at NOP (`0022100729`) lacks `open_implied_home,open_implied_away,open_home` for the all-features candidate; those candidates fall back to market probability.
- **Disclosed fallback:** 2022-01-31 TOR at ATL (`0022100767`) lacks `open_implied_home,open_implied_away,open_home` for the all-features candidate; those candidates fall back to market probability.
- **Disclosed fallback:** 2022-02-01 MIA at TOR (`0022100784`) lacks `open_implied_home,open_implied_away,open_home` for the all-features candidate; those candidates fall back to market probability.
- **Disclosed fallback:** 2022-02-06 BKN at DEN (`0022100804`) lacks `injx_home,injx_away,injx_diff,injout_home,injout_away,injout_diff` for the all-features candidate; those candidates fall back to market probability.
- **Disclosed fallback:** 2022-02-12 NYK at POR (`0022100845`) lacks `injx_home,injx_away,injx_diff,injout_home,injout_away,injout_diff` for the all-features candidate; those candidates fall back to market probability.
- **Disclosed fallback:** 2022-02-27 UTA at PHX (`0022100913`) lacks `injx_home,injx_away,injx_diff,injout_home,injout_away,injout_diff` for the all-features candidate; those candidates fall back to market probability.
- **Disclosed fallback:** 2022-02-28 TOR at BKN (`0022100726`) lacks `open_implied_home,open_implied_away,open_home` for the all-features candidate; those candidates fall back to market probability.
- **Disclosed fallback:** 2022-03-03 CHI at ATL (`0022100710`) lacks `open_implied_home,open_implied_away,open_home` for the all-features candidate; those candidates fall back to market probability.
- **Disclosed fallback:** 2022-03-20 POR at IND (`0022101062`) lacks `injx_home,injx_away,injx_diff,injout_home,injout_away,injout_diff` for the all-features candidate; those candidates fall back to market probability.
- Complete test rows per candidate: v3_only=615, plus_injuries=608, plus_minutes=615, all=608.

## Ablation — same locked 623-game Jan-Mar window

| config | features kept / candidates | pre-Dec CV LL | calibration | accuracy | flat ROI | P&L | Brier | market Brier | frozen gate | gate bets | gate ROI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v3_only | 18/56 | 0.607701 | none | 69.18% | -0.28% | -1.7330 | 0.207988 | 0.207814 | winner eps=0.01 | 247 | +3.87% |
| plus_injuries | 21/62 | 0.605264 | none | 68.38% | -1.17% | -7.2856 | 0.209421 | 0.207814 | winner eps=0.03 | 189 | -1.17% |
| plus_minutes | 19/61 | 0.595919 | none | 65.97% | -4.30% | -26.7675 | 0.212247 | 0.207814 | lean eps=0.02 | 498 | +1.01% |
| all | 19/67 | 0.602855 | none | 67.74% | -1.98% | -12.3515 | 0.209526 | 0.207814 | winner eps=0.03 | 220 | -3.82% |

Each candidate reruns backward selection, all three family grids, top-two voting, market shrinkage and chronological December calibration using only its permitted features. Training and test rows are held fixed across candidates. The primary configuration was selected by pre-December CV, not by this Jan-Mar table. `v3_only` keeps exactly v3's 56 features (which already include v3's own injury- and boxscore-derived columns); `plus_injuries` adds the new official-5PM-filing injury-impact features; `plus_minutes` adds the minutes-load features; `all` uses every feature.

## Monthly detail (primary)

| month | games | accuracy | flat ROI | P&L | Brier | market Brier | gate bets | gate ROI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-01 | 231 | 66.67% | -0.35% | -0.8029 | 0.211991 | 0.214521 | 194 | +13.98% |
| 2022-02 | 163 | 62.58% | -11.32% | -18.4533 | 0.219399 | 0.207844 | 128 | -4.25% |
| 2022-03 | 229 | 67.69% | -3.28% | -7.5113 | 0.207414 | 0.201028 | 176 | -9.46% |

## Edge gates (primary, all registered thresholds)

Let d = p_model_home − p_market_home. **Lean-side** bets home if d>0, away if d<0, only when |d|≥ε. **Winner-side conviction** bets the model's >0.5 winner only when the model exceeds the no-vig market probability on that same side by ≥ε and strictly >0. Exact zero edge never bets.

| variant | eps | bets | win rate | ROI | P&L |
| --- | --- | --- | --- | --- | --- |
| lean | 0.0 | 615 | 49.76% | -1.12% | -6.8951 |
| lean | 0.01 | 556 | 49.28% | -1.72% | -9.5460 |
| lean | 0.02 | 498 | 50.00% | +1.01% | +5.0273 |
| lean | 0.03 | 432 | 50.23% | -0.68% | -2.9286 |
| lean | 0.05 | 318 | 49.06% | -0.05% | -0.1665 |
| winner | 0.0 | 374 | 62.57% | -4.54% | -16.9951 |
| winner | 0.01 | 345 | 61.74% | -4.77% | -16.4460 |
| winner | 0.02 | 314 | 62.10% | -2.84% | -8.9227 |
| winner | 0.03 | 282 | 62.77% | -0.28% | -0.7786 |
| winner | 0.05 | 212 | 60.38% | +0.16% | +0.3335 |

**Deployable gate, frozen from December:** `lean`, eps=0.02; Dec 24-31 validation 40 bets, ROI -6.17%; Jan-Mar **498 bets, ROI +1.01%**. Minimum validation bets: 20; insufficient-sample fallback used: False.

**Descriptive Jan-Mar-best gate (hindsight, not selected for deployment):** `lean`, eps=0.02, 498 bets, ROI **+1.01%**. This is the best of ten reported Jan-Mar cells and is explicitly selection-biased.

## Quarter-Kelly (primary, frozen gate)

Stake fraction = clip[0.25·(p·decimal−1)/(decimal−1), 0, 0.05]; simultaneous games use start-of-Eastern-date bankroll. Bankroll multiple **0.759346x** (-24.07%), staked 17.218722, ROI on staked -1.40%, max daily drawdown 69.64%. Frictionless closing-price simulation, not executable intraday replay.

## Bootstrap 95% confidence intervals (primary, date-clustered)

1000 resamples, seeded at 202203; 84 whole Eastern calendar dates sampled with replacement. Games from the same day stay together. Percentile intervals condition on the fitted model and fixed December gate; they exclude model-selection and future-season uncertainty.

| metric | point estimate | 95% CI |
| --- | --- | --- |
| accuracy | 65.97% | [62.16%, 69.92%] |
| flat ROI | -4.30% | [-10.26%, +1.25%] |
| model Brier | 0.212247 | [0.195968, 0.227040] |
| market Brier | 0.207814 | [0.191957, 0.222055] |
| paired Brier − market | 0.004433 | [-0.001520, 0.009921] |
| December-frozen gate ROI | +1.01% | [-10.12%, +11.41%] |

## Verdict

The December-selected `plus_minutes` configuration posts **65.97%** accuracy and **-4.30%** flat ROI over the locked 623-game Jan-Mar 2022 window (623 games), versus v3's March-2022 reference of 65.5% accuracy / -4.52% flat ROI on its own 229-game window. The flat-ROI bootstrap does not establish positive profitability (95% CI [-10.26%, +1.25%]). Model Brier is worse than the closing market (0.212247 vs 0.207814). The December-frozen gate returns +1.01% on 498 Jan-Mar bets; the hindsight best gate is disclosed but must not be treated as an out-of-sample policy. One locked historical benchmark; no live betting recommendation.

## Reproduction and artifacts

```sh
JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home \
  /Users/x/.hermes/cache/scratch/msports-venv/bin/python -B backtest/v3_extended/run_extended.py
/Users/x/.hermes/cache/scratch/msports-venv/bin/python -B backtest/v3_extended/run_extended.py --report-only
```

Package versions: `{"lightgbm": "4.7.0", "nbainjuries": "1.1.1", "numpy": "2.4.6", "pandas": "2.2.3", "scikit-learn": "1.9.1", "scipy": "1.17.1", "xgboost": "3.2.0"}`.

- `backtest/v3_extended/nba_3month_predictions.csv`: primary per-game Jan-Mar predictions plus all four candidates' probabilities, settlement P&L and gates.
- `backtest/v3_extended/features_train.csv`, `features_test.csv`, `feature_audit.csv`, `feature_manifest.json`: captured features and temporal provenance.
- `backtest/v3_extended/metrics.json`: captured numbers — all candidates, gates, bootstrap intervals, source manifests.
- `backtest/v3_extended/experiment_plan.json`, `frozen_protocol.json`: registered grids, thresholds, code fingerprints and the pre-test frozen selection.
- `backtest/v3_extended/selection_*.json`, `cv_results_*.csv`, `validation_*.csv`, `model_*.joblib`, `predictions_*.csv`: per-candidate artifacts.
