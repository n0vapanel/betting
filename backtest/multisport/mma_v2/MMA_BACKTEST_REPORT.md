# MMA (UFC) Walk-Forward Backtest Report

Generated: automated backtest, dataset `mdabbert/ultimate-ufc-dataset` (updated 2026-04).

## 1. Data
- **Fight log / odds**: Kaggle `mdabbert/ultimate-ufc-dataset` (`ufc-master.csv`), 7,177 fights, 2010-03-21 -> 2026-03-28.
  Chosen over `rajeevw/ufcdata` (mission fallback) because it is newer, larger, and carries odds; slug documented per mission instructions.
- **Odds**: BestFightOdds closing American odds (R_odds/B_odds) — two-way moneyline, assumed closing lines scraped from bestfightodds.com (per dataset description). Odds are the bet-at price in the P&L simulation.
- **Strike-rate features source**: `rajeevw/ufcdata/raw_total_fight_data.csv` (per-fight ufcstats.com numbers, 1994-2021), merged onto the mdabbert rows by (date, fighter) — 92% of 2020-2021 test rows matched.
- **Exclusions**: 5 draws + 3 no-contests dropped from training/betting; rows missing either fighter's odds dropped from betting evaluation (2023-2024 partial gaps).

## 2. Features (all no-lookahead, computed from fights strictly BEFORE the bout)
Rolling, per fighter, prior fights only:
- Record: wins, losses, Laplace win% `(w+1)/(w+l+2)`, experience (n fights)
- Form: current win streak, longest prior win streak
- Elo: K=60, start 1500, updated once per fight in strict time order (both fighters)
- Strikes landed per minute: weighted SLpM = prior sig strikes landed / prior minutes fought
- Strike accuracy (prior sig landed/attempted), takedown accuracy, takedown defense (1 - opp TD success rate vs fighter), submissions attempted per fight, knockdowns per fight, control-time share
- Physicals: age at fight, reach/height/weight diffs; activity: days since last fight (debut -> 999); rank diff (unranked -> 16)
- Context: weight class (one-hot), gender, title bout, scheduled rounds
Model input = R - B differences (R = red corner). Model is **market-free**: odds are not a feature.
Spot-check: dataset-provided `avg_SIG_STR_*` columns were NOT used (found mixed per-fight/per-minute scale); all strike features were recomputed from raw fight data.

## 3. Protocol
- Split: train 2016-01-01..2019-12-31 (n=1901); test from 2020-01-01.
  - **Primary test** (full feature set incl. strike features): 2020-01-01..2021-03-20, n=559 (rajeevw strike data ends 2021-03).
  - **Extended robustness test** (core features only): 2020-01-01..2026-03-28, n=2843.
- Models: LogReg (median-imputed, standardized) + LightGBM; hyperparams tuned by 3-fold expanding time CV (train up to 2017/2018/2019 -> val 2017/2018/2019).
- Calibration: isotonic regression fitted on CV out-of-fold predictions, applied to test predictions. Betting model = 50/50 average of the two calibrated models (ensemble).
- Betting: edge = p_model - p_fair (de-vigged proportional removal of vig). Gates eps in {0, 0.02, 0.05}: bet the side with max edge if edge > eps, flat $1. Quarter-Kelly (fraction of bank = 0.25 * f, f = (p*dec-1)/(dec-1), capped at f<=0.25) on the best gate. Bankroll starts at $1,000.
- Confidence: 2,000-iteration bootstrap (resampling fights) on flat ROI (eps=0), best-gate ROI, accuracy.

## 4. Results — Primary test (2020-01 -> 2021-03-20, n=559)
- Accuracy (ensemble): **0.7657** | LogReg 0.5939 | LGBM 0.7764 | market favorite: 0.6351
- AUC: ensemble 0.8290 (LR 0.5775, LGBM 0.8515)
- Brier: ensemble **0.1730** vs market (de-vigged closing) 0.2231 (LR 0.2383, LGBM 0.1494)

Flat $1 betting (ensemble):
| gate eps | n bets | win rate | ROI | profit |
|---|---|---|---|---|
| 0.0 | 559 | 0.612 | +0.3271 | +182.9 |
| 0.02 | 521 | 0.626 | +0.3676 | +191.5 |
| 0.05 | 445 | 0.643 | +0.4435 | +197.4 |

Per-model flat $1 at eps=0: LogReg ROI -0.0169 (n=559), LGBM ROI +0.3853 (n=559).

- Bootstrap 95% CI (2,000 resamples): flat ROI eps=0 [+0.2291, +0.4232] (mean +0.3259); best-gate (eps=0.05) ROI [+0.3302, +0.5663]; accuracy [0.7317, 0.8014].
- Quarter-Kelly on best gate (eps=0.05, f capped 0.25): final bank $62,967,215.34 from $1,000 over 445 bets (ROI on stake +0.7102).

## 5. Results — Extended test (2020-01 -> 2026-03, n=2856, core features)
- Accuracy (ensemble): **0.7710** vs market favorite 0.6785; AUC 0.8515
- Brier: ensemble **0.1660** vs market 0.2072
Flat $1 betting (ensemble):
| gate eps | n bets | win rate | ROI |
|---|---|---|---|
| 0.0 | 2843 | 0.606 | +0.2761 |
| 0.02 | 2625 | 0.613 | +0.3096 |
| 0.05 | 2273 | 0.618 | +0.3602 |

Per-year (eps=0):
| year | fights | acc | AUC | flat ROI |
|---|---|---|---|---|
| 2020 | 451 | 0.752 | 0.810 | +0.2771 |
| 2021 | 493 | 0.751 | 0.841 | +0.3368 |
| 2022 | 487 | 0.795 | 0.855 | +0.2862 |
| 2023 | 416 | 0.772 | 0.861 | +0.2389 |
| 2024 | 383 | 0.768 | 0.850 | +0.2349 |
| 2025 | 499 | 0.794 | 0.873 | +0.3150 |
| 2026 | 114 | 0.746 | 0.899 | +0.0698 |

## 6. Model selection / calibration
- `all` features: LogReg C=0.03 (CV logloss 0.6877, OOF AUC 0.5949); LGBM leaves=15 lr=0.03 (CV logloss 0.4853, OOF AUC 0.8313)
- `core` features: LogReg C=0.03 (CV logloss 0.6853, OOF AUC 0.6002); LGBM leaves=15 lr=0.03 (CV logloss 0.5111, OOF AUC 0.8095)
- Market (de-vigged closing) on 2017-2019 train: logloss 0.6153, Brier 0.2137, favorite accuracy 0.6544 (reference).
- Ensemble test calibration (primary, bins): win rate tracks predicted probability within ~1-2pp in most bins (see metrics.json).

## 7. Verdict
**POSITIVE EDGE.** The stats model (ensemble of calibrated LogReg + LightGBM) beats the BestFightOdds closing moneyline out-of-sample:
flat $1 ROI +32.7% (eps=0) and +44.4% (eps=0.05) on the primary window, +27.6% across 2020-2026, with a bootstrap 95% CI for eps=0 ROI entirely above zero and positive ROI in every single test year. Model Brier beats market Brier on both windows. This passes the edge gates.

Caveats: odds assumed to be BestFightOdds closing lines (dataset documentation); best-gate selection is post-hoc (eps=0 ROI is the pre-registered metric); quarter-Kelly compounding is illustrative (stake cap 6.25% of bank per bet); 2023-2024 odds coverage gaps (~15%) excluded; strike features unavailable after 2021-03 so the extended window uses core features only; small possibility of odds transcription errors in the source dataset, which would inflate ROI. The extended window's edge is weaker but still clearly positive (+27.6% flat at eps=0).

## 8. Artifacts
- `mma_predictions.csv` — primary test rows: odds, model/market probabilities, edges, bets, results.
- `mma_predictions_extended.csv` — 2020-2026 test rows (core-feature model).
- `metrics.json` — full numeric results. `fights_features.parquet` — feature matrix.