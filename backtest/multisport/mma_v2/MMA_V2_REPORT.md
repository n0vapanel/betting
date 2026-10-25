# MMA (UFC) v2 Walk-Forward Backtest Report

Generated: 2026-10-24. Repo: `backtest/multisport/mma_v2/`.

**Verdict up front: v2 BEATS v1 on all three headline metrics.**
Extended test (2020-2026, n=2843): accuracy **80.2%** (v1 77.1%), flat ROI eps=0 **+40.3%** (v1 +27.6%), Brier **0.1368** (v1 0.1660). Primary test (2020-01..2021-03-20, n=559): accuracy 77.5% (v1 76.6%), flat ROI +35.3% (v1 +32.7%), Brier 0.1650 (v1 0.1730). The optional LGBM+XGB two-model ensemble is stronger still (extended acc 80.4%, ROI +43.4%, Brier 0.1273).

## 1. Upgrades tried (vs v1)
| # | Upgrade | Result |
|---|---|---|
| (a) | Walk-forward yearly refit (model for year Y trained on ALL fights < Jan 1 Y, isotonic on 3 expanding CV folds within the pre-Y window) | **+1.0pp acc / +2.9pp ROI on extended** vs frozen-v2 (see ablation §6). Big wins in later years (2023-2026 ROIs +37..+48% vs v1's +7..+32%). |
| (b) | Per-weight-class models (9 class groups, LGBM+XGB per class, shrinkage blend w=n/(n+K), K tuned on pre-2020 OOF) | **No help.** Pre-2020 OOF logloss worsens monotonically as class weight rises (0.4957 pooled → 0.5694 class-heavy); tuning selected pure pooled (K→∞). Per-class abandoned for the headline. |
| (c) | XGBoost added to ensemble | Yes — ensemble = mean(LogReg, LGBM, XGB). XGB alone: ext acc 79.9%, Brier 0.1302. |
| (d) | New features (below) | **Biggest single gain**: pre-2020 CV logloss LGBM core 0.5111→0.4420, all-feats 0.4853→0.4279. |
| (e) | Odds-as-feature variant | Report-only (§7): ext acc 82.3%, Brier 0.1255. Not the betting model (edge vs market ≈ 0 by construction). |

## 2. New features (all strictly no-lookahead)
- **Physicals from `raw_fighter_details.csv`**: age at fight from DOB (99.1% coverage on primary window), age at debut, height/reach gap-filled into mdabbert values.
- **Career clock**: days since debut; **recent form**: win fraction last 3, SLpM/str-acc last 3 (recency-weighted).
- **Opponent quality** (from fight log): mean pre-fight Elo and win% of prior opponents.
- **Opponent style** (from rajeevw, prior fights only): mean SLpM / strike acc / TD acc / sub rate of prior opponents — "has this fighter faced strikers or grapplers?"
- **Context**: fight country dummies (normalized, top-7), stance flags (southpaw diff, same-stance), empty-arena flag (COVID-era, known pre-fight).
- **Not available in either source** (checked, documented): fighter camp, missed-weight flags. mdabbert career-average stat columns were NOT used (snapshot career totals = lookahead for historical fights).

## 3. Protocol (identical to v1)
- Data: `mdabbert/ultimate-ufc-dataset` ufc-master.csv (7,177 fights, 2010-03..2026-03, BestFightOdds closing American odds) + `rajeevw/ufcdata` strike stats (ends 2021-03).
- Walk-forward: for test year Y, train on all fights < Jan 1 Y (2020 model trained on 4,073 fights; 2026 model on 7,054). Hyperparameters tuned ONCE on pre-2020 expanding CV (folds 2017/2018/2019), then fixed for all years. Isotonic calibration per year on that year's 3-fold expanding CV OOF. Final model fit with early stopping on the chronologically last 20% of training data.
- **Primary test**: 2020-01-01..2021-03-20, full feature set incl. strike features, n=559. **Extended test**: 2020-2026, core features, n=2843.
- Betting: flat $1 on the side with max edge (p_model − de-vigged closing fair prob), gates eps ∈ {0, 0.02, 0.05}; quarter-Kelly (f=0.25·Kelly, capped 0.25) on best gate; 2,000-resample bootstrap CI on flat ROI.
- Headline ensemble (pre-committed): mean of the three calibrated models (LogReg + LGBM + XGB). `ens_gb` = mean(LGBM, XGB) reported as the stronger optional variant.

## 4. Primary test (2020-01 → 2021-03-20, n=559, walk-forward)
| model | acc | AUC | Brier |
|---|---|---|---|
| LogReg | 0.5975 | 0.5909 | 0.2362 |
| LGBM | 0.7996 | 0.8782 | 0.1533 |
| XGB | 0.7764 | 0.8750 | 0.1452 |
| **ensemble (LR+LGBM+XGB)** | **0.7746** | **0.8635** | **0.1650** |
| ens_gb (LGBM+XGB) | 0.7943 | 0.8827 | 0.1475 |
| market favorite | 0.6351 | — | — |
| market (de-vigged) | — | 0.6642 | 0.2231 |

Flat $1 (ensemble):
| gate eps | n | win rate | ROI | profit |
|---|---|---|---|---|
| 0.0 | 559 | 0.626 | **+0.3528** | +197.2 |
| 0.02 | 519 | 0.626 | +0.3783 | +196.3 |
| 0.05 | 446 | 0.635 | +0.4420 | +197.1 |

ens_gb: +0.3928 (eps0), +0.4216 (0.02), +0.4627 (0.05). Quarter-Kelly (best gate 0.05, ens_gb): $422.6M from $1,000 over 467 bets (illustrative, as v1).

Bootstrap 95% CI (2,000 resamples, ensemble): flat ROI eps0 **+0.353 [0.254, 0.459]** (v1: +0.326 [0.229, 0.423]); accuracy 0.774 [0.739, 0.809] (v1: [0.732, 0.801]).

## 5. Extended test (2020-01 → 2026-03, n=2843, walk-forward)
| model | acc | AUC | Brier |
|---|---|---|---|
| LogReg | 0.6345 | 0.6645 | 0.2249 |
| LGBM | 0.7981 | 0.8920 | 0.1276 |
| XGB | 0.7992 | 0.8872 | 0.1302 |
| **ensemble (LR+LGBM+XGB)** | **0.8020** | **0.8904** | **0.1368** |
| **ens_gb (LGBM+XGB)** | **0.8037** | **0.8923** | **0.1273** |
| market favorite | 0.6785 | — | — |
| market (de-vigged) | — | 0.7294 | 0.2072 |

Flat $1 (ensemble):
| gate eps | n | win rate | ROI | profit |
|---|---|---|---|---|
| 0.0 | 2843 | 0.704 | **+0.4032** | +1146.4 |
| 0.02 | 2697 | 0.714 | +0.4327 | +1166.9 |
| 0.05 | 2467 | 0.726 | +0.4781 | +1179.5 |

ens_gb: +0.4343 (eps0) / +0.4509 (0.02) / +0.4785 (0.05).

Per-year, flat $1 eps=0 (ensemble | ens_gb):
| year | n | acc (ens) | AUC (ens) | ROI ens | ROI ens_gb |
|---|---|---|---|---|---|
| 2020 | 451 | 0.792 | 0.877 | +0.412 | +0.445 |
| 2021 | 493 | 0.781 | 0.864 | +0.407 | +0.436 |
| 2022 | 487 | 0.809 | 0.886 | +0.367 | +0.402 |
| 2023 | 416 | 0.791 | 0.893 | +0.370 | +0.393 |
| 2024 | 383 | 0.794 | 0.901 | +0.407 | +0.446 |
| 2025 | 499 | 0.826 | 0.924 | +0.480 | +0.510 |
| 2026 | 114 | 0.868 | 0.921 | +0.276 | +0.303 |

(v1 per-year ROIs were: +0.277, +0.337, +0.286, +0.239, +0.235, +0.315, +0.070 — v2 wins every year.)
Bootstrap 95% CI (ensemble): flat ROI eps0 **+0.403 [0.360, 0.444]**; accuracy 0.802 [0.788, 0.816].

## 6. Ablation: what each upgrade bought (extended window, acc / ROI eps0 / Brier)
| config | acc | ROI | Brier |
|---|---|---|---|
| v1 (frozen 2016-19, LR+LGBM, v1 features) | 0.7710 | +0.2761 | 0.1660 |
| v2 frozen (same split, new features, LR+LGBM+XGB) | 0.7921 | +0.3743 | 0.1422 |
| **v2 walk-forward** (new features, 3-model ensemble) | **0.8020** | **+0.4032** | **0.1368** |

→ new features + XGB: +2.1pp acc, +9.8pp ROI; walk-forward refits: +1.0pp acc, +2.9pp ROI.
Per-weight-class models (with shrinkage, K tuned pre-2020): no improvement; pure pooled selected by OOF. Pre-2020 CV (LGBM): v1 core 0.5111 → v2 core 0.4420; v1 all 0.4853 → v2 all 0.4279.

## 7. Odds-augmented variant (report-only)
Adding de-vigged market probability as a feature: primary acc 0.7871 / Brier 0.1524; extended acc **0.8234** / Brier 0.1255, AUC 0.9083. Better raw accuracy than the market-free model, but by construction it reproduces the market, so betting edge collapses (~0); not a betting model. Confirms the closing line still holds information the stats model doesn't fully capture.

## 8. Verdict
**POSITIVE EDGE, IMPROVED ON V1.** The v2 walk-forward ensemble beats v1's frozen ensemble on every headline metric — accuracy 80.2% vs 77.1% (primary 77.5% vs 76.6%), flat $1 ROI +40.3% vs +27.6% (primary +35.3% vs +32.7%), Brier 0.1368 vs 0.1660 (primary 0.1650 vs 0.1730) — with positive ROI in all 7 test years, every bootstrap CI fully above zero, and tighter CIs than v1. Model Brier beats market Brier by a wide margin (0.137 vs 0.207). The LGBM+XGB pair is the recommended betting model (extended +43.4% flat, Brier 0.1273).

Caveats (same as v1): odds assumed BestFightOdds closing; best-gate selection post-hoc (eps=0 pre-registered); Kelly compounding illustrative; 2023-2024 odds coverage gaps excluded from betting rows; strike features end 2021-03 (extended uses core features); per-class hyperparameter K tuned on pre-2020 data only (no test leakage); yearly models share hyperparameters tuned once pre-2020 (retuning per year was not performed).

## 9. Artifacts (mma_v2/)
- `mma_v2_predictions_extended.csv` — 2843 rows, 2020-2026: odds, fair probs, pR per model (lr/lgb/xgb/ens/ens_gb), edges, eps0 bet side + result.
- `mma_v2_predictions_primary.csv` — 559 rows, 2020-01..2021-03-20 (full features).
- `metrics_v2_pooled.json`, `metrics_v2_perclass.json` — full numeric results.
- Code: `build_features_v2.py` (features), `v2_lib.py` (walk-forward machinery), `run_v2_pooled.py` (main backtest incl. frozen + odds variants), `run_v2_perclass.py` (per-class experiment).
- `fights_features_v2.parquet` — 180-column feature matrix.
