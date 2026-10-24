# MULTI-SPORT PREDICTION BRAIN — Backtest Board (v1)

Generated 2026-10-24. Every sport: strict walk-forward (train strictly before test window), no-lookahead rolling features, settled at the book's CLOSING odds, flat-$1 stakes, edge gates on model-vs-devigged-market probability, bootstrap CIs clustered by date. Numbers below are independently re-verified from the per-game prediction CSVs by the coordinator (not just agent self-reports).

## Scoreboard (verified)

| Sport | Test window | n | Pick acc | Model flat ROI | Market-fav flat | Verdict |
|---|---|---|---|---|---|---|
| **MMA/UFC** | 2020-2026 | 2,843 | 77.1% (fav: 67.4%) | **+27.6%** (every year +) | ~0% | **REAL EDGE (candidate)** |
| **NBA** (v3) | Mar 2022 | 229 | 65.5% | -4.52% | -0.8% | market efficient |
| **NHL** | Mar-Jun 2022 | 555 | 59.1% | -0.26% | ~-0.5% | noise |
| **Soccer** (EPL+4) 1X2 | Mar-May 2022 | 571 | 54.1% | +7.82% (CI -10.5..+24.6) | -0.58% | not significant |
| **MLB** | May-Jun 2022 | 824 | 54.9% | -4.20% | ~-2% | market efficient |
| **Tennis ATP** | 2022 | 2,538 | 67.8% | -6.18% | ~-2% | market efficient |
| **NFL** | 2021 w15-2022 | running (Devin) | — | — | — | pending |

## The one that matters: MMA

- Source: Kaggle `mdabbert/ultimate-ufc-dataset` (7,177 fights → 2026) + ufcstats strike data merged from `rajeevw/ufcdata` (primary window only).
- Model: calibrated LogReg + LightGBM ensemble, trained ONCE on 2016-2019, frozen, then bet 2020-2026 (2,843 fights).
- Edge vs market: pick accuracy 77.1% vs favorite 67.4%; model Brier 0.166 vs market 0.207; positive flat ROI in **every single year** (2020 +27.7 / 2021 +33.7 / 2022 +28.6 / 2023 +23.9 / 2024 +23.5 / 2025 +31.5 / 2026 +7.0).
- Bootstrap 95% CI on primary window ROI: [+22.9%, +42.3%] — entirely above zero.
- Lookahead audit (coordinator): strike features use prior-fights-only aggregates (verified `prev_*` lags in build_features.py); Elo updated per-fight in strict time order; recompute of both CSVs from raw odds columns matches exactly (+32.7% / +27.6%).
- Residual risks (flagged honestly in MMA report): odds are BestFightOdds "closing" per dataset docs — transcription errors would inflate ROI; not true last-second sharps closers. Next step = live paper-trade on current cards to validate before real money.

## The honest rest

NBA/NHL/MLB/tennis closing lines are efficient: no model here beats them after vig (flat ROIs all negative, gates not significant after bootstrap). Soccer 1X2 shows +7.8% flat but the date-clustered CI straddles zero and it's EPL-driven (41 bets) — treat as noise. Soccer's stat engine is still real: expected-goals/shots regressors beat naive baselines on every quantity (shots MAE -2.2%, goals -1.4% vs team-season baseline) — the user-requested stats-vs-actual comparison.

## Structure

- `backtest/multisport/<sport>/` — each: `build_features.py`, `run_backtest.py`-style runner, `<SPORT>_BACKTEST_REPORT.md`, `<sport>_predictions.csv` (full per-game: odds, probs, edge, bet, result).
- Shared protocol is the NBA v3 harness convention; venv `~/.hermes/cache/scratch/msports-venv` (py3.11: nfl_data_py, pybaseball, kaggle, lightgbm, xgboost).
- Raw datasets gitignored (re-downloadable: football-data.co.uk CSVs, Retrosheet zips, moneypuck, Kaggle slugs documented per report).

## Next moves

1. NFL lane lands → add row, adjudicate.
2. MMA: build live card picker (pull tonight's card from BestFightOdds/UFC API, run frozen model, paper-trade).
3. Cross-model audit of the MMA pipeline (the multi-model lanes: Claude/Atria reviews once provider config unblocked).
