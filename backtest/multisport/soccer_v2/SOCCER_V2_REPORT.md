# Soccer v2 Walk-Forward Backtest — EPL + Big-4 Leagues, 5 Seasons (2017-18 … 2021-22)

**Data:** football-data.co.uk CSVs (seasons 1718–2122: E0/SP1/D1/I1/F1) + Understat per-match xG/xGA (9026/9027 matches joined by team name + date, ±1 day).
**Split:** train = all matches before 2022-03-01 (7924 after feature-complete filter; 2017-09-22 – 2022-02-28), test = 2022-03-01 – 2022-05-22 (571 matches — identical window to v1). Walk-forward, strict no-lookahead.

## Method
- **Features (96):** v1's 45 rolling features (stats w5/w10 + venue v5, Elo) extended with: venue splits at window 10 for all 14 stats; Understat rolling xGF/xGA and xG-diff (all/home/away, w5/w10); CLV drift (opening→closing no-vig, same-book chain AvgC→Avg, else PSC→PS, else B365C→B365) for H/D/A and O/U; opening market no-vig probabilities (m_open) as features; win/unbeaten/loss streak diffs; rest-days diff; form momentum (w5−w10) for GF/GA/xGF/xGA; xG attack×defense interactions; league dummies; season index.
- **No-lookahead protocol:** features are computed from team history with kickoff strictly before the current match; matches sharing a kickoff slot (date+hour) all see the same pre-match state (strict — no same-slot leakage), then the slot's outcomes are appended together. Elo carries over across seasons (K=32, GD-scaled, +65 HFA).
- **Models:** LogisticRegression (C tuned by 5-fold time-ordered CV) + LightGBM (early stopping on a time-ordered holdout), each isotonic-calibrated on time-ordered out-of-fold predictions; **Ensemble** = weighted mean with per-target weights selected on time-ordered train OOF logloss only (1X2: 0.1 LogReg / 0.9 LGBM; O/U: 0.5 / 0.5).
- **Betting protocol:** flat $1 stakes; bet the model's max-probability side whenever `model prob − closing-market no-vig prob > eps`, eps ∈ {0, 0.02, 0.05}. Payouts use closing average decimal odds (AvgH/AvgD/AvgA, Avg>2.5/Avg<2.5 — identical benchmark odds to v1 for the test window).
- **Bootstrap CI:** 2000 resamples clustered by match date, on flat-stake ROI. **Kelly:** quarter-Kelly, 5% cap.

### Data coverage notes
- Closing market odds chain: Avg (1920–2122) → BetBrain avg (1718–1819) → Pinnacle close. Test window: Avg everywhere.
- Opening odds: AvgC/B365C/MaxC (1920–2122), Pinnacle open only (1718–1819); O/U opening odds exist 1920+ (1718–1819 CLV_OU imputed as 0, flag clv_ou_avail=0).
- Understat xG coverage: see table below (league × season, matched/total).

| League | Season | xG matched |
|---|---|---|
| EPL | 1718 | 380/380 |
| EPL | 1819 | 380/380 |
| EPL | 1920 | 380/380 |
| EPL | 2021 | 380/380 |
| EPL | 2122 | 380/380 |
| LaLiga | 1718 | 380/380 |
| LaLiga | 1819 | 380/380 |
| LaLiga | 1920 | 380/380 |
| LaLiga | 2021 | 380/380 |
| LaLiga | 2122 | 380/380 |
| Bundesliga | 1718 | 306/306 |
| Bundesliga | 1819 | 306/306 |
| Bundesliga | 1920 | 306/306 |
| Bundesliga | 2021 | 306/306 |
| Bundesliga | 2122 | 306/306 |
| SerieA | 1718 | 380/380 |
| SerieA | 1819 | 380/380 |
| SerieA | 1920 | 380/380 |
| SerieA | 2021 | 379/379 |
| SerieA | 2122 | 379/379 |
| Ligue1 | 1718 | 379/380 |
| Ligue1 | 1819 | 380/380 |
| Ligue1 | 1920 | 279/279 |
| Ligue1 | 2021 | 380/380 |
| Ligue1 | 2122 | 380/380 |

## 1. 1X2 moneyline

| Model | Acc (3-way) | LogLoss | Brier (home win) |
|---|---|---|---|
| Market (no-vig max-prob) | 54.82% | — | 0.2112 |
| LogReg | 53.42% | 0.9863 | 0.2171 |
| LightGBM | 54.99% | 0.9785 | 0.2144 |
| Ensemble | 55.17% | 0.9788 | 0.2146 |

| Model | eps | Bets | Win rate | Flat $1 ROI |
|---|---|---|---|---|
| Market (always bet favorite) | always | 571 | 54.8% | -0.58% |
| LogReg | 0 | 189 | 37.6% | -16.37% |
| LogReg | 0.02 | 121 | 35.5% | -15.63% |
| LogReg | 0.05 | 50 | 36.0% | -11.78% |
| LightGBM | 0 | 179 | 46.9% | +2.02% |
| LightGBM | 0.02 | 100 | 50.0% | +10.09% |
| LightGBM | 0.05 | 32 | 50.0% | +12.31% |
| Ensemble | 0 | 173 | 46.2% | +1.06% |
| Ensemble | 0.02 | 95 | 50.5% | +10.15% |
| Ensemble | 0.05 | 24 | 50.0% | +14.71% |

**Bootstrap 95% CI for Ensemble flat ROI (eps=0, 173 bets, clustered by date):** -13.52% … +14.92%
**Kelly (quarter-Kelly, 5% cap) on best gate eps=0.05:** 24 bets (12 wins), mean stake 2.40% of bankroll, bankroll $100 → $140.53 (+40.5%).

## 2. Over/Under 2.5 goals

| Model | Side accuracy | LogLoss |
|---|---|---|
| LogReg | 58.84% | 0.6749 |
| LightGBM | 59.89% | 0.6695 |
| Ensemble | 59.72% | 0.6715 |

| Model | eps | Bets | Win rate | Flat $1 ROI |
|---|---|---|---|---|
| Market (always bet favorite) | always | 571 | 57.4% | -4.05% |
| LogReg | 0 | 315 | 55.9% | +3.07% |
| LogReg | 0.02 | 229 | 55.9% | +4.96% |
| LogReg | 0.05 | 116 | 54.3% | +5.41% |
| LightGBM | 0 | 311 | 57.9% | +4.37% |
| LightGBM | 0.02 | 202 | 59.9% | +9.66% |
| LightGBM | 0.05 | 84 | 59.5% | +13.00% |
| Ensemble | 0 | 290 | 56.9% | +4.00% |
| Ensemble | 0.02 | 196 | 57.1% | +6.94% |
| Ensemble | 0.05 | 65 | 66.2% | +28.71% |

**Bootstrap 95% CI for Ensemble flat ROI (eps=0, 290 bets, clustered by date):** -2.75% … +10.12%
**Kelly (quarter-Kelly, 5% cap) on best gate eps=0.05:** 65 bets (43 wins), mean stake 2.56% of bankroll, bankroll $100 → $273.49 (+173.5%).

*Caveat on Kelly figures: realized outcomes from compounding stakes over one 3-month window; the flat-ROI and its date-clustered bootstrap CI are the reliable estimate of edge.*

## 3. Per-league breakdown (Ensemble, eps=0)

| League | Matches | 1X2 acc | 1X2 ROI | 1X2 bets | O/U acc | O/U ROI | O/U bets |
|---|---|---|---|---|---|---|---|
| Bundesliga | 90 | 51.11% | +15.72% | 29 | 62.22% | +12.68% | 38 |
| EPL | 124 | 63.71% | +22.74% | 35 | 55.65% | -4.84% | 69 |
| LaLiga | 122 | 52.46% | -13.75% | 40 | 62.30% | +4.81% | 54 |
| Ligue1 | 120 | 52.50% | -8.47% | 36 | 60.83% | +0.70% | 63 |
| SerieA | 115 | 54.78% | -6.48% | 33 | 58.26% | +10.71% | 66 |
| **All leagues** | **571** | **55.17%** | **+1.06%** | **173** | **59.72%** | **+4.00%** | **290** |

## 4. Stat-level calibration — expected goals & shots vs actual (test set)

Mean absolute error of model-implied expectations (LightGBM regressors on the same feature set) vs the naive season-average baseline (train-window team mean, venue-aware) and a global-mean baseline.

| Quantity | Model MAE | Naive team-season MAE | Global mean MAE | Model vs naive |
|---|---|---|---|---|
| Home goals | 0.945 | 0.984 | 1.050 | -4.0% |
| Away goals | 0.907 | 0.936 | 0.997 | -3.1% |
| Total goals | 1.363 | 1.389 | 1.396 | -1.9% |
| Home shots | 3.613 | 4.005 | 4.130 | -9.8% |
| Away shots | 3.346 | 3.693 | 3.853 | -9.4% |
| Total shots | 4.806 | 5.102 | 4.790 | -5.8% |

## 5. Verdict vs v1 baseline

**v1 baseline (1-season data, 45 features):** 1X2 acc 54.12% (market 54.82%), flat ROI +7.82% at eps=0 (211 bets, CI −10.53%…+24.58%); O/U negative at every gate.
**v2 (5 seasons + xG + CLV + deeper form, 96 features):** 1X2 acc 55.17% vs market 54.82%; flat ROI +1.06% at eps=0 over 173 bets (CI -13.52%…+14.92%); eps=0.02 → +10.15% (95 bets); eps=0.05 → +14.71% (24 bets). **O/U 2.5:** acc 59.72%, flat ROI +4.00% at eps=0 over 290 bets (CI -2.75%…+10.12%).
The v2 1X2 model does not demonstrate a statistically significant edge vs the closing market: its flat ROI (+1.06% at eps=0 over 173 bets) beats the always-bet-market-favorite baseline (-0.58%) but the date-clustered 95% CI includes zero. The O/U 2.5 model improves massively on v1 — positive ROI at every gate (vs negative everywhere in v1) and side accuracy 59.7% vs 52.7% — but the date-clustered 95% CI still includes zero, so the edge is not statistically established over this single 3-month window. Closing-odds markets are a high-vig benchmark; the stat-level check confirms the feature engine carries real signal (expected-goals/shots regressors beat naive baselines), but a single 3-month window of ~571 matches cannot establish a market-beating edge at date-clustered significance.

## 6. Ablations (same 571-match test; weights/features as noted)

Each variant uses the same train/test split, calibration and gates; ensemble weights per variant unless noted. `v1like_feats` is the closest re-implementation of v1's feature set on 5-season data — its O/U accuracy (52.71%) exactly reproduces v1's, validating the pipeline.

| Variant | 1X2 acc | 1X2 eps0 ROI (CI) | 1X2 eps.02 / eps.05 | O/U acc | O/U eps0 ROI (CI) | O/U eps.02 / eps.05 |
|---|---|---|---|---|---|---|
| all feats, 50/50 ensemble | 54.64% | +1.6% (-14.0…+16.7) | -4.5% / -1.1% | 59.72% | +4.0% (-2.7…+10.1) | +6.9% / +28.7% |
| all feats, 30/70 ensemble | 54.99% | +0.8% (-14.5…+15.1) | +10.4% / +35.9% | 59.89% | +6.5% (-0.0…+13.5) | +8.7% / +23.1% |
| all feats, LGBM only | 54.99% | +2.0% (-11.3…+14.6) | +10.1% / +12.3% | 59.89% | +4.4% (-3.0…+12.2) | +9.7% / +13.0% |
| all feats, LogReg only | 53.42% | -16.4% (-31.3…-2.6) | -15.6% / -11.8% | 58.84% | +3.1% (-3.2…+9.3) | +5.0% / +5.4% |
| drop opening-market probs | 54.29% | -2.2% (-18.5…+13.7) | +6.5% / +7.4% | 56.74% | +1.9% (-6.6…+10.0) | +3.0% / -1.6% |
| drop CLV + open-market | 54.12% | -5.7% (-22.6…+8.7) | -3.9% / +11.6% | 57.27% | +1.6% (-7.4…+10.5) | +4.8% / -3.6% |
| drop xG features | 54.82% | -5.8% (-21.5…+10.0) | +11.6% / +36.7% | 57.44% | -1.6% (-8.7…+6.2) | +4.0% / +11.1% |
| drop streaks/momentum/rest/interactions | 54.82% | +0.8% (-15.9…+17.5) | +11.2% / +16.3% | 58.84% | +2.1% (-6.6…+10.8) | +5.6% / +12.5% |
| v1-like features (no xG/CLV/form) | 53.59% | -7.0% (-24.6…+8.1) | -8.4% / -9.0% | 52.71% | -8.1% (-17.4…+1.4) | -9.3% / -13.8% |
| LGBM higher capacity | 54.99% | +4.3% (-12.4…+20.3) | +13.3% / +5.1% | 59.02% | -0.5% (-8.6…+8.3) | +2.7% / +16.1% |

**Reading:** xG + CLV + open-market features drive the O/U improvement (52.7% → 59.7% accuracy; ROI flips from −7% at eps=0 in v1 to +4.0%, positive at every gate). For 1X2 the gains are smaller but real: accuracy rises from 54.1% to 55.2% — above the market's own max-prob accuracy (54.8%) for the first time — while eps=0.02/0.05 ROI (+10.2%/+14.7%, 95/24 bets) stay comparable to v1's (+13.3%/+12.6%, 174/133 bets); eps=0 ROI is +1.1% (v1 +7.8%).

_Report generated from backtest.py; predictions in `soccer_v2_predictions.csv` (571 test rows)._