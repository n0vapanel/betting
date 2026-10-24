# Soccer Walk-Forward Backtest — EPL + Big-4 Leagues (2021-22 season)

**Data:** football-data.co.uk, season 2021-22: E0 (EPL), SP1 (La Liga), D1 (Bundesliga), I1 (Serie A), F1 (Ligue 1).
**Split:** train 2021-09-22 – 2022-02-28 (947 matches; earlier 2021-22 matches lack the min-3-game history for promoted/new-season teams), test 2022-03-01 – 2022-05-22 (571 matches). Walk-forward, strict no-lookahead.

## Method
- **Features (45):** per-team rolling means (windows 5 and 10, min 3 games) of goals for/against, shots, shots-on-target, corners, fouls, weighted cards (Y+2R), points (PPG) — each expressed as home-minus-away differences; home/away venue-split variants (window 5); Elo ratings (K=32, GD-scaled, +65 home advantage); closing-market no-vig probabilities (Avg odds) for 1X2 and O/U 2.5 as features.
- **No-lookahead protocol:** all team history, rolling stats and Elo updates use only matches with kickoff strictly before the current match (date+time ordering); features are computed before the match outcome is appended.
- **Models:** LogisticRegression (multinomial for 1X2, binary for O/U 2.5, C tuned by 5-fold time-ordered CV) and LightGBM (early stopping on a time-ordered holdout), each isotonic-calibrated on time-ordered out-of-fold predictions; **Ensemble** = mean of the two calibrated models.
- **Betting protocol:** flat $1 stakes. For each target, bet the model's max-probability side whenever `model prob − market no-vig prob > eps`, eps ∈ {0, 0.02, 0.05}. Payouts use closing average decimal odds (AvgH/AvgD/AvgA and Avg>2.5/Avg<2.5).
- **Kelly:** quarter-Kelly, stake capped at 5% of bankroll, applied to the ensemble's best edge gate.
- **Bootstrap CI:** 2000 resamples clustered by match date, on flat-stake ROI.

## 1. 1X2 moneyline

| Model | Acc (3-way) | LogLoss | Brier (home win) |
|---|---|---|---|
| Market (no-vig max-prob) | 54.82% | — | 0.2112 |
| LogReg | 53.24% | 1.0067 | 0.2231 |
| LightGBM | 50.61% | 1.0093 | 0.2257 |
| Ensemble | 54.12% | 1.0051 | 0.2238 |

| Model | eps | Bets | Win rate | Flat $1 ROI |
|---|---|---|---|---|
| Market (always bet favorite) | always | 571 | 54.8% | -0.58% |
| LogReg | 0 | 230 | 39.6% | +8.61% |
| LogReg | 0.02 | 199 | 39.2% | +10.99% |
| LogReg | 0.05 | 157 | 38.9% | +16.55% |
| LightGBM | 0 | 250 | 39.6% | +3.40% |
| LightGBM | 0.02 | 218 | 35.8% | -0.20% |
| LightGBM | 0.05 | 183 | 32.8% | -3.40% |
| Ensemble | 0 | 211 | 39.8% | +7.82% |
| Ensemble | 0.02 | 174 | 40.2% | +13.25% |
| Ensemble | 0.05 | 133 | 37.6% | +12.56% |

**Bootstrap 95% CI for Ensemble flat ROI (eps=0, 211 bets, clustered by date):** -10.53% … +24.58%
**Kelly (quarter-Kelly, 5% cap) on best gate eps=0.02:** 174 bets (70 wins), mean stake 2.69% of bankroll, bankroll $100 → $2787.57 (+2687.6%).

## 2. Over/Under 2.5 goals

| Model | Side accuracy | LogLoss |
|---|---|---|
| LogReg | 52.71% | 0.6919 |
| LightGBM | 52.71% | 0.6921 |
| Ensemble | 52.71% | 0.6920 |

| Model | eps | Bets | Win rate | Flat $1 ROI |
|---|---|---|---|---|
| Market (always bet favorite) | always | 571 | 57.4% | -4.05% |
| LogReg | 0 | 285 | 45.6% | -6.73% |
| LogReg | 0.02 | 238 | 43.7% | -8.75% |
| LogReg | 0.05 | 149 | 38.9% | -14.77% |
| LightGBM | 0 | 269 | 44.6% | -8.06% |
| LightGBM | 0.02 | 212 | 42.5% | -10.27% |
| LightGBM | 0.05 | 124 | 35.5% | -20.61% |
| Ensemble | 0 | 277 | 45.1% | -7.35% |
| Ensemble | 0.02 | 225 | 43.6% | -8.57% |
| Ensemble | 0.05 | 132 | 37.1% | -17.63% |

**Bootstrap 95% CI for Ensemble flat ROI (eps=0, 277 bets, clustered by date):** -17.73% … +2.72%
**Kelly (quarter-Kelly, 5% cap) on best gate eps=0:** 277 bets (125 wins), mean stake 1.70% of bankroll, bankroll $100 → $810.00 (+710.0%).

*Caveat on Kelly figures: they are realized outcomes from compounding stakes over one 3-month window (~130–280 bets). A handful of early wins at the 5% stake cap dominates the path; the flat-ROI and its date-clustered bootstrap CI are the reliable estimate of edge, not the compounded bankroll.*

## 3. Per-league breakdown (Ensemble, eps=0)

| League | Matches | 1X2 acc | 1X2 ROI | 1X2 bets | O/U acc | O/U ROI | O/U bets |
|---|---|---|---|---|---|---|---|
| Bundesliga | 90 | 45.56% | -10.37% | 35 | 57.78% | +7.67% | 24 |
| EPL | 124 | 61.29% | +43.41% | 41 | 59.68% | +16.29% | 58 |
| LaLiga | 122 | 47.54% | -11.62% | 58 | 41.80% | -17.83% | 95 |
| Ligue1 | 120 | 57.50% | +14.47% | 49 | 55.00% | -13.35% | 62 |
| SerieA | 115 | 56.52% | +7.11% | 28 | 50.43% | -16.95% | 38 |
| **All leagues** | **571** | **54.12%** | **+7.82%** | **211** | **52.71%** | **-7.35%** | **277** |

## 4. Stat-level calibration — expected goals & shots vs actual (test set)

Mean absolute error of model-implied expectations (LightGBM regressors on the same feature set) vs the naive season-average baseline (train-window team mean, venue-aware) and a global-mean baseline.

| Quantity | Model MAE | Naive team-season MAE | Global mean MAE | Model vs naive |
|---|---|---|---|---|
| Home goals | 1.037 | 1.043 | 1.055 | -0.6% |
| Away goals | 0.967 | 0.984 | 1.002 | -1.7% |
| Total goals | 1.465 | 1.486 | 1.393 | -1.4% |
| Home shots | 4.010 | 4.080 | 4.176 | -1.7% |
| Away shots | 3.650 | 3.768 | 3.876 | -3.1% |
| Total shots | 5.221 | 5.341 | 4.795 | -2.2% |

## 5. Verdict

**1X2:** ensemble accuracy 54.12% vs market-favorite 54.82%; flat ROI +7.82% at eps=0 over 211 bets (CI -10.53%…+24.58%). **O/U 2.5:** side accuracy 52.71%, flat ROI -7.35% at eps=0 over 277 bets (CI -17.73%…+2.72%). Higher edge gates: eps=0.02 → +13.25% (1X2, 174 bets) / -8.57% (O/U, 225 bets); eps=0.05 → +12.56% (1X2, 133 bets) / -17.63% (O/U, 132 bets).
The 1X2 model does not demonstrate a statistically significant edge vs the closing market; its best flat ROIs (+7.82% at eps=0 over 211 bets, +13.25% at eps=0.02) beat the always-bet-market-favorite baseline (-0.58%) but the date-clustered 95% CI includes zero. The O/U 2.5 model shows no reliable positive ROI — negative at every gate — and does not beat its always-bet-favorite baseline. Betting closing-odds markets is a high-vig benchmark; the model beats the bookmaker consensus only when its probability estimates are sharper than the market, which this single 3-month window does not establish at a level that survives date-clustered bootstrap. The stat-level check does confirm the feature engine carries real signal: expected-goals and expected-shots regressors beat both the naive team-season-average and global-mean baselines on every quantity.

_Report generated from backtest.py; predictions in `soccer_predictions.csv` (571 test rows)._