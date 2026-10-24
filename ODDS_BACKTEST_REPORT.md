# Odds-Fed Backtest Report (real closing moneylines)

- **Window:** 2022-03-01 → 2022-03-31 (2021-22 season, fully final)
- **Games:** 229 (all with closing moneyline, zero missing)
- **Odds source:** Kaggle 'NBA Betting Data 2007-2026' (closing lines, SportsbookReview/ESPN)
- **Engine:** real `PredictionEngine.get_predictions` with real h2h moneylines (0.6 stats / 0.4 odds)
- **Stats:** no lookahead — strictly pre-game, prefix-array provider (same as neutral-odds harness)
- **P&L:** flat $1 staked on every predicted winner at its closing moneyline

## Headline

- Accuracy: **67.2%** (229 games)
- Flat-bet P&L: **-1.87** over $229 staked
- ROI: **-0.8%**

## Baselines (same 229 games, same staking)

- Always bet home: P&L -13.48 → ROI -5.9%
- Always bet closing favorite: P&L -9.40 → ROI -4.1%

## By tier

| Tier | Bets | Accuracy | P&L | ROI |
|------|-----:|---------:|----:|----:|
| Surefire | 0 | n/a | n/a | n/a |
| Middle | 78 | 74.4% | -7.44 | -9.5% |
| Hot | 151 | 63.6% | +5.57 | +3.7% |

## Calibration (predicted home win prob vs actual)

| Predicted home prob bin | Games | Mean predicted | Actual home win rate |
|-------------------------|------:|--------------:|---------------------:|
| 0.3-0.4 | 29 | 0.362 | 27.6% |
| 0.4-0.5 | 62 | 0.449 | 37.1% |
| 0.5-0.6 | 89 | 0.553 | 64.0% |
| 0.6-0.7 | 49 | 0.636 | 75.5% |

## Reproducing

```bash
.venv/bin/python backtest/odds_backtest_runner.py
```

`backtest/odds_predictions.csv` holds every game row.
