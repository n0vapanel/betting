# NBA Backtest Report

Backtest of `telegram_betting_bot_simple`'s real prediction engine (`PredictionEngine.get_predictions`) against historical NBA results.

- **Data source:** nba_api → stats.nba.com `LeagueGameFinder` (free, no API key)
- **Season:** 2025-26 regular season
- **Window replayed:** 2026-03-01 → 2026-03-31 (fully final results)
- **No-lookahead guarantee:** for every game, team stats (points for / points against / games played) are computed strictly from games played BEFORE that game's date. The game itself and all later games are invisible to the predictor.
- **Odds:** no odds API key available, so the odds term is the engine's neutral fallback `{home: 2.0, away: 2.0}` (contributes 50/50 under the 0.4 odds weight).

- **Games seen in window:** 239 (predicted: 239, skipped: 0)

## Summary

| Variant | Games | Correct | Accuracy |
|---------|------:|--------:|---------:|
| Engine default (0.6 stats / 0.4 odds, neutral odds) | 239 | 170 | 71.1% |
| Stats only (weight_odds = 0.0) | 239 | 170 | 71.1% |

Overall accuracy (engine default): **71.1%**
Overall accuracy (stats only): **71.1%**

Baseline for reference: NBA home teams won 60.3% of these games.

## Accuracy by tier

### Engine default (neutral odds)
| Tier | Predicted | Correct | Accuracy |
|------|----------:|--------:|---------:|
| Surefire | 0 | 0 | n/a |
| Middle | 0 | 0 | n/a |
| Hot | 239 | 170 | 71.1% |

### Stats only (weight_odds = 0.0)
| Tier | Predicted | Correct | Accuracy |
|------|----------:|--------:|---------:|
| Surefire | 0 | 0 | n/a |
| Middle | 0 | 0 | n/a |
| Hot | 239 | 170 | 71.1% |

> **Why no Middle/Surefire picks?** Both variants draw from the same pre-game stats; with neutral odds the 0.4 odds weight pulls every probability toward 0.5, and the raw stats-strength ratio never reaches the 0.6 / 0.75 tier thresholds. All picks land in the "Hot" tier, so tier-level accuracy equals overall accuracy here.

## Calibration (predicted home win probability vs actual home win rate)

### Engine default (neutral odds)
| Predicted home prob bin | Games | Mean predicted home prob | Actual home win rate |
|-------------------------|------:|-------------------------:|---------------------:|
| 0.4-0.5 | 108 | 0.491 | 38.0% |
| 0.5-0.6 | 131 | 0.508 | 78.6% |

### Stats only (weight_odds = 0.0)
| Predicted home prob bin | Games | Mean predicted home prob | Actual home win rate |
|-------------------------|------:|-------------------------:|---------------------:|
| 0.4-0.5 | 108 | 0.485 | 38.0% |
| 0.5-0.6 | 131 | 0.514 | 78.6% |

## Favorites vs underdogs

| Variant | Predicted favorite won | Underdog won | Favorite win rate |
|---------|-----------------------:|-------------:|------------------:|
| Engine default | 170 | 69 | 71.1% |
| Stats only | 170 | 69 | 71.1% |

## Games skipped

None.

## Reproducing

```bash
.venv/bin/pip install -r backtest/requirements.txt
.venv/bin/python backtest/backtest_runner.py
```

Raw game logs are cached in `backtest/cache/` (gitignored); delete the cache to force a re-fetch.

## Output

`backtest/predictions.csv` — one row per predicted game: `date, home, away, predicted_home_prob, predicted_winner, tier, actual_winner, correct` plus the stats-only variant columns (`stats_only_predicted_home_prob`, `stats_only_predicted_winner`, `stats_only_tier`, `stats_only_correct`).
