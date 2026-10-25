# Reproduce the NFL walk-forward backtest

All source, caches and outputs live in this directory. No git command, bookmaker account, API key or paid data is required.

## Run

Use the specified Python 3.11 interpreter:

```sh
cd "/Users/x/Desktop/x/Coding Projects/betting bot/backtest/multisport/nfl"
~/.hermes/cache/scratch/msports-venv/bin/python -B nfl_backtest.py
```

On the first run, missing local parquet is imported using nfl_data_py. Subsequent runs reuse `.cache/`. To prohibit network downloads:

```sh
~/.hermes/cache/scratch/msports-venv/bin/python -B nfl_backtest.py --offline
```

The CLI exits nonzero on missing required data, failed leakage assertions or invalid predictions. It prints game count, preselected-model accuracy/flat ROI, preselected best gate/ROI and output paths. It does not place bets.

## Verify

```sh
~/.hermes/cache/scratch/msports-venv/bin/python -B -m unittest -v test_nfl_backtest
~/.hermes/cache/scratch/msports-venv/bin/python -B nfl_backtest.py --verify-only
```

The tests cover odds/vig, push and ATS-sign settlement, snap-weighted injuries, lagged QB/depth, same/future-week mutation invariance, injury timestamp filtering, chronological CV/calibration, gate semantics, week-clustered bootstrap, price-aware simultaneous-week Kelly and tie scoring.

To regenerate only the two Markdown documents from frozen measured results, without downloading data, recalculating features, refitting or rescoring:

```sh
~/.hermes/cache/scratch/msports-venv/bin/python -B nfl_backtest.py --render-only
```

## Outputs

- `nfl_predictions.csv`: 335 completed test games, one row per game; both raw/calibrated model probabilities, no-vig market probability, source/cutoff audit fields, home/away feature values, all gate selections/unit profits and the preselected gate's quarter-Kelly path.
- `NFL_BACKTEST_REPORT.md`: formulas, source coverage, chronological model/gate selection, prediction/betting metrics, uncertainty and verdict.
- `README_nfl.md`: this reproduction guide.
- `.cache/evaluation.json`: frozen measured results, complete feature list, CV/fit audit, versions and source/prediction hashes.
- `.cache/features.parquet`: all lagged training/test feature rows.
- `.cache/validation_predictions.csv`: pre-test 2021 weeks 11–14 predictions used for model/gate selection.
- `.cache/closing_line_validation.csv`: schedule/PBP closing-spread and pregame-WP audit.

In the CSV, `result` is home score minus away score; positive `spread_line` means home is favoured. `primary_probability` is the preselected model's calibrated home-win probability. A gate's `_bet` is 0/1, `_side` is a team abbreviation or blank and `_profit` is net dollars from a $1 risk. `selected_gate_*` is frozen before test outcomes; `kelly_*` uses actual offered-price EV and start-of-week bankroll. `home_win_outcome=0.5` marks a tie, which is excluded from binary scoring but pushes a two-way moneyline bet. Current-game outcomes/odds/QB metadata are exported for auditing/settlement, never included in the explicit model feature list.

## Dependencies and scope

Runtime used Python 3.11.15 and:

```text
nfl_data_py==0.3.3
pandas==2.2.3
numpy==2.4.6
scipy==1.17.1
scikit-learn==1.9.1
lightgbm==4.7.0
pyarrow==26.0.0
```

All were already installed in the supplied venv; no packages were installed. Use `-B` to avoid Python bytecode outside this directory. CLI-managed temporary/cache locations are scoped inside this directory. This implementation changes no shared venv, repository configuration, other sport directory or git state.

The final evaluation uses closing moneyline; consult the report for complete assumptions. Historical injury/depth/NGS archives are not certified point-in-time snapshots. No claim of profitable deployment, incremental injury-feature benefit or future performance follows from these results.
