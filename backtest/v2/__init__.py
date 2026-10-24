"""v2 injury-aware walk-forward NBA backtest package.

Modules:
- data.py     : fetchers + caches (team/player logs, Kaggle moneylines,
                 official injury reports via nbainjuries).
- features.py : no-lookahead feature builders.
- model.py    : time-ordered CV model selection + isotonic calibration.
- run_v2.py   : CLI that runs the whole backtest and writes deliverables.
"""
