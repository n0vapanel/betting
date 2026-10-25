"""Evaluation protocol for the 3-month extension: flat $1 at closing ML, edge
gates, date-clustered bootstrap CIs, Brier vs market. Reuses the v3 functions
and adapts gate selection to the December 24-31 validation block.
"""

from __future__ import annotations

import numpy as np

from backtest.v3.evaluation_v3 import (
    EPS_GRID, N_BOOT, SEED,
    bet_frame, bet_metrics, evaluate, gate_table, hindsight_best,
    quarter_kelly, bootstrap_intervals,
)

VALIDATION_PERIOD = "2021-12-24..2021-12-31"
MINIMUM_VALIDATION_BETS = 20


def choose_gate(validation):
    candidates = gate_table(validation, validation.p_model_home.to_numpy())
    eligible = [c for c in candidates if c["n_bets"] >= MINIMUM_VALIDATION_BETS]
    if eligible:
        best = min(eligible, key=lambda c: (-c["roi"], -c["n_bets"], c["eps"], c["variant"]))
        return {**best, "selection_period": VALIDATION_PERIOD,
                "minimum_bets": MINIMUM_VALIDATION_BETS, "fallback": False}, candidates
    fallback = next(c for c in candidates if c["variant"] == "winner" and c["eps"] == 0)
    return {**fallback, "selection_period": VALIDATION_PERIOD,
            "minimum_bets": MINIMUM_VALIDATION_BETS, "fallback": True}, candidates
