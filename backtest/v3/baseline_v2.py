from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.v2 import data as D
from backtest.v2 import features as F
from backtest.v2 import model as M


HERE = Path(__file__).resolve().parent
SELECTED = [
    "off_eFG_home", "tov_home", "elo_diff", "b2b_home", "inj_home", "inj_diff",
    "market_home", "winpct_diff", "margin_diff",
]


def fingerprints(paths) -> dict:
    result = {}
    for path in paths:
        path = Path(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[str(path)] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return result


def reproduce_v2() -> dict:
    sources = [D.TEAMLOG_CACHE, D.PLAYERLOG_CACHE, D.INJURY_CACHE, D.KAGGLE_CSV,
               Path(F.__file__), Path(M.__file__), Path(D.__file__), Path(__file__)]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Read-only v2 reproduction requires existing cache: {source}")
    signature = fingerprints(sources)
    output = HERE / "v2_reproduction.json"
    predictions = HERE / "v2_reproduced_predictions.csv"
    if output.exists() and predictions.exists():
        saved = json.loads(output.read_text())
        if saved["input_fingerprints"] == signature:
            if not saved["sanity_gate_passed"]:
                raise AssertionError("Cached v2 reproduction failed the baseline sanity gate")
            print("[v2] Reusing fingerprint-verified reproduction", flush=True)
            return saved
    built = F.build_features(verbose=True)
    train, test = built["train"], built["test"]
    selection = {
        "best_kind": "lr", "lr_best_C": 0.05, "selected_cols": SELECTED,
        "hgb_best_params": {"max_iter": 50, "learning_rate": 0.03, "max_leaf_nodes": 7},
    }
    fitted = M.train_and_predict(
        train[F.FEATURE_NAMES].astype(float), train.home_won.astype(int), train.date,
        test[F.FEATURE_NAMES].astype(float), selection, verbose=True,
    )
    p = fitted["test_probs"]
    y = test.home_won.to_numpy(dtype=bool)
    predicted_home = p > 0.5
    correct = predicted_home == y
    price = np.where(predicted_home, test.dec_home, test.dec_away)
    pnl = np.where(correct, price - 1, -1)
    accuracy = float(correct.mean())
    roi = float(pnl.mean())
    model_brier = float(np.mean((p - y) ** 2))
    market_brier = float(np.mean((test.p_market_home.to_numpy() - y) ** 2))
    passed = (
        len(train) == 843 and len(test) == 229
        and round(accuracy * 100, 1) == 68.1
        and round(roi * 100, 2) == 0.10
        and round(model_brier, 4) == 0.2131
        and round(market_brier, 4) == 0.2010
    )
    result = {
        "n_train": len(train), "n_test": len(test), "accuracy": accuracy, "flat_roi": roi,
        "model_brier": model_brier, "market_brier": market_brier, "pnl": float(pnl.sum()),
        "selected_features": SELECTED, "model": "LogReg C=0.05", "calibration": fitted["calibrator"],
        "method": "Refit the frozen published v2 specification; no v2 hyperparameter search is rerun.",
        "known_limitations": [
            "v2 chooses calibration by in-sample February calibration loss; preserved for exact reproduction only.",
            "v2's 1PM early-tip routing can use a report at tip for a 1PM game; v3 requires strict snapshot < tip.",
            "v2 ORB denominator uses opposing OREB instead of opposing DREB; v3 corrects this.",
            "v2 injury minutes are double-lagged and season-to-date denominators are off by one; v3 corrects both.",
        ],
        "sanity_gate_passed": passed, "input_fingerprints": signature,
        "predictions_path": str(predictions),
    }
    frame = test[["date", "home", "away", "home_won", "dec_home", "dec_away", "p_market_home"]].copy()
    frame["p_model_home"] = p
    frame["correct"] = correct
    frame["pnl_1usd"] = pnl
    frame.to_csv(predictions, index=False)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"[v2] accuracy={accuracy:.6%}, ROI={roi:+.6%}, "
          f"Brier={model_brier:.6f}, market={market_brier:.6f}; gate={passed}", flush=True)
    if not passed:
        raise AssertionError(f"v2 sanity gate failed; inspect {output}")
    return result
