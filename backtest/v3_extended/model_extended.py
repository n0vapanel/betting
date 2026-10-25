"""Model selection + fitting for the 3-month extension.

Same registered methodology as v3 (greedy backward LogReg C=0.05 selection,
LogReg/LightGBM/XGBoost family grids, top-two soft vote, optional no-vig market
shrinkage, chronological calibration, final refit on all training rows), with
the extended walk-forward windows:

  training       2021-10-19 .. 2021-12-31   (frozen before Jan 1)
  development    < 2021-12-16               (5 date-grouped expanding folds)
  December holdout 2021-12-16 .. 2021-12-31 (calibration fit 16-23, choice 24-31)

Nothing after 2021-12-31 enters any selection decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from backtest.v3.baseline_v2 import fingerprints
from backtest.v3.model_v3 import (
    LR_C, LGB_GRID, XGB_GRID, VOTE_WEIGHTS, MODEL_WEIGHTS,
    log_loss, estimator, cv_evaluate, greedy_backward,
    apply_calibrator, fit_calibrator, raw_probability,
)

HERE = Path(__file__).resolve().parent
CALIBRATION_START = "2021-12-16"
CALIBRATION_SELECTION_START = "2021-12-24"
TRAIN_MIN, TRAIN_MAX = "2021-10-19", "2021-12-31"


def time_folds(dates, n_splits=5):
    dates = pd.Series(dates).reset_index(drop=True).astype(str).str[:10]
    if dates.max() >= CALIBRATION_START:
        raise ValueError("The December holdout is reserved; it is not a model-selection fold")
    if not dates.is_monotonic_increasing:
        raise ValueError("Model-selection dates must be sorted")
    from sklearn.model_selection import TimeSeriesSplit

    unique = np.array(sorted(dates.unique()))
    if len(unique) <= n_splits:
        raise ValueError("Insufficient unique dates for five time folds")
    folds = []
    for train_dates, validation_dates in TimeSeriesSplit(n_splits=n_splits).split(unique):
        train = np.flatnonzero(dates.isin(unique[train_dates]))
        validation = np.flatnonzero(dates.isin(unique[validation_dates]))
        if dates.iloc[train].max() >= dates.iloc[validation].min():
            raise AssertionError("A date crosses a training/validation boundary")
        folds.append((train, validation))
    return folds


def select_and_fit(train, columns, name):
    if train.date.max() > TRAIN_MAX or train.date.min() < TRAIN_MIN:
        raise ValueError("Training data outside the authorized October-December window")
    if not np.isfinite(train[columns].to_numpy(dtype=float)).all():
        raise ValueError("Training requires complete finite features")
    paths = [HERE / "features_train.csv", Path(__file__)]
    signature = fingerprints(paths)
    selection_path = HERE / f"selection_{name}.json"
    bundle_path = HERE / f"model_{name}.joblib"
    validation_path = HERE / f"validation_{name}.csv"
    if selection_path.exists() and bundle_path.exists() and validation_path.exists():
        saved = json.loads(selection_path.read_text())
        if saved["input_fingerprints"] == signature and saved["candidate_features"] == list(columns):
            print(f"[{name}] Reusing frozen, fingerprint-verified training result", flush=True)
            return joblib.load(bundle_path), saved, pd.read_csv(validation_path)
    development = train.loc[train.date.lt(CALIBRATION_START)].reset_index(drop=True)
    december = train.loc[train.date.ge(CALIBRATION_START)].reset_index(drop=True)
    if len(december) < 40:
        raise ValueError("December holdout is too small")
    folds = time_folds(development.date)
    y = development.home_won.to_numpy(dtype=int)
    with threadpool_limits(limits=1):
        selected, trace = greedy_backward(development[columns], y, columns, folds, name)
        X = development[selected].to_numpy(dtype=float)
        configurations = [{"kind": "logreg", "parameters": {"C": C}} for C in LR_C]
        configurations += [{"kind": "lightgbm", "parameters": parameters} for parameters in LGB_GRID]
        configurations += [{"kind": "xgboost", "parameters": parameters} for parameters in XGB_GRID]
        results = []
        for config in configurations:
            loss, predictions = cv_evaluate(config, X, y, folds)
            results.append({"config": config, "cv_log_loss": loss, "oof": predictions})
            print(f"[{name}] {config['kind']} {config['parameters']}: {loss:.6f}", flush=True)
        families = [
            min((r for r in results if r["config"]["kind"] == family), key=lambda r: r["cv_log_loss"])
            for family in ["logreg", "lightgbm", "xgboost"]
        ]
        families.sort(key=lambda r: (r["cv_log_loss"], r["config"]["kind"]))
        top = families[:2]
        covered = np.isfinite(top[0]["oof"]) & np.isfinite(top[1]["oof"])
        market = development.p_market_home.to_numpy(dtype=float)
        blends = []
        for weight in VOTE_WEIGHTS:
            vote = weight * top[0]["oof"] + (1 - weight) * top[1]["oof"]
            for model_weight in MODEL_WEIGHTS:
                blended = model_weight * vote + (1 - model_weight) * market
                blends.append({
                    "vote_weight": weight, "model_weight": model_weight,
                    "cv_log_loss": log_loss(y[covered], blended[covered]), "oof": blended,
                })
        chosen = min(blends, key=lambda b: (b["cv_log_loss"], b["model_weight"], -b["vote_weight"]))
        selection = {
            "name": name, "candidate_features": list(columns), "selected_features": selected,
            "backward_trace": trace, "top_two": [r["config"] for r in top],
            "vote_weight": chosen["vote_weight"], "model_weight": chosen["model_weight"],
            "cv_log_loss": chosen["cv_log_loss"],
            "cv_market_log_loss": log_loss(y[covered], market[covered]),
            "cv_results": [{"config": r["config"], "cv_log_loss": r["cv_log_loss"]} for r in results],
            "ensemble_grid": [{k: v for k, v in b.items() if k != "oof"} for b in blends],
            "folds": [
                {"train_start": development.date.iloc[tr].min(), "train_end": development.date.iloc[tr].max(),
                 "validation_start": development.date.iloc[va].min(), "validation_end": development.date.iloc[va].max(),
                 "n_train": len(tr), "n_validation": len(va)}
                for tr, va in folds
            ],
            "n_development": len(development), "n_december": len(december),
            "input_fingerprints": signature,
        }
        pre_dec_models = []
        for result in top:
            fitted = estimator(result["config"])
            fitted.fit(X, y)
            pre_dec_models.append(fitted)
        dec_raw = raw_probability(
            pre_dec_models, december[selected].to_numpy(dtype=float), december.p_market_home, selection
        )
        calibration_fit = december.date.lt(CALIBRATION_SELECTION_START).to_numpy()
        calibration_validation = ~calibration_fit
        if calibration_fit.sum() < 20 or calibration_validation.sum() < 20:
            raise ValueError("Both chronological December calibration blocks need at least 20 games")
        calibration_results = []
        for kind in ["none", "platt", "isotonic"]:
            fitted = fit_calibrator(kind, dec_raw[calibration_fit],
                                    december.home_won.to_numpy()[calibration_fit].astype(int))
            probabilities = apply_calibrator(fitted, dec_raw[calibration_validation])
            loss = log_loss(december.home_won.to_numpy()[calibration_validation], probabilities)
            calibration_results.append({"kind": kind, "validation_log_loss": loss,
                                        "calibrator": fitted, "validation_probability": probabilities})
        cal_choice = min(calibration_results, key=lambda r: (r["validation_log_loss"], r["kind"] != "none", r["kind"]))
        validation = december.loc[calibration_validation, [
            "date", "home", "away", "home_won", "dec_home", "dec_away", "p_market_home"
        ]].copy()
        validation["p_model_home"] = cal_choice["validation_probability"]
        validation["p_raw_home"] = dec_raw[calibration_validation]
        validation["base_fit_max_date"] = development.date.max()
        validation["calibration_fit_max_date"] = december.date.loc[calibration_fit].max()
        validation.to_csv(validation_path, index=False)
        selection["calibration"] = cal_choice["kind"]
        selection["calibration_validation"] = [
            {"kind": r["kind"], "validation_log_loss": r["validation_log_loss"]} for r in calibration_results
        ]
        selection["calibration_fit_rows"] = int(calibration_fit.sum())
        selection["calibration_validation_rows"] = int(calibration_validation.sum())
        selection["calibration_protocol"] = (
            "Base/feature/family/ensemble selection only before Dec 16. Fit calibrators Dec 16-23; "
            "choose using Dec 24-31 log loss. Gate validation uses those genuinely held-out predictions. "
            "Refit the chosen calibrator on all December out-of-sample raw predictions, then fit base "
            "models through Dec 31. No Jan-Mar labels enter any choice."
        )
        final_models = []
        for result in top:
            fitted = estimator(result["config"])
            fitted.fit(train[selected].to_numpy(dtype=float), train.home_won.to_numpy(dtype=int))
            final_models.append(fitted)
        calibrator = fit_calibrator(cal_choice["kind"], dec_raw, december.home_won.to_numpy(dtype=int))
        bundle = {"models": final_models, "calibrator": calibrator, "selection": selection}
        joblib.dump(bundle, bundle_path)
        selection_path.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
        pd.DataFrame([
            {"family": r["config"]["kind"], "parameters": json.dumps(r["config"]["parameters"]),
             "cv_log_loss": r["cv_log_loss"]}
            for r in results
        ]).to_csv(HERE / f"cv_results_{name}.csv", index=False)
        print(f"[{name}] frozen: {len(selected)} features; top two "
              f"{[r['config']['kind'] for r in top]}; vote={chosen['vote_weight']}, "
              f"model weight={chosen['model_weight']}; calibration={cal_choice['kind']}", flush=True)
    return bundle, selection, validation


def predict(bundle, frame, complete_mask):
    probabilities = frame.p_market_home.to_numpy(dtype=float).copy()
    complete_mask = np.asarray(complete_mask, dtype=bool)
    valid = frame.loc[complete_mask]
    selection = bundle["selection"]
    if len(valid):
        raw = raw_probability(
            bundle["models"], valid[selection["selected_features"]].to_numpy(dtype=float),
            valid.p_market_home.to_numpy(dtype=float), selection,
        )
        probabilities[complete_mask] = apply_calibrator(bundle["calibrator"], raw)
    if not np.isfinite(probabilities).all() or ((probabilities <= 0) | (probabilities >= 1)).any():
        raise AssertionError("Invalid model probabilities")
    return probabilities
