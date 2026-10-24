from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from backtest.v3.baseline_v2 import fingerprints


HERE = Path(__file__).resolve().parent
CALIBRATION_START = "2022-02-01"
CALIBRATION_SELECTION_START = "2022-02-15"
SEED = 42
LR_C = [0.003, 0.01, 0.03, 0.05, 0.1, 0.3, 1.0, 3.0]
LGB_GRID = [
    {"n_estimators": 60, "learning_rate": 0.03, "num_leaves": 3, "min_child_samples": 40, "reg_lambda": 10.0},
    {"n_estimators": 120, "learning_rate": 0.03, "num_leaves": 3, "min_child_samples": 40, "reg_lambda": 10.0},
    {"n_estimators": 80, "learning_rate": 0.05, "num_leaves": 5, "min_child_samples": 30, "reg_lambda": 10.0},
    {"n_estimators": 120, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 40, "reg_lambda": 30.0},
    {"n_estimators": 160, "learning_rate": 0.03, "num_leaves": 5, "min_child_samples": 50, "reg_lambda": 50.0},
]
XGB_GRID = [
    {"n_estimators": 60, "learning_rate": 0.03, "max_depth": 1, "min_child_weight": 5, "reg_lambda": 10.0},
    {"n_estimators": 120, "learning_rate": 0.03, "max_depth": 1, "min_child_weight": 5, "reg_lambda": 10.0},
    {"n_estimators": 80, "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 10, "reg_lambda": 30.0},
    {"n_estimators": 120, "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 20, "reg_lambda": 30.0},
    {"n_estimators": 160, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 20, "reg_lambda": 100.0},
]
VOTE_WEIGHTS = [0.25, 0.5, 0.75, 0.9]
MODEL_WEIGHTS = [0.25, 0.5, 0.75, 1.0]


def log_loss(y, probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))


def time_folds(dates, n_splits=5):
    dates = pd.Series(dates).reset_index(drop=True).astype(str).str[:10]
    if dates.max() >= CALIBRATION_START:
        raise ValueError("February is a reserved holdout, not a model-selection fold")
    if not dates.is_monotonic_increasing:
        raise ValueError("Model-selection dates must be sorted")
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


def estimator(config):
    kind, parameters = config["kind"], config["parameters"]
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=parameters["C"], max_iter=2000, tol=1e-6, solver="lbfgs", random_state=SEED
        ))
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            objective="binary", random_state=SEED, n_jobs=1, verbosity=-1,
            deterministic=True, force_col_wise=True, **parameters
        )
    if kind == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            random_state=SEED, n_jobs=1, subsample=1, colsample_bytree=1, **parameters
        )
    raise ValueError(f"Unknown model family {kind}")


def cv_evaluate(config, X, y, folds):
    predictions = np.full(len(y), np.nan)
    for training, validation in folds:
        fitted = estimator(config)
        fitted.fit(X[training], y[training])
        predictions[validation] = fitted.predict_proba(X[validation])[:, 1]
    covered = np.isfinite(predictions)
    return log_loss(y[covered], predictions[covered]), predictions


def greedy_backward(X, y, columns, folds, name):
    columns = list(columns)
    values = np.asarray(X, dtype=float)
    config = {"kind": "logreg", "parameters": {"C": 0.05}}
    positions = {column: i for i, column in enumerate(columns)}
    base, _ = cv_evaluate(config, values, y, folds)
    trace = [{"removed": None, "n_features": len(columns), "cv_log_loss": base}]
    print(f"[{name}] backward selection: {len(columns)} features, CV LL {base:.6f}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        while len(columns) > 3:
            def trial(removed):
                kept = [positions[c] for c in columns if c != removed]
                score, _ = cv_evaluate(config, values[:, kept], y, folds)
                return score, removed
            candidates = list(executor.map(trial, columns))
            score, removed = min(candidates, key=lambda result: (result[0], result[1]))
            if score >= base - 1e-5:
                break
            columns.remove(removed)
            base = score
            trace.append({"removed": removed, "n_features": len(columns), "cv_log_loss": score})
            print(f"[{name}] drop {removed}: CV LL {score:.6f} ({len(columns)} left)", flush=True)
    return columns, trace


def apply_calibrator(calibrator, values):
    p = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    if calibrator["kind"] == "none":
        return np.clip(p, 0.005, 0.995)
    if calibrator["kind"] == "platt":
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        return np.clip(calibrator["model"].predict_proba(logit)[:, 1], 0.005, 0.995)
    return np.clip(calibrator["model"].predict(p), 0.005, 0.995)


def fit_calibrator(kind, probability, y):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    if kind == "none":
        return {"kind": kind, "model": None}
    if kind == "platt":
        fitted = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        fitted.fit(np.log(p / (1 - p)).reshape(-1, 1), y)
    elif kind == "isotonic":
        fitted = IsotonicRegression(y_min=0.005, y_max=0.995, out_of_bounds="clip")
        fitted.fit(p, y)
    else:
        raise ValueError(f"Unknown calibrator {kind}")
    return {"kind": kind, "model": fitted}


def raw_probability(models, X, market, selection):
    predictions = [fitted.predict_proba(X)[:, 1] for fitted in models]
    vote = selection["vote_weight"] * predictions[0] + (1 - selection["vote_weight"]) * predictions[1]
    return selection["model_weight"] * vote + (1 - selection["model_weight"]) * np.asarray(market)


def select_and_fit(train, columns, name):
    if train.date.max() > "2022-02-28" or train.date.min() < "2021-10-20":
        raise ValueError("Training data outside the authorized October–February window")
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
    february = train.loc[train.date.ge(CALIBRATION_START)].reset_index(drop=True)
    if len(february) < 40:
        raise ValueError("February holdout is too small")
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
            "n_development": len(development), "n_february": len(february),
            "input_fingerprints": signature,
        }
        pre_feb_models = []
        for result in top:
            fitted = estimator(result["config"])
            fitted.fit(X, y)
            pre_feb_models.append(fitted)
        feb_raw = raw_probability(
            pre_feb_models, february[selected].to_numpy(dtype=float), february.p_market_home, selection
        )
        calibration_fit = february.date.lt(CALIBRATION_SELECTION_START).to_numpy()
        calibration_validation = ~calibration_fit
        if calibration_fit.sum() < 20 or calibration_validation.sum() < 20:
            raise ValueError("Both chronological February calibration blocks need at least 20 games")
        calibration_results = []
        for kind in ["none", "platt", "isotonic"]:
            fitted = fit_calibrator(kind, feb_raw[calibration_fit], february.home_won.to_numpy()[calibration_fit].astype(int))
            probabilities = apply_calibrator(fitted, feb_raw[calibration_validation])
            loss = log_loss(february.home_won.to_numpy()[calibration_validation], probabilities)
            calibration_results.append({"kind": kind, "validation_log_loss": loss,
                                        "calibrator": fitted, "validation_probability": probabilities})
        cal_choice = min(calibration_results, key=lambda r: (r["validation_log_loss"], r["kind"] != "none", r["kind"]))
        validation = february.loc[calibration_validation, [
            "date", "home", "away", "home_won", "dec_home", "dec_away", "p_market_home"
        ]].copy()
        validation["p_model_home"] = cal_choice["validation_probability"]
        validation["p_raw_home"] = feb_raw[calibration_validation]
        validation["base_fit_max_date"] = development.date.max()
        validation["calibration_fit_max_date"] = february.date.loc[calibration_fit].max()
        validation.to_csv(validation_path, index=False)
        selection["calibration"] = cal_choice["kind"]
        selection["calibration_validation"] = [
            {"kind": r["kind"], "validation_log_loss": r["validation_log_loss"]} for r in calibration_results
        ]
        selection["calibration_fit_rows"] = int(calibration_fit.sum())
        selection["calibration_validation_rows"] = int(calibration_validation.sum())
        selection["calibration_protocol"] = (
            "Base/feature/family/ensemble selection only before February. Fit calibrators Feb 1–14; "
            "choose using Feb 15–28 log loss. Gate validation uses those genuinely held-out predictions. "
            "Refit the chosen calibrator on all February out-of-sample raw predictions, then fit base models "
            "through Feb 28. No March labels enter any choice."
        )
        final_models = []
        for result in top:
            fitted = estimator(result["config"])
            fitted.fit(train[selected].to_numpy(dtype=float), train.home_won.to_numpy(dtype=int))
            final_models.append(fitted)
        calibrator = fit_calibrator(cal_choice["kind"], feb_raw, february.home_won.to_numpy(dtype=int))
        bundle = {"models": final_models, "calibrator": calibrator, "selection": selection}
        joblib.dump(bundle, bundle_path)
        selection_path.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
        pd.DataFrame([
            {"family": r["config"]["kind"], "parameters": json.dumps(r["config"]["parameters"]), "cv_log_loss": r["cv_log_loss"]}
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
