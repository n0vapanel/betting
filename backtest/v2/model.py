"""Model layer for the v2 backtest.

Walk-forward discipline — every choice is made on training-period data only:
1. Feature selection: greedy backward elimination minimizing time-ordered
   5-fold CV neg-log-loss (LR, C=0.01).
2. Hyperparameters: C grid re-tuned by the same CV on the selected subset;
   a regularized HGB grid is evaluated too, plus the LR/HGB soft blend.
3. Model family: best CV neg-log-loss among LR / HGB / blend.
4. Calibration: fitted on a genuine holdout (train on pre-Feb rows, predict
   Feb-2022 rows) — raw vs Platt vs isotonic chosen by Feb-holdout log loss.
5. Test rows are touched only by the final predict_proba.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LR_GRID_C = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
HGB_GRID = [
    {"max_iter": 50, "learning_rate": 0.03, "max_leaf_nodes": 7},
    {"max_iter": 50, "learning_rate": 0.05, "max_leaf_nodes": 7},
    {"max_iter": 100, "learning_rate": 0.03, "max_leaf_nodes": 7},
    {"max_iter": 50, "learning_rate": 0.03, "max_leaf_nodes": 15},
    {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 7},
    {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 15},
    {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15},
]

CALIBRATION_MONTH = "2022-02-01"  # holdout = training rows on/after this date


def _folds(X: pd.DataFrame, n_splits: int = 5):
    return list(TimeSeriesSplit(n_splits=n_splits).split(X))


def _log_loss(y_true, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _lr_factory(C: float):
    return lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    )


def _hgb_factory(**params) -> Callable:
    return lambda: HistGradientBoostingClassifier(
        random_state=42, early_stopping=False, **params
    )


def _cv_evaluate(
    estimator_factory: Callable, X: pd.DataFrame, y: pd.Series
) -> Tuple[float, np.ndarray]:
    """Mean neg-log-loss across time-ordered folds + out-of-fold probs."""
    oof = np.full(len(y), np.nan)
    losses = []
    for tr_idx, va_idx in _folds(X):
        est = estimator_factory()
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = est.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = p
        losses.append(_log_loss(y.iloc[va_idx].values, p))
    return float(np.mean(losses)), oof


def _cv_ll_lr(C: float, X: pd.DataFrame, y: pd.Series) -> float:
    return _cv_evaluate(_lr_factory(C), X, y)[0]


def _greedy_backward(X: pd.DataFrame, y: pd.Series, verbose: bool) -> list:
    """Greedy backward elimination minimizing CV neg-log-loss (LR C=0.01)."""
    cols = list(X.columns)
    base = _cv_ll_lr(0.01, X[cols], y)
    if verbose:
        print(f"[model] feature selection start: {len(cols)} feats, CV LL {base:.4f}")
    changed = True
    while changed and len(cols) > 3:
        changed = False
        best_ll, best_col = base, None
        for c in cols:
            trial = [x for x in cols if x != c]
            ll = _cv_ll_lr(0.01, X[trial], y)
            if ll < best_ll - 1e-5:
                best_ll, best_col = ll, c
        if best_col is not None:
            cols.remove(best_col)
            if verbose:
                print(f"[model] drop {best_col}: {base:.4f} -> {best_ll:.4f}")
            base = best_ll
            changed = True
    if verbose:
        print(f"[model] selected {len(cols)} features (CV LL {base:.4f}): {cols}")
    return cols


def select_model(X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> Dict:
    """Full training-period model selection (features + C + family + calibrator)."""
    selected_cols = _greedy_backward(X, y, verbose=verbose)
    Xs = X[selected_cols]

    results = []
    for C in LR_GRID_C:
        ll, oof = _cv_evaluate(_lr_factory(C), Xs, y)
        results.append({"kind": "lr", "name": f"LogReg C={C}", "ll": ll,
                        "oof": oof, "C": C})
    for params in HGB_GRID:
        ll, oof = _cv_evaluate(_hgb_factory(**params), Xs, y)
        name = f"HGB it={params['max_iter']} lr={params['learning_rate']} " \
               f"leaf={params['max_leaf_nodes']}"
        results.append({"kind": "hgb", "name": name, "ll": ll, "oof": oof,
                        "params": params})

    lr_best = min([r for r in results if r["kind"] == "lr"], key=lambda r: r["ll"])
    hgb_best = min([r for r in results if r["kind"] == "hgb"], key=lambda r: r["ll"])
    blend_oof = 0.5 * lr_best["oof"] + 0.5 * hgb_best["oof"]
    covered = ~np.isnan(blend_oof)  # TimeSeriesSplit leaves the tail uncovered
    blend_ll = _log_loss(y.values[covered], blend_oof[covered])
    results.append({"kind": "blend", "name": "Blend 0.5LR+0.5HGB", "ll": blend_ll,
                    "oof": blend_oof})

    best = min(results, key=lambda r: r["ll"])
    if verbose:
        for r in sorted(results, key=lambda r: r["ll"]):
            print(f"[model] CV log-loss {r['ll']:.4f}  {r['name']}")
        print(f"[model] chosen: {best['name']}")
    return {
        "best_kind": best["kind"],
        "best_name": best["name"],
        "best_ll": best["ll"],
        "oof_probs": best["oof"],
        "selected_cols": selected_cols,
        "lr_best_C": lr_best["C"],
        "lr_best_name": lr_best["name"],
        "hgb_best_params": hgb_best.get("params"),
        "hgb_best_name": hgb_best["name"],
        "cv_table": [(r["name"], r["ll"]) for r in results],
    }


def _make_estimator(kind: str, lr_C: float, hgb_params: Dict) -> object:
    if kind == "lr":
        return _lr_factory(lr_C)()
    return _hgb_factory(**hgb_params)()


def _predict(est, X: pd.DataFrame) -> np.ndarray:
    return est.predict_proba(X)[:, 1]


def _fit_pair(kind: str, lr_C: float, hgb_params: Dict):
    lr = _make_estimator("lr", lr_C, hgb_params)
    hgb = _make_estimator("hgb", lr_C, hgb_params)
    if kind == "blend":
        return (lr, hgb)
    if kind == "lr":
        return lr
    return hgb


def _fit(est_or_pair, kind: str, X, y) -> None:
    if kind == "blend":
        est_or_pair[0].fit(X, y)
        est_or_pair[1].fit(X, y)
    else:
        est_or_pair.fit(X, y)


def _preds(est_or_pair, kind: str, X) -> np.ndarray:
    if kind == "blend":
        return 0.5 * _predict(est_or_pair[0], X) + 0.5 * _predict(est_or_pair[1], X)
    return _predict(est_or_pair, X)


def _pick_calibrator(feb_preds: np.ndarray, feb_y: np.ndarray) -> Dict:
    """Choose raw vs Platt vs isotonic by Feb-holdout log loss."""
    p = np.clip(feb_preds, 1e-6, 1 - 1e-6)
    out = {"name": "none", "ll": _log_loss(feb_y, p), "obj": None}
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(feb_preds, feb_y)
    iso_p = np.clip(iso.predict(feb_preds), 1e-6, 1 - 1e-6)
    if _log_loss(feb_y, iso_p) < out["ll"]:
        out = {"name": "isotonic", "ll": _log_loss(feb_y, iso_p), "obj": iso}
    lo = np.log(p / (1 - p)).reshape(-1, 1)
    platt = LogisticRegression(max_iter=2000)
    platt.fit(lo, feb_y)
    pl_p = np.clip(platt.predict_proba(lo)[:, 1], 1e-6, 1 - 1e-6)
    if _log_loss(feb_y, pl_p) < out["ll"]:
        out = {"name": "platt", "ll": _log_loss(feb_y, pl_p), "obj": platt}
    return out


def train_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dates_train: pd.Series,
    X_test: pd.DataFrame,
    selection: Dict,
    verbose: bool = True,
) -> Dict:
    """Final fit + holdout-chosen calibration + test prediction."""
    kind = selection["best_kind"]
    lr_C = selection["lr_best_C"]
    hgb_params = selection["hgb_best_params"]
    cols = selection["selected_cols"]
    X_train = X_train[cols]
    X_test = X_test[cols]

    # Holdout: fit on pre-Feb rows, predict Feb rows out-of-sample.
    feb_mask = dates_train.values >= CALIBRATION_MONTH
    cal_fit_X, cal_fit_y = X_train[~feb_mask], y_train[~feb_mask]
    cal_pred_X, cal_pred_y = X_train[feb_mask], y_train[feb_mask]
    n_cal = int(len(cal_pred_y))
    calibrator = {"name": "none", "ll": None, "obj": None}
    if n_cal >= 20:
        cal_est = _fit_pair(kind, lr_C, hgb_params)
        _fit(cal_est, kind, cal_fit_X, cal_fit_y)
        feb_preds = _preds(cal_est, kind, cal_pred_X)
        calibrator = _pick_calibrator(feb_preds, cal_pred_y.values)
        if verbose:
            print(f"[model] calibration holdout: {n_cal} Feb-2022 rows; "
                  f"chosen {calibrator['name']} "
                  f"(holdout LL {calibrator['ll']:.4f})")
    elif verbose:
        print(f"[model] WARN: only {n_cal} Feb rows; skipping calibration")

    final = _fit_pair(kind, lr_C, hgb_params)
    _fit(final, kind, X_train, y_train)
    train_p = _preds(final, kind, X_train)
    test_p = _preds(final, kind, X_test)
    if calibrator["obj"] is not None:
        if calibrator["name"] == "platt":
            train_p = calibrator["obj"].predict_proba(
                np.log(np.clip(train_p, 1e-6, 1 - 1e-6)
                       / (1 - np.clip(train_p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            )[:, 1]
            test_p = calibrator["obj"].predict_proba(
                np.log(np.clip(test_p, 1e-6, 1 - 1e-6)
                       / (1 - np.clip(test_p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            )[:, 1]
        else:
            train_p = calibrator["obj"].predict(train_p)
            test_p = calibrator["obj"].predict(test_p)
    test_p = np.clip(test_p, 0.005, 0.995)
    return {
        "train_probs": train_p,
        "test_probs": test_p,
        "n_cal_rows": n_cal,
        "calibrator": calibrator["name"],
    }
