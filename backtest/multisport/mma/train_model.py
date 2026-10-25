"""
Train + export the frozen MMA ensemble models for the live-card picker.

train_model(history_df) is extracted from run_backtest.py (verified v1 backtest):
exactly the same hyperparams (LogReg C=0.03, LGBM num_leaves=15 lr=0.03),
the same expanding-CV OOF used for isotonic calibration, the same final fits,
and the same 50/50 (LR + LGBM) ensemble.

Two frozen models are exported to models/:
  1. primary_2016_2019.joblib  -- ALL features (incl. strike diffs), trained 2016-01-01..2019-12-31
  2. extended_core_thru_2025.joblib -- CORE features only, trained 2016-01-01..2025-12-31
     (strike-rate data ends ~2021, so live 2026 cards use core features)

Each artifact: dict(lr=..., iso_lr=..., gbm=..., iso_gbm=..., feats=[...],
                    train_lo, train_hi, feature_set, cv={...}, oof=...)
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import lightgbm as lgb

from feature_utils import add_model_features

base = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(base, "models")


# ---------- model factories (identical to run_backtest.py) ----------

def make_lr(C):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=C, max_iter=2000))])


def make_lr_pred(model, Xtr, ytr, Xva, yva):
    model.fit(Xtr, ytr)
    return model.predict_proba(Xva)[:, 1]


def make_lgb(leaves, lr):
    return lgb.LGBMClassifier(
        n_estimators=800, learning_rate=lr, num_leaves=leaves,
        min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=0, verbose=-1)


def make_lgb_pred(model, Xtr, ytr, Xva, yva):
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="logloss",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model.predict_proba(Xva)[:, 1]


# ---------- expanding CV (same as run_backtest.py cv_fit_eval) ----------

def cv_fit_eval(model_fn, X_tr_all, y_tr_all, dates, fold_bounds, val_bounds, make_pred):
    """model_fn() -> fresh model. Returns (oof_preds, mean_val_logloss, losses, iters)."""
    oof = np.full(len(X_tr_all), np.nan)
    losses = []
    iters = []
    for (tlo, thi), (vlo, vhi) in zip(fold_bounds, val_bounds):
        tr_idx = (dates >= tlo) & (dates < thi)
        va_idx = (dates >= vlo) & (dates < vhi)
        model = model_fn()
        pv = make_pred(model, X_tr_all[tr_idx], y_tr_all[tr_idx], X_tr_all[va_idx], y_tr_all[va_idx])
        oof[va_idx] = pv
        losses.append(log_loss(y_tr_all[va_idx], pv))
        iters.append(getattr(model, "best_iteration_", None))
    return oof, float(np.mean(losses)), losses, iters


# ---------- the extracted training function ----------

def train_model(history_df, feats, fold_bounds=None, val_bounds=None,
                lr_C=0.03, gbm_cfg=(15, 0.03)):
    """
    Fit the LogReg + LGBM ensemble exactly as run_backtest.py's final fit:
      - LogReg(C=lr_C) on the full history, isotonic-calibrated on expanding-CV OOF
      - LGBM(num_leaves, lr) refit on first 80% of history with early stopping on the
        last 20%, isotonic-calibrated on expanding-CV OOF
      - ensemble = 50/50 average of both calibrated models
    fold_bounds/val_bounds default to the v1 3-fold schedule (2016-2019).
    Returns dict with lr/iso_lr/gbm/iso_gbm, feats, cv metrics, oof arrays.
    """
    TRAIN_LO = pd.Timestamp("2016-01-01")
    if fold_bounds is None or val_bounds is None:
        fold_bounds = [(TRAIN_LO, pd.Timestamp("2017-01-01")),
                       (TRAIN_LO, pd.Timestamp("2018-01-01")),
                       (TRAIN_LO, pd.Timestamp("2019-01-01"))]
        val_bounds = [(pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")),
                      (pd.Timestamp("2018-01-01"), pd.Timestamp("2019-01-01")),
                      (pd.Timestamp("2019-01-01"), pd.Timestamp("2020-01-01"))]

    X = history_df[feats].values
    y = history_df["yR"].values
    dates = history_df["date_n"].values

    # CV for OOF (used ONLY for calibration targets)
    oof_lr, cv_lr, _, _ = cv_fit_eval(lambda: make_lr(lr_C), X, y, dates,
                                      fold_bounds, val_bounds, make_lr_pred)
    oof_gbm, cv_gbm, _, _ = cv_fit_eval(lambda: make_lgb(*gbm_cfg), X, y, dates,
                                        fold_bounds, val_bounds, make_lgb_pred)

    # final LogReg fit on full history
    lr = make_lr(lr_C)
    lr.fit(X, y)
    valid_lr = ~np.isnan(oof_lr)
    iso_lr = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    iso_lr.fit(oof_lr[valid_lr], y[valid_lr])

    # final LGBM refit with early stopping on last 20% of history
    n = len(X)
    cut = int(n * 0.8)
    gbm = make_lgb(*gbm_cfg)
    gbm.fit(X[:cut], y[:cut], eval_set=[(X[cut:], y[cut:])], eval_metric="logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    valid_gbm = ~np.isnan(oof_gbm)
    iso_gbm = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    iso_gbm.fit(oof_gbm[valid_gbm], y[valid_gbm])

    return {
        "lr": lr, "iso_lr": iso_lr, "gbm": gbm, "iso_gbm": iso_gbm,
        "feats": list(feats),
        "cv": {"lr_logloss": cv_lr, "gbm_logloss": cv_gbm,
               "lr_oof_auc": float(roc_auc_score(y[valid_lr], oof_lr[valid_lr])),
               "gbm_oof_auc": float(roc_auc_score(y[valid_gbm], oof_gbm[valid_gbm])),
               "lr_C": lr_C, "gbm_cfg": list(gbm_cfg),
               "fold_bounds": [(str(a.date()), str(b.date())) for a, b in fold_bounds]},
        "oof": {"lr": oof_lr, "gbm": oof_gbm},
    }


def predict_ensemble(models, X):
    """50/50 calibrated ensemble probability of a Red win."""
    p_lr = models["iso_lr"].predict(models["lr"].predict_proba(X)[:, 1])
    p_gbm = models["iso_gbm"].predict(models["gbm"].predict_proba(X)[:, 1])
    return (p_lr + p_gbm) / 2.0


# ---------- training driver ----------

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_parquet(os.path.join(base, "fights_features.parquet"))
    df = df[~df["Winner"].isin(["Draw", "No Contest"])].copy()
    df["yR"] = (df["Winner"] == "Red").astype(int)
    df, core_feats, all_feats = add_model_features(df)

    TRAIN_LO = pd.Timestamp("2016-01-01")

    # ---- 1) primary model: strike features, trained 2016-2019 (v1 exact) ----
    primary_hi = pd.Timestamp("2020-01-01")
    train_primary = df[(df["date_n"] >= TRAIN_LO) & (df["date_n"] < primary_hi)].copy()
    print(f"training PRIMARY  (all feats, 2016-2019): n={len(train_primary)}, feats={len(all_feats)}")
    m1 = train_model(train_primary, all_feats)  # default v1 fold schedule
    m1.update(train_lo=str(TRAIN_LO.date()), train_hi="2019-12-31", feature_set="all")
    print("  primary CV:", json.dumps(m1["cv"], default=str))

    # ---- 2) extended model: core features, trained through 2025-12-31 ----
    ext_hi = pd.Timestamp("2025-12-31")
    train_ext = df[(df["date_n"] >= TRAIN_LO) & (df["date_n"] <= ext_hi)].copy()
    years = list(range(2019, 2026))
    fold_bounds = [(TRAIN_LO, pd.Timestamp(f"{y}-01-01")) for y in years]
    val_bounds = [(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")) for y in years]
    print(f"training EXTENDED (core feats, 2016-01-01..2025-12-31): n={len(train_ext)}, "
          f"feats={len(core_feats)}, folds={years}")
    m2 = train_model(train_ext, core_feats, fold_bounds=fold_bounds, val_bounds=val_bounds)
    m2.update(train_lo=str(TRAIN_LO.date()), train_hi="2025-12-31", feature_set="core")
    print("  extended CV:", json.dumps(m2["cv"], default=str))

    # ---- save ----
    import joblib
    p1 = os.path.join(MODELS_DIR, "primary_2016_2019.joblib")
    p2 = os.path.join(MODELS_DIR, "extended_core_thru_2025.joblib")
    joblib.dump(m1, p1)
    joblib.dump(m2, p2)
    manifest = {
        "primary": {"file": os.path.basename(p1), "features": m1["feats"], "cv": m1["cv"]},
        "extended": {"file": os.path.basename(p2), "features": m2["feats"], "cv": m2["cv"]},
    }
    with open(os.path.join(MODELS_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    print("saved:", p1)
    print("saved:", p2)

    # ---- verify primary matches the v1 backtest numbers ----
    from backtest_lib import am_to_dec, fair_probs
    test_primary = df[(df["date_n"] >= pd.Timestamp("2020-01-01"))
                      & (df["date_n"] <= pd.Timestamp("2021-03-20"))].copy()
    test_primary = test_primary[test_primary["R_odds"].notna() & test_primary["B_odds"].notna()].copy()
    Xp = test_primary[all_feats].values
    p_ens = predict_ensemble(m1, Xp)
    acc = accuracy_score(test_primary["yR"].values, (p_ens >= 0.5).astype(int))
    auc = roc_auc_score(test_primary["yR"].values, p_ens)
    decR = am_to_dec(test_primary["R_odds"]); decB = am_to_dec(test_primary["B_odds"])
    fr, fb = fair_probs(decR, decB)
    print(f"\nVERIFY primary vs v1: v1 test acc=0.7657 auc(ens)~0.833 | refit acc={acc:.4f} auc={auc:.4f}")
    print(f"  expected OOF AUC LR=0.5949 GBM=0.8313 | got {m1['cv']['lr_oof_auc']:.4f} / "
          f"{m1['cv']['gbm_oof_auc']:.4f}")

    # extended holdout sanity: 2026-01..2026-03 fights (never seen in training)
    test_2026 = df[df["date_n"] >= pd.Timestamp("2026-01-01")].copy()
    test_2026 = test_2026[test_2026["R_odds"].notna() & test_2026["B_odds"].notna()].copy()
    if len(test_2026):
        Xe = test_2026[core_feats].values
        p_ens2 = predict_ensemble(m2, Xe)
        acc2 = accuracy_score(test_2026["yR"].values, (p_ens2 >= 0.5).astype(int))
        auc2 = roc_auc_score(test_2026["yR"].values, p_ens2)
        print(f"  extended holdout sanity (2026-01..03, n={len(test_2026)}): acc={acc2:.4f} auc={auc2:.4f}")


if __name__ == "__main__":
    main()
