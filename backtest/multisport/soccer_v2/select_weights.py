#!/usr/bin/env python
"""Select per-target ensemble weights and LGBM config via time-ordered OOF CV.

For each target (1X2, O/U 2.5): build 5-fold TSS OOF predictions for LogReg and
LGBM (base config and higher-capacity config), isotonic-calibrate each on its
OOF, then grid blend weights on calibrated-OOF logloss. Test set is never used.
"""
import json, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

BASE = "/Users/x/Desktop/x/Coding Projects/betting bot/backtest/multisport/soccer_v2"
full = pd.read_parquet(os.path.join(BASE, "cache_features.parquet"))
feat_cols = json.load(open(os.path.join(BASE, "cache_feat_cols.json")))
train = full[full["Date"] < pd.Timestamp("2022-03-01")].reset_index(drop=True)
train = train.dropna(subset=feat_cols).reset_index(drop=True)
X = train[feat_cols].values
sc = StandardScaler().fit(X)
X = sc.transform(X)
y1 = train["FTR"].map({"H": 0, "D": 1, "A": 2}).values
y2 = train["Over25"].values

LGB_BASE = dict(learning_rate=0.03, num_leaves=15, min_child_samples=50,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                reg_lambda=5.0, random_state=42, verbose=-1, n_jobs=4)
LGB_CAP = dict(LGB_BASE, num_leaves=31, min_child_samples=20)

def ll_mc(P, y):
    P = np.clip(P, 1e-9, 1 - 1e-9)
    return -np.log(P[np.arange(len(y)), y]).mean()

def ll_bin(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def oof_lr(X, y, binary):
    tss = TimeSeriesSplit(n_splits=5)
    n_out = 2 if binary else 3
    o = np.zeros((len(X), n_out))
    for tr, va in tss.split(X):
        Cbest, best = None, np.inf
        for C in [0.05, 0.2, 1.0, 5.0, 20.0]:
            m = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=42)
            m.fit(X[tr], y[tr])
            P = m.predict_proba(X[tr])
            s = ll_mc(P, y[tr]) if not binary else ll_bin(P[:, 1], y[tr])
            if s < best:
                best, Cbest = s, C
        m = LogisticRegression(C=Cbest, solver="lbfgs", max_iter=2000, random_state=42)
        m.fit(X[tr], y[tr])
        if binary:
            o[va, 1] = m.predict_proba(X[va])[:, 1]
            o[va, 0] = 1 - o[va, 1]
        else:
            o[va] = m.predict_proba(X[va])
    return o

def oof_lgb(X, y, binary, params):
    tss = TimeSeriesSplit(n_splits=5)
    n_out = 2 if binary else 3
    o = np.zeros((len(X), n_out))
    for tr, va in tss.split(X):
        cut = int(len(tr) * 0.85)
        cls = lgb.LGBMClassifier(objective="binary" if binary else "multiclass",
                                 num_class=None if binary else 3, n_estimators=600, **params)
        cls.fit(X[tr][:cut], y[tr][:cut], eval_set=[(X[tr][cut:], y[tr][cut:])],
                callbacks=[lgb.early_stopping(30, verbose=False)])
        n_est = int(cls.best_iteration_)
        m = lgb.LGBMClassifier(objective="binary" if binary else "multiclass",
                               num_class=None if binary else 3, n_estimators=n_est, **params)
        m.fit(X[tr], y[tr])
        if binary:
            o[va, 1] = m.predict_proba(X[va])[:, 1]
            o[va, 0] = 1 - o[va, 1]
        else:
            o[va] = m.predict_proba(X[va])
    return o

def cal_mc(o, y):
    isos = []
    for c in range(3):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(o[:, c], (y == c).astype(float))
        isos.append(iso)
    out = np.column_stack([np.clip(iso.predict(o[:, c]), 1e-6, 1 - 1e-6) for c, iso in enumerate(isos)])
    return out / out.sum(1, keepdims=True)

def cal_bin(o, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(o[:, 1], y.astype(float))
    return np.clip(iso.predict(o[:, 1]), 1e-6, 1 - 1e-6)

for label, y, binary in (("1X2", y1, False), ("OU", y2, True)):
    o_lr = oof_lr(X, y, binary)
    o_base = oof_lgb(X, y, binary, LGB_BASE)
    o_cap = oof_lgb(X, y, binary, LGB_CAP)
    if binary:
        c_lr, c_base, c_cap = cal_bin(o_lr, y), cal_bin(o_base, y), cal_bin(o_cap, y)
        ll = lambda p: ll_bin(p, y)
    else:
        c_lr, c_base, c_cap = cal_mc(o_lr, y), cal_mc(o_base, y), cal_mc(o_cap, y)
        ll = lambda P: ll_mc(P, y)
    print(f"== {label} ==  (OOF logloss: lr {ll(c_lr):.4f} | lgb_base {ll(c_base):.4f} | lgb_cap {ll(c_cap):.4f})")
    for name, c in (("base", c_base), ("cap", c_cap)):
        best_w, best_ll = None, np.inf
        for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            s = ll(w * c_lr + (1 - w) * c)
            if s < best_ll:
                best_ll, best_w = s, w
        print(f"   lgb={name}: best w_lr={best_w:.1f} -> blend OOF logloss {best_ll:.4f}")
