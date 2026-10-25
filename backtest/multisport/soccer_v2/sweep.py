#!/usr/bin/env python
"""Variant sweep for soccer v2 (uses cached feature parquet).

Explores: ensemble weights, model-only variants, feature ablations, LGBM capacity.
Prints compact comparison table for both targets; picks nothing automatically.
"""
import json, os, sys
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
test = full[full["Date"] >= pd.Timestamp("2022-03-01")].reset_index(drop=True)
train = train.dropna(subset=feat_cols).reset_index(drop=True)
meds = train[feat_cols].median()
t = test.copy()
for c in feat_cols:
    t[c] = t[c].fillna(meds[c])
test = t
print(f"train {len(train)} test {len(test)}")

y1_tr = train["FTR"].map({"H": 0, "D": 1, "A": 2}).values
y1_te = test["FTR"].map({"H": 0, "D": 1, "A": 2}).values
y2_tr = train["Over25"].values
y2_te = test["Over25"].values
mkt1_te = test[["m_H", "m_D", "m_A"]].values
odds1_te = test[["c_H", "c_D", "c_A"]].values
mkt2_te = test["m_Over"].values
odds2_te = test[["c_Over", "c_Under"]].values

LGB_BASE = dict(learning_rate=0.03, num_leaves=15, min_child_samples=50,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                reg_lambda=5.0, random_state=42, verbose=-1, n_jobs=4)

def logloss_mc(P, y):
    P = np.clip(P, 1e-9, 1 - 1e-9)
    return -np.log(P[np.arange(len(y)), y]).mean()

def pick_C(X, y, binary):
    tss = TimeSeriesSplit(n_splits=5)
    best_c, best = None, np.inf
    for C in [0.05, 0.2, 1.0, 5.0, 20.0]:
        scores = []
        for tr, va in tss.split(X):
            m = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=42)
            m.fit(X[tr], y[tr])
            P = m.predict_proba(X[va])
            if binary:
                P = np.column_stack([1 - P[:, 1], P[:, 1]])
            scores.append(logloss_mc(P, y[va]))
        s = float(np.mean(scores))
        if s < best:
            best, best_c = s, C
    return best_c, best

def best_iter(X, y, binary, params):
    cut = int(len(X) * 0.85)
    cls = lgb.LGBMClassifier(objective="binary" if binary else "multiclass",
                             num_class=None if binary else 3, n_estimators=600, **params)
    cls.fit(X[:cut], y[:cut], eval_set=[(X[cut:], y[cut:])],
            callbacks=[lgb.early_stopping(30, verbose=False)])
    return int(cls.best_iteration_)

def oof(model_fn, X, y, binary):
    tss = TimeSeriesSplit(n_splits=5)
    n_out = 2 if binary else 3
    o = np.zeros((len(X), n_out))
    for tr, va in tss.split(X):
        m = model_fn()
        m.fit(X[tr], y[tr])
        if binary:
            o[va, 1] = m.predict_proba(X[va])[:, 1]
            o[va, 0] = 1 - o[va, 1]
        else:
            o[va] = m.predict_proba(X[va])
    return o

def cal_mc(oof_, y):
    isos = []
    for c in range(3):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(oof_[:, c], (y == c).astype(float))
        isos.append(iso)
    def apply(P):
        out = np.column_stack([np.clip(iso.predict(P[:, c]), 1e-6, 1 - 1e-6) for c, iso in enumerate(isos)])
        return out / out.sum(1, keepdims=True)
    return apply

def cal_bin(oof_, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(oof_[:, 1], y.astype(float))
    return lambda P: np.clip(iso.predict(P[:, 1]), 1e-6, 1 - 1e-6)

def boot(dates, roi, n_iter=2000):
    uniq = np.unique(dates)
    rng = np.random.default_rng(7)
    out = []
    for _ in range(n_iter):
        s = rng.choice(uniq, size=len(uniq), replace=True)
        m = np.isin(dates, s)
        if m.sum():
            out.append(roi[m].mean())
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def run_variant(name, feat_sub, w_lr=0.5, lgb_extra=None, use_lr=True, use_lgb=True):
    params = dict(LGB_BASE)
    if lgb_extra:
        params.update(lgb_extra)
    Xtr = train[feat_sub].values
    Xte = test[feat_sub].values
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    res = {"name": name, "n_feats": len(feat_sub)}

    # ---- 1X2
    P_parts = []
    if use_lr:
        C, _ = pick_C(Xtr_s, y1_tr, False)
        o_lr = oof(lambda: LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=42), Xtr_s, y1_tr, False)
        c_lr = cal_mc(o_lr, y1_tr)
        m = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=42).fit(Xtr_s, y1_tr)
        P_parts.append(("lr", c_lr(m.predict_proba(Xte_s))))
    if use_lgb:
        it = best_iter(Xtr_s, y1_tr, False, params)
        o_lg = oof(lambda: lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=it, **params),
                   Xtr_s, y1_tr, False)
        c_lg = cal_mc(o_lg, y1_tr)
        m = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=it, **params).fit(Xtr_s, y1_tr)
        P_parts.append(("lgb", c_lg(m.predict_proba(Xte_s))))
    if len(P_parts) == 2:
        P = w_lr * P_parts[0][1] + (1 - w_lr) * P_parts[1][1]
    else:
        P = P_parts[0][1]
    side = P.argmax(1)
    p_side = P[np.arange(len(y1_te)), side]
    m_side = mkt1_te[np.arange(len(y1_te)), side]
    o_side = odds1_te[np.arange(len(y1_te)), side]
    win = side == y1_te
    res["acc1"] = float((side == y1_te).mean())
    res["ll1"] = logloss_mc(P, y1_te)
    gates1 = {}
    for eps in (0.0, 0.02, 0.05):
        bet = p_side - m_side > eps
        nb = int(bet.sum())
        roi = float(np.where(bet, np.where(win, o_side - 1, -1), 0).sum() / nb) if nb else np.nan
        gates1[eps] = (nb, roi)
    res["g1"] = gates1
    bet0 = p_side - m_side > 0.0
    roi0 = np.where(bet0, np.where(win, o_side - 1, -1), 0)
    lo, hi = boot(test["Date"].values[bet0], roi0[bet0]) if bet0.sum() else (np.nan, np.nan)
    res["ci1"] = (lo, hi)

    # ---- O/U
    P_parts2 = []
    if use_lr:
        C, _ = pick_C(Xtr_s, y2_tr, True)
        o_lr = oof(lambda: LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=42), Xtr_s, y2_tr, True)
        c_lr = cal_bin(o_lr, y2_tr)
        m = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=42).fit(Xtr_s, y2_tr)
        P_parts2.append(("lr", c_lr(m.predict_proba(Xte_s))))
    if use_lgb:
        it = best_iter(Xtr_s, y2_tr, True, params)
        o_lg = oof(lambda: lgb.LGBMClassifier(objective="binary", n_estimators=it, **params), Xtr_s, y2_tr, True)
        c_lg = cal_bin(o_lg, y2_tr)
        m = lgb.LGBMClassifier(objective="binary", n_estimators=it, **params).fit(Xtr_s, y2_tr)
        P_parts2.append(("lgb", c_lg(m.predict_proba(Xte_s))))
    if len(P_parts2) == 2:
        P2 = w_lr * P_parts2[0][1] + (1 - w_lr) * P_parts2[1][1]
    else:
        P2 = P_parts2[0][1]
    side2 = (P2 > 0.5).astype(int)
    p2 = np.where(side2 == 1, P2, 1 - P2)
    m2 = np.where(side2 == 1, mkt2_te, 1 - mkt2_te)
    o2 = np.where(side2 == 1, odds2_te[:, 0], odds2_te[:, 1])
    win2 = side2 == y2_te
    res["acc2"] = float((side2 == y2_te).mean())
    gates2 = {}
    for eps in (0.0, 0.02, 0.05):
        bet = p2 - m2 > eps
        nb = int(bet.sum())
        roi = float(np.where(bet, np.where(win2, o2 - 1, -1), 0).sum() / nb) if nb else np.nan
        gates2[eps] = (nb, roi)
    res["g2"] = gates2
    bet0 = p2 - m2 > 0.0
    roi0 = np.where(bet0, np.where(win2, o2 - 1, -1), 0)
    lo, hi = boot(test["Date"].values[bet0], roi0[bet0]) if bet0.sum() else (np.nan, np.nan)
    res["ci2"] = (lo, hi)
    return res

def keep(prefixes_keep):
    return [c for c in feat_cols if any(c.startswith(p) for p in prefixes_keep)]

ALL = feat_cols
NO_MOPEN = [c for c in ALL if not c.startswith("m_open_")]
NO_CLV = [c for c in ALL if not c.startswith(("m_open_", "clv_"))]
NO_XG = [c for c in ALL if not c.startswith(("xGF", "xGA", "xGd", "int_", "mom_xG"))]
NO_FORM = [c for c in ALL if not c.startswith(("win_streak", "unbeaten", "loss_streak", "rest_", "mom_", "int_"))]
V1LIKE = [c for c in ALL if c.startswith(("GF", "GA", "ShF", "ShA", "STF", "STA", "CF", "CA", "FF", "FA",
                                          "CardF", "CardA", "Pts", "OPts", "elo"))]

variants = [
    ("ens5050", ALL, dict(w_lr=0.5)),
    ("ens3070", ALL, dict(w_lr=0.3)),
    ("ens7030", ALL, dict(w_lr=0.7)),
    ("lgb_only", ALL, dict(use_lr=False)),
    ("lr_only", ALL, dict(use_lgb=False)),
    ("no_mopen", NO_MOPEN, dict(w_lr=0.5)),
    ("no_clv", NO_CLV, dict(w_lr=0.5)),
    ("no_xg", NO_XG, dict(w_lr=0.5)),
    ("no_form", NO_FORM, dict(w_lr=0.5)),
    ("v1like_feats", V1LIKE, dict(w_lr=0.5)),
    ("lgb_cap", ALL, dict(use_lr=False, lgb_extra=dict(num_leaves=31, min_child_samples=20))),
]

rows = []
for name, feats, kw in variants:
    r = run_variant(name, feats, **kw)
    rows.append(r)
    print(f"[{name}] nf={r['n_feats']} 1X2 acc={r['acc1']*100:.2f}% ll={r['ll1']:.4f} | "
          f"eps0: {r['g1'][0.0][0]}b {r['g1'][0.0][1]*100:+.1f}% CI[{r['ci1'][0]*100:+.1f},{r['ci1'][1]*100:+.1f}] | "
          f"eps.02: {r['g1'][0.02][0]}b {r['g1'][0.02][1]*100:+.1f}% | eps.05: {r['g1'][0.05][0]}b {r['g1'][0.05][1]*100:+.1f}%")
    print(f"        OU acc={r['acc2']*100:.2f}% | "
          f"eps0: {r['g2'][0.0][0]}b {r['g2'][0.0][1]*100:+.1f}% CI[{r['ci2'][0]*100:+.1f},{r['ci2'][1]*100:+.1f}] | "
          f"eps.02: {r['g2'][0.02][0]}b {r['g2'][0.02][1]*100:+.1f}% | eps.05: {r['g2'][0.05][0]}b {r['g2'][0.05][1]*100:+.1f}%")
    sys.stdout.flush()

json.dump([{**r, "g1": {str(k): v for k, v in r["g1"].items()},
            "g2": {str(k): v for k, v in r["g2"].items()}} for r in rows],
          open(os.path.join(BASE, "sweep_results.json"), "w"), indent=2)
print("\nSaved sweep_results.json")
