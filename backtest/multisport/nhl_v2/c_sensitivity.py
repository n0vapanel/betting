"""C-grid sensitivity: OOF Brier vs test metrics per C, full 43-feature model."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss

RNG = 42
np.random.seed(RNG)
VIG = 0.0476

df = pd.read_parquet("data/games_features_v2.parquet")
FEATS = [c for c in df.columns if c.startswith("diff_")]
train = df[df.date <= pd.Timestamp("2022-02-28")].reset_index(drop=True)
test = df[df.date >= pd.Timestamp("2022-03-01")].reset_index(drop=True)
Xtr, ytr = train[FEATS].values, train.home_win.values
Xte, yte = test[FEATS].values, test.home_win.values
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
tscv = TimeSeriesSplit(n_splits=5)


def cv_oof(C):
    oof = np.zeros(len(ytr))
    for tr_idx, va_idx in tscv.split(Xtr_s):
        m = LogisticRegression(C=C, max_iter=3000).fit(Xtr_s[tr_idx], ytr[tr_idx])
        oof[va_idx] = m.predict_proba(Xtr_s[va_idx])[:, 1]
    return oof


def platt(oof_in, y, pred):
    lo = LogisticRegression(C=1e6, max_iter=3000)
    lo.fit(oof_in.reshape(-1, 1), y)
    return lo.predict_proba(pred.reshape(-1, 1))[:, 1]


test = test.copy()
test["odds_h"] = 1 / (test.p_mkt_home * (1 + VIG))
test["odds_a"] = 1 / (1 - test.p_mkt_home) / (1 + VIG)
print(f"{'C':>6} {'OOFbrier':>9} {'acc':>7} {'flatROI':>8} {'brier':>7}")
for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    oof = cv_oof(C)
    m = LogisticRegression(C=C, max_iter=3000).fit(Xtr_s, ytr)
    p = platt(oof, ytr, m.predict_proba(Xte_s)[:, 1])
    side = np.where(p >= 0.5, "H", "A")
    o = np.where(side == "H", test.odds_h, test.odds_a)
    w_ = np.where(side == "H", test.home_win == 1, test.home_win == 0)
    r_ = np.where(w_ == 1, o - 1.0, -1.0)
    print(f"{C:>6} {brier_score_loss(ytr, oof):>9.4f} {w_.mean():>7.4f} {r_.mean():>+8.4f} "
          f"{brier_score_loss(test.home_win, p):>7.4f}")

# also: does the new-feature edge hold under 5-fold TSCV oof accuracy? print oof acc per C
print("\nOOF acc per C:")
for C in [0.003, 0.01, 0.03]:
    oof = cv_oof(C)
    acc = ((oof >= 0.5) == ytr).mean()
    print(f"  C={C}: oof acc {acc:.4f}")
