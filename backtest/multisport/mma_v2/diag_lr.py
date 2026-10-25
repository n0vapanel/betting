import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
import os
base = os.path.dirname(os.path.abspath(__file__))

df = pd.read_parquet(os.path.join(base, "fights_features.parquet"))
df = df[~df["Winner"].isin(["Draw", "No Contest"])].copy()
df["yR"] = (df["Winner"] == "Red").astype(int)
TRAIN_LO, TRAIN_HI = pd.Timestamp("2016-01-01"), pd.Timestamp("2020-01-01")
tr = df[(df["date_n"] >= TRAIN_LO) & (df["date_n"] < TRAIN_HI)].copy()
te = df[df["date_n"] >= TRAIN_HI].copy()
print("train", len(tr), "test", len(te))

for s in ["R", "B"]:
    df[f"win_pct_{s}"] = (df[f"w_prior_{s}"] + 1.0) / (df[f"n_prior_{s}"] + 2.0)
    df[f"days_since_{s}"] = df[f"days_since_{s}"].fillna(999.0)
    df[f"elo_before_{s}"] = df[f"elo_before_{s}"].fillna(1500.0)
    df[f"streak_pre_{s}"] = df[f"streak_pre_{s}"].fillna(0.0)
    df[f"longest_run_so_far_{s}"] = df[f"longest_run_so_far_{s}"].fillna(0.0)
    df[f"n_prior_{s}"] = df[f"n_prior_{s}"].fillna(0.0)
    df[f"w_prior_{s}"] = df[f"w_prior_{s}"].fillna(0.0)
    df[f"l_prior_{s}"] = df[f"l_prior_{s}"].fillna(0.0)
CORE = [("elo_before", "elo"), ("win_pct", "win_pct"), ("streak_pre", "streak"),
        ("longest_run_so_far", "longest_streak"), ("n_prior", "exp"), ("days_since", "days_since")]
for col, name in CORE:
    df[f"d_{name}"] = df[f"{col}_R"] - df[f"{col}_B"]
df["d_age"] = df["R_age"] - df["B_age"]
df["d_reach"] = df["R_Reach_cms"] - df["B_Reach_cms"]
df["d_height"] = df["R_Height_cms"] - df["B_Height_cms"]
df["d_weight"] = df["R_Weight_lbs"] - df["B_Weight_lbs"]
df["d_rank"] = df["B_match_weightclass_rank"].fillna(16) - df["R_match_weightclass_rank"].fillna(16)
for c in ["slpm", "str_acc", "td_acc", "td_def", "sub_per_f", "kd_per_f", "ctrl_share"]:
    df[f"d_{c}"] = df[f"feat_{c}_R"] - df[f"feat_{c}_B"]
tr = df[(df["date_n"] >= TRAIN_LO) & (df["date_n"] < TRAIN_HI)].copy()
te = df[df["date_n"] >= TRAIN_HI].copy()

feats = ["d_elo", "d_win_pct", "d_streak", "d_longest_streak", "d_exp", "d_days_since",
         "d_age", "d_reach", "d_height", "d_weight", "d_rank",
         "d_slpm", "d_str_acc", "d_td_acc", "d_td_def", "d_sub_per_f", "d_kd_per_f", "d_ctrl_share"]

# single-feature LR AUC on train (2017-2019 subset for comparability)
mask = tr["date_n"] >= pd.Timestamp("2017-01-01")
print("\nsingle-feature LR (imputed+scaled, C=1): train17-19 AUC | test AUC")
for f in feats:
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(tr[[f]].values)
    Xte = imp.transform(te[[f]].values)
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X, tr["yR"].values)
    a1 = roc_auc_score(tr.loc[mask, "yR"], lr.predict_proba(X[mask.values])[:, 1])
    a2 = roc_auc_score(te["yR"], lr.predict_proba(Xte)[:, 1])
    print(f"  {f:<18} {a1:.3f} | {a2:.3f}")

# full LR with scaling, C=0.03 vs C=1
for C in [0.03, 1.0]:
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    X = sc.fit_transform(imp.fit_transform(tr[feats].values))
    Xte = sc.transform(imp.transform(te[feats].values))
    lr = LogisticRegression(C=C, max_iter=2000)
    lr.fit(X, tr["yR"].values)
    pte = lr.predict_proba(Xte)[:, 1]
    print(f"\nfull LR C={C}: test AUC {roc_auc_score(te['yR'], pte):.4f} test logloss {log_loss(te['yR'], pte):.4f}")

# LGBM importances
X = tr[feats].values; y = tr["yR"].values
gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                         min_child_samples=30, random_state=0, verbose=-1)
gbm.fit(X, y)
imp = pd.Series(gbm.feature_importances_, index=feats).sort_values(ascending=False)
print("\nLGBM feature importances:")
print(imp.round(3).to_string())
