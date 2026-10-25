"""Ablation on the 2021 tune window ONLY (OOF Brier) to pick the v2 feature set.

Protocol-legal: decisions use 2021 data exclusively (TimeSeriesSplit OOF, point-in-time).
The winning variant is then run through the full walk-forward pipeline once.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from run_backtest_v2 import (make_logreg, make_lgbm, cv_oof_and_train, calibrate,
                             apply_cal, V1_DIFF_FEATS, V1_CTX_FEATS, V1_RAW_FEATS,
                             V2_DIFF_EXTRA, V2_RAW_EXTRA, V2_CTX_EXTRA)
from sklearn.metrics import brier_score_loss

f = pd.read_parquet("features_v2.parquet")
f["date"] = pd.to_datetime(f.date)
f = f.rename(columns={"home_win": "home_won"})
tr = f[f.date.dt.year == 2021].reset_index(drop=True)

CFG = dict(lr_c=0.03, lgb_num_leaves=63, lgb_lr=0.02, w_lgb=0.4, cal="isotonic")

SP_ST = [c for c in V2_DIFF_EXTRA + V2_RAW_EXTRA if "_st_" in c]
SP_L10 = [c for c in V2_DIFF_EXTRA + V2_RAW_EXTRA if "_l10_" in c]
BP_SC = [c for c in V2_DIFF_EXTRA + V2_RAW_EXTRA if c.startswith("bp_")]
OPS_WOBA = [c for c in V2_DIFF_EXTRA + V2_RAW_EXTRA if "ops" in c or "woba" in c]
CTX_EXTRA_PF = ["pf_basic5", "pf_3yr", "pf_hr5"]
CTX_EXTRA_TZ = ["tz_shift", "tz_abs", "away_tz7"]

VARIANTS = {
    "A_v2_full": (V2_DIFF_EXTRA, V2_RAW_EXTRA, V2_CTX_EXTRA),
    "B_v1_plus_park_tz_ops_bp": ([c for c in V2_DIFF_EXTRA if c not in SP_ST + SP_L10],
                                 [c for c in V2_RAW_EXTRA if c not in SP_ST + SP_L10],
                                 CTX_EXTRA_PF + CTX_EXTRA_TZ),
    "C_v1_plus_park_tz_ops": ([c for c in V2_DIFF_EXTRA if c not in SP_ST + SP_L10 + BP_SC],
                              [c for c in V2_RAW_EXTRA if c not in SP_ST + SP_L10 + BP_SC],
                              CTX_EXTRA_PF + CTX_EXTRA_TZ),
    "D_v1_plus_park_tz": ([c for c in V2_DIFF_EXTRA if c not in SP_ST + SP_L10 + BP_SC + OPS_WOBA],
                          [c for c in V2_RAW_EXTRA if c not in SP_ST + SP_L10 + BP_SC + OPS_WOBA],
                          CTX_EXTRA_PF + CTX_EXTRA_TZ),
    "E_v1_plus_park_only": ([], [], CTX_EXTRA_PF),
    "F_v1_plus_sp_only": (SP_ST + SP_L10, SP_ST + SP_L10, []),
    "G_v1_plus_bp_only": (BP_SC, BP_SC, []),
}


def oof_brier(diff_extra, raw_extra, ctx_extra):
    lo = list(dict.fromkeys(V1_DIFF_FEATS + diff_extra + V1_CTX_FEATS + ctx_extra))
    lg = list(dict.fromkeys(V1_DIFF_FEATS + diff_extra + V1_RAW_FEATS + raw_extra
                             + V1_CTX_FEATS + ctx_extra + ["park"]))
    oof_lr, _, cov = cv_oof_and_train(tr, lo, lambda: make_logreg(CFG["lr_c"]))
    oof_lgb, _, cov_l = cv_oof_and_train(tr, lg, lambda: make_lgbm(CFG["lgb_num_leaves"], CFG["lgb_lr"]), is_lgbm=True)
    ens = CFG["w_lgb"] * oof_lgb + (1 - CFG["w_lgb"]) * oof_lr
    m = cov & cov_l
    y = tr.home_won.to_numpy()[m]
    cal = calibrate(ens[m], y)
    p = apply_cal(cal, ens[m])
    return brier_score_loss(y, p), float(np.mean((oof_lr[m] - y) ** 2)), float(np.mean((oof_lgb[m] - y) ** 2))


print(f"{'variant':28s} {'ens_brier':>10s} {'lr_brier':>10s} {'lgb_brier':>10s}")
results = {}
for name, (d, r, c) in VARIANTS.items():
    b, bl, bg = oof_brier(d, r, c)
    results[name] = (b, bl, bg)
    print(f"{name:28s} {b:10.5f} {bl:10.5f} {bg:10.5f}")
best = min(results, key=lambda k: results[k][0])
print("BEST:", best, results[best])
