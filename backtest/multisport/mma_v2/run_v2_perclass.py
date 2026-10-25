"""
MMA v2 — per-weight-class models with pooled shrinkage (experiment vs pooled walk-forward).

For each year Y and class group g (HW/LHW/MW/WW/LW/FW/BW/FLW/W/OTH):
  - class LGBM + XGB trained on pre-Y fights in group g (if n_g >= MIN_N)
  - isotonic calibration on class-level expanding CV (last 3 val years)
  - shrinkage blend with pooled model: p = w * p_class + (1-w) * p_pooled,
    w = n_g / (n_g + K), K tuned once on pre-2020 expanding CV (no lookahead).
  - class ensemble = mean(LR_pooled, LGBM_blend, XGB_blend)

Reported vs the pooled walk-forward ensemble on the same protocol.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.isotonic import IsotonicRegression
from backtest_lib import brier_score, evaluate_bets, kelly_path
from v2_lib import (prepare_df, build_feature_frames, make_lr, make_lgb, make_xgb, model_builder,
                    cv_year, year_model, walk_forward, assemble, fit_predict)

base = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
MIN_N = 80

df = prepare_df(os.path.join(base, "fights_features_v2.parquet"))
df, core_feats, all_feats = build_feature_frames(df)

# hparams from pooled run
m = json.load(open(os.path.join(base, "metrics_v2_pooled.json")))
def hp_from(cfg):
    return dict(lr=dict(fn=model_builder("lr", dict(C=0.03))),
                lgb=dict(fn=model_builder("lgb", cfg["lgb"]["cfg"]), cfg=cfg["lgb"]["cfg"]),
                xgb=dict(fn=model_builder("xgb", cfg["xgb"]["cfg"]), cfg=cfg["xgb"]["cfg"]))
hp_core = hp_from(m["hparams"]["core"])
hp_all = hp_from(m["hparams"]["all"])
print("hparams:", json.dumps(m["hparams"]))

GROUPS = ["HW", "LHW", "MW", "WW", "LW", "FW", "BW", "FLW", "W", "OTH"]

# ---------------- tune shrinkage K on pre-2020 CV ----------------
def tune_k(feats, hp, tag):
    X = df[feats].values
    y = df["yR"].values
    d = df["date_n"]
    cg = df["cg"].values
    tr_all = d < pd.Timestamp("2020-01-01")
    Xtr_all, ytr_all, dtr_all, cgtr = X[tr_all], y[tr_all], d[tr_all], cg[tr_all]
    val_years = [2017, 2018, 2019]
    # pooled OOF
    pool_oof = {}
    for mn in ["lr", "lgb", "xgb"]:
        oof, _, _ = cv_year(Xtr_all, ytr_all, dtr_all, hp[mn]["fn"], val_years)
        pool_oof[mn] = oof
    # per-class OOF + class train sizes at each val row
    class_oof = {g: {mn: np.full(len(Xtr_all), np.nan) for mn in ["lgb", "xgb"]} for g in GROUPS}
    n_g_at = np.full(len(Xtr_all), np.nan)
    for vy in val_years:
        tr = dtr_all < pd.Timestamp(f"{vy}-01-01")
        va = dtr_all.dt.year == vy
        for g in GROUPS:
            gtr = tr & (cgtr == g)
            if gtr.sum() < MIN_N:
                continue
            n_g_at[va & (cgtr == g)] = gtr.sum()
            for mn in ["lgb", "xgb"]:
                model = hp[mn]["fn"]()
                pva, _, _ = fit_predict(model, Xtr_all[gtr], ytr_all[gtr], Xtr_all[va & (cgtr == g)], ytr_all[va & (cgtr == g)])
                class_oof[g][mn][va & (cgtr == g)] = pva
    # evaluate blend per K
    best_k, best_loss = None, np.inf
    for K in [100, 250, 500, 1000, 2500, 1e9]:
        blend = {}
        for mn in ["lgb", "xgb"]:
            b = pool_oof[mn].copy()
            for g in GROUPS:
                rows = ~np.isnan(n_g_at) & (cgtr == g) & ~np.isnan(class_oof[g][mn])
                w = n_g_at / (n_g_at + K)
                b[rows] = w[rows] * class_oof[g][mn][rows] + (1 - w[rows]) * pool_oof[mn][rows]
            blend[mn] = b
        ens_k = (pool_oof["lr"] + blend["lgb"] + blend["xgb"]) / 3.0
        valid = ~np.isnan(ens_k)
        loss = log_loss(ytr_all[valid], ens_k[valid])
        print(f"  [{tag}] K={K}: OOF logloss {loss:.4f}")
        if loss < best_loss:
            best_loss, best_k = loss, K
    print(f"  [{tag}] -> K={best_k} ({best_loss:.4f})")
    return best_k

K_core = tune_k(core_feats, hp_core, "core")
K_all = tune_k(all_feats, hp_all, "all")

# ---------------- walk-forward with per-class shrinkage ----------------
def wf_perclass(feats, hp, years, K):
    X = df[feats].values
    y = df["yR"].values
    d = df["date_n"]
    cg = df["cg"].values
    out = {}
    for Y in years:
        tr = d < pd.Timestamp(f"{Y}-01-01")
        te = d.dt.year == Y
        if te.sum() == 0:
            continue
        Xtr, ytr, dtr, cgtr = X[tr], y[tr], d[tr], cg[tr]
        Xte, cgte = X[te], cg[te]
        years_in = sorted(set(dtr.dt.year))
        val_years = years_in[-3:]
        # pooled models
        preds = {}
        for mn in ["lr", "lgb", "xgb"]:
            pte, iso, _, _ = year_model(Xtr, ytr, dtr, hp[mn]["fn"], val_years, Xte)
            preds[mn] = iso.predict(pte)
        # class models + shrink
        n_g = {}
        class_preds = {g: {mn: None for mn in ["lgb", "xgb"]} for g in GROUPS}
        for g in GROUPS:
            gtr = cgtr == g
            gte = cgte == g
            if gtr.sum() < MIN_N or gte.sum() == 0:
                continue
            n_g[g] = gtr.sum()
            for mn in ["lgb", "xgb"]:
                # class CV for isotonic
                oof = np.full(gtr.sum(), np.nan)
                for vy in val_years:
                    trv = dtr[gtr] < pd.Timestamp(f"{vy}-01-01")
                    vav = dtr[gtr].dt.year == vy
                    if vav.sum() == 0:
                        continue
                    model = hp[mn]["fn"]()
                    pva, _, _ = fit_predict(model, Xtr[gtr][trv], ytr[gtr][trv], Xtr[gtr][vav], ytr[gtr][vav])
                    oof[vav] = pva
                valid = ~np.isnan(oof)
                if valid.sum() >= 50:
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
                    iso.fit(oof[valid], ytr[gtr][valid])
                else:
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
                    iso.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
                ng = gtr.sum()
                cut = int(ng * 0.8)
                model = hp[mn]["fn"]()
                _, _, _ = fit_predict(model, Xtr[gtr][:cut], ytr[gtr][:cut], Xtr[gtr][cut:], ytr[gtr][cut:], Xte[gte])
                p_cls = iso.predict(model.predict_proba(Xte[gte])[:, 1])
                w = ng / (ng + K)
                preds[f"{mn}_{g}"] = np.full(te.sum(), np.nan)
                preds[f"{mn}_{g}"][gte] = p_cls
                class_preds[g][mn] = (gte, w)
        # apply shrinkage to get blend arrays
        for mn in ["lgb", "xgb"]:
            blend = preds[mn].copy()
            for g in GROUPS:
                if class_preds[g][mn] is None:
                    continue
                gte, w = class_preds[g][mn]
                blend[gte] = w * preds[f"{mn}_{g}"][gte] + (1 - w) * preds[mn][gte]
            preds[f"blend_{mn}"] = blend
        preds["ens_class"] = (preds["lr"] + preds["blend_lgb"] + preds["blend_xgb"]) / 3.0
        preds["ens_pooled"] = (preds["lr"] + preds["lgb"] + preds["xgb"]) / 3.0
        out[Y] = dict(preds=preds, y=y[te], idx=np.where(te)[0])
        print(f"  year {Y}: train={tr.sum()} test={te.sum()} class_sizes={ {g: n_g.get(g, 0) for g in GROUPS if g in n_g} }")
    return out

print("\n=== per-class walk-forward ALL feats (2020-2021) ===")
wfa = wf_perclass(all_feats, hp_all, [2020, 2021], K_all)
print("\n=== per-class walk-forward CORE feats (2020-2026) ===")
wfc = wf_perclass(core_feats, hp_core, list(range(2020, 2027)), K_core)

# ---------------- evaluate ----------------
def slice_odds(df, idx, ps, lo=None, hi=None):
    m = np.ones(len(idx), dtype=bool)
    if lo is not None:
        m &= df["date_n"].values[idx] >= lo
    if hi is not None:
        m &= df["date_n"].values[idx] <= hi
    vals = df.iloc[idx[m]].reset_index(drop=True)
    om = vals["R_odds"].notna().values & vals["B_odds"].notna().values
    return vals[om].reset_index(drop=True), {k: np.asarray(v)[m][om] for k, v in ps.items()}

pri_ps, _, pri_idx = assemble(wfa, [2020, 2021])
tp, pri_ps = slice_odds(df, pri_idx, pri_ps, pd.Timestamp("2020-01-01"), pd.Timestamp("2021-03-20"))
ext_ps, _, ext_idx = assemble(wfc, list(range(2020, 2027)))
ta, ext_ps = slice_odds(df, ext_idx, ext_ps)

def rep(tag, d, ps, names):
    y = d["yR"].values
    print(f"\n=== {tag} n={len(d)} ===")
    for nm in names:
        p = ps[nm]
        g0 = evaluate_bets(d, p, 0.0)
        print(f"  {nm:>10}: acc={accuracy_score(y, (p >= 0.5).astype(int)):.4f} auc={roc_auc_score(y, p):.4f} "
              f"brier={brier_score(p, y):.4f} roi0={g0['roi']:+.4f} (n={g0['n']})")

rep("PRIMARY", tp, pri_ps, ["ens_pooled", "ens_class"])
rep("EXTENDED", ta, ext_ps, ["ens_pooled", "ens_class"])

print("\nper-year extended (ens_class):")
for yr in range(2020, 2027):
    mm = ta["date_n"].dt.year == yr
    if mm.sum() == 0:
        continue
    p = ext_ps["ens_class"][mm]
    ysub = ta["yR"].values[mm]
    g = evaluate_bets(ta[mm].reset_index(drop=True), p, 0.0)
    print(f"  {yr}: n={mm.sum():>4} acc={accuracy_score(ysub, (p >= 0.5).astype(int)):.3f} "
          f"auc={roc_auc_score(ysub, p):.3f} roi0={g['roi']:+.3f}")

out = dict(K_core=K_core, K_all=K_all, runtime_sec=time.time() - t0)
json.dump(out, open(os.path.join(base, "metrics_v2_perclass.json"), "w"), indent=1, default=str)
print("\nwrote metrics_v2_perclass.json", out)
print(f"runtime {time.time()-t0:.0f}s")
