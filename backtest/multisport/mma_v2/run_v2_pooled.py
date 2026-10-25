"""
MMA v2 — pooled walk-forward yearly-refit backtest.

Upgrades vs v1:
- walk-forward refit per calendar year (model for year Y trained on all fights < Jan 1 Y)
- XGBoost added to the ensemble (LR + LGBM + XGB mean)
- new features: DOB age/age-at-debut/career days, fighter-details height/reach gap-fill,
  fight country, stance flags, opponent quality (Elo/win%), opponent style ratios,
  recent-3 form, empty-arena flag
- frozen-v2 ablation (single 2016-2019-trained model, v1-style) for direct comparison
- odds-augmented variant (market prob as feature) for reference only

Protocol identical to v1: primary 2020-01..2021-03-20 (all features, n=559),
extended 2020..2026 (core features), flat $1 at closing American odds, edge gates
0/0.02/0.05, quarter-Kelly capped 0.25, 2000-resample bootstrap CI.
"""
import os, json, time, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from backtest_lib import brier_score, evaluate_bets, kelly_path, bootstrap_ci
from v2_lib import (prepare_df, build_feature_frames, make_lr, make_lgb, make_xgb, model_builder,
                    cv_year, year_model, walk_forward, assemble)

base = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
t0 = time.time()

df = prepare_df(os.path.join(base, "fights_features_v2.parquet"))
df, core_feats, all_feats = build_feature_frames(df)
print(f"rows={len(df)} | core feats={len(core_feats)} | all feats={len(all_feats)}")

# ---------------- hyperparameter tuning (pre-2020 expanding CV, both feature sets) ----------------
def tune_gbm(feat_list, tag):
    X = df[feat_list].values
    y = df["yR"].values
    d = df["date_n"]
    tr = d < pd.Timestamp("2020-01-01")
    Xtr, ytr, dtr = X[tr], y[tr], d[tr]
    val_years = [2017, 2018, 2019]
    best_lgb, best_xgb = None, None
    bl, bx = np.inf, np.inf
    for leaves in [15, 31]:
        for lr in [0.03, 0.06]:
            _, loss, _ = cv_year(Xtr, ytr, dtr, lambda: make_lgb(leaves, lr), val_years)
            print(f"  [{tag}] LGBM leaves={leaves} lr={lr}: {loss:.4f}")
            if loss < bl:
                bl, best_lgb = loss, dict(leaves=leaves, lr=lr)
    for depth in [3, 4, 5]:
        for lr in [0.03, 0.06]:
            _, loss, _ = cv_year(Xtr, ytr, dtr, lambda: make_xgb(depth, lr), val_years)
            print(f"  [{tag}] XGB depth={depth} lr={lr}: {loss:.4f}")
            if loss < bx:
                bx, best_xgb = loss, dict(depth=depth, lr=lr)
    print(f"  [{tag}] -> LGBM {best_lgb} ({bl:.4f}), XGB {best_xgb} ({bx:.4f})")
    return dict(lr=dict(fn=model_builder("lr", dict(C=0.03)), cfg=dict(C=0.03)),
                lgb=dict(fn=model_builder("lgb", best_lgb), cfg=best_lgb, cv_loss=bl),
                xgb=dict(fn=model_builder("xgb", best_xgb), cfg=best_xgb, cv_loss=bx))

# ---------------- cache for expensive sections ----------------
CACHE = os.path.join(base, "pooled_cache.pkl")
cache = {}
if os.path.exists(CACHE):
    cache = pickle.load(open(CACHE, "rb"))
    print("loaded cache:", list(cache.keys()))

def cached(key, fn):
    if key in cache:
        print(f"[cache hit] {key}")
        return cache[key]
    val = fn()
    cache[key] = val
    pickle.dump(cache, open(CACHE, "wb"))
    return val

hp_core = cached("hp_core", lambda: tune_gbm(core_feats, "core"))
hp_all = cached("hp_all", lambda: tune_gbm(all_feats, "all"))

print("\n=== walk-forward ALL feats (primary window years) ===")
wf_all = cached("wf_all", lambda: walk_forward(df, all_feats, [2020, 2021], hp_all))

print("\n=== walk-forward CORE feats (extended window years) ===")
wf_core = cached("wf_core", lambda: walk_forward(df, core_feats, list(range(2020, 2027)), hp_core))

# ---------------- frozen variant (train < 2020, predict all 2020+) ----------------
def frozen(df, feats, hp, tag):
    print(f"\n=== frozen v2 ({tag}) ===")
    X = df[feats].values
    y = df["yR"].values
    d = df["date_n"]
    tr = d < pd.Timestamp("2020-01-01")
    te = d >= pd.Timestamp("2020-01-01")
    Xtr, ytr, dtr, Xte = X[tr], y[tr], d[tr], X[te]
    preds = {}
    for mname in ["lr", "lgb", "xgb"]:
        pte, iso, cv_loss, bi = year_model(Xtr, ytr, dtr, hp[mname]["fn"], [2017, 2018, 2019], Xte)
        preds[mname] = iso.predict(pte)
        print(f"  {mname}: CV loss {cv_loss:.4f}")
    preds["ens"] = (preds["lr"] + preds["lgb"] + preds["xgb"]) / 3.0
    preds["ens_gb"] = (preds["lgb"] + preds["xgb"]) / 2.0
    return preds, np.where(te)[0]

frozen_all_preds, frozen_all_idx = cached("frozen_all", lambda: frozen(df, all_feats, hp_all, "all feats"))
frozen_core_preds, frozen_core_idx = cached("frozen_core", lambda: frozen(df, core_feats, hp_core, "core feats"))

# ---------------- odds-augmented variant (report-only) ----------------
print("\n=== odds-augmented variant ===")
df["fairR_feat"] = df["fairR"]
odd_core_feats = core_feats + ["fairR_feat"]
odd_all_feats = all_feats + ["fairR_feat"]
wf_odd_core = cached("wf_odd_core", lambda: walk_forward(df, odd_core_feats, list(range(2020, 2027)), hp_core))
wf_odd_all = cached("wf_odd_all", lambda: walk_forward(df, odd_all_feats, [2020, 2021], hp_all))

# ---------------- evaluation helpers ----------------
def eval_set(test_df, ps, y):
    out = {}
    for nm, p in ps.items():
        out[nm] = dict(acc=float(accuracy_score(y, (p >= 0.5).astype(int))),
                       auc=float(roc_auc_score(y, p)), brier=brier_score(p, y))
    out["market_fav"] = dict(acc=float(accuracy_score(y, (test_df["fairR"].values >= 0.5).astype(int))))
    out["market"] = dict(brier=brier_score(test_df["fairR"].values, y),
                         auc=float(roc_auc_score(y, test_df["fairR"].values)))
    return out

def show_eval(test_df, ps, y):
    ev = eval_set(test_df, ps, y)
    for stat in ["acc", "auc", "brier"]:
        print(f"{stat}:", {k: round(v[stat], 4) for k, v in ev.items() if stat in v})
    return ev

def gates_table(test_df, p):
    gates = {eps: evaluate_bets(test_df, p, eps) for eps in [0.0, 0.02, 0.05]}
    rois = {eps: g["roi"] for eps, g in gates.items() if g["n"] >= 20 and not np.isnan(g["roi"])}
    best = max(rois, key=lambda e: rois[e]) if rois else None
    return gates, best

# primary window slice
def slice_window(df, idx_all, ps, lo, hi):
    m = (df["date_n"].values[idx_all] >= lo) & (df["date_n"].values[idx_all] <= hi)
    return {k: v[m] for k, v in ps.items()}, idx_all[m]

def apply_odds_filter(df, idx, ps):
    vals = df.iloc[idx].reset_index(drop=True)
    m = vals["R_odds"].notna().values & vals["B_odds"].notna().values
    return vals[m].reset_index(drop=True), {k: np.asarray(v)[m] for k, v in ps.items()}

pri_ps, _, pri_idx = assemble(wf_all, [2020, 2021])
pri_lo, pri_hi = pd.Timestamp("2020-01-01"), pd.Timestamp("2021-03-20")
pri_ps, pri_idx = slice_window(df, pri_idx, pri_ps, pri_lo, pri_hi)
tp, pri_ps = apply_odds_filter(df, pri_idx, pri_ps)
pri_y = tp["yR"].values
print(f"\nprimary test n={len(tp)} (2020-01..2021-03-20, all feats, walk-forward)")

pri_eval = show_eval(tp, pri_ps, pri_y)

ext_ps, _, ext_idx = assemble(wf_core, list(range(2020, 2027)))
ta, ext_ps = apply_odds_filter(df, ext_idx, ext_ps)
ext_y = ta["yR"].values
print(f"\nextended test n={len(ta)} (2020..2026, core feats, walk-forward)")
ext_eval = show_eval(ta, ext_ps, ext_y)

# frozen evals
fpa, frozen_all_preds = apply_odds_filter(df, frozen_all_idx, frozen_all_preds)
frozen_pri_mask = (fpa["date_n"] >= pri_lo) & (fpa["date_n"] <= pri_hi)
frozen_pri = fpa[frozen_pri_mask]
frozen_pri_preds = {k: v[frozen_pri_mask.values] for k, v in frozen_all_preds.items()}
fc, frozen_core_preds = apply_odds_filter(df, frozen_core_idx, frozen_core_preds)

# odds variant evals
odp_ps, _, odp_idx = assemble(wf_odd_all, [2020, 2021])
odp_ps, odp_idx = slice_window(df, odp_idx, odp_ps, pri_lo, pri_hi)
odp, odp_ps = apply_odds_filter(df, odp_idx, odp_ps)
ode_ps, _, ode_idx = assemble(wf_odd_core, list(range(2020, 2027)))
ode, ode_ps = apply_odds_filter(df, ode_idx, ode_ps)

# ---------------- betting ----------------
def betting_block(tag, test_df, p_ens):
    gates, best = gates_table(test_df, p_ens)
    print(f"\n=== betting {tag} ===")
    for eps, g in gates.items():
        print(f"  eps={eps}: n={g['n']:>4} win_rate={g['win_rate']:.3f} flat_ROI={g['roi']:+.4f} profit={g['profit']:+.1f}")
    if best is not None:
        k = kelly_path(test_df, p_ens, best, f_max=0.25)
        print(f"  best gate eps={best} -> qKelly bank={k['final_bank']:.2f} stakeROI={k['roi']} n={k['n_bets']}")
    return gates, best

pri_gates, pri_best = betting_block("PRIMARY (walk-forward ens)", tp, pri_ps["ens"])
ext_gates, ext_best = betting_block("EXTENDED (walk-forward ens)", ta, ext_ps["ens"])
betting_block("EXTENDED ens_gb (lgb+xgb)", ta, ext_ps["ens_gb"])

print("\n=== EXTENDED per-year (walk-forward ens) ===")
per_year = []
for yr in range(2020, 2027):
    m = ta["date_n"].dt.year == yr
    if m.sum() == 0:
        continue
    ysub = ta["yR"].values[m]
    psub = ext_ps["ens"][m]
    g = evaluate_bets(ta[m].reset_index(drop=True), psub, 0.0)
    row = dict(year=yr, n=int(m.sum()), acc=float(accuracy_score(ysub, (psub >= 0.5).astype(int))),
               auc=float(roc_auc_score(ysub, psub)), flat_roi_eps0=g["roi"], n_bets=g["n"])
    per_year.append(row)
    print(f"  {yr}: n={row['n']:>4} acc={row['acc']:.3f} auc={row['auc']:.3f} flatROI(eps0)={row['flat_roi_eps0']:+.3f} (n_bets={g['n']})")

# ---------------- bootstrap ----------------
def flat_roi_metric(d, p, eps):
    g = evaluate_bets(d, p, eps)
    return g["roi"] if g["n"] >= 20 else np.nan

def acc_metric(d, p, eps):
    return accuracy_score(d["yR"].values, (p >= 0.5).astype(int))

print("\nbootstrap (2000 resamples):")
for tag, d, p in [("primary", tp, pri_ps["ens"]), ("extended", ta, ext_ps["ens"])]:
    ci_roi0 = bootstrap_ci(flat_roi_metric, d, p, 0.0, rng=rng)
    ci_acc = bootstrap_ci(acc_metric, d, p, 0.0, rng=rng)
    print(f"  [{tag}] flat ROI eps0: {ci_roi0['mean']:+.4f} [{ci_roi0['lo']:+.4f}, {ci_roi0['hi']:+.4f}] | acc {ci_acc['mean']:.4f} [{ci_acc['lo']:.4f}, {ci_acc['hi']:.4f}]")

# ---------------- CSVs ----------------
def write_csv(fname, test_df, ps, models):
    out = pd.DataFrame({
        "date": test_df["date_n"].dt.strftime("%Y-%m-%d"),
        "red_fighter": test_df["R_fighter"], "blue_fighter": test_df["B_fighter"],
        "weight_class": test_df["weight_class"],
        "R_closing_odds_american": test_df["R_odds"], "B_closing_odds_american": test_df["B_odds"],
        "winner": test_df["Winner"],
        "fair_prob_R": test_df["fairR"], "fair_prob_B": test_df["fairB"],
    })
    for m in models:
        out[f"pR_{m}"] = ps[m]
    edge_R = ps["ens"] - test_df["fairR"].values
    edge_B = (1 - ps["ens"]) - test_df["fairB"].values
    side = np.where(edge_R >= edge_B, "R", "B")
    out["edge_R"], out["edge_B"], out["bet_side_eps0"] = edge_R, edge_B, side
    out["flat_bet_result_eps0"] = np.where(side == "R", (test_df["yR"].values == 1).astype(int),
                                           (test_df["yR"].values == 0).astype(int))
    out.to_csv(os.path.join(base, fname), index=False)
    print("wrote", fname, out.shape)

write_csv("mma_v2_predictions_primary.csv", tp, pri_ps, ["lr", "lgb", "xgb", "ens", "ens_gb"])
write_csv("mma_v2_predictions_extended.csv", ta, ext_ps, ["lr", "lgb", "xgb", "ens", "ens_gb"])

# ---------------- save artifacts ----------------
metrics = {
    "dataset": "mdabbert/ultimate-ufc-dataset + rajeevw/ufcdata (features)",
    "n_train_2020_model": int((df["date_n"] < pd.Timestamp('2020-01-01')).sum()),
    "primary": dict(n=int(len(tp)), eval={k: v for k, v in pri_eval.items() if k not in ("market",)},
                    market=pri_eval["market"],
                    gates={str(eps): {k: g[k] for k in ("n", "win_rate", "roi", "profit")} for eps, g in pri_gates.items()},
                    best_gate_eps=float(pri_best) if pri_best is not None else None),
    "extended": dict(n=int(len(ta)), eval=ext_eval,
                     gates={str(eps): {k: g[k] for k in ("n", "win_rate", "roi", "profit")} for eps, g in ext_gates.items()},
                     best_gate_eps=float(ext_best) if ext_best is not None else None,
                     per_year=per_year),
    "frozen_v2": dict(primary_acc=float(accuracy_score(frozen_pri["yR"].values, (frozen_pri_preds["ens"] >= 0.5).astype(int))),
                      primary_auc=float(roc_auc_score(frozen_pri["yR"].values, frozen_pri_preds["ens"])),
                      primary_brier=brier_score(frozen_pri_preds["ens"], frozen_pri["yR"].values),
                      primary_roi_eps0=evaluate_bets(frozen_pri, frozen_pri_preds["ens"], 0.0)["roi"],
                      core_ext_acc=float(accuracy_score(fc["yR"].values, (frozen_core_preds["ens"] >= 0.5).astype(int))),
                      core_ext_auc=float(roc_auc_score(fc["yR"].values, frozen_core_preds["ens"])),
                      core_ext_brier=brier_score(frozen_core_preds["ens"], fc["yR"].values),
                      core_ext_roi_eps0=evaluate_bets(fc, frozen_core_preds["ens"], 0.0)["roi"]),
    "odds_variant": dict(primary_acc=float(accuracy_score(odp["yR"].values, (odp_ps["ens"] >= 0.5).astype(int))),
                         primary_auc=float(roc_auc_score(odp["yR"].values, odp_ps["ens"])),
                         primary_brier=brier_score(odp_ps["ens"], odp["yR"].values),
                         extended_acc=float(accuracy_score(ode["yR"].values, (ode_ps["ens"] >= 0.5).astype(int))),
                         extended_auc=float(roc_auc_score(ode["yR"].values, ode_ps["ens"])),
                         extended_brier=brier_score(ode_ps["ens"], ode["yR"].values)),
    "hparams": dict(core=hp_core, all=hp_all),
    "features": dict(core=core_feats, strike=[c for c in all_feats if c not in core_feats]),
    "runtime_sec": float(time.time() - t0),
}
with open(os.path.join(base, "metrics_v2_pooled.json"), "w") as f:
    json.dump(metrics, f, indent=1, default=str)
with open(os.path.join(base, "wf_artifacts.pkl"), "wb") as f:
    pickle.dump(dict(df_idx_tp=tp.index.values, pri_ps=pri_ps, ext_ps=ext_ps,
                     frozen_core_preds=frozen_core_preds, hp_core=hp_core, hp_all=hp_all), f)
print("\nwrote metrics_v2_pooled.json + wf_artifacts.pkl")
print(f"\nKEY: primary n={len(tp)} acc={pri_eval['ens']['acc']:.4f} flat_roi0={pri_gates[0.0]['roi']:+.4f} | "
      f"extended n={len(ta)} acc={ext_eval['ens']['acc']:.4f} flat_roi0={ext_gates[0.0]['roi']:+.4f} | "
      f"brier_p={pri_eval['ens']['brier']:.4f} brier_e={ext_eval['ens']['brier']:.4f}")
print(f"runtime {time.time()-t0:.0f}s")
