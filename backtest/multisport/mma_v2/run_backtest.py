"""UFC walk-forward backtest — main flow (part 1: data, features, CV, fit)."""
import json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import lightgbm as lgb
from backtest_lib import am_to_dec, fair_probs, brier_score

base = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

df = pd.read_parquet(os.path.join(base, "fights_features.parquet"))
df = df[~df["Winner"].isin(["Draw", "No Contest"])].copy()
df["yR"] = (df["Winner"] == "Red").astype(int)

# ---------- features ----------
SIDE_CONT = [
    "elo_before", "n_prior", "w_prior", "l_prior", "streak_pre",
    "longest_run_so_far", "days_since",
]
for s in ["R", "B"]:
    df[f"win_pct_{s}"] = (df[f"w_prior_{s}"] + 1.0) / (df[f"n_prior_{s}"] + 2.0)
    df[f"days_since_{s}"] = df[f"days_since_{s}"].fillna(999.0)
    df[f"elo_before_{s}"] = df[f"elo_before_{s}"].fillna(1500.0)
    df[f"streak_pre_{s}"] = df[f"streak_pre_{s}"].fillna(0.0)
    df[f"longest_run_so_far_{s}"] = df[f"longest_run_so_far_{s}"].fillna(0.0)
    df[f"n_prior_{s}"] = df[f"n_prior_{s}"].fillna(0.0)
    df[f"w_prior_{s}"] = df[f"w_prior_{s}"].fillna(0.0)
    df[f"l_prior_{s}"] = df[f"l_prior_{s}"].fillna(0.0)

CORE_DIFF_PAIRS = [
    ("elo_before", "elo"),
    ("win_pct", "win_pct"),
    ("streak_pre", "streak"),
    ("longest_run_so_far", "longest_streak"),
    ("n_prior", "exp"),
    ("days_since", "days_since"),
]
STRIKE_DIFFS = ["slpm", "str_acc", "td_acc", "td_def", "sub_per_f", "kd_per_f", "ctrl_share"]

feat_names = []
for col, name in CORE_DIFF_PAIRS:
    df[f"d_{name}"] = df[f"{col}_R"] - df[f"{col}_B"]
    feat_names.append(f"d_{name}")

df["d_age"] = df["R_age"] - df["B_age"]; feat_names.append("d_age")
df["d_reach"] = df["R_Reach_cms"] - df["B_Reach_cms"]; feat_names.append("d_reach")
df["d_height"] = df["R_Height_cms"] - df["B_Height_cms"]; feat_names.append("d_height")
df["d_weight"] = df["R_Weight_lbs"] - df["B_Weight_lbs"]; feat_names.append("d_weight")
df["d_rank"] = df["B_match_weightclass_rank"].fillna(16) - df["R_match_weightclass_rank"].fillna(16)
feat_names.append("d_rank")

for c in STRIKE_DIFFS:
    df[f"d_{c}"] = df[f"feat_{c}_R"] - df[f"feat_{c}_B"]
    feat_names.append(f"d_" + c)

cat_names = []
df["is_female"] = (df["gender"] == "FEMALE").astype(int); cat_names.append("is_female")
df["is_title"] = df["title_bout"].astype(int); cat_names.append("is_title")
df["is_5rd"] = (df["no_of_rounds"] == 5).astype(int); cat_names.append("is_5rd")
wc_dummies = pd.get_dummies(df["weight_class"], prefix="wc")
df = pd.concat([df, wc_dummies.astype(int)], axis=1)
cat_names += list(wc_dummies.columns)

core_feats = feat_names[:11] + cat_names          # everything except strike diffs
all_feats = feat_names + cat_names
print("n core feats:", len(core_feats), "| n all feats:", len(all_feats))

# ---------- split ----------
TRAIN_LO, TRAIN_HI = pd.Timestamp("2016-01-01"), pd.Timestamp("2020-01-01")
TEST_LO = pd.Timestamp("2020-01-01")
PRIMARY_HI = pd.Timestamp("2021-03-20")
train = df[(df["date_n"] >= TRAIN_LO) & (df["date_n"] < TRAIN_HI)].copy()
test_all = df[df["date_n"] >= TEST_LO].copy()
test_primary = test_all[test_all["date_n"] <= PRIMARY_HI].copy()
print("train:", len(train), "| test primary (2020-01..2021-03):", len(test_primary),
      "| test extended (2020..2026):", len(test_all))

# odds for betting
for d in (train, test_primary, test_all):
    d["decR"] = am_to_dec(d["R_odds"]); d["decB"] = am_to_dec(d["B_odds"])
    fr, fb = fair_probs(d["decR"], d["decB"])
    d["fairR"], d["fairB"] = fr, fb
test_primary = test_primary[test_primary["R_odds"].notna() & test_primary["B_odds"].notna()].copy()
test_all = test_all[test_all["R_odds"].notna() & test_all["B_odds"].notna()].copy()
print("primary test with odds:", len(test_primary), "| extended with odds:", len(test_all))

# ---------- expanding CV ----------
fold_bounds = [(TRAIN_LO, pd.Timestamp("2017-01-01")),
               (TRAIN_LO, pd.Timestamp("2018-01-01")),
               (TRAIN_LO, pd.Timestamp("2019-01-01"))]
val_bounds = [(pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")),
              (pd.Timestamp("2018-01-01"), pd.Timestamp("2019-01-01")),
              (pd.Timestamp("2019-01-01"), TRAIN_HI)]

def cv_fit_eval(model_fn, X_tr_all, y_tr_all, dates, make_pred):
    """model_fn() -> fresh model. Returns (oof_preds, mean_val_logloss, best_model, extra)."""
    oof = np.full(len(X_tr_all), np.nan)
    losses = []
    iters = []
    for i, ((tlo, thi), (vlo, vhi)) in enumerate(zip(fold_bounds, val_bounds)):
        tr_idx = (dates >= tlo) & (dates < thi)
        va_idx = (dates >= vlo) & (dates < vhi)
        model = model_fn()
        pv = make_pred(model, X_tr_all[tr_idx], y_tr_all[tr_idx], X_tr_all[va_idx], y_tr_all[va_idx])
        oof[va_idx] = pv
        losses.append(log_loss(y_tr_all[va_idx], pv))
        iters.append(getattr(model, "best_iteration_", None))
    return oof, float(np.mean(losses)), losses, iters

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

X_all_train = train[all_feats].values
X_core_train = train[core_feats].values
y_train = train["yR"].values
d_train = train["date_n"].values
results = {}

for tag, X, feats in [("all", X_all_train, all_feats), ("core", X_core_train, core_feats)]:
    print(f"\n=== CV for feature set: {tag} ===")
    # LogReg grid
    best_lr, best_lr_score, best_lr_oof = None, np.inf, None
    for C in [0.03, 0.1, 0.3, 1.0, 3.0]:
        oof, score, _, _ = cv_fit_eval(lambda: make_lr(C), X, y_train, d_train, make_lr_pred)
        print(f"  LR C={C}: val logloss {score:.4f}")
        if score < best_lr_score:
            best_lr_score, best_lr_oof, best_lr = score, oof, C
    # LGBM grid
    best_gbm, best_gbm_score, best_gbm_oof, best_gbm_cfg = None, np.inf, None, None
    for leaves in [15, 31]:
        for lr in [0.03, 0.06]:
            oof, score, _, _ = cv_fit_eval(lambda: make_lgb(leaves, lr), X, y_train, d_train, make_lgb_pred)
            print(f"  LGBM leaves={leaves} lr={lr}: val logloss {score:.4f}")
            if score < best_gbm_score:
                best_gbm_score, best_gbm_oof, best_gbm_cfg = score, oof, (leaves, lr)
    results[tag] = dict(best_lr=best_lr, best_lr_score=best_lr_score, best_lr_oof=best_lr_oof,
                        best_gbm_cfg=best_gbm_cfg, best_gbm_score=best_gbm_score,
                        best_gbm_oof=best_gbm_oof, feats=feats)
    print(f"  -> LR C={best_lr} ({best_lr_score:.4f}), LGBM {best_gbm_cfg} ({best_gbm_score:.4f})")
    print(f"  NaN in OOF: LR={np.isnan(best_lr_oof).sum()} LGBM={np.isnan(best_gbm_oof).sum()}")
    print(f"  NaN in y: {np.isnan(y_train).sum()}")
    print(f"  NaN in X cols: {np.isnan(X).sum(axis=0)[:11]}")
    # OOF AUC (2016 rows never appear in a val fold -> their OOF stays NaN)
    v_lr = ~np.isnan(best_lr_oof)
    v_gb = ~np.isnan(best_gbm_oof)
    print(f"  OOF AUC  LR: {roc_auc_score(y_train[v_lr], best_lr_oof[v_lr]):.4f} | LGBM: {roc_auc_score(y_train[v_gb], best_gbm_oof[v_gb]):.4f}")
    results[tag]["v_lr"] = v_lr

mkt_mask = d_train >= pd.Timestamp("2017-01-01")
print(f"\nmarket (de-vigged closing) on 2017-2019 train rows: logloss {log_loss(y_train[mkt_mask], train['fairR'].values[mkt_mask]):.4f}, "
      f"brier {brier_score(train['fairR'].values[mkt_mask], y_train[mkt_mask]):.4f}, "
      f"acc {accuracy_score(y_train[mkt_mask], (train['fairR'].values[mkt_mask] >= 0.5).astype(int)):.4f}")

# ---------- final fits on full train + calibration ----------
def fit_final(tag):
    r = results[tag]
    X = train[r["feats"]].values
    # LR
    lr = make_lr(r["best_lr"])
    lr.fit(X, y_train)
    valid = ~np.isnan(r["best_lr_oof"])
    iso_lr = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    iso_lr.fit(r["best_lr_oof"][valid], y_train[valid])
    # LGBM: refit with early stopping on last 20% of train
    leaves, lr_rate = r["best_gbm_cfg"]
    n = len(X)
    cut = int(n * 0.8)
    gbm = make_lgb(leaves, lr_rate)
    gbm.fit(X[:cut], y_train[:cut], eval_set=[(X[cut:], y_train[cut:])], eval_metric="logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    valid_g = ~np.isnan(r["best_gbm_oof"])
    iso_gbm = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    iso_gbm.fit(r["best_gbm_oof"][valid_g], y_train[valid_g])
    return lr, iso_lr, gbm, iso_gbm

models = {tag: fit_final(tag) for tag in ["all", "core"]}
print("\nfinal fits done")
print(json.dumps({t: dict(lr_C=results[t]["best_lr"], gbm=results[t]["best_gbm_cfg"],
                          cv_lr=results[t]["best_lr_score"], cv_gbm=results[t]["best_gbm_score"]) for t in results}, indent=1))

# ---------- evaluation ----------
def predict_set(tag, d):
    X = d[results[tag]["feats"]].values
    lr, iso_lr, gbm, iso_gbm = models[tag]
    p_lr = iso_lr.predict(lr.predict_proba(X)[:, 1])
    p_gbm = iso_gbm.predict(gbm.predict_proba(X)[:, 1])
    return p_lr, p_gbm

def evaluate(d, tag):
    p_lr, p_gbm = predict_set(tag, d)
    p_ens = (p_lr + p_gbm) / 2.0
    y = d["yR"].values
    acc = {}
    for nm, p in [("lr", p_lr), ("gbm", p_gbm), ("ens", p_ens)]:
        acc[nm] = float(accuracy_score(y, (p >= 0.5).astype(int)))
    acc["market_fav"] = float(accuracy_score(y, (d["fairR"].values >= 0.5).astype(int)))
    auc = {nm: float(roc_auc_score(y, p)) for nm, p in [("lr", p_lr), ("gbm", p_gbm), ("ens", p_ens)]}
    brier = {nm: brier_score(p, y) for nm, p in [("lr", p_lr), ("gbm", p_gbm), ("ens", p_ens)]}
    brier["market"] = brier_score(d["fairR"].values, y)
    return p_lr, p_gbm, p_ens, acc, auc, brier

print("\n=== PRIMARY test (2020-01 -> 2021-03-20, strike features) ===")
p_lr1, p_gbm1, p_ens1, acc1, auc1, brier1 = evaluate(test_primary, "all")
print("acc:", {k: round(v, 4) for k, v in acc1.items()})
print("auc:", {k: round(v, 4) for k, v in auc1.items()})
print("brier:", {k: round(v, 4) for k, v in brier1.items()})

print("\n=== EXTENDED test (2020-01 -> 2026-03, core features) ===")
p_lr2, p_gbm2, p_ens2, acc2, auc2, brier2 = evaluate(test_all, "core")
print("acc:", {k: round(v, 4) for k, v in acc2.items()})
print("auc:", {k: round(v, 4) for k, v in auc2.items()})
print("brier:", {k: round(v, 4) for k, v in brier2.items()})

# ---------- betting ----------
from backtest_lib import evaluate_bets, kelly_path, bootstrap_ci

def betting_summary(d, p):
    gates = {}
    for eps in [0.0, 0.02, 0.05]:
        gates[eps] = evaluate_bets(d, p, eps)
    rois = {eps: g["roi"] for eps, g in gates.items() if g["n"] >= 20}
    best_eps = max(rois, key=lambda e: rois[e])
    kelly = kelly_path(d, p, best_eps, f_max=0.25)
    return gates, best_eps, kelly

print("\n=== betting: PRIMARY (ensemble prob) ===")
g1, best1, kell1 = betting_summary(test_primary, p_ens1)
for eps, g in g1.items():
    print(f"  eps={eps}: n={g['n']:>4}  win_rate={g['win_rate']:.3f}  flat_ROI={g['roi']:+.4f}  profit={g['profit']:+.1f}")
print(f"  best gate eps={best1} -> q-Kelly: final_bank={kell1['final_bank']:.2f} ROI={kell1['roi']} n={kell1['n_bets']}")

print("\n=== betting: EXTENDED (ensemble prob) ===")
g2, best2, kell2 = betting_summary(test_all, p_ens2)
for eps, g in g2.items():
    print(f"  eps={eps}: n={g['n']:>4}  win_rate={g['win_rate']:.3f}  flat_ROI={g['roi']:+.4f}  profit={g['profit']:+.1f}")
print(f"  best gate eps={best2} -> q-Kelly: final_bank={kell2['final_bank']:.2f} ROI={kell2['roi']} n={kell2['n_bets']}")

print("\n=== EXTENDED per-year (ensemble) ===")
per_year = []
for yr in range(2020, 2027):
    m = test_all["date_n"].dt.year == yr
    if m.sum() == 0:
        continue
    ysub = test_all["yR"].values[m]
    psub = p_ens2[m]
    g = evaluate_bets(test_all[m].reset_index(drop=True), psub, 0.0)
    row = dict(year=yr, n=int(m.sum()), acc=float(accuracy_score(ysub, (psub >= 0.5).astype(int))),
               auc=float(roc_auc_score(ysub, psub)), flat_roi_eps0=g["roi"], n_bets=g["n"])
    per_year.append(row)
    print(f"  {yr}: n={m.sum():>4} acc={row['acc']:.3f} auc={row['auc']:.3f} flatROI(eps0)={row['flat_roi_eps0']:+.3f} (n_bets={g['n']})")

# ---------- bootstrap CIs (primary) ----------
def flat_roi_metric(d, p, eps):
    g = evaluate_bets(d, p, eps)
    return g["roi"] if g["n"] >= 20 else np.nan

def acc_metric(d, p, eps):
    return accuracy_score(d["yR"].values, (p >= 0.5).astype(int))

ci_roi0 = bootstrap_ci(flat_roi_metric, test_primary, p_ens1, 0.0, rng=rng)
ci_roi_best = bootstrap_ci(flat_roi_metric, test_primary, p_ens1, best1, rng=rng)
ci_acc = bootstrap_ci(acc_metric, test_primary, p_ens1, 0.0, rng=rng)
print("\nbootstrap (2000):")
print(f"  flat ROI eps=0: mean={ci_roi0['mean']:+.4f} 95% CI [{ci_roi0['lo']:+.4f}, {ci_roi0['hi']:+.4f}]")
print(f"  flat ROI best gate ({best1}): mean={ci_roi_best['mean']:+.4f} 95% CI [{ci_roi_best['lo']:+.4f}, {ci_roi_best['hi']:+.4f}]")
print(f"  accuracy: mean={ci_acc['mean']:.4f} 95% CI [{ci_acc['lo']:.4f}, {ci_acc['hi']:.4f}]")

# ---------- calibration sanity (binned, primary) ----------
bins = pd.cut(p_ens1, np.arange(0, 1.001, 0.1))
cal = pd.DataFrame({"p": p_ens1, "y": test_primary["yR"].values}).groupby(bins, observed=True).agg(
    n=("y", "size"), winrate=("y", "mean"), pmean=("p", "mean"))
print("\ncalibration (ensemble, primary):")
print(cal.round(3).to_string())

# ---------- write mma_predictions.csv (primary test) ----------
tp = test_primary.reset_index(drop=True)
edge_R = p_ens1 - tp["fairR"].values
edge_B = (1 - p_ens1) - tp["fairB"].values
side = np.where(edge_R >= edge_B, "R", "B")
best_gate = g1[best1]
best_picks_full = np.full(len(tp), "", dtype=object)
best_picks_full[best_gate["bet_mask"]] = np.where(best_gate["picks"] == 0, "R", "B")
out = pd.DataFrame({
    "date": tp["date_n"].dt.strftime("%Y-%m-%d"),
    "red_fighter": tp["R_fighter"], "blue_fighter": tp["B_fighter"],
    "weight_class": tp["weight_class"],
    "R_closing_odds_american": tp["R_odds"], "B_closing_odds_american": tp["B_odds"],
    "winner": tp["Winner"],
    "pR_logreg": p_lr1, "pR_lightgbm": p_gbm1, "pR_ensemble": p_ens1,
    "fair_prob_R": tp["fairR"], "fair_prob_B": tp["fairB"],
    "edge_R": edge_R, "edge_B": edge_B,
    "bet_side_eps0": side,
    "bet_side_best_gate": best_picks_full,
    "flat_bet_result_eps0": np.where(side == "R", (tp["yR"] == 1).astype(int), (tp["yR"] == 0).astype(int)),
})
out.to_csv(os.path.join(base, "mma_predictions.csv"), index=False)
print("\nwrote mma_predictions.csv", out.shape)

# extended predictions CSV (core-feature model)
ta = test_all.reset_index(drop=True)
edge_Ra = p_ens2 - ta["fairR"].values
edge_Ba = (1 - p_ens2) - ta["fairB"].values
side_a = np.where(edge_Ra >= edge_Ba, "R", "B")
out2 = pd.DataFrame({
    "date": ta["date_n"].dt.strftime("%Y-%m-%d"),
    "red_fighter": ta["R_fighter"], "blue_fighter": ta["B_fighter"],
    "weight_class": ta["weight_class"],
    "R_closing_odds_american": ta["R_odds"], "B_closing_odds_american": ta["B_odds"],
    "winner": ta["Winner"],
    "pR_logreg": p_lr2, "pR_lightgbm": p_gbm2, "pR_ensemble": p_ens2,
    "fair_prob_R": ta["fairR"], "fair_prob_B": ta["fairB"],
    "edge_R": edge_Ra, "edge_B": edge_Ba,
    "bet_side_eps0": side_a,
})
out2.to_csv(os.path.join(base, "mma_predictions_extended.csv"), index=False)
print("wrote mma_predictions_extended.csv", out2.shape)

g_lr_only = evaluate_bets(test_primary, p_lr1, 0.0)
g_gbm_only = evaluate_bets(test_primary, p_gbm1, 0.0)

metrics = {
    "dataset": "mdabbert/ultimate-ufc-dataset",
    "odds_source": "BestFightOdds closing American odds (R_odds/B_odds)",
    "train_window": ["2016-01-01", "2019-12-31"],
    "primary_test_window": ["2020-01-01", "2021-03-20"],
    "extended_test_window": ["2020-01-01", "2026-03-28"],
    "n_train": int(len(train)), "n_test_primary": int(len(test_primary)),
    "n_test_extended": int(len(test_all)),
    "primary": dict(acc=acc1, auc=auc1, brier=brier1,
                    gates={str(eps): {k: g[k] for k in ("n", "win_rate", "roi", "profit")} for eps, g in g1.items()},
                    best_gate_eps=float(best1),
                    kelly=kell1, ci_roi0=ci_roi0, ci_roi_best=ci_roi_best, ci_acc=ci_acc,
                    per_model_flat_eps0=dict(lr={k: g_lr_only[k] for k in ("n", "win_rate", "roi", "profit")},
                                             gbm={k: g_gbm_only[k] for k in ("n", "win_rate", "roi", "profit")}),
                    cal=cal.rename(index=lambda i: str(i)).round(3).astype(str).to_dict()),
    "extended": dict(acc=acc2, auc=auc2, brier=brier2,
                     gates={str(eps): {k: g[k] for k in ("n", "win_rate", "roi", "profit")} for eps, g in g2.items()},
                     best_gate_eps=float(best2), kelly=kell2, per_year=per_year),
    "cv": {t: dict(lr_C=results[t]["best_lr"], lr_cv_logloss=results[t]["best_lr_score"],
                   gbm_cfg=results[t]["best_gbm_cfg"], gbm_cv_logloss=results[t]["best_gbm_score"],
                   oof_lr_auc=float(roc_auc_score(y_train[~np.isnan(results[t]["best_lr_oof"])],
                                                 results[t]["best_lr_oof"][~np.isnan(results[t]["best_lr_oof"])])),
                   oof_gbm_auc=float(roc_auc_score(y_train[~np.isnan(results[t]["best_gbm_oof"])],
                                                   results[t]["best_gbm_oof"][~np.isnan(results[t]["best_gbm_oof"])]))) for t in results},
    "features": dict(core=feat_names[:11], strike=["d_" + c for c in STRIKE_DIFFS], categorical=cat_names),
}
with open(os.path.join(base, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=1, default=str)
print("wrote metrics.json")
print("\nKEY: n_test=%d acc=%.4f flat_roi(eps0)=%+.4f best_gate=%s best_roi=%+.4f kelly_bank=%.2f" % (
    len(test_primary), acc1["ens"], g1[0.0]["roi"], best1, g1[best1]["roi"], kell1["final_bank"]))

