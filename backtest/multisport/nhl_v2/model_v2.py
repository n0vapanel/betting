"""NHL v2 walk-forward: LogReg + LGBM with 43 features (v1's 19 + goalie,
injuries, schedule density/travel, refined special teams, Kalman strength,
fast Elo). TSCV selection, Platt calibration, edge gates, quarter-Kelly,
bootstrap CI, ablations. Payouts = synthetic 2-way odds from de-vigged probs
@4.76% hold (same rule as v1)."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import lightgbm as lgb

RNG = 42
np.random.seed(RNG)
VIG = 0.0476

df = pd.read_parquet("data/games_features_v2.parquet")
FEATS = [c for c in df.columns if c.startswith("diff_")]
TRAIN_END = pd.Timestamp("2022-02-28")
TEST_START = pd.Timestamp("2022-03-01")

train = df[df.date <= TRAIN_END].copy().reset_index(drop=True)
test = df[df.date >= TEST_START].copy().reset_index(drop=True)
print(f"train: {len(train)} | test: {len(test)} (regular {(test.is_playoff == 0).sum()}, "
      f"playoff {(test.is_playoff == 1).sum()}) | n features: {len(FEATS)}")

Xtr, ytr = train[FEATS].values, train.home_win.values
Xte, yte = test[FEATS].values, test.home_win.values
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
tscv = TimeSeriesSplit(n_splits=5)


def cv_oof(make_model, X, y):
    oof = np.zeros(len(y))
    for tr_idx, va_idx in tscv.split(X):
        m = make_model()
        m.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict_proba(X[va_idx])[:, 1]
    return oof


def mk_logreg(C):
    return LogisticRegression(C=C, max_iter=3000, solver="lbfgs")


def mk_lgbm(lr, leaves, nest, mc):
    return lgb.LGBMClassifier(n_estimators=nest, learning_rate=lr, num_leaves=leaves,
                              subsample=0.9, colsample_bytree=0.9, min_child_samples=mc,
                              reg_lambda=1.0, random_state=RNG, verbosity=-1)


def platt(oof_in, y, pred):
    lo = LogisticRegression(C=1e6, max_iter=3000)
    lo.fit(oof_in.reshape(-1, 1), y)
    return lo.predict_proba(pred.reshape(-1, 1))[:, 1]


# ---- hyperparameter grid by OOF Brier ----
results = {}
for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    results[f"lr_C={C}"] = cv_oof(lambda C=C: mk_logreg(C), Xtr_s, ytr)
for lr_ in [0.03, 0.1]:
    for leaves in [7, 15, 31]:
        for nest in [100, 300]:
            for mc in [10, 20]:
                results[f"lgbm_lr={lr_}_lv={leaves}_ne={nest}_mc={mc}"] = cv_oof(
                    lambda lr_=lr_, leaves=leaves, nest=nest, mc=mc: mk_lgbm(lr_, leaves, nest, mc),
                    Xtr_s, ytr)
briers = {k: brier_score_loss(ytr, v) for k, v in results.items()}
# C selection with CV-noise floor (one-SE-style, train-only): among C's whose OOF
# Brier is within 0.001 of the best, pick the most regularized (largest C).
# Without the floor, a 0.0008-Brier hair selects C=0.003 (below CV noise).
lr_keys = sorted([k for k in results if k.startswith("lr_")],
                 key=lambda k: float(k.split("=")[1]))
best_oof_lr = min(briers[k] for k in lr_keys)
floor = best_oof_lr + 0.001
eligible = [k for k in lr_keys if briers[k] <= floor]
best_lr_key = eligible[-1]  # most regularized among statistically indistinguishable
best_lgb_key = min([k for k in results if k.startswith("lgbm_")], key=lambda k: briers[k])
print(f"C-grid OOF Briers: " + ", ".join(f"C={k.split('=')[1]}:{briers[k]:.4f}" for k in lr_keys))
print(f"OOF Brier: LogReg {best_lr_key} {briers[best_lr_key]:.4f} (noise floor {floor:.4f})"
      f" | LGBM {best_lgb_key} {briers[best_lgb_key]:.4f}"
      f" | market(train) {brier_score_loss(ytr, train.p_mkt_home):.4f}")
oof_lr = results[best_lr_key]
oof_lgb = results[best_lgb_key]


def oof_calibrated(make_model, cal_fn):
    n = len(ytr)
    splits = list(tscv.split(Xtr_s))
    oof_raw = np.zeros(n)
    for tr_idx, va_idx in splits:
        m = make_model()
        m.fit(Xtr_s[tr_idx], ytr[tr_idx])
        oof_raw[va_idx] = m.predict_proba(Xtr_s[va_idx])[:, 1]
    oof_cal = np.zeros(n)
    for tr_idx, va_idx in splits:
        oof_cal[va_idx] = cal_fn(oof_raw[tr_idx], ytr[tr_idx], oof_raw[va_idx])
    return oof_raw, oof_cal


import re
_, oof_lr_cal = oof_calibrated(lambda: mk_logreg(float(best_lr_key.split("=")[1])), platt)
nums = re.findall(r"[\d.]+", best_lgb_key)
_, oof_lgb_cal = oof_calibrated(
    lambda: mk_lgbm(float(nums[0]), int(float(nums[1])), int(float(nums[2])), int(float(nums[3]))),
    platt)

w_best, w_brier_best = None, np.inf
for w in [0, 0.25, 0.5, 0.75, 1.0]:
    b = brier_score_loss(ytr, w * oof_lr_cal + (1 - w) * oof_lgb_cal)
    if b < w_brier_best:
        w_best, w_brier_best = w, b
print(f"ensemble w={w_best} | OOF Brier cal-ens {w_brier_best:.4f} | "
      f"cal-LogReg {brier_score_loss(ytr, oof_lr_cal):.4f} | cal-LGBM {brier_score_loss(ytr, oof_lgb_cal):.4f}")

# ---- refit on full train ----
C_best = float(best_lr_key.split("=")[1])
lr_b = float(nums[0]); lv_b = int(float(nums[1])); ne_b = int(float(nums[2])); mc_b = int(float(nums[3]))
lr_model = mk_logreg(C_best).fit(Xtr_s, ytr)
lgb_model = mk_lgbm(lr_b, lv_b, ne_b, mc_b).fit(Xtr_s, ytr)
p_lr_raw = lr_model.predict_proba(Xte_s)[:, 1]
p_lgb_raw = lgb_model.predict_proba(Xte_s)[:, 1]
p_lr_cal = platt(oof_lr, ytr, p_lr_raw)
p_lgb_cal = platt(oof_lgb, ytr, p_lgb_raw)
p_model = w_best * p_lr_cal + (1 - w_best) * p_lgb_cal

test = test.copy()
test["p_model_home"] = p_model
test["p_lr_home"] = p_lr_cal
test["p_lgb_home"] = p_lgb_cal
test["p_model_away"] = 1 - p_model
test["p_mkt_away"] = 1 - test.p_mkt_home
test["odds_h_2way"] = 1 / (test.p_mkt_home * (1 + VIG))
test["odds_a_2way"] = 1 / (test.p_mkt_away * (1 + VIG))

# ---- evaluation ----
def run_eval(games, label):
    pick_side = np.where(games.p_model_home >= 0.5, "H", "A")
    pick_p = np.where(pick_side == "H", games.p_model_home, games.p_model_away)
    odds = np.where(pick_side == "H", games.odds_h_2way, games.odds_a_2way)
    mkt_p = np.where(pick_side == "H", games.p_mkt_home, games.p_mkt_away)
    win = np.where(pick_side == "H", games.home_win.values == 1, games.home_win.values == 0).astype(float)
    ret = np.where(win == 1, odds - 1.0, -1.0)
    edge = pick_p - mkt_p
    acc = win.mean()
    print(f"\n[{label}] n={len(games)} acc={acc:.4f} flatROI={ret.mean():+.4f} "
          f"Brier={brier_score_loss(games.home_win, games.p_model_home):.4f}")
    rows = {}
    for eps in [0, 0.02, 0.05]:
        bet = edge > eps
        n = bet.sum()
        rows[eps] = dict(n_bets=int(n),
                         roi=float(ret[bet].mean()) if n else None,
                         win_pct=float(win[bet].mean()) if n else None)
        print(f"  model-side gate eps={eps}: n={n} roi={rows[eps]['roi']} win%={rows[eps]['win_pct']}")
    for eps in [0.02, 0.05]:
        e_h = games.p_model_home - games.p_mkt_home
        e_a = games.p_model_away - games.p_mkt_away
        side = np.where((e_h > eps) & (e_h >= e_a), "H", np.where(e_a > eps, "A", "-"))
        bet = side != "-"
        o = np.where(side == "H", games.odds_h_2way, games.odds_a_2way)
        w_ = np.where(side == "H", games.home_win == 1, games.home_win == 0)
        r_ = np.where(w_ == 1, o - 1.0, -1.0)
        n = bet.sum()
        print(f"  either-side gate eps={eps}: n={n} roi={float(r_[bet].mean()) if n else None}")
    b = odds - 1.0
    f = np.asarray(0.25 * (b * pick_p - (1 - pick_p)) / b)
    bank = 1000.0
    for i in range(len(games)):
        if f[i] > 0:
            bank += min(f[i], 0.05) * bank * ret[i]
    print(f"  quarter-Kelly bankroll: {bank:.2f} ({bank/1000 - 1:+.1%})")
    return dict(acc=acc, flat_roi=float(ret.mean()), ret=ret, pick_side=pick_side,
                pick_p=pick_p, odds=odds, win=win, edge=edge, qkelly=bank,
                gates=rows)


m_all = run_eval(test, "TEST Mar-Jun 2022 (all)")
run_eval(test[test.is_playoff == 0], "TEST regular Mar-Apr")
run_eval(test[test.is_playoff == 1], "TEST playoffs May-Jun")

bh = test.home_win.mean()
roi_home = ((test.home_win == 1) * (test.odds_h_2way - 1) - (test.home_win == 0)).mean()
fav = np.where(test.p_mkt_home >= 0.5, "H", "A")
fodds = np.where(fav == "H", test.odds_h_2way, test.odds_a_2way)
fwin = np.where(fav == "H", test.home_win == 1, test.home_win == 0)
roi_fav = (fwin * (fodds - 1) - ~fwin).mean()
print(f"\nbaselines: home win% {bh:.4f} | always-home ROI {roi_home:+.4f} | "
      f"market-fav acc {fwin.mean():.4f} | market-fav ROI {roi_fav:+.4f}")
print(f"Brier: model {brier_score_loss(test.home_win, test.p_model_home):.4f} vs "
      f"market {brier_score_loss(test.home_win, test.p_mkt_home):.4f}")

# ---- ablations ----
v1_feats = ["diff_xg_share_5v5_10", "diff_xg_share_5v5_5", "diff_xg_share_5v5_20",
            "diff_cf_share_5v5", "diff_xg_share_all", "diff_gf_share_all",
            "diff_pdo", "diff_pdo_20", "diff_sh_pct", "diff_sv_pct",
            "diff_pp_eff", "diff_pk_eff", "diff_pen_mins", "diff_fo_pct",
            "diff_gf_net", "diff_pts_pct", "diff_rest", "diff_b2b", "diff_elo_before"]
groups = {
    "v1_features_only": v1_feats,
    "v1_plus_goalie": v1_feats + [c for c in FEATS if c.startswith("diff_gk_")],
    "v1_plus_injuries": v1_feats + [c for c in FEATS if c.startswith("diff_inj_")],
    "v1_plus_sched_density": v1_feats + [c for c in FEATS if
                                         c in ("diff_games7", "diff_games14", "diff_home7",
                                               "diff_away_streak", "diff_home_streak",
                                               "diff_travel", "diff_travel7")],
    "v1_plus_st_refined": v1_feats + [c for c in FEATS if
                                      c in ("diff_pp_eff_20", "diff_pk_eff_20",
                                            "diff_pp_opp_rate", "diff_pk_opp_rate", "diff_pp_net")],
    "v1_plus_kalman_elofast": v1_feats + [c for c in FEATS if
                                          c in ("diff_k_strength", "diff_kx_strength",
                                                "diff_elo_fast_before")],
    "full_model": FEATS,
}
print("\nablations (LogReg C=0.01, Platt cal, 2-way payouts):")
abl = {}
for name, cols in groups.items():
    Xa = scaler.fit_transform(train[cols].values)
    oof = cv_oof(lambda: mk_logreg(0.01), Xa, ytr)
    m = mk_logreg(0.01).fit(Xa, ytr)
    p = platt(oof, ytr, m.predict_proba(scaler.transform(test[cols].values))[:, 1])
    side = np.where(p >= 0.5, "H", "A")
    o = np.where(side == "H", test.odds_h_2way, test.odds_a_2way)
    w_ = np.where(side == "H", test.home_win == 1, test.home_win == 0)
    r_ = np.where(w_ == 1, o - 1.0, -1.0)
    brier_t = brier_score_loss(test.home_win, p)
    abl[name] = dict(n=len(cols), acc=float(w_.mean()), roi=float(r_.mean()), brier=float(brier_t))
    print(f"  {name}: nfeat={len(cols)} acc={w_.mean():.4f} roi={r_.mean():+.4f} brier={brier_t:.4f}")

# ---- bootstrap CI (cluster by date) ----
rng = np.random.default_rng(42)
dates = test.date.values
rois = []
for _ in range(2000):
    d = rng.choice(pd.unique(dates), size=test.date.nunique(), replace=True)
    idx = np.isin(dates, d)
    rois.append(m_all["ret"][idx].mean())
ci = np.percentile(rois, [2.5, 97.5])
print(f"\nflat ROI bootstrap 95% CI (date-clustered): [{ci[0]:+.4f}, {ci[1]:+.4f}]")

# gate CIs
gate_ci = {}
for eps in [0.02, 0.05]:
    bet = m_all["edge"] > eps
    rets = m_all["ret"][bet]
    if len(rets) == 0:
        gate_ci[eps] = None
        continue
    bdates = dates[bet]
    vals = []
    for _ in range(2000):
        d = rng.choice(pd.unique(bdates), size=len(pd.unique(bdates)), replace=True)
        vals.append(rets[np.isin(bdates, d)].mean())
    gate_ci[eps] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    print(f"gate eps={eps} bootstrap CI: [{gate_ci[eps][0]:+.4f}, {gate_ci[eps][1]:+.4f}]")

bins = np.linspace(0, 1, 11)
ece = 0.0
for i in range(10):
    msk = (test.p_model_home >= bins[i]) & (test.p_model_home < bins[i + 1])
    if msk.sum():
        ece += msk.sum() / len(test) * abs(test.home_win[msk].mean() - test.p_model_home[msk].mean())
print(f"ECE: {ece:.5f}")

# ---- save CSV ----
cols = ["date", "game_id", "home_name", "away_name", "is_playoff",
        "home_goals", "away_goals", "home_win",
        "p_model_home", "p_model_away", "p_lr_home", "p_lgb_home",
        "p_mkt_home", "p_mkt_away",
        "home_odds_raw", "away_odds_raw", "odds_h_2way", "odds_a_2way"] + FEATS
out = test.rename(columns={"home_odds": "home_odds_raw", "away_odds": "away_odds_raw"}).copy()
out["model_pick"] = m_all["pick_side"]
out["model_pick_odds"] = m_all["odds"]
out["model_pick_prob"] = m_all["pick_p"]
out["edge"] = m_all["edge"]
out["flat_ret"] = m_all["ret"]
out.to_csv("nhl_v2_predictions.csv", index=False)
print("wrote nhl_v2_predictions.csv", len(out))

metrics = dict(
    n_train=int(len(train)), n_test=int(len(test)),
    n_test_regular=int((test.is_playoff == 0).sum()), n_test_playoff=int((test.is_playoff == 1).sum()),
    test_accuracy=float(m_all["acc"]), flat_roi=float(m_all["flat_roi"]),
    model_brier=float(brier_score_loss(test.home_win, test.p_model_home)),
    market_brier=float(brier_score_loss(test.home_win, test.p_mkt_home)),
    roi_bootstrap_ci=[float(ci[0]), float(ci[1])],
    quarter_kelly_final=float(m_all["qkelly"]),
    gates={str(k): v for k, v in m_all["gates"].items()},
    gate_bootstrap_ci={str(k): v for k, v in gate_ci.items()},
    best_logreg=best_lr_key, best_lgbm=best_lgb_key, ensemble_weight=w_best,
    oof_brier_logreg=briers[best_lr_key], oof_brier_lgbm=briers[best_lgb_key],
    oof_brier_ensemble=w_brier_best, ece=float(ece),
    home_win_pct_test=float(bh), roi_always_home=float(roi_home),
    market_fav_accuracy=float(fwin.mean()), roi_market_fav=float(roi_fav),
    ablations=abl, hold=VIG, n_features=len(FEATS),
)
with open("metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=2)
print(json.dumps({k: metrics[k] for k in ["n_test", "test_accuracy", "flat_roi", "model_brier",
                                          "market_brier", "quarter_kelly_final",
                                          "roi_bootstrap_ci"]}, indent=2))
