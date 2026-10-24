"""MLB walk-forward betting backtest.

Walk-forward protocol (mirrors NBA v3 harness):
  train1: 2021 season + Mar-Apr 2022 through Apr 15  ->  validation: Apr 16-30 2022 (gate selection)
  train2: 2021 season + Mar-Apr 2022 (full)          ->  test:        May 1 - Jun 30 2022
Strictly no-lookahead rolling features (see features.py).
Models: LogisticRegression + LightGBM, time-ordered CV, isotonic/platt calibration,
        equal-weight ensemble.
Evaluation: accuracy, flat $1 ROI on predicted winner at closing ML, lean/winner edge
        gates (eps 0/0.01/0.02/0.03/0.05), quarter-Kelly on best gate,
        bootstrap 95% CI (cluster by date), model vs market Brier.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss
import lightgbm as lgb

warnings.filterwarnings("ignore")

EPS_GRID = [0.0, 0.01, 0.02, 0.03, 0.05]
N_BOOT = 1000
SEED = 202203
VAL_LO, VAL_HI = "2022-04-16", "2022-04-30"
TEST_LO, TEST_HI = "2022-05-01", "2022-06-30"

f = pd.read_parquet("features.parquet")
f["date"] = pd.to_datetime(f.date)
f = f.rename(columns={"home_win": "home_won"})

DIFF_FEATS = ["elo_diff", "rs_diff", "ra_diff", "rs_l10_diff", "ra_l10_diff",
              "winpct_diff", "obp_diff", "slg_diff", "krate_diff", "bbrate_diff",
              "ewrs_diff", "ewra_diff", "rest_diff", "inn3_diff",
              "xtra3_diff", "relw_diff", "rell_diff", "sp_exp_diff", "sp_wpct_diff",
              "sp_ra_diff", "sp_rest_diff"]
CTX_FEATS = ["day_night", "doubleheader"]
RAW_FEATS = ["elo_h", "elo_a", "rs_h", "rs_a", "ra_h", "ra_a", "rs_h_l10", "rs_a_l10",
             "ra_h_l10", "ra_a_l10", "winpct_h", "winpct_a", "obp_h", "obp_a",
             "slg_h", "slg_a", "krate_h", "krate_a", "bbrate_h", "bbrate_a",
             "ewrs_h", "ewrs_a", "ewra_h", "ewra_a", "rest_h", "rest_a", "inn3_h",
             "inn3_a", "xtra3_h", "xtra3_a", "relw_h", "relw_a", "rell_h", "rell_a",
             "sp_exp_h", "sp_exp_a", "sp_wpct_h", "sp_wpct_a", "sp_ra_h", "sp_ra_a",
             "sp_rest_h", "sp_rest_a", "games_h", "games_a"]

LOGR_FEATS = DIFF_FEATS + CTX_FEATS
LGBM_FEATS = DIFF_FEATS + RAW_FEATS + CTX_FEATS + ["park"]

y = f.home_won.to_numpy(dtype=float)


def split(mask_fn):
    tr = f[mask_fn("train")]
    va = f[mask_fn("val")]
    te = f[mask_fn("test")]
    return tr, va, te


# ---------------- model training (time-ordered CV + calibration) ----------------
def make_logreg(c=1.0):
    return Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=c, random_state=SEED)),
    ])


def make_lgbm(num_leaves=31, lr=0.03):
    return lgb.LGBMClassifier(
        n_estimators=400, learning_rate=lr, num_leaves=num_leaves, min_child_samples=40,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
        n_jobs=4, verbosity=-1,
    )


def cv_oof_and_train(train, X_cols, model_factory, is_lgbm=False):
    """TimeSeriesSplit OOF probs on train, then refit on full train. Returns (oof, fit_model, covered)."""
    X = train[X_cols].copy()
    if is_lgbm:
        X["park"] = X["park"].astype("category")
    yt = train.home_won.to_numpy()
    tscv = TimeSeriesSplit(n_splits=5)
    oof = np.full(len(train), np.nan)
    covered = np.zeros(len(train), dtype=bool)
    for tr_idx, va_idx in tscv.split(X):
        m = model_factory()
        if is_lgbm:
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            m.fit(Xtr, yt[tr_idx], categorical_feature=["park"])
        else:
            m.fit(X.iloc[tr_idx], yt[tr_idx])
        oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
        covered[va_idx] = True
    m = model_factory()
    if is_lgbm:
        m.fit(X, yt, categorical_feature=["park"])
    else:
        m.fit(X, yt)
    return oof, m, covered


def predict(m, X, is_lgbm=False):
    Xp = X.copy()
    if is_lgbm:
        Xp["park"] = Xp["park"].astype("category")
    return m.predict_proba(Xp)[:, 1]


def calibrate(oof, y_tr, platt_fallback=True):
    """Fit isotonic on OOF; fall back to Platt (LogReg on logit) if isotonic degenerates."""
    oof = np.nan_to_num(np.asarray(oof, dtype=float), nan=0.5)
    oof_c = np.clip(oof, 1e-6, 1 - 1e-6)
    iso = IsotonicRegression(out_of_bounds="clip")
    try:
        iso.fit(oof_c, y_tr)
        # degenerate check
        cal = iso.predict(oof_c)
        if np.std(cal) < 1e-6:
            raise ValueError("degenerate isotonic")
        return ("isotonic", iso)
    except Exception:
        logit = np.log(oof_c / (1 - oof_c)).reshape(-1, 1)
        pl = LogisticRegression(C=10.0, max_iter=2000)
        pl.fit(logit, y_tr)
        return ("platt", pl)


def apply_cal(cal, p):
    kind, mdl = cal
    p_c = np.clip(np.nan_to_num(np.asarray(p, dtype=float), nan=0.5), 1e-6, 1 - 1e-6)
    if kind == "none":
        return p_c
    if kind == "isotonic":
        return mdl.predict(p_c)
    return mdl.predict_proba(np.log(p_c / (1 - p_c)).reshape(-1, 1))[:, 1]


# ---------------- v3 evaluation port ----------------
def bet_frame(frame, probability, variant, eps=0.0):
    p = np.asarray(probability, dtype=float)
    market = frame.p_market_home.to_numpy(dtype=float)
    difference = p - market
    predicted_home = p > 0.5
    if variant == "flat":
        home = predicted_home
        active = np.ones(len(frame), dtype=bool)
    elif variant == "lean":
        home = difference > 0
        active = (np.abs(difference) >= eps) & (difference != 0)
    elif variant == "winner":
        home = predicted_home
        edge = np.where(home, difference, -difference)
        active = (edge >= eps) & (edge > 0)
    else:
        raise ValueError(f"Unknown gate {variant}")
    decimal = np.where(home, frame.dec_home.to_numpy(), frame.dec_away.to_numpy())
    won = home == frame.home_won.to_numpy(dtype=bool)
    return pd.DataFrame({
        "date": frame.date.to_numpy(), "bet": active, "side": np.where(home, "home", "away"),
        "p_side": np.where(home, p, 1 - p), "decimal": decimal, "won": won,
        "edge": np.where(home, difference, -difference),
        "pnl": np.where(active, np.where(won, decimal - 1, -1), 0.0),
    })


def bet_metrics(bets):
    active = bets.loc[bets.bet]
    n = len(active)
    return {"n_bets": n,
            "win_rate": float(active.won.mean()) if n else None,
            "roi": float(active.pnl.sum() / n) if n else None,
            "pnl": float(active.pnl.sum())}


def evaluate(frame, probability, label):
    p = np.asarray(probability, dtype=float)
    yf = frame.home_won.to_numpy(dtype=float)
    flat = bet_metrics(bet_frame(frame, p, "flat"))
    return {
        "label": label, "n_games": len(frame), "accuracy": flat["win_rate"],
        "flat_roi": flat["roi"], "flat_pnl": flat["pnl"],
        "brier": float(np.mean((p - yf) ** 2)),
        "market_brier": float(np.mean((frame.p_market_home.to_numpy(dtype=float) - yf) ** 2)),
        "log_loss": float(np.mean(-yf * np.log(np.clip(p, 1e-6, 1 - 1e-6))
                                  - (1 - yf) * np.log1p(-np.clip(p, 1e-6, 1 - 1e-6)))),
        "market_accuracy": float(((frame.p_market_home.to_numpy() > 0.5) == frame.home_won.to_numpy(bool)).mean()),
    }


def gate_table(frame, probability):
    return [
        {"variant": variant, "eps": eps, **bet_metrics(bet_frame(frame, probability, variant, eps))}
        for variant in ["lean", "winner"] for eps in EPS_GRID
    ]


def choose_gate(validation, p_val):
    candidates = gate_table(validation, p_val)
    eligible = [c for c in candidates if c["n_bets"] >= 30]
    if eligible:
        best = min(eligible, key=lambda c: (-c["roi"], -c["n_bets"], c["eps"], c["variant"]))
        return {**best, "selection_period": f"{VAL_LO}..{VAL_HI}", "minimum_bets": 30,
                "fallback": False}, candidates
    fallback = next(c for c in candidates if c["variant"] == "winner" and c["eps"] == 0)
    return {**fallback, "selection_period": f"{VAL_LO}..{VAL_HI}", "minimum_bets": 30,
            "fallback": True}, candidates


def quarter_kelly(bets):
    active = bets.loc[bets.bet].copy()
    bankroll, total_staked, n_staked = 1.0, 0.0, 0
    peak, max_drawdown = 1.0, 0.0
    for date, group in active.groupby("date", sort=True):
        decimal = group.decimal.to_numpy(dtype=float)
        p = group.p_side.to_numpy(dtype=float)
        fraction = np.clip(0.25 * (p * decimal - 1) / (decimal - 1), 0, 0.05)
        if fraction.sum() > 1:
            fraction /= fraction.sum()
        stakes = bankroll * fraction
        returns = np.where(group.won.to_numpy(dtype=bool), decimal - 1, -1)
        total_staked += float(stakes.sum())
        n_staked += int((stakes > 0).sum())
        bankroll += float(stakes @ returns)
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, 1 - bankroll / peak)
    return {
        "bankroll_multiple": bankroll, "net_return": bankroll - 1, "total_staked": total_staked,
        "roi_on_staked": (bankroll - 1) / total_staked if total_staked else None,
        "n_staked": n_staked, "n_gate_bets": len(active), "max_daily_drawdown": max_drawdown,
        "policy": "Quarter Kelly, 0-5% per bet; simultaneous games use start-of-Eastern-date bankroll; no within-day recycling.",
    }


def bootstrap_intervals(frame, probability, variant, eps, n_boot=N_BOOT):
    p = np.asarray(probability, dtype=float)
    flat = bet_frame(frame, p, "flat")
    gated = bet_frame(frame, p, variant, eps)
    yb = frame.home_won.to_numpy(dtype=float)
    work = pd.DataFrame({
        "date": frame.date.to_numpy(), "n": 1, "correct": flat.won.to_numpy(dtype=float),
        "flat_pnl": flat.pnl.to_numpy(), "brier": (p - yb) ** 2,
        "market_brier": (frame.p_market_home.to_numpy(dtype=float) - yb) ** 2,
        "gate_n": gated.bet.to_numpy(dtype=int), "gate_pnl": gated.pnl.to_numpy(),
        "gate_wins": (gated.bet & gated.won).to_numpy(dtype=int),
    })
    daily = work.groupby("date", sort=True).sum()
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(daily), size=(n_boot, len(daily)))
    sums = daily.to_numpy(dtype=float)[sampled].sum(axis=1)
    index = {c: i for i, c in enumerate(daily.columns)}
    n, gate_n = sums[:, index["n"]], sums[:, index["gate_n"]]
    gate_roi = np.divide(sums[:, index["gate_pnl"]], gate_n, out=np.full(n_boot, np.nan), where=gate_n > 0)
    gate_win = np.divide(sums[:, index["gate_wins"]], gate_n, out=np.full(n_boot, np.nan), where=gate_n > 0)
    values = {
        "accuracy": sums[:, index["correct"]] / n, "flat_roi": sums[:, index["flat_pnl"]] / n,
        "brier": sums[:, index["brier"]] / n, "market_brier": sums[:, index["market_brier"]] / n,
        "brier_minus_market": (sums[:, index["brier"]] - sums[:, index["market_brier"]]) / n,
        "gate_roi": gate_roi, "gate_win_rate": gate_win,
    }
    result = {}
    for key, samples in values.items():
        fin = samples[np.isfinite(samples)]
        result[key] = np.percentile(fin, [2.5, 97.5]).tolist() if len(fin) else [None, None]
    result["n_resamples"] = n_boot
    result["n_dates"] = len(daily)
    result["method"] = "Percentile bootstrap of whole Eastern calendar dates; fixed fitted model and fixed gate."
    return result


# ---------------- run pipeline ----------------
def run_phase(tr_mask, va_mask, te_mask, phase, cfg):
    tr = f[tr_mask].reset_index(drop=True)
    va = f[va_mask].reset_index(drop=True)
    te = f[te_mask].reset_index(drop=True)

    lr_factory = lambda: make_logreg(cfg["lr_c"])
    lgb_factory = lambda: make_lgbm(cfg["lgb_num_leaves"], cfg["lgb_lr"])
    oof_lr, m_lr, cov_lr = cv_oof_and_train(tr, LOGR_FEATS, lr_factory)
    oof_lgb, m_lgb, cov_lgb = cv_oof_and_train(tr, LGBM_FEATS, lgb_factory, is_lgbm=True)
    covered = cov_lr & cov_lgb
    oof_ens = cfg["w_lgb"] * oof_lgb + (1 - cfg["w_lgb"]) * oof_lr
    oof_c = oof_ens[covered]
    y_c = tr.home_won.to_numpy()[covered]

    cal = calibrate(oof_c, y_c) if cfg["cal"] != "none" else ("none", None)
    print(f"[{phase}] calibration: {cal[0]} | lr_c={cfg['lr_c']} lgb({cfg['lgb_num_leaves']},{cfg['lgb_lr']}) w_lgb={cfg['w_lgb']}")

    p_va_raw = (1 - cfg["w_lgb"]) * predict(m_lr, va[LOGR_FEATS]) + cfg["w_lgb"] * predict(m_lgb, va[LGBM_FEATS], is_lgbm=True)
    p_te_raw = (1 - cfg["w_lgb"]) * predict(m_lr, te[LOGR_FEATS]) + cfg["w_lgb"] * predict(m_lgb, te[LGBM_FEATS], is_lgbm=True)
    p_va = apply_cal(cal, p_va_raw)
    p_te = apply_cal(cal, p_te_raw)

    va = va.assign(p_model_home=p_va, p_model_home_raw=p_va_raw)
    te = te.assign(p_model_home=p_te, p_model_home_raw=p_te_raw)
    return tr, va, te, {"cal": cal[0], "n_oof": int(covered.sum()), "n_train": len(tr),
                        "oof_brier": brier_score_loss(y_c, oof_c),
                        "oof_brier_lr": brier_score_loss(y_c, oof_lr[covered]),
                        "oof_brier_lgb": brier_score_loss(y_c, oof_lgb[covered]),
                        "oof_logloss": float(np.mean(-y_c * np.log(np.clip(oof_c, 1e-6, 1 - 1e-6))
                                                     - (1 - y_c) * np.log1p(-np.clip(oof_c, 1e-6, 1 - 1e-6))))}


def tune(train):
    """Select hyperparameters + ensemble weight + calibration by OOF brier on the
    phase-1 training window (strictly before the validation period)."""
    best = None
    for lr_c in [0.03, 0.1, 0.3, 1.0, 3.0]:
        oof_lr, _, cov = cv_oof_and_train(train, LOGR_FEATS, lambda: make_logreg(lr_c))
        for nl, lr_l in [(15, 0.05), (31, 0.03), (63, 0.02)]:
            oof_lgb, _, cov_l = cv_oof_and_train(train, LGBM_FEATS, lambda: make_lgbm(nl, lr_l), is_lgbm=True)
            for w in [0.4, 0.5, 0.6]:
                ens = w * oof_lgb + (1 - w) * oof_lr
                m = cov & cov_l
                for cal_kind in ["isotonic", "platt", "none"]:
                    c = calibrate(ens[m], train.home_won.to_numpy()[m]) if cal_kind != "none" else ("none", None)
                    p = apply_cal(c, ens[m])
                    b = brier_score_loss(train.home_won.to_numpy()[m], p)
                    key = (lr_c, nl, lr_l, w, cal_kind)
                    if best is None or b < best[0]:
                        best = (b, key)
    b, (lr_c, nl, lr_l, w, cal_kind) = best
    print(f"tuned: brier={b:.4f} lr_c={lr_c} lgb({nl},{lr_l}) w_lgb={w} cal={cal_kind}")
    return {"lr_c": lr_c, "lgb_num_leaves": nl, "lgb_lr": lr_l, "w_lgb": w, "cal": cal_kind}


