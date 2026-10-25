"""Shared library for MMA v2 backtest: features, walk-forward yearly models, evaluation."""
from functools import partial
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from backtest_lib import am_to_dec, fair_probs, brier_score, evaluate_bets, kelly_path, bootstrap_ci

# ---------------- feature construction ----------------
WC_GROUPS = {"Heavyweight": "HW", "Light Heavyweight": "LHW", "Middleweight": "MW",
             "Welterweight": "WW", "Lightweight": "LW", "Featherweight": "FW",
             "Bantamweight": "BW", "Flyweight": "FLW"}

def class_group(wc):
    if wc in WC_GROUPS:
        return WC_GROUPS[wc]
    if "Women" in str(wc):
        return "W"
    return "OTH"

def prepare_df(path):
    df = pd.read_parquet(path)
    df = df[~df["Winner"].isin(["Draw", "No Contest"])].copy()
    df["yR"] = (df["Winner"] == "Red").astype(int)
    df["decR"] = am_to_dec(df["R_odds"])
    df["decB"] = am_to_dec(df["B_odds"])
    fr, fb = fair_probs(df["decR"], df["decB"])
    df["fairR"], df["fairB"] = fr, fb
    df["cg"] = df["weight_class"].map(class_group)
    return df

def build_feature_frames(df):
    """Returns (core_feats, all_feats, X_core, X_all, meta) on the full df."""
    for s in ["R", "B"]:
        df[f"win_pct_{s}"] = (df[f"w_prior_{s}"] + 1.0) / (df[f"n_prior_{s}"] + 2.0)
        df[f"days_since_{s}"] = df[f"days_since_{s}"].fillna(999.0)
        df[f"elo_before_{s}"] = df[f"elo_before_{s}"].fillna(1500.0)
        df[f"streak_pre_{s}"] = df[f"streak_pre_{s}"].fillna(0.0)
        df[f"longest_run_so_far_{s}"] = df[f"longest_run_so_far_{s}"].fillna(0.0)
        df[f"n_prior_{s}"] = df[f"n_prior_{s}"].fillna(0.0)
        df[f"w_prior_{s}"] = df[f"w_prior_{s}"].fillna(0.0)
        df[f"l_prior_{s}"] = df[f"l_prior_{s}"].fillna(0.0)
        df[f"career_days_{s}"] = df[f"career_days_{s}"].fillna(0.0)
        df[f"recent3_winpct_{s}"] = df[f"recent3_winpct_{s}"].fillna(0.0)
        df[f"opp_elo_avg_{s}"] = df[f"opp_elo_avg_{s}"].fillna(1500.0)
        df[f"opp_winpct_avg_{s}"] = df[f"opp_winpct_avg_{s}"].fillna(0.5)
        # physicals: DOB-based age first, else mdabbert; height/reach: mdabbert else fighter-details
        df[f"age_{s}"] = df[f"age_dob_{s}"].fillna(df[f"{s}_age"])
        df[f"age_debut_{s}"] = df[f"age_debut_{s}"].fillna(df[f"age_{s}"])
        df[f"height_{s}"] = df[f"{s}_Height_cms"].fillna(df[f"fd_height_{s}"])
        df[f"reach_{s}"] = df[f"{s}_Reach_cms"].fillna(df[f"fd_reach_{s}"])

    diff = lambda a, b: df[f"{a}_R"] - df[f"{a}_B"] if a != b else df[f"{a}_R"] - df[f"{a}_B"]
    def d_(name, a, b=None):
        b = b or a
        df[f"d_{name}"] = diff(a, b)

    for nm, (a, b) in {
        "elo": ("elo_before", "elo_before"), "win_pct": ("win_pct", "win_pct"),
        "streak": ("streak_pre", "streak_pre"), "longest_streak": ("longest_run_so_far", "longest_run_so_far"),
        "exp": ("n_prior", "n_prior"), "days_since": ("days_since", "days_since"),
        "age": ("age", "age"), "reach": ("reach", "reach"), "height": ("height", "height"),
        "career_days": ("career_days", "career_days"), "age_debut": ("age_debut", "age_debut"),
        "recent3_winpct": ("recent3_winpct", "recent3_winpct"),
        "opp_elo": ("opp_elo_avg", "opp_elo_avg"), "opp_winpct": ("opp_winpct_avg", "opp_winpct_avg"),
    }.items():
        d_(nm, a, b)
    df["d_weight"] = df["R_Weight_lbs"] - df["B_Weight_lbs"]
    df["d_rank"] = df["B_match_weightclass_rank"].fillna(16) - df["R_match_weightclass_rank"].fillna(16)

    # stance
    sp = lambda s: (df[f"stance_{s}"].fillna("").str.lower() == "southpaw").astype(int)
    df["d_sp"] = sp("R") - sp("B")
    df["same_stance"] = ((df["stance_R"].fillna("").str.lower() == df["stance_B"].fillna("").str.lower()) & (df["stance_R"].notna())).astype(int)

    # country dummies (top countries over full dataset; stable frequencies)
    top = df["country_n"].value_counts().head(7).index.tolist()
    for c in top:
        df[f"ct_{c.replace(' ', '_')}"] = (df["country_n"] == c).astype(int)

    # strike diffs
    strike_pairs = ["slpm", "str_acc", "td_acc", "td_def", "sub_per_f", "kd_per_f", "ctrl_share",
                    "slpm_r3", "str_acc_r3", "opp_slpm", "opp_str_acc", "opp_td_acc", "opp_sub"]
    for c in strike_pairs:
        df[f"d_{c}"] = df[f"feat_{c}_R"] - df[f"feat_{c}_B"]

    cat_names = []
    df["is_female"] = (df["gender"] == "FEMALE").astype(int); cat_names.append("is_female")
    df["is_title"] = df["title_bout"].astype(int); cat_names.append("is_title")
    df["is_5rd"] = (df["no_of_rounds"] == 5).astype(int); cat_names.append("is_5rd")
    wc_dummies = pd.get_dummies(df["weight_class"], prefix="wc")
    df = pd.concat([df, wc_dummies.astype(int)], axis=1)
    cat_names += list(wc_dummies.columns)
    ct_names = [f"ct_{c.replace(' ', '_')}" for c in top]
    cat_names += ct_names + ["empty_arena"]

    CORE_DIFF = ["d_elo", "d_win_pct", "d_streak", "d_longest_streak", "d_exp", "d_days_since",
                 "d_age", "d_reach", "d_height", "d_weight", "d_rank", "d_career_days",
                 "d_age_debut", "d_recent3_winpct", "d_opp_elo", "d_opp_winpct",
                 "d_sp", "same_stance"]
    STRIKE_DIFF = ["d_" + c for c in strike_pairs]
    core_feats = CORE_DIFF + cat_names
    all_feats = CORE_DIFF + STRIKE_DIFF + cat_names
    return df, core_feats, all_feats

# ---------------- models ----------------
def model_builder(name, cfg):
    """Picklable model constructor from a config dict."""
    if name == "lr":
        return partial(make_lr, cfg.get("C", 0.03))
    if name == "lgb":
        return partial(make_lgb, cfg["leaves"], cfg["lr"])
    if name == "xgb":
        return partial(make_xgb, cfg["depth"], cfg["lr"])
    raise ValueError(name)

def make_lr(C=0.03):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=C, max_iter=2000))])

def make_lgb(leaves=15, lr=0.03, n_estimators=800):
    return lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=lr, num_leaves=leaves,
        min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=0, verbose=-1)

def make_xgb(depth=4, lr=0.05, n_estimators=1200):
    return xgb.XGBClassifier(
        n_estimators=n_estimators, learning_rate=lr, max_depth=depth,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=0, n_jobs=4, verbosity=0,
        early_stopping_rounds=50)

# ---------------- walk-forward machinery ----------------
def fit_predict(model, Xtr, ytr, Xva=None, yva=None, Xte=None):
    """GBM-style fit with early stopping on val, or plain fit. Returns (pva, pte, best_iter)."""
    pva = pte = None
    if isinstance(model, Pipeline):
        model.fit(Xtr, ytr)
        if Xva is not None:
            pva = model.predict_proba(Xva)[:, 1]
        if Xte is not None:
            pte = model.predict_proba(Xte)[:, 1]
        return pva, pte, None
    kwargs = {}
    if Xva is not None:
        kwargs = dict(eval_set=[(Xva, yva)], eval_metric="logloss",
                      callbacks=[lgb.early_stopping(50, verbose=False)] if isinstance(model, lgb.LGBMClassifier)
                      else None)
        if isinstance(model, xgb.XGBClassifier):
            kwargs = dict(eval_set=[(Xva, yva)], verbose=False)
    model.fit(Xtr, ytr, **kwargs)
    if Xva is not None:
        pva = model.predict_proba(Xva)[:, 1]
    if Xte is not None:
        pte = model.predict_proba(Xte)[:, 1]
    return pva, pte, getattr(model, "best_iteration_", None)

def cv_year(Xtr, ytr, dtr, model_fn, val_years):
    """Expanding CV: for each val year, train on all fights before that year, validate on that year.
    Returns oof (aligned to Xtr), mean logloss, mean best_iter."""
    oof = np.full(len(Xtr), np.nan)
    losses, iters = [], []
    for vy in val_years:
        tr = dtr < pd.Timestamp(f"{vy}-01-01")
        va = (dtr.dt.year == vy)
        if va.sum() == 0:
            continue
        model = model_fn()
        pva, _, bi = fit_predict(model, Xtr[tr], ytr[tr], Xtr[va], ytr[va])
        oof[va] = pva
        losses.append(log_loss(ytr[va], pva))
        if bi is not None:
            iters.append(bi)
    return oof, float(np.mean(losses)) if losses else np.nan, (np.mean(iters) if iters else None)

def year_model(Xtr, ytr, dtr, model_fn, val_years, Xte):
    """Train + calibrate a model for one year: CV for isotonic, final fit w/ early stop on last 20%."""
    oof, cv_loss, iters = cv_year(Xtr, ytr, dtr, model_fn, val_years)
    valid = ~np.isnan(oof)
    iso = None
    if valid.sum() >= 50:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        iso.fit(oof[valid], ytr[valid])
    else:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        iso.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    # final fit: early stopping on last 20% of train (chronological)
    n = len(Xtr)
    cut = int(n * 0.8)
    model = model_fn()
    _, _, bi = fit_predict(model, Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:], Xte)
    if Xte is not None:
        pte = model.predict_proba(Xte)[:, 1]
    else:
        pte = None
    return pte, iso, cv_loss, bi

def walk_forward(df, feats, years, hparams, val_year_off=0):
    """Walk-forward yearly refits. hparams = dict(model->config). Returns preds dict {year: (ptes_by_model, y_te, idx_te)}."""
    X = df[feats].values
    y = df["yR"].values
    d = df["date_n"]
    out = {}
    for Y in years:
        tr = d < pd.Timestamp(f"{Y}-01-01")
        te = (d.dt.year == Y)
        if te.sum() == 0:
            continue
        Xtr, ytr, dtr = X[tr], y[tr], d[tr]
        Xte = X[te]
        years_in = sorted(set(dtr.dt.year))
        val_years = years_in[-3:]
        preds = {}
        for mname in ["lr", "lgb", "xgb"]:
            pte, iso, cv_loss, bi = year_model(Xtr, ytr, dtr, hparams[mname]["fn"], val_years, Xte)
            if pte is not None:
                pte = iso.predict(pte)
            preds[mname] = pte
            preds[mname + "_cv"] = cv_loss
        out[Y] = dict(preds=preds, y=y[te], idx=np.where(te)[0])
        print(f"  year {Y}: train n={tr.sum()}, test n={te.sum()}, CV losses "
              f"lr={preds['lr_cv']:.4f} lgb={preds['lgb_cv']:.4f} xgb={preds['xgb_cv']:.4f}")
    return out

def assemble(wf, years, mode="mean"):
    """Concatenate yearly preds into test arrays. Returns dict with keys per model and 'ens'."""
    ys, idxs = [], []
    for Y in years:
        ys.append(wf[Y]["y"]); idxs.append(wf[Y]["idx"])
    y_all = np.concatenate(ys)
    idx_all = np.concatenate(idxs)
    all_keys = [k for k in wf[years[0]]["preds"].keys()
                if isinstance(wf[years[0]]["preds"][k], np.ndarray) and wf[years[0]]["preds"][k].ndim >= 1]
    ps = {m: np.concatenate([wf[Y]["preds"][m] for Y in years]) for m in all_keys}
    if mode == "mean" and {"lr", "lgb", "xgb"} <= set(ps):
        ps["ens"] = (ps["lr"] + ps["lgb"] + ps["xgb"]) / 3.0
        ps["ens_gb"] = (ps["lgb"] + ps["xgb"]) / 2.0
    return ps, y_all, idx_all
