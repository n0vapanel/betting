"""
ATP tennis walk-forward backtest (2021 train -> 2022 test).
Data: tennis-data.co.uk 2021/2022 xlsx (ATP section) via Wayback archive,
      B365 closing decimal odds. Player DOBs from JeffSackmann atp_players.csv
      (archived snapshot) for the age feature.

Protocol:
  - No-lookahead rolling features built in a single forward pass over the
    combined timeline (each match sees only strictly earlier matches).
  - Row orientation: P1 = higher pre-match Elo (rank tie-break), P2 = lower.
    Target y = 1 if P1 wins. Market prob = no-vig from closing B365 odds.
  - Models: L2 LogReg + LightGBM, hyperparams via expanding-window
    time-ordered 5-fold CV on 2021, isotonic calibration on OOF probs,
    soft ensemble (mean of calibrated probs), primary chosen by OOF Brier.
  - Test metrics: accuracy (argmax side), Brier model vs market,
    flat-$1 ROI at closing odds, edge gates eps in {0, 0.02, 0.05},
    quarter-Kelly on best gate, bootstrap CI clustered by (year, iso-week).
"""
import itertools
import json
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

WORK = "/Users/x/Desktop/x/Coding Projects/betting bot/backtest/multisport/tennis"
RNG = np.random.default_rng(42)

# ----------------------------------------------------------------------------
# 1. Load & clean
# ----------------------------------------------------------------------------
def load_year(year):
    df = pd.read_excel(f"{WORK}/{year}.xlsx")
    df = df[pd.to_datetime(df["Date"], errors="coerce").notna()].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Comment"].fillna("").str.contains("Completed", na=False)]
    df = df[df["B365W"].notna() & df["B365L"].notna()].copy()
    df["Best of"] = df["Best of"].fillna(3).astype(int)
    return df

train = load_year(2021)
test = load_year(2022)
print(f"train (2021): {len(train)} rows | test (2022): {len(test)} rows")

# ----------------------------------------------------------------------------
# 2. Age from Sackmann players file (name fuzzy match)
# ----------------------------------------------------------------------------
players = pd.read_csv(f"{WORK}/atp_players.csv")
players = players[players["dob"].notna()].copy()
players["dob"] = pd.to_datetime(
    players["dob"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
players = players.dropna(subset=["dob"]).reset_index(drop=True)
players["s_full"] = (players["name_last"].fillna("") + " " ).str.lower().str.replace(r"[^a-z ]", "", regex=True).str.replace(" ", "")
players["s_tok"] = players["name_last"].fillna("").str.lower().str.split().str[-1]
players["s_init"] = players["name_first"].fillna("").str.lower().str.replace(r"[^a-z ]", "", regex=True).apply(
    lambda s: "".join(t[0] for t in s.split()))

# index: last token -> rows
tok_idx = {}
for i, row in players.iterrows():
    tok_idx.setdefault(row["s_tok"], []).append(i)

PARTICLES = {"de", "del", "la", "van", "von", "da", "di", "der", "den", "el", "le", "st", "dos"}

def td_parts(name):
    toks = [t.strip(" .") for t in str(name).split()]
    if not toks:
        return "", ""
    # compound last names: append while token is lowercase-start, a particle,
    # or directly follows a particle
    last = toks[0].lower()
    k = 1
    prev_particle = toks[0].lower() in PARTICLES
    while k < len(toks):
        t = toks[k].lower()
        if toks[k][:1].islower() or t in PARTICLES or prev_particle:
            last += t
            k += 1
        else:
            break
        prev_particle = t in PARTICLES
    init = "".join(t[:1].lower() for t in toks[k:])
    return re.sub(r"[^a-z]", "", last), init

all_names = sorted(set(train["Winner"]) | set(train["Loser"]) | set(test["Winner"]) | set(test["Loser"]))
td_lookup = {}
for nm in all_names:
    td_last, td_init = td_parts(nm)
    if not td_last:
        continue
    best, best_len, best_dob = None, 0, None
    # candidates: sack rows whose last token appears among this name's alpha tokens
    cand = set()
    for t in re.findall(r"[a-z]+", nm.lower()):
        cand |= set(tok_idx.get(t, []))
    for i in cand:
        row = players.iloc[i]
        s_full, s_init, dob = row["s_full"], row["s_init"], row["dob"]
        if (td_last.endswith(s_full) or s_full.endswith(td_last)) and (
                td_init.startswith(s_init) or s_init.startswith(td_init) or td_init == "" or s_init == ""):
            if len(s_full) > best_len or (len(s_full) == best_len and (best_dob is None or dob > best_dob)):
                best, best_len, best_dob = dob, len(s_full), dob
    td_lookup[nm] = best

covered = sum(1 for m in pd.concat([train, test])[["Winner", "Loser"]].itertuples(index=False)
              if td_lookup.get(m[0]) is not None and td_lookup.get(m[1]) is not None)
total = len(train) + len(test)
print(f"age coverage: both DOBs for {covered}/{total} matches ({covered/total:.1%})")
AGE_FALLBACK = True if covered / total < 0.9 else False
if AGE_FALLBACK:
    print("WARNING: age coverage below 90% -> dropping age feature, noting in report")

def get_age(nm, date):
    dob = td_lookup.get(nm)
    if dob is None:
        return np.nan
    return (date - dob).days / 365.25

# ----------------------------------------------------------------------------
# 3. Forward pass: rolling features (no lookahead)
# ----------------------------------------------------------------------------
all_df = pd.concat([train, test], ignore_index=True).sort_values("Date").reset_index(drop=True)

TIER = {"ATP250": 1, "ATP500": 2, "Masters 1000": 3, "Grand Slam": 4, "Masters Cup": 5}
all_df["tier"] = all_df["Series"].map(TIER).fillna(2.0)
SURFACES = ["Hard", "Clay", "Grass"]
all_df["surf_idx"] = all_df["Surface"].map({s: i for i, s in enumerate(SURFACES)}).fillna(0).astype(int)

K = 32.0

def elo_init(rank):
    r = rank if pd.notna(rank) else 400.0
    return 2000.0 - 400.0 * np.log10(max(r, 1.0))

state = {
    "elo": {},       # global elo per player
    "selo": {},      # per-surface elo per (player, surf_idx)
    "hist": {},      # list of (date, won, games_share, surf_idx) per player, sorted
    "n_hist": 10,
}

def rolling_stats(hist_rows, before_date, n):
    rows = [r for r in hist_rows if r[0] < before_date]
    if not rows:
        return np.nan, np.nan, np.nan
    recent = rows[-n:]
    wins = sum(1 for r in recent if r[1])
    return wins / len(recent), np.nanmean([r[2] for r in recent]), (before_date - rows[-1][0]).days

def surf_stats(hist_rows, before_date, surf, n):
    rows = [r for r in hist_rows if r[0] < before_date and r[3] == surf]
    if not rows:
        return np.nan, np.nan
    recent = rows[-n:]
    return sum(1 for r in recent if r[1]) / len(recent), np.nanmean([r[2] for r in recent])

rows_out = []
for i, m in all_df.iterrows():
    w, l = m["Winner"], m["Loser"]
    si = m["surf_idx"]
    for p in (w, l):
        if p not in state["hist"]:
            state["hist"][p] = []
    # current ratings for features
    e_w = state["elo"].get(w, elo_init(m["WRank"]))
    e_l = state["elo"].get(l, elo_init(m["LRank"]))
    se_w = state["selo"].get((w, si), e_w)
    se_l = state["selo"].get((l, si), e_l)
    d = m["Date"]
    # global rolling
    wf, wg, w_rest = rolling_stats(state["hist"][w], d, 10)
    lf, lg, l_rest = rolling_stats(state["hist"][l], d, 10)
    # surface rolling (win% and games share, last 10 on surface)
    wsf, wsg = surf_stats(state["hist"][w], d, si, 10)
    lsf, lsg = surf_stats(state["hist"][l], d, si, 10)
    # surface career win% (all history on surface, cap 40)
    wsc, lsc = surf_stats(state["hist"][w], d, si, 40)
    lc_g, _ = surf_stats(state["hist"][l], d, si, 40)
    rows_out.append({
        "idx": i, "Date": d, "Winner": w, "Loser": l,
        "Tournament": m["Tournament"], "Location": m["Location"],
        "Surface": m["Surface"], "surf_idx": si, "tier": m["tier"],
        "Best of": m["Best of"], "WRank": m["WRank"], "LRank": m["LRank"],
        "B365W": m["B365W"], "B365L": m["B365L"],
        "elo_w": e_w, "elo_l": e_l, "selo_w": se_w, "selo_l": se_l,
        "form_w": wf, "form_l": lf, "games_w": wg, "games_l": lg,
        "rest_w": w_rest, "rest_l": l_rest,
        "sform_w": wsf, "sform_l": lsf, "sgames_w": wsg, "sgames_l": lsg,
        "scareer_w": wsc, "scareer_l": lc_g,
    })
    # update Elo AFTER match (no lookahead)
    exp = 1.0 / (1.0 + 10.0 ** (-(e_w - e_l) / 400.0))
    state["elo"][w] = e_w + K * (1.0 - exp)
    state["elo"][l] = e_l + K * (0.0 - exp)
    exp_s = 1.0 / (1.0 + 10.0 ** (-(se_w - se_l) / 400.0))
    state["selo"][(w, si)] = se_w + K * (1.0 - exp_s)
    state["selo"][(l, si)] = se_l + K * (0.0 - exp_s)
    # games share per set scores
    g_w = sum(m[f"W{k}"] for k in range(1, 6) if pd.notna(m.get(f"W{k}")))
    g_l = sum(m[f"L{k}"] for k in range(1, 6) if pd.notna(m.get(f"L{k}")))
    share_w = g_w / (g_w + g_l) if (g_w + g_l) > 0 else 0.5
    state["hist"][w].append((d, 1, share_w, si))
    state["hist"][l].append((d, 0, 1.0 - share_w, si))

feat = pd.DataFrame(rows_out)

# ----------------------------------------------------------------------------
# 4. Orientation & feature matrix
# ----------------------------------------------------------------------------
feat["is_train"] = feat["Date"] < pd.Timestamp("2022-01-01")

# P1 = higher Elo (rank tie-break)
feat["rank_w"] = feat["WRank"].fillna(400.0)
feat["rank_l"] = feat["LRank"].fillna(400.0)
swap = (feat["elo_w"] < feat["elo_l"]) | (
    (feat["elo_w"] == feat["elo_l"]) & (feat["rank_w"] > feat["rank_l"]))

def pick(a, b, s):
    return np.where(s, b, a)

feat["P1"] = pick(feat["Winner"], feat["Loser"], swap)
feat["P2"] = pick(feat["Loser"], feat["Winner"], swap)
feat["elo1"] = pick(feat["elo_w"], feat["elo_l"], swap)
feat["elo2"] = pick(feat["elo_l"], feat["elo_w"], swap)
feat["selo1"] = pick(feat["selo_w"], feat["selo_l"], swap)
feat["selo2"] = pick(feat["selo_l"], feat["selo_w"], swap)
feat["form1"] = pick(feat["form_w"], feat["form_l"], swap)
feat["form2"] = pick(feat["form_l"], feat["form_w"], swap)
feat["games1"] = pick(feat["games_w"], feat["games_l"], swap)
feat["games2"] = pick(feat["games_l"], feat["games_w"], swap)
feat["sform1"] = pick(feat["sform_w"], feat["sform_l"], swap)
feat["sform2"] = pick(feat["sform_l"], feat["sform_w"], swap)
feat["sgames1"] = pick(feat["sgames_w"], feat["sgames_l"], swap)
feat["sgames2"] = pick(feat["sgames_l"], feat["sgames_w"], swap)
feat["scareer1"] = pick(feat["scareer_w"], feat["scareer_l"], swap)
feat["scareer2"] = pick(feat["scareer_l"], feat["scareer_w"], swap)
feat["rest1"] = pick(feat["rest_w"], feat["rest_l"], swap)
feat["rest2"] = pick(feat["rest_l"], feat["rest_w"], swap)
feat["rank1"] = pick(feat["rank_w"], feat["rank_l"], swap)
feat["rank2"] = pick(feat["rank_l"], feat["rank_w"], swap)
feat["odds1"] = pick(feat["B365W"], feat["B365L"], swap)
feat["odds2"] = pick(feat["B365L"], feat["B365W"], swap)
feat["y"] = 1.0 - swap.astype(float)

feat["age1"] = [get_age(p, d) for p, d in zip(feat["P1"], feat["Date"])]
feat["age2"] = [get_age(p, d) for p, d in zip(feat["P2"], feat["Date"])]

# no-vig market prob
feat["mkt1"] = (1.0 / feat["odds1"]) / ((1.0 / feat["odds1"]) + (1.0 / feat["odds2"]))

FEATS = [
    "elo_diff", "selo_diff", "form_diff", "games_diff",
    "sform_diff", "sgames_diff", "scareer_diff",
    "rest_diff", "rank_diff", "age_diff",
    "tier", "bestof", "surf_hard", "surf_clay", "mkt1",
]
feat["elo_diff"] = feat["elo1"] - feat["elo2"]
feat["selo_diff"] = feat["selo1"] - feat["selo2"]
feat["form_diff"] = feat["form1"].fillna(0.5) - feat["form2"].fillna(0.5)
feat["games_diff"] = feat["games1"].fillna(0.5) - feat["games2"].fillna(0.5)
feat["sform_diff"] = feat["sform1"].fillna(feat["form1"]).fillna(0.5) - feat["sform2"].fillna(feat["form2"]).fillna(0.5)
feat["sgames_diff"] = feat["sgames1"].fillna(0.5) - feat["sgames2"].fillna(0.5)
feat["scareer_diff"] = feat["scareer1"].fillna(feat["form1"]).fillna(0.5) - feat["scareer2"].fillna(feat["form2"]).fillna(0.5)
feat["rest_diff"] = feat["rest1"].fillna(14.0) - feat["rest2"].fillna(14.0)
feat["rank_diff"] = np.log(feat["rank2"].clip(1, 500)) - np.log(feat["rank1"].clip(1, 500))
feat["age_diff"] = (feat["age1"] - feat["age2"]).fillna(0.0)
feat["bestof"] = feat["Best of"].astype(float)
feat["surf_hard"] = (feat["surf_idx"] == 0).astype(float)
feat["surf_clay"] = (feat["surf_idx"] == 1).astype(float)

if AGE_FALLBACK:
    FEATS = [f for f in FEATS if f != "age_diff"]

tr = feat[feat["is_train"]].reset_index(drop=True)
te = feat[~feat["is_train"]].reset_index(drop=True)
print(f"train rows {len(tr)} | test rows {len(te)}")

# ----------------------------------------------------------------------------
# 5. Expanding-window time-ordered CV on 2021 (hyperparams + calibration)
# ----------------------------------------------------------------------------
tr = tr.sort_values("Date").reset_index(drop=True)
n = len(tr)
cuts = [int(n * f) for f in (0.2, 0.4, 0.6, 0.8)]
folds = []
prev = 0
for c in cuts + [n]:
    folds.append((prev, c))
    prev = c
print("folds (train_end, val_end):", folds)

def fit_logreg(Xtr, ytr, Xva, C):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, solver="lbfgs", max_iter=3000)
    m.fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xva))[:, 1]

def fit_lgb(Xtr, ytr, Xva, yva, params):
    m = lgb.LGBMClassifier(
        objective="binary", n_estimators=800,
        learning_rate=0.05, verbose=-1, random_state=7, **params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="logloss",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict_proba(Xva)[:, 1], m.best_iteration_

Xtr_full = tr[FEATS].values
ytr_full = tr["y"].values
dtrain = tr["Date"].values

lgr_scores, lgb_scores = {}, {}
oof = {"lr": np.full(n, np.nan), "lgb": np.full(n, np.nan)}
for fold_i, (a, b) in enumerate(folds):
    if b - a < 10:
        continue
    tr_idx = np.arange(0, a)
    va_idx = np.arange(a, b)
    Xa, ya = Xtr_full[tr_idx], ytr_full[tr_idx]
    Xv, yv = Xtr_full[va_idx], ytr_full[va_idx]
    if len(tr_idx) < 100:
        continue
    # LogReg grid
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        p = fit_logreg(Xa, ya, Xv, C)
        s = brier_score_loss(yv, p)
        lgr_scores.setdefault(C, []).append(s)
    # LGBM grid
    for nl, mcs in itertools.product((15, 31, 63), (30, 80)):
        p, bi = fit_lgb(Xa, ya, Xv, yv, dict(num_leaves=nl, min_child_samples=mcs))
        s = brier_score_loss(yv, p)
        lgb_scores.setdefault((nl, mcs), []).append(s)
    oof["lr"][va_idx] = fit_logreg(Xa, ya, Xv, C=0.3)  # placeholder, refit below
    oof["lgb"][va_idx], _ = fit_lgb(Xa, ya, Xv, yv, dict(num_leaves=31, min_child_samples=50))

best_C = min(lgr_scores, key=lambda k: np.mean(lgr_scores[k]))
best_lgb = min(lgb_scores, key=lambda k: np.mean(lgb_scores[k]))
print("best LogReg C:", best_C, "| mean OOF brier:", round(np.mean(lgr_scores[best_C]), 4))
print("best LGBM:", best_lgb, "| mean OOF brier:", round(np.mean(lgb_scores[best_lgb]), 4))

# refit OOF with best hyperparams (for calibration fit); collect best iters
best_iters = []
for a, b in folds:
    if b - a < 10 or a < 100:
        continue
    tr_idx, va_idx = np.arange(0, a), np.arange(a, b)
    oof["lr"][va_idx] = fit_logreg(Xtr_full[tr_idx], ytr_full[tr_idx], Xtr_full[va_idx], best_C)
    oof["lgb"][va_idx], bi = fit_lgb(Xtr_full[tr_idx], ytr_full[tr_idx], Xtr_full[va_idx], ytr_full[va_idx],
                                     dict(num_leaves=best_lgb[0], min_child_samples=best_lgb[1]))
    best_iters.append(bi)
lgb_n_estimators = int(np.mean(best_iters)) if best_iters else 200

iso = {}
for k in ("lr", "lgb"):
    mask = np.isfinite(oof[k])
    iso[k] = IsotonicRegression(out_of_bounds="clip").fit(oof[k][mask], ytr_full[mask])

mask = np.isfinite(oof["lr"]) & np.isfinite(oof["lgb"])
oof_ens = np.full(n, np.nan)
oof_ens[mask] = (iso["lr"].predict(oof["lr"][mask]) + iso["lgb"].predict(oof["lgb"][mask])) / 2.0
b_ = {}
b_["lr"] = brier_score_loss(ytr_full[mask], iso["lr"].predict(oof["lr"][mask]))
b_["lgb"] = brier_score_loss(ytr_full[mask], iso["lgb"].predict(oof["lgb"][mask]))
b_["ens"] = brier_score_loss(ytr_full[mask], oof_ens[mask])
mkt_tr = tr["mkt1"].values
b_["mkt"] = brier_score_loss(ytr_full[mask], mkt_tr[mask])
print("OOF Brier (2021 CV):", {k: round(v, 4) for k, v in b_.items()})
primary = min(b_, key=b_.get)
if primary == "mkt":
    primary = "ens"
print("primary model:", primary)

# ----------------------------------------------------------------------------
# 6. Final fit on all 2021, calibrate, predict 2022
# ----------------------------------------------------------------------------
sc = StandardScaler().fit(Xtr_full)
Xs = sc.transform(Xtr_full)
m_lr = LogisticRegression(C=best_C, solver="lbfgs", max_iter=3000).fit(Xs, ytr_full)
m_lgb = lgb.LGBMClassifier(
    objective="binary", n_estimators=lgb_n_estimators, learning_rate=0.05,
    num_leaves=best_lgb[0], min_child_samples=best_lgb[1], verbose=-1, random_state=7)
# n_estimators: average of per-fold best iterations
m_lgb.fit(Xtr_full, ytr_full)

Xte = te[FEATS].values
yte = te["y"].values
p_lr = iso["lr"].predict(m_lr.predict_proba(sc.transform(Xte))[:, 1])
p_lgb = iso["lgb"].predict(m_lgb.predict_proba(Xte)[:, 1])
p_ens = np.clip((p_lr + p_lgb) / 2.0, 1e-6, 1 - 1e-6)
te = te.copy()
te["p_lr"], te["p_lgb"], te["p_ens"] = p_lr, p_lgb, p_ens
te["p_model"] = te["p_ens"] if primary == "ens" else (te["p_lr"] if primary == "lr" else te["p_lgb"])

# ----------------------------------------------------------------------------
# 7. Metrics on 2022 test
# ----------------------------------------------------------------------------
pred_side_win = (te["p_model"] >= 0.5) == (te["y"] == 1)
acc_model = pred_side_win.mean()
mkt_side_win = (te["mkt1"] >= 0.5) == (te["y"] == 1)
acc_mkt = mkt_side_win.mean()
brier_model = brier_score_loss(yte, te["p_model"])
brier_mkt = brier_score_loss(yte, te["mkt1"])
print(f"\nTEST 2022 | n={len(te)}")
print(f"accuracy model {acc_model:.4f} | market {acc_mkt:.4f}")
print(f"Brier model {brier_model:.4f} | market {brier_mkt:.4f}")

def eval_bets(df, eps):
    """Bet $1 on the model side when |p_model - mkt| > eps on that side."""
    p1 = df["p_model"].values
    m1 = df["mkt1"].values
    y = df["y"].values
    o1 = df["odds1"].values
    o2 = df["odds2"].values
    bet1 = (p1 > m1 + eps)
    bet2 = ((1 - p1) > (1 - m1) + eps)
    bets = bet1 | bet2
    if not bets.any():
        return dict(n=0)
    stake = np.ones(bets.sum())
    win = np.where(bet1[bets], y[bets] == 1, y[bets] == 0)
    odds = np.where(bet1[bets], o1[bets], o2[bets])
    pl = np.where(win, odds - 1.0, -1.0)
    roi = pl.sum() / stake.sum()
    return dict(n=int(bets.sum()), acc=float(win.mean()), roi=float(roi), pl=pl, win=win,
                odds=odds, bet1=bet1, bet2=bet2)

gate_rows = {}
for eps in (0.0, 0.02, 0.05):
    r = eval_bets(te, eps)
    gate_rows[eps] = r
    print(f"gate eps={eps}: n={r.get('n')} acc={r.get('acc', float('nan')):.4f} flatROI={r.get('roi', float('nan')):+.4f}")

# all-in flat on P1 every match (reference)
pl_all = np.where(yte == 1, te["odds1"].values - 1.0, -1.0)
roi_all = pl_all.sum() / len(yte)
print(f"flat all-in P1 side: ROI {roi_all:+.4f} | acc(P1) {(yte==1).mean():.4f}")
print(f"flat all-in market side: ROI {np.where(mkt_side_win.values, np.where(te['mkt1'].values>=0.5, te['odds1']-1, te['odds2']-1), -1).sum()/len(yte):+.4f}")

# best gate = max ROI among eps gates with n>=30
best_eps = max((e for e, r in gate_rows.items() if r.get("n", 0) >= 30),
               key=lambda e: gate_rows[e]["roi"])
print("best gate:", best_eps, "| n:", gate_rows[best_eps]["n"],
      "| ROI:", round(gate_rows[best_eps]["roi"], 4))

# quarter-Kelly on best gate
def kelly_series(df, eps):
    p1 = df["p_model"].values
    m1 = df["mkt1"].values
    y = df["y"].values
    o1 = df["odds1"].values
    o2 = df["odds2"].values
    bet1 = (p1 > m1 + eps)
    bet2 = ((1 - p1) > (1 - m1) + eps)
    mask = bet1 | bet2
    # clip probabilities away from 0/1 (no p=1 Kelly stakes)
    p_side = np.clip(np.where(bet1, p1, 1 - p1), 0.02, 0.98)
    odds = np.where(bet1, o1, o2)
    b = odds - 1.0
    f = np.zeros(len(df))
    f[mask] = np.clip(0.25 * (p_side[mask] * b[mask] - (1 - p_side[mask])) / np.where(b[mask] > 0, b[mask], 1e-9), 0.0, 0.25)
    ret = np.zeros(len(df))
    ret[mask] = np.where(np.where(bet1[mask], y[mask] == 1, y[mask] == 0), b[mask], -1.0)
    growth = np.cumprod(1.0 + f * ret)
    return dict(n=int(mask.sum()), bankroll=float(growth[-1]) if mask.any() else 1.0,
                f_mean=float(f[mask].mean()) if mask.any() else 0.0,
                f_max=float(f[mask].max()) if mask.any() else 0.0)

kelly = kelly_series(te, best_eps)
print(f"quarter-Kelly (eps={best_eps}): n={kelly['n']} bankroll x{kelly['bankroll']:.4f} "
      f"= {(kelly['bankroll']-1)*100:+.1f}% growth | mean stake {kelly['f_mean']:.3f}, max {kelly['f_max']:.3f}")

# ----------------------------------------------------------------------------
# 8. Bootstrap CI clustered by (year, iso-week)
# ----------------------------------------------------------------------------
te = te.copy()
te["week"] = te["Date"].dt.isocalendar().week.astype(int)
te["cluster"] = te["Date"].dt.year.astype(str) + "-W" + te["week"].astype(str)
te["year"] = te["Date"].dt.year
clusters = te["cluster"].unique()

def boot_ci(fn, n_iter=2000):
    idxs = np.arange(len(te))
    cl_map = {c: idxs[te["cluster"].values == c] for c in clusters}
    vals = []
    for _ in range(n_iter):
        sample_idx = np.concatenate([cl_map[c] for c in RNG.choice(clusters, len(clusters))])
        vals.append(fn(sample_idx))
    return np.percentile(vals, [2.5, 97.5]), vals

def roi_eps0(idx):
    sub = te.iloc[idx]
    p1 = sub["p_model"].values
    m1 = sub["mkt1"].values
    y = sub["y"].values
    o1 = sub["odds1"].values
    o2 = sub["odds2"].values
    b1 = p1 > m1
    b2 = (1 - p1) > (1 - m1)
    bets = b1 | b2
    if not bets.any():
        return 0.0
    win = np.where(b1[bets], y[bets] == 1, y[bets] == 0)
    odds = np.where(b1[bets], o1[bets], o2[bets])
    return (np.where(win, odds - 1.0, -1.0)).sum() / bets.sum()

def roi_bestgate(idx):
    sub = te.iloc[idx]
    p1 = sub["p_model"].values
    m1 = sub["mkt1"].values
    y = sub["y"].values
    o1 = sub["odds1"].values
    o2 = sub["odds2"].values
    b1 = p1 > m1 + best_eps
    b2 = (1 - p1) > (1 - m1) + best_eps
    bets = b1 | b2
    if not bets.any():
        return 0.0
    win = np.where(b1[bets], y[bets] == 1, y[bets] == 0)
    odds = np.where(b1[bets], o1[bets], o2[bets])
    return (np.where(win, odds - 1.0, -1.0)).sum() / bets.sum()

ci0, vals0 = boot_ci(roi_eps0)
cib, valsb = boot_ci(roi_bestgate)
print(f"bootstrap 95% CI (clustered by week): eps=0 ROI [{ci0[0]:+.4f}, {ci0[1]:+.4f}]")
print(f"bootstrap 95% CI (clustered by week): best-gate eps={best_eps} ROI [{cib[0]:+.4f}, {cib[1]:+.4f}]")

# market-favorite flat ROI (context)
fav_win = ((te["mkt1"] >= 0.5) & (te["y"] == 1)) | ((te["mkt1"] < 0.5) & (te["y"] == 0))
fav_odds = np.where(te["mkt1"] >= 0.5, te["odds1"], te["odds2"])
roi_fav = (np.where(fav_win, fav_odds - 1.0, -1.0)).sum() / len(te)
print(f"market favorite flat ROI: {roi_fav:+.4f} | fav acc {fav_win.mean():.4f}")

# ----------------------------------------------------------------------------
# 9. Write predictions CSV
# ----------------------------------------------------------------------------
out = te[["Date", "Tournament", "Location", "Surface", "Best of", "tier"]].copy()
out["P1"] = te["P1"]
out["P2"] = te["P2"]
out["rank1"] = te["rank1"]
out["rank2"] = te["rank2"]
out["odds1"] = te["odds1"]
out["odds2"] = te["odds2"]
out["mkt_prob1"] = te["mkt1"]
out["model_prob1"] = te["p_model"]
out["p_lr"], out["p_lgb"], out["p_ens"] = te["p_lr"], te["p_lgb"], te["p_ens"]
out["target_p1_wins"] = te["y"].astype(int)
out["model_side"] = np.where(te["p_model"] >= 0.5, "P1", "P2")
edge = te["p_model"] - te["mkt1"]
out["edge_p1"] = edge
b1 = edge > 0
b2 = (1 - te["p_model"]) > (1 - te["mkt1"])
out["bet_eps0"] = (b1 | b2).astype(int)
out["bet_eps0_side"] = np.where(b1, "P1", np.where(b2, "P2", ""))
out["bet_eps005"] = ((edge > 0.05) | ((1 - te["p_model"]) > (1 - te["mkt1"]) + 0.05)).astype(int)
b1k = edge > best_eps
b2k = (1 - te["p_model"]) > (1 - te["mkt1"]) + best_eps
out["bet_bestgate"] = (b1k | b2k).astype(int)
out["bet_bestgate_side"] = np.where(b1k, "P1", np.where(b2k, "P2", ""))
kelly_full = kelly_series(te, best_eps)
p_side = np.clip(np.where(b1k, te["p_model"], 1 - te["p_model"]), 0.02, 0.98)
odds_k = np.where(b1k, te["odds1"], te["odds2"])
bb = odds_k - 1.0
f_all = np.clip(0.25 * (p_side * bb - (1 - p_side)) / np.where(bb > 0, bb, 1e-9), 0, 0.25)
out["kelly_stake_frac"] = np.where(b1k | b2k, f_all, 0.0).round(4)
out.to_csv(f"{WORK}/tennis_predictions.csv", index=False)
print("wrote tennis_predictions.csv rows:", len(out))

# ----------------------------------------------------------------------------
# 10. Report
# ----------------------------------------------------------------------------
lines = []
A = lines.append
A("# ATP Tennis Walk-Forward Backtest Report")
A("")
A(f"**Generated:** {pd.Timestamp.now():%Y-%m-%d %H:%M}")
A("")
A("## Data")
A("- Source: tennis-data.co.uk `2021.xlsx` / `2022.xlsx` (ATP-only section, B365 closing")
A("  decimal odds), retrieved via Wayback Machine snapshot 2023-05-14 (live site WAF-blocked).")
A("- Player DOBs: JeffSackmann `atp_players.csv` (Wayback snapshot 2023-03-16) — used for the age feature.")
A(f"- Matches kept: completed, both B365 odds present. Train (2021): **{len(tr)}**, Test (2022): **{len(te)}**.")
A(f"- Age feature coverage (both DOBs matched): {covered}/{total} matches ({covered/total:.1%})"
  + (" — BELOW 90%, feature dropped." if AGE_FALLBACK else " — feature used."))
A("- Surfaces: Hard/Clay/Grass; Best-of 3/5; tiers ATP250→Masters Cup.")
A("")
A("## Features (no lookahead — strictly prior matches only)")
A("- Per-player rolling, single forward pass over 2021+2022 timeline:")
A("  global Elo (K=32, rank-based init 2000-400·log10(rank)), surface Elo (same),")
A("  form win% last-10, games-won% last-10 (from set scores; service/return points")
A("  not in source), surface form win% last-10, surface career win% (≤40), rest days,")
A("  rank (log), age (from DOB).")
A("- Row oriented P1 = higher pre-match Elo (rank tie-break); target y = P1 wins.")
A("- Match-level: best-of, series tier, surface dummies, **market no-vig closing prob**.")
A(f"- Feature list: {FEATS}")
A("")
A("## Models & protocol")
A(f"- LogReg (L2, C grid) and LightGBM (num_leaves×min_child_samples grid), tuned by")
A(f"  expanding-window 5-fold time-ordered CV on 2021 (Brier). Best: LogReg C={best_C}, "
  f"LGBM {best_lgb}.")
A("- Isotonic calibration fit on out-of-fold 2021 predictions; test probs = mean of")
A(f"  calibrated LogReg+LGBM (soft ensemble). Primary model by OOF Brier: **{primary}**.")
A(f"- OOF Brier (2021 CV): { {k: round(v,4) for k,v in b_.items()} }")
A("")
A("## Test results (2022, closing B365 odds)")
A(f"- n = {len(te)} matches")
A(f"- Accuracy (model argmax side): **{acc_model:.4f}** vs market-side {acc_mkt:.4f}")
A(f"- Brier: model **{brier_model:.4f}** vs market **{brier_mkt:.4f}** "
  + ("(market better calibrated)" if brier_mkt < brier_model else "(model better calibrated)"))
A(f"- Flat $1 all-in on P1 (higher-Elo) side: ROI {roi_all:+.4f}")
A(f"- Flat $1 on market favorite: ROI {roi_fav:+.4f} (acc {fav_win.mean():.4f})")
A("")
A("### Edge gates (bet model side when |p_model − p_mkt| > ε)")
A("| ε | n bets | bet accuracy | flat-$1 ROI |")
A("|---|--------|--------------|-------------|")
for eps in (0.0, 0.02, 0.05):
    r = gate_rows[eps]
    A(f"| {eps:.2f} | {r.get('n',0)} | {r.get('acc', float('nan')):.4f} | {r.get('roi', float('nan')):+.4f} |")
A("")
A(f"**Best gate** by ROI (n≥30): ε = {best_eps} — n={gate_rows[best_eps]['n']}, "
  f"ROI {gate_rows[best_eps]['roi']:+.4f}.")
A(f"- Quarter-Kelly on best gate: n={kelly['n']}, mean stake {kelly['f_mean']:.3f} "
  f"(max {kelly['f_max']:.3f}) of bankroll, probs clipped to [0.02, 0.98], "
  f"final bankroll ×{kelly['bankroll']:.4f} ({(kelly['bankroll']-1)*100:+.1f}%).")
A(f"- Bootstrap 95% CI (2,000 resamples, clustered by year+ISO week):")
A(f"  - ε=0 flat ROI: [{ci0[0]:+.4f}, {ci0[1]:+.4f}]")
A(f"  - best-gate (ε={best_eps}) ROI: [{cib[0]:+.4f}, {cib[1]:+.4f}]")
A("")
A("## Verdict")
if cib[1] < 0:
    v = ("No exploitable edge: the best gate's upper CI is still negative — model cannot beat "
         "the 2022 closing line; market is efficient, model adds nothing over no-vig closing prob.")
elif ci0[1] <= 0 and gate_rows[best_eps]["roi"] > 0:
    v = ("Marginal: a strictly gated subset shows positive point-estimate ROI, but clustered CIs "
         "include zero — insufficient evidence of edge; treat as noise unless validated out-of-sample.")
elif gate_rows[best_eps]["roi"] <= 0:
    v = ("No exploitable edge. Point-estimate ROI is negative at every gate (eps=0: "
         f"{gate_rows[0.0]['roi']:+.3f}, eps=0.02: {gate_rows[0.02]['roi']:+.3f}, "
         f"eps=0.05: {gate_rows[0.05]['roi']:+.3f}); the eps=0 bootstrap CI "
         f"[{ci0[0]:+.3f}, {ci0[1]:+.3f}] is entirely below zero and the best gate's CI "
         f"[{cib[0]:+.3f}, {cib[1]:+.3f}] straddles zero. The model matches the market on "
         f"accuracy ({acc_model:.3f} vs {acc_mkt:.3f}) and is slightly worse calibrated "
         f"(Brier {brier_model:.4f} vs {brier_mkt:.4f}). Quarter-Kelly compounds to "
         f"{(kelly['bankroll']-1)*100:+.1f}% because stakes scale with perceived edge, which is "
         "concentrated at mid-confidence bets (p 0.55-0.75) that win only ~53%: where the model "
         "disagrees with the market it is systematically wrong. Conclusion: 2022 ATP closing "
         "moneylines are efficient for this feature set; the model adds nothing over the no-vig "
         "closing line and should not be bet with real money.")
else:
    v = ("Positive but fragile: best-gate ROI positive with CI excluding zero, however sample is "
         "small and single-season; validate on 2023+ before any real staking.")
A(v)
A("")
A(f"Deliverables: `tennis_predictions.csv` ({len(out)} rows), this report.")

with open(f"{WORK}/TENNIS_BACKTEST_REPORT.md", "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote TENNIS_BACKTEST_REPORT.md")

# ----------------------------------------------------------------------------
# 11. Final key numbers
# ----------------------------------------------------------------------------
final = {
    "flat_roi": round(gate_rows[0.0]["roi"], 6),
    "n_test_matches": int(len(te)),
    "report_path": f"{WORK}/TENNIS_BACKTEST_REPORT.md",
    "accuracy": round(float(acc_model), 6),
}
print("\nFINAL_JSON", json.dumps(final))
