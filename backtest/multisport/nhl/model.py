"""NHL 2021-22 walk-forward final: LogReg + LGBM, TSCV selection, isotonic+Platt
calibration, edge gates, quarter-Kelly, bootstrap CI, ablations. Payouts = synthetic
2-way odds from de-vigged probs @4.76% hold (raw archangel odds are 3-way-inflated)."""
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

df = pd.read_parquet('games_features.parquet')
FEATS = [c for c in df.columns if c.startswith('diff_')]
TRAIN_END = pd.Timestamp('2022-02-28')
TEST_START = pd.Timestamp('2022-03-01')

train = df[df.date <= TRAIN_END].copy().reset_index(drop=True)
test = df[df.date >= TEST_START].copy().reset_index(drop=True)
print(f"train: {len(train)} | test: {len(test)} (regular {(test.is_playoff == 0).sum()}, "
      f"playoff {(test.is_playoff == 1).sum()})")

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
    return LogisticRegression(C=C, max_iter=3000, solver='lbfgs')


def mk_lgbm(lr, leaves, nest, mc):
    return lgb.LGBMClassifier(n_estimators=nest, learning_rate=lr, num_leaves=leaves,
                              subsample=0.9, colsample_bytree=0.9, min_child_samples=mc,
                              reg_lambda=1.0, random_state=RNG, verbosity=-1)

# ---- hyperparameter grid by OOF Brier ----
results = {}
for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    results[f'lr_C={C}'] = cv_oof(lambda C=C: mk_logreg(C), Xtr_s, ytr)
for lr in [0.03, 0.1]:
    for leaves in [7, 15, 31]:
        for nest in [100, 300]:
            for mc in [10, 20]:
                results[f'lgbm_lr={lr}_lv={leaves}_ne={nest}_mc={mc}'] = cv_oof(
                    lambda lr=lr, leaves=leaves, nest=nest, mc=mc: mk_lgbm(lr, leaves, nest, mc),
                    Xtr_s, ytr)
briers = {k: brier_score_loss(ytr, v) for k, v in results.items()}
best_lr_key = min([k for k in results if k.startswith('lr_')], key=lambda k: briers[k])
best_lgb_key = min([k for k in results if k.startswith('lgbm_')], key=lambda k: briers[k])
print(f"OOF Brier: LogReg {best_lr_key} {briers[best_lr_key]:.4f} | LGBM {best_lgb_key} "
      f"{briers[best_lgb_key]:.4f} | market(train) {brier_score_loss(ytr, train.p_mkt_home):.4f}")
oof_lr = results[best_lr_key]
oof_lgb = results[best_lgb_key]

# ---- calibration on OOF: isotonic + Platt ----
def iso(oof_in, y, pred):
    iso = IsotonicRegression(out_of_bounds='clip', y_min=1e-4, y_max=1 - 1e-4).fit(oof_in, y)
    return iso.predict(pred)


def platt(oof_in, y, pred):
    lo = LogisticRegression(C=1e6, max_iter=3000)
    lo.fit(oof_in.reshape(-1, 1), y)
    return lo.predict_proba(pred.reshape(-1, 1))[:, 1]


# ---- ensemble weight by OOF Brier (on Platt-calibrated OOF via nested folds) ----
# simple approach: calibrate OOF with Platt fit on OOF itself is optimistic; use
# leave-one-fold-out calibration: for each test fold, fit calibrator on other folds' OOF.
def oof_calibrated(make_model, cal_fn):
    """OOF preds calibrated fold-out-of-fold."""
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


_, oof_lr_cal = oof_calibrated(lambda: mk_logreg(float(best_lr_key.split('=')[1])), platt)
import re
nums = re.findall(r'[\d.]+', best_lgb_key)
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
C_best = float(best_lr_key.split('=')[1])
lr_b = float(nums[0]); lv_b = int(float(nums[1])); ne_b = int(float(nums[2])); mc_b = int(float(nums[3]))
lr_model = mk_logreg(C_best).fit(Xtr_s, ytr)
lgb_model = mk_lgbm(lr_b, lv_b, ne_b, mc_b).fit(Xtr_s, ytr)
p_lr_raw = lr_model.predict_proba(Xte_s)[:, 1]
p_lgb_raw = lgb_model.predict_proba(Xte_s)[:, 1]
p_lr_cal = platt(oof_lr, ytr, p_lr_raw)
p_lgb_cal = platt(oof_lgb, ytr, p_lgb_raw)
p_model = w_best * p_lr_cal + (1 - w_best) * p_lgb_cal

test = test.copy()
test['p_model_home'] = p_model
test['p_lr_home'] = p_lr_cal
test['p_lgb_home'] = p_lgb_cal
test['p_model_away'] = 1 - p_model
test['p_mkt_away'] = 1 - test.p_mkt_home
test['odds_h_2way'] = 1 / (test.p_mkt_home * (1 + VIG))
test['odds_a_2way'] = 1 / (test.p_mkt_away * (1 + VIG))

# ---- evaluation ----
def run_eval(games, label):
    pick_side = np.where(games.p_model_home >= 0.5, 'H', 'A')
    pick_p = np.where(pick_side == 'H', games.p_model_home, games.p_model_away)
    odds = np.where(pick_side == 'H', games.odds_h_2way, games.odds_a_2way)
    mkt_p = np.where(pick_side == 'H', games.p_mkt_home, games.p_mkt_away)
    win = np.where(pick_side == 'H', games.home_win.values == 1, games.home_win.values == 0).astype(float)
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
    # either-side gate: bet any side whose model prob exceeds market by eps
    for eps in [0.02, 0.05]:
        e_h = games.p_model_home - games.p_mkt_home
        e_a = games.p_model_away - games.p_mkt_away
        side = np.where((e_h > eps) & (e_h >= e_a), 'H', np.where(e_a > eps, 'A', '-'))
        bet = side != '-'
        o = np.where(side == 'H', games.odds_h_2way, games.odds_a_2way)
        w_ = np.where(side == 'H', games.home_win == 1, games.home_win == 0)
        r_ = np.where(w_ == 1, o - 1.0, -1.0)
        n = bet.sum()
        print(f"  either-side gate eps={eps}: n={n} roi={float(r_[bet].mean()) if n else None}")
    # quarter-Kelly (eps=0 selection, capped 5% stake)
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

m_all = run_eval(test, 'TEST Mar-Jun 2022 (all)')
run_eval(test[test.is_playoff == 0], 'TEST regular Mar-Apr')
run_eval(test[test.is_playoff == 1], 'TEST playoffs May-Jun')

# baselines
bh = test.home_win.mean()
roi_home = ((test.home_win == 1) * (test.odds_h_2way - 1) - (test.home_win == 0)).mean()
fav = np.where(test.p_mkt_home >= 0.5, 'H', 'A')
fodds = np.where(fav == 'H', test.odds_h_2way, test.odds_a_2way)
fwin = np.where(fav == 'H', test.home_win == 1, test.home_win == 0)
roi_fav = (fwin * (fodds - 1) - ~fwin).mean()
print(f"\nbaselines: home win% {bh:.4f} | always-home ROI {roi_home:+.4f} | "
      f"market-fav acc {fwin.mean():.4f} | market-fav ROI {roi_fav:+.4f}")
print(f"Brier: model {brier_score_loss(test.home_win, test.p_model_home):.4f} vs "
      f"market {brier_score_loss(test.home_win, test.p_mkt_home):.4f}")

# ---- ablations: LogReg C=0.01 with feature groups ----
groups = {
    'xg_only': [c for c in FEATS if 'xg' in c],
    'corsi_only': [c for c in FEATS if 'cf' in c],
    'pdo_goal_only': [c for c in FEATS if any(s in c for s in ['pdo', 'sh_pct', 'sv_pct', 'gf_', 'pts'])],
    'special_teams_only': [c for c in FEATS if any(s in c for s in ['pp_', 'pk_', 'pen'])],
    'schedule_only': [c for c in FEATS if any(s in c for s in ['rest', 'b2b', 'fo'])],
    'elo_only': [c for c in FEATS if 'elo' in c],
}
print("\nablations (LogReg C=0.01, Platt cal, 2-way payouts):")
abl = {}
for name, cols in groups.items():
    Xa = scaler.fit_transform(train[cols].values)
    oof = cv_oof(lambda: mk_logreg(0.01), Xa, ytr)
    m = mk_logreg(0.01).fit(Xa, ytr)
    p = platt(oof, ytr, m.predict_proba(scaler.transform(test[cols].values))[:, 1])
    side = np.where(p >= 0.5, 'H', 'A')
    o = np.where(side == 'H', test.odds_h_2way, test.odds_a_2way)
    w_ = np.where(side == 'H', test.home_win == 1, test.home_win == 0)
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
    rois.append(m_all['ret'][idx].mean())
ci = np.percentile(rois, [2.5, 97.5])
print(f"\nflat ROI bootstrap 95% CI (date-clustered): [{ci[0]:+.4f}, {ci[1]:+.4f}]")

# ---- ECE ----
bins = np.linspace(0, 1, 11)
ece = 0.0
for i in range(10):
    msk = (test.p_model_home >= bins[i]) & (test.p_model_home < bins[i + 1])
    if msk.sum():
        ece += msk.sum() / len(test) * abs(test.home_win[msk].mean() - test.p_model_home[msk].mean())
print(f"ECE: {ece:.5f}")

# ---- save CSV ----
cols = ['date', 'game_id', 'home_name', 'away_name', 'is_playoff',
        'home_goals', 'away_goals', 'home_win',
        'p_model_home', 'p_model_away', 'p_lr_home', 'p_lgb_home',
        'p_mkt_home', 'p_mkt_away',
        'home_odds_raw', 'away_odds_raw', 'odds_h_2way', 'odds_a_2way'] + FEATS
out = test.rename(columns={'home_odds': 'home_odds_raw', 'away_odds': 'away_odds_raw'}).copy()
out['model_pick'] = m_all['pick_side']
out['model_pick_odds'] = m_all['odds']
out['model_pick_prob'] = m_all['pick_p']
out['edge'] = m_all['edge']
out['flat_ret'] = m_all['ret']
out.to_csv('nhl_predictions.csv', index=False)
print("wrote nhl_predictions.csv", len(out))

metrics = dict(
    n_train=int(len(train)), n_test=int(len(test)),
    n_test_regular=int((test.is_playoff == 0).sum()), n_test_playoff=int((test.is_playoff == 1).sum()),
    test_accuracy=float(m_all['acc']), flat_roi=float(m_all['flat_roi']),
    model_brier=float(brier_score_loss(test.home_win, test.p_model_home)),
    market_brier=float(brier_score_loss(test.home_win, test.p_mkt_home)),
    roi_bootstrap_ci=[float(ci[0]), float(ci[1])],
    quarter_kelly_final=float(m_all['qkelly']),
    gates={str(k): v for k, v in m_all['gates'].items()},
    best_logreg=best_lr_key, best_lgbm=best_lgb_key, ensemble_weight=w_best,
    oof_brier_logreg=briers[best_lr_key], oof_brier_lgbm=briers[best_lgb_key],
    oof_brier_ensemble=w_brier_best, ece=float(ece),
    home_win_pct_test=float(bh), roi_always_home=float(roi_home),
    market_fav_accuracy=float(fwin.mean()), roi_market_fav=float(roi_fav),
    ablations=abl, hold=VIG,
)
with open('metrics.json', 'w') as fh:
    json.dump(metrics, fh, indent=2)
print(json.dumps({k: metrics[k] for k in ['n_test', 'test_accuracy', 'flat_roi', 'model_brier',
                                          'market_brier', 'quarter_kelly_final',
                                          'roi_bootstrap_ci']}, indent=2))
