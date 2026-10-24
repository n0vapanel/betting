"""Verification: recompute headline numbers from nhl_predictions.csv; gate CI; lookahead spot-check."""
import numpy as np
import pandas as pd

pred = pd.read_csv('nhl_predictions.csv', parse_dates=['date'])
print("rows:", len(pred), "| dates:", pred.date.min().date(), "->", pred.date.max().date())
print("playoff rows:", pred.is_playoff.sum())

# --- recompute flat ROI from CSV ---
pick_odds = pred.model_pick_odds
win = (pred.model_pick == 'H') & (pred.home_win == 1) | (pred.model_pick == 'A') & (pred.home_win == 0)
ret = np.where(win, pick_odds - 1.0, -1.0)
print("recomputed flat ROI:", round(ret.mean(), 6), "| csv flat_ret mean:", round(pred.flat_ret.mean(), 6))
print("recomputed acc:", round(win.mean(), 6), "| pick counts:", pred.model_pick.value_counts().to_dict())
print("edge mean:", round(pred.edge.mean(), 4))

# --- gate CI (model-side eps=0.02) ---
edge = pred.edge.values
bet = edge > 0.02
r = ret[bet]
print(f"\ngate eps=0.02: n={bet.sum()} roi={r.mean():+.4f}")
rng = np.random.default_rng(7)
dates_all = pred.date.values
# bootstrap over dates, within the bet subset
bet_dates = pred.date.values[bet]
rois = []
for _ in range(2000):
    d = rng.choice(pd.unique(bet_dates), size=pd.unique(bet_dates).size, replace=True)
    idx = np.isin(bet_dates, d)
    rois.append(r[idx].mean())
ci = np.percentile(rois, [2.5, 97.5])
print(f"gate eps=0.02 bootstrap 95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
ci002 = [float(ci[0]), float(ci[1])]

# also eps=0.05
bet5 = edge > 0.05
r5 = ret[bet5]
print(f"gate eps=0.05: n={bet5.sum()} roi={r5.mean():+.4f}")
bet_dates5 = pred.date.values[bet5]
rois5 = []
for _ in range(2000):
    d = rng.choice(pd.unique(bet_dates5), size=pd.unique(bet_dates5).size, replace=True)
    idx = np.isin(bet_dates5, d)
    rois5.append(r5[idx].mean())
ci5 = np.percentile(rois5, [2.5, 97.5])
print(f"gate eps=0.05 bootstrap 95% CI: [{ci5[0]:+.4f}, {ci5[1]:+.4f}]")
ci005 = [float(ci5[0]), float(ci5[1])]

# --- lookahead spot-check: recompute features for 3 test games from prior games only ---
games = pd.read_parquet('games_full.parquet', columns=['game_id', 'date', 'home_code', 'away_code',
                                                       'home_win', 'home_goals', 'away_goals'])
tg = pd.read_parquet('team_games.parquet')
tg['date'] = pd.to_datetime(tg.date)
games['date'] = pd.to_datetime(games.date)

def prior_xg_share(team, before_date, n=10):
    hist = tg[(tg.team == team) & (tg.date < before_date)].sort_values('date').tail(n)
    if len(hist) == 0:
        return 0.5
    xgf = hist.xGoalsFor_5on5.sum()
    xga = hist.xGoalsAgainst_5on5.sum()
    return xgf / (xgf + xga) if (xgf + xga) > 0 else 0.5

def prior_rest(team, before_date):
    hist = tg[(tg.team == team) & (tg.date < before_date)]
    if len(hist) == 0:
        return 7
    return min((before_date - hist.date.max()).days, 7)

spot = pred.sample(3, random_state=3)
for _, r in spot.iterrows():
    h = r.home_name
    # map name->code via games_full
    gr = games[games.game_id == r.game_id].iloc[0]
    hx = prior_xg_share(gr.home_code, r.date)
    ax = prior_xg_share(gr.away_code, r.date)
    diff = hx - ax
    hr = prior_rest(gr.home_code, r.date)
    ar = prior_rest(gr.away_code, r.date)
    print(f"\nspot {r.date.date()} {gr.home_code} vs {gr.away_code}:")
    print(f"  csv diff_xg_share_5v5_10={r['diff_xg_share_5v5_10']:.4f} | recomputed={diff:.4f}")
    print(f"  csv diff_rest={r['diff_rest']:.2f} | recomputed={hr - ar:.2f}")

import json
with open('verify.json', 'w') as fh:
    json.dump(dict(flat_roi_recomputed=float(ret.mean()), acc_recomputed=float(win.mean()),
                   gate_002_ci=ci002, gate_005_ci=ci005), fh, indent=2)
print("\nwrote verify.json")
