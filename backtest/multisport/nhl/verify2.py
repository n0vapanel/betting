"""Exact-formula lookahead check: EWMA span-10 xG share + Elo + PDO for 3 spot games."""
import numpy as np
import pandas as pd
from collections import defaultdict

pred = pd.read_csv('nhl_predictions.csv', parse_dates=['date'])
games = pd.read_parquet('games_full.parquet')
games['date'] = pd.to_datetime(games.date)
tg = pd.read_parquet('team_games.parquet')
tg['date'] = pd.to_datetime(tg.date)
tg = tg.sort_values('date')

PRIOR = 0.5
state = {}
def push(t, key, val):
    state.setdefault(t, {}).setdefault(key, []).append(val)
    if len(state[t][key]) > 30:
        state[t][key].pop(0)

# walk all games BEFORE each spot date to build exact state
spot_ids = pred.sample(3, random_state=3).game_id.tolist()
spot_dates = {r.game_id: r.date for _, r in pred[pred.game_id.isin(spot_ids)].iterrows()}
max_d = max(spot_dates.values())

for _, r in tg[tg.date < max_d].iterrows():
    t = r.team
    xgf, xga = r.xGoalsFor_5on5, r.xGoalsAgainst_5on5
    push(t, 'xg5', xgf / (xgf + xga) if (xgf + xga) > 0 else 0.5)

def ewma(t, key, span=10):
    vals = state.get(t, {}).get(key, [PRIOR])
    return pd.Series(vals, dtype=float).ewm(span=span, adjust=True).mean().iloc[-1]

for gid in spot_ids:
    gr = games[games.game_id == gid].iloc[0]
    r = pred[pred.game_id == gid].iloc[0]
    st = {}
    for _, rr in tg[tg.date < gr.date].iterrows():
        st.setdefault(rr.team, {}).setdefault('xg5', []).append(
            (rr.xGoalsFor_5on5 / (rr.xGoalsFor_5on5 + rr.xGoalsAgainst_5on5))
            if (rr.xGoalsFor_5on5 + rr.xGoalsAgainst_5on5) > 0 else 0.5)
    def ew(t):
        vals = st.get(t, {}).get('xg5', [PRIOR])[-30:]
        return pd.Series(vals, dtype=float).ewm(span=10, adjust=True).mean().iloc[-1]
    diff = ew(gr.home_code) - ew(gr.away_code)
    print(f"{gr.date.date()} {gr.home_code} vs {gr.away_code}: csv={r['diff_xg_share_5v5_10']:.4f} exact_ewma={diff:.4f}")
