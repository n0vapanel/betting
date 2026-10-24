"""Rolling no-lookahead NHL features v2: xG/corsi share (multi-span), PDO, special teams,
points %, rest, Elo, b2b."""
import pandas as pd
import numpy as np
from collections import defaultdict

tg = pd.read_parquet('team_games.parquet')
g = pd.read_parquet('games_full.parquet')
g['date'] = pd.to_datetime(g.date)
tg['date'] = pd.to_datetime(tg.date)

PRIOR = dict(
    xg5=0.5, cf5=0.5, xga=0.5, gfa=0.5,
    sh=0.09, sv=0.905, pdo=0.995,
    ppg=0.55, pkga=0.55,
    pen=8.0, fo=0.5, gf=3.0, ga=3.0, pts=1.0,
)

win_map = {(r.game_id, r.home_code): int(r.home_win) for _, r in g.iterrows()}
win_map.update({(r.game_id, r.away_code): 1 - int(r.home_win) for _, r in g.iterrows()})

state = {}
last_date = {}
feat_rows = []
tg = tg.sort_values('date').reset_index(drop=True)


def ewma_last(vals, span):
    return pd.Series(vals, dtype=float).ewm(span=span, adjust=True).mean().iloc[-1]


for i, r in tg.iterrows():
    t = r.team
    if t not in state:
        state[t] = {}
    def ew(key, span=10):
        s = state[t].get(key)
        return ewma_last(s if s else [PRIOR[key]], span)

    feats = dict(
        gameId=r.gameId, team=t, home_or_away=r.home_or_away, date=r.date,
        xg_share_5v5_10=ew('xg5', 10), xg_share_5v5_5=ew('xg5', 5), xg_share_5v5_20=ew('xg5', 20),
        cf_share_5v5=ew('cf5', 10),
        xg_share_all=ew('xga', 10), gf_share_all=ew('gfa', 10),
        pdo=ew('pdo', 10), pdo_20=ew('pdo', 20), sh_pct=ew('sh', 10), sv_pct=ew('sv', 10),
        pp_eff=ew('ppg', 10), pk_eff=ew('pkga', 10),
        pen_mins=ew('pen', 10), fo_pct=ew('fo', 10),
        gf_net=ew('gf', 10) - ew('ga', 10),
        pts_pct=ew('pts', 10),
        rest=min((r.date - last_date[t]).days, 7) if t in last_date else 7,
        b2b=int(t in last_date and (r.date - last_date[t]).days == 1),
        elo=1500.0,  # placeholder; replaced by true pre-game Elo below
    )
    # elo handled separately below; placeholder replaced
    feat_rows.append(feats)
    last_date[t] = r.date

    gf, ga = r.goalsFor_all, r.goalsAgainst_all
    st = state[t]
    def push(key, val):
        st.setdefault(key, []).append(val)
        if len(st[key]) > 30:
            st[key].pop(0)

    xgf5, xga5 = r.xGoalsFor_5on5, r.xGoalsAgainst_5on5
    cff5, cfa5 = r.shotAttemptsFor_5on5, r.shotAttemptsAgainst_5on5
    xgf_a, xga_a = r.xGoalsFor_all, r.xGoalsAgainst_all
    push('xg5', xgf5 / (xgf5 + xga5) if (xgf5 + xga5) > 0 else 0.5)
    push('cf5', cff5 / (cff5 + cfa5) if (cff5 + cfa5) > 0 else 0.5)
    push('xga', xgf_a / (xgf_a + xga_a) if (xgf_a + xga_a) > 0 else 0.5)
    push('gfa', gf / (gf + ga) if (gf + ga) > 0 else 0.5)
    sf, sa = r.shotsOnGoalFor_all, r.shotsOnGoalAgainst_all
    shp = gf / sf if sf > 0 else 0.09
    svp = 1 - ga / sa if sa > 0 else 0.905
    push('sh', shp); push('sv', svp); push('pdo', shp + svp)
    pp_opps = r.iceTime_5on4 / 120.0
    pk_opps = r.iceTime_4on5 / 120.0
    push('ppg', r.goalsFor_5on4 / pp_opps if pp_opps > 0 else PRIOR['ppg'])
    push('pkga', r.goalsAgainst_4on5 / pk_opps if pk_opps > 0 else PRIOR['pkga'])
    push('pen', r.penalityMinutesFor_all)
    fow, foa = r.faceOffsWonFor_all, r.faceOffsWonAgainst_all
    push('fo', fow / (fow + foa) if (fow + foa) > 0 else 0.5)
    push('gf', gf); push('ga', ga)
    won = win_map.get((r.gameId, t), int(gf > ga))
    pts = 2.0 if won else (1.0 if gf == ga else 0.0)
    push('pts', pts / 2.0)

feats = pd.DataFrame(feat_rows)

# ---- Elo (chronological, true winners incl. SO, HCA in expectation) ----
elo = defaultdict(lambda: 1500.0)
elo_before = {}
res = g[['game_id', 'date', 'home_code', 'away_code', 'home_win', 'home_goals', 'away_goals']].sort_values('date')
for _, r in res.iterrows():
    h, a = r.home_code, r.away_code
    elo_before[(r.game_id, h)] = elo[h]
    elo_before[(r.game_id, a)] = elo[a]
    we_h = 1 / (1 + 10 ** ((elo[a] - (elo[h] + 35.0)) / 400))
    res_h = 1 if r.home_win == 1 else 0
    gd = abs(r.home_goals - r.away_goals)
    mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else 2.0)
    elo[h] += 6.0 * mult * (res_h - we_h)
    elo[a] += 6.0 * mult * ((1 - res_h) - (1 - we_h))

feats['elo_before'] = [elo_before.get((gid, t), 1500.0) for gid, t in zip(feats.gameId, feats.team)]

feat_cols = [c for c in feats.columns
             if c not in ('gameId', 'team', 'home_or_away', 'date', 'elo')]
home = feats[feats.home_or_away == 'HOME'].set_index('gameId')
away = feats[feats.home_or_away == 'AWAY'].set_index('gameId')
home = home.rename(columns={c: f'h_{c}' for c in feat_cols})
away = away.rename(columns={c: f'a_{c}' for c in feat_cols})
pair = home.join(away, how='inner', lsuffix='_h', rsuffix='_a')
for c in feat_cols:
    if c == 'elo_before':
        pair[f'diff_{c}'] = (pair[f'h_{c}'] + 35.0) - pair[f'a_{c}']
    else:
        pair[f'diff_{c}'] = pair[f'h_{c}'] - pair[f'a_{c}']
pair = pair.reset_index().rename(columns={'index': 'gameId'})

out = g.merge(pair[['gameId'] + [f'diff_{c}' for c in feat_cols]],
              left_on='game_id', right_on='gameId', how='inner')
assert len(out) == len(g)
out.to_parquet('games_features.parquet', index=False)
print("games with features:", len(out), "| n features:", len(feat_cols))
print([f'diff_{c}' for c in feat_cols])
