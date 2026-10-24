"""Final join: moneypuck games + archangel closing odds + mallari cross-check."""
import pandas as pd
from collections import defaultdict
from build_data import CODE2NAME

mp = pd.read_csv('all_teams.csv', low_memory=False)
mp = mp[mp.season == 2021].copy()
mp['date'] = pd.to_datetime(mp.gameDate.astype(str), format='%Y%m%d')

ar = pd.read_csv('kaggle_arch/games.csv', low_memory=False)
ar = ar[(ar.competition == 'NHL') & (ar.season == '2021-22')].copy()
ar['date'] = pd.to_datetime(ar.date)
real = ~ar.home_team.str.contains('Division') & ~ar.away_team.str.contains('Division')
ar = ar[real].reset_index(drop=True)
ar = ar[ar.date >= '2021-10-11']

# moneypuck game table
allrows = mp[mp.situation == 'all'].copy()
rows = []
for gid, grp in allrows.groupby('gameId'):
    home = grp[grp.home_or_away == 'HOME'].iloc[0]
    away = grp[grp.home_or_away == 'AWAY'].iloc[0]
    rows.append(dict(date=home.date, home_code=home.team, away_code=away.team,
                     home_name=CODE2NAME[home.team], away_name=CODE2NAME[away.team],
                     mp_hg=int(home.goalsFor), mp_ag=int(away.goalsFor),
                     game_id=gid, is_playoff=int(home.playoffGame)))
g = pd.DataFrame(rows)

lookup = defaultdict(list)
for i, r in ar.iterrows():
    lookup[r.date].append((r.home_team, r.away_team, r.home_score, r.away_score,
                           r.avg_home_odds, r.avg_away_odds, i))

def compat(mp_hg, mp_ag, hs, as_):
    if mp_hg == hs and mp_ag == as_:
        return 0
    if mp_hg == mp_ag and ((hs == mp_hg + 1 and as_ == mp_ag) or (as_ == mp_ag + 1 and hs == mp_hg)):
        return 1  # SO game
    return 2

recs = []
for i, r in g.iterrows():
    best = None
    for off in (0, 1, -1, 2, -2):
        d = r.date + pd.Timedelta(days=off)
        for (h, a, hs, as_, oh, oa, idx) in lookup.get(d, []):
            if {h, a} != {r.home_name, r.away_name}:
                continue
            if h == r.home_name and a == r.away_name:
                c = compat(r.mp_hg, r.mp_ag, hs, as_)
                cand = (abs(off), c, idx, False, off, hs, as_, oh, oa)
            else:
                c = compat(r.mp_hg, r.mp_ag, as_, hs)
                cand = (abs(off), c, idx, True, off, as_, hs, oa, oh)
            if best is None or (cand[1], cand[0]) < (best[1], best[0]):
                best = cand
    recs.append((i, best))

out = []
for i, b in recs:
    r = g.iloc[i]
    d = dict(game_id=r.game_id, date=r.date, home_code=r.home_code, away_code=r.away_code,
             home_name=r.home_name, away_name=r.away_name, is_playoff=r.is_playoff,
             mp_hg=r.mp_hg, mp_ag=r.mp_ag)
    if b is None:
        d.update(ar_idx=-1, compat=None, offset=None, home_goals=None, away_goals=None,
                 home_odds=None, away_odds=None)
    else:
        d.update(ar_idx=b[2], flip=b[3], offset=b[4], home_goals=b[5], away_goals=b[6],
                 home_odds=b[7], away_odds=b[8], compat=b[1])
    out.append(d)

res = pd.DataFrame(out)
res['winner'] = None
has_score = res.home_goals.notna()
res.loc[has_score, 'winner'] = (res.loc[has_score, 'home_goals'] > res.loc[has_score, 'away_goals']).astype(int)
res['home_win'] = res['winner']

print("games:", len(res))
print("with odds:", res.home_odds.notna().sum())
print("compat: 0=exact 1=SO-diff 2=conflict")
print(res.compat.value_counts(dropna=False).to_dict())
print("offsets:", res.offset.value_counts().sort_index().to_dict())
conf = res[res.compat == 2]
print("conflicts:", len(conf))
print(conf[['date', 'home_name', 'away_name', 'mp_hg', 'mp_ag', 'home_goals', 'away_goals']].to_string())
# SO games: moneypuck tied but archangel decided
so = res[res.compat == 1]
print("\nSO games:", len(so))
# de-vigged market probs
res['p_mkt_home_raw'] = 1 / res.home_odds
res['p_mkt_away_raw'] = 1 / res.away_odds
res['p_mkt_home'] = res.p_mkt_home_raw / (res.p_mkt_home_raw + res.p_mkt_away_raw)
res['p_mkt_away'] = 1 - res.p_mkt_home
res['mkt_fav'] = res.p_mkt_home.apply(lambda p: 'H' if p > 0.5 else ('A' if p < 0.5 else 'E'))

# ---- mallari cross-check (regular season, local dates, codes) ----
mall = pd.read_csv('kaggle_odds/sportsbook-nhl-2021-2022.csv')
mall['date'] = pd.to_datetime(mall.date)
MALLCODE = {'TB': 'TBL', 'LA': 'LAK', 'SJ': 'SJS', 'NJ': 'NJD', 'WAS': 'WSH'}
mall['h_code'] = mall.h__team.map(MALLCODE).fillna(mall.h__team)
mall['a_code'] = mall.a__team.map(MALLCODE).fillna(mall.a__team)
mall['h_tot'] = mall.h__goals_total.astype(int)
mall['a_tot'] = mall.a__goals_total.astype(int)
res2 = res.merge(mall[['date', 'h_code', 'a_code', 'h_tot', 'a_tot', 'moneyline', 'fav']],
                 left_on=['date', 'home_code', 'away_code'],
                 right_on=['date', 'h_code', 'a_code'], how='left')
print("\nmallari joined:", res2.moneyline.notna().sum(), "/", len(res2))
j = res2[res2.moneyline.notna()].copy()
# score check
ok = (j.home_goals == j.h_tot) & (j.away_goals == j.a_tot)
print("mallari score agreement:", ok.sum(), "/", len(j))
bad = j[~ok]
print(bad[['date', 'home_name', 'away_name', 'home_goals', 'h_tot', 'away_goals', 'a_tot', 'moneyline', 'fav']].head(12).to_string())
# favorite side check: mallari fav vs archangel implied fav
j2 = j.dropna(subset=['p_mkt_home'])
fav_ok = ((j2.fav == 'Home') & (j2.p_mkt_home > 0.5)) | ((j2.fav == 'Away') & (j2.p_mkt_home < 0.5))
print("favorite side agreement:", fav_ok.sum(), "/", len(j2))
# implied fav prob from mallari american odds vs archangel de-vig prob
j2['ml_abs'] = j2.moneyline.abs()
j2['imp_fav'] = j2.ml_abs / (j2.ml_abs + 100)
j2['arch_fav_prob'] = j2.p_mkt_home.where(j2.fav == 'Home', 1 - j2.p_mkt_home)
print("corr(implied fav prob mallari vs arch):", j2.imp_fav.corr(j2.arch_fav_prob).round(4))
res2.drop(columns=['h_code', 'a_code', 'h_tot', 'a_tot', 'moneyline', 'fav'], inplace=True)
res2.to_parquet('games_full.parquet', index=False)
print("\nsaved games_full.parquet")
print("test window (Mar-Jun):", res2[(res2.date >= '2022-03-01')].home_odds.notna().sum(), "of",
      (res2.date >= '2022-03-01').sum())
