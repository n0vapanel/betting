import pandas as pd
from collections import defaultdict
from build_data import CODE2NAME

res = pd.read_parquet('games_full.parquet')
used = set(res.ar_idx.tolist())
ar = pd.read_csv('kaggle_arch/games.csv', low_memory=False)
ar = ar[(ar.competition == 'NHL') & (ar.season == '2021-22')].copy()
ar['date'] = pd.to_datetime(ar.date)
real = ~ar.home_team.str.contains('Division') & ~ar.away_team.str.contains('Division')
ar = ar[real].reset_index(drop=True)
ar = ar[ar.date >= '2021-10-11']
unused = ar[~ar.index.isin(used)]
print("archangel rows:", len(ar), "| unused:", len(unused))
print(unused[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head(40).to_string())

# ---- odds quality: Brier of each source on regular-season games ----
res['date'] = pd.to_datetime(res.date)
reg = res[res.is_playoff == 0].copy()
reg['p_mkt_home'] = (1 / reg.home_odds) / ((1 / reg.home_odds) + (1 / reg.away_odds))
brier_arch = ((reg.p_mkt_home - reg.home_win) ** 2).mean()
print("\narchangel de-vig market Brier (regular):", round(brier_arch, 4),
      "| n =", len(reg), "| home win%:", round(reg.home_win.mean(), 4))
print("market favorite win% (arch):", round((reg.mkt_fav.replace({'H': 1, 'A': 0}).astype(float)
                                            .where(reg.mkt_fav != 'E') == reg.home_win
                                            .where(reg.mkt_fav != 'E')).mean(), 4))

mall = pd.read_csv('kaggle_odds/sportsbook-nhl-2021-2022.csv')
mall['date'] = pd.to_datetime(mall.date)
MALLCODE = {'TB': 'TBL', 'LA': 'LAK', 'SJ': 'SJS', 'NJ': 'NJD', 'WAS': 'WSH'}
mall['h_code'] = mall.h__team.map(MALLCODE).fillna(mall.h__team)
mall['a_code'] = mall.a__team.map(MALLCODE).fillna(mall.a__team)
m2 = reg.merge(mall, left_on=['date', 'home_code', 'away_code'],
               right_on=['date', 'h_code', 'a_code'], how='inner')
m2['p_fav'] = m2.moneyline.abs() / (m2.moneyline.abs() + 100)
m2['p_home_implied'] = m2.p_fav.where(m2.fav == 'Home', 1 - m2.p_fav)
m2['fav_win'] = m2.fav.map({'Home': 'home_win', 'Away': 'away_win'})
# de-vig mallari by assuming book = (p_fav + p_dog) with p_dog = 1-p_fav -> vig-free normalization
# implied home/away probs: p_fav on favorite side, 1-p_fav on dog side; normalize sum = p_fav + (1-p_fav) = 1 -> no vig to remove.
brier_mall = ((m2.p_home_implied - m2.home_win) ** 2).mean()
print("mallari implied market Brier (regular):", round(brier_mall, 4), "| n =", len(m2))
fwin = (m2.moneyline < 0) & ((m2.fav == 'Home') == (m2.home_win == 1))
print("mallari favorite win%:", round(fwin.mean(), 4))

# head-to-head prob comparison
d = m2[['date', 'home_name', 'away_name', 'p_mkt_home', 'p_home_implied', 'home_win']].copy()
d['diff'] = (d.p_mkt_home - d.p_home_implied).abs()
print("\nworst prob disagreements:")
print(d.sort_values('diff', ascending=False).head(10).to_string())
print("\nmedian |diff|:", round(d['diff'].median(), 4), "| mean |diff|:", round(d['diff'].mean(), 4))
