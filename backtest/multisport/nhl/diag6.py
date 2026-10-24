import pandas as pd

res = pd.read_parquet('games_full.parquet')
reg = res[res.is_playoff == 0].copy()
reg['mkt_fav_side'] = reg.p_mkt_home.apply(lambda p: 'H' if p > 0.5 else ('A' if p < 0.5 else 'E'))
fav_win = (reg.mkt_fav_side == 'H') == (reg.home_win == 1)
print("arch market favorite win% (excl E):", round(fav_win[reg.mkt_fav_side != 'E'].mean(), 4),
      "| n =", (reg.mkt_fav_side != 'E').sum())
print("avg |implied home prob - 0.5|:", round((reg.p_mkt_home - 0.5).abs().mean(), 4))
print("home_odds avg:", round(reg.home_odds.mean(), 3), "| away_odds avg:", round(reg.away_odds.mean(), 3))

mall = pd.read_csv('kaggle_odds/sportsbook-nhl-2021-2022.csv')
mall['date'] = pd.to_datetime(mall.date)
MALLCODE = {'TB': 'TBL', 'LA': 'LAK', 'SJ': 'SJS', 'NJ': 'NJD', 'WAS': 'WSH'}
mall['h_code'] = mall.h__team.map(MALLCODE).fillna(mall.h__team)
mall['a_code'] = mall.a__team.map(MALLCODE).fillna(mall.a__team)
m2 = reg.merge(mall, left_on=['date', 'home_code', 'away_code'],
               right_on=['date', 'h_code', 'a_code'], how='inner')
m2['p_fav'] = m2.moneyline.abs() / (m2.moneyline.abs() + 100)
m2.loc[m2.fav == 'Even', 'p_fav'] = 0.5
m2['p_home_implied'] = m2.p_fav.where(m2.fav == 'Home', 1 - m2.p_fav)
m2.loc[m2.fav == 'Even', 'p_home_implied'] = 0.5
brier_mall = ((m2.p_home_implied - m2.home_win) ** 2).mean()
print("mallari implied Brier:", round(brier_mall, 4))
fwin = (((m2.fav == 'Home') & (m2.home_win == 1)) | ((m2.fav == 'Away') & (m2.home_win == 0))).mean()
print("mallari favorite win% (incl Even as no-bet):", round(fwin, 4), "| Even count:", (m2.fav == 'Even').sum())

d = m2[['date', 'home_name', 'away_name', 'p_mkt_home', 'p_home_implied', 'home_win']].copy()
d['diff'] = (d.p_mkt_home - d.p_home_implied).abs()
print("\nmedian |diff|:", round(d['diff'].median(), 4), "| mean:", round(d['diff'].mean(), 4),
      "| corr:", round(d.p_mkt_home.corr(d.p_home_implied), 4))
print("\nworst 12 disagreements:")
print(d.sort_values('diff', ascending=False).head(12).to_string())
# where do big disagreements come from? check if arch probs are extreme
big = d[d['diff'] > 0.15]
print("\nbig disagreements:", len(big), "of", len(d))
