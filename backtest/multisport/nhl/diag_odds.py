import pandas as pd

res = pd.read_parquet('games_full.parquet')
res['date'] = pd.to_datetime(res.date)
mall = pd.read_csv('kaggle_odds/sportsbook-nhl-2021-2022.csv')
mall['date'] = pd.to_datetime(mall.date)
MALLCODE = {'TB': 'TBL', 'LA': 'LAK', 'SJ': 'SJS', 'NJ': 'NJD', 'WAS': 'WSH'}
mall['h_code'] = mall.h__team.map(MALLCODE).fillna(mall.h__team)
mall['a_code'] = mall.a__team.map(MALLCODE).fillna(mall.a__team)
m = res.merge(mall, left_on=['date', 'home_code', 'away_code'],
              right_on=['date', 'h_code', 'a_code'], how='inner')
m['imp_fav'] = m.moneyline.abs() / (m.moneyline.abs() + 100)
m.loc[m.fav == 'Even', 'imp_fav'] = 0.5
# archangel raw odds on the mallari-favorite side
m['arch_raw_fav'] = m.home_odds.where(m.fav == 'Home', m.away_odds)
# ratio of archangel raw odds to fair 2-way odds implied by mallari
m['ratio'] = m.arch_raw_fav / (1 / m.imp_fav)
print("n:", len(m))
print("ratio mean/median:", round(m['ratio'].mean(), 4), round(m['ratio'].median(), 4))
print("ratio by |moneyline| bucket:")
m['ml_abs'] = m.moneyline.abs()
for lo, hi in [(0, 110), (110, 140), (140, 180), (180, 250), (250, 1000)]:
    s = m[(m.ml_abs > lo) & (m.ml_abs <= hi)]
    print(f"  fav |ml| in ({lo},{hi}]: n={len(s)} ratio={s['ratio'].mean():.3f} arch_raw_fav={s.arch_raw_fav.mean():.3f}")
# implied 2-way fair odds from mallari = 1/imp_fav ; with 4.76% vig -> 1/(imp_fav*1.0476)
# compare with archangel raw
m['fair_2way_vig'] = 1 / (m.imp_fav * 1.0476)
print("\narch raw fav odds mean:", round(m.arch_raw_fav.mean(), 3),
      "| fair 2-way with vig mean:", round(m.fair_2way_vig.mean(), 3))
# also the de-vigged normalized arch prob vs mallari imp prob on fav side
m['arch_p_fav'] = m.p_mkt_home.where(m.fav == 'Home', 1 - m.p_mkt_home)
d = (m.arch_p_fav - m.imp_fav).abs()
print("|de-vig arch p_fav - mallari imp_fav| mean:", round(d.mean(), 4), "median:", round(d.median(), 4))
# check jonathanncoletti data
jh = pd.read_csv('kaggle_hist/nhl_data_plus.csv', low_memory=False)
jh = jh[jh.season == 2021].copy()
jh['date'] = pd.to_datetime(jh['date'], utc=True).dt.tz_localize(None)
print("\njh rows 2021:", len(jh), "| fav ml non-null:", jh.favorite_moneyline.notna().sum())
