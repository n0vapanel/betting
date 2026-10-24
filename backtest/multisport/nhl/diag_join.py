import pandas as pd
from build_data import CODE2NAME

mp = pd.read_csv('all_teams.csv', low_memory=False)
mp = mp[mp.season == 2021].copy()
mp['date'] = pd.to_datetime(mp.gameDate.astype(str), format='%Y%m%d')

ar = pd.read_csv('kaggle_arch/games.csv', low_memory=False)
ar = ar[(ar.competition == 'NHL') & (ar.season == '2021-22')].copy()
ar['date'] = pd.to_datetime(ar.date)
real = ~ar.home_team.str.contains('Division') & ~ar.away_team.str.contains('Division')
ar = ar[real]

ar_names = sorted(set(ar.home_team) | set(ar.away_team))
mp_names = sorted(set(CODE2NAME.values()))
print("archangel names:", len(ar_names), ar_names)
print("moneypuck names:", len(mp_names))
print("in arch not in mp map:", sorted(set(ar_names) - set(mp_names)))
print("in mp map not in arch:", sorted(set(mp_names) - set(ar_names)))

# sample same-date comparison
d = '2021-10-12'
print("\narchangel", d)
print(ar[ar.date == d][['date', 'home_team', 'away_team', 'home_score', 'away_score']].to_string())
