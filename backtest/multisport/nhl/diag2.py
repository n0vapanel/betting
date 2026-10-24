import pandas as pd

ar = pd.read_csv('kaggle_arch/games.csv', low_memory=False)
ar = ar[(ar.competition == 'NHL') & (ar.season == '2021-22')].copy()
ar['date'] = pd.to_datetime(ar.date)
real = ~ar.home_team.str.contains('Division') & ~ar.away_team.str.contains('Division')
ar = ar[real]
oct21 = ar[(ar.date >= '2021-10-01') & (ar.date <= '2021-10-20')]
print(oct21.date.value_counts().sort_index())
print("total rows oct:", len(oct21))
# show a few rows
print(oct21[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head(12).to_string())
# unique dates overall
print("n unique dates:", ar.date.nunique())
print("min/max:", ar.date.min(), ar.date.max())
# rows per month
print(ar.date.dt.to_period('M').value_counts().sort_index())
