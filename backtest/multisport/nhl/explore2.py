import pandas as pd

g = pd.read_csv('kaggle_arch/games.csv', low_memory=False)
s = g[(g.competition == 'NHL') & (g.season == '2021-22')]
names = sorted(set(s.home_team) | set(s.away_team))
print(len(names))
print(names)

# mallari codes for cross-check
m = pd.read_csv('kaggle_odds/sportsbook-nhl-2021-2022.csv')
print("mallari rows:", len(m))
print("mallari date range:", m.date.min(), "->", m.date.max())
print("mallari teams:", sorted(set(m.a__team) | set(m.h__team)))
