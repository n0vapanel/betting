import pandas as pd

df = pd.read_csv('all_teams.csv', low_memory=False)
print("cols:", len(df.columns), "rows:", len(df))
df21 = df[df.season == 2021]
print("2021-22 season rows:", len(df21))
print("gameDate range:", df21.gameDate.min(), "->", df21.gameDate.max())
print("situations:", df21.situation.value_counts(dropna=False).to_dict())
print("home_or_away:", df21.home_or_away.value_counts(dropna=False).to_dict())
print("playoffGame:", df21.playoffGame.value_counts(dropna=False).to_dict())
print("n games (gameId):", df21.gameId.nunique())
print("teams:", df21.team.nunique(), df21.team.unique()[:40])
# season summary teams file for name mapping
t = pd.read_csv('https://moneypuck.com/moneypuck/playerData/seasonSummary/2021/regular/teams.csv')
print(t.columns.tolist())
print(t[['team', 'name']].head(5).to_dict('records'))
