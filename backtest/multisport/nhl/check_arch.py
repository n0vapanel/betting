import pandas as pd

g = pd.read_csv('kaggle_arch/games.csv', low_memory=False)
nhl = g[g.competition == 'NHL'].copy()
s = nhl[nhl.season == '2021-22'].copy()
ids = set(s.game_id)
co = pd.read_csv('kaggle_arch/closing_odds.csv',
                 usecols=['game_id', 'bookmaker', 'market', 'home_odds', 'away_odds',
                          'last_update_utc', 'line_active'], low_memory=False)
co_nhl = co[co.game_id.isin(ids) & (co.market == 'moneyline')]
print("odds rows for NHL 2021-22:", len(co_nhl), "| distinct games:", co_nhl.game_id.nunique(), "of", len(ids))
print("bookmakers:", co_nhl.bookmaker.nunique())
co_nhl['last_update_utc'] = pd.to_datetime(co_nhl.last_update_utc, errors='coerce')
print("last_update range:", co_nhl.last_update_utc.min(), "->", co_nhl.last_update_utc.max())
s['date'] = pd.to_datetime(s.date)
test = s[s.date >= '2022-03-01']
print("test-period games:", len(test), "| with odds rows:", test.game_id.isin(co_nhl.game_id).sum())
# regular season only
s['month'] = s.date.dt.month
reg = s[(s.date >= '2021-10-01') & (s.date <= '2022-04-30')]
print("regular-ish games Oct-Apr:", len(reg), "| with odds rows:", reg.game_id.isin(co_nhl.game_id).sum())
print("sample row:")
print(s[['game_id', 'date', 'season', 'home_team', 'away_team', 'home_score', 'away_score',
         'avg_home_odds', 'avg_away_odds']].head(3).to_string())
