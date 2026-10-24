import pandas as pd

ar = pd.read_csv('kaggle_arch/games.csv', low_memory=False)
ar = ar[(ar.competition == 'NHL') & (ar.season == '2021-22')].copy()
ar['date'] = pd.to_datetime(ar.date)
# all archangel rows involving these teams around those dates
for d, h, a in [('2022-01-18', 'Philadelphia Flyers', 'New York Islanders'),
                ('2022-01-15', 'Arizona Coyotes', 'Colorado Avalanche'),
                ('2022-01-22', 'Minnesota Wild', 'Chicago Blackhawks'),
                ('2022-02-01', 'New Jersey Devils', 'Toronto Maple Leafs')]:
    d0 = pd.Timestamp(d)
    w = ar[(ar.date >= d0 - pd.Timedelta(days=3)) & (ar.date <= d0 + pd.Timedelta(days=3))]
    w = w[w.home_team.isin([h, a]) | w.away_team.isin([h, a])]
    print(w[['date', 'home_team', 'away_team', 'home_score', 'away_score',
             'avg_home_odds', 'avg_away_odds']].to_string())
    print('---')
