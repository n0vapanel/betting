"""Elo spot-check v2: rebuild Elo per spot date (strictly prior games only)."""
import pandas as pd
from collections import defaultdict

pred = pd.read_csv('nhl_predictions.csv', parse_dates=['date'])
games = pd.read_parquet('games_full.parquet')
games['date'] = pd.to_datetime(games.date)
games = games.sort_values('date')

spot_ids = pred.sample(3, random_state=5).game_id.tolist()

for gid in spot_ids:
    gr = games[games.game_id == gid].iloc[0]
    r = pred[pred.game_id == gid].iloc[0]
    elo = defaultdict(lambda: 1500.0)
    for p in games[games.date < gr.date].itertuples():
        we_h = 1 / (1 + 10 ** ((elo[p.away_code] - (elo[p.home_code] + 35.0)) / 400))
        res_h = 1 if p.home_win == 1 else 0
        gd = abs(p.home_goals - p.away_goals)
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else 2.0)
        elo[p.home_code] += 6.0 * mult * (res_h - we_h)
        elo[p.away_code] += 6.0 * mult * ((1 - res_h) - (1 - we_h))
    rec = (elo[gr.home_code] + 35.0) - elo[gr.away_code]
    print(f"{gr.date.date()} {gr.home_code} vs {gr.away_code}: csv={r['diff_elo_before']:.3f} rebuilt={rec:.3f}")
