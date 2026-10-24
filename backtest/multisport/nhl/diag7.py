import pandas as pd

t = pd.read_parquet('team_games.parquet')
print(t.columns.tolist())
print(t.shape)
print(t.head(2).to_string())
