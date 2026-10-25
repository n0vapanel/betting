import pandas as pd
import os
base = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(base, "raw_new", "UFC dataset", "Large set", "large_dataset.csv")
df = pd.read_csv(p, nrows=50)
print("large_dataset cols:", list(df.columns))
print()
odds = [c for c in df.columns if any(k in c.lower() for k in ["odds", "365", "closing", "money", "bet"])]
print("ODDS-LIKE:", odds)
print()
print(df.head(3).to_string())
print()
# check full file shape cheaply via chunk
tot = sum(1 for _ in pd.read_csv(p, usecols=[0], chunksize=100000))
print("total rows:", tot)
