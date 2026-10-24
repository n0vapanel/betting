import pandas as pd
import os
base = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(base, "raw_mdabbert", "ufc-master.csv")
df = pd.read_csv(p)
print("shape:", df.shape)
print("cols:", list(df.columns))
dcol = [c for c in df.columns if "date" in c.lower() or "Date" in c]
print("date cols:", dcol)
for c in dcol:
    print(c, "min/max:", df[c].min(), "/", df[c].max())
odds = [c for c in df.columns if any(k in c.lower() for k in ["odds", "365", "closing", "money", "bet"])]
print("ODDS-LIKE:", odds)
print(df.head(3).to_string())
