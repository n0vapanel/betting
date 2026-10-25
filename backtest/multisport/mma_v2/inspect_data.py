import pandas as pd
import glob, os
base = os.path.dirname(os.path.abspath(__file__))

for name in ["data.csv", "preprocessed_data.csv", "raw_fighter_details.csv", "raw_total_fight_data.csv"]:
    p = os.path.join(base, "raw", name)
    df = pd.read_csv(p, nrows=50)
    print(f"=== {name} shape(rows sampled 50) cols {len(df.columns)} ===")
    cols = list(df.columns)
    print(cols)
    odds = [c for c in cols if any(k in c.lower() for k in ["odds", "b365", "365", "closing", "money"])]
    print("ODDS-LIKE:", odds)
    print()

files = sorted(glob.glob(os.path.join(base, "raw_new", "**", "*"), recursive=True))
print("=== maksbasher files ===")
for f in files:
    if os.path.isfile(f):
        print(os.path.relpath(f, base), os.path.getsize(f))
