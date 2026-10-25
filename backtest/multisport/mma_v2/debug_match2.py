import pandas as pd
import numpy as np
import os
base = os.path.dirname(os.path.abspath(__file__))

raw = pd.read_csv(os.path.join(base, "raw", "raw_total_fight_data.csv"), sep=";")
raw.columns = [c.strip() for c in raw.columns]
raw["date"] = pd.to_datetime(raw["date"])

md = pd.read_csv(os.path.join(base, "raw_mdabbert", "ufc-master.csv"))
md["date"] = pd.to_datetime(md["date"])
md_old = md[md["date"] <= pd.Timestamp("2021-03-20")]
print("md rows <=2021-03-20:", len(md_old), "| min/max:", md_old["date"].min(), md_old["date"].max())

frames = []
for corner in ["R", "B"]:
    sub = raw[["date", f"{corner}_fighter", f"{corner}_SIG_STR."]].rename(
        columns={f"{corner}_fighter": "fighter", f"{corner}_SIG_STR.": "sig_landed"})
    frames.append(sub)
fights = pd.concat(frames, ignore_index=True)
fights = fights.dropna(subset=["fighter"])
fights["fighter"] = fights["fighter"].str.strip()
fights["sig_landed"] = pd.to_numeric(fights["sig_landed"], errors="coerce")
fights = fights.sort_values("date")
print("fights rows:", len(fights))

# pick 3 specific mdabbert rows and trace
for i in [0, 10, 100]:
    r = md_old.iloc[i]
    for corner, name in [("R", "R_fighter"), ("B", "B_fighter")]:
        f = str(r[name]).strip()
        sub = fights[(fights["date"] == r["date"]) & (fights["fighter"] == f)]
        print(f"md row {i} {r['date'].date()} {corner}='{f}' -> rajeevw rows at same date:", len(sub))
        if len(sub) == 0:
            sub3 = fights[(fights["fighter"] == f)]
            print("   fighter in rajeevw at any date:", len(sub3), "| dates:", sub3["date"].min() if len(sub3) else None, sub3["date"].max() if len(sub3) else None)
            dsub = fights[fights["date"] == r["date"]]
            print("   rajeevw rows on that date:", len(dsub), "| fighters:", dsub["fighter"].head(6).tolist())
