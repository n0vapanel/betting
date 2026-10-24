import pandas as pd
import numpy as np
import os
base = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base, "raw_mdabbert", "ufc-master.csv"))
df["date"] = pd.to_datetime(df["date"])
print("rows:", len(df), "| dates:", df["date"].min().date(), "->", df["date"].max().date())
print("Winner values:", df["Winner"].value_counts(dropna=False).to_dict())
print("draws:", (df["Winner"]=="Draw").sum())
print("odds NaN R/B:", df["R_odds"].isna().sum(), df["B_odds"].isna().sum())
# odds coverage by year
df["year"] = df["date"].dt.year
cov = df.groupby("year").agg(n=("R_odds","size"), odds_ok=("R_odds", lambda s: s.notna().sum()))
print(cov.to_string())
# duplicates
dup = df.duplicated(subset=["R_fighter","B_fighter","date"]).sum()
print("dup (R,B,date):", dup)
# implied prob vs win rate sanity (normalize vig out)
def am_to_dec(o):
    return np.where(o < 0, 1 + 100/np.abs(o), 1 + o/100)
m = df.dropna(subset=["R_odds","B_odds"]).copy()
m = m[m["Winner"]!="Draw"]
m["rd"] = am_to_dec(m["R_odds"]); m["bd"] = am_to_dec(m["B_odds"])
m["piR"] = 1/m["rd"]; m["piB"] = 1/m["bd"]
m["piR_n"] = m["piR"]/(m["piR"]+m["piB"])
m["R_win"] = (m["Winner"]=="Red").astype(int)
# binned calibration of market
m["bin"] = pd.cut(m["piR_n"], np.arange(0,1.05,0.05))
print("\nmarket implied prob (de-vigged) bins -> R win rate:")
print(m.groupby("bin", observed=True).agg(n=("R_win","size"), winrate=("R_win","mean"), piR=("piR_n","mean")).to_string())
print("\noverall R win rate:", m["R_win"].mean(), "| n:", len(m))
# R_odds vs B_odds: is R always one corner? check consistency: R_odds < B_odds -> R favorite rate
print("R favorite (lower odds) rate:", (m["R_odds"]<m["B_odds"]).mean())
# draws over time
print("\ndraws by year:", df[df["Winner"]=="Draw"].groupby("year").size().to_dict())
# gender / weight class
print("\nweight classes:", df["weight_class"].value_counts(dropna=False).head(15).to_dict())
print("gender:", df["gender"].value_counts(dropna=False).to_dict())
