import pandas as pd
import numpy as np
import os
base = os.path.dirname(os.path.abspath(__file__))

raw = pd.read_csv(os.path.join(base, "raw", "raw_total_fight_data.csv"), sep=";")
raw.columns = [c.strip() for c in raw.columns]
raw["date"] = pd.to_datetime(raw["date"])
md = pd.read_csv(os.path.join(base, "raw_mdabbert", "ufc-master.csv"))
md["date"] = pd.to_datetime(md["date"])

frames = []
for corner in ["R", "B"]:
    sub = raw[["date", f"{corner}_fighter", f"{corner}_SIG_STR."]].rename(
        columns={f"{corner}_fighter": "fighter", f"{corner}_SIG_STR.": "sig_landed"})
    frames.append(sub)
fights = pd.concat(frames, ignore_index=True)
fights = fights.dropna(subset=["fighter"])
fights["fighter"] = fights["fighter"].str.strip()
fights["sig_landed"] = pd.to_numeric(fights["sig_landed"], errors="coerce")
fights = fights.sort_values(["fighter", "date"])
g = fights.groupby("fighter")["sig_landed"]
fights["prev_avg_sig"] = (g.cumsum() - fights["sig_landed"]) / g.cumcount()

md_sub = md[md["date"] <= pd.Timestamp("2021-03-20")].copy()
md_sub["date_n"] = md_sub["date"].dt.normalize()
fights["date_n"] = fights["date"].dt.normalize()
print("fights dtypes:", fights["date_n"].dtype, fights["fighter"].dtype)
print("md_sub dtypes:", md_sub["date_n"].dtype, md_sub["R_fighter"].dtype)
print("date sets intersect:", len(set(md_sub["date_n"]) & set(fights["date_n"])))
print("fighter sets intersect:", len(set(md_sub["R_fighter"]) & set(fights["fighter"])))
t = md_sub[["date_n", "R_fighter", "R_avg_SIG_STR_landed"]].rename(columns={"R_fighter": "fighter", "R_avg_SIG_STR_landed": "md_avg"})
m = t.merge(fights[["date_n", "fighter", "prev_avg_sig"]], on=["date_n", "fighter"], how="inner")
print("merge rows:", len(m))
# try only fighter
m2 = t.merge(fights[["date_n", "fighter", "prev_avg_sig"]], on=["fighter"], how="inner")
print("merge on fighter only:", len(m2))
# try only date
m3 = t.merge(fights[["date_n", "fighter", "prev_avg_sig"]], on=["date_n"], how="inner")
print("merge on date only:", len(m3))
# direct comparison of one pair
r = md_sub.iloc[0]
sub = fights[(fights["date_n"] == r["date_n"]) & (fights["fighter"] == r["R_fighter"])]
print("direct pair check:", len(sub), r["date_n"], r["R_fighter"])
print("any fight rows with that date_n:", (fights["date_n"] == r["date_n"]).sum())
print("any fight rows with that fighter:", (fights["fighter"] == r["R_fighter"]).sum())
