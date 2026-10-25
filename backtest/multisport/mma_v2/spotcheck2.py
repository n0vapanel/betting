import pandas as pd
import numpy as np
import os
base = os.path.dirname(os.path.abspath(__file__))

raw = pd.read_csv(os.path.join(base, "raw", "raw_total_fight_data.csv"), sep=";")
raw.columns = [c.strip() for c in raw.columns]
raw["date"] = pd.to_datetime(raw["date"])
md = pd.read_csv(os.path.join(base, "raw_mdabbert", "ufc-master.csv"))
md["date"] = pd.to_datetime(md["date"])

def parse_t(t):
    s = str(t)
    if ":" not in s:
        return np.nan
    m, sec = s.split(":")
    return int(m) + int(sec) / 60

frames = []
for corner in ["R", "B"]:
    sub = raw[["date", f"{corner}_fighter", f"{corner}_SIG_STR.", "last_round", "last_round_time"]].rename(
        columns={f"{corner}_fighter": "fighter", f"{corner}_SIG_STR.": "sig_str"})
    frames.append(sub)
fights = pd.concat(frames, ignore_index=True)
fights = fights.dropna(subset=["fighter"])
fights["fighter"] = fights["fighter"].str.strip()
fights["sig_landed"] = pd.to_numeric(fights["sig_str"].astype(str).str.split(" of ").str[0], errors="coerce")
fights["lround"] = pd.to_numeric(fights["last_round"], errors="coerce")
fights["ltime"] = fights["last_round_time"].map(parse_t)
# fight minutes: rounds are 5 min (some 3-round fights early were 5 too; title 5)
fights["minutes"] = (fights["lround"] - 1) * 5 + fights["ltime"]
fights["slpm"] = fights["sig_landed"] / fights["minutes"]
fights = fights.sort_values(["fighter", "date"])
g = fights.groupby("fighter")
fights["prev_slpm_avg"] = (g["slpm"].cumsum() - fights["slpm"]) / g.cumcount()
fights["prev_slpm_wtd"] = (g["sig_landed"].cumsum() - fights["sig_landed"]) / (g["minutes"].cumsum() - fights["minutes"])

md_sub = md[md["date"] <= pd.Timestamp("2021-03-20")].copy()
md_sub["date_n"] = md_sub["date"].dt.normalize()
fights["date_n"] = fights["date"].dt.normalize()
out = []
for corner in ["R", "B"]:
    t = md_sub[["date_n", f"{corner}_fighter", f"{corner}_avg_SIG_STR_landed"]].rename(
        columns={f"{corner}_fighter": "fighter", f"{corner}_avg_SIG_STR_landed": "md_avg"})
    t = t.merge(fights[["date_n", "fighter", "prev_slpm_avg", "prev_slpm_wtd"]], on=["date_n", "fighter"], how="inner")
    out.append(t)
chk = pd.concat(out, ignore_index=True).dropna()
print("matched:", len(chk))
print("corr md vs avg-of-per-fight-SLpM:", round(chk["md_avg"].corr(chk["prev_slpm_avg"]), 4))
print("corr md vs weighted(total-landed/total-min):", round(chk["md_avg"].corr(chk["prev_slpm_wtd"]), 4))
print("mae vs avg:", round((chk["md_avg"] - chk["prev_slpm_avg"]).abs().mean(), 3),
      "| mae vs wtd:", round((chk["md_avg"] - chk["prev_slpm_wtd"]).abs().mean(), 3))
print(chk[["md_avg", "prev_slpm_avg", "prev_slpm_wtd"]].describe().round(2).to_string())
print(chk.head(8).to_string())
