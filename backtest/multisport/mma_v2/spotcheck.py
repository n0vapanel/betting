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
fights["sig_landed"] = fights["sig_landed"].astype(str).str.split(" of ").str[0]
fights["sig_landed"] = pd.to_numeric(fights["sig_landed"], errors="coerce")
fights = fights.sort_values(["fighter", "date"])
# cumulative mean of PRIOR fights per fighter
g = fights.groupby("fighter")["sig_landed"]
fights["prev_avg_sig"] = (g.cumsum() - fights["sig_landed"]) / g.cumcount()
fights.loc[fights["prev_avg_sig"] < 0, "prev_avg_sig"] = np.nan  # safety

# merge mdabbert (date<=2021-03-20) with fights on (date, fighter)
md_sub = md[md["date"] <= pd.Timestamp("2021-03-20")].copy()
md_sub["date_n"] = md_sub["date"].dt.normalize()
fights["date_n"] = fights["date"].dt.normalize()

out = []
for corner in ["R", "B"]:
    t = md_sub[["date_n", f"{corner}_fighter", f"{corner}_avg_SIG_STR_landed"]].rename(
        columns={f"{corner}_fighter": "fighter", f"{corner}_avg_SIG_STR_landed": "md_avg"})
    t = t.merge(fights[["date_n", "fighter", "prev_avg_sig"]], on=["date_n", "fighter"], how="inner")
    out.append(t)
chk = pd.concat(out, ignore_index=True)
chk = chk.dropna(subset=["md_avg", "prev_avg_sig"])
print("matched rows:", len(chk))
if len(chk) > 50:
    print("corr (md author-avg vs rajeevw pre-fight cumulative avg):", round(chk["md_avg"].corr(chk["prev_avg_sig"]), 4))
    print("mae:", round((chk["md_avg"] - chk["prev_avg_sig"]).abs().mean(), 3))
    print("mean md:", round(chk["md_avg"].mean(), 2), "mean raj:", round(chk["prev_avg_sig"].mean(), 2))
    print(chk.head(10).to_string())
