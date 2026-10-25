import pandas as pd
import os
base = os.path.dirname(os.path.abspath(__file__))
md = pd.read_csv(os.path.join(base, "raw_mdabbert", "ufc-master.csv"))
md["date"] = pd.to_datetime(md["date"])
md["year"] = md["date"].dt.year
cols = ["B_avg_SIG_STR_landed","B_avg_SIG_STR_pct","B_avg_TD_landed","B_avg_TD_pct","B_avg_SUB_ATT",
        "R_avg_SIG_STR_landed","R_avg_SIG_STR_pct","R_avg_TD_landed","R_avg_TD_pct","R_avg_SUB_ATT",
        "B_current_win_streak","B_wins","B_losses","R_age","B_age","R_Height_cms","R_Reach_cms"]
cov = md.groupby("year")[cols].apply(lambda s: s.notna().mean().round(2))
print(cov.to_string())
