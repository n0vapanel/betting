import pandas as pd
import os
base = os.path.dirname(os.path.abspath(__file__))
raw = pd.read_csv(os.path.join(base, "raw", "raw_total_fight_data.csv"), sep=";")
raw.columns = [c.strip() for c in raw.columns]
print(raw.columns.tolist())
cols = ["R_KD","R_SIG_STR.","R_SIG_STR_pct","R_TOTAL_STR.","R_TD","R_TD_pct","R_SUB_ATT","R_REV","R_CTRL","R_HEAD","R_DISTANCE","win_by","last_round","last_round_time","Format","Fight_type","Winner"]
print(raw[cols].head(6).to_string())
print(raw[cols].iloc[100:103].to_string())
print("\nvalue examples:")
for c in ["R_TD","R_TD_pct","R_SIG_STR_pct","R_CTRL","last_round_time","R_KD","R_SUB_ATT"]:
    print(c, "->", raw[c].dropna().unique()[:8])
