"""
Build no-lookahead rolling features for UFC walk-forward backtest.

Sources:
- mdabbert/ultimate-ufc-dataset (ufc-master.csv): fight log 2010-2026, BestFightOdds
  closing American odds (R_odds/B_odds), winner (Red/Blue/Draw/NC), age, reach, height,
  weight, ranks, weight class.
- rajeevw/ufcdata (raw_total_fight_data.csv): per-fight ufcstats numbers 1994-2021,
  used to build strike-rate features (SLpM, strike acc, TD acc, TD defense, sub att...).

All rolling features are computed strictly from fights BEFORE the bout (no lookahead).
Output: fights_features.parquet (one row per fight).

The feature code itself lives in feature_utils.py so ufc_picker.py and
train_model.py run the exact same builders.
"""
import os

from feature_utils import load_mdabbert, parse_rajeevw, build_rolling_features

base = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(base, "fights_features.parquet")


def main():
    md = load_mdabbert(base)
    print("md rows:", len(md))
    long = parse_rajeevw(base)
    print("rajeevw long rows:", len(long))
    merged = build_rolling_features(md, long)
    print("md merged rows:", len(merged))
    sub2021 = merged[(merged["date_n"] >= pd.Timestamp("2020-01-01")) & (merged["date_n"] <= pd.Timestamp("2021-03-20"))]
    print("2020-2021 rows:", len(sub2021), "| feat_slpm_R present:",
          sub2021["feat_slpm_R"].notna().mean().round(3))
    merged.to_parquet(OUT)
    print("saved", OUT, merged.shape)


if __name__ == "__main__":
    import pandas as pd
    main()
