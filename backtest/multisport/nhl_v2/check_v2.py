"""Sanity checks: v1 feature parity, goalie no-lookahead, kalman/travel sanity."""
import numpy as np
import pandas as pd

D = "data"
g1 = pd.read_parquet(f"../nhl/games_features.parquet")
g2 = pd.read_parquet(f"{D}/games_features_v2.parquet")

v1_feats = [c for c in g1.columns if c.startswith("diff_")]
shared = [c for c in v1_feats if c in g2.columns]
print(f"v1 features: {len(v1_feats)}, shared with v2: {len(shared)}")
m = g1.merge(g2[["game_id"] + shared], on="game_id", suffixes=("_v1", "_v2"))
print("parity check (max abs diff) on shared features:")
diffs = {}
for c in shared:
    d = (m[f"{c}_v1"] - m[f"{c}_v2"]).abs().max()
    diffs[c] = d
bad = {k: v for k, v in diffs.items() if v > 1e-9}
print("n features with any diff:", len(bad))
if bad:
    print({k: round(v, 4) for k, v in list(bad.items())[:10]})

# ---- goalie no-lookahead: reconstruct one goalie's EWMA manually ----
gk = pd.read_parquet(f"{D}/goalie_starts.parquet")
gkf = pd.read_parquet(f"{D}/goalie_lines.parquet")
vg = gk[gk.name == "A. Vasilevskiy"].sort_values("date")
print("\nVasilevskiy starts:", len(vg))
row = gkf[gkf.gameId == vg.iloc[10].gameId].iloc[0]
print("game", row.gameId, "line EWMA:", round(row.gk_sv_ewma, 4))
prev = vg.iloc[:10]
vals = [s / t if t > 0 else 0.905 for s, t in zip(prev.saves, prev.shots)]
ew = 0.905
for v in vals:
    ew = (2 / 11) * v + (1 - 2 / 11) * ew  # recursive span-10 alpha update (matches extract_goalies)
print("manual EWMA from 10 prior starts:", round(ew, 4), "-> match:", abs(ew - row.gk_sv_ewma) < 1e-6)

# ---- kalman sanity ----
for c in ["diff_k_strength", "diff_kx_strength", "diff_elo_before", "diff_elo_fast_before"]:
    print(c, "std:", round(g2[c].std(), 3), "range:", round(g2[c].min(), 2), round(g2[c].max(), 2))
print("corr(k_strength, elo):", round(g2.diff_k_strength.corr(g2.diff_elo_before), 3))
print("corr(kx_strength, elo):", round(g2.diff_kx_strength.corr(g2.diff_elo_before), 3))

# ---- travel sanity ----
print("\ntravel: mean", round(g2.diff_travel.mean(), 1), "max", round(g2.diff_travel.max(), 1))
print("travel7: mean", round(g2.diff_travel7.mean(), 1), "max", round(g2.diff_travel7.max(), 1))
# b2b home games should have travel 0 on the home side
h = g2[["home_code", "away_code"]]
print("sample travel values:", g2.diff_travel.sample(5, random_state=1).round(0).tolist())

# ---- injuries availability by month ----
print("\ninj_season test-window coverage:", (g2[(g2.date >= "2022-03-01") & (g2.diff_inj_season.abs() > 0)].shape[0]),
      "/", (g2.date >= "2022-03-01").sum())

# ---- NaN check ----
feats = [c for c in g2.columns if c.startswith("diff_")]
print("NaNs in any feature:", int(g2[feats].isna().sum().sum()))
print("games:", len(g2))
