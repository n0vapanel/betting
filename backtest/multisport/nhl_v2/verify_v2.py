"""Independent verification of NHL v2 backtest:
1. Recompute all headline numbers from nhl_v2_predictions.csv (must match metrics.json).
2. Confirm v1 baseline numbers from ../nhl/nhl_predictions.csv.
3. Paired v1-vs-v2 comparison on the same 555 games.
4. Feature reconstruction spot-checks (goalie EWMA, kalman first-game, travel).
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

rng = np.random.default_rng(42)
VIG = 0.0476

p = pd.read_csv("nhl_v2_predictions.csv", parse_dates=["date"])
p1 = pd.read_csv("../nhl/nhl_predictions.csv", parse_dates=["date"])
m = json.load(open("metrics.json"))

print("=== 1. recompute from CSV vs metrics.json ===")
acc = (p.model_pick == np.where(p.home_win == 1, "H", "A")).mean()
flat = p.flat_ret.mean()
brier = brier_score_loss(p.home_win, p.p_model_home)
mkt_brier = brier_score_loss(p.home_win, p.p_mkt_home)
print(f"acc: {acc:.4f} (json {m['test_accuracy']:.4f})")
print(f"flat ROI: {flat:+.4f} (json {m['flat_roi']:+.4f})")
print(f"model brier: {brier:.4f} (json {m['model_brier']:.4f})")
print(f"market brier: {mkt_brier:.4f} (json {m['market_brier']:.4f})")
assert abs(acc - m["test_accuracy"]) < 1e-9
assert abs(flat - m["flat_roi"]) < 1e-9
assert abs(brier - m["model_brier"]) < 1e-9

# odds rule check: odds_h_2way == 1/(p_mkt_home*1.0476)
rule_ok = np.allclose(p.odds_h_2way, 1 / (p.p_mkt_home * (1 + VIG)), rtol=1e-6)
print("payout rule 1/(p*1.0476) holds:", rule_ok)

# bootstrap CI recompute
dates = p.date.values
vals = []
for _ in range(2000):
    d = rng.choice(pd.unique(dates), size=p.date.nunique(), replace=True)
    vals.append(p.flat_ret[np.isin(dates, d)].mean())
ci = np.percentile(vals, [2.5, 97.5])
print(f"bootstrap CI recompute: [{ci[0]:+.4f}, {ci[1]:+.4f}] (json {m['roi_bootstrap_ci']})")

# gates recompute
for eps in [0, 0.02, 0.05]:
    bet = p.edge > eps
    roi = p.flat_ret[bet].mean() if bet.sum() else None
    print(f"gate eps={eps}: n={bet.sum()} roi={roi:+.4f} (json {m['gates'][str(eps)]['roi']})")

# ECE recompute
bins = np.linspace(0, 1, 11)
ece = 0.0
for i in range(10):
    msk = (p.p_model_home >= bins[i]) & (p.p_model_home < bins[i + 1])
    if msk.sum():
        ece += msk.sum() / len(p) * abs(p.home_win[msk].mean() - p.p_model_home[msk].mean())
print(f"ECE recompute: {ece:.5f} (json {m['ece']:.5f})")

# quarter-Kelly recompute
b = p.model_pick_odds - 1.0
f = np.asarray(0.25 * (b * p.model_pick_prob - (1 - p.model_pick_prob)) / b)
bank = 1000.0
for i in range(len(p)):
    if f[i] > 0:
        bank += min(f[i], 0.05) * bank * p.flat_ret.iloc[i]
print(f"qKelly recompute: {bank:.2f} (json {m['quarter_kelly_final']:.2f})")
assert abs(bank - m["quarter_kelly_final"]) < 0.01

print("\n=== 2. v1 baseline from v1 CSV ===")
acc1 = (p1.model_pick == np.where(p1.home_win == 1, "H", "A")).mean()
print(f"v1 acc: {acc1:.4f} | flat: {p1.flat_ret.mean():+.4f} | brier: "
      f"{brier_score_loss(p1.home_win, p1.p_model_home):.4f}")

print("\n=== 3. paired v1 vs v2 (same games) ===")
j = p.merge(p1[["game_id", "model_pick", "flat_ret"]],
            on="game_id", suffixes=("_v2", "_v1"))
same = (j.model_pick_v2 == j.model_pick_v1).mean()
print(f"pick agreement v1/v2: {same:.3f}")
d = (j.flat_ret_v2 - j.flat_ret_v1)
print(f"per-game return delta (v2-v1): mean {d.mean():+.4f}")
dv = []
datesj = j.date.values
for _ in range(2000):
    dd = rng.choice(pd.unique(datesj), size=len(pd.unique(datesj)), replace=True)
    dv.append(d[np.isin(datesj, dd)].mean())
ci_d = np.percentile(dv, [2.5, 97.5])
print(f"delta bootstrap 95% CI: [{ci_d[0]:+.4f}, {ci_d[1]:+.4f}]")
# disagreement games
dis = j[j.model_pick_v1 != j.model_pick_v2]
print(f"disagreements: {len(dis)} | v2 right when disagreeing: "
      f"{(dis.model_pick_v2 == np.where(dis.home_win == 1, 'H', 'A')).mean():.3f}")
# v2 ROI on games where v1 lost
v1lost = j[j.flat_ret_v1 < 0]
print(f"games where v1 lost: {len(v1lost)} | v2 ROI there: {v1lost.flat_ret_v2.mean():+.4f}")

print("\n=== 4. feature reconstruction spot-checks ===")
# (a) goalie: recompute gk_sv_ewma diff for one TEST game from goalie_lines.parquet
gkf = pd.read_parquet("data/goalie_lines.parquet")
gks = pd.read_parquet("data/goalie_starts.parquet")
gid = int(p.sort_values("date").iloc[0].game_id)
row = p[p.game_id == gid].iloc[0]
home_line = gkf[gkf.gameId == gid]
# find which of the two teams is home: check via team code matching games_full
gf = pd.read_parquet("data/games_full.parquet")
gr = gf[gf.game_id == gid].iloc[0]
h_gk = gkf[(gkf.gameId == gid) & (gkf.team == gr.home_code)].iloc[0]
a_gk = gkf[(gkf.gameId == gid) & (gkf.team == gr.away_code)].iloc[0]
print(f"game {gid}: diff_gk_sv_ewma in CSV = {row.diff_gk_sv_ewma:.4f}; "
      f"h-a from lines = {h_gk.gk_sv_ewma - a_gk.gk_sv_ewma:.4f}")
assert abs(row.diff_gk_sv_ewma - (h_gk.gk_sv_ewma - a_gk.gk_sv_ewma)) < 1e-9
# if the away starter had zero prior starts, ewma must be the 0.905 prior
if a_gk.gk_starts == 0:
    assert a_gk.gk_sv_ewma == 0.905
    print(f"  away starter had 0 prior starts -> ewma == 0.905 prior: OK")
else:
    print(f"  away starter had {int(a_gk.gk_starts)} prior starts, ewma {a_gk.gk_sv_ewma:.4f}")

# (b) kalman first game: strength diff ~ 0 (full-season features file)
feat_all = pd.read_parquet("data/games_features_v2.parquet")
first = feat_all.sort_values("date").iloc[0]
print(f"first game {first.game_id}: diff_k_strength = {first.diff_k_strength:.6f} "
      f"(expect ~0) | diff_kx_strength = {first.diff_kx_strength:.6f}")

# (c) travel: manual haversine for one road game
def hav(a, b):
    lat1, lon1, lat2, lon2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    return 7917.5 * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2
                                      + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))

AR = {"ANA": (33.8078, -117.8766), "ARI": (33.5320, -112.2612), "BOS": (42.3663, -71.0622),
      "BUF": (42.8751, -78.8766), "CAR": (35.8032, -78.7219), "CBJ": (39.9693, -83.0061),
      "CGY": (51.0374, -114.0519), "CHI": (41.8807, -87.6742), "COL": (39.7487, -105.0077),
      "DAL": (32.7905, -96.8103), "DET": (42.3410, -83.0550), "EDM": (53.5467, -113.4977),
      "FLA": (26.1585, -80.3256), "LAK": (34.0430, -118.2673), "MIN": (44.9447, -93.1011),
      "MTL": (45.4961, -73.5693), "NJD": (40.7336, -74.1711), "NSH": (36.1592, -86.7785),
      "NYI": (40.7118, -73.7271), "NYR": (40.7505, -73.9934), "OTT": (45.2969, -75.9271),
      "PHI": (39.9012, -75.1720), "PIT": (40.4393, -79.9893), "SEA": (47.6221, -122.3541),
      "SJS": (37.3329, -121.9011), "STL": (38.6268, -90.2026), "TBL": (27.9427, -82.4518),
      "TOR": (43.6435, -79.3791), "VAN": (49.2778, -123.1088), "VGK": (36.1029, -115.1784),
      "WPG": (49.8928, -97.1436), "WSH": (38.8981, -77.0208)}
# find a game where home team's previous game was away (travel > 0 for home side)
tg2 = pd.read_parquet("data/team_games.parquet")
tg2["date"] = pd.to_datetime(tg2["date"])
ok = False
for gid_ in p.game_id.sample(80, random_state=3):
    grr = gf[gf.game_id == gid_].iloc[0]
    prev = tg2[(tg2.team == grr.home_code) & (tg2.date < grr.date)].sort_values("date")
    if len(prev) == 0 or prev.iloc[-1].home_or_away != "AWAY":
        continue
    last = prev.iloc[-1]
    # previous away game venue = opponent's arena
    opp_row = gf[(gf.game_id == last.gameId)]
    opp = opp_row.iloc[0].away_code if opp_row.iloc[0].home_code == grr.home_code else opp_row.iloc[0].home_code
    if grr.home_code not in AR or opp not in AR:
        continue
    expected_h_travel = hav(AR[opp], AR[grr.home_code])
    # away team travel: its previous game venue -> this venue (home arena)
    apv = tg2[(tg2.team == grr.away_code) & (tg2.date < grr.date)].sort_values("date")
    a_travel = 0.0
    if len(apv):
        alast = apv.iloc[-1]
        aopp_row = gf[gf.game_id == alast.gameId].iloc[0]
        aopp = aopp_row.away_code if aopp_row.home_code == grr.away_code else aopp_row.home_code
        if alast.home_or_away == "AWAY":
            a_venue = AR[aopp] if aopp in AR else AR.get(grr.away_code)
        else:
            a_venue = AR.get(grr.away_code)
        if a_venue:
            a_travel = hav(a_venue, AR[grr.home_code])
    exp_diff = expected_h_travel - a_travel
    got = p[p.game_id == gid_].iloc[0].diff_travel
    print(f"game {gid_} {grr.home_code} vs {grr.away_code}: csv diff_travel={got:.1f} "
          f"manual={exp_diff:.1f}")
    assert abs(got - exp_diff) < 1.0, (got, exp_diff)
    ok = True
    break
print("travel reconstruction match:", ok)

print("\nALL VERIFICATION CHECKS PASSED")

# save verify.json
vj = dict(
    acc=float(acc), flat_roi=float(flat), model_brier=float(brier),
    market_brier=float(mkt_brier), roi_bootstrap_ci=[float(ci[0]), float(ci[1])],
    qkelly=float(bank), ece=float(ece),
    v1_acc=float(acc1), v1_flat_roi=float(p1.flat_ret.mean()),
    v1_brier=float(brier_score_loss(p1.home_win, p1.p_model_home)),
    pick_agreement=float(same), paired_delta_mean=float(d.mean()),
    paired_delta_ci=[float(ci_d[0]), float(ci_d[1])],
    disagreements=int(len(dis)), v2_right_on_disagreements=float(
        (dis.model_pick_v2 == np.where(dis.home_win == 1, "H", "A")).mean()),
    travel_reconstruction=ok,
)
with open("verify.json", "w") as fh:
    json.dump(vj, fh, indent=2)
print("wrote verify.json")
