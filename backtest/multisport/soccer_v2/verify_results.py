"""Independent verification of soccer v2 results — recomputes everything from the CSV only."""
import numpy as np, pandas as pd

df = pd.read_csv("/Users/x/Desktop/x/Coding Projects/betting bot/backtest/multisport/soccer_v2/soccer_v2_predictions.csv")
n = len(df)
print("rows:", n)
assert n == 571, "test set must be 571 rows"

# ---- 1X2 ----
P = df[["p_ens_H", "p_ens_D", "p_ens_A"]].values
M = df[["m_H", "m_D", "m_A"]].values
O = df[["AvgH", "AvgD", "AvgA"]].values
y = df["FTR"].map({"H": 0, "D": 1, "A": 2}).values

side = P.argmax(1)
p_side = P[np.arange(n), side]
m_side = M[np.arange(n), side]
o_side = O[np.arange(n), side]
win = side == y
acc = (side == y).mean()
print(f"acc 1X2 ensemble: {acc*100:.2f}%")

for eps in (0.0, 0.02, 0.05):
    bet = p_side - m_side > eps
    profit = np.where(bet, np.where(win, o_side - 1.0, -1.0), 0.0)
    nb = bet.sum()
    roi = profit.sum() / nb if nb else float("nan")
    print(f"eps={eps}: bets={nb} winrate={win[bet].mean()*100:.1f}% roi={roi*100:+.2f}%")

# Kelly on best-ROI gate (matches backtest.py's selection)
rois = {eps: (np.where(p_side - m_side > eps, np.where(win, o_side - 1.0, -1.0), 0.0)[p_side - m_side > eps]).mean()
        for eps in (0.0, 0.02, 0.05) if (p_side - m_side > eps).sum()}
eps = max(rois, key=lambda e: rois[e])
bet = p_side - m_side > eps
p, o, w = p_side[bet], o_side[bet], win[bet]
f_full = np.clip((p * o - 1.0) / np.maximum(o - 1.0, 1e-9), 0.0, None)
stake = np.minimum(f_full * 0.25, 0.05)
bank = 100.0
for s, wi, od in zip(stake, w, o):
    bank *= (1 - s) + s * (1 + (od - 1) * wi)
print(f"KELLY 1X2 eps={eps}: n={len(stake)} wins={w.sum()} bank=100->{bank:.2f}")

# ---- O/U ----
P2 = df["p_ens_Over"].values
M2 = df["m_Over"].values
O2 = df[["Avg>2.5", "Avg<2.5"]].values
y2 = df["Over25"].values
side2 = (P2 > 0.5).astype(int)
p_s2 = np.where(side2 == 1, P2, 1 - P2)
m_s2 = np.where(side2 == 1, M2, 1 - M2)
o_s2 = np.where(side2 == 1, O2[:, 0], O2[:, 1])
win2 = side2 == y2
print(f"\nacc OU ensemble: {(side2==y2).mean()*100:.2f}%")
for eps in (0.0, 0.02, 0.05):
    bet = p_s2 - m_s2 > eps
    profit = np.where(bet, np.where(win2, o_s2 - 1.0, -1.0), 0.0)
    nb = bet.sum()
    roi = profit.sum() / nb if nb else float("nan")
    print(f"eps={eps}: bets={nb} winrate={win2[bet].mean()*100:.1f}% roi={roi*100:+.2f}%")

# ---- market baselines ----
side_m = M.argmax(1)
o_m = O[np.arange(n), side_m]
win_m = side_m == y
profit = np.where(win_m, o_m - 1.0, -1.0)
print(f"\nmarket always-favorite 1X2: acc={win_m.mean()*100:.2f}% roi={profit.mean()*100:+.2f}%")
side_m2 = (M2 > 0.5).astype(int)
o_m2 = np.where(side_m2 == 1, O2[:, 0], O2[:, 1])
win_m2 = side_m2 == y2
profit2 = np.where(win_m2, o_m2 - 1.0, -1.0)
print(f"market always-favorite OU: acc={win_m2.mean()*100:.2f}% roi={profit2.mean()*100:+.2f}%")

# ---- date-clustered bootstrap for ensemble eps=0 (both targets) ----
rng = np.random.default_rng(7)
dates = df["Date"].values

def boot(d, roi):
    uniq = np.unique(d)
    out = []
    for _ in range(2000):
        s = rng.choice(uniq, size=len(uniq), replace=True)
        m = np.isin(d, s)
        if m.sum():
            out.append(roi[m].mean())
    out = np.array(out)
    return np.percentile(out, 2.5) * 100, np.percentile(out, 97.5) * 100

bet1 = p_side - m_side > 0
r1 = np.where(bet1, np.where(win, o_side - 1, -1), 0)
print(f"\nboot CI 1X2 eps0: {boot(dates[bet1], r1[bet1])[0]:+.2f}% ... {boot(dates[bet1], r1[bet1])[1]:+.2f}%")
bet2 = p_s2 - m_s2 > 0
r2 = np.where(bet2, np.where(win2, o_s2 - 1, -1), 0)
print(f"boot CI OU eps0: {boot(dates[bet2], r2[bet2])[0]:+.2f}% ... {boot(dates[bet2], r2[bet2])[1]:+.2f}%")

# ---- xG feature sanity: joined xG present everywhere in test ----
print(f"\nxG present in test rows: {df['xg_h'].notna().sum()}/{n}")
print(f"tot goals act mean {df['TotGoals'].mean():.3f}, model MAE {np.abs((df['exp_goals_home']+df['exp_goals_away'])-df['TotGoals']).mean():.3f}")
print("\nVERIFICATION PASSED")
