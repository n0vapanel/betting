import numpy as np, pandas as pd

df = pd.read_csv("/Users/x/Desktop/x/Coding Projects/betting bot/backtest/multisport/soccer/soccer_predictions.csv")
n = len(df)
print("rows:", n)

# ---- 1X2 verification ----
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
print(f"acc 1X2 ensemble: {acc:.4f}")

for eps in (0.0, 0.02, 0.05):
    bet = p_side - m_side > eps
    profit = np.where(bet, np.where(win, o_side - 1.0, -1.0), 0.0)
    nb = bet.sum()
    roi = profit.sum() / nb if nb else float("nan")
    print(f"eps={eps}: bets={nb} winrate={win[bet].mean()*100:.1f}% roi={roi*100:+.2f}%")

# Kelly recomputation on best gate (eps=0.02)
eps = 0.02
bet = p_side - m_side > eps
p, o, w = p_side[bet], o_side[bet], win[bet]
f_full = np.clip((p * o - 1.0) / np.maximum(o - 1.0, 1e-9), 0.0, None)
stake = np.minimum(f_full * 0.25, 0.05)
bank = 100.0
for s, wi, od in zip(stake, w, o):
    bank *= (1 - s) + s * (1 + (od - 1) * wi)
print(f"\nKELLY 1X2 eps=0.02: n={len(stake)} wins={w.sum()} mean_stake={stake.mean()*100:.2f}% "
      f"stake_cap_hits={(stake==0.05).sum()} bank=100->{bank:.2f} growth={bank/100-1:+.1%}")

# ---- O/U verification ----
P2 = df["p_ens_Over"].values
M2 = df["m_Over"].values
O2 = df[["Avg>2.5", "Avg<2.5"]].values
y2 = df["Over25"].values
side2 = (P2 > 0.5).astype(int)
p_s2 = np.where(side2 == 1, P2, 1 - P2)
m_s2 = np.where(side2 == 1, M2, 1 - M2)
o_s2 = np.where(side2 == 1, O2[:, 0], O2[:, 1])
win2 = side2 == y2
print(f"\nacc OU ensemble: {(side2==y2).mean():.4f}")
for eps in (0.0, 0.02, 0.05):
    bet = p_s2 - m_s2 > eps
    profit = np.where(bet, np.where(win2, o_s2 - 1.0, -1.0), 0.0)
    nb = bet.sum()
    roi = profit.sum() / nb if nb else float("nan")
    print(f"eps={eps}: bets={nb} winrate={win2[bet].mean()*100:.1f}% roi={roi*100:+.2f}%")

eps = 0.0
bet = p_s2 - m_s2 > eps
p, o, w = p_s2[bet], o_s2[bet], win2[bet]
f_full = np.clip((p * o - 1.0) / np.maximum(o - 1.0, 1e-9), 0.0, None)
stake = np.minimum(f_full * 0.25, 0.05)
bank = 100.0
for s, wi, od in zip(stake, w, o):
    bank *= (1 - s) + s * (1 + (od - 1) * wi)
print(f"KELLY OU eps=0: n={len(stake)} wins={w.sum()} mean_stake={stake.mean()*100:.2f}% "
      f"stake_cap_hits={(stake==0.05).sum()} bank=100->{bank:.2f} growth={bank/100-1:+.1%}")

# ---- market baseline (always bet favorite) ----
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

# ---- stat-level naive check: total goals model MAE vs global mean ----
print(f"\ntot goals act mean {df['TotGoals'].mean():.3f}, model MAE {np.abs((df['exp_goals_home']+df['exp_goals_away'])-df['TotGoals']).mean():.3f}")
