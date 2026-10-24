import pandas as pd
import numpy as np
import os, json
base = os.path.dirname(os.path.abspath(__file__))
out = pd.read_csv(os.path.join(base, "mma_predictions.csv"))
print("csv shape:", out.shape)
print(out.head(4).to_string())
# independent recompute of flat-$1 ROI at eps=0 from the CSV itself
def am_to_dec(o):
    o = np.asarray(o, dtype=float)
    return np.where(o < 0, 1 + 100 / np.abs(o), 1 + o / 100)
decR = am_to_dec(out["R_closing_odds_american"]); decB = am_to_dec(out["B_closing_odds_american"])
yR = (out["winner"] == "Red").astype(int).values
sideR = (out["bet_side_eps0"] == "R").values
dec = np.where(sideR, decR, decB)
win = (sideR == (yR == 1))
profit = np.where(win, dec - 1.0, -1.0)
print("recomputed flat ROI eps0:", round(profit.sum() / len(out), 4), "| profit:", round(profit.sum(), 1), "| n:", len(out))
# bets at best gate (0.05): side in bet_side_best_gate non-empty
bg = out[out["bet_side_best_gate"] != ""]
sideR_b = (bg["bet_side_best_gate"] == "R").values
dec_b = np.where(sideR_b, am_to_dec(bg["R_closing_odds_american"]), am_to_dec(bg["B_closing_odds_american"]))
y_b = (bg["winner"] == "Red").astype(int).values
win_b = (sideR_b == (y_b == 1))
profit_b = np.where(win_b, dec_b - 1.0, -1.0)
print("recomputed flat ROI best gate:", round(profit_b.sum() / len(bg), 4), "| n bets:", len(bg))
m = json.load(open(os.path.join(base, "metrics.json")))
print("\nmetrics.json keys:", list(m.keys()))
print("primary acc:", m["primary"]["acc"], "\nprimary gates:", {k: {kk: v[kk] for kk in ('n','roi')} for k, v in m["primary"]["gates"].items()})
print("\nreport exists:", os.path.exists(os.path.join(base, "MMA_BACKTEST_REPORT.md")),
      "| extended csv:", os.path.exists(os.path.join(base, "mma_predictions_extended.csv")))
