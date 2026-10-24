import numpy as np
import pandas as pd

pred = pd.read_csv('nhl_predictions.csv')
win = ((pred.model_pick == 'H') & (pred.home_win == 1)) | ((pred.model_pick == 'A') & (pred.home_win == 0))
for hold in [0.0238, 0.0476, 0.0650]:
    o_h = 1 / (pred.p_mkt_home * (1 + hold))
    o_a = 1 / (pred.p_mkt_away * (1 + hold))
    od = np.where(pred.model_pick == 'H', o_h, o_a)
    r = np.where(win, od - 1.0, -1.0)
    print(f"hold {hold}: flat ROI {r.mean():+.4f}")
