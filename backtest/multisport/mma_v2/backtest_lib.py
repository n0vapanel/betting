"""Shared helpers for the UFC backtest: odds conversion, betting evaluation, bootstrap."""
import numpy as np
import pandas as pd

def am_to_dec(o):
    o = np.asarray(o, dtype=float)
    return np.where(o < 0, 1 + 100 / np.abs(o), 1 + o / 100)

def fair_probs(decR, decB):
    """De-vigged fair probabilities from decimal odds."""
    piR = 1.0 / decR
    piB = 1.0 / decB
    tot = piR + piB
    return piR / tot, piB / tot

def evaluate_bets(df, pR, eps=0.0):
    """
    df: test rows with columns decR, decB, yR (R win = 1), pR (model prob R wins).
    Bets $1 flat on the side whose edge (p_model - p_fair) exceeds eps.
    Returns dict of metrics.
    """
    pB = 1.0 - pR
    fairR = df["fairR"].values
    fairB = df["fairB"].values
    decR = df["decR"].values
    decB = df["decB"].values
    yR = df["yR"].values

    edge_R = pR - fairR
    edge_B = pB - fairB
    # choose side: largest edge, must exceed eps
    side = np.where(edge_R >= edge_B, 0, 1)  # 0 = bet R, 1 = bet B
    max_edge = np.maximum(edge_R, edge_B)
    bet = max_edge > eps
    side = side[bet]
    yR_b = yR[bet]
    p_b = np.where(side == 0, pR[bet], pB[bet])
    dec_b = np.where(side == 0, decR[bet], decB[bet])

    if len(yR_b) == 0:
        return {"eps": eps, "n": 0, "roi": np.nan, "win_rate": np.nan, "profit": 0.0, "picks": side,
                "bet_mask": bet, "edges": max_edge}

    win = (side == 0) == (yR_b == 1)
    profit_per_bet = np.where(win, dec_b - 1.0, -1.0)
    profit = profit_per_bet.sum()
    roi = profit / len(yR_b)
    return {"eps": eps, "n": int(len(yR_b)), "roi": float(roi),
            "win_rate": float(win.mean()), "profit": float(profit),
            "picks": side, "bet_mask": bet, "edges": max_edge}

def kelly_path(df, pR, eps, bank_start=1000.0, frac=0.25, f_max=1.0):
    """Quarter-Kelly staking on the eps gate, sequential over test rows. Returns final bank, roi, path."""
    pB = 1.0 - pR
    fairR = df["fairR"].values
    fairB = df["fairB"].values
    decR = df["decR"].values
    decB = df["decB"].values
    yR = df["yR"].values

    edge_R = pR - fairR
    edge_B = pB - fairB
    side = np.where(edge_R >= edge_B, 0, 1)
    max_edge = np.maximum(edge_R, edge_B)

    bank = bank_start
    path = [bank]
    n_bets = 0
    total_staked = 0.0
    for i in range(len(df)):
        if max_edge[i] <= eps:
            path.append(bank)
            continue
        p = pR[i] if side[i] == 0 else pB[i]
        dec = decR[i] if side[i] == 0 else decB[i]
        if dec <= 1.001 or np.isnan(dec):
            path.append(bank)
            continue
        f = (p * dec - 1.0) / (dec - 1.0)
        f = np.clip(f, 0.0, f_max)
        stake = frac * f * bank
        if stake <= 0:
            path.append(bank)
            continue
        n_bets += 1
        total_staked += stake
        win = (side[i] == 0) == (yR[i] == 1)
        bank += stake * (dec - 1.0) if win else -stake
        path.append(bank)
    roi = (bank - bank_start) / total_staked if total_staked > 0 else np.nan
    return {"final_bank": float(bank), "roi": float(roi) if not np.isnan(roi) else None,
            "n_bets": n_bets, "total_staked": float(total_staked),
            "path": np.array(path), "start": bank_start}

def bootstrap_ci(metric_fn, df, pR, eps, n_iter=2000, seed=0, rng=None):
    rng = rng or np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(df))
    for _ in range(n_iter):
        b = rng.choice(idx, size=len(idx), replace=True)
        m = metric_fn(df.iloc[b], pR[b], eps)
        if np.isnan(m):
            continue
        vals.append(m)
    vals = np.array(vals)
    return {"mean": float(vals.mean()), "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)), "n_valid": int(len(vals))}

def brier_score(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))
