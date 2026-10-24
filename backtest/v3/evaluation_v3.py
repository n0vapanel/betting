from __future__ import annotations

import numpy as np
import pandas as pd


EPS_GRID = [0.0, 0.01, 0.02, 0.03, 0.05]
N_BOOT = 1000
SEED = 202203


def bet_frame(frame, probability, variant, eps=0.0):
    p = np.asarray(probability, dtype=float)
    if len(p) != len(frame) or not np.isfinite(p).all():
        raise ValueError("One finite probability is required per game")
    if ((p < 0) | (p > 1)).any() or eps < 0:
        raise ValueError("Invalid probabilities or epsilon")
    market = frame.p_market_home.to_numpy(dtype=float)
    difference = p - market
    predicted_home = p > 0.5
    if variant == "flat":
        home = predicted_home
        active = np.ones(len(frame), dtype=bool)
    elif variant == "lean":
        home = difference > 0
        active = (np.abs(difference) >= eps) & (difference != 0)
    elif variant == "winner":
        home = predicted_home
        edge = np.where(home, difference, -difference)
        active = (edge >= eps) & (edge > 0)
    else:
        raise ValueError(f"Unknown betting gate {variant}")
    decimal = np.where(home, frame.dec_home.to_numpy(), frame.dec_away.to_numpy())
    won = home == frame.home_won.to_numpy(dtype=bool)
    return pd.DataFrame({
        "date": frame.date.to_numpy(), "bet": active, "side": np.where(home, "home", "away"),
        "p_side": np.where(home, p, 1 - p), "decimal": decimal, "won": won,
        "edge": np.where(home, difference, -difference),
        "pnl": np.where(active, np.where(won, decimal - 1, -1), 0.0),
    })


def bet_metrics(bets):
    active = bets.loc[bets.bet]
    n = len(active)
    return {
        "n_bets": n, "win_rate": float(active.won.mean()) if n else None,
        "roi": float(active.pnl.sum() / n) if n else None, "pnl": float(active.pnl.sum()),
    }


def evaluate(frame, probability):
    p = np.asarray(probability, dtype=float)
    y = frame.home_won.to_numpy(dtype=float)
    flat = bet_metrics(bet_frame(frame, p, "flat"))
    return {
        "n_games": len(frame), "accuracy": flat["win_rate"], "flat_roi": flat["roi"], "flat_pnl": flat["pnl"],
        "brier": float(np.mean((p - y) ** 2)),
        "market_brier": float(np.mean((frame.p_market_home.to_numpy(dtype=float) - y) ** 2)),
        "log_loss": float(np.mean(-y * np.log(np.clip(p, 1e-6, 1 - 1e-6))
                                  - (1 - y) * np.log1p(-np.clip(p, 1e-6, 1 - 1e-6)))),
    }


def gate_table(frame, probability):
    return [
        {"variant": variant, "eps": eps, **bet_metrics(bet_frame(frame, probability, variant, eps))}
        for variant in ["lean", "winner"] for eps in EPS_GRID
    ]


def choose_gate(validation):
    candidates = gate_table(validation, validation.p_model_home.to_numpy())
    eligible = [c for c in candidates if c["n_bets"] >= 30]
    if eligible:
        best = min(eligible, key=lambda c: (-c["roi"], -c["n_bets"], c["eps"], c["variant"]))
        return {**best, "selection_period": "2022-02-15..2022-02-28", "minimum_bets": 30,
                "fallback": False}, candidates
    fallback = next(c for c in candidates if c["variant"] == "winner" and c["eps"] == 0)
    return {**fallback, "selection_period": "2022-02-15..2022-02-28", "minimum_bets": 30,
            "fallback": True}, candidates


def hindsight_best(gates):
    eligible = [g for g in gates if g["n_bets"] and g["roi"] is not None]
    return min(eligible, key=lambda c: (-c["roi"], -c["n_bets"], c["eps"], c["variant"]))


def quarter_kelly(bets):
    active = bets.loc[bets.bet].copy()
    bankroll, total_staked, n_staked = 1.0, 0.0, 0
    peak, max_drawdown = 1.0, 0.0
    for date, group in active.groupby("date", sort=True):
        decimal = group.decimal.to_numpy(dtype=float)
        p = group.p_side.to_numpy(dtype=float)
        fraction = np.clip(0.25 * (p * decimal - 1) / (decimal - 1), 0, 0.05)
        if fraction.sum() > 1:
            fraction /= fraction.sum()
        stakes = bankroll * fraction
        returns = np.where(group.won.to_numpy(dtype=bool), decimal - 1, -1)
        total_staked += float(stakes.sum())
        n_staked += int((stakes > 0).sum())
        bankroll += float(stakes @ returns)
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, 1 - bankroll / peak)
    return {
        "bankroll_multiple": bankroll, "net_return": bankroll - 1, "total_staked": total_staked,
        "roi_on_staked": (bankroll - 1) / total_staked if total_staked else None,
        "n_staked": n_staked, "n_gate_bets": len(active), "max_daily_drawdown": max_drawdown,
        "policy": "Quarter Kelly, 0–5% per bet; simultaneous games use start-of-Eastern-date bankroll; no within-day recycling.",
    }


def bootstrap_intervals(frame, probability, variant, eps, n_boot=N_BOOT):
    p = np.asarray(probability, dtype=float)
    flat = bet_frame(frame, p, "flat")
    gated = bet_frame(frame, p, variant, eps)
    y = frame.home_won.to_numpy(dtype=float)
    work = pd.DataFrame({
        "date": frame.date.to_numpy(), "n": 1, "correct": flat.won.to_numpy(dtype=float),
        "flat_pnl": flat.pnl.to_numpy(), "brier": (p - y) ** 2,
        "market_brier": (frame.p_market_home.to_numpy(dtype=float) - y) ** 2,
        "gate_n": gated.bet.to_numpy(dtype=int), "gate_pnl": gated.pnl.to_numpy(),
        "gate_wins": (gated.bet & gated.won).to_numpy(dtype=int),
    })
    daily = work.groupby("date", sort=True).sum()
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(daily), size=(n_boot, len(daily)))
    sums = daily.to_numpy(dtype=float)[sampled].sum(axis=1)
    index = {c: i for i, c in enumerate(daily.columns)}
    n, gate_n = sums[:, index["n"]], sums[:, index["gate_n"]]
    gate_roi = np.divide(sums[:, index["gate_pnl"]], gate_n, out=np.full(n_boot, np.nan), where=gate_n > 0)
    gate_win = np.divide(sums[:, index["gate_wins"]], gate_n, out=np.full(n_boot, np.nan), where=gate_n > 0)
    values = {
        "accuracy": sums[:, index["correct"]] / n, "flat_roi": sums[:, index["flat_pnl"]] / n,
        "brier": sums[:, index["brier"]] / n, "market_brier": sums[:, index["market_brier"]] / n,
        "brier_minus_market": (sums[:, index["brier"]] - sums[:, index["market_brier"]]) / n,
        "gate_roi": gate_roi, "gate_win_rate": gate_win,
    }
    result = {}
    for key, samples in values.items():
        finite = samples[np.isfinite(samples)]
        result[key] = np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else [None, None]
    result["n_resamples"] = n_boot
    result["n_dates"] = len(daily)
    result["method"] = "Percentile bootstrap of whole Eastern calendar dates; fixed fitted model and fixed gate."
    return result
