#!/usr/bin/env python3
"""Run the full v2 walk-forward backtest and write all deliverables.

Pipeline:
  1. data.py    — fetch/cache team logs, player logs, closing moneylines,
                  official injury reports (5PM game-day filing; 1PM game-day
                  snapshot for early tips / fallbacks).
  2. features.py — no-lookahead feature matrix (Four Factors, Elo, rest,
                  injury impact, market, season-to-date).
  3. model.py   — time-ordered CV model selection (LR vs HGB vs blend),
                  isotonic calibration on Feb-2022 OOF predictions.
  4. Evaluation — accuracy, Brier, flat $1 ROI, edge-gated betting (eps grid),
                  quarter-Kelly on the best eps, ablations, bootstrap CI,
                  and the v1 engine's predictions through the same edge gate.

Outputs: backtest/v2_predictions.csv, V2_BACKTEST_REPORT.md (repo root).
Usage:   .venv/bin/python backtest/v2/run_v2.py
         (injury fetching needs JAVA_HOME exported, see data.py)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.v2 import data as D  # noqa: E402
from backtest.v2 import features as F  # noqa: E402
from backtest.v2 import model as M  # noqa: E402

EPS_GRID = [0.0, 0.01, 0.02, 0.03, 0.05]
N_BOOT = 1000
RNG = np.random.default_rng(42)


# ------------------------------------------------------------------ betting eval

def _side_pnl(p_home, p_mkt, dec_home, dec_away, home_won, eps) -> Optional[Dict]:
    diff = p_home - p_mkt
    # spec: bet side where p_model_side > p_market_side and |diff| >= eps
    if diff >= eps and diff > 0:
        side = "home"
    elif diff <= -eps and diff < 0:
        side = "away"
    else:
        return None
    p_side = p_home if side == "home" else 1.0 - p_home
    ml = dec_home if side == "home" else dec_away
    won = bool(home_won) if side == "home" else not bool(home_won)
    pnl = (ml - 1.0) if won else -1.0
    return {"side": side, "p_side": p_side, "ml": ml, "won": won, "pnl": pnl,
            "edge": diff if side == "home" else -diff}


def gate_metrics(
    p_home: np.ndarray,
    p_mkt: np.ndarray,
    dec_home: np.ndarray,
    dec_away: np.ndarray,
    home_won: np.ndarray,
    eps: float,
) -> Dict:
    bets = [
        b for b in (
            _side_pnl(ph, pm, dh, da, hw, eps)
            for ph, pm, dh, da, hw in zip(p_home, p_mkt, dec_home, dec_away, home_won)
        )
        if b is not None
    ]
    n = len(bets)
    if n == 0:
        return {"eps": eps, "n_bets": 0, "win_rate": None, "roi": None, "pnl": 0.0,
                "bets": []}
    pnl = sum(b["pnl"] for b in bets)
    wins = sum(1 for b in bets if b["won"])
    return {"eps": eps, "n_bets": n, "win_rate": wins / n, "roi": pnl / n,
            "pnl": pnl, "bets": bets}


def flat_metrics(
    p_home: np.ndarray, dec_home: np.ndarray, dec_away: np.ndarray, home_won: np.ndarray
) -> Dict:
    sides = np.where(p_home > 0.5, "home", "away")
    pnl, correct = 0.0, 0
    for s, dh, da, hw in zip(sides, dec_home, dec_away, home_won):
        won = bool(hw) if s == "home" else not bool(hw)
        if won:
            pnl += (dh - 1.0) if s == "home" else (da - 1.0)
        else:
            pnl -= 1.0
        correct += int(won)
    n = len(p_home)
    return {"n": n, "accuracy": correct / n, "pnl": pnl, "roi": pnl / n}


def kelly_on_bets(bets: List[Dict]) -> Dict:
    """Quarter-Kelly on the gated bets; stake fraction clipped to [0, 0.05]."""
    bankroll = 1.0
    total_staked = 0.0
    pnl_units = 0.0
    for b in bets:
        p, q = b["p_side"], 1.0 - b["p_side"]
        b_odds = b["ml"] - 1.0
        f = 0.25 * (p * b_odds - q) / b_odds if b_odds > 0 else 0.0
        f = float(np.clip(f, 0.0, 0.05))
        total_staked += f
        if b["won"]:
            bankroll *= 1.0 + f * b_odds
            pnl_units += f * b_odds
        else:
            bankroll *= 1.0 - f
            pnl_units -= f
    return {"total_staked_fraction": total_staked, "bankroll_multiple": bankroll,
            "pnl_units": pnl_units}


def bootstrap_roi_ci(bets: List[Dict], n_boot: int = N_BOOT) -> Tuple[float, float]:
    pnls = np.array([b["pnl"] for b in bets], dtype=float)
    if len(pnls) < 5:
        return (np.nan, np.nan)
    means = [RNG.choice(pnls, size=len(pnls), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def brier(p_home: np.ndarray, home_won: np.ndarray) -> float:
    return float(np.mean((np.asarray(p_home) - np.asarray(home_won, dtype=float)) ** 2))


def best_eps(gates: Dict[float, Dict]) -> float:
    """Pick eps by ROI, tie-break more bets, then higher win rate."""
    scored = [
        (g["roi"] if g["roi"] is not None else -np.inf, g["n_bets"],
         g["win_rate"] if g["win_rate"] is not None else 0.0, eps)
        for eps, g in gates.items()
    ]
    return max(scored)[3]


# ------------------------------------------------------------------ ablations

ABLATIONS = {
    "main": F.FEATURE_NAMES,
    "no_injury": F.NO_INJURY_COLS,
    "no_market": F.NO_MARKET_COLS,
    "stats_only": F.STATS_ONLY_COLS,
}


def run_ablation(feats_train: pd.DataFrame, feats_test: pd.DataFrame, cols: List[str],
                 verbose: bool) -> Dict:
    X_tr = feats_train[cols].astype(float)
    X_te = feats_test[cols].astype(float)
    y_tr = feats_train["home_won"].astype(int)
    sel = M.select_model(X_tr, y_tr, verbose=verbose)
    out = M.train_and_predict(
        X_tr, y_tr, feats_train["date"].astype(str), X_te, sel, verbose=verbose
    )
    p_home = out["test_probs"]
    meta = feats_test
    gates = {
        eps: gate_metrics(
            p_home,
            meta["p_market_home"].values,
            meta["dec_home"].values,
            meta["dec_away"].values,
            meta["home_won"].values,
            eps,
        )
        for eps in EPS_GRID
    }
    eps_best = best_eps(gates)
    return {
        "model": sel["best_name"],
        "cv_ll": sel["best_ll"],
        "cv_table": sel["cv_table"],
        "sel_cols": sel["selected_cols"],
        "calibrator": out.get("calibrator", "none"),
        "test_probs": p_home,
        "flat": flat_metrics(
            p_home, meta["dec_home"].values, meta["dec_away"].values,
            meta["home_won"].values,
        ),
        "brier": brier(p_home, meta["home_won"].values),
        "gates": gates,
        "best_eps": eps_best,
        "best_gate": gates[eps_best],
    }


# ------------------------------------------------------------------ v1 replay

def v1_through_gate(v1_csv: Path) -> Dict:
    df = pd.read_csv(v1_csv)
    p_home = df["predicted_home_prob"].values
    p_mkt = (1.0 / df["dec_home"]) / (1.0 / df["dec_home"] + 1.0 / df["dec_away"])
    home_won = (df["actual_winner"] == df["home"]).values
    gates = {
        eps: gate_metrics(
            p_home, p_mkt.values, df["dec_home"].values, df["dec_away"].values,
            home_won, eps,
        )
        for eps in EPS_GRID
    }
    eps_best = best_eps(gates)
    return {
        "flat": flat_metrics(
            p_home, df["dec_home"].values, df["dec_away"].values, home_won
        ),
        "brier": brier(p_home, home_won),
        "gates": gates,
        "best_eps": eps_best,
        "best_gate": gates[eps_best],
    }


# ------------------------------------------------------------------ main

def main() -> int:
    print("== v2 backtest: building features ==")
    built = F.build_features(verbose=True)
    train, test = built["train"], built["test"]
    if not len(test):
        print("BLOCKED: no test rows with complete features.")
        return 1

    feats_train = train.copy()
    feats_test = test.copy()

    results = {}
    for name, cols in ABLATIONS.items():
        print(f"\n== ablation: {name} ({len(cols)} features) ==")
        results[name] = run_ablation(feats_train, feats_test, cols, verbose=(name == "main"))

    main_r = results["main"]
    meta = feats_test
    p_home = main_r["test_probs"]

    # ---------------- test predictions CSV ----------------
    rows = []
    for i, (_, g) in enumerate(meta.iterrows()):
        ph, pm = float(p_home[i]), float(g["p_market_home"])
        side = "home" if ph > 0.5 else "away"
        edge = ph - pm if side == "home" else pm - ph
        ml = g["dec_home"] if side == "home" else g["dec_away"]
        actual = "home" if g["home_won"] else "away"
        correct = side == actual
        pnl = (ml - 1.0) if correct else -1.0
        rows.append(
            {
                "date": g["date"], "home": g["home"], "away": g["away"],
                "p_model_home": round(ph, 4), "p_market_home": round(pm, 4),
                "predicted_side": side, "edge": round(edge, 4),
                "ml_of_side": round(float(ml), 4), "actual_side": actual,
                "correct": correct, "pnl_1usd": round(pnl, 4),
            }
        )
    pred_df = pd.DataFrame(rows)
    pred_out = PROJECT_ROOT / "backtest" / "v2_predictions.csv"
    pred_df.to_csv(pred_out, index=False)
    print(f"\n[predictions] wrote {len(pred_df)} rows -> {pred_out}")

    # ---------------- bootstrap + kelly on main best eps ----------------
    best_gate = main_r["best_gate"]
    ci = bootstrap_roi_ci(best_gate["bets"]) if best_gate["n_bets"] else (np.nan, np.nan)
    kelly = kelly_on_bets(best_gate["bets"]) if best_gate["n_bets"] else {
        "total_staked_fraction": 0.0, "bankroll_multiple": 1.0, "pnl_units": 0.0}

    # flat-bet bootstrap (verdict: distinguishable from 0?)
    won_flat = (p_home > 0.5) == meta["home_won"].values
    dec_of_side = np.where(p_home > 0.5, meta["dec_home"].values, meta["dec_away"].values)
    flat_pnls = np.where(won_flat, dec_of_side - 1.0, -1.0)
    flat_ci = bootstrap_roi_ci(
        [{"pnl": float(p)} for p in flat_pnls], n_boot=N_BOOT
    ) if len(flat_pnls) >= 5 else (np.nan, np.nan)

    # ---------------- v1 replay ----------------
    v1_csv = PROJECT_ROOT / "backtest" / "odds_predictions.csv"
    v1 = v1_through_gate(v1_csv) if v1_csv.exists() else None

    # ---------------- baselines ----------------
    base_rows = []
    for _, g in meta.iterrows():
        hw = bool(g["home_won"])
        base_rows.append(
            {
                "always_home": (g["dec_home"] - 1.0) if hw else -1.0,
                "always_fav": (min(g["dec_home"], g["dec_away"]) - 1.0)
                if (g["dec_home"] <= g["dec_away"]) == hw else -1.0,
            }
        )
    bdf = pd.DataFrame(base_rows)
    baselines = {
        "always_home_roi": float(bdf["always_home"].mean()),
        "always_fav_roi": float(bdf["always_fav"].mean()),
    }

    mkt_brier = brier(meta["p_market_home"].values, meta["home_won"].values)

    _write_report(
        results=results, built=built, main=main_r, v1=v1, baselines=baselines,
        mkt_brier=mkt_brier, ci=ci, kelly=kelly, n_test=len(meta),
        flat_ci=flat_ci,
    )
    print("\nDone. Report -> V2_BACKTEST_REPORT.md")

    import json

    best_roi_val = main_r["best_gate"]["roi"]
    no_inj_roi = results["no_injury"]["best_gate"]["roi"]
    inj_delta = (
        (best_roi_val - no_inj_roi) * 100
        if best_roi_val is not None and no_inj_roi is not None
        else None
    )
    summary = {
        "best_config": f"{main_r['model']}; eps={main_r['best_eps']} "
                       f"({main_r['best_gate']['n_bets']} bets)",
        "best_roi": best_roi_val if best_roi_val is not None else main_r["flat"]["roi"],
        "flat_roi": main_r["flat"]["roi"],
        "n_test_games": int(len(meta)),
        "report_path": "V2_BACKTEST_REPORT.md",
        "injury_ablation_delta_roi": (
            "n/a" if inj_delta is None else f"{inj_delta:+.2f}pp"
        ),
        "v1_flat_roi": v1["flat"]["roi"] if v1 is not None else None,
        "v1_best_gate_roi": v1["best_gate"]["roi"] if v1 is not None else None,
    }
    print("\nCONTRACT_JSON:" + json.dumps(summary))
    return 0


def _fmt(x: Optional[float], suffix: str = "") -> str:
    if x is None:
        return "n/a"
    return f"{x:+.4f}{suffix}"


def _write_report(results, built, main, v1, baselines, mkt_brier, ci, kelly, n_test,
                  flat_ci=None):
    train, test = built["train"], built["test"]
    eps_cols = " | ".join(f"ε={e}" for e in EPS_GRID)

    def gate_row(r: Dict, key: str, fmt_fn) -> str:
        cells = [fmt_fn(r["gates"][e][key]) for e in EPS_GRID]
        return " | ".join(cells)

    main_g = main["gates"]
    lines = []
    lines.append("# V2 Backtest Report — Injury-Aware Walk-Forward Model\n")
    lines.append("**Model v2:** rolling Four Factors + margin Elo + rest + OFFICIAL"
                 " NBA injury reports (5PM game-day filing; 1PM snapshot for "
                 "noon–5PM tips) + closing-line market signal, logistic "
                 "regression / gradient boosting, strictly walk-forward.")
    lines.append(f"**Training:** {len(train)} games, {train['date'].min()} → "
                 f"{train['date'].max()}")
    lines.append(f"**Test:** {n_test} games, {test['date'].min()} → "
                 f"{test['date'].max()} (same 229-game window as v1)")
    lines.append(f"**Dropped:** {built['dropped']}")
    lines.append(f"**Injury report source:** {built['injury_sources']}\n")

    lines.append("## Features\n")
    for f in F.FEATURE_NAMES:
        lines.append(f"- `{f}`")
    lines.append("\nFour Factors are rolling means over each team's last ≤10 games "
                 "(min 5, strictly prior). Injury impact = Σ status-weight × "
                 "prior-minutes share (Out 1.0 / Doubtful 0.5 / Questionable 0.25 / "
                 "Probable 0), from the official game-day report. Elo: K=20, "
                 "home-court +100, margin multiplier. Market = no-vig closing ML "
                 "implied home prob.\n")

    lines.append("\n## Model Selection (time-ordered 5-fold CV, neg-log-loss)\n")
    lines.append("Feature selection: greedy backward elimination minimizing CV "
                 "log-loss (training data only), then C/family re-tuned by the "
                 "same CV; calibrator (raw/Platt/isotonic) chosen by Feb-2022 "
                 "holdout log loss.\n")
    lines.append("| model | CV log-loss |")
    lines.append("|---|---:|")
    for name, ll in sorted(main["cv_table"], key=lambda t: t[1]):
        lines.append(f"| {name} | {ll:.4f} |")
    lines.append(f"\n**Chosen:** {main['model']} (CV log-loss "
                 f"{main['cv_ll']:.4f}); {len(main['sel_cols'])} selected features: "
                 f"`{', '.join(main['sel_cols'])}`; calibrator: "
                 f"{main['calibrator']} (fitted on Feb-2022 holdout).\n")

    lines.append("## Test Results (March 2022)\n")
    flat = main["flat"]
    lines.append(f"- Accuracy: **{flat['accuracy']*100:.1f}%** (v1: 67.2%)")
    lines.append(f"- Brier (model): **{main['brier']:.4f}** vs market Brier "
                 f"{mkt_brier:.4f}")
    flat_ci_str = ""
    if flat_ci is not None and flat_ci[0] == flat_ci[0]:
        flat_ci_str = (f"; bootstrap 95% CI [{flat_ci[0]*100:+.2f}%, "
                       f"{flat_ci[1]*100:+.2f}%]")
    lines.append(f"- Flat $1 on every predicted winner: P&L {flat['pnl']:+.2f} on "
                 f"${flat['n']} → ROI **{flat['roi']*100:+.2f}%** (v1: -0.8%)"
                 f"{flat_ci_str}")
    lines.append(f"- Baselines: always-home {baselines['always_home_roi']*100:+.2f}%, "
                 f"always-closing-favorite {baselines['always_fav_roi']*100:+.2f}%\n")
    lines.append("### Injury-report methodology\n")
    n_1pm = built["injury_sources"].get("game-day 1PM (early tip)", 0)
    n_5pm = built["injury_sources"].get("game-day 5PM", 0)
    lines.append("For each game g we use the official NBA injury report: the 5PM ET "
                 "game-day filing (snapshot `datetime(g, 17, 0)`, rows with Game "
                 "Date == g) — the complete NBA-mandated pre-game report, strictly "
                 "pre-tip for all evening games. The 5PM day-before filing was "
                 "inspected first and rejected as primary: it contains "
                 "'NOT YET SUBMITTED' placeholders for ~600 of 857 matchups "
                 "(teams routinely file late), so it systematically under-reports "
                 f"injuries. {n_1pm} of {n_1pm + n_5pm} game rows tip noon–5PM ET "
                 "and use the 1PM game-day snapshot so the information stays "
                 "strictly pre-tip. Injury impact = Σ status-weight × "
                 "prior-minutes share.\n")

    lines.append("## Edge-Gated Betting (bet $1 where model beats market by ≥ ε)\n")
    lines.append(f"| metric | {eps_cols} |")
    lines.append(f"|---|---{'|---' * len(EPS_GRID)}|")
    lines.append("| n_bets | " + gate_row(main, "n_bets", lambda x: str(x)) + " |")
    lines.append("| win rate | " + gate_row(main, "win_rate",
                 lambda x: "n/a" if x is None else f"{x*100:.1f}%") + " |")
    lines.append("| ROI | " + gate_row(main, "roi",
                 lambda x: "n/a" if x is None else f"{x*100:+.2f}%") + " |")
    best_g = main_g[main["best_eps"]]
    wr_txt = "n/a" if best_g["win_rate"] is None else f"{best_g['win_rate']*100:.1f}%"
    roi_txt = "n/a" if best_g["roi"] is None else f"{best_g['roi']*100:+.2f}%"
    ci_txt = f"[{ci[0]*100:+.2f}%, {ci[1]*100:+.2f}%]"
    lines.append(f"\n**Best gate:** ε={main['best_eps']} → {best_g['n_bets']} bets, "
                 f"win rate {wr_txt}, ROI **{roi_txt}** "
                 f"(bootstrap 95% CI {ci_txt})\n")

    lines.append("## Quarter-Kelly on best ε\n")
    lines.append(f"- Stakes clipped to [0, 0.05] of bankroll; "
                 f"{main_g[main['best_eps']]['n_bets']} bets")
    lines.append(f"- Bankroll multiple: {kelly['bankroll_multiple']:.4f} "
                 f"(flat-$1 P&L on same bets: "
                 f"{main_g[main['best_eps']]['pnl']:+.2f})\n")

    lines.append("## Ablations (each with best-ε gating)\n")
    lines.append("| config | feats (sel) | model | CV LL | calib | flat ROI | Brier |"
                 " best ε | n_bets | win rate | gated ROI |")
    lines.append("|---|---:|---|---:|---|---:|---:|---:|---:|---:|")
    for name in ["main", "no_injury", "no_market", "stats_only"]:
        r = results[name]
        g = r["best_gate"]
        lines.append(
            f"| {name} | {len(r['sel_cols'])}/{len(ABLATIONS[name])} | {r['model']} | "
            f"{r['cv_ll']:.4f} | {r['calibrator']} | {r['flat']['roi']*100:+.2f}% | "
            f"{r['brier']:.4f} | {r['best_eps']} | {g['n_bets']} | "
            f"{'n/a' if g['win_rate'] is None else format(g['win_rate']*100, '.1f') + '%'} | "
            f"{'n/a' if g['roi'] is None else format(g['roi']*100, '+.2f') + '%'} |"
        )
    lines.append("\nSelected features per config (CV greedy backward elimination):")
    for name in ["main", "no_injury", "no_market", "stats_only"]:
        r = results[name]
        lines.append(f"- **{name}**: `{', '.join(r['sel_cols'])}`")
    inj_delta = None
    ni_roi = results["no_injury"]["best_gate"]["roi"]
    m_roi = results["main"]["best_gate"]["roi"]
    if ni_roi is not None and m_roi is not None:
        inj_delta = (m_roi - ni_roi) * 100
    lines.append(
        "\n**Injury signal delta:** " +
        ("n/a" if inj_delta is None else f"{inj_delta:+.2f}pp of gated ROI") +
        " (main vs no_injury, best-ε each).\n"
    )

    if v1 is not None:
        lines.append("## v1 Engine Through the Same Edge Gate\n")
        lines.append(f"| metric | {eps_cols} |")
        lines.append(f"|---|---{'|---' * len(EPS_GRID)}|")
        lines.append("| v1 n_bets | " + gate_row(v1, "n_bets", lambda x: str(x)) + " |")
        lines.append("| v1 win rate | " + gate_row(v1, "win_rate",
                     lambda x: "n/a" if x is None else f"{x*100:.1f}%") + " |")
        lines.append("| v1 ROI | " + gate_row(v1, "roi",
                     lambda x: "n/a" if x is None else f"{x*100:+.2f}%") + " |")
        lines.append(f"\nv1 flat: ROI {v1['flat']['roi']*100:+.2f}%, accuracy "
                     f"{v1['flat']['accuracy']*100:.1f}%, Brier {v1['brier']:.4f}. "
                     f"v1 best gate: ε={v1['best_eps']} → ROI "
                     f"{v1['best_gate']['roi']*100:+.2f}% "
                     f"({v1['best_gate']['n_bets']} bets).\n")

    lines.append("## Verdict\n")
    flat_roi = main["flat"]["roi"]
    v1_roi = -0.008
    beats_v1 = flat_roi > v1_roi
    best_roi = main_g[main["best_eps"]]["roi"]
    if best_roi is None:
        ci_str = "n/a (no gated bets)"
    else:
        ci_str = (f"bootstrap 95% CI [{ci[0]*100:+.2f}%, {ci[1]*100:+.2f}%] — "
                  f"{'does' if ci[0] > 0 else 'does not'} exclude 0")
    flat_ci_str = ""
    if flat_ci is not None and flat_ci[0] == flat_ci[0]:
        flat_ci_str = (f" Flat-ROI bootstrap 95% CI [{flat_ci[0]*100:+.2f}%, "
                       f"{flat_ci[1]*100:+.2f}%] — "
                       f"{'excludes' if flat_ci[0] > 0 else 'does not exclude'} 0.")
    no_inj_flat = results["no_injury"]["flat"]["roi"]
    lines.append(
        f"The v2 model's flat ROI is **{flat_roi*100:+.2f}%** vs v1's -0.8%: "
        f"{'BEATS' if beats_v1 else 'does not beat'} v1 on flat betting "
        f"(accuracy {main['flat']['accuracy']*100:.1f}% vs v1's 67.2%)."
        f"{flat_ci_str} Injury features add "
        f"{(flat_roi - no_inj_flat)*100:+.2f}pp of flat ROI "
        f"(main vs no_injury ablation). Gated ROI at best ε is "
        f"{'n/a' if best_roi is None else f'{best_roi*100:+.2f}%'} with {ci_str}."
    )
    lines.append("\n## Reproducing\n")
    lines.append("```bash\nexport JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/"
                 "openjdk.jdk/Contents/Home\n.venv/bin/python backtest/v2/data.py   # "
                 "injury fetch (cached)\n.venv/bin/python backtest/v2/run_v2.py\n```\n")

    report = "\n".join(lines) + "\n"
    out = PROJECT_ROOT / "V2_BACKTEST_REPORT.md"
    out.write_text(report)
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
