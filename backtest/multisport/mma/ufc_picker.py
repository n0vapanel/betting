#!/usr/bin/env python3
"""
ufc_picker.py — live-card picks from the frozen MMA ensemble.

Input: a CSV of upcoming fights with columns
    date, red_fighter, blue_fighter, weight_class, R_odds, B_odds
(American closing odds; optional extra columns: title_bout, no_of_rounds, gender).

Or single-fight mode:
    --name1 "Fighter A" --name2 "Fighter B" --date 2026-04-04
    --odds1 -150 --odds2 +120 [--weight-class Lightweight]

What it does:
  1. Appends the card rows to the fight history (all fights strictly before the
     first card date — no future data).
  2. Runs the SAME feature builders as the v1 backtest (Elo, record, streak,
     days-since, strike stats) on the combined frame.
  3. Predicts with the frozen trained ensemble (LogReg + LGBM, isotonic-calibrated,
     50/50 average).
  4. De-vigs the input odds -> fair probabilities; edge = p_model - p_fair on each side.
  5. Prints the pick side, model prob, fair prob, edge, flat-$1 bet (edge > 0),
     quarter-Kelly stake (capped at 25% of bankroll) and writes a card report markdown.

Models: models/extended_core_thru_2025.joblib (default; core features, trained
through 2025-12-31) or models/primary_2016_2019.joblib (strike features, 2016-2019;
only use for cards where ufcstats strike data exists, i.e. <= 2021).
Run train_model.py first if models/ is missing.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

# lightgbm>=4.6 deprecation noise on eval_set (kept for v1 parity)
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)

from backtest_lib import am_to_dec, fair_probs          # noqa: E402
from feature_utils import (                             # noqa: E402
    load_mdabbert, parse_rajeevw, build_card_features, add_model_features)
from train_model import predict_ensemble                # noqa: E402

MODELS_DIR = os.path.join(base, "models")
CARD_COLS = ["date", "red_fighter", "blue_fighter", "weight_class", "R_odds", "B_odds"]
OPTIONAL_COLS = ["title_bout", "no_of_rounds", "gender"]


def load_model(name):
    import joblib
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        sys.exit(f"model not found: {path}\nRun: python {os.path.join(base, 'train_model.py')}")
    return joblib.load(path)


def load_card(card_path):
    df = pd.read_csv(card_path)
    missing = [c for c in CARD_COLS if c not in df.columns]
    if missing:
        sys.exit(f"card CSV missing required columns: {missing}\n"
                 f"required: {', '.join(CARD_COLS)}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for c in ["R_odds", "B_odds"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[df["R_odds"].isna() | df["B_odds"].isna()]
    if len(bad):
        sys.exit(f"non-numeric odds in rows:\n{bad[['date', 'red_fighter', 'blue_fighter']]}")
    df["red_fighter"] = df["red_fighter"].astype(str).str.strip()
    df["blue_fighter"] = df["blue_fighter"].astype(str).str.strip()
    df["weight_class"] = df["weight_class"].astype(str).str.strip()
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = None
    return df


def run_picks(card_df, model, as_of=None, bankroll=1000.0):
    """Returns (report_df, warnings, cutoff, meta)."""
    md = load_mdabbert(base)
    long = parse_rajeevw(base)

    merged, fids, cutoff = build_card_features(md, long, card_df,
                                               cutoff=pd.Timestamp(as_of).normalize()
                                               if as_of else None)
    if as_of is not None:
        asof_ts = pd.Timestamp(as_of).normalize()
        if asof_ts > pd.to_datetime(card_df["date"]).dt.normalize().min():
            sys.exit(f"--as-of {as_of} is after the first card date "
                     f"{pd.to_datetime(card_df['date']).dt.normalize().min().date()}; "
                     "that would leak future data")

    wc_cols = [f for f in model["feats"] if f.startswith("wc_")]
    feats_df, _, _ = add_model_features(merged, wc_columns=wc_cols)
    card_mask = feats_df["fid"].isin(fids)
    X = feats_df.loc[card_mask, model["feats"]].reindex(columns=model["feats"]).values

    pR = predict_ensemble(model, X)
    card = feats_df[card_mask].reset_index(drop=True).copy()
    decR = am_to_dec(card["R_odds"].values)
    decB = am_to_dec(card["B_odds"].values)
    fairR, fairB = fair_probs(decR, decB)
    pB = 1.0 - pR
    edge_R = pR - fairR
    edge_B = pB - fairB

    pick_R = edge_R >= edge_B
    p_pick = np.where(pick_R, pR, pB)
    fair_pick = np.where(pick_R, fairR, fairB)
    edge_pick = np.maximum(edge_R, edge_B)
    dec_pick = np.where(pick_R, decR, decB)

    # quarter-Kelly stake, capped at 25% of bankroll
    f_kelly = (p_pick * dec_pick - 1.0) / (dec_pick - 1.0)
    stake_frac = np.where(f_kelly > 0, np.minimum(0.25 * f_kelly, 0.25), 0.0)
    stake_usd = stake_frac * bankroll

    # warnings
    warnings = []
    for i, (r, b) in enumerate(zip(card["R_fighter"], card["B_fighter"])):
        for f, ncol in [(r, "n_prior_R"), (b, "n_prior_B")]:
            if card.iloc[i][ncol] == 0:
                warnings.append(f"{f}: no prior UFC fights in history (n_prior=0)")
        for col, label in [("R_age", f"{r} age"), ("B_age", f"{b} age"),
                           ("R_Reach_cms", f"{r} reach"), ("B_Reach_cms", f"{b} reach")]:
            if pd.isna(card.iloc[i][col]):
                warnings.append(f"{label}: missing -> median-imputed")
        if pd.isna(card.iloc[i]["feat_slpm_R"]) and "strike" in str(model.get("feature_set")):
            warnings.append(f"strike features missing for {r} vs {b}")

    out = pd.DataFrame({
        "date": card["date_n"].dt.strftime("%Y-%m-%d"),
        "red_fighter": card["R_fighter"],
        "blue_fighter": card["B_fighter"],
        "weight_class": card["weight_class"],
        "R_odds": card["R_odds"].astype(int),
        "B_odds": card["B_odds"].astype(int),
        "pR_model": pR, "pB_model": pB,
        "fairR": fairR, "fairB": fairB,
        "edge_R": edge_R, "edge_B": edge_B,
        "pick": np.where(pick_R, "RED", "BLUE"),
        "pick_fighter": np.where(pick_R, card["R_fighter"], card["B_fighter"]),
        "model_prob": p_pick, "fair_prob": fair_pick, "edge": edge_pick,
        "flat_bet": edge_pick > 0.0,
        "qkelly_frac": stake_frac, "qkelly_usd": stake_usd,
        "R_dec": decR, "B_dec": decB,
    })
    meta = {"cutoff": cutoff, "model_feature_set": model.get("feature_set"),
            "model_train_hi": model.get("train_hi"), "n_card": len(card),
            "bankroll": bankroll}
    return out, warnings, meta


def fmt_table(out):
    rows = []
    for _, r in out.iterrows():
        pick = f"{r['pick']} ({r['pick_fighter']})"
        flat = "BET $1" if r["flat_bet"] else "-"
        stake = f"{r['qkelly_frac'] * 100:.1f}% (${r['qkelly_usd']:.2f})" if r["qkelly_frac"] > 0 else "-"
        rows.append((r["date"], f"{r['red_fighter']} vs {r['blue_fighter']}", r["weight_class"],
                     f"{r['R_odds']:+d}/{r['B_odds']:+d}",
                     f"{r['model_prob']:.3f}", f"{r['fair_prob']:.3f}", f"{r['edge']:+.4f}",
                     pick, flat, stake))
    cols = ["date", "fight", "class", "odds(R/B)", "p_model", "p_fair", "edge", "pick", "flat$1", "qKelly"]
    return pd.DataFrame(rows, columns=cols)


def write_report(out, meta, warnings, path):
    lines = ["# UFC Card Picks — Model Report", ""]
    lines.append(f"- feature cutoff (no future data): **{meta['cutoff'].date()}**")
    lines.append(f"- model: **{meta['model_feature_set']}** features, trained through "
                 f"**{meta['model_train_hi']}**")
    lines.append(f"- fights: {meta['n_card']} | bankroll for stakes: ${meta['bankroll']:,.0f}")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
    lines.append("")
    lines.append("## Picks")
    lines.append("")
    lines.append("| date | fight | weight class | odds (R/B) | p_model | p_fair | edge | pick | flat $1 | q-Kelly stake |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in out.iterrows():
        pick = f"**{r['pick']}: {r['pick_fighter']}**"
        flat = "YES" if r["flat_bet"] else "no"
        stake = f"{r['qkelly_frac'] * 100:.1f}% (${r['qkelly_usd']:.2f})" if r["qkelly_frac"] > 0 else "—"
        lines.append(f"| {r['date']} | {r['red_fighter']} vs {r['blue_fighter']} | {r['weight_class']} | "
                     f"{r['R_odds']:+d} / {r['B_odds']:+d} | {r['model_prob']:.3f} | {r['fair_prob']:.3f} | "
                     f"{r['edge']:+.4f} | {pick} | {flat} | {stake} |")
    lines.append("")
    lines.append("edge = model prob − de-vigged fair prob. Flat bet when edge > 0; "
                 "stake = min(¼·Kelly, 25%) of bankroll.")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser(description="UFC live-card picker (frozen v1 ensemble)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--card", help="CSV of upcoming fights (date, red_fighter, blue_fighter, "
                                    "weight_class, R_odds, B_odds)")
    src.add_argument("--name1", help="red-corner fighter name (single-fight mode)")
    ap.add_argument("--name2", help="blue-corner fighter name (single-fight mode)")
    ap.add_argument("--date", help="fight date YYYY-MM-DD (single-fight mode)")
    ap.add_argument("--odds1", type=float, help="American odds for name1 (single-fight mode)")
    ap.add_argument("--odds2", type=float, help="American odds for name2 (single-fight mode)")
    ap.add_argument("--weight-class", help="e.g. 'Lightweight' (single-fight mode; inferred if omitted)")
    ap.add_argument("--title-bout", action="store_true", help="single-fight mode: title bout (5 rounds)")
    ap.add_argument("--rounds", type=int, help="single-fight mode: scheduled rounds (default 3, 5 if --title-bout)")
    ap.add_argument("--model", default="extended",
                    choices=["extended", "primary", "auto"],
                    help="extended = core features trained through 2025 (default); "
                         "primary = strike features 2016-2019; auto = primary if strike data present")
    ap.add_argument("--out", help="markdown report path (default: <card>_report.md)")
    ap.add_argument("--bankroll", type=float, default=1000.0, help="bankroll for stakes (default 1000)")
    ap.add_argument("--as-of", help="feature cutoff override (must be <= first card date)")
    args = ap.parse_args()

    if args.card:
        card_df = load_card(args.card)
        out_path = args.out or os.path.splitext(os.path.abspath(args.card))[0] + "_report.md"
    else:
        need = [args.name1, args.name2, args.date, args.odds1, args.odds2]
        if any(v is None for v in need):
            ap.error("single-fight mode needs --name1 --name2 --date --odds1 --odds2")
        if args.weight_class:
            wc = args.weight_class
        else:
            md = load_mdabbert(base)
            from feature_utils import fighter_last_known
            cutoff = pd.Timestamp(args.date).normalize()
            lk = fighter_last_known(md, cutoff)
            wcs = {lk.get(f, {}).get("weight_class") for f in (args.name1, args.name2)}
            wcs = {w for w in wcs if w}
            if len(wcs) != 1:
                ap.error(f"cannot infer weight class from history ({wcs or 'no history'}) — "
                         "pass --weight-class")
            wc = wcs.pop()
        title_bout = bool(args.title_bout)
        rounds = args.rounds if args.rounds else (5 if title_bout else 3)
        card_df = pd.DataFrame([{
            "date": args.date, "red_fighter": args.name1, "blue_fighter": args.name2,
            "weight_class": wc, "R_odds": args.odds1, "B_odds": args.odds2,
            "title_bout": title_bout, "no_of_rounds": rounds, "gender": None}])
        out_path = args.out

    if args.model == "auto":
        # use primary only when strike data exists for the whole card (pre-2022 cards)
        md_probe = load_mdabbert(base)
        long_probe = parse_rajeevw(base)
        mprobe, fids_p, _ = build_card_features(md_probe, long_probe, card_df,
                                                cutoff=pd.to_datetime(args.as_of).dt.normalize()
                                                if args.as_of else None)
        probe_mask = mprobe["fid"].isin(fids_p)
        has_strike = (mprobe.loc[probe_mask, "feat_slpm_R"].notna().all()
                      and mprobe.loc[probe_mask, "feat_slpm_B"].notna().all())
        model_name = ("primary_2016_2019.joblib" if has_strike
                      else "extended_core_thru_2025.joblib")
    else:
        model_name = {"extended": "extended_core_thru_2025.joblib",
                      "primary": "primary_2016_2019.joblib"}[args.model]

    model = load_model(model_name)
    out, warnings, meta = run_picks(card_df, model, as_of=args.as_of, bankroll=args.bankroll)

    print(f"model: {model_name} ({meta['model_feature_set']} feats, trained thru {meta['model_train_hi']})")
    print(f"feature cutoff: {meta['cutoff'].date()} (no future data)")
    print()
    print(fmt_table(out).to_string(index=False))
    if warnings:
        print("\nwarnings:")
        for w in warnings:
            print(" -", w)
    if out_path:
        write_report(out, meta, warnings, out_path)
        print(f"\nreport: {out_path}")
    print(f"\nflat bets (edge>0): {int(out['flat_bet'].sum())}/{len(out)} | "
          f"total q-Kelly staked: ${out['qkelly_usd'].sum():.2f}")


if __name__ == "__main__":
    main()
