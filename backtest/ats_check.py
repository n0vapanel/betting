#!/usr/bin/env python3
"""ATS check for March 2026: engine picks (stats-only variant) vs closing spread.

Moneyline is missing for the modern window, but the Kaggle dataset carries
closing spreads.  This scores the stats-only picks from `predictions.csv`
against the spread (ATS): picked team covers if its adjusted margin > 0.
Break-even ATS at -110 vig is ~52.4%.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

HERE = Path(__file__).resolve().parent
KAGGLE_CSV = (
    Path.home() / ".cache/kagglehub/datasets/"
    "cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024/versions/4/"
    "nba_2008-2026.csv"
)
CODE_MAP = {"gs": "GSW", "ny": "NYK", "no": "NOP", "sa": "SAS", "utah": "UTA", "wsh": "WAS"}


def kaggle_to_abbr(code: str) -> str:
    return CODE_MAP.get(code, code.upper())


def main() -> int:
    picks: Dict = {}
    with open(HERE / "predictions.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            picks[(r["date"], r["home"], r["away"])] = r["stats_only_predicted_winner"]

    df = pd.read_csv(KAGGLE_CSV)
    mar = df[
        (df["date"] >= "2026-03-01")
        & (df["date"] <= "2026-03-31")
        & (df["regular"] == True)  # noqa: E712
    ]

    rows: List[Dict] = []
    unmatched = 0
    for _, g in mar.iterrows():
        if pd.isna(g["spread"]) or pd.isna(g["score_home"]) or pd.isna(g["score_away"]):
            continue
        home, away = kaggle_to_abbr(str(g["home"])), kaggle_to_abbr(str(g["away"]))
        key = (str(g["date"])[:10], home, away)
        pick = picks.get(key)
        if pick is None:
            unmatched += 1
            continue
        spread = float(g["spread"])
        fav = home if str(g["whos_favored"]).lower() == "home" else away
        picked_score = float(g["score_home"]) if pick == home else float(g["score_away"])
        opp_score = float(g["score_away"]) if pick == home else float(g["score_home"])
        raw_margin = picked_score - opp_score
        # Favorite must win by MORE than the spread; underdog covers if it loses
        # by LESS than the spread (or wins outright).
        if pick == fav:
            margin = raw_margin - spread
        else:
            margin = raw_margin + spread
        rows.append(
            {
                "date": key[0],
                "home": home,
                "away": away,
                "pick": pick,
                "favorite": fav,
                "spread": spread,
                "pick_ats_margin": round(margin, 1),
                "covers": margin > 0,
                "push": abs(margin) < 1e-9,
            }
        )

    n = len(rows)
    covers = sum(1 for r in rows if r["covers"])
    pushes = sum(1 for r in rows if r["push"])
    decisive = n - pushes
    rate = covers / decisive if decisive else 0.0
    fav_pick = sum(1 for r in rows if r["pick"] == r["favorite"])

    with open(HERE / "ats_check.csv", "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["date", "home", "away", "pick", "favorite", "spread",
                            "pick_ats_margin", "covers", "push"]
        )
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r["date"], r["home"])):
            writer.writerow(r)

    print(f"ATS check (March 2026, stats-only picks vs closing spread):")
    print(f"  games matched: {n} (unmatched: {unmatched})")
    print(f"  covers: {covers} | pushes: {pushes} | ATS rate: {rate*100:.1f}% "
          f"(break-even ~52.4%)")
    print(f"  picks were the favorite {fav_pick}/{n} times")
    print("  -> backtest/ats_check.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
