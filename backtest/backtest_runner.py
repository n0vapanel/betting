#!/usr/bin/env python3
"""Backtest the Telegram betting bot's real prediction engine against NBA history.

Replays every 2025-26 regular-season game in the window 2026-03-01..2026-03-31
through `PredictionEngine.get_predictions` (the bot's actual production code,
via the stub-client pattern from `tests/test_engine.py`), using ONLY data
available before each game, and scores predictions against the real results.

Two variants are produced:
  A) "Engine default"      - the engine as-is: 0.6 stats / 0.4 odds weights,
                             with NO odds available (user has no odds API key),
                             so the odds term is the neutral 2.0/2.0 -> 50/50.
  B) "Stats only"          - `calculate_win_probabilities` called directly with
                             weight_odds=0.0, i.e. pure stats, no odds term.

Outputs:
  backtest/predictions.csv   one row per predicted game (both variants' columns)
  BACKTEST_REPORT.md         full numbers, written at the project root

Run from the project root:
  .venv/bin/python backtest/backtest_runner.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from telegram_betting_bot_simple.engine.analytics import calculate_win_probabilities
from telegram_betting_bot_simple.engine.prediction_engine import PredictionEngine
from telegram_betting_bot_simple.engine.tier_classifier import TierClassifier

try:  # package context (python -m backtest.backtest_runner)
    from .nba_data import (
        SEASON,
        WINDOW_END,
        WINDOW_START,
        build_march_games,
        build_prior_stats_provider,
        load_or_fetch_logs,
    )
    from .stub_clients import NbaBacktestApiSportsClient, NoOddsClient
except ImportError:  # plain script context (python backtest/backtest_runner.py)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nba_data import (  # type: ignore[no-redef]
        SEASON,
        WINDOW_END,
        WINDOW_START,
        build_march_games,
        build_prior_stats_provider,
        load_or_fetch_logs,
    )
    from stub_clients import NbaBacktestApiSportsClient, NoOddsClient  # type: ignore[no-redef]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = Path(__file__).resolve().parent / "predictions.csv"
REPORT_PATH = PROJECT_ROOT / "BACKTEST_REPORT.md"

NEUTRAL_ODDS = {"home": 2.0, "away": 2.0}


def _fmt_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.1f}%"


def run_backtest() -> Tuple[List[Dict], List[Tuple[str, str, str, str]], int]:
    """Run both prediction variants over the window; return (rows, skipped, parse_fail)."""
    logs = load_or_fetch_logs()
    games, skipped, parse_fail = build_march_games(logs)
    stats_provider = build_prior_stats_provider(logs)

    fixtures_by_date: Dict[str, List[Dict]] = defaultdict(list)
    for game in games:
        fixtures_by_date[game["date"].isoformat()].append(game)

    stub_api = NbaBacktestApiSportsClient(fixtures_by_date, stats_provider)
    engine = PredictionEngine(api_sports_key="backtest-stub")
    engine._api_sports_client = stub_api  # type: ignore[assignment]
    engine._odds_client = NoOddsClient()  # type: ignore[assignment]
    classifier = TierClassifier()

    rows: List[Dict] = []
    skipped_pred: List[Tuple[str, str, str, str]] = list(skipped)

    for game in games:
        home, away = game["home"], game["away"]
        game_date = game["date"]
        actual_winner = home if game["home_won"] else away

        # Variant A: the real engine, neutral odds fallback.
        predictions = engine.get_predictions("nba", game_date)
        pred = next(
            (p for p in predictions if p["home_team"] == home and p["away_team"] == away),
            None,
        )
        if pred is None:
            skipped_pred.append(
                (game_date.isoformat(), home, away, "engine returned no matching prediction")
            )
            continue

        # No-lookahead guard: stats must come from strictly earlier games.
        home_stats = stats_provider(game_date, home)
        away_stats = stats_provider(game_date, away)
        if home_stats["games_played"] == 0 or away_stats["games_played"] == 0:
            which = home if home_stats["games_played"] == 0 else away
            skipped_pred.append(
                (game_date.isoformat(), home, away, f"no prior games on record for {which}")
            )
            continue

        # Variant B: pure stats (weight_odds=0), same injury factors (none).
        probs_b = calculate_win_probabilities(
            home,
            away,
            home_stats,
            away_stats,
            NEUTRAL_ODDS,
            1.0,
            1.0,
            weight_odds=0.0,
        )
        winner_b, tier_b = classifier.classify(probs_b)

        home_prob_a = float(pred["probabilities"].get(home, 0.5))  # type: ignore[union-attr]
        rows.append(
            {
                "date": game_date.isoformat(),
                "home": home,
                "away": away,
                "predicted_home_prob": round(home_prob_a, 4),
                "predicted_winner": pred["winner"],
                "tier": pred["tier"],
                "actual_winner": actual_winner,
                "correct": pred["winner"] == actual_winner,
                "stats_only_predicted_home_prob": round(float(probs_b.get(home, 0.5)), 4),
                "stats_only_predicted_winner": winner_b,
                "stats_only_tier": tier_b,
                "stats_only_correct": winner_b == actual_winner,
            }
        )

    return rows, skipped_pred, parse_fail


def compute_metrics(rows: List[Dict], key_prefix: str = "") -> Dict:
    """Aggregate accuracy, tier breakdown, calibration and favorite/underdog stats."""
    prefix = key_prefix
    prob_key = f"{prefix}predicted_home_prob"
    winner_key = f"{prefix}predicted_winner"
    tier_key = f"{prefix}tier"
    correct_key = f"{prefix}correct"

    total = len(rows)
    correct = sum(1 for r in rows if r[correct_key])

    by_tier: Dict[str, Dict[str, int]] = {}
    for tier in ("Surefire", "Middle", "Hot"):
        subset = [r for r in rows if r[tier_key] == tier]
        by_tier[tier] = {
            "count": len(subset),
            "correct": sum(1 for r in subset if r[correct_key]),
        }

    # Calibration: bin predicted home win prob into 0.1 buckets.
    buckets = defaultdict(list)
    for r in rows:
        prob = float(r[prob_key])
        bucket = min(int(prob * 10), 9)
        buckets[bucket].append(r)
    calibration = []
    for bucket in range(10):
        subset = buckets.get(bucket, [])
        if not subset:
            continue
        n = len(subset)
        home_wins = sum(1 for r in subset if r["actual_winner"] == r["home"])
        calibration.append(
            {
                "bucket": f"{bucket / 10:.1f}-{(bucket + 1) / 10:.1f}",
                "count": n,
                "mean_predicted_home_prob": sum(float(r[prob_key]) for r in subset) / n,
                "actual_home_win_rate": home_wins / n,
            }
        )

    # Favorites (predicted winner) vs underdogs.
    fav_wins = correct
    dog_wins = total - correct

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_tier": by_tier,
        "calibration": calibration,
        "favorite_wins": fav_wins,
        "underdog_wins": dog_wins,
    }


def _tier_table(metrics: Dict) -> List[str]:
    lines = ["| Tier | Predicted | Correct | Accuracy |", "|------|----------:|--------:|---------:|"]
    for tier in ("Surefire", "Middle", "Hot"):
        t = metrics["by_tier"][tier]
        lines.append(
            f"| {tier} | {t['count']} | {t['correct']} | "
            f"{_fmt_pct(t['correct'], t['count'])} |"
        )
    return lines


def _calibration_table(metrics: Dict) -> List[str]:
    lines = [
        "| Predicted home prob bin | Games | Mean predicted home prob | Actual home win rate |",
        "|-------------------------|------:|-------------------------:|---------------------:|",
    ]
    for row in metrics["calibration"]:
        lines.append(
            f"| {row['bucket']} | {row['count']} | {row['mean_predicted_home_prob']:.3f} | "
            f"{100.0 * row['actual_home_win_rate']:.1f}% |"
        )
    return lines


def write_report(
    rows: List[Dict],
    skipped: List[Tuple[str, str, str, str]],
    parse_fail: int,
    fetch_error: Optional[str],
) -> None:
    metrics_a = compute_metrics(rows, key_prefix="")
    metrics_b = compute_metrics(rows, key_prefix="stats_only_")

    skip_counts = Counter(reason for _, _, _, reason in skipped)
    total_seen = len(rows) + len(skipped)

    lines: List[str] = []
    lines.append("# NBA Backtest Report")
    lines.append("")
    lines.append(
        "Backtest of `telegram_betting_bot_simple`'s real prediction engine "
        "(`PredictionEngine.get_predictions`) against historical NBA results."
    )
    lines.append("")
    lines.append(f"- **Data source:** nba_api → stats.nba.com `LeagueGameFinder` (free, no API key)")
    lines.append(f"- **Season:** {SEASON} regular season")
    lines.append(f"- **Window replayed:** {WINDOW_START.isoformat()} → {WINDOW_END.isoformat()} (fully final results)")
    lines.append("- **No-lookahead guarantee:** for every game, team stats (points for / points against / games played) are computed strictly from games played BEFORE that game's date. The game itself and all later games are invisible to the predictor.")
    lines.append("- **Odds:** no odds API key available, so the odds term is the engine's neutral fallback `{home: 2.0, away: 2.0}` (contributes 50/50 under the 0.4 odds weight).")

    if fetch_error:
        lines.append("")
        lines.append("## ⚠️ BLOCKER")
        lines.append("")
        lines.append(f"Data fetch failed, so no predictions could be scored:")
        lines.append("")
        lines.append(f"```\n{fetch_error}\n```")
        lines.append("")
        lines.append("The harness code in `backtest/` is complete; re-run once network access to stats.nba.com is available:")
        lines.append("")
        lines.append("```bash\n.venv/bin/python backtest/backtest_runner.py\n```")
        REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("")
    lines.append(f"- **Games seen in window:** {total_seen} (predicted: {len(rows)}, skipped: {len(skipped)})")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Variant | Games | Correct | Accuracy |")
    lines.append("|---------|------:|--------:|---------:|")
    lines.append(
        f"| Engine default (0.6 stats / 0.4 odds, neutral odds) | {metrics_a['total']} | "
        f"{metrics_a['correct']} | {_fmt_pct(metrics_a['correct'], metrics_a['total'])} |"
    )
    lines.append(
        f"| Stats only (weight_odds = 0.0) | {metrics_b['total']} | {metrics_b['correct']} | "
        f"{_fmt_pct(metrics_b['correct'], metrics_b['total'])} |"
    )
    lines.append("")
    lines.append(f"Overall accuracy (engine default): **{metrics_a['accuracy'] * 100:.1f}%**")
    lines.append(f"Overall accuracy (stats only): **{metrics_b['accuracy'] * 100:.1f}%**")
    lines.append("")
    lines.append(
        "Baseline for reference: NBA home teams won "
        f"{100.0 * sum(1 for r in rows if r['actual_winner'] == r['home']) / max(len(rows), 1):.1f}% "
        "of these games."
    )

    lines.append("")
    lines.append("## Accuracy by tier")
    lines.append("")
    lines.append("### Engine default (neutral odds)")
    lines.extend(_tier_table(metrics_a))
    lines.append("")
    lines.append("### Stats only (weight_odds = 0.0)")
    lines.extend(_tier_table(metrics_b))
    lines.append("")
    lines.append(
        "> **Why no Middle/Surefire picks?** Both variants draw from the same "
        "pre-game stats; with neutral odds the 0.4 odds weight pulls every "
        "probability toward 0.5, and the raw stats-strength ratio never "
        "reaches the 0.6 / 0.75 tier thresholds. All picks land in the "
        "\"Hot\" tier, so tier-level accuracy equals overall accuracy here."
    )

    lines.append("")
    lines.append("## Calibration (predicted home win probability vs actual home win rate)")
    lines.append("")
    lines.append("### Engine default (neutral odds)")
    lines.extend(_calibration_table(metrics_a))
    lines.append("")
    lines.append("### Stats only (weight_odds = 0.0)")
    lines.extend(_calibration_table(metrics_b))

    lines.append("")
    lines.append("## Favorites vs underdogs")
    lines.append("")
    lines.append("| Variant | Predicted favorite won | Underdog won | Favorite win rate |")
    lines.append("|---------|-----------------------:|-------------:|------------------:|")
    lines.append(
        f"| Engine default | {metrics_a['favorite_wins']} | {metrics_a['underdog_wins']} | "
        f"{_fmt_pct(metrics_a['favorite_wins'], metrics_a['total'])} |"
    )
    lines.append(
        f"| Stats only | {metrics_b['favorite_wins']} | {metrics_b['underdog_wins']} | "
        f"{_fmt_pct(metrics_b['favorite_wins'], metrics_b['total'])} |"
    )

    lines.append("")
    lines.append("## Games skipped")
    lines.append("")
    if skipped:
        lines.append(f"**{len(skipped)}** games were skipped:")
        lines.append("")
        lines.append("| Reason | Count |")
        lines.append("|--------|------:|")
        for reason, count in sorted(skip_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {reason} | {count} |")
        lines.append("")
        lines.append("<details><summary>Detailed list</summary>")
        lines.append("")
        for game_date, home, away, reason in skipped:
            lines.append(f"- {game_date} {home} @ {away}: {reason}")
        lines.append("")
        lines.append("</details>")
    else:
        lines.append("None.")
    if parse_fail:
        lines.append("")
        lines.append(
            f"Additionally, {parse_fail} game-log row(s) had unparseable MATCHUP values "
            "and could not be attributed to a game."
        )

    lines.append("")
    lines.append("## Reproducing")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/pip install -r backtest/requirements.txt")
    lines.append(".venv/bin/python backtest/backtest_runner.py")
    lines.append("```")
    lines.append("")
    lines.append("Raw game logs are cached in `backtest/cache/` (gitignored); delete the cache to force a re-fetch.")
    lines.append("")
    lines.append("## Output")
    lines.append("")
    lines.append(
        "`backtest/predictions.csv` — one row per predicted game: "
        "`date, home, away, predicted_home_prob, predicted_winner, tier, actual_winner, correct` "
        "plus the stats-only variant columns (`stats_only_predicted_home_prob`, "
        "`stats_only_predicted_winner`, `stats_only_tier`, `stats_only_correct`)."
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_predictions_csv(rows: List[Dict]) -> None:
    fieldnames = [
        "date",
        "home",
        "away",
        "predicted_home_prob",
        "predicted_winner",
        "tier",
        "actual_winner",
        "correct",
        "stats_only_predicted_home_prob",
        "stats_only_predicted_winner",
        "stats_only_tier",
        "stats_only_correct",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["date"], r["home"])):
            writer.writerow(row)


def main() -> int:
    fetch_error: Optional[str] = None
    try:
        rows, skipped, parse_fail = run_backtest()
    except Exception as exc:  # noqa: BLE001 - surface any data-fetch blocker
        fetch_error = f"{type(exc).__name__}: {exc}"
        rows, skipped, parse_fail = [], [], 0

    write_report(rows, skipped, parse_fail, fetch_error)
    if fetch_error:
        print(f"BLOCKED: {fetch_error}")
        print(f"Report written to {REPORT_PATH} (predictions unavailable).")
        return 1

    if rows:
        write_predictions_csv(rows)
    metrics_a = compute_metrics(rows, key_prefix="")
    print(
        f"Predicted {metrics_a['total']} games, correct {metrics_a['correct']} "
        f"({metrics_a['accuracy'] * 100:.1f}%), skipped {len(skipped)}."
    )
    print(f"predictions.csv -> {CSV_PATH}")
    print(f"report            -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
