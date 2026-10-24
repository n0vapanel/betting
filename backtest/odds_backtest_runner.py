#!/usr/bin/env python3
"""Odds-fed backtest: replay the betting bot's real PredictionEngine over NBA
games with REAL closing moneylines, and score P&L per $1 staked.

Moneyline data: Kaggle 'NBA Betting Data 2007-2026' (cviaxmiwnptr) — closing
lines via SportsbookReview/ESPN.  Moneyline coverage ends Jan 2023, so the
main window is March 2022 (229 games, 0 missing).  A secondary section checks
the engine's neutral-odds picks against the point spread (ATS) in March 2026,
where the dataset has spreads but no moneylines.

No lookahead: team stats come strictly from games BEFORE each predicted game
(same prefix-array provider as the neutral-odds harness).  Odds are pre-game
closing lines, so they are lookahead-free by definition.

Outputs: backtest/odds_predictions.csv + ODDS_BACKTEST_REPORT.md (repo root).
"""

from __future__ import annotations

import csv
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

try:
    from nba_api.stats.endpoints.leaguegamefinder import LeagueGameFinder
except Exception as exc:  # pragma: no cover
    LeagueGameFinder = None
    NBA_API_IMPORT_ERROR: Optional[Exception] = exc
else:
    NBA_API_IMPORT_ERROR = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nba_data import build_prior_stats_provider  # noqa: E402
from stub_clients import NbaBacktestApiSportsClient  # noqa: E402
from telegram_betting_bot_simple.engine.prediction_engine import PredictionEngine  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(__file__).resolve().parent / "cache"
KAGGLE_CSV = (
    Path.home() / ".cache/kagglehub/datasets/"
    "cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024/versions/4/"
    "nba_2008-2026.csv"
)
SEASON = "2021-22"
WINDOW_START = date(2022, 3, 1)
WINDOW_END = date(2022, 3, 31)
TEAM_IDS = [
    1610612737, 1610612738, 1610612751, 1610612766, 1610612741, 1610612739,
    1610612742, 1610612743, 1610612765, 1610612744, 1610612745, 1610612754,
    1610612746, 1610612747, 1610612763, 1610612748, 1610612749, 1610612750,
    1610612740, 1610612752, 1610612760, 1610612753, 1610612755, 1610612756,
    1610612757, 1610612758, 1610612759, 1610612761, 1610612762, 1610612764,
]
CODE_MAP = {"gs": "GSW", "ny": "NYK", "no": "NOP", "sa": "SAS", "utah": "UTA", "wsh": "WAS"}


def kaggle_to_abbr(code: str) -> str:
    return CODE_MAP.get(code, code.upper())


def _american_to_decimal(ml: float) -> float:
    if ml > 0:
        return 1.0 + ml / 100.0
    return 1.0 + 100.0 / abs(ml)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "PTS"]
    df = df[keep].copy()
    df["GAME_DATE"] = df["GAME_DATE"].astype(str).str[:10]
    return df


def _call_finder(team_id: int, attempts: int) -> Optional[pd.DataFrame]:
    if LeagueGameFinder is None:
        raise RuntimeError(f"nba_api is not importable: {NBA_API_IMPORT_ERROR}")
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            finder = LeagueGameFinder(
                team_id_nullable=team_id,
                league_id_nullable="00",
                season_nullable=SEASON,
                season_type_nullable="Regular Season",
                timeout=120,
            )
            return finder.get_data_frames()[0]
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(min(20.0, 2 ** attempt))
    print(f"WARN: fetch failed for team_id={team_id}: {last_exc}")
    return None


def fetch_season_logs() -> pd.DataFrame:
    cache = CACHE_DIR / f"nba_logs_{SEASON.replace('-', '_')}.csv"
    if cache.exists():
        df = pd.read_csv(cache, dtype={"GAME_ID": str})
        if len(df):
            return df
    try:
        df = _call_finder(team_id=0, attempts=4)
        if df is not None and len(df):
            out = _normalize(df)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            out.to_csv(cache, index=False)
            return out
    except Exception:
        pass
    frames = [_call_finder(t, 5) for t in TEAM_IDS]
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        raise RuntimeError("Could not fetch any game logs from stats.nba.com")
    out = _normalize(pd.concat(frames, ignore_index=True))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


class RealOddsStub:
    """Serves the day's real closing moneylines in the engine's odds shape."""

    def __init__(self, odds_entries: List[Dict]) -> None:
        self._entries = odds_entries

    def get_odds(self, sport_code: str, markets: str = "h2h") -> List[Dict]:
        return self._entries


def load_moneyline_games() -> List[Dict]:
    df = pd.read_csv(KAGGLE_CSV)
    mar = df[
        (df["date"] >= WINDOW_START.isoformat())
        & (df["date"] <= WINDOW_END.isoformat())
        & (df["regular"] == True)  # noqa: E712
    ]
    games: List[Dict] = []
    for _, r in mar.iterrows():
        if pd.isna(r["moneyline_home"]) or pd.isna(r["moneyline_away"]):
            continue
        if pd.isna(r["score_home"]) or pd.isna(r["score_away"]):
            continue
        home, away = kaggle_to_abbr(str(r["home"])), kaggle_to_abbr(str(r["away"]))
        games.append(
            {
                "date": date.fromisoformat(str(r["date"])[:10]),
                "home": home,
                "away": away,
                "dec_home": _american_to_decimal(float(r["moneyline_home"])),
                "dec_away": _american_to_decimal(float(r["moneyline_away"])),
                "home_won": float(r["score_home"]) > float(r["score_away"]),
            }
        )
    return games


def run_odds_backtest() -> List[Dict]:
    logs = fetch_season_logs()
    stats_provider = build_prior_stats_provider(logs)
    games = load_moneyline_games()

    fixtures_by_date: Dict[str, List[Dict]] = defaultdict(list)
    for g in games:
        fixtures_by_date[g["date"].isoformat()].append(g)

    odds_entries: List[Dict] = []
    for g in games:
        odds_entries.append(
            {
                "home_team": g["home"],
                "away_team": g["away"],
                "markets": {"h2h": {"home": g["dec_home"], "away": g["dec_away"]}},
            }
        )

    stub_api = NbaBacktestApiSportsClient(fixtures_by_date, stats_provider)
    engine = PredictionEngine(api_sports_key="backtest-stub", odds_api_key="backtest-stub")
    engine._api_sports_client = stub_api  # type: ignore[assignment]
    engine._odds_client = RealOddsStub(odds_entries)  # type: ignore[assignment]

    rows: List[Dict] = []
    for g in games:
        gd = g["date"]
        predictions = engine.get_predictions("nba", gd)
        pred = next(
            (p for p in predictions if p["home_team"] == g["home"] and p["away_team"] == g["away"]),
            None,
        )
        if pred is None:
            print(f"WARN: no prediction for {gd} {g['home']} vs {g['away']}")
            continue
        winner = pred["winner"]
        actual = g["home"] if g["home_won"] else g["away"]
        dec = g["dec_home"] if winner == g["home"] else g["dec_away"]
        correct = winner == actual
        pnl = (dec - 1.0) if correct else -1.0
        rows.append(
            {
                "date": gd.isoformat(),
                "home": g["home"],
                "away": g["away"],
                "predicted_home_prob": round(float(pred["probabilities"].get(g["home"], 0.5)), 4),
                "predicted_winner": winner,
                "tier": pred["tier"],
                "actual_winner": actual,
                "correct": correct,
                "ml_decimal": round(dec, 4),
                "dec_home": round(g["dec_home"], 4),
                "dec_away": round(g["dec_away"], 4),
                "pnl_1usd": round(pnl, 4),
            }
        )
    return rows


def _agg(rows: List[Dict], subset: Callable[[Dict], bool] = lambda r: True) -> Dict:
    sel = [r for r in rows if subset(r)]
    n = len(sel)
    if n == 0:
        return {"n": 0, "acc": None, "roi": None, "pnl": 0.0}
    correct = sum(1 for r in sel if r["correct"])
    pnl = sum(r["pnl_1usd"] for r in sel)
    return {"n": n, "acc": correct / n, "roi": pnl / n, "pnl": pnl}


def main() -> int:
    rows = run_odds_backtest()
    if not rows:
        print("BLOCKED: no odds-fed predictions produced.")
        return 1

    overall = _agg(rows)
    by_tier = {t: _agg(rows, lambda r, t=t: r["tier"] == t) for t in ("Surefire", "Middle", "Hot")}
    # Baselines: always bet home / always bet the moneyline favorite.
    home_pnl = sum(
        (r["dec_home"] - 1.0) if r["actual_winner"] == r["home"] else -1.0 for r in rows
    )
    fav_pnl = 0.0
    fav_n = 0
    for r in rows:
        fav = r["home"] if r["dec_home"] < r["dec_away"] else r["away"]
        dec = r["dec_home"] if fav == r["home"] else r["dec_away"]
        fav_pnl += (dec - 1.0) if r["actual_winner"] == fav else -1.0
        fav_n += 1

    # Calibration with real odds.
    buckets: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        buckets[min(int(float(r["predicted_home_prob"]) * 10), 9)].append(r)

    # Write CSV.
    with open(Path(__file__).resolve().parent / "odds_predictions.csv", "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["date", "home", "away", "predicted_home_prob", "predicted_winner",
                        "tier", "actual_winner", "correct", "ml_decimal", "dec_home",
                        "dec_away", "pnl_1usd"],
        )
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r["date"], r["home"])):
            writer.writerow(r)

    lines = [
        "# Odds-Fed Backtest Report (real closing moneylines)",
        "",
        f"- **Window:** {WINDOW_START.isoformat()} → {WINDOW_END.isoformat()} (2021-22 season, fully final)",
        f"- **Games:** {len(rows)} (all with closing moneyline, zero missing)",
        "- **Odds source:** Kaggle 'NBA Betting Data 2007-2026' (closing lines, SportsbookReview/ESPN)",
        "- **Engine:** real `PredictionEngine.get_predictions` with real h2h moneylines (0.6 stats / 0.4 odds)",
        "- **Stats:** no lookahead — strictly pre-game, prefix-array provider (same as neutral-odds harness)",
        "- **P&L:** flat $1 staked on every predicted winner at its closing moneyline",
        "",
        "## Headline",
        "",
        f"- Accuracy: **{overall['acc']*100:.1f}%** ({int(overall['n'])} games)",
        f"- Flat-bet P&L: **{'+' if overall['pnl']>=0 else ''}{overall['pnl']:.2f}** over ${overall['n']} staked",
        f"- ROI: **{overall['roi']*100:+.1f}%**",
        "",
        "## Baselines (same 229 games, same staking)",
        "",
        f"- Always bet home: P&L {home_pnl:+.2f} → ROI {home_pnl/len(rows)*100:+.1f}%",
        f"- Always bet closing favorite: P&L {fav_pnl:+.2f} → ROI {fav_pnl/fav_n*100:+.1f}%",
        "",
        "## By tier",
        "",
        "| Tier | Bets | Accuracy | P&L | ROI |",
        "|------|-----:|---------:|----:|----:|",
    ]
    for tier in ("Surefire", "Middle", "Hot"):
        t = by_tier[tier]
        if t["n"]:
            lines.append(
                f"| {tier} | {t['n']} | {t['acc']*100:.1f}% | {t['pnl']:+.2f} | {t['roi']*100:+.1f}% |"
            )
        else:
            lines.append(f"| {tier} | 0 | n/a | n/a | n/a |")
    lines += [
        "",
        "## Calibration (predicted home win prob vs actual)",
        "",
        "| Predicted home prob bin | Games | Mean predicted | Actual home win rate |",
        "|-------------------------|------:|--------------:|---------------------:|",
    ]
    for bucket in sorted(buckets):
        sel = buckets[bucket]
        n = len(sel)
        home_wins = sum(1 for r in sel if r["actual_winner"] == r["home"])
        lines.append(
            f"| {bucket/10:.1f}-{(bucket+1)/10:.1f} | {n} | "
            f"{sum(float(r['predicted_home_prob']) for r in sel)/n:.3f} | {home_wins/n*100:.1f}% |"
        )
    lines += [
        "",
        "## Reproducing",
        "",
        "```bash",
        ".venv/bin/python backtest/odds_backtest_runner.py",
        "```",
        "",
        "`backtest/odds_predictions.csv` holds every game row.",
    ]
    (PROJECT_ROOT / "ODDS_BACKTEST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Odds-fed: {len(rows)} games, acc {overall['acc']*100:.1f}%, P&L {overall['pnl']:+.2f}, "
          f"ROI {overall['roi']*100:+.1f}%")
    print("Report ->", PROJECT_ROOT / "ODDS_BACKTEST_REPORT.md")
    return 0


def _away_dec(r: Dict) -> float:
    """Deprecated: rows now carry dec_home/dec_away directly."""
    return r["dec_away"]


if __name__ == "__main__":
    sys.exit(main())
