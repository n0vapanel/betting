# MULTI-SPORT PREDICTION BRAIN — Backtest Board (v2, accuracy-max wave)

Generated 2026-10-24 (v2 wave). Every number below was independently recomputed by the coordinator from the per-game prediction CSVs (own odds math, own accuracy/pnl loops, own bootstrap where marked) — not taken from agent self-reports. Protocol everywhere: strict walk-forward, no-lookahead, settled at closing odds, flat $1 stakes, edge gates on model-vs-devigged-market probability, date/week-clustered bootstrap CIs.

## FINAL SCOREBOARD (verified)

| Sport | Test window | n | Pick acc | Flat ROI | Market acc | Verdict |
|---|---|---|---|---|---|---|
| **MMA/UFC v2** | 2020-2026 | 2,843 | **80.2%** | **+40.3%** (CI +36..+44) | 67.4% | **REAL EDGE** |
| Tennis v2 | 2022 | 2,538 | 67.5% | -5.2% | 68.1% | efficient |
| NBA v3-ext | Jan-Mar 2022 | 623 | 66.0% | -4.3% (v3-only -0.3%) | — | efficient |
| NFL (Devin) | 2021w15-2022 | 335 | 64.6% | +2.7% (CI straddles 0) | 66.1% | not significant |
| NHL v2 | Mar-Jun 2022 | 555 | 60.4% | +1.2% (CI straddles 0) | 64.9% | noise |
| Soccer v2 1X2 | Mar-May 2022 | 571 | 55.2% (market 54.8%) | +1.1% | 54.8% | not significant |
| Soccer v2 O/U 2.5 | Mar-May 2022 | 571 | **59.7%** | **+4.0% / +6.9% / +28.7%** (eps 0/.02/.05) | 57.4% | promising, CI not yet clean |
| MLB v2 | Apr-Jun 2022 | 1,099 | 56.1% | -3.3% | 60.0% | market beats model |

Every sport now has ≥3 months of backdated games (MMA 7 years, tennis 12 months, NFL 13 months, NHL 4, NBA/MLB/soccer 3).

## MMA v2 — the crown jewel (verified to the digit)

- Upgraded from v1: yearly walk-forward refits, XGBoost added, height/reach/age/country/stance/opponent-style/recent-3 features. Per-class models tried and honestly rejected (OOF logloss worsens).
- Frozen-train discipline maintained; all features from fights strictly before the bout.
- **Verified by coordinator**: acc 80.2%, flat ROI +40.3% (2,843 fights), Brier 0.1368 vs market 0.2072, day-clustered 1000-resample bootstrap CI [+36.1%, +44.4%] — entirely above zero. Positive every year 2020-2026 (2025: +48%).
- Recommended betting pair (LGBM+XGB): acc 80.4%, ROI +43.4%, Brier 0.1273.
- **UFC picker built**: `backtest/multisport/mma/ufc_picker.py` — feed tonight's card CSV (date, fighters, class, American odds) → feature build with cutoff, frozen-model probs, de-vigged edge, flat + quarter-Kelly stakes. Smoke-tested exit 0 on a real 2026 card (Fiziev +17.6% edge → 8.7% Kelly).
- Odds source for upcoming cards: bestfightodds.com event pages scrape cleanly with curl+UA (documented in PICKER_README.md); ufc.com API dead; tapology Cloudflare-blocked.
- Residual risk (unchanged): mdabbert odds are "closing per dataset docs". **Gate to real money = paper-trade live cards for a few weeks.**

## What the accuracy-max wave proved

- **More features ≈ +accuracy everywhere but ROI only in MMA.** Soccer O/U flipped from negative-everywhere to positive-everywhere (+7pp accuracy from xG/CLV/open-market features — ablation-verified). MMA jumped 77.1→80.2% acc and +27.6→+40.3% ROI. NBA/NHL/MLB/tennis added injuries/goalies/starters/park factors/xG — accuracy ticked up ~1pp, ROI stayed ~zero: those closing lines are efficient, full stop.
- **Honest ceilings**: favorites-only markets (NFL/NBA/NHL/MLB/tennis) cap out at market accuracy (60-68%) and won't clear vig. The soft market (UFC) is where the brain's money lives.
- Tennis lane caught its own h2h leak (constant feature encoding the target → fantasy 78%/+37%; fixed + lesson saved). Everything green here was re-verified independent of the builders.

## Structure

- `backtest/multisport/<sport>/` — v1 lanes; `<sport>_v2/` — accuracy-max lanes; each has REPORT.md + predictions CSV + code. `backtest/v3_extended/` = NBA 3-month. This file = the board.
- Shared venv `~/.hermes/cache/scratch/msports-venv` (py3.11). Raw data gitignored; download recipes in each report.
- Commits: `7d8b0ef` (v1 board), `f24f8fa` (accuracy-max wave), `fa491c7` (NFL).

## Next

1. **Paper-trade UFC live** (picker + real cards, tracked ledger) — the only result that converts to money.
2. Soccer O/U follow-up: 2 more seasons (2022-23, 2023-24) to see if the +28.7% gate is real or window luck.
3. Optional: NHL goalie-first model (v2 goalie features helped; CIs still straddle 0).
