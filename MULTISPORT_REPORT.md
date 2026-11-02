# MULTI-SPORT PREDICTION BRAIN — Backtest Board (v2, accuracy-max wave)

> **Correction (v3 learning-loop review, 2026-10-28): two of this board's "edges" are leakage, and once both are
> fixed nothing here beats the closing market.** The research is kept below as it was written; read it with these notes.
>
> 1. **MMA v2 (80.2% accuracy, +40.3% ROI) is target leakage.** `streak_pre` is forced to 0 in every fight the fighter
>    *loses* (`backtest/multisport/mma_v2/build_features_v2.py:213-217`; the same code is in `mma_v2/build_features.py:161-163`
>    and `mma/feature_utils.py:176-180`), and the `ffill()` isn't grouped by fighter, so "streak > 0" means "won this
>    fight". Only the tree models exploit it (LightGBM/XGBoost reach 80%, AUC 0.89). The clean baseline is logistic
>    regression alone on the same 2,843 fights: accuracy 63.6% against the market's 67.4%, Brier 0.2249 against 0.2072,
>    log loss 0.641 against 0.601 (Brier change +0.018, 95% range +0.013 to +0.023): **worse than the market.** The
>    picker's large edges (e.g. +17.6%) are artefacts of the same leak; do not bet them.
> 2. **Soccer v2's O/U gains (+4.0% / +6.9% / +28.7%) come from mixing up football-data.co.uk's odds columns.** The plain
>    columns (`AvgH`, `Avg>2.5`) are *pre-closing* prices (Friday/Tuesday afternoons); the `C` columns (`AvgCH`,
>    `AvgC>2.5`) are the *close*. `soccer_v2/backtest.py:115-121,165` used the pre-close as "closing" (feature and
>    settlement) and `:123-153,171-182` fed the true close in as "opening"/CLV features, so the model saw the real close
>    and bet at the earlier price. Betting the side the close moved toward, at the earlier price, makes +7.7% on O/U
>    (810 bets) and +6.6% on 1X2 (677 bets) on 2021-22 with no model at all: the same size as the reported gates.
>    Against the **true close** on the same 571 test matches the model has no edge: O/U Brier 0.2393 against 0.2377,
>    log loss 0.672 against 0.668 (Brier change +0.002, 95% range -0.002 to +0.005); home win Brier 0.2146 against
>    0.2112 (+0.003, -0.001 to +0.007).
>
> Clean lanes, recomputed from the committed prediction CSVs (model minus de-vigged close, Brier, date- or week-clustered
> 95% range): NBA v3-ext +0.004 (-0.002 to +0.010), MLB v2 +0.007 (+0.002 to +0.013), NHL v2 +0.013 (+0.003 to +0.025),
> NFL +0.014 (+0.003 to +0.024), tennis v2 +0.003 (+0.001 to +0.006). **The market wins in every lane.** The app's brain
> now leans on the market and tunes itself only behind statistical gates, judged walk-forward against the sharp close
> (`telegram_betting_bot_simple/learn/`, the Brain progress page); see `brain/README.md`.

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

> **Correction (2026-10-28):** the Soccer v2 O/U row (and the "soccer O/U flipped" line below) used football-data's
> pre-closing columns as the close and the true closing (`C`) columns as features. Against the true close the model has
> no edge (O/U Brier 0.2393 against 0.2377; change +0.002, 95% range -0.002 to +0.005). See the correction at the top.

Every sport now has ≥3 months of backdated games (MMA 7 years, tennis 12 months, NFL 13 months, NHL 4, NBA/MLB/soccer 3).

## MMA v2 — the crown jewel (verified to the digit)

> **Correction (2026-10-28):** these numbers come from a target leak (`streak_pre` is 0 in every fight the fighter loses,
> and the forward fill isn't grouped by fighter). The clean baseline, logistic regression alone, is worse than the
> market: accuracy 63.6% against 67.4%, Brier 0.2249 against 0.2072. See the correction at the top of this file.

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
