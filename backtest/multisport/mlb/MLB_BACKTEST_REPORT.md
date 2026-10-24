# MLB Walk-Forward Betting Backtest — May–Jun 2022

**Generated:** 2026-10-24
**Directory:** `backtest/multisport/mlb/`
**Protocol:** mirrors the NBA v3 harness (`backtest/v3/`) — strictly no-lookahead rolling features,
expanding-window walk-forward training, time-ordered CV, isotonic/Platt calibration, and the v3
betting-evaluation module (flat $1 bets, lean/winner edge gates, quarter-Kelly, date-clustered
bootstrap).

---

## 1. Verdict

**The model does not beat the market.**

| Metric (test: May 1 – Jun 30 2022) | Model | Market (closing ML consensus) |
|---|---|---|
| Games | 824 (786 with odds) | 786 |
| Accuracy (flat, predicted side) | **54.85%** (824 g) / 54.96% (786 g) | **59.03%** |
| Flat $1 ROI (predicted winner @ closing ML) | **−4.20%** | −3.24% (betting market favorite) |
| Brier score | **0.2471** | **0.2388** |
| Log loss | 0.6875 | 0.6701 |

The market beat the model on every dimension. The Brier difference is statistically significant:
bootstrap 95% CI (cluster by date, 1000 resamples) for model-minus-market Brier =
**[+0.0025, +0.0141]** (positive = model worse), excluding zero. The model's flat ROI
(−4.2%) has bootstrap CI [−0.096, +0.013]; it cannot be distinguished from the vig-bleed of the
market's own favorite strategy (−3.2%) and is nowhere near breakeven.

**Conclusion:** an Elo + rolling team/pitcher feature model built solely from free game-log data
predicts MLB winners at ~55% accuracy but is *under-calibrated and under-spread* relative to
closing lines: it compresses probabilities toward the ~54% home base rate, predicts the home team
~90% of the time (actual home rate in the window: 51.1%), and cannot identify heavy favorites
(market max p ≈ 0.81 vs model max p ≈ 0.71). All edge gates selected on a held-out validation
slice regressed to negative or statistically indistinguishable-from-zero ROI on test. **No
profitable betting strategy was found; the model adds no value over closing moneyline odds.**

---

## 2. Data

### 2.1 Game logs (no auth, verified)
- Retrosheet game logs `gl2021.txt` (2,429 games) and `gl2022.txt` (2,430 games),
  downloaded from retrosheet.org. Parsed with the official 161-field layout (`glfields.txt`).
- Per game: date, teams, scores, park, day/night, line scores, full box-score aggregates
  (AB/H/2B/3B/HR/RBI/SH/SF/HBP/BB/IBB/SO/SB/CS/GIDP/LOB), pitchers used, fielding stats,
  **winning/losing/saving pitchers**, **starting pitchers (both sides)**, managers, umpires.
- Cleaned: dropped forfeits and games without complete acquisition info; 30 unique teams;
  overall home-win rate 53.6% (sanity ✓).

### 2.2 Closing moneyline odds (free, GitHub)
- Source: `ArnavSaraogi/mlb-odds-scraper` release asset `mlb_odds_dataset.json`
  (80 MB, SportsBookReview scrape, 2021-04-01 → 2025-08-16).
- Closing `currentLine` moneyline from up to 6 books (FanDuel, DraftKings, Caesars, Bet365,
  BetMGM, BetRivers-NY).
- De-vigging: per-book implied probabilities normalized, then averaged across books
  → `p_market_home`. Decimals averaged across books for payout math.
- Join to Retrosheet by **date + home + away team (standard codes), verified by final score**
  (every join must reproduce the box-score result; doubleheaders matched sequentially by score).
- Coverage: 4,676 of 4,859 games joined (96.2%). Test window: **786 of 824 (95.4%)**; the 38
  unmatched test games are almost all doubleheader games that SBR lists once per day.

### 2.3 Note on season timing
The 2022 season started Apr 7 (lockout). "Mar–Apr 2022" training data is therefore
Apr 7 – Apr 30, 2022. No 2022 spring-training games were used.

---

## 3. Features (strictly no-lookahead)

Single chronological pass over all games; features for game *G* use only games completed
before *G* (same-day DH game 1 may inform game 2 — realistic and disclosed). All state is
updated *after* the feature row for *G* is written.

**Team level (home/away, plus home-minus-away diffs):**
- **Elo**: 538 MLB specification (K=4, home-field +24 baked into the rating update,
  margin multiplier ln(|M|+1)·2.2/((|Δ|+275)/1000)). Features use the raw rating diff
  (no HFA pre-baked; models learn home field from data).
- Runs scored/allowed per game: season-to-date, last-10, exponentially weighted (half-life ≈ 12 g).
- Win %, Pythagorean, games played.
- **Offense/pitching quality rates** (from box-score aggregates): season OBP, SLG,
  team strikeout rate and walk rate per 9 IP.
- **Bullpen fatigue**: team innings pitched over last 3 days (exact from game outs),
  extra-inning games in last 3 days, and reliever win/loss share over last 14 days
  (W/L by non-starters, exact from pitcher-of-record vs starter IDs).
- **Rest**: days since last game (both teams).
- Day/night and doubleheader flags; park ID (categorical, LightGBM only).

**Starting pitcher level:**
- Experience (season+prior starts), team win % in his starts, opponent runs per start
  (season-to-date), days since his last start. Sign conventions: `sp_ra_diff` = away SP RA − home
  SP RA so that higher = home SP advantage.

NaN handling: early-season all-NaN columns imputed with 0 in the LogReg pipeline
(constant imputation + standardization); LightGBM consumes NaNs natively.

**Leakage audit:** features are written before any state update in the same pass; SP/team/Elo
state is only mutated after the row is emitted. No box-score, odds, or outcome information from
game *G* or later is available to the features of game *G*.

---

## 4. Walk-forward protocol

| Phase | Train | Predict | Purpose |
|---|---|---|---|
| Tune | 2021-04-01 → 2022-04-15 | OOF (TimeSeriesSplit, 5 folds) | hyperparameters + calibration choice |
| Phase 1 | 2021-04-01 → 2022-04-15 | **Validation 2022-04-16 → 04-30** | gate selection |
| Phase 2 | 2021-04-01 → 2022-04-30 | **Test 2022-05-01 → 06-30** | final evaluation |

- **Models:** LogisticRegression (C grid) + LightGBM (leaves/lr grid), equal-data ensembled
  with tuned weight `w_lgb`. Selected config (OOF Brier on tune window): `lr_c=0.03`,
  `lgb(num_leaves=63, lr=0.02)`, `w_lgb=0.4`, **isotonic calibration** on OOF ensemble
  probabilities.
- Phase-2 training CV diagnostics (OOF, n=2,285): ensemble Brier 0.2484
  (LR alone 0.2470, LGBM alone 0.2700); log loss 0.6907.
- Calibration map (isotonic on OOF) is applied to validation and test predictions; validation
  probabilities used **only** for gate selection.

## 5. Betting evaluation (v3 module, adapted to MLB)

- **Flat:** $1 on the model's predicted winner at the consensus closing decimal.
- **Lean gate:** bet when |p − p_market| ≥ eps on the lean side.
- **Winner-conviction gate:** bet when edge on the model's predicted side ≥ eps.
- eps grid: {0, 0.01, 0.02, 0.03, 0.05}; gate chosen on validation (Apr 16–30, odds subset,
  min 30 bets): **winner, eps = 0.03** (80 bets, +8.3% ROI in validation).
- **Quarter-Kelly:** f = 0.25·(p·d − 1)/(d − 1), capped at 5% per bet, daily grouping,
  no intraday recycling.
- **Bootstrap:** 1,000 resamples of whole calendar dates (cluster by date), percentile CI.

### Test results (May–Jun 2022)

**Selected gate on test: winner eps=0.03 → 309 bets, win rate 47.9%, ROI −2.8%**
(bootstrap 95% CI for gate ROI: [−0.119, +0.069]). Validation's +8.3% did not survive
out-of-sample — a clean example of gate-selection noise.

| Test gate table (786 games) | n | win rate | ROI |
|---|---|---|---|
| lean eps 0.00 | 786 | 0.430 | −7.25% |
| lean eps 0.05 | 417 | 0.393 | −8.28% |
| winner eps 0.00 | 424 | 0.481 | −6.61% |
| **winner eps 0.03 (chosen)** | **309** | **0.479** | **−2.83%** |
| winner eps 0.05 | 238 | 0.445 | −4.73% |

**Quarter-Kelly (chosen gate):** bankroll multiple **0.691** (net −30.9% over the window),
ROI on staked −3.49%, max daily drawdown 52.7%. The policy loses money.

**Bootstrap 95% CIs (test, odds subset):**

| Quantity | CI |
|---|---|
| Flat accuracy | [0.519, 0.580] |
| Flat ROI | [−0.096, +0.013] |
| Gate ROI (winner eps 0.03) | [−0.119, +0.069] |
| Model Brier | [0.2418, 0.2520] |
| Market Brier | [0.2316, 0.2460] |
| Brier minus market | [+0.0025, +0.0141] ← market significantly better |

## 6. Why the model loses

1. **Compressed probabilities.** Model p-home distribution (std 0.056, max 0.71) is far narrower
   than the market's (std 0.097, max 0.81). The model cannot grade true mismatches (e.g.,
   ace-vs-replacement), because it has no per-pitcher talent priors beyond box-score aggregates.
2. **Structural home bias in a hostile window.** The model picked the home team in ~90% of test
   games; actual home win rate May–Jun 2022 was 51.1% (2021–22 overall: 53.6%). The market
   adapts its home-favorite rate continuously; the model's base rate lags.
3. **The market is strong.** Closing-line Brier 0.2388 and favorite accuracy 59% on this window
   are consistent with the literature: MLB closing moneylines are near-efficient, and
   line-only public features rarely yield exploitable edges.

## 7. Limitations

- Odds are consensus *closing* lines from one scrape (SportsBookReview); 4.6% of test games
  (mostly doubleheaders) have no odds and are excluded from betting metrics but included in
  accuracy/Brier counts.
- Features exclude play-by-play/Statcast data (per-pitcher IP, pitch counts, player-level
  projections, injuries, weather) — the bullpen-fatigue proxy uses team innings and reliever
  decision shares.
- Single test window (May–Jun 2022): 824 games is small for betting-edge detection; CIs are wide.
- Training on 2021+Apr 2022 only (≈2,700 games) limits model capacity.

## 8. Artifacts

| File | Content |
|---|---|
| `mlb_predictions.csv` | 824 test rows: date, teams, p_model_home, p_market_home, predicted side, American odds of side, correct flag, $1 PnL, decimal, source |
| `MLB_BACKTEST_REPORT.md` | this report |
| `backtest_results.json` | full metrics, gate tables, Kelly, bootstrap CIs, config |
| `parse_gamelogs.py`, `build_odds.py`, `features.py`, `run_backtest.py`, `main_backtest.py` | reproducible pipeline |
| `games_all.parquet`, `odds_closing.parquet`, `features.parquet` | intermediate data |

**Verdict: NO EDGE. The model does not beat the market; flat ROI −4.2% vs market −3.2%,
Brier worse by +0.0086 (CI excludes 0), selected gate ROI −2.8% with CI covering zero.**
