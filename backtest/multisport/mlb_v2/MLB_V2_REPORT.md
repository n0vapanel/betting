# MLB v2 Betting Backtest — Apr–Jun 2022 (3-month walk-forward, upgraded features)

**Generated:** 2026-10-24 | **Directory:** `backtest/multisport/mlb_v2/`
**Protocol:** strict walk-forward, no-lookahead rolling features, expanding-window training,
time-ordered CV, isotonic calibration, flat $1 bets at closing ML, edge gates selected on a
held-out 2021 validation window, quarter-Kelly, date-clustered bootstrap, Brier vs market.

---

## 1. Verdict

**v2 does not beat v1, and neither beats the market.** The upgrades (starting-pitcher quality
from Statcast, bullpen fatigue, park factors, rest/travel, lineup OPS) were all built and
validated; a tune-window-only ablation kept the Statcast SP block, which the LightGBM uses
heavily (12 of its top-18 features), but out-of-sample accuracy, Brier and ROI are statistically
indistinguishable from the v1 baseline, and the model remains clearly worse than the closing
moneyline.

| Metric (test: Apr 1 – Jun 30, 2022) | v1 original (May–Jun only) | v1 rerun (same 3-mo window) | **v2 final** | Market |
|---|---|---|---|---|
| Games (odds subset) | 786 | 1,099 | **1,099** | 1,099 |
| Accuracy, predicted side | 54.96% | 55.78% | **56.05%** | 59.96% |
| Flat $1 ROI @ closing ML | −4.20% | −2.58% | **−3.30%** | −1.37% |
| Brier score | 0.2471 | 0.24509 | **0.24554** | 0.23844 |
| Log loss | 0.6875 | 0.68337 | **0.68421** | 0.66958 |
| 2021 OOF Brier (training) | — | 0.24927 | **0.24880** | — |

v2 accuracy (56.05%) is +0.27 pp over the like-for-like v1 rerun and +1.1 pp over v1's original
2-month figure, but the bootstrap 95% CI for v2 accuracy is **[0.535, 0.587]** — v1's rerun point
estimate lies inside it. v2 flat ROI (−3.30%) is *worse* than the v1 rerun (−2.58%) and its CI
[−0.079, +0.013] covers both. Model-minus-market Brier on the odds subset:
**+0.0071, bootstrap 95% CI [+0.0019, +0.0121]** — the market is significantly better.

**Conclusion: the accuracy push failed honestly. No profitable betting strategy was found; the
upgraded model adds no value over closing moneyline odds.**

---

## 2. Walk-forward protocol (strict)

| Phase | Train | Predict | Purpose |
|---|---|---|---|
| Tune | 2021 full season (2,429 g) | OOF via TimeSeriesSplit(5) | hyperparams + calibration choice |
| Phase 1 | 2021-04-01 → 2021-08-15 | **Validation 2021-08-16 → 09-30** (573 g w/ odds) | gate selection |
| Phase 2 | 2021 + Mar 2022 | **Test 2022-04-01 → 06-30** (1,141 g; 1,099 w/ odds) | final evaluation |

- "Mar 2022" contains zero games (2022 lockout; season began Apr 7) — training is the full 2021
  regular season, the same size as v1's phase-2 training set minus April 2022.
- Models: LogisticRegression + LightGBM ensemble, tuned weight `w_lgb=0.4`, isotonic calibration
  on OOF. Tuning config identical to v1 (`lr_c=0.03`, `lgb(num_leaves=63, lr=0.02)`), re-tuned
  independently on the v2 feature set.
- Gate selection moved to a held-out late-2021 window (the old Apr-2022 validation slice is now
  inside the 3-month test window and unusable). **Every gate had negative validation ROI** in
  Aug–Sep 2021 (worst for both models); the protocol picked the least-bad (v2: winner-edge
  eps=0.02, 247 val bets, −24.8% val ROI). On test it again lost (488 bets, ROI bootstrap CI
  [−0.123, +0.036]); quarter-Kelly on that gate ended at bankroll multiple **0.280 (−72%)**.
  Same degeneration as v1 — documented, not hidden.
- Bootstrap: 1,000 resamples of whole calendar dates, percentile CI (85 test dates).
- Same odds source as v1 (below), joined by date+teams and **verified against the final score**.

## 3. Data sources

- **Game logs:** Retrosheet `gl2021.txt`/`gl2022.txt` (161-field layout; scores, starters,
  W/L pitchers, pitchers-used, box-score aggregates, linescores). 4,859 regular-season games.
- **Closing odds:** SportsBookReview scrape via `ArnavSaraogi/mlb-odds-scraper` release asset —
  identical to v1 (consensus of up to 6 books, de-vigged per book then averaged; decimals
  averaged). Test-window coverage 1,099/1,141 (96.3%); misses are mostly doubleheaders.
- **Statcast (new):** Baseball Savant pitch-level CSV, downloaded 2021-03-15 → 2022-06-30 in
  4-day chunks (savant caps 25k rows/query and its date filters are inclusive, so boundary-day
  games were de-duplicated), aggregated to 31,441 pitcher-game rows over all 3,570 scheduled
  games. MLB StatsAPI schedule (Final games only, start-time ordering) links game logs to
  `game_pk`; venue-swapped/make-up doubleheaders bucket-matched by date+team-set (e.g. the
  2022-05-10 OAK@DET DH). Join coverage: 2021 2,427/2,429; 2022 test window 1,139/1,141.
- **Park factors (new):** FanGraphs "Guts" park-factor page, season 2021 (5yr Basic, 3yr, HR),
  https://www.fangraphs.com/guts.aspx?type=pf&teamid=0&season=2021 — published before the 2022
  season, so using 2021 factors for 2022 features is lookahead-free. Sanity: Coors 115, Petco 94.

## 4. Features (all upgrades built; final set chosen by tune-window ablation)

Single chronological pass; features for game G use only games before G (same-day DH game 1 may
inform game 2). All state updated after the feature row is emitted.

**v1 features (kept):** Elo (538 spec), runs scored/allowed season-to-date/L10/EW, win%,
Pythagorean-equivalents, OBP/SLG/K%/BB% from box aggregates, team innings last 3 days,
extra-inning count, reliever W/L share 14d, rest days, day/night, DH flag, park ID (LGBM),
retro-based SP features (experience, team win% in his starts, team RA per start, days rest).

**New in v2 (all strictly no-lookahead):**
- **(b) Starting-pitcher quality** — from Statcast per-start IP, K, BB, HR, velo, batters faced
  + team RA in his start (game log). Rolled season-to-date and last-10: IP/start, K/9, BB/9,
  HR/9, avg velo, RA/start, starts, days rest. SP identity = actual starter (first pitcher of
  the top/bottom 1st). *Kept in final model.*
- **(c) Bullpen fatigue** — relief IP, pitches, K−BB, appearances per team over the previous
  3 days (Statcast, exact). *Computed; excluded by tune-window ablation (hurt OOF).*
- **(d) Park factors** — pf Basic(5yr)/3yr/HR for the home park (FanGraphs 2021). *Computed;
  excluded by ablation; park ID itself remains via v1.*
- **(e) Rest/travel** — rest days (v1) + away-team timezone shift (east = positive) and
  cumulative 7-day eastward shift (static team-TZ table, DST-aware). *Computed; excluded by ablation.*
- **(f) Lineup quality** — team OPS season-to-date and L10, wOBA approximation, exact from game
  logs (pybaseball `team_batting` is full-season and would leak; game-log OPS is the same signal,
  no lookahead). *Computed; excluded by ablation (v1's OBP/SLG already carry most of it).*

**Feature selection:** ablation of feature blocks on the 2021 tune window ONLY
(TimeSeriesSplit OOF Brier, fixed tuned config, isotonic-calibrated):

| Variant | 2021 OOF Brier |
|---|---|
| **F: v1 + Statcast SP (chosen)** | **0.24332** |
| A: v1 + everything | 0.24370 |
| E: v1 + park factors | 0.24379 |
| G: v1 + bullpen block | 0.24477 |
| B/C/D: v1 + non-SP combos | 0.24508–0.24520 |

F chosen; the F/A/E spread (0.0005) is within noise — the honest reading is "no block adds
materially". Decision used 2021 data only; the test set was touched once, after selection.

**Documented approximations:**
1. **SP runs allowed** = team RA in his starts (game-log). Retrosheet's individual-Earned-Runs
   list omits zero-ER pitchers (verified: list length equals pitchers-used in only 26/2,429
   games), so it cannot be attributed to starters.
2. **IP derived from out-events** (not `outs_when_up` deltas). Validated against Retrosheet
   LengthInOuts: median total-outs error −1 per game (~2%), 94% of games within ±2 — a small,
   uniform undercount, immaterial for ratio features.
3. **Park factors are 2021 values** applied to 2022 (2022 factors are published after the
   season — using them would leak). 2021 Blue Jays (Dunedin/Buffalo/Toronto) are all EDT, so a
   single timezone entry is exact for this data.
4. Odds are consensus *closing* lines from one scrape; ~3.7% of test games (mostly DH) lack odds.

## 5. Why v2 still loses (and didn't move the needle)

1. **The new features are real but the market already has them.** SP quality/velocity, bullpen
   workload and park factors are line-relevant public information; LightGBM leans on them
   (top importances: park, SP IP/K/BB/HR/velo st-d and L10), yet test Brier is unchanged
   (0.24554 vs 0.24509). The closing line is near-efficient on this window (market Brier
   0.23844, favorite accuracy 59.96%).
2. **Still compressed probabilities.** v2 p-home std 0.071, max 0.68 vs market std 0.095,
   max ~0.81 — the model still cannot grade true mismatches. It did break v1's home bias
   (predicted-home share 69% vs v1's ~90%; actual in-window home win rate 51.5%), but that
   reallocation didn't improve outcomes.
3. **Small data, hostile window.** 1,099 test games; the Aug–Sep 2021 validation window was
   uniformly brutal for every gate (market Brier there 0.1955 — favorites were hot), so gate
   selection was noise, exactly as in v1.
4. Accuracy differences vs baseline (+0.27 pp) are inside the bootstrap CI — not evidence of
   signal, especially after tune-window feature selection (mild selection bias inflates even this).

## 6. Artifacts

| File | Content |
|---|---|
| `mlb_v2_predictions.csv` | 1,141 test rows: date, teams, p_model_home, p_market_home, predicted side, American odds, correct, $1 PnL, decimal, source |
| `MLB_V2_REPORT.md` | this report |
| `backtest_v2.json`, `backtest_v1.json` | full metrics, gate tables, Kelly, bootstrap CIs, config (v2 final + v1 rerun on the same window) |
| `features_v2.py`, `features_v2.parquet` | rolling feature pipeline + data (152 cols; includes unused blocks) |
| `fetch_statcast.py`, `pitcher_game_stats.parquet` | Savant pitch download + per-pitcher-game aggregates |
| `fetch_schedule.py`, `schedule_gamepk.parquet` | MLB schedule → game_pk mapping |
| `park_factors_fg_2021.csv` | FanGraphs park factors (cited) |
| `run_backtest_v2.py`, `ablate_v2.py` | walk-forward runner (v1/v2 modes) + tune-window ablation |

## 7. Final key numbers

- **v2 final: 56.05% accuracy, −3.30% flat ROI, Brier 0.24554** (1,099 games w/ odds; 56.18% on all 1,141).
- **v1 like-for-like rerun: 55.78%, −2.58%, 0.24509.** v1 original 2-month: 54.96%, −4.20%, 0.2471.
- **Market: 59.96%, −1.37%, 0.23844.** Model-minus-market Brier +0.0071, CI [+0.0019, +0.0121] → market significantly better.
- Chosen gate (winner eps 0.02): 488 test bets, ROI CI [−0.123, +0.036]; quarter-Kelly bankroll 0.280.
- **Verdict: NO EDGE — v2 statistically ties the v1 baseline and loses to the closing line.**
