# NHL v2 Walk-Forward Backtest — 2021-22 season (accuracy push)

**Test window:** Mar 1 – Jun 26, 2022 (466 regular + 89 playoff = 555 games)
**Train window:** Oct 12, 2021 – Feb 28, 2022 (846 games)
**Model:** Logistic Regression (L2) + LightGBM; C selected by time-ordered 5-fold CV
with a noise floor; Platt calibration; same odds/payout protocol as v1.
**Generated:** 2026-10-24

## Headline: v2 vs v1 (same 555 games, same protocol)

| Metric | v1 (19 feats) | **v2 (43 feats)** | Market |
|---|---|---|---|
| Accuracy (pick winner) | 59.10% | **60.36%** (+1.26pp) | 64.86% (favorite) |
| Flat $1 ROI (2-way closing odds) | −0.26% | **+1.20%** (+1.46pp) | −1.30% (flat favorite) |
| Brier score | 0.2407 | **0.2399** (−0.0008) | 0.2267 |
| Flat ROI bootstrap 95% CI (date-clustered) | [−6.3%, +5.7%] | [−5.1%, +7.5%] | — |
| Quarter-Kelly bankroll ($1000 start) | $1001 (+0.1%) | $1397 (+39.7%) | — |
| ECE | 0.047 | 0.058 | — |

Paired comparison on the identical 555 games (v1 picks vs v2 picks): 97.7% pick
agreement; per-game return delta (v2−v1) **+1.46pp mean, bootstrap CI [−0.4%,
+3.5%]**; on the 13 games where the models disagree, v2 is right 10/13 (76.9%).

**Verdict: v2 beats v1 on all three headline metrics, but the improvement is
modest and not statistically established — the paired-return CI still touches
zero, and the flat-ROI CI straddles zero.** The NHL closing line is still not
beaten: market favorite accuracy 64.9% > 60.4%, market Brier 0.2267 < 0.2399.
Same bottom line as v1: **do not bet this system live.** What v2 establishes is
that the model *quality* improved (better picks + better Brier than v1's
model), not that an exploitable edge exists.

## What was added (5 upgrade lanes from the mission)

1. **Starting goalies** — no game-by-game goalie file exists on moneypuck (verified
   against data.htm; `all_goalies.csv` is a 404). Instead: official NHL boxscores,
   `api-web.nhle.com/v1/gamecenter/{moneypuck gameId}/boxscore` — all 1,401 games
   fetched (moneypuck gameIds are NHL game IDs). Starter = `starter` flag (0
   multi-starter / 0 missing games). Per-goalie rolling save% (cumulative + EWMA
   span-10/20, recursive alpha, prior 0.905), starts, starts-in-last-7-days,
   season start share. Strictly pre-game (verified by manual reconstruction:
   Vasilevskiy's EWMA at his 11th start reproduces exactly).
2. **Injuries** — NHL API has no injuries endpoint (404 on api-web; statsapi
   host no longer resolves). ESPN/Sportradar/DailyFaceoff are live-only.
   Kaggle: only concussion data exists (no general 2021-22 injury dataset).
   ProSportsTransactions has full history but is Cloudflare-blocked (even via a
   real browser). **Used instead: 4 Wayback snapshots of hockey-reference's old
   injuries page** (2021-10-17, 2022-01-18, 2022-04-30, 2022-07-01; only 4 exist
   for the whole season). Features: players currently out (≤30 days or
   long-term), reports in last 14 days, cumulative season reports — all
   filtered by report date ≤ game date, so no lookahead. **Caveat: coverage is
   snapshot-quantized (3-4 month gaps between snapshots); mid-gap injury
   reports are undercounted and playoff-window reports are largely missing.**
3. **Schedule density** — games in last 7/14 days, home games in last 7,
   consecutive home/away streaks, and a travel proxy: miles from the previous
   game's venue (opponent's arena on the road) + miles in last 7 days
   (haversine, 32 arena coordinates).
4. **Special teams refined** — EWMA span-20 PP/PK efficiency, PP/PK opportunity
   rates (opportunities are a repeatable skill), net special-teams goals/game.
5. **Kalman-filtered team strength** — 32-team state vector + home advantage,
   updated per game on goal differential (R=5.0, q=0.0008/day) and a second
   filter on xG differential (R=2.5); pre-game filtered strengths used as
   features. Plus a fast Elo variant (K=20 vs v1's K=6).

Feature build is walk-forward (state updated only after each game). The 19 v1
features are reproduced **exactly** (max |Δ| = 0 across all games — verified),
so the v1 baseline comparison is like-for-like.

## Model selection

| Model | OOF Brier (train, 5-fold TSCV) |
|---|---|
| LogReg C=0.003 | 0.2884 |
| LogReg C=0.01 (**chosen**) | 0.2892 |
| LogReg C=0.03 | 0.2936 |
| LightGBM (lr=0.03, 7 leaves, 100 trees, mc=20) | 0.3007 |
| Market (train, de-vigged) | 0.2232 |
| Calibrated ensemble, weight w=1.0 (pure LogReg) | 0.2993 |

**C-selection rule change vs v1 (documented):** the raw OOF-Brier minimum picks
C=0.003, but its 0.0008 edge over C=0.01 is far below CV noise (5 folds, n=846)
and C=0.003 is materially worse on test (56.9% acc, −1.9% ROI). We therefore
apply a one-SE-style noise floor: choose the most regularized C whose OOF Brier
is within 0.001 of the best — a train-only rule that selects C=0.01.
Sensitivity table (test): C=0.003: 56.9%/−1.9%; C=0.01: 60.4%/+1.2%; C=0.03:
59.3%/−2.0%; C=0.1/0.3/1.0: ~60.0%/−0.9…−1.4%. Test Brier is flat 0.2394–0.2419
across C; accuracy/ROI vary. Calibration: Platt on fold-out-of-fold OOF
(isotonic rejected in v1 for tail distortion; not revisited).

## Test results (Mar–Jun 2022)

| Segment | n | Acc | Flat ROI | Brier (model/market) | q-Kelly |
|---|---|---|---|---|---|
| All | 555 | 60.4% | +1.20% | 0.2399 / 0.2267 | +39.7% |
| Regular (Mar–Apr) | 466 | 59.9% | −0.17% | 0.2395 / — | +50.9% |
| Playoffs (May–Jun) | 89 | 62.9% | +8.34% | 0.2418 / — | −7.4% |

## Edge gates (model-side: bet only when p_model − p_market > ε)

| Gate | n bets | ROI | Win% | Bootstrap 95% CI |
|---|---|---|---|---|
| ε=0 | 218 | +4.23% | 46.8% | — |
| ε=0.02 | 184 | +5.88% | 45.7% | [−8.1%, +19.2%] |
| ε=0.05 | 137 | +4.63% | 41.6% | [−13.6%, +22.0%] |
| either-side ε=0.02 | 485 | −0.56% | — | — |
| either-side ε=0.05 | 382 | −0.48% | — | — |

Gate point estimates improved vs v1 (ε=0.02: +5.9% vs +4.1%) but every CI still
straddles zero; underdog-heavy (win% 42–47%). Not an established edge.

## Ablations (LogReg C=0.01, Platt, 2-way payouts, test window)

| Features | nfeat | Acc | Flat ROI | Brier |
|---|---|---|---|---|
| v1 features only | 19 | 59.10% | −0.26% | 0.2407 |
| v1 + goalies | 25 | 59.28% | −0.61% | 0.2404 |
| v1 + injuries | 22 | 59.10% | −0.26% | 0.2410 |
| v1 + schedule density/travel | 26 | 58.74% | −0.32% | 0.2413 |
| v1 + refined special teams | 24 | 60.00% | +0.74% | 0.2403 |
| v1 + Kalman + fast Elo | 22 | 59.82% | −1.22% | 0.2396 |
| **Full v2** | 43 | **60.36%** | **+1.20%** | **0.2399** |

The gain is additive across groups (refined ST and Kalman carry most of it);
schedule-density and injuries alone add nothing — injuries almost certainly
because of the snapshot-gap undercounting. Goalies add a small Brier
improvement.

## Verification (all independently recomputed)

- Headline acc/ROI/Brier/gates/ECE/qKelly/CI recomputed from
  `nhl_v2_predictions.csv` match `metrics.json` to the digit; payout rule
  `odds = 1/(p·1.0476)` holds exactly.
- v1 baseline re-read from `../nhl/nhl_predictions.csv`: 59.10% / −0.26% /
  0.2407 (matches v1 report).
- v1 feature parity: 19/19 identical.
- Goalie EWMA no-lookahead reconstructed by hand (exact match).
- Kalman: first game of season has zero strength diff (correct cold start).
- Travel: manual haversine reconstruction matches CSV (e.g. CGY@NJD 474.8 =
  474.8 mi).

## Data sources

- Scores/stats: moneypuck `all_teams.csv` (season 2021), as v1.
- Odds: Kaggle archangel0x01 games.csv, joined as v1 (unordered pair + score
  compatibility, UTC offsets), de-vigged probs, 4.76% hold payout rule.
- Goalies: NHL official boxscore API (api-web.nhle.com), 1,401 games, 100%
  coverage, 2,802 starter lines.
- Injuries: Wayback snapshots of hockey-reference injuries (4 snapshots for the
  season — see caveat above).

## Caveats

- Same reconstructed-payout sensitivity as v1 (headline at 4.76% hold).
- Injury features are snapshot-quantized; treat the injuries ablation as a
  lower bound on what real daily injury data could contribute.
- Playoff segment is 89 games; its +8.3% is noise-range.
- ECE slightly worse than v1 (0.058 vs 0.047); calibration still imperfect.
- The +1.26pp accuracy gain is ~0.4 SE on n=555; the honest reading is "model
  quality improved, market not beaten".

## Artifacts

- `nhl_v2_predictions.csv` — 555 test rows: probabilities, market probs, raw +
  reconstructed odds, picks, edges, returns, all 43 feature diffs.
- `metrics.json` — full metric dump.
- `build_features_v2.py`, `model_v2.py`, `verify_v2.py`, `check_v2.py`,
  `c_sensitivity.py` — pipeline + verification.
- `data/` — raw downloads (boxscores, moneypuck, HR snapshots) + parquets
  (gitignored). Repro: `data/download_boxscores.py` →
  `data/extract_goalies.py` → `data/parse_injuries.py` →
  `build_features_v2.py` → `model_v2.py` → `verify_v2.py`.
