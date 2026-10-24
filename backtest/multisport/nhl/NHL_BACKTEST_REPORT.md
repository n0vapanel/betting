# NHL Walk-Forward Backtest — 2021-22 season

**Test window:** Mar 1 – Jun 26, 2022 (466 regular + 89 playoff = 555 games)
**Train window:** Oct 12, 2021 – Feb 28, 2022 (846 games)
**Models:** Logistic Regression (L2, C=0.01) + LightGBM; ensemble weight selected by time-ordered CV
**Generated:** 2026-10-24

## Headline numbers (test, all 555 games)

| Metric | Model | Market | Baselines |
|---|---|---|---|
| Accuracy (pick winner) | **59.1%** | 64.9% (favorite) | 55.9% (always home) |
| Flat $1 ROI (2-way closing odds) | **−0.26%** | −1.30% (flat favorite) | −3.26% (always home) |
| Brier score | 0.2407 | **0.2267** | — |
| Flat ROI bootstrap 95% CI (date-clustered, 2000×) | [−6.3%, +5.7%] | — | — |
| Quarter-Kelly bankroll ($1000 start, 5% stake cap) | $1001.19 (+0.1%) | — | — |

## Verdict

**The model does not beat the NHL closing line.** Flat-betting every model pick at
2-way closing odds returns ≈0% (CI straddles zero); the model's Brier score is worse
than the de-vigged market (0.2407 vs 0.2267); the market favorite picks winners more
often (64.9% vs 59.1%). Edge-gated subsets show positive point estimates
(+4.1% at ε=0.02, 195 bets) but wide bootstrap CIs that include zero
([−8.1%, +16.0%]). The NHL moneyline market is efficient with respect to the public
statistical signals tested here. **Do not bet this system live.** The only
consistent finding is that the model's flat ROI (−0.26%) beats naive baselines
(always-home −3.26%, flat market-favorite −1.30%) by picking sides the market
under-prices relative to its own probability estimate — but the estimate itself is
worse-calibrated than the market's, so this survives only as a small pick-level
effect, not a bankable edge.

## Data

- **Scores & advanced stats — MoneyPuck** `playerData/careers/gameByGame/all_teams.csv`
  (team-per-game, all situations): 1,401 games for 2021-22 (1,312 regular + 89
  playoff), 2021-10-12 → 2022-06-26. Final scores cross-checked against two
  independent sources (below): **1,401/1,401 exact**, after accounting for MoneyPuck
  excluding shootout-deciding goals (102 SO games show the OT tie; winner taken from
  the odds source, which matches the 2-way moneyline definition).
- **Odds — Kaggle `archangel0x01/nba-nhl-and-college-basketball-results-and-odds`**
  (`games.csv`, Oddsportal-style average closing odds, both sides, full coverage
  incl. playoffs): all 1,401 games matched 1:1 on unordered team pair + score
  compatibility (1,245 matched at +1 day offset = UTC-shifted dates).
- **Cross-validation of odds — Kaggle `michaelmallari/sportsbook-odds-on-nhl-games`**
  (Sportsbookreview 2-way favorite moneylines, regular season only, n=1,311):
  final-score agreement 1,311/1,311; de-vigged probability agreement with archangel
  corr 0.996, median |Δp| 0.9pp; favorite side agreement 96% (disagreements are
  near-pick'ems). Closing favorite win rate 64.8% — sane for NHL.

### ⚠️ Payout reconstruction (important)

The archangel **raw** odds are inflated ~21% on average (ratio to SBR-implied fair
2-way price: 1.16×–1.25× depending on favorite strength) — 3-way/regulation prices
are mixed into the Oddsportal averages. The *de-vigged probabilities* (normalized
inverse odds) are unaffected (validated above). All ROI figures therefore use
**synthetic 2-way payouts rebuilt from the de-vigged probabilities at a standard
4.76% hold** (`odds = 1/(p·1.0476)`). Sensitivity: flat ROI = +2.1% at 2.38% hold,
−0.3% at 4.76%, −1.9% at 6.5% (headline uses 4.76%). Raw odds are kept in the
predictions CSV for transparency.

## Features (19 home-minus-away diffs, strictly pre-game)

| Group | Features |
|---|---|
| Process | xG share 5v5 (EWMA spans 5/10/20), Corsi share 5v5, xG share all-sit., GF share all-sit. |
| Goaltending/finishing | PDO (spans 10/20), shooting %, save % |
| Special teams | PP efficiency (PPG per 2-min opportunity, 5on4), PK efficiency (GA per 2-min opp, 4on5), penalty minutes/game |
| Schedule | rest days (capped 7), back-to-back flag, faceoff % |
| Results | net goals/game, points % (2 pts win, 1 pt OT/SO loss), Elo (K=6, margin-multiplied ×1/1.5/2, +35 home-ice in the diff) |

EWMA with span-10 default, neutral cold-start priors (0.5 shares, 1.0 PDO, 1500 Elo).
No lookahead: rolling state updated only after each game; Elo uses the true winner
incl. OT/SO. **Verified by independent reconstruction** on sampled games: xG share,
rest, and Elo features reproduced exactly (3/3 each), and headline ROI recomputed
from `nhl_predictions.csv` to the digit.

## Model selection (time-ordered 5-fold CV on train)

| Model | OOF Brier |
|---|---|
| LogReg C=0.01 (best of C∈{0.003…1.0}) | 0.2875 |
| LightGBM (lr=0.03, 7 leaves, 100 trees, mc=10) | 0.3003 |
| Market (train, de-vigged) | 0.2232 |
| Calibrated ensemble, weight w=1.0 (selected by calibrated-OOF Brier) | 0.2994 |

The market is already ~0.06 Brier better in-sample. Ensemble weight optimization
selected pure LogReg (w=1.0); LightGBM added no value on this data. Calibration:
Platt scaling fitted on fold-out-of-fold OOF predictions (isotonic was also tested —
it distorted the tails and produced a spurious −71% quarter-Kelly drawdown, hence
Platt). Test ECE 0.047.

## Test results (Mar–Jun 2022)

| Segment | n | Acc | Flat ROI | Brier (model/market) | q-Kelly |
|---|---|---|---|---|---|
| All | 555 | 59.1% | −0.26% | 0.2407 / 0.2267 | +0.1% |
| Regular (Mar–Apr) | 466 | 58.8% | −1.32% | 0.2404 / — | +17.3% |
| Playoffs (May–Jun) | 89 | 60.7% | +5.27% | 0.2421 / — | −14.6% |

## Edge gates (bet only when p_model − p_market > ε on the model's side)

| Gate | n bets | ROI | Win% | Bootstrap 95% CI (date-clustered) |
|---|---|---|---|---|
| ε=0 | 229 | +1.27% | 45.4% | — |
| ε=0.02 | 195 | +4.09% | 45.1% | [−8.1%, +16.0%] |
| ε=0.05 | 147 | +2.12% | 40.8% | [−13.1%, +17.4%] |
| either-side, ε=0.02 | 485 | −0.71% | — | — |
| either-side, ε=0.05 | 389 | −1.06% | — | — |

Gate bets are underdog-heavy (win% 40–45%) — the point estimates are positive but
every CI includes zero; the edge is not statistically established. The "either-side"
variant (bet whichever side has positive edge) is negative — the model's edge signal
lives on its own pick side only, and weakly.

## Ablations (LogReg C=0.01, Platt, 2-way payouts, test)

| Features | Acc | Flat ROI | Brier |
|---|---|---|---|
| Full model (19) | 59.1% | −0.26% | 0.2407 |
| schedule only (rest/b2b/FO) | 56.2% | −2.83% | 0.2430 |
| Elo only | 55.9% | −3.26% | 0.2442 |
| xG only | 55.9% | −3.26% | 0.2460 |
| PDO/goals only | 55.9% | −3.26% | 0.2460 |
| Corsi only | 55.9% | −3.26% | 0.2468 |
| Special teams only | 55.9% | −3.26% | 0.2477 |

Single groups collapse to the home baseline (55.9% — they just pick home); the
signal is additive across groups, reaching +3.2pp over baseline together. None of it
is enough to overcome the closing line.

## Quarter-Kelly

0.25-Kelly on all model-side-positive-edge bets (5% stake cap): final bankroll
$1001.19 from $1000 (+0.1%) on the full window. Regular-season window +17.3%,
playoffs −14.6% — both consistent with noise at these sample sizes.

## Caveats

- Payouts are reconstructed (see above) — ROI magnitudes depend on the assumed 4.76%
  hold, but the conclusion (market not beaten) is robust across 2.4%–6.5% holds.
- MoneyPuck's 'all' goals exclude shootout-deciding goals; winners used for targets
  and Elo come from the odds source's final score (2-way definition).
- 2021-22 was a COVID-disrupted season (heavy Dec-Jan postponements); train/test
  regime shift is real and calibration (ECE 0.047) is imperfect.
- Playoff segment is only 89 games — its +5.3% flat ROI is noise.

## Artifacts

- `nhl_predictions.csv` — 555 test-game rows: probabilities, market probs, raw +
  reconstructed odds, model picks, edges, per-game returns, all 19 feature diffs.
- `metrics.json` — full metric dump. `verify.json` — recomputed headline checks.
- `games_full.parquet`, `games_features.parquet`, `team_games.parquet` — build
  artifacts. `build_data.py`, `build2.py`, `features.py`, `model.py` — pipeline
  (repro: `msports-venv/bin/python build_data.py && python build2.py && python
  features.py && python model.py`).
