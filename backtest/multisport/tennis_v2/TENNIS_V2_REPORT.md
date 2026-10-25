# ATP Tennis v2 Backtest Report (accuracy push)

**Generated:** 2026-10-24 20:27

## Data
- Betting rows: tennis-data.co.uk `2021.xlsx`/`2022.xlsx` (ATP, B365 closing decimal odds) — identical test window to v1 (2022, n=2538).
- **Serve/return + duration:** JeffSackmann `tennis_atp` `atp_matches_2019-2022.csv` via GitHub fork `Kadantte/tennis_atp` (upstream repo deleted; fork kept current to 2026). Columns used: svpt/ace/df/1stIn/1stWon/2ndWon/bpSaved/bpFaced (+opponent side) and `minutes`. Joined exactly on Sackmann player_id; tennis-data name → player_id via the v1 fuzzy matcher (particle handling, initials, latest-DOB tie-break).
- Player-DOB coverage (both sides): 91.4% of model rows.
- Serve-stat coverage (both players have EWMA history): train 78.2%, test 80.1%. Minutes present for 86.3% of model rows.
- Fatigue inputs use ALL tennis-data rows (incl. Retired/Walkover) so retirements and partial matches still load the body; Elo/ratings update on completed rows only (v1 semantics preserved).

## Upgrades over v1
- **(a) H2H** (same-file prior meetings, completed matches only): count (log1p), recency-weighted win share (EWMA α=0.9 per meeting), last-meeting winner (±1), surface-specific share.
- **(b) Surface Elo**: v1's raw per-surface rating (K=32) shrunk toward the global rating: `blend = (n_surf·selo + 15·elo)/(n_surf+15)` — fixes v1's noisy low-sample surface ratings.
- **(c) Serve/return** (EWMA α=0.9 over prior matches, point-weighted, ≥30 points required): ace%, df%, 1st-serve win%, 2nd-serve win%, serve-points-won%, return-points-won%, break-points-saved%, break-point-conversion%; career-pooled versions of the two points-won rates.
- **(d) Fatigue**: matches/games/sets/minutes in last 14 days (per player), mean minutes of last 5 matches, retirement-as-loser within 30 days (injury flag).
- **(e) Duration**: match `minutes` (Sackmann) — not present in tennis-data xlsx.

## Protocol (unchanged from v1)
- Single forward pass over the 2021+2022 timeline; every feature uses strictly earlier matches only (no lookahead; same-day sack stats applied after the row).
- Orientation P1 = higher pre-match global Elo (rank tie-break), y = P1 wins, market prob = no-vig B365 closing.
- LogReg (L2, C grid) + LightGBM (num_leaves×min_child_samples grid), expanding-window time-ordered 5-fold CV on 2021 (Brier), isotonic on OOF, soft ensemble; primary model by OOF Brier (market excluded as a choice).
- Test 2022: accuracy, Brier vs market, flat-$1 ROI at closing odds, edge gates ε∈{0, 0.02, 0.05}, quarter-Kelly on best gate (n≥30), bootstrap CI clustered by year+ISO week (2,000 resamples).

## Results: ablation by feature group (test 2022, n=2538)
| group | OOF Brier (2021) | acc | Brier | ROI ε=0 | ROI ε=0.02 | ROI ε=0.05 |
|---|---|---|---|---|---|---|
| v1 baseline (prior run) | — | 67.8% | 0.2024 | −6.18% | — | −1.0% |
| V1 | 0.2023 | 67.5% | 0.2015 | -5.27% | -5.57% | -2.39% |
| V1+H2H | 0.2019 | 67.7% | 0.2032 | -8.18% | -6.14% | -4.59% |
| V1+FAT | 0.2020 | 67.7% | 0.2023 | -8.68% | -7.52% | -7.11% |
| V1+SRV | 0.2019 | 68.0% | 0.2001 | -4.39% | -2.99% | -1.90% |
| FULL | 0.2018 | 67.5% | 0.2012 | -5.74% | -4.42% | -3.98% |

**FULL model**: LogReg C=0.03, LGBM (15, 80), primary=ens; OOF Brier {'lr': 0.2045, 'lgb': 0.2043, 'ens': 0.2018, 'mkt': 0.2045}; test acc **67.5%** (market 68.1%), Brier **0.2012** (market 0.1980).

### FULL-model gates & staking (2022)
| ε | n bets | bet acc | flat-$1 ROI |
|---|---|---|---|
| 0.00 | 2538 | 0.5197 | -0.0574 |
| 0.02 | 1840 | 0.5212 | -0.0442 |
| 0.05 | 1031 | 0.5092 | -0.0398 |

- Best gate (by ROI, n≥30): ε=0.05, n=1031, ROI -0.0398.
- Quarter-Kelly on best gate: bankroll ×0.0872 (-91.3%).
- Bootstrap 95% CI (week-clustered): ε=0 ROI [-0.1006, -0.0091]; ε=0.05 ROI [-0.1137, +0.0381]; best-gate (ε=0.05) [-0.1134, +0.0357].
- Context: flat all-in P1 ROI -0.0274; market-favorite flat ROI -0.0563 (acc 0.6812).

## Verdict
v2 does NOT beat v1. FULL vs the in-run V1 ablation (same data, same protocol): accuracy 67.53% vs 67.53% (+0.00pp), Brier 0.2012 vs 0.2015 (-0.0003). Against v1's published numbers (67.8% acc, Brier 0.2024) the deltas are the same order of magnitude — noise, not improvement. Gate ROIs are negative at every ε for FULL (ε=0: -0.057, ε=0.02: -0.044, ε=0.05: -0.040); the ε=0 week-clustered CI [-0.1006, -0.0091] is entirely below zero and quarter-Kelly compounds to -91.3%. Ablation ranking: serve/return stats are the strongest single upgrade (V1+SRV acc 67.97%, Brier 0.2001, ε=0.05 gate -0.019, CI [-0.0604, +0.0286] — still negative/straddling zero), while h2h and fatigue add nothing on top. The 2022 ATP closing line prices all of it; no profitable betting rule emerges.

Deliverables: `tennis_v2_predictions.csv`, this report, `run_tennis_v2.py`, `raw/atp_matches_*.csv` (Sackmann fork), `probe_coverage.py`.
