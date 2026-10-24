# ATP Tennis Walk-Forward Backtest Report

**Generated:** 2026-10-24 19:29

## Data
- Source: tennis-data.co.uk `2021.xlsx` / `2022.xlsx` (ATP-only section, B365 closing
  decimal odds), retrieved via Wayback Machine snapshot 2023-05-14 (live site WAF-blocked).
- Player DOBs: JeffSackmann `atp_players.csv` (Wayback snapshot 2023-03-16) — used for the age feature.
- Matches kept: completed, both B365 odds present. Train (2021): **2389**, Test (2022): **2538**.
- Age feature coverage (both DOBs matched): 4503/4927 matches (91.4%) — feature used.
- Surfaces: Hard/Clay/Grass; Best-of 3/5; tiers ATP250→Masters Cup.

## Features (no lookahead — strictly prior matches only)
- Per-player rolling, single forward pass over 2021+2022 timeline:
  global Elo (K=32, rank-based init 2000-400·log10(rank)), surface Elo (same),
  form win% last-10, games-won% last-10 (from set scores; service/return points
  not in source), surface form win% last-10, surface career win% (≤40), rest days,
  rank (log), age (from DOB).
- Row oriented P1 = higher pre-match Elo (rank tie-break); target y = P1 wins.
- Match-level: best-of, series tier, surface dummies, **market no-vig closing prob**.
- Feature list: ['elo_diff', 'selo_diff', 'form_diff', 'games_diff', 'sform_diff', 'sgames_diff', 'scareer_diff', 'rest_diff', 'rank_diff', 'age_diff', 'tier', 'bestof', 'surf_hard', 'surf_clay', 'mkt1']

## Models & protocol
- LogReg (L2, C grid) and LightGBM (num_leaves×min_child_samples grid), tuned by
  expanding-window 5-fold time-ordered CV on 2021 (Brier). Best: LogReg C=0.03, LGBM (15, 80).
- Isotonic calibration fit on out-of-fold 2021 predictions; test probs = mean of
  calibrated LogReg+LGBM (soft ensemble). Primary model by OOF Brier: **ens**.
- OOF Brier (2021 CV): {'lr': 0.2032, 'lgb': 0.205, 'ens': 0.2022, 'mkt': 0.2044}

## Test results (2022, closing B365 odds)
- n = 2538 matches
- Accuracy (model argmax side): **0.6777** vs market-side 0.6812
- Brier: model **0.2024** vs market **0.1980** (market better calibrated)
- Flat $1 all-in on P1 (higher-Elo) side: ROI -0.0257
- Flat $1 on market favorite: ROI -0.0563 (acc 0.6812)

### Edge gates (bet model side when |p_model − p_mkt| > ε)
| ε | n bets | bet accuracy | flat-$1 ROI |
|---|--------|--------------|-------------|
| 0.00 | 2538 | 0.5449 | -0.0618 |
| 0.02 | 1833 | 0.5526 | -0.0489 |
| 0.05 | 955 | 0.5508 | -0.0098 |

**Best gate** by ROI (n≥30): ε = 0.05 — n=955, ROI -0.0098.
- Quarter-Kelly on best gate: n=955, mean stake 0.047 (max 0.235) of bankroll, probs clipped to [0.02, 0.98], final bankroll ×0.1140 (-88.6%).
- Bootstrap 95% CI (2,000 resamples, clustered by year+ISO week):
  - ε=0 flat ROI: [-0.1000, -0.0222]
  - best-gate (ε=0.05) ROI: [-0.0778, +0.0558]

## Verdict
No exploitable edge. Point-estimate ROI is negative at every gate (eps=0: -0.062, eps=0.02: -0.049, eps=0.05: -0.010); the eps=0 bootstrap CI [-0.100, -0.022] is entirely below zero and the best gate's CI [-0.078, +0.056] straddles zero. The model matches the market on accuracy (0.678 vs 0.681) and is slightly worse calibrated (Brier 0.2024 vs 0.1980). Quarter-Kelly compounds to -88.6% because stakes scale with perceived edge, which is concentrated at mid-confidence bets (p 0.55-0.75) that win only ~53%: where the model disagrees with the market it is systematically wrong. Conclusion: 2022 ATP closing moneylines are efficient for this feature set; the model adds nothing over the no-vig closing line and should not be bet with real money.

Deliverables: `tennis_predictions.csv` (2538 rows), this report.
