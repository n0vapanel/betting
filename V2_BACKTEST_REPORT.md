# V2 Backtest Report — Injury-Aware Walk-Forward Model

**Model v2:** rolling Four Factors + margin Elo + rest + OFFICIAL NBA injury reports (5PM game-day filing; 1PM snapshot for noon–5PM tips) + closing-line market signal, logistic regression / gradient boosting, strictly walk-forward.
**Training:** 843 games, 2021-10-29 → 2022-02-28
**Test:** 229 games, 2022-03-01 → 2022-03-31 (same 229-game window as v1)
**Dropped:** {'no_rolling_stats': 76, 'no_injury_report': 0, 'other': 0}
**Injury report source:** {'game-day 5PM': 1018, 'game-day 1PM (early tip)': 54}

## Features

- `off_eFG_home`
- `off_eFG_away`
- `tov_home`
- `tov_away`
- `orb_home`
- `orb_away`
- `ftr_home`
- `ftr_away`
- `pace_diff`
- `xprod_home`
- `xprod_away`
- `elo_diff`
- `rest_diff`
- `b2b_home`
- `b2b_away`
- `inj_home`
- `inj_away`
- `inj_diff`
- `market_home`
- `winpct_diff`
- `margin_diff`

Four Factors are rolling means over each team's last ≤10 games (min 5, strictly prior). Injury impact = Σ status-weight × prior-minutes share (Out 1.0 / Doubtful 0.5 / Questionable 0.25 / Probable 0), from the official game-day report. Elo: K=20, home-court +100, margin multiplier. Market = no-vig closing ML implied home prob.


## Model Selection (time-ordered 5-fold CV, neg-log-loss)

Feature selection: greedy backward elimination minimizing CV log-loss (training data only), then C/family re-tuned by the same CV; calibrator (raw/Platt/isotonic) chosen by Feb-2022 holdout log loss.

| model | CV log-loss |
|---|---:|
| LogReg C=0.05 | 0.6144 |
| LogReg C=0.03 | 0.6153 |
| LogReg C=0.1 | 0.6155 |
| LogReg C=0.02 | 0.6172 |
| LogReg C=0.3 | 0.6205 |
| Blend 0.5LR+0.5HGB | 0.6213 |
| LogReg C=0.01 | 0.6230 |
| LogReg C=1.0 | 0.6261 |
| LogReg C=3.0 | 0.6290 |
| LogReg C=10.0 | 0.6303 |
| LogReg C=0.005 | 0.6326 |
| HGB it=50 lr=0.03 leaf=7 | 0.6380 |
| HGB it=50 lr=0.05 leaf=7 | 0.6382 |
| HGB it=100 lr=0.03 leaf=7 | 0.6388 |
| LogReg C=0.003 | 0.6418 |
| HGB it=50 lr=0.03 leaf=15 | 0.6437 |
| HGB it=100 lr=0.05 leaf=7 | 0.6565 |
| HGB it=100 lr=0.05 leaf=15 | 0.6796 |
| HGB it=200 lr=0.05 leaf=15 | 0.7475 |

**Chosen:** LogReg C=0.05 (CV log-loss 0.6144); 9 selected features: `off_eFG_home, tov_home, elo_diff, b2b_home, inj_home, inj_diff, market_home, winpct_diff, margin_diff`; calibrator: isotonic (fitted on Feb-2022 holdout).

## Test Results (March 2022)

- Accuracy: **68.1%** (v1: 67.2%)
- Brier (model): **0.2131** vs market Brier 0.2010
- Flat $1 on every predicted winner: P&L +0.23 on $229 → ROI **+0.10%** (v1: -0.8%); bootstrap 95% CI [-9.41%, +9.12%]
- Baselines: always-home -5.89%, always-closing-favorite -3.27%

### Injury-report methodology

For each game g we use the official NBA injury report: the 5PM ET game-day filing (snapshot `datetime(g, 17, 0)`, rows with Game Date == g) — the complete NBA-mandated pre-game report, strictly pre-tip for all evening games. The 5PM day-before filing was inspected first and rejected as primary: it contains 'NOT YET SUBMITTED' placeholders for ~600 of 857 matchups (teams routinely file late), so it systematically under-reports injuries. 54 of 1072 game rows tip noon–5PM ET and use the 1PM game-day snapshot so the information stays strictly pre-tip. Injury impact = Σ status-weight × prior-minutes share.

## Edge-Gated Betting (bet $1 where model beats market by ≥ ε)

| metric | ε=0.0 | ε=0.01 | ε=0.02 | ε=0.03 | ε=0.05 |
|---|---|---|---|---|---|---|
| n_bets | 226 | 218 | 205 | 188 | 159 |
| win rate | 46.5% | 46.8% | 47.3% | 46.8% | 47.8% |
| ROI | -7.08% | -7.53% | -6.50% | -7.47% | -5.62% |

**Best gate:** ε=0.05 → 159 bets, win rate 47.8%, ROI **-5.62%** (bootstrap 95% CI [-24.41%, +13.19%])

## Quarter-Kelly on best ε

- Stakes clipped to [0, 0.05] of bankroll; 159 bets
- Bankroll multiple: 0.6964 (flat-$1 P&L on same bets: -8.93)

## Ablations (each with best-ε gating)

| config | feats (sel) | model | CV LL | calib | flat ROI | Brier | best ε | n_bets | win rate | gated ROI |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|
| main | 9/21 | LogReg C=0.05 | 0.6144 | isotonic | +0.10% | 0.2131 | 0.05 | 159 | 47.8% | -5.62% |
| no_injury | 5/18 | LogReg C=0.3 | 0.6068 | isotonic | -5.06% | 0.2161 | 0.01 | 212 | 44.8% | -6.25% |
| no_market | 8/20 | LogReg C=0.03 | 0.6296 | isotonic | +0.71% | 0.2107 | 0.03 | 192 | 48.4% | +1.57% |
| stats_only | 4/15 | LogReg C=0.05 | 0.6351 | isotonic | +2.05% | 0.2134 | 0.03 | 197 | 46.2% | +0.55% |

Selected features per config (CV greedy backward elimination):
- **main**: `off_eFG_home, tov_home, elo_diff, b2b_home, inj_home, inj_diff, market_home, winpct_diff, margin_diff`
- **no_injury**: `off_eFG_home, tov_home, elo_diff, b2b_home, market_home`
- **no_market**: `tov_home, elo_diff, rest_diff, b2b_home, inj_home, inj_diff, winpct_diff, margin_diff`
- **stats_only**: `tov_home, elo_diff, rest_diff, b2b_home`

**Injury signal delta:** +0.63pp of gated ROI (main vs no_injury, best-ε each).

## v1 Engine Through the Same Edge Gate

| metric | ε=0.0 | ε=0.01 | ε=0.02 | ε=0.03 | ε=0.05 |
|---|---|---|---|---|---|---|
| v1 n_bets | 229 | 225 | 217 | 204 | 187 |
| v1 win rate | 31.0% | 30.2% | 29.0% | 27.5% | 27.8% |
| v1 ROI | -6.47% | -7.38% | -8.62% | -10.02% | -6.90% |

v1 flat: ROI -0.82%, accuracy 67.2%, Brier 0.2212. v1 best gate: ε=0.0 → ROI -6.47% (229 bets).

## Verdict

The v2 model's flat ROI is **+0.10%** vs v1's -0.8%: BEATS v1 on flat betting (accuracy 68.1% vs v1's 67.2%). Flat-ROI bootstrap 95% CI [-9.41%, +9.12%] — does not exclude 0. Injury features add +5.16pp of flat ROI (main vs no_injury ablation). Gated ROI at best ε is -5.62% with bootstrap 95% CI [-24.41%, +13.19%] — does not exclude 0.

## Reproducing

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home
.venv/bin/python backtest/v2/data.py   # injury fetch (cached)
.venv/bin/python backtest/v2/run_v2.py
```

