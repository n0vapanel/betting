# NFL walk-forward backtest

Generated UTC: 2026-10-24T23:53:40.835468+00:00

## Executive result

- **Test:** 335 completed regular-season games, 2021-12-16 through 2023-01-08, in 22 season-weeks.
- **Preselected model:** lightgbm; **accuracy:** 64.56% on 333 decisive games. The 2 ties remain in the prediction file and betting denominator but are two-way moneyline pushes.
- **Bet accounting:** closing moneyline; $1 risked per flat bet, net profit / dollars risked. Moneyline-covered test games: 335; ATS fallback: 0.
- **Flat ROI:** 2.66%, week-clustered 95% CI [-5.40%, 10.53%].
- **Best gate selected before testing:** lean_eps_05, 242 test bets, ROI -1.21%, 95% CI [-18.21%, 16.81%].
- **Quarter-Kelly:** $1,000 → $884.14; total return -11.59%, bootstrapped 95% CI [-59.70%, 110.23%].

## Splits and no-lookahead protocol

Only REG games are modelled. Initial available training data are all 2020 regular-season games and 2021 weeks 1–14 (464 games; 462 decisive). Test games are 2021 weeks 15–18 and every completed 2022 regular-season game. There are 64 late-2021 and 271 2022 games. The cancelled 2022 BUF–CIN game has no completed result and is not a bet.

The initial training period is partitioned in advance:

1. **Tuning core:** 2020 plus 2021 weeks 1–4 (320 games). Three expanding time-ordered CV folds group whole season-weeks. Median imputation, missingness indicators and scaling are fitted inside each fold. Hyperparameters minimise mean validation log loss, not test ROI.
2. **Selection-model calibration:** 2021 weeks 5–10 (86 games), entirely excluded from that base model's fit. A regularised sigmoid maps logit(raw probability) to calibrated probability.
3. **Model/gate validation:** 2021 weeks 11–14 (58 games), not used to fit that base model or calibrator. Select model family by calibrated log loss; within that family choose the highest flat-ROI gate with at least 20 validation bets (ties resolved deterministically). Gate fallback used: False.
4. **Weekly test refits:** freeze hyperparameters, model choice and gate. Before each test season-week, use all earlier regular-season weeks; reserve its latest 6 observed weeks exclusively for a fresh sigmoid calibrator and fit each base model on the earlier remainder. First test calibration is 2021 weeks 9–14. Previously tested games may enter later training only after their week is over. Both families are predicted regardless of which was selected.

All games in a week receive features before that week's outcome, Elo, snap or NGS updates. Injury rows are the sole same-week player input and must have `date_modified < kickoff_utc`. Kickoffs are parsed in America/New_York and converted to UTC. Current-game QB names/IDs from schedules are **never** model features. Closing prices, spreads, totals, PBP win probabilities and current outcomes are **not** predictors. Prediction rows record base/calibration week cutoffs plus lagged feature-source weeks and injury timestamps.

Frozen hyperparameters: `{"lightgbm": {"num_leaves": 3}, "logreg": {"C": 0.1}}`. Feature count: 67. All selected numeric feature names are in `.cache/evaluation.json`.

## Feature formulas

### Team form, time and Elo

For each team, use its previous **4 and 8 games**, carrying observed form across the offseason, never same-week games:

- Offensive EPA/play = Σ offensive EPA / number of eligible offensive plays. Defensive EPA allowed/play uses the opponent's eligible offensive plays; lower is better.
- Eligible PBP is non-null EPA on `play_type ∈ {run, pass}`; penalty/no-play records, kicks, kneels and spikes are excluded. `qb_dropback=1` defines passing (including sacks and QB scrambles); other eligible plays are rushing. Report passing/rushing EPA/play and play shares.
- Points for/against and scrimmage yards for/against are rolling per-game averages; yards/play is Σ eligible yards / eligible plays. These are scrimmage-play yards, not an official all-purpose yardage total.
- Rest = calendar days since that team's previous completed regular-season game, capped at 30 for the model; 7 when no history exists. Include a season-opener indicator and historical game count.
- Elo starts at 1500; pregame P(home) = 1 / (1 + 10^(-(E_home − E_away + 55 H)/400)), where H=1 at a non-neutral home venue, otherwise 0. After a completed week, E_home += 20(actual − expected), E_away -= the same, with actual=0.5 on ties. Between seasons regress ratings one-third toward 1500. Include Elo difference/400, Elo home probability and H.

Models use home-minus-away differences of team/player form, explicit home/away source-availability flags and home-field/Elo terms. Availability is not silently treated as health or zero talent.

### Snap-count-weighted injury impact

For player p, obtain their **latest four prior appearances for the same team**, using only weeks before the target week. Source snap fractions are clipped to [0,1] to correct rounding such as 1.01 (3 regular-season source rows clipped). Let ō, d̄ and s̄ be mean offensive, defensive and special-team snap fractions; define ū_p = max(ō_p, d̄_p, s̄_p).

**I(team, week) = Σ_p a(status_p,week) × ū_p**, where a(Out)=1.0, a(Doubtful)=0.5, a(Questionable)=0.25 and otherwise 0. The sum is over that week's timestamp-eligible official injury-report players. Each player is counted once; if multiple report revisions exist, use the latest pre-kickoff one. Also include phase-specific Σ a×ō, Σ a×d̄, Σ a×s̄, number of positive-status reports, unmatched-player count and prior-snap match share.

These are **snap-equivalent lost-usage units**, not literal player-minutes or a medical absence forecast. Averaging prior appearances preserves a player's pre-injury role while absent; it is not zero-filled with subsequent missed games. No previous snaps means unknown usage: contribution is zero with explicit unmatched/missing flags, never an inferred league-average starter. Injured reserve and unlisted injuries absent from the weekly official report are not fabricated.

Player joins first use static PFR↔GSIS ID links, otherwise exact accent/punctuation/suffix-normalised full name within the same team. Ambiguous normalised names are not joined; there is no fuzzy matching. Only ID columns, not current team/age/draft attributes, are read from the contemporary crosswalk.

### Expected starter and quality/participation proxies

Use only the **latest prior-week depth snapshot in the same season**. A rank-1 QB is the expected starter; if unavailable, use the highest-attempt QB in that team's most recent prior NGS week. This is an auditable **lagged starter proxy**, not knowledge of the actual game-day announcement. `qb_on_report` flags that proxy appearing anywhere on the current pregame report (including a null game-status practice listing); `qb_injury_weight` uses the status weights above.

Prior depth rank-1 offense/defense players yield mean prior snap usage, mean usage × (1−injury weight), summed starter injury impact and starter snap-match coverage. This is **role/continuity quality**, not an independently estimated talent rating. For the expected QB use the last four prior observed NGS games: attempt-weighted CPOE and passer rating, and total passing yards / attempts.

Team NGS uses the last four observed prior team-weeks: attempt-weighted passing CPOE, rushing yards/attempt and rushing yards over expected/attempt; reception-weighted receiving YAC over expected and target-weighted separation. `week=0` full-season aggregates and postseason records are always discarded before feature calculation. NGS often covers qualifying players only, so these are covered-player aggregates, not complete official team totals.

## Predictive and flat-bet results

Intervals below are 95% percentile bootstraps resampling **whole season-weeks**, with 2,000 replicates and seed 20261024; all games/bets in a sampled week stay together. No iid-game bootstrap is used.

| Predictor | Accuracy [95% CI] | Brier | Log loss | Flat $1 ROI [95% CI] |
|---|---:|---:|---:|---:|
| logreg | 61.56% [58.10%, 65.17%] | 0.23476 | 0.66171 | -3.96% [-9.33%, 1.40%] |
| lightgbm (preselected) | 64.56% [60.06%, 68.86%] | 0.22312 | 0.63820 | 2.66% [-5.40%, 10.53%] |
| No-vig market | 65.77% [61.54%, 70.12%] | 0.20934 | 0.60581 | -4.50% [-10.81%, 2.48%] |

- **logreg:** Brier minus market = +0.02542; week-clustered 95% CI [+0.01470, +0.03570] (negative favours the model).
- **lightgbm:** Brier minus market = +0.01378; week-clustered 95% CI [+0.00277, +0.02416] (negative favours the model).

Brier/log loss/accuracy use decisive games only, matching two-way tie-refund moneyline semantics. Brier = mean((P(home win)−1_home_win)²). Market probabilities remove two-sided vig: q_home = (1/decimal_home) / ((1/decimal_home)+(1/decimal_away)). The market baseline bets its more probable winner at the actual offered price. No commission, limits, slippage, line shopping or transaction costs beyond quoted vig are modelled.

| Model | Test season | Games | Accuracy | Flat ROI | Brier |
|---|---|---:|---:|---:|---:|
| logreg | 2021 | 64 | 64.06% | -2.78% | 0.24236 |
| logreg | 2022 | 271 | 60.97% | -4.24% | 0.23295 |
| lightgbm | 2021 | 64 | 67.19% | 2.15% | 0.21321 |
| lightgbm | 2022 | 271 | 63.94% | 2.78% | 0.22548 |

## Edge gates

Let δ = P_model(home) − q_market(home). The lean backs home when δ>0, otherwise away, requiring |δ|>ε (strictly). Winner-conviction additionally requires that side to be the model's ≥50% winner. At equality there is no lean. ε ∈ {0, 0.02, 0.05} is in absolute probability points. A no-vig edge at ε=0 need not be positive EV at the vigged offered price.

| Model | Gate | Bets | Weeks with bets | Flat ROI | Week-clustered 95% CI |
|---|---|---:|---:|---:|---:|
| logreg | lean, ε=0.00 | 335 | 22 | -6.57% | [-19.06%, 6.89%] |
| logreg | lean, ε=0.02 | 286 | 22 | -7.20% | [-20.76%, 7.81%] |
| logreg | lean, ε=0.05 | 236 | 22 | -11.66% | [-26.99%, 4.13%] |
| logreg | winner_conviction, ε=0.00 | 141 | 22 | -6.74% | [-18.01%, 4.55%] |
| logreg | winner_conviction, ε=0.02 | 119 | 21 | -5.62% | [-17.83%, 7.65%] |
| logreg | winner_conviction, ε=0.05 | 98 | 21 | -11.13% | [-25.57%, 4.05%] |
| lightgbm | lean, ε=0.00 | 335 | 22 | -0.86% | [-13.81%, 12.77%] |
| lightgbm | lean, ε=0.02 | 293 | 22 | -2.77% | [-17.00%, 12.84%] |
| lightgbm | lean, ε=0.05 **preselected** | 242 | 22 | -1.21% | [-18.21%, 16.81%] |
| lightgbm | winner_conviction, ε=0.00 | 154 | 22 | 8.53% | [-6.51%, 24.72%] |
| lightgbm | winner_conviction, ε=0.02 | 137 | 22 | 8.00% | [-8.78%, 26.22%] |
| lightgbm | winner_conviction, ε=0.05 | 120 | 22 | 7.48% | [-9.63%, 25.33%] |

Pre-test-selected gate validation ROI was -9.16% on 42 bets. **Post-hoc descriptive maximum only:** lightgbm / winner_conviction_eps_00 returned 8.53% on 154 test bets. This test-selected maximum was not used for quarter-Kelly or the primary verdict; its nominal CI does not correct for searching gates/models.

For an unavailable moneyline pair only, q_market(home)=Φ(spread_line/13.5); ATS settles `side × (home_margin−spread_line)` at −110. Positive spread means home is favoured. Kelly ATS cover probability would be Φ(Φ⁻¹(P_home_win)−spread_line/13.5), an explicit latent-normal fallback rather than a directly fitted cover model. In this run the fallback was used for **0** test games.

## Quarter-Kelly on the preselected gate

For offered net decimal payout b and model probability p for the selected side, f = min(0.02, max(0, 0.25×(b p−(1−p))/b)). Even if the no-vig gate passes, negative offered-price EV produces **zero stake**. Cap each bet at 2% of bankroll; if weekly fractions ever exceed 100%, rescale proportionally. Every wager in the same week uses the **start-of-week bankroll**; do not finance a Sunday bet using a Thursday result.

- Actual positive-stake bets: 242; total staked: $5780.15; net profit / total staked: -2.00%.
- Initial/final bankroll: $1,000 / $884.14; total return: -11.59%.
- Week-end maximum drawdown: -45.80%; this does not measure intraday mark-to-market risk.
- 95% total-return interval: [-59.70%, 110.23%], from resampling realised weekly portfolio returns and compounding them. Neither this interval nor flat-bet intervals refit/reselect models, account for training uncertainty or guarantee future bankroll behaviour.

## Data coverage and closing-line audit

Raw source counts: `{"depth": 75267, "injuries": 11269, "ngs_passing": 1792, "ngs_receiving": 4561, "ngs_rushing": 1831, "snaps": 52849}`.

| Lagged/pregame source | Test team-games covered | Coverage |
|---|---:|---:|
| injury_report_available | 667/670 | 99.55% |
| snap_history_available | 670/670 | 100.00% |
| depth_available | 638/670 | 95.22% |
| ngs_passing_available | 670/670 | 100.00% |
| ngs_rushing_available | 670/670 | 100.00% |
| ngs_receiving_available | 670/670 | 100.00% |

- Positive-status test player-reports: 3225; matched to prior same-team snaps: 3128 (96.99%). Unknown players are flagged, not assigned hindsight usage.
- Injury rows excluded for non-pre-kickoff/missing timestamps: 3; report rows with no completed regular-season game: 17. The latter are not turned into observations.
- Injury/snap/depth imports cover **2021–2022 only**, as requested. The 2020 training rows have source-missing indicators; early 2021 has no 2020 snap/depth warm-up. A team without report rows may have no listed injuries or missing source data; the availability flag does not distinguish these cases.
- Depth charts and NGS are historical archives, not guaranteed contemporaneous immutable snapshots. Same-week depth charts are deliberately excluded. Weekly injury `date_modified` enforces a documented timestamp cutoff but cannot recover earlier report revisions that the archive no longer contains. PBP EPA and other nflverse fields may have later vendor/model corrections. This is therefore a **strictly lagged historical-data backtest, not a certified point-in-time production simulation**.
- Schedule betting columns inspected include `home_moneyline`, `away_moneyline`, `spread_line`, `home_spread_odds`, `away_spread_odds` and `total_line`; original columns are preserved in the source manifest. These are treated as the historical closing fields supplied by nflverse's schedule source, not independently timestamped sportsbook quotes.
- Schedule/PBP spread comparison: 799/799 games covered, 799 exact matches, 0 mismatches, maximum absolute difference 0.000 points.
- `vegas_line` column present: False. Available validation fields are `spread_line`, `vegas_wp`, `vegas_home_wp`; use only initial q1/3600-second `vegas_home_wp`, **not** its in-game values. Pregame comparisons: 799 games; MAE versus no-vig moneyline 0.02081, correlation 0.99452. That PBP value is a spread-informed win-probability model, not an independent moneyline quote or proof of closing-price timing.
- Actual schedule loader: `nfl_data_py.import_schedules`. Optional-source errors: `{}`.

Source parquet SHA-256 hashes, schemas, model-fold cutoffs, runtime versions and full feature names are frozen in `.cache/evaluation.json`; cached parquet remains under `.cache/`. Prediction CSV SHA-256: `92106f1cd35e66d70fefdf561872de2fbefd778d7b48e4f8e64423b74e73ba44`.

### Source endpoints

- nfl_data_py / nflverse schedules: http://www.habitatring.com/games.csv (official NFLverse fallback: https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv).
- PBP: `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet` for 2020–2022.
- NGS: `https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_{kind}.parquet` for passing, rushing and receiving; filter 2020–2022.
- Official injury-report archive: `https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.parquet` for 2021–2022.
- Snap participation: `https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{year}.parquet` for 2021–2022.
- Depth snapshots: `https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{year}.parquet` for 2021–2022.
- Static identity crosswalk: https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv.

These endpoints were read through installed nfl_data_py 0.3.3 on the run date. They are related data pipelines, not independent corroborating sources; the PBP spread agreement validates internal alignment only.

## Time-ordered tuning evidence

| Model | Candidate | Fold | Last training week | Validation weeks | Log loss |
|---|---|---:|---:|---|---:|
| logreg | `{"C": 0.1}` | 1 | 202009 | 202010–202013 | 0.58907 |
| logreg | `{"C": 0.1}` | 2 | 202013 | 202014–202017 | 0.64651 |
| logreg | `{"C": 0.1}` | 3 | 202017 | 202101–202104 | 0.79552 |
| logreg | `{"C": 1.0}` | 1 | 202009 | 202010–202013 | 0.84786 |
| logreg | `{"C": 1.0}` | 2 | 202013 | 202014–202017 | 0.69177 |
| logreg | `{"C": 1.0}` | 3 | 202017 | 202101–202104 | 0.86033 |
| logreg | `{"C": 10.0}` | 1 | 202009 | 202010–202013 | 1.21907 |
| logreg | `{"C": 10.0}` | 2 | 202013 | 202014–202017 | 0.74893 |
| logreg | `{"C": 10.0}` | 3 | 202017 | 202101–202104 | 0.88758 |
| lightgbm | `{"num_leaves": 3}` | 1 | 202009 | 202010–202013 | 0.63382 |
| lightgbm | `{"num_leaves": 3}` | 2 | 202013 | 202014–202017 | 0.65085 |
| lightgbm | `{"num_leaves": 3}` | 3 | 202017 | 202101–202104 | 0.68986 |
| lightgbm | `{"num_leaves": 7}` | 1 | 202009 | 202010–202013 | 0.63844 |
| lightgbm | `{"num_leaves": 7}` | 2 | 202013 | 202014–202017 | 0.65793 |
| lightgbm | `{"num_leaves": 7}` | 3 | 202017 | 202101–202104 | 0.70156 |

## Verdict

The preselected lightgbm model produced 64.56% decisive-game accuracy and 2.66% flat-stake ROI. Its preselected lean_eps_05 gate returned -1.21% on 242 bets, with a week-clustered 95% ROI interval of [-18.21%, 16.81%]. This does not establish a positive betting edge: the uncertainty interval does not exclude zero on the profitable side. The selected model's Brier score is 0.22312 versus 0.20934 for the no-vig market; lower is better. Even the least-losing pre-test gate had negative validation ROI in this run, so its selection is a comparison among the requested gates, not a recommendation to bet. Injury, starter and time-played features are explicitly present, but this experiment does not isolate their causal or incremental contribution; that requires a separately specified ablation and a new untouched evaluation period.
