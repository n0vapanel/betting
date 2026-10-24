# NBA v3: leakage-aware features and locked March-2022 evaluation

## Run

From this repository:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B backtest/v3/schema_discovery.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B backtest/v3/run_v3.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B backtest/v3/run_v3.py --report-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider backtest/v3 -q --basetemp=backtest/v3/test_tmp_final
```

The runner first reproduces the **frozen published v2 model** rather than
rerunning v2's feature search. It asserts the published rounded accuracy,
ROI, model Brier, market Brier and 843/229 sample sizes before any v3 fit.
The original v2 code and all caches remain read-only. No git command is used.
Bytecode and pytest cache writes are disabled.

LightGBM 4.6.0 and XGBoost 3.1.1 are installed in the existing symlinked
`.venv`. On this macOS installation the runner automatically re-execs with
sklearn's existing `.dylibs` directory on `DYLD_LIBRARY_PATH`, resolving
OpenMP without a Homebrew install or binary patch. Direct booster imports
outside the runner need that environment variable. Numerical threads are
limited to one; greedy candidate evaluations use four worker threads.

`--report-only` reads `backtest/v3/metrics.json` and renders the frozen numbers,
without source-data reads, feature recomputation or training. Normal runs
reuse feature/model captures only when source SHA-256 fingerprints match.
All exact output paths are recorded in `backtest/v3/output_manifest.json`.

## Components

| File | Boundary and reason for separation |
|---|---|
| `backtest/v3/schema_discovery.py` | Requested executable schema inspection; it streams both full CSVs, prints all columns, five rows and per-column statistics, and writes `v3_SCHEMA.md`. |
| `backtest/v3/features_v3.py` | Requested game-level home/away feature assembly. Unlike existing `player_features.py`, its inputs include paired teams, dated injury snapshots and two odds sources; it does not alter that verified team-game producer. |
| `backtest/v3/model_v3.py` | Requested time-ordered selection and training; it accepts training rows only and cannot inspect March labels. |
| `backtest/v3/evaluation_v3.py` | Probability-to-bets, settlement, Kelly and bootstrap formulas, independently tested without fitting models. |
| `backtest/v3/baseline_v2.py` | Calls unmodified v2 implementation with the published chosen specification and preserves exact reproduction evidence. |
| `backtest/v3/run_v3.py` | Requested composition root; freezes model and gate selection before its first March prediction, then captures results and renders the report. |
| `backtest/v3/test_pipeline_v3.py` | New temporal, price, betting, Kelly, bootstrap and schema contracts; complements the unchanged 23 player-feature tests. |

## Fixed experiment and missing-data policy

- Authorized training: **2021-10-20–2022-02-28**, complete finite predictor
  rows with genuine Kaggle closing moneylines. Five prior games are required;
  the supplied player features start November 1, so training naturally begins
  later. Missing values are never filled using later games.
- Five expanding CV folds use **only dates before 2022-02-01** and keep every
  date wholly in one partition. StandardScaler is fit inside each LogReg fold.
  Backward selection evaluates LogReg C=0.05, removes the best improving
  feature (CV log-loss improvement >0.00001), and stops at no improvement or
  three remaining features.
- Tune LogReg C, regularized shallow LightGBM and regularized shallow
  XGBoost on those same folds. The full grids are literal constants in
  `model_v3.py` and are captured before fitting in `experiment_plan.json`.
- Pick each family's best hyperparameters by pooled out-of-fold log loss;
  take the top **two distinct families**. Soft-vote weights for the first
  family are `{0.25, 0.5, 0.75, 0.9}`. For robustness, tune a transparent
  model/market probability mixture with model weight `{0.25, 0.5, 0.75, 1}`.
  Both choices use pre-February OOF log loss only. The market component is
  always the fixed Kaggle no-vig closing probability.
- Fit base models through January; obtain genuinely out-of-sample February
  raw probabilities. Fit Platt/isotonic on February 1–14, choose raw/Platt/
  isotonic by February 15–28 log loss. Then refit the chosen calibrator on all
  February **out-of-sample raw** probabilities, and refit base models through
  February 28 for March prediction. Probabilities are bounded to [.005,.995].
- Run the full pipeline for `all_features`, `no_players`, `no_clv`,
  `no_boxscores`, `no_injuries` on identical training and benchmark rows.
  **Primary configuration = smallest pre-February ensemble CV log loss**;
  ties prefer registration order. It is not picked by March accuracy/ROI.
- Choose each deployable gate on held-out February 15–28 predictions by
  highest ROI among the ten registered gates with at least 30 bets. Tie:
  more bets, lower epsilon, variant name. If none qualifies, use winner-side
  epsilon zero. February is reused for calibration choice and gate choice;
  its results are validation scores, not unbiased performance estimates.
- March **2022-03-01–2022-03-31 stays exactly 229 games**. March labels never
  enter any choice. Completed earlier March games may update historical
  statistics; March does not update fitted model weights.
- **Predeclared incomplete-test fallback:** any March row missing any
  registered predictor receives its Kaggle no-vig market probability for
  every configuration. Do not drop it, fabricate an opening, use a later
  injury report or choose a replacement using outcomes. This affects March
  3, ATL–CHI, whose source has no opening marker. Output `prediction_source`
  explicitly labels the fallback. Incomplete training rows are excluded.
- The December 31 BOS–PHX noon/afternoon injury source has no complete
  strictly-pre-tip snapshot; that training row is excluded, not marked healthy.
- All March gates/ablations are shown. A descriptive March-best gate is
  explicitly hindsight, never substituted for the February-selected policy.
  Bootstrap intervals for that hindsight gate are conditional and do not
  correct selection bias.

## Source schemas and availability

Full raw column names, five unmodified example rows, unique-value samples,
basic statistics and March 1 odds examples are in `v3_SCHEMA.md`.

`game_odds.csv` has:
`game_id, game_date, odds_date, last_fetched, decimal_home, decimal_away,
moneyline_home, moneyline_away, source_url`.
`odds_date` is **not a homogeneous timestamp**: it contains semantic `open`,
semantic `close`, UTC updates and some missing values. Game dates are midnight
encodings, not tipoff timestamps. `last_fetched` is a retrospective scrape
time, often in 2026 for 2022 games; it never determines historical availability.
American prices are converted directly, avoiding rounded decimal-price error.
NBA game IDs are normalized to ten-character strings.

Only the **semantic opening row** supplies current-game odds predictors.
It has no original publication timestamp; pregame availability is assumed
from the source's opening convention, not proved to an exact second.
Kaggle's supplied **closing** lines alone supply the current closing-market
signal and payout prices; they also lack publication timestamps.
This is an opening/closing-convention backtest, not a fully audited
point-in-time feed replay. No precision is invented for either source.

Current-game open→close movement, largest update move, number of updates and
TeamRankings close are saved in `odds_diagnostics.csv` **only**. Historical
versions use previous game dates only. For steam/update counts, UTC updates
are converted to `America/New_York`, restricted to that source game's local
calendar day, and sorted by their actual timestamps. None can come from the
target game's evolving line history.

Team/player boxscore `*_rank` fields, outcomes and availability flags are
excluded. Some may be retrospective season ranks. Raw completed-game
statistics are shifted by **strictly earlier calendar date**, not merely
one sorted row: multiple games on a date cannot leak into each other.
The advanced player file is scanned in **50,000-row chunks with `usecols`**;
only 2021–22 regular-season rows through March 31 are retained.
It is never loaded whole. Minute weights suppress tiny-appearance rate noise
without looking at future rotation membership.

Official injury `game_time` uses a twelve-hour ET clock without AM/PM.
Here 1–11 mean afternoon/evening, and 12 means noon (matching the cache's
NBA schedule). Select the latest complete snapshot with
`snapshot_ET < scheduled_tip_ET`, not ≤. Reject `NOT YET SUBMITTED`.
A 1PM report is not allowed for a 1PM tip; a complete previous-day report
is then required. Actual snapshot/tip strings are preserved per game.

For exact provenance, routing records the first cached game-clock label. The only amended matchup in this cache is 2022-02-03 MIA@SAS (20:30 versus 19:00 ET). Its selected 17:00 snapshot precedes both; this ambiguity affects no predictor or chosen snapshot in this benchmark.

The separately reproduced v2 implementation has a confirmed noon-routing issue: six 12:00/12:30 ET training games use its game-day 17:00 report. V3 instead uses strictly previous-day snapshots for all six. See `backtest/v3/baseline_temporal_audit.log`; v2's published numbers are reproduced unchanged, not presented as a perfectly leakage-free timing control.

`feature_audit.csv` records each team's previous-source maximum dates,
actual injury snapshot and tip, missing fields and opening-marker availability.
Source fingerprints and coverage counts live in `feature_manifest.json`.

## Feature formulas: all 56 registered candidates

Notation: `H`/`A` are home/away. `R10(x)` is the mean of the team's last up
to ten completed games, at least five nonmissing observations, **all strictly
before the target date**. `E10(x)` uses normalized finite-history EWMA weights
`w_j=2^(-(n-1-j)/10)`, `sum(w*x)/sum(w)`, with at least five observations
for newly computed team/advanced features. It is pandas
`ewm(halflife=10, adjust=True)`. Supplied player EWMAs retain their verified
producer's cold-start rules. Every `*_diff` below is home minus away.
No scaler, imputer or season-wide normalization is fit on test data.

### V2 statistical families, with explicit corrections

Per historical team-game:
`poss=FGA+.44*FTA-OREB+TOV`;
`off_eFG=(FGM+.5*FG3M)/FGA`;
`off_TOVpct=TOV/poss`; `off_FTr=FTM/FGA`;
`ORBpct=OREB/(OREB+opponent DREB)`.
Defensive metrics are the corresponding opponent rates;
`def_ORBpct=opponent OREB/(opponent OREB+DREB)`.
`pace=.5*(own poss+opponent poss)*240/teamlog MIN`.
The corrected offensive-rebound denominator is opposing **DREB**, not v2's
opposing OREB. Season-to-date means divide by the actual number of prior
games; v2's off-by-one denominator is not propagated.

| Feature | Formula |
|---|---|
| `off_eFG_home` | `R10(H off_eFG) - R10(A def_eFG)` |
| `off_eFG_away` | `R10(A off_eFG) - R10(H def_eFG)` |
| `tov_home` | `R10(A def_TOVpct) - R10(H off_TOVpct)` |
| `tov_away` | `R10(H def_TOVpct) - R10(A off_TOVpct)` |
| `orb_home` | `R10(H ORBpct) - R10(A def_ORBpct)` |
| `orb_away` | `R10(A ORBpct) - R10(H def_ORBpct)` |
| `ftr_home` | `R10(H off_FTr) - R10(A def_FTr)` |
| `ftr_away` | `R10(A off_FTr) - R10(H def_FTr)` |
| `pace_diff` | `R10(H pace) - R10(A pace)` |
| `xprod_home` | `off_eFG_home * orb_home` |
| `xprod_away` | `off_eFG_away * orb_away` |
| `elo_diff` | Pre-game home Elo minus away Elo; initialize 1500, K=20, +100 home expectation; update after completed games only. |
| `rest_diff` | Home minus away `(calendar days since previous game - 1)` |
| `b2b_home` | 1 if home last played on the previous date, otherwise 0 |
| `b2b_away` | 1 if away last played on the previous date, otherwise 0 |
| `winpct_diff` | Home minus away wins / number of strictly prior season games |
| `margin_diff` | Home minus away mean points margin over strictly prior season games |

Elo update, reused from v2:
`expected_H=1/(1+10^((E_A-E_H-100)/400))`;
`mult=(abs(score_H-score_A)+3)^.8/(7.5+.006*abs(E_H-E_A))`;
`delta=20*mult*(home_won-expected_H)`;
`E_H += delta`, `E_A -= delta` after the game's features are emitted.
NBA schedule team-games are at most one per date; tests independently ensure
the rolling aggregation excludes all same-day outcomes.

### Current pre-tip injury features (injuries ablation group)

For each listed player, use that player's last up to ten appearances for that
team with `GAME_DATE < target date`: `m_i=mean(MIN)`, `s_i=mean(PTS)`.
Status weights: Out 1, Doubtful .5, Questionable .25, Probable/Available 0.
The most recent actual prior appearance is included, fixing v2's double lag.
Unknown/new players without prior team appearances receive no estimated
minutes/points loss; no future player debut or full-season average is used.

| Feature | Formula |
|---|---|
| `inj_home` | `sum(status_weight_i * m_i/240)` for home |
| `inj_away` | Same for away |
| `inj_diff` | `inj_home - inj_away` |
| `inj_scoring_diff` | Home minus away `sum(status_weight_i * s_i)` |
| `inj_questionable_diff` | Home minus away `sum(1[Questionable] * m_i/240)` |

### Fixed closing market (retained in no-CLV ablation)

For American moneyline `a`, `decimal(a)=1+a/100` if `a>0`, else
`1+100/abs(a)`. `q_side=1/decimal(a_side)`.

| Feature | Formula |
|---|---|
| `market_home` | Kaggle no-vig `q_home/(q_home+q_away)` |
| `market_logit` | `log(market_home/(1-market_home))` |

### Supplied player/rotation features (players group)

For each team, use the precomputed same-team/game row without adding another
shift: it is already strictly pregame. Its verified implementation is
`backtest/v3/player_features.py`; detailed derivation remains in the unchanged
`backtest/v3/README.md`.
Prior-only player productivity is
`rating_per36=36*(E(PTS/MIN)+1.2E(REB/MIN)+1.5E(AST/MIN)+3E(STL/MIN)+3E(BLK/MIN)-E(TOV/MIN))`.
Rotation minutes `m_i` are EWMAs over the prior ten team games, zero for
absences; rank only historically observed players by these minutes.

| Feature | Team quantity, then home minus away |
|---|---|
| `player_team_rating_diff` | Top-eight minutes-weighted mean prior player `rating_per36` |
| `player_top8_share_diff` | Top-eight expected minutes / all eligible expected minutes |
| `player_depth_dropoff_diff` | Productivity of minutes-ranked player #5 minus player #9 |
| `player_pts_margin_ewma_diff` | `E(team points margin)` |
| `player_margin_std_ewma_diff` | Prior EW sample std: `sqrt(sum(w*(margin-mu)^2)/(W-sum(w^2)/W))` |
| `player_plus_minus_ewma_diff` | Top-eight minutes-weighted mean prior player plus/minus EWMA |

### Chunked advanced-player features (players group)

First compute each **completed historical team-game** quantity using that
game's players, then use `R10` and take home minus away. Current-game
participants, minutes, usage and ratings are never consulted.
These are historical team player-boxscore proxies, **not** announced lineups
or causal player impact estimates.

| Feature | Historical team-game quantity, then `R10(H)-R10(A)` |
|---|---|
| `player_adv_ts_diff` | Minutes-weighted mean `ts_pct` |
| `player_adv_pie_diff` | Minutes-weighted mean `pie` |
| `player_adv_net_diff` | Minutes-weighted mean `net_rating` |
| `player_adv_creation_diff` | Minutes-weighted mean `ast_pct` |
| `player_adv_turnover_diff` | Minutes-weighted mean `tm_tov_pct` (source percentage points) |
| `player_adv_usage_hhi_diff` | `sum((MIN_i*usg_pct_i / sum(MIN*usg_pct))^2)` |

### Opening and historical CLV proxies (CLV group)

| Feature | Formula |
|---|---|
| `open_implied_home` | `1/decimal(open American home ML)`, only `odds_date=="open"` |
| `open_implied_away` | Analogous away opening probability, including source vig |
| `open_home` | No-vig opening `open_implied_home/(open_implied_home+open_implied_away)` |
| `prior_clv_diff` | Home minus away `R10(close no-vig team p - open no-vig team p)` from prior games only |
| `prior_steam_diff` | Home minus away `R10(max adjacent absolute home-probability jump)` in prior games, using source opening then local-game-day timestamped updates |
| `prior_updates_diff` | Home minus away `R10(number of timestamped local-game-day updates)` in prior games |

Historical TeamRankings closes enter **only past-game** CLV; current
TeamRankings close never replaces the fixed Kaggle current market.
Largest absolute moves are the same magnitude from either side's no-vig
probability, while historical signed CLV reverses sign for the away team.
These are line-movement proxies; they are not realized CLV on a logged
historical order with a documented execution timestamp.

### Team-boxscore rolling features (boxscores group)

| Feature | Historical game quantity, then `R10(H)-R10(A)` unless stated |
|---|---|
| `box_pace_diff` | Source advanced possessions per 48 minutes, `pace` |
| `box_ts_diff` | Source true-shooting fraction `ts_pct` |
| `box_off_diff` | Source `off_rating`, points per 100 possessions |
| `box_def_diff` | Source `def_rating`, points allowed per 100 possessions; lower is better |
| `box_net_diff` | `off_rating-def_rating` |
| `box_off_proxy_diff` | `100*PTS/(FGA+.44*FTA-OREB+TOV)` |
| `box_def_proxy_diff` | `100*opponent PTS/(FGA+.44*FTA-OREB+TOV)`; same estimated own-possession denominator |
| `box_q_score_std_diff` | Population std of that prior game's four regulation-quarter point totals |
| `box_q_margin_std_diff` | Population std of that prior game's four regulation-quarter plus/minus values |
| `box_q4_margin_diff` | That prior game's fourth-quarter plus/minus |
| `box_three_rate_diff` | `FG3A/FGA` |
| `box_ast_to_diff` | `AST/(TOV+1)`; +1 avoids zero-turnover division |
| `box_net_ewma_diff` | `E10(H net rating)-E10(A net rating)` instead of rolling mean |
| `box_ts_ewma_diff` | `E10(H ts_pct)-E10(A ts_pct)` instead of rolling mean |

## Evaluation definitions

Flat accuracy uses predicted home when `p_home>0.5`, away otherwise, matching
v2's tie convention. Flat P&L is `decimal_price-1` on a correct $1 pick,
otherwise `-1`; ROI = total P&L / dollars staked. No stake is assigned to a
nonbet in gated evaluation. Brier is `mean((p_home-home_won)^2)`.
Predictions retain full precision; CSV `ml_of_side` is the signed **American**
closing moneyline and `decimal_of_side` is its payout conversion.

For `d=p_model_home-p_market_home` and epsilon `{0,.01,.02,.03,.05}`:

- Lean-side: bet home if `d>0`, away if `d<0`, requiring `abs(d)>=epsilon`.
- Winner-side conviction: bet the predicted winner only if its model minus
  no-vig-market probability is strictly positive and at least epsilon.
- No exact-zero edge bets. These definitions do not add a hidden
  vig-inclusive positive-EV filter.

Quarter-Kelly fraction =
`clip(.25*(p_side*decimal-1)/(decimal-1),0,.05)`.
Zero/negative EV gets zero Kelly stake. All games on a calendar date use
the day's starting bankroll; do not recycle an earlier CSV row's win into
an overlapping game. Total daily fractions are capped at one.

Bootstrap: **1,000** seeded percentile resamples of whole Eastern game dates,
preserving within-day correlation. Compute flat accuracy, flat ROI, model
Brier, market Brier, paired Brier difference and fixed-gate ROI/win rate.
The separate hindsight-best-gate CI is explicitly conditional and
selection-biased. Neither bootstrap includes fitting/selection uncertainty,
season shifts, bookmaker limits or execution friction.
