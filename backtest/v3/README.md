# NBA 2021–22 player features

## Run and output

From `/Users/x/Desktop/x/Coding Projects/bb-v3-data`:

```sh
.venv/bin/python backtest/v3/player_features.py
```

The script reads only these existing caches under
`/Users/x/Desktop/x/Coding Projects/betting bot/backtest/cache/`:

- `nba_playerlog_2021_22.csv`: player box scores and minutes.
- `nba_teamlog_2021_22.csv`: team points margins (`PLUS_MINUS`).
- `nba_logs_2021_22.csv`: scheduled team-game identifiers and dates only.

It writes `backtest/v3/player_features_2021_22.csv`: one row per scheduled
team-game with `GAME_DATE >= 2021-11-01`, sorted by date, game ID, and team.
The supplied regular-season schedule has 2,274 such rows (1,137 games),
through 2022-04-10. It prints ten rows, row count, date range, null counts,
numeric ranges/means, and the no-lookahead assertion result.
`GAME_ID` is preserved as a string, including leading zeros; read it with
`pd.read_csv(path, dtype={"GAME_ID": str})` when joining.

## Common EWMA definition

All exponential calculations use a **10-game half-life**, fixed in advance.
For an oldest-to-newest sequence of `n` eligible games, define

```text
q = 2^(-1/10)
w_j = q^(n - 1 - j), j = 0, ..., n - 1
E(x) = sum(w_j * x_j) / sum(w_j)
```

These are normalized, finite-history weights, equivalent to
`pandas.Series.ewm(halflife=10, adjust=True).mean()`, not the
`adjust=False` initialization convention. An observation ten eligible games
older has half the weight. The first observation initializes the mean;
there is no season-wide prior, fitted scaling, or future backfill.
Game age, not elapsed calendar days, controls decay.

## Player rolling ratings

Players are grouped by **`(TEAM_ABBREVIATION, PLAYER_ID)`**. Only appearances
for that team on dates strictly before the target date are included.
A traded player's other-team games never enter this team's player rating.
Player means use all available prior appearances, not just the last ten.

For each of `PTS`, `REB`, `AST`, `STL`, `BLK`, `TOV`, `FG_PCT`, and `FG3_PCT`:

```text
X_per_game_ewma = E(X)
X_per_min_ewma = sum(w_j * X_j / MIN_j for MIN_j > 0)
                 / sum(w_j for MIN_j > 0)
```

Thus the per-minute value is the **EWMA of game-level rates**, not the ratio
`E(X) / E(MIN)`. Zero-minute appearances still count in the game's age clock
and per-game means, but are excluded from rate numerators and denominators;
no division by zero occurs. A player with no positive-minute history has
undefined (`NaN`) per-minute ratings and cannot receive rotation weight.
Minutes are the cache's reported minutes, including its rounding.

Missing `FG_PCT`/`FG3_PCT` is set to zero **only** when `FGA`/`FG3A` is zero.
Missing percentages with nonzero attempts, other nonfinite inputs, invalid
percentage ranges, and duplicate input keys fail explicitly.
Percentages are fractions in `[0, 1]`; their per-minute versions literally
mean fraction/minute. These diagnostics are not conventional shooting-
efficiency estimators and are not mixed into a counting-stat composite.

Each player also has `player_min_ewma = E(MIN)` and
`plus_minus_ewma = E(PLUS_MINUS)` over their prior team-specific appearances.

The fixed scalar productivity rating is:

```text
rating_per36 = 36 * (PTS_per_min_ewma
                    + 1.2 * REB_per_min_ewma
                    + 1.5 * AST_per_min_ewma
                    + 3 * STL_per_min_ewma
                    + 3 * BLK_per_min_ewma
                    - TOV_per_min_ewma)
```

This is a transparent box-score productivity index, not an estimated impact
metric or a trained model. The weights and 36-minute scale do not depend on
the backtest outcomes. Shooting percentages remain separate diagnostics.
`player_rolling_ratings(history, target_date)` returns these per-player
values and **asserts** that every supplied history date is strictly prior.

## Rotation and team aggregation

For each team and target date, take its last **up to ten completed team
games strictly before that date**. For each previously observed player,
construct a minutes vector on those games, using zero for absences.
Apply `E(MIN)` within this ten-game window to obtain `m_i`, the player's
rotation minutes estimate. These are team-game weights, unlike the
appearance-based `player_min_ewma` above.

Only players with `m_i > 0` are eligible. Rank by descending `m_i`, breaking
ties by ascending string `PLAYER_ID`. Players absent for the entire window
are excluded, even if they played earlier in the season. The target game's
actual participants are never consulted. Let `R` be all eligible players,
`T` the first up to eight, and `M_T = sum(m_i for i in T)`.

| CSV column | Exact definition |
| --- | --- |
| `TEAM_ABBREVIATION` | Target team from the schedule. |
| `GAME_DATE` | Target calendar date, `YYYY-MM-DD`. |
| `GAME_ID` | Target string game identifier. |
| `team_rating` | `sum(m_i * rating_per36_i for i in T) / M_T`. |
| `top8_share` | `M_T / sum(m_i for i in R)`; concentration of expected minutes, bounded by 0–1. |
| `depth_dropoff` | `rating_per36` of rotation player #5 minus that of #9, using **minutes rank**, not rating rank. May be negative. |
| `pts_margin_ewma` | `E(team PLUS_MINUS)` over all prior team games. Team-log `PLUS_MINUS` is team PTS minus opponent PTS. |
| `margin_std_ewma` | Exponentially weighted **sample** standard deviation of those prior team margins; formula below. |
| `plus_minus_ewma` | `sum(m_i * player_plus_minus_ewma_i for i in T) / M_T`; distinct from the team's margin EWMA. |

The same top-eight minutes weights are applied separately to every player
per-game and per-minute EWMA, producing `lineup_X_per_game_ewma` and
`lineup_X_per_min_ewma`.

The player-level minutes/pace proxy is also computed:

```text
team_game_total_MIN_g = sum(MIN for every team player in prior game g)
team_min_ewma = E(team_game_total_MIN)
```

It uses **all** prior team games and all players, not just the top eight.
This mainly reflects regulation/overtime duration and rounding, not true
possession pace. To keep the requested **exact nine-column CSV schema**,
`team_min_ewma` and the detailed lineup aggregates are available in the
DataFrame returned by `build_player_features(...)`, but are not serialized
to this CSV. No extra feature CSVs are written.

## Margin consistency and missing history

For prior team margins `x_j`, with the weights above, let `W = sum(w_j)`,
`W2 = sum(w_j^2)`, and `mu = E(x)`. Then:

```text
margin_std_ewma = sqrt(sum(w_j * (x_j - mu)^2) / (W - W2 / W))
```

This is `ewm(halflife=10, adjust=True).std(bias=False)`.
With fewer than two historical team games the standard deviation is `NaN`.
With fewer than nine eligible rotation players `depth_dropoff` is `NaN`.
With no prior games, all numeric features are `NaN`; smaller nonempty
rotations use the available players. No missing feature is filled using
future data. Mismatched historical player/team game coverage raises an
error instead of silently treating a missing box score as a zero.

## Strict no-lookahead guarantees

1. Dates are normalized to calendar dates before comparison, so even an
   earlier time on the target date is ineligible.
2. **Both** source histories are explicitly filtered with
   `GAME_DATE < target_date` before any statistic or rotation is computed.
3. Candidate players, minutes shares, ranking, player ratings, team minutes,
   and team-margin moments all derive exclusively from these histories.
   No full-season roster, target-game player list, injury hindsight, or
   opponent/target score is used.
4. Schedule scores and outcomes are discarded; only the three key columns
   determine which rows are emitted. Historical October games are retained
   as warm-up even though output starts on November 1.
5. Each returned row includes in-memory `player_history_max_date` and
   `team_history_max_date`. `assert_no_lookahead(features)` asserts that both
   are missing or **strictly earlier** than that row's game date.
   These provenance columns are not written to the fixed-schema CSV.
6. Regression tests replace, add, remove, and shuffle current/future player
   and team observations and change target scores/outcomes. Every returned
   value must remain exactly unchanged. Tests also cover same-date games,
   intraday timestamps, traded players, stale rotation members, zero minutes,
   ten-game half-life, hand-computed formulas, and cold starts.
7. All source caches are opened read-only; the script writes only the
   feature CSV beside itself. It makes no network calls or git calls.

This is a **historical rotation proxy**, not a known pregame lineup.
Injury absences, new signings, and departures cannot be inferred from
future appearances: a departed player can linger until their last positive
minutes leave the ten-game window, and a newcomer cannot enter until after
their first observed team appearance.

## Focused verification

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest \
  -p no:cacheprovider backtest/v3/test_player_features.py -q
```

The flags keep bytecode and pytest-cache writes out of the rest of the repo.
