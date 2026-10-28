# brain/: research only (retired as a design)

**The brain's learning loop now lives in the app: `telegram_betting_bot_simple/learn/`.** Follow that design, not this
folder's. Nothing here runs, and two of its rules are retired:

- **The promotion gate** in `RESEARCH_PIPELINE.md` §3.3 ("promote iff candidate Brier ≤ incumbent and log loss ≤
  incumbent") has no significance test. Two near-identical candidates differ by noise, so it would promote about half
  of them. The app promotes a small nudge only when it is non-inferior out of sample (the upper end of the day-block
  95% range of the change in log loss is at most +0.0005, on at least 300 unseen games) and a structural change only
  when it is better (one-sided Diebold-Mariano with the Harvey-Leybourne-Newbold correction, Holm across the week's
  candidates, the range entirely below 0, then two weeks in shadow). Every change rolls back on its own when it gets
  worse (after 200 games, or a CUSUM alarm), and the owner can undo any change.
- **The ECE alert** ("ECE > 0.03 with n ≥ 200") sits below the noise floor: a perfectly calibrated forecaster scores a
  median ECE of 0.070 at n = 200 and 0.032 at n = 1,000. The app shows calibration as a CORP reliability curve with its
  consistency band and the calibration slope and intercept with 95% ranges.

The baselines `BRAIN_BLUEPRINT.md` and `RESEARCH_PIPELINE.md` quote for MMA and soccer come from two leaks (MMA's
`streak_pre` and soccer v2's swapped closing columns); see the correction at the top of `../MULTISPORT_REPORT.md`.
Once both are fixed, no offline backtest beats the closing market.

Where to look instead:

- `learn/history.py`: the gentle nightly backfill of old games (football-data.co.uk, nflverse, ESPN) and the grader's
  live capture, point in time;
- `learn/evaluate.py`: walk-forward scores against the sharp market and simple baselines (log loss, Brier, count log
  score, CRPS, PIT, calibration);
- `learn/tune.py` and `learn/registry.py`: the allow-listed settings, the refits, the gates, rollbacks and safe mode;
- `learn/loop.py` and `learn/analyst.py`: the schedule (nightly, weekly, monthly) and the linked AI, which proposes
  bounded changes judged only on games played after its proposal and never forecasts old games;
- the app's Brain progress page (`GET /api/brain/progress`) and the bot's `/brain`.

The research documents, scripts and the `ledger.py` / `metrics.py` spine are kept as they were.
