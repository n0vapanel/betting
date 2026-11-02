# brain/ — Continuous-Learning Pipeline for the Multi-Sport Betting Brain

> **Retired as a design (2026-10-28):** the learning loop lives in `telegram_betting_bot_simple/learn/`; this document's promotion gate and ECE alert are retired. See `brain/README.md`.

Status: RESEARCH + DESIGN (this document). Nothing here is wired yet except the directory layout.

Context: the repo already has per-sport walk-forward backtests under `backtest/multisport/<sport>[_v2]/` with build/run scripts and per-game prediction CSVs. The verified scoreboard (`MULTISPORT_REPORT.md`, commit `7d8b0ef` / `f24f8fa`) shows exactly one money lane: **MMA v2 (+40.3% flat ROI, CI +36..+44, 2,843 fights 2020-26)** and one candidate lane: **Soccer v2 O/U 2.5 (+28.7% at the ε=0.05 gate, CI not yet clean)**. Everything else (NBA/NHL/MLB/NFL/tennis) is market-efficient and is kept only as a monitoring lane. This design reflects that: heavy machinery where the money is, cheap monitoring everywhere else.

The core loop, in one line: **collect → drift-check → retrain (locked hyperparams) → promote-or-rollback on newest window → paper-trade ledger → monthly scoreboard** — the "live paper-trade + monthly scoreboard" loop, backed by what production shops actually do.

---

## 0. Design principles (hard rules)

1. **Locked hyperparameters, fresh data.** Production sports-modeling shops (e.g. QuantSport's public MLOps writeups) never re-run grid search on a schedule. Hyperparams are locked by macro-level walk-forward validation; scheduled refits only update feature states and expand the dataset — "give the algorithm current market context, not a new brain." Concretely for us: MMA stays LGBM+XGB soft-vote (validated by the v2 yearly walk-forward); soccer O/U keeps its locked config. Annual walk-forward replay re-validates the locks.
2. **Promote on calibration, not ROI.** Newest-window samples are tiny (MMA ≈ 20-40 fights/month). Brier/ECE are promotable signals at n≈50; ROI is not (CI width swamps it). ROI on the newest window triggers *investigation*, never automatic promotion.
3. **Every detector maps to a deterministic playbook.** The #1 production lesson from drift literature: alerts without a mitigation playbook create alert fatigue. Each alert in §5 has a fixed next action.
4. **Idempotent everything.** Every script must be re-runnable: tmp-file + atomic rename, dedupe-on-read (upsert by event/game id), `.done` stamps in `brain/state/`. No partial artifacts, ever.
5. **No new infra.** One macOS box, launchd + hermes cron. Airflow would be overkill by an order of magnitude (§6.3).

---

## 1. Directory layout

```
brain/
  RESEARCH_PIPELINE.md    # this doc
  collect/<sport>_*.py    # idempotent fetch wrappers (thin; reuse existing fetch scripts where they exist)
  train/train_<sport>.py  # walk-forward refit → versioned artifact → promote/rollback
  eval/eval_<sport>.py    # newest-window Brier/ECE/gate + PSI/CSI drift per sport
  drift/drift_report.py   # aggregate brain/state/drift.json, apply thresholds, emit alerts
  ledger/bets.csv         # append-only bet ledger (paper + real, mode column) — GITIGNORED
  ledger/scoreboard_YYYY-MM.md   # monthly scoreboards — committed to git
  models/<sport>/<YYYY-MM-DD>/   # versioned model artifacts + feature manifest + hyperparams json
  state/registry.json     # current model version per sport, last-fetch stamps, source hashes
  state/drift.json        # latest drift metrics per sport
  logs/                   # per-job stdout/stderr
  launchd/*.plist         # launchd job definitions (installed to ~/Library/LaunchAgents)
  run.sh                  # venv + env wrapper (JAVA_HOME, PYTHONPATH hygiene), logs to brain/logs
```

Conventions:
- **Raw data stays where it already lives** — `backtest/multisport/<sport>[/data]/` (already gitignored). `brain/collect/` writes there, it does not duplicate a 127 MB moneypuck CSV.
- **Models, ledger, state, logs are gitignored.** Scripts + this doc + scoreboards are committed.
- Shared venv: `~/.hermes/cache/scratch/msports-venv/bin/python` (verified 2026-10: py 3.11.15, numpy 2.4.6, lightgbm 4.7.0 — health-checked, no cp314 contamination).
- **Do not run the venv python through a parent process that injects Hermes's PYTHONPATH** (points at py3.14 site-packages). Always run via `terminal`/launchd/cron, never via an `execute_code` subprocess. `run.sh` sets `PYTHONDONTWRITEBYTECODE=1` and clears PYTHONPATH.

---

## 2. Data collection: cadence per source and cheapest refresh

Verified publish cadences (sources checked 2026-10):

| Sport | Source | Publisher cadence | Our refresh | Cheapest method |
|---|---|---|---|---|
| Soccer | football-data.co.uk CSVs (`mmz4281/<season>/<code>.csv`) | ~weekly, after each matchweek; per-season files rewritten in place | **Mon 06:00** | Full re-download of current-season CSVs (KB-scale). No incremental protocol exists — the file IS the delta. Existing fetcher: `backtest/multisport/soccer_v2/fetch_footballdata.py` |
| Soccer | understat `main/getLeagueData/<slug>/<season>` gzip JSON | per-match (daily in season) | **Mon 06:00** | Full league-season re-download (small). Existing fetcher: `fetch_understat.py`. Join on team name + date ±1 day as v2 does |
| NHL | moneypuck `playerData/careers/gameByGame/all_teams.csv` | next morning, daily in season ("usually doesn't update until the next morning") | **Tue 06:15** | Full 127 MB re-download (weekly is fine; daily would also work but wastes bandwidth). Filter `season==start_year` |
| NHL | `api-web.nhle.com/v1/gamecenter/{gameId}/boxscore` (gameId = moneypuck gameId = NHL gameId) | live/next-day per game | **Tue 06:15** | Incremental: fetch only gameIds not present in `nhl_v2/data/boxscores/`. Existing: `nhl_v2/data/download_boxscores.py` |
| NBA | `nba_api` (stats.nba.com) game/player logs | after each game | **daily 06:30 (Tue–Sun, in season)** | Incremental per-season append; `LeagueGameFinder(season_nullable=...)`; per-team loop on 429s. Existing: `backtest/v2/data.py` |
| NBA | `nbainjuries` (official injury reports) | official report files 5PM ET on game days; day-before has 'NOT YET SUBMITTED' placeholders | **daily 18:30 ET (game days)** | Incremental snapshot append (1PM early-tip snapshot, 5PM official). **Requires `export JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home`** (jpype) |
| NFL | nflverse releases via `nfl_data_py` | pbp rebuilt nightly after game days; **Thursday's files are the cleanest** (stat corrections land Mon–Wed); schedules every 5 min; injuries/rosters daily 7AM UTC | **Thu 06:00** | Full season parquet re-import (package overwrites its cache — this is the documented in-season refresh pattern). Existing: `nfl_backtest.py` calls `import_weekly_data` / `import_schedules`, caches in `.cache/` |
| NFL ⚠️ | — | `nfl_data_py` repo **archived Sep 2025**, deprecated in favor of **`nflreadpy`** | — | **Migration debt:** new code should use `nflreadpy` (Polars → `.to_pandas()`). Same nflverse release files underneath |
| MLB | `statsapi.mlb.com/api/v1/schedule` | daily | **daily 06:00 (Mar–Nov)** | Incremental by date range. Existing: `mlb_v2/fetch_schedule.py` |
| MLB | Statcast | daily | **daily 06:00 (Mar–Nov)** | 4-day chunks + dedupe with inclusive date filters (25k-row API cap). Existing: `mlb_v2/fetch_statcast.py` |
| MLB | Retrosheet `gl<year>.zip` | **annual** (season released in offseason) | once at season start | Full zip re-download when the new season posts. Existing: `mlb/parse_gamelogs.py` + `glfields.txt` |
| Tennis | tennis-data.co.uk `2021.xlsx…` | weekly publisher cadence, but **live site WAF-blocks bots** (serves parked-domain HTML) | **Mon 06:45 best-effort** | Full season xlsx via Wayback (`web.archive.org/web/20230514150435id_/http://www.Tennis-Data.co.uk/2021/2021.xlsx` pattern) or curl+UA; **tolerate misses** — this lane is degraded by design |
| Tennis | Kadantte/tennis_atp (Sackmann mirror; upstream deleted) | periodic (through 2026) | **Mon 06:45 best-effort** | Per-year CSVs from raw.githubusercontent.com **with a User-Agent** (404 without one) |
| MMA | Kaggle `mdabbert/ultimate-ufc-dataset` `ufc-master.csv` | **irregular, post-event** (changelog-driven; also mirrored at github.com/shortlikeafox/ultimate_ufc_dataset, updated through 2026) | **Tue + Fri 07:00 (cheap version check)** | Full `ufc-master.csv` re-download via kaggle CLI (`kaggle datasets download -d mdabbert/ultimate-ufc-dataset -f ufc-master.csv`; token at `~/.kaggle/access_token`); GitHub raw mirror as fallback. ~tens of MB, re-download weekly is trivially cheap |

**General refresh rule:** where the source rewrites a file in place (football-data, understat, moneypuck, nflverse), full re-download IS the incremental approach — diffing is more code than bandwidth. Where the source is a per-event API (NHL boxscores, MLB Statcast, NBA logs), fetch only unseen keys and dedupe on read. Record per-source freshness (`rows_added`, source hash/etag, latest event date) in `brain/state/` so a silent source stall trips a drift alert instead of quietly aging the model.

---

## 3. Retraining architecture

### 3.1 Cadence per sport

| Sport | Lane | Feature-state refresh | Model refit | Refit trigger extras |
|---|---|---|---|---|
| **MMA** | MONEY | after every event lands in data (weekly collect) | **monthly (1st Monday 07:00)**, plus event-triggered if ≥10 new fights since last refit | Manual refit if gate ROI negative for 3+ consecutive events (mirrors the community-standard UFC rule: retrain monthly ~every 8-10 events, or immediately when ROI drops 3 events running) |
| **Soccer O/U** | candidate | weekly (Mon, same job as collect) | monthly (1st Monday) | The +28.7% gate must be re-confirmed on 2022-23/2023-24 before promotion; until then it is a watched lane |
| NBA / NHL / MLB / NFL / tennis | monitor | weekly | monthly in-season; none in offseason | Refit exists to keep drift monitors meaningful; no betting money here |

**Why not more frequent?** Weekly refits add noise risk without adding games (NBA/NHL add ~40-80 games/week; the QuantSport pattern decouples data state from architecture). Monthly matches the natural event cycle (one MMA card per ~2 weeks; one soccer matchweek per week) and keeps every refit reproducible.

**Warm-starting: not needed now.** All datasets are small (MMA 7k fights, soccer ~9k matches, NHL 1.4k games/season) — retraining from scratch takes minutes. Adopt LGBM `init_model` warm-start only if a refit exceeds ~30 min wall-clock (NFL is the only plausible candidate). Reproducibility beats compute at this scale.

### 3.2 Refit protocol (train/train_<sport>.py)

1. Load data with cutoff = last settled game/fight date (never today — settlement lag).
2. Rebuild features with the **existing feature builder** (same code path as the backtest — one canonical builder for train and predict; the repo already enforces this pattern with `frozen_protocol.json` in v3_extended).
3. Train candidate with **locked hyperparams** from the walk-forward-validated config.
4. Write versioned artifact `brain/models/<sport>/<YYYY-MM-DD>-<short_hash>/` containing: model file, feature manifest (columns + builder git hash), hyperparams json, train-window summary (n, date range, Brier on last-season holdout).
5. Run the promotion gate (§3.3) against the incumbent; update `brain/state/registry.json` (`current_version`), repoint `models/<sport>/latest` symlink, or roll back (keep incumbent untouched). **Never delete old versions.**
6. `registry.json` per sport: `{current_version, created_at, train_until, refit_history[{version, brier_newest_window, promoted|rolled_back}]}`.

### 3.3 Promotion gate (evaluate-on-newest-window)

For each refit, score the newest settled window (last 30 days, or last 3 events for MMA) with BOTH candidate and incumbent, plus the market as reference:

- **Promote** iff candidate newest-window Brier ≤ incumbent Brier AND candidate logloss ≤ incumbent logloss AND candidate accuracy within −1pp of incumbent.
- **Roll back** (keep incumbent) otherwise — including ties, because the incumbent already passed this gate.
- ROI/gate P&L on the newest window is **reported, never promoted on** (n too small). If incumbent gate ROI is negative 3 events running, that fires the manual investigation job, not auto-promotion.
- Annual: replay the full yearly walk-forward to re-validate locked hyperparams and to re-check the two money lanes' CI story (this is the existing v2 protocol, just re-run on new data).

This is the production-standard pattern: analysts.cloud's NFL playbook (nightly light retrain / weekly full retrain / drift-triggered retrain / shadow evals), QuantSport's weekly-refit-with-locked-architecture, and hiro.solutions' rollback-to-validated-checkpoint rule all converge on the same shape.

---

## 4. Ledger: paper-trade first, CLV as the fast signal

### 4.1 Schema — `brain/ledger/bets.csv` (append-only, one row per bet)

```
ts_placed, ts_settled, mode, sport, market, event_id, selection,
model_version, p_model, p_market_devig, odds_taken_dec, closing_odds_dec,
stake, result, pnl, clv_pp
```

- `mode`: `paper` | `real`. One file, one pipeline; real-money rows are the same schema with a flag. **Paper-trade UFC for several weeks before any real stake** — this is the gating requirement already recorded in `MULTISPORT_REPORT.md`.
- `result`: `W | L | P | V` (push/void). Corrections are **new rows with a `void`-ed original, never edits** (2+2 driveline-template convention; keeps the ledger auditable).
- Append-only means the settle job (§6.2) fills `ts_settled/result/pnl` on open rows by joining `event_id` to collected results — same join keys the backtests already use.

### 4.2 CLV computation

```
clv_pp = (1/odds_taken_dec − 1/closing_odds_dec) × 100   # implied-prob points, de-vig the close
```

- Record `closing_odds_dec` from the **same source** as `odds_taken` (or a sharp, low-margin book), else you measure book-margin gaps, not line value. For MMA the closing source is BestFightOdds (same as the dataset's odds); soccer uses football-data closing (B365/Avg) columns.
- Summary stats that matter (from r/sportsbook's best long-run trackers and Apify's CLV grader): **mean CLV**, **beat-the-close rate** (share of bets with `clv_pp > 0`), and CLV broken down by sport/market/`time_to_start`. CLV is the *leading* indicator — it validates edge **before** outcomes settle, which is exactly why it's the right gate for promoting MMA from paper to real.
- Review cadence: monthly (FootyMetrics + the best tracker threads agree: monthly review, never per-bet reactions). Sample-size honesty: 200 paper bets ≠ proven edge; but 200 bets with positive mean CLV and negative P&L mean "process ahead of market, variance behind" — opposite advice to 200 bets with negative CLV and positive P&L.

### 4.3 Monthly scoreboard

`eval/*.py` + the ledger feed `brain/ledger/scoreboard_YYYY-MM.md`, committed to git, extending the existing `MULTISPORT_REPORT.md` board style: per sport: n, pick acc, flat ROI (date-clustered bootstrap CI where n permits), **Brier model vs market**, mean CLV, beat-close rate, verdict. The hermes cron job (§6.2) generates and posts it.

---

## 5. Drift detection (probability models)

Two independent layers, run weekly for money lanes (MMA, soccer) and monthly for monitor lanes. All outputs land in `brain/state/drift.json`; `brain/drift/drift_report.py` applies thresholds.

### 5.1 Feature drift — PSI / CSI

- **PSI** on continuous features, **CSI** on categoricals (weight class, home/away, gate tier). Reference distribution = the locked training window; current = last 30 days of settled games. 10 quantile/equal-width bins (standard: 10 bins, both distributions binned on reference edges; add small epsilon to zero bins).
- **Thresholds (standard production bands, confirmed across sources incl. Harvard's ML-Systems book):**
  - PSI < 0.10 → stable, no action
  - 0.10 ≤ PSI < 0.25 → WARNING: log, include in weekly digest
  - PSI ≥ 0.25 → ALERT → **playbook: queue an out-of-cycle refit** (drift-triggered retrain), then re-check; if PSI stays ≥ 0.25 after refit, open an investigation (source change, roster/rule change, broken fetch)
- Also track **KS statistic on the model score distribution** (current vs reference): KS > 0.2 = drift alert (analysts.cloud production threshold).
- Track max-PSI + count of features over 0.1 (a broad shift across many features = source-level break, not sport-level change).

### 5.2 Performance drift — Brier / logloss / ECE

Per sport, monthly windows, tracked as a time series in `state/drift.json`:

- **Brier (model), Brier (market), gap = Brier_model − Brier_market.** The repo's own health metric; every backtest already reports it. Alert when the 30-day gap exceeds the trailing 6-month mean gap by **+0.02** — the model is losing ground *relative to the market*, which is the only comparison that pays.
- **ECE (expected calibration error):** alert at ECE > 0.03 with n ≥ 200 in window (the community threshold is 0.015 for high-volume systems; at our 20-40 bets/month per sport, 0.03 with the n-floor avoids noise alarms). Playbook: recalibrate (Platt/isotonic re-fit on recent window) before refitting the whole model.
- **Gate health (money lanes only):** rolling gate ROI over last 3 events; negative 3 consecutive → manual review + refit (UFC community rule).
- Every alert maps to a playbook (rollback / refit / recalibrate / investigate source) — never a bare notification.

### 5.3 The three drift types, mapped

Covariate drift → PSI/CSI. Label drift (season-phase changes in target distribution) → outcome-rate chart vs market-implied rate. Concept drift (feature→target relationship changing) → Brier-gap/ECE. Response per type is fixed in the playbook table; no ad-hoc reactions.

---

## 6. Orchestration: launchd + hermes cron (NOT airflow)

### 6.1 Why not Airflow

The literature is unanimous for our scale: Airflow exists for 20+ interdependent DAG tasks, retries, backfills, and an ops team. For one box running ≤10 independent jobs with no interdependencies, cron/launchd wins on everything that matters (zero infra, instant start, easy debugging, no metadata DB to maintain) — the commonly cited cost figure is 70-90% overhead for basic use. The hybrid pattern (start with cron, escalate only if real orchestration needs appear) is the standard recommendation (QuestDB's pipeline comparison, SO guidance). Decision: **launchd for deterministic scripts, hermes cron for the two agent jobs.** Do not add Airflow/Dagster/Prefect unless job interdependence actually appears.

### 6.2 The job table

**launchd** (deterministic; plists live in `brain/launchd/`, installed to `~/Library/LaunchAgents/` with `launchctl bootstrap gui/$(id -u) <plist>`). All jobs call `brain/run.sh <script> <args>`, which sets the venv, clears PYTHONPATH, exports JAVA_HOME, and logs to `brain/logs/`. Times are local (EDT).

| Job (plist name) | Schedule (StartCalendarInterval) | Command |
|---|---|---|
| `com.brain.collect.soccer` | Mon 06:00 | `backtest/multisport/soccer_v2/fetch_footballdata.py && backtest/multisport/soccer_v2/fetch_understat.py` |
| `com.brain.collect.nhl` | Tue 06:15 | `curl -fsSL -o backtest/multisport/nhl_v2/data/all_teams.csv https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv && backtest/multisport/nhl_v2/data/download_boxscores.py` |
| `com.brain.collect.nba` | daily 06:30 (Tue–Sun, Oct–Jun) | `backtest/v2/data.py` (nba_api game/player logs) |
| `com.brain.collect.nba.injuries` | daily 18:30 ET (game days) | `backtest/v2/data.py --injuries` (nbainjuries; 1PM early-tip snapshot) |
| `com.brain.collect.nfl` | Thu 06:00 | `brain/collect/nfl_refresh.py` (import_weekly_data/import_schedules; migrate to nflreadpy — §2 note) |
| `com.brain.collect.mlb` | daily 06:00 (Mar–Nov) | `backtest/multisport/mlb_v2/fetch_schedule.py && backtest/multisport/mlb_v2/fetch_statcast.py` |
| `com.brain.collect.mlb.retrosheet` | Jan 15 yearly | `curl -fsSL -o backtest/multisport/mlb/gl<year>.zip https://www.retrosheet.org/gamelogs/gl<year>.zip && backtest/multisport/mlb/parse_gamelogs.py <year>` |
| `com.brain.collect.tennis` | Mon 06:45 | `brain/collect/tennis_refresh.py` (best-effort: tennis-data via Wayback/curl+UA + Kadantte mirror CSVs with UA) |
| `com.brain.collect.mma` | Tue + Fri 07:00 | `brain/collect/mma_refresh.py` (`kaggle datasets download -d mdabbert/ultimate-ufc-dataset -f ufc-master.csv -p backtest/multisport/mma_v2/ --force`; GitHub mirror fallback) |
| `com.brain.train.monthly` | 1st Mon 07:30 | `for s in mma soccer o_u nba nhl mlb nfl tennis; do brain/train/train_$s.py; done` (each does its own promote/rollback) |
| `com.brain.eval.drift` | Sun 20:00 | `brain/eval/eval_all.py && brain/drift/drift_report.py` (writes `state/drift.json`, sets exit 2 on any ALERT) |
| `com.brain.ledger.settle` | daily 22:00 | `brain/ledger/settle.py` (join open bets to results, fill result/pnl/clv) |

Example plist (in `brain/launchd/com.brain.collect.mma.plist`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.brain.collect.mma</string>
  <key>ProgramArguments</key><array>
    <string>/Users/x/Desktop/x/Coding Projects/betting bot/brain/run.sh</string>
    <string>collect/mma_refresh.py</string>
  </array>
  <key>StartCalendarInterval</key><array>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>StandardOutPath</key><string>/Users/x/Desktop/x/Coding Projects/betting bot/brain/logs/mma_collect.log</string>
  <key>StandardErrorPath</key><string>/Users/x/Desktop/x/Coding Projects/betting bot/brain/logs/mma_collect.err</string>
</dict></plist>
```

**hermes cron** (agent jobs — reads state, writes reports, delivers digests). Note: hermes cron runs have a 3-minute hard interrupt, so only the two light digest jobs go here; heavy collect/train stays on launchd.

| Job | Schedule | What it does |
|---|---|---|
| `brain-weekly-digest` | `every monday 9am`, workdir = repo | Read `brain/state/drift.json` + ledger tail. Post a compact per-sport health table (PSI max, Brier gap, ECE, gate P&L, rows added by collect) and surface any ALERT with its playbook. |
| `brain-monthly-scoreboard` | `0 9 1 * *`, workdir = repo | Generate `brain/ledger/scoreboard_YYYY-MM.md` from ledger + eval outputs, update the board section of `MULTISPORT_REPORT.md`, post the scoreboard. |

`hermes cron add "every monday 9am" --name brain-weekly-digest --workdir "/Users/x/Desktop/x/Coding Projects/betting bot" ...` (see `hermes cron --help` / `/cron`; per-job `no_agent` not needed here — these two ARE agent jobs).

### 6.3 Idempotency & failure rules (enforced by `run.sh` conventions)

- Fetch scripts download to `*.tmp`, then atomic `mv` into place; dedupe on read by event/game id (the repo's existing upsert patterns).
- Each job writes `brain/state/last_run_<job>.json` (`{ts, exit, rows_added, source_hash}`); the digest job flags any collect job whose `rows_added` is 0 for 2+ consecutive runs while the sport is in season (silent source stall = the most common real-world failure).
- launchd gives exit-code-aware scheduling and one log pair per job; no retry needed because every job is idempotent — rerunning is the retry.
- Backfill = run the collect script manually with a date range; nothing to orchestrate.

---

## 7. Rollout plan (in order, each step independently valuable)

1. **Phase 0 (now):** `brain/` layout exists. Write `run.sh` (venv+env wrapper). Create `bets.csv` header.
2. **Phase 1:** `mma_refresh.py` + `mma` ledger flow — run the UFC picker (`backtest/multisport/mma/ufc_picker.py`) on every real card, log picks to `bets.csv` (paper), settle with collected results. **This is the gate to real money** (paper-trade for several weeks, watch mean CLV + beat-close rate, then review).
3. **Phase 2:** soccer O/U follow-up (2 more seasons per `MULTISPORT_REPORT.md` "Next") + weekly drift for MMA/soccer.
4. **Phase 3:** the other collect jobs + monthly monitor refits + both hermes cron jobs.

---

## 8. Known gaps & risks (be honest about these in every scoreboard)

- **Tennis live feed is degraded** (WAF-blocked site, Wayback-dependent history). Tennis stays a monitor lane with best-effort refresh until a current-season odds source appears.
- **mdabbert update cadence is irregular** — the Tue/Fri version check catches events within ~4 days; the GitHub mirror (shortlikeafox) has been more recent than Kaggle at times. Check both.
- **nfl_data_py is archived** (Sep 2025); pin 0.3.3 until `nflreadpy` migration is done, then switch.
- **Hermes cron 3-min interrupt** — never put training/fetching there; launchd owns deterministic work.
- **MMA odds are "closing per dataset docs"** — the paper-trade ledger is also the audit that proves whether the +40% survives real closers. CLV column exists precisely to catch this.
- **Small-sample honesty:** monthly windows are too small for ROI promotion decisions; the scoreboard must always show bootstrap CIs or refuse to draw conclusions.

---

## Sources (verified 2026-10)

- QuantSport MLOps post (weekly retrain, locked walk-forward hyperparams, health-guard deployment blocks) — linkedin.com/posts/quantsport-ai
- analysts.cloud "Operationalizing Game-Day Predictive Models" (retrain cadence: nightly light / weekly full / drift-triggered; KS>0.2 and 10%-calibration-drop alert thresholds; shadow evals)
- hiro.solutions "Deploying Self-Learning Prediction Models" (prequential eval, Brier/logloss/ECE, ADWIN/DDM, rollback-to-checkpoint, "every drift detector maps to a playbook")
- wagerproof.ghost.io "When to Retrain Betting Models" (ECE>0.015 alert, weekly KS drift tests, CLV as leading edge indicator, monthly ~8-10-event UFC retrain rule, 3-events-negative-ROI trigger)
- sportbotai.com "Reddit's Sports Betting Spreadsheet" (closing-line column as the upgrade, implied-prob CLV math, beat-close rate, same-book closing)
- footymetrics.com "How to track your bets" (7-field minimum schema, monthly review cadence)
- Apify Bet CLV Tracker docs (no-vig fair closing prob, clvPercent, beatClosingLine, EV-at-close fields)
- forumserver.twoplustwo.com Driveline tracker thread (bet/book/date/price/line/risk/grade/bankroll/Kelly schema; grade-sequentially caveat)
- Harvard CS249r "Machine Learning Systems" (PSI bands: warn >0.1, critical >0.25; incident severity mapping P0–P3)
- nflverse data schedule vignette (pbp nightly, Thursday cleanest, injuries daily 7AM UTC) + nfl_data_py README (cache-overwrite weekly refresh) + nflanalytic.com (nfl_data_py archived → nflreadpy)
- moneypuck freshness: hubot-hockey issue "doesn't update until the next morning"; PuckAPI comparison "updated every morning in season"
- Kaggle mdabbert/ultimate-ufc-dataset page + GitHub shortlikeafox/ultimate_ufc_dataset (changelog cadence, upcoming.csv)
- QuestDB blog "Automating Workflows" + LinkedIn Airflow-vs-cron threads + SO 79959389 (cron for ≤few jobs: 70-90% less overhead, MAILTO/exit-code discipline, jitter)
- Repo-internal: `MULTISPORT_REPORT.md`, `betting-bot-ops` skill (verified per-sport sources, env gotchas: JAVA_HOME/nbainjuries, msports-venv, PYTHONPATH hygiene, lgbm eval_metric)
