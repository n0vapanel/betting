# BRAIN BLUEPRINT — the forever-learning layer

Status: spine built (ledger + metrics + collect), research inbound (3 lanes: GitHub recon,
techniques, pipeline). Sections marked **[PENDING-RESEARCH]** get filled when those land.

## Goal

One continuously improving multi-sport prediction system:
**collect → retrain → evaluate → bet → settle → scoreboard → repeat.**
No manual re-runs: the brain re-learns on its own cadence and gets measurably better
or worse over time, tracked in a single ledger + metrics history.

## Loop (the spine — DONE)

1. **collect** — `brain/collect/fetch_all.sh` refreshes all sports' data (idempotent).
2. **retrain** — per-sport retrain scripts (walk-forward refit; MMA yearly-refit pattern).
   [PENDING-RESEARCH: cadence per sport from pipeline lane]
3. **eval** — newest-window walk-forward evaluation vs closing lines, appends to
   `brain/metrics_history.jsonl` via `brain/metrics.py`.
4. **bet** — `backtest/multisport/mma/ufc_picker.py` + per-sport pickers; every bet logged
   in `brain/ledger.jsonl` (paper or live) via `brain/ledger.py`.
5. **settle** — results filled in after events; report computes ROI/winrate/avg-edge/CLV.
6. **drift watch** — `metrics.py trend <sport>` shows whether accuracy/ROI/Brier are
   degrading across retrains; degradation triggers a feature/architecture review.
   [PENDING-RESEARCH: PSI/CSI thresholds from pipeline lane]

## Directory

```
brain/
  ledger.py            # paper/live bet ledger (JSONL, idempotent)
  metrics.py           # retrain-eval history + trend/drift view
  collect/fetch_all.sh # per-sport data refresh
  schedule/            # cron specs (hermes cron add lines)
  BRAIN_BLUEPRINT.md   # this file
  RESEARCH_*.md        # 3 research lanes' findings
```

## Scheduling (single macOS box)

Hermes cron (proposal — one silent watcher, pings only on change):
- Daily 06:00 UTC: fetch_all.sh (incremental)
- Monday 07:00 UTC: MMA + soccer retrain-and-eval, append metrics, report only if
  ROI/acc moves >1pp vs previous row (silent otherwise — matches user preference)
- Weekly: settle open ledger bets; send ledger delta
Exact `hermes cron add ...` lines land with the pipeline research.

## What the research lanes feed into

- **GitHub recon** → candidate techniques to port (xG/xT, ratings systems, market tooling)
- **Techniques** → model upgrades (Dixon-Coles for soccer goals, Bayesian/Kalman ratings,
  CLV-as-feature, stacking)
- **Pipeline** → cadences + drift thresholds + cron spec above

## Verified baselines (the starting point of the learning curve)

From MULTISPORT_REPORT.md (all coordinator-verified): MMA v2 80.2% acc / +40.3% flat ROI /
Brier 0.1368 (2,843 fights 2020-26); soccer O/U 59.7% / +4.0→28.7% gates; NFL 64.6%;
NBA/NHL/MLB/tennis ≈ market. The forever-learning loop starts here and every future
retrain must append a metrics row — improvements are proven, regressions caught.
