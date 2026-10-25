# UFC Picker — Live-Card Predictions from the Frozen v1 MMA Ensemble

Makes model picks for upcoming UFC cards using the exact ensemble from the
verified v1 backtest (`run_backtest.py`): LogReg (C=0.03) + LightGBM
(num_leaves=15, lr=0.03), each isotonic-calibrated on expanding-CV OOF
predictions, blended 50/50.

```
backtest/multisport/mma/
├── feature_utils.py       # shared feature builders (extracted from build_features.py)
├── build_features.py      # v1 script, refactored to call feature_utils (output verified identical)
├── train_model.py         # train_model(history_df) extracted from run_backtest.py + export driver
├── ufc_picker.py          # the live-card CLI (this file's subject)
├── models/
│   ├── primary_2016_2019.joblib        # ALL features incl. strike stats, trained 2016→2019
│   ├── extended_core_thru_2025.joblib  # CORE features, trained 2016→2025-12-31 (default)
│   └── manifest.json
├── smoke_card_2026_01_31.csv           # smoke-test card (3 real 2026-01-31 fights, real odds)
└── picker_smoke_report.md              # smoke-test results + verification details
```

## Quick start

```bash
# 1. (once) train/export the frozen models — reproduces v1 backtest numbers exactly
python train_model.py

# 2. pick a card
python ufc_picker.py --card upcoming.csv

# single fight
python ufc_picker.py --name1 "Fighter A" --name2 "Fighter B" \
    --date 2026-04-04 --odds1 -150 --odds2 +120 --weight-class Lightweight [--title-bout]
```

Use the shared venv: `~/.hermes/cache/scratch/msports-venv/bin/python`.

## Card CSV format

Required columns: `date, red_fighter, blue_fighter, weight_class, R_odds, B_odds`
(American closing odds, e.g. `-162`, `+136`). Optional columns (defaulted when
absent): `title_bout` (False), `no_of_rounds` (3, or 5 for title bouts),
`gender` (inferred from "Women's" weight classes).

```csv
date,red_fighter,blue_fighter,weight_class,R_odds,B_odds,title_bout,no_of_rounds
2026-01-31,Alexander Volkanovski,Diego Lopes,Featherweight,-162,136,True,5
```

Output: console table + markdown report (`<card>_report.md`, override with
`--out`). Per fight: pick side, model prob, de-vigged fair prob, **edge =
p_model − p_fair**, flat-$1 bet when edge > 0, and a quarter-Kelly stake
`min(¼·f_Kelly, 25%)` of bankroll (`--bankroll`, default $1,000) when Kelly is
positive.

## How it works (no future data)

1. History = every fight in `raw_mdabbert/ufc-master.csv` with date **strictly
   before the first card date** (override with `--as-of`, must be ≤ first card
   date — the picker refuses anything later). Outcomes of the card fights
   themselves never enter the computation.
2. Card rows are appended (`Winner=NaN`) and the **same** builders as the v1
   backtest run over the combined frame: Elo (K=60, start 1500), record,
   streak, longest run, days-since, and ufcstats strike stats. Rows without an
   outcome read their Elo but never update it.
3. Model features (diffs + categoricals) are added exactly as in
   `run_backtest.py`; the picker reindexes to the frozen model's feature list.
4. Prediction = 50/50 average of the two calibrated models.

## Model choice

- `--model extended` (default): core features, trained through 2025-12-31.
  Use for any live card — strike data (ufcstats) ends ~2021, so 2026 cards have
  no strike features.
- `--model primary`: strike + core features, trained 2016-2019 (the v1 primary
  model). Only for cards where ufcstats strike data exists (≤ 2021).
- `--model auto`: primary if every fight has strike features, else extended.

## Known live-vs-backtest caveats

- **Rank drift**: `*_match_weightclass_rank` uses the fighter's last known rank
  before the card; the day-of rank is unknowable live.
- **v1 streak quirk**: in training data, the losing corner's `streak_pre` is 0
  by construction. The picker gives both corners their true pre-fight streak —
  the only honest live choice, but slightly off the training distribution.
- **Static attrs** (age, reach, height, weight) come from each fighter's last
  known fight; age is advanced to the card date. Missing values are
  median-imputed by the model pipeline.
- If a fighter has no prior UFC fight in the history, record/streak/Elo
  features default (Elo 1500, 0 fights) and a warning is printed.

## Verification status

- Refactored `build_features.py` output is **bit-identical** to the v1
  `fights_features.parquet` (assert_frame_equal passes).
- `train_model.py` primary model reproduces v1 metrics to 15 decimals:
  OOF AUC LR 0.594935 / GBM 0.831257; v1 primary test acc 0.7657 reproduced.
- Extended model: OOF GBM AUC 0.8631; 2026-01..03 holdout acc 0.7895, AUC 0.9199.
- Smoke test (3 real 2026-01-31 fights, real odds): exit 0; rolling features
  match the v1 parquet for the same fights; details in `picker_smoke_report.md`.

## Upcoming-card data sources (probe results)

| source | status |
|---|---|
| bestfightodds.com | ✅ curl + browser UA works, no auth; event pages expose `bestbet">±108` odds spans — scriptable |
| ufc.com | ⚠️ `/api/v1/events` JSON API dead; `ufc.com/events` SSR lists events but no odds |
| tapology.com | ❌ 403 + Cloudflare/captcha for curl |

No authenticated/paywalled scraping. Manual CSV is always the fallback.
