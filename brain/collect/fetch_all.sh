#!/usr/bin/env bash
# brain/collect/fetch_all.sh — refresh every sport's data (idempotent, re-runnable).
# Run from repo root: bash brain/collect/fetch_all.sh
# Sizes: full refresh ~1.5GB cached (gitignored). Incremental where possible.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=~/.hermes/cache/scratch/msports-venv/bin/python
cd "$ROOT"

# --- MMA: Kaggle mdabbert/ultimate-ufc-dataset (fights + BFO closing odds) ---
echo "== MMA =="
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-KGAT_c77c97fe666f687ee1863c29c574ebe9}"
mkdir -p backtest/multisport/mma/raw_mdabbert
if ! kaggle datasets download -f ufc-master.csv mdabbert/ultimate-ufc-dataset \
     -p backtest/multisport/mma/raw_mdabbert 2>/dev/null; then
  echo "  MMA: kaggle CLI missing or failed (run: pip install kaggle)"; fi

# --- Soccer: football-data.co.uk (scores + closing odds + shots/corners) ---
echo "== SOCCER =="
mkdir -p backtest/multisport/soccer_v2/data
for L in E0 SP1 D1 I1 F1; do
  for S in 1718 1819 1920 2021 2122; do
    F="backtest/multisport/soccer_v2/data/${L}_${S}.csv"
    [ -s "$F" ] || curl -sL -m 30 -o "$F" "https://www.football-data.co.uk/mmz4281/${S}/${L}.csv" || true
  done
done

# --- MLB: Retrosheet game logs (scores + pitchers) ---
echo "== MLB =="
mkdir -p backtest/multisport/mlb_v2/data
for Y in 2021 2022; do
  [ -s "backtest/multisport/mlb_v2/data/gl${Y}.zip" ] || \
    curl -sL -m 60 -o "backtest/multisport/mlb_v2/data/gl${Y}.zip" \
      "https://www.retrosheet.org/gamelogs/gl${Y}.zip" || true
done

# --- NHL: moneypuck game-by-game (needed by nhl_v2 builder) ---
echo "== NHL =="
mkdir -p backtest/multisport/nhl_v2/data
[ -s "backtest/multisport/nhl_v2/data/all_teams.csv" ] || \
  curl -sL -m 60 -o "backtest/multisport/nhl_v2/data/all_teams.csv" \
    "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv" || true

# --- Tennis: tennis-data.co.uk via Wayback (B365 closing odds) ---
echo "== TENNIS =="
mkdir -p backtest/multisport/tennis_v2/raw
for Y in 2021 2022; do
  [ -s "backtest/multisport/tennis_v2/raw/${Y}.xlsx" ] || \
    curl -sL -m 90 -o "backtest/multisport/tennis_v2/raw/${Y}.xlsx" \
      "https://web.archive.org/web/20230514150435id_/http://www.Tennis-Data.co.uk/${Y}/${Y}.xlsx" || true
done

# --- NFL: nfl_data_py parquet releases (schedules/PBP/injuries/snaps/depth) ---
echo "== NFL =="
"$PY" - <<'EOF'
import nfl_data_py as nfl, os
os.makedirs("backtest/multisport/nfl/.cache", exist_ok=True)
for y in (2020, 2021, 2022):
    try:
        nfl.import_schedules([y])
        nfl.import_injuries([y])
        nfl.import_snap_counts([y])
    except Exception as e:
        print(f"  NFL {y}: {e}")
EOF

echo "== done =="
