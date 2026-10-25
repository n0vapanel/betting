"""Download Statcast pitch-level CSVs (Baseball Savant) and aggregate to per-pitcher-game stats.

Needed for: starting-pitcher quality (IP, K, BB, HR, velo, batters faced per start) and
bullpen fatigue (relief IP/pitches per team per day) with strict no-lookahead rolling.

Fetches raw CSVs in ~3-week chunks, aggregates, saves pitcher_game_stats.parquet.
"""
import io
import os
import sys
import time
import datetime as dt

import pandas as pd
import requests

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE = os.path.join(OUT_DIR, "pitcher_game_stats.parquet")

# date ranges: pre-season pad so early games have some relief history
RANGES = [
    ("2021-03-15", "2021-11-15"),   # full 2021 (incl. playoffs; we filter by schedule join)
    ("2022-03-15", "2022-06-30"),   # 2022 through test window end
]

USE_COLS = ["game_pk", "game_date", "pitcher", "home_team", "away_team",
            "inning", "inning_topbot", "outs_when_up", "pitch_number",
            "at_bat_number", "events", "description", "type", "release_speed"]

# events that record outs, with the number of outs they record
OUT_EVENTS = {
    "strikeout": 1, "strikeout_double_play": 2, "field_out": 1, "force_out": 1,
    "grounded_into_double_play": 2, "double_play": 2, "triple_play": 3,
    "sac_fly": 1, "sac_fly_double_play": 2, "fielders_choice_out": 1,
    "caught_stealing_2b": 1, "caught_stealing_3b": 1, "caught_stealing_home": 1,
    "pickoff_1b": 1, "pickoff_2b": 1, "pickoff_3b": 1,
    "pickoff_caught_stealing_2b": 2, "pickoff_caught_stealing_3b": 2,
    "pickoff_caught_stealing_home": 2, "other_out": 1,
}
K_EVENTS = {"strikeout", "strikeout_double_play"}
BB_EVENTS = {"walk", "intent_walk"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (research backtest; contact: local)"})


def chunk_dates(lo, hi, days=4):
    out, cur = [], dt.date.fromisoformat(lo)
    hi_d = dt.date.fromisoformat(hi)
    while cur < hi_d:
        nxt = min(cur + dt.timedelta(days=days), hi_d)
        out.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt
    return out


def fetch(start, end, retries=4):
    url = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details"
           f"&hfGT=R%7C&player_type=pitcher&game_date_gt={start}&game_date_lt={end}"
           "&min_pitches=0&min_results=0&group_by=name&sort_col=pitches"
           "&player_event_sort=game_date&sort_order=desc&min_pa=0")
    for attempt in range(retries):
        try:
            r = S.get(url, timeout=300)
            r.raise_for_status()
            if "<html" in r.text[:200].lower():
                raise ValueError("got HTML page, not CSV")
            return r.text
        except Exception as e:
            print(f"  retry {attempt+1}/{retries} {start}->{end}: {type(e).__name__} {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def main():
    rows = []
    for lo, hi in RANGES:
        for start, end in chunk_dates(lo, hi):
            t0 = time.time()
            text = fetch(start, end)
            if text is None:
                print(f"FAILED chunk {start}->{end}", flush=True)
                sys.exit(1)
            try:
                df = pd.read_csv(io.StringIO(text), usecols=USE_COLS)
            except ValueError as e:
                print(f"parse err {start}->{end}: {e}", flush=True)
                sys.exit(1)
            df = df[df.pitch_number.notna()].copy()
            if len(df) >= 24800:
                print(f"  WARNING {start}->{end}: {len(df)} rows (possible savant truncation)", flush=True)
            df["game_date"] = pd.to_datetime(df["game_date"])
            topbot_ord = df["inning_topbot"].map({"Top": 0, "Bot": 1}).fillna(2)
            df["_sortkey"] = list(zip(df["inning"].fillna(0), topbot_ord,
                                      df["at_bat_number"].fillna(0),
                                      df["pitch_number"].fillna(0)))
            df = df.sort_values("_sortkey", kind="mergesort")
            df["_ord"] = df.groupby("game_pk").cumcount()
            ev = df["events"].fillna("").astype(str)
            outs = ev.map(OUT_EVENTS).fillna(0).astype(int)
            g = df.assign(
                n_pitches=1,
                outs=outs,
                k=ev.isin(K_EVENTS).astype(int),
                bb=ev.isin(BB_EVENTS).astype(int),
                hbp=(ev == "hit_by_pitch").astype(int),
                hr=(ev == "home_run").astype(int),
                hits=ev.isin(HIT_EVENTS).astype(int),
                bf=(df.pitch_number == 1).astype(int),
                _ord=df["_ord"],
            ).groupby(["game_pk", "game_date", "pitcher", "home_team", "away_team"],
                      as_index=False).agg(
                pitches=("n_pitches", "sum"), outs=("outs", "sum"),
                k=("k", "sum"), bb=("bb", "sum"), hbp=("hbp", "sum"),
                hr=("hr", "sum"), hits=("hits", "sum"), bf=("bf", "sum"),
                avg_velo=("release_speed", "mean"),
                min_ord=("_ord", "min"),
                first_topbot=("inning_topbot", "first"),
            )
            rows.append(g)
            print(f"{start}->{end}: {len(df):>7} pitches, {g.pitcher.nunique():>4} pitcher-games "
                  f"({time.time()-t0:.0f}s)", flush=True)
            time.sleep(1.5)
    full = pd.concat(rows, ignore_index=True)
    # Savant's date filters are inclusive, so boundary-day games appear in two
    # adjacent chunks -> drop duplicate (game_pk, pitcher) aggregate rows.
    full = full.drop_duplicates(["game_pk", "pitcher"], keep="first").reset_index(drop=True)
    full["ip"] = full["outs"] / 3.0
    full.to_parquet(SAVE, index=False)
    print(f"\nSaved {SAVE}: {len(full)} rows, {full.game_pk.nunique()} games, "
          f"{full.pitcher.nunique()} pitchers", flush=True)


if __name__ == "__main__":
    main()
