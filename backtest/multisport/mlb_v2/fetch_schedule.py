"""Rebuild schedule_gamepk.parquet with robust doubleheader numbering.

MLB StatsAPI lists games with unreliable gameNumber/doubleHeader flags and
duplicate rows. Robust approach: keep only Final-status games, use officialDate,
dedupe by game_pk, and for same (date, home, away) pairs order by start time
(gameDate) to assign dh_num 1/2 — matching Retrosheet's DoubleHeader convention.
"""
import requests
import pandas as pd

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0"})

rows = []
for lo, hi in [("2021-03-15", "2021-10-15"), ("2022-03-15", "2022-06-30")]:
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={lo}"
           f"&endDate={hi}&gameType=R&hydrate=team")
    j = S.get(url, timeout=120).json()
    n = 0
    for d in j.get("dates", []):
        for g in d["games"]:
            status = (g.get("status") or {}).get("codedGameState", "")
            if status != "F":
                continue
            off = g.get("officialDate")
            if not off or not (lo <= off <= hi):
                continue
            rows.append(dict(
                game_pk=g["gamePk"], date=off,
                start=g.get("gameDate"),  # UTC ISO
                home=g["teams"]["home"]["team"].get("abbreviation"),
                away=g["teams"]["away"]["team"].get("abbreviation")))
            n += 1
    print(lo, "->", hi, "final games:", n)

sched = pd.DataFrame(rows)
sched["start"] = pd.to_datetime(sched["start"], utc=True)
sched = sched.sort_values(["date", "home", "away", "start"]).drop_duplicates(
    ["date", "home", "away", "game_pk"], keep="first")
# dh numbering within (date, home, away)
sched["dh_num"] = sched.groupby(["date", "home", "away"]).cumcount() + 1
sched["dh_num"] = sched.groupby(["date", "home", "away"])["dh_num"].transform(
    lambda s: s if len(s) > 1 else 0)
sched["start_et"] = sched.start.dt.tz_convert("America/New_York").dt.strftime("%H:%M")
sched = sched.drop(columns="start")
sched.to_parquet("schedule_gamepk.parquet", index=False)
print("total:", len(sched))
print("multi-game date+teams pairs:", (sched.dh_num > 0).sum() // 2, "DHs")
print(sched[sched.dh_num > 0].head(6).to_string())
