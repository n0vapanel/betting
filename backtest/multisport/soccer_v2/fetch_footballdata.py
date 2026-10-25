#!/usr/bin/env python
"""Download football-data.co.uk CSVs for 5 leagues x 5 seasons into data/.
Polite: 1.2s between requests. Reports per-file column availability for
opening-odds columns (CLV features)."""
import os, time, urllib.request

BASE = "/Users/x/Desktop/x/Coding Projects/betting bot/backtest/multisport/soccer_v2"
DATA = os.path.join(BASE, "data")
LEAGUES = {"E0": "EPL", "SP1": "LaLiga", "D1": "Bundesliga", "I1": "SerieA", "F1": "Ligue1"}
SEASONS = ["1718", "1819", "1920", "2021", "2122"]

def fetch(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research backtest)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    return data

def main():
    import csv
    os.makedirs(DATA, exist_ok=True)
    for code in LEAGUES:
        for season in SEASONS:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            path = os.path.join(DATA, f"{code}_{season}.csv")
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                print(f"cached  {code} {season}")
                continue
            try:
                data = fetch(url, path)
                print(f"ok      {code} {season}  {len(data)} bytes")
            except Exception as e:
                print(f"FAIL    {code} {season}  {e}")
            time.sleep(1.2)
    print("\nColumn availability (opening odds):")
    for code in LEAGUES:
        for season in SEASONS:
            path = os.path.join(DATA, f"{code}_{season}.csv")
            try:
                with open(path, encoding="utf-8-sig") as f:
                    cols = next(csv.reader(f))
                want = ["AvgH", "AvgCH", "PSH", "PSCH", "B365H", "B365CH", "MaxH", "MaxCH",
                        "Avg>2.5", "AvgC>2.5", "PS>2.5", "PSC>2.5", "BbAv>2.5", "BbAvH"]
                have = [c for c in want if c in cols]
                miss = [c for c in want if c not in cols]
                print(f"{code} {season}: HAVE {have} | MISS {miss}")
            except Exception as e:
                print(f"{code} {season}: {e}")

if __name__ == "__main__":
    main()
