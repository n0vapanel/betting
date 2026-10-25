#!/usr/bin/env python
"""Fetch Understat league data (per-match xG/xGA for every match) for 5 leagues x 5 seasons.
Endpoint: https://understat.com/main/getLeagueData/<slug>/<season>  (gzip JSON).
Season label mapping: football-data 2122 -> understat 2021, ..., 1718 -> 2017.
Polite: 1.2s between requests, retries with backoff. Caches to data/understat/.
"""
import json, os, time, gzip, urllib.request, urllib.error

BASE = "/Users/x/Desktop/x/Coding Projects/betting bot/backtest/multisport/soccer_v2"
DATA = os.path.join(BASE, "data", "understat")
LEAGUES = {"EPL": "EPL", "La_Liga": "LaLiga", "Bundesliga": "Bundesliga",
           "Serie_A": "SerieA", "Ligue_1": "Ligue1"}
SEASONS = [("2122", "2021"), ("2021", "2020"), ("1920", "2019"),
           ("1819", "2018"), ("1718", "2017")]
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
      "X-Requested-With": "XMLHttpRequest"}

def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        raw = r.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw

def main():
    os.makedirs(DATA, exist_ok=True)
    for slug, lname in LEAGUES.items():
        for fd_season, us_season in SEASONS:
            path = os.path.join(DATA, f"{lname}_{fd_season}.json")
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                print(f"cached  {lname} {fd_season}")
                continue
            url = f"https://understat.com/main/getLeagueData/{slug}/{us_season}"
            ok = False
            for attempt in range(4):
                try:
                    raw = fetch(url)
                    d = json.loads(raw)
                    dates = d.get("dates", [])
                    if not dates:
                        raise ValueError("empty dates")
                    with open(path, "w") as f:
                        json.dump(d, f)
                    print(f"ok      {lname} {fd_season}: {len(dates)} matches, {len(d.get('teams', {}))} teams")
                    ok = True
                    break
                except Exception as e:
                    print(f"retry {attempt+1} {lname} {fd_season}: {e}")
                    time.sleep(4 * (attempt + 1))
            if not ok:
                print(f"FAIL    {lname} {fd_season}")
            time.sleep(1.2)

if __name__ == "__main__":
    main()
