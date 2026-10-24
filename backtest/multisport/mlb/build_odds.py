"""Build closing-odds table from the SportsBookReview JSON dump (ArnavSaraogi/mlb-odds-scraper).

Consensus across available sportsbooks of de-vigged closing moneyline probabilities.
"""
import json
import pandas as pd
import numpy as np

# Retrosheet team code -> standard MLB abbreviation (as used by the odds dump)
RETRO_TO_STD = {
    "ANA": "LAA", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHA": "CWS", "CHN": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "HOU": "HOU", "KCA": "KC", "LAN": "LAD", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYA": "NYY", "NYN": "NYM", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDN": "SD", "SEA": "SEA", "SFN": "SF",
    "SLN": "STL", "TBA": "TB", "TEX": "TEX", "TOR": "TOR", "WAS": "WSH",
}

D = json.load(open("/Users/x/.hermes/cache/scratch/mlb-odds-scraper/mlb_odds_dataset.json"))


def am_to_dec(ml):
    ml = float(ml)
    return (1 + ml / 100) if ml > 0 else (1 - 100 / ml)


def am_to_implied(ml):
    ml = float(ml)
    return (-ml / (-ml + 100)) if ml < 0 else (100 / (ml + 100))


rows = []
for date, games in D.items():
    for gm in games:
        gv = gm.get("gameView", {})
        ml_entries = gm.get("odds", {}).get("moneyline", [])
        if not ml_entries:
            continue
        home, away = gv.get("homeTeam", {}).get("shortName"), gv.get("awayTeam", {}).get("shortName")
        hs, as_ = gv.get("homeTeamScore"), gv.get("awayTeamScore")
        status = gv.get("gameStatusText", "")
        # closing lines only
        p_home_l, dec_home_l, dec_away_l = [], [], []
        for e in ml_entries:
            cur = e.get("currentLine", {})
            try:
                ho, ao = float(cur.get("homeOdds")), float(cur.get("awayOdds"))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(ho) and np.isfinite(ao)) or ho == 0 or ao == 0:
                continue
            ph = am_to_implied(ho)
            pa = am_to_implied(ao)
            p_home_l.append(ph / (ph + pa))  # de-vig each book
            dec_home_l.append(am_to_dec(ho))
            dec_away_l.append(am_to_dec(ao))
        if not p_home_l:
            continue
        rows.append({
            "date": date, "home": home, "away": away,
            "home_score": hs, "away_score": as_, "status": status,
            "n_books": len(p_home_l),
            "p_market_home": float(np.mean(p_home_l)),
            "dec_home": float(np.mean(dec_home_l)),
            "dec_away": float(np.mean(dec_away_l)),
        })

odds = pd.DataFrame(rows)
odds["date"] = pd.to_datetime(odds["date"], format="%Y-%m-%d")
odds.to_parquet("odds_closing.parquet")
print(odds.shape)
print(odds.groupby(odds.date.dt.year).agg(n=("date", "size"), first=("date", "min"), last=("date", "max")))
print("unique team codes in odds:", sorted(set(odds.home) | set(odds.away)))
print("games with non-final status:", odds[~odds.status.str.startswith("Final")].shape[0])
print(odds.head(3).to_string())
