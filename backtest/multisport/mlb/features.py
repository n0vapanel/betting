"""Rolling feature engineering with strict no-lookahead.

One pass over games in date order; features for game G use only games strictly before G
(same-date doubleheader game 1 may inform game 2).
Also joins closing odds (verified by final score match).

Outputs: features.parquet with columns for home team perspective + target home_win.
"""
import pandas as pd
import numpy as np
from collections import defaultdict

games = pd.read_parquet("games_all.parquet")
odds = pd.read_parquet("odds_closing.parquet")

RETRO_TO_STD = {
    "ANA": "LAA", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHA": "CHW", "CHN": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "HOU": "HOU", "KCA": "KC", "LAN": "LAD", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYA": "NYY", "NYN": "NYM", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDN": "SD", "SEA": "SEA", "SFN": "SF",
    "SLN": "STL", "TBA": "TB", "TEX": "TEX", "TOR": "TOR", "WAS": "WAS",
}
games["home_std"] = games.HomeTeam.map(RETRO_TO_STD)
games["away_std"] = games.VisitingTeam.map(RETRO_TO_STD)

# ---------------- odds join (score-verified) ----------------
odds = odds[odds.status.str.startswith("Final")].copy()
odds_used = set()
join_rows = []
for gi, g in games.iterrows():
    d = g["date"].strftime("%Y-%m-%d")
    cands = odds[(odds.date == g["date"]) & (odds.home == g.home_std) & (odds.away == g.away_std)]
    match = None
    for ci, c in cands.iterrows():
        if ci in odds_used:
            continue
        if (pd.isna(c.home_score) or pd.isna(c.away_score)
                or (int(c.home_score) == int(g.HomeRunsScore) and int(c.away_score) == int(g.VisitorRunsScored))):
            match = ci
            break
    if match is not None:
        odds_used.add(match)
        join_rows.append((gi, match))
    else:
        join_rows.append((gi, -1))

jo = pd.DataFrame(join_rows, columns=["gi", "oi"])
matched = jo[jo.oi >= 0].set_index("gi")["oi"].to_dict()
gcols = ["home_std", "away_std"]
# build merge frame with NaN rows for unmatched
mr = pd.DataFrame({
    "oi": [matched.get(i, np.nan) for i in range(len(games))],
}, index=games.index).merge(
    odds[["p_market_home", "dec_home", "dec_away", "n_books"]],
    left_on="oi", right_index=True, how="left"
)
games = games.join(mr[["p_market_home", "dec_home", "dec_away", "n_books"]])
games["has_odds"] = games.p_market_home.notna()

# ---------------- rolling features ----------------
# Elo (538 MLB parameters): K=4, home adv 24, margin multiplier ln(|m|+1)*2.2/((|d|+275)/1000)
HFA = 24.0
K = 4.0

team_state = defaultdict(lambda: {
    "elo": 1500.0,
    "rs": [], "ra": [], "wins": 0, "games": 0,
    "last_date": None,
    "so": [], "bb": [], "h": [], "ab": [], "d": [], "t": [], "hr": [], "hbp": [], "sf": [], "outs": [],
    "inn_3d": [],        # (date, outs) last-3-day workloads
    "xtra_3d": [],
    "games_14d": [],     # dates of games in last 14 days
    "relief_w_14d": [],  # dates of games where a reliever got the W
    "relief_l_14d": [],  # dates of games where a reliever got the L
})
sp_state = defaultdict(lambda: {
    "starts": 0, "wins_in_starts": 0, "ra_in_starts": 0.0, "last_start": None,
})

rows = []
for _, g in games.iterrows():
    date = g["date"]
    home, away = g.HomeTeam, g.VisitingTeam
    hs, vs = g.HomeRunsScore, g.VisitorRunsScored
    h = team_state[home]
    a = team_state[away]

    def roll(vals, n):
        arr = np.array(vals[-n:], dtype=float)
        return arr.mean() if len(arr) else np.nan

    def ew(vals, alpha):
        if not vals:
            return np.nan
        arr = np.array(vals, dtype=float)
        w = (1 - alpha) ** np.arange(len(arr) - 1, -1, -1)
        return float(np.sum(arr * w) / np.sum(w))

    def inn3(st, today):
        cutoff = today - pd.Timedelta(days=3)
        keep = [(d_, o) for d_, o in st["inn_3d"] if d_ >= cutoff]
        st["inn_3d"] = keep
        return sum(o for _, o in keep) / 3.0

    def xtra3(st, today):
        cutoff = today - pd.Timedelta(days=3)
        st["xtra_3d"] = [d_ for d_ in st["xtra_3d"] if d_ >= cutoff]
        return float(len(st["xtra_3d"]))

    def rest(st, today):
        if st["last_date"] is None:
            return 5.0
        return max((today - st["last_date"]).days, 0.0)

    def win_share(vals, games_list):
        if not games_list:
            return np.nan
        return float(len(vals) / len(games_list))

    def sp_feat(pid, team):
        s = sp_state[pid]
        if s["starts"] == 0:
            return dict(sp_exp=0.0, sp_wpct=np.nan, sp_ra=np.nan, sp_rest=np.nan)
        return dict(
            sp_exp=float(s["starts"]),
            sp_wpct=float(s["wins_in_starts"] / s["starts"]),
            sp_ra=float(s["ra_in_starts"] / s["starts"]),
            sp_rest=float((date - s["last_start"]).days) if s["last_start"] is not None else np.nan,
        )

    def rates(st):
        h = np.array(st["h"], dtype=float); ab = np.array(st["ab"], dtype=float)
        bb = np.array(st["bb"], dtype=float); hbp = np.array(st["hbp"], dtype=float)
        sf = np.array(st["sf"], dtype=float); d = np.array(st["d"], dtype=float)
        t = np.array(st["t"], dtype=float); hr = np.array(st["hr"], dtype=float)
        so = np.array(st["so"], dtype=float); outs = np.array(st["outs"], dtype=float)
        if len(ab) == 0:
            return dict(obp=np.nan, slg=np.nan, krate=np.nan, bbrate=np.nan)
        obp = (h.sum() + bb.sum() + hbp.sum()) / max(ab.sum() + bb.sum() + hbp.sum() + sf.sum(), 1)
        slg = (h.sum() + d.sum() + 2 * t.sum() + 3 * hr.sum()) / max(ab.sum(), 1)
        inn = outs.sum() / 3.0
        krate = so.sum() / inn if inn > 0 else np.nan
        bbrate = bb.sum() / inn if inn > 0 else np.nan
        return dict(obp=float(obp), slg=float(slg), krate=float(krate), bbrate=float(bbrate))

    hspp = sp_feat(g.HomeStartingPitcherID, home)
    aspp = sp_feat(g.VisitingStartingPitcherID, away)
    rh = rates(h)
    ra_ = rates(a)

    feat = {
        "date": date, "season": g.season, "doubleheader": int(g.DoubleHeader),
        "home": home, "away": away, "home_std": g.home_std, "away_std": g.away_std,
        "park": g.ParkID, "day_night": 1 if g.DayNight == "D" else 0,
        "home_win": int(g.home_win),
        "home_runs": int(hs), "away_runs": int(vs),
        "p_market_home": g.p_market_home, "dec_home": g.dec_home, "dec_away": g.dec_away,
        "has_odds": int(g.has_odds), "n_books": g.n_books,
        # elo (raw diff, no home-field baked in)
        "elo_h": h["elo"], "elo_a": a["elo"],
        "elo_diff": h["elo"] - a["elo"],
        # season to date
        "rs_h": roll(h["rs"], 10**6) or np.nan, "rs_a": roll(a["rs"], 10**6) or np.nan,
        "ra_h": roll(h["ra"], 10**6) or np.nan, "ra_a": roll(a["ra"], 10**6) or np.nan,
        "rs_h_l10": roll(h["rs"], 10), "rs_a_l10": roll(a["rs"], 10),
        "ra_h_l10": roll(h["ra"], 10), "ra_a_l10": roll(a["ra"], 10),
        "winpct_h": h["wins"] / h["games"] if h["games"] else np.nan,
        "winpct_a": a["wins"] / a["games"] if a["games"] else np.nan,
        "games_h": h["games"], "games_a": a["games"],
        # exponentially weighted (half-life ~12 games)
        "ewrs_h": ew(h["rs"], 0.056), "ewrs_a": ew(a["rs"], 0.056),
        "ewra_h": ew(h["ra"], 0.056), "ewra_a": ew(a["ra"], 0.056),
        # rate stats (season to date)
        "obp_h": rh["obp"], "obp_a": ra_["obp"],
        "slg_h": rh["slg"], "slg_a": ra_["slg"],
        "krate_h": rh["krate"], "krate_a": ra_["krate"],
        "bbrate_h": rh["bbrate"], "bbrate_a": ra_["bbrate"],
        # rest
        "rest_h": rest(h, date), "rest_a": rest(a, date),
        # bullpen fatigue
        "inn3_h": inn3(h, date), "inn3_a": inn3(a, date),
        "xtra3_h": xtra3(h, date), "xtra3_a": xtra3(a, date),
        # relief win/loss share last 14 days
        "relw_h": win_share(h["relief_w_14d"], h["games_14d"]), "relw_a": win_share(a["relief_w_14d"], a["games_14d"]),
        "rell_h": win_share(h["relief_l_14d"], h["games_14d"]), "rell_a": win_share(a["relief_l_14d"], a["games_14d"]),
        # starting pitcher
        "sp_exp_h": hspp["sp_exp"], "sp_exp_a": aspp["sp_exp"],
        "sp_wpct_h": hspp["sp_wpct"], "sp_wpct_a": aspp["sp_wpct"],
        "sp_ra_h": hspp["sp_ra"], "sp_ra_a": aspp["sp_ra"],
        "sp_rest_h": hspp["sp_rest"], "sp_rest_a": aspp["sp_rest"],
    }
    rows.append(feat)

    # ---------------- update state ----------------
    def upd(tm, rs_for, ra_for, won, outs, today, sp_pid, wp_id, stats):
        st = team_state[tm]
        st["rs"].append(rs_for)
        st["ra"].append(ra_for)
        st["games"] += 1
        st["wins"] += int(won)
        st["last_date"] = today
        st["so"].append(stats["so"]); st["bb"].append(stats["bb"])
        st["h"].append(stats["h"]); st["ab"].append(stats["ab"])
        st["d"].append(stats["d"]); st["t"].append(stats["t"]); st["hr"].append(stats["hr"])
        st["hbp"].append(stats["hbp"]); st["sf"].append(stats["sf"]); st["outs"].append(outs)
        st["inn_3d"].append((today, outs))
        if outs > 54:
            st["xtra_3d"].append(today)
        # relief win/loss: winning/losing pitcher not the starter
        cutoff = today - pd.Timedelta(days=14)
        for key in ("relief_w_14d", "relief_l_14d", "games_14d"):
            st[key] = [d_ for d_ in st[key] if d_ >= cutoff]
        st["games_14d"].append(today)
        if won and wp_id != sp_pid:
            st["relief_w_14d"].append(today)
        if (not won) and g.LosingPitcherID != sp_pid:
            st["relief_l_14d"].append(today)

    hstats = dict(so=int(g.HomeStrikeouts), bb=int(g.HomeWalks), h=int(g.HomeHits),
                  ab=int(g.HomeAtBats), d=int(g.HomeDoubles), t=int(g.HomeTriples),
                  hr=int(g.HomeHomeruns), hbp=int(g.HomeHitByPitch), sf=int(g.HomeSacrificeFlies))
    astats = dict(so=int(g.VisitorStrikeouts), bb=int(g.VisitorWalks), h=int(g.VisitorHits),
                  ab=int(g.VisitorAtBats), d=int(g.VisitorDoubles), t=int(g.VisitorTriples),
                  hr=int(g.VisitorHomeruns), hbp=int(g.VisitorHitByPitch), sf=int(g.VisitorSacrificeFlies))
    upd(home, hs, vs, g.home_win, int(g.LengthInOuts), date, g.HomeStartingPitcherID, g.WinningPitcherID, hstats)
    upd(away, vs, hs, not g.home_win, int(g.LengthInOuts), date, g.VisitingStartingPitcherID, g.WinningPitcherID, astats)

    # starting pitcher state
    for pid, won, ra_ in [(g.HomeStartingPitcherID, bool(g.home_win), float(vs)),
                          (g.VisitingStartingPitcherID, bool(not g.home_win), float(hs))]:
        s = sp_state[pid]
        s["starts"] += 1
        s["wins_in_starts"] += int(won)
        s["ra_in_starts"] += ra_
        s["last_start"] = date

    # elo update (538)
    def elo_update(st_h, st_a, margin, home_won):
        d = st_h["elo"] + HFA - st_a["elo"]
        mult = np.log(abs(margin) + 1.0) * 2.2 / ((abs(d) + 275.0) / 1000.0)
        e = 1.0 / (1.0 + 10.0 ** (-d / 400.0))
        shift = K * mult * (int(home_won) - e)
        st_h["elo"] += shift
        st_a["elo"] -= shift

    elo_update(h, a, hs - vs, g.home_win)

feat = pd.DataFrame(rows)
feat["date"] = pd.to_datetime(feat.date)

# diff features (home - away; positive = home advantage)
feat["rs_diff"] = feat.rs_h - feat.rs_a
feat["ra_diff"] = feat.ra_h - feat.ra_a
feat["rs_l10_diff"] = feat.rs_h_l10 - feat.rs_a_l10
feat["ra_l10_diff"] = feat.ra_h_l10 - feat.ra_a_l10
feat["winpct_diff"] = feat.winpct_h - feat.winpct_a
feat["obp_diff"] = feat.obp_h - feat.obp_a
feat["slg_diff"] = feat.slg_h - feat.slg_a
feat["krate_diff"] = feat.krate_h - feat.krate_a
feat["bbrate_diff"] = feat.bbrate_h - feat.bbrate_a
feat["ewrs_diff"] = feat.ewrs_h - feat.ewrs_a
feat["ewra_diff"] = feat.ewra_h - feat.ewra_a
feat["rest_diff"] = feat.rest_h - feat.rest_a
feat["inn3_diff"] = feat.inn3_a - feat.inn3_h   # away workload minus home -> + = home fresher
feat["xtra3_diff"] = feat.xtra3_a - feat.xtra3_h
feat["relw_diff"] = feat.relw_h - feat.relw_a
feat["rell_diff"] = feat.rell_h - feat.rell_a
feat["sp_exp_diff"] = feat.sp_exp_h - feat.sp_exp_a
feat["sp_wpct_diff"] = feat.sp_wpct_h - feat.sp_wpct_a
feat["sp_ra_diff"] = feat.sp_ra_a - feat.sp_ra_h   # + = home SP better
feat["sp_rest_diff"] = feat.sp_rest_h - feat.sp_rest_a

feat.to_parquet("features.parquet")
print(feat.shape)
print("games with odds:", feat.has_odds.sum(), "of", len(feat))
print("odds coverage by month (2022):")
print(feat[feat.season == 2022].groupby(feat.date.dt.to_period("M")).agg(n=("date", "size"), n_odds=("has_odds", "sum")))
print(feat.columns.tolist())
