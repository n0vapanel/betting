"""MLB v2 features: v1 rolling features + SP quality (Statcast), bullpen fatigue
(Statcast relief workload), park factors (FanGraphs), rest/travel (timezones),
lineup OPS (game-log exact), with strict no-lookahead.

One chronological pass over games; features for game G use only games completed
strictly before G (same-day DH game 1 may inform game 2, as in v1).

Outputs: features_v2.parquet (home perspective, target home_win).
"""
import os

import numpy as np
import pandas as pd
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MLB1 = os.environ.get("MLB1_DIR", os.path.join(os.path.dirname(HERE), "mlb"))
SC_PATH = os.environ.get("SC_PATH", os.path.join(HERE, "pitcher_game_stats.parquet"))
OUT_PATH = os.environ.get("OUT_PATH", os.path.join(HERE, "features_v2.parquet"))

games = pd.read_parquet(os.path.join(MLB1, "games_all.parquet"))
odds = pd.read_parquet(os.path.join(MLB1, "odds_closing.parquet"))
sched = pd.read_parquet(os.path.join(HERE, "schedule_gamepk.parquet"))

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
games = games.sort_values(["date", "DoubleHeader"], kind="stable").reset_index(drop=True)

# ---------------- odds join (score-verified; identical to v1) ----------------
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
mr = pd.DataFrame({"oi": [matched.get(i, np.nan) for i in range(len(games))]},
                  index=games.index).merge(
    odds[["p_market_home", "dec_home", "dec_away", "n_books"]],
    left_on="oi", right_index=True, how="left")
games = games.join(mr[["p_market_home", "dec_home", "dec_away", "n_books"]])
games["has_odds"] = games.p_market_home.notna()

# ---------------- schedule -> game_pk join (Statcast link) ----------------
sched = sched.dropna(subset=["date"]).copy()
sched["date"] = pd.to_datetime(sched["date"])
sched = sched.drop_duplicates(["date", "home", "away", "dh_num"], keep="first")
# statsapi abbreviations -> standard (CHW/WAS/ARI)
SCHED_TO_STD = {"CWS": "CHW", "WSH": "WAS", "AZ": "ARI"}
sched["home"] = sched.home.map(lambda x: SCHED_TO_STD.get(x, x))
sched["away"] = sched.away.map(lambda x: SCHED_TO_STD.get(x, x))
sched = sched.sort_values(["date", "start_et"]).reset_index(drop=True)

# Pass 1: exact (date, home, away, dh_num) match
sched_map = {}
for _, r in sched.iterrows():
    sched_map[(r.date, r.home, r.away, int(r.dh_num))] = r.game_pk
games["game_pk"] = [sched_map.get((g.date, g.home_std, g.away_std, int(g.DoubleHeader)))
                    for _, g in games.iterrows()]

# Pass 2: bucket match for venue-swapped/glitched games (rare: swapped DHs, makeup games)
#   per (date, frozenset(teams)) pair remaining rows in start-time order with
#   Retrosheet order (DoubleHeader 1 then 2). Handles e.g. 2022-05-10 OAK@DET DH.
unmatched_idx = games.index[games.game_pk.isna()]
if len(unmatched_idx):
    used = set(sched_map.values())
    api_left = {}
    for _, r in sched.iterrows():
        if r.game_pk in used:
            continue
        api_left.setdefault((r.date, frozenset((r.home, r.away))), []).append(r.game_pk)
    retro_left = {}
    for gi in unmatched_idx:
        g = games.loc[gi]
        retro_left.setdefault((g.date, frozenset((g.home_std, g.away_std))), []).append(gi)
    fixed = {}
    for key, gis in retro_left.items():
        apis = api_left.get(key, [])
        if len(apis) == len(gis):
            for gi, gpk in zip(sorted(gis, key=lambda i: games.loc[i].DoubleHeader), apis):
                fixed[gi] = gpk
    if fixed:
        for gi, gpk in fixed.items():
            games.loc[gi, "game_pk"] = gpk
        print(f"pass-2 bucket matches: {len(fixed)} games")

print("schedule join coverage:", int(games.game_pk.notna().sum()), "of", len(games))
for y in (2021, 2022):
    sub = games[games.season == y]
    print(f"  unmatched {y}:", int((~sub.game_pk.notna()).sum()), "of", len(sub))

# ---------------- Statcast pitcher-game aggregates ----------------
sc = pd.read_parquet(SC_PATH)
sc["game_date"] = pd.to_datetime(sc["game_date"])
ord0 = sc[sc.min_ord == 0][["game_pk", "pitcher"]].rename(columns={"pitcher": "home_sp_mlbam"})
top = sc[sc.first_topbot == "Bot"].sort_values("min_ord").groupby("game_pk").first()["pitcher"].rename("away_sp_mlbam")
sp_by_game = ord0.merge(top, on="game_pk", how="left")
sp_by_game["away_sp_mlbam"] = sp_by_game.away_sp_mlbam.fillna(-1).astype(int)

sc2 = sc.merge(sp_by_game, on="game_pk", how="left")
sc2["is_sp"] = (sc2.pitcher == sc2.away_sp_mlbam) | (sc2.pitcher == sc2.home_sp_mlbam)
# Top halves are pitched by the HOME team, bottom halves by the AWAY team.
sc2["side"] = np.where(sc2.first_topbot == "Top", "home", "away")

# SP line per game per side
def sp_line(side):
    d = sc2[(sc2.is_sp) & (sc2.side == side)].copy()
    d = d.rename(columns={"pitcher": "sp_mlbam", "pitches": "sp_pitches", "outs": "sp_outs",
                          "k": "sp_k", "bb": "sp_bb", "hr": "sp_hr", "hits": "sp_hits",
                          "bf": "sp_bf", "avg_velo": "sp_velo"})
    return d[["game_pk", "sp_mlbam", "sp_pitches", "sp_outs", "sp_k", "sp_bb",
              "sp_hr", "sp_hits", "sp_bf", "sp_velo"]].copy()

gsc = sp_line("home").rename(columns={c: c + "_h" for c in sp_line("home").columns if c != "game_pk"})
gsc = gsc.merge(sp_line("away").rename(columns={c: c + "_a" for c in sp_line("away").columns if c != "game_pk"}),
                on="game_pk", how="outer")

# relief workload per game per side
rel = sc2[~sc2.is_sp].copy()
relg = rel.groupby(["game_pk", "side"]).agg(
    rel_outs=("outs", "sum"), rel_pitches=("pitches", "sum"),
    rel_k=("k", "sum"), rel_bb=("bb", "sum"), rel_app=("pitcher", "nunique")).reset_index()
for side in ("home", "away"):
    r = relg[relg.side == side].rename(columns={
        "rel_outs": f"rel_outs_{side}", "rel_pitches": f"rel_pitches_{side}",
        "rel_k": f"rel_k_{side}", "rel_bb": f"rel_bb_{side}", "rel_app": f"rel_app_{side}"}).drop(columns="side")
    gsc = gsc.merge(r, on="game_pk", how="outer")
games = games.merge(gsc, on="game_pk", how="left")
print("statcast game coverage (via game_pk):",
      int(games.sp_outs_h.notna().sum()), "of", len(games))

# ---------------- park factors (FanGraphs guts, 2021) ----------------
FG_ABBR = {
    "Angels": "LAA", "Orioles": "BAL", "Red Sox": "BOS", "White Sox": "CHW",
    "Cleveland": "CLE", "Tigers": "DET", "Royals": "KC", "Twins": "MIN",
    "Yankees": "NYY", "Athletics": "OAK", "Mariners": "SEA", "Rays": "TB",
    "Rangers": "TEX", "Blue Jays": "TOR", "Diamondbacks": "ARI", "Braves": "ATL",
    "Cubs": "CHC", "Reds": "CIN", "Rockies": "COL", "Marlins": "MIA",
    "Astros": "HOU", "Dodgers": "LAD", "Brewers": "MIL", "Nationals": "WAS",
    "Mets": "NYM", "Phillies": "PHI", "Pirates": "PIT", "Cardinals": "STL",
    "Padres": "SD", "Giants": "SF",
}
pf = pd.read_csv(os.path.join(HERE, "park_factors_fg_2021.csv"))
pf["team"] = pf.Team.map(FG_ABBR)
pf = pf[["team", "Basic (5yr)", "3yr", "HR"]].rename(
    columns={"Basic (5yr)": "pf_basic5", "3yr": "pf_3yr", "HR": "pf_hr5"})
for c in ("pf_basic5", "pf_3yr", "pf_hr5"):
    pf[c] = pf[c] / 100.0
games = games.merge(pf, left_on="home_std", right_on="team", how="left").drop(columns="team")
print("park factor coverage:", int(games.pf_basic5.notna().sum()), "of", len(games))

# ---------------- timezone table ----------------
# standard-time UTC offsets (east positive). DST +1 (US, Mar-Nov); ARI has no DST.
TZ_STD = {
    "LAA": -8, "LAD": -8, "OAK": -8, "SD": -8, "SEA": -8, "SF": -8,  # PT
    "ARI": -7,                                                         # MT, no DST
    "COL": -7,                                                         # MT
    "CHW": -6, "CHC": -6, "HOU": -6, "KC": -6, "MIL": -6, "MIN": -6,
    "STL": -6, "TEX": -6,                                              # CT
    "ATL": -5, "BAL": -5, "BOS": -5, "CLE": -5, "CIN": -5, "DET": -5,
    "MIA": -5, "NYM": -5, "NYY": -5, "PHI": -5, "PIT": -5, "TB": -5,
    "TOR": -5, "WAS": -5,                                              # ET
}


def tz_offset(team, date):
    base = TZ_STD[team]
    if team == "ARI":
        return base
    # US DST: 2021 Mar 14 - Nov 7; 2022 Mar 13 - Nov 6
    if date.year == 2021:
        lo, hi = pd.Timestamp("2021-03-14"), pd.Timestamp("2021-11-07")
    else:
        lo, hi = pd.Timestamp("2022-03-13"), pd.Timestamp("2022-11-06")
    return base + (1 if lo <= date < hi else 0)


# ---------------- rolling features ----------------
HFA = 24.0
K = 4.0

team_state = defaultdict(lambda: {
    "elo": 1500.0,
    "rs": [], "ra": [], "wins": 0, "games": 0, "last_date": None,
    "so": [], "bb": [], "h": [], "ab": [], "d": [], "t": [], "hr": [], "hbp": [], "sf": [], "outs": [],
    "inn_3d": [], "xtra_3d": [], "games_14d": [], "relief_w_14d": [], "relief_l_14d": [],
    # v2: relief workload history (statcast), ops per game, tz shift history (away perspective)
    "rel_hist": [], "ops": [], "tz_hist": [],
})
sp_state = defaultdict(lambda: {"starts": []})
sp_retro = defaultdict(lambda: {
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
        st["inn_3d"] = [(d_, o) for d_, o in st["inn_3d"] if d_ >= cutoff]
        return sum(o for _, o in st["inn_3d"]) / 3.0

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

    def rel3(st, today):
        cutoff = today - pd.Timedelta(days=3)
        st["rel_hist"] = [x for x in st["rel_hist"] if x["date"] >= cutoff]
        if not st["rel_hist"]:
            return dict(ip=np.nan, pit=np.nan, kbb=np.nan, app=np.nan)
        ip = sum(x["outs"] for x in st["rel_hist"]) / 3.0
        pit = sum(x["pitches"] for x in st["rel_hist"])
        kbb = sum(x["k"] for x in st["rel_hist"]) - sum(x["bb"] for x in st["rel_hist"])
        app = sum(x["app"] for x in st["rel_hist"])
        return dict(ip=ip, pit=pit, kbb=kbb, app=app)

    def tz7(st, today):
        cutoff = today - pd.Timedelta(days=7)
        st["tz_hist"] = [x for x in st["tz_hist"] if x[0] >= cutoff]
        if not st["tz_hist"]:
            return 0.0
        return float(sum(x[1] for x in st["tz_hist"]))

    def rates(st):
        h_ = np.array(st["h"], dtype=float); ab = np.array(st["ab"], dtype=float)
        bb = np.array(st["bb"], dtype=float); hbp = np.array(st["hbp"], dtype=float)
        sf = np.array(st["sf"], dtype=float); d = np.array(st["d"], dtype=float)
        t = np.array(st["t"], dtype=float); hr = np.array(st["hr"], dtype=float)
        so = np.array(st["so"], dtype=float); outs = np.array(st["outs"], dtype=float)
        if len(ab) == 0:
            return dict(obp=np.nan, slg=np.nan, krate=np.nan, bbrate=np.nan, ops=np.nan, woba=np.nan)
        obp = (h_.sum() + bb.sum() + hbp.sum()) / max(ab.sum() + bb.sum() + hbp.sum() + sf.sum(), 1)
        slg = (h_.sum() + d.sum() + 2 * t.sum() + 3 * hr.sum()) / max(ab.sum(), 1)
        inn = outs.sum() / 3.0
        krate = so.sum() / inn if inn > 0 else np.nan
        bbrate = bb.sum() / inn if inn > 0 else np.nan
        pa = ab.sum() + bb.sum() + hbp.sum() + sf.sum()
        singles = h_.sum() - d.sum() - t.sum() - hr.sum()
        woba = (0.69 * bb.sum() + 0.72 * hbp.sum() + 0.88 * singles + 1.247 * d.sum()
                + 1.578 * t.sum() + 2.031 * hr.sum()) / max(pa, 1)
        return dict(obp=float(obp), slg=float(slg), krate=float(krate), bbrate=float(bbrate),
                    ops=float(obp + slg), woba=float(woba))

    # ---------- statcast SP features (season-to-date / last-10, per mlbam id) ----------
    def sp_sc_feats(mlbam, year, today):
        hist = sp_state[mlbam]["starts"] if mlbam and mlbam >= 0 else []
        st = [x for x in hist if x["date"].year == year]
        l10 = hist[-10:]
        out = {}
        for tag, src in (("st", st), ("l10", l10)):
            if not src:
                out.update({f"sp_sc_ip_{tag}": np.nan, f"sp_sc_k9_{tag}": np.nan,
                            f"sp_sc_bb9_{tag}": np.nan, f"sp_sc_hr9_{tag}": np.nan,
                            f"sp_sc_velo_{tag}": np.nan, f"sp_sc_ra_{tag}": np.nan,
                            f"sp_sc_starts_{tag}": 0.0})
                continue
            ip = sum(x["outs"] for x in src) / 3.0
            k = sum(x["k"] for x in src); bb = sum(x["bb"] for x in src)
            hr = sum(x["hr"] for x in src); ra = sum(x["team_ra"] for x in src)
            velos = [x["velo"] for x in src if x["velo"] == x["velo"]]
            out[f"sp_sc_ip_{tag}"] = ip / len(src) if tag == "l10" else ip
            out[f"sp_sc_k9_{tag}"] = 9 * k / ip if ip > 0 else np.nan
            out[f"sp_sc_bb9_{tag}"] = 9 * bb / ip if ip > 0 else np.nan
            out[f"sp_sc_hr9_{tag}"] = 9 * hr / ip if ip > 0 else np.nan
            out[f"sp_sc_velo_{tag}"] = float(np.mean(velos)) if velos else np.nan
            out[f"sp_sc_ra_{tag}"] = ra / len(src)
            out[f"sp_sc_starts_{tag}"] = float(len(src))
        out["sp_sc_rest"] = (today - hist[-1]["date"]).days if hist else np.nan
        return out

    hs_sc = sp_sc_feats(int(g.sp_mlbam_h) if pd.notna(g.sp_mlbam_h) else -1, date.year, date)
    as_sc = sp_sc_feats(int(g.sp_mlbam_a) if pd.notna(g.sp_mlbam_a) else -1, date.year, date)

    # ---------- retro SP features (v1) ----------
    def sp_feat(pid, today):
        s = sp_retro[pid]
        if s["starts"] == 0:
            return dict(sp_exp=0.0, sp_wpct=np.nan, sp_ra=np.nan, sp_rest=np.nan)
        return dict(
            sp_exp=float(s["starts"]),
            sp_wpct=float(s["wins_in_starts"] / s["starts"]),
            sp_ra=float(s["ra_in_starts"] / s["starts"]),
            sp_rest=float((today - s["last_start"]).days) if s["last_start"] is not None else np.nan,
        )

    hspp = sp_feat(g.HomeStartingPitcherID, date)
    aspp = sp_feat(g.VisitingStartingPitcherID, date)
    rh = rates(h)
    ra_ = rates(a)
    bph = rel3(h, date)
    bpa = rel3(a, date)
    shift_away = tz_offset(g.home_std, date) - tz_offset(g.away_std, date)
    tz7_away = tz7(a, date)

    feat = {
        "date": date, "season": g.season, "doubleheader": int(g.DoubleHeader),
        "home": home, "away": away, "home_std": g.home_std, "away_std": g.away_std,
        "park": g.ParkID, "day_night": 1 if g.DayNight == "D" else 0,
        "home_win": int(g.home_win),
        "home_runs": int(hs), "away_runs": int(vs),
        "p_market_home": g.p_market_home, "dec_home": g.dec_home, "dec_away": g.dec_away,
        "has_odds": int(g.has_odds), "n_books": g.n_books,
        # elo
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
        "ewrs_h": ew(h["rs"], 0.056), "ewrs_a": ew(a["rs"], 0.056),
        "ewra_h": ew(h["ra"], 0.056), "ewra_a": ew(a["ra"], 0.056),
        # rate stats
        "obp_h": rh["obp"], "obp_a": ra_["obp"],
        "slg_h": rh["slg"], "slg_a": ra_["slg"],
        "krate_h": rh["krate"], "krate_a": ra_["krate"],
        "bbrate_h": rh["bbrate"], "bbrate_a": ra_["bbrate"],
        # v2: lineup quality
        "ops_h": rh["ops"], "ops_a": ra_["ops"],
        "ops_l10_h": roll(h["ops"], 10), "ops_l10_a": roll(a["ops"], 10),
        "woba_h": rh["woba"], "woba_a": ra_["woba"],
        # rest / travel
        "rest_h": rest(h, date), "rest_a": rest(a, date),
        "tz_shift": shift_away, "tz_abs": abs(shift_away), "away_tz7": tz7_away,
        # bullpen fatigue v1
        "inn3_h": inn3(h, date), "inn3_a": inn3(a, date),
        "xtra3_h": xtra3(h, date), "xtra3_a": xtra3(a, date),
        "relw_h": win_share(h["relief_w_14d"], h["games_14d"]), "relw_a": win_share(a["relief_w_14d"], a["games_14d"]),
        "rell_h": win_share(h["relief_l_14d"], h["games_14d"]), "rell_a": win_share(a["relief_l_14d"], a["games_14d"]),
        # bullpen fatigue v2 (statcast relief workload, last 3 days)
        "bp_ip3_h": bph["ip"], "bp_ip3_a": bpa["ip"],
        "bp_pit3_h": bph["pit"], "bp_pit3_a": bpa["pit"],
        "bp_kbb3_h": bph["kbb"], "bp_kbb3_a": bpa["kbb"],
        "bp_app3_h": bph["app"], "bp_app3_a": bpa["app"],
        # starting pitcher v1 (retro)
        "sp_exp_h": hspp["sp_exp"], "sp_exp_a": aspp["sp_exp"],
        "sp_wpct_h": hspp["sp_wpct"], "sp_wpct_a": aspp["sp_wpct"],
        "sp_ra_h": hspp["sp_ra"], "sp_ra_a": aspp["sp_ra"],
        "sp_rest_h": hspp["sp_rest"], "sp_rest_a": aspp["sp_rest"],
        # starting pitcher v2 (statcast)
        **{k + "_h": v for k, v in hs_sc.items()},
        **{k + "_a": v for k, v in as_sc.items()},
        # park factors (game level, home park)
        "pf_basic5": g.pf_basic5, "pf_3yr": g.pf_3yr, "pf_hr5": g.pf_hr5,
    }
    rows.append(feat)

    # ---------------- update state (after feature row) ----------------
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

    # v2: relief history + ops + tz history
    for tm, side in ((home, "home"), (away, "away")):
        st = team_state[tm]
        r = dict(date=date,
                 outs=int(g[f"rel_outs_{side}"]) if pd.notna(g[f"rel_outs_{side}"]) else 0,
                 pitches=int(g[f"rel_pitches_{side}"]) if pd.notna(g[f"rel_pitches_{side}"]) else 0,
                 k=int(g[f"rel_k_{side}"]) if pd.notna(g[f"rel_k_{side}"]) else 0,
                 bb=int(g[f"rel_bb_{side}"]) if pd.notna(g[f"rel_bb_{side}"]) else 0,
                 app=int(g[f"rel_app_{side}"]) if pd.notna(g[f"rel_app_{side}"]) else 0)
        st["rel_hist"].append(r)
    # ops per game (from this game's box, appended after feature emission)
    h["ops"].append(float(hs))  # placeholder: real OPS appended below via box stats
    a["ops"].append(float(vs))
    h["ops"][-1] = float((hstats["h"] + hstats["bb"] + hstats["hbp"])
                         / max(hstats["ab"] + hstats["bb"] + hstats["hbp"] + hstats["sf"], 1)
                         + (hstats["h"] + hstats["d"] + 2 * hstats["t"] + 3 * hstats["hr"])
                         / max(hstats["ab"], 1))
    a["ops"][-1] = float((astats["h"] + astats["bb"] + astats["hbp"])
                         / max(astats["ab"] + astats["bb"] + astats["hbp"] + astats["sf"], 1)
                         + (astats["h"] + astats["d"] + 2 * astats["t"] + 3 * astats["hr"])
                         / max(astats["ab"], 1))
    # tz history for away team only (travel shifts)
    a["tz_hist"].append((date, shift_away))

    # retro SP state (v1)
    for pid, won, ra_for in [(g.HomeStartingPitcherID, bool(g.home_win), float(vs)),
                             (g.VisitingStartingPitcherID, bool(not g.home_win), float(hs))]:
        s = sp_retro[pid]
        s["starts"] += 1
        s["wins_in_starts"] += int(won)
        s["ra_in_starts"] += ra_for
        s["last_start"] = date

    # statcast SP state
    for mlbam, won, team_ra, outs, kk, bb, hr, velo, pitches in [
        (g.sp_mlbam_h, bool(g.home_win), float(vs), g.sp_outs_h, g.sp_k_h, g.sp_bb_h, g.sp_hr_h, g.sp_velo_h, g.sp_pitches_h),
        (g.sp_mlbam_a, bool(not g.home_win), float(hs), g.sp_outs_a, g.sp_k_a, g.sp_bb_a, g.sp_hr_a, g.sp_velo_a, g.sp_pitches_a),
    ]:
        if pd.isna(mlbam) or int(mlbam) < 0:
            continue
        st = sp_state[int(mlbam)]
        st["starts"].append(dict(date=date, outs=int(outs) if pd.notna(outs) else 0,
                                 k=int(kk) if pd.notna(kk) else 0,
                                 bb=int(bb) if pd.notna(bb) else 0,
                                 hr=int(hr) if pd.notna(hr) else 0,
                                 team_ra=team_ra, velo=float(velo) if pd.notna(velo) else np.nan))

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
DIFF_PAIRS = [("rs", "rs_h", "rs_a"), ("ra", "ra_h", "ra_a"),
              ("rs_l10", "rs_h_l10", "rs_a_l10"), ("ra_l10", "ra_h_l10", "ra_a_l10")]
for c in ["winpct", "obp", "slg", "krate", "bbrate", "ewrs", "ewra", "rest", "inn3",
          "xtra3", "relw", "rell", "sp_exp", "sp_wpct", "sp_ra", "sp_rest", "ops",
          "ops_l10", "woba", "bp_ip3", "bp_pit3", "bp_kbb3", "bp_app3",
          "sp_sc_ip_st", "sp_sc_k9_st", "sp_sc_bb9_st", "sp_sc_hr9_st",
          "sp_sc_velo_st", "sp_sc_ra_st", "sp_sc_starts_st", "sp_sc_ip_l10",
          "sp_sc_k9_l10", "sp_sc_bb9_l10", "sp_sc_hr9_l10", "sp_sc_velo_l10",
          "sp_sc_ra_l10", "sp_sc_starts_l10", "sp_sc_rest"]:
    DIFF_PAIRS.append((c, f"{c}_h", f"{c}_a"))
for name, hc, ac in DIFF_PAIRS:
    if hc in feat.columns and ac in feat.columns:
        feat[f"{name}_diff"] = feat[hc] - feat[ac]

feat.to_parquet(OUT_PATH)
print(feat.shape)
print("games with odds:", int(feat.has_odds.sum()), "of", len(feat))
print("statcast SP coverage 2022:", int(feat[feat.season == 2022].sp_sc_ip_st_h.notna().sum()),
      "of", int((feat.season == 2022).sum()))
print("odds coverage by month (2022):")
print(feat[feat.season == 2022].groupby(feat.date.dt.to_period("M")).agg(
    n=("date", "size"), n_odds=("has_odds", "sum")).to_string())
