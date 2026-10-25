"""NHL v2 features: v1 (xG/corsi/PDO/ST/sched/Elo) + starting-goalie save%,
injury-report counts (HR snapshots), schedule density + travel, refined special
teams (20-game windows, PP/PK opportunity rates), Kalman-filtered team strength
(goals + xG), fast Elo. Strict no-lookahead walk-forward.
"""
import os

import numpy as np
import pandas as pd
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

tg = pd.read_parquet(os.path.join(D, "team_games.parquet"))
g = pd.read_parquet(os.path.join(D, "games_full.parquet"))
gk = pd.read_parquet(os.path.join(D, "goalie_lines.parquet"))
inj = pd.read_parquet(os.path.join(D, "injuries.parquet"))

g["date"] = pd.to_datetime(g["date"])
tg["date"] = pd.to_datetime(tg["date"])

PRIOR = dict(
    xg5=0.5, cf5=0.5, xga=0.5, gfa=0.5,
    sh=0.09, sv=0.905, pdo=0.995,
    ppg=0.55, pkga=0.55, ppopp=2.5, pkopp=2.5,
    pen=8.0, fo=0.5, gf=3.0, ga=3.0, pts=1.0,
)

# ---------- arena coordinates (lat, lon) for travel proxy ----------
ARENAS = {
    "ANA": (33.8078, -117.8766), "ARI": (33.5320, -112.2612), "BOS": (42.3663, -71.0622),
    "BUF": (42.8751, -78.8766), "CAR": (35.8032, -78.7219), "CBJ": (39.9693, -83.0061),
    "CGY": (51.0374, -114.0519), "CHI": (41.8807, -87.6742), "COL": (39.7487, -105.0077),
    "DAL": (32.7905, -96.8103), "DET": (42.3410, -83.0550), "EDM": (53.5467, -113.4977),
    "FLA": (26.1585, -80.3256), "LAK": (34.0430, -118.2673), "MIN": (44.9447, -93.1011),
    "MTL": (45.4961, -73.5693), "NJD": (40.7336, -74.1711), "NSH": (36.1592, -86.7785),
    "NYI": (40.7118, -73.7271), "NYR": (40.7505, -73.9934), "OTT": (45.2969, -75.9271),
    "PHI": (39.9012, -75.1720), "PIT": (40.4393, -79.9893), "SEA": (47.6221, -122.3541),
    "SJS": (37.3329, -121.9011), "STL": (38.6268, -90.2026), "TBL": (27.9427, -82.4518),
    "TOR": (43.6435, -79.3791), "VAN": (49.2778, -123.1088), "VGK": (36.1029, -115.1784),
    "WPG": (49.8928, -97.1436), "WSH": (38.8981, -77.0208),
}


def haversine(a, b):
    lat1, lon1 = np.radians(a[0]), np.radians(a[1])
    lat2, lon2 = np.radians(b[0]), np.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 7917.5 * np.arcsin(np.sqrt(h))  # earth diameter miles


win_map = {(r.game_id, r.home_code): int(r.home_win) for _, r in g.iterrows()}
win_map.update({(r.game_id, r.away_code): 1 - int(r.home_win) for _, r in g.iterrows()})
opp_map = {(r.game_id, r.home_code): r.away_code for _, r in g.iterrows()}
opp_map.update({(r.game_id, r.away_code): r.home_code for _, r in g.iterrows()})

state = {}
last_date = {}
last_venue = {}
last_venues7 = {}
feat_rows = []
tg = tg.sort_values("date").reset_index(drop=True)


def ewma_last(vals, span):
    return pd.Series(vals, dtype=float).ewm(span=span, adjust=True).mean().iloc[-1]


# ---------- Kalman filter on goal differential ----------
TEAMS = sorted(g.home_code.unique())
TIDX = {t: i for i, t in enumerate(TEAMS)}
N = len(TEAMS)
x = np.zeros(N + 1)
P = np.eye(N + 1) * 0.25
R_GOALS = 5.0
Q_PER_DAY = 0.0008
last_kal_date = None
k_strength_before = {}
k_hadv_before = {}
k_pred_before = {}

# second Kalman on xG differential
x2 = np.zeros(N + 1)
P2 = np.eye(N + 1) * 0.25
R_XG = 2.5


def kalman_update(xx, PP, H, z, R, q_dt):
    S = float(H @ PP @ H.T) + R
    K = PP @ H.T / S
    resid = float(z - H @ xx)
    xx = xx + K * resid
    PP = (np.eye(len(xx)) - np.outer(K, H)) @ PP
    PP += np.eye(len(xx)) * q_dt
    return xx, PP


for i, r in tg.iterrows():
    t = r.team

    def ew(key, span=10):
        s = state[t].get(key) if t in state else None
        return ewma_last(s if s else [PRIOR[key]], span)

    feats = dict(
        gameId=r.gameId, team=t, home_or_away=r.home_or_away, date=r.date,
        xg_share_5v5_10=ew("xg5", 10), xg_share_5v5_5=ew("xg5", 5), xg_share_5v5_20=ew("xg5", 20),
        cf_share_5v5=ew("cf5", 10),
        xg_share_all=ew("xga", 10), gf_share_all=ew("gfa", 10),
        pdo=ew("pdo", 10), pdo_20=ew("pdo", 20), sh_pct=ew("sh", 10), sv_pct=ew("sv", 10),
        pp_eff=ew("ppg", 10), pk_eff=ew("pkga", 10),
        pen_mins=ew("pen", 10), fo_pct=ew("fo", 10),
        gf_net=ew("gf", 10) - ew("ga", 10),
        pts_pct=ew("pts", 10),
        rest=min((r.date - last_date[t]).days, 7) if t in last_date else 7,
        b2b=int(t in last_date and (r.date - last_date[t]).days == 1),
        # ---- v2: special teams refined ----
        pp_eff_20=ew("ppg", 20), pk_eff_20=ew("pkga", 20),
        pp_opp_rate=ew("ppopp", 10), pk_opp_rate=ew("pkopp", 10),
        pp_net=ew("ppg", 10) * ew("ppopp", 10) - ew("pkga", 10) * ew("pkopp", 10),
    )
    # ---- v2: schedule density ----
    # venue where THIS game is played
    if r.home_or_away == "HOME":
        this_venue = ARENAS.get(t, (45.0, -75.0))
    else:
        opp = opp_map.get((r.gameId, t), t)
        this_venue = ARENAS.get(opp, (45.0, -75.0))
    if t in last_date:
        g7 = [d for d in state[t].get("dates7", []) if (r.date - d).days < 7]
        g14 = [d for d in state[t].get("dates14", []) if (r.date - d).days < 14]
        hom7 = [d for d in state[t].get("homedates7", []) if (r.date - d).days < 7]
        feats["games7"] = len(g7)
        feats["games14"] = len(g14)
        feats["home7"] = len(hom7)
        feats["away_streak"] = state[t].get("away_streak", 0) if r.home_or_away == "AWAY" else 0
        feats["home_streak"] = state[t].get("home_streak", 0) if r.home_or_away == "HOME" else 0
        # travel: distance from previous game venue to this venue
        prev_venue = last_venue[t]
        feats["travel"] = haversine(prev_venue, this_venue) if prev_venue is not None else 0.0
        v7 = [(d, v) for d, v in last_venues7.get(t, []) if (r.date - d).days < 7]
        travel7 = sum(haversine(v7[k][1], v7[k + 1][1]) for k in range(len(v7) - 1)) if len(v7) > 1 else 0.0
        feats["travel7"] = min(travel7, 4000.0)
    else:
        feats.update(games7=0, games14=0, home7=0, away_streak=0, home_streak=0,
                     travel=0.0, travel7=0.0)
    feat_rows.append(feats)
    last_date[t] = r.date

    # ---- update state with this game ----
    if t not in state:
        state[t] = {}
    st = state[t]

    def push(key, val, cap=30):
        st.setdefault(key, []).append(val)
        if len(st[key]) > cap:
            st[key].pop(0)

    gf, ga = r.goalsFor_all, r.goalsAgainst_all
    xgf5, xga5 = r.xGoalsFor_5on5, r.xGoalsAgainst_5on5
    cff5, cfa5 = r.shotAttemptsFor_5on5, r.shotAttemptsAgainst_5on5
    xgf_a, xga_a = r.xGoalsFor_all, r.xGoalsAgainst_all
    push("xg5", xgf5 / (xgf5 + xga5) if (xgf5 + xga5) > 0 else 0.5)
    push("cf5", cff5 / (cff5 + cfa5) if (cff5 + cfa5) > 0 else 0.5)
    push("xga", xgf_a / (xgf_a + xga_a) if (xgf_a + xga_a) > 0 else 0.5)
    push("gfa", gf / (gf + ga) if (gf + ga) > 0 else 0.5)
    sf, sa = r.shotsOnGoalFor_all, r.shotsOnGoalAgainst_all
    shp = gf / sf if sf > 0 else 0.09
    svp = 1 - ga / sa if sa > 0 else 0.905
    push("sh", shp)
    push("sv", svp)
    push("pdo", shp + svp)
    pp_opps = r.iceTime_5on4 / 120.0
    pk_opps = r.iceTime_4on5 / 120.0
    push("ppg", r.goalsFor_5on4 / pp_opps if pp_opps > 0 else PRIOR["ppg"])
    push("pkga", r.goalsAgainst_4on5 / pk_opps if pk_opps > 0 else PRIOR["pkga"])
    push("ppopp", pp_opps)
    push("pkopp", pk_opps)
    push("pen", r.penalityMinutesFor_all)
    fow, foa = r.faceOffsWonFor_all, r.faceOffsWonAgainst_all
    push("fo", fow / (fow + foa) if (fow + foa) > 0 else 0.5)
    push("gf", gf)
    push("ga", ga)
    won = win_map.get((r.gameId, t), int(gf > ga))
    pts = 2.0 if won else (1.0 if gf == ga else 0.0)
    push("pts", pts / 2.0)
    # schedule bookkeeping
    st.setdefault("dates7", []).append(r.date)
    st.setdefault("dates14", []).append(r.date)
    if len(st["dates7"]) > 10:
        st["dates7"].pop(0)
    if len(st["dates14"]) > 20:
        st["dates14"].pop(0)
    if r.home_or_away == "HOME":
        st.setdefault("homedates7", []).append(r.date)
        if len(st["homedates7"]) > 10:
            st["homedates7"].pop(0)
        st["home_streak"] = st.get("home_streak", 0) + 1
        st["away_streak"] = 0
    else:
        st["away_streak"] = st.get("away_streak", 0) + 1
        st["home_streak"] = 0
    venue = this_venue
    st.setdefault("venues7", []).append((r.date, venue))
    if len(st["venues7"]) > 12:
        st["venues7"].pop(0)
    last_venue[t] = venue
    last_venues7[t] = list(st["venues7"])

feats = pd.DataFrame(feat_rows)

# ---------- Kalman filter (chronological pass over games) ----------
g_sorted = g.sort_values("date")
kal_rows = []
for _, r in g_sorted.iterrows():
    ih, ia = TIDX[r.home_code], TIDX[r.away_code]
    z_gd = r.mp_hg - r.mp_ag
    # xG diff from team_games
    hrow = tg[(tg.gameId == r.game_id) & (tg.team == r.home_code)]
    arow = tg[(tg.gameId == r.game_id) & (tg.team == r.away_code)]
    if len(hrow) and len(arow):
        z_xg = hrow.iloc[0].xGoalsFor_all - arow.iloc[0].xGoalsFor_all
    else:
        z_xg = 0.0
    kal_rows.append(dict(game_id=r.game_id, home_code=r.home_code, away_code=r.away_code,
                         k_h=x[ih], k_a=x[ia], k_hadv=x[N], k_pred=float(x[ih] - x[ia] + x[N]),
                         kx_h=x2[ih], kx_a=x2[ia], kx_pred=float(x2[ih] - x2[ia] + x2[N])))
    H = np.zeros(N + 1)
    H[ih] = 1.0
    H[ia] = -1.0
    H[N] = 1.0
    dt = 1.0
    x, P = kalman_update(x, P, H, z_gd, R_GOALS, Q_PER_DAY * dt)
    x2, P2 = kalman_update(x2, P2, H, z_xg, R_XG, Q_PER_DAY * dt)
kal = pd.DataFrame(kal_rows)

# ---------- Elo (v1 K=6) + fast Elo (K=20) ----------
elo = defaultdict(lambda: 1500.0)
elo_fast = defaultdict(lambda: 1500.0)
elo_before = {}
elo_fast_before = {}
for _, r in g_sorted.iterrows():
    h, a = r.home_code, r.away_code
    elo_before[(r.game_id, h)] = elo[h]
    elo_before[(r.game_id, a)] = elo[a]
    elo_fast_before[(r.game_id, h)] = elo_fast[h]
    elo_fast_before[(r.game_id, a)] = elo_fast[a]
    res_h = 1 if r.home_win == 1 else 0
    gd = abs(r.home_goals - r.away_goals)
    mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else 2.0)
    we_h = 1 / (1 + 10 ** ((elo[a] - (elo[h] + 35.0)) / 400))
    elo[h] += 6.0 * mult * (res_h - we_h)
    elo[a] += 6.0 * mult * ((1 - res_h) - (1 - we_h))
    we_h = 1 / (1 + 10 ** ((elo_fast[a] - (elo_fast[h] + 35.0)) / 400))
    elo_fast[h] += 20.0 * mult * (res_h - we_h)
    elo_fast[a] += 20.0 * mult * ((1 - res_h) - (1 - we_h))

feats["elo_before"] = [elo_before.get((gid, t), 1500.0) for gid, t in zip(feats.gameId, feats.team)]
feats["elo_fast_before"] = [elo_fast_before.get((gid, t), 1500.0)
                            for gid, t in zip(feats.gameId, feats.team)]
feats["k_strength"] = [kal.set_index("game_id").loc[gid, ("k_h" if side == "HOME" else "k_a")]
                       if gid in kal.game_id.values else 0.0
                       for gid, side in zip(feats.gameId, feats.home_or_away)]
feats["kx_strength"] = [kal.set_index("game_id").loc[gid, ("kx_h" if side == "HOME" else "kx_a")]
                        if gid in kal.game_id.values else 0.0
                        for gid, side in zip(feats.gameId, feats.home_or_away)]

# ---------- goalie join ----------
gk_join = gk[["gameId", "team", "gk_sv_cum", "gk_sv_ewma", "gk_sv_ewma20",
              "gk_starts", "gk_starts7"]].copy()
feats = feats.merge(gk_join, on=["gameId", "team"], how="left")
# gk_share = starts / team games before this game
tg_sorted = tg.sort_values("date")
team_games_before = {}
share_rows = []
for _, r in tg_sorted.iterrows():
    tgb = team_games_before.get(r.team, 0)
    share_rows.append(dict(gameId=r.gameId, team=r.team, tgb=tgb))
    team_games_before[r.team] = tgb + 1
sh = pd.DataFrame(share_rows)
feats = feats.merge(sh, on=["gameId", "team"], how="left")
feats["gk_share"] = feats["gk_starts"] / feats["tgb"].clip(lower=1)
feats.drop(columns=["tgb"], inplace=True)
gk_missing = feats.gk_sv_ewma.isna().sum()
feats["gk_sv_ewma"] = feats["gk_sv_ewma"].fillna(0.905)
feats["gk_sv_ewma20"] = feats["gk_sv_ewma20"].fillna(0.905)
feats["gk_sv_cum"] = feats["gk_sv_cum"].fillna(0.905)
feats["gk_starts"] = feats["gk_starts"].fillna(0)
feats["gk_starts7"] = feats["gk_starts7"].fillna(0)
feats["gk_share"] = feats["gk_share"].fillna(0.0)
print("goalie joins missing (filled w/ prior):", gk_missing)

# ---------- injuries (snapshot-quantized, no lookahead) ----------
inj = inj.sort_values("snapshot_date")
snap_ts = sorted(inj.snapshot_date.unique())
inj_feats = {}
inj_long = inj.note.str.lower().str.contains("season|indefinite", na=False)
inj["long"] = inj_long.values


def inj_for(team, d):
    """Counts for team as of game date d using the latest snapshot <= d."""
    snaps_le = [s for s in snap_ts if s <= d]
    if not snaps_le:
        return dict(inj_out=0, inj_recent14=0, inj_season=0, inj_avail=0)
    s = snaps_le[-1]
    sub = inj[(inj.snapshot_date == s) & (inj.team == team) & (inj.injury_date <= d)]
    if len(sub) == 0:
        return dict(inj_out=0, inj_recent14=0, inj_season=0, inj_avail=1)
    days = (d - sub.injury_date).dt.days
    out = ((days <= 30) | sub.long).sum()
    recent = ((days > 0) & (days <= 14)).sum()
    return dict(inj_out=int(out), inj_recent14=int(recent), inj_season=int(len(sub)),
                inj_avail=1)


ij_rows = []
for i, r in feats.iterrows():
    ij_rows.append(inj_for(r.team, r.date))
ij = pd.DataFrame(ij_rows, index=feats.index)
feats = pd.concat([feats, ij], axis=1)
print("inj feature counts:")
print(feats[["inj_out", "inj_recent14", "inj_season"]].describe().round(1).to_string())

# ---------- home/away diffs ----------
feat_cols = [c for c in feats.columns
             if c not in ("gameId", "team", "home_or_away", "date", "inj_avail")]
home = feats[feats.home_or_away == "HOME"].set_index("gameId")
away = feats[feats.home_or_away == "AWAY"].set_index("gameId")
home = home.rename(columns={c: f"h_{c}" for c in feat_cols})
away = away.rename(columns={c: f"a_{c}" for c in feat_cols})
pair = home.join(away, how="inner", lsuffix="_h", rsuffix="_a")
for c in feat_cols:
    if c == "elo_before" or c == "elo_fast_before":
        pair[f"diff_{c}"] = (pair[f"h_{c}"] + 35.0) - pair[f"a_{c}"]
    else:
        pair[f"diff_{c}"] = pair[f"h_{c}"] - pair[f"a_{c}"]
pair = pair.reset_index().rename(columns={"index": "gameId"})

out = g.merge(pair[["gameId"] + [f"diff_{c}" for c in feat_cols]],
              left_on="game_id", right_on="gameId", how="inner")
assert len(out) == len(g), (len(out), len(g))
out.to_parquet(os.path.join(D, "games_features_v2.parquet"), index=False)
print("games with features:", len(out), "| n features:", len(feat_cols))
print("feature list:")
print(sorted(f"diff_{c}" for c in feat_cols))
