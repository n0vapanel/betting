"""
ATP tennis v2 backtest (2021 train -> 2022 test). Accuracy push over v1.

Upgrades over v1 (backtest/multisport/tennis/run_tennis_backtest.py):
  (a) head-to-head history: same-file prior meetings (count, recency-weighted
      win share, last-meeting winner, surface-specific share).
  (b) surface Elo: v1 kept a raw per-surface rating; v2 shrinks it toward the
      global rating by surface sample size (blend = (n*selo + 15*elo)/(n+15)).
  (c) serve/return stats: JeffSackmann tennis_atp atp_matches 2019-2022 CSVs
      (GitHub fork Kadantte/tennis_atp, repo deleted upstream). Columns
      w_svpt/w_ace/w_df/w_1stIn/w_1stWon/w_2ndWon/w_bpSaved/w_bpFaced (+l_*).
      Joined exactly by player_id; td-name->player_id via fuzzy matcher.
  (d) fatigue: matches/games/sets/minutes in last 14 days, retirement (as
      loser) within 30 days, mean minutes of last 5 matches.
  (e) match duration (minutes) from the Sackmann CSVs (not in tennis-data).

Protocol unchanged from v1: single forward pass, no lookahead (every feature
uses strictly earlier matches), P1 = higher pre-match global Elo (rank
tie-break), y = P1 wins, market prob = no-vig B365 closing odds. LogReg+LGBM
soft ensemble, isotonic on OOF, expanding-window time-ordered 5-fold CV on
2021 for hyperparams + primary-model choice. Test 2022, flat $1 at B365
closing decimal odds, edge gates eps 0/0.02/0.05, quarter-Kelly on best gate,
bootstrap CI clustered by (year, iso-week).
"""
import itertools
import json
import re
from collections import deque, defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

WORK = "/Users/x/Desktop/x/Coding Projects/betting bot/backtest/multisport/tennis_v2"
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Load tennis-data (all rows incl. retirements/walkovers for fatigue)
# ---------------------------------------------------------------------------
def load_td(year):
    df = pd.read_excel(f"{WORK}/{year}.xlsx")
    df = df[pd.to_datetime(df["Date"], errors="coerce").notna()].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df

td = pd.concat([load_td(2021), load_td(2022)], ignore_index=True)
td = td.sort_values("Date").reset_index(drop=True)
td["Best of"] = td["Best of"].fillna(3).astype(int)
td["completed"] = td["Comment"].fillna("").str.contains("Completed", na=False)
model_mask = td["completed"] & td["B365W"].notna() & td["B365L"].notna()
print(f"td rows {len(td)} | model rows {model_mask.sum()} "
      f"(train {(model_mask & (td['Date'] < pd.Timestamp('2022-01-01'))).sum()}, "
      f"test {(model_mask & (td['Date'] >= pd.Timestamp('2022-01-01'))).sum()})")

# ---------------------------------------------------------------------------
# 2. Players: concat archived + fork file; v1 fuzzy matcher td name -> row
# ---------------------------------------------------------------------------
players = pd.read_csv(f"{WORK}/raw/atp_players_archived.csv")
# note: atp_players.csv (Kadantte fork, 66k rows) is NOT used for matching —
# archived-only reproduces v1's exact age values; fork file kept in raw/ for reference.
players = players[players["dob"].notna()].copy()
players["dob"] = pd.to_datetime(
    players["dob"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
players = players.dropna(subset=["dob"]).reset_index(drop=True)
players["s_full"] = (players["name_last"].fillna("") + " ").str.lower().str.replace(
    r"[^a-z ]", "", regex=True).str.replace(" ", "")
players["s_tok"] = players["name_last"].fillna("").str.lower().str.split().str[-1]
players["s_init"] = players["name_first"].fillna("").str.lower().str.replace(
    r"[^a-z ]", "", regex=True).apply(lambda s: "".join(t[0] for t in s.split()))

tok_idx = defaultdict(list)
for i, row in players.iterrows():
    tok_idx[row["s_tok"]].append(i)

PARTICLES = {"de", "del", "la", "van", "von", "da", "di", "der", "den", "el", "le", "st", "dos"}


def td_parts(name):
    toks = [t.strip(" .") for t in str(name).split()]
    if not toks:
        return "", ""
    last = toks[0].lower()
    k = 1
    prev_particle = toks[0].lower() in PARTICLES
    while k < len(toks):
        t = toks[k].lower()
        if toks[k][:1].islower() or t in PARTICLES or prev_particle:
            last += t
            k += 1
        else:
            break
        prev_particle = t in PARTICLES
    init = "".join(t[:1].lower() for t in toks[k:])
    return re.sub(r"[^a-z]", "", last), init


all_names = sorted(set(td["Winner"]) | set(td["Loser"]))
name_map = {}          # td name -> dict(last, init, dob, player_id)
pid_of = {}            # td name -> player_id
for nm in all_names:
    td_last, td_init = td_parts(nm)
    if not td_last:
        continue
    best, best_len, best_dob, best_pid = None, 0, None, None
    cand = set()
    for t in re.findall(r"[a-z]+", nm.lower()):
        cand |= set(tok_idx.get(t, []))
    for i in cand:
        row = players.iloc[i]
        s_full, s_init, dob = row["s_full"], row["s_init"], row["dob"]
        if (td_last.endswith(s_full) or s_full.endswith(td_last)) and (
                td_init.startswith(s_init) or s_init.startswith(td_init)
                or td_init == "" or s_init == ""):
            if len(s_full) > best_len or (len(s_full) == best_len
                                          and (best_dob is None or dob > best_dob)):
                best, best_len, best_dob, best_pid = dob, len(s_full), dob, row["player_id"]
    if best is not None:
        name_map[nm] = dict(dob=best, player_id=best_pid)
        pid_of[nm] = best_pid

cov = sum(1 for m in td[model_mask][["Winner", "Loser"]].itertuples(index=False)
          if m[0] in name_map and m[1] in name_map)
print(f"dob/pid coverage: both {cov}/{model_mask.sum()} ({cov/model_mask.sum():.1%})")


def get_age(nm, date):
    e = name_map.get(nm)
    return np.nan if e is None else (date - e["dob"]).days / 365.25


# ---------------------------------------------------------------------------
# 3. Sackmann serve/return + minutes per (player_id, date)
# ---------------------------------------------------------------------------
sack = {}  # (player_id, date) -> stat dict for that player in that match
for y in (2019, 2020, 2021, 2022):
    df = pd.read_csv(f"{WORK}/raw/atp_matches_{y}.csv", low_memory=False)
    for _, r in df.iterrows():
        d = pd.to_datetime(str(int(r["tourney_date"])), format="%Y%m%d")
        wstat = dict(
            svpt=r["w_svpt"], ace=r["w_ace"], df=r["w_df"], firstIn=r["w_1stIn"],
            firstWon=r["w_1stWon"], secondWon=r["w_2ndWon"],
            bpFaced=r["w_bpFaced"], bpSaved=r["w_bpSaved"],
            o_svpt=r["l_svpt"], o_firstWon=r["l_1stWon"], o_secondWon=r["l_2ndWon"],
            o_bpFaced=r["l_bpFaced"], o_bpSaved=r["l_bpSaved"], minutes=r["minutes"])
        lstat = dict(
            svpt=r["l_svpt"], ace=r["l_ace"], df=r["l_df"], firstIn=r["l_1stIn"],
            firstWon=r["l_1stWon"], secondWon=r["l_2ndWon"],
            bpFaced=r["l_bpFaced"], bpSaved=r["l_bpSaved"],
            o_svpt=r["w_svpt"], o_firstWon=r["w_1stWon"], o_secondWon=r["w_2ndWon"],
            o_bpFaced=r["w_bpFaced"], o_bpSaved=r["w_bpSaved"], minutes=r["minutes"])
        if pd.notna(wstat["svpt"]) and wstat["svpt"] >= 20:
            sack[(r["winner_id"], d)] = wstat
        if pd.notna(lstat["svpt"]) and lstat["svpt"] >= 20:
            sack[(r["loser_id"], d)] = lstat
# sack stats per player, sorted by date (for fast-forward between td matches)
sack_by_pid = defaultdict(list)
for (pid, d), st in sack.items():
    sack_by_pid[pid].append((d, st))
for pid in sack_by_pid:
    sack_by_pid[pid].sort(key=lambda x: x[0])
print(f"sack matches with stats: {len(sack)} | players: {len(sack_by_pid)}")


def stat_ratios(st):
    """Per-match ratios + point weights for a player's serve record."""
    out = {}
    svpt, o_svpt = st["svpt"], st["o_svpt"]
    svWon = st["firstWon"] + st["secondWon"]
    retWon = o_svpt - (st["o_firstWon"] + st["o_secondWon"])
    out["svPtsWon"] = (svWon / svpt, svpt)
    out["retPtsWon"] = (retWon / o_svpt, o_svpt)
    out["acePct"] = (st["ace"] / svpt, svpt)
    out["dfPct"] = (st["df"] / svpt, svpt)
    out["firstWin"] = ((st["firstWon"] / st["firstIn"], st["firstIn"])
                       if pd.notna(st["firstIn"]) and st["firstIn"] > 5 else (np.nan, 0))
    s2 = svpt - st["firstIn"]
    out["secondWin"] = ((st["secondWon"] / s2, s2)
                        if pd.notna(s2) and s2 > 5 else (np.nan, 0))
    out["bpSavedPct"] = ((st["bpSaved"] / st["bpFaced"], st["bpFaced"])
                         if pd.notna(st["bpFaced"]) and st["bpFaced"] > 2 else (np.nan, 0))
    obf = st["o_bpFaced"]
    out["bpConvPct"] = (((obf - st["o_bpSaved"]) / obf, obf)
                        if pd.notna(obf) and obf > 2 else (np.nan, 0))
    return out


# ---------------------------------------------------------------------------
# 4. Forward pass over td timeline
# ---------------------------------------------------------------------------
TIER = {"ATP250": 1, "ATP500": 2, "Masters 1000": 3, "Grand Slam": 4, "Masters Cup": 5}
td["tier"] = td["Series"].map(TIER).fillna(2.0)
SURFACES = ["Hard", "Clay", "Grass"]
td["surf_idx"] = td["Surface"].map({s: i for i, s in enumerate(SURFACES)}).fillna(0).astype(int)
K = 32.0
ALPHA = 0.9  # EWMA decay per match for serve stats


def elo_init(rank):
    r = rank if pd.notna(rank) else 400.0
    return 2000.0 - 400.0 * np.log10(max(r, 1.0))


state = {
    "elo": {}, "selo": {},          # per player / per (player, surf)
    "nsurf": defaultdict(int),      # completed matches per (player, surf)
    "hist": defaultdict(list),      # (date, won, games_share, surf) per player
    # fatigue
    "fatq": defaultdict(deque),     # (date, sets, games, minutes, n_match)
    "fatsum": defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]),  # sets, games, minutes, matches
    "last5min": defaultdict(deque),  # last-5 match minutes
    "hurt_d": {},                   # last retirement date (as loser) per player
    # serve EWMA + career (per player_id)
    "s_ewma": {},                   # pid -> {stat: [num, den]}
    "s_car": {},                    # pid -> {stat: [num, den]}
    "s_ptr": defaultdict(int),      # pid -> sack index pointer
    # h2h
    "h2h": {},                      # pair -> [wsum, n, count, last_winner]
    "h2hs": {},                     # (pair, surf) -> [wsum, n]
}

STATS = ["svPtsWon", "retPtsWon", "acePct", "dfPct", "firstWin", "secondWin",
         "bpSavedPct", "bpConvPct"]


def ewma_add(pid, st):
    if pid is None:
        return
    e = state["s_ewma"].setdefault(pid, {s: [0.0, 0.0] for s in STATS})
    c = state["s_car"].setdefault(pid, {s: [0.0, 0.0] for s in STATS})
    for s, (v, w) in stat_ratios(st).items():
        if pd.notna(v) and w > 0:
            e[s][0] = ALPHA * e[s][0] + w * v
            e[s][1] = ALPHA * e[s][1] + w
            c[s][0] += w * v
            c[s][1] += w


def ewma_feat(pid, career=False):
    if pid is None:
        return {s: np.nan for s in STATS}
    src = state["s_car"] if career else state["s_ewma"]
    e = src.get(pid, {})
    return {s: (e[s][0] / e[s][1] if s in e and e[s][1] >= 30 else np.nan)
            for s in STATS}


def fastforward_sack(pid, before):
    """Apply sack matches with date < before (not yet applied)."""
    if pid is None or pid not in sack_by_pid:
        return
    lst = sack_by_pid[pid]
    p = state["s_ptr"][pid]
    while p < len(lst) and lst[p][0] < before:
        ewma_add(pid, lst[p][1])
        p += 1
    state["s_ptr"][pid] = p


def apply_sack_today(pid, d):
    """Apply the sack entry exactly at date d (the current match itself), once."""
    if pid is None or pid not in sack_by_pid:
        return
    lst = sack_by_pid[pid]
    p = state["s_ptr"][pid]
    if p < len(lst) and lst[p][0] == d:
        ewma_add(pid, lst[p][1])
        state["s_ptr"][pid] = p + 1


def rolling_stats(hist_rows, before_date, n):
    rows = [r for r in hist_rows if r[0] < before_date]
    if not rows:
        return np.nan, np.nan, np.nan
    recent = rows[-n:]
    wins = sum(1 for r in recent if r[1])
    return wins / len(recent), np.nanmean([r[2] for r in recent]), (before_date - rows[-1][0]).days


def surf_stats(hist_rows, before_date, surf, n):
    rows = [r for r in hist_rows if r[0] < before_date and r[3] == surf]
    if not rows:
        return np.nan, np.nan
    recent = rows[-n:]
    return sum(1 for r in recent if r[1]) / len(recent), np.nanmean([r[2] for r in recent])


def fatigue_feat(pname, d):
    q = state["fatq"].get(pname)
    if q:
        while q and (d - q[0][0]).days > 14:
            old = q.popleft()
            s = state["fatsum"][pname]
            s[0] -= old[1]; s[1] -= old[2]; s[2] -= old[3]; s[3] -= old[4]
        s = state["fatsum"][pname]
        sets14, games14, min14, match14 = s[0], s[1], s[2], s[3]
    else:
        sets14 = games14 = min14 = match14 = 0.0
    lm = state["last5min"].get(pname)
    avg5 = np.nanmean(list(lm)) if lm else np.nan
    hurt = 0
    hd = state["hurt_d"].get(pname)
    if hd is not None and (d - hd).days <= 30:
        hurt = 1
    return match14, games14, sets14, min14, avg5, hurt


rows_out = []
for i in td.index:
    m = td.loc[i]
    w, l = m["Winner"], m["Loser"]
    si = m["surf_idx"]
    d = m["Date"]
    is_model = bool(model_mask.loc[i])
    # fast-forward sack serve history to strictly before this match
    fastforward_sack(pid_of.get(w), d)
    fastforward_sack(pid_of.get(l), d)
    # current ratings
    e_w = state["elo"].get(w, elo_init(m["WRank"]))
    e_l = state["elo"].get(l, elo_init(m["LRank"]))
    se_w = state["selo"].get((w, si), e_w)
    se_l = state["selo"].get((l, si), e_l)
    nw, nl = state["nsurf"][(w, si)], state["nsurf"][(l, si)]
    b_w = (nw * se_w + 15.0 * e_w) / (nw + 15.0)
    b_l = (nl * se_l + 15.0 * e_l) / (nl + 15.0)
    # rolling form (v1 semantics, completed matches only)
    wf, wg, w_rest = rolling_stats(state["hist"][w], d, 10)
    lf, lg, l_rest = rolling_stats(state["hist"][l], d, 10)
    wsf, wsg = surf_stats(state["hist"][w], d, si, 10)
    lsf, lsg = surf_stats(state["hist"][l], d, si, 10)
    wsc, _ = surf_stats(state["hist"][w], d, si, 40)
    lsc, _ = surf_stats(state["hist"][l], d, si, 40)
    # fatigue
    wm14, wg14, ws14, wmin14, wavg5, whurt = fatigue_feat(w, d)
    lm14, lg14, ls14, lmin14, lavg5, lhurt = fatigue_feat(l, d)
    # serve EWMA + career
    wse = ewma_feat(pid_of.get(w))
    lse = ewma_feat(pid_of.get(l))
    wsc2 = ewma_feat(pid_of.get(w), career=True)
    lsc2 = ewma_feat(pid_of.get(l), career=True)
    # h2h
    key = tuple(sorted((w, l)))
    first = key[0]
    h = state["h2h"].get(key)
    if h is not None:
        h2h_n = h[2]
        share_first = h[0] / h[1] if h[1] > 0 else np.nan
        h2h_share = share_first if w == first else (1.0 - share_first if pd.notna(share_first) else np.nan)
        h2h_last = 1 if h[3] == w else -1
    else:
        h2h_n, h2h_share, h2h_last = 0, np.nan, 0
    hs = state["h2hs"].get((key, si))
    if hs is not None and hs[1] > 0:
        share_first_s = hs[0] / hs[1]
        h2h_ss = share_first_s if w == first else 1.0 - share_first_s
    else:
        h2h_ss = np.nan
    rows_out.append({
        "idx": i, "Date": d, "Winner": w, "Loser": l, "Surface": m["Surface"],
        "surf_idx": si, "tier": m["tier"], "Best of": m["Best of"],
        "WRank": m["WRank"], "LRank": m["LRank"],
        "B365W": m["B365W"], "B365L": m["B365L"],
        "Tournament": m["Tournament"], "Location": m["Location"],
        "is_model": is_model,
        "elo_w": e_w, "elo_l": e_l, "selo_w": se_w, "selo_l": se_l,
        "selob_w": b_w, "selob_l": b_l,
        "form_w": wf, "form_l": lf, "games_w": wg, "games_l": lg,
        "rest_w": w_rest, "rest_l": l_rest,
        "sform_w": wsf, "sform_l": lsf, "sgames_w": wsg, "sgames_l": lsg,
        "scareer_w": wsc, "scareer_l": lsc,
        "m14_w": wm14, "m14_l": lm14, "g14_w": wg14, "g14_l": lg14,
        "s14_w": ws14, "s14_l": ls14, "min14_w": wmin14, "min14_l": lmin14,
        "avg5_w": wavg5, "avg5_l": lavg5, "hurt_w": whurt, "hurt_l": lhurt,
        "h2h_n": h2h_n, "h2h_share": h2h_share, "h2h_last": h2h_last,
        "h2h_ss": h2h_ss,
        **{f"s_{s}_w": wse[s] for s in STATS},
        **{f"s_{s}_l": lse[s] for s in STATS},
        **{f"c_{s}_w": wsc2[s] for s in STATS},
        **{f"c_{s}_l": lsc2[s] for s in STATS},
    })
    # ---- update state AFTER features (no lookahead) ----
    # fatigue: all td rows (retirements/walkovers also consume energy)
    games = sum(m.get(f"W{k}") for k in range(1, 6) if pd.notna(m.get(f"W{k}"))) + \
        sum(m.get(f"L{k}") for k in range(1, 6) if pd.notna(m.get(f"L{k}")))
    sets_p = (m["Wsets"] if pd.notna(m["Wsets"]) else 0) + \
        (m["Lsets"] if pd.notna(m["Lsets"]) else 0)
    minutes_cur = None
    pidw, pidl = pid_of.get(w), pid_of.get(l)
    stw = sack.get((pidw, d)) if pidw else None
    stl = sack.get((pidl, d)) if pidl else None
    if stw is not None and pd.notna(stw["minutes"]):
        minutes_cur = stw["minutes"]
    elif stl is not None and pd.notna(stl["minutes"]):
        minutes_cur = stl["minutes"]
    for pname in (w, l):
        q = state["fatq"][pname]
        q.append((d, sets_p, games, minutes_cur or 0.0, 1))
        s = state["fatsum"][pname]
        s[0] += sets_p; s[1] += games; s[2] += minutes_cur or 0.0; s[3] += 1
        if minutes_cur is not None:
            lm = state["last5min"][pname]
            lm.append(minutes_cur)
            if len(lm) > 5:
                lm.popleft()
    # retirement flag (loser retired -> injury risk for loser)
    if m["completed"] is not True and "retir" in str(m["Comment"]).lower():
        state["hurt_d"][l] = d
    # apply the current match's sack stats for FUTURE matches
    apply_sack_today(pidw, d)
    apply_sack_today(pidl, d)
    # h2h update (completed matches only)
    if is_model:
        if key not in state["h2h"]:
            state["h2h"][key] = [0.0, 0.0, 0, None]
        hh = state["h2h"][key]
        hh[0] = ALPHA * hh[0] + (1.0 if w == first else 0.0)
        hh[1] = ALPHA * hh[1] + 1.0
        hh[2] += 1
        hh[3] = w
        if (key, si) not in state["h2hs"]:
            state["h2hs"][(key, si)] = [0.0, 0.0]
        hs = state["h2hs"][(key, si)]
        hs[0] = ALPHA * hs[0] + (1.0 if w == first else 0.0)
        hs[1] = ALPHA * hs[1] + 1.0
    # Elo + games-share history (v1 semantics: completed model rows only)
    if is_model:
        exp = 1.0 / (1.0 + 10.0 ** (-(e_w - e_l) / 400.0))
        state["elo"][w] = e_w + K * (1.0 - exp)
        state["elo"][l] = e_l + K * (0.0 - exp)
        exp_s = 1.0 / (1.0 + 10.0 ** (-(se_w - se_l) / 400.0))
        state["selo"][(w, si)] = se_w + K * (1.0 - exp_s)
        state["selo"][(l, si)] = se_l + K * (0.0 - exp_s)
        state["nsurf"][(w, si)] += 1
        state["nsurf"][(l, si)] += 1
        g_w = sum(m[f"W{k}"] for k in range(1, 6) if pd.notna(m.get(f"W{k}")))
        g_l = sum(m[f"L{k}"] for k in range(1, 6) if pd.notna(m.get(f"L{k}")))
        share_w = g_w / (g_w + g_l) if (g_w + g_l) > 0 else 0.5
        state["hist"][w].append((d, 1, share_w, si))
        state["hist"][l].append((d, 0, 1.0 - share_w, si))

feat = pd.DataFrame(rows_out)
feat = feat[feat["is_model"]].reset_index(drop=True)
feat["is_train"] = feat["Date"] < pd.Timestamp("2022-01-01")
print(f"model rows: {len(feat)} | train {feat['is_train'].sum()} | test {(~feat['is_train']).sum()}")

# serve feature coverage (share of test rows with both players' EWMA present)
sv_cov = (feat["s_svPtsWon_w"].notna() & feat["s_svPtsWon_l"].notna()).mean()
print(f"serve EWMA coverage (both sides): {sv_cov:.1%}")

# ---------------------------------------------------------------------------
# 5. Orientation & feature matrix
# ---------------------------------------------------------------------------
feat["rank_w"] = feat["WRank"].fillna(400.0)
feat["rank_l"] = feat["LRank"].fillna(400.0)
swap = (feat["elo_w"] < feat["elo_l"]) | (
    (feat["elo_w"] == feat["elo_l"]) & (feat["rank_w"] > feat["rank_l"]))


def pick(a, b, s):
    return np.where(s, b, a)


feat["P1"] = pick(feat["Winner"], feat["Loser"], swap)
feat["P2"] = pick(feat["Loser"], feat["Winner"], swap)
for c in ["elo", "selo", "selob", "form", "games", "sform", "sgames", "scareer",
          "rest", "rank"]:
    feat[f"{c}1"] = pick(feat[f"{c}_w"], feat[f"{c}_l"], swap)
    feat[f"{c}2"] = pick(feat[f"{c}_l"], feat[f"{c}_w"], swap)
for c in ["m14", "g14", "s14", "min14", "avg5"]:
    feat[f"{c}_1"] = pick(feat[f"{c}_w"], feat[f"{c}_l"], swap)
    feat[f"{c}_2"] = pick(feat[f"{c}_l"], feat[f"{c}_w"], swap)
for c in ["hurt"]:
    feat[f"{c}1"] = pick(feat[f"{c}_w"], feat[f"{c}_l"], swap)
    feat[f"{c}2"] = pick(feat[f"{c}_l"], feat[f"{c}_w"], swap)
for s in STATS:
    feat[f"s_{s}1"] = pick(feat[f"s_{s}_w"], feat[f"s_{s}_l"], swap)
    feat[f"s_{s}2"] = pick(feat[f"s_{s}_l"], feat[f"s_{s}_w"], swap)
    feat[f"c_{s}1"] = pick(feat[f"c_{s}_w"], feat[f"c_{s}_l"], swap)
    feat[f"c_{s}2"] = pick(feat[f"c_{s}_l"], feat[f"c_{s}_w"], swap)
feat["odds1"] = pick(feat["B365W"], feat["B365L"], swap)
feat["odds2"] = pick(feat["B365L"], feat["B365W"], swap)
feat["y"] = 1.0 - swap.astype(float)
feat["h2h_share1"] = np.where(swap, 1 - feat["h2h_share"], feat["h2h_share"])
feat["h2h_last1"] = np.where(swap, -feat["h2h_last"], feat["h2h_last"])
feat["h2h_ss1"] = np.where(swap, 1 - feat["h2h_ss"], feat["h2h_ss"])

feat["age1"] = [get_age(p, d) for p, d in zip(feat["P1"], feat["Date"])]
feat["age2"] = [get_age(p, d) for p, d in zip(feat["P2"], feat["Date"])]
feat["mkt1"] = (1.0 / feat["odds1"]) / ((1.0 / feat["odds1"]) + (1.0 / feat["odds2"]))

feat["elo_diff"] = feat["elo1"] - feat["elo2"]
feat["selo_diff"] = feat["selo1"] - feat["selo2"]
feat["selob_diff"] = feat["selob1"] - feat["selob2"]
feat["form_diff"] = feat["form1"].fillna(0.5) - feat["form2"].fillna(0.5)
feat["games_diff"] = feat["games1"].fillna(0.5) - feat["games2"].fillna(0.5)
feat["sform_diff"] = feat["sform1"].fillna(feat["form1"]).fillna(0.5) - feat["sform2"].fillna(feat["form2"]).fillna(0.5)
feat["sgames_diff"] = feat["sgames1"].fillna(0.5) - feat["sgames2"].fillna(0.5)
feat["scareer_diff"] = feat["scareer1"].fillna(feat["form1"]).fillna(0.5) - feat["scareer2"].fillna(feat["form2"]).fillna(0.5)
feat["rest_diff"] = feat["rest1"].fillna(14.0) - feat["rest2"].fillna(14.0)
feat["rank_diff"] = np.log(feat["rank2"].clip(1, 500)) - np.log(feat["rank1"].clip(1, 500))
feat["age_diff"] = (feat["age1"] - feat["age2"]).fillna(0.0)
feat["bestof"] = feat["Best of"].astype(float)
feat["surf_hard"] = (feat["surf_idx"] == 0).astype(float)
feat["surf_clay"] = (feat["surf_idx"] == 1).astype(float)
# v2 features
feat["h2h_count_log"] = np.log1p(feat["h2h_n"].fillna(0))
feat["h2h_share_d"] = feat["h2h_share1"].fillna(0.5) - 0.5
feat["h2h_last_d"] = feat["h2h_last1"].fillna(0)
feat["h2h_ss_d"] = feat["h2h_ss1"].fillna(0.5) - 0.5
feat["m14_diff"] = feat["m14_1"] - feat["m14_2"]
feat["g14_diff"] = feat["g14_1"] - feat["g14_2"]
feat["s14_diff"] = feat["s14_1"] - feat["s14_2"]
feat["min14_diff"] = feat["min14_1"] - feat["min14_2"]
feat["avg5_diff"] = feat["avg5_1"].fillna(feat["avg5_2"]).fillna(120) - feat["avg5_2"].fillna(feat["avg5_1"]).fillna(120)
feat["hurt1"] = feat["hurt1"].fillna(0).astype(float)
feat["hurt2"] = feat["hurt2"].fillna(0).astype(float)
for s in STATS:
    feat[f"s_{s}_diff"] = (feat[f"s_{s}1"].fillna(feat[f"c_{s}1"]).fillna(feat[f"s_{s}2"])
                           - feat[f"s_{s}2"].fillna(feat[f"c_{s}2"]).fillna(feat[f"s_{s}1"])).fillna(0.0)
for s in ("svPtsWon", "retPtsWon"):
    feat[f"c_{s}_diff"] = (feat[f"c_{s}1"].fillna(feat[f"s_{s}1"]).fillna(feat[f"c_{s}2"])
                           - feat[f"c_{s}2"].fillna(feat[f"s_{s}2"]).fillna(feat[f"c_{s}1"])).fillna(0.0)

V1_FEATS = [
    "elo_diff", "selo_diff", "form_diff", "games_diff",
    "sform_diff", "sgames_diff", "scareer_diff",
    "rest_diff", "rank_diff", "age_diff",
    "tier", "bestof", "surf_hard", "surf_clay", "mkt1",
]
H2H_FEATS = ["h2h_count_log", "h2h_share_d", "h2h_last_d", "h2h_ss_d", "selob_diff"]
FAT_FEATS = ["m14_diff", "g14_diff", "s14_diff", "min14_diff", "avg5_diff", "hurt1", "hurt2"]
SRV_FEATS = [f"s_{s}_diff" for s in STATS] + ["c_svPtsWon_diff", "c_retPtsWon_diff"]
FEAT_GROUPS = {
    "V1": V1_FEATS,
    "V1+H2H": V1_FEATS + H2H_FEATS,
    "V1+FAT": V1_FEATS + FAT_FEATS,
    "V1+SRV": V1_FEATS + SRV_FEATS,
    "FULL": V1_FEATS + H2H_FEATS + FAT_FEATS + SRV_FEATS,
}
print("feature counts:", {k: len(v) for k, v in FEAT_GROUPS.items()})

tr = feat[feat["is_train"]].sort_values("Date").reset_index(drop=True)
te = feat[~feat["is_train"]].sort_values("Date").reset_index(drop=True)
n = len(tr)
cuts = [int(n * f) for f in (0.2, 0.4, 0.6, 0.8)]
folds = []
prev = 0
for c in cuts + [n]:
    folds.append((prev, c))
    prev = c
dtrain = tr["Date"].values


# ---------------------------------------------------------------------------
# 6. Model pipeline (v1 protocol, parameterized by feature list)
# ---------------------------------------------------------------------------
def fit_logreg(Xtr, ytr, Xva, C):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, solver="lbfgs", max_iter=3000)
    m.fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xva))[:, 1]


def fit_lgb(Xtr, ytr, Xva, yva, params):
    m = lgb.LGBMClassifier(
        objective="binary", n_estimators=800,
        learning_rate=0.05, verbose=-1, random_state=7, **params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="logloss",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict_proba(Xva)[:, 1], m.best_iteration_


def run_pipeline(feats, do_bootstrap=False, n_iter=2000):
    Xtr_full = tr[feats].values
    ytr_full = tr["y"].values
    lgr_scores, lgb_scores = {}, {}
    oof = {"lr": np.full(n, np.nan), "lgb": np.full(n, np.nan)}
    for a, b in folds:
        if b - a < 10 or a < 100:
            continue
        tr_idx, va_idx = np.arange(0, a), np.arange(a, b)
        Xa, ya = Xtr_full[tr_idx], ytr_full[tr_idx]
        Xv, yv = Xtr_full[va_idx], ytr_full[va_idx]
        for C in (0.03, 0.1, 0.3, 1.0, 3.0):
            p = fit_logreg(Xa, ya, Xv, C)
            lgr_scores.setdefault(C, []).append(brier_score_loss(yv, p))
        for nl, mcs in itertools.product((15, 31, 63), (30, 80)):
            p, bi = fit_lgb(Xa, ya, Xv, yv, dict(num_leaves=nl, min_child_samples=mcs))
            lgb_scores.setdefault((nl, mcs), []).append(brier_score_loss(yv, p))
    best_C = min(lgr_scores, key=lambda k: np.mean(lgr_scores[k]))
    best_lgb = min(lgb_scores, key=lambda k: np.mean(lgb_scores[k]))
    best_iters = []
    for a, b in folds:
        if b - a < 10 or a < 100:
            continue
        tr_idx, va_idx = np.arange(0, a), np.arange(a, b)
        oof["lr"][va_idx] = fit_logreg(Xtr_full[tr_idx], ytr_full[tr_idx],
                                       Xtr_full[va_idx], best_C)
        oof["lgb"][va_idx], bi = fit_lgb(Xtr_full[tr_idx], ytr_full[tr_idx],
                                         Xtr_full[va_idx], ytr_full[va_idx],
                                         dict(num_leaves=best_lgb[0], min_child_samples=best_lgb[1]))
        best_iters.append(bi)
    lgb_n_est = int(np.mean(best_iters)) if best_iters else 200
    iso = {}
    for k in ("lr", "lgb"):
        mask = np.isfinite(oof[k])
        iso[k] = IsotonicRegression(out_of_bounds="clip").fit(oof[k][mask], ytr_full[mask])
    mask = np.isfinite(oof["lr"]) & np.isfinite(oof["lgb"])
    oof_ens = np.full(n, np.nan)
    oof_ens[mask] = (iso["lr"].predict(oof["lr"][mask]) + iso["lgb"].predict(oof["lgb"][mask])) / 2.0
    b_ = {kk: brier_score_loss(ytr_full[mask], iso[kk].predict(oof[kk][mask]))
          for kk in ("lr", "lgb")}
    b_["ens"] = brier_score_loss(ytr_full[mask], oof_ens[mask])
    b_["mkt"] = brier_score_loss(ytr_full[mask], tr["mkt1"].values[mask])
    primary = min(b_, key=b_.get)
    if primary == "mkt":
        primary = "ens"
    # final fit on all 2021 -> predict 2022
    sc = StandardScaler().fit(Xtr_full)
    m_lr = LogisticRegression(C=best_C, solver="lbfgs", max_iter=3000).fit(sc.transform(Xtr_full), ytr_full)
    m_lgb = lgb.LGBMClassifier(
        objective="binary", n_estimators=lgb_n_est, learning_rate=0.05,
        num_leaves=best_lgb[0], min_child_samples=best_lgb[1], verbose=-1, random_state=7)
    m_lgb.fit(Xtr_full, ytr_full)
    Xte = te[feats].values
    yte = te["y"].values
    p_lr = iso["lr"].predict(m_lr.predict_proba(sc.transform(Xte))[:, 1])
    p_lgb = iso["lgb"].predict(m_lgb.predict_proba(Xte)[:, 1])
    p_ens = np.clip((p_lr + p_lgb) / 2.0, 1e-6, 1 - 1e-6)
    p_model = {"lr": p_lr, "lgb": p_lgb, "ens": p_ens}[primary]
    acc_model = ((p_model >= 0.5) == (yte == 1)).mean()
    acc_mkt = ((te["mkt1"] >= 0.5) == (yte == 1)).mean()
    brier_model = brier_score_loss(yte, p_model)
    brier_mkt = brier_score_loss(yte, te["mkt1"])
    # gates
    def eval_bets(pmod, eps):
        p1, m1 = pmod, te["mkt1"].values
        y, o1, o2 = yte, te["odds1"].values, te["odds2"].values
        b1 = (p1 > m1 + eps)
        b2 = ((1 - p1) > (1 - m1) + eps)
        bets = b1 | b2
        if not bets.any():
            return dict(n=0)
        win = np.where(b1[bets], y[bets] == 1, y[bets] == 0)
        odds = np.where(b1[bets], o1[bets], o2[bets])
        pl = np.where(win, odds - 1.0, -1.0)
        return dict(n=int(bets.sum()), acc=float(win.mean()), roi=float(pl.sum() / bets.sum()),
                    b1=b1, b2=b2)
    gates = {eps: eval_bets(p_model, eps) for eps in (0.0, 0.02, 0.05)}
    res = dict(best_C=best_C, best_lgb=best_lgb, oof=b_, primary=primary,
               acc=float(acc_model), acc_mkt=float(acc_mkt),
               brier=float(brier_model), brier_mkt=float(brier_mkt),
               gates=gates, p_model=p_model, p_ens=p_ens, p_lr=p_lr, p_lgb=p_lgb)
    return res


results = {}
for gname, gfeats in FEAT_GROUPS.items():
    print(f"\n=== {gname} ({len(gfeats)} feats) ===")
    results[gname] = run_pipeline(gfeats)
    r = results[gname]
    print(f"OOF brier: { {k: round(v,4) for k,v in r['oof'].items()} } | primary {r['primary']}")
    print(f"TEST acc {r['acc']:.4f} (mkt {r['acc_mkt']:.4f}) | Brier {r['brier']:.4f} (mkt {r['brier_mkt']:.4f})")
    for eps in (0.0, 0.02, 0.05):
        g = r["gates"][eps]
        print(f"  gate eps={eps}: n={g.get('n',0)} acc={g.get('acc', float('nan')):.4f} flatROI={g.get('roi', float('nan')):+.4f}")

FULL = results["FULL"]
p_model = FULL["p_model"]
yte = te["y"].values

# ---------------------------------------------------------------------------
# 7. Bootstrap CI (full model) + quarter-Kelly
# ---------------------------------------------------------------------------
te = te.copy()
te["p_model"] = p_model
te["p_ens"] = FULL["p_ens"]
te["p_lr"] = FULL["p_lr"]
te["p_lgb"] = FULL["p_lgb"]
te["week"] = te["Date"].dt.isocalendar().week.astype(int)
te["cluster"] = te["Date"].dt.year.astype(str) + "-W" + te["week"].astype(str)
clusters = te["cluster"].unique()


def boot_ci(fn, n_iter=2000):
    idxs = np.arange(len(te))
    cl_map = {c: idxs[te["cluster"].values == c] for c in clusters}
    vals = []
    for _ in range(n_iter):
        sample_idx = np.concatenate([cl_map[c] for c in RNG.choice(clusters, len(clusters))])
        vals.append(fn(sample_idx))
    return np.percentile(vals, [2.5, 97.5]), vals


def roi_eps_vec(pvec, idx, eps):
    sub = te.iloc[idx]
    p1 = pvec[idx]
    m1 = sub["mkt1"].values
    y = sub["y"].values
    o1 = sub["odds1"].values
    o2 = sub["odds2"].values
    b1 = p1 > m1 + eps
    b2 = (1 - p1) > (1 - m1) + eps
    bets = b1 | b2
    if not bets.any():
        return 0.0
    win = np.where(b1[bets], y[bets] == 1, y[bets] == 0)
    odds = np.where(b1[bets], o1[bets], o2[bets])
    return (np.where(win, odds - 1.0, -1.0)).sum() / bets.sum()


def roi_eps(idx, eps):
    return roi_eps_vec(te["p_model"].values, idx, eps)


ci0, _ = boot_ci(lambda idx: roi_eps(idx, 0.0))
ci005, _ = boot_ci(lambda idx: roi_eps(idx, 0.05))
print(f"\nbootstrap 95% CI (clustered by week) eps=0: [{ci0[0]:+.4f}, {ci0[1]:+.4f}]")
print(f"bootstrap 95% CI (clustered by week) eps=0.05: [{ci005[0]:+.4f}, {ci005[1]:+.4f}]")

# V1+SRV ablation CIs (best single upgrade)
p_srv = results["V1+SRV"]["p_model"]
ci_srv0, _ = boot_ci(lambda idx: roi_eps_vec(p_srv, idx, 0.0))
ci_srv005, _ = boot_ci(lambda idx: roi_eps_vec(p_srv, idx, 0.05))
print(f"V1+SRV CI eps=0: [{ci_srv0[0]:+.4f}, {ci_srv0[1]:+.4f}] | "
      f"eps=0.05: [{ci_srv005[0]:+.4f}, {ci_srv005[1]:+.4f}]")

best_eps = max((e for e, r in FULL["gates"].items() if r.get("n", 0) >= 30),
               key=lambda e: FULL["gates"][e]["roi"])
cib, _ = boot_ci(lambda idx: roi_eps(idx, best_eps))
print(f"best gate eps={best_eps} ROI {FULL['gates'][best_eps]['roi']:+.4f} | CI [{cib[0]:+.4f}, {cib[1]:+.4f}]")

# quarter-Kelly on best gate
b1k = FULL["gates"][best_eps]["b1"]
b2k = FULL["gates"][best_eps]["b2"]
mask_k = b1k | b2k
p_side = np.clip(np.where(b1k, p_model, 1 - p_model), 0.02, 0.98)
odds_k = np.where(b1k, te["odds1"].values, te["odds2"].values)
bb = odds_k - 1.0
f = np.zeros(len(te))
f[mask_k] = np.clip(0.25 * (p_side[mask_k] * bb[mask_k] - (1 - p_side[mask_k])) / np.where(bb[mask_k] > 0, bb[mask_k], 1e-9), 0.0, 0.25)
ret = np.zeros(len(te))
ret[mask_k] = np.where(np.where(b1k[mask_k], yte[mask_k] == 1, yte[mask_k] == 0), bb[mask_k], -1.0)
growth = np.cumprod(1.0 + f * ret)
kelly_bankroll = float(growth[-1]) if mask_k.any() else 1.0
print(f"quarter-Kelly (eps={best_eps}): n={int(mask_k.sum())} bankroll x{kelly_bankroll:.4f} "
      f"({(kelly_bankroll-1)*100:+.1f}%) | mean stake {f[mask_k].mean():.3f} max {f[mask_k].max():.3f}")

# market context
fav_win = ((te["mkt1"] >= 0.5) & (yte == 1)) | ((te["mkt1"] < 0.5) & (yte == 0))
fav_odds = np.where(te["mkt1"] >= 0.5, te["odds1"], te["odds2"])
roi_fav = (np.where(fav_win, fav_odds - 1.0, -1.0)).sum() / len(te)
pl_all = np.where(yte == 1, te["odds1"].values - 1.0, -1.0)
roi_all = pl_all.sum() / len(yte)
print(f"flat all-in P1: ROI {roi_all:+.4f} | market fav: ROI {roi_fav:+.4f} (acc {fav_win.mean():.4f})")

# ---------------------------------------------------------------------------
# 8. Predictions CSV
# ---------------------------------------------------------------------------
out = te[["Date", "Tournament", "Location", "Surface", "Best of", "tier"]].copy()
out["P1"], out["P2"] = te["P1"], te["P2"]
out["rank1"], out["rank2"] = te["rank1"], te["rank2"]
out["odds1"], out["odds2"] = te["odds1"], te["odds2"]
out["mkt_prob1"], out["model_prob1"] = te["mkt1"], te["p_model"]
out["p_lr"], out["p_lgb"], out["p_ens"] = te["p_lr"], te["p_lgb"], te["p_ens"]
out["target_p1_wins"] = yte.astype(int)
out["model_side"] = np.where(te["p_model"] >= 0.5, "P1", "P2")
edge = te["p_model"] - te["mkt1"]
out["edge_p1"] = edge
out["bet_eps0"] = ((edge > 0) | ((1 - te["p_model"]) > (1 - te["mkt1"]))).astype(int)
out["bet_eps005"] = ((edge > 0.05) | ((1 - te["p_model"]) > (1 - te["mkt1"]) + 0.05)).astype(int)
out["bet_bestgate"] = (b1k | b2k).astype(int)
out["kelly_stake_frac"] = np.where(mask_k, f, 0.0).round(4)
out.to_csv(f"{WORK}/tennis_v2_predictions.csv", index=False)
print("wrote tennis_v2_predictions.csv rows:", len(out))

# ---------------------------------------------------------------------------
# 9. Report
# ---------------------------------------------------------------------------
sv_cov_tr = (tr["s_svPtsWon_w"].notna() & tr["s_svPtsWon_l"].notna()).mean()
sv_cov_te = (te["s_svPtsWon_w"].notna() & te["s_svPtsWon_l"].notna()).mean()
min_cov = float(feat["avg5_w"].notna().mean())

L = []
A = L.append
A("# ATP Tennis v2 Backtest Report (accuracy push)")
A("")
A(f"**Generated:** {pd.Timestamp.now():%Y-%m-%d %H:%M}")
A("")
A("## Data")
A("- Betting rows: tennis-data.co.uk `2021.xlsx`/`2022.xlsx` (ATP, B365 closing decimal"
  " odds) — identical test window to v1 (2022, n=2538).")
A("- **Serve/return + duration:** JeffSackmann `tennis_atp` `atp_matches_2019-2022.csv`"
  " via GitHub fork `Kadantte/tennis_atp` (upstream repo deleted; fork kept current to"
  " 2026). Columns used: svpt/ace/df/1stIn/1stWon/2ndWon/bpSaved/bpFaced (+opponent"
  " side) and `minutes`. Joined exactly on Sackmann player_id; tennis-data name →"
  " player_id via the v1 fuzzy matcher (particle handling, initials, latest-DOB"
  " tie-break).")
A(f"- Player-DOB coverage (both sides): {cov/model_mask.sum():.1%} of model rows.")
A(f"- Serve-stat coverage (both players have EWMA history): train {sv_cov_tr:.1%},"
  f" test {sv_cov_te:.1%}. Minutes present for {min_cov:.1%} of model rows.")
A("- Fatigue inputs use ALL tennis-data rows (incl. Retired/Walkover) so retirements"
  " and partial matches still load the body; Elo/ratings update on completed rows only"
  " (v1 semantics preserved).")
A("")
A("## Upgrades over v1")
A("- **(a) H2H** (same-file prior meetings, completed matches only): count (log1p),"
  " recency-weighted win share (EWMA α=0.9 per meeting), last-meeting winner (±1),"
  " surface-specific share.")
A("- **(b) Surface Elo**: v1's raw per-surface rating (K=32) shrunk toward the global"
  " rating: `blend = (n_surf·selo + 15·elo)/(n_surf+15)` — fixes v1's noisy low-sample"
  " surface ratings.")
A("- **(c) Serve/return** (EWMA α=0.9 over prior matches, point-weighted, ≥30 points"
  " required): ace%, df%, 1st-serve win%, 2nd-serve win%, serve-points-won%,"
  " return-points-won%, break-points-saved%, break-point-conversion%; career-pooled"
  " versions of the two points-won rates.")
A("- **(d) Fatigue**: matches/games/sets/minutes in last 14 days (per player), mean"
  " minutes of last 5 matches, retirement-as-loser within 30 days (injury flag).")
A("- **(e) Duration**: match `minutes` (Sackmann) — not present in tennis-data xlsx.")
A("")
A("## Protocol (unchanged from v1)")
A("- Single forward pass over the 2021+2022 timeline; every feature uses strictly"
  " earlier matches only (no lookahead; same-day sack stats applied after the row).")
A("- Orientation P1 = higher pre-match global Elo (rank tie-break), y = P1 wins,"
  " market prob = no-vig B365 closing.")
A("- LogReg (L2, C grid) + LightGBM (num_leaves×min_child_samples grid), expanding-"
  "window time-ordered 5-fold CV on 2021 (Brier), isotonic on OOF, soft ensemble;"
  " primary model by OOF Brier (market excluded as a choice).")
A("- Test 2022: accuracy, Brier vs market, flat-$1 ROI at closing odds, edge gates"
  " ε∈{0, 0.02, 0.05}, quarter-Kelly on best gate (n≥30), bootstrap CI clustered by"
  " year+ISO week (2,000 resamples).")
A("")
A("## Results: ablation by feature group (test 2022, n=2538)")
A("| group | OOF Brier (2021) | acc | Brier | ROI ε=0 | ROI ε=0.02 | ROI ε=0.05 |")
A("|---|---|---|---|---|---|---|")
A("| v1 baseline (prior run) | — | 67.8% | 0.2024 | −6.18% | — | −1.0% |")
for gname in ("V1", "V1+H2H", "V1+FAT", "V1+SRV", "FULL"):
    r = results[gname]
    g = {eps: r["gates"][eps] for eps in (0.0, 0.02, 0.05)}
    A(f"| {gname} | {r['oof']['ens']:.4f} | {r['acc']*100:.1f}% | {r['brier']:.4f} | "
      f"{g[0.0].get('roi', float('nan')):+.2%} | {g[0.02].get('roi', float('nan')):+.2%} | "
      f"{g[0.05].get('roi', float('nan')):+.2%} |")
A("")
r = FULL
A(f"**FULL model**: LogReg C={r['best_C']}, LGBM {r['best_lgb']}, primary={r['primary']};"
  f" OOF Brier { {k: round(v, 4) for k, v in r['oof'].items()} }; test acc **{r['acc']*100:.1f}%**"
  f" (market {r['acc_mkt']*100:.1f}%), Brier **{r['brier']:.4f}** (market {r['brier_mkt']:.4f}).")
A("")
A("### FULL-model gates & staking (2022)")
A("| ε | n bets | bet acc | flat-$1 ROI |")
A("|---|---|---|---|")
for eps in (0.0, 0.02, 0.05):
    g = r["gates"][eps]
    A(f"| {eps:.2f} | {g.get('n',0)} | {g.get('acc', float('nan')):.4f} | {g.get('roi', float('nan')):+.4f} |")
A("")
A(f"- Best gate (by ROI, n≥30): ε={best_eps}, n={r['gates'][best_eps]['n']},"
  f" ROI {r['gates'][best_eps]['roi']:+.4f}.")
A(f"- Quarter-Kelly on best gate: bankroll ×{kelly_bankroll:.4f}"
  f" ({(kelly_bankroll-1)*100:+.1f}%).")
A(f"- Bootstrap 95% CI (week-clustered): ε=0 ROI [{ci0[0]:+.4f}, {ci0[1]:+.4f}];"
  f" ε=0.05 ROI [{ci005[0]:+.4f}, {ci005[1]:+.4f}]; best-gate (ε={best_eps})"
  f" [{cib[0]:+.4f}, {cib[1]:+.4f}].")
A(f"- Context: flat all-in P1 ROI {roi_all:+.4f}; market-favorite flat ROI {roi_fav:+.4f}"
  f" (acc {fav_win.mean():.4f}).")
A("")
A("## Verdict")
v1_run = results["V1"]
sr = results["V1+SRV"]
d_acc = (r["acc"] - v1_run["acc"]) * 100
d_brier = r["brier"] - v1_run["brier"]
verdict = (
    f"v2 does NOT beat v1. FULL vs the in-run V1 ablation (same data, same protocol): "
    f"accuracy {r['acc']*100:.2f}% vs {v1_run['acc']*100:.2f}% ({d_acc:+.2f}pp), "
    f"Brier {r['brier']:.4f} vs {v1_run['brier']:.4f} ({d_brier:+.4f}). Against v1's "
    f"published numbers (67.8% acc, Brier 0.2024) the deltas are the same order of "
    f"magnitude — noise, not improvement. Gate ROIs are negative at every ε for FULL "
    f"(ε=0: {r['gates'][0.0]['roi']:+.3f}, ε=0.02: {r['gates'][0.02]['roi']:+.3f}, "
    f"ε=0.05: {r['gates'][0.05]['roi']:+.3f}); the ε=0 week-clustered CI "
    f"[{ci0[0]:+.4f}, {ci0[1]:+.4f}] is entirely below zero and quarter-Kelly compounds "
    f"to {(kelly_bankroll-1)*100:+.1f}%. Ablation ranking: serve/return stats are the "
    f"strongest single upgrade (V1+SRV acc {sr['acc']*100:.2f}%, Brier {sr['brier']:.4f}, "
    f"ε=0.05 gate {sr['gates'][0.05]['roi']:+.3f}, CI [{ci_srv005[0]:+.4f}, {ci_srv005[1]:+.4f}] "
    f"— still negative/straddling zero), while h2h and fatigue add nothing on top. "
    f"The 2022 ATP closing line prices all of it; no profitable betting rule emerges.")
A(verdict)
A("")
A("Deliverables: `tennis_v2_predictions.csv`, this report, `run_tennis_v2.py`,"
  " `raw/atp_matches_*.csv` (Sackmann fork), `probe_coverage.py`.")

with open(f"{WORK}/TENNIS_V2_REPORT.md", "w") as fh:
    fh.write("\n".join(L) + "\n")
print("wrote TENNIS_V2_REPORT.md")

# ---------------------------------------------------------------------------
# 10. Final key numbers
# ---------------------------------------------------------------------------
final = {
    "n_test_matches": int(len(te)),
    "accuracy": round(float(r["acc"]), 6),
    "brier": round(float(r["brier"]), 6),
    "v1_accuracy": 0.678,
    "v1_brier": 0.2024,
    "flat_roi_eps0": round(FULL["gates"][0.0]["roi"], 6),
    "best_gate_eps": float(best_eps),
    "best_gate_roi": round(FULL["gates"][best_eps]["roi"], 6),
    "ci_eps0": [round(float(ci0[0]), 6), round(float(ci0[1]), 6)],
    "ci_bestgate": [round(float(cib[0]), 6), round(float(cib[1]), 6)],
    "oof_brier_full": round(FULL["oof"]["ens"], 6),
    "report_path": f"{WORK}/TENNIS_V2_REPORT.md",
}
print("\nFINAL_JSON", json.dumps(final))
