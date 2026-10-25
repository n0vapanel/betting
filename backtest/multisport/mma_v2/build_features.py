"""
Build no-lookahead rolling features for UFC walk-forward backtest.

Sources:
- mdabbert/ultimate-ufc-dataset (ufc-master.csv): fight log 2010-2026, BestFightOdds
  closing American odds (R_odds/B_odds), winner (Red/Blue/Draw/NC), age, reach, height,
  weight, ranks, weight class.
- rajeevw/ufcdata (raw_total_fight_data.csv): per-fight ufcstats numbers 1994-2021,
  used to build strike-rate features (SLpM, strike acc, TD acc, TD defense, sub att...).

All rolling features are computed strictly from fights BEFORE the bout (no lookahead).
Output: fights_features.parquet (one row per fight).
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(base, "fights_features.parquet")

# ---------------- load mdabbert ----------------
md = pd.read_csv(os.path.join(base, "raw_mdabbert", "ufc-master.csv"))
md["fid"] = md.index
md["date"] = pd.to_datetime(md["date"]).dt.normalize()
md["date_n"] = md["date"]
md["R_fighter"] = md["R_fighter"].astype(str).str.strip()
md["B_fighter"] = md["B_fighter"].astype(str).str.strip()
print("md rows:", len(md))

# ---------------- parse rajeevw per-fight data ----------------
raw = pd.read_csv(os.path.join(base, "raw", "raw_total_fight_data.csv"), sep=";")
raw.columns = [c.strip() for c in raw.columns]
raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()

def landed_att(s):
    parts = s.astype(str).str.split(" of ", expand=True)
    return pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")

def pct(s):
    return pd.to_numeric(s.astype(str).str.replace("%", "").replace("---", np.nan), errors="coerce")

def mmss(s):
    s = s.astype(str)
    out = np.full(len(s), np.nan)
    mask = s.str.contains(":")
    parts = s[mask].str.split(":", expand=True)
    out[mask] = pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")
    return out

frames = []
for corner in ["R", "B"]:
    opp = "B" if corner == "R" else "R"
    l_sig, a_sig = landed_att(raw[f"{corner}_SIG_STR."])
    l_td, a_td = landed_att(raw[f"{corner}_TD"])
    sub = pd.DataFrame({
        "fight_id": raw.index,
        "date": raw["date"],
        "fighter": raw[f"{corner}_fighter"].astype(str).str.strip(),
        "opp": raw[f"{opp}_fighter"].astype(str).str.strip(),
        "kd": pd.to_numeric(raw[f"{corner}_KD"], errors="coerce"),
        "sig_landed": l_sig,
        "sig_att": a_sig,
        "td_landed": l_td,
        "td_att": a_td,
        "sub_att": pd.to_numeric(raw[f"{corner}_SUB_ATT"], errors="coerce"),
        "ctrl_sec": mmss(raw[f"{corner}_CTRL"]),
        "win": (raw["Winner"].astype(str).str.strip() == raw[f"{corner}_fighter"].astype(str).str.strip()).astype(float),
        "last_round": pd.to_numeric(raw["last_round"], errors="coerce"),
        "last_round_time": mmss(raw["last_round_time"]),
    })
    frames.append(sub)
long = pd.concat(frames, ignore_index=True)
long = long[(long["fighter"] != "") & (long["fighter"] != "nan") & long["fighter"].notna()]
long["minutes"] = (long["last_round"] - 1) * 5 + long["last_round_time"] / 60
print("rajeevw long rows:", len(long))

# opponent stats (for TD defense)
opp_stats = long[["fight_id", "fighter", "td_landed", "td_att"]].rename(
    columns={"fighter": "opp", "td_landed": "opp_td_landed", "td_att": "opp_td_att"})
long = long.merge(opp_stats, on=["fight_id", "opp"], how="left")

# ---------------- rolling prior stats per fighter ----------------
long = long.sort_values(["fighter", "date", "fight_id"]).reset_index(drop=True)
g = long.groupby("fighter")
prev = lambda col: g[col].cumsum() - long[col]
long["prev_n"] = g.cumcount()
long["prev_sig_landed"] = prev("sig_landed")
long["prev_sig_att"] = prev("sig_att")
long["prev_minutes"] = prev("minutes")
long["prev_td_landed"] = prev("td_landed")
long["prev_td_att"] = prev("td_att")
long["prev_opp_td_landed"] = prev("opp_td_landed")
long["prev_opp_td_att"] = prev("opp_td_att")
long["prev_ctrl_sec"] = prev("ctrl_sec")
long["prev_sub_sum"] = prev("sub_att")
long["prev_kd_sum"] = prev("kd")

def safe_div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, np.nan), np.nan)

long["feat_slpm"] = safe_div(long["prev_sig_landed"], long["prev_minutes"])
long["feat_str_acc"] = safe_div(long["prev_sig_landed"], long["prev_sig_att"])
long["feat_td_acc"] = safe_div(long["prev_td_landed"], long["prev_td_att"])
long["feat_td_def"] = 1 - safe_div(long["prev_opp_td_landed"], long["prev_opp_td_att"])
long["feat_sub_per_f"] = safe_div(long["prev_sub_sum"], long["prev_n"])
long["feat_kd_per_f"] = safe_div(long["prev_kd_sum"], long["prev_n"])
long["feat_ctrl_share"] = safe_div(long["prev_ctrl_sec"], long["prev_minutes"] * 60)
print("rajeevw feat coverage (>=1 prior fight):", long["feat_slpm"].notna().mean().round(3))

rj = long[["date", "fighter", "feat_slpm", "feat_str_acc", "feat_td_acc", "feat_td_def",
           "feat_sub_per_f", "feat_kd_per_f", "feat_ctrl_share"]].copy()
rj = rj.drop_duplicates(subset=["date", "fighter"], keep="first")

# ---------------- mdabbert fight-log rolling: record/streak/Elo/days ----------------
m = md[~md["Winner"].isin(["Draw", "No Contest"])].copy()
m = m.sort_values(["date", "fid"]).reset_index(drop=True)
m["y"] = (m["Winner"] == "Red").astype(int)

lm = []
for corner in ["R", "B"]:
    side = 1 if corner == "R" else 0
    lm.append(pd.DataFrame({
        "fid": m["fid"],
        "corner": corner,
        "date": m["date"],
        "fighter": m[f"{corner}_fighter"],
        "opp": m[f"{'B' if corner == 'R' else 'R'}_fighter"],
        "won": (m["y"] == side).astype(int),
    }))
lm = pd.concat(lm, ignore_index=True)
lm = lm.sort_values(["date", "fid"]).reset_index(drop=True)
print("md log long rows:", len(lm))

# Elo pass in strict time order (K=60, start 1500): update each FIGHT once
elo = {}
K = 60
eloR = np.full(len(m), np.nan)
eloB = np.full(len(m), np.nan)
for i in range(len(m)):
    rf, bf, y = m.at[i, "R_fighter"], m.at[i, "B_fighter"], m.at[i, "y"]
    er = elo.get(rf, 1500.0)
    eb = elo.get(bf, 1500.0)
    eloR[i] = er
    eloB[i] = eb
    exp_r = 1 / (1 + 10 ** ((eb - er) / 400))
    elo[rf] = er + K * (y - exp_r)
    elo[bf] = eb + K * ((1 - y) - (1 - exp_r))
elo_lookup = pd.DataFrame({"fid": m["fid"], "eloR": eloR, "eloB": eloB})
lm = lm.merge(elo_lookup, on="fid", how="left")
lm["elo_before"] = np.where(lm["corner"] == "R", lm["eloR"], lm["eloB"])
lm = lm.drop(columns=["eloR", "eloB"])

# per-fighter priors
lm = lm.sort_values(["fighter", "date", "fid"]).reset_index(drop=True)
g = lm.groupby("fighter")
lm["n_prior"] = g.cumcount()
lm["w_prior"] = g["won"].cumsum() - lm["won"]
lm["l_prior"] = lm["n_prior"] - lm["w_prior"]
wins_run = g["won"].cumsum()
loss_flag = (lm["won"] == 0)
last_loss_winrun = wins_run.where(loss_flag).ffill().fillna(0)
streak_incl = wins_run - last_loss_winrun
lm["streak_pre"] = streak_incl - lm["won"]               # consecutive wins BEFORE this fight
lm["longest_run_so_far"] = lm.groupby("fighter")["streak_pre"].transform(
    lambda s: s.shift(1).cummax()).fillna(0)
lm["prev_date"] = g["date"].shift(1)
lm["days_since"] = (lm["date"] - lm["prev_date"]).dt.days

log_feats = lm[["fid", "fighter", "elo_before", "n_prior", "w_prior", "l_prior",
                "streak_pre", "longest_run_so_far", "days_since"]]

# ---------------- merge onto mdabbert rows ----------------
merged = md.merge(log_feats, how="left", left_on=["fid", "R_fighter"], right_on=["fid", "fighter"])
merged = merged.drop(columns=["fighter"])
merged = merged.rename(columns={c: f"{c}_R" for c in
    ["elo_before", "n_prior", "w_prior", "l_prior", "streak_pre", "longest_run_so_far", "days_since"]})
merged = merged.merge(log_feats, how="left", left_on=["fid", "B_fighter"], right_on=["fid", "fighter"])
merged = merged.drop(columns=["fighter"])
merged = merged.rename(columns={c: f"{c}_B" for c in
    ["elo_before", "n_prior", "w_prior", "l_prior", "streak_pre", "longest_run_so_far", "days_since"]})
print("md merged rows:", len(merged))

for corner in ["R", "B"]:
    rjc = rj.rename(columns={c: f"{c}_{corner}" for c in rj.columns if c not in ("date", "fighter")})
    merged = merged.merge(rjc, how="left", left_on=["date_n", f"{corner}_fighter"],
                          right_on=["date", "fighter"])
    merged = merged.drop(columns=["date", "fighter"], errors="ignore")

sub2021 = merged[(merged["date_n"] >= pd.Timestamp("2020-01-01")) & (merged["date_n"] <= pd.Timestamp("2021-03-20"))]
print("2020-2021 rows:", len(sub2021), "| feat_slpm_R present:",
      sub2021["feat_slpm_R"].notna().mean().round(3))

merged.to_parquet(OUT)
print("saved", OUT, merged.shape)
