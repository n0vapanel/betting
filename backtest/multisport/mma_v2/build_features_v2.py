"""
Build no-lookahead rolling features for UFC walk-forward backtest — V2.

Same sources as v1 (mdabbert ufc-master.csv + rajeevw raw_total_fight_data.csv),
plus rajeevw raw_fighter_details.csv (DOB/Height/Reach/Stance) for the new
physical features.

New vs v1 (all strictly no-lookahead, computed from data BEFORE the bout):
- age at fight from DOB (fallback: mdabbert R_age), age at debut, career days
- height/reach gap-filled from raw_fighter_details.csv
- fight location country (normalized, top-country dummies)
- stance flags (same stance, southpaw advantage)
- opponent quality: mean Elo and mean win% of prior opponents faced
- opponent style (rajeevw): mean prior opponents' SLpM / str_acc / TD_acc / sub rate
- recent form: win fraction last 3 fights; SLpM last 3 fights (recent-weighted)
- empty_arena flag (COVID-era, known pre-fight)

Note: mdabbert career-average stat columns (avg_SIG_STR_*, wins, losses...) are
snapshot career totals -> lookahead for historical fights; NOT used.
No missed-weight / camp fields exist in either source.

Output: fights_features_v2.parquet
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(base, "fights_features_v2.parquet")

# ---------------- load mdabbert ----------------
md = pd.read_csv(os.path.join(base, "raw_mdabbert", "ufc-master.csv"))
md["fid"] = md.index
md["date"] = pd.to_datetime(md["date"]).dt.normalize()
md["date_n"] = md["date"]
md["R_fighter"] = md["R_fighter"].astype(str).str.strip()
md["B_fighter"] = md["B_fighter"].astype(str).str.strip()
md["country_n"] = md["country"].astype(str).str.strip().str.lower()
md["empty_arena"] = md["empty_arena"].fillna(0).astype(int)
print("md rows:", len(md))

# ---------------- load fighter details (static physicals) ----------------
fd = pd.read_csv(os.path.join(base, "raw", "raw_fighter_details.csv"))
fd["fighter_name"] = fd["fighter_name"].astype(str).str.strip()

def parse_height(s):
    s = s.astype(str)
    out = np.full(len(s), np.nan)
    m = s.str.match(r"^(\d+)'\s*(\d+)\"?$")
    feet = pd.to_numeric(s.str.extract(r"^(\d+)'")[0], errors="coerce")
    inch = pd.to_numeric(s.str.extract(r"'?\s*(\d+)\"?$")[0], errors="coerce")
    ok = s.str.contains("'") & inch.notna() & feet.notna()
    out[ok] = feet[ok] * 30.48 + inch[ok] * 2.54
    return out

def parse_reach(s):
    s = s.astype(str)
    out = np.full(len(s), np.nan)
    m = s.str.match(r'^\d+(\.\d+)?$')
    out[m] = pd.to_numeric(s[m], errors="coerce") * 2.54
    return out

fd["fd_height_cm"] = parse_height(fd["Height"])
fd["fd_reach_cm"] = parse_reach(fd["Reach"])
fd["dob"] = pd.to_datetime(fd["DOB"], format="%b %d, %Y", errors="coerce")
fd["stance_fd"] = fd["Stance"].astype(str).str.strip()
fd = fd[["fighter_name", "fd_height_cm", "fd_reach_cm", "dob", "stance_fd"]].copy()

# ---------------- parse rajeevw per-fight data (as v1) ----------------
raw = pd.read_csv(os.path.join(base, "raw", "raw_total_fight_data.csv"), sep=";")
raw.columns = [c.strip() for c in raw.columns]
raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()

def landed_att(s):
    parts = s.astype(str).str.split(" of ", expand=True)
    return pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")

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

opp_stats = long[["fight_id", "fighter", "td_landed", "td_att"]].rename(
    columns={"fighter": "opp", "td_landed": "opp_td_landed", "td_att": "opp_td_att"})
long = long.merge(opp_stats, on=["fight_id", "opp"], how="left")

# rolling prior stats per fighter (strictly before this fight)
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

# recent-3 weighted SLpM (recency-weighted form)
def recent3(df, col):
    s = df.groupby("fighter")[col].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    return s
long["feat_slpm_r3"] = recent3(long, "sig_landed") / recent3(long, "minutes").where(
    recent3(long, "minutes") > 0, np.nan)
long["feat_str_acc_r3"] = recent3(long, "sig_landed") / recent3(long, "sig_att").where(
    recent3(long, "sig_att") > 0, np.nan)

# opponent style at each fight (opponent's rolling stats), then per-fighter mean over priors
opp_roll = long[["fight_id", "fighter", "feat_slpm", "feat_str_acc", "feat_td_acc", "feat_sub_per_f"]].rename(
    columns={"fighter": "opp", "feat_slpm": "o_slpm", "feat_str_acc": "o_str_acc",
             "feat_td_acc": "o_td_acc", "feat_sub_per_f": "o_sub"})
long = long.merge(opp_roll, on=["fight_id", "opp"], how="left")
g = long.groupby("fighter")   # refresh groupby after merge
for c in ["o_slpm", "o_str_acc", "o_td_acc", "o_sub"]:
    long[f"prev_{c}_sum"] = g[c].cumsum() - long[c]
    long[f"prev_{c}_n"] = g[c].transform(lambda x: x.shift(1).notna().cumsum()).fillna(0)
    long[f"feat_opp_{c[2:]}"] = safe_div(long[f"prev_{c}_sum"], long[f"prev_{c}_n"])

rj = long[["date", "fighter", "feat_slpm", "feat_str_acc", "feat_td_acc", "feat_td_def",
           "feat_sub_per_f", "feat_kd_per_f", "feat_ctrl_share",
           "feat_slpm_r3", "feat_str_acc_r3", "feat_opp_slpm", "feat_opp_str_acc",
           "feat_opp_td_acc", "feat_opp_sub"]].copy()
rj = rj.drop_duplicates(subset=["date", "fighter"], keep="first")
print("rajeevw feat coverage (>=1 prior fight):", rj["feat_slpm"].notna().mean().round(3))

# ---------------- mdabbert fight-log rolling: record/streak/Elo/days + new ----------------
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

# Elo pass in strict time order (K=60, start 1500)
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

lm = lm.sort_values(["fighter", "date", "fid"]).reset_index(drop=True)
g = lm.groupby("fighter")
lm["n_prior"] = g.cumcount()
lm["w_prior"] = g["won"].cumsum() - lm["won"]
lm["l_prior"] = lm["n_prior"] - lm["w_prior"]
wins_run = g["won"].cumsum()
loss_flag = (lm["won"] == 0)
last_loss_winrun = wins_run.where(loss_flag).ffill().fillna(0)
streak_incl = wins_run - last_loss_winrun
lm["streak_pre"] = streak_incl - lm["won"]
lm["longest_run_so_far"] = lm.groupby("fighter")["streak_pre"].transform(
    lambda s: s.shift(1).cummax()).fillna(0)
lm["prev_date"] = g["date"].shift(1)
lm["days_since"] = (lm["date"] - lm["prev_date"]).dt.days
lm["debut_date"] = g["date"].transform("min")
lm["career_days"] = (lm["date"] - lm["debut_date"]).dt.days
lm["recent3_winpct"] = g["won"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

# opponent quality: mean pre-fight Elo / win% of prior opponents
oppq = lm[["fid", "fighter", "elo_before", "won"]].rename(
    columns={"fighter": "opp", "elo_before": "opp_elo", "won": "opp_won"})
lm = lm.merge(oppq, on=["fid", "opp"], how="left")
g2 = lm.groupby("fighter")
lm["prev_opp_elo_sum"] = g2["opp_elo"].cumsum() - lm["opp_elo"]
lm["prev_opp_elo_n"] = g2["opp_elo"].transform(lambda x: x.shift(1).notna().cumsum()).fillna(0)
lm["opp_elo_avg"] = safe_div(lm["prev_opp_elo_sum"], lm["prev_opp_elo_n"])
lm["prev_opp_win_sum"] = g2["opp_won"].cumsum() - lm["opp_won"]
lm["opp_winpct_avg"] = safe_div(lm["prev_opp_win_sum"], lm["prev_opp_elo_n"])

# age from DOB (merge once, per-fighter static)
lm = lm.merge(fd[["fighter_name", "dob"]], how="left",
              left_on="fighter", right_on="fighter_name")
lm = lm.drop(columns=["fighter_name"])
lm["age_dob"] = (lm["date"] - lm["dob"]).dt.days / 365.25
lm["age_debut"] = (lm["debut_date"] - lm["dob"]).dt.days / 365.25
lm = lm.drop(columns=["dob"])

log_feats = lm[["fid", "fighter", "elo_before", "n_prior", "w_prior", "l_prior",
                "streak_pre", "longest_run_so_far", "days_since", "career_days",
                "recent3_winpct", "opp_elo_avg", "opp_winpct_avg",
                "age_dob", "age_debut"]]

# ---------------- merge onto mdabbert rows ----------------
merged = md.merge(log_feats, how="left", left_on=["fid", "R_fighter"], right_on=["fid", "fighter"])
merged = merged.drop(columns=["fighter"])
merged = merged.rename(columns={c: f"{c}_R" for c in
    ["elo_before", "n_prior", "w_prior", "l_prior", "streak_pre", "longest_run_so_far",
     "days_since", "career_days", "recent3_winpct", "opp_elo_avg", "opp_winpct_avg",
     "age_dob", "age_debut"]})
merged = merged.merge(log_feats, how="left", left_on=["fid", "B_fighter"], right_on=["fid", "fighter"])
merged = merged.drop(columns=["fighter"])
merged = merged.rename(columns={c: f"{c}_B" for c in
    ["elo_before", "n_prior", "w_prior", "l_prior", "streak_pre", "longest_run_so_far",
     "days_since", "career_days", "recent3_winpct", "opp_elo_avg", "opp_winpct_avg",
     "age_dob", "age_debut"]})
print("md merged rows:", len(merged))

# ---------------- merge fighter-details height/reach/stance (static) ----------------
fdh = fd[["fighter_name", "fd_height_cm", "fd_reach_cm", "stance_fd"]].drop_duplicates("fighter_name")
for corner in ["R", "B"]:
    merged = merged.merge(fdh, how="left", left_on=f"{corner}_fighter", right_on="fighter_name")
    merged = merged.drop(columns=["fighter_name"])
    merged = merged.rename(columns={"fd_height_cm": f"fd_height_{corner}",
                                    "fd_reach_cm": f"fd_reach_{corner}",
                                    "stance_fd": f"stance_{corner}"})

# ---------------- merge rajeevw rolling strike stats ----------------
for corner in ["R", "B"]:
    rjc = rj.rename(columns={c: f"{c}_{corner}" for c in rj.columns if c not in ("date", "fighter")})
    merged = merged.merge(rjc, how="left", left_on=["date_n", f"{corner}_fighter"],
                          right_on=["date", "fighter"])
    merged = merged.drop(columns=["date", "fighter"], errors="ignore")

sub2021 = merged[(merged["date_n"] >= pd.Timestamp("2020-01-01")) & (merged["date_n"] <= pd.Timestamp("2021-03-20"))]
print("2020-2021 rows:", len(sub2021), "| feat_slpm_R present:", sub2021["feat_slpm_R"].notna().mean().round(3))
print("DOB coverage R (2020-2021):", sub2021["age_dob_R"].notna().mean().round(3))
print("fd height coverage R (2020-2021):", sub2021["fd_height_R"].notna().mean().round(3))

merged.to_parquet(OUT)
print("saved", OUT, merged.shape)
