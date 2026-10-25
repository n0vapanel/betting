"""
Shared feature-building code for the MMA (UFC) model.

Extracted from build_features.py (rolling no-lookahead features) and
run_backtest.py (model feature columns) so that both the backtest and the
live-card picker (ufc_picker.py) run the SAME feature builders.

Key functions:
  load_mdabbert(base)              -> master fight log (mdabbert/ultimate-ufc-dataset)
  parse_rajeevw(base)              -> long per-fighter fight stats (rajeevw/ufcdata)
  build_rolling_features(md, long) -> merged df, one row per fight, all rolling feats
  add_model_features(df, ...)      -> diffs + categoricals used by the models
  build_card_features(md, long, card_rows) -> history (< first card date) + card rows,
                                               features computed with NO future data
"""
import os

import numpy as np
import pandas as pd

# ---------------- sources ----------------


def load_mdabbert(base):
    """Load ufc-master.csv and normalize (fid, date, date_n, stripped names)."""
    md = pd.read_csv(os.path.join(base, "raw_mdabbert", "ufc-master.csv"))
    md["fid"] = md.index
    md["date"] = pd.to_datetime(md["date"]).dt.normalize()
    md["date_n"] = md["date"]
    md["R_fighter"] = md["R_fighter"].astype(str).str.strip()
    md["B_fighter"] = md["B_fighter"].astype(str).str.strip()
    return md


def parse_rajeevw(base):
    """Parse rajeevw/ufcdata raw_total_fight_data.csv into long per-fighter stats."""
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

    # opponent stats (for TD defense)
    opp_stats = long[["fight_id", "fighter", "td_landed", "td_att"]].rename(
        columns={"fighter": "opp", "td_landed": "opp_td_landed", "td_att": "opp_td_att"})
    long = long.merge(opp_stats, on=["fight_id", "opp"], how="left")

    # rolling prior stats per fighter (strictly before each fight)
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
    return long


# ---------------- rolling features on the fight log ----------------


def build_rolling_features(md, long):
    """
    Compute the full set of rolling features (Elo, record, streak, days since,
    strike-rate stats) on the master log. `md` may contain appended card rows
    (Winner = NaN) — those get features from strictly prior fights only.
    Returns the merged feature df (same schema as fights_features.parquet).
    """
    # strike features per fighter/date (from rajeevw)
    rj = long[["date", "fighter", "feat_slpm", "feat_str_acc", "feat_td_acc", "feat_td_def",
               "feat_sub_per_f", "feat_kd_per_f", "feat_ctrl_share"]].copy()
    rj = rj.drop_duplicates(subset=["date", "fighter"], keep="first")

    # mdabbert fight-log rolling: record/streak/Elo/days
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
            "has_winner": m["Winner"].notna().astype(int),  # 0 for appended card rows
        }))
    lm = pd.concat(lm, ignore_index=True)
    lm = lm.sort_values(["date", "fid"]).reset_index(drop=True)

    # Elo pass in strict time order (K=60, start 1500). Card rows (no outcome yet)
    # read their Elo but do NOT update the ratings — no fake results propagate.
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
        if not pd.notna(m.at[i, "Winner"]):
            continue  # upcoming fight, no outcome: do not update ratings
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
    lm["streak_pre"] = streak_incl - lm["won"]
    # Card rows (no outcome yet) were counted as losses above; replace their
    # streak with the consecutive-win run through their last PRIOR fight.
    card_rows = lm["has_winner"] == 0
    if card_rows.any():
        prev_winrun = g["won"].cumsum().shift(1).fillna(0)
        prev_llw = last_loss_winrun.shift(1).fillna(0)
        lm.loc[card_rows, "streak_pre"] = (prev_winrun - prev_llw)[card_rows]
    lm["longest_run_so_far"] = lm.groupby("fighter")["streak_pre"].transform(
        lambda s: s.shift(1).cummax()).fillna(0)
    lm["prev_date"] = g["date"].shift(1)
    lm["days_since"] = (lm["date"] - lm["prev_date"]).dt.days

    log_feats = lm[["fid", "fighter", "elo_before", "n_prior", "w_prior", "l_prior",
                    "streak_pre", "longest_run_so_far", "days_since"]]

    merged = md.merge(log_feats, how="left", left_on=["fid", "R_fighter"], right_on=["fid", "fighter"])
    merged = merged.drop(columns=["fighter"])
    merged = merged.rename(columns={c: f"{c}_R" for c in
        ["elo_before", "n_prior", "w_prior", "l_prior", "streak_pre", "longest_run_so_far", "days_since"]})
    merged = merged.merge(log_feats, how="left", left_on=["fid", "B_fighter"], right_on=["fid", "fighter"])
    merged = merged.drop(columns=["fighter"])
    merged = merged.rename(columns={c: f"{c}_B" for c in
        ["elo_before", "n_prior", "w_prior", "l_prior", "streak_pre", "longest_run_so_far", "days_since"]})

    for corner in ["R", "B"]:
        rjc = rj.rename(columns={c: f"{c}_{corner}" for c in rj.columns if c not in ("date", "fighter")})
        merged = merged.merge(rjc, how="left", left_on=["date_n", f"{corner}_fighter"],
                              right_on=["date", "fighter"])
        merged = merged.drop(columns=["date", "fighter"], errors="ignore")
    return merged


# ---------------- model feature columns (from run_backtest.py) ----------------

CORE_DIFF_PAIRS = [
    ("elo_before", "elo"),
    ("win_pct", "win_pct"),
    ("streak_pre", "streak"),
    ("longest_run_so_far", "longest_streak"),
    ("n_prior", "exp"),
    ("days_since", "days_since"),
]
STRIKE_DIFFS = ["slpm", "str_acc", "td_acc", "td_def", "sub_per_f", "kd_per_f", "ctrl_share"]


def add_model_features(df, wc_columns=None):
    """
    Apply the exact feature transforms from run_backtest.py.
    Returns (df, core_feats, all_feats). wc_columns: optional list of weight-class
    dummy columns to force (for cards with classes not seen in this slice).
    """
    df = df.copy()
    for s in ["R", "B"]:
        df[f"win_pct_{s}"] = (df[f"w_prior_{s}"] + 1.0) / (df[f"n_prior_{s}"] + 2.0)
        df[f"days_since_{s}"] = df[f"days_since_{s}"].fillna(999.0)
        df[f"elo_before_{s}"] = df[f"elo_before_{s}"].fillna(1500.0)
        df[f"streak_pre_{s}"] = df[f"streak_pre_{s}"].fillna(0.0)
        df[f"longest_run_so_far_{s}"] = df[f"longest_run_so_far_{s}"].fillna(0.0)
        df[f"n_prior_{s}"] = df[f"n_prior_{s}"].fillna(0.0)
        df[f"w_prior_{s}"] = df[f"w_prior_{s}"].fillna(0.0)
        df[f"l_prior_{s}"] = df[f"l_prior_{s}"].fillna(0.0)

    feat_names = []
    for col, name in CORE_DIFF_PAIRS:
        df[f"d_{name}"] = df[f"{col}_R"] - df[f"{col}_B"]
        feat_names.append(f"d_{name}")

    df["d_age"] = df["R_age"] - df["B_age"]; feat_names.append("d_age")
    df["d_reach"] = df["R_Reach_cms"] - df["B_Reach_cms"]; feat_names.append("d_reach")
    df["d_height"] = df["R_Height_cms"] - df["B_Height_cms"]; feat_names.append("d_height")
    df["d_weight"] = df["R_Weight_lbs"] - df["B_Weight_lbs"]; feat_names.append("d_weight")
    df["d_rank"] = df["B_match_weightclass_rank"].fillna(16) - df["R_match_weightclass_rank"].fillna(16)
    feat_names.append("d_rank")

    for c in STRIKE_DIFFS:
        df[f"d_{c}"] = df[f"feat_{c}_R"] - df[f"feat_{c}_B"]
        feat_names.append("d_" + c)

    cat_names = []
    df["is_female"] = (df["gender"] == "FEMALE").astype(int); cat_names.append("is_female")
    df["is_title"] = df["title_bout"].astype(int); cat_names.append("is_title")
    df["is_5rd"] = (df["no_of_rounds"] == 5).astype(int); cat_names.append("is_5rd")
    wc_dummies = pd.get_dummies(df["weight_class"], prefix="wc")
    if wc_columns is not None:
        for c in wc_columns:
            if c not in wc_dummies.columns:
                wc_dummies[c] = 0
        wc_dummies = wc_dummies[wc_columns]
    df = pd.concat([df, wc_dummies.astype(int)], axis=1)
    cat_names += list(wc_dummies.columns)

    core_feats = feat_names[:11] + cat_names
    all_feats = feat_names + cat_names
    return df, core_feats, all_feats


# ---------------- card helpers (ufc_picker) ----------------

STATIC_ATTRS = ["R_age", "R_Reach_cms", "R_Height_cms", "R_Weight_lbs", "R_match_weightclass_rank",
                "B_age", "B_Reach_cms", "B_Height_cms", "B_Weight_lbs", "B_match_weightclass_rank"]


def fighter_last_known(md, cutoff):
    """Last known static attributes per fighter from fights strictly before `cutoff`,
    in chronological order so the most recent fight wins. Also records the date of the
    last fight (for age adjustment to the card date)."""
    hist = md[md["date_n"] < cutoff].sort_values(["date_n", "fid"])
    lookup = {}
    for _, r in hist.iterrows():
        for corner in ["R", "B"]:
            f = r[f"{corner}_fighter"]
            entry = lookup.setdefault(f, {})
            for attr, col in [("age", f"{corner}_age"), ("Reach_cms", f"{corner}_Reach_cms"),
                              ("Height_cms", f"{corner}_Height_cms"),
                              ("Weight_lbs", f"{corner}_Weight_lbs"),
                              ("match_weightclass_rank", f"{corner}_match_weightclass_rank")]:
                val = r.get(col, np.nan)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    entry[attr] = val
            entry["last_fight_date"] = r["date_n"]
            if pd.notna(r.get("weight_class")):
                entry["weight_class"] = r["weight_class"]
            if pd.notna(r.get("gender")):
                entry["gender"] = r["gender"]
    return lookup


def card_rows_to_md(card_df, md, cutoff=None):
    """
    Convert an upcoming-card DataFrame (date, red_fighter, blue_fighter, weight_class,
    R_odds, B_odds [, title_bout, no_of_rounds, gender]) into rows matching the master
    schema, with static attrs filled from each fighter's last known fight before the card.
    Returns (card_md_rows, card_fids).
    """
    if cutoff is None:
        cutoff = pd.to_datetime(card_df["date"]).dt.normalize().min()
    else:
        cutoff = pd.Timestamp(cutoff).normalize()
    lookup = fighter_last_known(md, cutoff)

    next_fid = int(md["fid"].max()) + 1
    rows = []
    fids = []
    for _, c in card_df.iterrows():
        date = pd.Timestamp(c["date"]).normalize()
        wc = str(c["weight_class"])
        gender = c.get("gender") if isinstance(c.get("gender"), str) and c["gender"] else None
        if gender is None:
            gender = "FEMALE" if "women" in wc.lower() else "MALE"
        title = bool(c.get("title_bout", False)) if pd.notna(c.get("title_bout", False)) else False
        rounds = c.get("no_of_rounds")
        if rounds is None or pd.isna(rounds):
            rounds = 5 if title else 3
        row = {"fid": next_fid, "date": date, "date_n": date,
               "R_fighter": str(c["red_fighter"]).strip(), "B_fighter": str(c["blue_fighter"]).strip(),
               "weight_class": wc, "gender": gender, "title_bout": bool(title),
               "no_of_rounds": int(rounds),
               "R_odds": float(c["R_odds"]), "B_odds": float(c["B_odds"]),
               "R_ev": np.nan, "B_ev": np.nan, "Winner": None}
        for corner in ["R", "B"]:
            f = row[f"{corner}_fighter"]
            lk = lookup.get(f, {})
            for attr, col in [("age", f"{corner}_age"), ("Reach_cms", f"{corner}_Reach_cms"),
                              ("Height_cms", f"{corner}_Height_cms"), ("Weight_lbs", f"{corner}_Weight_lbs"),
                              ("match_weightclass_rank", f"{corner}_match_weightclass_rank")]:
                row[col] = lk.get(attr, np.nan)
            # age is "age at last fight" in the source data: shift it forward to the
            # card date for a live-fight estimate
            if pd.notna(row.get(f"{corner}_age")) and lk.get("last_fight_date") is not None:
                yrs = (date - lk["last_fight_date"]).days / 365.25
                row[f"{corner}_age"] = row[f"{corner}_age"] + yrs
        rows.append(row)
        fids.append(next_fid)
        next_fid += 1
    template = md.iloc[0:0].copy()
    card_md = pd.concat([template, pd.DataFrame(rows)], ignore_index=True)
    return card_md, fids


def build_card_features(md, long, card_df, cutoff=None):
    """
    History (fights strictly before the cutoff — default: first card date) + card
    rows appended, then the SAME rolling feature builders run over the combined frame.
    No future data: card rows only see pre-cutoff fights. Returns
    (merged_features_df, card_fids, cutoff).
    """
    if cutoff is None:
        cutoff = pd.to_datetime(card_df["date"]).dt.normalize().min()
    else:
        cutoff = pd.Timestamp(cutoff).normalize()
    hist = md[md["date_n"] < cutoff].copy()
    card_md, fids = card_rows_to_md(card_df, md, cutoff=cutoff)
    combined = pd.concat([hist, card_md], ignore_index=True)
    merged = build_rolling_features(combined, long)
    return merged, fids, cutoff
