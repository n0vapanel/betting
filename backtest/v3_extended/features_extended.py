"""Extended feature builder: v3's 56 features rebuilt on a 3-month walk-forward
window, plus two new feature groups.

New group 1 — official-filing injury impact (INJURY_EXT_FEATURES):
    Routing: the NBA-mandated 5PM ET filing. Evening tips (>=18:00 ET) use the
    game-day 5PM official report (snapshot_date == game_date, 17:00); earlier
    tips use the day-before 5PM report (snapshot_date == game_date - 1),
    which may contain 'NOT YET SUBMITTED' rows — those teams are dropped as
    no-information. Weights: Out 1.0, Doubtful 0.5, Questionable 0.25, all
    multiplied by the player's prior-minutes share of the team rotation
    (10-game-halflife EWMA minutes over the team's last <=10 games, zeros for
    absences, divided by the rotation total). 'injout' is the Out-only EWMA
    minutes share of a 240-minute team game.

New group 2 — minutes load (MINUTES_FEATURES) from the 766MB cached player
    boxscores, read in 50k-row chunks and aggregated to one compact row per
    team-game: total minutes, top-8 minutes, and the top-5-by-minutes player
    id set. Rolling (strictly prior dates only):
      load7        total team minutes over games in the prior 7 calendar days
                   (0.0 when the window has games=0, e.g. post All-Star break;
                   NaN only at true cold start with no prior games at all);
      top8share    mean over the team's last 5 games of top-8-minutes / team
                   minutes (rotation concentration);
      stability    mean over the last 6 games' consecutive pairs of
                   |top5_g ∩ top5_prev| / 5 (lineup continuity).

Windows: train 2021-10-19..2021-12-31 (model frozen before Jan 1), test
2022-01-01..2022-03-31 (locked 623-game benchmark, Kaggle closing moneylines).
Every dated source is asserted strictly earlier than its target date.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.v2 import data as D
from backtest.v2 import features as F2
from backtest.v3 import features_v3 as F3
from backtest.v3.baseline_v2 import fingerprints

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = ROOT / "backtest/cache"

TRAIN_START, TRAIN_END = "2021-10-19", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2022-03-31"
EXPECTED_TEST_GAMES = 623

STATUS_WEIGHTS = {"Out": 1.0, "Doubtful": 0.5, "Questionable": 0.25}

V3_FEATURES = list(F3.FEATURE_NAMES)  # the locked 56-feature v3 set
MINUTES_FEATURES = ["load7_home", "load7_away", "load7_diff",
                    "top8share_diff", "stability_diff"]
INJURY_EXT_FEATURES = ["injx_home", "injx_away", "injx_diff",
                       "injout_home", "injout_away", "injout_diff"]
ALL_FEATURES = V3_FEATURES + MINUTES_FEATURES + INJURY_EXT_FEATURES
FEATURE_GROUPS = dict(F3.FEATURE_GROUPS)
FEATURE_GROUPS.update({name: "minutes_load" for name in MINUTES_FEATURES})
FEATURE_GROUPS.update({name: "injuries_ext" for name in INJURY_EXT_FEATURES})

CANDIDATE_SPECS = {
    "v3_only": V3_FEATURES,
    "plus_injuries": V3_FEATURES + INJURY_EXT_FEATURES,
    "plus_minutes": V3_FEATURES + MINUTES_FEATURES,
    "all": V3_FEATURES + MINUTES_FEATURES + INJURY_EXT_FEATURES,
}

ADVANCED_COLUMNS = ["game_id", "game_date", "player_id", "team_abbreviation", "min",
                    "ts_pct", "usg_pct", "pie", "net_rating", "ast_pct", "tm_tov_pct"]


# ------------------------------------------------------------------ minutes load

def compact_team_minutes(players):
    """One row per team-game: team minutes, top-8 minutes, top-5 player id set."""
    frames = []
    for (gid, team, date), group in players.groupby(
            ["game_id", "team_abbreviation", "game_date"], sort=False):
        minutes = group["min"].to_numpy(dtype=float)
        ids = group["player_id"].astype(str).to_numpy()
        order = np.lexsort((ids, -minutes))
        minutes, ids = minutes[order], ids[order]
        frames.append({
            "game_id": gid, "team": team, "date": str(date)[:10],
            "team_min": float(minutes.sum()), "top8_min": float(minutes[:8].sum()),
            "top5_ids": "|".join(ids[:5]),
        })
    compact = pd.DataFrame(frames)
    if compact.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate compact team-minute rows")
    return compact


def minutes_load_features(compact):
    """Rolling minutes-load features with strictly-prior histories."""
    rows = []
    for team, group in compact.groupby("team", sort=False):
        group = group.sort_values(["date", "game_id"]).reset_index(drop=True)
        dates = pd.to_datetime(group["date"])
        for i in range(len(group)):
            prior = group.iloc[:i]
            if not len(prior):
                rows.append({"team": team, "game_id": group.at[i, "game_id"],
                             "date": group.at[i, "date"], "load7": np.nan,
                             "top8share": np.nan, "stability": np.nan,
                             "history_max_date": None, "n_prior": 0, "n_prior_7d": 0})
                continue
            cutoff = (dates.iloc[i] - pd.Timedelta(days=7)).to_datetime64()
            window7 = prior[dates.iloc[:i].to_numpy() >= cutoff]
            load7 = float(window7.team_min.sum())
            recent5 = prior.tail(5)
            top8share = float((recent5.top8_min / recent5.team_min).mean())
            last6 = prior.tail(6)
            if len(last6) >= 2:
                sets = [set(s.split("|")) for s in last6.top5_ids]
                stability = float(np.mean([len(a & b) / 5.0 for a, b in zip(sets[1:], sets[:-1])]))
            else:
                stability = np.nan
            rows.append({"team": team, "game_id": group.at[i, "game_id"],
                         "date": group.at[i, "date"], "load7": load7,
                         "top8share": top8share, "stability": stability,
                         "history_max_date": prior.date.max(), "n_prior": len(prior),
                         "n_prior_7d": len(window7)})
    result = pd.DataFrame(rows)
    known = result.history_max_date.notna()
    if not (result.loc[known, "history_max_date"] < result.loc[known, "date"]).all():
        raise AssertionError("Minutes-load feature source is not strictly earlier than target date")
    return result


# ------------------------------------------------------------------ injury ext

def _hour24(value):
    match = re.search(r"(\d{1,2}):(\d{2})", str(value))
    if not match:
        return 99
    hours = int(match.group(1))
    if hours < 12:
        hours += 12
    return hours


def official_filing(injuries, date, matchup):
    """Rows of the official 5PM ET filing for one game, or None.

    Primary: game-day 5PM report (snapshot (g, 17:00), Game Date == g) for
    evening tips (>=18:00 ET). Otherwise the day-before 5PM report
    (snapshot (g-1, 17:00), Game Date == g) with 'NOT YET SUBMITTED' rows
    dropped as no-information.
    """
    cand = injuries[(injuries.game_date == date) & (injuries.matchup == matchup)]
    gd17 = cand[(cand.snapshot_date == date) & (cand.snapshot_time == "17:00")]
    early = False
    if len(gd17):
        hours = [_hour24(t) for t in gd17.game_time.unique()]
        early = any(h < 18 for h in hours)
        if not early:
            return gd17, "game-day 5PM official filing", 0
    previous = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    day_before = injuries[(injuries.snapshot_date == previous)
                          & (injuries.snapshot_time == "17:00")
                          & (injuries.game_date == date)
                          & (injuries.matchup == matchup)]
    if len(day_before):
        nyss = day_before.reason.fillna("").str.contains("NOT YET SUBMITTED", case=False)
        cleaned = day_before.loc[~nyss]
        if len(cleaned):
            label = ("day-before 5PM official filing (early tip)" if early
                     else "day-before 5PM official filing (game-day missing)")
            return cleaned, label, int(nyss.sum())
        return None, "day-before 5PM filing has only NOT YET SUBMITTED rows", int(nyss.sum())
    if early:
        return None, "early tip and no usable day-before 5PM filing", 0
    return None, "no official 5PM filing rows", 0


def rotation_ewma(by_team_date, team_dates, team, target):
    """{player_norm: ewma minutes} over the team's last <=10 prior games."""
    dates = team_dates.get(team, [])
    prior = [d for d in dates if d < target][-10:]
    if not prior:
        return {}, 0
    names = set()
    frames = {}
    for d in prior:
        series = by_team_date.get((team, d))
        frames[d] = series if series is not None else pd.Series(dtype=float)
        names.update(frames[d].index)
    matrix = pd.DataFrame({d: frames[d].reindex(sorted(names)).fillna(0.0) for d in prior})
    n = len(prior)
    q = 2 ** (-1 / 10)
    weights = q ** np.arange(n - 1, -1, -1)
    minutes = matrix.to_numpy(dtype=float) @ weights / weights.sum()
    return dict(zip(matrix.index, minutes)), n


def injury_extended_features(games, injuries, playerlog, teamlog):
    """injx (weighted impact) and injout (Out minutes share) per side + diffs."""
    injuries = F2.prepare_injury_cache(injuries, teamlog)
    players = playerlog.copy()
    players["player_norm"] = players.PLAYER_NAME.apply(F2.norm_player_name)
    by_team_date = {}
    for (team, date), group in players.groupby(["TEAM_ABBREVIATION", "GAME_DATE"], sort=False):
        by_team_date[(team, str(date)[:10])] = group.set_index("player_norm")["MIN"].astype(float)
    team_dates = {}
    for team, group in players.groupby("TEAM_ABBREVIATION", sort=False):
        team_dates[team] = sorted(group.GAME_DATE.astype(str).str[:10].unique())
    rows, audits = [], []
    for game in games.itertuples(index=False):
        matchup = f"{game.away}@{game.home}"
        filing, source, nyss = official_filing(injuries, game.date, matchup)
        if filing is None or not len(filing):
            rows.append({"game_id": game.game_id, **{c: np.nan for c in INJURY_EXT_FEATURES}})
            audits.append({"game_id": game.game_id, "injury_ext_source": source,
                           "injury_ext_stamp": None, "injury_ext_rows": 0,
                           "injury_ext_nyss_dropped": nyss})
            continue
        reports = filing.drop_duplicates(["team_abbr", "player_norm"], keep="last")
        stamp = f"{filing.snapshot_date.iloc[0]} {filing.snapshot_time.iloc[0]}"
        values, prior_counts = {}, {}
        for side, team in [("home", game.home), ("away", game.away)]:
            rotation, n_prior = rotation_ewma(by_team_date, team_dates, team, game.date)
            total = sum(rotation.values())
            impact = out_min = 0.0
            for report in reports.loc[reports.team_abbr.eq(team)].itertuples(index=False):
                status = str(report.status).strip()
                weight = STATUS_WEIGHTS.get(status, 0.0)
                if weight <= 0:
                    continue
                minutes = rotation.get(report.player_norm, 0.0)
                share = minutes / total if total > 0 else 0.0
                impact += weight * share
                if status == "Out":
                    out_min += minutes
            values[side] = (impact, out_min / 240.0)
            prior_counts[side] = n_prior
        h, a = values["home"], values["away"]
        rows.append({"game_id": game.game_id, "injx_home": h[0], "injx_away": a[0],
                     "injx_diff": h[0] - a[0], "injout_home": h[1], "injout_away": a[1],
                     "injout_diff": h[1] - a[1]})
        audits.append({"game_id": game.game_id, "injury_ext_source": source,
                       "injury_ext_stamp": stamp, "injury_ext_rows": int(len(reports)),
                       "injury_ext_nyss_dropped": nyss,
                       "inj_rotation_prior_home": prior_counts["home"],
                       "inj_rotation_prior_away": prior_counts["away"]})
    return pd.DataFrame(rows), pd.DataFrame(audits)


def close_agreement(diagnostics, games):
    joined = games.merge(diagnostics, on=["game_id", "date"], how="left", validate="one_to_one")
    rows = {}
    for name, subset in [("training", joined.loc[joined.date.le(TRAIN_END)]),
                         ("test_window", joined.loc[joined.date.ge(TEST_START)])]:
        usable = subset.loc[subset.source_close_home.notna()]
        h = (usable.source_close_ml_home - usable.ml_home).abs()
        a = (usable.source_close_ml_away - usable.ml_away).abs()
        error = (usable.source_close_home - usable.p_market_home).abs()
        rows[name] = {
            "n_games": len(subset), "n_matched": len(usable),
            "exact_both_moneylines": int((h.eq(0) & a.eq(0)).sum()),
            "fraction_exact_both": float((h.eq(0) & a.eq(0)).mean()),
            "fraction_within_10_both": float((h.le(10) & a.le(10)).mean()),
            "mean_abs_home_probability_difference": float(error.mean()),
            "median_abs_home_probability_difference": float(error.median()),
            "max_abs_home_probability_difference": float(error.max()),
            "favorite_side_agreement": float(usable.source_close_home.gt(0.5).eq(usable.p_market_home.gt(0.5)).mean()),
        }
    return rows


# ------------------------------------------------------------------ assembly

def build_features():
    inputs = [
        CACHE / "v3/game_odds.csv", CACHE / "v3/team_boxscores.csv", CACHE / "v3/player_boxscores.csv",
        D.TEAMLOG_CACHE, D.PLAYERLOG_CACHE, D.INJURY_CACHE, D.KAGGLE_CSV,
        F3.HERE / "player_features_2021_22.csv", F3.HERE / "player_features.py",
        Path(F3.__file__), Path(__file__),
    ]
    signature = fingerprints(inputs)
    manifest_path = HERE / "feature_manifest.json"
    train_path, test_path = HERE / "features_train.csv", HERE / "features_test.csv"
    if manifest_path.exists() and train_path.exists() and test_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["input_fingerprints"] == signature:
            print("[features] Reusing fingerprint-verified feature capture", flush=True)
            return {
                "train": pd.read_csv(train_path, dtype={"game_id": str}),
                "test": pd.read_csv(test_path, dtype={"game_id": str}), "manifest": saved,
            }
    teamlog = pd.read_csv(D.TEAMLOG_CACHE, dtype={"GAME_ID": str})
    teamlog = teamlog.loc[teamlog.GAME_DATE.le(TEST_END)].copy()
    playerlog = pd.read_csv(D.PLAYERLOG_CACHE, dtype={"GAME_ID": str})
    playerlog = playerlog.loc[playerlog.GAME_DATE.le(TEST_END)].copy()
    base, schedule = F3.base_team_features(teamlog)
    kaggle = D.load_kaggle_games(TRAIN_START, TEST_END)
    games = schedule.merge(kaggle, on=["date", "home", "away"], validate="one_to_one",
                           suffixes=("", "_kaggle"))
    if not (games.score_home.eq(games.score_home_kaggle) & games.score_away.eq(games.score_away_kaggle)).all():
        raise AssertionError("NBA and Kaggle outcome mismatch")
    games["p_market_home"] = F3.no_vig(games.ml_home, games.ml_away)
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    odds, odds_stats = F3.load_filtered_csv(CACHE / "v3/game_odds.csv", [
        "game_id", "game_date", "odds_date", "moneyline_home", "moneyline_away",
    ])
    diagnostics = F3.summarize_odds(odds)
    diagnostics.to_csv(HERE / "odds_diagnostics.csv", index=False)
    box_columns = ["game_id", "game_date", "team_abbreviation", "pace", "ts_pct", "off_rating", "def_rating",
                   "fga", "fta", "oreb", "tov", "pts", "fg3a", "ast"] + [
                       f"q{i}_{metric}" for i in range(1, 5) for metric in ["pts", "plus_minus"]
                   ]
    box, box_stats = F3.load_filtered_csv(CACHE / "v3/team_boxscores.csv", box_columns)
    players, player_stats = F3.load_filtered_csv(CACHE / "v3/player_boxscores.csv", ADVANCED_COLUMNS)
    if players.duplicated(["game_id", "team_abbreviation", "player_id"]).any():
        raise ValueError("Duplicate advanced player records")
    compact = compact_team_minutes(players)
    minutes_roll = minutes_load_features(compact)
    precomputed = pd.read_csv(F3.HERE / "player_features_2021_22.csv", dtype={"GAME_ID": str})
    precomputed = precomputed.rename(columns={"GAME_ID": "game_id", "GAME_DATE": "date",
                                              "TEAM_ABBREVIATION": "team"})
    precomputed["game_id"] = F3.game_ids(precomputed.game_id)
    if precomputed.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate supplied player features")
    box_roll = F3.box_team_features(box)
    advanced_roll = F3.advanced_player_features(players)
    history_roll = F3.odds_history_features(diagnostics, schedule)
    injury, injury_audit = F3.injury_features(
        games, pd.read_csv(D.INJURY_CACHE, dtype=str), playerlog, teamlog
    )
    injury_ext, injury_ext_audit = injury_extended_features(
        games, pd.read_csv(D.INJURY_CACHE, dtype=str), playerlog, teamlog
    )
    assembled = games.merge(F3.opening_features(odds), on="game_id", how="left", validate="one_to_one")
    assembled = assembled.merge(injury, on="game_id", validate="one_to_one")
    assembled = assembled.merge(injury_audit, on="game_id", validate="one_to_one")
    assembled = assembled.merge(injury_ext, on="game_id", validate="one_to_one")
    assembled = assembled.merge(injury_ext_audit, on="game_id", validate="one_to_one")
    for name, table in [("base", base), ("box", box_roll), ("advanced", advanced_roll),
                        ("odds_history", history_roll), ("player", precomputed),
                        ("minutes", minutes_roll)]:
        for side in ["home", "away"]:
            rename = {c: f"{name}_{c}_{side}" for c in table if c not in ["game_id", "team", "date"]}
            part = table.rename(columns=rename).drop(columns="date", errors="ignore")
            assembled = assembled.merge(
                part, left_on=["game_id", side], right_on=["game_id", "team"], how="left",
                validate="one_to_one"
            ).drop(columns="team")
    for side, other in [("home", "away"), ("away", "home")]:
        assembled[f"off_eFG_{side}"] = assembled[f"base_off_eFG_mean_{side}"] - assembled[f"base_def_eFG_mean_{other}"]
        assembled[f"tov_{side}"] = assembled[f"base_def_TOVpct_mean_{other}"] - assembled[f"base_off_TOVpct_mean_{side}"]
        assembled[f"orb_{side}"] = assembled[f"base_ORBpct_mean_{side}"] - assembled[f"base_def_ORBpct_mean_{other}"]
        assembled[f"ftr_{side}"] = assembled[f"base_off_FTr_mean_{side}"] - assembled[f"base_def_FTr_mean_{other}"]
        assembled[f"xprod_{side}"] = assembled[f"off_eFG_{side}"] * assembled[f"orb_{side}"]
        assembled[f"b2b_{side}"] = assembled[f"base_b2b_{side}"]
    for feature, field in [("pace_diff", "pace_mean"), ("rest_diff", "rest"),
                           ("winpct_diff", "winpct"), ("margin_diff", "s2d_margin")]:
        assembled[feature] = assembled[f"base_{field}_home"] - assembled[f"base_{field}_away"]
    assembled["market_home"] = assembled.p_market_home
    assembled["market_logit"] = np.log(assembled.market_home / (1 - assembled.market_home))
    for value in F3.PLAYER_VALUES:
        assembled[f"player_{value}_diff"] = assembled[f"player_{value}_home"] - assembled[f"player_{value}_away"]
    for value in F3.ADV_VALUES:
        assembled[f"player_{value}_diff"] = assembled[f"advanced_{value}_mean_home"] - assembled[f"advanced_{value}_mean_away"]
    for value in F3.BOX_VALUES:
        assembled[f"{value}_diff"] = assembled[f"box_{value}_mean_home"] - assembled[f"box_{value}_mean_away"]
    for value in ["box_net", "box_ts"]:
        assembled[f"{value}_ewma_diff"] = assembled[f"box_{value}_ewma_home"] - assembled[f"box_{value}_ewma_away"]
    for value in F3.ODDS_HISTORY:
        assembled[f"{value}_diff"] = assembled[f"odds_history_{value}_mean_home"] - assembled[f"odds_history_{value}_mean_away"]
    assembled["load7_home"] = assembled.minutes_load7_home
    assembled["load7_away"] = assembled.minutes_load7_away
    assembled["load7_diff"] = assembled.load7_home - assembled.load7_away
    assembled["top8share_diff"] = assembled.minutes_top8share_home - assembled.minutes_top8share_away
    assembled["stability_diff"] = assembled.minutes_stability_home - assembled.minutes_stability_away
    history_columns = [c for c in assembled if "history_max_date" in c]
    for column in history_columns:
        known = assembled[column].notna()
        if not assembled.loc[known, column].lt(assembled.loc[known, "date"]).all():
            raise AssertionError(f"Temporal leakage in {column}")
    known_injury = assembled.injury_snapshot_et.notna()
    if not assembled.loc[known_injury, "injury_snapshot_et"].lt(assembled.loc[known_injury, "tip_et"]).all():
        raise AssertionError("Post-tip or at-tip injury snapshot")
    known_ext = assembled.injury_ext_stamp.notna()
    if not assembled.loc[known_ext, "injury_ext_stamp"].lt(assembled.loc[known_ext, "date"] + " 23:59:59").all():
        raise AssertionError("Official-filing stamp is not on or before the target calendar date")
    complete_all = np.isfinite(assembled[ALL_FEATURES].to_numpy(dtype=float)).all(axis=1)
    train_mask = assembled.date.between(TRAIN_START, TRAIN_END)
    test_mask = assembled.date.between(TEST_START, TEST_END)
    train = assembled.loc[train_mask & complete_all].copy()
    test = assembled.loc[test_mask].copy()
    if len(test) != EXPECTED_TEST_GAMES or test.duplicated(["date", "home", "away"]).any():
        raise AssertionError(f"Locked 3-month benchmark must contain {EXPECTED_TEST_GAMES} "
                             f"distinct games, got {len(test)}")
    output_columns = ["game_id", "date", "home", "away", "home_won", "dec_home", "dec_away",
                      "p_market_home"] + ALL_FEATURES
    train = train[output_columns]
    test = test[output_columns]
    missing = {
        c: int(assembled.loc[train_mask, c].isna().sum())
        for c in ALL_FEATURES if assembled.loc[train_mask, c].isna().any()
    }
    audit = assembled[["game_id", "date", "home", "away", "injury_snapshot_et", "tip_et",
                       "injury_source_issue", "injury_ext_source", "injury_ext_stamp",
                       "injury_ext_rows", "injury_ext_nyss_dropped",
                       "inj_rotation_prior_home", "inj_rotation_prior_away"] + history_columns].copy()
    audit["complete_all_features"] = complete_all
    audit["missing_features"] = assembled[ALL_FEATURES].isna().apply(
        lambda r: ",".join(r.index[r]), axis=1)
    audit.to_csv(HERE / "feature_audit.csv", index=False)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    filing_counts = audit.loc[audit.injury_ext_source.notna(), "injury_ext_source"].value_counts().to_dict()
    manifest = {
        "input_fingerprints": signature, "n_train": len(train), "n_test": len(test),
        "n_features": len(ALL_FEATURES), "train_date_min": train.date.min(), "train_date_max": train.date.max(),
        "test_date_min": test.date.min(), "test_date_max": test.date.max(),
        "training_candidates_with_closing_lines": int(train_mask.sum()),
        "training_dropped_incomplete": int((train_mask & ~complete_all).sum()),
        "training_missing_by_feature": missing,
        "test_incomplete_games": audit.loc[test_mask & ~complete_all, [
            "game_id", "date", "home", "away", "missing_features"
        ]].to_dict("records"),
        "test_complete_per_candidate": {
            name: int(np.isfinite(test[columns].to_numpy(dtype=float)).all(axis=1).sum())
            for name, columns in CANDIDATE_SPECS.items()
        },
        "test_fallback_policy": (
            "Before evaluation, any Jan-Mar test row missing a candidate's features is assigned "
            "Kaggle no-vig market probability for that candidate. Never remove a benchmark game "
            "or fabricate a filing/quote."
        ),
        "feature_groups": FEATURE_GROUPS, "odds_read": odds_stats, "boxscore_read": box_stats,
        "player_read": player_stats,
        "closing_line_agreement": close_agreement(diagnostics, games),
        "injury_snapshot_counts": assembled.injury_snapshot_et.str[-5:].value_counts().to_dict(),
        "official_filing_counts": filing_counts,
        "official_filing_policy": (
            "Official NBA injury report = the 5PM ET filing. Evening tips (>=18:00 ET) use the "
            "game-day 5PM report (snapshot_date == game_date); earlier tips use the day-before "
            "5PM report (snapshot_date == game_date - 1) with 'NOT YET SUBMITTED' teams dropped. "
            "Impact = status weight (Out 1.0, Doubtful 0.5, Questionable 0.25) x player's "
            "prior-minutes share (10-game-halflife EWMA minutes over the team's last <=10 games, "
            "zeros for absences, divided by rotation minutes). injout = Out-only EWMA minutes / 240."
        ),
        "minutes_load_policy": (
            "From the cached player boxscores (chunked 50k-row reads, single pass, columns "
            "game_id/game_date/player_id/team_abbreviation/min). load7 = team minutes over prior "
            "7 calendar days (0.0 when the window contains zero games, e.g. after the All-Star "
            "break; NaN only at cold start). top8share = mean over the team's last 5 games of "
            "top-8 minutes / team minutes. stability = mean over the last 6 games' consecutive "
            "pairs of |top5_g AND top5_prev| / 5. All histories strictly prior to the target date."
        ),
        "timestamp_policy": {
            "statistics": "Completed games with source date strictly less than target date; rolling last 10, minimum 5, EWMA half-life 10.",
            "opening": "Only odds_date == open; no marker publication timestamp exists. Semantic historical opening, not exact point-in-time provable.",
            "closing": "Kaggle supplied closing moneylines are the sole closing market input and settlement price; actual publication timestamps unavailable.",
            "movement": "Current game close/move/steam/update-count fields are never predictors. Historical versions use strictly previous game dates.",
            "injuries_v3": "Latest complete official snapshot with Eastern timestamp strictly less than scheduled Eastern tip (v3 formulation, part of the 56).",
            "injuries_ext": "Official 5PM ET game-day filing for evening tips; day-before 5PM filing for earlier tips with NOT-YET-SUBMITTED teams dropped.",
            "minutes": "Player boxscore histories strictly prior to the target date; 7-day window excludes the target day.",
            "test_window": "Model weights, calibration, mixture and gate freeze on 2021-12-31. Completed Jan-Mar games update statistics only, never model parameters.",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"[features] {len(train)} complete training games, {len(test)} test games, "
          f"{len(ALL_FEATURES)} features; {len(manifest['test_incomplete_games'])} "
          f"all-feature test fallback(s)", flush=True)
    return {"train": train.reset_index(drop=True), "test": test.reset_index(drop=True),
            "manifest": manifest}
