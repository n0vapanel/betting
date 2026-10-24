from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.v2 import data as D
from backtest.v2 import features as F2
from backtest.v3.baseline_v2 import fingerprints


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = ROOT / "backtest/cache"
TRAIN_START, TRAIN_END = "2021-10-20", "2022-02-28"
TEST_START, TEST_END = "2022-03-01", "2022-03-31"
PLAYER_VALUES = ["team_rating", "top8_share", "depth_dropoff", "pts_margin_ewma",
                 "margin_std_ewma", "plus_minus_ewma"]
BOX_VALUES = ["box_pace", "box_ts", "box_off", "box_def", "box_net", "box_off_proxy",
              "box_def_proxy", "box_q_score_std", "box_q_margin_std", "box_q4_margin",
              "box_three_rate", "box_ast_to"]
ADV_VALUES = ["adv_ts", "adv_pie", "adv_net", "adv_creation", "adv_turnover", "adv_usage_hhi"]
ODDS_HISTORY = ["prior_clv", "prior_steam", "prior_updates"]
CLV_FEATURES = ["open_implied_home", "open_implied_away", "open_home"] + [f"{x}_diff" for x in ODDS_HISTORY]
PLAYER_FEATURES = [f"player_{x}_diff" for x in PLAYER_VALUES] + [f"player_{x}_diff" for x in ADV_VALUES]
BOX_FEATURES = [f"{x}_diff" for x in BOX_VALUES] + ["box_net_ewma_diff", "box_ts_ewma_diff"]
INJURY_FEATURES = ["inj_home", "inj_away", "inj_diff", "inj_scoring_diff", "inj_questionable_diff"]
FEATURE_NAMES = list(F2.FEATURE_NAMES) + ["market_logit"] + PLAYER_FEATURES + CLV_FEATURES + BOX_FEATURES + INJURY_FEATURES[3:]
FEATURE_GROUPS = {
    name: ("players" if name in PLAYER_FEATURES else "clv" if name in CLV_FEATURES
           else "boxscores" if name in BOX_FEATURES else "injuries" if name in INJURY_FEATURES
           else "market" if name in ["market_home", "market_logit"] else "v2_statistics")
    for name in FEATURE_NAMES
}
BASE_METRICS = ["off_eFG", "off_TOVpct", "off_FTr", "ORBpct", "def_eFG",
                "def_TOVpct", "def_FTr", "def_ORBpct", "pace"]


def american_decimal(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values == 0).any():
        raise ValueError("Moneylines must be finite and nonzero")
    return 1 + np.where(values > 0, values / 100, 100 / np.abs(values))


def no_vig(home, away):
    ih, ia = 1 / american_decimal(home), 1 / american_decimal(away)
    return ih / (ih + ia)


def game_ids(values):
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def load_filtered_csv(path, columns, date_column="game_date", chunksize=50000):
    kept, count = [], 0
    dtype = {c: str for c in ["game_id", "player_id"] if c in columns}
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, dtype=dtype):
        count += len(chunk)
        dates = chunk[date_column].astype(str).str[:10]
        selected = chunk.loc[dates.between("2021-10-01", TEST_END)].copy()
        if "game_id" in selected:
            selected["game_id"] = game_ids(selected["game_id"])
            selected = selected.loc[selected.game_id.str.startswith("00221")]
        kept.append(selected)
    result = pd.concat(kept, ignore_index=True)
    return result, {"rows_scanned": count, "rows_retained": len(result), "chunksize": chunksize,
                    "columns_read": columns}


def lagged_aggregates(frame, columns, window=10, min_periods=5):
    source = frame.copy()
    source["date"] = source["date"].astype(str).str[:10]
    if source.duplicated(["team", "game_id"]).any():
        raise ValueError("Duplicate team-game rows")
    pieces = []
    for team, group in source.groupby("team", sort=False):
        group = group.sort_values(["date", "game_id"]).reset_index(drop=True)
        numeric = group[columns].astype(float)
        roll = numeric.rolling(window, min_periods=min_periods)
        aggregates = pd.concat([
            roll.mean().add_suffix("_mean"),
            numeric.ewm(halflife=10, adjust=True, min_periods=min_periods).mean().add_suffix("_ewma"),
            roll.std(ddof=1).add_suffix("_std"),
        ], axis=1)
        aggregates["date"] = group.date
        previous = aggregates.groupby("date", sort=False).tail(1).set_index("date").shift(1)
        dates = pd.Series(group.date.drop_duplicates().to_numpy())
        prior_dates = dict(zip(dates, dates.shift(1)))
        out = group[["team", "date", "game_id"]].merge(previous, left_on="date", right_index=True, how="left")
        out["history_max_date"] = out.date.map(prior_dates)
        out["n_prior"] = out.date.map(group.groupby("date").size().cumsum().shift(1).fillna(0))
        pieces.append(out)
    result = pd.concat(pieces, ignore_index=True)
    known = result.history_max_date.notna()
    if not (result.loc[known, "history_max_date"] < result.loc[known, "date"]).all():
        raise AssertionError("Rolling feature source is not strictly earlier than target date")
    return result


def opening_features(odds):
    opened = odds.loc[odds.odds_date.eq("open")].copy()
    opened["game_id"] = game_ids(opened.game_id)
    if opened.duplicated("game_id").any():
        raise ValueError("Conflicting or duplicate opening markers")
    opened["open_implied_home"] = 1 / american_decimal(opened.moneyline_home)
    opened["open_implied_away"] = 1 / american_decimal(opened.moneyline_away)
    opened["open_home"] = opened.open_implied_home / (opened.open_implied_home + opened.open_implied_away)
    return opened[["game_id", "open_implied_home", "open_implied_away", "open_home"]].reset_index(drop=True)


def summarize_odds(odds):
    rows = []
    for gid, group in odds.groupby("game_id", sort=False):
        opened = group.loc[group.odds_date.eq("open")]
        closed = group.loc[group.odds_date.eq("close")]
        if len(opened) > 1 or len(closed) > 1:
            raise ValueError(f"Duplicate odds markers for {gid}")
        date = group.game_date.iloc[0][:10]
        open_p = float(no_vig(opened.moneyline_home, opened.moneyline_away)[0]) if len(opened) else np.nan
        close_p = float(no_vig(closed.moneyline_home, closed.moneyline_away)[0]) if len(closed) else np.nan
        timed = group.loc[~group.odds_date.isin(["open", "close"])].copy()
        timed["timestamp"] = pd.to_datetime(timed.odds_date, utc=True, errors="coerce", format="mixed")
        local_dates = timed.timestamp.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        intraday = timed.loc[local_dates.eq(date)].sort_values("timestamp")
        prices = no_vig(intraday.moneyline_home, intraday.moneyline_away) if len(intraday) else np.array([])
        sequence = np.r_[open_p, prices] if np.isfinite(open_p) else prices
        steam = float(np.abs(np.diff(sequence)).max()) if len(sequence) > 1 else 0.0
        rows.append({
            "game_id": str(gid).zfill(10), "date": date, "open_home": open_p,
            "source_close_home": close_p, "move_home": close_p - open_p,
            "move_away": open_p - close_p, "steam": steam, "n_updates": len(intraday),
            "first_update_utc": str(intraday.timestamp.min()) if len(intraday) else None,
            "last_update_utc": str(intraday.timestamp.max()) if len(intraday) else None,
            "source_close_ml_home": float(closed.moneyline_home.iloc[0]) if len(closed) else np.nan,
            "source_close_ml_away": float(closed.moneyline_away.iloc[0]) if len(closed) else np.nan,
            "opening_timestamp": "semantic open marker; publication timestamp absent" if len(opened) else None,
        })
    return pd.DataFrame(rows)


def injury_snapshot(injuries, date, home, away):
    rows = injuries.loc[injuries.game_date.eq(date) & injuries.matchup.eq(f"{away}@{home}")].copy()
    if rows.empty:
        raise ValueError(f"No official injury snapshot for {date} {away}@{home}")
    hours, minutes = map(int, re.search(r"(\d{1,2}):(\d{2})", str(rows.game_time.iloc[0])).groups())
    if hours < 12:
        hours += 12
    tip = f"{date} {hours:02d}:{minutes:02d}"
    rows["snapshot"] = rows.snapshot_date + " " + rows.snapshot_time
    rows = rows.loc[rows.snapshot.lt(tip)]
    if rows.empty:
        raise ValueError(f"No strictly pre-tip injury report for {date} {away}@{home}")
    for stamp in sorted(rows.snapshot.unique(), reverse=True):
        snapshot = rows.loc[rows.snapshot.eq(stamp)]
        if "reason" not in snapshot or not snapshot.reason.fillna("").str.contains("NOT YET SUBMITTED", case=False).any():
            return snapshot.copy(), stamp, tip
    raise ValueError(f"Only incomplete pre-tip injury reports for {date} {away}@{home}")


def base_team_features(teamlog):
    frame = teamlog.copy()
    frame["game_id"] = game_ids(frame.GAME_ID)
    frame["date"] = frame.GAME_DATE.astype(str).str[:10]
    frame["team"] = frame.TEAM_ABBREVIATION
    frame["is_home"] = frame.MATCHUP.str.contains(" vs. ", regex=False)
    frame["poss"] = frame.FGA + 0.44 * frame.FTA - frame.OREB + frame.TOV
    frame["off_eFG"] = (frame.FGM + 0.5 * frame.FG3M) / frame.FGA
    frame["off_TOVpct"] = frame.TOV / frame.poss
    frame["off_FTr"] = frame.FTM / frame.FGA
    frame["win"] = frame.WL.eq("W").astype(float)
    opponent = frame[["game_id", "team", "OREB", "DREB", "PTS", "poss", "off_eFG", "off_TOVpct", "off_FTr"]].rename(
        columns={c: "opp_" + c for c in ["team", "OREB", "DREB", "PTS", "poss", "off_eFG", "off_TOVpct", "off_FTr"]}
    )
    frame = frame.merge(opponent, on="game_id")
    frame = frame.loc[frame.team.ne(frame.opp_team)].copy()
    if frame.duplicated(["team", "game_id"]).any():
        raise ValueError("Non-unique paired team logs")
    frame["ORBpct"] = frame.OREB / (frame.OREB + frame.opp_DREB)
    frame["def_ORBpct"] = frame.opp_OREB / (frame.opp_OREB + frame.DREB)
    for metric in ["eFG", "TOVpct", "FTr"]:
        frame["def_" + metric] = frame["opp_off_" + metric]
    frame["margin"] = frame.PTS - frame.opp_PTS
    frame["pace"] = 0.5 * (frame.poss + frame.opp_poss) * 240 / frame.MIN
    rolled = lagged_aggregates(frame, BASE_METRICS)
    prior = []
    for team, group in frame.groupby("team", sort=False):
        group = group.sort_values(["date", "game_id"])
        dates = group.groupby("date").agg(wins=("win", "sum"), margins=("margin", "sum"), n=("win", "size"))
        dates["winpct"] = dates.wins.cumsum().shift(1) / dates.n.cumsum().shift(1)
        dates["s2d_margin"] = dates.margins.cumsum().shift(1) / dates.n.cumsum().shift(1)
        dates["rest"] = pd.to_datetime(dates.index).to_series(index=dates.index).diff().dt.days - 1
        dates["b2b"] = dates.rest.eq(0).astype(float)
        values = group[["game_id", "team", "date"]].merge(
            dates[["winpct", "s2d_margin", "rest", "b2b"]], left_on="date", right_index=True
        )
        prior.append(values)
    rolled = rolled.merge(pd.concat(prior), on=["game_id", "team", "date"], validate="one_to_one")
    home = frame.loc[frame.is_home, ["game_id", "date", "team", "PTS"]].rename(columns={"team": "home", "PTS": "score_home"})
    away = frame.loc[~frame.is_home, ["game_id", "team", "PTS"]].rename(columns={"team": "away", "PTS": "score_away"})
    games = home.merge(away, on="game_id", validate="one_to_one").sort_values(["date", "game_id"])
    elo = F2.build_elo_gaps(games.rename(columns={"game_id": "GAME_ID"})).rename(columns={"GAME_ID": "game_id"})
    return rolled, elo


def box_team_features(box):
    frame = box.rename(columns={"team_abbreviation": "team"}).copy()
    frame["date"] = frame.game_date.str[:10]
    frame["box_pace"], frame["box_ts"] = frame.pace, frame.ts_pct
    frame["box_off"], frame["box_def"] = frame.off_rating, frame.def_rating
    frame["box_net"] = frame.off_rating - frame.def_rating
    frame["estimated_poss"] = frame.fga + 0.44 * frame.fta - frame.oreb + frame.tov
    opp = frame[["game_id", "team", "pts"]].rename(columns={"team": "opponent", "pts": "opp_pts"})
    frame = frame.merge(opp, on="game_id")
    frame = frame.loc[frame.team.ne(frame.opponent)].copy()
    frame["box_off_proxy"] = 100 * frame.pts / frame.estimated_poss
    frame["box_def_proxy"] = 100 * frame.opp_pts / frame.estimated_poss
    frame["box_q_score_std"] = frame[[f"q{i}_pts" for i in range(1, 5)]].std(axis=1, ddof=0)
    frame["box_q_margin_std"] = frame[[f"q{i}_plus_minus" for i in range(1, 5)]].std(axis=1, ddof=0)
    frame["box_q4_margin"] = frame.q4_plus_minus
    frame["box_three_rate"] = frame.fg3a / frame.fga
    frame["box_ast_to"] = frame.ast / (frame.tov + 1)
    return lagged_aggregates(frame, BOX_VALUES)


def advanced_player_features(players):
    rows = []
    for (team, gid, date), group in players.groupby(["team_abbreviation", "game_id", "game_date"], sort=False):
        weight = group["min"].to_numpy(dtype=float)
        if weight.sum() <= 0:
            raise ValueError(f"No played minutes for {gid} {team}")
        usage = weight * group.usg_pct.to_numpy(dtype=float)
        hhi = float(np.sum((usage / usage.sum()) ** 2)) if usage.sum() > 0 else 0.0
        row = {"team": team, "game_id": gid, "date": date[:10], "adv_usage_hhi": hhi}
        for name, column in {"adv_ts": "ts_pct", "adv_pie": "pie", "adv_net": "net_rating",
                             "adv_creation": "ast_pct", "adv_turnover": "tm_tov_pct"}.items():
            row[name] = float(np.average(group[column].to_numpy(dtype=float), weights=weight))
        rows.append(row)
    return lagged_aggregates(pd.DataFrame(rows), ADV_VALUES)


def odds_history_features(diagnostics, schedule):
    rows = []
    lookup = diagnostics.set_index("game_id")
    for game in schedule.itertuples(index=False):
        if game.game_id in lookup.index:
            quote = lookup.loc[game.game_id]
            movement, steam, updates = quote.move_home, quote.steam, quote.n_updates
        else:
            movement = steam = updates = np.nan
        for team, sign in [(game.home, 1), (game.away, -1)]:
            rows.append({"team": team, "game_id": game.game_id, "date": game.date,
                         "prior_clv": sign * movement, "prior_steam": steam, "prior_updates": updates})
    return lagged_aggregates(pd.DataFrame(rows), ODDS_HISTORY)


def injury_features(games, injuries, playerlog, teamlog):
    injuries = F2.prepare_injury_cache(injuries, teamlog)
    players = playerlog.copy()
    players["player_norm"] = players.PLAYER_NAME.apply(F2.norm_player_name)
    history = {}
    for key, group in players.groupby(["TEAM_ABBREVIATION", "player_norm"], sort=False):
        group = group.sort_values(["GAME_DATE", "GAME_ID"])
        history[key] = group[["GAME_DATE", "MIN", "PTS"]].reset_index(drop=True)
    rows, audits = [], []
    for game in games.itertuples(index=False):
        try:
            reports, stamp, tip = injury_snapshot(injuries, game.date, game.home, game.away)
        except ValueError as error:
            rows.append({"game_id": game.game_id, **{c: np.nan for c in INJURY_FEATURES}})
            audits.append({"game_id": game.game_id, "injury_snapshot_et": None, "tip_et": None,
                           "injury_source_issue": str(error)})
            continue
        reports = reports.drop_duplicates(["team_abbr", "player_norm"], keep="last")
        values = {}
        for side, team in [("home", game.home), ("away", game.away)]:
            impact = scoring = questionable = 0.0
            for report in reports.loc[reports.team_abbr.eq(team)].itertuples(index=False):
                source = history.get((team, report.player_norm))
                if source is None:
                    continue
                past = source.loc[source.GAME_DATE.lt(game.date)].tail(10)
                if past.empty:
                    continue
                status = str(report.status).strip()
                weight = D.STATUS_WEIGHTS.get(status, 0.0)
                share = float(past.MIN.mean()) / 240
                impact += weight * share
                scoring += weight * float(past.PTS.mean())
                questionable += float(status == "Questionable") * share
            values[side] = (impact, scoring, questionable)
        h, a = values["home"], values["away"]
        rows.append({"game_id": game.game_id, "inj_home": h[0], "inj_away": a[0],
                     "inj_diff": h[0] - a[0], "inj_scoring_diff": h[1] - a[1],
                     "inj_questionable_diff": h[2] - a[2]})
        audits.append({"game_id": game.game_id, "injury_snapshot_et": stamp, "tip_et": tip,
                       "injury_source_issue": None})
    return pd.DataFrame(rows), pd.DataFrame(audits)


def close_agreement(diagnostics, games):
    joined = games.merge(diagnostics, on=["game_id", "date"], how="left", validate="one_to_one")
    rows = {}
    for name, subset in [("training", joined.loc[joined.date.le(TRAIN_END)]),
                         ("march", joined.loc[joined.date.ge(TEST_START)])]:
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


def build_features():
    inputs = [
        CACHE / "v3/game_odds.csv", CACHE / "v3/team_boxscores.csv", CACHE / "v3/player_boxscores.csv",
        D.TEAMLOG_CACHE, D.PLAYERLOG_CACHE, D.INJURY_CACHE, D.KAGGLE_CSV,
        HERE / "player_features_2021_22.csv", HERE / "player_features.py", Path(__file__),
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
    base, schedule = base_team_features(teamlog)
    kaggle = D.load_kaggle_games(TRAIN_START, TEST_END)
    games = schedule.merge(kaggle, on=["date", "home", "away"], validate="one_to_one", suffixes=("", "_kaggle"))
    if not (games.score_home.eq(games.score_home_kaggle) & games.score_away.eq(games.score_away_kaggle)).all():
        raise AssertionError("NBA and Kaggle outcome mismatch")
    games["p_market_home"] = no_vig(games.ml_home, games.ml_away)
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    odds, odds_stats = load_filtered_csv(CACHE / "v3/game_odds.csv", [
        "game_id", "game_date", "odds_date", "moneyline_home", "moneyline_away",
    ])
    diagnostics = summarize_odds(odds)
    diagnostics.to_csv(HERE / "odds_diagnostics.csv", index=False)
    box_columns = ["game_id", "game_date", "team_abbreviation", "pace", "ts_pct", "off_rating", "def_rating",
                   "fga", "fta", "oreb", "tov", "pts", "fg3a", "ast"] + [
                       f"q{i}_{metric}" for i in range(1, 5) for metric in ["pts", "plus_minus"]
                   ]
    box, box_stats = load_filtered_csv(CACHE / "v3/team_boxscores.csv", box_columns)
    advanced_columns = ["game_id", "game_date", "player_id", "team_abbreviation", "min",
                        "ts_pct", "usg_pct", "pie", "net_rating", "ast_pct", "tm_tov_pct"]
    players, player_stats = load_filtered_csv(CACHE / "v3/player_boxscores.csv", advanced_columns)
    if players.duplicated(["game_id", "team_abbreviation", "player_id"]).any():
        raise ValueError("Duplicate advanced player records")
    precomputed = pd.read_csv(HERE / "player_features_2021_22.csv", dtype={"GAME_ID": str})
    precomputed = precomputed.rename(columns={"GAME_ID": "game_id", "GAME_DATE": "date", "TEAM_ABBREVIATION": "team"})
    precomputed["game_id"] = game_ids(precomputed.game_id)
    if precomputed.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate supplied player features")
    box_roll = box_team_features(box)
    advanced_roll = advanced_player_features(players)
    history_roll = odds_history_features(diagnostics, schedule)
    injury, injury_audit = injury_features(
        games, pd.read_csv(D.INJURY_CACHE, dtype=str), playerlog, teamlog
    )
    assembled = games.merge(opening_features(odds), on="game_id", how="left", validate="one_to_one")
    assembled = assembled.merge(injury, on="game_id", validate="one_to_one")
    assembled = assembled.merge(injury_audit, on="game_id", validate="one_to_one")
    for name, table in [("base", base), ("box", box_roll), ("advanced", advanced_roll),
                        ("odds_history", history_roll), ("player", precomputed)]:
        for side in ["home", "away"]:
            rename = {c: f"{name}_{c}_{side}" for c in table if c not in ["game_id", "team", "date"]}
            part = table.rename(columns=rename).drop(columns="date", errors="ignore")
            assembled = assembled.merge(
                part, left_on=["game_id", side], right_on=["game_id", "team"], how="left", validate="one_to_one"
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
    for value in PLAYER_VALUES:
        assembled[f"player_{value}_diff"] = assembled[f"player_{value}_home"] - assembled[f"player_{value}_away"]
    for value in ADV_VALUES:
        assembled[f"player_{value}_diff"] = assembled[f"advanced_{value}_mean_home"] - assembled[f"advanced_{value}_mean_away"]
    for value in BOX_VALUES:
        assembled[f"{value}_diff"] = assembled[f"box_{value}_mean_home"] - assembled[f"box_{value}_mean_away"]
    for value in ["box_net", "box_ts"]:
        assembled[f"{value}_ewma_diff"] = assembled[f"box_{value}_ewma_home"] - assembled[f"box_{value}_ewma_away"]
    for value in ODDS_HISTORY:
        assembled[f"{value}_diff"] = assembled[f"odds_history_{value}_mean_home"] - assembled[f"odds_history_{value}_mean_away"]
    history_columns = [c for c in assembled if "history_max_date" in c]
    for column in history_columns:
        known = assembled[column].notna()
        if not assembled.loc[known, column].lt(assembled.loc[known, "date"]).all():
            raise AssertionError(f"Temporal leakage in {column}")
    known_injury = assembled.injury_snapshot_et.notna()
    if not assembled.loc[known_injury, "injury_snapshot_et"].lt(assembled.loc[known_injury, "tip_et"]).all():
        raise AssertionError("Post-tip or at-tip injury snapshot")
    complete = np.isfinite(assembled[FEATURE_NAMES].to_numpy(dtype=float)).all(axis=1)
    train_mask = assembled.date.between(TRAIN_START, TRAIN_END)
    test_mask = assembled.date.between(TEST_START, TEST_END)
    train = assembled.loc[train_mask & complete].copy()
    test = assembled.loc[test_mask].copy()
    if len(test) != 229 or test.duplicated(["date", "home", "away"]).any():
        raise AssertionError(f"The locked March benchmark must contain 229 distinct games, got {len(test)}")
    train = train[["game_id", "date", "home", "away", "home_won", "dec_home", "dec_away", "p_market_home"] + FEATURE_NAMES]
    test = test[["game_id", "date", "home", "away", "home_won", "dec_home", "dec_away", "p_market_home"] + FEATURE_NAMES]
    missing = {
        c: int(assembled.loc[train_mask, c].isna().sum())
        for c in FEATURE_NAMES if assembled.loc[train_mask, c].isna().any()
    }
    audit = assembled[["game_id", "date", "home", "away", "injury_snapshot_et", "tip_et", "injury_source_issue"] + history_columns].copy()
    audit["complete_features"] = complete
    audit["missing_features"] = assembled[FEATURE_NAMES].isna().apply(lambda r: ",".join(r.index[r]), axis=1)
    audit["opening_timestamp"] = np.where(
        assembled.open_home.notna(), "semantic open marker; publication timestamp absent", "missing"
    )
    audit.to_csv(HERE / "feature_audit.csv", index=False)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    manifest = {
        "input_fingerprints": signature, "n_train": len(train), "n_test": len(test),
        "n_features": len(FEATURE_NAMES), "train_date_min": train.date.min(), "train_date_max": train.date.max(),
        "training_candidates_with_closing_lines": int(train_mask.sum()),
        "training_dropped_incomplete": int((train_mask & ~complete).sum()),
        "training_missing_by_feature": missing,
        "unusable_injury_snapshots": audit.loc[audit.injury_source_issue.notna(), [
            "game_id", "date", "home", "away", "injury_source_issue"
        ]].to_dict("records"),
        "test_incomplete_games": audit.loc[test_mask & ~complete, ["game_id", "date", "home", "away", "missing_features"]].to_dict("records"),
        "test_fallback_policy": "Before evaluation, any incomplete March feature row is assigned Kaggle no-vig market probability for every ablation. Never remove a benchmark game or fabricate an opening quote.",
        "feature_groups": FEATURE_GROUPS, "odds_read": odds_stats, "boxscore_read": box_stats,
        "player_read": player_stats, "closing_line_agreement": close_agreement(diagnostics, games),
        "injury_snapshot_counts": assembled.injury_snapshot_et.str[-5:].value_counts().to_dict(),
        "timestamp_policy": {
            "statistics": "Completed games with source date strictly less than target date; rolling last 10, minimum 5, EWMA half-life 10.",
            "opening": "Only odds_date == open; no marker publication timestamp exists. Semantic historical opening, not exact point-in-time provable.",
            "closing": "Kaggle supplied closing moneylines are the sole current closing market input and settlement price; actual publication timestamps unavailable.",
            "movement": "Current game close/move/steam/update-count fields are never predictors. Historical versions use strictly previous game dates; timed updates restricted to that game's Eastern calendar day.",
            "injuries": "Latest complete official snapshot with Eastern timestamp strictly less than scheduled Eastern tip; 1–11 in the 12-hour cache are afternoon/evening.",
            "march": "Weights and selections freeze before March. Earlier completed March games update statistics, never model weights.",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"[features] {len(train)} complete training games, {len(test)} March games, "
          f"{len(FEATURE_NAMES)} features; {len(manifest['test_incomplete_games'])} disclosed market fallback(s)", flush=True)
    return {"train": train.reset_index(drop=True), "test": test.reset_index(drop=True), "manifest": manifest}
