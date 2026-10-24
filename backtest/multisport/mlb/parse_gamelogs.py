"""Parse Retrosheet game logs 2021-2022 into a clean games table."""
import pandas as pd
import numpy as np

COLS = [
    "Date", "DoubleHeader", "DayOfWeek",
    "VisitingTeam", "VisitingTeamLeague", "VisitingTeamGameNumber",
    "HomeTeam", "HomeTeamLeague", "HomeTeamGameNumber",
    "VisitorRunsScored", "HomeRunsScore", "LengthInOuts", "DayNight",
    "CompletionInformation", "ForfeitInformation", "ProtestInformation",
    "ParkID", "Attendance", "TimeOfGame",
    "VisitorLineScore", "HomeLineScore",
    "VisitorAtBats", "VisitorHits", "VisitorDoubles", "VisitorTriples",
    "VisitorHomeruns", "VisitorRBI", "VisitorSacrificeHits",
    "VisitorSacrificeFlies", "VisitorHitByPitch", "VisitorWalks",
    "VisitorIntentionalWalks", "VisitorStrikeouts", "VisitorStolenBases",
    "VisitorCaughtStealing", "VisitorGroundedIntoDoublePlays",
    "VisitorCatchersInterference", "VisitorLeftOnBase",
    "VisitorPitchersUsed", "VisitorIndividualEarnedRuns",
    "VisitorTeamEarnedRuns", "VisitorWildPitches", "VisitorBalks",
    "VisitorPutouts", "VisitorAssists", "VisitorErrors", "VisitorPassedBalls",
    "VisitorDoublePlays", "VisitorTriplePlays",
    "HomeAtBats", "HomeHits", "HomeDoubles", "HomeTriples", "HomeHomeruns",
    "HomeRBI", "HomeSacrificeHits", "HomeSacrificeFlies", "HomeHitByPitch",
    "HomeWalks", "HomeIntentionalWalks", "HomeStrikeouts", "HomeStolenBases",
    "HomeCaughtStealing", "HomeGroundedIntoDoublePlays",
    "HomeCatchersInterference", "HomeLeftOnBase",
    "HomePitchersUsed", "HomeIndividualEarnedRuns", "HomeTeamEarnedRuns",
    "HomeWildPitches", "HomeBalks",
    "HomePutouts", "HomeAssists", "HomeErrors", "HomePassedBalls",
    "HomeDoublePlays", "HomeTriplePlays",
    "HomePlateUmpireID", "HomePlateUmpireName",
    "FirstBaseUmpireID", "FirstBaseUmpireName",
    "SecondBaseUmpireID", "SecondBaseUmpireName",
    "ThirdBaseUmpireID", "ThirdBaseUmpireName",
    "LeftFieldUmpireID", "LeftFieldUmpireName",
    "RightFieldUmpireID", "RightFieldUmpireName",
    "VisitingTeamManagerID", "VisitingTeamManagerName",
    "HomeTeamManagerID", "HomeTeamManagerName",
    "WinningPitcherID", "WinningPitcherName",
    "LosingPitcherID", "LosingPitcherName",
    "SavingPitcherID", "SavingPitcherName",
    "GameWinningRBIBatterID", "GameWinningRBIBatterName",
    "VisitingStartingPitcherID", "VisitingStartingPitcherName",
    "HomeStartingPitcherID", "HomeStartingPitcherName",
]
# batting orders: 9 batters x (ID, Name, Position) each side
for side in ["Visiting", "Home"]:
    for i in range(1, 10):
        COLS += [f"{side}Batter{i}ID", f"{side}Batter{i}Name", f"{side}Batter{i}Position"]
COLS += ["AdditionalInformation", "AcquisitionInformation"]
assert len(COLS) == 161, len(COLS)

NUMERIC = [
    "VisitingTeamGameNumber", "HomeTeamGameNumber",
    "VisitorRunsScored", "HomeRunsScore", "LengthInOuts", "Attendance", "TimeOfGame",
    "VisitorAtBats", "VisitorHits", "VisitorDoubles", "VisitorTriples",
    "VisitorHomeruns", "VisitorRBI", "VisitorSacrificeHits",
    "VisitorSacrificeFlies", "VisitorHitByPitch", "VisitorWalks",
    "VisitorIntentionalWalks", "VisitorStrikeouts", "VisitorStolenBases",
    "VisitorCaughtStealing", "VisitorGroundedIntoDoublePlays",
    "VisitorCatchersInterference", "VisitorLeftOnBase",
    "VisitorPitchersUsed", "VisitorIndividualEarnedRuns",
    "VisitorTeamEarnedRuns", "VisitorWildPitches", "VisitorBalks",
    "VisitorPutouts", "VisitorAssists", "VisitorErrors", "VisitorPassedBalls",
    "VisitorDoublePlays", "VisitorTriplePlays",
    "HomeAtBats", "HomeHits", "HomeDoubles", "HomeTriples", "HomeHomeruns",
    "HomeRBI", "HomeSacrificeHits", "HomeSacrificeFlies", "HomeHitByPitch",
    "HomeWalks", "HomeIntentionalWalks", "HomeStrikeouts", "HomeStolenBases",
    "HomeCaughtStealing", "HomeGroundedIntoDoublePlays",
    "HomeCatchersInterference", "HomeLeftOnBase",
    "HomePitchersUsed", "HomeIndividualEarnedRuns", "HomeTeamEarnedRuns",
    "HomeWildPitches", "HomeBalks",
    "HomePutouts", "HomeAssists", "HomeErrors", "HomePassedBalls",
    "HomeDoublePlays", "HomeTriplePlays",
    "DoubleHeader",
]


def load(year):
    df = pd.read_csv(f"gl{year}.txt", header=None, names=COLS, dtype=str)
    return df


frames = []
for year in (2021, 2022):
    df = load(year)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["season"] = year
    frames.append(df)

g = pd.concat(frames, ignore_index=True)
g["date"] = pd.to_datetime(g["Date"], format="%Y%m%d")
g = g.sort_values(["date", "DoubleHeader", "ParkID", "TimeOfGame"], kind="stable").reset_index(drop=True)

# drop games not fully acquired or forfeits (rare)
g = g[g["AcquisitionInformation"] == "Y"].copy()
g = g[g["ForfeitInformation"].isna()].copy()

# keep regular season only for 2021-2022 (game logs contain all games incl. playoffs);
# regular season = teams within the 30-team set and before playoff start.
g["home_win"] = (g["HomeRunsScore"] > g["VisitorRunsScored"]).astype(int)

keep = [
    "season", "date", "DoubleHeader", "VisitingTeam", "HomeTeam", "ParkID",
    "VisitorRunsScored", "HomeRunsScore", "LengthInOuts", "DayNight",
    "VisitorLineScore", "HomeLineScore", "Attendance",
    "VisitingStartingPitcherID", "VisitingStartingPitcherName",
    "HomeStartingPitcherID", "HomeStartingPitcherName",
    "WinningPitcherID", "WinningPitcherName",
    "LosingPitcherID", "LosingPitcherName",
    "VisitingTeamGameNumber", "HomeTeamGameNumber",
    "VisitorPitchersUsed", "HomePitchersUsed",
    "VisitorStrikeouts", "VisitorWalks", "VisitorHits", "VisitorAtBats",
    "VisitorDoubles", "VisitorTriples", "VisitorHomeruns", "VisitorHitByPitch",
    "VisitorSacrificeFlies",
    "HomeStrikeouts", "HomeWalks", "HomeHits", "HomeAtBats",
    "HomeDoubles", "HomeTriples", "HomeHomeruns", "HomeHitByPitch",
    "HomeSacrificeFlies",
    "home_win",
]
out = g[keep].reset_index(drop=True)
out.to_parquet("games_all.parquet")
print(out.shape)
print(out.groupby("season").agg(n=("date", "size"), first=("date", "min"), last=("date", "max")))
print(out.head(3).to_string())
print("home win rate:", out.home_win.mean())
print("n teams:", len(set(out.VisitingTeam) | set(out.HomeTeam)))
