import soccerdata as sd
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors

from datetime import datetime
from collections import defaultdict
from _commons import initPlotting, initFolders, flattenMultiCol, justifyText


# Functions
def isEasyMatch(team_row, mode, offStrong, offWeak, defStrong, defWeak):
    if mode == "DEF":
        return team_row["opponent"] in offWeak  # and team_row["team"] in defStrong
    else:  # mode == "OFF"
        return team_row["opponent"] in defWeak  # and team_row["team"] in offStrong


# Initialization
initPlotting()
outputFolder, dataFolder = initFolders(imageSubFolder="table")
CACHING_PATH = os.path.join(
    dataFolder, f"{datetime.today().strftime('%Y%m%d')}_sorareFixturePairing.pkl"
)
VISUAL_FILENAME = "sorareFixturePairings"

# Consts
PERCENTILE_THRESHOLD = 60  # 0-100
MIN_GOOD_GWS_NUMBER = 5
CURRENT_YEAR = 2025
CURRENT_SEASON = "2526"

fbref = sd.FBref(leagues="USA-Major League Soccer", seasons=2025)

if not os.path.exists(CACHING_PATH):
    df = fbref.read_schedule().reset_index()
    df.columns = flattenMultiCol(df.columns)
    df.to_pickle(CACHING_PATH)
else:
    df = pd.read_pickle(CACHING_PATH)

playedFrame = df.copy()
playedFrame = playedFrame[playedFrame["score"].notna()]

toBePlayedFrame = df.copy()
toBePlayedFrame = toBePlayedFrame[toBePlayedFrame["score"].isna()].reset_index(
    drop=True
)

# Find teams of current season
teams = sorted(pd.unique(toBePlayedFrame["home_team"].values))

# Build dictionary with scores for each team
playedFrame[["hg", "ag"]] = playedFrame["score"].str.split("–", expand=True)
playedFrame["hg"] = playedFrame["hg"].astype(int)
playedFrame["ag"] = playedFrame["ag"].astype(int)

# Check if there are enough matches to use current season data
currentSeasonPlayedFrame = playedFrame[playedFrame["season"] == CURRENT_SEASON]
if (len(currentSeasonPlayedFrame) / len(teams)) > 9:
    usedFrame = currentSeasonPlayedFrame.reset_index(drop=True)
else:
    print("Not enough fixtures")
    exit()

teamScores = defaultdict(dict)
promotedTeams = []

for team in teams:
    teamFrame = usedFrame.copy()

    hFrame = teamFrame[teamFrame["home_team"] == team]
    aFrame = teamFrame[teamFrame["away_team"] == team]

    if len(hFrame) + len(aFrame) < 9:
        promotedTeams.append(team)
        continue

    offScore = (hFrame["hg"].sum() + aFrame["ag"].sum()) / (len(hFrame) + len(aFrame))
    defScore = (hFrame["ag"].sum() + aFrame["hg"].sum()) / (len(hFrame) + len(aFrame))

    teamScores[team] = {
        "offScore": round(float(offScore), 2),
        "defScore": round(float(defScore), 2),
    }

# Fix promoted teams scores, setting 3rd worst value as their value
sortedOff = sorted(teamScores.items(), key=lambda x: x[1]["offScore"])
sortedDef = sorted(teamScores.items(), key=lambda x: x[1]["defScore"], reverse=True)
for team in promotedTeams:
    teamScores[team]["offScore"] = sortedOff[2][1]["offScore"]
    teamScores[team]["defScore"] = sortedDef[2][1]["defScore"]
teamScoresFrame = pd.DataFrame.from_dict(teamScores, orient="index").reset_index()
teamScoresFrame.columns = ["team", "offScore", "defScore"]


# Process matches to be played and find good matches
normalizedMatches = []

# Normalizes matches to be played with a standard format (team/opponent instead of home/away)
for _, row in toBePlayedFrame.iterrows():
    normalizedMatches.append(
        {
            "team": row["home_team"],
            "opponent": row["away_team"],
            "is_home": True,
        }
    )
    normalizedMatches.append(
        {
            "team": row["away_team"],
            "opponent": row["home_team"],
            "is_home": False,
        }
    )

normalizedFrame = pd.DataFrame(normalizedMatches)
normalizedFrame = normalizedFrame.merge(teamScoresFrame, on="team", how="left")
normalizedFrame = normalizedFrame.merge(
    teamScoresFrame, left_on="opponent", right_on="team", suffixes=("", "_opp")
)
normalizedFrame = normalizedFrame.drop(columns=["team_opp"])

offScores = [v["offScore"] for v in teamScores.values()]
defScores = [v["defScore"] for v in teamScores.values()]
offScores = sorted(offScores)
defScores = sorted(defScores)
offTargetThreshold = round(np.percentile(offScores, PERCENTILE_THRESHOLD), 2)
defTargetThreshold = round(np.percentile(defScores, 100 - PERCENTILE_THRESHOLD), 2)
offOppositionThreshold = round(np.percentile(offScores, 100 - PERCENTILE_THRESHOLD), 2)
defOppositionThreshold = round(np.percentile(defScores, PERCENTILE_THRESHOLD), 2)
offStrongTeams = teamScoresFrame[teamScoresFrame["offScore"] >= offTargetThreshold][
    "team"
].tolist()
offWeakTeams = teamScoresFrame[teamScoresFrame["offScore"] <= offOppositionThreshold][
    "team"
].tolist()
defStrongTeams = teamScoresFrame[teamScoresFrame["defScore"] <= defTargetThreshold][
    "team"
].tolist()
defWeakTeams = teamScoresFrame[teamScoresFrame["defScore"] >= defOppositionThreshold][
    "team"
].tolist()

print(defStrongTeams)
print(defWeakTeams)
print(offStrongTeams)
print(offWeakTeams)

# Note that every team is processed offensively even without explicitly iterating on the "OFF" target mode
maxBestPairings = 0
for targetMode in ["DEF", "OFF"]:
    easyFixturesDict = defaultdict(int)
    for targetTeam in teams:
        targetGames = normalizedFrame[normalizedFrame["team"] == targetTeam]
        pairingCounts = {}

        for _, tgtRow in targetGames.iterrows():
            if isEasyMatch(
                tgtRow,
                targetMode,
                offStrongTeams,
                offWeakTeams,
                defStrongTeams,
                defWeakTeams,
            ):
                easyFixturesDict[targetTeam] += 1

    sortedEasyFixturesTeams = sorted(
        easyFixturesDict.items(), key=lambda x: x[1], reverse=True
    )

    print("")
    print(targetMode)
    for pair in sortedEasyFixturesTeams:
        print(f"{pair[0]} -> {pair[1]}")
