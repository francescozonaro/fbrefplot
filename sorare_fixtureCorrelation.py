import soccerdata as sd
import numpy as np
import pandas as pd
import os


from _commons import initPlotting, initFolders, flattenMultiCol


def isGoodMatch(team_row, mode, offT, offO, defT, defO):
    if mode == "DEF":
        return team_row["defScore"] <= defT and team_row["offScore_opp"] <= offO
    else:  # mode == "OFF"
        return team_row["offScore"] >= offT and team_row["defScore_opp"] >= defO


# Initialization
initPlotting()
outputFolder, dataFolder = initFolders(imageSubFolder="table")
CACHING_PATH = os.path.join(dataFolder, f"jpn24to25.pkl")

# Consts
CURRENT_SEASON = "2526"
PERCENTILE_THRESHOLD = 60
MIN_GOOD_GWS_NUMBER = 4

# Fbref
fbref = sd.FBref(leagues="JPN-JLeague", seasons=[2024, 2025])

if not os.path.exists(CACHING_PATH):
    df = fbref.read_schedule().reset_index()
    df.columns = flattenMultiCol(df.columns)
    df.to_pickle(CACHING_PATH)
else:
    df = pd.read_pickle(CACHING_PATH)

playedFrame = df.copy()
playedFrame = playedFrame[playedFrame["score"].notna()]
playedFrame[["hg", "ag"]] = playedFrame["score"].str.split("–", expand=True)
playedFrame["hg"] = playedFrame["hg"].astype(int)
playedFrame["ag"] = playedFrame["ag"].astype(int)

currentSeasonPlayedFrame = playedFrame[playedFrame["season"] == CURRENT_SEASON]
teams = sorted(pd.unique(currentSeasonPlayedFrame["home_team"].values))

if (len(currentSeasonPlayedFrame) / len(teams)) > 9:
    usedFrame = currentSeasonPlayedFrame.reset_index(drop=True)
else:
    usedFrame = playedFrame.reset_index(drop=True)

teamScores = {}
promotedTeams = []

for team in teams:
    teamFrame = usedFrame.copy()

    if len(teamFrame) < 9:
        promotedTeams.append(team)
        continue

    hFrame = teamFrame[teamFrame["home_team"] == team]
    aFrame = teamFrame[teamFrame["away_team"] == team]

    offScore = (hFrame["hg"].sum() + aFrame["ag"].sum()) / (len(hFrame) + len(aFrame))
    defScore = (hFrame["ag"].sum() + aFrame["hg"].sum()) / (len(hFrame) + len(aFrame))

    teamScores[team] = {
        "offScore": round(float(offScore), 2),
        "defScore": round(float(defScore), 2),
    }

# Fix promoted teams scores
sortedOff = sorted(teamScores.items(), key=lambda x: x[1]["offScore"])
sortedDef = sorted(teamScores.items(), key=lambda x: x[1]["defScore"], reverse=True)
for team in promotedTeams:
    teamScores[team]["offScore"] = sortedOff[2][1]["offScore"]
    teamScores[team]["defScore"] = sortedDef[2][1]["defScore"]


toBePlayedFrame = df.copy()
toBePlayedFrame = toBePlayedFrame[toBePlayedFrame["score"].isna()].reset_index(
    drop=True
)
normalizedMatches = []

for _, row in toBePlayedFrame.iterrows():
    normalizedMatches.append(
        {
            "week": row["week"],
            "team": row["home_team"],
            "opponent": row["away_team"],
            "is_home": True,
        }
    )
    normalizedMatches.append(
        {
            "week": row["week"],
            "team": row["away_team"],
            "opponent": row["home_team"],
            "is_home": False,
        }
    )

normalizedFrame = pd.DataFrame(normalizedMatches)
teamScoreDf = pd.DataFrame.from_dict(teamScores, orient="index").reset_index()
teamScoreDf.columns = ["team", "offScore", "defScore"]
normalizedFrame = normalizedFrame.merge(teamScoreDf, on="team", how="left")
normalizedFrame = normalizedFrame.merge(
    teamScoreDf, left_on="opponent", right_on="team", suffixes=("", "_opp")
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

print(offScores, "*********", offTargetThreshold, offOppositionThreshold)
print(defScores, "*********", defTargetThreshold, defOppositionThreshold)

for targetTeam in teams:
    for targetMode in ["OFF", "DEF"]:
        targetGames = normalizedFrame[normalizedFrame["team"] == targetTeam]
        pairingCounts = {}

        for _, tgtRow in targetGames.iterrows():
            week = tgtRow["week"]

            if not isGoodMatch(
                tgtRow,
                targetMode,
                offTargetThreshold,
                offOppositionThreshold,
                defTargetThreshold,
                defOppositionThreshold,
            ):
                continue

            sameWeekMatches = normalizedFrame[
                (normalizedFrame["week"] == week)
                & (normalizedFrame["team"] != targetTeam)
            ]

            for _, matchRow in sameWeekMatches.iterrows():
                oppositeTargetMode = "OFF" if targetMode == "DEF" else "DEF"
                if isGoodMatch(
                    matchRow,
                    oppositeTargetMode,
                    offTargetThreshold,
                    offOppositionThreshold,
                    defTargetThreshold,
                    defOppositionThreshold,
                ):
                    team = matchRow["team"]
                    pairingCounts[team] = pairingCounts.get(team, 0) + 1

        sortedPairings = sorted(pairingCounts.items(), key=lambda x: x[1], reverse=True)

        if sortedPairings:
            highestCount = sortedPairings[0][1]
            if highestCount >= MIN_GOOD_GWS_NUMBER:
                print("")
                print(targetTeam, targetMode)
                for team, count in sortedPairings:
                    if count >= MIN_GOOD_GWS_NUMBER:
                        print(f"{team}: {count}")
