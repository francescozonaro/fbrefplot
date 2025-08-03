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
        return team_row["team"] in defStrong and team_row["opponent"] in offWeak
    else:  # mode == "OFF"
        return team_row["team"] in offStrong and team_row["opponent"] in defWeak


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

fbref = sd.FBref(
    leagues="BEL-Belgian Pro League", seasons=[CURRENT_YEAR - 1, CURRENT_YEAR]
)

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
playedFrame = playedFrame[playedFrame["round"] == "Regular season"]
playedFrame["hg"] = playedFrame["hg"].astype(int)
playedFrame["ag"] = playedFrame["ag"].astype(int)

# Check if there are enough matches to use current season data
currentSeasonPlayedFrame = playedFrame[playedFrame["season"] == CURRENT_SEASON]
if (len(currentSeasonPlayedFrame) / len(teams)) > 9:
    usedFrame = currentSeasonPlayedFrame.reset_index(drop=True)
else:
    usedFrame = playedFrame.reset_index(drop=True)

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

# Note that every team is processed offensively even without explicitly iterating on the "OFF" target mode
filteredPairings = {}
maxBestPairings = 0
for targetTeam in teams:
    for targetMode in ["DEF"]:  # ["DEF", "OFF"]
        targetGames = normalizedFrame[normalizedFrame["team"] == targetTeam]
        pairingCounts = {}

        for _, tgtRow in targetGames.iterrows():
            week = tgtRow["week"]

            if not isEasyMatch(
                tgtRow,
                targetMode,
                offStrongTeams,
                offWeakTeams,
                defStrongTeams,
                defWeakTeams,
            ):
                continue

            sameWeekMatches = normalizedFrame[
                (normalizedFrame["week"] == week)
                & (normalizedFrame["team"] != targetTeam)
            ]

            for _, matchRow in sameWeekMatches.iterrows():
                oppositeTargetMode = "OFF" if targetMode == "DEF" else "DEF"
                if isEasyMatch(
                    matchRow,
                    oppositeTargetMode,
                    offStrongTeams,
                    offWeakTeams,
                    defStrongTeams,
                    defWeakTeams,
                ):
                    team = matchRow["team"]
                    pairingCounts[team] = pairingCounts.get(team, 0) + 1

        sortedPairings = sorted(pairingCounts.items(), key=lambda x: x[1], reverse=True)

        if sortedPairings:
            highestCount = sortedPairings[0][1]
            maxBestPairings = max(maxBestPairings, sortedPairings[0][1])
            if highestCount >= MIN_GOOD_GWS_NUMBER:
                filteredPairings[targetTeam] = pd.DataFrame(
                    sortedPairings, columns=["Team", "Count"]
                )


rows = int(round(len(filteredPairings) / 2))

fig, axs = plt.subplots(rows, 2, figsize=(12, 16), sharex=True, dpi=72)
fig.subplots_adjust(wspace=0.1, hspace=0.3)
fig.patch.set_facecolor("#eeeeee")

teamKeys = list(filteredPairings.keys())

for i, ax in enumerate(axs.flatten()):

    if i >= len(filteredPairings):
        fig.delaxes(ax)
        continue

    ax.tick_params(axis="x", labelbottom=True)
    dark_green = "#004d00"
    light_green = "#ffb703"

    # Create the colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "green_gradient", [light_green, dark_green]
    )

    plotFrame = filteredPairings.get(teamKeys[i])

    norm = plt.Normalize(vmin=1, vmax=maxBestPairings)
    colors = [cmap(norm(value)) for value in plotFrame["Count"]]

    bars = ax.barh(
        plotFrame["Team"],
        plotFrame["Count"],
        color=colors,
        edgecolor="grey",
        alpha=0.8,
    )

    ax.set_yticks([])
    for bar, team in zip(bars, plotFrame["Team"]):
        text = ax.text(
            x=0.2,  # A bit of padding from left edge
            y=bar.get_y() + bar.get_height() / 2,
            s=team,
            va="center",
            ha="left",
            fontweight="bold",
            fontsize=11,
            color="white",
        )

        text.set_path_effects(
            [
                path_effects.withStroke(linewidth=1.75, foreground="black"),
            ]
        )

    ax.axvline(0, color="black", lw=1)
    ax.set_title(
        f"{teamKeys[i]}",
        fontsize=13,
        color="#5A5A5A",
        fontweight="bold",
        pad=8,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.set_facecolor("#eeeeee")
    ax.invert_yaxis()

HORIZONTAL_ALIGN = 0.125

fig.text(
    HORIZONTAL_ALIGN,
    1.052,
    "Jupiler League best Sorare mini-stack pairings [25/26]",
    ha="left",
    va="bottom",
    fontsize=20,
    weight="bold",
    color="black",
)

subtitleText = "Each subplot represents a defensively solid Jupiler League team. Each bar represent the number of gameweeks where both the defensive team and the candidate strong attacking team have easy fixtures, respectively against offensively weak and defensively weak teams. Offensive and defensive strength is based on goals scored and conceded during the 2024-25 regular season. Data: FBRef | @francescozonaro"
charsPerLine = 110
justifiedText = justifyText(subtitleText, charsPerLine)

txt = fig.text(
    HORIZONTAL_ALIGN,
    1.03,
    justifiedText,
    size=10,
    color="#5A5A5A",
    va="top",
)
txt.set_linespacing(1.5)

# Build display text
category_text = (
    f"Defensively Strong: [{', '.join(defStrongTeams)}]\n"
    f"Defensively Weak: [{', '.join(defWeakTeams)}]\n"
    f"Offensively Strong: [{', '.join(offStrongTeams)}]\n"
    f"Offensively Weak: [{', '.join(offWeakTeams)}]"
)


legend = fig.text(
    HORIZONTAL_ALIGN,
    0.965,
    category_text,
    size=10,
    color="#5A5A5A",
    va="top",
    ha="left",
)
legend.set_linespacing(1.5)

plt.savefig(
    f"{outputFolder}/{VISUAL_FILENAME}.png",
    dpi=600,
    facecolor="#eeeeee",
    bbox_inches="tight",
    pad_inches=0.3,
    edgecolor="none",
    transparent=False,
)


# Offensive and defensive scores (and consequently teams) where computed automatically (best and worst 40%) based on last season data. I've no in depth knowledge regarding JPL, so it may be possible to get more accurate results with custom inputs.
