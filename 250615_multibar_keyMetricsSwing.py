import pandas as pd
import matplotlib.pyplot as plt
import soccerdata as sd
import os
import numpy as np
from PIL import Image
import urllib.request
import numpy as np
import textwrap
from sklearn.preprocessing import StandardScaler


from collections import defaultdict
from _commons import initPlotting, initFolders, flattenMultiCol, justifyText

# Initialization
initPlotting()
outputFolder, dataFolder = initFolders(imageSubFolder="multibar")

# Constants
TEAM_NAME = "Tottenham"
LINEUP_TEAM_NAME = "Tottenham Hotspur"
# TEAM_NAME = "Manchester Utd"
# LINEUP_TEAM_NAME = "Manchester United"
# TEAM_NAME = "Chelsea"
# LINEUP_TEAM_NAME = "Chelsea"
CACHING_PATH = os.path.join(dataFolder, f"caching.pkl")
VISUAL_FILENAME = "250614_totPlayerDif"

# Fbref
fbref = sd.FBref(leagues="ENG-Premier League", seasons=2024)

if not os.path.exists(CACHING_PATH):
    df = fbref.read_schedule().reset_index()
    df = flattenMultiCol(df)

    df = df[(df["home_team"] == TEAM_NAME) | (df["away_team"] == TEAM_NAME)]
    df[["home_score", "away_score"]] = (
        df["score"]
        .str.replace("[–—−]", "-", regex=True)
        .str.split("-", expand=True)
        .astype(int)
    )
    df = df.reset_index(drop=True)

    playerData = defaultdict(
        lambda: {
            "matches": 0,
            "gA": 0,
            "xGA": 0.0,
            "gS": 0,
            "xG": 0.0,
            "matches_not": 0,
            "gA_not": 0,
            "xGA_not": 0.0,
            "gS_not": 0,
            "xG_not": 0.0,
        }
    )

    lineupDict = {}
    allPlayersOnTeam = set()
    for index, row in df.iterrows():
        lineup = fbref.read_lineup(match_id=row.game_id)
        if index == 0:
            print(lineup)
        teamPlayers = lineup[
            (lineup["team"] == LINEUP_TEAM_NAME) & (lineup["position"] != "GK")
        ]["player"].unique()
        allPlayersOnTeam.update(teamPlayers)
        lineupDict[row.game_id] = lineup

    for index, row in df.iterrows():
        rowGA = row.away_score if row.home_team == TEAM_NAME else row.home_score
        rowxGA = row.away_xg if row.home_team == TEAM_NAME else row.home_xg
        rowGS = row.home_score if row.home_team == TEAM_NAME else row.away_score
        rowxG = row.home_xg if row.home_team == TEAM_NAME else row.away_xg

        lineup = lineupDict[row.game_id]
        starters = lineup[
            (lineup["team"] == LINEUP_TEAM_NAME)
            & (lineup["position"] != "GK")
            & (lineup["is_starter"] == True)
        ]
        startersSet = set(starters["player"])

        for _, player_row in starters.iterrows():
            pName = player_row["player"]
            playerData[pName]["matches"] += 1
            playerData[pName]["gA"] += rowGA
            playerData[pName]["xGA"] += rowxGA
            playerData[pName]["gS"] += rowGS
            playerData[pName]["xG"] += rowxG

        for pName in allPlayersOnTeam:
            if pName in startersSet:
                playerData[pName]["matches"] += 1
                playerData[pName]["gA"] += rowGA
                playerData[pName]["xGA"] += rowxGA
                playerData[pName]["gS"] += rowGS
                playerData[pName]["xG"] += rowxG
            else:
                playerData[pName]["matches_not"] += 1
                playerData[pName]["gA_not"] += rowGA
                playerData[pName]["xGA_not"] += rowxGA
                playerData[pName]["gS_not"] += rowGS
                playerData[pName]["xG_not"] += rowxG

    pdf = pd.DataFrame.from_dict(playerData, orient="index")
    pdf.index.name = "player"
    pdf.reset_index(inplace=True)
    pdf.to_pickle(CACHING_PATH)

else:
    pdf = pd.read_pickle(CACHING_PATH)


pdf = pdf[pdf["matches"] >= 10]
pdf["ga90"] = pdf["gA"] / pdf["matches"]
pdf["xga90"] = pdf["xGA"] / pdf["matches"]
pdf["gs90"] = pdf["gS"] / pdf["matches"]
pdf["xg90"] = pdf["xG"] / pdf["matches"]
pdf["ga90_not"] = pdf["gA_not"] / pdf["matches_not"]
pdf["xga90_not"] = pdf["xGA_not"] / pdf["matches_not"]
pdf["gs90_not"] = pdf["gS_not"] / pdf["matches_not"]
pdf["xg90_not"] = pdf["xG_not"] / pdf["matches_not"]

# Metrics to plot
metrics = ["gs90", "xg90", "ga90", "xga90"]
betterMetricNames = {
    "gs90": "Goals scored",
    "xg90": "xG",
    "ga90": "Goals conceded",
    "xga90": "xGA",
}

fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex=True, dpi=72)
fig.subplots_adjust(wspace=0.5, hspace=0.2)
fig.patch.set_facecolor("#eeeeee")

globalMinX = 999
globalMaxX = 0

for i, ax in enumerate(axs.flatten()):
    ax.tick_params(axis="x", labelbottom=True)
    metric = metrics[i]
    better_name = betterMetricNames.get(metric)

    diff = pdf[metric] - pdf[f"{metric}_not"]

    if metric in ["gs90", "xg90"]:
        sorted_indices = np.argsort(diff)[::-1]
        cmap = plt.cm.RdYlGn
    else:
        sorted_indices = np.argsort(diff)
        cmap = plt.cm.RdYlGn_r

    sorted_diff = diff.iloc[sorted_indices]
    sorted_players = pdf["player"].iloc[sorted_indices]

    norm = plt.Normalize(vmin=min(diff), vmax=max(diff))
    colors = [cmap(norm(value)) for value in diff]
    sorted_colors = [colors[idx] for idx in sorted_indices]

    ax.barh(
        sorted_players,
        sorted_diff,
        color=sorted_colors,
        edgecolor="grey",
        alpha=0.8,
    )
    ax.axvline(0, color="black", lw=1)
    ax.set_title(
        f"{better_name} per 90 (Starting - Not Starting)",
        fontsize=10,
        pad=8,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.set_facecolor("#eeeeee")
    ax.invert_yaxis()

pdf["diff_gs90"] = pdf["gs90"] - pdf["gs90_not"]
pdf["diff_xg90"] = pdf["xg90"] - pdf["xg90_not"]
pdf["diff_ga90"] = pdf["ga90_not"] - pdf["ga90"]
pdf["diff_xga90"] = pdf["xga90_not"] - pdf["xga90"]

scaler = StandardScaler()
pdf[
    [
        "norm_diff_gs90",
        "norm_diff_xg90",
        "norm_diff_ga90",
        "norm_diff_xga90",
    ]
] = scaler.fit_transform(
    pdf[
        [
            "diff_gs90",
            "diff_xg90",
            "diff_ga90",
            "diff_xga90",
        ]
    ]
)

pdf["total_diff"] = (
    pdf["norm_diff_gs90"]
    + pdf["norm_diff_xg90"]
    + pdf["norm_diff_ga90"]
    + pdf["norm_diff_xga90"]
)

# # Sum all diffs for an overall score
# pdf["total_diff"] = (
#     pdf["diff_gs90"] + pdf["diff_xg90"] + pdf["diff_ga90"] + pdf["diff_xga90"]
# )

# Sort players by total_diff ascending to find the worst overall (lowest)
overall = pdf.sort_values("total_diff")[["player", "total_diff", "matches_not"]]
print(overall)

for ax in axs.flatten():
    yticks = ax.get_yticklabels()
    for label in yticks:
        if label.get_text() in list(overall.tail(4)["player"]):
            facecolor = "#0D9B3C"
            label.set_bbox(
                dict(
                    facecolor=facecolor,
                    alpha=0.5,
                    edgecolor="none",
                    boxstyle="round,pad=0.15",
                )
            )
        elif label.get_text() in list(overall.head(4)["player"]):
            facecolor = "#dd4019"
            label.set_bbox(
                dict(
                    facecolor=facecolor,
                    alpha=0.5,
                    edgecolor="none",
                    boxstyle="round,pad=0.15",
                )
            )

fig.text(
    0.01,
    1.0,
    "Who made the difference? Tottenham's key players (24/25)",
    ha="left",
    va="bottom",
    fontsize=20,
    weight="bold",
    color="black",
)

subtitleText = "Difference in performance key metrics per 90 minutes when a player is starting versus not starting for Tottenham in 2024-25. These plots analyze changes in Goals Scored, Expected Goals (xG), Goals Conceded, and Expected Goals Against (xGA) per 90 minutes, revealing individual player impact on team dynamics. Highlighted players indicate the top and bottom four performers, based on their aggregated total normalized difference across all four metrics. Data from FBRef | @francescozonaro"
charsPerLine = 125
justifiedText = justifyText(subtitleText, charsPerLine)
# wrapped_subtitle_text = textwrap.fill(long_subtitle_text, width=max_chars_per_line)

txt = fig.text(
    0.01,
    0.925,
    justifiedText,
    size=10,
    color="#5A5A5A",
    va="bottom",
)
txt.set_linespacing(1.5)

league_logo = "https://images.fotmob.com/image_resources/logo/leaguelogo/47.png"
league_icon = Image.open(urllib.request.urlopen(league_logo)).convert("LA")
logo_ax = fig.add_axes([0.8, 1, 0.025, 0.025], anchor="C")
logo_ax.imshow(league_icon)
logo_ax.axis("off")

team_logo_url = "https://images.fotmob.com/image_resources/logo/teamlogo/8586.png"
team_logo_image = Image.open(urllib.request.urlopen(team_logo_url)).convert("LA")
team_logo_axes = fig.add_axes([0.83, 1, 0.025, 0.025], anchor="C")
team_logo_axes.imshow(team_logo_image)
team_logo_axes.axis("off")

plt.savefig(
    f"{outputFolder}/{VISUAL_FILENAME}.png",
    dpi=600,
    facecolor="#eeeeee",
    bbox_inches="tight",
    pad_inches=0.3,
    edgecolor="none",
    transparent=False,
)
