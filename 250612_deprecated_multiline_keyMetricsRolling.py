import pandas as pd
import matplotlib.pyplot as plt
import soccerdata as sd
import os
import numpy as np
import urllib.request
import matplotlib.patheffects as path_effects
import datetime

from PIL import Image
from _commons import initPlotting, initFolders, flattenMultiCol, justifyText

# Initialization
initPlotting()
outputFolder, dataFolder = initFolders(imageSubFolder="multiline")

# Data
os.makedirs(dataFolder, exist_ok=True)
fbref = sd.FBref(leagues="ENG-Premier League", seasons=range(2017, 2025))

SALTY_SALT = datetime.date.today().strftime("%Y%m%d")
TEAM_NAME = "Aston Villa"
OUTPUT_NAME = f"{SALTY_SALT}_{TEAM_NAME.replace(' ', '')}_rollingPerformances"
TITLE_TEXT = "Aston Villa's rise through the years"
SUBTITLE_TEXT = "A look at Joan García's 2024-25 season at Espanyol, consistently overperforming his Post-Shot Expected Goals (PSxG) faced, as shown by the rolling gap (10-game window) between goals conceded and PSxG."
CHARS_PER_LINE = 80
DATA_CACHING_PATH = os.path.join(dataFolder, f"{SALTY_SALT}_{TEAM_NAME}_data.pkl")
ROLLING_WINDOW = 38

if not os.path.exists(DATA_CACHING_PATH):
    df = fbref.read_schedule().reset_index()
    df.columns = flattenMultiCol(df.columns)

    df = df[(df["home_team"] == TEAM_NAME) | (df["away_team"] == TEAM_NAME)]
    df[["home_score", "away_score"]] = (
        df["score"]
        .str.replace("[–—−]", "-", regex=True)
        .str.split("-", expand=True)
        .astype(int)
    )
    df = df.reset_index(drop=True)

    gls = []
    glsAg = []
    xg = []
    xgAg = []
    points = []
    seasons = []

    for index, row in df.iterrows():
        match_id = row.game_id
        gls.append(row.home_score if row.home_team == TEAM_NAME else row.away_score)
        glsAg.append(row.away_score if row.home_team == TEAM_NAME else row.home_score)
        xg.append(row.home_xg if row.home_team == TEAM_NAME else row.away_xg)
        xgAg.append(row.away_xg if row.home_team == TEAM_NAME else row.home_xg)
        points.append(
            3
            if (row.home_score > row.away_score and row.home_team == TEAM_NAME)
            or (row.away_score > row.home_score and row.away_team == TEAM_NAME)
            else 1 if row.home_score == row.away_score else 0
        )
        seasons.append(row.season)

    data = {
        "goals": gls,
        "goals_against": glsAg,
        "xG": xg,
        "xG_against": xgAg,
        "points": points,
        "season": seasons,
    }

    ff = pd.DataFrame(data)
    ff.to_pickle(DATA_CACHING_PATH)
else:
    ff = pd.read_pickle(DATA_CACHING_PATH)

tdf = ff.copy()
aff = ff.copy()
compare_metrics = ["points", "goals", "xG", "goals_against", "xG_against"]
ff[compare_metrics] = (
    ff[compare_metrics].rolling(window=ROLLING_WINDOW, min_periods=10).mean()
)
aff[compare_metrics] = aff[compare_metrics].expanding(min_periods=10).mean()


betterMetricNames = {
    "points": "Points",
    "goals": "Goals Scored",
    "xG": "xG",
    "goals_against": "Goals Conceded",
    "xG_against": "xGA",
}
colors = {
    "points": "#D382DD",
    "goals": "#57b3ec",
    "goals_against": "#f46161",
    "xG": "#2a9d8f",
    "xG_against": "#f19823",
}

# Subplots
fig, axs = plt.subplots(
    nrows=len(compare_metrics), ncols=1, figsize=(10, 14), sharex=True, dpi=600
)
fig.subplots_adjust(hspace=0.4)
fig.patch.set_facecolor("#eeeeee")

x = range(1, len(ff) + 1)

for ax, metric in zip(axs, compare_metrics):
    ax.set_facecolor("#eeeeee")
    ax.grid(visible=True, ls="--", color="lightgrey")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.plot(
        x,
        ff[metric],
        label=betterMetricNames.get(metric),
        color="#CCCCCC",
        linewidth=2.5,
    )
    ax.plot(
        x,
        aff[metric],
        label=betterMetricNames.get(metric),
        color=colors[metric],
        linewidth=1.5,
    )

    # mean_value = ff[metric].mean()
    meanValue = tdf[metric].expanding(min_periods=10).mean().iloc[-1]
    ax.axhline(
        meanValue,
        linestyle="--",
        color=colors[metric],
        linewidth=1,
        alpha=0.6,
        label="Mean",
    )

    ax.set_ylabel(f"{betterMetricNames.get(metric)}", labelpad=10)
    # ax.set_yticks(np.arange(0.0, 2.01, 0.5))
    ax.legend(loc="upper right", fontsize="small", frameon=False)

print(ff)
seasonChangeIndices = ff[ff["season"] != ff["season"].shift()].index.tolist()
tick_positions = []
tick_labels = []

for x in seasonChangeIndices:
    if x != 0:
        tick_positions.append(x)
        label = f"Start of\n20{ff.iloc[x].season[:2]}/{ff.iloc[x].season[2:]}"
        tick_labels.append(label)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, ha="center", fontsize=7, weight="bold")

# axs[-1].set_xlabel("Gameday", labelpad=10)
# axs[-1].set_xticks(range(0, len(ff) + 1, 10))

fig.text(
    0.13,
    0.94,
    TITLE_TEXT,
    ha="left",
    va="bottom",
    fontsize=15,
    weight="bold",
    color="black",
)

txt = fig.text(
    0.13,
    0.9,
    justifyText(SUBTITLE_TEXT, CHARS_PER_LINE),
    ha="left",
    va="bottom",
    fontsize=9,
    color="#5A5A5A",
)
txt.set_linespacing(1.5)

fig.text(
    1,
    0.1,
    "@francescozonaro | Data from FBRef",
    transform=axs[-3].transAxes,
    ha="right",
    va="top",
    fontsize=8,
    color="#5A5A5A",
    family="Monospace",
)

league_logo = "https://images.fotmob.com/image_resources/logo/teamlogo/10252.png"
league_icon = Image.open(urllib.request.urlopen(league_logo)).convert("LA")
logo_ax = fig.add_axes([0.825, 0.895, 0.075, 0.075], anchor="C")
logo_ax.imshow(league_icon)
logo_ax.axis("off")

plt.savefig(
    f"{outputFolder}/{OUTPUT_NAME}",
    dpi=600,
    facecolor="#eeeeee",
    bbox_inches="tight",
    pad_inches=0.3,
    edgecolor="none",
    transparent=False,
)
