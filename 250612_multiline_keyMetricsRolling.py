import pandas as pd
import matplotlib.pyplot as plt
import soccerdata as sd
import os
import math
import numpy as np
import urllib.request
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.patheffects as path_effects

from PIL import Image
from scipy.stats import poisson

# # Initialization
plt.rcParams["font.family"] = "Monospace"
outputFolder = os.path.join("imgs/", str(f"multilineplot"))
os.makedirs(outputFolder, exist_ok=True)

# Data
dataFolderName = "fbrefData"
os.makedirs(dataFolderName, exist_ok=True)
picklePath = os.path.join(dataFolderName, f"20250610.pkl")
fbref = sd.FBref(leagues="ENG-Premier League", seasons=[2023, 2024])

TEAM_NAME = "Tottenham"

if not os.path.exists(picklePath):
    df = fbref.read_schedule()
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip("_") for col in df.columns.values]

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

    data = {
        "goals": gls,
        "goals_against": glsAg,
        "xG": xg,
        "xG_against": xgAg,
        "points": points,
    }

    ff = pd.DataFrame(data)
    ff.to_pickle(picklePath)
else:
    ff = pd.read_pickle(picklePath)

ff = ff.rolling(window=10, min_periods=10).mean()

compare_metrics = ["points", "goals", "xG", "goals_against", "xG_against"]
# compare_metrics = ["points", "goals", "goals_against"]
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
    nrows=len(compare_metrics), ncols=1, figsize=(10, 14), sharex=True, dpi=72
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
        color=colors[metric],
        linewidth=2.5,
    )

    mean_value = ff[metric].mean()
    ax.axhline(
        mean_value,
        linestyle="--",
        color=colors[metric],
        linewidth=1,
        alpha=0.6,
        label="Mean",
    )

    ax.set_ylabel(f"{betterMetricNames.get(metric)}", labelpad=10)
    ax.set_yticks(np.arange(0.0, 3.00, 0.5))
    ax.legend(loc="upper left", fontsize="small", frameon=False)

    ax.plot(
        [38, 38],
        [ax.get_ylim()[0], ax.get_ylim()[1]],
        color="black",
        alpha=0.35,
        zorder=2,
        ls="--",
        lw=2,
    )

text_ = axs[0].annotate(
    xy=(38, 2.3),
    text="Start of\n2024/25",
    color="black",
    size=9,
    va="center",
    ha="center",
    weight="bold",
    zorder=4,
)
text_.set_path_effects(
    [
        path_effects.Stroke(linewidth=1.5, foreground="white"),
        path_effects.Normal(),
    ]
)

axs[-1].set_xlabel("Gameday", labelpad=10)
axs[-1].set_xticks(range(0, len(ff) + 1, 5))

fig.text(
    0.13,
    0.94,
    "Tottenham's EPL performances under Ange (2023 to 2025)",
    ha="left",
    va="bottom",
    fontsize=15,
    weight="bold",
    color="black",
)

txt = fig.text(
    0.13,
    0.9,
    "The 10-game rolling average of Spurs' EPL form under Ange — a story that started with\npromise, slipped into struggle, and somehow still ended with silver in hand.",
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

league_logo = "https://images.fotmob.com/image_resources/logo/teamlogo/8586.png"
league_icon = Image.open(urllib.request.urlopen(league_logo)).convert("LA")
logo_ax = fig.add_axes([0.825, 0.895, 0.075, 0.075], anchor="C")
logo_ax.imshow(league_icon)
logo_ax.axis("off")

plt.savefig(
    f"{outputFolder}/20250610_{TEAM_NAME}.png",
    dpi=600,
    facecolor="#eeeeee",
    bbox_inches="tight",
    pad_inches=0.3,
    edgecolor="none",
    transparent=False,
)
