import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import soccerdata as sd
import matplotlib.colors as mcolors
import os
import urllib.request
import numpy as np
import datetime

from PIL import Image
from _commons import (
    addTitleSubAndLogo,
    initPlotting,
    initFolders,
    flattenMultiCol,
    justifyText,
)


# Initialization
initPlotting()
outputFolder, dataFolder = initFolders(imageSubFolder="line")

TEAM_NAME = "Espanyol"
PLAYER_NAME = "Joan García"
SALTY_SALT = datetime.date.today().strftime("%Y%m%d")
OUTPUT_NAME = f"{SALTY_SALT}_{PLAYER_NAME.replace(' ', '')}PSXG"
TITLE_TEXT = "Joan García | PSxG minus goals against rolling gap"
SUBTITLE_TEXT = "A look at Joan García's 2024-25 season at Espanyol, consistently overperforming his Post-Shot Expected Goals (PSxG) faced, as shown by the rolling gap (10-game window) between goals conceded and PSxG."
CHARS_PER_LINE = 80
DATA_CACHING_PATH = os.path.join(dataFolder, f"{SALTY_SALT}_data.pkl")
ROLLING_WINDOW = 10

fbref = sd.FBref(leagues="ESP-La Liga", seasons=2024)

df = []

if not os.path.exists(DATA_CACHING_PATH):
    sdf = fbref.read_schedule().reset_index()
    sdf.columns = flattenMultiCol(sdf.columns)

    sdf = sdf[(sdf["home_team"] == TEAM_NAME) | (sdf["away_team"] == TEAM_NAME)]
    sdf[["home_score", "away_score"]] = (
        sdf["score"]
        .str.replace("[–—−]", "-", regex=True)
        .str.split("-", expand=True)
        .astype(int)
    )
    sdf = sdf.reset_index(drop=True)

    for idx, row in sdf.iterrows():
        gameId = row["game_id"]

        try:
            mdf = fbref.read_player_match_stats(
                stat_type="keepers", match_id=gameId
            ).reset_index()
            mdf.columns = flattenMultiCol(mdf.columns)
        except Exception as e:
            print(f"Failed to get stats for match {gameId}: {e}")
            continue

        gkMatchRows = mdf[mdf["player"] == PLAYER_NAME]
        df.append(gkMatchRows)

    df = pd.concat(df, ignore_index=True)
    os.makedirs(dataFolder, exist_ok=True)
    df.to_pickle(DATA_CACHING_PATH)
else:
    df = pd.read_pickle(DATA_CACHING_PATH)

df["psxg_rolling"] = (
    df["shot stopping_psxg"].rolling(window=ROLLING_WINDOW, min_periods=0).mean()
)
df["ga_rolling"] = (
    df["shot stopping_ga"].rolling(window=ROLLING_WINDOW, min_periods=0).mean()
)
df["diff_rolling"] = df["psxg_rolling"] - df["ga_rolling"]
df = df[
    [
        "shot stopping_psxg",
        "shot stopping_ga",
        "diff_rolling",
        "psxg_rolling",
        "ga_rolling",
        "game",
        "season",
    ]
]
seasonChangeIndices = df[df["season"] != df["season"].shift()].index.tolist()

# Visual
fig = plt.figure(figsize=(8, 8), dpi=100)
ax = plt.subplot()
ax.set_facecolor("#eeeeee")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.grid(axis="y", ls="--", color="lightgrey")
ax.set_xlabel("Match index", size=10, labelpad=10)
ax.set_ylim(0, max(df["ga_rolling"].max(), df["psxg_rolling"].max()) + 0.1)
ax.set_xlim(-0.5, df.shape[0])

ax.plot(
    df.index,
    df["ga_rolling"],
    label="Goals Against",
    color="#c1121f",
    linestyle="-",
    linewidth=1.5,
)
ax.plot(
    df.index,
    df["psxg_rolling"],
    label="Post-shot xG",
    color="#386641",
    linestyle="-",
    linewidth=1.25,
)


def highlightColor(base_color, blend_color, diff, vmin, vmax):
    """
    Returns a color close to base_color when |diff| is large,
    and closer to blend_color when diff is near zero.
    """
    mag = abs(diff) / max(abs(vmin), abs(vmax)) if vmax != 0 or vmin != 0 else 0
    mag = 0.65 + 0.35 * mag

    # Interpolate from blend_color (when mag=0) to base_color (when mag=1)
    c_base = np.array(mcolors.to_rgb(base_color))
    c_blend = np.array(mcolors.to_rgb(blend_color))
    rgb = mag * c_base + (1 - mag) * c_blend
    return mcolors.to_hex(rgb)


vmin = df["diff_rolling"].min()
vmax = df["diff_rolling"].max()
for i in range(1, len(df["game"])):
    diff = df["diff_rolling"].iloc[i]
    if diff > 0:
        color = highlightColor("#386641", "#c1121f", diff, vmin, vmax)
    else:
        color = highlightColor("#c1121f", "#386641", diff, vmin, vmax)

    ax.fill_between(
        [i - 1, i],
        [df["psxg_rolling"].iloc[i - 1], df["psxg_rolling"].iloc[i]],
        [df["ga_rolling"].iloc[i - 1], df["ga_rolling"].iloc[i]],
        color=color,
        zorder=3,
        alpha=0.5,
        linewidth=0.1,
    )

ax.legend(markerscale=2, loc="upper left", fontsize="x-small", frameon=False)

for x in [ROLLING_WINDOW]:
    ax.fill_between(
        x=[-0.5, ROLLING_WINDOW],
        y1=ax.get_ylim()[0],
        y2=ax.get_ylim()[1],
        alpha=0.15,
        color="black",
        ec="None",
        zorder=2,
    )
    text_ = ax.annotate(
        xy=(x, 0.2),
        text=f"{ROLLING_WINDOW} games\nwindow",
        color="black",
        size=7,
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

for x in seasonChangeIndices:
    if x != 0:
        text = f"First match of the\n20{df.iloc[x].season[:2]}/{df.iloc[x].season[2:]} season"

        ax.plot(
            [x, x],
            [ax.get_ylim()[0], ax.get_ylim()[1]],
            color="black",
            alpha=0.35,
            zorder=2,
            ls="dashdot",
            lw=0.95,
        )

        text_ = ax.annotate(
            xy=(x, 2.1),
            text=text,
            color="black",
            size=7,
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

addTitleSubAndLogo(
    fig,
    ax,
    title=TITLE_TEXT,
    titleFontSize=15,
    titleLineSpacing=1,
    subtitle=SUBTITLE_TEXT,
    subtitleFontSize=9,
    subtitleLineSpacing=1.5,
    spacing=0.03,
    source="Data: FBRef | @francescozonaro",
    logo="https://images.fotmob.com/image_resources/logo/leaguelogo/87.png",
)

plt.savefig(
    f"{outputFolder}/{OUTPUT_NAME}.png",
    dpi=600,
    facecolor="#eee",
    bbox_inches="tight",
    edgecolor="none",
    pad_inches=0.3,
    transparent=False,
)
