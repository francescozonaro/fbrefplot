import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import soccerdata as sd
import matplotlib.colors as mcolors
import os
import urllib.request
import numpy as np

from PIL import Image

# Initialization
plt.rcParams["font.family"] = "Monospace"
outputFolder = os.path.join("imgs/", "lineplot")
os.makedirs(outputFolder, exist_ok=True)

TEAM_NAME = "Torino"
PLAYER_NAME = "Vanja Milinković-Savić"
OUTPUT_NAME = "milinkovicSavicRollingPSXG"
ROLLING_WINDOW = 10

# Data
folder_name = "fbrefData"
schedule_path = os.path.join(folder_name, "31052025_schedule.pkl")
data_path = os.path.join(folder_name, "31052025_data.pkl")
fbref = sd.FBref(leagues="ITA-Serie A", seasons=range(2023, 2025))

if not os.path.exists(schedule_path):
    schedule_df = fbref.read_schedule()
    os.makedirs(folder_name, exist_ok=True)
    schedule_df.to_pickle(schedule_path)
else:
    schedule_df = pd.read_pickle(schedule_path)

schedule_df = schedule_df.reset_index()
if isinstance(schedule_df.columns, pd.MultiIndex):
    schedule_df.columns = [
        "_".join(col).strip("_") for col in schedule_df.columns.values
    ]

schedule_df = schedule_df[
    schedule_df[["home_team", "away_team"]].isin([f"{TEAM_NAME}"]).any(axis=1)
].reset_index(drop=True)

df_rows = []

if os.path.exists(data_path):
    df = pd.read_pickle(data_path)
else:
    for idx, row in schedule_df.iterrows():
        game_id = row["game_id"]

        try:
            match_df = fbref.read_player_match_stats(
                stat_type="keepers", match_id=game_id
            )
            match_df = match_df.reset_index()
            if isinstance(match_df.columns, pd.MultiIndex):
                match_df.columns = [
                    "_".join(col).strip("_") for col in match_df.columns.values
                ]
            match_df.columns = [col.lower() for col in match_df.columns]
        except Exception as e:
            print(f"Failed to get stats for match {game_id}: {e}")
            continue

        keeper_match_rows = match_df[match_df["player"] == PLAYER_NAME]
        df_rows.append(keeper_match_rows)

    df = pd.concat(df_rows, ignore_index=True)
    os.makedirs(folder_name, exist_ok=True)
    df.to_pickle(data_path)

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
ax.set_ylim(0, 2.2)
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

ax.legend(markerscale=2, loc="upper right", fontsize="x-small", frameon=False)

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
        xy=(x, 2.1),
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

league_logo = "https://images.fotmob.com/image_resources/logo/leaguelogo/55.png"
team_icon = Image.open(urllib.request.urlopen(league_logo)).convert("LA")
logo_ax = fig.add_axes([0.05, 0.98, 0.04, 0.04], anchor="C")
logo_ax.imshow(team_icon)
logo_ax.axis("off")

team_logo = "https://images.fotmob.com/image_resources/logo/teamlogo/9804.png"
team_icon = Image.open(urllib.request.urlopen(team_logo)).convert("LA")
team_ax = fig.add_axes([0.05, 0.93, 0.04, 0.04], anchor="C")
team_ax.imshow(team_icon)
team_ax.axis("off")


ax.text(
    0,
    1.15,
    "Vanja 'the wall' Milinković-Savić",
    ha="left",
    va="bottom",
    fontsize=15,
    weight="bold",
    color="black",
    transform=ax.transAxes,
)

txt = ax.text(
    0,
    1.06,
    "A look at the rolling gap between goals conceded and PSxG faced (using 10-game\nwindows), highlighting Milinković-Savić's standout shot-stopping metrics.",
    ha="left",
    va="bottom",
    fontsize=9,
    color="#5A5A5A",
    transform=ax.transAxes,
)
txt.set_linespacing(1.5)

ax.text(
    1.0,
    0.035,
    "@francescozonaro | Data from FBRef",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=8,
    color="#5A5A5A",
    family="Monospace",
)

plt.savefig(
    f"{outputFolder}/{OUTPUT_NAME}.png",
    dpi=600,
    facecolor="#eee",
    bbox_inches="tight",
    edgecolor="none",
    pad_inches=0.2,
    transparent=False,
)

plt.savefig(
    f"{outputFolder}/{OUTPUT_NAME}_ig.png",
    dpi=600,
    facecolor="#eee",
    bbox_inches="tight",
    edgecolor="none",
    pad_inches=0.3,
    transparent=False,
)
