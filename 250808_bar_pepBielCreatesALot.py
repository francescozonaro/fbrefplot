import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors

import os
import soccerdata as sd
import pandas as pd
import urllib.request

from _commons import flattenMultiCol
from adjustText import adjust_text
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap


IMAGE_SUB_FOLDER = "biel"
VISUAL_NAME = "250808_bar_wastedCreators_charlotte"
FBREF_FOLDER = "fbrefData"
CACHE_PATH = f"{FBREF_FOLDER}/{VISUAL_NAME}.pkl"
OUTPUT_FOLDER = f"imgs/{IMAGE_SUB_FOLDER}"

plt.rcParams["font.family"] = "Monospace"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FBREF_FOLDER, exist_ok=True)

if not os.path.exists(CACHE_PATH):
    fbref = sd.FBref(leagues="USA-Major League Soccer", seasons=2025)
    df = fbref.read_player_season_stats(stat_type="passing").reset_index()
    df.columns = flattenMultiCol(df.columns)
    df.to_pickle(CACHE_PATH)
else:
    df = pd.read_pickle(CACHE_PATH)


# df = df[df["90s"] > 10].reset_index(drop=True)
df = df[df["team"] == "Charlotte"]
df = df[df["expected_xa"] != 0]
df = df.nlargest(30, "expected_xa").reset_index(drop=True)
df["target"] = df["ast"] - df["expected_xa"]
df = df.sort_values(by="target", ascending=False).reset_index(drop=True)

fig = plt.figure(figsize=(7, 9), dpi=600)
ax = plt.subplot()
ax.set_facecolor("#eeeeee")
ax.grid(visible=True, ls="--", color="lightgrey")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("Assists - xAssists", labelpad=10)

colors = ["#C2B7B7", "#288ece"]
custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)
norm = mcolors.Normalize(vmin=df["target"].min(), vmax=df["target"].max())

ax.barh(
    df.index,
    df["target"],
    zorder=3,
    color=custom_cmap(norm(df["target"])),
)

for i, (player, val) in enumerate(zip(df["player"], df["target"])):

    label = f"{player}"
    xval = 0.2 if val > 0 else -0.2
    haval = "left" if val > 0 else "right"

    ax.text(
        xval,
        i,
        label,
        va="center",
        ha=haval,
        fontsize=9,
        zorder=4,
        fontweight="bold",
        path_effects=[path_effects.withStroke(linewidth=1, foreground="white")],
    )

ax.spines["left"].set_position(("data", 0))
ax.set_xticks(range(-5, 6, 1))
ax.set_yticks([])

league_logo = "https://images.fotmob.com/image_resources/logo/teamlogo/1323940.png"
league_icon = Image.open(urllib.request.urlopen(league_logo))  # .convert("LA")
logo_ax = fig.add_axes([0.4825, 0.9, 0.06, 0.06], anchor="C")
logo_ax.imshow(league_icon)
logo_ax.axis("off")

plt.savefig(
    f"{OUTPUT_FOLDER}/{VISUAL_NAME}.png",
    facecolor="#eeeeee",
    bbox_inches="tight",
    pad_inches=0.3,
    edgecolor="none",
    transparent=False,
)
