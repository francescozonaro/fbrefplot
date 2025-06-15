import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import soccerdata as sd
import matplotlib.colors as mcolors
import os
import urllib.request

from PIL import Image
from highlight_text import fig_text
from matplotlib.colors import LinearSegmentedColormap

# Initialization
plt.rcParams["font.family"] = "Monospace"
outputFolder = os.path.join("imgs/", str(f"barplot"))
os.makedirs(outputFolder, exist_ok=True)

# Data
folder_name = "fbrefData"
os.makedirs(folder_name, exist_ok=True)
pickle_path = os.path.join(folder_name, f"venues.pkl")
if not os.path.exists(pickle_path):
    fbref = sd.FBref(leagues="ENG-Premier League", seasons=2024)
    df = fbref.read_schedule()
    df.to_pickle(pickle_path)
else:
    df = pd.read_pickle(pickle_path)
df = df.reset_index()
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ["_".join(col).strip("_") for col in df.columns.values]

df = df[df["venue"].notna() & df["score"].notna()]
df[["home_goals", "away_goals"]] = (
    df["score"]
    .str.replace("[–—−]", "-", regex=True)
    .str.split("-", expand=True)
    .astype(int)
)
df["total_goals"] = df["home_goals"] + df["away_goals"]
latestVenues = df.groupby("home_team")["venue"].last().to_dict()
df = (
    df.groupby("home_team")
    .agg(
        total_goals=("total_goals", "sum"),
        games_counted=("game_id", "count"),
    )
    .reset_index()
)
df["goals90"] = df["total_goals"] / df["games_counted"]
df["venue"] = df["home_team"].map(latestVenues)
df = df.sort_values(by="goals90", ascending=True).reset_index(drop=True)

# Visual
fig = plt.figure(figsize=(8, 8), dpi=100)
ax = plt.subplot()
ax.set_facecolor("#eeeeee")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.grid(ls="--", lw=1, color="lightgrey", axis="x")
ax.set_xlabel("Goals per 90", size=10, labelpad=10)
ax.set_yticks([])

colors = ["#748cab", "#e26d5c"]
custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)
norm = mcolors.Normalize(vmin=df["goals90"].min(), vmax=df["goals90"].max())

ax.barh(
    df.index,
    df["goals90"],
    ec="#efe9e6",
    color=custom_cmap(norm(df["goals90"])),
    zorder=3,
)
ax.plot([0, 0], [ax.get_ylim()[0], 19.5], color="black", lw=0.75, zorder=3)


for i, (team, venue, val) in enumerate(
    zip(df["home_team"], df["venue"], df["goals90"])
):
    label = f"{venue.split('-')[0].strip()}"
    ax.text(
        0.05,
        i,
        label,
        va="center",
        ha="left",
        fontsize=8,
        fontweight="bold",
        path_effects=[path_effects.withStroke(linewidth=1, foreground="white")],
        zorder=4,
    )

league_logo = "https://images.fotmob.com/image_resources/logo/leaguelogo/47.png"
league_icon = Image.open(urllib.request.urlopen(league_logo)).convert("LA")
logo_ax = fig.add_axes([0.8, 0.975, 0.075, 0.075], anchor="C")
logo_ax.imshow(league_icon)
logo_ax.axis("off")

fig_text(
    x=0.125,
    y=1,
    s="Box Office Football: The venues\nthat never missed",
    size=18,
    family="Monospace",
    va="bottom",
    weight="bold",
)

fig_text(
    x=0.125,
    y=0.925,
    s="The 2024/25 most <entertaining> theatres of football — and\nwhere it's been a bit <less lively>.",
    size=10,
    color="#5A5A5A",
    va="bottom",
    family="Monospace",
    weight="normal",
    highlight_textprops=[
        {
            "color": "w",
            "weight": "bold",
            "bbox": {"facecolor": "#e26d5c", **{"linewidth": 0, "pad": 1.5}},
        },
        {
            "color": "w",
            "weight": "bold",
            "bbox": {"facecolor": "#748cab", **{"linewidth": 0, "pad": 1.5}},
        },
    ],
)

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
    f"{outputFolder}/boxOfficeVenues.png",
    dpi=600,
    facecolor="#eee",
    bbox_inches="tight",
    edgecolor="none",
    pad_inches=0.2,
    transparent=False,
)

plt.savefig(
    f"{outputFolder}/boxOfficeVenues_ig.png",
    dpi=600,
    facecolor="#eee",
    bbox_inches="tight",
    edgecolor="none",
    pad_inches=0.3,
    transparent=False,
)
