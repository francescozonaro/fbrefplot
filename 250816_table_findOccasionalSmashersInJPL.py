import matplotlib.pyplot as plt
import os
import soccerdata as sd
import pandas as pd
import time
from PIL import Image
import urllib.request
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from _commons import flattenMultiCol, calc_trend_from_values, addTitleSubAndLogo
from _fbref_commons import normalize_fbref_schedule

IMAGE_SUB_FOLDER = "JPL"
VISUAL_NAME = "250816_underdogSmashersForSorareJPL"
FBREF_FOLDER = "fbrefData"
CACHE_PATH = f"{FBREF_FOLDER}/{VISUAL_NAME}.pkl"
OUTPUT_FOLDER = f"imgs/{IMAGE_SUB_FOLDER}"

FBREF_TEAM_TO_FOTMOB_ID = {
    "Antwerp": "9988",
    "Dender": "7947",
    "La Louvière": "1218969",
    "Zulte Waregem": "10000",
    "Anderlecht": "8635",
    "Club Brugge": "8342",
    "OH Leuven": "1773",
    "Sint-Truiden": "9997",
    "Mechelen": "8203",
    "Gent": "9991",
    "Standard Liège": "9985",
    "Westerlo": "10001",
    "Cercle Brugge": "9984",
    "Charleroi": "9986",
    "Genk": "9987",
    "Union SG": "7978",
}

plt.rcParams["font.family"] = "Monospace"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FBREF_FOLDER, exist_ok=True)

if not os.path.exists(CACHE_PATH):
    fbrefPast = sd.FBref(
        leagues="BEL-Belgian Pro League",
        seasons=[1718, 1819, 1920, 2021, 2122, 2223, 2324, 2425],
    )
    df = fbrefPast.read_schedule().reset_index()
    df.columns = flattenMultiCol(df.columns)
    df.to_pickle(CACHE_PATH)
else:
    df = pd.read_pickle(CACHE_PATH)

df = df[df["round"] == "Regular season"]
df = normalize_fbref_schedule(df)
df = df[
    [
        "round",
        "team",
        "season",
        "opponent",
        "goals",
        "opponent_goals",
        "game_id",
        "at_home",
    ]
]

games_per_season = df.groupby(["team"])["game_id"].count()

mask = (df["goals"] >= 2) & (df["opponent_goals"] == 0)
df = df[mask]

# This allows to print not only the sum of the "good games", but also the seasons where they were obtained.
res = (
    df.groupby(["team", "season"])["game_id"].count().unstack()
)  # TODO what does this actually do
res = res[res.index.isin(FBREF_TEAM_TO_FOTMOB_ID)]
res["total"] = res.sum(axis=1)
res["cs_perc"] = res.index.map(games_per_season)  # map total games played
res["cs_perc"] = (res["total"] / res["cs_perc"] * 100).round(2)

season_cols = [c for c in res.columns if str(c).isdigit()]
res["trend"] = res.apply(
    lambda row: calc_trend_from_values(row[season_cols[-5:]].astype(float).values),
    axis=1,
)
res = res.sort_values(by="cs_perc", ascending=False)

HEADER_MAPPINGS = {
    "1718": "17/18",
    "1819": "18/19",
    "1920": "19/20",
    "2021": "20/21",
    "2122": "21/22",
    "2223": "22/23",
    "2324": "23/24",
    "2425": "24/25",
    "total": "Total",
    "cs_perc": "%",
    "trend": "Last 5",
}

# Let's print the result in a nice format
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.set_facecolor("#eeeeee")
ax.set_axis_off()
ax.set_xlim(0, 1)

colors = ["#eaeaea", "#d5d5d5"]
row_colors = [colors[i % 2] for i in range(len(res))]
# This identifies cols position so that each takes the same space
left_marg, right_marg = 0.3, 1
col_space = (right_marg - left_marg) / (len(res.columns) - 1)
col_positions = [
    left_marg + i * col_space - col_space / 2 for i in range(len(res.columns))
]

colors = ["#e76f51", "#588157"]  # example green
cmap = LinearSegmentedColormap.from_list("custom_red_green", colors)
cs_perc_values = res["cs_perc"]
norm = Normalize(vmin=cs_perc_values.min(), vmax=cs_perc_values.max())

for i, (team, row) in enumerate(res.iterrows()):
    # time.sleep(1)
    y = 0.9 - i * 0.04
    ax.fill_between([0, 1], y - 0.02, y + 0.02, color=row_colors[i], zorder=-1)

    team_logo_url = f"https://images.fotmob.com/image_resources/logo/teamlogo/{FBREF_TEAM_TO_FOTMOB_ID[team]}.png"
    team_icon = Image.open(urllib.request.urlopen(team_logo_url)).convert("RGBA")
    team_image = OffsetImage(team_icon, zoom=0.08, resample=True)
    # Add logo
    ab = AnnotationBbox(team_image, (0.02, y), frameon=False, box_alignment=(0.5, 0.5))
    ax.add_artist(ab)
    # Add text
    ax.text(0.06, y, team, ha="left", va="center", fontweight="bold", fontsize=9)

    for j, col in enumerate(res.columns):

        if pd.isna(row[col]):
            formattedVal = "-"
        else:
            formattedVal = str(int(row[col]))

            if col == "cs_perc":
                formattedVal = f"{round(row[col], 2)}%"
                bg_color = cmap(norm(row[col]))
                ax.fill_between(
                    [
                        col_positions[j] - (col_space / 2),
                        col_positions[j] + (col_space / 2),
                    ],
                    y - 0.02,
                    y + 0.02,
                    color=bg_color,
                    zorder=-1,
                )

            if col == "trend":
                slope = row[col]

                if row.name not in ["Hamburger SV", "St. Pauli"]:
                    dx = 0.04  # arrow length in x
                    dy = 0.02 * slope  # arrow length in y, scaled by slope
                    x0 = col_positions[j] - dx / 2
                    y0 = y - dy / 2

                    arrowColors = ["#e76f51", "#e7c451", "#588157"]
                    color = arrowColors[0 if slope < 0 else 1 if slope == 0 else 2]

                    ax.arrow(
                        x0,
                        y0,
                        dx,
                        dy,
                        head_width=0.005,
                        head_length=0.005,
                        fc=color,
                        ec=color,
                        linewidth=1,
                        length_includes_head=True,
                    )

                formattedVal = ""

        ax.text(
            col_positions[j],
            y,
            formattedVal,
            ha="center",
            va="center",
            fontsize=9,
        )

# Headers
for j, col in enumerate(res.columns):
    ax.text(
        col_positions[j],
        0.95,
        HEADER_MAPPINGS[col],
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

ax.text(
    x=0,  # right edge
    y=0,  # bottom edge
    s="Mask: 2+ scored, 0 conceded",
    transform=ax.transAxes,  # use axes coordinates
    ha="left",
    va="bottom",
    fontsize=9,
    alpha=0.85,
)


ax.text(
    x=1,  # right edge
    y=0,  # bottom edge
    s="@francescozonaro",
    transform=ax.transAxes,  # use axes coordinates
    ha="right",
    va="bottom",
    fontsize=9,
    alpha=0.85,
)

plt.savefig(
    f"{OUTPUT_FOLDER}/{VISUAL_NAME}.png",
    facecolor="#eceff4",
    bbox_inches="tight",
    pad_inches=0.3,
    edgecolor="none",
    transparent=False,
)
