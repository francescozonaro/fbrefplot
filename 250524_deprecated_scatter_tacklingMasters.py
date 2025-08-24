import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
import scipy.stats as stats
import soccerdata as sd
import urllib.request
import os

from _commons import addTitleSubAndLogo
from adjustText import adjust_text
from PIL import Image

# Initialization
plt.rcParams["font.family"] = "Monospace"
outputFolder = os.path.join("imgs/", str(f"scatterplot"))
os.makedirs(outputFolder, exist_ok=True)

# Data
fbref = sd.FBref(leagues="ITA-Serie B", seasons=2024)
df = fbref.read_player_season_stats(stat_type="defense")
df = df.reset_index()
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ["_".join(col).strip("_") for col in df.columns.values]

df["tkl90"] = df["Tackles_Tkl"] / df["90s"]
df["tklW90"] = df["Tackles_TklW"] / df["90s"]
df["tklW%"] = (df["tklW90"] / df["tkl90"]).round(2)

df = df[(df["90s"] >= df["90s"].median()) & (df["tkl90"] >= df["tkl90"].median())]
df = df[["team", "player", "pos", "90s", "tkl90", "tklW90", "tklW%"]]
df = df.sort_values(by="tkl90", ascending=False)
df = df.reset_index(drop=True)
df["zscore"] = stats.zscore(df["tkl90"]) * 0.5 + stats.zscore(df["tklW%"]) * 0.5
df["annotated"] = [
    True if x > df["zscore"].quantile(0.85) else False for x in df["zscore"]
]

# Visual
fig = plt.figure(
    figsize=(8, 8),
    dpi=100,
)
ax = plt.subplot()
ax.set_facecolor("#eeeeee")
ax.grid(visible=True, ls="--", color="lightgrey")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_ylabel("Tackles won %", labelpad=10)
ax.set_xlabel("Tackles attempted per 90", labelpad=10)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))

ax.scatter(
    df["tkl90"],
    df["tklW%"],
    c=df["zscore"],
    cmap="Reds",
    zorder=3,
    ec="grey",
    s=55,
    alpha=0.8,
)

texts = []
annotated = df[df["annotated"]].reset_index(drop=True)
for index in range(annotated.shape[0]):
    texts += [
        ax.text(
            x=annotated["tkl90"].iloc[index],
            y=annotated["tklW%"].iloc[index],
            s=f"{annotated['player'].iloc[index]}",
            path_effects=[
                path_effects.Stroke(linewidth=2, foreground=fig.get_facecolor()),
                path_effects.Normal(),
            ],
            color="black",
            weight="bold",
            size=8,
        )
    ]

adjust_text(
    texts,
    only_move={"static": "y"},
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
)

addTitleSubAndLogo(
    fig,
    ax,
    title="Serie B's tackling gladiators | 2024-25",
    titleFontSize=15,
    titleLineSpacing=1,
    subtitle="This scatterplot showcases Serie B players who have played at least the median number of minutes and attempt at least the median number of tackles per 90 minutes during the 2024-25 season. The plot highlights those who excel not only in the frequency of tackles attempted but also in the percentage of tackles won, combining these metrics into a composite score to identify the league's most effective and active defensive players.",
    subtitleFontSize=9,
    subtitleLineSpacing=1.5,
    spacing=0.02,
    source="Data: FBRef | @francescozonaro",
    logo="https://images.fotmob.com/image_resources/logo/leaguelogo/86.png",
)

plt.savefig(
    f"{outputFolder}/gladiators.png",
    dpi=600,
    facecolor="#eeeeee",
    bbox_inches="tight",
    pad_inches=0.4,
    edgecolor="none",
    transparent=False,
)
