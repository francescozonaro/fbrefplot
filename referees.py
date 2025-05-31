import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import soccerdata as sd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import os
import urllib.request

from PIL import Image

plt.rcParams["font.family"] = "Monospace"
outputFolder = os.path.join("imgs/", "barplot")
os.makedirs(outputFolder, exist_ok=True)

folder_name = "fbrefData"
schedule_path = os.path.join(folder_name, "30052025_schedule.pkl")
referee_stats_path = os.path.join(folder_name, "30052025_data.pkl")
fbref = sd.FBref(leagues="ENG-Premier League", seasons=2024)

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

if os.path.exists(referee_stats_path):
    df = pd.read_pickle(referee_stats_path)
else:
    referee_stats = {}

    for idx, row in schedule_df.iterrows():
        game_id = row["game_id"]
        referee_name = row.get("referee", "Unknown")

        try:
            match_df = fbref.read_player_match_stats(stat_type="misc", match_id=game_id)
            match_df = match_df.reset_index()
            if isinstance(match_df.columns, pd.MultiIndex):
                match_df.columns = [
                    "_".join(col).strip("_") for col in match_df.columns.values
                ]
            match_df.columns = [col.lower() for col in match_df.columns]
        except Exception as e:
            print(f"Failed to get misc stats for match {game_id}: {e}")
            continue

        if "performance_fls" in match_df.columns:
            total_fouls = match_df["performance_fls"].sum()

            if referee_name not in referee_stats:
                referee_stats[referee_name] = {"fouls": 0, "games": 0}

            referee_stats[referee_name]["fouls"] += total_fouls
            referee_stats[referee_name]["games"] += 1
        else:
            print(f"Missing Performance_Fls or min data in match {game_id}")

    df = pd.DataFrame.from_dict(referee_stats, orient="index")
    df.index.name = "referee"
    df = df.reset_index()

    os.makedirs(folder_name, exist_ok=True)
    df.to_pickle(referee_stats_path)

df["fouls90"] = df["fouls"] / df["games"]
df = df[df["games"] >= 10]
df = df.sort_values(by="fouls90", ascending=True).reset_index(drop=True)
print(df)

###

fig = plt.figure(figsize=(8, 8), dpi=100)
ax = plt.subplot()
ax.set_facecolor("#eeeeee")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.grid(ls="--", lw=1, color="lightgrey", axis="x")
ax.set_xlabel("Fouls per 90", size=10, labelpad=10)
ax.set_yticks([])

colors = ["#a3b18a", "#e26d5c"]
custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)
norm = mcolors.Normalize(vmin=df["fouls90"].min(), vmax=df["fouls90"].max())

ax.barh(
    df.index,
    df["fouls90"],
    ec="#efe9e6",
    color=custom_cmap(norm(df["fouls90"])),
    zorder=3,
)
ax.plot([0, 0], [ax.get_ylim()[0], len(df)], color="black", lw=0.75, zorder=3)


for i, (ref, val) in enumerate(zip(df["referee"], df["fouls90"])):
    label = ref
    ax.text(
        0.25,
        i,
        label,
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold",
        path_effects=[path_effects.withStroke(linewidth=1, foreground="white")],
        zorder=4,
    )

league_logo = "https://images.fotmob.com/image_resources/logo/leaguelogo/47.png"
league_icon = Image.open(urllib.request.urlopen(league_logo)).convert("LA")
logo_ax = fig.add_axes([0.8, 0.975, 0.075, 0.075], anchor="C")
logo_ax.imshow(league_icon)
logo_ax.axis("off")

ax.text(
    0,
    1.15,
    "Referee's fouls called per 90s",
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
    "It was an exciting race in Serie A, with Napoli pulling ahead in the \nxPts favored Inter throughout — a reminder that xG doesn't",
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
    f"{outputFolder}/strictRef.png",
    dpi=600,
    facecolor="#eee",
    bbox_inches="tight",
    edgecolor="none",
    pad_inches=0.2,
    transparent=False,
)

plt.savefig(
    f"{outputFolder}/strictRef_ig.png",
    dpi=600,
    facecolor="#eee",
    bbox_inches="tight",
    edgecolor="none",
    pad_inches=0.3,
    transparent=False,
)
