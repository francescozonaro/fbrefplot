import soccerdata as sd
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors

from datetime import datetime
from collections import defaultdict
from _commons import flattenMultiCol
from _fbref_commons import (
    separate_score,
    filter_regular_season,
    normalize_fbref_schedule,
)


# Functions
def isEasyMatch(team_row, mode, offStrong, offWeak, defStrong, defWeak):
    if mode == "DEF":
        return team_row["team"] in defStrong and team_row["opponent"] in offWeak
    else:
        return team_row["team"] in offStrong and team_row["opponent"] in defWeak


# Init
FBREF_FOLDER = "fbrefData"
TARGET_LEAGUE = "BEL-Belgian Pro League"
TODAY = datetime.today().strftime("%Y%m%d")
VISUAL_NAME = f"{TODAY}_{TARGET_LEAGUE}_sorareFixtureCorrelation"
CACHE_PATH = f"{FBREF_FOLDER}/{VISUAL_NAME}.pkl"
OUTPUT_FOLDER = f"imgs/{TARGET_LEAGUE}"

plt.rcParams["font.family"] = "Monospace"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FBREF_FOLDER, exist_ok=True)

# Consts
PERCENTILE_THRESHOLD = 60  # 0-100
MIN_GOOD_GWS_NUMBER = 5
PREV_SEASON = "2425"
CUR_SEASON = "2526"
USE_CUSTOM_INPUTS = True

fbref = sd.FBref(
    leagues=TARGET_LEAGUE,
    seasons=[PREV_SEASON, CUR_SEASON],
)

if not os.path.exists(CACHE_PATH):
    df = fbref.read_schedule().reset_index()
    df.columns = flattenMultiCol(df.columns)
    df.to_pickle(CACHE_PATH)
else:
    df = pd.read_pickle(CACHE_PATH)

df_played = df.copy()
df_played = df_played[df_played["score"].notna()]
df_played["hg"], df_played["ag"] = separate_score(df_played["score"])
df_played = df_played.pipe(filter_regular_season, TARGET_LEAGUE)

df_played_pre = (
    df_played[df_played["season"] == PREV_SEASON].copy().reset_index(drop=True)
)
df_played_cur = (
    df_played[df_played["season"] == CUR_SEASON].copy().reset_index(drop=True)
)

df_future = df.copy()
df_future = df_future[df_future["score"].isna()].reset_index(drop=True)

# Find teams of current season
teams_previous_season = sorted(pd.unique(df_played_pre["home_team"].values))
teams_current_season = sorted(pd.unique(df_future["home_team"].values))

if USE_CUSTOM_INPUTS:
    offStrong = [
        "Anderlecht",
        "Club Brugge",
        "Genk",
        "Gent",
        "Sint-Truiden",
        "Union SG",
    ]
    offWeak = [
        "Cercle Brugge",
        "Charleroi",
        "La Louvière",
        "OH Leuven",
        "Westerlo",
        "Zulte Waregem",
    ]
    defStrong = [
        "Anderlecht",
        "Club Brugge",
        "Genk",
        "Gent",
        "Mechelen",
        "Sint-Truiden",
        "Union SG",
    ]
    defWeak = [
        "Cercle Brugge",
        "Charleroi",
        "La Louvière",
        "OH Leuven",
        "Westerlo",
        "Zulte Waregem",
    ]
else:
    # Check if there are enough matches to use current season data
    if len(df_played_cur) > 80:
        # Use only current season matches
        df_used = df_played_cur.reset_index(drop=True)
    else:
        # Use both current season matches and previous season matches
        df_used = df_played.reset_index(drop=True)

    teamScores = defaultdict(dict)
    promotedTeams = []

    for team in teams_current_season:
        teamFrame = df_used.copy()
        hFrame = teamFrame[teamFrame["home_team"] == team]
        aFrame = teamFrame[teamFrame["away_team"] == team]
        num_played_matches = len(hFrame) + len(aFrame)
        if team not in teams_previous_season:
            promotedTeams.append(team)
            continue
        offScore = (hFrame["hg"].sum() + aFrame["ag"].sum()) / num_played_matches
        defScore = (hFrame["ag"].sum() + aFrame["hg"].sum()) / num_played_matches
        teamScores[team] = {
            "offScore": round(float(offScore), 2),
            "defScore": round(float(defScore), 2),
        }

    # Fix promoted teams scores, setting 3rd worst value as their value
    sortedOff = sorted(teamScores.items(), key=lambda x: x[1]["offScore"])
    sortedDef = sorted(teamScores.items(), key=lambda x: x[1]["defScore"], reverse=True)
    for team in promotedTeams:
        teamScores[team]["offScore"] = sortedOff[2][1]["offScore"]
        teamScores[team]["defScore"] = sortedDef[2][1]["defScore"]

    offScores = sorted([v["offScore"] for v in teamScores.values()])
    defScores = sorted([v["defScore"] for v in teamScores.values()])
    offThreshold = round(np.percentile(offScores, PERCENTILE_THRESHOLD), 2)
    defThreshold = round(np.percentile(defScores, 100 - PERCENTILE_THRESHOLD), 2)
    offOppThreshold = round(np.percentile(offScores, 100 - PERCENTILE_THRESHOLD), 2)
    defOppThreshold = round(np.percentile(defScores, PERCENTILE_THRESHOLD), 2)
    offStrong = [t for t, v in teamScores.items() if v["offScore"] >= offThreshold]
    offWeak = [t for t, v in teamScores.items() if v["offScore"] <= offOppThreshold]
    defStrong = [t for t, v in teamScores.items() if v["defScore"] <= defThreshold]
    defWeak = [t for t, v in teamScores.items() if v["defScore"] >= defOppThreshold]

home_cols = {
    "home_team": "team",
    "away_team": "opponent",
}
away_cols = {
    "home_team": "opponent",
    "away_team": "team",
}
df_norm = normalize_fbref_schedule(df_future, home_cols, away_cols)

# Note that every team is processed offensively even without explicitly iterating on the "OFF" target mode
res = {}
maxBestPairings = 0
target_mode = "DEF"
opposite_mode = "OFF"

df_norm["easy_DEF"] = df_norm.apply(
    lambda r: isEasyMatch(r, target_mode, offStrong, offWeak, defStrong, defWeak),
    axis=1,
)
df_norm["easy_OFF"] = df_norm.apply(
    lambda r: isEasyMatch(r, opposite_mode, offStrong, offWeak, defStrong, defWeak),
    axis=1,
)

easy_target = df_norm[df_norm[f"easy_{target_mode}"]].copy()
easy_opposite = df_norm[df_norm[f"easy_{opposite_mode}"]].copy()

merged = easy_target.merge(easy_opposite, on="week", suffixes=("_tgt", "_opp"))
merged = merged[merged["team_tgt"] != merged["team_opp"]]

counts = merged.groupby(["team_tgt", "team_opp"]).size().reset_index(name="Count")

for team in teams_current_season:
    df_res_team = counts[counts["team_tgt"] == team].sort_values(
        "Count", ascending=False
    )
    if not df_res_team.empty:
        highest_count = df_res_team.iloc[0]["Count"]
        maxBestPairings = max(maxBestPairings, highest_count)
        if highest_count >= MIN_GOOD_GWS_NUMBER:
            res[team] = df_res_team.rename(columns={"team_opp": "Team"})

rows = int(round(len(res) / 2))

fig, axs = plt.subplots(rows, 2, figsize=(12, 16), sharex=True, dpi=600)
fig.subplots_adjust(wspace=0.1, hspace=0.3)
fig.patch.set_facecolor("#eceff4")

teamKeys = list(res.keys())

for i, ax in enumerate(axs.flatten()):

    if i >= len(res):
        fig.delaxes(ax)
        continue

    ax.tick_params(axis="x", labelbottom=True)
    dark_green = "#226f54"
    light_green = "#c3d011"

    # Create the colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "green_gradient", [light_green, dark_green]
    )

    plotFrame = res.get(teamKeys[i])

    norm = plt.Normalize(vmin=1, vmax=maxBestPairings)
    colors = [cmap(norm(value)) for value in plotFrame["Count"]]

    bars = ax.barh(
        plotFrame["Team"],
        plotFrame["Count"],
        color=colors,
        edgecolor="grey",
        alpha=0.8,
    )

    ax.set_yticks([])
    for bar, team in zip(bars, plotFrame["Team"]):
        text = ax.text(
            x=0.2,  # A bit of padding from left edge
            y=bar.get_y() + bar.get_height() / 2,
            s=team,
            va="center",
            ha="left",
            fontweight="bold",
            fontsize=11,
            color="white",
        )

        text.set_path_effects(
            [
                path_effects.withStroke(linewidth=1.75, foreground="black"),
            ]
        )

    ax.axvline(0, color="black", lw=1)
    ax.set_title(
        f"{teamKeys[i]}",
        fontsize=13,
        # color="#5A5A5A",
        fontweight="bold",
        pad=8,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.set_facecolor("#eceff4")
    ax.invert_yaxis()


print("reaching")
plt.savefig(
    f"{OUTPUT_FOLDER}/{VISUAL_NAME}.png",
    dpi=600,
    facecolor="#eceff4",
    bbox_inches="tight",
    pad_inches=0.3,
    edgecolor="none",
    transparent=False,
)
