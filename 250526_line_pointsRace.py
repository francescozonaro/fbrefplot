import pandas as pd
import matplotlib.pyplot as plt
import soccerdata as sd
import os
import numpy as np
import urllib.request

from PIL import Image
from scipy.stats import poisson


def calculateXpts(home_xg, away_xg, max_goals=5):
    home_probs = poisson.pmf(range(0, max_goals + 1), home_xg)
    away_probs = poisson.pmf(range(0, max_goals + 1), away_xg)

    match_probs = np.outer(home_probs, away_probs)
    p_home_win = np.sum(np.tril(match_probs, -1))
    p_draw = np.sum(np.diag(match_probs))
    p_away_win = np.sum(np.triu(match_probs, 1))

    home_xpts = (3 * p_home_win) + (1 * p_draw)
    away_xpts = (3 * p_away_win) + (1 * p_draw)

    return home_xpts, away_xpts


# Initialization
plt.rcParams["font.family"] = "Monospace"
outputFolder = os.path.join("imgs/", str(f"lineplot"))
os.makedirs(outputFolder, exist_ok=True)

# Data
dataFolderName = "fbrefData"
os.makedirs(dataFolderName, exist_ok=True)
picklePath = os.path.join(dataFolderName, f"pointsRace.pkl")
if not os.path.exists(picklePath):
    fbref = sd.FBref(leagues="ITA-Serie A", seasons=2024)
    df = fbref.read_schedule()
    df.to_pickle(picklePath)
else:
    df = pd.read_pickle(picklePath)
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


teams = ["Inter", "Napoli"]
cumulativeData = {team: {"actual": [], "expected": []} for team in teams}
runningTotals = {team: {"actual": 0, "expected": 0} for team in teams}

for _, row in df.iterrows():
    homeTeam, awayTeam = row["home_team"], row["away_team"]
    home_xg, away_xg = row["home_xg"], row["away_xg"]
    homeGoals, awayGoals = row["home_goals"], row["away_goals"]
    home_xpts, away_xpts = calculateXpts(home_xg, away_xg)

    if homeGoals > awayGoals:
        homePts, awayPts = 3, 0
    elif homeGoals == awayGoals:
        homePts, awayPts = 1, 1
    else:
        homePts, awayPts = 0, 3

    for team in teams:
        if team == homeTeam:
            runningTotals[team]["actual"] += homePts
            runningTotals[team]["expected"] += home_xpts
        elif team == awayTeam:
            runningTotals[team]["actual"] += awayPts
            runningTotals[team]["expected"] += away_xpts
        else:
            continue

        cumulativeData[team]["actual"].append(runningTotals[team]["actual"])
        cumulativeData[team]["expected"].append(runningTotals[team]["expected"])

# Visual
fig = plt.figure(figsize=(10, 6), dpi=100)
ax = plt.subplot()
ax.set_facecolor("#eeeeee")
ax.grid(visible=True, ls="--", color="lightgrey")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xticks(range(0, len(cumulativeData[teams[0]]["actual"]) + 1, 2))
ax.set_xlim(left=1)
ax.set_ylabel("Points", labelpad=10)
ax.set_xlabel("Gameday", labelpad=10)

colors = {"Inter": "#1d3557", "Napoli": "#669bbc"}
for team in teams:
    rounds = range(1, len(cumulativeData[team]["actual"]) + 1)
    ax.plot(
        rounds,
        cumulativeData[team]["actual"],
        label=f"{team} actual points",
        color=colors[team],
        linewidth=2,
    )
    ax.plot(
        rounds,
        cumulativeData[team]["expected"],
        label=f"{team} expected points",
        color=colors[team],
        linestyle="-.",
    )

ax.legend(markerscale=1, loc="upper left", fontsize="x-small", frameon=False)

ax.text(
    0,
    1.15,
    "Inter's data-driven Scudetto (too bad it's not real)",
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
    "It was an exciting race in Serie A, with Napoli pulling ahead in the final stretch, while\nxPts favored Inter throughout — a reminder that xG doesn't lift trophies (yet).",
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

league_logo = "https://images.fotmob.com/image_resources/logo/leaguelogo/55.png"
league_icon = Image.open(urllib.request.urlopen(league_logo)).convert("LA")
logo_ax = fig.add_axes([0.825, 0.95, 0.075, 0.075], anchor="C")
logo_ax.imshow(league_icon)
logo_ax.axis("off")

plt.savefig(
    f"{outputFolder}/serie_a_race.png",
    dpi=600,
    facecolor="#eeeeee",
    bbox_inches="tight",
    pad_inches=0.2,
    edgecolor="none",
    transparent=False,
)

plt.savefig(
    f"{outputFolder}/serie_a_race_ig.png",
    dpi=600,
    facecolor="#eeeeee",
    bbox_inches="tight",
    pad_inches=0.3,
    edgecolor="none",
    transparent=False,
)
