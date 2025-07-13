import soccerdata as sd
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from tabulate import tabulate


from _commons import initPlotting, initFolders, flattenMultiCol

# Initialization
initPlotting()
outputFolder, dataFolder = initFolders(imageSubFolder="table")
CACHING_PATH = os.path.join(dataFolder, f"jleague2025.pkl")

# Fbref
fbref = sd.FBref(leagues="JPN-JLeague", seasons=2025)

if not os.path.exists(CACHING_PATH):
    df = fbref.read_schedule().reset_index()
    df.columns = flattenMultiCol(df.columns)
    df.to_pickle(CACHING_PATH)
else:
    df = pd.read_pickle(CACHING_PATH)

pdf = df.copy()
pdf = pdf[pdf["score"].notna()]
pdf[["home_goals", "away_goals"]] = pdf["score"].str.split("–", expand=True)
pdf["home_goals"] = pdf["home_goals"].astype(int)
pdf["away_goals"] = pdf["away_goals"].astype(int)

# print(pdf.head(10))
# print(pdf.columns)

teams = sorted(pd.unique(pdf["home_team"].values))

standings = pd.DataFrame(
    index=teams, columns=["MP", "GF", "GA", "PTS"], dtype=np.int64
).fillna(0)

for _, row in pdf.iterrows():
    home = row["home_team"]
    away = row["away_team"]
    hg = row["home_goals"]
    ag = row["away_goals"]

    standings.at[home, "MP"] += 1
    standings.at[away, "MP"] += 1
    standings.at[home, "GF"] += hg
    standings.at[home, "GA"] += ag
    standings.at[away, "GF"] += ag
    standings.at[away, "GA"] += hg

    if hg > ag:
        standings.at[home, "PTS"] += 3
    elif hg < ag:
        standings.at[away, "PTS"] += 3
    else:
        standings.at[home, "PTS"] += 1
        standings.at[away, "PTS"] += 1


standings["GF"] = standings["GF"] / standings["MP"]
standings["GA"] = standings["GA"] / standings["MP"]
standings["PTS"] = standings["PTS"] / standings["MP"]


tbdf = df.copy()
tbdf = tbdf[tbdf["score"].isna()].reset_index(drop=True)

print("DEFENSIVE STACK PAIRED WITH OFFENSIVE")
for team in teams:
    if standings.at[team, "GA"] < 1.35:
        goodPairing = {}
        for week, gdf in tbdf.groupby("week"):
            targetHasEasyMatch = False
            for _, row in gdf.iterrows():
                if row["home_team"] == team:
                    if standings.at[row["away_team"], "GF"] < 1.2:
                        targetHasEasyMatch = True
                if row["away_team"] == team:
                    if standings.at[row["home_team"], "GF"] < 1.2:
                        targetHasEasyMatch = True

            if targetHasEasyMatch:
                for _, row in gdf.iterrows():
                    if standings.at[row["away_team"], "GA"] > 1:
                        goodPairing[row["home_team"]] = (
                            goodPairing.get(row["home_team"], 0) + 1
                        )
                    if standings.at[row["home_team"], "GA"] > 1:
                        goodPairing[row["away_team"]] = (
                            goodPairing.get(row["away_team"], 0) + 1
                        )

        filteredPairings = {
            pair: count
            for pair, count in goodPairing.items()
            if count > 4 and pair != team
        }
        if filteredPairings:
            sortedPairings = dict(
                sorted(filteredPairings.items(), key=lambda item: item[1], reverse=True)
            )

            print(team)
            for pairTeam, count in sortedPairings.items():
                print(f"    {pairTeam}: {count}")
            print("")

# print("OFFENSIVE STACK PAIRED WITH DEFENSIVE")
# for team in teams:
#     if standings.at[team, "GF"] > 1.1:
#         goodPairing = {}
#         for week, gdf in tbdf.groupby("week"):
#             targetHasEasyMatch = False
#             for _, row in gdf.iterrows():
#                 if row["home_team"] == team:
#                     if standings.at[row["away_team"], "GA"] > 1.2:
#                         targetHasEasyMatch = True
#                 if row["away_team"] == team:
#                     if standings.at[row["home_team"], "GA"] > 1.2:
#                         targetHasEasyMatch = True

#             if targetHasEasyMatch:
#                 for _, row in gdf.iterrows():
#                     if standings.at[row["away_team"], "GF"] < 1.2:
#                         goodPairing[row["home_team"]] = (
#                             goodPairing.get(row["home_team"], 0) + 1
#                         )
#                     if standings.at[row["home_team"], "GF"] < 1.2:
#                         goodPairing[row["away_team"]] = (
#                             goodPairing.get(row["away_team"], 0) + 1
#                         )

#         filteredPairings = {
#             pair: count
#             for pair, count in goodPairing.items()
#             if count > 5 and pair != team
#         }
#         if filteredPairings:
#             sortedPairings = dict(
#                 sorted(filteredPairings.items(), key=lambda item: item[1], reverse=True)
#             )

#             print(team)
#             for pairTeam, count in sortedPairings.items():
#                 print(f"    {pairTeam}: {count}")
#             print("")
