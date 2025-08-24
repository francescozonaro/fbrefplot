import matplotlib.pyplot as plt
import os
import soccerdata as sd
import pandas as pd

from _commons import flattenMultiCol


IMAGE_SUB_FOLDER = "biel"
VISUAL_NAME = "250811_idk"
FBREF_FOLDER = "fbrefData"
CACHE_PATH = f"{FBREF_FOLDER}/{VISUAL_NAME}.pkl"
OUTPUT_FOLDER = f"imgs/{IMAGE_SUB_FOLDER}"

plt.rcParams["font.family"] = "Monospace"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FBREF_FOLDER, exist_ok=True)

fbref = sd.FBref(leagues="USA-Major League Soccer", seasons=2025)
if not os.path.exists(CACHE_PATH):
    df = fbref.read_schedule().reset_index()
    df.columns = flattenMultiCol(df.columns)
    df.to_pickle(CACHE_PATH)
else:
    df = pd.read_pickle(CACHE_PATH)

TARGET_TEAM = "Seattle Sounders"
team_df = df[(df["home_team"] == TARGET_TEAM) | (df["away_team"] == TARGET_TEAM)]


def checkifPlayerIsStarter(row, player_name):
    game_id = row["game_id"]
    lineup = fbref.read_lineup(match_id=game_id)
    filteredLineup = lineup.loc[lineup["player"] == player_name, "is_starter"]

    if not filteredLineup.empty:
        isStarter = filteredLineup.iloc[0]
    else:
        isStarter = False

    return isStarter


team_df["playerIsStarter"] = team_df.apply(
    lambda row: checkifPlayerIsStarter(row, player_name="Yeimar Gómez Andrade"), axis=1
)

print(team_df)
