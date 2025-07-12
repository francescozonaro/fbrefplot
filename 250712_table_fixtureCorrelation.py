import soccerdata as sd
import numpy as np
import pandas as pd
from collections import defaultdict


from _commons import initPlotting, initFolders, flattenMultiCol

# Initialization
initPlotting()
outputFolder, dataFolder = initFolders(imageSubFolder="multibar")

# Fbref
fbref = sd.FBref(leagues="JPN-JLeague", seasons=2025)
df = fbref.read_schedule().reset_index()
df.columns = flattenMultiCol(df.columns)

print(df)
