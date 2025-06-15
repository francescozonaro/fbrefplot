import matplotlib.pyplot as plt
import os
import pandas as pd
import textwrap


def initPlotting(fontFamily="Monospace"):
    plt.rcParams["font.family"] = fontFamily


def initFolders(imageSubFolder: str):
    outFolder = os.path.join("imgs", imageSubFolder)
    os.makedirs(outFolder, exist_ok=True)
    dataFolder = "fbrefData"
    os.makedirs(dataFolder, exist_ok=True)
    return outFolder, dataFolder


def flattenMultiCol(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, col)).strip("_") for col in df.columns.values]
    return df


def justifyText(text, width):
    lines = textwrap.wrap(text, width)
    justified_lines = []
    for line in lines[:-1]:  # Don't justify the last line
        words = line.split()
        if len(words) == 1:
            justified_lines.append(line)
            continue
        total_spaces = width - sum(len(word) for word in words)
        spaces_between_words = len(words) - 1
        space_width, extra = divmod(total_spaces, spaces_between_words)
        justified_line = ""
        for i, word in enumerate(words[:-1]):
            justified_line += word + " " * (space_width + (1 if i < extra else 0))
        justified_line += words[-1]
        justified_lines.append(justified_line)
    justified_lines.append(lines[-1])  # Last line as-is
    return "\n".join(justified_lines)
