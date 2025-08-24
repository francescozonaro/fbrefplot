import matplotlib.pyplot as plt
import os
import pandas as pd
import textwrap
import urllib.request

from PIL import Image


def flattenMultiCol(columns):
    if isinstance(columns, pd.MultiIndex):
        return ["_".join(map(str, col)).strip("_").lower() for col in columns.values]
    return columns


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


def estimateTextHeight(fig, fontsize, numChars, charsPerLine, linespacing=1.0):
    numLines = (numChars + charsPerLine - 1) // charsPerLine
    fontHeightInches = fontsize / 72
    totalHeightInches = numLines * fontHeightInches * linespacing
    figHeightInches = fig.get_size_inches()[1]
    return totalHeightInches / figHeightInches


def addTitleSubAndLogo(
    fig,
    ax,
    title,
    titleFontSize,
    titleLineSpacing,
    subtitle,
    subtitleFontSize,
    subtitleLineSpacing,
    spacing=0.02,
    source=None,
    logo=None,
):

    axPosition = ax.get_position()
    leftEdge = axPosition.x0
    rightEdge = axPosition.x1
    figWidth = fig.get_size_inches()[0]

    titleCharWidth = (subtitleFontSize * 0.68) / 72
    titleCharNumber = int(figWidth / titleCharWidth)
    titleJustified = justifyText(title, titleCharNumber)

    subCharWidth = (subtitleFontSize * 0.68) / 72
    subtitleCharNumber = int((figWidth * rightEdge - leftEdge) / subCharWidth)
    subJustified = justifyText(subtitle, subtitleCharNumber)

    titleHeight = estimateTextHeight(
        fig, titleFontSize, len(title), titleCharNumber, titleLineSpacing
    )
    subtitleHeight = estimateTextHeight(
        fig, subtitleFontSize, len(subtitle), subtitleCharNumber, subtitleLineSpacing
    )

    totalheight = subtitleHeight

    # Shift all axes down
    for ax in fig.axes:
        pos = ax.get_position()
        new_pos = [pos.x0, pos.y0, pos.width, pos.height]
        new_pos[1] -= totalheight  # shift up
        ax.set_position(new_pos)

    # Add the actual text
    y = 1.0
    fig.text(
        leftEdge,
        y,
        titleJustified,
        ha="left",
        va="top",
        fontsize=titleFontSize,
        weight="bold",
    )
    y -= titleHeight + spacing
    fig.text(
        leftEdge,
        y,
        subJustified,
        ha="left",
        va="top",
        fontsize=subtitleFontSize,
        color="#5A5A5A",
        linespacing=subtitleLineSpacing,
    )

    if logo:
        team_icon = Image.open(urllib.request.urlopen(logo)).convert("LA")
        logo_ax = ax.inset_axes([0.95, 0.95, 0.05, 0.05], transform=ax.transAxes)
        logo_ax.imshow(team_icon)
        logo_ax.axis("off")

    if source:
        ax.text(
            x=1,
            y=0.025,
            s=source,
            transform=ax.transAxes,
            ha="right",
            fontsize=9,
            alpha=0.7,
        )


# NEW RELEASE

import numpy as np
from scipy.stats import linregress


def calc_trend_from_values(values):

    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) < 2 or np.all(values == 0):
        return 0.0

    x = np.arange(len(values))
    slope, _, _, _, _ = linregress(x, values)
    slope_normalized = slope / (np.mean(values) + 1e-6)

    return slope_normalized
