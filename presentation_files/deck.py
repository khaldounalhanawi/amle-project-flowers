"""Shared colours and Matplotlib style so every chart matches the slides."""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

NAVY = "#23364D"
BLUE = "#66849A"
TEAL = "#72A99A"
BG = "#F7F8F6"
TEXT = "#252B32"
GREY = "#78828C"
OCHRE = "#D6AD67"
RULE = "#D9DDDC"

TEAL_MAP = LinearSegmentedColormap.from_list("deck", [BG, TEAL, NAVY])


def setup():
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "Arial"],
            "font.size": 14,
            "text.color": TEXT,
            "axes.labelcolor": GREY,
            "axes.edgecolor": RULE,
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "axes.titlecolor": NAVY,
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.color": GREY,
            "ytick.color": GREY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 13,
        }
    )
