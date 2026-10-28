"""Figure style and labels shared by every stage (from plot_style.py).

The polymer-specific parts of plot_style.py (short names, family names)
now come from each dataset's config; configure() loads them at the start of
a run.
"""

import contextlib

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

METHOD_LABELS = {"borutashap": "BorutaShap", "lasso": "Lasso",
                 "xgboost": "XGBoost"}
METHOD_COLORS = {"borutashap": "#0072B2", "lasso": "#E69F00",
                 "xgboost": "#009E73"}            # Okabe-Ito
TICK_ROTATION = 45
# embeddable, editable text in vector output (required by most publishers)
VECTOR_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}

# Light-to-dark ramp for classes of the target (plot_style._ENERGY_RAMP).
CLASS_RAMP = ["#9ecae1", "#c9e6df", "#a5d5cb", "#7ac2b6", "#4dae9f",
              "#2d9687", "#1f7d70", "#186459", "#124b43", "#0b322d"]


def class_colors(n_classes):
    """n colours sampled evenly from CLASS_RAMP, lightest first
    (plot_style.energy_colors)."""
    idx = np.linspace(0, len(CLASS_RAMP) - 1, n_classes).round().astype(int)
    return [CLASS_RAMP[i] for i in idx]

# plot_style.apply_style(), with the closed box (all four spines) built in.
PAPER_RC = {
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 13, "font.weight": "normal",
    "axes.titlesize": 15, "axes.titleweight": "normal",
    "axes.labelsize": 14, "axes.labelweight": "normal",
    "axes.linewidth": 1.3, "axes.edgecolor": "black",
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "xtick.color": "black", "ytick.color": "black",
    "text.color": "black", "axes.labelcolor": "black",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": False,
    "legend.fontsize": 13, "legend.title_fontsize": 14,
    "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
}

_short_names = {}
_blocks = {}


def configure(cfg):
    """Load this dataset's labels. Called once at the start of a run."""
    _short_names.clear()
    _short_names.update(cfg["labels"]["short_names"])
    _blocks.clear()
    _blocks.update(cfg["labels"]["blocks"])


def short(name):
    """Axis label for a descriptor: short_names, else the block prefix
    rule ("Polymer-x" -> "P: x"), else the column name itself."""
    if name in _short_names:
        return _short_names[name]
    for prefix, tag in _blocks.items():
        if name.startswith(prefix):
            return f"{tag}: {name[len(prefix):]}"
    return name


@contextlib.contextmanager
def paper_style():
    """For figures the original scripts drew after plot_style.apply_style()."""
    with mpl.rc_context(PAPER_RC):
        yield


@contextlib.contextmanager
def default_style():
    """For figures the original scripts drew with matplotlib's defaults."""
    with mpl.rc_context():
        mpl.rcdefaults()
        yield


def rotate_xticklabels(ax, rotation=None, fontsize=11, bold=False):
    """Tilted labels for a categorical x axis, ending at their ticks."""
    rot = TICK_ROTATION if rotation is None else rotation
    tilted = 0 < rot < 90
    plt.setp(ax.get_xticklabels(), rotation=rot,
             ha="right" if tilted else "center",
             rotation_mode="anchor" if tilted else None,
             fontsize=fontsize, fontweight="bold" if bold else "normal")


def frame_axes(ax, linewidth=None):
    """Closed box: all four spines, optionally with one line width."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        if linewidth is not None:
            ax.spines[side].set_linewidth(linewidth)
