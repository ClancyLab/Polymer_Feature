"""Histograms stage: each descriptor's histogram, stacked by target class.

Ported from histogram_polymer_analysis_part3.py and _part4.py, which differ
only in how they bin a descriptor. The target is split into classes at
class_edges; a value on an edge belongs to the class above it, as in the
scripts and the cascade stage. Each descriptor of the full dataset is cut
into n_bins bins:
  * custom_bins: the edges given for that descriptor (part4's CUSTOM_BINS);
  * bin_range dense (part4, the default): equal widths over the dense
    range, the lower_q to upper_q quantiles, plus an open-ended bin for
    each tail longer than tail_frac x that range;
  * bin_range full (part3): equal widths from the minimum to the maximum;
  * value_bins: a descriptor with at most n_bins distinct values gets one
    bin per value instead.
A bin "a to b" holds a < value <= b, and the first bin also holds a
(pandas.cut with include_lowest, as in the scripts).

For each descriptor the stage counts the rows of each class in each bin,
takes the percentages per bin and the target's statistics per bin, and
tests whether bin and class are independent (chi-square). run() writes the
tables; plot() draws each descriptor's stacked counts and stacked
percentages from bin_counts.csv and bin_percentages.csv, so `fsphysics
plot` restyles them without counting again.

Outputs (Feature is the descriptor, Descriptor_Bin the bin's label):
  bin_counts.csv        per descriptor and bin: rows of each class, and
                        Total_Count
  bin_percentages.csv   the same as percentages of the bin's rows
  bin_statistics.csv    per descriptor and bin: the target's mean, median,
                        std, min and max (the scripts' Descriptor_Summary)
  bins.csv              per descriptor and bin: its exact edges, and how
                        the bins were chosen (Rule)
  classes.csv           per class: its range of the target, and its rows
  chi_square.csv        per descriptor: the test, the rows in it and
                        outside the bins, Cramer's V
  histograms.xlsx       the scripts' workbook (workbook: true)
  suggested_custom_bins.yaml   the automatic bins, rounded, to freeze
  histograms_summary.txt       classes, bin rules, descriptors ranked by V

Differences from the scripts, all deliberate:
  * One stage for both scripts: bin_range dense gives part4's bins, full
    part3's.
  * Every descriptor of the dataset (features: all), not seven listed ones.
  * The classes come from class_edges, auto by default: n_classes classes
    of about equal size, cut at the target's quantiles (3 significant
    figures). The paper's config sets the scripts' ten ranges, their
    labels and the legend title; their colours are the default for ten
    classes. class_order, class_labels and the colours are settings, and
    the legend adds data.target_unit to the labels (legend_units).
  * Bin labels gain decimals until neighbouring edges print differently.
    The scripts stopped once the labels differed, which can leave a bin
    labelled "0.9 to 0.9" (part3 does, on a range like the paper's
    hardness, 0.299 to 1.501); "-0" prints as "0".
  * A descriptor with at most n_bins distinct values gets one bin per value
    (value_bins). One whose dense range is a single value (most rows
    equal) gets equal widths over its full range; part4 never finished
    such a descriptor, as its labels could not be told apart.
  * The rows outside custom bins are counted in chi_square.csv (N_Outside)
    and the summary; part4 printed a warning, part3 had no custom bins.
  * chi_square.csv adds the rows tested (N), Cramer's V and the share of
    expected counts below 5, and the summary ranks the descriptors by V.
  * The tables are also CSV files, in long format. The workbook keeps the
    scripts' sheets, with the overview sheets first and the class and bin
    tables added; a descriptor's sheets are named by its file name
    (file_names, else the descriptor), cut to Excel's 31 characters.
  * part4 printed its automatic bins as entries to paste into CUSTOM_BINS;
    they are in suggested_custom_bins.yaml, with the lowest and highest
    closed edges rounded outward so that no row falls outside. part4
    rounded them to the nearest value, which can drop the extreme rows;
    its own comments move four descriptors' edges by hand for that reason.
  * The figures go to plots/ (the scripts printed "./plots/" but wrote to
    another folder), as <name>_stacked_counts.png and
    <name>_stacked_percentages.png; save_formats can add PDF or SVG, with
    editable text.
"""

import math
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.colors import is_color_like, to_hex
from scipy.stats import chi2_contingency

from . import outputs, style

# The scripts' class colours (ENERGY_COLORS): tab10, ordered so that darker
# means lower values. Listed from the highest class to the lowest.
RANGE_COLORS = ["#bcbd22", "#17becf", "#e377c2", "#ff7f0e", "#7f7f7f",
                "#2ca02c", "#9467bd", "#1f77b4", "#8c564b", "#d62728"]
RULE_TEXT = {                      # how a descriptor's bins were chosen
    "custom": "custom bins",
    "dense": "equal widths over the dense range, open bins for long tails",
    "full": "equal widths from the minimum to the maximum",
    "values": "one bin per value",
}
_RESERVED = ("Feature", "Descriptor_Bin", "Total_Count")   # table columns
_CHOICES = {"bin_range": ("dense", "full"),
            "class_order": ("descending", "ascending")}
_PLOTS = {"counts": ("stacked_counts", "count_label"),
          "percentages": ("stacked_percentages", "percent_label")}
_LEGEND_LOCS = ("best", "upper right", "upper left", "lower left",
                "lower right", "right", "center left", "center right",
                "lower center", "upper center", "center")
_SHEET_CHARS = "[]:*?/\\"          # not allowed in Excel sheet names
_SHEET_BASE = 24                   # 31 characters, less "_Counts"
_EXTRA_DECIMALS = 20               # most decimals added to tell edges apart


# =============================================================================
# Settings
# =============================================================================
def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _whole(value, low):
    return _number(value) and value == int(value) and value >= low


def _numbers(value, sizes):
    return (isinstance(value, (list, tuple)) and len(value) in sizes
            and all(_number(v) for v in value))


def _settings(cfg):
    """The histograms settings, checked as far as they can be without the
    data."""
    h = cfg["histograms"]
    for key, allowed in _CHOICES.items():
        if h[key] not in allowed:
            raise ValueError(f"histograms.{key}: {h[key]!r} (use "
                             f"{' | '.join(allowed)})")
    tests = {
        "n_bins": (lambda v: _whole(v, 3), "a whole number, 3 or more"),
        "lower_q": (lambda v: _number(v) and 0 <= v < 1,
                    "a fraction from 0 up to, not including, 1"),
        "upper_q": (lambda v: _number(v) and 0 < v <= 1,
                    "a fraction above 0, up to 1"),
        "tail_frac": (lambda v: _number(v) and v >= 0, "a number, 0 or more"),
        "value_bins": (lambda v: isinstance(v, bool), "true or false"),
        "n_classes": (lambda v: _whole(v, 2), "a whole number, 2 or more"),
        "workbook": (lambda v: isinstance(v, bool), "true or false"),
    }
    for key, (ok, need) in tests.items():
        if not ok(h[key]):
            raise ValueError(f"histograms.{key}: {h[key]!r} (use {need})")
    if not h["lower_q"] < h["upper_q"]:
        raise ValueError(f"histograms.lower_q ({h['lower_q']}) must be below "
                         f"upper_q ({h['upper_q']})")
    labels = h["class_labels"]
    if labels != "auto":
        if (not isinstance(labels, (list, tuple))
                or not all(isinstance(t, str) and t.strip() for t in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError(f"histograms.class_labels: {labels!r} (use auto, "
                             f"or a list of different texts, one per class)")
        bad = [t for t in labels if t in _RESERVED or t.startswith("_")]
        if bad:
            raise ValueError(f"histograms.class_labels: {bad} can't be used "
                             f"(the tables' own columns are "
                             f"{', '.join(_RESERVED)}, and matplotlib hides "
                             f"labels starting with _)")
    for key in ("custom_bins", "file_names"):
        if not isinstance(h[key], dict):
            raise ValueError(f"histograms.{key}: a section, e.g. "
                             f"{{descriptor: ...}}")
    for name, text in h["file_names"].items():
        if (not isinstance(text, str) or not text.strip()
                or any(ch in text for ch in outputs.BAD_FILE_CHARS)):
            raise ValueError(f"histograms.file_names.{name}: {text!r} (use a "
                             f"name without any of "
                             f"{' '.join(outputs.BAD_FILE_CHARS)})")
    return h, _figure_settings(cfg)


def _figure_settings(cfg):
    """histograms.figure as attributes, after checking them."""
    f = cfg["histograms"]["figure"]
    formats = FigureCanvasBase.get_supported_filetypes()
    tests = {
        "plots": (lambda v: isinstance(v, (list, tuple))
                  and len(set(v)) == len(v) and all(p in _PLOTS for p in v),
                  "a list of counts and/or percentages"),
        "figsize": (lambda v: _numbers(v, (2,)) and min(v) > 0,
                    "[width, height] in inches"),
        "dpi": (lambda v: _number(v) and v > 0, "a number above 0"),
        "label_fontsize": (lambda v: _number(v) and v > 0,
                           "a number above 0"),
        "tick_rotation": (_number, "an angle in degrees"),
        "count_label": (lambda v: isinstance(v, str), "a text"),
        "percent_label": (lambda v: isinstance(v, str), "a text"),
        "legend_title": (lambda v: isinstance(v, str), "auto, or a text"),
        "legend_loc": (lambda v: v in _LEGEND_LOCS,
                       " | ".join(_LEGEND_LOCS)),
        "legend_anchor": (lambda v: v is None or _numbers(v, (2, 4)),
                          "[x, y] in axes fractions, or null"),
        "legend_reverse": (lambda v: isinstance(v, bool), "true or false"),
        "legend_units": (lambda v: isinstance(v, bool), "true or false"),
        "class_colors": (lambda v: v == "auto" or (
            isinstance(v, (list, tuple)) and all(is_color_like(c)
                                                  for c in v)),
            "auto, or a list of colours"),
        "axis_labels": (lambda v: isinstance(v, dict) and all(
            isinstance(t, str) for t in v.values()),
            "a section {descriptor: label}"),
        "save_formats": (lambda v: isinstance(v, (list, tuple)) and v
                         and all(isinstance(e, str) and e.lower() in formats
                                 for e in v),
                         "a list of formats such as png, pdf, svg"),
    }
    for key, (ok, need) in tests.items():
        if not ok(f[key]):
            raise ValueError(f"histograms.figure.{key}: {f[key]!r} (use "
                             f"{need})")
    return SimpleNamespace(**f)


def _check_names(h, X):
    """The descriptors named in the settings exist."""
    sections = {"custom_bins": h["custom_bins"],
                "file_names": h["file_names"],
                "figure.axis_labels": h["figure"]["axis_labels"]}
    for where, section in sections.items():
        missing = [n for n in section if n not in X.columns]
        if missing:
            raise KeyError(f"histograms.{where}: {missing} are not "
                           f"descriptors of this dataset.")


def _features(h, X):
    if h["features"] == "all":
        return list(X.columns)
    feats = h["features"]
    if (not isinstance(feats, (list, tuple)) or not feats
            or not all(isinstance(f, str) for f in feats)
            or len(set(feats)) != len(feats)):
        raise ValueError(f"histograms.features: {feats!r} (use all, or a "
                         f"list of different descriptors)")
    missing = [f for f in feats if f not in X.columns]
    if missing:
        raise KeyError(f"histograms.features: {missing} are not descriptors "
                       f"of this dataset.")
    return list(feats)


def _edge(value, where):
    """One custom edge: a number, or -inf / inf for an open end."""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"{where}: {value!r} is not a number (write -inf "
                             f"or inf for an open end)") from None
    if not _number(value) or math.isnan(value):
        raise ValueError(f"{where}: {value!r} is not a number")
    return float(value)


def _custom_bins(h):
    """{descriptor: (edges, decimals)}, checked."""
    out = {}
    for name, spec in h["custom_bins"].items():
        where = f"histograms.custom_bins.{name}"
        if (not isinstance(spec, dict) or "edges" not in spec
                or set(spec) - {"edges", "decimals"}):
            raise ValueError(f"{where}: {spec!r} (use {{edges: [...], "
                             f"decimals: n}})")
        raw = spec["edges"]
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            raise ValueError(f"{where}.edges: {raw!r} (use a list of two or "
                             f"more numbers)")
        edges = [_edge(v, f"{where}.edges") for v in raw]
        if (any(np.isneginf(e) for e in edges[1:])
                or any(np.isposinf(e) for e in edges[:-1])):
            raise ValueError(f"{where}.edges: -inf can only be the first "
                             f"edge, and inf only the last")
        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError(f"{where}.edges: {raw!r} must increase")
        decimals = spec.get("decimals", 0)
        if not _whole(decimals, 0):
            raise ValueError(f"{where}.decimals: {decimals!r} (use a whole "
                             f"number, 0 or more)")
        out[name] = (np.array(edges, dtype=float), int(decimals))
    return out


# =============================================================================
# Printing numbers in labels
# =============================================================================
def _printed(value, spec):
    """A number in a format such as ".2f" or ".6g", without the minus sign
    of a value that prints as zero ("-0.0")."""
    text = f"{value:{spec}}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _apart(values, specs):
    """The values printed in the first format that keeps neighbours apart
    (each printed value above the one before), or None."""
    for spec in specs:
        texts = [_printed(v, spec) for v in values]
        numbers = [float(t) for t in texts]
        if all(a < b for a, b in zip(numbers, numbers[1:])):
            return texts, spec
    return None


# =============================================================================
# Classes of the target
# =============================================================================
def class_edges(h, y):
    """The class edges in target units, increasing. auto: n_classes classes
    of about equal size, cut at the target's quantiles (taken as values of
    the target, so a target with a few values can get a class per value)
    to 3 significant figures, more where needed to keep them apart; an edge
    that would leave a class without rows (ties in the target) is
    dropped."""
    y = np.asarray(y, dtype=float)
    setting = h["class_edges"]
    if setting != "auto":
        if (not isinstance(setting, (list, tuple)) or not setting
                or not all(_number(e) and math.isfinite(e) for e in setting)
                or len(set(setting)) != len(setting)):
            raise ValueError(f"histograms.class_edges: {setting!r} (use auto, "
                             f"or a list of one or more different numbers)")
        return sorted(float(e) for e in setting)
    k = h["n_classes"]
    quantiles = np.unique(np.quantile(y, np.arange(1, k) / k, method="lower"))
    for digits in range(3, 18):
        edges = np.unique([float(f"{q:.{digits}g}") for q in quantiles])
        if len(edges) == len(quantiles):
            break
    edges = [float(e) for e in edges]
    while edges:
        rows = np.bincount(np.digitize(y, edges), minlength=len(edges) + 1)
        empty = np.flatnonzero(rows == 0)
        if not len(empty):
            break
        edges.pop(min(int(empty[0]), len(edges) - 1))
    if not edges:
        raise ValueError("The target has a single value, so it has no "
                         "classes to compare.")
    return edges


def make_classes(h, y):
    """The classes in table order (Class, Lower, Upper, Rows) and each row's
    class. The table order is class_order; a row's class is the one whose
    Lower <= value < Upper."""
    edges = class_edges(h, y)
    k = len(edges) + 1
    bounds = [-np.inf] + edges + [np.inf]
    shown = _apart(edges, [f".{d}g" for d in range(6, 18)])[0]
    auto = ([f"< {shown[0]}"]
            + [f"{a} to {b}" for a, b in zip(shown, shown[1:])]
            + [f"> {shown[-1]}"])
    order = list(range(k))
    if h["class_order"] == "descending":
        order.reverse()
    if h["class_labels"] == "auto":
        labels = [auto[i] for i in order]
    else:
        labels = list(h["class_labels"])
        if len(labels) != k:
            raise ValueError(f"histograms.class_labels: {len(labels)} labels "
                             f"for {k} classes (class edges {edges})")
    codes = np.digitize(np.asarray(y, dtype=float), edges)    # 0 = lowest
    position = np.empty(k, dtype=int)
    position[order] = np.arange(k)
    row_class = np.asarray(labels, dtype=object)[position[codes]]
    rows = np.bincount(codes, minlength=k)
    table = pd.DataFrame({"Class": labels,
                          "Lower": [bounds[i] for i in order],
                          "Upper": [bounds[i + 1] for i in order],
                          "Rows": [int(rows[i]) for i in order]})
    return table, row_class


def class_colors(F, classes):
    """One colour per class, in table order. auto: the scripts' ten colours,
    sampled evenly for fewer classes (viridis for more), darker for lower
    values whatever the order."""
    k = len(classes)
    if F.class_colors != "auto":
        colors = list(F.class_colors)
        if len(colors) != k:
            raise ValueError(f"histograms.figure.class_colors: {len(colors)} "
                             f"colours for {k} classes")
        return colors
    if k <= len(RANGE_COLORS):
        idx = np.linspace(0, len(RANGE_COLORS) - 1, k).round().astype(int)
        by_value = [RANGE_COLORS[i] for i in idx]          # highest first
    else:
        cmap = mpl.colormaps["viridis"]
        by_value = [to_hex(cmap(t)) for t in np.linspace(1, 0, k)]
    rank = classes["Lower"].rank(ascending=False, method="first")
    return [by_value[int(r) - 1] for r in rank]


# =============================================================================
# Bins of a descriptor
# =============================================================================
def range_labels(edges, decimals):
    """part4's make_labels: "< b" and "> a" for the open bins, else
    "a to b"."""
    spec = f".{decimals}f"
    labels = []
    for a, b in zip(edges[:-1], edges[1:]):
        if np.isneginf(a):
            labels.append(f"< {_printed(b, spec)}")
        elif np.isposinf(b):
            labels.append(f"> {_printed(a, spec)}")
        else:
            labels.append(f"{_printed(a, spec)} to {_printed(b, spec)}")
    return labels


def label_decimals(edges, start, name):
    """The fewest decimals, from start, at which neighbouring finite edges
    print differently (so no two labels match, and no label reads
    "a to a")."""
    finite = [e for e in edges if np.isfinite(e)]
    found = _apart(finite, [f".{n}f" for n in
                            range(start, start + _EXTRA_DECIMALS + 1)])
    if found is None:
        raise ValueError(f"{name}: its bin edges are too close together to "
                         f"label ({list(edges)}); set histograms.custom_bins "
                         f"for it.")
    return int(found[1][1:-1])


def value_labels(values):
    """Labels for one bin per value: the values to 6 significant figures,
    more where needed to tell them apart."""
    found = _apart(values, [f".{d}g" for d in range(6, 18)])
    return found[0] if found else [repr(float(v)) for v in values]


def dense_edges(values, h):
    """part4's auto_bins: equal widths over the dense range (the lower_q to
    upper_q quantiles), with an open-ended bin for each tail longer than
    tail_frac x that range. None if the dense range is a single value."""
    vmin, vmax = values.min(), values.max()
    lo, hi = values.quantile(h["lower_q"]), values.quantile(h["upper_q"])
    span = hi - lo
    if not span > 0:
        return None
    open_lower = (lo - vmin) > h["tail_frac"] * span
    open_upper = (vmax - hi) > h["tail_frac"] * span
    lo_edge = lo if open_lower else vmin
    hi_edge = hi if open_upper else vmax
    n_closed = h["n_bins"] - int(open_lower) - int(open_upper)
    edges = list(np.linspace(lo_edge, hi_edge, n_closed + 1))
    if open_lower:
        edges = [-np.inf] + edges
    if open_upper:
        edges = edges + [np.inf]
    return np.array(edges)


def make_bins(name, values, h, custom):
    """How one descriptor is binned: rule, edges (range bins) or values (one
    bin per value), labels, label decimals and a note."""
    if name in custom:
        edges, start = custom[name]
        n = label_decimals(edges, start, name)
        return SimpleNamespace(rule="custom", edges=edges, values=None,
                               labels=range_labels(edges, n), decimals=n,
                               note=None)
    distinct = np.unique(values.to_numpy())
    if h["value_bins"] and len(distinct) <= h["n_bins"]:
        return SimpleNamespace(rule="values", edges=None, values=distinct,
                               labels=value_labels(distinct), decimals=None,
                               note=None)
    rule, note, edges = h["bin_range"], None, None
    if rule == "dense":
        edges = dense_edges(values, h)
        if edges is None:
            rule = "full"
            note = (f"its {h['lower_q']:g} and {h['upper_q']:g} quantiles "
                    f"are the same value (most rows share it), so its bins "
                    f"have equal widths over its full range instead")
    if edges is None:
        edges = np.linspace(values.min(), values.max(), h["n_bins"] + 1)
    n = label_decimals(edges, 0, name)
    return SimpleNamespace(rule=rule, edges=edges, values=None,
                           labels=range_labels(edges, n), decimals=n,
                           note=note)


def assign(values, bins):
    """Each row's bin, as an ordered categorical (missing outside the
    bins)."""
    if bins.rule == "values":
        codes = np.searchsorted(bins.values, values.to_numpy())
        return pd.Series(pd.Categorical.from_codes(
            codes, categories=bins.labels, ordered=True), index=values.index,
            name=values.name)
    return pd.cut(values, bins=bins.edges, labels=bins.labels,
                  include_lowest=True, duplicates="drop")


def bin_rows(name, bins):
    """bins.csv rows for one descriptor."""
    if bins.rule == "values":
        lower = upper = list(bins.values)
    else:
        lower, upper = list(bins.edges[:-1]), list(bins.edges[1:])
    return [{"Feature": name, "Descriptor_Bin": label, "Lower": float(a),
             "Upper": float(b), "Rule": bins.rule}
            for label, a, b in zip(bins.labels, lower, upper)]


def suggested_edges(bins):
    """part4's suggested CUSTOM_BINS edges for automatic bins: rounded to
    the labels' decimals (at least 1), with the closed lowest and highest
    edges rounded outward, so every row stays inside."""
    d = max(bins.decimals, 1)
    last = len(bins.edges) - 1
    while True:
        out = []
        for i, e in enumerate(bins.edges):
            e = float(e)
            if not math.isfinite(e):
                out.append(e)
                continue
            v = round(e, d)
            if i == 0 and v > e:
                v = round(v - 10.0 ** -d, d)
            elif i == last and v < e:
                v = round(v + 10.0 ** -d, d)
            out.append(v)
        if all(a < b for a, b in zip(out, out[1:])):
            return out
        d += 1


# =============================================================================
# Tables
# =============================================================================
def tables_for(name, values, y, bins, row_class, class_labels, target):
    """One descriptor's tables, as the scripts compute them: counts and
    percentages (bins x classes), the target's statistics per bin, and the
    chi-square row."""
    binned = assign(values, bins)
    n_out = int(values.notna().sum() - binned.notna().sum())
    temp = pd.DataFrame({"Descriptor_Bin": binned})
    temp["_class"] = row_class
    temp["_target"] = np.asarray(y, dtype=float)
    temp = temp.dropna(subset=["Descriptor_Bin"])

    counts = pd.crosstab(temp["Descriptor_Bin"], temp["_class"])
    counts = counts.reindex(index=bins.labels, columns=class_labels,
                            fill_value=0)
    counts["Total_Count"] = counts.sum(axis=1)
    pct = (counts.drop(columns=["Total_Count"])
           .div(counts["Total_Count"], axis=0) * 100).fillna(0.0)
    pct["Total_Count"] = counts["Total_Count"]

    stats = (temp.groupby("Descriptor_Bin", observed=False)["_target"]
             .agg(**{f"Mean_{target}": "mean", f"Median_{target}": "median",
                     f"Std_{target}": "std", f"Min_{target}": "min",
                     f"Max_{target}": "max", "Count": "size"})
             .reset_index())
    stats.insert(0, "Feature", name)

    chi_table = counts.drop(columns=["Total_Count"])
    chi_table = chi_table.loc[chi_table.sum(axis=1) > 0]
    chi_table = chi_table.loc[:, chi_table.sum(axis=0) > 0]
    try:
        chi2, p_value, dof, expected = chi2_contingency(chi_table)
    except ValueError:
        chi2 = p_value = dof = np.nan
        expected = None
    n_rows = int(chi_table.to_numpy().sum())
    cramers_v = below5 = np.nan
    if expected is not None:
        below5 = 100.0 * float((np.asarray(expected) < 5).mean())
        r, c = chi_table.shape
        if min(r, c) > 1:
            plain = chi2_contingency(chi_table, correction=False)[0]
            cramers_v = math.sqrt(plain / (n_rows * (min(r, c) - 1)))
    chi_row = {"Feature": name, "Chi2": chi2, "Degrees_of_Freedom": dof,
               "P_Value": p_value, "N": n_rows, "N_Outside": n_out,
               "Cramers_V": cramers_v, "Pct_Expected_Below_5": below5}
    return counts, pct, stats, chi_row


def _long(table, name):
    out = table.reset_index()
    out.insert(0, "Feature", name)
    return out


def _sheet_bases(stems):
    """A workbook sheet name base per descriptor: its file name without the
    characters Excel refuses, cut to fit 31 characters with "_Counts",
    different from the others (Excel ignores case)."""
    used, out = set(), {}
    for name, stem in stems.items():
        base = "".join("_" if ch in _SHEET_CHARS else ch for ch in stem)
        base = base.lstrip("'")[:_SHEET_BASE] or "descriptor"
        candidate, i = base, 2
        while candidate.lower() in used:
            tag = f"~{i}"
            candidate = base[:_SHEET_BASE - len(tag)] + tag
            i += 1
        used.add(candidate.lower())
        out[name] = candidate
    return out


def file_stems(h, features):
    """Each descriptor's figure file name (file_names, else the descriptor
    name made safe), checked for clashes; Windows ignores case."""
    stems = {f: h["file_names"].get(f) or outputs.file_stem(f)
             for f in features}
    seen = {}
    for name, stem in stems.items():
        other = seen.setdefault(stem.lower(), name)
        if other != name:
            raise ValueError(f"Descriptors {other!r} and {name!r} would share "
                             f"the file name {stem!r} (file names ignore case "
                             f"on Windows); give one of them another name in "
                             f"histograms.file_names.")
    return stems


def _p_text(p):
    if not np.isfinite(p):
        return "n/a"
    return "<1e-300" if p < 1e-300 else f"{p:.2g}"


_KEEP = "\u00a0"            # joins words that a line must not split


def _wrap(text, indent=""):
    return textwrap.fill(text, width=79, initial_indent=indent,
                         subsequent_indent=indent + "    ",
                         break_on_hyphens=False).replace(_KEEP, " ")


def _items(texts):
    """A list for _wrap, each item kept on one line."""
    return ", ".join(t.replace(" ", _KEEP) for t in texts)


def class_ranges(classes):
    """Each class's range of the target, e.g. "-1.5 <= value < -1"."""
    edges = sorted({float(e) for e in np.r_[classes["Lower"], classes["Upper"]]
                    if math.isfinite(e)})
    shown = dict(zip(edges, _apart(edges, [f".{d}g" for d in
                                           range(6, 18)])[0]))
    out = []
    for lo, hi in zip(classes["Lower"], classes["Upper"]):
        if not math.isfinite(lo):
            out.append(f"value < {shown[hi]}")
        elif not math.isfinite(hi):
            out.append(f"value >= {shown[lo]}")
        else:
            out.append(f"{shown[lo]} <= value < {shown[hi]}")
    return out


def _summary(cfg, h, n_rows, classes, all_bins, chi, notes, stems):
    unit = cfg["data"]["target_unit"]
    first = "highest" if h["class_order"] == "descending" else "lowest"
    width = max(len(t) for t in classes["Class"])
    lines = [f"Rows: {n_rows} (the full dataset); descriptors: {len(chi)}",
             f"Classes of {cfg['data']['target']}"
             + (f" ({unit})" if unit else "")
             + f", from the {first} (the bottom of each bar):"]
    lines += [f"  {r.Class:<{width}}  {rng:<26} {r.Rows:>7} rows"
              for r, rng in zip(classes.itertuples(), class_ranges(classes))]
    lines += ["", _wrap(f"Bins (up to {h['n_bins']} per descriptor; \"a to "
                        f"b\" holds a < value <= b, the first bin also a):")]
    rule_text = dict(RULE_TEXT, dense=(
        f"dense range ({h['lower_q']:g} to {h['upper_q']:g} quantiles), "
        f"open bins for tails over {h['tail_frac']:g} x its width"))
    for rule in RULE_TEXT:
        names = [style.short(n) for n, b in all_bins.items() if b.rule == rule]
        if names:
            lines.append(_wrap(f"{rule_text[rule]}: {_items(names)}", "  "))
    renamed = {n: s for n, s in stems.items() if s != n}
    if renamed:
        lines += ["", "File names of figures and workbook sheets:"]
        lines += [f"  {n}: {s}" for n, s in renamed.items()]
    lines += [
        "",
        _wrap("Chi-square test of bin x class (empty bins and classes left "
              "out), strongest association first by Cramer's V (0: the class "
              "does not depend on the bin, 1: the bin fixes it). Rows that "
              "share a descriptor value, such as one polymer in many "
              "solvents, are not independent, so the p-values are optimistic; "
              "compare descriptors by V. exp<5: the share of expected counts "
              "below 5 (the test is approximate above about 20%)."),
        f"  {'V':>5} {'chi2':>9} {'dof':>4} {'p':>8} {'rows':>6} "
        f"{'outside':>7} {'exp<5':>5}  descriptor"]
    ranked = chi.sort_values("Cramers_V", ascending=False, na_position="last",
                             kind="stable")
    for r in ranked.itertuples():
        v = "n/a" if not np.isfinite(r.Cramers_V) else f"{r.Cramers_V:.3f}"
        c2 = "n/a" if not np.isfinite(r.Chi2) else f"{r.Chi2:.1f}"
        dof = ("n/a" if not np.isfinite(r.Degrees_of_Freedom)
               else f"{int(r.Degrees_of_Freedom)}")
        b5 = ("n/a" if not np.isfinite(r.Pct_Expected_Below_5)
              else f"{r.Pct_Expected_Below_5:.0f}%")
        lines.append(f"  {v:>5} {c2:>9} {dof:>4} {_p_text(r.P_Value):>8} "
                     f"{r.N:>6} {r.N_Outside:>7} {b5:>5}  "
                     f"{style.short(r.Feature)}")
    if notes:
        lines += ["", "Notes:"] + [_wrap(n, "  ") for n in notes]
    return "\n".join(lines) + "\n"


def _write_workbook(path, overview, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, frame in overview:
            frame.to_excel(writer, sheet_name=sheet, index=False)
        for base, counts, pct in sheets:
            counts.reset_index().to_excel(writer, sheet_name=f"{base}_Counts",
                                          index=False)
            pct.reset_index().to_excel(writer, sheet_name=f"{base}_Pcts",
                                       index=False)


def _write_suggestions(path, entries):
    header = (
        "# The bins the histograms stage chose by itself, rounded to their\n"
        "# labels' decimals (at least one); the lowest and highest closed\n"
        "# edges are rounded outward, so every row stays in a bin. To\n"
        "# freeze a descriptor's bins, copy its entry under\n"
        "# histograms: custom_bins: in the dataset's config (round the edges\n"
        "# further if you like).\n")
    body = yaml.safe_dump({"custom_bins": entries}, sort_keys=False,
                          default_flow_style=None, allow_unicode=True,
                          width=79)
    path.write_text(header + body, encoding="utf-8")


# =============================================================================
# Stage entry points
# =============================================================================
def run(cfg, X, y, out_dir):
    out_dir = Path(out_dir)
    h, F = _settings(cfg)
    _check_names(h, X)
    features = _features(h, X)
    custom = _custom_bins(h)
    stems = file_stems(h, features)
    classes, row_class = make_classes(h, y)
    class_colors(F, classes)
    class_labels = classes["Class"].tolist()
    target = cfg["data"]["target"]

    all_bins, results, notes = {}, {}, []
    for name in features:
        values = X[name].astype(float)
        b = make_bins(name, values, h, custom)
        all_bins[name] = b
        results[name] = tables_for(name, values, y, b, row_class,
                                   class_labels, target)
        n_out = results[name][3]["N_Outside"]
        if n_out:
            notes.append(f"{name}: {n_out} rows lie outside its custom bins "
                         f"and are left out.")
        if b.note:
            notes.append(f"{name}: {b.note}.")
    if h["class_edges"] == "auto" and len(classes) < h["n_classes"]:
        notes.insert(0, f"The target's ties allow only {len(classes)} "
                        f"classes of the {h['n_classes']} asked for.")
    empty = classes.loc[classes["Rows"] == 0, "Class"].tolist()
    if empty:
        notes.insert(0, f"Classes without rows: {', '.join(empty)}.")

    outputs.start(out_dir, cfg)
    chi = pd.DataFrame([results[n][3] for n in features])
    counts_long = pd.concat([_long(results[n][0], n) for n in features],
                            ignore_index=True)
    pct_long = pd.concat([_long(results[n][1], n) for n in features],
                         ignore_index=True)
    stats = pd.concat([results[n][2] for n in features], ignore_index=True)
    bins_table = pd.DataFrame([row for n in features
                               for row in bin_rows(n, all_bins[n])])
    counts_long.to_csv(out_dir / "bin_counts.csv", index=False)
    pct_long.to_csv(out_dir / "bin_percentages.csv", index=False)
    stats.to_csv(out_dir / "bin_statistics.csv", index=False)
    bins_table.to_csv(out_dir / "bins.csv", index=False)
    classes.to_csv(out_dir / "classes.csv", index=False)
    chi.to_csv(out_dir / "chi_square.csv", index=False)
    if h["workbook"]:
        bases = _sheet_bases(stems)
        _write_workbook(
            out_dir / "histograms.xlsx",
            [("ChiSquare_Results", chi), ("Descriptor_Summary", stats),
             ("Classes", classes), ("Bins", bins_table)],
            [(bases[n], results[n][0], results[n][1]) for n in features])
    auto = {n: {"edges": suggested_edges(b), "decimals": b.decimals}
            for n, b in all_bins.items() if b.rule in ("dense", "full")}
    if auto:
        _write_suggestions(out_dir / "suggested_custom_bins.yaml", auto)
    (out_dir / "histograms_summary.txt").write_text(
        _summary(cfg, h, len(X), classes, all_bins, chi, notes, stems),
        encoding="utf-8")

    unit = cfg["data"]["target_unit"]
    print(f"{len(X)} rows, {len(features)} descriptors; classes of {target}"
          + (f" ({unit})" if unit else "") + ": "
          + ", ".join(f"{r.Class} ({r.Rows})" for r in classes.itertuples()))
    counts_by_rule = {r: sum(b.rule == r for b in all_bins.values())
                      for r in ("custom", "dense", "full", "values")}
    print("Bins: " + ", ".join(f"{n} {r}" for r, n in counts_by_rule.items()
                               if n))
    for note in notes:
        print(f"  note: {note}")
    top = chi.dropna(subset=["Cramers_V"]).nlargest(5, "Cramers_V")
    if len(top):
        print("Strongest association with the classes (Cramer's V): "
              + ", ".join(f"{style.short(r.Feature)} {r.Cramers_V:.2f}"
                          for r in top.itertuples()))


def _plain_value(value):
    """Tuples as lists, as a value reads back from config_used.yaml."""
    if isinstance(value, dict):
        return {k: _plain_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(v) for v in value]
    return value


def changed_since_run(cfg, out_dir):
    """The settings outside histograms.figure that differ from those the
    saved tables were made with (config_used.yaml)."""
    path = Path(out_dir) / "config_used.yaml"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        used = (yaml.safe_load(fh) or {}).get("histograms", {})
    now = cfg["histograms"]
    return [key for key in now if key != "figure"
            and _plain_value(used.get(key)) != _plain_value(now[key])]


def read_table(path, text=("Feature", "Descriptor_Bin")):
    """A table of this stage as written: text columns as text (a label such
    as "NA" stays text), only empty cells missing, and numbers exactly
    (pandas' default parser can be one unit in the last place off)."""
    return pd.read_csv(path, dtype={c: str for c in text},
                       keep_default_na=False, na_values=[""],
                       float_precision="round_trip")


def plot(cfg, out_dir):
    out_dir = Path(out_dir)
    if not (out_dir / "bin_counts.csv").exists():
        raise FileNotFoundError(f"{out_dir / 'bin_counts.csv'} not found; run "
                                f"the histograms stage first.")
    h, F = _settings(cfg)
    changed = changed_since_run(cfg, out_dir)
    if changed:
        print(f"note: histograms.{', histograms.'.join(changed)} changed "
              f"since the stage ran; the figures show the saved tables. Run "
              f"the stage again to apply them.")
    classes = read_table(out_dir / "classes.csv", text=("Class",))
    labels = classes["Class"].tolist()
    colors = class_colors(F, classes)
    tables = {"counts": read_table(out_dir / "bin_counts.csv"),
              "percentages": read_table(out_dir / "bin_percentages.csv")}
    features = list(dict.fromkeys(tables["counts"]["Feature"]))
    stems = file_stems(h, features)
    unit = cfg["data"]["target_unit"]
    legend_labels = [f"{t} {unit}" if unit and F.legend_units else t
                     for t in labels]
    title = F.legend_title
    if title == "auto":
        title = (f"{cfg['data']['target_label']} range"
                 + (f" ({unit})" if unit else ""))
    if not F.plots:
        print("histograms.figure.plots is empty, so no figures.")
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    with style.default_style(), mpl.rc_context(style.VECTOR_RC):
        for name in features:
            for kind in F.plots:
                suffix, ylabel = _PLOTS[kind]
                table = tables[kind]
                frame = (table[table["Feature"] == name]
                         .set_index("Descriptor_Bin")[labels])
                try:
                    _draw(frame, colors, F,
                          F.axis_labels.get(name) or style.short(name),
                          getattr(F, ylabel), legend_labels, title,
                          plots_dir / f"{stems[name]}_{suffix}")
                except ValueError as exc:      # e.g. a broken $...$ label
                    plt.close("all")
                    raise ValueError(f"The {kind} figure of {name} could not "
                                     f"be drawn: {exc}") from exc
    print(f"Figures: {len(features)} descriptors x {len(F.plots)} "
          f"({', '.join(F.plots)}) in {plots_dir}")


def _draw(frame, colors, F, xlabel, ylabel, legend_labels, title, base):
    """One stacked bar chart, drawn as the scripts draw it."""
    ax = frame.plot(kind="bar", stacked=True, figsize=tuple(F.figsize),
                    color=colors)
    plt.xlabel(xlabel, fontsize=F.label_fontsize)
    plt.ylabel(ylabel, fontsize=F.label_fontsize)
    plt.xticks(rotation=F.tick_rotation)
    handles, _ = ax.get_legend_handles_labels()
    shown = list(legend_labels)
    if F.legend_reverse:
        handles, shown = handles[::-1], shown[::-1]
    where = {"loc": F.legend_loc}
    if F.legend_anchor is not None:
        where["bbox_to_anchor"] = tuple(F.legend_anchor)
    plt.legend(handles, shown, title=title, **where)
    plt.tight_layout()
    for fmt in F.save_formats:
        plt.savefig(f"{base}.{fmt.lower()}", dpi=F.dpi, bbox_inches="tight")
    plt.close()
